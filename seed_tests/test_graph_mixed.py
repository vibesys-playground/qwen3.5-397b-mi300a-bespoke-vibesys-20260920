"""CPU tests for `SEED_MIXED_GRAPH` (graph_mixed.py): a captured mixed decode+prefill step
must produce the separate decode and prefill steps' logits and lane state, and a scheduler over
it the same greedy tokens and prefix reuse as with the flag off.

The capture backend is test_graph_capture.py's `EagerBackend` (a "replay" re-runs the static
step on the same buffers), so these check the static step's arithmetic and buffer handling;
whether it captures on the device is `prepare`'s own startup validation on the GPU.

    <python-with-torch> -m pytest seed_tests/test_graph_mixed.py -q -o addopts=
"""

import json
import random
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import graph_decode  # noqa: E402
import graph_mixed  # noqa: E402
from graph_decode import GraphDecodeRunner, pad_slot_for  # noqa: E402
from graph_mixed import MixedShape, mixed_shape_for, parse_mixed_shapes  # noqa: E402
from graph_prefill import Shape  # noqa: E402
from scheduler import Request, Scheduler  # noqa: E402
from session_cache import SessionCache  # noqa: E402
from test_batched_decode import build, prompt_of  # noqa: E402
from test_graph_capture import CorruptingBackend, EagerBackend  # noqa: E402
from test_graph_prefill import lane_state, peer_rejects, sessions  # noqa: E402
from test_scheduler import Sink, drain  # noqa: E402
from test_seed_parity import build_hf, write_checkpoint  # noqa: E402

DECODE, PREFILL = "2,4", "1x4,2x4,1x8"
MAX_BATCH = 5


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("tiny-mixed-graph")
    write_checkpoint(build_hf(), out, mxfp4=False)
    return out


def test_flag_defaults_to_off() -> None:
    assert graph_mixed.MIXED_GRAPH is False


def test_parse_is_the_product_sorted_by_area() -> None:
    got = parse_mixed_shapes("4, 2,99", "1x4,2x4,8x4", max_batch=8, max_seq=64)
    assert [s.name for s in got] == ["2+1x4", "4+1x4", "2+2x4", "4+2x4"]
    with pytest.raises(ValueError, match="bad entry"):
        parse_mixed_shapes("x", "1x4", 8, 64)


def test_totals_give_one_row_per_decode_bucket() -> None:
    got = parse_mixed_shapes("16,48,64", "", 64, 4096, totals_spec="64,128")
    assert [s.name for s in got] == ["16+1x48", "48+1x16", "16+1x112", "48+1x80", "64+1x64"]
    assert {s.area for s in got} == {64, 128}
    two = parse_mixed_shapes("48", "", 64, 4096, totals_spec="2x256")
    assert [s.name for s in two] == ["48+2x104"]
    with pytest.raises(ValueError, match="bad entry"):
        parse_mixed_shapes("48", "", 64, 4096, totals_spec="ax256")


def test_shape_for_holds_both_halves() -> None:
    shapes = parse_mixed_shapes("16,48", "1x16,1x256,2x64", 64, 4096)
    assert mixed_shape_for(10, [5], shapes) == MixedShape(16, Shape(1, 16))
    assert mixed_shape_for(17, [5], shapes) == MixedShape(48, Shape(1, 16))
    assert mixed_shape_for(10, [20], shapes) == MixedShape(16, Shape(2, 64))
    assert mixed_shape_for(10, [65], shapes) == MixedShape(16, Shape(1, 256))
    assert mixed_shape_for(49, [5], shapes) is None
    assert mixed_shape_for(10, [257], shapes) is None
    assert mixed_shape_for(10, [1, 1, 1], shapes) is None


def test_pad_slot_prefers_a_free_lane_then_the_other_half() -> None:
    assert pad_slot_for([0, 1], 4, avoid=[2]) == 3
    assert pad_slot_for([0, 1], 3, avoid=[2]) == 2  # no free lane: the other half's lane
    assert pad_slot_for([0, 1], 2) == 0  # unchanged without `avoid`


def runner_for(checkpoint: Path, monkeypatch, backend=None, max_batch=MAX_BATCH):  # noqa: ANN001, ANN201
    monkeypatch.setenv("SEED_MIXED_GRAPH_DECODE", DECODE)
    monkeypatch.setenv("SEED_MIXED_GRAPH_PREFILL", PREFILL)
    monkeypatch.setenv("SEED_MIXED_GRAPH_TOTALS", "")
    model = build(checkpoint, max_batch=max_batch)
    runner = GraphDecodeRunner(
        model, backend=backend or EagerBackend(), prefill_graphs=False, mixed_graphs=True
    )
    runner.prepare()
    return model, runner


def test_prepare_captures_and_validates_every_shape(checkpoint: Path, monkeypatch) -> None:  # noqa: ANN001
    _, runner = runner_for(checkpoint, monkeypatch)
    assert runner.mixed_runner.enabled
    assert sorted(runner.mixed_runner.graphs) == sorted(
        parse_mixed_shapes(DECODE, PREFILL, MAX_BATCH, 96)
    )


def test_a_replay_that_computes_nothing_is_rejected(checkpoint: Path, monkeypatch) -> None:  # noqa: ANN001
    _, runner = runner_for(checkpoint, monkeypatch, backend=CorruptingBackend())
    assert runner.mixed_runner is not None and not runner.mixed_runner.enabled
    assert runner.mixed_fit(1) is None


def test_a_shape_one_rank_rejects_is_dropped_on_every_rank(checkpoint: Path, monkeypatch) -> None:  # noqa: ANN001
    shapes = parse_mixed_shapes(DECODE, PREFILL, MAX_BATCH, 96)
    target = shapes[1]
    peer_rejects(monkeypatch, graph_mixed.MixedGraphRunner, target)
    _, runner = runner_for(checkpoint, monkeypatch)
    mr = runner.mixed_runner
    assert mr.enabled
    assert mr.shapes == [s for s in shapes if s != target]
    assert target not in mr.graphs


@pytest.mark.parametrize(
    ("decode", "chunks", "chained"),
    [
        ([(0, 3)], [(1, 4, 0)], False),  # one decode lane, a full-width chunk from scratch
        ([(0, 5), (3, 2), (4, 7)], [(1, 3, 2)], False),  # 3 decode rows in bucket 4, resumed
        ([(2, 4)], [(0, 4, 3), (4, 1, 5)], False),  # two prefill rows: 2x4
        ([(0, 2), (1, 3), (2, 4)], [(3, 2, 1), (4, 4, 0)], False),  # full pool: pads fall back
        # `SEED_AR_RMSNORM_FUSED`'s chained norm loop (on CPU each call is the unfused pair)
        ([(0, 5), (3, 2), (4, 7)], [(1, 3, 2)], True),
    ],
)
def test_replay_matches_separate_decode_and_prefill(
    checkpoint: Path,
    monkeypatch,
    decode,
    chunks,  # noqa: ANN001
    chained: bool,
) -> None:
    """Same decode logits, prefill logits, DeltaNet state and KV rows as one eager decode step
    followed by one eager packed prefill; lanes outside the step are untouched."""
    monkeypatch.setattr(graph_decode, "AR_RMSNORM_FUSED", chained)
    results = []
    for mode in ("separate", "graph"):
        model, runner = runner_for(checkpoint, monkeypatch)
        for slot in range(MAX_BATCH):
            runner.begin(slot)
        slots, tokens, positions = [], [], []
        for slot, prefix in decode:
            runner.prefill(slot, prompt_of(20 + slot, prefix), 0)
            slots.append(slot)
            tokens.append(prompt_of(40 + slot, 1)[0])
            positions.append(prefix)
        calls = []
        for slot, n, start in chunks:
            if start:
                runner.prefill(slot, prompt_of(10 + slot, start), 0)
            calls.append((slot, prompt_of(50 + slot, n), start))
        used = set(slots) | {c[0] for c in chunks}
        idle = [s for s in range(MAX_BATCH) if s not in used]
        for s in idle:
            runner.prefill(s, prompt_of(99 + s, 3), 0)
        before_idle = [lane_state(model, s, 3) for s in idle]
        if mode == "graph":
            shape = runner.mixed_runner.replayable(slots, positions, calls)
            assert shape is not None
            before = runner.mixed_runner.replays
            dec, pre = runner.decode_mixed(slots, tokens, positions, calls)
            assert runner.mixed_runner.replays == before + 1
        else:
            dec = model.decode(slots, tokens, positions)
            pre = model.prefill_batch(calls)
        for s, was in zip(idle, before_idle, strict=True):
            for a, b in zip(was, lane_state(model, s, 3), strict=True):
                assert torch.equal(a, b), "an idle lane's state changed"
        states = [lane_state(model, s, p + 1) for s, p in zip(slots, positions, strict=True)]
        states += [lane_state(model, slot, start + n) for slot, n, start in chunks]
        results.append((dec, pre, states))
    (want_dec, want_pre, want_states), (got_dec, got_pre, got_states) = results
    torch.testing.assert_close(got_dec.reshape(want_dec.shape), want_dec, atol=1e-4, rtol=1e-4)
    for w, g in zip(want_pre, got_pre, strict=True):
        torch.testing.assert_close(g, w, atol=1e-4, rtol=1e-4)
    for ws, gs in zip(want_states, got_states, strict=True):
        for w, g in zip(ws, gs, strict=True):
            torch.testing.assert_close(g, w, atol=1e-4, rtol=1e-4)


def test_steps_no_shape_holds_are_not_replayed(checkpoint: Path, monkeypatch) -> None:  # noqa: ANN001
    _, runner = runner_for(checkpoint, monkeypatch)
    mr = runner.mixed_runner
    assert mr.replayable([0], [3], [(1, [1] * 9, 0)]) is None  # wider than every shape
    assert mr.replayable([0], [3], [(0, [1], 0)]) is None  # a lane in both halves
    assert mr.replayable([0], [3], [(1, [1], 0), (2, [1], 0), (3, [1], 0)]) is None  # 3 rows
    assert mr.replayable([0], [3], []) is None
    assert mr.replayable([0], [3], [(1, [1] * 5, 0)]) == MixedShape(2, Shape(1, 8))
    fits, width, tokens = runner.mixed_fit(3)
    assert (width, tokens) == (8, 8) and fits([8]) and fits([4, 4]) and not fits([5, 1])
    assert runner.mixed_fit(5) is None  # no bucket holds 5 decode rows


@pytest.mark.parametrize("overlap", [False, True], ids=["serial", "overlap"])
def test_scheduler_multi_turn_matches_generate_with_flag_on_and_off(
    checkpoint: Path,
    monkeypatch,
    overlap: bool,  # noqa: ANN001
) -> None:
    """Greedy tokens across multi-turn sessions with stop tokens equal the one-sequence
    `generate` with mixed graphs on and off, and prefix reuse (the `end` events) is the same:
    a mixed step publishes the same boundaries and snapshots as a separate prefill.

    The snapshot and KV pools get room so nothing is evicted: mixing finishes requests in a
    different order than alternation, and with the CPU floor (one snapshot slot per lane) LRU
    eviction then drops a different prefix (the eager `SEED_MIXED_BATCH` path shows the same)."""
    import model as model_module

    monkeypatch.setattr(model_module, "KV_POOL_MIN_CAPACITY_FACTOR", 4.0)
    monkeypatch.setattr(model_module, "SNAPSHOT_POOL_MIN_CAPACITY_FACTOR", 8.0)
    ref = build(checkpoint, max_batch=1)
    specs = sessions(ref, random.Random(3))
    ends = {}
    for mixed in (False, True):
        model, runner = runner_for(checkpoint, monkeypatch)
        cache = SessionCache(model.block_allocator, model.block_size, model.num_snapshots)
        sched = Scheduler(runner, cache, prefill_chunk=4, overlap=overlap, mixed_graph=mixed)
        got = []
        for turn in range(3):
            sinks = []
            for k, sess in enumerate(specs):
                prompt, max_new, stop, suffix_len, _ = sess[turn]
                sink = Sink()
                sched.submit(Request(list(prompt), max_new, 0.0, stop, sink, suffix_len))
                sinks.append(sink)
                if k == 1:
                    sched.step()  # let two sessions start decoding before the rest arrive
                    sched.step()
            drain(sched)
            for sess, sink in zip(specs, sinks, strict=True):
                _, _, stop, _, reply = sess[turn]
                assert sink.error is None, sink.error
                assert sink.tokens == [t for t in reply if t not in stop], (mixed, turn)
                got.append(sink.end)
        if mixed:
            assert runner.mixed_runner.replays > 0, "no mixed step replayed"
        else:
            assert runner.mixed_runner.replays == 0
        ends[mixed] = got
    assert ends[True] == ends[False]


# ---------------------------------------------------------------- four ranks, real gloo group


@pytest.fixture(scope="module")
def tp_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from test_seed_parity import tiny_config
    from test_tensor_parallel import TP_AXES

    out = tmp_path_factory.mktemp("tp-mixed-graph")
    write_checkpoint(build_hf(cfg=tiny_config(**TP_AXES)), out, mxfp4=False)
    return out


def test_four_gloo_ranks_mixed_graphs_match_generate(
    tp_checkpoint: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Four processes, every rank replaying the same mixed shapes (the decision reads only the
    broadcast step) with rank 0 reserving both halves' blocks: tokens equal the unsharded
    `generate`, finish events equal the flag-off four-rank run. max_batch is 2, so every mixed
    step fills the pool and the decode bucket's padding row takes the fallback lane."""
    import subprocess

    from test_overlap_sched import seed_model_unsharded
    from test_tensor_parallel import WORLD, free_port
    from tp_gloo_rank import MAX_SEQ

    monkeypatch.setenv("SEED_MIXED_GRAPH_DECODE", "1,2")
    monkeypatch.setenv("SEED_MIXED_GRAPH_PREFILL", "1x4")
    monkeypatch.setenv("SEED_MIXED_GRAPH_TOTALS", "")
    ref = seed_model_unsharded(tp_checkpoint, MAX_SEQ)
    base = [prompt_of(930 + i, 3 + i) for i in range(3)]
    greedy = [list(ref.generate(p, 6, 0.0, frozenset())) for p in base]
    specs = [
        [base[0], 8, [], 0],
        [base[1], 6, [greedy[1][2]], 2],  # arrives while lane 0 decodes, stops mid-batch
        [base[2], 4, [], 3],  # waits for a lane (max_batch 2)
    ]
    want = [list(ref.generate(p, n, 0.0, frozenset(s))) for p, n, s, _ in specs]
    # Follow-up turn, arriving once both runs are idle: mixing changes how many steps the
    # first turns take, so an earlier arrival would see a different set of finished turns.
    specs.append([base[0] + want[0] + [9], 3, [], 200])
    want.append(list(ref.generate(specs[-1][0], 3, 0.0, frozenset())))
    specs_file = tmp_path / "specs.json"
    specs_file.write_text(json.dumps(specs))

    script = Path(__file__).resolve().parent / "overlap_gloo_rank.py"

    def run(flag: int) -> list:
        out = tmp_path / f"out-{flag}.json"
        base_cmd = [sys.executable, str(script), "--checkpoint", str(tp_checkpoint)]
        base_cmd += ["--world", str(WORLD), "--port", str(free_port()), "--specs", str(specs_file)]
        base_cmd += ["--overlap", "1", "--mixed-graph", str(flag)]
        procs = [
            subprocess.Popen(
                [*base_cmd, "--rank", str(r), *(["--out", str(out)] if r == 0 else [])]
            )  # noqa: S603
            for r in range(WORLD)
        ]
        try:
            codes = [p.wait(timeout=900) for p in procs]
        finally:
            for p in procs:
                if p.poll() is None:
                    p.kill()
        assert codes == [0] * WORLD, f"rank exit codes {codes}"
        return json.loads(out.read_text())

    on, off = run(1), run(0)
    replays = on.pop()
    assert replays > 0, "no mixed step replayed under TP"
    for i, ((_, _, stop, _), ref_tokens) in enumerate(zip(specs, want, strict=True)):
        streamed = [t for t in ref_tokens if t not in stop]
        assert off[i][0] == streamed, f"req {i}: flag-off four-rank run diverged"
        assert on[i][0] == streamed, f"req {i}: flag-on four-rank run diverged"
        assert on[i][1] == off[i][1], f"req {i}: finish event (reuse) differs"

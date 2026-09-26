"""CPU tests for `SEED_PREFILL_GRAPHS` (graph_prefill.py): captured small-prefill shapes must
produce the eager prefill's logits and state, and a scheduler over them the eager tokens.

The capture backend is test_graph_capture.py's `EagerBackend` (a "replay" re-runs the static
step on the same buffers), so these check the static step's arithmetic and buffer handling;
whether it captures on the device is `prepare`'s own startup validation on the GPU.

    <python-with-torch> -m pytest seed_tests/test_graph_prefill.py -q -o addopts=
"""

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import graph_decode  # noqa: E402
import graph_prefill  # noqa: E402
from graph_decode import GraphDecodeRunner  # noqa: E402
from graph_prefill import Shape, parse_shapes, shape_for  # noqa: E402
from scheduler import Request, Scheduler  # noqa: E402
from session_cache import SessionCache  # noqa: E402
from test_batched_decode import build, prompt_of  # noqa: E402
from test_graph_capture import CorruptingBackend, EagerBackend  # noqa: E402
from test_scheduler import Sink, drain  # noqa: E402
from test_seed_parity import build_hf, write_checkpoint  # noqa: E402

SHAPES = "1x4,2x4,3x8"


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("tiny-prefill-graph")
    write_checkpoint(build_hf(), out, mxfp4=False)
    return out


def test_flag_defaults_to_off() -> None:
    assert graph_prefill.PREFILL_GRAPHS is False


def test_parse_shapes_sorts_by_area_and_drops_unusable() -> None:
    got = parse_shapes("4x64, 1x16,2x16,9x16,1x999", max_batch=8, max_seq=128)
    assert got == [Shape(1, 16), Shape(2, 16), Shape(4, 64)]
    with pytest.raises(ValueError, match="bad entry"):
        parse_shapes("4by64", 8, 128)


def test_shape_for_picks_the_smallest_area_that_holds_the_call() -> None:
    shapes = parse_shapes(graph_prefill.DEFAULT_SHAPES, 48, 4096)
    assert shape_for([5], shapes) == Shape(1, 16)
    assert shape_for([5, 16, 3], shapes) == Shape(4, 16)
    assert shape_for([17], shapes) == Shape(1, 64)
    assert shape_for([17, 1, 1], shapes) == Shape(4, 64)
    assert shape_for([100, 120], shapes) == Shape(2, 128)
    assert shape_for([200], shapes) == Shape(1, 256)
    assert shape_for([300], shapes) == Shape(1, 384)
    assert shape_for([385], shapes) is None
    assert shape_for([1] * 9, shapes) is None


def runner_for(checkpoint: Path, monkeypatch: pytest.MonkeyPatch, backend=None, max_batch=3):  # noqa: ANN001, ANN201
    monkeypatch.setenv("SEED_PREFILL_GRAPH_SHAPES", SHAPES)
    model = build(checkpoint, max_batch=max_batch)
    runner = GraphDecodeRunner(model, backend=backend or EagerBackend(), prefill_graphs=True)
    runner.prepare()
    return model, runner


def test_prepare_captures_and_validates_every_shape(checkpoint: Path, monkeypatch) -> None:  # noqa: ANN001
    _, runner = runner_for(checkpoint, monkeypatch)
    assert runner.prefill_runner.enabled
    assert sorted(runner.prefill_runner.graphs) == sorted(parse_shapes(SHAPES, 3, 96))


def test_a_replay_that_computes_nothing_is_rejected(checkpoint: Path, monkeypatch) -> None:  # noqa: ANN001
    _, runner = runner_for(checkpoint, monkeypatch, backend=CorruptingBackend())
    assert not runner.prefill_runner.enabled
    assert runner.prefill_runner.replayable([(0, [1, 2], 0)]) is None


def peer_rejects(monkeypatch, runner_cls, target) -> None:  # noqa: ANN001
    """Make the vote right after `runner_cls._check(target)` come back False, as if another
    rank's check of that shape failed while this rank's passed."""
    real_check, real_vote = runner_cls._check, graph_decode.all_ranks
    last = {}

    def check(self, shape):  # noqa: ANN001, ANN202
        last["shape"] = shape
        return real_check(self, shape)

    def vote(tp, ok):  # noqa: ANN001, ANN202
        return real_vote(tp, ok) and last.pop("shape", None) != target

    monkeypatch.setattr(runner_cls, "_check", check)
    monkeypatch.setattr(graph_decode, "all_ranks", vote)


def test_a_shape_one_rank_rejects_is_dropped_on_every_rank(checkpoint: Path, monkeypatch) -> None:  # noqa: ANN001
    """The verdict per shape is agreed before it is acted on: a shape another rank rejects is
    off here too (this rank passed it), and validation goes on to the next shape instead of
    returning while the peers are still in a replay's all-reduce."""
    peer_rejects(monkeypatch, graph_prefill.PrefillGraphRunner, Shape(2, 4))
    _, runner = runner_for(checkpoint, monkeypatch)
    pr = runner.prefill_runner
    assert pr.enabled
    assert pr.shapes == [Shape(1, 4), Shape(3, 8)]
    assert sorted(pr.graphs) == [Shape(1, 4), Shape(3, 8)]
    assert pr.replayable([(0, [1, 2], 0), (1, [3], 0)]) == Shape(3, 8)


def drifting_step(monkeypatch, perturb) -> None:  # noqa: ANN001
    """Make the static step (captured and uncaptured alike) post-process its logits with
    `perturb`, so replay and uncaptured step still agree and only the eager check can fail."""
    real = graph_prefill.prefill_step

    def step_for(model, buf):  # noqa: ANN001, ANN202
        inner = real(model, buf)

        def step() -> None:
            inner()
            buf.out.copy_(perturb(buf.out))

        return step

    monkeypatch.setattr(graph_prefill, "prefill_step", step_for)


def test_rounding_drift_against_eager_is_accepted(checkpoint: Path, monkeypatch) -> None:  # noqa: ANN001
    """A 3% max-abs gap (what 60 bf16 layers of different kernels give on the GPU, see
    `EAGER_MIN_CORR`) used to fail the 2% gate and disable every shape."""

    def drift(out: torch.Tensor) -> torch.Tensor:
        idx = torch.arange(out.shape[-1], dtype=out.dtype)
        return out + 0.03 * out.abs().amax(-1, keepdim=True) * torch.sin(idx)

    drifting_step(monkeypatch, drift)
    _, runner = runner_for(checkpoint, monkeypatch)
    assert runner.prefill_runner.enabled


def test_decorrelated_logits_are_rejected(checkpoint: Path, monkeypatch) -> None:  # noqa: ANN001
    """What a wrong mask, position or lane looks like: logits unrelated to the eager ones."""
    drifting_step(monkeypatch, lambda out: out.roll(7, dims=-1))
    _, runner = runner_for(checkpoint, monkeypatch)
    assert not runner.prefill_runner.enabled


def test_corr_and_top_agree() -> None:
    a = torch.tensor([0.0, 1.0, 5.0, 2.0])
    assert graph_prefill._corr(a, a) == pytest.approx(1.0)
    assert graph_prefill._corr(a, -a) == pytest.approx(-1.0)
    assert graph_prefill._corr(torch.zeros(3), torch.zeros(3)) == 1.0
    assert graph_prefill._corr(torch.zeros(4), a) == 0.0
    assert graph_prefill._top_agree(a, torch.tensor([0.0, 1.0, 4.9, 5.0]))
    wide = torch.arange(10.0)
    assert not graph_prefill._top_agree(wide, wide.flip(0))


def test_gemm_widths_are_the_shape_areas(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("SEED_PREFILL_GRAPH_SHAPES", "1x16,2x16,4x16,1x64,2x64")
    monkeypatch.setattr(graph_prefill, "PREFILL_GRAPHS", True)
    assert graph_prefill.gemm_widths(48, 4096) == (16, 32, 64, 128)
    monkeypatch.setattr(graph_prefill, "TUNE", False)
    assert graph_prefill.gemm_widths(48, 4096) == ()


def test_prefill_sp_keeps_only_a_residual_shard_and_chains_full_norms(monkeypatch) -> None:  # noqa: ANN001
    """The prefill SP loop has graph_mixed's shape contract and matches full residual math."""
    hidden, rows, width = 4, 2, 2
    embed = torch.arange(32, dtype=torch.float32).reshape(8, hidden) / 16
    layers = [
        {"in_norm": torch.ones(hidden), "post_norm": torch.ones(hidden)} for _ in range(2)
    ]

    class FakeTP:
        rank, world, custom_reduce = 0, 4, None

        @staticmethod
        def all_reduce_residual_norm(partial, residual, weight, eps, norm):  # noqa: ANN001
            updated = residual + partial
            return updated, norm(updated, weight, eps)

    model = SimpleNamespace(
        cfg=SimpleNamespace(hidden=hidden, eps=1e-6, layer_types=["full_attention"] * 2),
        embed=embed,
        layers=layers,
        final_norm=torch.ones(hidden),
        tp=FakeTP(),
        moe=lambda _i, h: h * 0.25,
        unembed=lambda h: h,
    )
    buf = SimpleNamespace(
        tokens=torch.tensor([[1, 2], [3, 4]]),
        length=torch.tensor([2, 1]),
        out=torch.zeros(rows, hidden),
    )
    monkeypatch.setattr(graph_decode, "AR_RMSNORM_FUSED", True)
    monkeypatch.setattr(graph_prefill, "attn_prefill_static", lambda _m, i, h, _b: h * (i + 1))
    monkeypatch.setattr(graph_prefill, "_sp_rows_ok", lambda _m, _rows: False)
    graph_prefill.prefill_step(model, buf)()
    expected = buf.out.clone()

    class FakeCR:
        def __init__(self) -> None:
            self.full = torch.nn.functional.embedding(buf.tokens, embed).reshape(rows * width, hidden)
            self.calls = 0

        def sp_ar_add_rmsnorm(self, partial, shard, weight, eps):  # noqa: ANN001
            assert partial.shape == (rows * width, hidden)
            assert shard.shape == (rows * width // 4, hidden)
            assert torch.equal(shard, self.full[: rows * width // 4])
            self.full = self.full + partial
            self.calls += 1
            return self.full[: rows * width // 4].clone(), graph_prefill.rmsnorm(
                self.full, weight, eps
            )

    cr = FakeCR()
    model.tp.custom_reduce = cr
    monkeypatch.setattr(graph_prefill, "_sp_rows_ok", lambda _m, _rows: True)
    buf.out.zero_()
    graph_prefill.prefill_step(model, buf)()
    assert cr.calls == 2 * len(layers)
    assert torch.equal(buf.out, expected)
    monkeypatch.setattr(graph_prefill, "TUNE", True)
    monkeypatch.setattr(graph_prefill, "PREFILL_GRAPHS", False)
    assert graph_prefill.gemm_widths(48, 4096) == ()


def test_fused_in_proj_prefill_matches_split(checkpoint: Path, monkeypatch) -> None:  # noqa: ANN001
    """`SEED_PREFILL_FUSED_IN_PROJ`: eager prefill through `in_proj_all` equals the four GEMMs."""
    import model as model_module

    out = []
    for fused in (False, True):
        monkeypatch.setattr(model_module, "PREFILL_FUSED_IN_PROJ", fused)
        model = build(checkpoint, max_batch=2)
        assert "in_proj_all" in next(w for w in model.layers if "in_proj_qkv" in w)
        model.begin(0)
        first = model.prefill(0, prompt_of(3, 7), 0)
        second = model.prefill(0, prompt_of(4, 5), 7)
        out.append((first, second, lane_state(model, 0, 12)))
    (a1, a2, sa), (b1, b2, sb) = out
    torch.testing.assert_close(b1, a1, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(b2, a2, atol=1e-5, rtol=1e-5)
    for x, y in zip(sa, sb, strict=True):
        torch.testing.assert_close(y, x, atol=1e-5, rtol=1e-5)


def lane_state(model, slot: int, upto: int) -> list[torch.Tensor]:  # noqa: ANN001
    """Lane `slot`'s DeltaNet state and its first `upto` KV rows, every layer."""
    out = []
    for i, pool in enumerate(model.pool):
        if "conv" in pool:
            out += [pool["conv"][slot].clone(), pool["rec"][slot].clone()]
        else:
            rows = model._physical_rows_range(slot, 0, upto, pool["k"].device)
            out += [pool["k"][rows].clone(), pool["v"][rows].clone()]
        del i
    return out


@pytest.mark.parametrize(
    "chunks",
    [
        [(0, 4, 0)],  # one full-width row from scratch
        [(0, 1, 5), (1, 3, 2)],  # resumed rows, shorter than the width, no padding row
        [(2, 7, 9), (0, 1, 4)],  # a 3x8 shape with one padding row, a lane > row index
    ],
)
def test_replay_matches_the_eager_prefill(checkpoint: Path, monkeypatch, chunks) -> None:  # noqa: ANN001
    """Same logits, DeltaNet state and KV rows as the eager call, from a resumed prefix; the
    lanes not in the call (padding rows' filler lane included) are untouched."""
    results = []
    for graph in (False, True):
        model, runner = runner_for(checkpoint, monkeypatch)
        for slot in range(3):
            runner.begin(slot)
        calls = []
        for slot, n, start in chunks:
            if start:
                runner.prefill(slot, prompt_of(10 + slot, start), 0)
            calls.append((slot, prompt_of(50 + slot, n), start))
        idle = [s for s in range(3) if s not in {c[0] for c in chunks}]
        if idle:
            runner.prefill(idle[0], prompt_of(99, 3), 0)
        before_idle = [lane_state(model, s, 3) for s in idle[:1]]
        if graph:
            assert runner._prefill_shape(calls) is not None
            before = runner.prefill_runner.replays
            logits = runner.prefill_batch(calls)
            assert runner.prefill_runner.replays == before + 1
        else:
            runner.prefill_runner.enabled = False
            logits = runner.prefill_batch(calls)
        states = [lane_state(model, slot, start + n) for slot, n, start in chunks]
        for s, before in zip(idle[:1], before_idle, strict=True):
            for a, b in zip(before, lane_state(model, s, 3), strict=True):
                assert torch.equal(a, b), "an idle lane's state changed"
        results.append((logits, states))
    (want, want_states), (got, got_states) = results
    for w, g in zip(want, got, strict=True):
        torch.testing.assert_close(g, w, atol=1e-4, rtol=1e-4)
    for ws, gs in zip(want_states, got_states, strict=True):
        for w, g in zip(ws, gs, strict=True):
            torch.testing.assert_close(g, w, atol=1e-4, rtol=1e-4)


def test_calls_no_shape_holds_stay_eager(checkpoint: Path, monkeypatch) -> None:  # noqa: ANN001
    _, runner = runner_for(checkpoint, monkeypatch)
    pr = runner.prefill_runner
    assert pr.replayable([(0, [1] * 9, 0)]) is None  # wider than every shape
    assert pr.replayable([(0, [1], 0), (0, [2], 1)]) is None  # a lane twice
    assert pr.replayable([(0, [1] * 4, 94)]) is None  # past max_seq
    assert pr.replayable([(0, [1] * 5, 0), (1, [1], 0)]) == Shape(3, 8)


def sessions(ref, rng: random.Random):  # noqa: ANN001, ANN201
    """Two raw and two chat sessions, three turns each, with stop tokens; prompts short enough
    that most chunks fit a shape (prefill_chunk 4 splits the longer ones)."""
    out = []
    for s in range(4):
        chat = s % 2 == 1
        suffix = [7, 8] if chat else []
        history = prompt_of(700 + s, 3 + s)
        specs = []
        for _ in range(3):
            prompt = history + suffix
            greedy = list(ref.generate(prompt, 5, 0.0, frozenset()))
            stop = frozenset({greedy[rng.randint(1, 3)]}) if rng.random() < 0.6 else frozenset()
            reply = list(ref.generate(prompt, 5, 0.0, stop))
            specs.append((prompt, 5, stop, len(suffix), reply))
            history = history + ([3] if chat else []) + reply + prompt_of(rng.randint(1, 99), 2)
        out.append(specs)
    return out


@pytest.mark.parametrize("overlap", [False, True], ids=["serial", "overlap"])
def test_scheduler_multi_turn_matches_generate_with_flags_on_and_off(
    checkpoint: Path,
    monkeypatch,
    overlap: bool,  # noqa: ANN001
) -> None:
    """Greedy tokens across multi-turn sessions with stop tokens equal the one-sequence
    `generate`, with prefill graphs and turn-close deferral each on and off; with both off vs
    prefill graphs alone, prefix reuse is identical too (graphs change no scheduling)."""
    ref = build(checkpoint, max_batch=1)
    specs = sessions(ref, random.Random(1))
    ends = {}
    for graphs in (False, True):
        for defer in (False, True):
            model, runner = runner_for(checkpoint, monkeypatch)
            if not graphs:
                runner.prefill_runner = None
            cache = SessionCache(model.block_allocator, model.block_size, model.num_snapshots)
            sched = Scheduler(
                runner, cache, prefill_chunk=4, overlap=overlap, defer_turn_close=defer
            )
            got = []
            for turn in range(3):
                sinks = []
                for sess in specs:
                    prompt, max_new, stop, suffix_len, _ = sess[turn]
                    sink = Sink()
                    sched.submit(Request(list(prompt), max_new, 0.0, stop, sink, suffix_len))
                    sinks.append(sink)
                drain(sched)
                for sess, sink in zip(specs, sinks, strict=True):
                    _, _, stop, _, reply = sess[turn]
                    assert sink.error is None, sink.error
                    assert sink.tokens == [t for t in reply if t not in stop], (graphs, defer)
                    got.append(sink.end)
            if graphs:
                assert runner.prefill_runner.replays > 0, "no prefill call replayed"
            ends[(graphs, defer)] = got
    assert ends[(True, False)] == ends[(False, False)]
    assert ends[(True, True)] == ends[(False, True)]


# ---------------------------------------------------------------- four ranks, real gloo group


@pytest.fixture(scope="module")
def tp_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from test_seed_parity import tiny_config
    from test_tensor_parallel import TP_AXES

    out = tmp_path_factory.mktemp("tp-prefill-graph")
    write_checkpoint(build_hf(cfg=tiny_config(**TP_AXES)), out, mxfp4=False)
    return out


def test_four_gloo_ranks_prefill_graphs_and_deferral_match_generate(
    tp_checkpoint: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Four processes, every rank replaying the same captured prefill shapes (the decision
    reads only the broadcast call list) with rank 0 reserving blocks: tokens equal the
    unsharded `generate`, finish events equal the flags-off four-rank run."""
    import subprocess

    from test_overlap_sched import seed_model_unsharded
    from test_tensor_parallel import WORLD, free_port
    from tp_gloo_rank import MAX_SEQ

    monkeypatch.setenv("SEED_PREFILL_GRAPH_SHAPES", "1x4,2x4")
    ref = seed_model_unsharded(tp_checkpoint, MAX_SEQ)
    base = [prompt_of(930 + i, 3 + i) for i in range(3)]
    greedy = [list(ref.generate(p, 6, 0.0, frozenset())) for p in base]
    specs = [
        [base[0], 5, [], 0],
        [base[1], 6, [greedy[1][2]], 0],  # stops mid-batch
        [base[2], 4, [], 2],  # waits for a lane (max_batch 2)
    ]
    want = [list(ref.generate(p, n, 0.0, frozenset(s))) for p, n, s, _ in specs]
    specs.append([base[0] + want[0] + [9], 3, [], 40])  # follow-up turn
    want.append(list(ref.generate(specs[-1][0], 3, 0.0, frozenset())))
    specs_file = tmp_path / "specs.json"
    specs_file.write_text(json.dumps(specs))

    script = Path(__file__).resolve().parent / "overlap_gloo_rank.py"

    def run(flags: int) -> list:
        out = tmp_path / f"out-{flags}.json"
        base_cmd = [sys.executable, str(script), "--checkpoint", str(tp_checkpoint)]
        base_cmd += ["--world", str(WORLD), "--port", str(free_port()), "--specs", str(specs_file)]
        base_cmd += ["--overlap", "1", "--prefill-graphs", str(flags), "--defer", str(flags)]
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
    assert replays > 0, "no prefill call replayed under TP"
    for i, ((_, _, stop, _), ref_tokens) in enumerate(zip(specs, want, strict=True)):
        streamed = [t for t in ref_tokens if t not in stop]
        assert off[i][0] == streamed, f"req {i}: flags-off four-rank run diverged"
        assert on[i][0] == streamed, f"req {i}: flags-on four-rank run diverged"
        # ["end", [reason, generated, reused]]; reuse differs by the deferred closing token
        assert on[i][1][0] == off[i][1][0] and on[i][1][1][:2] == off[i][1][1][:2], i

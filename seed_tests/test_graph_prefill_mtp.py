"""CPU tests for captured prefill under MTP (`SEED_PREFILL_GRAPHS` with `SEED_MTP_SERVE`):
the captured step's MTP tail must leave the MTP KV and `hidden_scratch` as the eager MTP
prefill does, touch no other lane, and a scheduler serving MTP rounds over captured prefill
(with `SEED_PREFILL_ACCUM`) must stream the plain greedy tokens.

The capture backend re-runs the static step (`EagerBackend`), so these check the step's
arithmetic and buffer handling; on the GPU, `PrefillGraphRunner._check` validates each captured
shape (tail included) at boot on every rank.

    /tmp/torchenv/bin/python -m pytest seed_tests/test_graph_prefill_mtp.py -q -o addopts=
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import graph_mtp  # noqa: E402
import graph_prefill  # noqa: E402
from graph_decode import GraphDecodeRunner  # noqa: E402
from scheduler import Scheduler  # noqa: E402
from session_cache import SessionCache  # noqa: E402
from test_graph_capture import CorruptingBackend, EagerBackend  # noqa: E402
from test_graph_prefill import lane_state  # noqa: E402
from test_mtp import prompt_of  # noqa: E402
from test_mtp_serve import (  # noqa: E402
    mtp_model,
    patch_eager_draft,
    patch_graph_draft,
    plain_model,
    resolve_plan,
    run_conversations,
)
from tp_driver import pool_handshake  # noqa: E402

SHAPES = "1x4,2x4,3x8"


@pytest.fixture(autouse=True)
def roomy_kv_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    import model as seed_model

    monkeypatch.setattr(seed_model, "KV_POOL_MIN_CAPACITY_FACTOR", 16.0)
    monkeypatch.setattr(seed_model, "SNAPSHOT_POOL_MIN_CAPACITY_FACTOR", 16.0)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from test_mtp import write_mtp_shard
    from test_seed_parity import build_hf, tiny_config, write_checkpoint

    out = tmp_path_factory.mktemp("tiny-prefill-mtp")
    cfg = tiny_config()
    write_checkpoint(build_hf(cfg=cfg), out, mxfp4=False)
    write_mtp_shard(cfg, out)
    return out


def runner_for(checkpoint: Path, monkeypatch: pytest.MonkeyPatch, backend=None):  # noqa: ANN001, ANN201
    monkeypatch.setenv("SEED_PREFILL_GRAPH_SHAPES", SHAPES)
    model = mtp_model(checkpoint, 3)
    runner = GraphDecodeRunner(model, backend=backend or EagerBackend(), prefill_graphs=True)
    runner.prepare()
    return model, runner


def mtp_state(model, slot: int, upto: int) -> list[torch.Tensor]:  # noqa: ANN001
    rows = model._physical_rows_range(slot, 0, upto, model.hidden_scratch.device)
    return [
        model.hidden_scratch[slot].clone(),
        model.mtp.pool["k"][rows].clone(),
        model.mtp.pool["v"][rows].clone(),
    ]


def test_full_residual_reassembles_rank_shards() -> None:
    """Row-sharded residual: the all-gather along the last dim, reshaped, is the full
    `[rows, hidden]` in rank order, identical on every rank."""
    world, rows, hidden = 4, 8, 3
    full = torch.arange(rows * hidden, dtype=torch.float32).view(rows, hidden)
    shards = full.view(world, rows // world, hidden)

    for rank in range(world):
        tp = SimpleNamespace(
            rank=rank,
            world=world,
            all_gather_last=lambda x: torch.cat(list(shards), dim=-1),
        )
        model = SimpleNamespace(tp=tp)
        got = graph_prefill._full_residual(model, shards[rank], rows, True)
        assert torch.equal(got, full)
    model = SimpleNamespace(tp=None)
    assert torch.equal(
        graph_prefill._full_residual(model, full.view(2, 4, hidden), rows, False), full
    )


def test_prepare_captures_every_shape_under_mtp(checkpoint: Path, monkeypatch) -> None:  # noqa: ANN001
    model, runner = runner_for(checkpoint, monkeypatch)
    assert model.mtp is not None
    ok, why = runner.prefill_runner.supported()
    assert ok, why
    assert runner.prefill_runner.enabled
    assert sorted(runner.prefill_runner.graphs) == sorted(graph_prefill.parse_shapes(SHAPES, 3, 96))


def test_a_replay_without_the_mtp_tail_is_rejected(checkpoint: Path, monkeypatch) -> None:  # noqa: ANN001
    """Validation catches a captured step that skips the tail: the MTP KV rows are zeroed on
    rewind, so the replay cannot pass on the eager run's leftovers."""
    real_batch, real_tail = graph_prefill.PrefillGraphRunner.prefill_batch, graph_prefill._mtp_tail
    replaying = [False]

    def prefill_batch(self, shape, calls):  # noqa: ANN001, ANN202
        replaying[0] = True
        try:
            return real_batch(self, shape, calls)
        finally:
            replaying[0] = False

    def tail(*args):  # noqa: ANN002, ANN202
        if not replaying[0]:
            real_tail(*args)

    monkeypatch.setattr(graph_prefill.PrefillGraphRunner, "prefill_batch", prefill_batch)
    monkeypatch.setattr(graph_prefill, "_mtp_tail", tail)
    _, runner = runner_for(checkpoint, monkeypatch)
    assert not runner.prefill_runner.enabled


def test_a_corrupt_replay_is_rejected_under_mtp(checkpoint: Path, monkeypatch) -> None:  # noqa: ANN001
    _, runner = runner_for(checkpoint, monkeypatch, backend=CorruptingBackend())
    assert not runner.prefill_runner.enabled


@pytest.mark.parametrize(
    "chunks",
    [
        [(0, 4, 0)],  # one full-width row from scratch (zero seed)
        [(0, 1, 5), (1, 3, 2)],  # resumed rows (seed from the prefix), no padding row
        [(2, 7, 9), (0, 1, 4)],  # a 3x8 shape with one padding row, a lane > row index
    ],
)
def test_replay_matches_the_eager_mtp_prefill(checkpoint: Path, monkeypatch, chunks) -> None:  # noqa: ANN001
    """Same logits, DeltaNet state, target KV, MTP KV and `hidden_scratch` as the eager MTP
    prefill, from a resumed prefix; the idle lane (a padding row's filler) is untouched."""
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
        before_idle = [lane_state(model, s, 3) + mtp_state(model, s, 3) for s in idle[:1]]
        if graph:
            assert runner._prefill_shape(calls) is not None
            before = runner.prefill_runner.replays
            logits = runner.prefill_batch(calls)
            assert runner.prefill_runner.replays == before + 1
        else:
            runner.prefill_runner.enabled = False
            logits = runner.prefill_batch(calls)
        states = [
            lane_state(model, slot, start + n) + mtp_state(model, slot, start + n)
            for slot, n, start in chunks
        ]
        for s, before in zip(idle[:1], before_idle, strict=True):
            after = lane_state(model, s, 3) + mtp_state(model, s, 3)
            for a, b in zip(before, after, strict=True):
                assert torch.equal(a, b), "an idle lane's state changed"
        results.append((logits, states))
    (want, want_states), (got, got_states) = results
    for w, g in zip(want, got, strict=True):
        torch.testing.assert_close(g, w, atol=1e-4, rtol=1e-4)
    for ws, gs in zip(want_states, got_states, strict=True):
        for w, g in zip(ws, gs, strict=True):
            assert w.abs().sum() > 0 or g.abs().sum() == 0
            torch.testing.assert_close(g, w, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("accum", [False, True])
def test_mtp_serving_over_captured_prefill_matches_greedy(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch, accum: bool
) -> None:
    """Multi-turn MTP serving with every fitting prefill replayed (and, with `accum`, held
    by `SEED_PREFILL_ACCUM`) streams the plain greedy tokens and finish events."""
    monkeypatch.setenv("SEED_PREFILL_GRAPH_SHAPES", SHAPES)
    plan, oracle = resolve_plan(checkpoint)
    plain = plain_model(checkpoint, 2)
    pool_handshake(plain)
    want = run_conversations(
        Scheduler(
            plain,
            SessionCache(plain.block_allocator, plain.block_size, plain.num_snapshots),
            prefill_chunk=4,
            spec_decode=False,
        ),
        plan,
    )

    patch_eager_draft(monkeypatch, oracle)
    model = mtp_model(checkpoint, 2)
    runner = graph_mtp.GraphMTPRunner(model, backend=EagerBackend())
    runner.want_prefill_graphs = True
    runner.prepare()
    assert runner.mtp_runner.enabled and runner.prefill_runner.enabled
    patch_graph_draft(runner.mtp_runner, oracle)
    pool_handshake(model)
    cache = SessionCache(model.block_allocator, model.block_size, model.num_snapshots)
    clock = [0.0]
    sched = Scheduler(
        runner,
        cache,
        prefill_chunk=4,
        spec_decode=True,
        batched_prefill=True,
        prefill_fit_graph=True,
        prefill_accum=accum,
        prefill_accum_min_decode=1,
        prefill_accum_max_wait_ms=0.0,
        clock=lambda: clock[0],
    )
    assert sched.prefill_accum is accum
    got = run_conversations(sched, plan)
    assert got == want
    assert runner.prefill_runner.replays > 0, "no prefill replayed a captured shape"


def _packed_layout(lens: list[int], starts: list[int], lanes: list[int], area: int, align: int):  # noqa: ANN202
    """`PrefillGraphRunner._fill_packed`'s stream layout, host side."""
    tok_lane, tok_off, tok_pos, real = [0] * area, [1 << 30] * area, [0] * area, [False] * area
    seg_last, at = [], 0
    for n, start, lane in zip(lens, starts, lanes, strict=True):
        span = -(-n // align) * align
        for o in range(span):
            tok_lane[at + o], tok_off[at + o] = lane, o
            tok_pos[at + o], real[at + o] = start + min(o, n - 1), o < n
        seg_last.append(at + n - 1)
        at += span
    return tok_lane, tok_off, tok_pos, real, seg_last


@settings(max_examples=60, deadline=None)
@given(
    segs=st.lists(st.tuples(st.integers(1, 9), st.integers(0, 20)), min_size=1, max_size=4),
    spare=st.integers(0, 2),
)
def test_packed_mtp_tail_matches_per_segment_prefill(segs, spare) -> None:  # noqa: ANN001
    """Regression (round 15): packed captured prefill shapes (`SEED_PREFILL_PACK`) ran the
    per-row MTP tail on the packed stream and failed boot validation, so MTP boots lost every
    packed shape. Per segment, the packed tail must feed `cache_target_rows` what an eager
    per-sequence MTP prefill feeds it: previous residual = the lane's seed at the segment's
    first token, the stream's previous token after; and leave the residual at the segment's
    last real token in the lane's `hidden_scratch` row."""
    align, hidden, lanes_n = 4, 3, 8
    lens = [n for n, _ in segs]
    starts = [s for _, s in segs]
    lanes = list(range(1, len(segs) + 1))
    used = sum(-(-n // align) * align for n in lens)
    area = used + spare * align
    tok_lane, tok_off, tok_pos, real, seg_last = _packed_layout(lens, starts, lanes, area, align)
    n_segs = len(segs) + 1  # one idle segment slot
    seg_lanes = [*lanes, 0]
    seg_active = [True] * len(segs) + [False]
    seg_last = [*seg_last, 0]
    hs = torch.randn(lanes_n, hidden)
    x = torch.randn(area, hidden)
    tokens = torch.randint(0, 50, (area,))
    buf = SimpleNamespace(
        packed=True, shape=SimpleNamespace(area=area, rows=1, width=area),
        tokens=tokens.view(1, area), q_pos=torch.tensor(tok_pos), write_rows=torch.arange(area) + 100,
        real=torch.tensor(real).view(1, area), tok_lane=torch.tensor(tok_lane),
        tok_off=torch.tensor(tok_off, dtype=torch.int32), seg_last=torch.tensor(seg_last),
        seg_lanes=torch.tensor(seg_lanes), seg_active=torch.tensor(seg_active),
    )
    assert len(seg_lanes) == n_segs
    model = SimpleNamespace(hidden_scratch=hs.clone(), mtp=object())
    seen = {}

    def record(_m, _mtp, tok, prev, pos, rows, keep, embedding_out=None):  # noqa: ANN001, ANN202, ARG001
        seen.update(tok=tok.clone(), prev=prev.clone(), pos=pos.clone(), rows=rows.clone(), keep=keep.clone())

    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(graph_prefill.mtp_mod, "cache_target_rows", record)
        graph_prefill._mtp_tail(model, buf, x)
    finally:
        mp.undo()
    keep = seen["keep"].view(-1)
    assert keep.tolist() == real
    at = 0
    for n, start, lane in zip(lens, starts, lanes, strict=True):
        want_prev = torch.cat([hs[lane][None], x[at : at + n - 1]])
        assert torch.equal(seen["prev"].view(area, hidden)[at : at + n], want_prev)
        assert seen["pos"].view(-1)[at : at + n].tolist() == list(range(start, start + n))
        assert torch.equal(seen["tok"].view(-1)[at : at + n], tokens[at : at + n])
        assert torch.equal(model.hidden_scratch[lane], x[at + n - 1])
        at += -(-n // align) * align
    untouched = [j for j in range(lanes_n) if j not in lanes]
    assert torch.equal(model.hidden_scratch[untouched], hs[untouched])

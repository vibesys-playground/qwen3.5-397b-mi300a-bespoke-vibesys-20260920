"""CPU tests for overlap scheduling of MTP rounds (`SEED_OVERLAP_SCHED` with `SEED_MTP_SERVE`,
mtp_overlap.py): a scheduler that launches round N+1 before reading round N must stream the
plain greedy tokens and finish events, with stops and budgets landing inside accepted drafts
(so some lanes finish in round N while round N+1 is already launched with them as lookahead
rows).

    /tmp/torchenv/bin/python -m pytest seed_tests/test_mtp_overlap.py -q -o addopts=
"""

import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mtp_overlap  # noqa: E402
import scheduler as sched_mod  # noqa: E402
from model import LOOKAHEAD_TOKEN  # noqa: E402
from scheduler import Scheduler  # noqa: E402
from session_cache import SessionCache  # noqa: E402
from test_graph_capture import EagerBackend  # noqa: E402
from test_mtp_forced_drafts import FakeSpecRunner, K  # noqa: E402
from test_mtp_serve import (  # noqa: E402
    mtp_model,
    patch_eager_draft,
    patch_graph_draft,
    plain_model,
    resolve_plan,
    run_conversations,
)
from test_scheduler import drain, make, request, serial  # noqa: E402
from tp_driver import pool_handshake  # noqa: E402


@pytest.fixture(autouse=True)
def roomy_kv_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    import model as seed_model

    monkeypatch.setattr(seed_model, "KV_POOL_MIN_CAPACITY_FACTOR", 16.0)
    monkeypatch.setattr(seed_model, "SNAPSHOT_POOL_MIN_CAPACITY_FACTOR", 16.0)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from test_mtp import write_mtp_shard
    from test_seed_parity import build_hf, tiny_config, write_checkpoint

    out = tmp_path_factory.mktemp("tiny-mtp-overlap")
    cfg = tiny_config()
    write_checkpoint(build_hf(cfg=cfg), out, mxfp4=False)
    write_mtp_shard(cfg, out)
    return out


def plain_reference(checkpoint: Path, plan: list) -> dict:
    plain = plain_model(checkpoint, 3)
    pool_handshake(plain)
    cache = SessionCache(plain.block_allocator, plain.block_size, plain.num_snapshots)
    return run_conversations(Scheduler(plain, cache, prefill_chunk=4, spec_decode=False), plan)


@pytest.mark.parametrize("prefill_graphs", [False, True])
def test_overlapped_mtp_rounds_match_greedy(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch, prefill_graphs: bool
) -> None:
    monkeypatch.setenv("SEED_PREFILL_GRAPH_SHAPES", "1x4,2x4,3x8")
    plan, oracle = resolve_plan(checkpoint)
    want = plain_reference(checkpoint, plan)

    patch_eager_draft(monkeypatch, oracle)
    model = mtp_model(checkpoint, 3)
    runner = mtp_overlap.OverlapMTPRunner(model, backend=EagerBackend())
    runner.want_prefill_graphs = prefill_graphs
    runner.prepare()
    assert runner.mtp_runner.enabled
    patch_graph_draft(runner.mtp_runner, oracle)

    launches = []
    real = runner.speculative_launch

    def spy(slots, tokens, positions, budgets, stops):  # noqa: ANN001, ANN202
        launches.append(list(tokens))
        return real(slots, tokens, positions, budgets, stops)

    runner.speculative_launch = spy
    inactive = []
    real_resolve = runner.lane_state.resolve

    def resolve(buf, look):  # noqa: ANN001, ANN202
        before = buf.active.clone()
        real_resolve(buf, look)
        inactive.append(int((before & ~buf.active).sum()))

    runner.lane_state.resolve = resolve

    pool_handshake(model)
    cache = SessionCache(model.block_allocator, model.block_size, model.num_snapshots)
    sched = Scheduler(runner, cache, prefill_chunk=4, spec_decode=True, overlap=True)
    assert sched.overlap
    got = run_conversations(sched, plan)
    assert got == want
    assert any(LOOKAHEAD_TOKEN in t for t in launches), "no round launched behind another"
    assert any(inactive), "no lane finished inside a round that had its successor in flight"


def test_overlap_stays_off_without_an_async_mtp_runner(checkpoint: Path) -> None:
    model = mtp_model(checkpoint, 2)
    pool_handshake(model)
    cache = SessionCache(model.block_allocator, model.block_size, model.num_snapshots)
    sched = Scheduler(model, cache, spec_decode=True, overlap=True)
    assert not sched.overlap


def test_lane_tables_record_active_rows_only() -> None:
    """`update` writes next token/position/budget/live for active rows' lanes only;
    `resolve` takes them for lookahead rows and drops rows whose lane is not live."""
    from types import SimpleNamespace

    tables = mtp_overlap.LaneTables(4, torch.device("cpu"))
    buf = SimpleNamespace(
        capacity=3,
        slot_rows=torch.tensor([2, 0, 3]),
        active=torch.tensor([True, True, False]),
        accept_len=torch.tensor([1, 2, 0]),
        step_argmax=torch.tensor([[5, 6, 7], [8, 9, 4], [1, 1, 1]]),
        budget=torch.tensor([5, 3, 9]),
        pos=torch.tensor([10, 20, 30]),
        stops=torch.tensor([[-1], [9], [-1]]),
        token_matrix=torch.zeros(3, 3, dtype=torch.long),
    )
    tables.update(buf)
    assert tables.next_tok.tolist() == [4, 0, 6, 0]
    assert tables.next_pos.tolist() == [23, 0, 12, 0]
    assert tables.budget_left.tolist() == [0, 0, 3, 0]
    assert tables.live.tolist() == [False, False, True, False]

    nxt = SimpleNamespace(
        slot_rows=torch.tensor([2, 0, 1]),
        active=torch.tensor([True, True, True]),
        token_matrix=torch.tensor([[0, 0, 0], [0, 0, 0], [7, 0, 0]]),
        pos=torch.tensor([10, 20, 40]),
        budget=torch.tensor([5, 3, 2]),
    )
    tables.resolve(nxt, torch.tensor([True, True, False]))
    assert nxt.token_matrix[:, 0].tolist() == [6, 4, 7]
    assert nxt.pos.tolist() == [12, 23, 40]
    assert nxt.budget.tolist() == [3, 0, 2]
    assert nxt.active.tolist() == [True, False, True]


# ---------------------------------------------------------------- fake-runner property test


class _FakeRound:
    def __init__(self, committed: list[list[int]]) -> None:
        self._committed = committed

    def committed(self) -> list[list[int]]:
        return self._committed


def _fake_overlap_runner_cls():  # noqa: ANN202
    """A `FakeSpecRunner` (perfect drafts, forced feed) whose rounds launch asynchronously
    with `mtp_overlap`'s lane-table contract: a `LOOKAHEAD` row takes its token, position and
    budget from the lane's previous round, and runs inactive (garbage result, no state change)
    when that round ended on a stop or its budget."""

    class FakeOverlapRunner(FakeSpecRunner):
        def __init__(self, *a, **kw) -> None:  # noqa: ANN002, ANN003
            super().__init__(*a, **kw)
            self.lane_tab: dict[int, tuple[int, int, int, bool]] = {}
            self.lookahead_rows = 0
            self.inactive_rows = 0

        def speculative_launch_ok(self, lanes, stops) -> bool:  # noqa: ANN001
            return True

        def speculative_launch(self, lanes, tokens, positions, budgets, stops, forced=None):  # noqa: ANN001, ANN202
            out = []
            for j, lane in enumerate(lanes):
                tok, pos, budget = tokens[j], positions[j], budgets[j]
                f = list(forced[j]) if forced is not None else []
                if tok == LOOKAHEAD_TOKEN:
                    assert not f, "a lookahead row never carries forced feed"
                    self.lookahead_rows += 1
                    tok, pos, budget, live = self.lane_tab[lane]
                    if not live:
                        self.inactive_rows += 1
                        out.append([424242])  # garbage: the host must never emit it
                        continue
                committed = self.speculative_decode(
                    [lane], [tok], [pos], [budget], [stops[j]], [f] if f else None
                )[0]
                self.rounds.pop()
                used = 1 if f else len(committed)
                left = budget - used
                self.lane_tab[lane] = (
                    committed[-1],
                    pos + len(committed),
                    left,
                    committed[-1] not in stops[j] and left > 0,
                )
                out.append(committed)
            assert K + 1 == self.decode_tokens_per_step()
            return _FakeRound(out)

    return FakeOverlapRunner


@settings(max_examples=60, deadline=None)
@given(
    suffixes=st.lists(st.integers(min_value=0, max_value=5), min_size=1, max_size=4),
    max_new=st.integers(min_value=1, max_value=12),
    stop_tok=st.one_of(st.none(), st.integers(min_value=1, max_value=997)),
    forced_on=st.booleans(),
)
def test_overlapped_rounds_match_serial_with_forced_suffixes(
    suffixes, max_new, stop_tok, forced_on
) -> None:  # noqa: ANN001
    """Concurrent requests (some draining folded suffixes of 0..5 ids, stops and budgets
    landing inside rounds) through overlapped rounds equal one-at-a-time generation."""
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(sched_mod, "MTP_FORCED_DRAFTS", forced_on)
        mp.setattr("test_scheduler.FakeRunner", _fake_overlap_runner_cls())
        sched, runner, _ = make(
            max_batch=4, prefill_chunk=8, spec_decode=True, fold_turn_suffix=True, overlap=True
        )
        assert sched.overlap
        stop = frozenset() if stop_tok is None else frozenset({stop_tok})
        reqs = []
        for i, n in enumerate(suffixes):
            prompt = [10 * i + 1, 10 * i + 2, *[900 + 10 * i + j for j in range(n)]]
            req, sink = request(prompt, max_new, stop, suffix_len=n)
            sched.submit(req)
            reqs.append((prompt, sink))
        drain(sched)
        for prompt, sink in reqs:
            want = serial(prompt, max_new, stop)
            stopped = bool(want) and want[-1] in stop
            assert sink.error is None
            assert sink.tokens == (want[:-1] if stopped else want)  # a stop id is not streamed
            assert sink.end[:2] == ("stop" if stopped else "length", len(want))
    finally:
        mp.undo()


def test_fake_overlap_exercises_lookahead_and_inactive_rows() -> None:
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(sched_mod, "MTP_FORCED_DRAFTS", True)
        mp.setattr("test_scheduler.FakeRunner", _fake_overlap_runner_cls())
        sched, runner, _ = make(
            max_batch=4, prefill_chunk=8, spec_decode=True, fold_turn_suffix=True, overlap=True
        )
        reqs = []
        for i, (n, max_new) in enumerate([(5, 11), (0, 7), (3, 9), (1, 12)]):
            prompt = [10 * i + 1, 10 * i + 2, *[900 + 10 * i + j for j in range(n)]]
            req, sink = request(prompt, max_new, suffix_len=n)
            sched.submit(req)
            reqs.append((prompt, max_new, sink))
        drain(sched)
        for prompt, max_new, sink in reqs:
            assert sink.tokens == serial(prompt, max_new)
        assert runner.lookahead_rows > 0
        assert runner.inactive_rows > 0
        assert not runner.batches, "an overlapped step fell back to plain decode"
    finally:
        mp.undo()

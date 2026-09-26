"""CPU tests for serving decode through MTP rounds (`SEED_MTP_SERVE=1`).

The property: a `Scheduler` driving MTP rounds (1..k+1 tokens per lane per step) produces the
same streamed tokens and the same finish events (reason, generated count, reused prompt tokens)
as a `Scheduler` driving plain greedy decode, on a multi-turn workload whose later turns reuse
the earlier turns' turn-close cache entries, with stop tokens landing inside accepted drafts.

The tiny checkpoint's MTP head has random weights, so its own drafts are almost never accepted.
To exercise multi-token advance, the drafter is replaced by an oracle built from the plain
model's greedy trajectories, corrupted at a position that cycles with the round's position so
every accept length 0..k occurs. Correctness never depends on the drafter (verify is exact);
the oracle only makes long accepts, and stops inside them, actually happen.

    /tmp/torchenv/bin/python -m pytest seed_tests/test_mtp_serve.py -q -o addopts=
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import graph_mtp  # noqa: E402
import mtp  # noqa: E402
from scheduler import Request, Scheduler  # noqa: E402
from session_cache import SessionCache  # noqa: E402
from test_mtp import build, prompt_of, write_mtp_shard  # noqa: E402
from test_seed_parity import VOCAB, build_hf, tiny_config, write_checkpoint  # noqa: E402
from tp_driver import pool_handshake  # noqa: E402

K = 2
POOL_FACTOR = 16.0
"""KV and snapshot pools = 16x the CPU floor (128 blocks, 32 snapshots at max_batch 2): no cache
entry is evicted, so reuse counts are a function of the workload alone and must match exactly
between the MTP and plain runs (MTP finishes turns in a different order, which would change
LRU eviction)."""


@pytest.fixture(autouse=True)
def roomy_kv_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    import model as seed_model

    monkeypatch.setattr(seed_model, "KV_POOL_MIN_CAPACITY_FACTOR", POOL_FACTOR)
    monkeypatch.setattr(seed_model, "SNAPSHOT_POOL_MIN_CAPACITY_FACTOR", POOL_FACTOR)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("tiny-mtp-serve")
    cfg = tiny_config()
    write_checkpoint(build_hf(cfg=cfg), out, mxfp4=False)
    write_mtp_shard(cfg, out)
    return out


class Sink:
    def __init__(self) -> None:
        self.tokens: list[int] = []
        self.end = None
        self.all_ids: list[int] = []  # streamed tokens plus the stop token, as `output_ids`

    def __call__(self, event: tuple) -> None:
        kind, val = event
        if kind == "tok":
            self.tokens.append(val)
        else:
            self.end = (kind, val)


class Oracle:
    """Drafts from known greedy trajectories: key `(pos, token)` -> the next `k` tokens.

    `corrupt` flips draft `pos % (k + 1)` (when `< k`), so accept lengths cycle through 0..k.
    Unknown keys draft zeros (a legal, wrong draft)."""

    def __init__(self, k: int, corrupt: bool = True) -> None:
        self.k, self.corrupt = k, corrupt
        self.table: dict[tuple[int, int], list[int]] = {}

    def add(self, seq: list[int], start: int) -> None:
        for i in range(start, len(seq)):
            nxt = seq[i + 1 : i + 1 + self.k]
            self.table.setdefault((i, seq[i]), nxt + [0] * (self.k - len(nxt)))

    def drafts(self, tokens: list[int], positions: list[int]) -> list[list[int]]:
        out = []
        for tok, pos in zip(tokens, positions, strict=True):
            d = list(self.table.get((pos, tok), [0] * self.k))
            c = pos % (self.k + 1)
            if self.corrupt and c < self.k:
                d[c] = (d[c] + 1) % VOCAB
            out.append(d)
        return out


def patch_eager_draft(monkeypatch: pytest.MonkeyPatch, oracle: Oracle) -> None:
    def draft(model, mtp_, hidden, tokens, slots, positions):  # noqa: ANN001, ANN202
        return oracle.drafts(list(tokens), list(positions))

    monkeypatch.setattr(mtp, "draft", draft)


def patch_graph_draft(runner: graph_mtp.MTPVerifyRunner, oracle: Oracle) -> None:
    """Replace each bucket's captured draft with the oracle writing the same buffer columns."""
    for graph in runner.graphs.values():
        buf = graph.buf

        def draft(buf: graph_mtp.MTPVerifyBuffers = buf) -> None:
            tokens = buf.token_matrix[:, 0].tolist()
            positions = buf.pos.tolist()
            buf.token_matrix[:, 1:].copy_(torch.tensor(oracle.drafts(tokens, positions)))

        graph.draft = draft


def plain_model(checkpoint: Path, max_batch: int):  # noqa: ANN201
    return build(checkpoint, max_batch=max_batch, mtp_env=False, k=K)


def mtp_model(checkpoint: Path, max_batch: int):  # noqa: ANN201
    return build(checkpoint, max_batch=max_batch, mtp_env=True, k=K)


# Conversations: (first prompt, [(max_new, stop index into this turn's greedy reply or None,
# follow-up user tokens)]...). A stop index picks that reply token as the stop id, so the
# stop lands wherever the round structure puts it, often inside an accepted draft.
CONVERSATIONS = [
    (prompt_of(11, 6), [(8, 4, [5, 6, 7]), (7, None, [9, 3]), (6, 2, [])]),
    (prompt_of(12, 7), [(9, 6, [4, 4]), (8, 3, [])]),
    (prompt_of(13, 5), [(5, None, [8, 2, 6]), (9, 5, [])]),
]


def run_conversations(sched: Scheduler, plan: list) -> dict[tuple[int, int], tuple]:
    """Drive every conversation to completion, submitting each follow-up turn as soon as the
    previous turn ends (`prompt + prompt reply + stop token + user tokens`, the turn-close
    node's own token span), interleaved across conversations. Returns (tokens, end) keyed by
    (conversation, turn): MTP finishes turns sooner, so completion order differs."""
    results: dict[tuple[int, int], tuple] = {}
    state = []  # per conversation: [turn index, prompt, sink or None]
    for prompt, turns in plan:
        state.append([0, list(prompt), None, turns])

    def submit(conv: list) -> None:
        turn, prompt, _, turns = conv
        max_new, stop, _user = turns[turn]
        sink = Sink()
        conv[2] = sink
        sched.submit(Request(list(prompt), max_new, 0.0, frozenset(stop), sink))

    for conv in state:
        submit(conv)
    for _ in range(2000):
        progressed = sched.step()
        for conv in state:
            sink = conv[2]
            if sink is None or sink.end is None:
                continue
            turn, prompt, _, turns = conv
            results[(state.index(conv), turn)] = (sink.tokens, sink.end)
            conv[2] = None
            max_new, stop, user = turns[turn]
            reply = sink.tokens + ([next(iter(stop))] if sink.end[1][0] == "stop" else [])
            if turn + 1 < len(turns):
                conv[0], conv[1] = turn + 1, prompt + reply + user
                submit(conv)
        if not progressed and all(c[2] is None for c in state):
            return results
    raise AssertionError("scheduler did not drain")


def resolve_plan(checkpoint: Path) -> tuple[list, Oracle]:
    """Turn stop indices into stop ids by running each turn greedily on the plain model (the
    reference), and build the oracle from those same trajectories."""
    ref = plain_model(checkpoint, 1)
    oracle = Oracle(K)
    plan = []
    for prompt, turns in CONVERSATIONS:
        resolved, cur = [], list(prompt)
        for max_new, stop_at, user in turns:
            full = list(ref.generate(cur, max_new + K, 0.0, frozenset()))
            oracle.add(cur + full, len(cur) - 1)
            stop = [] if stop_at is None else [full[stop_at]]
            reply = list(ref.generate(cur, max_new, 0.0, frozenset(stop)))
            resolved.append((max_new, stop, user))
            cur = cur + reply + user
        plan.append((prompt, resolved))
    return plan, oracle


def assert_turn_close_reuse(plan: list, got: dict) -> None:
    """Every follow-up turn resumed from the previous turn's turn-close node: its reused count
    is the previous prompt plus the previous reply (streamed tokens plus the stop token)."""
    for c, (prompt, turns) in enumerate(plan):
        prompt_len = len(prompt)
        for turn, (_max_new, stop, user) in enumerate(turns):
            tokens, end = got[(c, turn)]
            reply = len(tokens) + (len(stop) if end[1][0] == "stop" else 0)
            if turn + 1 < len(turns):
                assert got[(c, turn + 1)][1][1][2] == prompt_len + reply, (c, turn)
            prompt_len += reply + len(user)


def serve(runner, model, plan: list, spec: bool) -> list:  # noqa: ANN001
    pool_handshake(model)  # the scheduler owns blocks from here, as in `server.py`
    cache = SessionCache(model.block_allocator, model.block_size, model.num_snapshots)
    sched = Scheduler(runner, cache, prefill_chunk=4, spec_decode=spec)
    out = run_conversations(sched, plan)
    assert cache.root is not None
    return out


def test_commit_limit_matches_tensor_form() -> None:
    step_argmax = torch.tensor([[5, 6, 7], [5, 6, 7], [5, 6, 7], [9, 2, 3], [5, 6, 7]])
    accept = torch.tensor([2, 2, 1, 0, 2])
    budget = torch.tensor([9, 2, 9, 9, 1])
    stops = torch.tensor([[6, -1], [-1, -1], [7, -1], [9, 2], [-1, -1]])
    got = graph_mtp._commit_limit(accept, step_argmax, budget, stops).tolist()
    want = [
        mtp.commit_limit(
            int(accept[j]),
            step_argmax[j].tolist(),
            int(budget[j]),
            [s for s in stops[j].tolist() if s >= 0],
        )
        for j in range(5)
    ]
    assert got == want == [1, 1, 1, 0, 0]


def test_mtp_serving_matches_greedy_multi_turn_eager(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, oracle = resolve_plan(checkpoint)
    plain = plain_model(checkpoint, 2)
    want = serve(plain, plain, plan, spec=False)

    truncations = []
    real_limit = mtp.commit_limit

    def counting_limit(accept, step_argmax, budget, stops):  # noqa: ANN001, ANN202
        got = real_limit(accept, step_argmax, budget, stops)
        if got < accept:
            truncations.append((accept, got))
        return got

    monkeypatch.setattr(mtp, "commit_limit", counting_limit)
    patch_eager_draft(monkeypatch, oracle)
    model = mtp_model(checkpoint, 2)
    got = serve(model, model, plan, spec=True)

    assert got == want
    assert truncations, "no stop/budget landed inside an accepted draft; the test lost coverage"
    assert_turn_close_reuse(plan, got)


def test_mtp_serving_matches_greedy_multi_turn_graph(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same workload through `GraphMTPRunner` (captured draft+verify round, eager capture
    backend), with the scheduler owning blocks: `fill` must never allocate."""
    from test_graph_mtp import EagerBackend

    plan, oracle = resolve_plan(checkpoint)
    plain = plain_model(checkpoint, 2)
    want = serve(plain, plain, plan, spec=False)

    patch_eager_draft(monkeypatch, oracle)
    model = mtp_model(checkpoint, 2)
    runner = graph_mtp.GraphMTPRunner(model, backend=EagerBackend())
    runner.prepare()
    assert runner.mtp_runner.enabled
    patch_graph_draft(runner.mtp_runner, oracle)
    paths = []
    real = runner.mtp_runner.speculative_decode

    def spy(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        paths.append(runner.mtp_runner.replayable(args[0], args[4] if len(args) > 4 else None))
        return real(*args, **kwargs)

    runner.mtp_runner.speculative_decode = spy
    got = serve(runner, model, plan, spec=True)
    assert got == want
    assert paths and all(paths), "every MTP round should have replayed a captured bucket"


def test_idle_lanes_are_never_grown_by_a_captured_round(checkpoint: Path) -> None:
    """Reviewer finding: the round's `fill` used to grow every lane's block table with the
    model's own allocator. Under TP that allocation is rank-local; now only the round's lanes
    are checked (not grown) once the scheduler owns blocks."""
    from test_graph_mtp import EagerBackend

    model = mtp_model(checkpoint, 3)
    runner = graph_mtp.MTPVerifyRunner(model, model.mtp, backend=EagerBackend())
    assert runner.prepare() is True
    pool_handshake(model)
    prompt = prompt_of(77, 5)
    runner.begin(0)
    model.extend_blocks([(0, b) for b in model.block_allocator.alloc(1)])
    model.prefill(0, prompt, 0)
    free = model.block_allocator.free_count
    committed = runner.speculative_decode([0], [prompt[-1]], [len(prompt)], [5], [[]])
    assert 1 <= len(committed[0]) <= K + 1
    assert model.block_allocator.free_count == free
    assert [len(t.blocks) for t in model.block_tables[1:]] == [0, 0]
    with pytest.raises(RuntimeError, match="scheduler must reserve"):
        runner.speculative_decode([0], [prompt[-1]], [model.block_size - K], [5], [[]])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

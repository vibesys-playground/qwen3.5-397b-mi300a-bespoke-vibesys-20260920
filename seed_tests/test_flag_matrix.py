"""Scheduler flag matrix: every combination of `SEED_OVERLAP_SCHED`, `SEED_MIXED_BATCH` and
`SEED_BATCHED_PREFILL` (8 combos), on the eager model and on `GraphDecodeRunner` (eager
capture backend), must stream the same greedy tokens as the seed's one-sequence `generate`.

The workload is multi-turn with prefix reuse: three sessions, three turns each, every
follow-up turn is the previous prompt plus its reply plus new user tokens, so it resumes from
the previous turn's turn-close node. `max_batch=2` with staggered session starts keeps lanes
contended, so follow-ups are admitted while other lanes decode (mixed steps under
`SEED_MIXED_BATCH`, lookahead flushes under `SEED_OVERLAP_SCHED`).

    <python-with-torch> -m pytest seed_tests/test_flag_matrix.py -q -o addopts=
"""

import itertools
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import model as seed_model  # noqa: E402
from graph_decode import GraphDecodeRunner  # noqa: E402
from scheduler import Request, Scheduler  # noqa: E402
from session_cache import SessionCache  # noqa: E402
from test_batched_decode import build, prompt_of  # noqa: E402
from test_graph_capture import EagerBackend  # noqa: E402
from test_scheduler import Sink  # noqa: E402
from test_seed_parity import build_hf, write_checkpoint  # noqa: E402

SESSIONS = 3
TURNS = 3
MAX_NEW = (6, 5, 4)  # per turn
START_STEP = (0, 1, 3)  # per session
COMBOS = list(itertools.product([False, True], repeat=3))  # (overlap, mixed, batched_prefill)
SNAPSHOTS = 16
"""On CPU the snapshot pool sizes to its floor (`max_batch` slots), too few for any turn-close
node to survive until its follow-up turn; this gives the test room to exercise reuse."""


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("tiny-flag-matrix")
    write_checkpoint(build_hf(), out, mxfp4=False)
    return out


@pytest.fixture(scope="module")
def reference(checkpoint: Path):  # noqa: ANN201
    """`generate(prompt, max_new)`, memoized: one unbatched model shared by every combo."""
    model = build(checkpoint, max_batch=1)
    memo: dict[tuple, list[int]] = {}

    def ref(prompt: list[int], max_new: int) -> list[int]:
        key = (tuple(prompt), max_new)
        if key not in memo:
            memo[key] = list(model.generate(prompt, max_new, 0.0, frozenset()))
        return memo[key]

    return ref


def play(runner, overlap: bool, mixed: bool, batched: bool) -> tuple[list[list[tuple]], dict]:  # noqa: ANN001
    """Drive the scheduler step by step. Returns, per session, one (prompt, tokens, end) per
    turn, plus how many mixed steps and lookahead launches ran. A follow-up turn is submitted
    on the step after its previous turn ends."""
    model = getattr(runner, "model", runner)
    cache = SessionCache(model.block_allocator, model.block_size, model.num_snapshots)
    sched = Scheduler(
        runner,
        cache,
        prefill_chunk=4,
        token_budget=6,
        batched_prefill=batched,
        overlap=overlap,
        mixed_batch=mixed,
    )
    counts = {"mixed": 0, "launch": 0}
    for name in ("_mixed_step", "_launch_overlap"):
        inner = getattr(sched, name)

        def counted(*a, _inner=inner, _key=name, **k):  # noqa: ANN002, ANN003, ANN202
            counts["mixed" if _key == "_mixed_step" else "launch"] += 1
            return _inner(*a, **k)

        setattr(sched, name, counted)
    history: list[list[tuple]] = [[] for _ in range(SESSIONS)]
    live: list[tuple[list[int], Sink] | None] = [None] * SESSIONS
    step = 0
    while True:
        for s in range(SESSIONS):
            cur = live[s]
            if cur is not None and (cur[1].end is not None or cur[1].error is not None):
                prompt, sink = cur
                assert sink.error is None, f"session {s}: {sink.error}"
                history[s].append((prompt, list(sink.tokens), sink.end))
                live[s] = None
            turn = len(history[s])
            if live[s] is None and turn < TURNS and step >= START_STEP[s]:
                if turn == 0:
                    prompt = prompt_of(500 + s, 7 + 2 * s)
                else:
                    prev_prompt, prev_tokens, _ = history[s][-1]
                    prompt = prev_prompt + prev_tokens + prompt_of(600 + 10 * s + turn, 3)
                sink = Sink()
                sched.submit(Request(list(prompt), MAX_NEW[turn], 0.0, frozenset(), sink))
                live[s] = (prompt, sink)
        progressed = sched.step()
        if not progressed and all(len(h) == TURNS for h in history):
            break
        step += 1
        assert step < 10_000, "scheduler did not finish"
    assert sched._launched is None and not sched.decoding and not sched.prefill_q
    return history, counts


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_every_flag_combination_streams_the_reference_tokens(
    checkpoint: Path,
    reference,
    graph: bool,
    monkeypatch: pytest.MonkeyPatch,  # noqa: ANN001
) -> None:
    size_pools = seed_model.Model._size_pools
    monkeypatch.setattr(
        seed_model.Model, "_size_pools", lambda self, dev: (size_pools(self, dev)[0], SNAPSHOTS)
    )
    results = {}
    for overlap, mixed, batched in COMBOS:
        # `Model.prefill_batch` reads its own copy of the flag; the scheduler takes it as an
        # argument. Both layers get the same value, as the env var would give them.
        monkeypatch.setattr(seed_model, "BATCHED_PREFILL", batched)
        model = build(checkpoint, max_batch=2)
        runner = model
        if graph:
            runner = GraphDecodeRunner(model, backend=EagerBackend())
            assert runner.prepare()
        history, counts = play(runner, overlap, mixed, batched)
        combo = f"overlap={overlap} mixed={mixed} batched_prefill={batched}"
        # The workload must actually exercise the flags it claims to cover.
        assert (counts["mixed"] > 0) == mixed, (combo, counts)
        assert (counts["launch"] > 0) == overlap, (combo, counts)
        for s, turns in enumerate(history):
            for t, (prompt, tokens, end) in enumerate(turns):
                assert tokens == reference(prompt, MAX_NEW[t]), f"{combo}: session {s} turn {t}"
                assert end[0] == "length" and end[1] == MAX_NEW[t], (combo, s, t, end)
        # Every follow-up turn must resume from its previous turn's turn-close node.
        for s, turns in enumerate(history):
            for t in range(1, TURNS):
                prev_prompt, prev_tokens, _ = turns[t - 1]
                want = len(prev_prompt) + len(prev_tokens)
                assert turns[t][2][2] == want, f"{combo}: session {s} turn {t} reused {turns[t][2]}"
        results[(overlap, mixed, batched)] = [[tokens for _, tokens, _ in h] for h in history]
    baseline = results[(False, False, False)]
    for combo, got in results.items():
        assert got == baseline, f"{combo} streamed different tokens than all-off"

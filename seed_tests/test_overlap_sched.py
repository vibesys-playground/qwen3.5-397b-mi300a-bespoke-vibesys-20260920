"""CPU tests for `SEED_OVERLAP_SCHED` (scheduler.py `OVERLAP_SCHED`): one-step-lookahead
decode must produce exactly what the serial scheduler produces.

Two layers:

- `OverlapFakeRunner` (test_scheduler.py's `FakeRunner` plus a device-side per-lane token
  table and a deferred-read handle) under many randomized interleavings: arrivals between
  steps, stop tokens mid-batch, `max_tokens`, chat-suffix boundaries, turn-close publishes
  that later requests resume from, and lane pressure. Executing each call eagerly in call
  order is exactly what a single device stream does; what the fake models is *host*
  visibility (ids only through `PendingFake.tokens()`), which is the thing overlap changes.
  The fake's `prefill`/`decode` assert every position against the lane's history, so a
  double-forwarded or skipped token fails loudly.
- The tiny real `Model` (and `GraphDecodeRunner` with the eager stand-in backend): overlap
  on vs off vs the seed's own one-sequence `generate`.

    <python-with-torch> -m pytest seed_tests/test_overlap_sched.py -q -o addopts=
"""

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import block_pool  # noqa: E402
from scheduler import LOOKAHEAD, Request, Scheduler  # noqa: E402
from session_cache import SessionCache  # noqa: E402
from test_scheduler import BLOCK_SIZE, FakeRunner, Sink, drain, next_token, serial  # noqa: E402


class PendingFake:
    def __init__(self, ids: list[int]) -> None:
        self._ids = ids
        self.read = False

    def tokens(self) -> list[int]:
        self.read = True
        return list(self._ids)

    def row(self, i: int) -> list[int]:
        return [self._ids[i]]


class OverlapFakeRunner(FakeRunner):
    """`FakeRunner` plus `decode_launch`: a lane's `LOOKAHEAD` input resolves to the id this
    lane sampled last, which lives only in `lane_tok` (the device) until read back."""

    def __init__(self, *a, **kw) -> None:  # noqa: ANN002, ANN003
        super().__init__(*a, **kw)
        self.lane_tok = [None] * self.max_batch
        self.launches: list[list[int]] = []  # tokens argument of every launch
        self.fail_launch_at: int | None = None

    def decode_launch(self, lanes, tokens, positions, temperatures):  # noqa: ANN001, ANN201
        if self.fail_launch_at is not None and len(self.launches) == self.fail_launch_at:
            self.launches.append(list(tokens))
            raise RuntimeError("launch blew up")
        self.launches.append(list(tokens))
        resolved = []
        for lane, token in zip(lanes, tokens, strict=True):
            if token == LOOKAHEAD:
                assert self.lane_tok[lane] is not None, f"lane {lane}: lookahead with no token"
                token = self.lane_tok[lane]
            resolved.append(token)
        out = self.decode(list(lanes), resolved, list(positions))
        for lane, token in zip(lanes, out, strict=True):
            self.lane_tok[lane] = token
        return PendingFake(out)

    def begin(self, lane: int) -> None:
        super().begin(lane)
        self.lane_tok[lane] = None


def make(overlap: bool, max_batch: int, num_blocks: int = 512, num_snapshots: int = 64, **kw):
    allocator = block_pool.BlockAllocator(num_blocks, BLOCK_SIZE)
    cache = SessionCache(allocator, BLOCK_SIZE, num_snapshots)
    cls = OverlapFakeRunner if overlap else FakeRunner
    runner = cls(max_batch, allocator, BLOCK_SIZE)
    return Scheduler(runner, cache, overlap=overlap, **kw), runner, cache, allocator


# ---------------------------------------------------------------- randomized interleavings


def workload(seed: int) -> dict:
    """One random scenario: requests (some continuing a session, so they resume from a
    published node), each with an arrival step, a budget, maybe a stop token drawn from its
    own serial output (so it fires mid-batch), maybe a chat suffix."""
    rng = random.Random(seed)
    reqs = []
    history: list[list[int]] = []
    for i in range(rng.randint(1, 9)):
        if history and rng.random() < 0.4:
            base = list(rng.choice(history))  # a follow-up turn: resumes a published node
        else:
            base = [rng.randint(1, 900) for _ in range(rng.randint(1, 9))]
        prompt = base + [rng.randint(1, 900) for _ in range(rng.randint(0, 5))]
        if not prompt:
            prompt = [rng.randint(1, 900)]
        max_new = rng.randint(0, 9)
        full = serial(prompt, max(max_new, 1))
        stop: frozenset[int] = frozenset()
        if max_new and rng.random() < 0.5:
            stop = frozenset({rng.choice(full[:max_new])})
        suffix = rng.randint(1, min(3, len(prompt))) if rng.random() < 0.25 else 0
        arrival = rng.randint(0, 12) if i else 0
        reqs.append((prompt, max_new, stop, suffix, arrival))
        history.append(prompt + serial(prompt, max_new, stop))
    return {
        "reqs": reqs,
        "max_batch": rng.randint(1, 4),
        "prefill_chunk": rng.randint(1, 8),
        "batched_prefill": rng.random() < 0.5,
        "num_snapshots": rng.choice([4, 64]),
        "num_blocks": rng.choice([48, 512]),
    }


def play(w: dict, overlap: bool, early: bool = False):
    sched, runner, cache, allocator = make(
        overlap,
        w["max_batch"],
        prefill_early_launch=early,
        num_blocks=w["num_blocks"],
        num_snapshots=w["num_snapshots"],
        prefill_chunk=w["prefill_chunk"],
        batched_prefill=w["batched_prefill"],
    )
    sinks = []
    pending = sorted(enumerate(w["reqs"]), key=lambda x: x[1][4])
    step = 0
    order: list[tuple[int, Sink]] = []
    while True:
        while pending and pending[0][1][4] <= step:
            i, (prompt, max_new, stop, suffix, _) = pending.pop(0)
            sink = Sink()
            sched.submit(Request(list(prompt), max_new, 0.0, stop, sink, suffix))
            order.append((i, sink))
        progressed = sched.step()
        step += 1
        if not progressed and not pending:
            break
        if step > 100_000:
            raise AssertionError("scheduler did not finish")
    order.sort(key=lambda x: x[0])
    sinks = [s for _, s in order]
    return sched, runner, cache, allocator, sinks


@pytest.mark.parametrize("early", [False, True])
@pytest.mark.parametrize("seed", range(400))
def test_randomized_interleavings_match_the_serial_scheduler(seed: int, early: bool) -> None:
    """`early`: `SEED_PREFILL_EARLY_LAUNCH` (prefill enqueued before the in-flight step is read)."""
    w = workload(seed)
    _, off_runner, off_cache, _, off_sinks = play(w, overlap=False)
    assert_cache_is_consistent(off_cache, off_runner)
    sched, runner, cache, allocator, on_sinks = play(w, overlap=True, early=early)
    for i, (req, off, on) in enumerate(zip(w["reqs"], off_sinks, on_sinks, strict=True)):
        prompt, max_new, stop = req[0], req[1], req[2]
        if off.error is not None:
            # A tight block pool can exhaust even after eviction (both modes alike; seed 5718
            # of a 10k sweep). `test_pool_exhaustion_...` below covers that case directly.
            assert "block pool exhausted" in off.error, off.error
            continue
        assert on.error is None, on.error
        want = serial(prompt, max_new, stop)
        assert off.tokens == [t for t in want if t not in stop], f"req {i}: baseline broken"
        assert on.tokens == off.tokens, f"req {i}: overlap changed the streamed tokens"
        assert on.end is not None and off.end is not None
        assert on.end[:2] == off.end[:2], f"req {i}: overlap changed the finish reason/count"
        # `reused` is not compared: arrivals are pinned to step indices, and the overlap
        # scheduler's step count differs, so whether a follow-up turn's prefix is already
        # published when it arrives legitimately differs. What must hold is below: every
        # published node holds exactly the state of its own token path.
    assert_cache_is_consistent(cache, runner)
    # Nothing leaked: every lane is free, nothing is in flight, and every block still held is
    # held by the cache alone.
    assert sorted(sched.free_lanes) == list(range(w["max_batch"]))
    assert sched._launched is None and not sched.decoding and not sched.prefill_q
    cache.evict_all()
    assert allocator.free_count == allocator.usable_blocks


def assert_cache_is_consistent(cache: SessionCache, runner: FakeRunner) -> None:
    """Every trie node's snapshot, cached logits and KV rows are those of its own token path
    (the fake's state is the literal token history, so any publish of the wrong state, e.g.
    a lookahead token forwarded twice or not at all, shows up here).

    A node whose snapshot was reclaimed (`snapshot == -1`, see `session_cache.py`) has no
    state to check, only KV rows; it must still have children, since nothing can resume from
    a stateless leaf and the cache removes one as soon as it appears."""
    stack = [(cache.root, ())]
    while stack:
        node, path = stack.pop()
        if not node.is_root() and node.snapshot < 0:
            assert not node.is_leaf(), f"stateless leaf at {len(path)}"
        elif not node.is_root():
            assert runner.snapshots[node.snapshot] == list(path), f"node at {len(path)}"
            assert runner.node_logits[node.snapshot] == [next_token(list(path))]
        if not node.is_root():
            for pos, tok in enumerate(path):
                block = node.blocks[pos // BLOCK_SIZE]
                assert runner.block_content[block][pos % BLOCK_SIZE] == tok, f"KV row {pos}"
        for children in node.children.values():
            for child in children:
                stack.append((child, path + child.edge))


def test_consistency_check_accepts_reclaimed_snapshots_under_overlap() -> None:
    """Regression: with a small snapshot pool, multi-turn sessions reclaim superseded
    internal snapshots (`snapshot == -1`); the check indexed the fake's snapshot table with
    -1 and raised KeyError. Every stateful node must still match its path, and every
    stateless one must keep children."""
    for overlap in (False, True):
        sched, runner, cache, _ = make(overlap, max_batch=2, num_snapshots=3, prefill_chunk=4)
        for session in range(3):
            prompt = [session * 100 + j for j in range(1, 5)]
            for _turn in range(4):
                sink = Sink()
                req = Request(list(prompt), 3, 0.0, frozenset(), sink)
                sched.submit(req)
                drain(sched)
                assert sink.tokens == serial(prompt, 3)
                prompt = prompt + sink.tokens + [session * 100 + 50 + len(prompt)]
                assert_cache_is_consistent(cache, runner)
        assert cache.snapshot_stats().reclaims > 0, "setup: nothing was reclaimed"


def test_sequential_turns_reuse_the_same_prefixes_as_the_serial_scheduler() -> None:
    """With each turn drained before the next arrives, prefix reuse cannot depend on timing,
    so it must match exactly, including turns that end on a stop token (lookahead publish)."""
    for seed in range(40):
        rng = random.Random(seed)
        ends = {}
        for overlap in (False, True):
            sched, runner, cache, _ = make(overlap, max_batch=2, prefill_chunk=rng.randint(1, 8))
            prompt = [rng.randint(1, 900) for _ in range(rng.randint(1, 6))]
            got = []
            for _turn in range(4):
                full = serial(prompt, 6)
                stop = frozenset({full[rng.randint(0, 5)]}) if rng.random() < 0.6 else frozenset()
                sink = run_all(sched, [(prompt, 6, stop)])[0]
                got.append(sink.end)
                prompt = prompt + serial(prompt, 6, stop) + [rng.randint(1, 900)]
            assert_cache_is_consistent(cache, runner)
            ends[overlap] = got
            rng = random.Random(seed)  # replay the same draws for the other mode
        assert ends[True] == ends[False]


# ---------------------------------------------------------------- targeted cases


def run_all(sched: Scheduler, reqs: list[tuple]) -> list[Sink]:
    sinks = []
    for prompt, max_new, stop in reqs:
        sink = Sink()
        sched.submit(Request(list(prompt), max_new, 0.0, stop, sink))
        sinks.append(sink)
    drain(sched)
    return sinks


def test_the_next_step_is_launched_before_the_previous_ids_are_read() -> None:
    """The point of the feature: steady-state launches feed `LOOKAHEAD`, not host ids."""
    sched, runner, *_ = make(True, max_batch=2, prefill_chunk=8)
    run_all(sched, [([1, 2, 3], 8, frozenset()), ([4, 5], 8, frozenset())])
    lookahead_rows = sum(t == LOOKAHEAD for launch in runner.launches for t in launch)
    host_rows = sum(t != LOOKAHEAD for launch in runner.launches for t in launch)
    assert host_rows == 2, "only each request's first decode input comes from the host"
    assert lookahead_rows == 2 * 8 - 2 - 2  # 8 tokens: 1 from prefill, 7 decoded, last not fed


def test_fold_turn_suffix_never_feeds_lookahead_while_draining() -> None:
    """Regression: a flight draining its forced chat-suffix queue (`SEED_FOLD_TURN_SUFFIX`)
    must never be fed `LOOKAHEAD` (the device's own last-sampled id) in place of the queued
    suffix id -- `_launch_overlap`'s `outstanding` check has no notion of a flight whose real
    sampled token is meant to be discarded. `OverlapFakeRunner` resolves `LOOKAHEAD` to
    whatever the fake model actually produced, so a wrong id here diverges from `serial()`
    from the second suffix tick onward (the first tick isn't yet `outstanding`); pre-fix this
    failed even for a single, otherwise-idle flight."""
    boundary = [1, 2, 3, 4]
    suffix = [990, 991, 992]  # long enough to need more than one launch while draining
    prompt = boundary + suffix
    sched, runner, cache, _ = make(True, max_batch=2, prefill_chunk=8, fold_turn_suffix=True)
    sink = Sink()
    sched.submit(Request(list(prompt), 6, 0.0, frozenset(), sink, suffix_len=len(suffix)))
    drain(sched)
    assert sink.tokens == serial(prompt, 6)
    assert len(runner.launches) > 1
    assert runner.launches[: len(suffix)] == [[token] for token in suffix]

    # Same check with a second, concurrently-decoding lane, so the overlap pipeline is
    # actively launching steps for another flight while this one drains its suffix.
    sched2, runner2, *_ = make(True, max_batch=2, prefill_chunk=8, fold_turn_suffix=True)
    other_sink = Sink()
    sched2.submit(Request([7, 8], 10, 0.0, frozenset(), other_sink))
    chat_sink = Sink()
    sched2.submit(Request(list(prompt), 6, 0.0, frozenset(), chat_sink, suffix_len=len(suffix)))
    drain(sched2)
    assert other_sink.tokens == serial([7, 8], 10)
    assert chat_sink.tokens == serial(prompt, 6)
    # The second lane keeps using device lookahead while the chat lane is fed its exact
    # host-known suffix. Folding must not disable overlap for the whole batch.
    assert any(LOOKAHEAD in launch for launch in runner2.launches)


def test_max_tokens_costs_no_extra_step() -> None:
    """`max_new` is known on the host, so the request's final in-flight token is never fed
    back: the decode count equals the serial scheduler's."""
    off, off_runner, *_ = make(False, max_batch=1, prefill_chunk=8)
    on, on_runner, *_ = make(True, max_batch=1, prefill_chunk=8)
    run_all(off, [([1, 2, 3], 6, frozenset())])
    run_all(on, [([1, 2, 3], 6, frozenset())])
    assert on_runner.batches == off_runner.batches == [1] * 5


def test_a_stop_token_mid_batch_costs_one_discarded_token_and_is_not_streamed() -> None:
    expected = serial([1, 2, 3], 50)
    stop = frozenset({expected[3]})
    sched, runner, cache, _ = make(True, max_batch=2, prefill_chunk=8)
    sinks = run_all(sched, [([1, 2, 3], 50, stop), ([7, 8], 12, frozenset())])
    assert sinks[0].tokens == expected[:3]
    assert sinks[0].end == ("stop", 4, 0)
    assert sinks[1].tokens == serial([7, 8], 12)
    # 3 decode steps produce tokens 2..4 (the stop token is the 4th); one more ran with the
    # stop token as input before the host saw it, and its output was dropped.
    assert sum(1 for launch in runner.launches if len(launch) == 2) >= 4


def test_turn_close_after_a_lookahead_stop_publishes_the_same_state_without_a_reforward() -> None:
    """The lookahead step's input *was* the final token, so turn-close publishes that step's
    state and logits instead of re-forwarding it; the node must be identical, and an exact
    repeat of prompt+reply must still be served with zero prefill."""
    expected = serial([1, 2, 3], 50)
    stop = frozenset({expected[4]})
    results = {}
    for overlap in (False, True):
        sched, runner, cache, _ = make(overlap, max_batch=2, prefill_chunk=8)
        run_all(sched, [([1, 2, 3], 50, stop)])
        full = [1, 2, 3] + expected[:5]
        node = cache.lookup(full)
        assert node.depth == len(full), "turn-close was not published"
        results[overlap] = (runner.snapshots[node.snapshot], runner.node_logits[node.snapshot])
        before = runner.prefilled
        sinks = run_all(sched, [(full, 3, frozenset())])
        assert sinks[0].tokens == serial(full, 3)
        assert sinks[0].end[2] == len(full)
        if overlap:
            # +1: request 2's own turn-close re-forward (max_tokens path), as in the serial
            # scheduler (test_scheduler.py); the repeat itself costs nothing.
            assert runner.prefilled == before + 1, "exact repeat must cost zero prefill"
    assert results[True] == results[False]


def test_a_decode_launch_failure_fails_the_batch_and_frees_the_lanes() -> None:
    sched, runner, _, _ = make(True, max_batch=2, prefill_chunk=8)
    runner.fail_launch_at = 2
    sinks = run_all(sched, [([1, 2, 3], 9, frozenset()), ([4, 5], 9, frozenset())])
    assert all(s.error is not None and "launch blew up" in s.error for s in sinks)
    assert sorted(sched.free_lanes) == [0, 1]
    # The scheduler keeps serving afterwards.
    ok = run_all(sched, [([6, 7], 4, frozenset())])
    assert ok[0].tokens == serial([6, 7], 4)


def test_pool_exhaustion_at_launch_matches_the_serial_scheduler() -> None:
    """Blocks for step N+1 are reserved before it launches (`_launch_overlap`). A lane whose
    reservation fails is left out of N+1 and failed only after step N completes, so the token
    it had in flight is still streamed first: the streamed tokens and the error must equal
    the serial scheduler's, and nothing may leak."""
    reqs = [([1, 2, 3], 40, frozenset()), ([4, 5, 6], 40, frozenset())]
    results = {}
    for overlap in (False, True):
        # 7 usable blocks of 4 tokens: two lanes of ~43 tokens each cannot both fit.
        sched, runner, cache, allocator = make(overlap, max_batch=2, num_blocks=8, prefill_chunk=8)
        sinks = run_all(sched, reqs)
        assert any(s.error is not None and "block pool exhausted" in s.error for s in sinks)
        assert sorted(sched.free_lanes) == [0, 1]
        assert sched._launched is None
        cache.evict_all()
        assert allocator.free_count == allocator.usable_blocks
        results[overlap] = [(s.tokens, s.end, s.error) for s in sinks]
    assert results[True] == results[False]


def test_overlap_is_off_when_mtp_is_on() -> None:
    sched, *_ = make(True, max_batch=1, spec_decode=True)
    assert sched.overlap is False


# ---------------------------------------------------------------- the tiny real model

torch = pytest.importorskip("torch")

from graph_decode import GraphDecodeRunner  # noqa: E402
from test_batched_decode import build, prompt_of  # noqa: E402
from test_graph_capture import EagerBackend  # noqa: E402
from test_seed_parity import build_hf, write_checkpoint  # noqa: E402


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("tiny-overlap")
    write_checkpoint(build_hf(), out, mxfp4=False)
    return out


def real_specs(checkpoint: Path) -> tuple[list[tuple], list[list[int]]]:
    """Requests with staggered arrivals, a stop token mid-batch, `max_tokens`, and a follow-up
    turn that resumes from the first request's turn-close node; plus each one's reference
    from the seed's own one-sequence `generate`."""
    ref = build(checkpoint, max_batch=1)
    base = [prompt_of(300 + i, 4 + 2 * i) for i in range(4)]
    greedy = [list(ref.generate(p, 10, 0.0, frozenset())) for p in base]
    specs = [
        (base[0], 7, frozenset(), 0),
        (base[1], 10, frozenset({greedy[1][3]}), 0),  # stops mid-batch
        (base[2], 5, frozenset(), 2),
        (base[3], 9, frozenset({greedy[3][5]}), 1),  # stops mid-batch, arrives mid-decode
    ]
    want = [list(ref.generate(p, n, 0.0, stop)) for p, n, stop, _ in specs]
    specs += [
        (base[0] + want[0] + [5, 6], 6, frozenset(), 30),  # resumes req 0's turn close
        (base[1] + want[1], 4, frozenset(), 40),  # repeat of a lookahead publish (if still
        # cached: the tiny CPU KV pool is 18 blocks, so eviction is likely; same in both modes)
    ]
    want += [list(ref.generate(p, n, 0.0, stop)) for p, n, stop, _ in specs[4:]]
    return specs, want


def play_real(runner, specs, overlap: bool) -> list[Sink]:  # noqa: ANN001
    model = getattr(runner, "model", runner)
    cache = SessionCache(model.block_allocator, model.block_size, model.num_snapshots)
    sched = Scheduler(runner, cache, prefill_chunk=4, overlap=overlap)
    sinks = [Sink() for _ in specs]
    step, pending = 0, sorted(range(len(specs)), key=lambda i: specs[i][3])
    while True:
        while pending and specs[pending[0]][3] <= step:
            i = pending.pop(0)
            prompt, max_new, stop, _ = specs[i]
            sched.submit(Request(list(prompt), max_new, 0.0, stop, sinks[i]))
        if not sched.step() and not pending:
            return sinks
        step += 1


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_real_model_overlap_matches_serial_and_the_reference(checkpoint: Path, graph: bool) -> None:
    specs, want = real_specs(checkpoint)
    results = {}
    for overlap in (False, True):
        model = build(checkpoint, max_batch=3)
        runner = model
        if graph:
            runner = GraphDecodeRunner(model, backend=EagerBackend())
            runner.min_batch = 1  # replay every batch, so the captured path is what is tested
            assert runner.prepare()
        results[overlap] = play_real(runner, specs, overlap)
    for i, ((_, _, stop, _), ref) in enumerate(zip(specs, want, strict=True)):
        off, on = results[False][i], results[True][i]
        assert off.error is None and on.error is None, (off.error, on.error)
        streamed = [t for t in ref if t not in stop]
        assert off.tokens == streamed, f"req {i}: serial scheduler diverged from generate"
        assert on.tokens == streamed, f"req {i}: overlap diverged"
        assert on.end == off.end, f"req {i}: finish/reuse differ ({on.end} vs {off.end})"
    assert results[True][4].end[2] > 0, "the follow-up turn did not resume from a cached node"


def test_lookahead_publish_holds_the_same_state_as_the_reforward(checkpoint: Path) -> None:
    """A stop token read one step late: overlap publishes the lookahead step's state (the stop
    token forwarded by *decode*) where the serial path re-forwards it with a one-token
    *prefill*. Same arithmetic, different kernels, so equal to float tolerance."""
    ref = build(checkpoint, max_batch=1)
    prompt = prompt_of(77, 6)
    stop = frozenset({list(ref.generate(prompt, 8, 0.0, frozenset()))[4]})
    states = {}
    for overlap in (False, True):
        model = build(checkpoint, max_batch=2)
        cache = SessionCache(model.block_allocator, model.block_size, model.num_snapshots)
        sched = Scheduler(model, cache, prefill_chunk=4, overlap=overlap)
        sink = Sink()
        sched.submit(Request(list(prompt), 8, 0.0, stop, sink))
        drain(sched)
        full = prompt + sink.tokens + sorted(stop)
        node = cache.lookup(full)
        assert node.depth == len(full)
        snap = [
            {k: v.clone() for k, v in layer.items()}
            for layer in model.snapshot_row(node.snapshot)
            if layer is not None
        ]
        states[overlap] = (snap, model.cached_node_logits(node.snapshot).clone())
        repeat = Sink()
        sched.submit(Request(list(full), 3, 0.0, frozenset(), repeat))
        drain(sched)
        assert repeat.end[2] == len(full), "exact repeat was not served from the cache"
        assert repeat.tokens == list(ref.generate(full, 3, 0.0, frozenset()))
    (snap_off, logits_off), (snap_on, logits_on) = states[False], states[True]
    torch.testing.assert_close(logits_on, logits_off, atol=1e-4, rtol=1e-4)
    for a, b in zip(snap_off, snap_on, strict=True):
        for k in a:
            torch.testing.assert_close(b[k], a[k], atol=1e-4, rtol=1e-4)


def test_real_model_overlap_samples_with_temperature(checkpoint: Path) -> None:
    model = build(checkpoint, max_batch=2)
    cache = SessionCache(model.block_allocator, model.block_size, model.num_snapshots)
    sched = Scheduler(model, cache, prefill_chunk=4, overlap=True)
    sinks = [Sink(), Sink()]
    sched.submit(Request(prompt_of(5, 5), 6, 0.8, frozenset(), sinks[0]))
    sched.submit(Request(prompt_of(6, 5), 6, 0.0, frozenset(), sinks[1]))
    drain(sched)
    assert sinks[0].error is None and len(sinks[0].tokens) == 6
    assert sinks[1].tokens == list(
        build(checkpoint, 1).generate(prompt_of(6, 5), 6, 0.0, frozenset())
    )


# ---------------------------------------------------------------- four ranks, real gloo group


@pytest.fixture(scope="module")
def tp_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from test_seed_parity import tiny_config
    from test_tensor_parallel import TP_AXES

    out = tmp_path_factory.mktemp("tp-overlap")
    write_checkpoint(build_hf(cfg=tiny_config(**TP_AXES)), out, mxfp4=False)
    return out


def run_ranks(
    checkpoint: Path, specs_file: Path, out: Path, overlap: bool, mixed: bool = False
) -> list:
    import subprocess

    from test_tensor_parallel import WORLD, free_port

    script = Path(__file__).resolve().parent / "overlap_gloo_rank.py"
    base = [sys.executable, str(script), "--checkpoint", str(checkpoint), "--world", str(WORLD)]
    base += ["--port", str(free_port()), "--specs", str(specs_file), "--overlap", str(int(overlap))]
    base += ["--mixed", str(int(mixed))]
    procs = [
        subprocess.Popen([*base, "--rank", str(r), *(["--out", str(out)] if r == 0 else [])])  # noqa: S603
        for r in range(WORLD)
    ]
    try:
        codes = [p.wait(timeout=900) for p in procs]
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
    assert codes == [0] * WORLD, f"rank exit codes {codes}"
    import json

    return json.loads(out.read_text())


def test_four_gloo_ranks_overlap_matches_serial(tp_checkpoint: Path, tmp_path: Path) -> None:
    """Four processes: `Op.DECODE_LAUNCH` over the gloo command group, the sampled ids shared
    by a broadcast inside the step, lookahead publish, max_tokens and a late arrival, all
    through `Broadcaster`/`serve_worker`. Tokens must equal the unsharded `generate`, and the
    finish events must equal the same four-rank run with overlap off."""
    import json

    from tp_gloo_rank import MAX_SEQ

    ref = seed_model_unsharded(tp_checkpoint, MAX_SEQ)
    base = [prompt_of(900 + i, 5 + i) for i in range(4)]
    greedy = [list(ref.generate(p, 8, 0.0, frozenset())) for p in base]
    specs = [
        [base[0], 6, [], 0],
        [base[1], 8, [greedy[1][2]], 0],  # stop mid-batch
        [base[2], 5, [], 1],  # waits for a lane (max_batch 2), then joins
        [base[3], 7, [greedy[3][4]], 3],
    ]
    want = [list(ref.generate(p, n, 0.0, frozenset(s))) for p, n, s, _ in specs]
    # A follow-up of req 1's lookahead publish. The CPU KV pool is a dozen blocks, so the node
    # is usually evicted by then; `end` (which carries the reuse count) equals the serial run.
    specs.append([base[1] + want[1] + [7], 4, [], 60])
    want.append(list(ref.generate(specs[-1][0], 4, 0.0, frozenset())))
    specs_file = tmp_path / "specs.json"
    specs_file.write_text(json.dumps(specs))

    on = run_ranks(tp_checkpoint, specs_file, tmp_path / "on.json", overlap=True)
    off = run_ranks(tp_checkpoint, specs_file, tmp_path / "off.json", overlap=False)
    for i, ((_, _, stop, _), ref_tokens) in enumerate(zip(specs, want, strict=True)):
        streamed = [t for t in ref_tokens if t not in stop]
        assert off[i][0] == streamed, f"req {i}: serial four-rank run diverged"
        assert on[i][0] == streamed, f"req {i}: overlap four-rank run diverged"
        assert on[i][1] == off[i][1], f"req {i}: finish events differ"


def test_four_gloo_ranks_mixed_batch_with_and_without_overlap(
    tp_checkpoint: Path, tmp_path: Path
) -> None:
    """`SEED_MIXED_BATCH` under TP: `Op.DECODE_MIXED` carries decode rows and prefill chunks in
    one broadcast, and both halves' blocks are reserved on rank 0 (`EXTEND_BLOCKS`) before it.
    Late arrivals force mixed steps while other lanes decode; with overlap on, each mixed step
    first completes the in-flight lookahead step. Tokens must equal the unsharded `generate`."""
    import json

    from tp_gloo_rank import MAX_SEQ

    ref = seed_model_unsharded(tp_checkpoint, MAX_SEQ)
    base = [prompt_of(950 + i, 6 + i) for i in range(4)]
    specs = [[base[0], 8, [], 0], [base[1], 6, [], 2], [base[2], 7, [], 4], [base[3], 5, [], 5]]
    want = [list(ref.generate(p, n, 0.0, frozenset())) for p, n, _, _ in specs]
    specs_file = tmp_path / "specs.json"
    specs_file.write_text(json.dumps(specs))
    for overlap in (False, True):
        out = tmp_path / f"mixed-{int(overlap)}.json"
        got = run_ranks(tp_checkpoint, specs_file, out, overlap=overlap, mixed=True)
        for i, ref_tokens in enumerate(want):
            assert got[i][0] == ref_tokens, f"overlap={overlap} req {i}: mixed four-rank diverged"
            assert got[i][1][0] == "end", got[i][1]


def seed_model_unsharded(checkpoint: Path, max_seq: int):  # noqa: ANN201
    import model as seed_model

    return seed_model.Model(checkpoint, ["cpu"], torch.float32, max_seq, 1)


def test_early_launch_enqueues_prefill_before_reading_the_in_flight_step() -> None:
    """`SEED_PREFILL_EARLY_LAUNCH`: with a decode step in flight, a new request's prefill is
    forwarded before that step's ids are read back."""
    sched, runner, _, _ = make(True, max_batch=2, prefill_early_launch=True)
    first = Sink()
    sched.submit(Request([5, 6, 7], 8, 0.0, frozenset(), first))
    while sched._launched is None:
        sched.step()
    inflight = sched._launched.handle
    seen: list[bool] = []
    orig = runner.prefill_batch

    def spy(calls):  # noqa: ANN001, ANN202
        seen.append(inflight.read)
        return orig(calls)

    runner.prefill_batch = spy
    second = Sink()
    sched.submit(Request([9, 10], 4, 0.0, frozenset(), second))
    drain(sched)
    assert seen and seen[0] is False
    assert first.tokens == serial([5, 6, 7], 8) and second.tokens == serial([9, 10], 4)

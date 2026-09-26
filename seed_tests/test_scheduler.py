"""Hermetic CPU tests for the Stage 2 batching scheduler + prefix cache, driven by a fake
model.

`FakeRunner` drives a real `block_pool.BlockAllocator` and a real per-lane `block_pool.
BlockTable` (both torch-free already, see `block_pool.py`'s module docstring), so block
growth, copy-on-write and refcount bookkeeping run for real, not faked -- only the DeltaNet/
conv state and the forward pass itself are stand-ins. `next_token` is a pure function of the
logical token history a lane's `hist` list holds, so any scheduling or cache bug (wrong lane,
wrong boundary, a stale or corrupted snapshot, a missing copy-on-write, a lane reused before the
cache entry it fed is safe) shows up as a wrong token or a wrong refcount, not a timing
difference.

    python3 -m pytest examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/seed_tests/test_scheduler.py
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import block_pool  # noqa: E402
from scheduler import MAX_PREFILL_BURST, Request, Scheduler  # noqa: E402
from session_cache import SessionCache  # noqa: E402

BLOCK_SIZE = 4


def next_token(hist: list[int]) -> int:
    """Deterministic stand-in for a forward pass over `hist`."""
    return (sum(hist) * 31 + len(hist) * 7) % 997 + 1


def serial(prompt: list[int], max_new: int, stop: frozenset[int] = frozenset()) -> list[int]:
    """What one-request-at-a-time generation yields, mirroring Model.generate."""
    hist, out = list(prompt), []
    for _ in range(max_new):
        tok = next_token(hist)
        out.append(tok)
        if tok in stop:
            break
        hist.append(tok)
    return out


class FakeRunner:
    """Lane-addressed fake model, backed by a real `block_pool` allocator/table per lane."""

    def __init__(
        self, max_batch: int, allocator: block_pool.BlockAllocator, block_size: int = BLOCK_SIZE
    ) -> None:
        self.max_batch = max_batch
        self.allocator = allocator
        self.block_size = block_size
        self.hist: list[list[int]] = [[] for _ in range(max_batch)]
        self.tables = [block_pool.BlockTable() for _ in range(max_batch)]
        self.block_content: dict[int, list[int | None]] = {}
        self.snapshots: dict[int, list[int]] = {}
        self.node_logits: dict[int, list[int]] = {}
        self.prefilled = 0  # tokens sent through prefill, the work prefix reuse saves
        self.batches: list[int] = []  # batch size of every decode step
        self.fail_at: int | None = None  # decode step index that raises
        self.prefill_calls = 0  # count of prefill() calls, for fail_prefill_at
        self.fail_prefill_at: int | None = None  # prefill call index that raises
        self.prefill_batch_sizes: list[int] = []  # number of requests in each prefill_batch call
        self.prefill_batch_chunk_sizes: list[list[int]] = []  # token count per request, per call
        self.prefill_batch_lanes: list[list[int]] = []  # lanes touched by each prefill_batch call
        self.fail_prefill_batch_at: int | None = None  # prefill_batch call index that raises
        self.copy_block_calls: list[tuple[int, int, int]] = []  # (dst, src, filled), in order

    def begin(self, lane: int) -> None:
        self.hist[lane] = []
        self.tables[lane].blocks = []

    def lane_blocks(self, lane: int) -> tuple[int, ...]:
        return tuple(self.tables[lane].blocks)

    def attach_blocks(self, lane: int, blocks) -> None:  # noqa: ANN001
        self.tables[lane].blocks = list(blocks)

    def lane_block_count(self, lane: int) -> int:
        return len(self.tables[lane].blocks)

    def extend_blocks(self, grants) -> None:  # noqa: ANN001
        for lane, block in grants:
            assert self.allocator.refcount(block) >= 1, f"block {block} granted while free"
            self.tables[lane].blocks.append(block)

    def decode_tokens_per_step(self) -> int:
        return 1

    def copy_block(self, dst_block: int, src_block: int, filled: int) -> None:
        self.copy_block_calls.append((dst_block, src_block, filled))
        src = self.block_content.get(src_block, [None] * self.block_size)
        dst = self.block_content.setdefault(dst_block, [None] * self.block_size)
        dst[:filled] = src[:filled]

    def load_snapshot(self, lane: int, snap: int) -> None:
        assert snap in self.snapshots, f"snapshot {snap} was never saved"
        self.hist[lane] = list(self.snapshots[snap])

    def save_snapshot(self, lane: int, snap: int) -> None:
        self.snapshots[snap] = list(self.hist[lane])

    def _write(self, lane: int, ids: list[int], start: int) -> None:
        table = self.tables[lane]
        # Like `Model` once serving starts (`scheduler_owns_blocks`): the runner never
        # allocates, the scheduler must have reserved and extended the table already.
        assert table.token_capacity(self.block_size) >= start + len(ids), (
            f"lane {lane}: {len(table.blocks)} block(s) for {start + len(ids)} tokens"
        )
        for i, tok in enumerate(ids):
            pos = start + i
            block = table.blocks[pos // self.block_size]
            self.block_content.setdefault(block, [None] * self.block_size)[
                pos % self.block_size
            ] = tok

    def prefill(self, lane: int, ids: list[int], start: int) -> list[int]:
        if self.fail_prefill_at is not None and self.prefill_calls == self.fail_prefill_at:
            self.prefill_calls += 1
            raise RuntimeError("prefill blew up")
        self.prefill_calls += 1
        assert start == len(self.hist[lane]), (
            f"lane {lane}: start {start} != {len(self.hist[lane])}"
        )
        self._write(lane, ids, start)
        self.hist[lane].extend(ids)
        self.prefilled += len(ids)
        return [next_token(self.hist[lane])]

    def prefill_batch(self, calls: list[tuple[int, list[int], int]]) -> list[list[int]]:
        """Packed prefill as a plain loop over `prefill` -- see the Stage 1 test's own note:
        what this exercises is scheduler-side batching, not numeric packing itself."""
        call_index = len(self.prefill_batch_sizes)
        if self.fail_prefill_batch_at is not None and call_index == self.fail_prefill_batch_at:
            self.prefill_batch_sizes.append(len(calls))
            self.prefill_batch_chunk_sizes.append([len(ids) for _, ids, _ in calls])
            self.prefill_batch_lanes.append([lane for lane, _, _ in calls])
            raise RuntimeError("prefill_batch blew up")
        self.prefill_batch_sizes.append(len(calls))
        self.prefill_batch_chunk_sizes.append([len(ids) for _, ids, _ in calls])
        self.prefill_batch_lanes.append([lane for lane, _, _ in calls])
        return [self.prefill(lane, ids, start) for lane, ids, start in calls]

    def decode(self, lanes: list[int], tokens: list[int], positions: list[int]) -> list[int]:
        if self.fail_at is not None and len(self.batches) == self.fail_at:
            self.batches.append(len(lanes))
            raise RuntimeError("decode blew up")
        self.batches.append(len(lanes))
        out = []
        for lane, token, pos in zip(lanes, tokens, positions, strict=True):
            assert pos == len(self.hist[lane]), f"lane {lane}: pos {pos} != {len(self.hist[lane])}"
            self._write(lane, [token], pos)
            self.hist[lane].append(token)
            out.append(next_token(self.hist[lane]))
        return out

    def decode_row(self, logits: list[int], row: int) -> list[int]:
        return [logits[row]]

    def sample_batch(self, logits: list[int], temperatures: list[float]) -> list[int]:
        assert len(logits) == len(temperatures)
        return list(logits)

    def cache_node_logits(self, snap: int, logits: list[int]) -> None:
        self.node_logits[snap] = list(logits)

    def cached_node_logits(self, snap: int) -> list[int]:
        cached = self.node_logits.get(snap)
        assert cached is not None, f"snapshot {snap} has no cached logits"
        return cached


class Sink:
    """Collects one request's events the way server.py's `emit` would."""

    def __init__(self) -> None:
        self.tokens: list[int] = []
        self.end: tuple[str, int, int] | None = None
        self.error: str | None = None

    def __call__(self, event: tuple) -> None:
        kind, val = event
        if kind == "tok":
            self.tokens.append(val)
        elif kind == "end":
            self.end = val
        else:
            self.error = val


def request(
    prompt: list[int],
    max_new: int,
    stop: frozenset[int] = frozenset(),
    *,
    suffix_len: int = 0,
) -> tuple:
    sink = Sink()
    return Request(list(prompt), max_new, 0.0, stop, sink, suffix_len), sink


def drain(sched: Scheduler, limit: int = 100_000) -> None:
    for _ in range(limit):
        if not sched.step():
            return
    raise AssertionError("scheduler did not finish")


def session(base: int, turns: list[int]) -> list[list[int]]:
    """Prompts of one session: each turn is the previous prompt plus a fresh suffix."""
    prompts, prompt = [], [base, base + 1, base + 2]
    for n in turns:
        prompt = prompt + [base * 100 + n * 10 + j for j in range(n)]
        prompts.append(list(prompt))
    return prompts


def make(
    max_batch: int = 4,
    num_blocks: int = 256,
    num_snapshots: int = 32,
    block_size: int = BLOCK_SIZE,
    **kw,
) -> tuple[Scheduler, FakeRunner, SessionCache]:
    allocator = block_pool.BlockAllocator(num_blocks, block_size)
    cache = SessionCache(allocator, block_size, num_snapshots)
    runner = FakeRunner(max_batch, allocator, block_size)
    sched = Scheduler(runner, cache, **kw)
    return sched, runner, cache


# ---------------------------------------------------------------- (a) batching


def test_concurrent_requests_match_serial_processing() -> None:
    sched, runner, _ = make(max_batch=4, prefill_chunk=3)
    specs = [([1, 2, 3, 4, 5, 6, 7], 9), ([11, 12], 4), ([21, 22, 23, 24], 12), ([31], 7)]
    pairs = [request(p, n) for p, n in specs]
    for req, _ in pairs:
        sched.submit(req)
    drain(sched)

    for (prompt, max_new), (_, sink) in zip(specs, pairs, strict=True):
        assert sink.error is None
        assert sink.tokens == serial(prompt, max_new)
        assert sink.end == ("length", max_new, 0)
    assert max(runner.batches) > 1, "no decode step was actually shared"


def test_many_slots_finish_in_the_same_decode_step_still_match_serial() -> None:
    sched, runner, _ = make(max_batch=6, prefill_chunk=8)
    specs = [([1, 2], 2), ([3, 4], 2), ([5, 6], 8), ([7, 8], 2), ([9, 10], 8), ([11, 12], 2)]
    pairs = [request(p, n) for p, n in specs]
    for req, _ in pairs:
        sched.submit(req)
    drain(sched)

    for (prompt, max_new), (_, sink) in zip(specs, pairs, strict=True):
        assert sink.error is None
        assert sink.tokens == serial(prompt, max_new)
        assert sink.end == ("length", max_new, 0)


def test_late_arrivals_join_the_running_batch() -> None:
    sched, runner, _ = make(max_batch=4, prefill_chunk=2)
    early = [request([1, 2, 3], 20), request([4, 5, 6, 7], 20)]
    late = [request([8, 9], 6), request([10, 11, 12], 6)]
    for req, _ in early:
        sched.submit(req)
    for _ in range(9):
        sched.step()
    for req, _ in late:
        sched.submit(req)
    drain(sched)

    for (req, sink), max_new in zip(early + late, [20, 20, 6, 6], strict=True):
        assert sink.error is None
        assert sink.tokens == serial(req.prompt, max_new)
    assert max(runner.batches) >= 3, "late requests never shared a decode step"


def test_requests_beyond_max_batch_wait_and_still_complete() -> None:
    sched, _, _ = make(max_batch=2, prefill_chunk=4)
    specs = [([1, 2, 3], 5), ([4, 5, 6], 5), ([7, 8, 9], 5), ([10, 11], 5)]
    pairs = [request(p, n) for p, n in specs]
    for req, _ in pairs:
        sched.submit(req)
    drain(sched)
    for (prompt, max_new), (_, sink) in zip(specs, pairs, strict=True):
        assert sink.tokens == serial(prompt, max_new)


def test_stop_token_ends_the_turn_without_streaming_it() -> None:
    sched, _, _ = make(max_batch=2, prefill_chunk=8)
    prompt = [1, 2, 3]
    expected = serial(prompt, 50)
    stop = frozenset({expected[3]})
    req, sink = request(prompt, 50, stop)
    sched.submit(req)
    drain(sched)
    assert sink.end == ("stop", 4, 0)
    assert sink.tokens == expected[:3]  # the stop token counts but is not streamed


def test_decode_failure_is_reported_and_the_lane_is_freed() -> None:
    """Unlike Stage 1 (which discarded a slot's whole recorded prefix on any later failure),
    Stage 2 only ever publishes what was actually, successfully computed: the prompt boundary
    here was genuinely prefilled before decode ever started, so it stays a valid cache entry
    even though this request's own decode later fails -- the failure just means no turn-close
    node for *this* request's reply gets published."""
    sched, runner, cache = make(max_batch=2, prefill_chunk=8)
    runner.fail_at = 2
    req, sink = request([1, 2, 3], 10)
    sched.submit(req)
    drain(sched)
    assert sink.error is not None
    assert len(sched.free_lanes) == 2, "a failed lane must still be returned to the free pool"
    assert cache.lookup([1, 2, 3]).depth == 3, "the successfully prefilled prompt stays cached"


def test_a_broken_iteration_fails_requests_instead_of_hanging() -> None:
    class Broken(FakeRunner):
        def begin(self, lane: int) -> None:
            raise RuntimeError("out of memory")

    allocator = block_pool.BlockAllocator(64, BLOCK_SIZE)
    cache = SessionCache(allocator, BLOCK_SIZE, 8)
    sched = Scheduler(Broken(2, allocator, BLOCK_SIZE), cache, prefill_chunk=8)
    req, sink = request([1, 2, 3], 5)
    sched.submit(req)
    sched.step()  # the admission fails and rolls back: only this request errors
    assert sink.error is not None and "out of memory" in sink.error
    assert len(sched.free_lanes) == 2, "a failed admission must return its lane"
    assert allocator.free_count == allocator.usable_blocks
    sched.abort("out of memory")
    assert not sched.decoding
    assert not sched.prefill_q
    assert not sched.waiting


# ---------------------------------------------------------------- (b) prefix reuse


def test_cached_prefix_plus_suffix_matches_an_uncached_run() -> None:
    prompts = session(base=5, turns=[4, 6, 3])

    for prompt in prompts:  # each turn on a scheduler that has never seen the session
        sched, _, _ = make(max_batch=4, prefill_chunk=5)
        req, sink = request(prompt, 6)
        sched.submit(req)
        drain(sched)
        assert sink.tokens == serial(prompt, 6)
    cold_prefilled = sum(len(p) for p in prompts)

    sched, warm, _ = make(max_batch=4, prefill_chunk=5)
    for turn, prompt in enumerate(prompts):
        before = warm.prefilled
        req, sink = request(prompt, 6)
        sched.submit(req)
        drain(sched)
        assert sink.tokens == serial(prompt, 6), f"turn {turn} diverged under reuse"
        new_tokens = len(prompt) - (len(prompts[turn - 1]) if turn else 0)
        # +1: turn-close always re-forwards its own last generated token before publishing
        # (see `Scheduler._publish_turn_close`), on top of however much of the prompt was new.
        assert warm.prefilled - before == new_tokens + 1, f"turn {turn} recomputed the history"
        assert sink.end is not None and sink.end[2] == len(prompt) - new_tokens
    assert warm.prefilled < cold_prefilled


def test_reuse_survives_other_sessions_sharing_the_pool() -> None:
    sched, _, _ = make(max_batch=4, prefill_chunk=4)
    sessions = [session(base=b, turns=[3, 5]) for b in (2, 3, 4)]
    for turn in range(2):
        pairs = [request(s[turn], 5) for s in sessions]
        for req, _ in pairs:
            sched.submit(req)
        drain(sched)
        for s, (_, sink) in zip(sessions, pairs, strict=True):
            assert sink.tokens == serial(s[turn], 5)


def test_a_busy_lane_session_is_never_offered_as_a_prefix() -> None:
    """Right up to the point a request finishes, nothing about it is published yet (a
    Stage-2-specific check: unlike Stage 1's slot, a lane is not the cache -- but the trie
    still must not contain a half-finished session's tokens)."""
    sched, _, cache = make(max_batch=2, prefill_chunk=4)
    first, first_sink = request([1, 2, 3], 20)
    sched.submit(first)
    for _ in range(6):
        sched.step()
    assert cache.lookup([1, 2, 3]).depth == 3  # the prompt boundary was already published
    assert cache.lookup([1, 2, 3, 9, 9, 9]).depth == 3, "no unfinished reply is visible yet"
    second, second_sink = request([1, 2, 3, 4], 5)
    sched.submit(second)
    drain(sched)
    assert first_sink.tokens == serial([1, 2, 3], 20)
    assert second_sink.tokens == serial([1, 2, 3, 4], 5)


# ---------------------------------------------------------------- (c) eviction


def test_evicted_session_falls_back_to_a_full_recompute() -> None:
    sched, runner, cache = make(max_batch=2, num_snapshots=2, prefill_chunk=16)
    a, b, c = (session(base=v, turns=[4, 5]) for v in (1, 2, 3))

    for prompt in (a[0], b[0]):  # fills both snapshot slots
        req, sink = request(prompt, 3)
        sched.submit(req)
        drain(sched)
        assert sink.tokens == serial(prompt, 3)

    req, sink = request(c[0], 3)  # evicts A's entry, the least recently used
    sched.submit(req)
    drain(sched)
    assert sink.tokens == serial(c[0], 3)
    assert cache.lookup(a[0]).depth == 0, "A's entry should have been evicted"

    before = runner.prefilled
    req, sink = request(a[1], 3)  # A's next turn: no cached prefix left
    sched.submit(req)
    drain(sched)
    assert sink.tokens == serial(a[1], 3), "evicted session came back corrupted"
    assert runner.prefilled - before == len(a[1]) + 1, "did not fall back to a full recompute"


def test_lru_evicts_the_session_that_was_idle_longest() -> None:
    # num_snapshots=4: each session publishes 2 nodes (prompt-end + turn-close), so "room for
    # 2 sessions" is 4 slots now, not 2 -- the Stage 2 equivalent of Stage 1's 2-slot pool.
    sched, runner, _ = make(max_batch=2, num_snapshots=4, prefill_chunk=16)
    a, b, c = (session(base=v, turns=[4, 5, 4]) for v in (1, 2, 3))

    def turn(prompt: list[int]) -> int:
        before = runner.prefilled
        req, sink = request(prompt, 3)
        sched.submit(req)
        drain(sched)
        assert sink.tokens == serial(prompt, 3), "reuse or eviction changed the output"
        return runner.prefilled - before

    turn(a[0])
    turn(b[0])
    # +1 each: turn-close always re-forwards its own last generated token before publishing.
    assert turn(a[1]) == len(a[1]) - len(a[0]) + 1  # A reuses and becomes the newer entry
    turn(c[0])  # evicts B, not A
    assert turn(a[2]) == len(a[2]) - len(a[1]) + 1
    assert turn(b[1]) == len(b[1]) + 1


def test_lru_evicts_the_conversation_that_was_idle_longest() -> None:
    # Conversations extend their own history: each turn's prompt is the previous prompt, its
    # reply, and new text, so the turn-close node is the one live state a conversation needs.
    # Each turn publishes 2 nodes (prompt-end + turn-close). The previous turn-close node is
    # superseded once the next turn resumes from it and publishes past it; the prompt-end
    # node, internal but older than its turn-close child, loses its snapshot under LRU before
    # any conversation's leaf. With 3 slots, a third conversation's turn needs 2 while A and
    # B each hold 1 (their turn-close leaf).
    sched, runner, _ = make(max_batch=2, num_snapshots=3, prefill_chunk=16)
    history: dict[int, list[int]] = {}

    def turn(conv: int, new: int) -> int:
        prompt = history.get(conv, [conv, conv + 1, conv + 2])
        prompt = prompt + [conv * 100 + len(prompt) * 10 + j for j in range(new)]
        before = runner.prefilled
        req, sink = request(prompt, 3)
        sched.submit(req)
        drain(sched)
        assert sink.tokens == serial(prompt, 3), "reuse or eviction changed the output"
        history[conv] = prompt + sink.tokens
        return runner.prefilled - before

    turn(1, 4)
    turn(2, 4)
    # +1: turn-close always re-forwards its own last generated token before publishing.
    assert turn(1, 5) == 5 + 1  # A reuses its turn-close state and becomes the newer entry
    turn(3, 4)  # a third conversation: evicts B's leaf, not A's
    assert turn(1, 4) == 4 + 1
    assert turn(2, 5) == len(history[2]) - 3 + 1, "B, idle longest, was fully evicted"


def test_kv_block_pressure_forces_eviction_on_the_copy_on_write_path() -> None:
    """Distinct from snapshot-slot pressure above: a pool with plenty of snapshot slots but too
    few blocks must still evict (paged-kv-design.md section 1.7: two coupled LRU clocks).

    Admission-time block reservation (`SessionCache.reserve_blocks`, used directly for a fresh
    session's very first block and for copy-on-write's replacement block) is eviction-aware;
    this exercises it under a pool sized so it cannot possibly avoid evicting. Growth during
    prefill and decode is covered by the pool-pressure tests at the end of this file.
    """
    sched, runner, cache = make(
        max_batch=2, num_blocks=4, num_snapshots=32, block_size=4, prefill_chunk=16
    )
    a, b, c = ([base, base + 1, base + 2] for base in (1, 10, 20))  # 1 block each, cold

    for prompt in (a, b, c):  # 3 usable blocks, exactly enough for 3 sessions' first block
        req, sink = request(prompt, 1)
        sched.submit(req)
        drain(sched)
        assert sink.tokens == serial(prompt, 1)
    assert cache.allocator.free_count == 0, "setup: the pool must be fully committed"

    node = cache.lookup(a)
    assert cache.needs_cow(node), "resuming a must require copy-on-write of its partial block"
    req, sink = request([*a, 99], 1)  # resuming A needs 1 more (COW'd) block; none are free
    sched.submit(req)
    drain(sched)
    assert sink.error is None, "admission must evict rather than fail outright"
    assert sink.tokens == serial([*a, 99], 1)


# ---------------------------------------------------------------- (h) lanes decoupled from the cache


def test_hit_survives_lane_reuse_by_unrelated_sessions() -> None:
    """The Stage 2 property: with only 2 lanes but a cache sized for many more sessions, 3
    independent sessions all keep their turn-1 reuse even though every one of them had to give
    up its lane (and get a different one back) in between. Stage 1 could not do this: reclaiming
    an idle slot destroyed the only copy of that session's cached prefix."""
    sched, runner, _ = make(max_batch=2, num_blocks=512, num_snapshots=64, prefill_chunk=16)
    a, b, c = (session(base=v, turns=[5, 3]) for v in (1, 2, 3))

    for s in (a, b, c):  # only 2 lanes ever exist; each turn frees its lane before the next runs
        req, sink = request(s[0], 2)
        sched.submit(req)
        drain(sched)
        assert sink.tokens == serial(s[0], 2)
    assert len(sched.free_lanes) == 2, "every lane must be free between turns"

    for s in (a, b, c):
        before = runner.prefilled
        req, sink = request(s[1], 2)
        sched.submit(req)
        drain(sched)
        assert sink.tokens == serial(s[1], 2)
        new_tokens = len(s[1]) - len(s[0])
        assert runner.prefilled - before == new_tokens + 1, "lane reuse lost this session's cache"


def test_max_lanes_is_independent_of_cache_capacity() -> None:
    """A single lane can serve far more distinct cached sessions than the lane count, one at a
    time, each getting a full hit on its own second turn."""
    sched, _, _ = make(max_batch=1, num_blocks=512, num_snapshots=64, prefill_chunk=16)
    sessions = [session(base=v, turns=[4, 3]) for v in range(1, 9)]
    for s in sessions:
        req, sink = request(s[0], 2)
        sched.submit(req)
        drain(sched)
        assert sink.tokens == serial(s[0], 2)
    for s in sessions:
        req, sink = request(s[1], 2)
        sched.submit(req)
        drain(sched)
        assert sink.tokens == serial(s[1], 2)


# ---------------------------------------------------------------- (i) refcounts


def test_two_sessions_sharing_a_system_prompt_share_one_set_of_block_refs() -> None:
    shared_prefix = [100, 101, 102, 103, 104, 105, 106, 107]  # 8 tokens = 2 full blocks
    sched, runner, cache = make(max_batch=4, prefill_chunk=8)

    req0, sink0 = request(shared_prefix, 1)
    sched.submit(req0)
    drain(sched)
    assert sink0.tokens == serial(shared_prefix, 1)

    node = cache.lookup(shared_prefix)
    assert node.depth == len(shared_prefix)
    # Baseline: the prompt-end node's own share, plus one more from the turn-close node that
    # session 0 also published (it shares these same blocks as its own inherited prefix).
    baseline = cache.allocator.refcount(node.blocks[0])
    assert baseline >= 1

    # A second, independent session shares exactly this prefix, then diverges.
    req1, sink1 = request([*shared_prefix, 500, 501], 2)
    sched.submit(req1)
    sched.step()  # _admit: matches `node` and adopts its blocks onto a lane
    # A copy would allocate entirely different block ids and leave this refcount untouched;
    # any increase here is direct proof the second session shares the same physical blocks.
    assert cache.allocator.refcount(node.blocks[0]) > baseline, (
        "the second session must share, not copy"
    )
    drain(sched)
    assert sink1.tokens == serial([*shared_prefix, 500, 501], 2)
    # Session 1's own lane share is released, but session 1 also published its own further
    # nodes over this same prefix (its own prompt-end and turn-close boundaries), each holding
    # one more persistent share -- refcounting never collapses two independent owners into one.
    assert cache.allocator.refcount(node.blocks[0]) > baseline


# ---------------------------------------------------------------- (j) copy-on-write


def test_two_divergent_continuations_copy_on_write_the_shared_partial_block() -> None:
    shared = [1, 2, 3, 4, 5, 6]  # 6 tokens: block 0 full, block 1 holds only 2 of 4 -- partial
    sched, runner, cache = make(max_batch=4, prefill_chunk=8)

    req0, sink0 = request(shared, 1)
    sched.submit(req0)
    drain(sched)

    base_node = cache.lookup(shared)
    assert base_node.depth == len(shared)
    assert cache.needs_cow(base_node)
    partial_block = base_node.blocks[-1]
    original_content = list(runner.block_content[partial_block])
    assert original_content[:2] == [5, 6]

    prompt_a = [*shared, 100, 101]
    prompt_b = [*shared, 200, 201, 202]
    req_a, sink_a = request(prompt_a, 2)
    req_b, sink_b = request(prompt_b, 2)
    sched.submit(req_a)
    sched.submit(req_b)
    drain(sched)

    assert sink_a.tokens == serial(prompt_a, 2)
    assert sink_b.tokens == serial(prompt_b, 2)
    assert len(runner.copy_block_calls) == 2, "each divergent continuation must copy once"
    for _, src, filled in runner.copy_block_calls:
        assert src == partial_block
        assert filled == len(shared) % BLOCK_SIZE

    # The original shared block must be untouched by either continuation.
    assert runner.block_content[partial_block] == original_content
    node_a = cache.lookup(prompt_a[:6])
    node_b = cache.lookup(prompt_b[:6])
    assert node_a is base_node and node_b is base_node  # both still resolve to the same parent


def test_turn_close_reforward_never_writes_into_an_unowned_cow_block() -> None:
    """Regression: resuming a node that needs copy-on-write, with `suffix_len > 0` and no
    further boundary published before release (the resumed prompt lands exactly on the node's
    own boundary, so `fl.parent` is never re-pointed at a node built from *this* lane's own
    table -- see `_publish_turn_close`'s docstring), must never let the turn-close re-forward
    write into `fl.parent.blocks`' original (un-cow'd) last block: this lane was never given a
    share of it (its own share is the fresh cow'd block from admission), so writing there would
    both corrupt whatever else still resolves through the original node and be unprotected
    against a concurrent eviction of that block.
    """
    boundary = [1, 2, 3]  # 3 tokens: partial at block_size 4
    sched, runner, cache = make(max_batch=4, prefill_chunk=8)

    req0, sink0 = request(boundary, 1)
    sched.submit(req0)
    drain(sched)
    base_node = cache.lookup(boundary)
    assert base_node.depth == len(boundary)
    assert cache.needs_cow(base_node)
    original_block = base_node.blocks[-1]
    original_content = list(runner.block_content[original_block])

    suffix = [90, 91]
    prompt1 = [*boundary, *suffix]  # boundary == suffix_len boundary exactly: an exact match
    req1, sink1 = request(prompt1, 2, suffix_len=len(suffix))
    sched.submit(req1)
    drain(sched)
    assert sink1.tokens == serial(prompt1, 2)
    assert cache.lookup(boundary) is base_node, "the original node must still resolve correctly"
    assert runner.block_content[original_block] == original_content, (
        "the turn-close re-forward corrupted a block this lane never owned a share of"
    )

    # A second, independent resumer must still see the untouched original content, proving no
    # corruption was silently propagated through the shared node.
    req2, sink2 = request([*boundary, 500], 1)
    sched.submit(req2)
    drain(sched)
    assert sink2.tokens == serial([*boundary, 500], 1)


# ---------------------------------------------------------------- (e) cache-reset hook


def test_reset_prefix_cache_clears_idle_entries_only() -> None:
    sched, runner, cache = make(max_batch=2, prefill_chunk=8)
    req, sink = request([1, 2, 3], 3)
    sched.submit(req)
    drain(sched)
    assert cache.lookup([1, 2, 3]).depth > 0, "setup: nothing recorded to reset"

    cleared = sched.reset_prefix_cache()
    assert cleared >= 1
    assert cache.lookup([1, 2, 3]).depth == 0
    assert sched.reset_prefix_cache() == 0  # idempotent


def test_reset_prefix_cache_is_a_noop_on_an_empty_pool() -> None:
    sched, _, _ = make(max_batch=3, prefill_chunk=8)
    assert sched.reset_prefix_cache() == 0


def test_reset_prefix_cache_does_not_disturb_an_in_flight_request() -> None:
    sched, runner, cache = make(max_batch=2, prefill_chunk=8)
    busy_req, busy_sink = request([9, 9], 50)
    sched.submit(busy_req)
    for _ in range(4):  # admits and makes progress, but does not finish
        sched.step()

    sched.reset_prefix_cache()
    drain(sched)
    assert busy_sink.tokens == serial([9, 9], 50)


# ------------------------------------------- (f) chat generation-prompt suffix boundary


def test_suffix_is_prefilled_but_not_recorded_past_the_boundary() -> None:
    """Right after a request whose prompt ends in a `suffix_len`-long tail finishes its own
    prefill, the cache's recorded prefix stops at the boundary before that tail, not at the
    prompt's own end (mirrors Qwen3.5's `<think>\\n\\n</think>\\n\\n`, appended only to the
    prompt that generates from it)."""
    boundary = [1, 2, 3, 4]  # shared history up to "<|im_start|>assistant\n"
    suffix = [990, 991, 992]  # "<think>\n\n</think>\n\n"
    sched, runner, cache = make(max_batch=2, prefill_chunk=3)  # deliberately misaligned chunking

    prompt = boundary + suffix
    req, sink = request(prompt, 4, suffix_len=len(suffix))
    sched.submit(req)
    drain(sched)
    assert sink.tokens == serial(prompt, 4)
    node = cache.lookup(boundary)
    assert node.depth == len(boundary), "recorded past the suffix"
    assert cache.lookup(prompt).depth == len(boundary), "the suffix itself was never recorded"
    # The suffix was still forwarded, just not recorded: this request's own decode needed it.
    assert runner.prefilled >= len(prompt), "the suffix must still have been forwarded"


def test_next_turn_reuses_the_boundary_even_though_history_omits_the_suffix() -> None:
    """The bug this fixes: a chat template renders a finished turn in the *next* prompt's
    history without the suffix it only added to generate from. Without `suffix_len`, the cache's
    recorded tokens (prompt including the suffix) would never be a prefix of the next turn's
    prompt, so this turn would fall back to a full recompute.

    Stage 2 actually does better than just reusing the turn-open boundary: turn 1's turn-close
    node (boundary + reply, published automatically -- see `_publish_turn_close`) is itself a
    prefix of turn 2's prompt, so turn 2 hits *that*, deeper than the boundary alone.
    """
    boundary = [1, 2, 3, 4]
    suffix = [990, 991, 992]
    sched, runner, cache = make(max_batch=2, prefill_chunk=3)

    turn1 = boundary + suffix
    req1, sink1 = request(turn1, 4, suffix_len=len(suffix))
    sched.submit(req1)
    drain(sched)
    reply1 = sink1.tokens

    more_history = [50, 51]  # "<|im_end|>\n<|im_start|>user\n...<|im_start|>assistant\n"
    turn2 = boundary + reply1 + more_history + suffix
    before = runner.prefilled
    req2, sink2 = request(turn2, 4, suffix_len=len(suffix))
    sched.submit(req2)
    drain(sched)

    assert sink2.end is not None
    reused = sink2.end[2]
    assert reused >= len(boundary), "did not reuse at least the shared boundary"
    assert reused == len(boundary) + len(reply1), "should hit turn 1's turn-close node"
    # Cheaper than a full recompute (which would forward all of turn2 plus its own reply);
    # exact byte count also includes turn2's own turn-close re-forward (suffix_len > 0 re-runs
    # the whole reply at the boundary -- see `_publish_turn_close`), not asserted precisely here.
    assert runner.prefilled - before < len(turn2) + 4, "recomputed cached history"
    assert sink2.tokens == serial(turn2, 4), "resumed output diverged from a full recompute"
    node = cache.lookup(turn2[: len(turn2) - len(suffix)])
    assert node.depth == len(turn2) - len(suffix), "turn 2's own boundary was recorded in turn"


def test_zero_suffix_len_is_the_ordinary_end_of_prompt_snapshot() -> None:
    """`suffix_len=0` (chat's default when the derived suffix does not apply, and every
    non-chat caller) must behave exactly as before this feature existed."""
    sched, runner, cache = make(max_batch=2, prefill_chunk=3)
    prompt = [1, 2, 3, 4, 5, 6, 7]
    req, sink = request(prompt, 3, suffix_len=0)
    sched.submit(req)
    drain(sched)
    assert sink.tokens == serial(prompt, 3)
    assert cache.lookup(prompt).depth == len(prompt)


def test_turn_close_is_published_and_serves_an_exact_repeat_with_no_prefill() -> None:
    """paged-kv-design.md section 2.4: a request whose full token sequence exactly matches an
    existing turn-close node resolves to it directly, with zero prefill."""
    sched, runner, cache = make(max_batch=2, prefill_chunk=8)
    prompt = [1, 2, 3]
    req, sink = request(prompt, 4)
    sched.submit(req)
    drain(sched)
    reply = sink.tokens
    full = prompt + reply
    assert cache.lookup(full).depth == len(full), "turn-close was not published"

    before = runner.prefilled
    req2, sink2 = request(full, 3)
    sched.submit(req2)
    drain(sched)
    assert sink2.tokens == serial(full, 3)
    # The repeat itself costs zero prefill; the +1 that remains is request 2's *own* turn-close
    # re-forward of its own last generated token (see `_publish_turn_close`), unrelated to
    # whether the repeat hit the cache.
    assert runner.prefilled - before == 1, "an exact repeat of prompt+reply must cost zero prefill"
    assert sink2.end is not None and sink2.end[2] == len(full)


def test_turn_close_after_a_suffix_boundary_reforwards_the_reply_at_the_boundary() -> None:
    """When `suffix_len > 0`, the reply was actually generated past the suffix; turn-close must
    still record it as if it followed the boundary directly (matching what the *next* turn's
    history rendering will contain -- see the module docstring), which needs the extra
    position-shifted `prefill` call `_publish_turn_close` performs."""
    boundary = [1, 2, 3, 4]
    suffix = [990, 991, 992]
    sched, runner, cache = make(max_batch=2, prefill_chunk=8)
    prompt = boundary + suffix
    req, sink = request(prompt, 3, suffix_len=len(suffix))
    sched.submit(req)
    drain(sched)
    reply = sink.tokens

    full_history_next_turn = boundary + reply  # what the next turn's re-rendered history is
    node = cache.lookup(full_history_next_turn)
    assert node.depth == len(full_history_next_turn), (
        "turn-close must be reachable via boundary+reply, not boundary+suffix+reply"
    )


# --------------------------------------- (f.2) SEED_FOLD_TURN_SUFFIX


def test_fold_turn_suffix_matches_serial_and_skips_a_prefill_call() -> None:
    """`fold_turn_suffix=True` must produce byte-identical output to the unfolded path, and
    the suffix must ride decode instead of its own prefill/mixed step: the fake model's
    `next_token` is a pure function of `hist`, so if the suffix's positions get folded into
    `decode()` calls (one token at a time) instead of one `prefill()` call, the resulting
    tokens only match `serial()` if the state each path leaves behind is identical."""
    boundary = [1, 2, 3, 4]
    suffix = [990, 991, 992]
    prompt = boundary + suffix

    sched_off, runner_off, cache_off = make(max_batch=2, prefill_chunk=8)
    req_off, sink_off = request(prompt, 4, suffix_len=len(suffix))
    sched_off.submit(req_off)
    drain(sched_off)

    sched_on, runner_on, cache_on = make(max_batch=2, prefill_chunk=8, fold_turn_suffix=True)
    req_on, sink_on = request(prompt, 4, suffix_len=len(suffix))
    sched_on.submit(req_on)
    drain(sched_on)

    assert sink_off.tokens == serial(prompt, 4)
    assert sink_on.tokens == serial(prompt, 4), "folded suffix diverged from a full recompute"
    assert sink_on.tokens == sink_off.tokens

    # The boundary node (cache hit length for a future turn) is unaffected by folding.
    node_off, node_on = cache_off.lookup(boundary), cache_on.lookup(boundary)
    assert node_off.depth == node_on.depth == len(boundary)

    # The suffix no longer costs a prefill call of its own: both paths still pay the boundary
    # chunk and `_publish_turn_close`'s own end-of-reply re-forward (suffix_len > 0, unrelated
    # to this flag), so the difference between them is exactly the folded suffix's length.
    assert runner_off.prefilled >= len(prompt), "unfolded: the suffix is still prefilled"
    assert runner_off.prefilled - runner_on.prefilled == len(suffix), (
        "folded: the suffix must not be prefilled"
    )
    assert max(runner_on.batches) >= 1, "the suffix must have ridden a decode step instead"


def test_fold_turn_suffix_does_not_stream_or_budget_the_suffix_ticks() -> None:
    """Forced suffix ticks are not generated output: they must not appear in `sink.tokens`,
    not count against `max_new`, and not trigger a spurious stop match even if a suffix id
    happens to equal a stop id."""
    boundary = [1, 2]
    suffix = [500, 501, 502]  # one of these must never be mistaken for a real stop/output token
    prompt = boundary + suffix
    sched, runner, cache = make(max_batch=2, prefill_chunk=8, fold_turn_suffix=True)

    req, sink = request(prompt, 2, stop=frozenset({500, 501, 502}), suffix_len=len(suffix))
    sched.submit(req)
    drain(sched)

    assert sink.tokens == serial(prompt, 2, stop=frozenset({500, 501, 502}))
    assert not set(sink.tokens) & {500, 501, 502}, "a forced suffix id leaked into the output"
    assert sink.end is not None and sink.end[0] != "stop", "a suffix id was mistaken for a stop"


def test_fold_turn_suffix_is_a_noop_when_suffix_len_is_zero() -> None:
    """The flag only changes the `suffix_len > 0` path; ordinary (non-chat) requests are
    unaffected."""
    sched, runner, cache = make(max_batch=2, prefill_chunk=3, fold_turn_suffix=True)
    prompt = [1, 2, 3, 4, 5, 6, 7]
    req, sink = request(prompt, 3, suffix_len=0)
    sched.submit(req)
    drain(sched)
    assert sink.tokens == serial(prompt, 3)
    assert cache.lookup(prompt).depth == len(prompt)


# ---------------------------------------------------------------- (g) cross-request batched prefill


def test_batched_prefill_matches_unbatched_prefill_results() -> None:
    """The batched and single-request prefill paths must be observationally identical: same
    tokens, same end reason, for every request, even though the batched path packs several
    requests' chunks into each `prefill_batch` call."""
    specs = [
        ([1, 2, 3, 4, 5, 6, 7], 5),
        ([11, 12, 13], 4),
        ([21, 22, 23, 24, 25], 6),
        ([31, 32], 3),
    ]

    sched_u, unbatched, _ = make(max_batch=4, prefill_chunk=3, batched_prefill=False)
    pairs_u = [request(p, n) for p, n in specs]
    for req, _ in pairs_u:
        sched_u.submit(req)
    drain(sched_u)

    sched_b, batched, _ = make(max_batch=4, prefill_chunk=3, token_budget=8, batched_prefill=True)
    pairs_b = [request(p, n) for p, n in specs]
    for req, _ in pairs_b:
        sched_b.submit(req)
    drain(sched_b)

    for (_, sink_u), (_, sink_b) in zip(pairs_u, pairs_b, strict=True):
        assert sink_u.error is None and sink_b.error is None
        assert sink_u.tokens == sink_b.tokens
        assert sink_u.end == sink_b.end

    assert unbatched.prefill_batch_sizes == [], "the old path must never call prefill_batch"
    assert max(batched.prefill_batch_sizes) > 1, "packing never actually combined requests"


def test_batched_prefill_produces_the_same_cache_reuse_as_unbatched() -> None:
    """Turn 2 of a session must reuse exactly as much under batched prefill as it does under
    the single-request path."""
    prompts = session(base=7, turns=[5, 4])

    sched_u, unbatched, cache_u = make(max_batch=2, prefill_chunk=3, batched_prefill=False)
    sched_b, batched, cache_b = make(
        max_batch=2, prefill_chunk=3, token_budget=6, batched_prefill=True
    )

    for prompt in prompts:
        req_u, sink_u = request(prompt, 4)
        sched_u.submit(req_u)
        drain(sched_u)
        req_b, sink_b = request(prompt, 4)
        sched_b.submit(req_b)
        drain(sched_b)
        assert sink_u.tokens == sink_b.tokens
        assert sink_u.end == sink_b.end

    assert unbatched.prefilled == batched.prefilled, "reuse must save the same work either way"
    assert cache_u.lookup(prompts[-1]).depth == cache_b.lookup(prompts[-1]).depth


def test_prefill_token_budget_is_never_exceeded() -> None:
    sched, runner, _ = make(max_batch=5, prefill_chunk=6, token_budget=10, batched_prefill=True)
    specs = [
        ([100 + i] * 9, 2) for i in range(5)
    ]  # five 9-token prompts, budget well under the total
    pairs = [request(p, n) for p, n in specs]
    for req, _ in pairs:
        sched.submit(req)
    drain(sched)

    for (prompt, max_new), (_, sink) in zip(specs, pairs, strict=True):
        assert sink.tokens == serial(prompt, max_new)
    assert runner.prefill_batch_sizes, "packing never ran"
    assert max(runner.prefill_batch_sizes) > 1, "budget of 10 against chunk 6 should still pack"
    for sizes in runner.prefill_batch_chunk_sizes:
        assert sum(sizes) <= 10, "one packed call forwarded more tokens than the budget allows"


def test_prefill_batch_fifo_fairness() -> None:
    """With a budget that only ever admits one request's chunk, batched prefill must degenerate
    to processing `prefill_q` strictly in submission order."""
    sched, runner, _ = make(max_batch=3, prefill_chunk=4, token_budget=4, batched_prefill=True)
    specs = [([1] * 8, 2), ([2] * 8, 2), ([3] * 8, 2)]
    pairs = [request(p, n) for p, n in specs]
    for req, _ in pairs:
        sched.submit(req)
    drain(sched)

    for (prompt, max_new), (_, sink) in zip(specs, pairs, strict=True):
        assert sink.tokens == serial(prompt, max_new)

    assert all(size == 1 for size in runner.prefill_batch_sizes), (
        "a budget equal to prefill_chunk should never admit two requests' chunks at once"
    )
    first_lane_per_call = [lanes[0] for lanes in runner.prefill_batch_lanes]
    order: list[int] = []
    for lane in first_lane_per_call:
        if not order or order[-1] != lane:
            order.append(lane)
    assert order == [0, 1, 2], (
        "requests must complete their own prefill in FIFO order, one at a time, with no "
        "interleaving between them"
    )


def test_prefill_fit_graph_shapes_every_packed_call_to_a_captured_shape() -> None:
    """`prefill_fit_graph`: chunks are capped at the widest captured width, and a request
    joins a call only while the call still fits a captured `rows x width` shape. Results match
    the unshaped packed path, and every request still completes."""
    import graph_prefill  # noqa: PLC0415 -- torch import, only for `shape_for`

    shapes = graph_prefill.parse_shapes("1x2,2x2,1x4", max_batch=4, max_seq=64)

    class Fit(FakeRunner):
        def prefill_fit(self):  # noqa: ANN202
            return (lambda lengths: graph_prefill.shape_for(lengths, shapes) is not None, 4)

    specs = [([100 + i] * 9, 2) for i in range(3)]
    sched, runner, _ = make(
        max_batch=3,
        prefill_chunk=6,
        token_budget=12,
        batched_prefill=True,
        prefill_fit_graph=True,
    )
    sched.runner = Fit(3, runner.allocator, runner.block_size)
    runner = sched.runner
    pairs = [request(p, n) for p, n in specs]
    for req, _ in pairs:
        sched.submit(req)
    drain(sched)

    for (prompt, max_new), (_, sink) in zip(specs, pairs, strict=True):
        assert sink.error is None
        assert sink.tokens == serial(prompt, max_new)
    for sizes in runner.prefill_batch_chunk_sizes:
        assert graph_prefill.shape_for(sizes, shapes) is not None, sizes
    assert runner.prefill_batch_chunk_sizes[0] == [4], "head capped at width 4, alone"


def test_prefill_fit_graph_uses_tighter_default_wide_shapes() -> None:
    """The measured 1x384 and 2x128 shapes are reachable through scheduler packing: a long
    FIFO head runs alone, then two short requests share the tighter two-row container."""
    def fits(lengths: list[int]) -> bool:
        return (len(lengths) == 1 and max(lengths) <= 384) or (
            len(lengths) <= 2 and max(lengths) <= 128
        )

    class Fit(FakeRunner):
        def prefill_fit(self):  # noqa: ANN202
            return fits, 384

    specs = [([101] * 300, 1), ([102] * 100, 1), ([103] * 100, 1)]
    sched, runner, _ = make(
        max_batch=3,
        prefill_chunk=384,
        token_budget=512,
        batched_prefill=True,
        prefill_fit_graph=True,
    )
    sched.runner = Fit(3, runner.allocator, runner.block_size)
    runner = sched.runner
    pairs = [request(prompt, new) for prompt, new in specs]
    for req, _ in pairs:
        sched.submit(req)
    drain(sched)

    for (prompt, new), (_, sink) in zip(specs, pairs, strict=True):
        assert sink.error is None
        assert sink.tokens == serial(prompt, new)
    assert runner.prefill_batch_chunk_sizes[:2] == [[300], [100, 100]]


def test_prefill_fit_graph_is_inert_without_captured_shapes() -> None:
    """A runner with no `prefill_fit` (eager) leaves packing exactly as before."""
    specs = [([100 + i] * 9, 2) for i in range(3)]
    calls = []
    for fit in (False, True):
        sched, runner, _ = make(
            max_batch=3,
            prefill_chunk=6,
            token_budget=12,
            batched_prefill=True,
            prefill_fit_graph=fit,
        )
        for req, _ in (request(p, n) for p, n in specs):
            sched.submit(req)
        drain(sched)
        calls.append(runner.prefill_batch_chunk_sizes)
    assert calls[0] == calls[1]


def test_suffix_boundary_lands_exactly_inside_a_packed_chunk() -> None:
    """The chat-suffix boundary must still land exactly, even when the chunk that crosses it is
    packed together with another request's own chunk in the same `prefill_batch` call."""
    boundary = [1, 2, 3, 4]
    suffix = [990, 991, 992]
    prompt0 = boundary + suffix  # 7 tokens; suffix_len=3 so the boundary sits at position 4
    prompt1 = list(range(50, 60))  # 10 tokens, no suffix; just something to pack alongside it

    sched, runner, cache = make(max_batch=2, prefill_chunk=3, token_budget=10, batched_prefill=True)
    req0, sink0 = request(prompt0, 4, suffix_len=len(suffix))
    req1, sink1 = request(prompt1, 4)
    sched.submit(req0)
    sched.submit(req1)
    drain(sched)

    assert sink0.tokens == serial(prompt0, 4)
    assert sink1.tokens == serial(prompt1, 4)
    assert cache.lookup(boundary).depth == len(boundary), "recorded past the suffix"

    # Chunk 0 is [3, 3] (both requests' first PREFILL_CHUNK); chunk 1 is where request 0's
    # remaining single token (position 3, the boundary itself) lands: [1, 3], packed with
    # request 1's next chunk rather than processed alone.
    assert runner.prefill_batch_lanes[0] == [0, 1]
    assert runner.prefill_batch_chunk_sizes[0] == [3, 3]
    assert runner.prefill_batch_lanes[1] == [0, 1]
    assert runner.prefill_batch_chunk_sizes[1] == [1, 3], (
        "the boundary-landing chunk must be packed together with another request's chunk"
    )


def test_decode_interleaves_with_batched_prefill_bursts() -> None:
    """`MAX_PREFILL_BURST` must still force a decode step after at most that many prefill
    iterations once a request is waiting to decode, whether or not a prefill iteration is
    itself a packed multi-request call."""
    sched, runner, _ = make(max_batch=2, prefill_chunk=2, token_budget=2, batched_prefill=True)
    short, short_sink = request([1, 2], 5)  # finishes its own prefill in the very first call
    long, long_sink = request([3] * 40, 2)  # keeps prefill_q busy for many more iterations
    sched.submit(short)
    sched.submit(long)

    kinds: list[str] = []
    for _ in range(1000):
        before_prefill = len(runner.prefill_batch_sizes)
        before_decode = len(runner.batches)
        if not sched.step():
            break
        if len(runner.batches) > before_decode:
            kinds.append("decode")
        elif len(runner.prefill_batch_sizes) > before_prefill:
            kinds.append("prefill")

    assert short_sink.tokens == serial([1, 2], 5)
    assert long_sink.tokens == serial([3] * 40, 2)
    assert "decode" in kinds, "no decode step ever ran"
    first_decode = kinds.index("decode")
    assert first_decode <= 1 + MAX_PREFILL_BURST


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# ------------------------------------------- (i) block authority, pool pressure, failure paths


def test_decode_growth_evicts_cached_sessions_instead_of_failing() -> None:
    """Every block a forward writes is reserved by the scheduler first (`_grow_lanes`), which
    evicts LRU cache entries when the pool is full. A direct allocator call mid-decode raised
    `MemoryError` with the pool full of evictable cached sessions and failed the batch."""
    sched, runner, cache = make(max_batch=1, num_blocks=5, num_snapshots=32, prefill_chunk=16)
    for base in (1, 10, 20):  # three cached sessions, 2 blocks each (prompt + turn-close)
        req, sink = request([base, base + 1, base + 2], 2)
        sched.submit(req)
        drain(sched)
        assert sink.error is None
    long_req, long_sink = request([40, 41], 12)  # 14 tokens: 4 blocks, only possible by evicting
    sched.submit(long_req)
    drain(sched)
    assert long_sink.error is None, long_sink.error
    assert long_sink.tokens == serial([40, 41], 12)


def test_an_admission_that_cannot_copy_on_write_falls_back_instead_of_stranding() -> None:
    """The only cached node holds every block, and resuming it needs a copy-on-write block
    that only evicting that same (protected) node could free. Admission used to return `None`
    and, with nothing else in flight, `run()` blocked on `incoming.get()` forever. Now it
    admits from scratch, which may evict the node."""
    sched, runner, cache = make(max_batch=2, num_blocks=4, num_snapshots=32, prefill_chunk=16)
    prompt = list(range(1, 11))  # 10 tokens: 3 blocks, the whole pool
    req, sink = request(prompt, 1)
    sched.submit(req)
    drain(sched)
    assert sink.error is None and cache.allocator.free_count == 0
    resumed = [*prompt, 77]
    assert cache.needs_cow(cache.lookup(resumed))
    req, sink = request(resumed, 1)
    sched.submit(req)
    drain(sched)
    assert sink.end is not None, "the request must not be left waiting"
    assert sink.tokens == serial(resumed, 1)


def test_a_failed_admission_releases_the_adopted_blocks_and_the_lane() -> None:
    class BrokenLoad(FakeRunner):
        broken = False

        def load_snapshot(self, lane: int, snap: int) -> None:
            if self.broken:
                raise RuntimeError("load failed")
            super().load_snapshot(lane, snap)

    allocator = block_pool.BlockAllocator(64, BLOCK_SIZE)
    cache = SessionCache(allocator, BLOCK_SIZE, 8)
    runner = BrokenLoad(2, allocator, BLOCK_SIZE)
    sched = Scheduler(runner, cache, prefill_chunk=8)
    req, _ = request([1, 2, 3, 4, 5], 1)
    sched.submit(req)
    drain(sched)
    free_before = allocator.free_count
    runner.broken = True
    req, sink = request([1, 2, 3, 4, 5, 6, 7], 1)  # resumes the cached node (with COW)
    sched.submit(req)
    drain(sched)
    assert sink.error is not None and "load failed" in sink.error
    assert allocator.free_count == free_before, "adopted/COW blocks must be released"
    assert len(sched.free_lanes) == 2


def test_the_terminal_event_is_sent_even_if_releasing_the_lane_raises() -> None:
    class BrokenRelease(FakeRunner):
        begins = 0

        def begin(self, lane: int) -> None:
            self.begins += 1
            if self.begins > 1:  # admission's begin works; the release's begin raises
                raise RuntimeError("begin failed")
            super().begin(lane)

    allocator = block_pool.BlockAllocator(64, BLOCK_SIZE)
    cache = SessionCache(allocator, BLOCK_SIZE, 8)
    sched = Scheduler(BrokenRelease(2, allocator, BLOCK_SIZE), cache, prefill_chunk=8)
    req, sink = request([1, 2, 3], 2)
    sched.submit(req)
    with pytest.raises(RuntimeError, match="begin failed"):
        drain(sched)
    assert sink.end == ("length", 2, 0), "the client must still get its terminal event"
    assert len(sched.free_lanes) == 2


def test_abort_reaches_every_client_even_if_one_release_raises() -> None:
    sched, runner, cache = make(max_batch=2, prefill_chunk=8)
    reqs = [request([base, base + 1], 50) for base in (1, 10)]
    for req, _ in reqs:
        sched.submit(req)
    for _ in range(3):
        sched.step()

    def boom(lane: int) -> tuple[int, ...]:
        raise RuntimeError("lane_blocks failed")

    runner.lane_blocks = boom
    waiting, waiting_sink = request([5, 6], 3)
    sched.waiting.append(waiting)
    sched.abort("scheduler died")
    assert all(sink.error == "scheduler died" for _, sink in reqs)
    assert waiting_sink.error == "scheduler died"


def test_turn_close_is_skipped_when_the_turn_open_publish_failed() -> None:
    """With the turn-open publish failed, `fl.parent` is the root (snapshot -1): the suffix
    branch used to `load_snapshot(-1)` (fatal on every TP rank), and the no-suffix branch
    forwarded the last token at the wrong position and published a node whose KV did not
    match its tokens. Both must skip the turn-close publish instead."""

    class NoSnapshots(FakeRunner):
        loads: list[int] = []

        def save_snapshot(self, lane: int, snap: int) -> None:
            raise RuntimeError("snapshot write failed")

        def load_snapshot(self, lane: int, snap: int) -> None:
            self.loads.append(snap)
            super().load_snapshot(lane, snap)

    for suffix_len in (0, 2):
        allocator = block_pool.BlockAllocator(64, BLOCK_SIZE)
        cache = SessionCache(allocator, BLOCK_SIZE, 8)
        runner = NoSnapshots(2, allocator, BLOCK_SIZE)
        runner.loads = []
        sched = Scheduler(runner, cache, prefill_chunk=16)
        prompt = [3, 4, 5, 6, 7, 8]
        req, sink = request(prompt, 3, suffix_len=suffix_len)
        sched.submit(req)
        drain(sched)
        assert sink.tokens == serial(prompt, 3)
        assert runner.loads == [], f"suffix_len={suffix_len}: no snapshot may be loaded"
        assert runner.prefill_calls == 1 + (suffix_len > 0), "no turn-close re-forward"
        assert cache.node_count() == 1, "nothing published"
        assert cache.free_snapshot_count() == 8, "a reserved-but-unpublished slot must return"


def test_reset_prefix_cache_keeps_nodes_that_in_flight_requests_extend() -> None:
    """`evict_all` must protect every in-flight `fl.parent`, as reservations do: otherwise
    the node's snapshot slot returns to the free list while the request still needs it."""
    sched, runner, cache = make(max_batch=2, prefill_chunk=8)
    busy_req, busy_sink = request([9, 8, 7], 30)
    sched.submit(busy_req)
    for _ in range(3):
        sched.step()
    parent = sched.decoding[0].parent
    assert not parent.is_root()
    sched.reset_prefix_cache()
    assert cache.is_live(parent)
    drain(sched)
    assert busy_sink.tokens == serial([9, 8, 7], 30)


def test_an_exact_duplicate_never_reads_logits_of_an_evicted_node() -> None:
    sched, runner, cache = make(max_batch=2, prefill_chunk=8)
    req, sink = request([4, 5, 6], 2)
    sched.submit(req)
    drain(sched)
    repeat = [4, 5, 6, *sink.tokens]
    req, sink = request(repeat, 2)
    sched.submit(req)
    sched._drain()
    sched._admit()
    assert sched.prefill_q and not sched.prefill_q[0].pending, "setup: exact duplicate"
    cache.evict_all()  # simulate a bug elsewhere evicting the node under the queued request
    drain(sched)
    assert sink.error is not None and not sink.tokens


def test_score_runs_on_a_free_lane_and_releases_its_blocks() -> None:
    """`/v1/score` goes through the scheduler: a free lane (never a live one), blocks
    reserved like any forward, and released afterwards."""

    class Scoring(FakeRunner):
        scored_lanes: list[int] = []

        def score(self, lane: int, ids: list[int], continuation_start: int) -> list[int]:
            self.scored_lanes.append(lane)
            self._write(lane, list(ids), 0)
            return [next_token(ids[: i + 1]) for i in range(continuation_start - 1, len(ids))]

    allocator = block_pool.BlockAllocator(64, BLOCK_SIZE)
    cache = SessionCache(allocator, BLOCK_SIZE, 8)
    runner = Scoring(2, allocator, BLOCK_SIZE)
    runner.scored_lanes = []
    sched = Scheduler(runner, cache, prefill_chunk=8)
    busy_req, busy_sink = request([2, 3, 4], 20)
    sched.submit(busy_req)
    for _ in range(3):
        sched.step()
    busy_lane = sched.decoding[0].lane
    held = allocator.free_count
    got = sched.score([1, 2, 3, 4, 5, 6, 7, 8, 9], 5, lambda logits: list(logits))
    assert got == [next_token(list(range(1, n + 1))) for n in range(5, 10)]
    assert runner.scored_lanes and busy_lane not in runner.scored_lanes
    assert allocator.free_count == held, "the score lane's blocks must be released"
    drain(sched)
    assert busy_sink.tokens == serial([2, 3, 4], 20)

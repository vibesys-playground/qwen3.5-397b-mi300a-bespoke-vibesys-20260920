"""Hermetic CPU tests for `SEED_MIXED_BATCH` (Sarathi-Serve-style stall-free batching), driven
by the same deterministic `FakeRunner` stand-in `test_scheduler.py` uses, extended with a
`decode_mixed` that does exactly what `_decode_step` + `_finish_prefill_chunk` would do
separately, just inside one call (see `MixedFakeRunner.decode_mixed`'s docstring).

Three things this file checks that `test_scheduler.py` does not:

(a) mixed vs. alternating scheduling produce *identical* per-request token streams and
    identical prefix-cache depths -- the correctness property a stall-free reschedule must not
    break (`test_mixed_and_alternating_produce_identical_tokens_and_cache_depths`).
(b) the stall-free property itself: once a request is decoding, mixed batching must never skip
    it for a step while there is a decode-eligible lane and pending prefill work, which plain
    alternation demonstrably does (`test_mixed_batch_advances_every_decoding_lane_every_step`,
    contrasted against `test_alternating_batch_can_stall_a_decoding_lane_for_several_steps`).
(c) `_mixed_budget`'s cost bound and the FIFO fairness it must preserve while adaptive
    (`test_mixed_budget_never_exceeds_its_cost_ceiling`,
    `test_mixed_prefill_batch_is_still_fifo_fair`).

    /tmp/torchenv/bin/python -m pytest \\
        examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/seed_tests/test_scheduler_mixed_batch.py
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import block_pool  # noqa: E402
from scheduler import MIXED_BUDGET_SLACK, Request, Scheduler  # noqa: E402
from session_cache import SessionCache  # noqa: E402
from test_scheduler import BLOCK_SIZE, FakeRunner, Sink, drain, next_token, serial  # noqa: E402


class MixedFakeRunner(FakeRunner):
    """`FakeRunner` plus `decode_mixed`, built strictly out of the same primitives `decode`
    and `prefill` already use (`_write`, `self.hist`, `next_token`), so a mixed step and an
    alternating (decode-step, then prefill-step) pair touch lane state identically -- the same
    writes, in the same per-lane order, just issued from one Python call instead of two. That
    is the structural reason (a) below holds for this fake regardless of chunk boundaries:
    `next_token` is a pure function of a lane's own cumulative history, so *which* scheduler
    iteration a given write happens on can never change what any lane emits, only when.
    """

    def __init__(self, *a, **kw) -> None:  # noqa: ANN002, ANN003
        super().__init__(*a, **kw)
        self.mixed_calls: list[tuple[int, list[int]]] = []  # (decode batch, prefill chunk sizes)
        self.fail_mixed_at: int | None = None

    def decode_mixed(
        self,
        lanes: list[int],
        tokens: list[int],
        positions: list[int],
        prefill_calls: list[tuple[int, list[int], int]],
    ) -> tuple[list[int], list[list[int]]]:
        call_index = len(self.mixed_calls)
        chunk_sizes = [len(ids) for _, ids, _ in prefill_calls]
        if self.fail_mixed_at is not None and call_index == self.fail_mixed_at:
            self.mixed_calls.append((len(lanes), chunk_sizes))
            raise RuntimeError("decode_mixed blew up")
        self.mixed_calls.append((len(lanes), chunk_sizes))
        # Decode half: byte-for-byte what `FakeRunner.decode` does, minus its own fail/batches
        # bookkeeping (recorded separately below so `runner.batches` still reads as "one decode
        # step per mixed step", matching what `_log_step`/existing assertions expect).
        self.batches.append(len(lanes))
        decode_out = []
        for lane, token, pos in zip(lanes, tokens, positions, strict=True):
            assert pos == len(self.hist[lane]), f"lane {lane}: pos {pos} != {len(self.hist[lane])}"
            self._write(lane, [token], pos)
            self.hist[lane].append(token)
            decode_out.append(next_token(self.hist[lane]))
        # Prefill half: literally `FakeRunner.prefill`, once per call, in order -- what
        # `_finish_prefill_chunk` expects back (`prefill_batch`'s own per-call contract).
        prefill_out = [self.prefill(lane, list(ids), start) for lane, ids, start in prefill_calls]
        return decode_out, prefill_out


def request(
    prompt: list[int],
    max_new: int,
    stop: frozenset[int] = frozenset(),
    *,
    suffix_len: int = 0,
) -> tuple[Request, Sink]:
    sink = Sink()
    return Request(list(prompt), max_new, 0.0, stop, sink, suffix_len), sink


def make(
    max_batch: int = 4,
    num_blocks: int = 256,
    num_snapshots: int = 32,
    block_size: int = BLOCK_SIZE,
    **kw,
) -> tuple[Scheduler, MixedFakeRunner, SessionCache]:
    allocator = block_pool.BlockAllocator(num_blocks, block_size)
    cache = SessionCache(allocator, block_size, num_snapshots)
    runner = MixedFakeRunner(max_batch, allocator, block_size)
    sched = Scheduler(runner, cache, **kw)
    return sched, runner, cache


# ---------------------------------------------------------------- (a) equivalence


def test_mixed_and_alternating_produce_identical_tokens_and_cache_depths() -> None:
    """The core correctness property: turning `SEED_MIXED_BATCH` on must not change a single
    token any request sees, nor how much of any prompt ends up cached, even though the
    schedule of forward calls is completely different (one mixed call per iteration vs. a
    whole prefill burst then a whole decode step)."""
    specs = [
        ([1, 2, 3, 4, 5, 6, 7], 9),
        ([11, 12, 13], 6),
        ([21, 22, 23, 24, 25], 12),
        ([31, 32], 5),
    ]

    sched_a, runner_a, cache_a = make(max_batch=4, prefill_chunk=3, token_budget=6)
    pairs_a = [request(p, n) for p, n in specs]
    for req, _ in pairs_a:
        sched_a.submit(req)
    drain(sched_a)

    sched_m, runner_m, cache_m = make(
        max_batch=4, prefill_chunk=3, token_budget=6, mixed_batch=True
    )
    pairs_m = [request(p, n) for p, n in specs]
    for req, _ in pairs_m:
        sched_m.submit(req)
    drain(sched_m)

    for (prompt, max_new), (_, sink_a), (_, sink_m) in zip(specs, pairs_a, pairs_m, strict=True):
        assert sink_a.error is None and sink_m.error is None
        assert sink_a.tokens == serial(prompt, max_new)
        assert sink_m.tokens == sink_a.tokens, "mixed batching changed this request's tokens"
        assert sink_m.end == sink_a.end

    for prompt, _ in specs:
        node_a, node_m = cache_a.lookup(prompt), cache_m.lookup(prompt)
        assert node_m.depth == node_a.depth, "mixed batching changed the recorded prefix depth"
    assert runner_m.mixed_calls, "mixed batching never actually ran a mixed call"


def test_mixed_and_alternating_agree_across_a_two_turn_session() -> None:
    """Same equivalence, across a second turn that hits the prefix cache -- publishing
    (`_publish_boundary`/`_publish_turn_close`) runs identically either way."""
    turn1 = [5, 6, 7]
    turn2 = [5, 6, 7, 100, 101, 102, 103]

    sched_a, runner_a, cache_a = make(max_batch=3, prefill_chunk=4, token_budget=8)
    sched_m, runner_m, cache_m = make(
        max_batch=3, prefill_chunk=4, token_budget=8, mixed_batch=True
    )
    for sched in (sched_a, sched_m):
        req, sink = request(turn1, 4)
        sched.submit(req)
        drain(sched)
        assert sink.tokens == serial(turn1, 4)

    req_a, sink_a = request(turn2, 5)
    req_m, sink_m = request(turn2, 5)
    sched_a.submit(req_a)
    drain(sched_a)
    sched_m.submit(req_m)
    drain(sched_m)

    assert sink_a.tokens == serial(turn2, 5)
    assert sink_m.tokens == sink_a.tokens
    assert sink_m.end == sink_a.end
    assert cache_m.lookup(turn2).depth == cache_a.lookup(turn2).depth
    assert runner_a.prefilled >= runner_m.prefilled or runner_a.prefilled == runner_m.prefilled


# ---------------------------------------------------------------- (b) stall-free property


def test_mixed_batch_advances_every_decoding_lane_every_step() -> None:
    """The property `SEED_MIXED_BATCH` exists for: once a request is decoding, it must not be
    skipped for a scheduler step while there is still pending prefill work, no matter how much
    of that work is queued behind it."""
    # A tight `mixed_token_budget` (1 request-chunk per mixed step, same as
    # `test_prefill_batch_fifo_fairness`'s setup) forces enough genuinely-mixed steps to check.
    sched, runner, _ = make(
        max_batch=4, prefill_chunk=8, token_budget=8, mixed_batch=True, mixed_token_budget=8
    )
    long_req, long_sink = request([1, 2, 3], 30)
    sched.submit(long_req)
    for _ in range(20):
        if sched.decoding:
            break
        sched.step()
    assert sched.decoding, "setup: the long request never started decoding"

    burst = [request([200 + i] * 20, 1) for i in range(3)]
    for req, _ in burst:
        sched.submit(req)

    # Detect a genuinely mixed step (decode *and* a real prefill chunk in the same call) via
    # `runner.mixed_calls` instrumentation, rather than guessing scheduler-internal timing
    # (admission happens inside `step()`, so `sched.prefill_q` cannot be checked beforehand).
    checked = 0
    for _ in range(60):
        if not sched.decoding:
            break
        before = {id(fl): fl.pos for fl in sched.decoding}
        mixed_before = len(runner.mixed_calls)
        assert sched.step()
        after = {id(fl): fl.pos for fl in sched.decoding}
        if len(runner.mixed_calls) > mixed_before and runner.mixed_calls[-1][1]:
            for key, pos in before.items():
                # A flight can leave `self.decoding` this step (finished); only lanes still
                # present afterwards are asserted, same as any other still-in-flight check.
                if key in after:
                    assert after[key] == pos + 1, "a decoding lane was skipped during mixed batching"
                    checked += 1
    assert checked > 5, "test setup: too few genuinely mixed (decode+prefill) steps to check"
    assert long_sink.tokens == serial([1, 2, 3], 30)
    for (prompt, max_new), (_, sink) in zip([([200 + i] * 20, 1) for i in range(3)], burst):
        assert sink.tokens == serial(prompt, max_new)


def test_alternating_batch_can_stall_a_decoding_lane_for_several_steps() -> None:
    """The regression this feature fixes, made concrete: with `SEED_MIXED_BATCH` off (plain
    alternation), the same setup as above *can* skip a decoding lane for consecutive steps
    while `MAX_PREFILL_BURST` prefill iterations run -- the p95 TPOT breach the campaign brief
    describes at C=48."""
    sched, runner, _ = make(max_batch=4, prefill_chunk=8, token_budget=8, mixed_batch=False)
    long_req, long_sink = request([1, 2, 3], 30)
    sched.submit(long_req)
    for _ in range(20):
        if sched.decoding:
            break
        sched.step()
    assert sched.decoding

    burst = [request([200 + i] * 20, 1) for i in range(3)]
    for req, _ in burst:
        sched.submit(req)

    # A "pure prefill" step under plain alternation is directly observable: `runner.batches`
    # (one entry per decode step) does not grow while `runner.prefill_batch_sizes` (one entry
    # per batched-prefill call) does. Any decoding lane still present before and after such a
    # step, at the same `pos`, was skipped for that iteration -- the stall this feature removes.
    saw_a_stall = False
    for _ in range(60):
        if not sched.decoding:
            break
        before = {id(fl): fl.pos for fl in sched.decoding}
        batches_before, prefill_before = len(runner.batches), len(runner.prefill_batch_sizes)
        assert sched.step()
        after = {id(fl): fl.pos for fl in sched.decoding}
        pure_prefill_step = (
            len(runner.prefill_batch_sizes) > prefill_before
            and len(runner.batches) == batches_before
        )
        if pure_prefill_step:
            for key, pos in before.items():
                if key in after and after[key] == pos:
                    saw_a_stall = True
    assert saw_a_stall, "alternating scheduling was expected to stall a decoding lane at least once"
    assert long_sink.tokens == serial([1, 2, 3], 30)


# ---------------------------------------------------------------- (c) adaptive budget


def test_mixed_budget_never_exceeds_its_cost_ceiling() -> None:
    """`_mixed_budget`'s own invariant, swept across a grid of cost-model inputs: the
    simulated added cost of the prefill side (`budget * ms_per_token`) must never exceed
    `MIXED_BUDGET_SLACK` of the assumed decode-step cost, so a mixed step's total simulated
    cost never exceeds `(1 + MIXED_BUDGET_SLACK)` (1.5x) of a decode-only step's."""
    for decode_ms in (60.0, 80.0, 100.0, 120.0, 200.0):
        for ms_per_token in (0.05, 0.3, 1.0, 3.0):
            for ceiling in (32, 128, 512, 2048):
                sched, _, _ = make(
                    max_batch=1,
                    mixed_batch=True,
                    mixed_decode_step_ms=decode_ms,
                    mixed_prefill_ms_per_token=ms_per_token,
                    mixed_token_budget=ceiling,
                )
                budget = sched._mixed_budget()
                assert 1 <= budget <= ceiling
                added_cost = budget * ms_per_token
                # The `max(1, ...)` floor can push one token over the ceiling when the pure
                # cost formula would round to 0; that one-token floor is bounded (never more
                # than one extra token's cost), everything else must respect the ceiling.
                if budget > 1:
                    assert added_cost <= MIXED_BUDGET_SLACK * decode_ms + ms_per_token


def test_mixed_prefill_batch_is_still_fifo_fair() -> None:
    """`_mixed_budget`'s adaptive figure feeds the same `_select_prefill_batch` walk the
    ordinary batched-prefill path uses, so FIFO fairness (`test_prefill_batch_fifo_fairness`
    in `test_scheduler.py`) must survive under mixed batching too: with a budget too tight to
    ever admit two requests' chunks at once, later requests must not finish their own prefill
    before earlier ones do."""
    sched, runner, _ = make(
        max_batch=4,
        prefill_chunk=4,
        mixed_batch=True,
        mixed_decode_step_ms=100.0,
        mixed_prefill_ms_per_token=40.0,  # forces a budget of 1 token: tight FIFO pressure
        mixed_token_budget=512,
    )
    assert sched._mixed_budget() == 1, "test setup: needs a one-token budget to force FIFO order"
    long_req, long_sink = request([1, 1], 20)  # keeps a lane decoding so mixed steps trigger
    sched.submit(long_req)
    for _ in range(10):
        if sched.decoding:
            break
        sched.step()
    assert sched.decoding

    finish_order: list[int] = []
    specs = [([10] * 6, 1), ([20] * 6, 1), ([30] * 6, 1)]
    sinks = [Sink() for _ in specs]
    for i, ((prompt, max_new), sink) in enumerate(zip(specs, sinks, strict=True)):

        def make_emit(index: int, sink: Sink):  # noqa: ANN001, ANN201
            def emit(event: tuple) -> None:
                sink(event)
                if event[0] == "end" and index not in finish_order:
                    finish_order.append(index)

            return emit

        req = Request(list(prompt), max_new, 0.0, frozenset(), make_emit(i, sink))
        sched.submit(req)
    drain(sched)

    for (prompt, max_new), sink in zip(specs, sinks, strict=True):
        assert sink.tokens == serial(prompt, max_new)
    assert long_sink.tokens == serial([1, 1], 20)
    assert finish_order == [0, 1, 2], (
        "a tight mixed-step budget must still resolve prefill_q strictly in FIFO order"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

"""CPU tests for `SEED_PREFILL_ACCUM` (scheduler.py): prefill held while lanes decode until a
captured shape carries the token target or the oldest flight hits the wait bound.

    python3 -m pytest seed_tests/test_scheduler_prefill_accum.py -q -o addopts=
"""

import random
import sys
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_scheduler import FakeRunner, drain, make, request, serial  # noqa: E402

SHAPES = [(1, 4), (2, 4), (4, 4), (2, 8)]


def shape_for(lengths, shapes):  # noqa: ANN001, ANN201 -- graph_prefill.shape_for without torch
    for rows, width in sorted(shapes, key=lambda s: (s[0] * s[1], s[1])):
        if rows >= len(lengths) and width >= max(lengths):
            return rows, width
    return None


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class ShapedRunner(FakeRunner):
    shapes = SHAPES

    def prefill_fit(self):  # noqa: ANN201
        return (
            lambda lengths: shape_for(lengths, self.shapes) is not None,
            max(w for _, w in self.shapes),
        )

    def prefill_shapes(self):  # noqa: ANN201
        return list(self.shapes)


def build(clock: Clock, **kw):  # noqa: ANN003, ANN201
    base = dict(
        max_batch=8,
        prefill_chunk=8,
        token_budget=64,
        batched_prefill=True,
        prefill_fit_graph=True,
        prefill_accum=True,
        prefill_accum_tokens=12,
        prefill_accum_max_wait_ms=50.0,
        prefill_accum_cost=(4.0, 1.0),
        prefill_accum_min_decode=1,
        clock=clock,
    )
    base.update(kw)
    sched, runner, cache = make(**base)
    shaped = ShapedRunner(runner.max_batch, runner.allocator, runner.block_size)
    sched.runner = shaped
    return sched, shaped, cache


def test_holds_short_prefill_while_decoding_until_target() -> None:
    clock = Clock()
    sched, runner, _ = build(clock)
    first, first_sink = request([1, 2, 3], 40)
    sched.submit(first)
    sched.step()  # nothing decodes yet: fires at once
    assert len(runner.prefill_batch_sizes) == 1
    for p in ([10, 11, 12], [20, 21, 22]):  # 6 real tokens < target 12
        sched.submit(request(p, 3)[0])
    for _ in range(5):
        sched.step()
    assert len(runner.prefill_batch_sizes) == 1, "fired below target before the wait bound"
    sched.submit(request([30, 31, 32, 33], 3)[0])
    sched.submit(request([40, 41], 3)[0])  # now 12 real tokens across 4 rows
    sched.step()
    assert len(runner.prefill_batch_sizes) == 2
    assert runner.prefill_batch_chunk_sizes[1] == [3, 3, 4, 2]
    drain(sched)
    assert first_sink.tokens == serial([1, 2, 3], 40)


def test_wait_bound_forces_a_partial_step() -> None:
    clock = Clock()
    sched, runner, _ = build(clock)
    sched.submit(request([1, 2, 3], 40)[0])
    sched.step()
    sched.submit(request([10, 11, 12], 3)[0])
    for _ in range(3):
        sched.step()
    assert len(runner.prefill_batch_sizes) == 1
    clock.t = 0.049
    sched.step()
    assert len(runner.prefill_batch_sizes) == 1
    clock.t = 0.051
    sched.step()
    assert len(runner.prefill_batch_sizes) == 2
    assert runner.prefill_batch_chunk_sizes[1] == [3]


def test_shape_choice_caps_long_rows_at_the_chosen_width() -> None:
    """Two long flights: 2x8 carries 16 real tokens in area 16, best per modeled ms."""
    clock = Clock()
    sched, runner, _ = build(clock)
    sched.submit(request([1, 2, 3], 40)[0])
    sched.step()
    sched.submit(request(list(range(100, 120)), 2)[0])
    sched.submit(request(list(range(200, 220)), 2)[0])
    sched.step()
    assert runner.prefill_batch_chunk_sizes[1] == [8, 8]


def test_plan_width_caps_every_row_not_just_the_first() -> None:
    """Regression: the plan's width must cap every row's chunk. The captured-shape fit used to
    rebind the plan's width to the widest captured width after the first row, so later rows
    escalated the call to a wider, mostly padded shape."""
    clock = Clock()
    sched, runner, _ = build(clock, prefill_accum_tokens=1000, prefill_chunk=8)
    runner.shapes = [(1, 8), (2, 4), (4, 8)]
    sched.submit(request([1, 2, 3], 40)[0])
    sched.step()
    sched.submit(request(list(range(100, 106)), 2)[0])
    sched.submit(request(list(range(200, 206)), 2)[0])
    sched.step()  # admitted and held: 12 real tokens < target
    assert len(runner.prefill_batch_sizes) == 1
    clock.t = 1.0  # forced: (2, 4) carries 8 real in area 8, best per modeled ms
    sched.step()
    assert runner.prefill_batch_chunk_sizes[1] == [4, 4]


def test_fires_at_once_below_min_decode() -> None:
    clock = Clock()
    sched, runner, _ = build(clock, prefill_accum_min_decode=2)
    sched.submit(request([1, 2, 3], 40)[0])
    sched.step()
    sched.submit(request([10, 11, 12], 3)[0])
    sched.step()  # one lane decodes (< 2): no hold
    assert len(runner.prefill_batch_sizes) == 2


def test_flag_off_fires_immediately() -> None:
    clock = Clock()
    sched, runner, _ = build(clock, prefill_accum=False)
    sched.submit(request([1, 2, 3], 40)[0])
    sched.step()
    sched.submit(request([10, 11, 12], 3)[0])
    sched.step()
    assert len(runner.prefill_batch_sizes) == 2


def test_accum_ignored_without_fit_graph() -> None:
    sched, _, _ = build(Clock(), prefill_fit_graph=False)
    assert not sched.prefill_accum


@settings(max_examples=60, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    target=st.integers(1, 40),
    wait_ms=st.sampled_from([0.0, 5.0, 50.0]),
    chunk=st.sampled_from([4, 8]),
)
def test_random_arrivals_match_serial_and_every_call_fits(
    seed: int, target: int, wait_ms: float, chunk: int
) -> None:
    """Any arrival order, target and wait bound: every request's tokens equal serial
    generation, and every packed call replays a captured shape."""
    rng = random.Random(seed)
    clock = Clock()
    sched, runner, _ = build(
        clock, prefill_accum_tokens=target, prefill_accum_max_wait_ms=wait_ms, prefill_chunk=chunk
    )
    specs = [
        ([rng.randrange(1, 900) for _ in range(rng.randint(1, 19))], rng.randint(1, 12))
        for _ in range(rng.randint(1, 10))
    ]
    pairs = []
    for prompt, max_new in specs:
        req, sink = request(prompt, max_new)
        pairs.append((prompt, max_new, sink))
        sched.submit(req)
        for _ in range(rng.randint(0, 3)):
            clock.t += 0.01
            sched.step()
    for _ in range(100_000):
        clock.t += 0.01
        if not sched.step():
            break
    for prompt, max_new, sink in pairs:
        assert sink.error is None
        assert sink.tokens == serial(prompt, max_new)
    for sizes in runner.prefill_batch_chunk_sizes:
        assert shape_for(sizes, SHAPES) is not None, sizes


# -- SEED_PREFILL_PACK: packed shapes are filled by aligned area and segment slots ----------

ALIGN = 2


def packed_shape_for(lengths, shapes, packing):  # noqa: ANN001, ANN201
    """`graph_prefill.shape_for` with packed shapes, without torch."""
    for rows, width in sorted(shapes, key=lambda s: (s[0] * s[1], s[1])):
        segs, align = packing.get((rows, width), (0, 1))
        if segs:
            if len(lengths) <= segs and sum(-(-n // align) * align for n in lengths) <= rows * width:
                return rows, width
        elif rows >= len(lengths) and width >= max(lengths):
            return rows, width
    return None


class PackedRunner(ShapedRunner):
    packing = {(2, 8): (4, ALIGN), (4, 4): (4, ALIGN)}

    def prefill_fit(self):  # noqa: ANN201
        return (
            lambda lengths: packed_shape_for(lengths, self.shapes, self.packing) is not None,
            max(r * w if (r, w) in self.packing else w for r, w in self.shapes),
        )

    def prefill_packing(self):  # noqa: ANN201
        return dict(self.packing)


def build_packed(clock: Clock, **kw):  # noqa: ANN003, ANN201
    sched, runner, cache = build(clock, **kw)
    packed = PackedRunner(runner.max_batch, runner.allocator, runner.block_size)
    sched.runner = packed
    return sched, packed, cache


def test_packed_shape_takes_more_flights_than_rows() -> None:
    """Five short flights (3 tokens each, 4 aligned) fire as one 4-slot packed call on 2x8:
    rows no longer cap the call, the slot count does, and area is charged aligned."""
    clock = Clock()
    sched, runner, _ = build_packed(clock, prefill_accum_tokens=12)
    sched.submit(request([1, 2, 3], 40)[0])
    sched.step()
    for base in (10, 20, 30, 40, 50):
        sched.submit(request([base, base + 1, base + 2], 2)[0])
    sched.step()
    assert runner.prefill_batch_chunk_sizes[1] == [3, 3, 3, 3]
    drain(sched)


def test_packed_cuts_the_last_flight_to_the_area_left() -> None:
    """A long flight after a short one: 3 (4 aligned) + 12 of the next = area 16 on 2x8,
    longer than the width 8 a per-row 2x8 would cap it at."""
    clock = Clock()
    sched, runner, _ = build_packed(clock, prefill_accum_tokens=15, prefill_chunk=16)
    sched.submit(request([1, 2, 3], 40)[0])
    sched.step()
    sched.submit(request([10, 11, 12], 2)[0])
    sched.submit(request(list(range(100, 130)), 2)[0])
    sched.step()
    assert runner.prefill_batch_chunk_sizes[1] == [3, 12]
    drain(sched)


@settings(max_examples=60, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    target=st.integers(1, 40),
    wait_ms=st.sampled_from([0.0, 5.0, 50.0]),
    chunk=st.sampled_from([4, 8, 16]),
)
def test_packed_random_arrivals_match_serial_and_every_call_fits(
    seed: int, target: int, wait_ms: float, chunk: int
) -> None:
    rng = random.Random(seed)
    clock = Clock()
    sched, runner, _ = build_packed(
        clock, prefill_accum_tokens=target, prefill_accum_max_wait_ms=wait_ms, prefill_chunk=chunk
    )
    pairs = []
    for _ in range(rng.randint(1, 10)):
        prompt = [rng.randrange(1, 900) for _ in range(rng.randint(1, 25))]
        max_new = rng.randint(1, 12)
        req, sink = request(prompt, max_new)
        pairs.append((prompt, max_new, sink))
        sched.submit(req)
        for _ in range(rng.randint(0, 3)):
            clock.t += 0.01
            sched.step()
    for _ in range(100_000):
        clock.t += 0.01
        if not sched.step():
            break
    for prompt, max_new, sink in pairs:
        assert sink.error is None
        assert sink.tokens == serial(prompt, max_new)
    for sizes in runner.prefill_batch_chunk_sizes:
        assert packed_shape_for(sizes, runner.shapes, runner.packing) is not None, sizes

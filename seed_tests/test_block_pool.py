"""Hermetic CPU tests for the paged-KV block bookkeeping (`block_pool.py`).

`block_pool.py` is pure Python (no torch), so these tests import only `block_pool` and
`pytest`. They cover the allocator's refcounted free-list accounting (fresh allocation,
double-free detection, shared-block refcounting), `BlockTable`'s block-count math and
row-addressing arithmetic (including a table whose block ids are deliberately not a
contiguous run), and a steady-state simulation across many lanes and turns that pins the
"no leak, no double allocation" property `paged-kv-design.md` requires of the allocator.
Needs no torch.

    python3 -m pytest examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/seed_tests/test_block_pool.py
"""

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import block_pool  # noqa: E402


def _expected_rows(blocks: list[int], length: int, block_size: int) -> list[int]:
    """Independent re-derivation of `physical_rows`' formula, for hand-checking its output."""
    return [blocks[pos // block_size] * block_size + pos % block_size for pos in range(length)]


# ---------------------------------------------------------------- BlockAllocator: basics


def test_allocator_constructor_validates_its_arguments() -> None:
    with pytest.raises(ValueError):
        block_pool.BlockAllocator(1, 4)  # needs at least 1 usable block plus the reserved pad
    with pytest.raises(ValueError):
        block_pool.BlockAllocator(4, 0)
    allocator = block_pool.BlockAllocator(2, 4)  # smallest legal pool: 1 usable block
    assert allocator.usable_blocks == 1
    assert allocator.free_count == 1


def test_capacity_tokens_is_usable_blocks_times_block_size() -> None:
    allocator = block_pool.BlockAllocator(9, 4)  # usable = 8
    assert allocator.capacity_tokens() == 8 * 4


def test_alloc_returns_distinct_fresh_ids_at_refcount_one() -> None:
    allocator = block_pool.BlockAllocator(9, 4)  # usable = 8
    ids = allocator.alloc(5)
    assert len(ids) == 5
    assert len(set(ids)) == 5, "alloc handed out a duplicate id"
    assert block_pool.RESERVED_BLOCK not in ids
    for b in ids:
        assert allocator.refcount(b) == 1
    assert allocator.free_count == allocator.usable_blocks - 5


def test_alloc_zero_returns_an_empty_list_and_touches_nothing() -> None:
    allocator = block_pool.BlockAllocator(4, 4)
    before = allocator.free_count
    assert allocator.alloc(0) == []
    assert allocator.free_count == before


def test_alloc_rejects_a_negative_count() -> None:
    allocator = block_pool.BlockAllocator(4, 4)
    with pytest.raises(ValueError):
        allocator.alloc(-1)


def test_decref_returns_blocks_to_the_free_list() -> None:
    allocator = block_pool.BlockAllocator(9, 4)  # usable = 8
    ids = allocator.alloc(5)
    allocator.decref(ids)
    assert allocator.free_count == allocator.usable_blocks
    for b in ids:
        assert allocator.refcount(b) == 0


def test_reserved_block_is_never_allocated_and_is_guarded() -> None:
    allocator = block_pool.BlockAllocator(5, 4)  # usable = 4: ids 1..4
    seen = set()
    for _ in range(4):
        (b,) = allocator.alloc(1)
        seen.add(b)
    assert block_pool.RESERVED_BLOCK not in seen
    assert seen == {1, 2, 3, 4}

    with pytest.raises(ValueError):
        allocator.refcount(block_pool.RESERVED_BLOCK)
    with pytest.raises(ValueError):
        allocator.decref([block_pool.RESERVED_BLOCK])
    with pytest.raises(ValueError):
        allocator.incref([block_pool.RESERVED_BLOCK])
    with pytest.raises(ValueError):
        allocator.refcount(5)  # out of range: valid ids are [0, 5)


def test_alloc_exhaustion_does_not_partially_allocate() -> None:
    allocator = block_pool.BlockAllocator(5, 4)  # usable = 4
    before = allocator.free_count
    with pytest.raises(MemoryError):
        allocator.alloc(5)
    assert allocator.free_count == before, "a failed alloc must not consume any free blocks"

    # the pool must still be fully usable afterward -- no blocks were silently marked resident
    got = allocator.alloc(before)
    assert len(got) == before
    assert allocator.free_count == 0


def test_decref_past_zero_is_a_double_free() -> None:
    allocator = block_pool.BlockAllocator(4, 4)
    (b,) = allocator.alloc(1)
    allocator.decref([b])
    assert allocator.refcount(b) == 0
    with pytest.raises(RuntimeError, match="double free"):
        allocator.decref([b])
    # and it must not have gone negative or been re-added to the free list twice
    assert allocator.free_count == allocator.usable_blocks


def test_incref_a_free_block_raises_instead_of_hiding_a_double_free() -> None:
    allocator = block_pool.BlockAllocator(4, 4)
    (b,) = allocator.alloc(1)
    allocator.decref([b])
    with pytest.raises(RuntimeError):
        allocator.incref([b])


def test_incref_requires_two_decrefs_before_a_shared_block_frees() -> None:
    allocator = block_pool.BlockAllocator(4, 4)
    (b,) = allocator.alloc(1)
    assert allocator.refcount(b) == 1
    before_free = allocator.free_count

    allocator.incref([b])
    assert allocator.refcount(b) == 2

    allocator.decref([b])  # one owner's worth of decref: still resident
    assert allocator.refcount(b) == 1
    assert allocator.free_count == before_free, "a doubly-referenced block must not free early"

    allocator.decref([b])  # the second owner's decref: now it frees
    assert allocator.refcount(b) == 0
    assert allocator.free_count == before_free + 1


# ---------------------------------------------------------------- free functions


def test_max_blocks_per_lane_is_ceil_division() -> None:
    assert block_pool.max_blocks_per_lane(16384, 16) == 1024
    assert block_pool.max_blocks_per_lane(1, 16) == 1
    assert block_pool.max_blocks_per_lane(16, 16) == 1
    assert block_pool.max_blocks_per_lane(17, 16) == 2


def test_valid_block_count_is_ceil_division_but_zero_at_zero_tokens() -> None:
    assert block_pool.valid_block_count(0, 4) == 0
    assert block_pool.valid_block_count(1, 4) == 1
    assert block_pool.valid_block_count(4, 4) == 1
    assert block_pool.valid_block_count(5, 4) == 2
    with pytest.raises(ValueError):
        block_pool.valid_block_count(-1, 4)


# ---------------------------------------------------------------- BlockTable: construction


def test_block_table_token_capacity() -> None:
    assert block_pool.BlockTable(blocks=[3, 7, 1]).token_capacity(4) == 12
    assert block_pool.BlockTable().token_capacity(4) == 0


def test_reset_is_idempotent_on_an_empty_table() -> None:
    allocator = block_pool.BlockAllocator(4, 4)
    table = block_pool.BlockTable()
    table.reset(allocator)  # must not raise or touch the allocator
    assert table.blocks == []
    assert allocator.free_count == allocator.usable_blocks


def test_reset_releases_every_block_back_to_the_allocator() -> None:
    allocator = block_pool.BlockAllocator(9, 4)  # usable = 8
    table = block_pool.BlockTable()
    table.grow_to(12, 4, allocator, max_blocks=8)
    assert len(table.blocks) == 3
    table.reset(allocator)
    assert table.blocks == []
    assert allocator.free_count == allocator.usable_blocks


def test_grow_to_allocates_exactly_the_blocks_needed() -> None:
    allocator = block_pool.BlockAllocator(41, 4)  # usable = 40
    table = block_pool.BlockTable()

    table.grow_to(10, 4, allocator, max_blocks=20)  # ceil(10/4) = 3
    assert len(table.blocks) == 3
    assert allocator.free_count == allocator.usable_blocks - 3

    table.grow_to(12, 4, allocator, max_blocks=20)  # capacity (12 tokens) already suffices
    assert len(table.blocks) == 3, "growing to a length already covered must be a no-op"
    assert allocator.free_count == allocator.usable_blocks - 3

    table.grow_to(13, 4, allocator, max_blocks=20)  # ceil(13/4) = 4: exactly one more block
    assert len(table.blocks) == 4
    assert allocator.free_count == allocator.usable_blocks - 4


def test_grow_to_zero_tokens_is_a_no_op_on_a_fresh_table() -> None:
    allocator = block_pool.BlockAllocator(9, 4)
    table = block_pool.BlockTable()
    table.grow_to(0, 4, allocator, max_blocks=8)
    assert table.blocks == []
    assert allocator.free_count == allocator.usable_blocks


def test_grow_to_rejects_negative_tokens() -> None:
    allocator = block_pool.BlockAllocator(9, 4)
    table = block_pool.BlockTable()
    with pytest.raises(ValueError):
        table.grow_to(-1, 4, allocator, max_blocks=8)


def test_grow_to_past_max_blocks_raises_without_partial_growth() -> None:
    allocator = block_pool.BlockAllocator(41, 4)  # usable = 40, plenty of raw blocks
    table = block_pool.BlockTable()
    before_free = allocator.free_count

    with pytest.raises(MemoryError):
        table.grow_to(100, 4, allocator, max_blocks=4)  # needs 25 blocks, lane cap is 4
    assert table.blocks == [], "a failed grow_to must leave a fresh table untouched"
    assert allocator.free_count == before_free

    table.grow_to(8, 4, allocator, max_blocks=4)  # within cap: fine
    assert len(table.blocks) == 2
    before_free2 = allocator.free_count

    with pytest.raises(MemoryError):
        table.grow_to(20, 4, allocator, max_blocks=4)  # needs 5 total, cap is still 4
    assert len(table.blocks) == 2, "a failed grow_to must not partially extend an existing table"
    assert allocator.free_count == before_free2


def test_grow_to_raises_when_the_allocator_itself_is_exhausted() -> None:
    allocator = block_pool.BlockAllocator(5, 4)  # usable = 4
    table = block_pool.BlockTable()
    table.grow_to(16, 4, allocator, max_blocks=10)  # consumes all 4 usable blocks
    assert allocator.free_count == 0

    other = block_pool.BlockTable()
    with pytest.raises(MemoryError):
        other.grow_to(4, 4, allocator, max_blocks=10)  # needs 1 more, none free
    assert other.blocks == []


# ---------------------------------------------------------------- BlockTable: addressing


@pytest.mark.parametrize(
    "blocks,block_size",
    [
        ([3, 4, 5], 4),  # a contiguous run of ids, for contrast
        ([5, 12, 1, 100], 4),  # deliberately non-contiguous ids
        ([1], 8),  # a single block
    ],
    ids=["contiguous-ids", "non-contiguous-ids", "single-block"],
)
def test_physical_rows_and_physical_row_address_correctly(
    blocks: list[int], block_size: int
) -> None:
    """`physical_row`/`physical_rows` must not secretly assume `blocks` is a contiguous run."""
    table = block_pool.BlockTable(blocks=list(blocks))
    max_len = len(blocks) * block_size

    for length in range(0, max_len + 1):
        assert table.physical_rows(length, block_size) == _expected_rows(blocks, length, block_size)

    for pos in range(max_len):
        block_idx, offset = divmod(pos, block_size)
        assert table.physical_row(pos, block_size) == blocks[block_idx] * block_size + offset


def test_padded_row_right_pads_with_reserved_block() -> None:
    table = block_pool.BlockTable(blocks=[3, 7])
    r = block_pool.RESERVED_BLOCK
    assert table.padded_row(5) == [3, 7, r, r, r]
    assert table.padded_row(2) == [3, 7]  # exactly full: no padding needed


def test_padded_row_raises_if_the_table_already_exceeds_max_blocks() -> None:
    table = block_pool.BlockTable(blocks=[1, 2, 3])
    with pytest.raises(MemoryError):
        table.padded_row(2)


# ---------------------------------------------------------------- fragmentation


def test_allocator_can_produce_a_fragmented_block_table() -> None:
    """A reset-then-regrow pattern across sibling lanes hands a lane back a table whose block
    ids are not a contiguous run -- the case `test_paged_attn.py`'s kernel test needs a
    non-contiguous block table for.

    Traced step by step against `BlockAllocator`'s actual free-list mechanics (`alloc` pops
    from the end of `_free`, `decref` appends in the order given -- see its module docstring,
    "order is irrelevant; LIFO is cheapest"): with usable ids 1..8,

        a.grow_to(12) -> alloc(3) pops 8,7,6           -> a=[8,7,6]  free=[1,2,3,4,5]
        b.grow_to(8)  -> alloc(2) pops 5,4             -> b=[5,4]    free=[1,2,3]
        c.grow_to(4)  -> alloc(1) pops 3               -> c=[3]      free=[1,2]
        a.reset()     -> decref([8,7,6]) appends 8,7,6 ->            free=[1,2,8,7,6]
        b.reset()     -> decref([5,4]) appends 5,4     ->            free=[1,2,8,7,6,5,4]
        d.grow_to(16) -> alloc(4) pops 4,5,6,7         -> d=[4,5,6,7] free=[1,2,8]
        a.grow_to(8)  -> alloc(2) pops 8,2             -> a=[8,2]    free=[1]

    landing `a` on `[8, 2]`: two ids six apart, not a contiguous run.
    """
    block_size = 4
    allocator = block_pool.BlockAllocator(9, block_size)  # usable ids 1..8
    a, b, c, d = (block_pool.BlockTable() for _ in range(4))

    a.grow_to(12, block_size, allocator, max_blocks=8)
    b.grow_to(8, block_size, allocator, max_blocks=8)
    c.grow_to(4, block_size, allocator, max_blocks=8)

    a.reset(allocator)
    b.reset(allocator)

    d.grow_to(16, block_size, allocator, max_blocks=8)

    a.grow_to(8, block_size, allocator, max_blocks=8)  # regrown from scratch: needs 2 blocks

    assert len(a.blocks) == 2
    assert len(set(a.blocks)) == 2
    span = max(a.blocks) - min(a.blocks)
    assert span >= len(a.blocks), f"expected a fragmented (non-contiguous) table, got {a.blocks}"
    assert a.blocks == [8, 2], "documenting the exact ids from the traced sequence above"

    # physical_rows/physical_row must still address correctly through the fragmented table.
    length = a.token_capacity(block_size)
    expected = _expected_rows(a.blocks, length, block_size)
    assert a.physical_rows(length, block_size) == expected
    assert expected != sorted(expected), "the addressed rows should reflect the fragmentation too"


# ---------------------------------------------------------------- steady state over many turns


def test_block_allocator_steady_state_over_many_turns() -> None:
    """Many lanes, many turns of reset-then-regrow: no leak, no double allocation, ever.

    Mirrors test_state_memory.py's steady-state pattern (serve many turns, check the
    footprint stays consistent after every one), specialized to `BlockAllocator`/`BlockTable`
    alone with no `Scheduler`/`Model` involved.
    """
    num_lanes = 6
    block_size = 4
    per_lane_max_blocks = 9  # upper bound on how large any one lane's table gets in this test
    max_blocks_per_lane = 10  # the lane's fixed device-buffer row width; never hit here
    # generous headroom (3x the worst case where every lane is simultaneously maxed out) so
    # this test is purely about free_count bookkeeping, not about exercising exhaustion.
    total_blocks = num_lanes * per_lane_max_blocks * 3 + 1
    turns = 240

    allocator = block_pool.BlockAllocator(total_blocks, block_size)
    tables = [block_pool.BlockTable() for _ in range(num_lanes)]
    rng = random.Random(0xB10C_C0DE)

    for _ in range(turns):
        lane = rng.randrange(num_lanes)
        tables[lane].reset(allocator)  # session-end / eviction for whatever this lane held
        tokens = rng.randrange(0, per_lane_max_blocks * block_size + 1)  # a new session's length
        tables[lane].grow_to(tokens, block_size, allocator, max_blocks_per_lane)

        busy_blocks = sum(len(t.blocks) for t in tables)  # tracked independently of the allocator
        assert allocator.free_count == allocator.usable_blocks - busy_blocks
        assert 0 <= allocator.free_count <= allocator.usable_blocks

    for t in tables:
        t.reset(allocator)
    assert allocator.free_count == allocator.usable_blocks, "leaked blocks after every lane reset"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

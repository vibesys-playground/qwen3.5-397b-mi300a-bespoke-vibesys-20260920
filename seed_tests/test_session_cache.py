"""Hermetic CPU tests for the Stage 2 prefix trie (`session_cache.py`) on its own, apart from
the scheduler that drives it in practice (`test_scheduler.py` covers the end-to-end behavior
with a `FakeRunner`). Pure Python, real `block_pool.BlockAllocator`, no torch.

    python3 -m pytest examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/seed_tests/test_session_cache.py
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import block_pool  # noqa: E402
from session_cache import SessionCache, SessionCacheError, SnapshotStats  # noqa: E402

BLOCK_SIZE = 4


def make(
    num_blocks: int = 64, num_snapshots: int = 8, block_size: int = BLOCK_SIZE
) -> SessionCache:
    allocator = block_pool.BlockAllocator(num_blocks, block_size)
    return SessionCache(allocator, block_size, num_snapshots)


def test_lookup_on_an_empty_cache_returns_the_root() -> None:
    cache = make()
    node = cache.lookup([1, 2, 3])
    assert node is cache.root
    assert node.depth == 0


def test_publish_then_lookup_finds_the_exact_node() -> None:
    cache = make()
    blocks = tuple(cache.reserve_blocks(1))
    snap = cache.reserve_snapshot()
    node = cache.publish(cache.root, (1, 2, 3), blocks, snap)
    assert cache.lookup([1, 2, 3]) is node
    assert cache.lookup([1, 2, 3, 9, 9]) is node  # longest prefix, not exact-length only
    assert cache.lookup([1, 2, 9]) is cache.root  # diverges before the edge ends: no match


def test_two_sessions_sharing_a_prefix_incref_the_same_blocks_once_each() -> None:
    cache = make()
    blocks = tuple(cache.reserve_blocks(2))  # 2 blocks * 4 tokens = 8-token shared edge
    snap = cache.reserve_snapshot()
    shared = cache.publish(cache.root, tuple(range(8)), blocks, snap)
    # `publish` always takes its own share on top of whatever the caller already held
    # (`reserve_blocks` above), mirroring the scheduler's real flow where a lane's own
    # temporary share is released separately (`release_lane_blocks`) after publishing;
    # simulate that release here to reach the steady state a node holds on its own.
    cache.release_lane_blocks(blocks)
    assert cache.allocator.refcount(blocks[0]) == 1

    lane_a = cache.adopt(shared)
    assert cache.allocator.refcount(blocks[0]) == 2
    lane_b = cache.adopt(shared)
    assert cache.allocator.refcount(blocks[0]) == 3

    cache.release_lane_blocks(lane_a)
    assert cache.allocator.refcount(blocks[0]) == 2
    cache.release_lane_blocks(lane_b)
    assert cache.allocator.refcount(blocks[0]) == 1  # only the node's own share remains


def test_needs_cow_exactly_for_a_non_block_aligned_boundary() -> None:
    cache = make()
    aligned = cache.publish(
        cache.root, (1, 2, 3, 4), tuple(cache.reserve_blocks(1)), cache.reserve_snapshot()
    )
    assert not cache.needs_cow(aligned)
    partial = cache.publish(
        aligned, (5, 6), tuple(cache.reserve_blocks(1)), cache.reserve_snapshot()
    )
    assert cache.needs_cow(partial)
    assert not cache.needs_cow(cache.root)  # depth 0, no blocks at all


def test_adopt_with_replacement_skips_increfing_the_old_last_block() -> None:
    cache = make()
    first = tuple(cache.reserve_blocks(1))
    node = cache.publish(cache.root, (1, 2, 3), first, cache.reserve_snapshot())  # partial: 3 % 4
    cache.release_lane_blocks(first)  # see the sharing test above for why this step is needed
    assert cache.needs_cow(node)
    fresh = cache.reserve_blocks(1)[0]
    lane_blocks = cache.adopt(node, replacement_last_block=fresh)
    assert lane_blocks == (fresh,)
    assert cache.allocator.refcount(first[0]) == 1, "old partial block must not gain a reference"
    assert cache.allocator.refcount(fresh) == 1  # only the reservation's own share so far


def test_protect_stops_reservation_from_evicting_the_node_being_extended() -> None:
    """Regression: `reserve_blocks`/`reserve_snapshot` must never evict the very node a caller
    is mid-way through extending (admitting onto a lane, or publishing a child under) just
    because that node happens to still be a childless leaf at that exact moment -- see
    `scheduler.py`'s `_acquire` and `_publish_boundary`, which pass `protect=` for exactly this.
    Without `protect`, a tight pool would pick `node` itself (the only leaf), corrupting
    whatever the caller was about to build on top of it instead of failing cleanly.
    """
    cache = make(num_blocks=3, num_snapshots=2)  # 2 usable blocks
    reservation = tuple(cache.reserve_blocks(2))
    node = cache.publish(cache.root, (1, 2), reservation, cache.reserve_snapshot())
    # `publish` takes its own share on top of the caller's reservation (see
    # test_two_sessions_sharing_a_prefix_incref_the_same_blocks_once_each); release the
    # caller's temporary share to reach the steady state a node holds on its own.
    cache.release_lane_blocks(reservation)
    assert cache.allocator.free_count == 0

    with pytest.raises(SessionCacheError):
        cache.reserve_blocks(1, protect=node)
    assert cache.lookup([1, 2]) is node, "the protected node must not have been evicted"
    assert cache.allocator.refcount(node.blocks[0]) == 1, "its blocks must be untouched"


def test_protect_still_allows_evicting_a_different_leaf() -> None:
    cache = make(num_blocks=16, num_snapshots=2)
    protected = cache.publish(
        cache.root, (1, 2), tuple(cache.reserve_blocks(1)), cache.reserve_snapshot()
    )
    other = cache.publish(
        cache.root, (3, 4), tuple(cache.reserve_blocks(1)), cache.reserve_snapshot()
    )
    assert cache.free_snapshot_count() == 0

    snap = cache.reserve_snapshot(protect=protected)
    assert cache.lookup([1, 2]) is protected, "the protected node must survive"
    assert cache.lookup([3, 4]) is cache.root, "the other node must have been evicted instead"
    assert snap == other.snapshot


def test_only_leaf_nodes_are_evicted() -> None:
    cache = make(num_blocks=16, num_snapshots=3)
    parent = cache.publish(
        cache.root, (1, 2, 3, 4), tuple(cache.reserve_blocks(1)), cache.reserve_snapshot()
    )
    child = cache.publish(
        parent, (5, 6, 7, 8), tuple(cache.reserve_blocks(1)), cache.reserve_snapshot()
    )
    cache.lookup([1, 2, 3, 4])  # touch parent more recently than child's own publish stamp...
    # ...but parent still has a child, so it must never be chosen over an actual leaf.
    victim = cache._evict_one_leaf()
    assert victim is child
    assert parent in [n for bucket in cache.root.children.values() for n in bucket]


def test_eviction_frees_both_the_snapshot_slot_and_the_blocks() -> None:
    cache = make(num_blocks=16, num_snapshots=2)
    a = cache.publish(cache.root, (1, 2), tuple(cache.reserve_blocks(1)), cache.reserve_snapshot())
    cache.publish(cache.root, (2, 3), tuple(cache.reserve_blocks(1)), cache.reserve_snapshot())
    assert cache.free_snapshot_count() == 0
    # Reserving one more must evict the LRU leaf (`a`, touched first) to make room.
    snap = cache.reserve_snapshot()
    assert cache.lookup([1, 2]) is cache.root, "the evicted node must no longer be reachable"
    assert snap == a.snapshot, "the freed slot must be the one the evicted node held"


def test_snapshot_stats_report_high_water_pressure_eviction_and_exhaustion() -> None:
    cache = make(num_blocks=16, num_snapshots=2)
    protected = cache.publish(
        cache.root, (1, 2), tuple(cache.reserve_blocks(1)), cache.reserve_snapshot()
    )
    other = cache.publish(
        cache.root, (3, 4), tuple(cache.reserve_blocks(1)), cache.reserve_snapshot()
    )
    assert cache.snapshot_stats() == SnapshotStats(
        capacity=2,
        used=2,
        high_watermark=2,
        pressure_evictions=0,
        reservation_failures=0,
    )

    assert cache.reserve_snapshot(protect=protected) == other.snapshot
    with pytest.raises(SessionCacheError):
        cache.reserve_snapshot(protect=protected)
    assert cache.snapshot_stats() == SnapshotStats(
        capacity=2,
        used=2,
        high_watermark=2,
        pressure_evictions=1,
        reservation_failures=1,
    )


def test_reserve_blocks_raises_when_nothing_is_left_to_evict() -> None:
    cache = make(num_blocks=3, num_snapshots=4)  # 2 usable blocks (block 0 is reserved)
    cache.reserve_blocks(2)  # exhaust the pool with nothing published (nothing evictable)
    with pytest.raises(SessionCacheError):
        cache.reserve_blocks(1)


def test_evict_all_clears_every_entry_and_is_a_noop_on_an_empty_cache() -> None:
    cache = make()
    cache.publish(cache.root, (1, 2), tuple(cache.reserve_blocks(1)), cache.reserve_snapshot())
    cache.publish(cache.root, (3, 4), tuple(cache.reserve_blocks(1)), cache.reserve_snapshot())
    assert cache.evict_all() == 2
    assert cache.node_count() == 1  # root only
    assert cache.evict_all() == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

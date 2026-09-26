"""Block-paged KV bookkeeping for the full-attention layers (paged-KV design, Stage 1).

Design: see `paged-kv-design.md` sections 1.3 ("Block size and full-attention KV pool"), 3.1
("Paged decode attention"), 5 ("Risks" -- refcounted-unit eviction), and "Stage 1" in section 6.
This module is deliberately torch-free (pure Python, no device tensors): it is the host-side
bookkeeping half of the design, exercised by the CPU-hermetic tests
(`seed_tests/test_block_pool.py`), the same posture `scheduler.py` already takes toward the
CPU-hermetic tests in `test_scheduler.py`. `model.py` is the only caller that touches torch:
it owns the actual `[num_blocks * block_size, kv_heads, head_dim]` K/V pool tensors (one pair
per full-attention layer, all indexed by the same block ids from the one allocator here) and
the fixed-shape device buffers (`block_table`, `block_valid`) a captured graph replays from.

Stage 1 vs Stage 2, and why this module does not change shape between them. Stage 1 keys a
`BlockTable` by *lane* (today's whole-session scheduler slot: `scheduler.py`'s `_Slot`, whose
lifetime is unchanged by this design), so in Stage 1 no two lanes ever point at the same block
and every `decref` you'll see here drops a block straight back to the free list. Stage 2 (a
session directory decoupled from lanes, a prefix trie with nodes shared across sessions) keys a
`BlockTable` by *session* instead and lets two sessions' tables both hold the same block id for
a shared-prefix edge -- at that point `incref`/`decref`'s refcounting, unused in Stage 1 beyond
the trivial 0/1 case, starts doing real work, with no change to `BlockAllocator` itself. That is
the point of refcounting every block from day one rather than adding it when Stage 2 needs it:
Stage 2 becomes "key `BlockTable` by session id and call `incref` at a shared trie node," not a
rewrite of the allocator's accounting.

Block id 0 is reserved and never handed out by `alloc`: it is the pad sentinel a fixed-width
device block-table row uses for a lane's unused (beyond its `len(blocks)`) entries, so that a
kernel's address arithmetic for an out-of-range block-table slot always lands on real, allocated
storage (block 0) even though the per-lane valid-count predicate means its *contents* are never
read. Defense in depth against a masking bug reading physically out-of-bounds memory, not a
correctness requirement on its own (see `paged_attn.py`'s module docstring for the load-side
half of this).
"""

from __future__ import annotations

from dataclasses import dataclass, field

RESERVED_BLOCK = 0
"""Never allocated; the pad value for an unused block-table row entry. See module docstring."""


class BlockAllocator:
    """Refcounted free-list over `num_blocks` opaque block ids (`RESERVED_BLOCK` excluded).

    A block's refcount is 0 (free, on the free list), or >= 1 (resident, owned by however many
    `BlockTable`s currently list it -- exactly one in Stage 1, potentially more once Stage 2's
    prefix trie shares a node's blocks across sessions). `alloc` only ever hands out
    refcount-0 blocks and sets them to 1; growing a table's *share* of an already-resident
    block (Stage 2's shared-prefix case) is `incref`, not `alloc`.
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks < 2:  # at least one usable block plus the reserved pad
            raise ValueError(f"block pool needs at least 2 blocks (1 usable), got {num_blocks}")
        if block_size < 1:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._refcount = [0] * num_blocks
        self._refcount[RESERVED_BLOCK] = -1  # sentinel: never touched by alloc/incref/decref
        self._free: list[int] = list(range(1, num_blocks))  # order is irrelevant; LIFO is cheapest

    @property
    def free_count(self) -> int:
        return len(self._free)

    @property
    def usable_blocks(self) -> int:
        """`num_blocks` minus the one reserved for `RESERVED_BLOCK`."""
        return self.num_blocks - 1

    def capacity_tokens(self) -> int:
        """Total token capacity if every usable block were resident and full."""
        return self.usable_blocks * self.block_size

    def alloc(self, n: int) -> list[int]:
        """Allocate `n` fresh blocks, each starting at refcount 1.

        Raises `MemoryError` (not silently under-allocating) if fewer than `n` are free, per
        this design's "fails while loading / fails admission, not mid-decode" contract (see
        `paged-kv-design.md` section 1.2 point 5 and section 4's admission-by-free-blocks note).
        """
        if n < 0:
            raise ValueError(f"cannot allocate a negative count: {n}")
        if n > len(self._free):
            raise MemoryError(
                f"block pool exhausted: need {n} block(s), {len(self._free)} free of "
                f"{self.usable_blocks} usable"
            )
        got = [self._free.pop() for _ in range(n)]
        for b in got:
            assert self._refcount[b] == 0, f"block {b} handed out by alloc while still resident"
            self._refcount[b] = 1
        return got

    def incref(self, block_ids) -> None:  # noqa: ANN001 - Iterable[int], kept loose like scheduler.py
        """Add one reference to each id (Stage 2: a second session's table adopts a shared edge).

        Every id must already be resident (refcount >= 1); incref-ing a free block would hide
        a double-free bug behind what looks like ordinary sharing, so this raises instead.
        """
        for b in block_ids:
            self._check_resident(b, "incref")
            self._refcount[b] += 1

    def decref(self, block_ids) -> None:  # noqa: ANN001
        """Drop one reference from each id; a block whose count reaches 0 returns to the free list.

        Raises on a decref past zero (double free) rather than silently going negative, which
        is exactly the accounting bug the steady-state CPU test exists to catch (see module
        docstring and `seed_tests/test_block_pool.py`).
        """
        for b in block_ids:
            self._check_resident(b, "decref")
            self._refcount[b] -= 1
            if self._refcount[b] == 0:
                self._free.append(b)

    def refcount(self, block_id: int) -> int:
        self._check_not_reserved(block_id)
        return self._refcount[block_id]

    def _check_not_reserved(self, block_id: int) -> None:
        if block_id == RESERVED_BLOCK:
            raise ValueError("RESERVED_BLOCK is not a real allocation")
        if not 0 <= block_id < self.num_blocks:
            raise ValueError(f"block id {block_id} out of range [0, {self.num_blocks})")

    def _check_resident(self, block_id: int, op: str) -> None:
        self._check_not_reserved(block_id)
        if self._refcount[block_id] <= 0:
            raise RuntimeError(f"{op}: block {block_id} is not resident (refcount 0): double free?")


@dataclass(slots=True)
class BlockTable:
    """One sequence's ordered block ids, append-only as its token count grows.

    Stage 1: one `BlockTable` per scheduler lane (slot), living exactly as long as the slot's
    session does; `reset` is called from the same place `Model.reset`/`begin` already are.
    Stage 2: the same class, keyed by session id instead of lane index, with `blocks` able to
    start as a *copy* of a shared trie-edge's block list (each id `incref`'d once for the new
    owner) rather than always starting empty -- nothing here has to change for that, since
    `grow_to` only ever appends and `reset` only ever decrefs exactly what this table listed.
    """

    blocks: list[int] = field(default_factory=list)

    def token_capacity(self, block_size: int) -> int:
        """Tokens this table can currently hold without allocating another block."""
        return len(self.blocks) * block_size

    def reset(self, allocator: BlockAllocator) -> None:
        """Release every block this table holds and forget them. Idempotent on an empty table."""
        if self.blocks:
            allocator.decref(self.blocks)
        self.blocks = []

    def grow_to(self, tokens: int, block_size: int, allocator: BlockAllocator, max_blocks: int) -> None:
        """Allocate fresh blocks, if needed, so this table can hold `tokens` tokens.

        `max_blocks` is the lane's fixed device-buffer row width (`ceil(max_context /
        block_size)`, see `paged-kv-design.md` section 3.1): exceeding it is a configuration
        error (a request longer than `--max-seq-len`/`--max-context` admits), not an allocator
        failure, so it raises `MemoryError` same as running out of blocks, distinguished only by
        message for whoever is reading the log.
        """
        if tokens < 0:
            raise ValueError(f"tokens must be non-negative, got {tokens}")
        needed = -(-tokens // block_size) - len(self.blocks)  # ceil(tokens / block_size) - have
        if needed <= 0:
            return
        if len(self.blocks) + needed > max_blocks:
            raise MemoryError(
                f"sequence needs {len(self.blocks) + needed} blocks for {tokens} tokens, "
                f"lane cap is {max_blocks} ({max_blocks * block_size} tokens)"
            )
        self.blocks.extend(allocator.alloc(needed))

    def physical_row(self, pos: int, block_size: int) -> int:
        """The pool row (into a `[num_blocks * block_size, ...]` tensor) holding token `pos`."""
        block_idx, offset = divmod(pos, block_size)
        return self.blocks[block_idx] * block_size + offset

    def physical_rows(self, length: int, block_size: int) -> list[int]:
        """Pool rows for tokens `[0, length)`, in order -- what a prefix gather indexes by."""
        full_blocks, rem = divmod(length, block_size)
        rows = [
            b * block_size + t for b in self.blocks[:full_blocks] for t in range(block_size)
        ]
        if rem:
            rows.extend(self.blocks[full_blocks] * block_size + t for t in range(rem))
        return rows

    def padded_row(self, max_blocks: int) -> list[int]:
        """This table's block ids, right-padded to `max_blocks` with `RESERVED_BLOCK`.

        What gets copied into one row of the fixed-shape device `block_table` buffer
        (`model.Model.block_table`/`graph_decode.Buffers.block_table`) every replay -- same
        "every input buffer is rewritten on every replay, unconditionally" contract
        `graph_decode.py`'s module docstring already holds `pos`/`active` to.
        """
        if len(self.blocks) > max_blocks:
            raise MemoryError(f"table holds {len(self.blocks)} blocks, lane cap is {max_blocks}")
        return self.blocks + [RESERVED_BLOCK] * (max_blocks - len(self.blocks))


def max_blocks_per_lane(max_context: int, block_size: int) -> int:
    """`ceil(max_context / block_size)`: the fixed row width of the device block-table buffer."""
    return -(-max_context // block_size)


def valid_block_count(num_tokens: int, block_size: int) -> int:
    """`ceil(num_tokens / block_size)`: how many of a lane's block-table entries are live.

    `num_tokens` is `pos + 1` (the position about to be written, inclusive) for a decode step,
    or a prefix length for the gather-once prefill path. Kept as a free function, not a
    `BlockTable` method, because the paged-attention kernel's per-replay `block_valid` buffer is
    filled from the scheduler's `positions` list directly (see `graph_decode.Buffers.fill`), not
    from a table object.
    """
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    return -(-num_tokens // block_size) if num_tokens else 0

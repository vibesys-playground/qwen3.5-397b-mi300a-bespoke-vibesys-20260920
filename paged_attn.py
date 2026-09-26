"""Fused Triton kernel for paged decode attention over the full-attention layers (Stage 1).

Design: `paged-kv-design.md` section 3.1. What this replaces and why it has to be a real
kernel, not a gather-then-SDPA call, is that section's argument in full; the short version:
once KV storage is block-paged, a session's resident tokens are not a contiguous view any more
(`model._KVWindow`'s whole trick), so a decode step either gathers every referenced block into
a contiguous scratch buffer every step (a bandwidth non-starter at long context, see the design
doc's ~1.9 GiB/rank/step number at 128k) or reads blocks in place inside the attention kernel
via the block table. This module is the second option.

One program per `(lane, kv_head)`. Query heads within a KV group are folded into an extra
"row" axis exactly the way `model.decode_attention` folds them into the query *length*
(`enable_gqa` under a mask is the thing both of them avoid; see that function's docstring for
the measured cost of not doing this) -- here the group axis is just carried through the kernel
directly instead of being reshaped into the query axis, since there is no SDPA call underneath
to fold it into. Every kv head's block is read exactly once per program regardless of the
group size, same property, different mechanism.

Capture-safety (`graph_decode.py`'s module docstring lists the constraints this obeys):

- **Static grid, device-side loop trip count.** The launch grid is `(lanes, kv_heads)`, fixed
  for a given `(max_batch, kv_heads)`. The inner loop over the block table runs
  `n_valid_blocks` times, a value the kernel loads from `bvalid` in device memory. That is
  capture-safe: a replay re-reads the buffer, and no launch argument, grid, or host-side
  branch depends on it. Looping `MAX_BLOCKS` (`ceil(max_seq / block_size)`, 1024 at
  `--max-seq-len 16384`) times instead made every lane pay for the longest possible context
  on every step: the masked iterations skip the loads but still run the `[GROUP, BLOCK_SIZE,
  HEAD_DIM]` products and the online-softmax update, serially, in one program per lane.
  `SEED_PAGED_ATTN_STATIC_LOOP=1` restores the fixed `MAX_BLOCKS` trip count for A/B.
- **Predicated loads, not post-hoc masking.** Each iteration reads a per-lane `block_valid`
  count (refreshed every replay exactly like `graph_decode.Buffers.pos`/`.active`) and uses it
  to *mask the `tl.load` itself* for both the block-table read and the K/V pool read once a
  block index is out of range -- a masked Triton load does not perform the memory transaction
  for masked lanes, so an out-of-range block-table slot never touches the K/V pool at all, per
  the design doc's explicit "skip the global memory load ... not merely mask the result after
  loading" requirement. `block_pool.RESERVED_BLOCK` (always allocated, never freed) is the
  `other=` value for a masked block-table load, so even a masking bug reads real, allocated
  storage rather than an arbitrary address.
- **Online softmax**, not a materialized `[group, context]` score tensor: running max/sum/
  accumulator carried across the block loop (the standard flash-attention recurrence), updated
  once per block. This is what keeps per-program scratch at `[group, head_dim]` plus a few
  `[group]` vectors regardless of how many blocks a lane has, rather than growing with context
  length the way a single masked-softmax-over-everything call would.

Precision follows `model.decode_attention`'s own masked ("written out") branch: q/k loaded and
promoted to fp32, the score product and softmax done in fp32, the final store rounded once to
the pool's storage dtype (bf16 in the deployed model) -- the same single-rounding-point
discipline `rmsnorm_fused.py`/`deltanet_fused.py` hold their kernels to. Not bit-identical to
`decode_attention`'s SDPA-backed dense path (different reduction order, a Triton `tl.sum`
reduction instead of SDPA's, exactly the caveat `deltanet_fused.py`'s module docstring already
carries for its own kernels), but the same fp32 chain, same single rounding.

Prefill is not this module's concern: `paged-kv-design.md` section 3.2 keeps prefill on a
gather-once-into-contiguous-then-`prefill_attention`-unchanged path for Stage 1 (see
`model.Model._gather_cached_prefix`), which is plain torch indexing, not a kernel.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - exercised only where triton is absent
    HAVE_TRITON = False


def available(device: torch.device) -> bool:
    """Same contract as `deltanet_fused.available`/`rmsnorm_fused.available`.

    `SEED_FUSED_PAGED_ATTN=0` forces... there is no fallback path in this module (Stage 1 has
    no second, torch-only paged-attention spelling to fall back to -- the whole point of a
    paged pool is that there is no contiguous view a torch call could read), so this flag
    exists only for the tests to force CPU/interpreter execution explicitly and for a future
    bisect; it does not gate correctness of anything else in `model.py`.
    """
    if not HAVE_TRITON or os.environ.get("SEED_FUSED_PAGED_ATTN", "1") == "0":
        return False
    return device.type == "cuda"


STATIC_LOOP = os.environ.get("SEED_PAGED_ATTN_STATIC_LOOP", "0") == "1"
"""A/B knob: loop `MAX_BLOCKS` times per lane (the Stage 1 behavior) instead of the lane's
own valid block count. See the module docstring's capture-safety notes."""


if HAVE_TRITON:

    @triton.jit
    def _paged_attn_decode_kernel(
        q_ptr,
        kpool_ptr,
        vpool_ptr,
        bt_ptr,
        bvalid_ptr,
        pos_ptr,
        o_ptr,
        sq_b,
        sq_h,
        sq_g,
        skp_row,
        skp_h,
        svp_row,
        svp_h,
        sbt_b,
        so_b,
        so_h,
        so_g,
        scale,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        MAX_BLOCKS: tl.constexpr,
        GROUP: tl.constexpr,
        STATIC_LOOP: tl.constexpr,
    ):
        """One decode step, one `(lane, kv_head)` program. See module docstring for the recurrence.

        `q` is `[lanes, kv_heads, GROUP, HEAD_DIM]` (the query heads of this kv head's group,
        already folded by the wrapper -- see `decode_attention_paged`). `kpool`/`vpool` are
        `[num_blocks * block_size, kv_heads, HEAD_DIM]`: row `r` holds token `r % block_size` of
        block `r // block_size`. `bt` is `[lanes, MAX_BLOCKS]` int32, this lane's block ids,
        right-padded with `block_pool.RESERVED_BLOCK` (0). `bvalid` is `[lanes]` int32, `ceil((pos
        + 1) / block_size)`. `pos` is `[lanes]`, the lane's current position (0-indexed): the
        fine within-the-last-block mask compares against `pos + 1` directly, so no separate
        "tokens in the last block" buffer is needed.
        """
        lane = tl.program_id(0)
        head = tl.program_id(1)
        g = tl.arange(0, GROUP)
        d = tl.arange(0, HEAD_DIM)

        q = tl.load(q_ptr + lane * sq_b + head * sq_h + g[:, None] * sq_g + d[None, :]).to(
            tl.float32
        )

        valid_len = tl.load(pos_ptr + lane) + 1
        n_valid_blocks = tl.load(bvalid_ptr + lane)

        m_i = tl.full((GROUP,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((GROUP,), dtype=tl.float32)
        acc = tl.zeros((GROUP, HEAD_DIM), dtype=tl.float32)

        t = tl.arange(0, BLOCK_SIZE)
        if STATIC_LOOP:
            n_iter = MAX_BLOCKS
        else:
            n_iter = tl.minimum(n_valid_blocks, MAX_BLOCKS)
        for blk in range(0, n_iter):
            active = blk < n_valid_blocks
            block_id = tl.load(bt_ptr + lane * sbt_b + blk, mask=active, other=0)
            logical_pos = blk * BLOCK_SIZE + t
            tok_live = active & (logical_pos < valid_len)
            row = block_id * BLOCK_SIZE + t

            k = tl.load(
                kpool_ptr + row[:, None] * skp_row + head * skp_h + d[None, :],
                mask=tok_live[:, None],
                other=0.0,
            ).to(tl.float32)
            v = tl.load(
                vpool_ptr + row[:, None] * svp_row + head * svp_h + d[None, :],
                mask=tok_live[:, None],
                other=0.0,
            ).to(tl.float32)

            scores = tl.sum(q[:, None, :] * k[None, :, :], axis=2) * scale  # [GROUP, BLOCK_SIZE]
            scores = tl.where(tok_live[None, :], scores, float("-inf"))

            blk_max = tl.max(scores, axis=1)
            new_m = tl.maximum(m_i, blk_max)
            alpha = tl.exp(m_i - new_m)
            p = tl.exp(scores - new_m[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * v[None, :, :], axis=1)
            m_i = new_m

        out = acc / l_i[:, None]
        tl.store(
            o_ptr + lane * so_b + head * so_h + g[:, None] * so_g + d[None, :],
            out.to(o_ptr.dtype.element_ty),
        )


def _row_stride(x: torch.Tensor, name: str) -> tuple[int, ...]:
    if x.stride(-1) != 1:
        raise ValueError(f"{name} must be contiguous along its last axis, got stride {x.stride()}")
    return x.stride()[:-1]


def decode_attention_paged(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_table: torch.Tensor,
    block_valid: torch.Tensor,
    positions: torch.Tensor,
    block_size: int,
    scale: float,
) -> torch.Tensor:
    """`model.decode_attention`'s masked branch, reading K/V through a block table.

    Shapes: `q` is `[B, heads, 1, head_dim]` (the plain per-slot query, not pre-folded -- the
    fold into `[B, kv_heads, group, head_dim]` happens here, same reshape
    `model.decode_attention` does, so call sites swap between the two functions with no reshape
    of their own). `k_pool`/`v_pool` are `[num_blocks * block_size, kv_heads, head_dim]`.
    `block_table` is `[B, max_blocks]` int32 (`block_pool.BlockTable.padded_row`'s output,
    stacked). `block_valid` is `[B]` int32 (`block_pool.valid_block_count` per lane).
    `positions` is `[B]` (any int dtype `tl.load` accepts as an index), the lane's current
    position. Returns `[B, heads, 1, head_dim]`, matching `decode_attention`'s return shape.

    No mask argument, unlike `decode_attention`: every call here is inherently "masked" (a
    per-lane prefix length, read through a lane-specific block table), so there is no unmasked/
    uniform-position fast path to choose between -- see `model.decode_attention`'s docstring for
    why that split exists on the dense pool and does not apply once storage is paged.
    """
    b, heads, one, head_dim = q.shape
    if one != 1:
        raise ValueError(f"decode_attention_paged is decode-only (T=1), got q with T={one}")
    kv_heads = k_pool.shape[1]
    if heads % kv_heads:
        raise ValueError(f"heads {heads} is not a multiple of kv_heads {kv_heads}")
    group = heads // kv_heads
    max_blocks = block_table.shape[1]

    grouped_q = q.reshape(b, kv_heads, group, head_dim)
    sq_b, sq_h, sq_g = _row_stride(grouped_q, "q")
    skp_row, skp_h = _row_stride(k_pool, "k_pool")
    svp_row, svp_h = _row_stride(v_pool, "v_pool")
    if block_table.stride(-1) != 1:
        raise ValueError("block_table must be contiguous along its last axis")

    out = torch.empty(b, kv_heads, group, head_dim, dtype=q.dtype, device=q.device)
    so_b, so_h, so_g = _row_stride(out, "out")
    _paged_attn_decode_kernel[(b, kv_heads)](
        grouped_q,
        k_pool,
        v_pool,
        block_table,
        block_valid,
        positions,
        out,
        sq_b,
        sq_h,
        sq_g,
        skp_row,
        skp_h,
        svp_row,
        svp_h,
        block_table.stride(0),
        so_b,
        so_h,
        so_g,
        scale,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        MAX_BLOCKS=max_blocks,
        GROUP=group,
        STATIC_LOOP=STATIC_LOOP,
    )
    return out.reshape(b, heads, 1, head_dim)

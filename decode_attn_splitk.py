"""Length-aware split-K paged decode attention: `SEED_DECODE_ATTN_SPLITK=1`.

Idea: Flash-Decoding (Dao, Haziza, Massa, Sizov, PyTorch blog, 2023): split each sequence's KV
range across several programs, each producing a partial online-softmax state `(m, l, acc)`, then
merge them in a second kernel. Cited as an idea only; no code is taken from any implementation.

Why the base kernel (`paged_attn._paged_attn_decode_kernel`) scales so badly with context:
one program per `(lane, kv_head)` walks the lane's blocks serially, 16 tokens per iteration,
with the q.k and p.v products written as broadcast-multiply-reduce over a `[GROUP, 16, 256]`
fp32 tile. Offline gfx942 compile: 316 VGPRs (1 wave/SIMD), no MFMA, ~3.7k VALU instructions
per 16-token block. At TP=4 the grid is only `lanes` programs wide (1 local KV head), so at
b16 16 of 228 CUs do all the work, each issue-bound in one wave: time per step grows linearly
with the longest lane's block count, independent of HBM bandwidth.

`decode_attn_v2.py` (`SEED_DECODE_ATTN_V2`) splits by a fixed block range derived from the
row width (`max_blocks`), so at `--max-seq-len 16384` a 2k-token lane still lands entirely in
split 0 of 8: no parallelism where it matters, plus it keeps the VALU-bound inner loop.

This kernel changes both things:

- **Split by each lane's own length.** The grid is `(lanes, kv_heads, NUM_SPLITS)`, fixed per
  captured bucket. Each program reads `pos[lane]` from device memory and takes the
  `split`-th of `NUM_SPLITS` equal slices of *that lane's* `ceil((pos + 1) / TILE)` tiles, so a
  2k lane spreads across all splits and a 256-token lane across a few (the rest see an empty
  range and store an empty partial). Nothing about the grid or any launch argument depends on
  device data, so the kernels are capture-safe and a replay picks up new positions.
- **MFMA inner loop.** `TILE` tokens (several 16-token pages, gathered through the block
  table per token) per iteration; `q.k^T` and `p.v` are `tl.dot` on bf16 operands with fp32
  accumulation. The query group (8 heads) is padded to 16 rows, the smallest MFMA M.

`NUM_SPLITS` is chosen once per call from the bucket's lane count so that `lanes * kv_heads *
NUM_SPLITS` is about `SPLITS_PER_CU` programs per CU (`_num_splits`); it is a host function of
shapes only, so a captured bucket always replays the same grid.

Precision: q.k products are exact in fp32 (bf16 x bf16) with fp32 accumulation; the softmax
recurrence is fp32; `p` is rounded to bf16 before the p.v MFMA (standard FlashAttention
practice), which is the one extra rounding versus the base kernel. Partials are fp32; the
output is rounded once to the pool dtype in the combine kernel.
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
    """`SEED_DECODE_ATTN_SPLITK=1` switches `model.decode_attention_paged` to this module."""
    if not HAVE_TRITON or os.environ.get("SEED_DECODE_ATTN_SPLITK", "0") != "1":
        return False
    return device.type == "cuda"


TILE = int(os.environ.get("SEED_DECODE_ATTN_SPLITK_TILE", "32"))
"""Tokens per inner-loop iteration (a multiple of the KV block size)."""

SPLITS_PER_CU = float(os.environ.get("SEED_DECODE_ATTN_SPLITK_PER_CU", "2"))
"""Target programs per CU when sizing `NUM_SPLITS` for a bucket."""

MAX_SPLITS = int(os.environ.get("SEED_DECODE_ATTN_SPLITK_MAX_SPLITS", "64"))

KERNEL_WARPS = int(os.environ.get("SEED_DECODE_ATTN_SPLITK_WARPS", "0"))
"""`num_warps` for the split kernel; 0 keeps Triton's default (4)."""

KERNEL_STAGES = int(os.environ.get("SEED_DECODE_ATTN_SPLITK_STAGES", "0"))
"""`num_stages` for the split kernel; 0 keeps Triton's default.

Round 15 (captured B96 decode step, node05): `1` is bit exact against the default and cut
the 15-layer attention core from 3.65 to 2.42 ms at pos 2048 and 6.62 to 4.32 ms at pos 4096;
with `SEED_DECODE_ATTN_SPLITK_PER_CU=8` (rounding order only: more splits) the step fell by
2.5 ms at pos 2048 and 3.7 ms at pos 4096, and by 1.6/0.6/0.5 ms at B80/B64/B48 (pos 2048)."""

_NUM_CUS: dict[int, int] = {}


def _num_cus(device: torch.device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _NUM_CUS:
        _NUM_CUS[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _NUM_CUS[idx]


def _num_splits(lanes: int, kv_heads: int, max_tokens: int, num_cus: int) -> int:
    """Splits per lane for a `lanes`-wide bucket: ~`SPLITS_PER_CU` programs per CU, at most
    one split per tile of the widest possible lane, capped at `MAX_SPLITS`. Shapes only, so it
    is fixed for a captured bucket."""
    want = -(-int(SPLITS_PER_CU * num_cus) // max(1, lanes * kv_heads))
    max_tiles = max(1, -(-max_tokens // TILE))
    return max(1, min(want, MAX_SPLITS, max_tiles))


if HAVE_TRITON:

    @triton.jit
    def _splitk_kernel(
        q_ptr,
        kpool_ptr,
        vpool_ptr,
        bt_ptr,
        pos_ptr,
        m_ptr,
        l_ptr,
        acc_ptr,
        sq_b,
        sq_h,
        sq_g,
        skp_row,
        skp_h,
        svp_row,
        svp_h,
        sbt_b,
        sp_b,
        sp_h,
        sp_s,
        scale,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        GROUP: tl.constexpr,
        GROUP_PAD: tl.constexpr,
        TILE: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
    ):
        """One `(lane, kv_head, split)` program over this split's slice of the lane's tiles.

        Partials are `[lanes, kv_heads, NUM_SPLITS, GROUP(, HEAD_DIM)]` fp32; `sp_*` are the
        `m`/`l` strides and `acc`'s strides are those times `HEAD_DIM`.
        """
        lane = tl.program_id(0)
        head = tl.program_id(1)
        split = tl.program_id(2)
        g = tl.arange(0, GROUP_PAD)
        d = tl.arange(0, HEAD_DIM)
        t = tl.arange(0, TILE)
        g_live = g < GROUP

        n_tok = tl.load(pos_ptr + lane) + 1
        n_tiles = (n_tok + TILE - 1) // TILE
        per_split = (n_tiles + NUM_SPLITS - 1) // NUM_SPLITS
        t0 = split * per_split
        n_local = tl.maximum(tl.minimum(per_split, n_tiles - t0), 0)

        q = tl.load(
            q_ptr + lane * sq_b + head * sq_h + g[:, None] * sq_g + d[None, :],
            mask=g_live[:, None],
            other=0.0,
        )

        m_i = tl.full((GROUP_PAD,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((GROUP_PAD,), dtype=tl.float32)
        acc = tl.zeros((GROUP_PAD, HEAD_DIM), dtype=tl.float32)

        for j in range(0, n_local):
            tok = (t0 + j) * TILE + t
            live = tok < n_tok
            bid = tl.load(bt_ptr + lane * sbt_b + tok // BLOCK_SIZE, mask=live, other=0)
            row = bid * BLOCK_SIZE + tok % BLOCK_SIZE
            k = tl.load(
                kpool_ptr + row[:, None] * skp_row + head * skp_h + d[None, :],
                mask=live[:, None],
                other=0.0,
            )
            v = tl.load(
                vpool_ptr + row[:, None] * svp_row + head * svp_h + d[None, :],
                mask=live[:, None],
                other=0.0,
            )
            s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale  # [GROUP_PAD, TILE] fp32
            s = tl.where(live[None, :], s, float("-inf"))
            # Every iterated tile holds >= 1 live token (tile < n_tiles), so new_m is finite.
            new_m = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - new_m)
            p = tl.exp(s - new_m[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision="ieee")
            m_i = new_m

        base = lane * sp_b + head * sp_h + split * sp_s
        tl.store(m_ptr + base + g, m_i, mask=g_live)
        tl.store(l_ptr + base + g, l_i, mask=g_live)
        tl.store(
            acc_ptr + base * HEAD_DIM + g[:, None] * HEAD_DIM + d[None, :],
            acc,
            mask=g_live[:, None],
        )

    @triton.jit
    def _combine_kernel(
        m_ptr,
        l_ptr,
        acc_ptr,
        pos_ptr,
        o_ptr,
        sp_b,
        sp_h,
        sp_s,
        so_b,
        so_h,
        so_g,
        HEAD_DIM: tl.constexpr,
        GROUP: tl.constexpr,
        TILE: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
    ):
        """One `(lane, kv_head)` program: merge the non-empty splits' partials (the same
        online-softmax rule the split loop uses) and store the output in the pool dtype."""
        lane = tl.program_id(0)
        head = tl.program_id(1)
        g = tl.arange(0, GROUP)
        d = tl.arange(0, HEAD_DIM)
        n_tiles = (tl.load(pos_ptr + lane) + TILE) // TILE  # ceil((pos + 1) / TILE)
        per_split = (n_tiles + NUM_SPLITS - 1) // NUM_SPLITS
        n_used = (n_tiles + per_split - 1) // per_split  # non-empty splits, same split rule

        m_i = tl.full((GROUP,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((GROUP,), dtype=tl.float32)
        acc = tl.zeros((GROUP, HEAD_DIM), dtype=tl.float32)
        for s in range(0, n_used):
            base = lane * sp_b + head * sp_h + s * sp_s
            m_s = tl.load(m_ptr + base + g)
            l_s = tl.load(l_ptr + base + g)
            a_s = tl.load(acc_ptr + base * HEAD_DIM + g[:, None] * HEAD_DIM + d[None, :])
            new_m = tl.maximum(m_i, m_s)
            alpha = tl.exp(m_i - new_m)
            beta = tl.exp(m_s - new_m)
            l_i = l_i * alpha + l_s * beta
            acc = acc * alpha[:, None] + a_s * beta[:, None]
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


def decode_attention_paged_splitk(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_table: torch.Tensor,
    block_valid: torch.Tensor,
    positions: torch.Tensor,
    block_size: int,
    scale: float,
    num_splits: int | None = None,
) -> torch.Tensor:
    """Drop-in for `paged_attn.decode_attention_paged` (same arguments and return shape).

    `block_valid` is accepted for signature parity and not read: the split ranges come from
    `positions`, and every tile below `ceil((pos + 1) / TILE)` maps to a block index below
    `ceil((pos + 1) / block_size)`. `num_splits` overrides `_num_splits` (tests, benches).
    """
    del block_valid
    b, heads, one, head_dim = q.shape
    if one != 1:
        raise ValueError(f"decode_attention_paged_splitk is decode-only (T=1), got T={one}")
    kv_heads = k_pool.shape[1]
    if heads % kv_heads:
        raise ValueError(f"heads {heads} is not a multiple of kv_heads {kv_heads}")
    if TILE % block_size:
        raise ValueError(f"SEED_DECODE_ATTN_SPLITK_TILE {TILE} is not a multiple of {block_size}")
    if block_table.stride(-1) != 1:
        raise ValueError("block_table must be contiguous along its last axis")
    group = heads // kv_heads
    max_tokens = block_table.shape[1] * block_size
    if num_splits is None:
        num_splits = _num_splits(b, kv_heads, max_tokens, _num_cus(q.device))

    grouped_q = q.reshape(b, kv_heads, group, head_dim)
    sq_b, sq_h, sq_g = _row_stride(grouped_q, "q")
    skp_row, skp_h = _row_stride(k_pool, "k_pool")
    svp_row, svp_h = _row_stride(v_pool, "v_pool")

    dev = q.device
    m_part = torch.empty(b, kv_heads, num_splits, group, dtype=torch.float32, device=dev)
    l_part = torch.empty_like(m_part)
    acc_part = torch.empty(
        b, kv_heads, num_splits, group, head_dim, dtype=torch.float32, device=dev
    )
    sp_b, sp_h, sp_s = m_part.stride()[:-1]

    _splitk_kernel[(b, kv_heads, num_splits)](
        grouped_q,
        k_pool,
        v_pool,
        block_table,
        positions,
        m_part,
        l_part,
        acc_part,
        sq_b,
        sq_h,
        sq_g,
        skp_row,
        skp_h,
        svp_row,
        svp_h,
        block_table.stride(0),
        sp_b,
        sp_h,
        sp_s,
        scale,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        GROUP=group,
        GROUP_PAD=max(16, triton.next_power_of_2(group)),
        TILE=TILE,
        NUM_SPLITS=num_splits,
        **({"num_warps": KERNEL_WARPS} if KERNEL_WARPS else {}),
        **({"num_stages": KERNEL_STAGES} if KERNEL_STAGES else {}),
    )

    out = torch.empty(b, kv_heads, group, head_dim, dtype=q.dtype, device=dev)
    so_b, so_h, so_g = _row_stride(out, "out")
    _combine_kernel[(b, kv_heads)](
        m_part,
        l_part,
        acc_part,
        positions,
        out,
        sp_b,
        sp_h,
        sp_s,
        so_b,
        so_h,
        so_g,
        HEAD_DIM=head_dim,
        GROUP=group,
        TILE=TILE,
        NUM_SPLITS=num_splits,
    )
    return out.reshape(b, heads, 1, head_dim)

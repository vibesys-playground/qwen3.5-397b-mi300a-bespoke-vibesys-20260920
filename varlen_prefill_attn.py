"""Fused Triton kernel for varlen paged prefill attention (`SEED_VARLEN_PREFILL_ATTN=1`).

Idea: FlashAttention-2's variable-length ("varlen") forward (Dao, "FlashAttention-2: Faster
Attention with Better Parallelism and Work Partitioning," 2023) -- pack several sequences'
queries end to end with no padding, address each one's own span via `cu_seqlens`, and run one
online-softmax recurrence per query tile instead of materializing a `[heads, T, S]` score
tensor -- combined with this bundle's own paged-KV block-table read (`paged_attn.py`, same
idea `decode_attention_paged` already applies to decode) so a resumed prefix's keys/values are
read straight out of the block-paged pool, in place, rather than gathered into a contiguous
scratch tensor first. Cited as an idea only; nothing here is copied from FlashAttention-2,
sglang, vllm, TensorRT-LLM, or aiter.

What this replaces: `model.py`'s `full_attention_packed` today loops `prefill_attention`
(an SDPA call) once per sequence in the packed batch, and `full_attention` itself first
materializes a *contiguous* `[1, kv_heads, start + t, head_dim]` K/V view per sequence
(`Model._gather_cached_prefix`, gather-once into `_prefill_scratch`) before that SDPA call can
run at all -- `paged-kv-design.md` section 3.2's deliberate Stage-1 simplification, justified
there by prefill not being graph-captured either way. That gather is one strided-index copy of
the whole resumed prefix per sequence per layer, and the per-sequence SDPA loop is `len(seqs)`
separate host dispatches and kernel launches per layer where one packed call would do (measured
~1 ms/token against a ~0.03 ms/token roofline SOL per `COMMON_BRIEF.md`). This kernel removes
both: no gather (every key/value read goes through `block_table` directly, cached-prefix rows
and this-chunk's freshly-written rows alike -- by the time this kernel runs, `full_attention`'s
own write-before-read ordering has already put this chunk's new K/V into the pool at the rows
its own block table maps them to, so "cached prefix" and "new tokens" are not actually two
different read paths here, just two different *ranges* of the same paged sequence, discriminated
only by the causal mask), and one launch covers every sequence in the packed batch.

GQA folding, same principle `paged_attn.py`'s decode kernel and `model.prefill_attention`'s own
docstring both already use: every kv head's key/value tile is loaded from the pool exactly
once per outer key-block iteration and shared across every query head in its group, rather than
re-reading it once per query head (`enable_gqa` under an explicit mask's fallback path -- see
`model.prefill_attention`'s docstring for the measured cost of that). Concretely, the kernel's
grid has a `kv_head` axis (this rank's *local* KV heads, 1 at TP=4 -- see `tp.kv_shard`), and
each program folds the group's query heads into a Python-level unrolled loop *inside* the key-
block loop, so the K/V tile loaded for one key block is reused by every group member before the
next block is read. `q`/`k`/`v` at TP=4: `pl.q.count == 8` query heads, `pl.kv.count == 1` KV
head, `GROUP = 8`, `head_dim == 256` (derived from `tp.kv_shard`/`tp.even`, not hardcoded here;
see `varlen_prefill_attention_paged`'s docstring for the exact per-rank arithmetic).

Precision: Q/K/V stay `bf16` into the `tl.dot` calls (unlike `paged_attn.py`'s decode kernel,
which promotes to fp32 before an elementwise product -- decode's `GROUP` rows are too few to
benefit from MFMA, prefill's `GROUP * BLOCK_M` query rows are not), so `S = Q @ K^T` is a native
bf16-in/fp32-accumulate MFMA op with no extra rounding on the way in. The softmax itself (max,
`exp`, sum) runs in fp32, standard for a numerically stable online-softmax recurrence. `P` is
cast to bf16 immediately before `P @ V` -- the one extra rounding point beyond the final output
cast, and the standard FlashAttention practice for feeding the second matmul through MFMA rather
than a slower fp32 dot; every published flash-attention kernel this idea is cited from does the
same. Final output is rounded once more, to the pool's storage dtype, matching every other fused
kernel in this bundle's single-rounding-at-the-store discipline for everything upstream of that
one already-standard extra P rounding.

Capture-safety does not apply: prefill is never graph-captured (`paged-kv-design.md` section
3.2, `paged_attn.py`'s own module docstring), so the launch grid here is sized per call from
live packed-batch metadata (`n_seqs`, each sequence's own chunk length), the same way
`full_attention_packed`'s existing per-sequence Python loop already varies per call.
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
    """Same contract as `paged_attn.available`/`decode_attn_v2.available`. `SEED_
    VARLEN_PREFILL_ATTN=0` (default) keeps the existing per-sequence gather + SDPA path."""
    if not HAVE_TRITON or os.environ.get("SEED_VARLEN_PREFILL_ATTN", "0") == "0":
        return False
    return device.type == "cuda"


BLOCK_M = int(os.environ.get("SEED_VARLEN_PREFILL_ATTN_BLOCK_M", "16"))
"""Query rows per program, before the GQA fold (`GM = GROUP * BLOCK_M` is the kernel's actual
`tl.dot` row count -- 128 at `GROUP = 8`, TP=4's per-rank head count). Offline `gfx942` compile
(`isa_attn_kernels.py`) is what set this default, not a guess: at `GROUP = 8`, `HEAD_DIM = 256`,
`num_warps = 8`, `BLOCK_M = 64` (`GM = 512`) allocates the full 512-register file and spills to
scratch memory (`scratch_load`/`scratch_store` both > 0 in the compiled ISA); `BLOCK_M = 16`
(`GM = 128`) is the widest tile in the swept grid {64, 32, 16} x {`num_warps` 4, 8} that
compiles with zero scratch traffic (`next_free_vgpr = 236` of the 512-register file, no
spill) -- see `varlen_prefill_attention_paged`'s docstring for the exact swept numbers."""

BLOCKS_PER_ITER = int(os.environ.get("SEED_VARLEN_PREFILL_ATTN_BLOCKS_PER_ITER", "1"))
"""Paged-pool blocks read together per key-tile iteration (`KEY_TILE = BLOCKS_PER_ITER *
block_size`), so the K/V matmul's `N` dimension is wide enough for MFMA to be worth issuing --
one block at a time (`block_size` = 16 tokens by default, `SEED_KV_BLOCK_SIZE`) would make
`N = 16`, the narrowest legal MFMA tile and mostly launch/reduction overhead. Block ids inside
one iteration need not be contiguous (the paged pool never promises that): the kernel loads
`BLOCKS_PER_ITER` block-table entries, builds each one's own `block_size`-wide row range, and
flattens the two axes into one `KEY_TILE`-long gather, exactly the multi-row gather `paged_attn.
py`'s single-block loop already does, just `BLOCKS_PER_ITER` blocks at a time instead of one.
Default `1` (so `KEY_TILE = 16`) for the same register-pressure reason `BLOCK_M`'s default is
16, not 64: at `BLOCK_M = 16`, `BLOCKS_PER_ITER = 2` already spills (105/59
`scratch_load`/`scratch_store`); `= 1` is the widest that does not, in the same sweep."""

NUM_WARPS = int(os.environ.get("SEED_VARLEN_PREFILL_ATTN_NUM_WARPS", "8"))
"""`num_warps` for the kernel launch. `8` (not Triton's default `4`) halves `next_free_vgpr`
at every `BLOCK_M`/`BLOCKS_PER_ITER` pair in the sweep (more warps sharing the same `GM`-row
tile means fewer rows, and so fewer live accumulator registers, per thread) -- `BLOCK_M = 16`,
`BLOCKS_PER_ITER = 1` goes from 489 registers (no spill, but within 23 of the 512-register
ceiling) at `num_warps = 4` to 236 (no spill, comfortable headroom) at `8`."""

if HAVE_TRITON:
    _INTERPRET = tl.constexpr(os.environ.get("TRITON_INTERPRET") == "1")
    """Same fix `mxfp4_gemv.py`'s `_BW_INTERPRET` already applies, for the same reason: the CPU
    interpreter's `tl.dot` on `bf16` operands is wrong (confirmed directly: a tiny isolated
    `tl.dot` under `TRITON_INTERPRET=1` at bf16 is off by ~1e10 against a plain `torch.matmul`
    reference, while the identical call at fp32 or fp16 is exact/near-exact -- numpy has no
    native bf16, and this Triton version's interpreter path through it does not round-trip
    correctly). `tl.dot`'s operands are fp32 under the interpreter only; the real compiled path
    (gfx942, what actually ships) still runs `bf16` MFMA. A logic-only substitution, not a
    numerics difference in what the GPU runs -- offline `gfx942` compilation (`isa.py`-style) is
    what confirms the real path still lowers to `v_mfma_f32_..._bf16`, not this workaround.
    """

    @triton.jit
    def _varlen_prefill_attn_kernel(
        q_ptr,
        kpool_ptr,
        vpool_ptr,
        bt_ptr,
        seq_start_ptr,
        seq_len_ptr,
        seq_prefix_ptr,
        o_ptr,
        sq_row,
        sq_h,
        sq_g,
        skp_row,
        skp_h,
        svp_row,
        svp_h,
        sbt_row,
        so_row,
        so_h,
        so_g,
        scale,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        GROUP: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCKS_PER_ITER: tl.constexpr,
    ):
        """One `(sequence, query-tile, kv_head)` program. See module docstring for the design.

        `q`/`o` are `[total_T, kv_heads, GROUP, HEAD_DIM]` (already folded by the wrapper, `total_
        T` packed end to end across `meta.seqs`, row-major within a sequence -- see
        `varlen_prefill_attention_paged`). `kpool`/`vpool` are `[num_blocks * block_size, kv_heads,
        HEAD_DIM]`, same convention as `paged_attn.py`. `bt` is `[n_seqs, max_blocks]` int32, this
        sequence's block ids, right-padded with `block_pool.RESERVED_BLOCK` (0) -- identical layout
        to the decode kernel's `bt`, just indexed by sequence instead of by lane. `seq_start` is
        `[n_seqs]`, the packed row this sequence's queries begin at; `seq_len` is `[n_seqs]`, this
        chunk's own new-token count (`t` in `full_attention`'s terms); `seq_prefix` is `[n_seqs]`,
        the resumed-prefix length already resident in the pool before this chunk (`start`). Query
        row `m` of this sequence sits at absolute position `seq_prefix + m`; the pool already holds
        every key up to `seq_prefix + seq_len` by the time this kernel runs (`full_attention`'s
        existing write-before-attend order), so causal visibility is exactly `key_pos <= query_pos`
        over that whole resident range -- no separate "prefix" vs. "new" branch inside the kernel.
        """
        seq = tl.program_id(0)
        qblk = tl.program_id(1)
        head = tl.program_id(2)

        t_i = tl.load(seq_len_ptr + seq)
        if qblk * BLOCK_M >= t_i:
            return

        row0 = tl.load(seq_start_ptr + seq) + qblk * BLOCK_M
        prefix = tl.load(seq_prefix_ptr + seq)
        valid_len = prefix + t_i  # keys resident for this sequence: cached prefix + this chunk

        # GQA fold: `GROUP` query heads become an extra, group-major leading axis merged into
        # the row dimension (`GM = GROUP * BLOCK_M` rows total, row `g * BLOCK_M + m` is group
        # `g`'s copy of local query position `m`) -- the same merge `model.prefill_attention`'s
        # own docstring already uses for its SDPA call, here built directly as one `tl.dot`
        # operand instead of a reshape on the torch side. Triton's AST front end does not
        # support a Python list of per-group tensors updated in a `GROUP`-trip loop (`.append`
        # inside a `@triton.jit` function is not a supported AST node, unlike the CPU
        # interpreter, which just runs the Python and does not care), so this fold is what
        # keeps the whole recurrence one vectorized flow: every `tl.dot` below multiplies the
        # *entire* `GM`-row query tile against one key tile in a single MFMA-mapped call, and a
        # key/value tile loaded once per iteration is shared across every group and every row
        # in the tile, not just every row.
        g = tl.arange(0, GROUP)
        m = tl.arange(0, BLOCK_M)
        q_row_ok_1d = (qblk * BLOCK_M + m) < t_i  # [BLOCK_M]
        q_abs_pos_1d = prefix + qblk * BLOCK_M + m  # [BLOCK_M]
        q_row_ok = tl.broadcast_to(q_row_ok_1d[None, :], (GROUP, BLOCK_M)).reshape(GROUP * BLOCK_M)
        q_abs_pos = tl.broadcast_to(q_abs_pos_1d[None, :], (GROUP, BLOCK_M)).reshape(
            GROUP * BLOCK_M
        )

        dot_dtype = tl.float32 if _INTERPRET else tl.bfloat16  # see module docstring, `_INTERPRET`
        d = tl.arange(0, HEAD_DIM)
        q_addr = (g[:, None] * sq_g + (row0 + m)[None, :] * sq_row).reshape(GROUP * BLOCK_M)
        q_tile = tl.load(
            q_ptr + q_addr[:, None] + head * sq_h + d[None, :],
            mask=q_row_ok[:, None],
            other=0.0,
        ).to(dot_dtype)  # [GM, HEAD_DIM]

        m_i = tl.full((GROUP * BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((GROUP * BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((GROUP * BLOCK_M, HEAD_DIM), dtype=tl.float32)

        tt = tl.arange(0, BLOCK_SIZE)
        bi = tl.arange(0, BLOCKS_PER_ITER)
        n_valid_blocks = tl.cdiv(valid_len, BLOCK_SIZE)
        n_iters = tl.cdiv(n_valid_blocks, BLOCKS_PER_ITER)
        for it in range(0, n_iters):
            blk_ids = it * BLOCKS_PER_ITER + bi
            blk_active = blk_ids < n_valid_blocks
            bids = tl.load(bt_ptr + seq * sbt_row + blk_ids, mask=blk_active, other=0)

            row_2d = bids[:, None] * BLOCK_SIZE + tt[None, :]  # [BLOCKS_PER_ITER, BLOCK_SIZE]
            logical_pos_2d = blk_ids[:, None] * BLOCK_SIZE + tt[None, :]
            resident_2d = tl.broadcast_to(blk_active[:, None], (BLOCKS_PER_ITER, BLOCK_SIZE)) & (
                logical_pos_2d < valid_len
            )
            row = row_2d.reshape(BLOCKS_PER_ITER * BLOCK_SIZE)
            logical_pos = logical_pos_2d.reshape(BLOCKS_PER_ITER * BLOCK_SIZE)
            resident = resident_2d.reshape(BLOCKS_PER_ITER * BLOCK_SIZE)

            k = tl.load(
                kpool_ptr + row[:, None] * skp_row + head * skp_h + d[None, :],
                mask=resident[:, None],
                other=0.0,
            ).to(dot_dtype)
            v = tl.load(
                vpool_ptr + row[:, None] * svp_row + head * svp_h + d[None, :],
                mask=resident[:, None],
                other=0.0,
            ).to(dot_dtype)
            k_t = tl.trans(k)  # [HEAD_DIM, KEY_TILE]

            scores = tl.dot(q_tile, k_t) * scale  # [GM, KEY_TILE], fp32 accumulate
            causal = logical_pos[None, :] <= q_abs_pos[:, None]
            live = resident[None, :] & causal & q_row_ok[:, None]
            scores = tl.where(live, scores, float("-inf"))

            blk_max = tl.max(scores, axis=1)
            new_m = tl.maximum(m_i, blk_max)
            # A fully masked tile for a still-live query row (e.g. this row's own causal
            # frontier has not reached any key in this key-tile yet) leaves new_m == -inf;
            # `exp(-inf - -inf)` is nan, not 0, so guard alpha/p the same way a fully-empty
            # split does in `decode_attn_v2.py`'s reduce kernel.
            new_m_safe = tl.where(new_m == float("-inf"), 0.0, new_m)
            alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - new_m_safe))
            p = tl.where(scores == float("-inf"), 0.0, tl.exp(scores - new_m_safe[:, None]))
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(dot_dtype), v)
            m_i = new_m

        safe_l = tl.where(l_i == 0.0, 1.0, l_i)  # unread row (masked out entirely): avoid 0/0
        out = acc / safe_l[:, None]
        # A fresh address tensor, not `q_addr`: `o`'s strides (`so_g`/`so_row`) need not equal
        # `q`'s (`sq_g`/`sq_row`) unless both happen to be default-contiguous with the same
        # shape, which the wrapper does not guarantee for an arbitrary caller's `q`.
        o_addr = (g[:, None] * so_g + (row0 + m)[None, :] * so_row).reshape(GROUP * BLOCK_M)
        ptrs = o_ptr + o_addr[:, None] + head * so_h + d[None, :]
        tl.store(ptrs, out.to(o_ptr.dtype.element_ty), mask=q_row_ok[:, None])


def _row_stride(x: torch.Tensor, name: str) -> tuple[int, ...]:
    if x.stride(-1) != 1:
        raise ValueError(f"{name} must be contiguous along its last axis, got stride {x.stride()}")
    return x.stride()[:-1]


def varlen_prefill_attention_paged(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_table: torch.Tensor,
    seq_start: torch.Tensor,
    seq_len: torch.Tensor,
    seq_prefix: torch.Tensor,
    block_size: int,
    scale: float,
) -> torch.Tensor:
    """One packed varlen prefill call over `n_seqs` sequences, reading every key/value through
    `block_table` instead of a per-sequence contiguous gather. Drop-in for the attention half of
    `model.full_attention_packed`'s per-sequence loop (projections, RoPE, and the paged KV write
    stay in `model.py`; this call replaces only the `prefill_attention`/SDPA step, run once for
    the whole packed batch instead of once per sequence).

    `q` is `[total_T, heads, head_dim]`, packed end to end in `meta.spans()` order (`heads` is
    this rank's *local* query head count -- `pl.q.count`, 8 at TP=4 for `heads=32` unsharded
    over 4 ranks via `tp.even`). `k_pool`/`v_pool` are `[num_blocks * block_size, kv_heads,
    head_dim]` (`kv_heads` local, 1 at TP=4 -- `tp.kv_shard` replicates the 2 global KV heads
    over `world // kv_heads == 2` ranks each, so every rank's query block reads exactly one
    local KV head; see `tp.kv_shard`'s docstring for the derivation). `block_table` is
    `[n_seqs, max_blocks]` int32, one row per sequence (`block_pool.BlockTable.padded_row`,
    same convention as decode's), already grown to cover `seq_prefix + seq_len` tokens by the
    caller (`full_attention`'s `table.grow_to` call, unchanged). `seq_start`/`seq_len`/
    `seq_prefix` are `[n_seqs]` int32: `seq_start[i]` is `meta.spans()[i][0]`, `seq_len[i]` is
    `meta.seqs[i].length`, `seq_prefix[i]` is `meta.seqs[i].start`. Returns `[total_T, heads,
    head_dim]`, matching `q`'s packed layout.

    `BLOCK_M`/`BLOCKS_PER_ITER`/`NUM_WARPS` defaults, from an offline `gfx942` compile
    (`GROUP=8`, `HEAD_DIM=256`, `BLOCK_SIZE=16`, TP=4's served shape) sweeping tile size against
    `next_free_vgpr` and scratch (spill) traffic in the compiled ISA:

    | BLOCK_M | BLOCKS_PER_ITER | num_warps | vgpr | scratch_load/store |
    |--------:|-----------------:|----------:|-----:|--------------------|
    |      64 |                4 |         4 |  512 | 1238 / 1227 (spills) |
    |      32 |                1 |         4 |  512 |   219 / 224 (spills) |
    |      16 |                1 |         4 |  489 |          0 / 0       |
    |      32 |                1 |         8 |  256 |    78 / 64 (spills)  |
    |  **16** |            **1** |     **8** |  **236** |      **0 / 0**   |
    |       8 |                1 |         8 |  320 |          0 / 0       |

    `next_free_vgpr` reports the combined VGPR+AGPR file (gfx942 CDNA3, `accum_offset` splits
    it -- 256 architected VGPRs, 256 AGPRs for MFMA accumulation, 512 total); anything at 512
    is out of registers and spilling to scratch memory, a real, measured cost (extra global
    memory traffic on every spilled access), not a modeling estimate. `16/1/8` is the widest
    tile in the swept grid with zero scratch traffic, so it is the default; `8/1/8` is a safer
    fallback with more headroom (320 of 512) if the real compiler's allocator behaves
    differently at other shapes the sweep did not cover.
    """
    total_t, heads, head_dim = q.shape
    kv_heads = k_pool.shape[1]
    if heads % kv_heads:
        raise ValueError(f"heads {heads} is not a multiple of kv_heads {kv_heads}")
    group = heads // kv_heads
    n_seqs = seq_start.shape[0]
    if block_table.stride(-1) != 1:
        raise ValueError("block_table must be contiguous along its last axis")

    grouped_q = q.reshape(total_t, kv_heads, group, head_dim)
    sq_row, sq_h, sq_g = _row_stride(grouped_q, "q")
    skp_row, skp_h = _row_stride(k_pool, "k_pool")
    svp_row, svp_h = _row_stride(v_pool, "v_pool")

    out = torch.empty(total_t, kv_heads, group, head_dim, dtype=q.dtype, device=q.device)
    so_row, so_h, so_g = _row_stride(out, "out")

    max_chunk = int(seq_len.max().item()) if n_seqs else 0
    max_q_blocks = -(-max_chunk // BLOCK_M) if max_chunk else 0
    if max_q_blocks == 0:
        return out.reshape(total_t, heads, head_dim)

    _varlen_prefill_attn_kernel[(n_seqs, max_q_blocks, kv_heads)](
        grouped_q,
        k_pool,
        v_pool,
        block_table,
        seq_start,
        seq_len,
        seq_prefix,
        out,
        sq_row,
        sq_h,
        sq_g,
        skp_row,
        skp_h,
        svp_row,
        svp_h,
        block_table.stride(0),
        so_row,
        so_h,
        so_g,
        scale,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        GROUP=group,
        BLOCK_M=BLOCK_M,
        BLOCKS_PER_ITER=BLOCKS_PER_ITER,
        num_warps=NUM_WARPS,
    )
    return out.reshape(total_t, heads, head_dim)

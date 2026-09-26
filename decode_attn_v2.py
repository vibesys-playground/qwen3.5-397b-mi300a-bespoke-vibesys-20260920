"""Split-KV ("flash-decoding") paged decode attention: `SEED_DECODE_ATTN_V2=1`.

Idea: Dao, Haziza, Massa, Sizov, "Flash-Decoding for long-context inference" (PyTorch blog,
2023), applied on top of the online-softmax recurrence FlashAttention-2 (Dao, 2023) already
uses -- both cited as ideas only, nothing here is copied from either implementation or from
any sglang/vllm/TensorRT-LLM/aiter source.

`paged_attn.py`'s base kernel (`_paged_attn_decode_kernel`) launches one program per `(lane,
kv_head)` and loops that program serially over every valid block of the lane's context
(6b3cf965 made that loop skip past `n_valid_blocks`, not `MAX_BLOCKS`, on the common path).
That is fine when the grid itself has enough programs to fill the device -- `lanes * kv_heads`
at TP=4 with `kv_heads` local = 1, so the grid is just `lanes` wide. At `b48` that is 48
programs, plenty. At batch 1 (a single resumed session, the "32k/128k single-sequence decode"
shape this module's docstring is asked to predict) the grid is *one* program, whatever the
context length: one CU active, the other seven idle for the length of a 2,048- or 8,192-block
serial loop. Splitting that one lane's context across several programs, each covering a
disjoint slice of blocks and writing a *partial* online-softmax state, turns "grid width =
batch size" into "grid width = batch size x splits" -- occupancy that scales with context
length instead of collapsing at low batch, exactly Flash-Decoding's argument. A second,
lightweight kernel (`_decode_attn_v2_reduce_kernel`) merges each lane's partial states with the
same online-softmax combine rule the block loop already uses internally, one extra read of
`NUM_SPLITS` small partials per lane instead of the full context.

Same capture-safety posture as the base kernel (see its module docstring): the *grid* here is
`(lanes, kv_heads, NUM_SPLITS)` for the split kernel and `(lanes, kv_heads)` for the reduce
kernel, both fixed by `max_batch`/`kv_heads`/`NUM_SPLITS` -- none of which are device data, so
neither shape changes between replays. Each split's own inner loop trips `n_local` times, a
*runtime* value (`min(max(n_valid_blocks - split_start, 0), BLOCKS_PER_SPLIT)`, read from
`bvalid_ptr`), not the `BLOCKS_PER_SPLIT` constexpr -- confirmed via offline `gfx942` compile
(`isa_attn_kernels.py`) that this compiles to one real device-side loop whose static code size
and register count do not change with `BLOCKS_PER_SPLIT` (checked at 128/64/32/16, all
identical), the same property `paged_attn.py`'s own base kernel gets from bounding its loop by
`n_iter` (a `tl.minimum(...)` value) instead of a compile-time trip count -- a `range()` over a
plain `tl.constexpr` upper bound, by contrast, is what `mxfp4_gemv.py`'s `_BW_INTERPRET` note
calls out the interpreter needing a workaround for, and is also what this split kernel used to
do before that offline compile caught it fully unrolling to ~49k instructions and spilling to
scratch memory. A split whose whole range sits past a lane's live context runs `n_local = 0`
iterations (not `BLOCKS_PER_SPLIT` masked no-op ones) and writes an empty partial (`m = -inf`,
`l = 0`), which the reduce kernel discards via the same `exp(-inf - max) = 0` identity a
flash-attention combine already relies on. Every kernel input buffer here is rewritten on every
replay by the same `GraphDecodeRunner.fill`/`Buffers.fill` contract `graph_decode.py`'s module
docstring already documents for the base kernel's `bt`/`bvalid`/`pos`; this module changes none
of that, only which kernel(s) `model.decode_attention_paged` dispatches to.

`NUM_SPLITS`/`BLOCKS_PER_SPLIT` are derived once, host-side, from `block_table.shape[1]` (the
lane row's fixed width, `block_pool.max_blocks_per_lane(max_seq, block_size)` -- a deploy-time
constant, not live position), not from any lane's actual `block_valid`: a captured graph must
pick one grid and stick to it, so "only split for long contexts" here means "only split when
the *served* `max-seq-len` is long enough to be worth it" (`_split_config`'s docstring), not "only
split lanes that happen to be long on this particular replay." A lane that is actually short
(most of the early decode steps in a long-max-seq deployment) still launches every split's
program; the ones past its own `n_valid_blocks` do one cheap masked pass and contribute nothing,
which is the same trade-off the base kernel's own non-`STATIC_LOOP` default already makes at
the single-program level (predicated loads over a fixed-width row, see its docstring) --
this module just adds a second, coarser level of the same trade-off, sized so it stays a net
win once `max-seq-len` clears ~8k tokens (`_split_config`'s threshold; see its docstring for
the derivation and this module's predicted numbers in the campaign report).

Precision: same fp32 online-softmax chain as the base kernel (q/k promoted to fp32, scores and
the softmax recurrence in fp32), with one extra rounding point -- the split kernel's partial
`acc`/`l`/`m` are stored in fp32 (never rounded to bf16, so the reduce kernel's combine sees the
same numbers the single-program kernel's internal recurrence would have) and the final output
is rounded once, in the reduce kernel, to the pool's storage dtype. Two kernel launches instead
of one changes the reduction *order* (across-split combine vs. one program's serial fold) but
not the arithmetic itself, the same reduction-order caveat `paged_attn.py`'s own docstring
already carries against `decode_attention`'s SDPA-backed path.
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
    """Same contract as `paged_attn.available`. `SEED_DECODE_ATTN_V2=0` (default) keeps the
    base single-pass kernel; `=1` switches `model.decode_attention_paged` to this module."""
    if not HAVE_TRITON or os.environ.get("SEED_DECODE_ATTN_V2", "0") == "0":
        return False
    return device.type == "cuda"


LONG_CONTEXT_TOKENS = int(os.environ.get("SEED_DECODE_ATTN_V2_THRESHOLD_TOKENS", "8192"))
"""Below this served `max-seq-len`, splitting is not worth a second kernel launch's overhead
(a short-context deployment's serial block loop is already cheap); see `_split_config`."""

MAX_SPLITS = int(os.environ.get("SEED_DECODE_ATTN_V2_MAX_SPLITS", "8"))
"""Cap on parallel splits per lane. 8 matches the Flash-Decoding blog post's own default range
and keeps the reduce kernel's per-lane fan-in small (one short unrolled loop, not a device-side
one) -- see `_decode_attn_v2_reduce_kernel`."""

MIN_BLOCKS_PER_SPLIT = int(os.environ.get("SEED_DECODE_ATTN_V2_MIN_BLOCKS_PER_SPLIT", "64"))
"""Below this many blocks, a lane's row does not get split at all (`_split_config` returns
`NUM_SPLITS=1`): 64 blocks is 1,024 tokens at the default 16-token block, i.e. splitting stays
off until a served context meaningfully exceeds one program's already-cheap serial loop."""


def _split_config(max_blocks: int, block_size: int) -> tuple[int, int]:
    """`(NUM_SPLITS, BLOCKS_PER_SPLIT)` for a lane row `max_blocks` wide, derived once from the
    deploy-time row width (see module docstring for why this cannot depend on live position).

    `NUM_SPLITS=1` (every split-kernel program covers the whole row, degenerating to exactly the
    base kernel's own loop, just wrapped in a second, near-instant reduce launch) whenever the
    served context is under `LONG_CONTEXT_TOKENS` or the row is too narrow to clear
    `MIN_BLOCKS_PER_SPLIT`; otherwise as many `MIN_BLOCKS_PER_SPLIT`-sized splits as fit, capped
    at `MAX_SPLITS`.
    """
    if max_blocks * block_size < LONG_CONTEXT_TOKENS or max_blocks < MIN_BLOCKS_PER_SPLIT:
        return 1, max_blocks
    num_splits = min(MAX_SPLITS, max_blocks // MIN_BLOCKS_PER_SPLIT)
    blocks_per_split = -(-max_blocks // num_splits)  # ceil
    return num_splits, blocks_per_split


if HAVE_TRITON:

    @triton.jit
    def _decode_attn_v2_split_kernel(
        q_ptr,
        kpool_ptr,
        vpool_ptr,
        bt_ptr,
        bvalid_ptr,
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
        sacc_b,
        sacc_h,
        sacc_s,
        sacc_g,
        sml_b,
        sml_h,
        sml_s,
        scale,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        GROUP: tl.constexpr,
        BLOCKS_PER_SPLIT: tl.constexpr,
    ):
        """One `(lane, kv_head, split)` program: the base kernel's own block loop, restricted to
        this split's `[split * BLOCKS_PER_SPLIT, (split + 1) * BLOCKS_PER_SPLIT)` block range,
        writing a partial `(m, l, acc)` instead of a final output. See module docstring.
        """
        lane = tl.program_id(0)
        head = tl.program_id(1)
        split = tl.program_id(2)
        g = tl.arange(0, GROUP)
        d = tl.arange(0, HEAD_DIM)

        q = tl.load(q_ptr + lane * sq_b + head * sq_h + g[:, None] * sq_g + d[None, :]).to(
            tl.float32
        )

        valid_len = tl.load(pos_ptr + lane) + 1
        n_valid_blocks = tl.load(bvalid_ptr + lane)
        split_start = split * BLOCKS_PER_SPLIT
        n_local = tl.minimum(
            tl.maximum(n_valid_blocks - split_start, 0), BLOCKS_PER_SPLIT
        )

        m_i = tl.full((GROUP,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((GROUP,), dtype=tl.float32)
        acc = tl.zeros((GROUP, HEAD_DIM), dtype=tl.float32)

        t = tl.arange(0, BLOCK_SIZE)
        # Bounded by the runtime value `n_local`, not the `BLOCKS_PER_SPLIT` constexpr: a
        # `range()` over a `tl.constexpr` gets fully unrolled at compile time (confirmed via
        # offline `gfx942` compilation -- `BLOCKS_PER_SPLIT=128` at the deployed `--max-seq-len`
        # unrolled to ~49k instructions and spilled to scratch memory), where a runtime-valued
        # bound compiles to an actual device-side loop, exactly `paged_attn.py`'s own base
        # kernel's `for blk in range(0, n_iter)` (`n_iter` a `tl.minimum(...)` value, not a
        # constexpr). Every `j` here already satisfies `j < n_local <= BLOCKS_PER_SPLIT`, so
        # the base kernel's separate `active` predicate (needed there for its `STATIC_LOOP`
        # A/B mode, absent here) is not needed: every iteration this loop actually runs reads a
        # real, in-range block.
        for j in range(0, n_local):
            blk = split_start + j
            block_id = tl.load(bt_ptr + lane * sbt_b + blk)
            logical_pos = blk * BLOCK_SIZE + t
            tok_live = logical_pos < valid_len
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

            scores = tl.sum(q[:, None, :] * k[None, :, :], axis=2) * scale
            scores = tl.where(tok_live[None, :], scores, float("-inf"))

            blk_max = tl.max(scores, axis=1)
            new_m = tl.maximum(m_i, blk_max)
            # Unlike the base kernel's per-lane loop (always >= 1 valid block for any lane at
            # a real position), a split here can be entirely past `n_valid_blocks` -- every
            # `tok_live` in it is False, so `blk_max` and `new_m` are both `-inf`. Unguarded,
            # `exp(m_i - new_m)` is then `exp(-inf - -inf) = exp(nan) = nan`, poisoning `l_i`/
            # `acc` for what must stay an all-zero contribution. Guard exactly at that one
            # `-inf - -inf` case, same fix `varlen_prefill_attn.py`'s kernel needs for the same
            # reason (a query row whose causal frontier has not reached a key-tile yet).
            new_m_safe = tl.where(new_m == float("-inf"), 0.0, new_m)
            alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - new_m_safe))
            p = tl.where(scores == float("-inf"), 0.0, tl.exp(scores - new_m_safe[:, None]))
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * v[None, :, :], axis=1)
            m_i = new_m

        base = lane * sacc_b + head * sacc_h + split * sacc_s
        tl.store(m_ptr + lane * sml_b + head * sml_h + split * sml_s + g, m_i)
        tl.store(l_ptr + lane * sml_b + head * sml_h + split * sml_s + g, l_i)
        tl.store(acc_ptr + base + g[:, None] * sacc_g + d[None, :], acc)

    @triton.jit
    def _decode_attn_v2_reduce_kernel(
        m_ptr,
        l_ptr,
        acc_ptr,
        o_ptr,
        sacc_b,
        sacc_h,
        sacc_s,
        sacc_g,
        sml_b,
        sml_h,
        sml_s,
        so_b,
        so_h,
        so_g,
        HEAD_DIM: tl.constexpr,
        GROUP: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
    ):
        """One `(lane, kv_head)` program: combine `NUM_SPLITS` partial online-softmax states
        into the final output, via the standard flash-attention merge rule (global max, rescale
        each split's numerator/denominator by `exp(local_max - global_max)`, sum). `NUM_SPLITS`
        is small (`<= MAX_SPLITS`) and a `tl.constexpr`, so this loads every split at once as one
        extra leading axis rather than looping.
        """
        lane = tl.program_id(0)
        head = tl.program_id(1)
        s = tl.arange(0, NUM_SPLITS)
        g = tl.arange(0, GROUP)
        d = tl.arange(0, HEAD_DIM)

        m = tl.load(m_ptr + lane * sml_b + head * sml_h + s[:, None] * sml_s + g[None, :])
        l = tl.load(l_ptr + lane * sml_b + head * sml_h + s[:, None] * sml_s + g[None, :])
        acc = tl.load(
            acc_ptr
            + lane * sacc_b
            + head * sacc_h
            + s[:, None, None] * sacc_s
            + g[None, :, None] * sacc_g
            + d[None, None, :]
        )

        global_m = tl.max(m, axis=0)  # [GROUP]
        alpha = tl.exp(m - global_m[None, :])  # [NUM_SPLITS, GROUP]
        l_total = tl.sum(l * alpha, axis=0)  # [GROUP]
        acc_total = tl.sum(acc * alpha[:, :, None], axis=0)  # [GROUP, HEAD_DIM]

        out = acc_total / l_total[:, None]
        tl.store(
            o_ptr + lane * so_b + head * so_h + g[:, None] * so_g + d[None, :],
            out.to(o_ptr.dtype.element_ty),
        )


def _row_stride(x: torch.Tensor, name: str) -> tuple[int, ...]:
    if x.stride(-1) != 1:
        raise ValueError(f"{name} must be contiguous along its last axis, got stride {x.stride()}")
    return x.stride()[:-1]


def decode_attention_paged_v2(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_table: torch.Tensor,
    block_valid: torch.Tensor,
    positions: torch.Tensor,
    block_size: int,
    scale: float,
) -> torch.Tensor:
    """Split-KV drop-in for `paged_attn.decode_attention_paged`: identical signature, shapes,
    and return value (`[B, heads, 1, head_dim]`), so `model.decode_attention_paged` can switch
    between them with no reshape at the call site. See module docstring for the design.
    """
    b, heads, one, head_dim = q.shape
    if one != 1:
        raise ValueError(f"decode_attention_paged_v2 is decode-only (T=1), got q with T={one}")
    kv_heads = k_pool.shape[1]
    if heads % kv_heads:
        raise ValueError(f"heads {heads} is not a multiple of kv_heads {kv_heads}")
    group = heads // kv_heads
    max_blocks = block_table.shape[1]
    if block_table.stride(-1) != 1:
        raise ValueError("block_table must be contiguous along its last axis")

    num_splits, blocks_per_split = _split_config(max_blocks, block_size)

    grouped_q = q.reshape(b, kv_heads, group, head_dim)
    sq_b, sq_h, sq_g = _row_stride(grouped_q, "q")
    skp_row, skp_h = _row_stride(k_pool, "k_pool")
    svp_row, svp_h = _row_stride(v_pool, "v_pool")

    dev = q.device
    m_part = torch.empty(b, kv_heads, num_splits, group, dtype=torch.float32, device=dev)
    l_part = torch.empty(b, kv_heads, num_splits, group, dtype=torch.float32, device=dev)
    acc_part = torch.empty(
        b, kv_heads, num_splits, group, head_dim, dtype=torch.float32, device=dev
    )
    sacc_b, sacc_h, sacc_s, sacc_g = acc_part.stride()[:-1]
    sml_b, sml_h, sml_s = m_part.stride()[:-1]

    _decode_attn_v2_split_kernel[(b, kv_heads, num_splits)](
        grouped_q,
        k_pool,
        v_pool,
        block_table,
        block_valid,
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
        sacc_b,
        sacc_h,
        sacc_s,
        sacc_g,
        sml_b,
        sml_h,
        sml_s,
        scale,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        GROUP=group,
        BLOCKS_PER_SPLIT=blocks_per_split,
    )

    out = torch.empty(b, kv_heads, group, head_dim, dtype=q.dtype, device=dev)
    so_b, so_h, so_g = _row_stride(out, "out")
    _decode_attn_v2_reduce_kernel[(b, kv_heads)](
        m_part,
        l_part,
        acc_part,
        out,
        sacc_b,
        sacc_h,
        sacc_s,
        sacc_g,
        sml_b,
        sml_h,
        sml_s,
        so_b,
        so_h,
        so_g,
        HEAD_DIM=head_dim,
        GROUP=group,
        NUM_SPLITS=num_splits,
    )
    return out.reshape(b, heads, 1, head_dim)

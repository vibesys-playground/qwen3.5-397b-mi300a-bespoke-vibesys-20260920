"""Capture-safe BF16 routed experts for the MTP draft layer on MI300A.

The MTP checkpoint stores dense BF16 experts while the target layers use MXFP4.  Gathering
one full expert tensor per assignment is especially costly under TP: at the production shape
(`B=96`, top-k=10, H=4096, I=1024), it copies 24 GiB of weights per rank before either GEMM
and copies remote assignments too.  These kernels leave the rank's `[128, ...]` expert
tensors resident and use the routed expert id only as an indirect base address.

The launch grids depend on tensor shapes, not routing values, so both launches are HIP graph
capturable.  A program tests ownership and row activity before loading an expert weight.  The
gate/up launch writes one fixed scratch row per assignment.  The down launch owns one token's
whole top-k reduction, applies each routing weight once, and writes without atomics.  Duplicate
experts therefore have the same semantics as distinct assignments and deterministic combine
order.

Roofline, one TP rank at B=96: uniform routing gives about 240 local assignments.  Each local
assignment streams `2 * I * H + H * I` BF16 values, 24 MiB, or about 5.6 GiB total.  MI300A's
5.3 TB/s peak gives a 1.1 ms HBM floor per draft step before launch and cache effects.  The
old gather plus bmm path moves at least 48 GiB (24 GiB gather writes plus 24 GiB bmm reads),
an 8.6 ms ideal floor.  Both paths do the same useful expert arithmetic; this path removes
remote work and the materialized-weight round trip.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - accelerator-only optional dependency
    HAVE_TRITON = False


BLOCK_N = 32
BLOCK_K = 64
DOT_M = 16


def available(h: torch.Tensor, experts: dict) -> bool:
    """Whether the dense kernel supports this input and resident expert layout."""
    return (
        HAVE_TRITON
        and h.device.type == "cuda"
        and h.dtype == torch.bfloat16
        and set(experts) == {"gate_up", "down"}
        and experts["gate_up"].dtype == torch.bfloat16
        and experts["down"].dtype == torch.bfloat16
        and h.is_contiguous()
        and experts["gate_up"].is_contiguous()
        and experts["down"].is_contiguous()
    )


def scratch_bytes(max_tokens: int, top_k: int, intermediate: int, hidden: int) -> int:
    """Persistent scratch required by :func:`fused_moe` (BF16 bytes)."""
    return 2 * (max_tokens * top_k * intermediate + max_tokens * hidden)


if HAVE_TRITON:

    @triton.jit
    def _gather_rows_kernel(
        src_ptr,
        row_ptr,
        out_ptr,
        src_stride,
        out_stride,
        cols,
        BLOCK: tl.constexpr,
    ):
        dst_row = tl.program_id(0)
        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        src_row = tl.load(row_ptr + dst_row).to(tl.int64)
        value = tl.load(src_ptr + src_row * src_stride + col, mask=col < cols)
        tl.store(out_ptr + dst_row * out_stride + col, value, mask=col < cols)

    @triton.jit
    def _scatter_time_active_rows_kernel(
        dst_ptr,
        row_ptr,
        src_ptr,
        time_ptr,
        active_ptr,
        dst_stride,
        src_batch_stride,
        src_time_stride,
        cols,
        BLOCK: tl.constexpr,
    ):
        src_row = tl.program_id(0)
        if tl.load(active_ptr + src_row):
            col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
            dst_row = tl.load(row_ptr + src_row).to(tl.int64)
            time = tl.load(time_ptr + src_row).to(tl.int64)
            value = tl.load(
                src_ptr + src_row * src_batch_stride + time * src_time_stride + col,
                mask=col < cols,
            )
            tl.store(dst_ptr + dst_row * dst_stride + col, value, mask=col < cols)

    @triton.jit
    def _scatter_active_rows_kernel(
        dst_ptr,
        row_ptr,
        src_ptr,
        active_ptr,
        dst_stride,
        src_stride,
        cols,
        ACTIVE_REPEAT: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        src_row = tl.program_id(0)
        if tl.load(active_ptr + src_row // ACTIVE_REPEAT):
            col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
            dst_row = tl.load(row_ptr + src_row).to(tl.int64)
            value = tl.load(src_ptr + src_row * src_stride + col, mask=col < cols)
            tl.store(dst_ptr + dst_row * dst_stride + col, value, mask=col < cols)

    @triton.jit
    def _dense_gate_up_kernel(
        x_ptr,
        gate_up_ptr,
        expert_ptr,
        weight_ptr,
        active_ptr,
        inter_ptr,
        expert_lo,
        expert_hi,
        K: tl.constexpr,
        INTER: tl.constexpr,
        TOP_K: tl.constexpr,
        HAS_ACTIVE: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        BM: tl.constexpr,
    ):
        assignment = tl.program_id(0)
        expert = tl.load(expert_ptr + assignment)
        combine = tl.load(weight_ptr + assignment)
        token = assignment // TOP_K
        live = combine != 0.0
        if HAS_ACTIVE:
            live = live & tl.load(active_ptr + token)

        # Uniform scalar branch per program. Remote assignments return before either the
        # activation or the selected expert's weight base is dereferenced.
        if (expert >= expert_lo) & (expert < expert_hi) & live:
            local = (expert - expert_lo).to(tl.int64)
            n = tl.program_id(1) * BN + tl.arange(0, BN)
            n_ok = n < INTER
            rows = tl.arange(0, BM)
            acc_gate = tl.zeros((BN, BM), dtype=tl.float32)
            acc_up = tl.zeros((BN, BM), dtype=tl.float32)
            base = local * (2 * INTER) * K
            for k0 in range(0, K, BK):
                k = k0 + tl.arange(0, BK)
                x = tl.load(x_ptr + token * K + rows[:, None] * 0 + k[None, :])
                gate = tl.load(
                    gate_up_ptr + base + n[:, None] * K + k[None, :],
                    mask=n_ok[:, None],
                    other=0.0,
                )
                up = tl.load(
                    gate_up_ptr + base + (n[:, None] + INTER) * K + k[None, :],
                    mask=n_ok[:, None],
                    other=0.0,
                )
                acc_gate = tl.dot(gate, tl.trans(x), acc_gate)
                acc_up = tl.dot(up, tl.trans(x), acc_up)
            dt = inter_ptr.dtype.element_ty
            # Every MFMA output column is identical because ``x`` is broadcast over BM.
            # Triton 3.2 cannot index a tensor with ``[:, 0]``; reduce the identical columns.
            gate0 = (tl.sum(acc_gate, axis=1) / BM).to(dt)
            up0 = (tl.sum(acc_up, axis=1) / BM).to(dt)
            gatef = gate0.to(tl.float32)
            silu = (gatef / (1.0 + tl.exp(-gatef))).to(dt)
            act = (silu.to(tl.float32) * up0.to(tl.float32)).to(dt)
            tl.store(inter_ptr + assignment * INTER + n, act, mask=n_ok)

    @triton.jit
    def _dense_down_combine_kernel(
        inter_ptr,
        down_ptr,
        expert_ptr,
        weight_ptr,
        active_ptr,
        out_ptr,
        expert_lo,
        expert_hi,
        K: tl.constexpr,
        HIDDEN: tl.constexpr,
        TOP_K: tl.constexpr,
        HAS_ACTIVE: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        BM: tl.constexpr,
    ):
        token = tl.program_id(0)
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        n_ok = n < HIDDEN
        token_live = True
        if HAS_ACTIVE:
            token_live = tl.load(active_ptr + token)
        total = tl.zeros((BN,), dtype=tl.float32)
        if token_live:
            for j in range(TOP_K):
                assignment = token * TOP_K + j
                expert = tl.load(expert_ptr + assignment)
                combine = tl.load(weight_ptr + assignment).to(tl.float32)
                if (expert >= expert_lo) & (expert < expert_hi) & (combine != 0.0):
                    local = (expert - expert_lo).to(tl.int64)
                    rows = tl.arange(0, BM)
                    acc = tl.zeros((BN, BM), dtype=tl.float32)
                    base = local * HIDDEN * K
                    for k0 in range(0, K, BK):
                        k = k0 + tl.arange(0, BK)
                        act = tl.load(inter_ptr + assignment * K + rows[:, None] * 0 + k[None, :])
                        w = tl.load(
                            down_ptr + base + n[:, None] * K + k[None, :],
                            mask=n_ok[:, None],
                            other=0.0,
                        )
                        acc = tl.dot(w, tl.trans(act), acc)
                    total += (tl.sum(acc, axis=1) / BM) * combine
        tl.store(out_ptr + token * HIDDEN + n, total, mask=n_ok)


def gather_rows(src: torch.Tensor, rows: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Capture-safe ``out.copy_(src[rows])`` for contiguous two-dimensional tensors.

    ``rows`` may be a strided view (a column of a ``[B, t]`` buffer): the kernel reads it as a
    dense vector, so it is made contiguous first. Without that, row ``j`` read flat element
    ``j`` of the parent buffer instead of its own."""
    rows = rows.reshape(-1).contiguous()
    if src.dim() != 2 or out.shape != (rows.numel(), src.shape[1]):
        raise ValueError(f"gather_rows shape mismatch: {src.shape}, {rows.shape}, {out.shape}")
    if src.stride(1) != 1 or out.stride(1) != 1:
        raise ValueError("gather_rows requires a contiguous column axis")
    if src.device.type != "cuda" or not HAVE_TRITON:
        out.copy_(src[rows])
        return out
    block = 256
    _gather_rows_kernel[(rows.numel(), triton.cdiv(src.shape[1], block))](
        src,
        rows,
        out,
        src.stride(0),
        out.stride(0),
        src.shape[1],
        BLOCK=block,
        num_warps=4,
    )
    return out


def gather_embeddings(
    table: torch.Tensor, ids: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """Capture-safe embedding lookup into caller-owned fixed-shape storage."""
    expected = (*ids.shape, table.shape[1])
    if out.shape != expected:
        raise ValueError(f"gather_embeddings shape mismatch: {table.shape}, {ids.shape}, {out.shape}")
    gather_rows(table, ids.reshape(-1), out.reshape(-1, table.shape[1]))
    return out


def gather_first_dim(src: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Capture-safe advanced-index replacement for a contiguous tensor's first axis."""
    if not src.is_contiguous():
        raise ValueError("gather_first_dim requires a contiguous source")
    out = torch.empty((rows.numel(), *src.shape[1:]), dtype=src.dtype, device=src.device)
    gather_rows(src.view(src.shape[0], -1), rows, out.view(rows.numel(), -1))
    return out


def scatter_active_first_dim(
    dst: torch.Tensor,
    rows: torch.Tensor,
    src: torch.Tensor,
    active: torch.Tensor,
    active_repeat: int = 1,
) -> None:
    """Write source rows to dynamic destination rows only for active logical lanes.

    ``rows``/``active`` may be strided views; the kernel reads them as dense vectors, so they
    are made contiguous first (see `gather_rows`)."""
    rows, active = rows.reshape(-1).contiguous(), active.reshape(-1).contiguous()
    if not dst.is_contiguous() or not src.is_contiguous():
        raise ValueError(
            "scatter_active_first_dim requires contiguous tensors: "
            f"dst={dst.shape}/{dst.stride()}, src={src.shape}/{src.stride()}"
        )
    if src.shape[0] != rows.numel() or rows.numel() != active.numel() * active_repeat:
        raise ValueError("scatter_active_first_dim row/active shape mismatch")
    if src.shape[1:] != dst.shape[1:]:
        raise ValueError("scatter_active_first_dim trailing shape mismatch")
    dst2 = dst.view(dst.shape[0], -1)
    src2 = src.view(src.shape[0], -1)
    if dst.device.type != "cuda" or not HAVE_TRITON:
        expanded = active.repeat_interleave(active_repeat)
        dst[rows[expanded]] = src[expanded]
        return
    block = 256
    _scatter_active_rows_kernel[(rows.numel(), triton.cdiv(dst2.shape[1], block))](
        dst2,
        rows,
        src2,
        active,
        dst2.stride(0),
        src2.stride(0),
        dst2.shape[1],
        ACTIVE_REPEAT=active_repeat,
        BLOCK=block,
        num_warps=4,
    )


def scatter_time_active_rows(
    dst: torch.Tensor,
    rows: torch.Tensor,
    src: torch.Tensor,
    time: torch.Tensor,
    active: torch.Tensor,
) -> None:
    """Write ``src[b, time[b]]`` to active lane ``rows[b]`` with one fixed-grid kernel.

    Index vectors are made contiguous first (see `gather_rows`)."""
    rows, time, active = rows.contiguous(), time.contiguous(), active.contiguous()
    if dst.dim() != 2 or src.dim() != 3 or src.shape[0] != rows.numel():
        raise ValueError(f"scatter_time_active_rows shape mismatch: {dst.shape}, {rows.shape}, {src.shape}")
    if src.shape[2] != dst.shape[1] or time.shape != rows.shape or active.shape != rows.shape:
        raise ValueError("scatter_time_active_rows requires aligned rows/time/active and columns")
    if dst.stride(1) != 1 or src.stride(2) != 1:
        raise ValueError("scatter_time_active_rows requires contiguous columns")
    if dst.device.type != "cuda" or not HAVE_TRITON:
        selected = src[torch.arange(src.shape[0], device=src.device), time]
        dst[rows] = torch.where(active[:, None], selected, dst[rows])
        return
    block = 256
    _scatter_time_active_rows_kernel[(rows.numel(), triton.cdiv(dst.shape[1], block))](
        dst,
        rows,
        src,
        time,
        active,
        dst.stride(0),
        src.stride(0),
        src.stride(1),
        dst.shape[1],
        BLOCK=block,
        num_warps=4,
    )


def fused_moe(
    h: torch.Tensor,
    experts: dict,
    routing: tuple[torch.Tensor, torch.Tensor],
    top_k: int,
    expert_range: tuple[int, int],
    inter_scratch: torch.Tensor,
    out_scratch: torch.Tensor,
    active: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run resident dense experts into caller-owned, address-stable scratch tensors.

    ``routing`` is ``(global expert ids, normalized weights)``. ``active`` is an optional
    per-token bool mask for graph bucket padding. Inactive rows produce zero routed output.
    """
    if not available(h, experts):
        raise ValueError("mtp_dense_moe requires contiguous BF16 CUDA inputs and dense experts")
    expert, weight = routing
    tokens, hidden = h.shape
    assignments = tokens * top_k
    intermediate = experts["gate_up"].shape[1] // 2
    if expert.shape != (tokens, top_k) or weight.shape != (tokens, top_k):
        raise ValueError(f"routing must be [{tokens}, {top_k}], got {expert.shape}, {weight.shape}")
    if experts["gate_up"].shape[2] != hidden:
        raise ValueError("gate_up reduction dimension does not match h")
    if experts["down"].shape[1:] != (hidden, intermediate):
        raise ValueError("down expert shape does not match hidden/intermediate")
    if inter_scratch.shape[0] < assignments or inter_scratch.shape[1] != intermediate:
        raise ValueError("MTP expert intermediate scratch is too small")
    if out_scratch.shape[0] < tokens or out_scratch.shape[1] != hidden:
        raise ValueError("MTP expert output scratch is too small")
    if active is not None and (active.shape != (tokens,) or active.dtype != torch.bool):
        raise ValueError(f"active must be bool [{tokens}]")

    inter = inter_scratch[:assignments]
    out = out_scratch[:tokens]
    lo, hi = expert_range
    active_ptr = active if active is not None else weight
    _dense_gate_up_kernel[(assignments, triton.cdiv(intermediate, BLOCK_N))](
        h,
        experts["gate_up"],
        expert,
        weight,
        active_ptr,
        inter,
        lo,
        hi,
        K=hidden,
        INTER=intermediate,
        TOP_K=top_k,
        HAS_ACTIVE=active is not None,
        BN=BLOCK_N,
        BK=BLOCK_K,
        BM=DOT_M,
        num_warps=4,
    )
    _dense_down_combine_kernel[(tokens, triton.cdiv(hidden, BLOCK_N))](
        inter,
        experts["down"],
        expert,
        weight,
        active_ptr,
        out,
        lo,
        hi,
        K=intermediate,
        HIDDEN=hidden,
        TOP_K=top_k,
        HAS_ACTIVE=active is not None,
        BN=BLOCK_N,
        BK=BLOCK_K,
        BM=DOT_M,
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------- expert-grouped variant
#
# `fused_moe` above reads one expert's weights once per (token, expert) assignment: at B96
# that is ~240 local assignments, 5.6 GiB per rank per draft step. Only ~108 of a rank's 128
# experts are distinct among them (960 assignments over 512 experts), so reading each touched
# expert once and applying it to all of its tokens cuts the stream to ~2.6 GiB.
# `grouped_moe` does that with four fixed-grid launches (capture safe, no host read):
#
#   prep      one program per local expert: its live assignments, compacted in arrival order
#   gate_up   (local expert, N tile): the expert's tokens in BM-row chunks, then SiLU * up
#   down      (local expert, N tile): fp32 per-assignment down projection
#   combine   (token, N tile): sum over the token's top-k in routing order, times the weight
#
# Arithmetic per assignment is `fused_moe`'s: the same `tl.dot` tile shapes (BN x BK times
# BK x BM, fp32 accumulate) over the same K order, bf16 rounding of the SiLU product, fp32
# down output scaled by the routing weight and summed over j = 0..top_k-1 in order, one bf16
# rounding at the store. The one difference is that `fused_moe` broadcasts a single token over
# the BM MFMA columns and averages the identical columns back, while here each column is a
# different token; that average can differ from the column value in the last fp32 bit.

if HAVE_TRITON:

    @triton.jit
    def _grouped_prep_kernel(
        expert_ptr,
        weight_ptr,
        active_ptr,
        list_ptr,
        count_ptr,
        expert_lo,
        ASSIGN,
        ASSIGN_PAD: tl.constexpr,
        TOP_K: tl.constexpr,
        HAS_ACTIVE: tl.constexpr,
        MAXC: tl.constexpr,
    ):
        local = tl.program_id(0)
        a = tl.arange(0, ASSIGN_PAD)
        a_ok = a < ASSIGN
        expert = tl.load(expert_ptr + a, mask=a_ok, other=-1)
        combine = tl.load(weight_ptr + a, mask=a_ok, other=0.0)
        mine = a_ok & (expert == expert_lo + local) & (combine != 0.0)
        if HAS_ACTIVE:
            mine = mine & (tl.load(active_ptr + a // TOP_K, mask=a_ok, other=0) != 0)
        flag = mine.to(tl.int32)
        slot = tl.cumsum(flag, 0) - 1
        tl.store(list_ptr + local * MAXC + slot, a, mask=mine)
        tl.store(count_ptr + local, tl.sum(flag, 0))

    @triton.jit
    def _grouped_gate_up_kernel(
        x_ptr,
        gate_up_ptr,
        list_ptr,
        count_ptr,
        inter_ptr,
        K: tl.constexpr,
        INTER: tl.constexpr,
        TOP_K: tl.constexpr,
        MAXC: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        BM: tl.constexpr,
    ):
        local = tl.program_id(0)
        count = tl.load(count_ptr + local)
        if count > 0:
            n = tl.program_id(1) * BN + tl.arange(0, BN)
            n_ok = n < INTER
            base = local.to(tl.int64) * (2 * INTER) * K
            dt = inter_ptr.dtype.element_ty
            for c0 in range(0, count, BM):
                r = c0 + tl.arange(0, BM)
                r_ok = r < count
                asg = tl.load(list_ptr + local * MAXC + r, mask=r_ok, other=0).to(tl.int64)
                token = asg // TOP_K
                acc_gate = tl.zeros((BN, BM), dtype=tl.float32)
                acc_up = tl.zeros((BN, BM), dtype=tl.float32)
                for k0 in range(0, K, BK):
                    k = k0 + tl.arange(0, BK)
                    x = tl.load(x_ptr + token[:, None] * K + k[None, :], mask=r_ok[:, None], other=0.0)
                    gate = tl.load(
                        gate_up_ptr + base + n[:, None] * K + k[None, :],
                        mask=n_ok[:, None],
                        other=0.0,
                    )
                    up = tl.load(
                        gate_up_ptr + base + (n[:, None] + INTER) * K + k[None, :],
                        mask=n_ok[:, None],
                        other=0.0,
                    )
                    acc_gate = tl.dot(gate, tl.trans(x), acc_gate)
                    acc_up = tl.dot(up, tl.trans(x), acc_up)
                gatef = acc_gate.to(dt).to(tl.float32)
                silu = (gatef / (1.0 + tl.exp(-gatef))).to(dt)
                act = (silu.to(tl.float32) * acc_up.to(dt).to(tl.float32)).to(dt)
                tl.store(
                    inter_ptr + asg[None, :] * INTER + n[:, None],
                    act,
                    mask=n_ok[:, None] & r_ok[None, :],
                )

    @triton.jit
    def _grouped_down_kernel(
        inter_ptr,
        down_ptr,
        list_ptr,
        count_ptr,
        y_ptr,
        K: tl.constexpr,
        HIDDEN: tl.constexpr,
        MAXC: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        BM: tl.constexpr,
    ):
        local = tl.program_id(0)
        count = tl.load(count_ptr + local)
        if count > 0:
            n = tl.program_id(1) * BN + tl.arange(0, BN)
            n_ok = n < HIDDEN
            base = local.to(tl.int64) * HIDDEN * K
            for c0 in range(0, count, BM):
                r = c0 + tl.arange(0, BM)
                r_ok = r < count
                asg = tl.load(list_ptr + local * MAXC + r, mask=r_ok, other=0).to(tl.int64)
                acc = tl.zeros((BN, BM), dtype=tl.float32)
                for k0 in range(0, K, BK):
                    k = k0 + tl.arange(0, BK)
                    act = tl.load(inter_ptr + asg[:, None] * K + k[None, :], mask=r_ok[:, None], other=0.0)
                    w = tl.load(
                        down_ptr + base + n[:, None] * K + k[None, :],
                        mask=n_ok[:, None],
                        other=0.0,
                    )
                    acc = tl.dot(w, tl.trans(act), acc)
                tl.store(
                    y_ptr + asg[None, :] * HIDDEN + n[:, None],
                    acc,
                    mask=n_ok[:, None] & r_ok[None, :],
                )

    @triton.jit
    def _grouped_combine_kernel(
        y_ptr,
        expert_ptr,
        weight_ptr,
        active_ptr,
        out_ptr,
        expert_lo,
        expert_hi,
        HIDDEN: tl.constexpr,
        TOP_K: tl.constexpr,
        HAS_ACTIVE: tl.constexpr,
        BN: tl.constexpr,
    ):
        token = tl.program_id(0)
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        n_ok = n < HIDDEN
        token_live = True
        if HAS_ACTIVE:
            token_live = tl.load(active_ptr + token)
        total = tl.zeros((BN,), dtype=tl.float32)
        if token_live:
            for j in range(TOP_K):
                assignment = token * TOP_K + j
                expert = tl.load(expert_ptr + assignment)
                combine = tl.load(weight_ptr + assignment).to(tl.float32)
                if (expert >= expert_lo) & (expert < expert_hi) & (combine != 0.0):
                    y = tl.load(y_ptr + assignment.to(tl.int64) * HIDDEN + n, mask=n_ok, other=0.0)
                    total += y * combine
        tl.store(out_ptr + token * HIDDEN + n, total, mask=n_ok)


GROUPED = {
    "gu_bn": 64,
    "gu_bk": 128,
    "gu_warps": 4,
    "gu_stages": 1,
    "dn_bn": 64,
    "dn_bk": 128,
    "dn_warps": 4,
    "dn_stages": 1,
    "combine_bn": 256,
}
"""Launch shapes for `grouped_moe`, from a 98-point sweep at B96 on one MI300A (uniform
routing, 247 local assignments over 109 distinct experts): 1.05 ms per call versus 1.50 ms
at the per-assignment kernel's shapes (BN 32, BK 64, 2 stages) and 3.0 ms for `fused_moe`.
That is about 2.5 TB/s on the 2.6 GB of BF16 expert weights read, near this node's measured
2.85 TB/s copy bandwidth. Outputs were bit-identical to `fused_moe` at every shape.
BM stays `DOT_M` (the smallest MFMA M; most experts see one or two tokens a step)."""


def grouped_scratch(
    max_tokens: int, top_k: int, local_experts: int, hidden: int, device: torch.device
) -> dict:
    """Address-stable scratch for :func:`grouped_moe` beyond `fused_moe`'s two buffers."""
    return {
        # One expert can hold every assignment when a token repeats an expert (`fused_moe`'s
        # contract allows duplicates), so the per-expert list is `max_tokens * top_k` wide.
        "list": torch.zeros(local_experts, max_tokens * top_k, dtype=torch.int32, device=device),
        "count": torch.zeros(local_experts, dtype=torch.int32, device=device),
        "y": torch.empty(max_tokens * top_k, hidden, dtype=torch.float32, device=device),
    }


def grouped_moe(
    h: torch.Tensor,
    experts: dict,
    routing: tuple[torch.Tensor, torch.Tensor],
    top_k: int,
    expert_range: tuple[int, int],
    inter_scratch: torch.Tensor,
    out_scratch: torch.Tensor,
    scratch: dict,
    active: torch.Tensor | None = None,
) -> torch.Tensor:
    """`fused_moe`'s contract, reading each touched local expert once (see the section note).

    ``scratch`` comes from :func:`grouped_scratch` with ``max_tokens >= h.shape[0]``.
    """
    if not available(h, experts):
        raise ValueError("mtp_dense_moe requires contiguous BF16 CUDA inputs and dense experts")
    expert, weight = routing
    tokens, hidden = h.shape
    assignments = tokens * top_k
    local_experts = experts["gate_up"].shape[0]
    intermediate = experts["gate_up"].shape[1] // 2
    if expert.shape != (tokens, top_k) or weight.shape != (tokens, top_k):
        raise ValueError(f"routing must be [{tokens}, {top_k}], got {expert.shape}, {weight.shape}")
    if not (expert.is_contiguous() and weight.is_contiguous()):
        raise ValueError("routing tensors must be contiguous")
    if inter_scratch.shape[0] < assignments or inter_scratch.shape[1] != intermediate:
        raise ValueError("MTP expert intermediate scratch is too small")
    if out_scratch.shape[0] < tokens or out_scratch.shape[1] != hidden:
        raise ValueError("MTP expert output scratch is too small")
    maxc = scratch["list"].shape[1]
    if scratch["list"].shape[0] != local_experts or maxc < assignments:
        raise ValueError("grouped scratch list is too small")
    if scratch["y"].shape[0] < assignments or scratch["y"].shape[1] != hidden:
        raise ValueError("grouped scratch y is too small")
    if active is not None and (active.shape != (tokens,) or active.dtype != torch.bool):
        raise ValueError(f"active must be bool [{tokens}]")

    inter = inter_scratch[:assignments]
    out = out_scratch[:tokens]
    y = scratch["y"][:assignments]
    lo, hi = expert_range
    active_ptr = active if active is not None else weight
    _grouped_prep_kernel[(local_experts,)](
        expert,
        weight,
        active_ptr,
        scratch["list"],
        scratch["count"],
        lo,
        ASSIGN=assignments,
        ASSIGN_PAD=triton.next_power_of_2(assignments),
        TOP_K=top_k,
        HAS_ACTIVE=active is not None,
        MAXC=maxc,
        num_warps=4,
    )
    cfg = GROUPED
    _grouped_gate_up_kernel[(local_experts, triton.cdiv(intermediate, cfg["gu_bn"]))](
        h,
        experts["gate_up"],
        scratch["list"],
        scratch["count"],
        inter,
        K=hidden,
        INTER=intermediate,
        TOP_K=top_k,
        MAXC=maxc,
        BN=cfg["gu_bn"],
        BK=cfg["gu_bk"],
        BM=DOT_M,
        num_warps=cfg["gu_warps"],
        num_stages=cfg["gu_stages"],
    )
    _grouped_down_kernel[(local_experts, triton.cdiv(hidden, cfg["dn_bn"]))](
        inter,
        experts["down"],
        scratch["list"],
        scratch["count"],
        y,
        K=intermediate,
        HIDDEN=hidden,
        MAXC=maxc,
        BN=cfg["dn_bn"],
        BK=cfg["dn_bk"],
        BM=DOT_M,
        num_warps=cfg["dn_warps"],
        num_stages=cfg["dn_stages"],
    )
    _grouped_combine_kernel[(tokens, triton.cdiv(hidden, cfg["combine_bn"]))](
        y,
        expert,
        weight,
        active_ptr,
        out,
        lo,
        hi,
        HIDDEN=hidden,
        TOP_K=top_k,
        HAS_ACTIVE=active is not None,
        BN=cfg["combine_bn"],
        num_warps=4,
    )
    return out

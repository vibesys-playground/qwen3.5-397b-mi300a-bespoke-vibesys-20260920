"""Fused Triton kernel for `model.rmsnorm`, the plain (non-gated) RMSNorm.

`model.rmsnorm` backs five call sites -- `in_norm`, `post_norm`, `q_norm`, `k_norm`, and the
final pre-LM-head norm -- 151 calls in one decode step at the deployed batch (60 + 60 + 1 + 15
+ 15). Each call is six unfused torch ops (`x.float()`, `.pow(2)`, `.mean(-1)`, `+eps` folded
into `rsqrt`, two multiplies, one final `.type_as` cast), which on a step whose kernels are
already this cheap is a dispatch cost, the same story `deltanet_fused.py` tells for the DeltaNet
mixer: an isolated microbenchmark at the five real shapes (batch 48) measured the unfused chain
at 45.8-76.9us of device time a call and this kernel at 1.86-2.64us, a 28.7-32.8x per-call cut,
about 10.7ms of device time over a 151-call step. See rmsnorm_fusion_poc.py and
rmsnorm_fusion_poc_result.json for that measurement; this module is what it validated before
being wired in.

Not a reformulation with a different rounding order, unlike `deltanet_fused`'s kernels: nothing
here changes reduction order (Triton's `tl.sum` still is not bit-identical to `torch.sum`'s,
the same caveat `deltanet_fused` carries), but the arithmetic is otherwise the same fp32 chain
`model.rmsnorm` runs, at the same single rounding point (the final store to the output dtype;
`model.rmsnorm` has one rounding, unlike `gated_rmsnorm`'s three, so there is no intermediate
cast to round out of order).

One kernel, one wrapper (`rmsnorm`), used everywhere `model.rmsnorm` is: the eager single-
sequence path (`Model.layer`, `Model.full_attention`), the eager batched-decode path
(`Model.decode_layer`, `Model.attn_decode`), and the captured/static path
(`graph_decode.decode_layer_static`, `attn_decode_static`, `segment_step`) all call
`model.rmsnorm`, which dispatches here when available -- see its docstring. This is the same
single-dispatch-point shape `deltanet_fused.available` uses, and deliberately so: the graph
port's own history has fusion wins land in the eager path and get silently missed by the static
one (`deltanet_decode_static`, `attn_decode_static`'s TP-sharding fix), and the fix both times
was to make the shared function itself capture-aware rather than special-case each call site.

`model.rmsnorm` takes `x` of any shape ending in the normalized axis (`[..., cols]`), not just
`[rows, heads, cols]` like `gated_rmsnorm`'s callers hand it: `in_norm`/`post_norm`/`final_norm`
see `[B, 1, hidden]`, `q_norm`/`k_norm` see `[B, 1, heads, head_dim]` (`heads` folded into
`rows`, since a plain RMSNorm's reduction is per-row regardless of how many head axes lead up
to it -- there is no per-head gate tensor here to keep index-aligned with, unlike
`gated_rmsnorm`, which is the only reason that kernel keeps a separate `heads` axis at all).
`rmsnorm` here flattens every leading axis into one row axis via `.flatten(0, -2)`, which is a
view whenever the leading axes are regularly strided -- true for every real call site, including
`q_norm`/`k_norm`'s operands, which arrive as `.chunk(2, dim=-1)` slices of a wider projection
(non-contiguous in the parent's row width, but still regularly strided over the merged axis) --
and only copies if the caller ever hands it something irregular.
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
    """True when the fused kernel can run for tensors on `device`.

    Same contract as `deltanet_fused.available`: `SEED_FUSED_RMSNORM=0` forces the torch path
    for an ablation or a bisect, and the function below does not consult it, so tests can drive
    it directly on CPU under `TRITON_INTERPRET=1`.
    """
    if not HAVE_TRITON or os.environ.get("SEED_FUSED_RMSNORM", "1") == "0":
        return False
    return device.type == "cuda"


if HAVE_TRITON:

    @triton.jit
    def _rmsnorm_kernel(
        x_ptr,
        w_ptr,
        o_ptr,
        sx_row,
        so_row,
        cols,
        eps,
        BLOCK: tl.constexpr,
    ):
        """One program per row. Non-gated sibling of `deltanet_fused._gated_rmsnorm_kernel`.

        Mirrors `model.rmsnorm` exactly: normalize in fp32, scale by `(1 + w)` (the "+1" is
        `model.rmsnorm`'s own weight convention, not `gated_rmsnorm`'s plain `w`), round to the
        storage dtype once at the store -- there is no intermediate cast to `dt` before the
        scale the way `gated_rmsnorm` has, because `model.rmsnorm` does not have one either.
        """
        row = tl.program_id(0)
        c = tl.arange(0, BLOCK)
        live = c < cols
        dt = o_ptr.dtype.element_ty

        x = tl.load(x_ptr + row * sx_row + c, mask=live, other=0.0).to(tl.float32)
        y = x * tl.rsqrt(tl.sum(x * x) / cols + eps)
        w = tl.load(w_ptr + c, mask=live, other=0.0).to(tl.float32)
        out = (y * (1.0 + w)).to(dt)
        tl.store(o_ptr + row * so_row + c, out, mask=live)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """`model.rmsnorm` in one kernel, over an `x` of any shape ending in the normalized axis.

    `x`'s last axis must be contiguous (true of every real call site: it is either the model's
    native layout or a `.chunk(dim=-1)` slice, both of which keep the last axis packed). The
    leading axes fold into one row axis via `flatten(0, -2)`, a view whenever they are regularly
    strided, which every real call site is.
    """
    if x.stride(-1) != 1:
        raise ValueError(f"rmsnorm_fused: x's last axis must be contiguous, got stride {x.stride()}")
    cols = x.shape[-1]
    flat = x.flatten(0, -2) if x.dim() > 1 else x.unsqueeze(0)
    rows = flat.shape[0]
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    out_flat = out.flatten(0, -2) if x.dim() > 1 else out.unsqueeze(0)
    _rmsnorm_kernel[(rows,)](
        flat,
        w,
        out_flat,
        flat.stride(0),
        out_flat.stride(0),
        cols,
        eps,
        BLOCK=triton.next_power_of_2(cols),
    )
    return out


if HAVE_TRITON:

    @triton.jit
    def _add_rmsnorm_kernel(
        x_ptr,
        r_ptr,
        w_ptr,
        xo_ptr,
        o_ptr,
        sx_row,
        sr_row,
        sxo_row,
        so_row,
        cols,
        eps,
        BLOCK: tl.constexpr,
    ):
        """`xo = x + r` (rounded to the storage dtype, as torch's bf16 `add` rounds), then
        `_rmsnorm_kernel`'s exact body on the rounded `xo`: same load shape, same fp32 chain,
        same `tl.sum`, so `o` is bit-identical to `rmsnorm(x + r)` run as two kernels."""
        row = tl.program_id(0)
        c = tl.arange(0, BLOCK)
        live = c < cols
        dt = o_ptr.dtype.element_ty

        a = tl.load(x_ptr + row * sx_row + c, mask=live, other=0.0).to(tl.float32)
        b = tl.load(r_ptr + row * sr_row + c, mask=live, other=0.0).to(tl.float32)
        s = (a + b).to(dt)
        tl.store(xo_ptr + row * sxo_row + c, s, mask=live)
        x = s.to(tl.float32)
        y = x * tl.rsqrt(tl.sum(x * x) / cols + eps)
        w = tl.load(w_ptr + c, mask=live, other=0.0).to(tl.float32)
        out = (y * (1.0 + w)).to(dt)
        tl.store(o_ptr + row * so_row + c, out, mask=live)


def add_rmsnorm(
    x: torch.Tensor, r: torch.Tensor, w: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(x + r, rmsnorm(x + r, w, eps))` in one kernel (`SEED_ADD_RMSNORM`).

    The decode step's residual update and the RMSNorm that reads it, twice a layer: the
    mixer's all-reduced partial into `post_norm`, and the MoE's into the next layer's
    `in_norm` (or the final norm). Replaces an elementwise `add` plus `rmsnorm` launch with
    one; the rounding points are the unfused sequence's (see `_add_rmsnorm_kernel`). `x` and
    `r` share a shape whose last axis is contiguous and whose leading axes flatten to rows.
    """
    if x.shape != r.shape or x.stride(-1) != 1 or r.stride(-1) != 1:
        raise ValueError(f"add_rmsnorm: shapes {x.shape}/{r.shape} strides {x.stride()}/{r.stride()}")
    cols = x.shape[-1]
    xf, rf = x.reshape(-1, cols), r.reshape(-1, cols)
    xo = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    xof, of = xo.view(-1, cols), out.view(-1, cols)
    _add_rmsnorm_kernel[(xf.shape[0],)](
        xf,
        rf,
        w,
        xof,
        of,
        xf.stride(0),
        rf.stride(0),
        xof.stride(0),
        of.stride(0),
        cols,
        eps,
        BLOCK=triton.next_power_of_2(cols),
    )
    return xo, out


if HAVE_TRITON:

    @triton.jit
    def _ar_add_rmsnorm_kernel(
        m_ptr,
        r_ptr,
        w_ptr,
        xo_ptr,
        o_ptr,
        b0,
        b1,
        b2,
        b3,
        f0,
        f1,
        f2,
        f3,
        ctr_ptr,
        sm_row,
        sr_row,
        sxo_row,
        so_row,
        cols,
        eps,
        max_elems,
        SELF: tl.constexpr,
        MAX_FLAG_BLOCKS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Graph-safe one-shot all-reduce of `m` over 4 ranks, then `_add_rmsnorm_kernel`.

        Same protocol, buffers, flags and device call counter as `allreduce_custom`'s
        `oneshot_ar_kernel` (read that comment first), so the two kernels interleave on one
        counter: program `row` reads `ctr[0] = e`, publishes its row of `m` into slot `e & 1`
        of this rank's IPC buffer (`b<SELF>`), sets its flag `f<SELF>[slot][row] = e + 1`
        (system-scope release), waits for each peer's same flag, sums the four rows in rank
        order in fp32 and rounds to bf16 (`oneshot_ar_kernel`'s exact arithmetic), then runs
        `_add_rmsnorm_kernel`'s body on `(r, reduced)`. The last program to retire advances
        the counter. Bit-exact against `oneshot` + `add_rmsnorm` (same `tl.sum` layout; see
        `CustomAllReduce.ar_add_rmsnorm` for what was checked). Fixed at 4 ranks. No spin
        watchdog, unlike the HIP kernels: a peer that never arrives hangs this rank.
        """
        row = tl.program_id(0)
        c = tl.arange(0, BLOCK)
        live = c < cols
        dt = o_ptr.dtype.element_ty

        e = tl.load(ctr_ptr, volatile=True)
        slot = e & 1
        value = e + 1
        off = slot * max_elems + row * cols
        foff = slot * MAX_FLAG_BLOCKS + row
        m = tl.load(m_ptr + row * sm_row + c, mask=live, other=0.0)
        if SELF == 0:
            tl.store(b0 + off + c, m, mask=live)
        elif SELF == 1:
            tl.store(b1 + off + c, m, mask=live)
        elif SELF == 2:
            tl.store(b2 + off + c, m, mask=live)
        else:
            tl.store(b3 + off + c, m, mask=live)
        # Every wave's stores done (per-wave vmcnt wait), then the block barrier, then one
        # system-scope release on the flag: the whole row is visible to peers before the flag.
        tl.inline_asm_elementwise(
            "s_waitcnt vmcnt(0)", "=v,v", [c], dtype=tl.int32, is_pure=False, pack=1
        )
        tl.debug_barrier()
        if SELF == 0:
            tl.atomic_xchg(f0 + foff, value, sem="release", scope="sys")
        elif SELF == 1:
            tl.atomic_xchg(f1 + foff, value, sem="release", scope="sys")
        elif SELF == 2:
            tl.atomic_xchg(f2 + foff, value, sem="release", scope="sys")
        else:
            tl.atomic_xchg(f3 + foff, value, sem="release", scope="sys")
        if SELF != 0:
            while tl.atomic_add(f0 + foff, 0, sem="acquire", scope="sys") != value:
                pass
        if SELF != 1:
            while tl.atomic_add(f1 + foff, 0, sem="acquire", scope="sys") != value:
                pass
        if SELF != 2:
            while tl.atomic_add(f2 + foff, 0, sem="acquire", scope="sys") != value:
                pass
        if SELF != 3:
            while tl.atomic_add(f3 + foff, 0, sem="acquire", scope="sys") != value:
                pass
        tl.debug_barrier()
        acc = tl.load(b0 + off + c, mask=live, other=0.0, volatile=True).to(tl.float32)
        acc += tl.load(b1 + off + c, mask=live, other=0.0, volatile=True).to(tl.float32)
        acc += tl.load(b2 + off + c, mask=live, other=0.0, volatile=True).to(tl.float32)
        acc += tl.load(b3 + off + c, mask=live, other=0.0, volatile=True).to(tl.float32)
        red = acc.to(dt)

        a = tl.load(r_ptr + row * sr_row + c, mask=live, other=0.0).to(tl.float32)
        b = red.to(tl.float32)
        s = (a + b).to(dt)
        tl.store(xo_ptr + row * sxo_row + c, s, mask=live)
        x = s.to(tl.float32)
        y = x * tl.rsqrt(tl.sum(x * x) / cols + eps)
        w = tl.load(w_ptr + c, mask=live, other=0.0).to(tl.float32)
        out = (y * (1.0 + w)).to(dt)
        tl.store(o_ptr + row * so_row + c, out, mask=live)

        done = tl.atomic_add(ctr_ptr + 1, 1, sem="acq_rel", scope="gpu")
        if done == tl.num_programs(0) - 1:
            tl.store(ctr_ptr + 1, 0)
            tl.store(ctr_ptr, e + 1)


if HAVE_TRITON:

    @triton.jit
    def _push_ar_add_rmsnorm_kernel(
        m_ptr,
        r_ptr,
        w_ptr,
        xo_ptr,
        o_ptr,
        p0,
        p1,
        p2,
        p3,
        g0,
        g1,
        g2,
        g3,
        ctr_ptr,
        sm_row,
        sr_row,
        sxo_row,
        so_row,
        cols,
        eps,
        push_elems,
        SELF: tl.constexpr,
        MAXR: tl.constexpr,
        BLOCK: tl.constexpr,
        SYS_FENCE: tl.constexpr = True,
    ):
        """`_ar_add_rmsnorm_kernel` with pushes instead of remote reads (`SEED_AR_PUSH`).

        Program `row` writes its row of `m` into every peer's receive buffer
        (`p<t>[slot][SELF][row]`, remote stores, posted), sets `g<t>[slot][SELF][row] = e + 1`
        (system-scope release), then spins on its own local flags `g<SELF>[slot][q][row]` and
        reads the three peers' rows from local memory. The reduction (four rows in rank order
        in fp32, rounded to bf16) and the add + norm tail are `_ar_add_rmsnorm_kernel`'s, so
        the outputs are bit-identical to it. Same device call counter, so it interleaves with
        the other one-shot kernels. Slot reuse is the SP kernel's argument: rank r's call k+2
        push to rank t follows rank r seeing rank t's call k+1 push, which rank t issues only
        after its call k finished reading."""
        row = tl.program_id(0)
        c = tl.arange(0, BLOCK)
        live = c < cols
        dt = o_ptr.dtype.element_ty

        e = tl.load(ctr_ptr, volatile=True)
        slot = e & 1
        value = e + 1
        off = (slot * 4 + SELF) * push_elems + row * cols
        m = tl.load(m_ptr + row * sm_row + c, mask=live, other=0.0)
        if SELF != 0:
            tl.store(p0 + off + c, m, mask=live)
        if SELF != 1:
            tl.store(p1 + off + c, m, mask=live)
        if SELF != 2:
            tl.store(p2 + off + c, m, mask=live)
        if SELF != 3:
            tl.store(p3 + off + c, m, mask=live)
        tl.inline_asm_elementwise(
            "s_waitcnt vmcnt(0)", "=v,v", [c], dtype=tl.int32, is_pure=False, pack=1
        )
        tl.debug_barrier()
        foff = (slot * 4 + SELF) * MAXR + row
        if SYS_FENCE:
            if SELF != 0:
                tl.atomic_xchg(g0 + foff, value, sem="release", scope="sys")
            if SELF != 1:
                tl.atomic_xchg(g1 + foff, value, sem="release", scope="sys")
            if SELF != 2:
                tl.atomic_xchg(g2 + foff, value, sem="release", scope="sys")
            if SELF != 3:
                tl.atomic_xchg(g3 + foff, value, sem="release", scope="sys")
        else:
            # Diagnostic: no L2 writeback before the flag; relies on the fine-grained pages'
            # uncached remote stores being complete at the vmcnt wait above.
            if SELF != 0:
                tl.atomic_xchg(g0 + foff, value, sem="relaxed", scope="sys")
            if SELF != 1:
                tl.atomic_xchg(g1 + foff, value, sem="relaxed", scope="sys")
            if SELF != 2:
                tl.atomic_xchg(g2 + foff, value, sem="relaxed", scope="sys")
            if SELF != 3:
                tl.atomic_xchg(g3 + foff, value, sem="relaxed", scope="sys")
        if SELF == 0:
            p_me = p0
            g_me = g0
        elif SELF == 1:
            p_me = p1
            g_me = g1
        elif SELF == 2:
            p_me = p2
            g_me = g2
        else:
            p_me = p3
            g_me = g3
        for q in tl.static_range(4):
            if q != SELF:
                if SYS_FENCE:
                    while tl.atomic_add(g_me + (slot * 4 + q) * MAXR + row, 0, sem="acquire", scope="sys") != value:
                        pass
                else:
                    while tl.atomic_add(g_me + (slot * 4 + q) * MAXR + row, 0, sem="relaxed", scope="sys") != value:
                        pass
        tl.debug_barrier()
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for q in tl.static_range(4):
            if q == SELF:
                v = m.to(tl.float32)
            else:
                v = tl.load(
                    p_me + (slot * 4 + q) * push_elems + row * cols + c,
                    mask=live,
                    other=0.0,
                    volatile=True,
                ).to(tl.float32)
            if q == 0:
                acc = v
            else:
                acc += v
        red = acc.to(dt)

        a = tl.load(r_ptr + row * sr_row + c, mask=live, other=0.0).to(tl.float32)
        b = red.to(tl.float32)
        s = (a + b).to(dt)
        tl.store(xo_ptr + row * sxo_row + c, s, mask=live)
        x = s.to(tl.float32)
        y = x * tl.rsqrt(tl.sum(x * x) / cols + eps)
        w = tl.load(w_ptr + c, mask=live, other=0.0).to(tl.float32)
        out = (y * (1.0 + w)).to(dt)
        tl.store(o_ptr + row * so_row + c, out, mask=live)

        done = tl.atomic_add(ctr_ptr + 1, 1, sem="acq_rel", scope="gpu")
        if done == tl.num_programs(0) - 1:
            tl.store(ctr_ptr + 1, 0)
            tl.store(ctr_ptr, e + 1)


def push_ar_add_rmsnorm(
    m: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    peers: tuple,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`ar_add_rmsnorm`'s result through `_push_ar_add_rmsnorm_kernel` (`SEED_AR_PUSH`).
    `peers` is `(recv, flags, ctr, self_rank, push_elems, max_rows)` from
    `allreduce_custom.CustomAllReduce.ar_add_rmsnorm`, which checks the preconditions."""
    recv, flags, ctr, rank, push_elems, max_rows = peers[:6]
    sys_fence = peers[6] if len(peers) > 6 else True
    cols = m.shape[-1]
    mf, rf = m.reshape(-1, cols), r.reshape(-1, cols)
    xo = torch.empty(r.shape, dtype=r.dtype, device=r.device)
    out = torch.empty(r.shape, dtype=r.dtype, device=r.device)
    xof, of = xo.view(-1, cols), out.view(-1, cols)
    _push_ar_add_rmsnorm_kernel[(mf.shape[0],)](
        mf,
        rf,
        w,
        xof,
        of,
        *recv,
        *flags,
        ctr,
        mf.stride(0),
        rf.stride(0),
        xof.stride(0),
        of.stride(0),
        cols,
        eps,
        push_elems,
        SELF=rank,
        MAXR=max_rows,
        BLOCK=triton.next_power_of_2(cols),
        SYS_FENCE=sys_fence,
    )
    return xo, out


class RawPtr:
    """A device address Triton can take as a pointer argument: it only reads `data_ptr()` and
    `dtype`. Wraps the IPC-mapped peer buffers/flags, which are not torch tensors."""

    def __init__(self, ptr: int, dtype: torch.dtype) -> None:
        self.ptr, self.dtype = ptr, dtype

    def data_ptr(self) -> int:
        return self.ptr


def ar_add_rmsnorm(
    m: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    peers: tuple[list[RawPtr], list[RawPtr], torch.Tensor, int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(r + all_reduce(m), rmsnorm(r + all_reduce(m), w, eps))` in one kernel.

    `peers` is `(bufs, flags, ctr, self_rank, max_elems, max_flag_blocks)` from
    `allreduce_custom.CustomAllReduce.ar_add_rmsnorm`, which owns the IPC state and checks
    the preconditions (4 ranks, bf16, rows within the flag table)."""
    bufs, flags, ctr, rank, max_elems, max_flag_blocks = peers
    cols = m.shape[-1]
    mf, rf = m.reshape(-1, cols), r.reshape(-1, cols)
    xo = torch.empty(r.shape, dtype=r.dtype, device=r.device)
    out = torch.empty(r.shape, dtype=r.dtype, device=r.device)
    xof, of = xo.view(-1, cols), out.view(-1, cols)
    _ar_add_rmsnorm_kernel[(mf.shape[0],)](
        mf,
        rf,
        w,
        xof,
        of,
        *bufs,
        *flags,
        ctr,
        mf.stride(0),
        rf.stride(0),
        xof.stride(0),
        of.stride(0),
        cols,
        eps,
        max_elems,
        SELF=rank,
        MAX_FLAG_BLOCKS=max_flag_blocks,
        BLOCK=triton.next_power_of_2(cols),
    )
    return xo, out


if HAVE_TRITON:

    @triton.jit
    def _sp_push_row(dst_ptr, src, off, c, live):
        tl.store(dst_ptr + off + c, src, mask=live)

    @triton.jit
    def _sp_flag(ptr, value, t: tl.constexpr, ONE_RELEASE: tl.constexpr, FIRST_PEER: tl.constexpr):
        if ONE_RELEASE and t != FIRST_PEER:
            tl.atomic_xchg(ptr, value, sem="relaxed", scope="sys")
        else:
            tl.atomic_xchg(ptr, value, sem="release", scope="sys")

    @triton.jit
    def _sp_flags(f0, f1, f2, f3, off, value, SELF: tl.constexpr, ONE_RELEASE: tl.constexpr, FIRST_PEER: tl.constexpr):
        """Set this program's flag in every peer's table. Every store is a system-scope release
        unless `ONE_RELEASE`: then only the first peer's is (it makes every prior store of the
        program visible), and the other two, issued after it, are relaxed."""
        if SELF != 0:
            _sp_flag(f0 + off, value, 0, ONE_RELEASE, FIRST_PEER)
        if SELF != 1:
            _sp_flag(f1 + off, value, 1, ONE_RELEASE, FIRST_PEER)
        if SELF != 2:
            _sp_flag(f2 + off, value, 2, ONE_RELEASE, FIRST_PEER)
        if SELF != 3:
            _sp_flag(f3 + off, value, 3, ONE_RELEASE, FIRST_PEER)

    @triton.jit
    def _sp_wait(ptr, value, RELAXED_SPIN: tl.constexpr):
        if RELAXED_SPIN:
            while tl.atomic_add(ptr, 0, sem="relaxed", scope="sys") != value:
                pass
            tl.atomic_add(ptr, 0, sem="acquire", scope="sys")
        else:
            while tl.atomic_add(ptr, 0, sem="acquire", scope="sys") != value:
                pass

    @triton.jit
    def _sp_ar_add_rmsnorm_kernel(
        m_ptr,
        r_ptr,
        w_ptr,
        xo_ptr,
        h_ptr,
        rs0,
        rs1,
        rs2,
        rs3,
        ag0,
        ag1,
        ag2,
        ag3,
        fa0,
        fa1,
        fa2,
        fa3,
        fb0,
        fb1,
        fb2,
        fb3,
        ctr_ptr,
        sm_row,
        sr_row,
        sxo_row,
        sh_row,
        S,
        cols,
        eps,
        slot_elems,
        SELF: tl.constexpr,
        MAXS: tl.constexpr,
        BLOCK: tl.constexpr,
        COPY_AG: tl.constexpr = True,
        RELAXED_SPIN: tl.constexpr = False,
        ONE_RELEASE: tl.constexpr = False,
        FIRST_PEER: tl.constexpr = 0,
        R: tl.constexpr = 1,
    ):
        """Reduce-scatter, residual add, RMSNorm, all-gather: `_ar_add_rmsnorm_kernel`'s result
        with the residual kept sharded by rows (`SEED_AR_SP`).

        Rank `SELF` owns rows `[SELF*S, (SELF+1)*S)` of the residual. Program `prog` handles
        shard rows `p = prog*R .. prog*R + R-1` (`R` = `SEED_AR_SP_ROWS_PER_PROG` on wide calls,
        else 1) with one flag round per phase for all of them; for each row `p`:
        1. pushes row `t*S + p` of its partial `m` into rank `t`'s reduce-scatter buffer
           (`rs<t>[slot][SELF][p]`) for each peer `t`, then sets `fa<t>[slot][SELF][p]`
           (system-scope release) -- remote writes, no remote reads;
        2. waits on its own `fa<SELF>[slot][q][p]` for every peer `q` (local spin);
        3. sums the four partials of its row `SELF*S + p` in rank order in fp32 and rounds to
           bf16 (`oneshot_ar_kernel`'s arithmetic), then `_add_rmsnorm_kernel`'s body against
           the residual shard row `p`: writes `xo[p]` and the normed row;
        4. pushes the normed row into every peer's all-gather buffer and `h` locally, sets
           `fb<t>[slot][SELF][p]`, waits on its own `fb`, and copies the peers' normed rows
           `q*S + p` into `h`.
        So each rank moves 1.5x the payload over xGMI (0.75 in, 0.75 out) instead of the one-
        shot's 3x reads, and the norm runs on `S` rows. Every value is the same computation as
        `_ar_add_rmsnorm_kernel`'s for that row, so the outputs are bit-identical to it.
        Own device counter (`ctr[0]`, `ctr[1]` retired programs) and own buffers, so it
        interleaves freely with the other all-reduce kernels. Slot reuse: rank r's call k+2
        push to rank t happens after rank r's call k+1 saw rank t's call k+1 push, which rank t
        issues only after its call k finished reading. No spin watchdog (same as
        `_ar_add_rmsnorm_kernel`)."""
        prog = tl.program_id(0)
        c = tl.arange(0, BLOCK)
        live = c < cols
        dt = h_ptr.dtype.element_ty

        e = tl.load(ctr_ptr, volatile=True)
        slot = e & 1
        value = e + 1
        base = slot * slot_elems
        # 1. push partial rows to their owners (`R` shard rows per program)
        for j in tl.static_range(R):
            p = prog * R + j
            if SELF != 0:
                _sp_push_row(rs0, tl.load(m_ptr + (0 * S + p) * sm_row + c, mask=live, other=0.0), base + (SELF * S + p) * cols, c, live)
            if SELF != 1:
                _sp_push_row(rs1, tl.load(m_ptr + (1 * S + p) * sm_row + c, mask=live, other=0.0), base + (SELF * S + p) * cols, c, live)
            if SELF != 2:
                _sp_push_row(rs2, tl.load(m_ptr + (2 * S + p) * sm_row + c, mask=live, other=0.0), base + (SELF * S + p) * cols, c, live)
            if SELF != 3:
                _sp_push_row(rs3, tl.load(m_ptr + (3 * S + p) * sm_row + c, mask=live, other=0.0), base + (SELF * S + p) * cols, c, live)
        tl.inline_asm_elementwise(
            "s_waitcnt vmcnt(0)", "=v,v", [c], dtype=tl.int32, is_pure=False, pack=1
        )
        tl.debug_barrier()
        foff_me = slot * 4 * MAXS + SELF * MAXS + prog
        _sp_flags(fa0, fa1, fa2, fa3, foff_me, value, SELF, ONE_RELEASE, FIRST_PEER)
        # 2. wait for the three peers' pushes of this program's rows (local flags)
        if SELF == 0:
            fa_me = fa0
            rs_me = rs0
            fb_me = fb0
            ag_me = ag0
        elif SELF == 1:
            fa_me = fa1
            rs_me = rs1
            fb_me = fb1
            ag_me = ag1
        elif SELF == 2:
            fa_me = fa2
            rs_me = rs2
            fb_me = fb2
            ag_me = ag2
        else:
            fa_me = fa3
            rs_me = rs3
            fb_me = fb3
            ag_me = ag3
        for q in tl.static_range(4):
            if q != SELF:
                _sp_wait(fa_me + slot * 4 * MAXS + q * MAXS + prog, value, RELAXED_SPIN)
        tl.debug_barrier()
        # 3. reduce in rank order, residual add, norm, push the normed row to every peer
        for j in tl.static_range(R):
            p = prog * R + j
            row = SELF * S + p
            acc = tl.zeros([BLOCK], dtype=tl.float32)
            for q in tl.static_range(4):
                if q == SELF:
                    v = tl.load(m_ptr + row * sm_row + c, mask=live, other=0.0).to(tl.float32)
                else:
                    v = tl.load(rs_me + base + (q * S + p) * cols + c, mask=live, other=0.0, volatile=True).to(tl.float32)
                if q == 0:
                    acc = v
                else:
                    acc += v
            red = acc.to(dt)
            a = tl.load(r_ptr + p * sr_row + c, mask=live, other=0.0).to(tl.float32)
            s = (a + red.to(tl.float32)).to(dt)
            tl.store(xo_ptr + p * sxo_row + c, s, mask=live)
            x = s.to(tl.float32)
            y = x * tl.rsqrt(tl.sum(x * x) / cols + eps)
            w = tl.load(w_ptr + c, mask=live, other=0.0).to(tl.float32)
            out = (y * (1.0 + w)).to(dt)
            tl.store(h_ptr + row * sh_row + c, out, mask=live)
            if SELF != 0:
                _sp_push_row(ag0, out, base + row * cols, c, live)
            if SELF != 1:
                _sp_push_row(ag1, out, base + row * cols, c, live)
            if SELF != 2:
                _sp_push_row(ag2, out, base + row * cols, c, live)
            if SELF != 3:
                _sp_push_row(ag3, out, base + row * cols, c, live)
        tl.inline_asm_elementwise(
            "s_waitcnt vmcnt(0)", "=v,v", [c], dtype=tl.int32, is_pure=False, pack=1
        )
        tl.debug_barrier()
        _sp_flags(fb0, fb1, fb2, fb3, foff_me, value, SELF, ONE_RELEASE, FIRST_PEER)
        for q in tl.static_range(4):
            if q != SELF:
                _sp_wait(fb_me + slot * 4 * MAXS + q * MAXS + prog, value, RELAXED_SPIN)
        tl.debug_barrier()
        # 4. copy the peers' normed rows into `h`
        if COPY_AG:
            for j in tl.static_range(R):
                p = prog * R + j
                for q in tl.static_range(4):
                    if q != SELF:
                        rq = q * S + p
                        v = tl.load(ag_me + base + rq * cols + c, mask=live, other=0.0, volatile=True)
                        tl.store(h_ptr + rq * sh_row + c, v, mask=live)

        done = tl.atomic_add(ctr_ptr + 1, 1, sem="acq_rel", scope="gpu")
        if done == tl.num_programs(0) - 1:
            tl.store(ctr_ptr + 1, 0)
            tl.store(ctr_ptr, e + 1)


SP_COPY_AG = True

SP_RELAXED_SPIN = os.environ.get("SEED_AR_SP_RELAXED_SPIN", "0") not in ("0", "", "false", "False")
"""`SEED_AR_SP_RELAXED_SPIN=1`: the SP kernel polls its local flags with relaxed atomics and
issues one system-scope acquire after the flag matches, instead of an acquire (an L2
invalidate) on every poll. Same ordering guarantee; same numbers."""

SP_ROWS_PER_PROG = int(os.environ.get("SEED_AR_SP_ROWS_PER_PROG", "1"))
"""`SEED_AR_SP_ROWS_PER_PROG=R`: each SP program handles `R` shard rows (one flag round per
phase for all of them), when the shard has at least `SP_ROWS_PER_PROG_MIN` rows. Fewer
programs means fewer system-scope releases per call; per-row arithmetic is unchanged."""

SP_ROWS_PER_PROG_MIN = int(os.environ.get("SEED_AR_SP_ROWS_PER_PROG_MIN", "128"))

SP_ONE_RELEASE = os.environ.get("SEED_AR_SP_ONE_RELEASE", "0") not in ("0", "", "false", "False")
"""`SEED_AR_SP_ONE_RELEASE=1`: of the three flag stores after each push, only the first is a
system-scope release (L2 writeback + wait for prior stores); the other two are relaxed and are
issued after it, so every row the flags announce is already visible."""
"""Diagnostic only: False skips copying the peers' normed rows into `h` (wrong output; times
the copy)."""


def sp_ar_add_rmsnorm(
    m: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    peers: tuple,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(r + all_reduce(m)[own rows], rmsnorm(r_full + all_reduce(m)))` with `r` this rank's
    row shard (`SEED_AR_SP`): returns the new residual shard `[S, cols]` and the full normed
    `[R, cols]`. `peers` is `(rs, ag, fa, fb, ctr, self_rank, slot_elems, maxs, num_warps)`
    from `allreduce_custom.CustomAllReduce.sp_ar_add_rmsnorm`, which checks preconditions."""
    rs, ag, fa, fb, ctr, rank, slot_elems, maxs, num_warps = peers
    cols = m.shape[-1]
    mf = m.reshape(-1, cols)
    rf = r.reshape(-1, cols)
    s_rows = rf.shape[0]
    xo = torch.empty_like(rf)
    h = torch.empty(mf.shape, dtype=m.dtype, device=m.device)
    rpp = SP_ROWS_PER_PROG if s_rows % SP_ROWS_PER_PROG == 0 and s_rows >= SP_ROWS_PER_PROG_MIN else 1
    _sp_ar_add_rmsnorm_kernel[(s_rows // rpp,)](
        mf,
        rf,
        w,
        xo,
        h,
        *rs,
        *ag,
        *fa,
        *fb,
        ctr,
        mf.stride(0),
        rf.stride(0),
        xo.stride(0),
        h.stride(0),
        s_rows,
        cols,
        eps,
        slot_elems,
        SELF=rank,
        MAXS=maxs,
        BLOCK=triton.next_power_of_2(cols),
        num_warps=num_warps,
        COPY_AG=SP_COPY_AG,
        RELAXED_SPIN=SP_RELAXED_SPIN,
        ONE_RELEASE=SP_ONE_RELEASE,
        FIRST_PEER=1 if rank == 0 else 0,
        R=rpp,
    )
    return xo, h

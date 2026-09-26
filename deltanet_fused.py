"""Fused Triton kernels for the Gated DeltaNet decode step.

What this replaces, and why it is worth a kernel. At one token per slot the DeltaNet mixer
computes almost nothing: its whole weight and state traffic is about 10 GB per step, roughly
3 ms at the bandwidth this part delivers. It measured 64.12 ms, 34% of the decode step, the
largest single term. The reason is dispatch count, not work: the torch spelling issues about
35 `aten` calls per layer over 45 layers, and the step is host-issue-bound (the host takes
186 ms to issue a step the device finishes 2.16 ms later, with the device idle 35% of the
time). In that regime a call costs what it costs to *issue*, nearly independent of what it
computes, so collapsing twelve small ops into one kernel removes eleven real milliseconds.
See `TP_BYTELUT_BOTTLENECK_2026-09-22.md` sections 1 and 5.

Four kernels, all decode-only (`T == 1`, one token per slot):

- `gated_rmsnorm` replaces twelve torch ops on a few-hundred-KB tensor with one.
- `delta_rule_decode` replaces the per-token recurrence's op chain, its `l2norm`/cast
  preamble, the `repeat_interleave` that widens q/k onto the value heads, and the gate
  activations (`sigmoid`, `softplus`, two `exp`) with one. Besides the dispatches, it is the
  only spelling that reads the recurrent state once: the torch form touches `rec` five times
  (`mul_`, the `mem` reduction, `add_`'s read and write, the output reduction), and `rec` is
  50 MB per layer per rank at 48 slots, so that is ~15 GB/step of avoidable traffic. Its
  optional `active` argument (`SEED_FUSE_GLUE=1`) additionally removes the `clone()` +
  `torch.where` + `copy_` `graph_decode.deltanet_decode_static` used to wrap it in just to
  leave an inactive captured-replay row's state alone -- see the kernel's docstring.
- `masked_row_copy` is the same "leave an inactive row alone, in the store, for free" trick
  applied to `graph_decode.causal_conv_static`'s conv-state update, which was its own
  `torch.where` + `copy_` over the small `[B, conv_dim, conv_k - 1]` state.
- `causal_conv_decode` (`SEED_DN_CONV_INPLACE=1`) replaces the whole conv step -- gather,
  `cat`, masked state copy, `F.conv1d`, `SiLU`, scatter, plus the transposes around it -- with
  one kernel that reads and advances the lane pool directly by `lanes`, the same in-place
  trick `delta_rule_decode`'s `lanes` applies to `rec`.

All three are reformulations of the torch code in `model.py`, not approximations, and the
tests in `seed_tests/test_deltanet_fused.py` hold them to that against the torch chain. They
are not bit-identical to it: `tl.sum` reduces in a different order than `torch.sum`, which
moves the last bits of the two fp32 reductions. Everything that torch rounds to the storage
dtype mid-expression is rounded here too, so the difference is reduction order alone.

Prefill keeps the torch path. It runs many tokens per call, where the ops are large enough to
be worth their own launch and the recurrence is a Python loop over positions regardless.
"""

from __future__ import annotations

import os

import deltanet_prefill_chunked
import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - exercised only where triton is absent
    HAVE_TRITON = False

L2_EPS = 1e-6
"""`model.l2norm`'s default epsilon, which the DeltaNet path always takes."""

SOFTPLUS_THRESHOLD = 20.0
"""`F.softplus`'s default linear threshold. Above it `log1p(exp(z))` is `z` to fp32."""

BLOCK_V = int(os.environ.get("SEED_DELTANET_BLOCK_V", "32"))
"""Value channels one `delta_rule_decode` program owns. Tuning knob, not a correctness one.

The recurrence is independent per value channel -- `mem`, `delta`, the state update and the
output all index `v` alone -- so the state tile a program holds is `[k_dim, BLOCK_V]` and the
grid gets a second axis. At the real shape (`k_dim` 128, `v_dim` 128, 16 local value heads, 48
slots) 32 puts 3,072 programs on the device against a 128x32 fp32 tile per program.
"""


def available(device: torch.device) -> bool:
    """True when the fused decode kernels can run for tensors on `device`.

    Same contract and same escape hatch as `mxfp4_gemv.available`: Triton compiles per target
    architecture and these are only reachable on an accelerator, so CPU tensors keep the torch
    path. `SEED_FUSED_DELTANET=0` forces the torch path on a machine that could run the
    kernels, which is what an ablation benchmark or a bisect of a suspected kernel bug uses.
    The functions below do not consult this, so the tests can drive them directly on CPU
    tensors under `TRITON_INTERPRET=1`.
    """
    if not HAVE_TRITON or os.environ.get("SEED_FUSED_DELTANET", "1") == "0":
        return False
    return device.type == "cuda"


BLOCK_V_PREFILL = int(os.environ.get("SEED_DELTANET_PREFILL_BLOCK_V", "32"))
"""Value channels one `fused_recurrent_prefill` program owns. Same tuning knob as `BLOCK_V`,
kept separate because the two kernels have different register pressure per program: this one
holds its `[k_dim, BLOCK_V_PREFILL]` state tile live across the whole `T` loop instead of one
step, so occupancy trades off against per-program live range rather than program count alone.
At the real shape (`k_dim` 128, 16 local value heads) 32 gives 4 V-tiles per head, 64 programs
per sequence."""


PREFILL_PREFETCH = os.environ.get("SEED_DELTANET_PREFILL_PREFETCH", "0") not in ("0", "", "false")
"""`SEED_DELTANET_PREFILL_PREFETCH=1`: `fused_recurrent_prefill` loads position t+1's operands
before position t's math (`prefetch=` overrides per call). The recurrence is one dependent
chain of `T` steps per program (64 programs per sequence at the real shape, a quarter of the
CUs), so each step's global-load latency is paid serially unless it is overlapped. Off by
default until measured; `bench_prefill_moe.py` prints the recurrence's us/token."""


PREFILL_WARPS = int(os.environ.get("SEED_DELTANET_PREFILL_WARPS", "4"))
"""Warps per `fused_recurrent_prefill` program (4 = Triton's default, the behavior before
this knob). Each step does four reductions over `k_dim`; with 4 warps each is a cross-warp
LDS round trip with barriers (12 `s_barrier` per step in the gfx942 ISA), which sits on the
serial per-step critical path. Fewer warps trade those for intra-wave reductions and more
registers per lane; `bench_prefill_moe.py` sweeps it."""


def available_prefill(device: torch.device) -> bool:
    """True when `fused_recurrent_prefill` can run for tensors on `device`.

    Same escape hatch as `available`, under its own flag (`SEED_FUSED_RECURRENT_PREFILL`) so
    an ablation can disable the prefill kernel alone and keep the decode kernel, or vice versa.
    """
    if not HAVE_TRITON or os.environ.get("SEED_FUSED_RECURRENT_PREFILL", "1") == "0":
        return False
    return device.type == "cuda"


if HAVE_TRITON:
    _L2 = tl.constexpr(L2_EPS)
    _SOFTPLUS_MAX = tl.constexpr(SOFTPLUS_THRESHOLD)
    """The two constants above again, as Triton constexprs.

    A `@triton.jit` body may only read globals that are `tl.constexpr` instances. The
    reference interpreter is laxer and accepts the plain floats, so a run that only ever went
    through `TRITON_INTERPRET=1` would not catch this; the same note is on `mxfp4_gemv._MX`.
    """

    @triton.jit
    def _gated_rmsnorm_kernel(
        x_ptr,
        g_ptr,
        w_ptr,
        o_ptr,
        sx_b,
        sx_h,
        sg_b,
        sg_h,
        so_b,
        so_h,
        heads,
        cols,
        eps,
        BLOCK: tl.constexpr,
        ROUND_IN: tl.constexpr = False,
    ):
        """One program per `(batch, head)` row of `model.gated_rmsnorm`.

        The three roundings to the storage dtype are deliberate and mirror the torch
        expression exactly: it normalizes in fp32, rounds before scaling by `w`, rounds the
        product, and rounds again after the `silu` gate.
        """
        pid = tl.program_id(0)
        row, head = pid // heads, pid % heads
        c = tl.arange(0, BLOCK)
        live = c < cols
        dt = o_ptr.dtype.element_ty

        x = tl.load(x_ptr + row * sx_b + head * sx_h + c, mask=live, other=0.0)
        if ROUND_IN:  # fp32 input: the caller's `.to(dt)`, done here instead of as a launch
            x = x.to(dt)
        x = x.to(tl.float32)
        y = (x * tl.rsqrt(tl.sum(x * x) / cols + eps)).to(dt)
        w = tl.load(w_ptr + c, mask=live, other=0.0).to(tl.float32)
        y = (y.to(tl.float32) * w).to(dt)
        g = tl.load(g_ptr + row * sg_b + head * sg_h + c, mask=live, other=0.0).to(tl.float32)
        out = (y.to(tl.float32) * (g * tl.sigmoid(g))).to(dt)
        tl.store(o_ptr + row * so_b + head * so_h + c, out, mask=live)

    @triton.jit
    def _delta_rule_decode_kernel(  # noqa: PLR0915 - one fused expression, not a procedure
        q_ptr,
        k_ptr,
        v_ptr,
        a_ptr,
        b_ptr,
        rec_ptr,
        alog_ptr,
        dtb_ptr,
        o_ptr,
        active_ptr,
        lane_ptr,
        sq_b,
        sq_h,
        sk_b,
        sk_h,
        sv_b,
        sv_h,
        sa_b,
        sb_b,
        sr_b,
        sr_h,
        sr_k,
        so_b,
        so_h,
        heads,
        rep,
        k_dim,
        v_dim,
        scale,
        HAS_ACTIVE: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_C: tl.constexpr,
        HAS_LANES: tl.constexpr = False,
    ):
        """One decode step of the gated delta rule, for one `(slot, value head, v block)`.

        `q`/`k` are indexed at `head // rep`, which is what `repeat_interleave` materialized;
        the gate activations are computed here from the raw projections, so `a` and `b` come
        in as `in_proj_a` and `in_proj_b` produced them.

        `HAS_ACTIVE`/`active_ptr` (`SEED_FUSE_GLUE=1`, `graph_decode.deltanet_decode_static`):
        an optional per-row `int32` mask, 1 for a row this step's batch actually holds and 0
        for a padding row of a captured replay. When given, the `rec` store is additionally
        gated on it, so an inactive row's state is left untouched *by the kernel itself* --
        the caller no longer needs to `clone()` the whole state, run the kernel, and
        `torch.where` the update back in, which is 2-3 full `[rows, heads, k_dim, v_dim]`
        HBM round trips per layer for a tensor the module docstring measures at ~50 MB/layer/
        rank at 48 slots. `other=0.0` on the `rec` load already zeros a bounds-masked lane, so
        ANDing the store mask with `act` is exact: a False lane never had its `rec_off`
        touched, identical to the `torch.where(active, updated, original)` it replaces. The
        output row is still written unconditionally (needed to fill `o_ptr` at every row so
        its shape stays static-batch-wide); an inactive row's mixer output is discarded by the
        caller exactly as it always was.
        """
        pid = tl.program_id(0)
        row, head = pid // heads, pid % heads
        kh = head // rep
        kc = tl.arange(0, BLOCK_K)
        k_live = kc < k_dim
        vc = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        v_live = vc < v_dim

        q = tl.load(q_ptr + row * sq_b + kh * sq_h + kc, mask=k_live, other=0.0).to(tl.float32)
        k = tl.load(k_ptr + row * sk_b + kh * sk_h + kc, mask=k_live, other=0.0).to(tl.float32)
        q = q * tl.rsqrt(tl.sum(q * q) + _L2) * scale
        k = k * tl.rsqrt(tl.sum(k * k) + _L2)
        v = tl.load(v_ptr + row * sv_b + head * sv_h + vc, mask=v_live, other=0.0).to(tl.float32)

        # beta rounds through the projection's own dtype because `.sigmoid()` runs there.
        raw = tl.load(b_ptr + row * sb_b + head)
        beta = tl.sigmoid(raw.to(tl.float32)).to(b_ptr.dtype.element_ty).to(tl.float32)
        z = tl.load(a_ptr + row * sa_b + head).to(tl.float32) + tl.load(dtb_ptr + head)
        softplus = tl.where(z > _SOFTPLUS_MAX, z, tl.log(1.0 + tl.exp(z)))
        decay = tl.exp(-tl.exp(tl.load(alog_ptr + head)) * softplus)

        rec_row = row
        if HAS_LANES:
            # `SEED_DN_STATE_INPLACE=1`: `rec` is the whole `[max_batch, ...]` lane pool and
            # row `row` owns lane `lane_ptr[row]`, so the state is read and advanced in place
            # instead of through a gathered copy.
            rec_row = tl.load(lane_ptr + row).to(tl.int64)
        rec_off = rec_ptr + rec_row * sr_b + head * sr_h + kc[:, None] * sr_k + vc[None, :]
        live = k_live[:, None] & v_live[None, :]
        rec = tl.load(rec_off, mask=live, other=0.0) * decay
        delta = (v - tl.sum(rec * k[:, None], axis=0)) * beta
        rec = rec + k[:, None] * delta[None, :]
        store_mask = live
        if HAS_ACTIVE:
            act = tl.load(active_ptr + row) != 0
            store_mask = live & act
        tl.store(rec_off, rec, mask=store_mask)
        tl.store(
            o_ptr + row * so_b + head * so_h + vc, tl.sum(rec * q[:, None], axis=0), mask=v_live
        )

    @triton.jit
    def _delta_rule_verify_exact_kernel(  # noqa: PLR0915
        q_ptr,
        k_ptr,
        v_ptr,
        a_ptr,
        b_ptr,
        rec_ptr,
        alog_ptr,
        dtb_ptr,
        o_ptr,
        active_ptr,
        lane_ptr,
        length_ptr,
        sq_b,
        sq_t,
        sq_h,
        sk_b,
        sk_t,
        sk_h,
        sv_b,
        sv_t,
        sv_h,
        sa_b,
        sa_t,
        sb_b,
        sb_t,
        sr_b,
        sr_h,
        sr_k,
        so_b,
        so_t,
        so_h,
        heads,
        rep,
        k_dim,
        v_dim,
        scale,
        HAS_ACTIVE: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_C: tl.constexpr,
        T: tl.constexpr,
        HAS_LANES: tl.constexpr = False,
        HAS_LENGTHS: tl.constexpr = False,
        MATERIALIZE_STEPS: tl.constexpr = True,
    ):
        """Fixed-width verify recurrence with decode's exact per-step operation ordering.

        One program applies `T` unrolled decode updates. The strict specialization
        materializes fp32 state between steps, matching the store/reload boundary of separate
        decode launches and preventing the compiler from carrying extra precision or changing
        contraction across that boundary. Raw gates are activated inside the loop exactly as in
        `_delta_rule_decode_kernel`; this is the semantic difference from the generic packed
        prefill kernel, whose gates are precomputed by torch operations.
        """
        pid = tl.program_id(0)
        row, head = pid // heads, pid % heads
        kh = head // rep
        kc = tl.arange(0, BLOCK_K)
        k_live = kc < k_dim
        vc = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        v_live = vc < v_dim
        live = k_live[:, None] & v_live[None, :]

        rec_row = row
        if HAS_LANES:
            rec_row = tl.load(lane_ptr + row).to(tl.int64)
        rec_off = rec_ptr + rec_row * sr_b + head * sr_h + kc[:, None] * sr_k + vc[None, :]
        rec = tl.load(rec_off, mask=live, other=0.0).to(tl.float32)

        for step in range(T):
            step_live = True
            if HAS_LENGTHS:
                step_live = step < tl.load(length_ptr + row)
            q = tl.load(
                q_ptr + row * sq_b + step * sq_t + kh * sq_h + kc,
                mask=k_live,
                other=0.0,
            ).to(tl.float32)
            k = tl.load(
                k_ptr + row * sk_b + step * sk_t + kh * sk_h + kc,
                mask=k_live,
                other=0.0,
            ).to(tl.float32)
            q = q * tl.rsqrt(tl.sum(q * q) + _L2) * scale
            k = k * tl.rsqrt(tl.sum(k * k) + _L2)
            v = tl.load(
                v_ptr + row * sv_b + step * sv_t + head * sv_h + vc,
                mask=v_live,
                other=0.0,
            ).to(tl.float32)

            raw = tl.load(b_ptr + row * sb_b + step * sb_t + head)
            beta = tl.sigmoid(raw.to(tl.float32)).to(b_ptr.dtype.element_ty).to(tl.float32)
            z = (
                tl.load(a_ptr + row * sa_b + step * sa_t + head).to(tl.float32)
                + tl.load(dtb_ptr + head)
            )
            softplus = tl.where(z > _SOFTPLUS_MAX, z, tl.log(1.0 + tl.exp(z)))
            decay = tl.exp(-tl.exp(tl.load(alog_ptr + head)) * softplus)

            next_rec = rec * decay
            delta = (v - tl.sum(next_rec * k[:, None], axis=0)) * beta
            next_rec = next_rec + k[:, None] * delta[None, :]
            if HAS_LENGTHS:
                rec = tl.where(step_live, next_rec, rec)
            else:
                rec = next_rec
            if MATERIALIZE_STEPS:
                step_store = live
                if HAS_ACTIVE:
                    step_store = live & (tl.load(active_ptr + row) != 0)
                tl.store(rec_off, rec, mask=step_store)
                tl.debug_barrier()
                rec = tl.load(rec_off, mask=live, other=0.0).to(tl.float32)
            tl.store(
                o_ptr + row * so_b + step * so_t + head * so_h + vc,
                tl.sum(rec * q[:, None], axis=0),
                mask=v_live,
            )

        store_mask = live
        if HAS_ACTIVE:
            act = tl.load(active_ptr + row) != 0
            store_mask = live & act
        tl.store(rec_off, rec, mask=store_mask)

    @triton.jit
    def _conv_tap_window(x, state_base, w_base, sw_k, ss_k, live, CONV_K: tl.constexpr):
        """`_causal_conv_decode_kernel`'s per-channel window and SiLU, exact same op order,
        rounded to bf16 as that kernel's output store rounds it. No state write."""
        acc = tl.zeros(x.shape, dtype=tl.float32)
        for j in range(CONV_K - 1):
            sj = tl.load(state_base + j * ss_k, mask=live, other=0.0).to(tl.float32)
            wj = tl.load(w_base + j * sw_k, mask=live, other=0.0).to(tl.float32)
            acc += sj * wj
        wlast = tl.load(w_base + (CONV_K - 1) * sw_k, mask=live, other=0.0).to(tl.float32)
        acc += x * wlast
        return (acc * tl.sigmoid(acc)).to(tl.bfloat16).to(tl.float32)

    @triton.jit
    def _conv_state_shift(x, state_base, ss_k, mask, CONV_K: tl.constexpr):
        """The window's state advance: tap `j` moves to `j - 1`, `x` enters at `CONV_K - 2`."""
        for j in range(1, CONV_K - 1):
            sj = tl.load(state_base + j * ss_k, mask=mask, other=0.0)
            tl.store(state_base + (j - 1) * ss_k, sj, mask=mask)
        tl.store(state_base + (CONV_K - 2) * ss_k, x.to(state_base.dtype.element_ty), mask=mask)

    @triton.jit
    def _dn_decode_fused_kernel(  # noqa: PLR0915
        qkv_ptr,
        cw_ptr,
        cs_ptr,
        a_ptr,
        b_ptr,
        rec_ptr,
        alog_ptr,
        dtb_ptr,
        z_ptr,
        nw_ptr,
        os_ptr,
        o_ptr,
        active_ptr,
        lane_ptr,
        cnt_ptr,
        sqkv_b,
        sw_c,
        sw_k,
        ss_l,
        ss_c,
        ss_k,
        sa_b,
        sb_b,
        sr_b,
        sr_h,
        sr_k,
        sz_b,
        sz_h,
        so_b,
        so_h,
        heads,
        rep,
        k_heads,
        k_dim,
        v_dim,
        scale,
        eps,
        CONV_K: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_C: tl.constexpr,
        NVB: tl.constexpr,
        NORM_BLOCK: tl.constexpr,
        CNT_STRIDE: tl.constexpr,
    ):
        """`causal_conv_decode` + `delta_rule_decode` (`HAS_ACTIVE`, `HAS_LANES`) +
        `gated_rmsnorm` (`ROUND_IN`) as one launch (`SEED_DN_DECODE_FUSED`), same grid as
        `delta_rule_decode`: program `(row * heads + head, vb)` owns value channels
        `vb * BLOCK_C ...` of head `head`.

        Conv: each program computes the q/k windows of its key head (shared by `rep * NVB`
        programs) and its own value channels' window from the lane's *old* conv state, with
        `_causal_conv_decode_kernel`'s arithmetic. Its value channels are its alone, so it
        advances their state itself. The q/k state is advanced by whichever of the `rep * NVB`
        sharers arrives last at `cnt[0][row, k head]` (acq_rel), after every sharer has read
        it. Recurrence: `_delta_rule_decode_kernel`'s body unchanged. Norm: every program
        writes its fp32 output slice to `os`, and the last of the `NVB` programs of a
        `(row, head)` to arrive at `cnt[1]` reads the whole row back and runs
        `_gated_rmsnorm_kernel`'s body. Counters and `os` return to zero, so replays and
        layers reuse them. Bit-exact against the three kernels (output, conv pool, recurrent
        pool; checked on MI300A at B 1/16/48, eager and replayed).

        Measured per layer at the TP=4 shapes, isolated in a graph: B1 9.6 vs 12.1 us, B16
        23.8 vs 23.4, B48 54.3 vs 48.5. The q/k windows are recomputed by all `rep * NVB`
        sharers, which costs more than the two launches it saves once B is large."""
        pid = tl.program_id(0)
        row, head = pid // heads, pid % heads
        kh = head // rep
        kc = tl.arange(0, BLOCK_K)
        k_live = kc < k_dim
        vc = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        v_live = vc < v_dim
        lane = tl.load(lane_ptr + row).to(tl.int64)
        act = tl.load(active_ptr + row) != 0
        key_dim = k_heads * k_dim
        cs_lane = cs_ptr + lane * ss_l

        cq = kh * k_dim + kc
        ck = key_dim + kh * k_dim + kc
        cv = 2 * key_dim + head * v_dim + vc
        xq = tl.load(qkv_ptr + row * sqkv_b + cq, mask=k_live, other=0.0).to(tl.float32)
        xk = tl.load(qkv_ptr + row * sqkv_b + ck, mask=k_live, other=0.0).to(tl.float32)
        xv = tl.load(qkv_ptr + row * sqkv_b + cv, mask=v_live, other=0.0).to(tl.float32)
        q = _conv_tap_window(
            xq, cs_lane + cq * ss_c, cw_ptr + cq * sw_c, sw_k, ss_k, k_live, CONV_K
        )
        k = _conv_tap_window(
            xk, cs_lane + ck * ss_c, cw_ptr + ck * sw_c, sw_k, ss_k, k_live, CONV_K
        )
        v = _conv_tap_window(
            xv, cs_lane + cv * ss_c, cw_ptr + cv * sw_c, sw_k, ss_k, v_live, CONV_K
        )
        _conv_state_shift(xv, cs_lane + cv * ss_c, ss_k, v_live & act, CONV_K)

        q = q * tl.rsqrt(tl.sum(q * q) + _L2) * scale
        k = k * tl.rsqrt(tl.sum(k * k) + _L2)

        raw = tl.load(b_ptr + row * sb_b + head)
        beta = tl.sigmoid(raw.to(tl.float32)).to(b_ptr.dtype.element_ty).to(tl.float32)
        zz = tl.load(a_ptr + row * sa_b + head).to(tl.float32) + tl.load(dtb_ptr + head)
        softplus = tl.where(zz > _SOFTPLUS_MAX, zz, tl.log(1.0 + tl.exp(zz)))
        decay = tl.exp(-tl.exp(tl.load(alog_ptr + head)) * softplus)

        rec_off = rec_ptr + lane * sr_b + head * sr_h + kc[:, None] * sr_k + vc[None, :]
        live = k_live[:, None] & v_live[None, :]
        rec = tl.load(rec_off, mask=live, other=0.0) * decay
        delta = (v - tl.sum(rec * k[:, None], axis=0)) * beta
        rec = rec + k[:, None] * delta[None, :]
        tl.store(rec_off, rec, mask=live & act)
        o = tl.sum(rec * q[:, None], axis=0)
        # Cross-program hand-off through agent-scope *relaxed* atomics (int32 bit patterns),
        # which MI300's per-XCD L2s keep coherent; an acquire/release pair would add a
        # whole-L2 writeback (`buffer_wbl2`) per program instead (measured 7x slower).
        # `os` is all-zero between launches, so an integer add deposits the bits exactly.
        tl.atomic_add(
            os_ptr + row * so_b + head * so_h + vc,
            o.to(tl.int32, bitcast=True),
            mask=v_live,
            sem="relaxed",
            scope="gpu",
        )

        # Every wave's loads and atomics done, then one arrival per program.
        tl.inline_asm_elementwise(
            "s_waitcnt vmcnt(0)", "=v,v", [kc], dtype=tl.int32, is_pure=False, pack=1
        )
        tl.debug_barrier()
        n_rows = tl.num_programs(0) // heads
        # One counter per 128-byte line: same-line atomics serialize in L2 (measured 7x slower).
        c_qk = cnt_ptr + (row * k_heads + kh) * CNT_STRIDE
        if tl.atomic_add(c_qk, 1, sem="relaxed", scope="gpu") == rep * NVB - 1:
            tl.atomic_xchg(c_qk, 0, sem="relaxed", scope="gpu")
            _conv_state_shift(xq, cs_lane + cq * ss_c, ss_k, k_live & act, CONV_K)
            _conv_state_shift(xk, cs_lane + ck * ss_c, ss_k, k_live & act, CONV_K)
        c_n = cnt_ptr + (n_rows * k_heads + row * heads + head) * CNT_STRIDE
        if tl.atomic_add(c_n, 1, sem="relaxed", scope="gpu") == NVB - 1:
            tl.atomic_xchg(c_n, 0, sem="relaxed", scope="gpu")
            nc = tl.arange(0, NORM_BLOCK)
            n_live = nc < v_dim
            dt = o_ptr.dtype.element_ty
            # `and 0` returns the bits and re-zeroes `os` for the next launch.
            xi = tl.atomic_and(
                os_ptr + row * so_b + head * so_h + nc,
                tl.zeros([NORM_BLOCK], tl.int32),
                mask=n_live,
                sem="relaxed",
                scope="gpu",
            )
            x = xi.to(tl.float32, bitcast=True)
            x = x.to(dt).to(tl.float32)
            y = (x * tl.rsqrt(tl.sum(x * x) / v_dim + eps)).to(dt)
            w = tl.load(nw_ptr + nc, mask=n_live, other=0.0).to(tl.float32)
            y = (y.to(tl.float32) * w).to(dt)
            g = tl.load(z_ptr + row * sz_b + head * sz_h + nc, mask=n_live, other=0.0).to(
                tl.float32
            )
            out = (y.to(tl.float32) * (g * tl.sigmoid(g))).to(dt)
            tl.store(o_ptr + row * so_b + head * so_h + nc, out, mask=n_live)

    @triton.jit
    def _causal_conv_decode_kernel(
        x_ptr,
        w_ptr,
        state_ptr,
        lane_ptr,
        active_ptr,
        o_ptr,
        sx_b,
        sw_c,
        sw_k,
        ss_l,
        ss_c,
        ss_k,
        so_b,
        channels,
        conv_k: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """One program per `(row, channel-block)`: `graph_decode.causal_conv_static`'s depthwise
        causal conv (kernel width `conv_k`) plus `SiLU`, reading and advancing the lane's state
        directly in the pool -- no gather, no scatter, no channel-major transpose.

        `x` is `qkv` at `[B, 1, C]` row-major (`C` contiguous): a depthwise conv's per-channel
        window is a sum of `conv_k` scalars, which needs no `F.conv1d`-style channel-major
        layout the way the torch path's `.transpose(1, 2)` existed to feed.

        `state_ptr` is the *whole* `[L, C, conv_k - 1]` lane pool; row `row` reads and updates
        lane `lane_ptr[row]` in place -- the same trick `_delta_rule_decode_kernel`'s
        `HAS_LANES` applies to the recurrent state (`SEED_DN_STATE_INPLACE`), here unconditional
        (`SEED_DN_CONV_INPLACE` has no non-lanes mode). Tap `j`'s value is stored to its shifted
        slot `j - 1` in the same unrolled iteration it is loaded in (`j == 0` has nowhere to
        shift to -- it is dropped, the oldest sample falling off the window -- and the loop
        never revisits a position after storing it: iteration `j` only ever writes slot `j - 1`,
        strictly behind the slot every later iteration `j' > j` reads), so a lane's own state is
        never read after this program has partly overwritten it. The out-of-range address
        computed for `c >= channels` lanes is never dereferenced (mask-only, standard Triton
        semantics -- no OOB access).

        `active_ptr` gates the state write only, exactly like `masked_row_copy`: a padding row's
        shared filler lane is read (safe -- many padding-row readers, no writers among them) but
        never written, so the store mask is `live & active[row]`. The conv output itself is
        computed and stored for every row unconditionally, matching `causal_conv_static`, whose
        `mixed` is likewise computed for every row regardless of `active`.

        Precision: fp32 accumulate throughout (state/weight/x loaded and multiplied in fp32,
        summed in fp32), one rounding to the storage dtype at the `SiLU`'d output and at each
        stored state tap -- see `causal_conv_decode`'s docstring for why this matches
        `F.conv1d`'s bf16 behavior closely enough for the campaign's reformulation bar.
        """
        row = tl.program_id(0)
        c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        live = c < channels
        lane = tl.load(lane_ptr + row).to(tl.int64)
        act = tl.load(active_ptr + row) != 0

        x = tl.load(x_ptr + row * sx_b + c, mask=live, other=0.0).to(tl.float32)
        state_base = state_ptr + lane * ss_l + c * ss_c

        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        store_mask = live & act
        for j in range(conv_k - 1):  # conv_k is tl.constexpr: unrolled, not a device-side loop
            s = tl.load(state_base + j * ss_k, mask=live, other=0.0).to(tl.float32)
            wj = tl.load(w_ptr + c * sw_c + j * sw_k, mask=live, other=0.0).to(tl.float32)
            acc += s * wj
            if j > 0:
                tl.store(
                    state_base + (j - 1) * ss_k, s.to(state_ptr.dtype.element_ty), mask=store_mask
                )
        wlast = tl.load(w_ptr + c * sw_c + (conv_k - 1) * sw_k, mask=live, other=0.0).to(tl.float32)
        acc += x * wlast
        if conv_k > 1:
            tl.store(
                state_base + (conv_k - 2) * ss_k, x.to(state_ptr.dtype.element_ty), mask=store_mask
            )

        out = acc * tl.sigmoid(acc)
        tl.store(o_ptr + row * so_b + c, out.to(o_ptr.dtype.element_ty), mask=live)

    @triton.jit
    def _causal_conv_verify_exact_kernel(
        x_ptr,
        w_ptr,
        state_ptr,
        length_ptr,
        active_ptr,
        o_ptr,
        sx_b,
        sx_t,
        sw_c,
        sw_k,
        ss_b,
        ss_c,
        ss_k,
        so_b,
        so_t,
        channels,
        conv_k: tl.constexpr,
        T: tl.constexpr,
        HAS_LENGTHS: tl.constexpr,
        HAS_ACTIVE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Fixed-width causal conv with decode's per-step kernel ordering."""
        row = tl.program_id(0)
        c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        live = c < channels
        state_base = state_ptr + row * ss_b + c * ss_c
        w_base = w_ptr + c * sw_c
        row_active = True
        if HAS_ACTIVE:
            row_active = tl.load(active_ptr + row) != 0

        for step in range(T):
            step_live = row_active
            if HAS_LENGTHS:
                step_live = step_live & (step < tl.load(length_ptr + row))
            x = tl.load(
                x_ptr + row * sx_b + step * sx_t + c,
                mask=live,
                other=0.0,
            ).to(tl.float32)
            out = _conv_tap_window(x, state_base, w_base, sw_k, ss_k, live, conv_k)
            tl.store(
                o_ptr + row * so_b + step * so_t + c,
                out.to(o_ptr.dtype.element_ty),
                mask=live,
            )
            _conv_state_shift(x, state_base, ss_k, live & step_live, conv_k)
            # Match the state store/reload boundary between ordinary decode launches.
            tl.debug_barrier()

    @triton.jit
    def _masked_row_copy_kernel(
        src_ptr,
        dst_ptr,
        active_ptr,
        s_row,
        s_ch,
        d_row,
        d_ch,
        channels,
        width: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """`dst[row] = src[row] if active[row] else dst[row]`, for one `[row, channel-block]`.

        `src`/`dst` are `[rows, channels, width]` with the `width` axis contiguous (stride 1)
        in both but independently strided `row`/`channel` axes -- `causal_conv_static` calls
        this with `src` a suffix slice of `torch.cat([state, x])` (channel stride `conv_k`,
        one wider than `width == conv_k - 1`) and `dst` the plain contiguous state buffer
        (channel stride `width`), so a flat `[rows, channels * width]` reshape would silently
        force a copy of the strided side. Reading `src` and writing `dst` through their own
        strides instead keeps both sides views, no extra HBM traffic beyond the copy itself.

        The `torch.where(active, src, dst)` this replaces reads `dst`'s current value only to
        hand it right back unchanged on an inactive row; gating the store on `active` instead
        means an inactive row's `dst` bytes are never touched at all -- no read, no write --
        rather than read-and-write-the-same-value, which is what `torch.where` compiles to
        with no way to skip it from Python (see `deltanet_fused.delta_rule_decode`'s
        `HAS_ACTIVE` for the same trick applied to the DeltaNet recurrent state itself).
        """
        row = tl.program_id(0)
        ch = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        live = ch < channels
        act = tl.load(active_ptr + row) != 0
        for wi in range(width):  # width is tl.constexpr: unrolled, not a device-side loop
            val = tl.load(src_ptr + row * s_row + ch * s_ch + wi, mask=live, other=0.0)
            tl.store(dst_ptr + row * d_row + ch * d_ch + wi, val, mask=live & act)

    @triton.jit
    def _prefill_step_loads(
        q_ptr,
        k_ptr,
        v_ptr,
        g_ptr,
        beta_ptr,
        row,
        head,
        kc,
        vc,
        k_ok,
        v_ok,
        ok,
        sq_b,
        sq_h,
        sk_b,
        sk_h,
        sv_b,
        sv_h,
        sg_b,
        sbeta_b,
    ):
        """One position's `q`, `k` (fp32 `[BLOCK_K]`), `v` (fp32 `[BLOCK_C]`), `g`, `beta`."""
        q = tl.load(q_ptr + row * sq_b + head * sq_h + kc, mask=k_ok, other=0.0).to(tl.float32)
        k = tl.load(k_ptr + row * sk_b + head * sk_h + kc, mask=k_ok, other=0.0).to(tl.float32)
        v = tl.load(v_ptr + row * sv_b + head * sv_h + vc, mask=v_ok, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + row * sg_b + head, mask=ok, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + row * sbeta_b + head, mask=ok, other=0.0).to(tl.float32)
        return q, k, v, g, beta

    @triton.jit
    def _prefill_step(rec, q, k, v, g, beta, scale, o_ptrs, v_live):
        """`model.delta_rule_recurrent`'s one step on a `[k_dim, BLOCK_C]` state tile: decay,
        delta update, query read-out (stored). Same expressions as before the split."""
        q = q * tl.rsqrt(tl.sum(q * q) + _L2) * scale
        k = k * tl.rsqrt(tl.sum(k * k) + _L2)
        rec = rec * tl.exp(g)
        delta = (v - tl.sum(rec * k[:, None], axis=0)) * beta
        rec = rec + k[:, None] * delta[None, :]
        tl.store(o_ptrs, tl.sum(rec * q[:, None], axis=0), mask=v_live)
        return rec

    @triton.jit
    def _fused_recurrent_prefill_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        g_ptr,
        beta_ptr,
        rec_ptr,
        cu_ptr,
        o_ptr,
        sq_b,
        sq_h,
        sk_b,
        sk_h,
        sv_b,
        sv_h,
        sg_b,
        sbeta_b,
        sr_s,
        sr_h,
        sr_k,
        so_b,
        so_h,
        heads,
        k_dim,
        v_dim,
        scale,
        BLOCK_K: tl.constexpr,
        BLOCK_C: tl.constexpr,
        PREFETCH: tl.constexpr,
    ):
        """Every prefill step of the gated delta rule, for one `(sequence, value head, v block)`.

        One program owns the whole `[k_dim, BLOCK_C]` state tile for its `(sequence, head,
        v-block)` in registers for the entire sequence: it reads `rec_ptr` once, loops
        `t = 0 .. seq_len-1` doing exactly the per-step math `model.delta_rule_recurrent` does
        (decay, then the delta update, then the query read-out), and writes `rec_ptr` back
        once at the end. That is the whole point of fusing the loop into the kernel instead of
        launching `delta_rule_decode` once per token: the state never round-trips through HBM
        mid-sequence.

        `q`/`k`/`v` are `[T_total, heads, dim]`, packed across sequences along the row axis
        (varlen; a single sequence is the `n_seqs == 1` case). `g` is the already-computed
        per-token log-decay (`model.py`'s `g = -A_log.exp() * softplus(...)`, `<= 0`) and
        `beta` is the already-sigmoid'd gate, both `[T_total, heads]`: unlike
        `_delta_rule_decode_kernel`, this kernel does not recompute either from raw
        projections, because the prefill call site (`Model.delta_rule`) already has them in
        that form for every position, not just one. `q`/`k` are l2-normalized here (`q`
        additionally scaled by `k_dim ** -0.5`), matching `model._delta_rule_inputs`. `rec_ptr`
        is `[n_seqs, heads, k_dim, v_dim]` fp32, read and written in place: `cu_ptr` (int32,
        `[n_seqs + 1]`) gives each sequence's row range into the packed `T_total` axis.
        """
        pid = tl.program_id(0)
        seq, head = pid // heads, pid % heads
        seq_start = tl.load(cu_ptr + seq)
        seq_len = tl.load(cu_ptr + seq + 1) - seq_start

        kc = tl.arange(0, BLOCK_K)
        k_live = kc < k_dim
        vc = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        v_live = vc < v_dim
        live = k_live[:, None] & v_live[None, :]

        rec_off = rec_ptr + seq * sr_s + head * sr_h + kc[:, None] * sr_k + vc[None, :]
        rec = tl.load(rec_off, mask=live, other=0.0).to(tl.float32)

        if PREFETCH:
            # Step t+1's operands are loaded before step t's math (they never depend on the
            # state), so their latency overlaps the step instead of heading it. The last step
            # re-reads its own row: in bounds, unused.
            ok = seq_len > 0
            q, k, v, g, beta = _prefill_step_loads(
                q_ptr,
                k_ptr,
                v_ptr,
                g_ptr,
                beta_ptr,
                seq_start,
                head,
                kc,
                vc,
                k_live & ok,
                v_live & ok,
                ok,
                sq_b,
                sq_h,
                sk_b,
                sk_h,
                sv_b,
                sv_h,
                sg_b,
                sbeta_b,
            )
            for t in range(0, seq_len):
                row = seq_start + t
                nxt = seq_start + tl.minimum(t + 1, seq_len - 1)
                qn, kn, vn, gn, bn = _prefill_step_loads(
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    g_ptr,
                    beta_ptr,
                    nxt,
                    head,
                    kc,
                    vc,
                    k_live,
                    v_live,
                    True,
                    sq_b,
                    sq_h,
                    sk_b,
                    sk_h,
                    sv_b,
                    sv_h,
                    sg_b,
                    sbeta_b,
                )
                o_row = o_ptr + row * so_b + head * so_h + vc
                rec = _prefill_step(rec, q, k, v, g, beta, scale, o_row, v_live)
                q, k, v, g, beta = qn, kn, vn, gn, bn
        else:
            for t in range(0, seq_len):
                row = seq_start + t
                q, k, v, g, beta = _prefill_step_loads(
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    g_ptr,
                    beta_ptr,
                    row,
                    head,
                    kc,
                    vc,
                    k_live,
                    v_live,
                    True,
                    sq_b,
                    sq_h,
                    sk_b,
                    sk_h,
                    sv_b,
                    sv_h,
                    sg_b,
                    sbeta_b,
                )
                o_row = o_ptr + row * so_b + head * so_h + vc
                rec = _prefill_step(rec, q, k, v, g, beta, scale, o_row, v_live)

        tl.store(rec_off, rec, mask=live)


def _rows(x: torch.Tensor, name: str) -> tuple[int, int]:
    """`(stride(0), stride(1))` of a `[rows, heads, channels]` operand, last axis contiguous."""
    if x.dim() != 3:
        raise ValueError(f"{name} must be [rows, heads, channels], got {tuple(x.shape)}")
    if x.stride(-1) != 1:
        raise ValueError(f"{name} must be contiguous along its last axis")
    return x.stride(0), x.stride(1)


def gated_rmsnorm(
    x: torch.Tensor,
    gate: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """`model.gated_rmsnorm` in one kernel, over `[rows, heads, channels]` operands.

    `out_dtype` (default `x.dtype`): with an fp32 `x` and a narrower `out_dtype`, the kernel
    first rounds `x` to `out_dtype`, so the result is bit-identical to
    `gated_rmsnorm(x.to(out_dtype), ...)` without the separate cast launch
    (`SEED_DN_NORM_F32_IN`).

    `x` and `gate` need the same shape and a contiguous last axis; their row and head strides
    are free, which is what lets the caller pass a slice of the concatenated input projection
    as the gate without first making it contiguous. The result is fresh and contiguous.
    """
    if x.shape != gate.shape:
        raise ValueError(f"shape mismatch: x {tuple(x.shape)}, gate {tuple(gate.shape)}")
    sx = _rows(x, "x")
    sg = _rows(gate, "gate")
    rows, heads, cols = x.shape
    out_dtype = out_dtype or x.dtype
    out = torch.empty(x.shape, dtype=out_dtype, device=x.device)
    _gated_rmsnorm_kernel[(rows * heads,)](
        x,
        gate,
        w,
        out,
        *sx,
        *sg,
        *out.stride()[:2],
        heads,
        cols,
        eps,
        BLOCK=triton.next_power_of_2(cols),
        ROUND_IN=out_dtype != x.dtype,
    )
    return out


def delta_rule_decode(
    qk: tuple[torch.Tensor, torch.Tensor],
    v: torch.Tensor,
    gate: tuple[torch.Tensor, torch.Tensor],
    rec: torch.Tensor,
    params: tuple[torch.Tensor, torch.Tensor],
    active: torch.Tensor | None = None,
    lanes: torch.Tensor | None = None,
) -> torch.Tensor:
    """One token per slot of the gated delta rule, in one kernel. Advances `rec` in place.

    `lanes` (`[B]` int, optional; `SEED_DN_STATE_INPLACE=1`): `rec` is then the full lane pool
    `[L, v_heads, k_dim, v_dim]` and row `r` reads and writes lane `lanes[r]` directly. Requires
    `active`: rows sharing a lane (graph padding rows share one filler lane) must be inactive,
    so no two stores hit one lane.

    Operands, all with a contiguous last axis:

    | | shape | what it is |
    | --- | --- | --- |
    | `qk` | `[B, k_heads, k_dim]` each | `in_proj_qkv`'s q and k, *before* `repeat_interleave` |
    | `v` | `[B, v_heads, v_dim]` | `in_proj_qkv`'s v |
    | `gate` | `[B, v_heads]` each | raw `in_proj_a` and `in_proj_b` outputs |
    | `rec` | `[B, v_heads, k_dim, v_dim]` fp32 | recurrent state, updated in place |
    | `params` | `[v_heads]` each fp32 | `A_log` and `dt_bias` |
    | `active` | `[B]` int32/bool, optional | 1/True for a row whose `rec` should advance |

    Returns fp32 `[B, v_heads, v_dim]`, which is what `delta_rule_recurrent` returns once its
    singleton time axis is dropped. `v_heads` must be a multiple of `k_heads`; the kernel
    indexes q and k at `head // rep` rather than having the caller widen them.

    `active`, when given (`SEED_FUSE_GLUE=1`'s `graph_decode.deltanet_decode_static`), makes
    the `rec` write itself a no-op for a row whose `active` entry is 0/False -- see the
    kernel's docstring for why that is exact and what it replaces (a `clone()` of the whole
    state, the kernel, then a `torch.where` select back in: 2-3 extra full-state HBM round
    trips this collapses to zero). `None` (every other call site: eager decode only ever runs
    the kernel over rows it means to advance) keeps today's unconditional-write behavior
    exactly, so this is purely additive.
    """
    q, k = qk
    a, b = gate
    a_log, dt_bias = params
    sq, sk = _rows(q, "q"), _rows(k, "k")
    sv = _rows(v, "v")
    rows, v_heads, v_dim = v.shape
    k_dim = q.shape[-1]
    if lanes is not None:
        if active is None or lanes.shape != (rows,):
            raise ValueError("lanes requires active and must be [rows]")
        if rec.shape[1:] != (v_heads, k_dim, v_dim) or rec.stride(-1) != 1:
            raise ValueError(f"rec must be [L, {v_heads}, {k_dim}, {v_dim}] contiguous rows")
    elif rec.shape != (rows, v_heads, k_dim, v_dim) or rec.stride(-1) != 1:
        raise ValueError(f"rec must be [{rows}, {v_heads}, {k_dim}, {v_dim}] with contiguous rows")
    if v_heads % q.shape[1]:
        raise ValueError(f"v_heads {v_heads} is not a multiple of k_heads {q.shape[1]}")
    if active is not None and active.shape != (rows,):
        raise ValueError(f"active must be [{rows}], got {tuple(active.shape)}")

    out = torch.empty((rows, v_heads, v_dim), dtype=torch.float32, device=v.device)
    block_c = min(BLOCK_V, triton.next_power_of_2(v_dim))
    active_arg = active if active is not None else q  # unused dummy pointer when HAS_ACTIVE=False
    _delta_rule_decode_kernel[(rows * v_heads, triton.cdiv(v_dim, block_c))](
        q,
        k,
        v,
        a,
        b,
        rec,
        a_log,
        dt_bias,
        out,
        active_arg,
        lanes if lanes is not None else active_arg,
        *sq,
        *sk,
        *sv,
        a.stride(0),
        b.stride(0),
        *rec.stride()[:3],
        *out.stride()[:2],
        v_heads,
        v_heads // q.shape[1],
        k_dim,
        v_dim,
        k_dim**-0.5,
        HAS_ACTIVE=active is not None,
        BLOCK_K=triton.next_power_of_2(k_dim),
        BLOCK_C=block_c,
        HAS_LANES=lanes is not None,
    )
    return out


def delta_rule_verify_exact(
    qk: tuple[torch.Tensor, torch.Tensor],
    v: torch.Tensor,
    gate: tuple[torch.Tensor, torch.Tensor],
    rec: torch.Tensor,
    params: tuple[torch.Tensor, torch.Tensor],
    active: torch.Tensor | None = None,
    lanes: torch.Tensor | None = None,
    lengths: torch.Tensor | None = None,
    materialize_steps: bool = True,
) -> torch.Tensor:
    """Two to four decode-ordered recurrence steps in one MTP verification launch.

    Inputs add a fixed time axis to `delta_rule_decode`: q/k are `[B, T, k_heads, k_dim]`,
    v is `[B, T, v_heads, v_dim]`, and raw a/b gates are `[B, T, v_heads]`, with compile-time
    `T` in 2..4. The state and optional graph-padding arguments have the same contract as
    `delta_rule_decode`. `lengths` optionally limits each row to its accepted prefix for rollback. `materialize_steps`
    preserves the separate decode kernels' fp32 state boundary. Keeping the width fixed makes
    the loop fully unrolled and graph safe.
    """
    q, k = qk
    a, b = gate
    if q.shape != k.shape or q.dim() != 4 or not 2 <= q.shape[1] <= 4:
        raise ValueError(f"q/k must have matching [B, T, heads, dim], T=2..4; got {q.shape}/{k.shape}")
    rows, width, k_heads, k_dim = q.shape
    if v.dim() != 4 or v.shape[:2] != (rows, width):
        raise ValueError(f"v must be [B, T, heads, dim], got {tuple(v.shape)}")
    v_heads, v_dim = v.shape[2:]
    if a.shape != (rows, width, v_heads) or b.shape != a.shape:
        raise ValueError(f"a/b must be [{rows}, {width}, {v_heads}], got {a.shape}/{b.shape}")
    if any(x.stride(-1) != 1 for x in (q, k, v, a, b)):
        raise ValueError("q/k/v/a/b must be contiguous along their last axis")
    if v_heads % k_heads:
        raise ValueError(f"v_heads {v_heads} is not a multiple of k_heads {k_heads}")
    if lanes is not None:
        if active is None or lanes.shape != (rows,):
            raise ValueError("lanes requires active and must be [rows]")
        if rec.shape[1:] != (v_heads, k_dim, v_dim) or rec.stride(-1) != 1:
            raise ValueError(f"rec must be [L, {v_heads}, {k_dim}, {v_dim}] contiguous rows")
    elif rec.shape != (rows, v_heads, k_dim, v_dim) or rec.stride(-1) != 1:
        raise ValueError(f"rec must be [{rows}, {v_heads}, {k_dim}, {v_dim}] contiguous rows")
    if active is not None and active.shape != (rows,):
        raise ValueError(f"active must be [{rows}], got {tuple(active.shape)}")
    if lengths is not None and lengths.shape != (rows,):
        raise ValueError(f"lengths must be [{rows}], got {tuple(lengths.shape)}")

    a_log, dt_bias = params
    out = torch.empty((rows, width, v_heads, v_dim), dtype=torch.float32, device=v.device)
    block_c = min(BLOCK_V, triton.next_power_of_2(v_dim))
    active_arg = active if active is not None else q
    _delta_rule_verify_exact_kernel[(rows * v_heads, triton.cdiv(v_dim, block_c))](
        q,
        k,
        v,
        a,
        b,
        rec,
        a_log,
        dt_bias,
        out,
        active_arg,
        lanes if lanes is not None else active_arg,
        lengths if lengths is not None else active_arg,
        *q.stride()[:3],
        *k.stride()[:3],
        *v.stride()[:3],
        *a.stride()[:2],
        *b.stride()[:2],
        *rec.stride()[:3],
        *out.stride()[:3],
        v_heads,
        v_heads // k_heads,
        k_dim,
        v_dim,
        k_dim**-0.5,
        HAS_ACTIVE=active is not None,
        BLOCK_K=triton.next_power_of_2(k_dim),
        BLOCK_C=block_c,
        T=width,
        HAS_LANES=lanes is not None,
        HAS_LENGTHS=lengths is not None,
        MATERIALIZE_STEPS=materialize_steps,
    )
    return out


_DN_COUNTERS: dict[tuple[int, int], torch.Tensor] = {}
_DN_OUT_SCRATCH: dict[tuple[int, int], torch.Tensor] = {}
_CNT_STRIDE = 32  # int32s per counter: one 128-byte line each


def dn_decode_fused(
    qkv: torch.Tensor,
    conv: tuple[torch.Tensor, torch.Tensor],
    gate: tuple[torch.Tensor, torch.Tensor],
    rec: torch.Tensor,
    params: tuple[torch.Tensor, torch.Tensor],
    rows: tuple[torch.Tensor, torch.Tensor],
    norm: tuple[torch.Tensor, torch.Tensor, float],
    heads: tuple[int, int, int, int],
) -> torch.Tensor:
    """`gated_rmsnorm(delta_rule_decode(causal_conv_decode(qkv, ...) split into q/k/v, ...),
    z, w, eps, bf16)` as one kernel (`SEED_DN_DECODE_FUSED`), advancing the lane pools' conv
    and recurrent state in place exactly as `causal_conv_decode` and `delta_rule_decode(...,
    active=, lanes=)` do.

    `qkv` `[B, C]` or `[B, 1, C]` bf16 (last axis contiguous), `conv` = (weight `[C, 1, K]`,
    state pool `[L, C, K - 1]`), `gate` = raw (`a`, `b`) `[B, v_heads]`, `rec` the `[L,
    v_heads, k_dim, v_dim]` fp32 pool, `params` = (`A_log`, `dt_bias`), `rows` = (`active`,
    `lanes`) `[B]`, `norm` = (`z` `[B, v_heads, v_dim]`, `dn_norm` weight, eps), `heads` =
    (`k_heads`, `v_heads`, `k_dim`, `v_dim`). Returns bf16 `[B, v_heads, v_dim]`."""
    if qkv.dim() == 3:
        qkv = qkv.reshape(qkv.shape[0], qkv.shape[2])
    cw, cs = conv
    a, b = gate
    a_log, dt_bias = params
    active, lanes = rows
    z, nw, eps = norm
    k_heads, v_heads, k_dim, v_dim = heads
    bsz = qkv.shape[0]
    if qkv.stride(-1) != 1 or z.stride(-1) != 1 or rec.stride(-1) != 1 or cs.stride(-1) != 1:
        raise ValueError("dn_decode_fused: last axes must be contiguous")
    block_c = min(BLOCK_V, triton.next_power_of_2(v_dim))
    nvb = triton.cdiv(v_dim, block_c)
    key = (qkv.device.index or 0, torch.cuda.current_stream(qkv.device).cuda_stream)
    cnt = _DN_COUNTERS.get(key)
    need = bsz * (k_heads + v_heads) * _CNT_STRIDE
    if cnt is None or cnt.numel() < need:
        cnt = _DN_COUNTERS[key] = torch.zeros(
            max(need, 64 * 20 * _CNT_STRIDE), dtype=torch.int32, device=qkv.device
        )
    o32 = _DN_OUT_SCRATCH.get(key)
    if o32 is None or o32.numel() < bsz * v_heads * v_dim:
        o32 = torch.zeros(max(bsz, 64) * v_heads * v_dim, dtype=torch.int32, device=qkv.device)
        _DN_OUT_SCRATCH[key] = o32
    o32 = o32[: bsz * v_heads * v_dim].view(bsz, v_heads, v_dim)
    out = torch.empty((bsz, v_heads, v_dim), dtype=qkv.dtype, device=qkv.device)
    _dn_decode_fused_kernel[(bsz * v_heads, nvb)](
        qkv,
        cw,
        cs,
        a,
        b,
        rec,
        a_log,
        dt_bias,
        z,
        nw,
        o32,
        out,
        active,
        lanes,
        cnt,
        qkv.stride(0),
        cw.stride(0),
        cw.stride(2),
        *cs.stride()[:3],
        a.stride(0),
        b.stride(0),
        *rec.stride()[:3],
        *z.stride()[:2],
        *out.stride()[:2],
        v_heads,
        v_heads // k_heads,
        k_heads,
        k_dim,
        v_dim,
        k_dim**-0.5,
        eps,
        CONV_K=cw.shape[-1],
        BLOCK_K=triton.next_power_of_2(k_dim),
        BLOCK_C=block_c,
        NVB=nvb,
        NORM_BLOCK=triton.next_power_of_2(v_dim),
        CNT_STRIDE=_CNT_STRIDE,
    )
    return out


def masked_row_copy(dst: torch.Tensor, src: torch.Tensor, active: torch.Tensor) -> None:
    """`dst[row] = src[row]` for every row `active` marks, in place; every other row untouched.

    `dst`/`src` are `[rows, channels, width]`, `width` (the last axis) contiguous (stride 1)
    in both but otherwise independently strided -- `causal_conv_static` passes a suffix slice
    of `torch.cat([state, x])` as `src` (whose channel stride is one wider than `width`, from
    the dropped oldest column) and the plain contiguous state buffer as `dst`, so this never
    materializes a copy of either side just to make them flat-reshapable. `active` is
    `[rows]`, bool or int.
    """
    if dst.shape != src.shape:
        raise ValueError(f"shape mismatch: dst {tuple(dst.shape)}, src {tuple(src.shape)}")
    if dst.dim() != 3:
        raise ValueError(f"dst/src must be [rows, channels, width], got {tuple(dst.shape)}")
    if dst.stride(-1) != 1 or src.stride(-1) != 1:
        raise ValueError("dst/src must be contiguous along their last (width) axis")
    if active.shape != (dst.shape[0],):
        raise ValueError(f"active must be [{dst.shape[0]}], got {tuple(active.shape)}")
    rows, channels, width = dst.shape
    block = min(1024, triton.next_power_of_2(channels))
    _masked_row_copy_kernel[(rows, triton.cdiv(channels, block))](
        src,
        dst,
        active,
        src.stride(0),
        src.stride(1),
        dst.stride(0),
        dst.stride(1),
        channels,
        width=width,
        BLOCK=block,
    )


def causal_conv_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    state_pool: torch.Tensor,
    lanes: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """`graph_decode.causal_conv_static` in one kernel (`SEED_DN_CONV_INPLACE=1`).

    Replaces the gather (`st["conv"][rows]`) -> `torch.cat` -> masked state copy ->
    `F.conv1d` (MIOpen) -> `F.silu` -> scatter (`st["conv"][rows] = ...`) chain, plus the
    `.transpose(1, 2)` on the way in and the `[:, :, 0]` slice on the way out, with one launch
    that reads and advances `state_pool[lanes[row]]` in place -- the same trick
    `delta_rule_decode`'s `lanes` argument (`SEED_DN_STATE_INPLACE`) already applies to the
    DeltaNet recurrent state. See the kernel's docstring for the read-before-write ordering
    that makes the in-place shift safe and for why a padding row's shared filler lane is never
    written.

    Idea: a fused per-token shortconv-decode step (windowed dot product per channel, one
    program per row) is the same shape the causal_conv1d "step" kernels published for
    Mamba/GatedDeltaNet-style architectures use -- cited for the shape only, no code taken.

    `x` is `[B, 1, C]` or `[B, C]` (row-major, `qkv` straight off `F.linear`, no transpose:
    a depthwise conv's per-channel window is a sum over `conv_k` scalars, not a matmul).
    `weight` is `[C, 1, conv_k]`, `F.conv1d`'s own layout, unchanged. `state_pool` is the whole
    `[L, C, conv_k - 1]` lane pool (`model.py`'s `pool[i]["conv"]`); `lanes` (`[B]` int) is this
    row's lane (`buf.slot_rows`), `active` (`[B]`) gates the state write only -- the conv output
    is computed for every row regardless, matching `causal_conv_static`. Returns `[B, C]`
    contiguous: `deltanet_decode_static`'s `q`/`k`/`v` slices of it are views, unlike the old
    path's `mixed[:, :, 0]` slice of a `[B, C, 1]` `F.conv1d` output.

    Precision: `F.conv1d` on bf16 inputs runs its MIOpen reduction with fp32 accumulation and
    rounds once at the output (the standard ROCm/cuDNN behavior for reduced-precision conv
    operands); this kernel does the same -- state/weight/x loaded and multiplied in fp32,
    summed in fp32, rounded once at the `SiLU`'d output and once at each stored state tap. The
    two are not bit-identical (different tap grouping/reduction order, the same caveat every
    other kernel in this module carries for `tl.sum` vs. `torch`/MIOpen reductions), so
    `seed_tests/test_deltanet_fused.py` holds the bf16 case to `BF16_TOL` like the others.
    """
    if x.dim() == 3:
        if x.shape[1] != 1:
            raise ValueError(f"x must be [B, 1, C] or [B, C], got {tuple(x.shape)}")
        x = x.reshape(x.shape[0], x.shape[2])
    elif x.dim() != 2:
        raise ValueError(f"x must be [B, 1, C] or [B, C], got {tuple(x.shape)}")
    if x.stride(-1) != 1:
        raise ValueError("x must be contiguous along its last (channel) axis")
    b, channels = x.shape
    if weight.dim() != 3 or weight.shape[0] != channels:
        raise ValueError(f"weight must be [{channels}, 1, conv_k], got {tuple(weight.shape)}")
    conv_k = weight.shape[-1]
    if state_pool.dim() != 3 or state_pool.shape[1:] != (channels, conv_k - 1):
        raise ValueError(
            f"state_pool must be [L, {channels}, {conv_k - 1}], got {tuple(state_pool.shape)}"
        )
    if state_pool.stride(-1) != 1:
        raise ValueError("state_pool must be contiguous along its last axis")
    if lanes.shape != (b,) or active.shape != (b,):
        raise ValueError(
            f"lanes/active must be [{b}], got {tuple(lanes.shape)}/{tuple(active.shape)}"
        )

    out = torch.empty((b, channels), dtype=x.dtype, device=x.device)
    block = min(1024, triton.next_power_of_2(channels))
    _causal_conv_decode_kernel[(b, triton.cdiv(channels, block))](
        x,
        weight,
        state_pool,
        lanes,
        active,
        out,
        x.stride(0),
        weight.stride(0),
        weight.stride(2),
        *state_pool.stride()[:3],
        out.stride(0),
        channels,
        conv_k=conv_k,
        BLOCK=block,
    )
    return out


def causal_conv_verify_exact(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    *,
    lengths: torch.Tensor | None = None,
    active: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run two to four causal-conv decode steps in one launch.

    The fixed-width loop uses the same tap order, bf16 output rounding, and state
    store boundary as :func:`causal_conv_decode`. ``lengths`` limits each row's
    persistent-state advance for accepted-prefix rollback. ``active`` prevents
    graph padding rows from modifying their state.
    """
    if x.dim() != 3 or not 2 <= x.shape[1] <= 4:
        raise ValueError(f"x must be [B, T, C] with T=2..4, got {tuple(x.shape)}")
    rows, width, channels = x.shape
    if x.stride(-1) != 1:
        raise ValueError("x must be contiguous along its last (channel) axis")
    if weight.dim() != 3 or weight.shape[0] != channels:
        raise ValueError(f"weight must be [{channels}, 1, conv_k], got {tuple(weight.shape)}")
    conv_k = weight.shape[-1]
    if state.shape != (rows, channels, conv_k - 1) or state.stride(-1) != 1:
        raise ValueError(
            f"state must be [{rows}, {channels}, {conv_k - 1}] contiguous rows, "
            f"got {tuple(state.shape)}"
        )
    if lengths is not None and lengths.shape != (rows,):
        raise ValueError(f"lengths must be [{rows}], got {tuple(lengths.shape)}")
    if active is not None and active.shape != (rows,):
        raise ValueError(f"active must be [{rows}], got {tuple(active.shape)}")

    out = torch.empty_like(x)
    block = min(1024, triton.next_power_of_2(channels))
    placeholder = x
    _causal_conv_verify_exact_kernel[(rows, triton.cdiv(channels, block))](
        x,
        weight,
        state,
        lengths if lengths is not None else placeholder,
        active if active is not None else placeholder,
        out,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(2),
        *state.stride()[:3],
        out.stride(0),
        out.stride(1),
        channels,
        conv_k=conv_k,
        T=width,
        HAS_LENGTHS=lengths is not None,
        HAS_ACTIVE=active is not None,
        BLOCK=block,
    )
    return out


def _rows2(x: torch.Tensor, name: str) -> int:
    """`stride(0)` of a `[rows, heads]` operand, contiguous along `heads`."""
    if x.dim() != 2:
        raise ValueError(f"{name} must be [rows, heads], got {tuple(x.shape)}")
    if x.stride(-1) != 1:
        raise ValueError(f"{name} must be contiguous along its last axis")
    return x.stride(0)


def fused_recurrent_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    *,
    prefetch: bool | None = None,
    num_warps: int | None = None,
    block_v: int | None = None,
    chunked: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The whole gated-delta-rule prefill loop (every `t`), in one kernel. One `t`-loop per
    `(sequence, value head, v-block)` program, the state tile held in registers throughout.

    Same math and precision as `model.delta_rule_recurrent`'s `for s in range(T)` loop and the
    same per-step formulation `delta_rule_decode` uses, just run for every position inside the
    kernel instead of once per Python-level call: decay the state by `g`'s already-computed
    log-decay, then the delta update, then the query read-out, all in fp32.

    Operands, all with a contiguous last axis:

    | | shape | what it is |
    | --- | --- | --- |
    | `q`, `k` | `[T, heads, k_dim]` | raw `q`/`k` (*after* `repeat_interleave` onto value heads, *before* l2-norm) |
    | `v` | `[T, heads, v_dim]` | raw `v` |
    | `g` | `[T, heads]` fp32 | per-token log-decay, `<= 0` (`model.py`'s `-A_log.exp() * softplus(...)`) |
    | `beta` | `[T, heads]` | per-token gate, already `sigmoid`'d |
    | `initial_state` | `[n_seqs, heads, k_dim, v_dim]` fp32 | recurrent state; updated in place |
    | `cu_seqlens` | `[n_seqs + 1]` int32, optional | packed varlen row offsets into `T` |

    `T` is the packed length across all sequences (`cu_seqlens[-1]`); omitting `cu_seqlens`
    means one sequence of the whole length, i.e. `cu_seqlens = [0, T]`, which is what every
    call site needs today (`Model.delta_rule` prefills one sequence per call; the varlen
    batched-prefill path still loops one sequence at a time, see `deltanet_tp.py`) -- the grid
    already has a sequence axis so a caller with real packed batches does not need a new entry
    point, only to build `cu_seqlens` and a `[n_seqs, ...]` `initial_state`.

    Keyword overrides (tests, `bench_prefill_moe.py`): `prefetch`, `num_warps` and `block_v`
    default to `SEED_DELTANET_PREFILL_PREFETCH` / `_WARPS` / `_BLOCK_V`. `chunked=True` hands the whole
    call to the chunked (WY) kernels (`deltanet_prefill_chunked`); only the model's prefill
    call sites pass it, via `deltanet_prefill_chunked.use_chunked`, never the captured MTP
    verify path (many 2-3 row sequences, and the chunk table is built on the host).

    Returns `(out, final_state)`: `out` is fp32 `[T, heads, v_dim]`, and `final_state` is
    `initial_state` itself (same storage), so either name can be used after the call.
    """
    sq, sk, sv = _rows(q, "q"), _rows(k, "k"), _rows(v, "v")
    rows, heads, v_dim = v.shape
    k_dim = q.shape[-1]
    if q.shape[:2] != (rows, heads) or k.shape[:2] != (rows, heads):
        raise ValueError("q, k, v must share [T, heads]")
    sg, sbeta = _rows2(g, "g"), _rows2(beta, "beta")
    if g.shape != (rows, heads) or beta.shape != (rows, heads):
        raise ValueError(f"g, beta must be [{rows}, {heads}]")
    if initial_state.dim() != 4 or initial_state.shape[1:] != (heads, k_dim, v_dim):
        raise ValueError(f"initial_state must be [n_seqs, {heads}, {k_dim}, {v_dim}]")
    if initial_state.stride(-1) != 1:
        raise ValueError("initial_state must be contiguous along its last axis")
    n_seqs = initial_state.shape[0]
    if cu_seqlens is None:
        if n_seqs != 1:
            raise ValueError(
                "cu_seqlens is required when initial_state holds more than one sequence"
            )
        cu_seqlens = torch.tensor([0, rows], dtype=torch.int32, device=q.device)
    elif cu_seqlens.shape != (n_seqs + 1,):
        raise ValueError(f"cu_seqlens must be [{n_seqs + 1}]")

    if chunked:
        # SEED_DELTANET_PREFILL_CHUNKED, opted into by the prefill call sites only
        # (`deltanet_prefill_chunked.use_chunked`): the chunked (WY) form, same contract.
        return deltanet_prefill_chunked.chunked_prefill(q, k, v, g, beta, initial_state, cu_seqlens)
    out = torch.empty((rows, heads, v_dim), dtype=torch.float32, device=v.device)
    block_c = min(block_v or BLOCK_V_PREFILL, triton.next_power_of_2(v_dim))
    _fused_recurrent_prefill_kernel[(n_seqs * heads, triton.cdiv(v_dim, block_c))](
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
        out,
        *sq,
        *sk,
        *sv,
        sg,
        sbeta,
        *initial_state.stride()[:3],
        *out.stride()[:2],
        heads,
        k_dim,
        v_dim,
        k_dim**-0.5,
        BLOCK_K=triton.next_power_of_2(k_dim),
        BLOCK_C=block_c,
        PREFETCH=PREFILL_PREFETCH if prefetch is None else prefetch,
        num_warps=num_warps or PREFILL_WARPS,
    )
    return out, initial_state

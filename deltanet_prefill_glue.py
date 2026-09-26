"""One kernel for the graph-prefill DeltaNet glue (`SEED_DN_PREFILL_GLUE_FUSED=1`, default off).

What it replaces, per DeltaNet layer of every captured prefill and mixed step
(`graph_prefill.deltanet_prefill_core`, `[rows, width]` layout): the `cat` of the gathered
conv window with `qkv`, the depthwise `conv1d` (MIOpen's naive kernel) and its `silu`, the
conv-window `gather`/`where`/scatter back into the pool, the `q`/`k`/`v` split and `v`'s
contiguous copy, `beta = where(real, sigmoid(b), 0)`, `g = where(real, -exp(A_log) *
softplus(a + dt_bias), 0)` (five kernels), and the two `repeat_interleave`s that widen
`q`/`k` from the key heads to the value heads: about 18 launches per layer, each a pass over
a `[rows * width, <=3072]` tensor, down to one launch plus the conv-window gather. The
chunked kernels then read `q`/`k` at their key-head width (`deltanet_prefill_chunked`
takes key-head `q`/`k`), so the widened copies are never written.

Numerics match the torch sequence: the conv accumulates in fp32 (MIOpen's naive kernel in
double; both are exact to well below a bf16 ulp for a 4-tap filter) and is rounded to bf16
before the `silu`, which is computed in fp32 and rounded again, as `F.silu` on a bf16 tensor
does. `beta` is bf16 (sigmoid in fp32, rounded), `g` fp32.

Conv window state: the kernel reads the window from a gathered copy (`pool["conv"][lanes]`),
never from the pool it writes, so there is no read-after-write hazard between programs.
Rows with `active == 0` (padding rows) write nothing, which is what `torch.where(active,
new, pre)` scattered back amounts to.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

ENABLED = os.environ.get("SEED_DN_PREFILL_GLUE_FUSED", "0") not in ("0", "", "false", "False")

BT = 32
"""Tokens per program."""

BC = 128
"""Conv channels per program; must divide the key width (q and k blocks never straddle)."""


def available(device: torch.device) -> bool:
    return ENABLED and HAVE_TRITON and device.type == "cuda"


if HAVE_TRITON:

    @triton.jit
    def _dn_glue_kernel(
        proj_ptr,
        s_pr,
        s_pt,
        conv_w_ptr,
        pre_ptr,
        pool_ptr,
        lanes_ptr,
        length_ptr,
        active_ptr,
        real_ptr,
        s_rr,
        a_log_ptr,
        dt_bias_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        beta_ptr,
        g_ptr,
        T,
        C: tl.constexpr,
        KEYD: tl.constexpr,
        OFF_B: tl.constexpr,
        OFF_A: tl.constexpr,
        H: tl.constexpr,
        KC: tl.constexpr,
        BT: tl.constexpr,
        BC: tl.constexpr,
    ):
        r = tl.program_id(0)
        pid_t = tl.program_id(1)
        pid_c = tl.program_id(2)
        tok = pid_t * BT + tl.arange(0, BT)
        ch = pid_c * BC + tl.arange(0, BC)
        tok_ok = tok < T
        W: tl.constexpr = KC - 1
        prow = proj_ptr + r.to(tl.int64) * s_pr
        pre = pre_ptr + r.to(tl.int64) * (C * W)

        acc = tl.zeros([BT, BC], tl.float32)
        for d in tl.static_range(KC):
            src = tok + d - W  # position in this row's qkv; < 0 reads the carried window
            x_new = tl.load(
                prow + src[:, None] * s_pt + ch[None, :],
                mask=(src[:, None] >= 0) & tok_ok[:, None],
                other=0.0,
            )
            x_old = tl.load(
                pre + ch[None, :] * W + (src[:, None] + W),
                mask=(src[:, None] < 0) & tok_ok[:, None],
                other=0.0,
            )
            wv = tl.load(conv_w_ptr + ch * KC + d).to(tl.float32)
            acc += (x_new.to(tl.float32) + x_old.to(tl.float32)) * wv[None, :]
        y = acc.to(tl.bfloat16).to(tl.float32)
        y = (y / (1.0 + tl.exp(-y))).to(tl.bfloat16)

        orow = (r * T + tok).to(tl.int64)
        m2 = tok_ok[:, None]
        if pid_c * BC < KEYD:
            tl.store(q_ptr + orow[:, None] * KEYD + ch[None, :], y, mask=m2)
        elif pid_c * BC < 2 * KEYD:
            tl.store(k_ptr + orow[:, None] * KEYD + (ch[None, :] - KEYD), y, mask=m2)
        else:
            VD: tl.constexpr = C - 2 * KEYD
            tl.store(v_ptr + orow[:, None] * VD + (ch[None, :] - 2 * KEYD), y, mask=m2)

        if pid_c == 0:
            h = tl.arange(0, H)
            real = tl.load(real_ptr + r * s_rr + tok, mask=tok_ok, other=0) != 0
            b = tl.load(prow + tok[:, None] * s_pt + OFF_B + h[None, :], mask=m2, other=0.0)
            beta = 1.0 / (1.0 + tl.exp(-b.to(tl.float32)))
            beta = tl.where(real[:, None], beta, 0.0).to(tl.bfloat16)
            tl.store(beta_ptr + orow[:, None] * H + h[None, :], beta, mask=m2)
            a = tl.load(prow + tok[:, None] * s_pt + OFF_A + h[None, :], mask=m2, other=0.0)
            xa = a.to(tl.float32) + tl.load(dt_bias_ptr + h)[None, :]
            sp = tl.where(xa > 20.0, xa, tl.log(1.0 + tl.exp(tl.minimum(xa, 20.0))))
            g = -tl.exp(tl.load(a_log_ptr + h))[None, :] * sp
            g = tl.where(real[:, None], g, 0.0)
            tl.store(g_ptr + orow[:, None] * H + h[None, :], g, mask=m2)

        if pid_t == 0:
            if tl.load(active_ptr + r) != 0:
                lane = tl.load(lanes_ptr + r).to(tl.int64)
                n = tl.load(length_ptr + r)
                for i in tl.static_range(W):
                    pos = n + i  # index into cat(window, qkv)
                    from_new = tl.load(
                        prow + (pos - W) * s_pt + ch, mask=(pos >= W) & (ch < C), other=0.0
                    )
                    from_old = tl.load(pre + ch * W + pos, mask=(pos < W) & (ch < C), other=0.0)
                    val = tl.where(pos >= W, from_new, from_old)
                    tl.store(pool_ptr + lane * (C * W) + ch * W + i, val)


    @triton.jit
    def _dn_glue_packed_kernel(
        proj_ptr,
        s_pt,
        conv_w_ptr,
        pre_ptr,
        tok_seg_ptr,
        tok_off_ptr,
        real_ptr,
        a_log_ptr,
        dt_bias_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        beta_ptr,
        g_ptr,
        T,
        C: tl.constexpr,
        KEYD: tl.constexpr,
        OFF_B: tl.constexpr,
        OFF_A: tl.constexpr,
        H: tl.constexpr,
        KC: tl.constexpr,
        BT: tl.constexpr,
        BC: tl.constexpr,
    ):
        """`_dn_glue_kernel`'s per-token work on a packed stream of `T` tokens: token `tok`
        belongs to segment `tok_seg[tok]` at offset `tok_off[tok]` inside it, and its conv
        taps before the segment start read that segment's carried window (`pre[seg]`) instead
        of the previous segment's tokens. Same arithmetic and rounding points per token."""
        pid_t = tl.program_id(0)
        pid_c = tl.program_id(1)
        tok = pid_t * BT + tl.arange(0, BT)
        ch = pid_c * BC + tl.arange(0, BC)
        tok_ok = tok < T
        W: tl.constexpr = KC - 1
        seg = tl.load(tok_seg_ptr + tok, mask=tok_ok, other=0).to(tl.int64)
        off = tl.load(tok_off_ptr + tok, mask=tok_ok, other=0)
        pre = pre_ptr + seg * (C * W)

        acc = tl.zeros([BT, BC], tl.float32)
        for d in tl.static_range(KC):
            rel = off + d - W  # position in this segment; < 0 reads its carried window
            src = tok + d - W
            x_new = tl.load(
                proj_ptr + src[:, None] * s_pt + ch[None, :],
                mask=((rel >= 0) & (src >= 0) & tok_ok)[:, None],
                other=0.0,
            )
            x_old = tl.load(
                pre[:, None] + ch[None, :] * W + (rel[:, None] + W),
                mask=((rel < 0) & tok_ok)[:, None],
                other=0.0,
            )
            wv = tl.load(conv_w_ptr + ch * KC + d).to(tl.float32)
            acc += (x_new.to(tl.float32) + x_old.to(tl.float32)) * wv[None, :]
        y = acc.to(tl.bfloat16).to(tl.float32)
        y = (y / (1.0 + tl.exp(-y))).to(tl.bfloat16)

        orow = tok.to(tl.int64)
        m2 = tok_ok[:, None]
        if pid_c * BC < KEYD:
            tl.store(q_ptr + orow[:, None] * KEYD + ch[None, :], y, mask=m2)
        elif pid_c * BC < 2 * KEYD:
            tl.store(k_ptr + orow[:, None] * KEYD + (ch[None, :] - KEYD), y, mask=m2)
        else:
            VD: tl.constexpr = C - 2 * KEYD
            tl.store(v_ptr + orow[:, None] * VD + (ch[None, :] - 2 * KEYD), y, mask=m2)

        if pid_c == 0:
            h = tl.arange(0, H)
            real = tl.load(real_ptr + tok, mask=tok_ok, other=0) != 0
            b = tl.load(proj_ptr + tok[:, None] * s_pt + OFF_B + h[None, :], mask=m2, other=0.0)
            beta = 1.0 / (1.0 + tl.exp(-b.to(tl.float32)))
            beta = tl.where(real[:, None], beta, 0.0).to(tl.bfloat16)
            tl.store(beta_ptr + orow[:, None] * H + h[None, :], beta, mask=m2)
            a = tl.load(proj_ptr + tok[:, None] * s_pt + OFF_A + h[None, :], mask=m2, other=0.0)
            xa = a.to(tl.float32) + tl.load(dt_bias_ptr + h)[None, :]
            sp = tl.where(xa > 20.0, xa, tl.log(1.0 + tl.exp(tl.minimum(xa, 20.0))))
            g = -tl.exp(tl.load(a_log_ptr + h))[None, :] * sp
            g = tl.where(real[:, None], g, 0.0)
            tl.store(g_ptr + orow[:, None] * H + h[None, :], g, mask=m2)

    @triton.jit
    def _dn_window_packed_kernel(
        proj_ptr,
        s_pt,
        pre_ptr,
        pool_ptr,
        lanes_ptr,
        seg_start_ptr,
        seg_len_ptr,
        active_ptr,
        C: tl.constexpr,
        KC: tl.constexpr,
        BC: tl.constexpr,
    ):
        """Segment `s`'s new conv window, the last `KC - 1` inputs of `cat(pre[s], its
        tokens)`, written to its lane when active (the per-row kernel's `pid_t == 0` tail)."""
        s = tl.program_id(0)
        ch = tl.program_id(1) * BC + tl.arange(0, BC)
        W: tl.constexpr = KC - 1
        if tl.load(active_ptr + s) != 0:
            lane = tl.load(lanes_ptr + s).to(tl.int64)
            n = tl.load(seg_len_ptr + s)
            row0 = tl.load(seg_start_ptr + s).to(tl.int64)
            pre = pre_ptr + s.to(tl.int64) * (C * W)
            for i in tl.static_range(W):
                pos = n + i  # index into cat(window, segment tokens)
                from_new = tl.load(
                    proj_ptr + (row0 + pos - W) * s_pt + ch, mask=(pos >= W) & (ch < C), other=0.0
                )
                from_old = tl.load(pre + ch * W + pos, mask=(pos < W) & (ch < C), other=0.0)
                val = tl.where(pos >= W, from_new, from_old)
                tl.store(pool_ptr + lane * (C * W) + ch * W + i, val)


def prefill_glue(
    proj: torch.Tensor,
    conv_w: torch.Tensor,
    conv_pool: torch.Tensor,
    lanes: torch.Tensor,
    length: torch.Tensor,
    active: torch.Tensor,
    real: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    key_dim: int,
    heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`proj` `[r, t, P]` (`in_proj_all`'s output; last axis contiguous) -> `(q, k, v, beta, g)`
    as `[r*t, key_dim]`, `[r*t, key_dim]`, `[r*t, C - 2*key_dim]` bf16, `[r*t, heads]` bf16
    and fp32, and this step's conv window written into `conv_pool` at each active row's lane.
    `conv_w` `[C, 1, K]`, `conv_pool` `[lanes, C, K-1]`, `real` `[r, t]` bool."""
    r, t, p = proj.shape
    c, _, kc = conv_w.shape
    if proj.stride(-1) != 1:
        raise ValueError("proj must be contiguous along its last axis")
    if key_dim % BC or (c - 2 * key_dim) % BC or heads & (heads - 1):
        raise ValueError("prefill_glue: channel blocks must not straddle q/k/v")
    pre = conv_pool[lanes]  # gathered copy: the kernel reads this, writes the pool
    dev = proj.device
    q = torch.empty(r * t, key_dim, dtype=proj.dtype, device=dev)
    k = torch.empty(r * t, key_dim, dtype=proj.dtype, device=dev)
    v = torch.empty(r * t, c - 2 * key_dim, dtype=proj.dtype, device=dev)
    beta = torch.empty(r * t, heads, dtype=proj.dtype, device=dev)
    g = torch.empty(r * t, heads, dtype=torch.float32, device=dev)
    real_i = real.view(torch.int8) if real.dtype == torch.bool else real  # free reinterpret
    active_i = active.view(torch.int8) if active.dtype == torch.bool else active
    if not (conv_w.is_contiguous() and conv_pool.is_contiguous() and real_i.stride(-1) == 1):
        raise ValueError("prefill_glue: conv weight, conv pool and real must be contiguous")
    off_b = c + (c - 2 * key_dim)  # after qkv and z (z is v-width)
    _dn_glue_kernel[(r, triton.cdiv(t, BT), c // BC)](
        proj,
        proj.stride(0),
        proj.stride(1),
        conv_w,
        pre,
        conv_pool,
        lanes,
        length,
        active_i,
        real_i,
        real_i.stride(0),
        a_log,
        dt_bias,
        q,
        k,
        v,
        beta,
        g,
        t,
        C=c,
        KEYD=key_dim,
        OFF_B=off_b,
        OFF_A=off_b + heads,
        H=heads,
        KC=kc,
        BT=BT,
        BC=BC,
        num_warps=4,
    )
    return q, k, v, beta, g


def prefill_glue_packed(
    proj: torch.Tensor,
    conv_w: torch.Tensor,
    conv_pool: torch.Tensor,
    seg_lanes: torch.Tensor,
    seg_start: torch.Tensor,
    seg_len: torch.Tensor,
    seg_active: torch.Tensor,
    tok_seg: torch.Tensor,
    tok_off: torch.Tensor,
    real: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    key_dim: int,
    heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`prefill_glue` for a packed stream (`SEED_PREFILL_PACK`): `proj` `[T, P]` (or any shape
    flattening to it) holds several segments; per token `tok_seg`/`tok_off` (int32 `[T]`) give
    its segment slot and offset; per slot `seg_lanes`, `seg_start` (stream index of its first
    token), `seg_len`, `seg_active`. Outputs are `[T, ...]` as in `prefill_glue`, and each
    active segment's window is written to its lane. Tokens outside every segment (`tok_off`
    large, `real` false) read only the stream and produce finite, unused values."""
    p = proj.shape[-1]
    proj = proj.reshape(-1, p)
    t = proj.shape[0]
    c, _, kc = conv_w.shape
    if proj.stride(-1) != 1:
        raise ValueError("proj must be contiguous along its last axis")
    if key_dim % BC or (c - 2 * key_dim) % BC or heads & (heads - 1):
        raise ValueError("prefill_glue: channel blocks must not straddle q/k/v")
    pre = conv_pool[seg_lanes]  # gathered copy: the kernels read this, write the pool
    dev = proj.device
    q = torch.empty(t, key_dim, dtype=proj.dtype, device=dev)
    k = torch.empty(t, key_dim, dtype=proj.dtype, device=dev)
    v = torch.empty(t, c - 2 * key_dim, dtype=proj.dtype, device=dev)
    beta = torch.empty(t, heads, dtype=proj.dtype, device=dev)
    g = torch.empty(t, heads, dtype=torch.float32, device=dev)
    real_i = real.reshape(-1)
    real_i = real_i.view(torch.int8) if real_i.dtype == torch.bool else real_i
    active_i = seg_active.view(torch.int8) if seg_active.dtype == torch.bool else seg_active
    if not (conv_w.is_contiguous() and conv_pool.is_contiguous() and real_i.is_contiguous()):
        raise ValueError("prefill_glue: conv weight, conv pool and real must be contiguous")
    off_b = c + (c - 2 * key_dim)
    _dn_glue_packed_kernel[(triton.cdiv(t, BT), c // BC)](
        proj,
        proj.stride(0),
        conv_w,
        pre,
        tok_seg,
        tok_off,
        real_i,
        a_log,
        dt_bias,
        q,
        k,
        v,
        beta,
        g,
        t,
        C=c,
        KEYD=key_dim,
        OFF_B=off_b,
        OFF_A=off_b + heads,
        H=heads,
        KC=kc,
        BT=BT,
        BC=BC,
        num_warps=4,
    )
    _dn_window_packed_kernel[(seg_lanes.shape[0], triton.cdiv(c, BC))](
        proj,
        proj.stride(0),
        pre,
        conv_pool,
        seg_lanes,
        seg_start,
        seg_len,
        active_i,
        C=c,
        KC=kc,
        BC=BC,
        num_warps=4,
    )
    return q, k, v, beta, g

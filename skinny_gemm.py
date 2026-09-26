"""`SEED_SKINNY_GEMM`: a weight-streaming bf16 GEMM for the decode step's skinny dense layers.

Every dense projection in a decode step is `[M, K] @ [N, K]^T` with M = batch <= 64 and
K, N in the thousands, so its cost is the weight bytes (`2 * N * K`) over HBM bandwidth, not
FLOPs. hipBLASLt's TunableOp-selected Tensile solutions reach 0.4-2.2 TB/s on these shapes
(measured r5 profile, b48) against MI300A's ~5.3 TB/s peak; the tiles are sized for M.

This kernel is the standard split-K GEMV/skinny-GEMM shape (the idea is Stream-K / split-K
with a serial fixup: Osama et al., "Stream-K: Work-centric Parallel Decomposition for Dense
Matrix-Matrix Multiplication on the GPU", PPoPP 2023; and the "last block reduces" pattern of
CUDA's threadFenceReduction sample): the grid is `(N / BLOCK_N, SPLIT_K)`, each program
streams one `[BLOCK_N, K / SPLIT_K]` slab of the weight exactly once, multiplies it against
the whole (padded) activation block with bf16 MFMA, and, when `SPLIT_K > 1`, writes an fp32
partial; the last program to finish an N tile (an atomic arrival counter) sums the partials
in fixed `split` order and stores bf16, then resets the counter so the next launch (and a
graph replay) starts at zero. The fixed summation order makes the result deterministic run to
run. It is not bit-identical to hipBLASLt (a different fp32 reduction order), so callers that
need bit-exact logits leave the flag off.

Two optional activation prologues fold the elementwise op that produces the GEMM input into
the load of that input, removing its kernel(s) (`SEED_SKINNY_GEMM` glue folding):

- `ACT_SWIGLU`: input is `silu(g) * u` with `g, u` the two halves of the shared expert's
  gate_up output, rounded to bf16 at the same two points `swiglu_mlp`'s torch ops round.
- `ACT_SIGMOID_GATE`: input is `out * sigmoid(gate)` (attention's output gate), with `gate`
  read through a per-head stride so the `.chunk()` view needs no `.reshape` copy.
"""

from __future__ import annotations

import os
from typing import Literal

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - exercised only where triton is absent
    HAVE_TRITON = False

ENABLED = os.environ.get("SEED_SKINNY_GEMM", "0") not in ("0", "false", "False")
"""Route every eligible decode `linear` here. Off: measured slower than TunableOp's hipBLASLt
on most shapes (see `scratchpad/skinny_bench.py` results in the module docstring)."""
SWIGLU = os.environ.get("SEED_SKINNY_SWIGLU", "0") not in ("0", "false", "False")
"""Run only the shared expert's down projection here, with `silu(gate) * up` folded into its
input load (`swiglu_mlp`). Removes the `silu` and `mul` launches (2 a layer); the down GEMM
itself (K = 256, 2 MB) runs at hipBLASLt's speed on this kernel."""
MAX_M = 64
"""Largest M the kernel takes; wider calls (prefill) keep `F.linear`."""

Act = Literal["none", "swiglu", "sigmoid_gate"]
_ACT_CODE = {"none": 0, "swiglu": 1, "sigmoid_gate": 2}


def available(x: torch.Tensor, w: torch.Tensor, force: bool = False) -> bool:
    """True when `skinny_linear` may replace `F.linear(x, w)`. `force` skips the
    `SEED_SKINNY_GEMM` check (a caller gated on its own flag)."""
    return (
        (ENABLED or force)
        and HAVE_TRITON
        and x.device.type == "cuda"
        and x.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
        and w.dim() == 2
        and w.stride(1) == 1
        and x.numel() // max(x.shape[-1], 1) <= MAX_M
    )


if HAVE_TRITON:

    @triton.jit
    def _load_act(
        x_ptr,
        g_ptr,
        rm,
        rk,
        M,
        K,
        sxm,
        sgm,
        sgh,
        ACT: tl.constexpr,
        HD: tl.constexpr,
    ):
        """The `[BM, BK]` activation tile, with the prologue op applied (bf16 out)."""
        mask = (rm < M)[:, None] & (rk < K)[None, :]
        dt = x_ptr.dtype.element_ty
        if ACT == 0:
            return tl.load(x_ptr + rm[:, None] * sxm + rk[None, :], mask=mask, other=0.0)
        elif ACT == 1:
            g = tl.load(x_ptr + rm[:, None] * sxm + rk[None, :], mask=mask, other=0.0)
            u = tl.load(x_ptr + rm[:, None] * sxm + K + rk[None, :], mask=mask, other=0.0)
            gf = g.to(tl.float32)
            s = (gf / (1.0 + tl.exp(-gf))).to(dt)  # F.silu, rounded to bf16
            return (s.to(tl.float32) * u.to(tl.float32)).to(dt)
        else:
            o = tl.load(x_ptr + rm[:, None] * sxm + rk[None, :], mask=mask, other=0.0)
            goff = rm[:, None] * sgm + (rk // HD)[None, :] * sgh + (rk % HD)[None, :]
            gf = tl.load(g_ptr + goff, mask=mask, other=0.0).to(tl.float32)
            sg = (1.0 / (1.0 + tl.exp(-gf))).to(dt)  # torch.sigmoid, rounded
            return (o.to(tl.float32) * sg.to(tl.float32)).to(dt)

    @triton.jit
    def _skinny_kernel(
        x_ptr,
        g_ptr,
        w_ptr,
        o_ptr,
        ws_ptr,
        cnt_ptr,
        M,
        N,
        K,
        sxm,
        sgm,
        sgh,
        swn,
        som,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        SPLIT_K: tl.constexpr,
        K_PER_SPLIT: tl.constexpr,
        ACT: tl.constexpr,
        HD: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        rn = pid_n * BN + tl.arange(0, BN)
        rm = tl.arange(0, BM)
        n_ok = rn < N
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        k_lo = pid_k * K_PER_SPLIT
        for kk in range(0, K_PER_SPLIT, BK):
            rk = k_lo + kk + tl.arange(0, BK)
            w = tl.load(
                w_ptr + rn[:, None] * swn + rk[None, :],
                mask=n_ok[:, None] & (rk < K)[None, :],
                other=0.0,
            )
            x = _load_act(x_ptr, g_ptr, rm, rk, M, K, sxm, sgm, sgh, ACT, HD)
            acc += tl.dot(x, tl.trans(w))
        out_mask = (rm < M)[:, None] & n_ok[None, :]
        o_off = rm[:, None] * som + rn[None, :]
        if SPLIT_K == 1:
            tl.store(o_ptr + o_off, acc.to(o_ptr.dtype.element_ty), mask=out_mask)
        else:
            tile = BM * BN
            p_off = rm[:, None] * BN + (rn - pid_n * BN)[None, :]
            tl.store(ws_ptr + (pid_n * SPLIT_K + pid_k) * tile + p_off, acc)
            arrived = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
            if arrived == SPLIT_K - 1:
                total = tl.zeros((BM, BN), dtype=tl.float32)
                for s in tl.static_range(SPLIT_K):
                    total += tl.load(ws_ptr + (pid_n * SPLIT_K + s) * tile + p_off, cache_modifier=".cg")
                tl.store(o_ptr + o_off, total.to(o_ptr.dtype.element_ty), mask=out_mask)
                tl.atomic_xchg(cnt_ptr + pid_n, 0, sem="relaxed", scope="gpu")


WS_FLOATS = 1 << 22
"""fp32 split-K partials per device (16 MB); `config` never asks for more at M <= 64."""
CNT_SLOTS = 1 << 16
_WS: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}


def _scratch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed-size per-device split-K workspace and arrival counters, allocated once.

    Fixed, never regrown: a captured graph holds these pointers, so replacing the tensor
    would leave an earlier graph writing freed memory. Allocated on first use, an eager
    warmup step before any capture (`GraphDecodeRunner` warms before capturing). Counters
    start at zero and every launch leaves them at zero (the last arriving program resets its
    tile's counter).
    """
    bufs = _WS.get(device)
    if bufs is None:
        bufs = (
            torch.empty(WS_FLOATS, dtype=torch.float32, device=device),
            torch.zeros(CNT_SLOTS, dtype=torch.int32, device=device),
        )
        _WS[device] = bufs
    return bufs


def config(m: int, n: int, k: int) -> tuple[int, int, int, int, int]:
    """(BM, BN, BK, SPLIT_K, num_warps) for one shape. Chosen to put >= ~2 programs per CU
    on the 228-CU MI300A while keeping each program's weight slab >= ~32 KB."""
    bm = 16 if m <= 16 else (32 if m <= 32 else 64)
    bn = 16
    bk = min(256, triton.next_power_of_2(k))
    env = os.environ.get("SEED_SKINNY_CFG")
    tiles = triton.cdiv(n, bn)
    target = 456
    split = 1
    while tiles * split < target and k // (split * 2) >= 512 and k % (split * 2 * bk) == 0:
        split *= 2
    nw = 4
    if env:
        bn, bk, split, nw = (int(v) for v in env.split(","))
    return bm, bn, bk, split, nw


def skinny_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    act: Act = "none",
    gate: torch.Tensor | None = None,
    head_dim: int = 1,
    cfg: tuple[int, int, int, int, int] | None = None,
) -> torch.Tensor:
    """`F.linear(x', w)` where `x'` is `x` after the `act` prologue. `w` is `[N, K]`.

    `act="swiglu"`: `x` is `[..., 2K]` (gate then up); `x' = silu(gate) * up`.
    `act="sigmoid_gate"`: `x` is `[..., K]`, `gate` any `[..., K // head_dim, head_dim]`-shaped
    view with unit last stride; `x' = x * sigmoid(gate)`.
    """
    n, k = w.shape
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    m = x2.shape[0]
    if gate is not None:
        g2 = gate.reshape(m, -1, head_dim) if gate.dim() != 3 else gate
        sgm, sgh = g2.stride(0), g2.stride(1)
    else:
        g2, sgm, sgh = x2, 0, 0
    cfg = cfg or config(m, n, k)
    bm, bn, bk, split, nw = cfg
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    tiles = triton.cdiv(n, bn)
    if split > 1:
        ws, cnt = _scratch(x.device)
        if tiles * split * bm * bn > ws.numel() or tiles > cnt.numel():
            raise ValueError(f"skinny_gemm: split-K workspace too small for {(m, n, k, cfg)}")
    else:
        ws, cnt = out, out
    _skinny_kernel[(tiles, split)](
        x2,
        g2,
        w,
        out,
        ws,
        cnt,
        m,
        n,
        k,
        x2.stride(0),
        sgm,
        sgh,
        w.stride(0),
        out.stride(0),
        BM=bm,
        BN=bn,
        BK=bk,
        SPLIT_K=split,
        K_PER_SPLIT=k // split,
        ACT=_ACT_CODE[act],
        HD=head_dim,
        num_warps=nw,
    )
    return out.reshape(*lead, n)


def linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """`F.linear(x, w)`, through the skinny kernel when `SEED_SKINNY_GEMM` applies."""
    if available(x, w):
        return skinny_linear(x, w)
    return F.linear(x, w)

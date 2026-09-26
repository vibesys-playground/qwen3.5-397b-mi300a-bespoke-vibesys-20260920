"""Small fused elementwise kernels for the decode step's glue (`SEED_ELEMWISE_FUSED`).

`swiglu_mlp`'s `F.silu(gate) * up` is two torch launches on a `.chunk()` view (the `silu`
runs the non-vectorized strided elementwise path: 5.7 us at b48 in the r5 profile, plus 4.6 us
for the `mul`), 60 a step each. `silu_mul` is one launch with the same two bf16 rounding
points: `silu` computed in fp32 and rounded, then the product computed in fp32 and rounded.

Attention's output gate, `out * torch.sigmoid(gate.reshape(b, 1, -1))` with `gate` a `.chunk()`
view of `q_proj`'s output, is three launches (the `reshape` copy, `sigmoid`, `mul`), 15 a step
each; `sigmoid_gate_mul` is one, reading `gate` through its head stride, with torch's two
rounding points (`sigmoid` rounded to bf16, then the product).
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

ELEMWISE_FUSED = os.environ.get("SEED_ELEMWISE_FUSED", "0") not in ("0", "false", "False")


def available(x: torch.Tensor) -> bool:
    return ELEMWISE_FUSED and HAVE_TRITON and x.device.type == "cuda"


if HAVE_TRITON:

    @triton.jit
    def _silu_mul_kernel(gu_ptr, o_ptr, s_row, so_row, half, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        live = c < half
        dt = o_ptr.dtype.element_ty
        g = tl.load(gu_ptr + row * s_row + c, mask=live, other=0.0).to(tl.float32)
        u = tl.load(gu_ptr + row * s_row + half + c, mask=live, other=0.0).to(tl.float32)
        s = (g / (1.0 + tl.exp(-g))).to(dt)
        tl.store(o_ptr + row * so_row + c, (s.to(tl.float32) * u).to(dt), mask=live)


def silu_mul(gu: torch.Tensor) -> torch.Tensor:
    """`F.silu(g) * u` for `g, u = gu.chunk(2, -1)`, one launch. `gu` has a contiguous last axis."""
    half = gu.shape[-1] // 2
    flat = gu.reshape(-1, gu.shape[-1])
    out = torch.empty((*gu.shape[:-1], half), dtype=gu.dtype, device=gu.device)
    of = out.view(-1, half)
    block = min(1024, triton.next_power_of_2(half))
    _silu_mul_kernel[(flat.shape[0], triton.cdiv(half, block))](
        flat, of, flat.stride(0), of.stride(0), half, BLOCK=block
    )
    return out


if HAVE_TRITON:

    @triton.jit
    def _sigmoid_gate_mul_kernel(
        o_ptr, g_ptr, y_ptr, so_row, sg_row, sg_head, sy_row, HD: tl.constexpr, BLOCK: tl.constexpr
    ):
        row = tl.program_id(0)
        c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        dt = y_ptr.dtype.element_ty
        o = tl.load(o_ptr + row * so_row + c).to(tl.float32)
        g = tl.load(g_ptr + row * sg_row + (c // HD) * sg_head + c % HD).to(tl.float32)
        s = (1.0 / (1.0 + tl.exp(-g))).to(dt)
        tl.store(y_ptr + row * sy_row + c, (o * s.to(tl.float32)).to(dt))


def sigmoid_gate_mul(out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """`out * torch.sigmoid(gate.reshape(out.shape))` in one launch.

    `out` is `[b, 1, heads * hd]` with a contiguous last axis; `gate` is `[b, 1, heads, hd]`
    with a unit last stride (any row/head stride). `heads * hd` must be a multiple of 256.
    """
    b, heads, hd = gate.shape[0], gate.shape[-2], gate.shape[-1]
    width = heads * hd
    o2 = out.reshape(b, width)
    y = torch.empty((b, width), dtype=out.dtype, device=out.device)
    block = 256
    if width % block or o2.stride(-1) != 1 or gate.stride(-1) != 1:
        raise ValueError(f"sigmoid_gate_mul: width {width}, strides {o2.stride()}/{gate.stride()}")
    _sigmoid_gate_mul_kernel[(b, width // block)](
        o2, gate, y, o2.stride(0), gate.stride(0), gate.stride(-2), y.stride(0), HD=hd, BLOCK=block
    )
    return y.view(out.shape)

"""Decode-glue fusions against the torch sequences they replace: `rmsnorm_fused.add_rmsnorm`
(`SEED_ADD_RMSNORM`), `decode_glue.silu_mul`/`sigmoid_gate_mul` (`SEED_ELEMWISE_FUSED`),
`deltanet_fused.gated_rmsnorm(out_dtype=...)` (`SEED_DN_NORM_F32_IN`), and the
`skinny_gemm.skinny_linear` split-K path.

On an accelerator every fusion must be bit-identical to the unfused sequence (the GEMM only
close: it reorders the fp32 reduction). Under `TRITON_INTERPRET=1` on CPU the interpreter's
reduction and transcendental order differ from torch's, so that arm checks a tolerance.

    TRITON_INTERPRET=1 python -m pytest seed_tests/test_decode_glue.py -p no:cacheprovider --no-cov
"""

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import decode_glue  # noqa: E402
import deltanet_fused  # noqa: E402
import rmsnorm_fused  # noqa: E402
import skinny_gemm  # noqa: E402

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.bfloat16 if DEVICE.type == "cuda" else torch.float16
"""The interpreter mis-handles bf16 `tl.dot` operands, so the CPU arm runs fp16."""

pytestmark = pytest.mark.skipif(
    not rmsnorm_fused.HAVE_TRITON or not (torch.cuda.is_available() or INTERPRET),
    reason="needs triton plus either an accelerator or TRITON_INTERPRET=1",
)


def same(got: torch.Tensor, want: torch.Tensor, tol: float = 2e-2) -> None:
    if DEVICE.type == "cuda":
        assert torch.equal(got, want)
    else:
        scale = max(want.float().abs().max().item(), 1e-30)
        assert (got.float() - want.float()).abs().max().item() / scale < tol


@pytest.mark.parametrize("b", [1, 3])
def test_add_rmsnorm_matches_add_then_rmsnorm(b: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(b, 1, 256, dtype=DT, device=DEVICE)
    r = torch.randn(b, 1, 256, dtype=DT, device=DEVICE)
    w = torch.randn(256, dtype=DT, device=DEVICE) * 0.1
    xo, h = rmsnorm_fused.add_rmsnorm(x, r, w, 1e-6)
    assert torch.equal(xo, x + r)
    same(h, rmsnorm_fused.rmsnorm(x + r, w, 1e-6))


def test_silu_mul_matches_chunked_torch() -> None:
    torch.manual_seed(1)
    gu = torch.randn(5, 2 * 96, dtype=DT, device=DEVICE) * 3
    g, u = gu.chunk(2, dim=-1)
    same(decode_glue.silu_mul(gu), F.silu(g) * u)


def test_sigmoid_gate_mul_reads_the_chunked_gate_view() -> None:
    torch.manual_seed(2)
    b, heads, hd = 3, 2, 128
    q_and_gate = torch.randn(b, 1, heads, 2 * hd, dtype=DT, device=DEVICE)
    _, gate = q_and_gate.chunk(2, dim=-1)
    out = torch.randn(b, 1, heads * hd, dtype=DT, device=DEVICE)
    same(decode_glue.sigmoid_gate_mul(out, gate), out * torch.sigmoid(gate.reshape(b, 1, -1)))


def test_gated_rmsnorm_rounds_an_fp32_input_in_kernel() -> None:
    torch.manual_seed(3)
    x = torch.randn(3, 4, 64, device=DEVICE)
    gate = torch.randn(3, 4, 64, dtype=DT, device=DEVICE)
    w = torch.randn(64, dtype=DT, device=DEVICE)
    got = deltanet_fused.gated_rmsnorm(x, gate, w, 1e-6, DT)
    assert got.dtype == DT
    same(got, deltanet_fused.gated_rmsnorm(x.to(DT), gate, w, 1e-6))


@pytest.mark.parametrize("split", [1, 4])
def test_skinny_linear_split_k_is_close_and_deterministic(split: int) -> None:
    torch.manual_seed(4)
    x = torch.randn(5, 512, dtype=DT, device=DEVICE)
    w = torch.randn(40, 512, dtype=DT, device=DEVICE)
    cfg = (16, 16, 64, split, 4)
    got = skinny_gemm.skinny_linear(x, w, cfg=cfg)
    want = F.linear(x.float(), w.float())
    assert ((got.float() - want).abs().max() / want.abs().max()).item() < 1e-2
    assert torch.equal(got, skinny_gemm.skinny_linear(x, w, cfg=cfg))
    if split > 1:  # every launch leaves its arrival counters at zero for the next one
        assert int(skinny_gemm._scratch(x.device)[1].abs().sum()) == 0

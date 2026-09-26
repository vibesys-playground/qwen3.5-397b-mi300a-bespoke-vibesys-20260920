"""`SEED_MOE_ROUTE_FUSED`'s two building blocks, against the torch/kernel chain they replace.

1. `router_fused.route(..., index_dtype=torch.int32)`: the same kernel `test_router_fused.py`
   already covers, now writing `top_i` as int32 (`mxfp4_gemv.bw_prep`'s `a_expert` dtype, so
   `_moe_hip_route_fused` needs no separate `top_i.to(torch.int32)` cast).
2. `mxfp4_gemv.bw_combine_glue`: `_bw_combine_kernel` (`bw_combine`) plus the shared-expert
   sigmoid-gate-multiply-add `Model.moe` otherwise does as three separate torch ops, folded
   into one kernel.

Two ways to run it, same as this campaign's other fused kernels:

    # real kernels, needs an accelerator
    python -m pytest .../seed_tests/test_moe_route_fused.py -p no:cacheprovider --no-cov

    # logic only, no GPU: Triton's reference interpreter, on CPU tensors
    TRITON_INTERPRET=1 python -m pytest ... -p no:cacheprovider --no-cov
"""

import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "seed_tests"))

import mxfp4_gemv  # noqa: E402
import router_fused  # noqa: E402

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pytestmark = pytest.mark.skipif(
    not router_fused.HAVE_TRITON or not (torch.cuda.is_available() or INTERPRET),
    reason="needs triton plus either an accelerator or TRITON_INTERPRET=1",
)

TOL = 1e-5


# ---- 1. router_fused.route(index_dtype=torch.int32) ----------------------------------------


def torch_route(routing: torch.Tensor, experts_out: int, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`Model.moe`'s own router prelude, transcribed as the oracle (same as `test_router_fused.py`)."""
    probs = routing[:, :experts_out].softmax(-1, dtype=torch.float)
    top_w, top_i = probs.topk(top_k, dim=-1)
    top_w = top_w / top_w.sum(-1, keepdim=True)
    return top_w, top_i


@pytest.mark.parametrize(
    "t,experts_out,top_k",
    [
        (1, 4, 1),
        (5, 8, 2),
        (3, 512, 10),  # the real model's shape
        (48, 512, 10),  # a real decode batch
    ],
)
def test_route_int32_matches_route_int64(t: int, experts_out: int, top_k: int) -> None:
    """`index_dtype=torch.int32` must select the same experts and weights as the default int64
    path (`test_router_fused.py`'s `test_route_matches_torch_on_well_separated_logits`), just
    in a narrower integer type -- `_fused_route_kernel` itself is untouched; only the output
    tensor's dtype differs. Widely separated logits, so there is no top-k tie to break either
    way (see `router_fused.py`'s module docstring for that separate, documented risk)."""
    gen = torch.Generator().manual_seed(t * 1000 + experts_out * 10 + top_k)
    routing = (torch.randn(t, experts_out + 1, generator=gen) * 4.0).to(DEVICE)

    want_w, want_i = torch_route(routing, experts_out, top_k)
    got_w64, got_i64 = router_fused.route(routing, experts_out, top_k)
    got_w32, got_i32 = router_fused.route(routing, experts_out, top_k, index_dtype=torch.int32)

    assert got_i32.dtype == torch.int32
    assert got_w32.dtype == torch.float32
    # int32 and int64 runs must agree with each other bit for bit (same kernel, same inputs).
    assert torch.equal(got_i32.to(torch.int64), got_i64)
    assert torch.equal(got_w32, got_w64)

    # And both must match the torch oracle as a *set* per row (top-k order/tie-break is not
    # part of the contract -- see `test_router_fused.py`).
    want_order = want_i.argsort(dim=-1)
    got_order = got_i32.argsort(dim=-1)
    want_i_sorted = want_i.gather(-1, want_order)
    got_i_sorted = got_i32.to(torch.int64).gather(-1, got_order)
    assert torch.equal(got_i_sorted, want_i_sorted)
    want_w_sorted = want_w.gather(-1, want_order)
    got_w_sorted = got_w32.gather(-1, got_order)
    assert (got_w_sorted - want_w_sorted).abs().max().item() < TOL


def test_route_int32_rejects_bad_index_dtype() -> None:
    routing = torch.randn(2, 8, device=DEVICE)
    with pytest.raises(ValueError, match="index_dtype"):
        router_fused.route(routing, 8, 2, index_dtype=torch.float32)


# ---- 2. mxfp4_gemv.bw_combine_glue ----------------------------------------------------------

TOP_K = 10
HIDDEN = 4096


def _random_combine_inputs(tokens: int, span: tuple[int, int], *, seed: int, zero_rows=()):
    from test_mxfp4_fused_gemv import random_routing

    a_expert, a_weight = random_routing(tokens, 64, TOP_K, seed=seed, zero_rows=zero_rows)
    g = torch.Generator().manual_seed(seed + 1)
    y = torch.randn(tokens * TOP_K, HIDDEN, generator=g).to(torch.bfloat16).to(DEVICE)
    shared = torch.randn(tokens, HIDDEN, generator=g).to(torch.bfloat16).to(DEVICE)
    gate = torch.randn(tokens, 1, generator=g).to(torch.bfloat16).to(DEVICE)
    return a_expert, a_weight, y, shared, gate


@pytest.mark.parametrize("tokens,span", [(1, (16, 32)), (7, (16, 32)), (48, (16, 32))])
def test_bw_combine_glue_matches_bw_combine_plus_glue(tokens: int, span: tuple[int, int]) -> None:
    """`bw_combine_glue`'s `routed_sum + sigmoid(gate) * shared` must match `bw_combine`
    followed by the three separate torch ops `Model.moe` runs today, to the same
    fp32-reduction-order bar the rest of this codebase's fused kernels hold (see
    `_bw_combine_glue_kernel`'s docstring: the only difference is *when* the routed sum rounds
    to bf16, not what it sums)."""
    a_expert, a_weight, y, shared, gate = _random_combine_inputs(tokens, span, seed=3, zero_rows=(0,) if tokens > 1 else ())

    out_unfused = torch.empty(tokens, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    mxfp4_gemv.bw_combine(y, a_expert, a_weight, span, TOP_K, out_unfused)
    want = out_unfused.float() + torch.sigmoid(gate.float()) * shared.float()

    out_fused = torch.empty(tokens, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    mxfp4_gemv.bw_combine_glue(y, a_expert, a_weight, span, TOP_K, shared, gate, out_fused)

    err = (out_fused.float() - want).abs().max().item() / want.abs().max().clamp_min(1e-6).item()
    assert err < 2e-2


def test_bw_combine_glue_reads_gate_through_its_own_row_stride() -> None:
    """`gate` as a non-contiguous view (`routing[:, experts_out:]`'s real shape) must still
    read the right column -- the kernel takes `gate.stride(0)` explicitly rather than
    assuming a tight `[tokens, 1]` layout."""
    tokens, span = 5, (16, 32)
    a_expert, a_weight, y, shared, _ = _random_combine_inputs(tokens, span, seed=9)
    wide = torch.randn(tokens, 4, generator=torch.Generator().manual_seed(10)).to(torch.bfloat16).to(DEVICE)
    gate_view = wide[:, 3:]  # stride(0) == 4, not 1
    assert gate_view.stride(0) == 4

    out_unfused = torch.empty(tokens, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    mxfp4_gemv.bw_combine(y, a_expert, a_weight, span, TOP_K, out_unfused)
    want = out_unfused.float() + torch.sigmoid(gate_view.float()) * shared.float()

    out_fused = torch.empty(tokens, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    mxfp4_gemv.bw_combine_glue(y, a_expert, a_weight, span, TOP_K, shared, gate_view, out_fused)

    err = (out_fused.float() - want).abs().max().item() / want.abs().max().clamp_min(1e-6).item()
    assert err < 2e-2


def test_bw_combine_glue_rejects_mismatched_shared_shape() -> None:
    tokens, span = 3, (16, 32)
    a_expert, a_weight, y, shared, gate = _random_combine_inputs(tokens, span, seed=1)
    out = torch.empty(tokens, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    with pytest.raises(ValueError, match="shared"):
        mxfp4_gemv.bw_combine_glue(y, a_expert, a_weight, span, TOP_K, shared[:-1], gate, out)

"""`router_fused.route` against the torch chain it replaces (`Model.moe`'s router prelude).

Same two ways to run it as the rest of this campaign's fused kernels:

    # real kernel, needs an accelerator
    python -m pytest .../seed_tests/test_router_fused.py -p no:cacheprovider --no-cov

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

import router_fused  # noqa: E402

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pytestmark = pytest.mark.skipif(
    not router_fused.HAVE_TRITON or not (torch.cuda.is_available() or INTERPRET),
    reason="needs triton plus either an accelerator or TRITON_INTERPRET=1",
)

TOL = 1e-5


def torch_route(routing: torch.Tensor, experts_out: int, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`Model.moe`'s own router prelude, transcribed as the oracle."""
    probs = routing[:, :experts_out].softmax(-1, dtype=torch.float)
    top_w, top_i = probs.topk(top_k, dim=-1)
    top_w = top_w / top_w.sum(-1, keepdim=True)
    return top_w, top_i


SHAPES = pytest.mark.parametrize(
    "t,experts_out,top_k",
    [
        (1, 4, 1),  # one token, one expert selected
        (5, 8, 2),  # deltanet_tp_fixture's tiny shape
        (3, 512, 10),  # the real model's shape
        (4, 17, 3),  # experts_out not a power of two, so the softmax block is masked
    ],
)


@SHAPES
def test_route_matches_torch_on_well_separated_logits(t: int, experts_out: int, top_k: int) -> None:
    """Logits spread widely enough that top-k selection has no near-ties: only the case this
    kernel's docstring claims to match exactly (fp32-rounding-order noise only, not a
    different *selection*). `routing` is wider than `experts_out` -- `Model.moe`'s real
    router output also carries the shared-expert gate past column `experts_out` -- so this
    also checks the kernel reads `routing`'s row stride rather than assuming a tight `[t,
    experts_out]` layout.
    """
    gen = torch.Generator().manual_seed(t * 1000 + experts_out * 10 + top_k)
    routing = (torch.randn(t, experts_out + 3, generator=gen) * 4.0).to(DEVICE)

    want_w, want_i = torch_route(routing, experts_out, top_k)
    got_w, got_i = router_fused.route(routing, experts_out, top_k)

    assert got_w.shape == want_w.shape and got_i.shape == want_i.shape
    assert got_i.dtype == want_i.dtype == torch.int64
    # Sort each row's (index, weight) pairs before comparing: top-k order between torch's and
    # the kernel's own tie-break-free selection need not match position for position, only
    # the *set* of selected experts and each one's weight (see the module docstring).
    want_order = want_i.argsort(dim=-1)
    got_order = got_i.argsort(dim=-1)
    want_i_sorted = want_i.gather(-1, want_order)
    got_i_sorted = got_i.gather(-1, got_order)
    assert torch.equal(got_i_sorted, want_i_sorted), "selected a different set of experts"
    want_w_sorted = want_w.gather(-1, want_order)
    got_w_sorted = got_w.float().gather(-1, got_order)
    err = (got_w_sorted - want_w_sorted).abs().max().item()
    assert err < TOL
    assert (got_w.sum(-1) - 1.0).abs().max().item() < TOL, "weights must renormalize to 1"


def test_route_rejects_top_k_over_the_cap() -> None:
    routing = torch.randn(2, 8, device=DEVICE)
    with pytest.raises(ValueError, match="MAX_TOP_K"):
        router_fused.route(routing, 8, router_fused.MAX_TOP_K + 1)

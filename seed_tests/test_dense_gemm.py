"""`fuse_rows`/`fuse_moe_dense` equivalence, and the shape set `blas_tune` tunes.

The GEMM solution selection itself is a ROCm facility with no CPU spelling, so what is
testable here is the part that decides *what* to tune (`decode_gemms`, `tuned_batches`) and the
weight concatenations that reshape the decode step's GEMMs. The selection's own effect is a
hardware measurement and lives in the commit message and `blas_tune`'s docstring.
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import blas_tune  # noqa: E402
import model as seed_model  # noqa: E402


def _layer(seed: int, hidden: int, inter: int, experts: int) -> dict:
    g = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g)

    return {
        "router": randn(experts, hidden),
        "shared_gate": randn(1, hidden),
        "shared_expert.gate_proj": randn(inter, hidden),
        "shared_expert.up_proj": randn(inter, hidden),
        "shared_expert.down_proj": randn(hidden, inter),
    }


# ---------------------------------------------------------------- fuse_rows


def test_fuse_rows_keeps_the_named_parts_as_views_of_one_tensor() -> None:
    layer = _layer(0, 16, 8, 4)
    fused = seed_model.fuse_moe_dense(layer)

    for name in (*seed_model.ROUTER_GATE_ORDER, *seed_model.GATE_UP_ORDER):
        torch.testing.assert_close(fused[name], layer[name], atol=0, rtol=0)
    # A view, not a copy: the concatenation is the only storage of these weights.
    assert fused["router"].data_ptr() == fused["router_gate"].data_ptr()
    assert fused["shared_expert.gate_proj"].data_ptr() == (
        fused["shared_expert.gate_up_proj"].data_ptr()
    )
    assert fused["router_gate"].shape == (5, 16)
    assert fused["shared_expert.gate_up_proj"].shape == (16, 16)


def test_fuse_rows_passes_through_a_layer_that_lacks_the_parts() -> None:
    # This is what makes one call safe for both mixer types: a full-attention layer holds no
    # DeltaNet input projections and must come back untouched rather than raise.
    attention = {"q_proj": torch.zeros(4, 4)}
    assert seed_model.fuse_in_proj(attention) is attention
    assert "in_proj_all" not in seed_model.fuse_in_proj(attention)


def test_fused_moe_dense_keeps_the_routing_bit_exact() -> None:
    """The router's columns must not move: they pick the experts, and a nudge could reorder.

    The shared gate's column may move (BLAS reduces a 1-column GEMV differently from a wide
    GEMM), which is why `test_vectorized_moe_matches_old_per_expert_loop` allows 1e-4. The
    router's do not, because they are a whole macro tile's worth of columns either way.
    """
    layer = seed_model.fuse_moe_dense(_layer(1, 32, 16, 6))
    h = torch.randn(9, 32, generator=torch.Generator().manual_seed(7))

    routing = F.linear(h, layer["router_gate"])

    assert torch.equal(routing[:, :6], F.linear(h, layer["router"]))


@pytest.mark.parametrize("tokens", [1, 5, 40])
def test_swiglu_against_the_concatenation_matches_two_separate_projections(tokens: int) -> None:
    layer = seed_model.fuse_moe_dense(_layer(2, 32, 16, 4))
    h = torch.randn(tokens, 32, generator=torch.Generator().manual_seed(11))

    got = seed_model.swiglu_mlp(
        h, layer["shared_expert.gate_up_proj"], layer["shared_expert.down_proj"]
    )
    gate = F.linear(h, layer["shared_expert.gate_proj"])
    up = F.linear(h, layer["shared_expert.up_proj"])
    want = F.linear(F.silu(gate) * up, layer["shared_expert.down_proj"])

    # Not bit-exact: the same products, reduced in whatever order BLAS blocks a 2*inter-column
    # GEMM versus an inter-column one. What must hold is that the fused form computes the same
    # function, so the bound is float32's, not a fitted number.
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------- blas_tune


def test_decode_gemms_collects_each_distinct_projection_once() -> None:
    shared = seed_model.fuse_moe_dense(_layer(3, 32, 16, 4))
    attention = {**shared, "q_proj": torch.zeros(8, 32), "o_proj": torch.zeros(32, 8)}
    deltanet = {**shared, "in_proj_all": torch.zeros(12, 32), "out_proj": torch.zeros(32, 12)}
    fake = SimpleNamespace(
        layers=[attention, deltanet, dict(attention)], lm_head=torch.zeros(64, 32), max_batch=8
    )

    shapes = {(int(w.shape[0]), int(w.shape[1])) for w in blas_tune.decode_gemms(fake)}

    # Deduplicated across the repeated attention layer, and the views into a concatenation
    # (`router`, `shared_expert.gate_proj`, ...) are not tuned separately from their parent.
    assert shapes == {(5, 32), (32, 32), (32, 16), (8, 32), (32, 8), (12, 32), (32, 12), (64, 32)}


def test_tuned_batches_covers_a_single_request_and_the_full_slot_pool() -> None:
    assert blas_tune.tuned_batches(48) == (1, 48)
    assert blas_tune.tuned_batches(1) == (1,)


def test_tune_is_a_no_op_without_an_accelerator(monkeypatch: pytest.MonkeyPatch) -> None:
    # The hermetic tests and any CPU-only run reach this; TunableOp is a device BLAS facility.
    monkeypatch.setenv("SEED_BLAS_TUNE", "0")
    assert not blas_tune.enabled()
    assert blas_tune.tune(SimpleNamespace(layers=[], lm_head=torch.zeros(4, 4), max_batch=1)) == 0

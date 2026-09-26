"""Hermetic CPU tests for `hot_experts.py` (`SEED_HOT_EXPERTS=1`) on small synthetic MXFP4
weights (not the real model): numerics of the hot bf16 dense path against a plain-torch
per-expert MXFP4 reference, and routing-split correctness (the hot/cold assignment split never
double-counts or drops an assignment).

Run with a python that has torch (see seed_tests/test_seed_parity.py for the recipe):
    /tmp/torchenv/bin/python -m pytest examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/seed_tests/test_hot_experts.py -p no:cacheprovider --no-cov
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import hot_experts  # noqa: E402
from mxfp4 import dequant_mxfp4  # noqa: E402
from test_moe_vectorize import build_layer, quantize_mxfp4  # noqa: E402,F401


def reference_moe(
    ex: dict, h: torch.Tensor, top_i: torch.Tensor, top_w: torch.Tensor
) -> torch.Tensor:
    """Full per-assignment MXFP4 reference: dequantize every activated expert, no caching."""
    out = torch.zeros_like(h)
    for e in top_i.unique().tolist():
        tok, slot = (top_i == e).nonzero(as_tuple=True)
        gate_up = dequant_mxfp4(ex["gate_up"][e], ex["gate_up_scale"][e], h.dtype)
        down = dequant_mxfp4(ex["down"][e], ex["down_scale"][e], h.dtype)
        gate, up = F.linear(h[tok], gate_up).chunk(2, dim=-1)
        y = F.linear(F.silu(gate) * up, down) * top_w[tok, slot, None]
        out.index_add_(0, tok, y.to(out.dtype))
    return out


def cold_reference(
    ex: dict, h: torch.Tensor, top_i: torch.Tensor, top_w: torch.Tensor
) -> torch.Tensor:
    """Same as `reference_moe`, but skips zero-weight assignments -- the cold path's contract
    after the hot path has zeroed the assignments it consumed (mirrors `mxfp4_gemv.py`'s
    `a_weight != 0` gate that `model.Model.moe` relies on)."""
    out = torch.zeros_like(h)
    for e in top_i.unique().tolist():
        tok, slot = (top_i == e).nonzero(as_tuple=True)
        live = top_w[tok, slot] != 0
        if not live.any():
            continue
        tok, slot = tok[live], slot[live]
        gate_up = dequant_mxfp4(ex["gate_up"][e], ex["gate_up_scale"][e], h.dtype)
        down = dequant_mxfp4(ex["down"][e], ex["down_scale"][e], h.dtype)
        gate, up = F.linear(h[tok], gate_up).chunk(2, dim=-1)
        y = F.linear(F.silu(gate) * up, down) * top_w[tok, slot, None]
        out.index_add_(0, tok, y.to(out.dtype))
    return out


def route(h: torch.Tensor, router: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    probs = F.linear(h, router).softmax(-1, dtype=torch.float)
    top_w, top_i = probs.topk(top_k, dim=-1)
    return top_w / top_w.sum(-1, keepdim=True), top_i


# -- pure logic -----------------------------------------------------------------------------


def test_select_hot_local_picks_highest_counts_deterministically() -> None:
    counts = torch.tensor([5, 1, 5, 3, 0, 5])
    got = hot_experts.select_hot_local(counts, 3)
    # ties at count=5 (indices 0, 2, 5): argsort(stable) keeps ascending index order among them.
    assert got.tolist() == [0, 2, 5]


def test_select_hot_local_h_zero_or_more_than_available() -> None:
    counts = torch.tensor([1, 2, 3])
    assert hot_experts.select_hot_local(counts, 0).numel() == 0
    assert hot_experts.select_hot_local(counts, 100).tolist() == [0, 1, 2]


def test_budget_h_per_layer_clips_and_zeroes() -> None:
    # 1 GiB, 10 layers, 100 MiB/expert/layer budget -> floor(1 GiB / 10 / 100 MiB) experts.
    per_layer_bytes = (1.0 * 1024**3) / 10
    bytes_per_expert = 100 * 1024**2
    h = hot_experts.budget_h_per_layer(1.0, 10, bytes_per_expert, local_experts=128)
    assert h == int(per_layer_bytes // bytes_per_expert)
    assert hot_experts.budget_h_per_layer(0.0, 10, bytes_per_expert, 128) == 0
    assert hot_experts.budget_h_per_layer(1000.0, 1, 1, local_experts=4) == 4  # clipped


def test_per_expert_bf16_bytes_matches_hand_computed_shape() -> None:
    ex = {"gate_up": torch.empty(6, 64, 16), "down": torch.empty(6, 32, 8)}  # [E, rows, K/2]
    # gate_up: 64*16*2=2048 values -> *2 bytes; down: 32*8*2=512 values -> *2 bytes.
    assert hot_experts.per_expert_bf16_bytes(ex) == (64 * 16 * 2 + 32 * 8 * 2) * 2


# -- numerics: hot bf16 path vs the MXFP4 reference ------------------------------------------


@pytest.mark.parametrize(
    "seed,tokens,hidden,inter,experts,top_k,n_hot",
    [
        (0, 6, 32, 32, 8, 2, 2),
        (1, 12, 64, 32, 12, 3, 4),
        (2, 20, 64, 32, 16, 4, 8),
        (3, 9, 32, 32, 8, 8, 0),  # no hot experts: hot path must contribute exactly zero
        (4, 9, 32, 32, 8, 8, 8),  # every expert hot: cold path must contribute exactly zero
    ],
)
def test_hot_path_plus_cold_path_matches_full_reference(
    seed: int, tokens: int, hidden: int, inter: int, experts: int, top_k: int, n_hot: int
) -> None:
    dtype = torch.float32  # exact-numerics comparison, not a bf16 tolerance check
    layer = build_layer(seed, hidden, inter, experts, mxfp4=True, dtype=dtype)
    ex = layer["experts"]
    g = torch.Generator().manual_seed(seed + 2000)
    h = torch.randn(tokens, hidden, generator=g, dtype=dtype)
    top_w, top_i = route(h, layer["router"], top_k)

    want = reference_moe(ex, h, top_i, top_w)

    counts = torch.bincount(top_i.reshape(-1), minlength=experts)
    hot_ids = hot_experts.select_hot_local(counts, n_hot)
    cache = hot_experts.build_layer_cache(ex, hot_ids, dtype)

    hot_out, consumed = hot_experts.hot_expert_forward(h, top_i, top_w, lo=0, cache=cache)
    spliced_w = top_w.masked_fill(consumed, 0.0)
    cold_out = cold_reference(ex, h, top_i, spliced_w)
    got = hot_out + cold_out

    assert got.shape == want.shape
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)
    if n_hot == 0:
        assert not consumed.any()
        torch.testing.assert_close(hot_out, torch.zeros_like(hot_out))
    if n_hot == experts:
        assert consumed.all()
        torch.testing.assert_close(cold_out, torch.zeros_like(cold_out))


def test_routing_split_never_double_counts_or_drops_weight() -> None:
    """Every (token, slot) assignment is consumed by exactly the hot path or the cold path,
    never both and never neither, and the consumed slots' weight is exactly what the hot path
    used (nothing left over for the cold path to also apply)."""
    dtype = torch.float32
    hidden, inter, experts, top_k, tokens = 32, 32, 10, 4, 15
    layer = build_layer(7, hidden, inter, experts, mxfp4=True, dtype=dtype)
    g = torch.Generator().manual_seed(7001)
    h = torch.randn(tokens, hidden, generator=g, dtype=dtype)
    top_w, top_i = route(h, layer["router"], top_k)

    counts = torch.bincount(top_i.reshape(-1), minlength=experts)
    hot_ids = hot_experts.select_hot_local(counts, 3)
    cache = hot_experts.build_layer_cache(layer["experts"], hot_ids, dtype)
    _, consumed = hot_experts.hot_expert_forward(h, top_i, top_w, lo=0, cache=cache)

    hot_global = set(hot_ids.tolist())
    want_consumed = torch.tensor([[int(e.item()) in hot_global for e in row] for row in top_i])
    assert torch.equal(consumed, want_consumed)

    spliced_w = top_w.masked_fill(consumed, 0.0)
    # every zeroed slot was a hot assignment, every nonzero slot was not.
    assert torch.equal(spliced_w == 0, consumed | (top_w == 0))
    assert torch.equal((spliced_w != 0), (top_w != 0) & ~consumed)


def test_hot_experts_disabled_is_a_pure_no_op_in_model_moe() -> None:
    """`SEED_HOT_EXPERTS=0` (the module default): `hot_experts.ENABLED` is False, so
    `model.Model.moe`'s hot-path branch never runs regardless of `hot_cache` contents. This
    pins the gate itself, independent of the numerics tests above.
    """
    assert hot_experts.ENABLED is False


def test_hot_cache_from_bw_shuffled_experts_matches_the_load_layout() -> None:
    """`SEED_MOE_BW_VARIANT` shuffle variants rewrite the MXFP4 bytes in place at load; the
    hot cache (built later, at warmup) must unshuffle before dequant and so hold exactly the
    bf16 weights it would build from the original layout."""
    import mxfp4_moe_bw2

    g = torch.Generator().manual_seed(7)
    e, hidden, inter = 4, 256, 128
    ex = {
        "gate_up": torch.randint(
            0, 256, (e, 2 * inter, hidden // 2), generator=g, dtype=torch.uint8
        ),
        "gate_up_scale": torch.randint(
            120, 130, (e, 2 * inter, hidden // 32), generator=g, dtype=torch.uint8
        ),
        "down": torch.randint(0, 256, (e, hidden, inter // 2), generator=g, dtype=torch.uint8),
        "down_scale": torch.randint(
            120, 130, (e, hidden, inter // 32), generator=g, dtype=torch.uint8
        ),
    }
    shuffled = {k: v.clone() for k, v in ex.items()}
    mxfp4_moe_bw2.bw_shuffle(shuffled)
    assert "bw_layout" in shuffled and not torch.equal(shuffled["gate_up"], ex["gate_up"])
    ids = torch.tensor([3, 1])
    want = hot_experts.build_layer_cache(ex, ids, torch.bfloat16)
    got = hot_experts.build_layer_cache(shuffled, ids, torch.bfloat16)
    assert torch.equal(got.gate_up, want.gate_up) and torch.equal(got.down, want.down)

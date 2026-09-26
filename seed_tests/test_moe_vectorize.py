"""Hermetic CPU regression test: vectorized MoE (model.Model.moe) vs the old per-expert-loop
implementation it replaced, on small random synthetic weights/activations/scales (not the real
model). No GPU, no real checkpoint.

`old_moe` below is a verbatim copy of the pre-vectorization `Model.moe`/`expert_weights` (one
Python-level `for e in top_i.unique().tolist()` loop, one `dequant_mxfp4` call per activated
expert), kept only as the regression oracle for this test.

Run with a python that has torch (see seed_tests/test_seed_parity.py for the recipe):
    /tmp/torchenv/bin/python -m pytest examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/seed_tests/test_moe_vectorize.py -p no:cacheprovider --no-cov
"""

import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import model as seed_model  # noqa: E402
from mxfp4 import dequant_mxfp4  # noqa: E402

FP4 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def quantize_mxfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Test-local MXFP4 quantizer: returns (packed u8, e8m0 scale u8) for a [..., K] tensor."""
    *lead, k = w.shape
    blocks = w.reshape(*lead, k // 32, 32)
    exp = torch.floor(torch.log2(blocks.abs().amax(-1).clamp_min(1e-30))) - 2
    scale = torch.exp2(exp)[..., None]
    mags = (blocks.abs() / scale)[..., None] - torch.tensor(FP4)
    code = mags.abs().argmin(-1) + 8 * (blocks < 0)
    code = code.reshape(*lead, k).to(torch.uint8)
    packed = code[..., 0::2] | (code[..., 1::2] << 4)
    return packed, (exp + 127).reshape(*lead, k // 32).to(torch.uint8)


def old_expert_weights(ex: dict, e: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Verbatim copy of the removed per-expert `Model.expert_weights`."""
    if "gate_up_scale" not in ex:
        return ex["gate_up"][e], ex["down"][e]
    return (
        dequant_mxfp4(ex["gate_up"][e], ex["gate_up_scale"][e], dtype),
        dequant_mxfp4(ex["down"][e], ex["down_scale"][e], dtype),
    )


def old_moe(cfg, w: dict, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Verbatim copy of the removed per-expert-loop `Model.moe` body."""
    h = x.reshape(-1, cfg.hidden)
    probs = F.linear(h, w["router"]).softmax(-1, dtype=torch.float)
    top_w, top_i = probs.topk(cfg.top_k, dim=-1)
    top_w = (top_w / top_w.sum(-1, keepdim=True)).to(h.dtype)
    out = torch.zeros_like(h)
    for e in top_i.unique().tolist():
        tok, slot = (top_i == e).nonzero(as_tuple=True)
        gate_up, down = old_expert_weights(w["experts"], e, dtype)
        gate, up = F.linear(h[tok], gate_up).chunk(2, dim=-1)
        y = F.linear(F.silu(gate) * up, down) * top_w[tok, slot, None]
        out.index_add_(0, tok, y.to(out.dtype))
    # Deliberately the unfused spelling: one `F.linear` per projection, which is what
    # `fuse_moe_dense` and `swiglu_mlp`'s concatenated form have to reproduce.
    gate = F.linear(h, w["shared_expert.gate_proj"])
    up = F.linear(h, w["shared_expert.up_proj"])
    shared = F.linear(F.silu(gate) * up, w["shared_expert.down_proj"])
    out = out + torch.sigmoid(F.linear(h, w["shared_gate"])) * shared
    return out.reshape(x.shape)


def build_layer(
    seed: int, hidden: int, inter: int, experts: int, *, mxfp4: bool, dtype: torch.dtype
) -> dict:
    g = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g, dtype=torch.float32)

    gate_up_dense = randn(experts, 2 * inter, hidden)
    down_dense = randn(experts, hidden, inter)
    ex: dict[str, torch.Tensor] = {}
    if mxfp4:
        ex["gate_up"], ex["gate_up_scale"] = quantize_mxfp4(gate_up_dense)
        ex["down"], ex["down_scale"] = quantize_mxfp4(down_dense)
    else:
        ex["gate_up"], ex["down"] = gate_up_dense.to(dtype), down_dense.to(dtype)
    return seed_model.fuse_moe_dense(
        {
            "router": randn(experts, hidden).to(dtype),
            "shared_gate": randn(1, hidden).to(dtype),
            "shared_expert.gate_proj": randn(inter, hidden).to(dtype),
            "shared_expert.up_proj": randn(inter, hidden).to(dtype),
            "shared_expert.down_proj": randn(hidden, inter).to(dtype),
            "experts": ex,
        }
    )


def fake_model(cfg, layer: dict, dtype: torch.dtype) -> SimpleNamespace:
    """A minimal stand-in self for calling `Model.moe` as an unbound method.

    `moe` on the torch path reads only `self.cfg`, `self.layers[i]`, `self.dtype`,
    `self.expert_range` and the two routed-expert methods, so a SimpleNamespace duck-typing
    those exercises the production code path without constructing a full Model (weight
    loading, embeddings, ...). These tests are CPU-only, where `mxfp4_gemv.available` is
    False, so `moe` always takes the grouped torch branch here and the fused kernels have
    their own file.

    `expert_range` is the whole axis: this file is about the vectorized routing, and the
    expert-parallel narrowing of that range has its own tests in test_tensor_parallel.py.
    """
    experts = layer["experts"]["gate_up"].shape[0]
    fake_self = SimpleNamespace(cfg=cfg, layers=[layer], dtype=dtype, expert_range=(0, experts))
    for name in ("expert_weights_batch", "_routed_grouped"):
        method = getattr(seed_model.Model, name)
        setattr(fake_self, name, partial(method, fake_self))
    return fake_self


def moe_via_model(cfg, layer: dict, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Call the real `Model.moe` on a fake self, allocating its own accumulator."""
    return seed_model.Model.moe(fake_model(cfg, layer, dtype), 0, x)


@pytest.mark.parametrize("mxfp4", [False, True])
@pytest.mark.parametrize(
    "seed,tokens,hidden,inter,experts,top_k",
    [
        (0, 1, 32, 32, 4, 2),  # single token, degenerate batch dim
        (1, 5, 32, 32, 4, 2),  # more tokens than experts: most/all experts touched
        (2, 17, 64, 32, 8, 3),  # top_k > 1, uneven per-expert counts
        (3, 40, 64, 32, 6, 1),  # top_k = 1
        (4, 9, 32, 32, 8, 8),  # top_k = experts: every expert touched by every token
    ],
)  # hidden and inter must be multiples of BLOCK=32 for the mxfp4 quantizer above
def test_vectorized_moe_matches_old_per_expert_loop(
    seed: int, tokens: int, hidden: int, inter: int, experts: int, top_k: int, mxfp4: bool
) -> None:
    dtype = torch.float32
    cfg = SimpleNamespace(hidden=hidden, top_k=top_k)
    layer = build_layer(seed, hidden, inter, experts, mxfp4=mxfp4, dtype=dtype)
    x = torch.randn(1, tokens, hidden, generator=torch.Generator().manual_seed(seed + 1000))

    want = old_moe(cfg, layer, x, dtype)
    got = moe_via_model(cfg, layer, x, dtype)

    assert got.shape == want.shape
    assert torch.isfinite(got).all()
    # 1e-4, not the 1e-5 this asserted before `fuse_moe_dense`, and only because of the shared
    # gate. Concatenating a projection onto a wider one leaves each output column's dot product
    # the same set of products, but BLAS reduces a 1-column GEMV and a 7-column GEMM in
    # different orders, so the gate logit moves by a few 1e-6 relative and the sigmoid carries
    # that into every hidden channel of the shared expert's contribution. The router's own
    # columns stay bit-exact, which is what `test_fused_moe_dense_keeps_the_routing_bit_exact`
    # pins, so routing cannot change; this is a magnitude nudge far below bf16's epsilon, in a
    # test that runs in float32 to make it visible at all.
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)

"""Regression test: `Model._routed_fused` must not compute the Triton `order` permutation at
prefill token counts.

`mxfp4_gemv.fused_expert_order`'s kernel grid is one program with `ASSIGNMENTS` (`t * top_k`)
as a `tl.constexpr`. Decode calls it at a small, batch-bounded `t` that repeats step to step,
so the compiled kernel is reused. A prefill chunk's `t` varies call to call and can reach
`max_seq * top_k` (production: up to ~32k assignments in one program), so calling the same
path there means a Triton recompile per distinct prefill length plus one huge single-program
launch. The fix bounds the fused-order path to `t <= max_batch` (decode shapes); prefill takes
`order=None`, the pre-existing arrival-order path.

Hermetic CPU test: `_routed_fused` is called on a `SimpleNamespace` fake `self` (only the
attributes it reads are duck-typed, same pattern as `test_moe_vectorize.py`'s `fake_model`),
and `mxfp4_gemv.fused_expert_order`/`fused_moe`/`dedup_available` are monkeypatched so no
Triton or GPU is required. What's under test is the dispatch decision (was `fused_expert_order`
called at all), not kernel arithmetic.

Run with a python that has torch (see seed_tests/test_seed_parity.py for the recipe):
    /tmp/torchenv/bin/python -m pytest \
        examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/seed_tests/test_moe_fused_routing_dispatch.py \
        -p no:cacheprovider --no-cov
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import model as seed_model  # noqa: E402
import mxfp4_gemv  # noqa: E402

HIDDEN = 8
TOP_K = 2
EXPERTS = 4


def fake_self(max_batch: int) -> SimpleNamespace:
    """A minimal stand-in `self` for calling `Model._routed_fused` as an unbound method.

    `_routed_fused` on the path exercised here (moe_bw off, dedup unavailable) reads only
    `self.cfg`, `self.max_batch`, `self.moe_inter_scratch`, `self.moe_dedup_y_scratch`,
    `self.moe_route_cursor`, `self.layers`, and `self.expert_range`.
    """
    cfg = SimpleNamespace(top_k=TOP_K, experts=EXPERTS)
    return SimpleNamespace(
        cfg=cfg,
        max_batch=max_batch,
        moe_bw=False,
        moe_hip=False,
        moe_inter_scratch=[torch.empty(0, HIDDEN)],
        moe_dedup_y_scratch=[torch.empty(0, HIDDEN)],
        moe_route_cursor=[torch.empty(EXPERTS, dtype=torch.int32)],
        layers=[{"experts": {"gate_up_scale": torch.empty(0)}}],
        expert_range=(0, EXPERTS),
    )


def run_routed_fused(monkeypatch: pytest.MonkeyPatch, t: int, max_batch: int) -> dict[str, int]:
    """Call `_routed_fused` with the fused kernels stubbed out; return call counts."""
    calls = {"order": 0, "moe": 0}

    def fake_order(a_expert: torch.Tensor, num_experts: int, *, cursor=None) -> torch.Tensor:
        calls["order"] += 1
        return torch.arange(a_expert.numel(), dtype=torch.int32)

    def fake_moe(x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        calls["moe"] += 1
        return torch.zeros(x.shape[0], HIDDEN, dtype=x.dtype)

    monkeypatch.setattr(mxfp4_gemv, "fused_expert_order", fake_order)
    monkeypatch.setattr(mxfp4_gemv, "fused_moe", fake_moe)
    monkeypatch.setattr(mxfp4_gemv, "dedup_available", lambda device: False)

    self = fake_self(max_batch)
    h = torch.randn(t, HIDDEN)
    top_i = torch.randint(0, EXPERTS, (t, TOP_K))
    top_w = torch.rand(t, TOP_K)

    got = seed_model.Model._routed_fused(self, 0, h, top_i, top_w, None)
    assert got.shape == (t, HIDDEN)
    return calls


@pytest.mark.parametrize(
    "t,max_batch,expect_order",
    [
        (1, 48, True),  # smallest decode shape
        (48, 48, True),  # exactly at the max_batch boundary: still decode-sized
        (49, 48, False),  # one past max_batch: prefill-shaped, must skip `order`
        (2048, 48, False),  # a real prefill chunk
    ],
)
def test_fused_order_only_computed_for_decode_sized_batches(
    monkeypatch: pytest.MonkeyPatch, t: int, max_batch: int, expect_order: bool
) -> None:
    calls = run_routed_fused(monkeypatch, t, max_batch)
    assert calls["moe"] == 1, "fused_moe must still run the routed-expert GEMV either way"
    assert calls["order"] == (1 if expect_order else 0)


def test_fused_order_disabled_via_env_var_even_at_decode_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEED_FUSED_MOE_ROUTING", "0")
    calls = run_routed_fused(monkeypatch, t=8, max_batch=48)
    assert calls["order"] == 0

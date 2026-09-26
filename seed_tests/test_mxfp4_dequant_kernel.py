"""Hermetic CPU regression test: the bandwidth-optimized `dequant_mxfp4` (int32 gather indices,
broadcast scale) vs the old implementation it replaced, on random synthetic packed/scale tensors.

`old_dequant_mxfp4` below is a verbatim copy of the pre-optimization implementation (int64 gather
indices via `.long()`, `repeat_interleave` to materialize the full-size scale tensor), kept only
as the regression oracle for this test. See mxfp4.py's docstring and
resources/skills/serving-systems/references/platforms/rocm/aiter.md ("Dequant-to-bf16-scratch at
prefill M") for the roofline finding that motivated the rewrite.

Run with a python that has torch (see seed_tests/test_seed_parity.py for the recipe):
    /tmp/torchenv/bin/python -m pytest examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/seed_tests/test_mxfp4_dequant_kernel.py -p no:cacheprovider --no-cov
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mxfp4 import _FP4_VALUES, BLOCK, dequant_mxfp4  # noqa: E402


def old_dequant_mxfp4(
    packed: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Verbatim copy of the pre-optimization `dequant_mxfp4` (the regression oracle)."""
    lut = torch.tensor(_FP4_VALUES, dtype=torch.float32, device=packed.device)
    lo = lut[(packed & 0xF).long()]
    hi = lut[(packed >> 4).long()]
    values = torch.stack([lo, hi], dim=-1).flatten(-2)
    scales = torch.exp2(scale.float() - 127.0).repeat_interleave(BLOCK, dim=-1)
    return (values * scales).to(dtype)


def random_packed_and_scale(
    shape: tuple[int, ...], k: int, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random synthetic packed nibbles and e8m0 scales for leading `shape`, inner dim `k`."""
    g = torch.Generator().manual_seed(seed)
    packed = torch.randint(0, 256, (*shape, k // 2), dtype=torch.uint8, generator=g)
    scale = torch.randint(100, 155, (*shape, k // BLOCK), dtype=torch.uint8, generator=g)
    return packed, scale


SHAPES = [
    ((), 32),  # single block, no leading dims
    ((8,), 64),  # 1D leading dim (a single expert's weight row), 2 blocks
    ((5, 16), 128),  # 2D leading dims (a batch of experts x rows), like ex["gate_up"][idx]
    ((3, 1, 40), 4096),  # a leading dim of size 1, and a large K matching the real gate_up K
]


@pytest.mark.parametrize("shape,k", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_new_dequant_matches_old_dequant(
    shape: tuple[int, ...], k: int, dtype: torch.dtype
) -> None:
    packed, scale = random_packed_and_scale(shape, k, seed=hash((shape, k, dtype)) & 0xFFFFFFFF)
    old = old_dequant_mxfp4(packed, scale, dtype)
    new = dequant_mxfp4(packed, scale, dtype)
    assert new.shape == old.shape
    assert new.dtype == old.dtype
    assert torch.equal(new, old), "new dequant must be bit-exact with the old implementation"


def test_new_dequant_matches_old_dequant_including_nan_scale() -> None:
    """e8m0 255 is NaN; both implementations must propagate it identically."""
    packed, scale = random_packed_and_scale((4,), 64, seed=7)
    scale[..., 0] = 255
    old = old_dequant_mxfp4(packed, scale, torch.float32)
    new = dequant_mxfp4(packed, scale, torch.float32)
    assert torch.equal(new.isnan(), old.isnan())
    finite = ~old.isnan()
    assert torch.equal(new[finite], old[finite])


def test_new_dequant_index_dtype_is_narrower_than_int64() -> None:
    """Regression guard for the specific waste this rewrite removes: no int64 gather index."""
    packed = torch.randint(0, 256, (4, 64), dtype=torch.uint8)
    assert (packed & 0xF).int().dtype == torch.int32
    assert (packed & 0xF).int().element_size() == 4  # half the old .long() index's 8 bytes


def test_new_dequant_does_not_materialize_full_size_scale_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard: no tensor of full output size K should come from expanding `scale`."""
    packed = torch.randint(0, 256, (4, 32), dtype=torch.uint8)  # K = 64
    scale = torch.randint(100, 155, (4, 2), dtype=torch.uint8)
    seen_repeat_interleave = False
    real_repeat_interleave = torch.Tensor.repeat_interleave

    def spy(self: torch.Tensor, *a: object, **kw: object) -> torch.Tensor:
        nonlocal seen_repeat_interleave
        seen_repeat_interleave = True
        return real_repeat_interleave(self, *a, **kw)

    monkeypatch.setattr(torch.Tensor, "repeat_interleave", spy)
    dequant_mxfp4(packed, scale, torch.float32)
    assert not seen_repeat_interleave, "dequant_mxfp4 should broadcast, not repeat_interleave"

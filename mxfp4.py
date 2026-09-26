"""OCP MXFP4 dequantization in plain torch.

A weight of logical shape [..., K] is stored as uint8 [..., K/2] (two fp4 e2m1
values per byte, low nibble = even element, verified against the FP8 source
checkpoint, see reference/README.md) plus uint8 e8m0 scales [..., K/32].
"""

import torch

BLOCK = 32
# fp4 e2m1: bit 3 is the sign, bits 0-2 index the magnitudes below.
_FP4_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def dequant_mxfp4(packed: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Return the dense [..., K] tensor for packed [..., K/2] and scale [..., K/32].

    Two waste sources in a naive torch LUT dequant, per this hardware's measured roofline
    (resources/skills/serving-systems/references/platforms/rocm/aiter.md, "Dequant-to-bf16-scratch
    at prefill M"): int64 gather indices (8 bytes/element just to select a 16-entry LUT), and
    `repeat_interleave` eagerly materializing a full-size fp32 scale tensor. Index with int32
    (the narrowest dtype `Tensor.__getitem__`/`index_select` accept; uint8 is read as a bool
    mask) and broadcast the scale over the block dimension instead of expanding it.
    """
    lut = torch.tensor(_FP4_VALUES, dtype=torch.float32, device=packed.device)
    lo = lut[(packed & 0xF).int()]
    hi = lut[(packed >> 4).int()]
    values = torch.stack([lo, hi], dim=-1).flatten(-2)  # interleave: lo, hi, lo, hi, ...
    *lead, k = values.shape
    blocks = values.reshape(*lead, k // BLOCK, BLOCK)
    scales = torch.exp2(scale.float() - 127.0).unsqueeze(
        -1
    )  # e8m0 (255 = NaN, ignored); broadcasts
    return (blocks * scales).reshape(*lead, k).to(dtype)

"""`decode_attn_splitk`'s length-aware split-K kernel against `test_paged_attn`'s dense oracle.

    # real kernel, needs an accelerator
    python -m pytest .../seed_tests/test_decode_attn_splitk.py -p no:cacheprovider --no-cov

    # logic only, no GPU: Triton's reference interpreter, on CPU tensors
    TRITON_INTERPRET=1 python -m pytest ... -p no:cacheprovider --no-cov

`TILE` is patched to 16 (four 4-token blocks) so tiny cases span several tiles, and
`num_splits` is swept so lanes get empty, partial and multi-tile splits.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import decode_attn_splitk  # noqa: E402
from test_paged_attn import FP32_TOL, make_case, reference_decode, relative_error  # noqa: E402

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"

NEEDS_KERNEL = pytest.mark.skipif(
    not decode_attn_splitk.HAVE_TRITON or not (torch.cuda.is_available() or INTERPRET),
    reason="needs triton plus either an accelerator or TRITON_INTERPRET=1",
)


@pytest.mark.parametrize(
    ("lanes", "kv_heads", "max_tokens", "cus", "want"),
    [
        (16, 1, 16384, 228, 29),  # ceil(456 / 16)
        (48, 1, 16384, 228, 10),
        (1, 1, 16384, 228, 64),  # MAX_SPLITS
        (1, 1, 64, 228, 2),  # one split per 32-token tile of the widest lane
        (512, 1, 16384, 228, 1),
    ],
)
def test_num_splits(lanes, kv_heads, max_tokens, cus, want, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(decode_attn_splitk, "TILE", 32)
    monkeypatch.setattr(decode_attn_splitk, "SPLITS_PER_CU", 2.0)
    monkeypatch.setattr(decode_attn_splitk, "MAX_SPLITS", 64)
    assert decode_attn_splitk._num_splits(lanes, kv_heads, max_tokens, cus) == want


def _lanes() -> list[dict]:
    return [
        {"block_ids": [3], "valid_len": 1},  # length one: most splits empty
        {"block_ids": [5, 6, 7], "valid_len": 10},  # one partial tile
        {"block_ids": [9, 2, 15, 4, 20, 1, 8, 13, 6, 11], "valid_len": 38},  # 3 tiles, fragmented
        {"block_ids": list(range(21, 37)), "valid_len": 64},  # exactly 4 full tiles
    ]


CASES = [
    pytest.param(dict(head_dim=16, kv_heads=1, group=1), id="mha"),
    pytest.param(dict(head_dim=32, kv_heads=2, group=4), id="gqa-kv2-group4"),
    pytest.param(dict(head_dim=16, kv_heads=1, group=8), id="gqa-group8"),
]


@NEEDS_KERNEL
@pytest.mark.parametrize("shape", CASES)
@pytest.mark.parametrize("num_splits", [1, 2, 3, 5])
def test_splitk_matches_reference(shape, num_splits, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(decode_attn_splitk, "TILE", 16)
    case = make_case(block_size=4, lanes=_lanes(), seed=7 + num_splits, **shape)
    got = decode_attn_splitk.decode_attention_paged_splitk(
        case["q"], case["k_pool"], case["v_pool"], case["block_table"], case["block_valid"],
        case["positions"], case["block_size"], case["scale"], num_splits=num_splits,
    )
    want = reference_decode(case)
    assert got.shape == want.shape
    err = relative_error(got, want)
    assert err < FP32_TOL, f"rel_err {err:.3e} >= {FP32_TOL:.3e}"


@NEEDS_KERNEL
def test_model_decode_attention_paged_dispatches_to_splitk(monkeypatch) -> None:  # noqa: ANN001
    import model as seed_model

    monkeypatch.setattr(decode_attn_splitk, "TILE", 16)
    case = make_case(head_dim=16, block_size=4, kv_heads=1, group=2, lanes=_lanes(), seed=11)
    args = (
        case["q"], case["k_pool"], case["v_pool"], case["block_table"],
        case["block_valid"], case["positions"], case["block_size"], case["scale"],
    )
    monkeypatch.setattr(decode_attn_splitk, "available", lambda _dev: False)
    want = seed_model.decode_attention_paged(*args)
    monkeypatch.setattr(decode_attn_splitk, "available", lambda _dev: True)
    monkeypatch.setattr(decode_attn_splitk, "_num_cus", lambda _dev: 4)
    got = seed_model.decode_attention_paged(*args)
    assert relative_error(got, want) < FP32_TOL


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

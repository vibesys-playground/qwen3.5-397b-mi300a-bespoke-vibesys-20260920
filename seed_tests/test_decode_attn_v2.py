"""`decode_attn_v2`'s split-KV ("flash-decoding") kernel against the same independent
dense-causal-attention oracle `test_paged_attn.py` uses for the base kernel, plus its own
`_split_config` unit tests -- `make_case`/`reference_decode` are imported from `test_paged_attn`
rather than duplicated, so both kernels are checked against the exact same oracle.

Run modes, same as `test_paged_attn.py`:

    # real kernel, needs an accelerator
    python -m pytest .../seed_tests/test_decode_attn_v2.py -p no:cacheprovider --no-cov

    # logic only, no GPU: Triton's reference interpreter, on CPU tensors
    TRITON_INTERPRET=1 python -m pytest ... -p no:cacheprovider --no-cov

Every kernel-dependent test here forces a small `MIN_BLOCKS_PER_SPLIT`/`LONG_CONTEXT_TOKENS`
via monkeypatch so tiny test shapes actually exercise `NUM_SPLITS > 1` (the module's real
default threshold, 8k served tokens, is far larger than any shape a CPU-hermetic/interpreter
test can afford) -- both the degenerate `NUM_SPLITS == 1` path and a genuinely split path are
covered, since `_split_config`'s own docstring promises the two are numerically identical.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import decode_attn_v2  # noqa: E402
from test_paged_attn import FP32_TOL, make_case, reference_decode, relative_error  # noqa: E402

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NEEDS_KERNEL = pytest.mark.skipif(
    not decode_attn_v2.HAVE_TRITON or not (torch.cuda.is_available() or INTERPRET),
    reason="needs triton plus either an accelerator or TRITON_INTERPRET=1",
)


def _force_splitting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tiny-shape stand-in for the real ~8k-token threshold: any row with >= 2 blocks splits
    into (at most) 2-block pieces, so a handful of test-sized block tables (4-11 blocks) yield
    `NUM_SPLITS` in [1, 4] depending on the case, covering both the degenerate and real paths.
    """
    monkeypatch.setattr(decode_attn_v2, "LONG_CONTEXT_TOKENS", 0)
    monkeypatch.setattr(decode_attn_v2, "MIN_BLOCKS_PER_SPLIT", 2)
    monkeypatch.setattr(decode_attn_v2, "MAX_SPLITS", 4)


# ---------------------------------------------------------------- _split_config


@pytest.mark.parametrize(
    ("max_blocks", "block_size", "long_ctx", "min_blocks", "max_splits", "want"),
    [
        (8, 4, 8192, 64, 8, (1, 8)),  # under LONG_CONTEXT_TOKENS: no split
        (63, 4, 8192, 64, 8, (1, 63)),  # 252 tok, under LONG_CONTEXT_TOKENS=8192: no split
        (2048, 4, 4096, 64, 8, (8, 256)),  # 2048 // 64 = 32, capped at MAX_SPLITS=8
        (65, 4, 0, 2, 4, (4, 17)),  # ceil(65 / 4) = 17 blocks/split, 4 splits (capped)
    ],
)
def test_split_config(
    max_blocks: int,
    block_size: int,
    long_ctx: int,
    min_blocks: int,
    max_splits: int,
    want: tuple[int, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(decode_attn_v2, "LONG_CONTEXT_TOKENS", long_ctx)
    monkeypatch.setattr(decode_attn_v2, "MIN_BLOCKS_PER_SPLIT", min_blocks)
    monkeypatch.setattr(decode_attn_v2, "MAX_SPLITS", max_splits)
    assert decode_attn_v2._split_config(max_blocks, block_size) == want


def test_split_config_covers_every_block_at_least_once() -> None:
    """`NUM_SPLITS * BLOCKS_PER_SPLIT >= max_blocks` for every config `_split_config` can
    produce -- the split kernel's per-program loop trip count must not silently drop blocks
    past the last split's nominal range (this is the same invariant `test_varlen_prefill_attn.
    py`'s `BLOCKS_PER_ITER` overhang case checks for the prefill kernel)."""
    for max_blocks in (1, 2, 7, 64, 65, 127, 1024, 8191, 8192):
        for block_size in (4, 16):
            num_splits, blocks_per_split = decode_attn_v2._split_config(max_blocks, block_size)
            assert num_splits * blocks_per_split >= max_blocks


# ---------------------------------------------------------------- the kernel


CASES = [
    pytest.param(
        lambda: make_case(
            head_dim=16, block_size=4, kv_heads=1, group=1,
            lanes=[{"block_ids": [3], "valid_len": 1}], seed=1,
        ),
        id="length-one",
    ),
    pytest.param(
        lambda: make_case(
            head_dim=16, block_size=4, kv_heads=1, group=1,
            lanes=[
                {"block_ids": [3], "valid_len": 4},
                {"block_ids": [5, 6, 7], "valid_len": 10},
                {"block_ids": [9, 10], "valid_len": 7},
            ],
            seed=2,
        ),
        id="multi-lane-mixed-lengths",
    ),
    pytest.param(
        lambda: make_case(
            head_dim=16, block_size=4, kv_heads=1, group=1,
            # partial last block: only 1 of 4 tokens in the second block are live
            lanes=[{"block_ids": [4, 8], "valid_len": 5}], seed=3,
        ),
        id="partial-last-block-crosses-boundary",
    ),
    pytest.param(
        lambda: make_case(
            head_dim=16, block_size=4, kv_heads=1, group=1,
            # non-ascending, non-contiguous block ids, "resumed prefix" shape: several full
            # blocks plus a partial tail, exactly what a resumed session's block table looks like
            lanes=[{"block_ids": [9, 2, 15, 4, 20, 1, 8, 13, 6, 11], "valid_len": 38}],
            seed=4,
        ),
        id="resumed-prefix-fragmented-10-blocks",
    ),
    pytest.param(
        lambda: make_case(
            head_dim=32, block_size=4, kv_heads=2, group=4,
            lanes=[
                {"block_ids": [3, 6], "valid_len": 8},
                {"block_ids": [11, 2, 9, 7, 14, 5, 1, 10], "valid_len": 30},
            ],
            seed=5,
        ),
        id="gqa-kv-heads-2-group-4",
    ),
]


@NEEDS_KERNEL
@pytest.mark.parametrize("case_fn", CASES)
def test_decode_attn_v2_matches_independent_reference(
    case_fn, monkeypatch: pytest.MonkeyPatch  # noqa: ANN001
) -> None:
    _force_splitting(monkeypatch)
    case = case_fn()
    got = decode_attn_v2.decode_attention_paged_v2(
        case["q"],
        case["k_pool"],
        case["v_pool"],
        case["block_table"],
        case["block_valid"],
        case["positions"],
        case["block_size"],
        case["scale"],
    )
    want = reference_decode(case)
    assert got.shape == want.shape
    err = relative_error(got, want)
    assert err < FP32_TOL, f"rel_err {err:.3e} >= {FP32_TOL:.3e}"


@NEEDS_KERNEL
@pytest.mark.parametrize("case_fn", CASES)
def test_decode_attn_v2_matches_the_base_kernel_no_split(
    case_fn, monkeypatch: pytest.MonkeyPatch  # noqa: ANN001
) -> None:
    """`NUM_SPLITS == 1` (the module's default at short-context shapes) must reproduce
    `paged_attn.decode_attention_paged`'s own output bit-for-bit in the online-softmax sense
    (same recurrence, same reduction order at `NUM_SPLITS == 1`): the split/reduce split is
    purely an occupancy change, never a numerics change, at the degenerate split count.
    """
    import paged_attn

    case = case_fn()
    args = (
        case["q"], case["k_pool"], case["v_pool"], case["block_table"],
        case["block_valid"], case["positions"], case["block_size"], case["scale"],
    )
    base = paged_attn.decode_attention_paged(*args)
    v2 = decode_attn_v2.decode_attention_paged_v2(*args)
    err = relative_error(v2, base)
    assert err < FP32_TOL, f"rel_err {err:.3e} >= {FP32_TOL:.3e}"


@NEEDS_KERNEL
def test_available_respects_the_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEED_DECODE_ATTN_V2", "0")
    assert not decode_attn_v2.available(torch.device("cuda"))


@NEEDS_KERNEL
def test_model_decode_attention_paged_dispatches_to_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    """`model.decode_attention_paged`'s dispatch, v2 branch vs. the torch fallback -- mirrors
    `test_paged_attn.test_model_decode_attention_paged_dispatches_to_the_kernel`."""
    import model as seed_model

    _force_splitting(monkeypatch)
    case = make_case(
        head_dim=16, block_size=4, kv_heads=1, group=2,
        lanes=[
            {"block_ids": [3], "valid_len": 4},
            {"block_ids": [5, 6, 7], "valid_len": 10},
        ],
        seed=6,
    )
    args = (
        case["q"], case["k_pool"], case["v_pool"], case["block_table"],
        case["block_valid"], case["positions"], case["block_size"], case["scale"],
    )
    monkeypatch.setattr(decode_attn_v2, "available", lambda _dev: False)
    want = seed_model.decode_attention_paged(*args)
    monkeypatch.setattr(decode_attn_v2, "available", lambda _dev: True)
    got = seed_model.decode_attention_paged(*args)
    assert relative_error(got, want) < FP32_TOL


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

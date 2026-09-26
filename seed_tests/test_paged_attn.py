"""`paged_attn`'s fused Triton kernel, and `model.decode_attention_paged`'s dispatch, against an
independent dense-causal-attention reference written directly in this file.

The oracle here is deliberately *not* `model.decode_attention_paged`'s own torch-fallback
branch (that would just test the fallback against itself). Instead every case below gathers
each lane's real, logical KV sequence by walking its block table up to its valid length, and
runs a plain masked-softmax attention over that gather -- independent of both `paged_attn.py`
and `model.py`.

Two ways to run the kernel-dependent tests, as for `test_deltanet_fused.py`:

    # real kernel, needs an accelerator
    python -m pytest .../seed_tests/test_paged_attn.py -p no:cacheprovider --no-cov

    # logic only, no GPU: Triton's reference interpreter, on CPU tensors
    TRITON_INTERPRET=1 python -m pytest ... -p no:cacheprovider --no-cov

`test_model_decode_attention_paged_uses_the_torch_fallback_on_cpu` is the exception: it needs
neither Triton nor CUDA and runs in plain CI with no special invocation, because it is what
lets every other CPU-hermetic decode test in this directory (test_batched_decode.py,
test_tensor_parallel.py, ...) assume `model.decode_attention_paged`'s fallback branch is
correct. Everything else in this module is gated behind `NEEDS_KERNEL` (applied per test, not
as a module-wide `pytestmark`, precisely so that one test escapes the gate).
"""

import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import block_pool  # noqa: E402
import model as seed_model  # noqa: E402
import paged_attn  # noqa: E402

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NEEDS_KERNEL = pytest.mark.skipif(
    not paged_attn.HAVE_TRITON or not (torch.cuda.is_available() or INTERPRET),
    reason="needs triton plus either an accelerator or TRITON_INTERPRET=1",
)

FP32_TOL = 1e-4
"""Looser than rmsnorm/deltanet's 1e-5: this is an online-softmax recurrence over several
blocks, so there is more fp32 reduction-order slack against the reference's one-shot softmax,
even though both are the same fp32 chain with a single rounding at the end."""


def relative_error(got: torch.Tensor, want: torch.Tensor) -> float:
    scale = want.float().abs().max().item()
    return ((got.float() - want.float()).abs().max() / max(scale, 1e-30)).item()


# ---------------------------------------------------------------- fixtures


def make_case(
    *,
    head_dim: int,
    block_size: int,
    kv_heads: int,
    group: int,
    lanes: list[dict],
    seed: int,
    device: torch.device = DEVICE,
) -> dict:
    """Build `(q, k_pool, v_pool, block_table, block_valid, positions)` for one test case.

    `lanes` is a list of `{"block_ids": [...], "valid_len": n}`: `block_ids` are this lane's
    real, resident block ids (never `block_pool.RESERVED_BLOCK`, never assumed contiguous or
    ascending), `valid_len` is the number of live tokens (not necessarily a multiple of
    `block_size`). `block_table`/`block_valid` are built with `block_pool.BlockTable.padded_row`
    and `block_pool.valid_block_count`, the same functions `model.py` uses, so the fixture
    matches the real call convention exactly.
    """
    heads = kv_heads * group
    max_block_id = max(max(spec["block_ids"]) for spec in lanes)
    num_blocks = max_block_id + 1  # ids are [0, num_blocks); id 0 is RESERVED_BLOCK, unused here
    max_blocks_row = max(len(spec["block_ids"]) for spec in lanes)

    gen = torch.Generator().manual_seed(seed)
    k_pool = torch.randn(num_blocks * block_size, kv_heads, head_dim, generator=gen).to(device)
    v_pool = torch.randn(num_blocks * block_size, kv_heads, head_dim, generator=gen).to(device)
    q = torch.randn(len(lanes), heads, 1, head_dim, generator=gen).to(device)

    block_table = torch.tensor(
        [
            block_pool.BlockTable(blocks=list(spec["block_ids"])).padded_row(max_blocks_row)
            for spec in lanes
        ],
        dtype=torch.int32,
        device=device,
    )
    block_valid = torch.tensor(
        [block_pool.valid_block_count(spec["valid_len"], block_size) for spec in lanes],
        dtype=torch.int32,
        device=device,
    )
    positions = torch.tensor(
        [spec["valid_len"] - 1 for spec in lanes], dtype=torch.int32, device=device
    )

    return {
        "q": q,
        "k_pool": k_pool,
        "v_pool": v_pool,
        "block_table": block_table,
        "block_valid": block_valid,
        "positions": positions,
        "block_size": block_size,
        "scale": head_dim**-0.5,
        "lanes": lanes,
        "kv_heads": kv_heads,
        "group": group,
    }


def reference_decode(case: dict) -> torch.Tensor:
    """Independent dense-causal-attention oracle: gather the real logical KV sequence per lane
    via its block table, up to its valid length, and do a straightforward masked-softmax
    attention. Written from scratch here, not derived from `paged_attn.py` or `model.py`.
    """
    q, k_pool, v_pool = case["q"], case["k_pool"], case["v_pool"]
    kv_heads, group, block_size = case["kv_heads"], case["group"], case["block_size"]
    lanes = case["lanes"]
    b, heads, _, head_dim = q.shape
    dev = q.device

    out = torch.zeros(b, heads, head_dim, dtype=torch.float32, device=dev)
    for lane_idx, lane in enumerate(lanes):
        length = lane["valid_len"]
        rows = []
        for tok in range(length):
            blk, off = divmod(tok, block_size)
            rows.append(lane["block_ids"][blk] * block_size + off)
        rows_t = torch.tensor(rows, dtype=torch.long, device=dev)
        k = k_pool[rows_t].float()  # [length, kv_heads, head_dim]
        v = v_pool[rows_t].float()
        for kvh in range(kv_heads):
            qh = q[lane_idx, kvh * group : (kvh + 1) * group, 0].float()  # [group, head_dim]
            scores = (qh @ k[:, kvh, :].T) * case["scale"]  # [group, length]
            probs = scores.softmax(dim=-1)
            out[lane_idx, kvh * group : (kvh + 1) * group] = probs @ v[:, kvh, :]
    return out.reshape(b, heads, 1, head_dim)


# ---------------------------------------------------------------- the cases


def case_single_lane_one_full_block() -> dict:
    return make_case(
        head_dim=16,
        block_size=4,
        kv_heads=1,
        group=1,
        lanes=[{"block_ids": [3], "valid_len": 4}],
        seed=1,
    )


def case_multi_lane_mixed_lengths() -> dict:
    return make_case(
        head_dim=16,
        block_size=4,
        kv_heads=1,
        group=1,
        lanes=[
            {"block_ids": [3], "valid_len": 4},
            {"block_ids": [5, 6, 7], "valid_len": 10},
            {"block_ids": [9, 10], "valid_len": 7},
        ],
        seed=2,
    )


def case_partial_last_block() -> dict:
    return make_case(
        head_dim=16,
        block_size=4,
        kv_heads=1,
        group=1,
        # 2 blocks resident, but valid_len=5 means only token 0 of the second block is live
        lanes=[{"block_ids": [4, 8], "valid_len": 5}],
        seed=3,
    )


def case_fragmented_block_table() -> dict:
    return make_case(
        head_dim=16,
        block_size=4,
        kv_heads=1,
        group=1,
        # non-ascending, non-contiguous block ids, plus a partial last block
        lanes=[{"block_ids": [9, 2, 15], "valid_len": 11}],
        seed=4,
    )


def case_gqa_kv_heads_two_group_four() -> dict:
    return make_case(
        head_dim=32,
        block_size=4,
        kv_heads=2,
        group=4,
        lanes=[
            {"block_ids": [3, 6], "valid_len": 8},
            {"block_ids": [11, 2, 9], "valid_len": 9},  # fragmented + partial, under GQA
        ],
        seed=5,
    )


def case_gqa_kv_heads_one_group_two() -> dict:
    return make_case(
        head_dim=16,
        block_size=4,
        kv_heads=1,
        group=2,
        lanes=[{"block_ids": [5, 12], "valid_len": 6}],
        seed=6,
    )


CASES = [
    pytest.param(case_single_lane_one_full_block, id="single-lane-one-full-block"),
    pytest.param(case_multi_lane_mixed_lengths, id="multi-lane-mixed-lengths"),
    pytest.param(case_partial_last_block, id="partial-last-block"),
    pytest.param(case_fragmented_block_table, id="fragmented-block-table"),
    pytest.param(case_gqa_kv_heads_two_group_four, id="gqa-kv-heads-2-group-4"),
    pytest.param(case_gqa_kv_heads_one_group_two, id="gqa-kv-heads-1-group-2"),
]


# ---------------------------------------------------------------- the kernel


@NEEDS_KERNEL
@pytest.mark.parametrize("case_fn", CASES)
def test_paged_attn_kernel_matches_independent_reference(case_fn) -> None:  # noqa: ANN001
    case = case_fn()
    got = paged_attn.decode_attention_paged(
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
@pytest.mark.parametrize("static_loop", [False, True], ids=["valid-blocks-loop", "static-loop"])
def test_paged_attn_kernel_short_lanes_in_a_wide_table(
    static_loop: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The serving shape: each lane's row is `max_blocks_per_lane` wide (1024 at max_seq
    16384) but holds a few live blocks. The kernel loops over the lane's own valid block count
    by default and over the whole row under `SEED_PAGED_ATTN_STATIC_LOOP=1`; both must match."""
    monkeypatch.setattr(paged_attn, "STATIC_LOOP", static_loop)
    case = make_case(
        head_dim=16,
        block_size=4,
        kv_heads=1,
        group=2,
        lanes=[
            {"block_ids": [7], "valid_len": 1},
            {"block_ids": [3, 12, 5], "valid_len": 10},
            {"block_ids": [20, 21], "valid_len": 8},
        ],
        seed=7,
    )
    wide = 64
    bt = case["block_table"]
    pad = torch.full((bt.shape[0], wide - bt.shape[1]), block_pool.RESERVED_BLOCK, dtype=bt.dtype)
    case["block_table"] = torch.cat([bt, pad.to(bt.device)], dim=1).contiguous()
    got = paged_attn.decode_attention_paged(
        case["q"],
        case["k_pool"],
        case["v_pool"],
        case["block_table"],
        case["block_valid"],
        case["positions"],
        case["block_size"],
        case["scale"],
    )
    err = relative_error(got, reference_decode(case))
    assert err < FP32_TOL, f"rel_err {err:.3e} >= {FP32_TOL:.3e}"


@NEEDS_KERNEL
def test_available_respects_the_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEED_FUSED_PAGED_ATTN", "0")
    assert not paged_attn.available(torch.device("cuda"))


@NEEDS_KERNEL
def test_model_decode_attention_paged_dispatches_to_the_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`model.decode_attention_paged`'s dispatch itself, both arms, against each other --
    mirrors test_rmsnorm_fused.test_model_rmsnorm_dispatches_to_the_fused_kernel."""
    case = case_multi_lane_mixed_lengths()
    args = (
        case["q"],
        case["k_pool"],
        case["v_pool"],
        case["block_table"],
        case["block_valid"],
        case["positions"],
        case["block_size"],
        case["scale"],
    )
    monkeypatch.setattr(paged_attn, "available", lambda _dev: False)
    want = seed_model.decode_attention_paged(*args)
    monkeypatch.setattr(paged_attn, "available", lambda _dev: True)
    got = seed_model.decode_attention_paged(*args)

    assert relative_error(got, want) < FP32_TOL


# ---------------------------------------------------------------- the torch fallback (no gate)


def test_model_decode_attention_paged_uses_the_torch_fallback_on_cpu() -> None:
    """No accelerator, no `TRITON_INTERPRET`: `model.decode_attention_paged` must take its
    pure-torch branch and match the same independent reference the kernel tests use above.
    Runs in plain CI with no special environment, unlike every other test in this file.
    """
    cpu = torch.device("cpu")
    assert not paged_attn.available(cpu), "sanity: this must actually exercise the fallback"

    case = make_case(
        head_dim=16,
        block_size=4,
        kv_heads=2,
        group=2,
        lanes=[
            {"block_ids": [3, 6], "valid_len": 8},
            {"block_ids": [11, 2, 9], "valid_len": 9},  # fragmented + partial last block
        ],
        seed=100,
        device=cpu,
    )
    got = seed_model.decode_attention_paged(
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

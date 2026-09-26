"""Regression test: `model.verify_attention_paged`'s torch fallback must bound its K/V
gather to the batch's actual occupancy (`block_valid`'s max), not the block table's fixed
`max_blocks_per_lane` width.

Before the fix, the gather always spanned the full block-table width (sized for `max_seq`),
even when every lane's live context is far shorter -- at b48/16k context that reads ~0.8 GB
of unwritten K/V per layer. `block_valid` (unused before the fix, despite being passed in)
says how many of a lane's block-table columns are actually live; the fix reads it to slice
`block_table` down to the batch's longest lane before gathering.

Hermetic CPU test, no GPU or Triton: a small random K/V pool, several lanes at different
occupancies, one lane much shorter than the block table's fixed width. `old_verify_attention_paged`
below is a verbatim copy of the pre-fix body (the full-width gather), kept only as this test's
correctness oracle -- the fix must not change the result, only how much it reads to get there.

    /tmp/torchenv/bin/python -m pytest \\
        examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/seed_tests/test_verify_attention_paged_bounded_gather.py \\
        -p no:cacheprovider --no-cov
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import block_pool  # noqa: E402
import model as seed_model  # noqa: E402

BLOCK_SIZE = 4
MAX_BLOCKS_PER_LANE = 20  # stands in for a long max_seq's fixed block-table width
HEADS, KV_HEADS, HEAD_DIM = 4, 2, 8
T = 3  # MTP round width (mtp.k + 1)


def old_verify_attention_paged(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_table: torch.Tensor,
    block_valid: torch.Tensor,  # noqa: ARG001 - unused, same as the pre-fix body
    base_positions: torch.Tensor,
    block_size: int,
    scale: float,
) -> torch.Tensor:
    """Verbatim copy of `verify_attention_paged`'s pre-fix body: always gathers the full
    `block_table.shape[1]` width. This test's correctness oracle only."""
    b, heads, t, head_dim = q.shape
    kv_heads = k_pool.shape[1]
    group = heads // kv_heads
    max_blocks = block_table.shape[1]
    dev = q.device

    tok = torch.arange(block_size, device=dev)
    logical_pos = (
        torch.arange(max_blocks, device=dev)[:, None] * block_size + tok[None, :]
    ).reshape(-1)
    query_pos = base_positions[:, None] + torch.arange(t, device=dev)[None, :]
    live = logical_pos[None, None, :] <= query_pos[:, :, None]
    rows = (block_table[:, :, None] * block_size + tok[None, None, :]).reshape(b, -1)
    safe_rows = torch.where(live[:, -1, :], rows, torch.zeros_like(rows))

    k = k_pool[safe_rows].permute(0, 2, 1, 3).float()
    v = v_pool[safe_rows].permute(0, 2, 1, 3).float()
    grouped_q = q.reshape(b, kv_heads, group, t, head_dim).float()

    scores = torch.einsum("bkgtd,bksd->bkgts", grouped_q, k) * scale
    probs = scores.masked_fill(~live[:, None, None, :, :], float("-inf")).softmax(-1)
    out = torch.einsum("bkgts,bksd->bkgtd", probs, v)
    return out.reshape(b, heads, t, head_dim).to(q.dtype)


def build_batch(seed: int, base_positions: list[int]) -> dict[str, torch.Tensor]:
    """A random pool plus a `[B, MAX_BLOCKS_PER_LANE]` block table, one lane per
    `base_positions` entry, right-padded with `block_pool.RESERVED_BLOCK` past each lane's
    own live block count -- the same shape `Model._paged_read_buffers` builds."""
    g = torch.Generator().manual_seed(seed)
    b = len(base_positions)
    num_blocks = MAX_BLOCKS_PER_LANE + 1  # +1 keeps RESERVED_BLOCK (id 0) unused by real data
    k_pool = torch.randn(num_blocks * BLOCK_SIZE, KV_HEADS, HEAD_DIM, generator=g)
    v_pool = torch.randn(num_blocks * BLOCK_SIZE, KV_HEADS, HEAD_DIM, generator=g)

    valid = [block_pool.valid_block_count(p + T, BLOCK_SIZE) for p in base_positions]
    rows = [
        list(range(1, n + 1)) + [block_pool.RESERVED_BLOCK] * (MAX_BLOCKS_PER_LANE - n)
        for n in valid
    ]
    return {
        "q": torch.randn(b, HEADS, T, HEAD_DIM, generator=g),
        "k_pool": k_pool,
        "v_pool": v_pool,
        "block_table": torch.tensor(rows, dtype=torch.int64),
        "block_valid": torch.tensor(valid, dtype=torch.int32),
        "base_positions": torch.tensor(base_positions, dtype=torch.int64),
    }


def einsum_gathered_width(monkeypatch: pytest.MonkeyPatch, batch: dict) -> tuple[torch.Tensor, int]:
    """Call the real `verify_attention_paged`, spying on `torch.einsum` to capture the
    gathered K/V width `S` from the `"bkgtd,bksd->bkgts"` scores einsum."""
    real_einsum = torch.einsum
    widths: list[int] = []

    def spy_einsum(equation: str, *operands: torch.Tensor) -> torch.Tensor:
        if equation == "bkgtd,bksd->bkgts":
            widths.append(operands[1].shape[-2])  # k's S axis
        return real_einsum(equation, *operands)

    monkeypatch.setattr(torch, "einsum", spy_einsum)
    out = seed_model.verify_attention_paged(
        batch["q"],
        batch["k_pool"],
        batch["v_pool"],
        batch["block_table"],
        batch["block_valid"],
        batch["base_positions"],
        BLOCK_SIZE,
        HEAD_DIM**-0.5,
    )
    assert widths, "verify_attention_paged never hit the scores einsum"
    return out, widths[0]


def test_gather_bounded_to_batch_max_valid_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    # Lane 0 is short (little live context); lanes 1-2 are much longer, close to the fixed
    # table width -- the bound must track the longest lane, not the shortest or the fixed
    # per-lane ceiling.
    base_positions = [1, 60, 55]
    batch = build_batch(seed=0, base_positions=base_positions)

    got, gathered_s = einsum_gathered_width(monkeypatch, batch)
    want = old_verify_attention_paged(
        batch["q"],
        batch["k_pool"],
        batch["v_pool"],
        batch["block_table"],
        batch["block_valid"],
        batch["base_positions"],
        BLOCK_SIZE,
        HEAD_DIM**-0.5,
    )

    assert got.shape == want.shape
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)

    full_width = MAX_BLOCKS_PER_LANE * BLOCK_SIZE
    expected_bound = int(batch["block_valid"].max().item()) * BLOCK_SIZE
    assert expected_bound < full_width, "test scenario must actually exercise a smaller bound"
    assert gathered_s == expected_bound
    assert gathered_s < full_width


def test_gather_matches_full_width_when_every_lane_is_near_max(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Edge case: when the busiest lane's occupancy fills the table, the bound degrades to
    the old full-width behavior rather than clipping anything."""
    base_positions = [MAX_BLOCKS_PER_LANE * BLOCK_SIZE - T, 3, 5]
    batch = build_batch(seed=1, base_positions=base_positions)

    got, gathered_s = einsum_gathered_width(monkeypatch, batch)
    want = old_verify_attention_paged(
        batch["q"],
        batch["k_pool"],
        batch["v_pool"],
        batch["block_table"],
        batch["block_valid"],
        batch["base_positions"],
        BLOCK_SIZE,
        HEAD_DIM**-0.5,
    )

    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
    assert gathered_s == MAX_BLOCKS_PER_LANE * BLOCK_SIZE


def test_accelerator_path_does_not_read_a_device_scalar_on_the_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Graph replay must read its context length on device, not freeze warmup's width."""
    batch = build_batch(seed=2, base_positions=[1, 40, 60])
    calls = []

    monkeypatch.setattr(seed_model.paged_attn, "available", lambda _device: True)

    def fake_decode(q, *args):  # noqa: ANN001, ANN002, ANN202
        calls.append((q, args))
        return q

    monkeypatch.setattr(seed_model.paged_attn, "decode_attention_paged", fake_decode)

    def forbidden_item(_self):  # noqa: ANN001, ANN202
        raise AssertionError("captured verify converted a device scalar to a Python integer")

    monkeypatch.setattr(torch.Tensor, "item", forbidden_item)
    got = seed_model.verify_attention_paged(
        batch["q"],
        batch["k_pool"],
        batch["v_pool"],
        batch["block_table"],
        batch["block_valid"],
        batch["base_positions"],
        BLOCK_SIZE,
        HEAD_DIM**-0.5,
    )
    torch.testing.assert_close(got, batch["q"])
    assert len(calls) == T
    for step, (q, args) in enumerate(calls):
        assert q.shape[2] == 1
        torch.testing.assert_close(args[4], batch["base_positions"] + step)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

"""Regression: the MTP row helpers read their index vectors as dense arrays.

`graph_mtp.draft_round_step` passes columns of `[B, t]` buffers (`token_matrix[:, :1]` as the
step-0 token ids, `write_rows[:, s]` as the step-s KV rows). Before the fix the Triton kernels
read element `j` of the parent buffer for row `j`, so the captured draft embedded the wrong
step-0 tokens and wrote draft K/V into other lanes' rows (B96 acceptance 1.43 vs 1.99 for the
eager draft). Needs an accelerator.

    python -m pytest seed_tests/test_mtp_row_helpers_strided.py -p no:cacheprovider
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mtp_dense_moe  # noqa: E402

NEEDS_GPU = pytest.mark.skipif(
    not (mtp_dense_moe.HAVE_TRITON and torch.cuda.is_available()), reason="needs an accelerator"
)


def _strided_index(gen: torch.Generator, b: int, t: int, col: int, high: int) -> torch.Tensor:
    grid = torch.randperm(high, generator=gen)[: b * t].view(b, t)  # distinct: no write races
    return grid.cuda()[:, col]  # [b], stride t


@NEEDS_GPU
@pytest.mark.parametrize("seed", range(6))
def test_gather_rows_strided_index(seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    b, t, cols = int(torch.randint(1, 97, (1,), generator=gen)), int(torch.randint(2, 5, (1,), generator=gen)), 300
    col = int(torch.randint(0, t, (1,), generator=gen))
    src = torch.randn(512, cols, generator=gen).cuda()
    rows = _strided_index(gen, b, t, col, 512)
    assert not rows.is_contiguous() or b == 1
    out = torch.empty(b, cols, device="cuda")
    mtp_dense_moe.gather_rows(src, rows, out)
    assert torch.equal(out, src[rows])
    ids = rows[:, None]  # the draft's `token_matrix[:, :1]` shape
    emb = torch.empty(b, 1, cols, device="cuda")
    mtp_dense_moe.gather_embeddings(src, ids, emb)
    assert torch.equal(emb[:, 0], src[rows])


@NEEDS_GPU
@pytest.mark.parametrize("seed", range(6))
def test_scatter_active_first_dim_strided_index(seed: int) -> None:
    gen = torch.Generator().manual_seed(100 + seed)
    b, t = int(torch.randint(1, 97, (1,), generator=gen)), int(torch.randint(2, 5, (1,), generator=gen))
    col = int(torch.randint(0, t, (1,), generator=gen))
    dst = torch.randn(1024, 2, 8, generator=gen).cuda()
    want = dst.clone()
    rows = _strided_index(gen, b, t, col, 1024)
    src = torch.randn(b, 2, 8, generator=gen).cuda()
    active_grid = (torch.rand(b, t, generator=gen) > 0.3).cuda()
    active = active_grid[:, col]
    mtp_dense_moe.scatter_active_first_dim(dst, rows, src, active)
    want[rows[active]] = src[active]
    assert torch.equal(dst, want)


@NEEDS_GPU
def test_scatter_time_active_rows_strided_index() -> None:
    gen = torch.Generator().manual_seed(7)
    b, t, cols = 40, 4, 64
    dst = torch.randn(256, cols, generator=gen).cuda()
    want = dst.clone()
    rows = _strided_index(gen, b, t, 2, 256)
    time = torch.randint(0, t, (b, 3), generator=gen).cuda()[:, 1]
    active = (torch.rand(b, 2, generator=gen) > 0.2).cuda()[:, 0]
    src = torch.randn(b, t, cols, generator=gen).cuda()
    mtp_dense_moe.scatter_time_active_rows(dst, rows, src, time, active)
    sel = src[torch.arange(b, device="cuda"), time]
    want[rows[active]] = sel[active]
    assert torch.equal(dst, want)

"""`rmsnorm_fused` against the torch chain it replaces, and `model.rmsnorm`'s dispatch.

The oracle is `model.rmsnorm` itself -- the same posture test_deltanet_fused.py takes toward
`model.gated_rmsnorm`. Same two ways to run it:

    # real kernel, needs an accelerator
    python -m pytest .../seed_tests/test_rmsnorm_fused.py -p no:cacheprovider --no-cov

    # logic only, no GPU: Triton's reference interpreter, on CPU tensors
    TRITON_INTERPRET=1 python -m pytest ... -p no:cacheprovider --no-cov
"""

import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import model as seed_model  # noqa: E402
import rmsnorm_fused  # noqa: E402
from test_seed_parity import build_hf, write_checkpoint  # noqa: E402

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pytestmark = pytest.mark.skipif(
    not rmsnorm_fused.HAVE_TRITON or not (torch.cuda.is_available() or INTERPRET),
    reason="needs triton plus either an accelerator or TRITON_INTERPRET=1",
)

FP32_TOL = 1e-5
BF16_TOL = 5e-2 if INTERPRET else 1e-2
"""Same bar seed_tests/test_deltanet_fused.py holds gated_rmsnorm to; see its module docstring
for why the interpreter arm gets a looser one (an extra rounding bias, not a real regression)."""


def tolerance(dtype: torch.dtype) -> float:
    return BF16_TOL if dtype is torch.bfloat16 else FP32_TOL


def relative_error(got: torch.Tensor, want: torch.Tensor) -> float:
    scale = want.float().abs().max().item()
    return ((got.float() - want.float()).abs().max() / max(scale, 1e-30)).item()


# ---------------------------------------------------------------- the kernel, generic shapes

REAL_SHAPES = [
    # (label, rows, heads, cols) at the deployed batch (48) -- heads folded into rows below,
    # since rmsnorm_fused takes any [..., cols] shape, not [rows, heads, cols].
    ("in_norm/post_norm/final_norm", 48, 1, 4096),
    ("q_norm", 48, 8, 256),
    ("k_norm", 48, 1, 256),
]
GENERIC_SHAPES = [
    ("generic", 1, 1, 8),
    ("generic", 4, 3, 16),
    ("generic", 5, 2, 24),
    ("generic", 2, 4, 64),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("name,rows,heads,cols", REAL_SHAPES + GENERIC_SHAPES)
def test_rmsnorm_fused_matches_torch(
    name: str, rows: int, heads: int, cols: int, dtype: torch.dtype
) -> None:
    """One kernel against `model.rmsnorm`'s six torch ops, folding heads into rows."""
    gen = torch.Generator().manual_seed(rows * 1000 + heads * 100 + cols)
    x = torch.randn(rows, heads, cols, generator=gen).to(DEVICE).to(dtype)
    w = torch.randn(cols, generator=gen).to(DEVICE).to(dtype)
    eps = 1e-6

    want = seed_model.rmsnorm(x, w, eps)
    got = rmsnorm_fused.rmsnorm(x, w, eps)
    assert got.shape == want.shape and got.dtype == want.dtype
    err = relative_error(got, want)
    assert err < tolerance(dtype), f"{name}: rel_err {err:.3e} >= {tolerance(dtype):.3e}"


def test_rmsnorm_fused_reads_strided_operands() -> None:
    """q_norm/k_norm's real inputs: a `.chunk(dim=-1)` slice of a wider projection, so the
    row-to-row stride is wider than `cols`. Mirrors
    test_deltanet_fused.test_gated_rmsnorm_reads_strided_operands."""
    rows, heads, cols = 4, 3, 16
    gen = torch.Generator().manual_seed(11)
    wide = torch.randn(rows, heads, cols * 3, generator=gen).to(DEVICE)
    x = wide[:, :, cols : 2 * cols]
    assert x.stride(-1) == 1 and x.stride(1) != cols, "the slice must not be tightly packed"
    w = torch.randn(cols, generator=gen).to(DEVICE)

    want = seed_model.rmsnorm(x, w, 1e-6)
    got = rmsnorm_fused.rmsnorm(x, w, 1e-6)
    assert relative_error(got, want) < FP32_TOL


def test_rmsnorm_fused_rejects_a_noncontiguous_last_axis() -> None:
    x = torch.randn(2, 8, device=DEVICE).t()  # transpose: last axis no longer contiguous
    with pytest.raises(ValueError, match="contiguous"):
        rmsnorm_fused.rmsnorm(x, torch.randn(2, device=DEVICE), 1e-6)


def test_available_respects_the_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEED_FUSED_RMSNORM", "0")
    assert not rmsnorm_fused.available(torch.device("cuda"))


# ---------------------------------------------------------------- model.rmsnorm's dispatch


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("tiny-rmsnorm")
    write_checkpoint(build_hf(), out, mxfp4=False)
    return out


def test_model_rmsnorm_dispatches_to_the_fused_kernel(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`model.rmsnorm` itself, both arms, against each other -- the dispatch this whole fusion
    depends on for every call site (eager and static) to pick it up automatically."""
    x = torch.randn(3, 1, 64, device=DEVICE)
    w = torch.randn(64, device=DEVICE)

    monkeypatch.setattr(rmsnorm_fused, "available", lambda _dev: False)
    want = seed_model.rmsnorm(x, w, 1e-6)
    monkeypatch.setattr(rmsnorm_fused, "available", lambda _dev: True)
    got = seed_model.rmsnorm(x, w, 1e-6)

    assert relative_error(got, want) < FP32_TOL


def test_decode_matches_the_torch_path_with_rmsnorm_fused(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A whole batched `Model.decode` step, fused rmsnorm on vs off, output and state both.

    Exercises `Model.decode_layer` -> `rmsnorm` at every one of the five call sites through a
    real forward pass, not just the kernel in isolation. Mirrors
    test_deltanet_fused.test_decode_matches_the_torch_path.
    """
    torch.manual_seed(5)
    model = seed_model.Model(checkpoint, [str(DEVICE)], torch.float32, 32, 4)
    slots = [0, 1, 2, 3]
    for s in slots:
        model.begin(s)
        model.prefill(s, [7, 8, 9], 0)

    monkeypatch.setattr(rmsnorm_fused, "available", lambda _dev: False)
    want = model.decode(slots, [11, 12, 13, 14], [3, 3, 3, 3])

    for s in slots:
        model.begin(s)
        model.prefill(s, [7, 8, 9], 0)
    monkeypatch.setattr(rmsnorm_fused, "available", lambda _dev: True)
    got = model.decode(slots, [11, 12, 13, 14], [3, 3, 3, 3])

    assert relative_error(got, want) < FP32_TOL
    assert torch.equal(got.argmax(-1), want.argmax(-1)), "greedy token choice must not move"

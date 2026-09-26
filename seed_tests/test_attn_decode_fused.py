"""`attn_decode_fused.rope_and_kv_write` against the torch chain it replaces.

The oracle is `model.apply_rope` plus a gather/`torch.where`/scatter into the pool, exactly
`graph_decode.attn_decode_static`'s `SEED_FUSE_GLUE=0` path. Same two ways to run it as the
rest of this campaign's fused kernels:

    # real kernel, needs an accelerator
    python -m pytest .../seed_tests/test_attn_decode_fused.py -p no:cacheprovider --no-cov

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

import attn_decode_fused  # noqa: E402
import model as seed_model  # noqa: E402

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pytestmark = pytest.mark.skipif(
    not attn_decode_fused.HAVE_TRITON or not (torch.cuda.is_available() or INTERPRET),
    reason="needs triton plus either an accelerator or TRITON_INTERPRET=1",
)

TOL = 1e-5


def relative_error(got: torch.Tensor, want: torch.Tensor) -> float:
    scale = want.float().abs().max().clamp(min=1e-30).item()
    return ((got.float() - want.float()).abs().max() / scale).item()


def torch_rope_and_kv_write(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    pool_k: torch.Tensor,
    pool_v: torch.Tensor,
    write_rows: torch.Tensor,
    active: torch.Tensor | None,
) -> torch.Tensor:
    """`attn_decode_static`'s `SEED_FUSE_GLUE=0` spelling, on the kernel's own operand shapes."""
    qr = seed_model.apply_rope(q[:, :, None], cos[:, None, None], sin[:, None, None])[:, :, 0]
    kr = seed_model.apply_rope(k[:, :, None], cos[:, None, None], sin[:, None, None])[:, :, 0]
    if active is None:
        pool_k[write_rows] = kr
        pool_v[write_rows] = v
    else:
        keep = active[:, None, None]
        pool_k[write_rows] = torch.where(keep, kr, pool_k[write_rows])
        pool_v[write_rows] = torch.where(keep, v, pool_v[write_rows])
    return qr


SHAPES = pytest.mark.parametrize(
    "b,nq,nkv,head_dim,rot_dim",
    [
        (1, 1, 1, 16, 16),  # simplest case, full rotary
        (5, 8, 2, 32, 32),  # the real model's q/kv head ratio, full rotary
        (4, 4, 1, 64, 32),  # partial rotary: half the channels pass through unrotated
        (3, 2, 2, 8, 8),  # head_dim not a power of two multiple of anything special
    ],
)


@SHAPES
def test_rope_and_kv_write_matches_torch(
    b: int, nq: int, nkv: int, head_dim: int, rot_dim: int
) -> None:
    gen = torch.Generator().manual_seed(b * 1000 + nq * 10 + head_dim)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen).to(DEVICE)

    q = randn(b, nq, head_dim)
    k = randn(b, nkv, head_dim)
    v = randn(b, nkv, head_dim)
    cos = randn(b, rot_dim)
    sin = randn(b, rot_dim)
    pool_rows = b + 3
    pool_k_a = randn(pool_rows, nkv, head_dim)
    pool_v_a = randn(pool_rows, nkv, head_dim)
    pool_k_b, pool_v_b = pool_k_a.clone(), pool_v_a.clone()
    write_rows = torch.randperm(pool_rows, generator=gen)[:b].to(DEVICE)

    want_q = torch_rope_and_kv_write(q, k, v, cos, sin, pool_k_a, pool_v_a, write_rows, None)
    got_q = attn_decode_fused.rope_and_kv_write(
        q, k, v, cos, sin, rot_dim, pool_k_b, pool_v_b, write_rows, active=None
    )

    assert relative_error(got_q[:, :, 0], want_q) < TOL
    assert relative_error(pool_k_b, pool_k_a) < TOL
    assert relative_error(pool_v_b, pool_v_a) < TOL


@SHAPES
def test_rope_and_kv_write_active_mask_matches_torch(
    b: int, nq: int, nkv: int, head_dim: int, rot_dim: int
) -> None:
    """The `SEED_FUSE_GLUE=1` path `attn_decode_static` actually takes: `active` gates the
    cache write only -- `q`'s RoPE output is unconditional either way (see the kernel's
    docstring for why: only the *cache* has a "leave an inactive row alone" requirement)."""
    gen = torch.Generator().manual_seed(b * 37 + head_dim)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen).to(DEVICE)

    q = randn(b, nq, head_dim)
    k = randn(b, nkv, head_dim)
    v = randn(b, nkv, head_dim)
    cos = randn(b, rot_dim)
    sin = randn(b, rot_dim)
    pool_rows = b + 2
    pool_k_a = randn(pool_rows, nkv, head_dim)
    pool_v_a = randn(pool_rows, nkv, head_dim)
    pool_k_before, pool_v_before = pool_k_a.clone(), pool_v_a.clone()
    pool_k_b, pool_v_b = pool_k_a.clone(), pool_v_a.clone()
    write_rows = torch.randperm(pool_rows, generator=gen)[:b].to(DEVICE)
    active = torch.rand(b, generator=gen) > 0.5
    if not active.any():
        active[0] = True
    if active.all() and b > 1:
        active[0] = False

    want_q = torch_rope_and_kv_write(q, k, v, cos, sin, pool_k_a, pool_v_a, write_rows, active)
    got_q = attn_decode_fused.rope_and_kv_write(
        q, k, v, cos, sin, rot_dim, pool_k_b, pool_v_b, write_rows, active=active
    )

    assert relative_error(got_q[:, :, 0], want_q) < TOL
    assert relative_error(pool_k_b, pool_k_a) < TOL
    assert relative_error(pool_v_b, pool_v_a) < TOL
    # The inactive rows' cache bytes must be genuinely untouched (bit-identical), not just
    # numerically close to "unchanged" -- same bar `test_deltanet_fused.py` holds `active` to.
    inactive_rows = write_rows[~active]
    assert torch.equal(pool_k_b[inactive_rows], pool_k_before[inactive_rows])
    assert torch.equal(pool_v_b[inactive_rows], pool_v_before[inactive_rows])


# ---------------------------------------------------------------- rmsnorm_rope_and_kv_write


def torch_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """`model.rmsnorm`'s own torch chain (not the fused kernel): normalize in fp32, scale by
    `(1 + w)`, round once to `x`'s dtype."""
    y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (y * (1.0 + w.float())).type_as(x)


def torch_rmsnorm_rope_and_kv_write(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    pool_k: torch.Tensor,
    pool_v: torch.Tensor,
    write_rows: torch.Tensor,
    active: torch.Tensor | None,
) -> torch.Tensor:
    """`attn_decode_static`'s norm -> RoPE -> KV-write chain, unfused, as the oracle: the same
    two rounding points (`rmsnorm`'s own single rounding, then RoPE's) the fused kernel keeps."""
    qn = torch_rmsnorm(q, q_norm, eps)
    kn = torch_rmsnorm(k, k_norm, eps)
    return torch_rope_and_kv_write(qn, kn, v, cos, sin, pool_k, pool_v, write_rows, active)


@SHAPES
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rmsnorm_rope_and_kv_write_matches_torch(
    b: int, nq: int, nkv: int, head_dim: int, rot_dim: int, dtype: torch.dtype
) -> None:
    gen = torch.Generator().manual_seed(b * 1000 + nq * 10 + head_dim + 3)
    eps = 1e-6

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen).to(DEVICE).to(dtype)

    q = randn(b, nq, head_dim)
    k = randn(b, nkv, head_dim)
    v = randn(b, nkv, head_dim)
    q_norm = randn(head_dim)
    k_norm = randn(head_dim)
    cos = randn(b, rot_dim)
    sin = randn(b, rot_dim)
    pool_rows = b + 3
    pool_k_a = randn(pool_rows, nkv, head_dim)
    pool_v_a = randn(pool_rows, nkv, head_dim)
    pool_k_b, pool_v_b = pool_k_a.clone(), pool_v_a.clone()
    write_rows = torch.randperm(pool_rows, generator=gen)[:b].to(DEVICE)
    active = torch.rand(b, generator=gen) > 0.5
    if not active.any():
        active[0] = True
    if active.all() and b > 1:
        active[0] = False

    want_q = torch_rmsnorm_rope_and_kv_write(
        q, k, v, q_norm, k_norm, cos, sin, eps, pool_k_a, pool_v_a, write_rows, active
    )
    got_q = attn_decode_fused.rmsnorm_rope_and_kv_write(
        q,
        k,
        v,
        q_norm,
        k_norm,
        cos,
        sin,
        rot_dim,
        eps,
        pool_k_b,
        pool_v_b,
        write_rows,
        active=active,
    )

    tol = 1e-5 if dtype is torch.float32 else 1e-2
    assert relative_error(got_q[:, :, 0], want_q) < tol
    assert relative_error(pool_k_b, pool_k_a) < tol
    assert relative_error(pool_v_b, pool_v_a) < tol
    inactive_rows = write_rows[~active]
    # The inactive rows' cache bytes must be genuinely untouched, same bar as the plain kernel.
    assert torch.equal(pool_k_b[inactive_rows], pool_k_a[inactive_rows])
    assert torch.equal(pool_v_b[inactive_rows], pool_v_a[inactive_rows])


def test_rmsnorm_rope_and_kv_write_accepts_pre_transpose_layout() -> None:
    """`q`/`k`/`v` as `[B, 1, heads, head_dim]`, the natural shape straight off `F.linear(...)
    .view(...)` -- no `.transpose(1, 2)` needed, unlike `rope_and_kv_write`."""
    b, nq, nkv, head_dim, rot_dim = 3, 2, 1, 16, 16
    gen = torch.Generator().manual_seed(5)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen).to(DEVICE)

    q = randn(b, 1, nq, head_dim)
    k = randn(b, 1, nkv, head_dim)
    v = randn(b, 1, nkv, head_dim)
    q_norm, k_norm = randn(head_dim), randn(head_dim)
    cos, sin = randn(b, rot_dim), randn(b, rot_dim)
    pool_k = randn(b, nkv, head_dim)
    pool_v = randn(b, nkv, head_dim)
    write_rows = torch.arange(b, device=DEVICE)

    got_q = attn_decode_fused.rmsnorm_rope_and_kv_write(
        q, k, v, q_norm, k_norm, cos, sin, rot_dim, 1e-6, pool_k, pool_v, write_rows
    )
    assert got_q.shape == (b, nq, 1, head_dim)

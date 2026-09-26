"""`deltanet_prefill_glue` (SEED_DN_PREFILL_GLUE_FUSED) against the torch glue it replaces,
and the chunked kernels on key-head q/k against `repeat_interleave`d copies. GPU only.

    <python-with-torch> -m pytest seed_tests/test_deltanet_prefill_glue.py -q -o addopts=
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["SEED_DN_PREFILL_GLUE_FUSED"] = "1"

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (Triton)")

KH, KD, VH, VD, KC = 4, 128, 16, 128, 4


def torch_glue(proj, conv_w, pool_conv, lanes, length, active, real, a_log, dt_bias):
    """`graph_prefill.deltanet_prefill_core`'s glue, verbatim."""
    r, t = proj.shape[:2]
    key_dim, val_dim = KH * KD, VH * VD
    qkv, z, beta_raw, a_raw = proj.split([2 * key_dim + val_dim, val_dim, VH, VH], dim=-1)
    pre_conv = pool_conv[lanes]
    full = torch.cat([pre_conv, qkv.transpose(1, 2)], dim=-1)
    mixed = F.silu(F.conv1d(full, conv_w, groups=full.shape[1]))
    win = torch.arange(pre_conv.shape[-1], device=proj.device)
    idx = (length[:, None] + win[None, :])[:, None, :].expand(-1, full.shape[1], -1)
    new_conv = torch.gather(full, -1, idx)
    pool_conv[lanes] = torch.where(active[:, None, None], new_conv, pre_conv)
    q, k, v = mixed.transpose(1, 2).split([key_dim, key_dim, val_dim], dim=-1)
    rl = real[:, :, None]
    beta = torch.where(rl, beta_raw.sigmoid(), 0.0)
    g = torch.where(rl, -a_log.exp() * F.softplus(a_raw.float() + dt_bias), 0.0)
    return (
        q.reshape(r * t, key_dim),
        k.reshape(r * t, key_dim),
        v.reshape(r * t, val_dim),
        beta.reshape(r * t, VH),
        g.reshape(r * t, VH),
    )


@pytest.mark.parametrize("t", [32, 64, 208])
def test_glue_matches_torch(t: int) -> None:
    import deltanet_prefill_glue

    torch.manual_seed(t)
    dev = torch.device("cuda")
    r, n_lanes = 3, 8
    c = 2 * KH * KD + VH * VD
    p = c + VH * VD + 2 * VH
    proj = torch.randn(r, t, p, device=dev, dtype=torch.bfloat16)
    conv_w = torch.randn(c, 1, KC, device=dev, dtype=torch.bfloat16) * 0.5
    pool = torch.randn(n_lanes, c, KC - 1, device=dev, dtype=torch.bfloat16)
    lanes = torch.tensor([5, 2, 7], device=dev)
    length = torch.tensor([t, 2, 1], device=dev)
    active = torch.tensor([True, True, False], device=dev)
    real = torch.arange(t, device=dev)[None, :] < (length * active)[:, None]
    a_log = torch.randn(VH, device=dev)
    dt_bias = torch.randn(VH, device=dev)
    pool_ref = pool.clone()
    want = torch_glue(proj, conv_w, pool_ref, lanes, length, active, real, a_log, dt_bias)
    got = deltanet_prefill_glue.prefill_glue(
        proj, conv_w, pool, lanes, length, active, real, a_log, dt_bias, KH * KD, VH
    )
    for name, a, b in zip(("q", "k", "v", "beta", "g"), got, want, strict=True):
        diff = (a.float() - b.float()).abs().max().item()
        assert diff <= 1e-2 * b.float().abs().max().item() + 1e-6, (name, diff)
    assert torch.equal(pool, pool_ref)


def test_chunked_key_head_qk_matches_repeat_interleave() -> None:
    import deltanet_prefill_chunked

    torch.manual_seed(1)
    dev = torch.device("cuda")
    n, t = 2, 64
    q = torch.randn(n * t, KH, KD, device=dev, dtype=torch.bfloat16)
    k = torch.randn(n * t, KH, KD, device=dev, dtype=torch.bfloat16)
    v = torch.randn(n * t, VH, VD, device=dev, dtype=torch.bfloat16)
    g = -torch.rand(n * t, VH, device=dev) * 0.1
    beta = torch.rand(n * t, VH, device=dev, dtype=torch.bfloat16)
    st = torch.randn(n, VH, KD, VD, device=dev) * 0.1
    rep = VH // KH
    s1, s2 = st.clone(), st.clone()
    want, _ = deltanet_prefill_chunked.chunked_prefill(
        q.repeat_interleave(rep, 1), k.repeat_interleave(rep, 1), v, g, beta, s1, None, uniform=t
    )
    got, _ = deltanet_prefill_chunked.chunked_prefill(q, k, v, g, beta, s2, None, uniform=t)
    assert torch.equal(got, want)
    assert torch.equal(s1, s2)


@pytest.mark.parametrize("rows,t", [(1, 64), (2, 256), (4, 64), (3, 128)])
def test_chunked_inplace_lanes_match_gather_scatter(rows: int, t: int) -> None:
    """`SEED_PREFILL_DN_STATE_INPLACE`: lane-indexed in-place state with the in-kernel uniform
    table is bit-identical to gather, clone, run, `where(active)` scatter back. A padding row
    (inactive) shares a filler lane with nothing active and must leave it untouched."""
    import deltanet_prefill_chunked

    torch.manual_seed(rows * 1000 + t)
    dev = torch.device("cuda")
    q = torch.randn(rows * t, KH, KD, device=dev, dtype=torch.bfloat16)
    k = torch.randn(rows * t, KH, KD, device=dev, dtype=torch.bfloat16)
    v = torch.randn(rows * t, VH, VD, device=dev, dtype=torch.bfloat16)
    g = -torch.rand(rows * t, VH, device=dev) * 0.1
    beta = torch.rand(rows * t, VH, device=dev, dtype=torch.bfloat16)
    pool = torch.randn(9, VH, KD, VD, device=dev) * 0.1
    lanes = torch.tensor([6, 1, 3, 8][:rows], device=dev)
    active = torch.tensor([True, True, True, False][:rows], device=dev)
    if rows == 1:
        active = torch.tensor([True], device=dev)
    ref_pool = pool.clone()
    pre = ref_pool[lanes]
    rec = pre.clone()
    want, _ = deltanet_prefill_chunked.chunked_prefill(q, k, v, g, beta, rec, None, uniform=t)
    ref_pool[lanes] = torch.where(active[:, None, None, None], rec, pre)
    got, _ = deltanet_prefill_chunked.chunked_prefill(
        q, k, v, g, beta, pool, None, uniform=t, lanes=lanes, active=active, table_free=True
    )
    assert torch.equal(got, want)
    assert torch.equal(pool, ref_pool)


def test_prefill_fused_gated_norm_matches_torch() -> None:
    """`SEED_PREFILL_DN_NORM_FUSED`: fp32 recurrence output and a strided `z` slice of the
    projection through `deltanet_fused.gated_rmsnorm` against `.to(bf16)` + the torch norm."""
    import deltanet_fused
    from model import gated_rmsnorm

    torch.manual_seed(3)
    dev = torch.device("cuda")
    r, t = 2, 256
    out = torch.randn(r * t, VH, VD, device=dev)
    proj = torch.randn(r, t, 3 * VH * VD + 32, device=dev, dtype=torch.bfloat16)
    z = proj[..., VH * VD : 2 * VH * VD]
    w = (torch.rand(VD, device=dev) + 0.5).to(torch.bfloat16)
    want = gated_rmsnorm(out.to(torch.bfloat16).reshape(-1, VD), z.reshape(-1, VD), w, 1e-6)
    got = deltanet_fused.gated_rmsnorm(
        out, z.reshape(r * t, VH, VD), w, 1e-6, torch.bfloat16
    ).reshape(-1, VD)
    diff = (got.float() - want.float()).abs()
    # rounding-order only: at most one bf16 ulp on a small fraction of elements
    assert diff.max().item() <= 2e-2 * want.float().abs().max().item()
    assert (diff > 0).float().mean().item() < 0.05

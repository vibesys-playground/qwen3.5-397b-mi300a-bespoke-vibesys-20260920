"""`prefill_moe` (SEED_PREFILL_GROUPED_MOE) against the fp32 dequantize-then-`bmm` oracle.

    # logic only, no GPU (Triton >= 3.5 interpreter; 3.4's is not trusted here), < 2 min
    TRITON_INTERPRET=1 python -m pytest seed_tests/test_prefill_moe.py -p no:cacheprovider

    # real kernels on an accelerator (bf16 activations, adds the real-dimension case)
    python -m pytest seed_tests/test_prefill_moe.py -p no:cacheprovider

Under the interpreter everything is fp32 (its bf16 `tl.dot` is wrong), so the bar is fp32
reduction-order noise. On a GPU the activations, `inter` and `y` are bf16, so the bar is a
few bf16 ulps of the output's largest magnitude.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mxfp4_gemv  # noqa: E402
import prefill_moe  # noqa: E402
from mxfp4 import dequant_mxfp4  # noqa: E402
from test_mxfp4_fused_gemv import (  # noqa: E402
    DEVICE,
    random_experts,
    random_routing,
    relative_error,
    shard_experts,
)

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DTYPE = torch.float32 if INTERPRET else torch.bfloat16
TOL = 1e-4 if INTERPRET else 2e-2

pytestmark = pytest.mark.skipif(
    not prefill_moe.HAVE_TRITON or not (torch.cuda.is_available() or INTERPRET),
    reason="needs triton plus either an accelerator or TRITON_INTERPRET=1",
)


def _case(tokens, experts, top_k, hidden, inter, span, *, seed, zero_rows=()):
    ex = random_experts(experts, hidden, inter, seed=seed)
    a_expert, a_weight = random_routing(tokens, experts, top_k, seed=seed, zero_rows=zero_rows)
    a_expert, a_weight = a_expert.to(DEVICE), a_weight.to(DEVICE)
    g = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(tokens, hidden, generator=g).to(DEVICE, DTYPE)
    local = shard_experts(ex, span)
    want = mxfp4_gemv.reference_moe(
        x, local, (a_expert, a_weight), top_k, span, dequant_mxfp4, torch.float32
    )
    return x, local, (a_expert, a_weight), want


def test_align_is_a_partition_of_live_assignments() -> None:
    """Every live assignment sits in exactly one row of a block of its own expert, and
    `assign_slot` is the inverse of `slot_assign` on exactly the live ones."""
    top_k, span, bm = 4, (3, 11), 16
    a_expert, a_weight = random_routing(200, 16, top_k, seed=3, zero_rows=(5, 17))
    slot_assign, block_expert, assign_slot = prefill_moe.prefill_align(a_expert, a_weight, span, bm)
    lo, hi = span
    live = (a_expert >= lo) & (a_expert < hi) & (a_weight != 0)
    assert block_expert.numel() == prefill_moe.block_count(a_expert.numel(), hi - lo, bm)
    held = slot_assign[slot_assign >= 0].long()
    assert sorted(held.tolist()) == list(range(a_expert.numel()))  # each exactly once
    for s in range(slot_assign.numel()):
        a = int(slot_assign[s])
        if a < 0 or not live[a]:
            continue
        assert int(block_expert[s // bm]) == int(a_expert[a]) - lo
        assert int(assign_slot[a]) == s
    assert (assign_slot[~live] == -1).all()
    # dropped assignments only ever share blocks with the sentinel expert
    for s in range(slot_assign.numel()):
        a = int(slot_assign[s])
        if a >= 0 and not live[a]:
            assert int(block_expert[s // bm]) == hi - lo


@pytest.mark.parametrize("prefetch", [True, False])
@pytest.mark.parametrize(
    ("tokens", "span", "bm"),
    [(40, (2, 6), 16), (23, (0, 8), 32)],
)
def test_prefill_moe_matches_fp32_oracle(prefetch, tokens, span, bm) -> None:
    """Expert-parallel slice (non-local assignments dropped), zero-weight rows skipped,
    partial blocks, several N tiles and an even number of K trips in both GEMMs."""
    hidden, inter, top_k = 256, 128, 3
    x, local, routing, want = _case(
        tokens, 8, top_k, hidden, inter, span, seed=tokens, zero_rows=(1, 7)
    )
    got = prefill_moe.prefill_moe(
        x,
        local,
        routing,
        top_k,
        span,
        block_m=bm,
        block_n=32,
        block_n_down=64,
        block_k=64,
        prefetch=prefetch,
    )
    assert got.shape == want.shape
    assert torch.isfinite(got.float()).all()
    assert relative_error(got, want) < TOL
    # zero-weight tokens and tokens with no local expert come back exactly zero
    assert (got[1].float() == 0).all() and (got[7].float() == 0).all()


def test_rejects_odd_trip_count_when_prefetching() -> None:
    x, local, routing, _ = _case(8, 4, 2, 64, 64, (0, 4), seed=1)
    with pytest.raises(ValueError, match="even"):
        prefill_moe.prefill_moe(
            x,
            local,
            routing,
            2,
            (0, 4),
            block_m=16,
            block_n=32,
            block_n_down=64,
            block_k=64,
            prefetch=True,
        )


@pytest.mark.skipif(INTERPRET, reason="real dimensions: GPU only")
def test_real_dims_match_grouped_path() -> None:
    """Model dimensions, 512 tokens, rank 1 of 4: same answer as `_grouped_moe`."""
    hidden, inter, top_k, experts, span = 4096, 1024, 10, 512, (128, 256)
    ex = shard_experts(random_experts(experts, hidden, inter, seed=7), span)
    a_expert, a_weight = random_routing(512, experts, top_k, seed=7)
    routing = (a_expert.to(DEVICE), a_weight.to(DEVICE))
    x = torch.randn(512, hidden, generator=torch.Generator().manual_seed(8)).to(DEVICE, DTYPE)
    got = prefill_moe.prefill_moe(x, ex, routing, top_k, span)
    old = mxfp4_gemv.fused_moe(x, ex, routing, top_k, span, grouped=True)
    assert relative_error(got, old) < TOL


# -- SEED_DELTANET_PREFILL_PREFETCH: the other prefill lever in PREFILL_ROOFLINE.md ----------


@pytest.mark.parametrize("lengths", [(7, 0, 1, 5), (12,)])
def test_deltanet_prefill_prefetch_matches_recurrence(lengths) -> None:
    """`fused_recurrent_prefill(prefetch=True)` against the plain kernel (bit-identical: same
    expressions, only the load order moves) and against `model.delta_rule_recurrent` per
    sequence. Covers an empty sequence (the pre-loop load is masked) and a length-1 one (the
    prefetch re-reads its own row)."""
    import deltanet_fused  # noqa: PLC0415
    import model as seed_model  # noqa: PLC0415

    heads, dk, dv = 3, 16, 32
    total = sum(lengths)
    g = torch.Generator().manual_seed(len(lengths))
    q = torch.randn(total, heads, dk, generator=g).to(DEVICE)
    k = torch.randn(total, heads, dk, generator=g).to(DEVICE)
    v = torch.randn(total, heads, dv, generator=g).to(DEVICE)
    gl = (-torch.rand(total, heads, generator=g) * 0.5).to(DEVICE)
    beta = torch.rand(total, heads, generator=g).to(DEVICE)
    state0 = torch.randn(len(lengths), heads, dk, dv, generator=g).to(DEVICE) * 0.1
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32).to(DEVICE)

    outs = {}
    for pf in (False, True):
        st = state0.clone()
        o, _ = deltanet_fused.fused_recurrent_prefill(
            q, k, v, gl, beta, st, cu, prefetch=pf, chunked=False
        )
        outs[pf] = (o, st)
    assert torch.equal(outs[True][0], outs[False][0])
    assert torch.equal(outs[True][1], outs[False][1])

    for s, (lo, hi) in enumerate(zip(cu[:-1].tolist(), cu[1:].tolist(), strict=True)):
        rec = state0[s : s + 1].clone().cpu()
        if hi > lo:
            want = seed_model.delta_rule_recurrent(
                q[lo:hi].cpu()[None],
                k[lo:hi].cpu()[None],
                v[lo:hi].cpu()[None],
                gl[lo:hi].cpu()[None],
                beta[lo:hi].cpu()[None],
                rec,
            )[0]
            assert relative_error(outs[True][0][lo:hi].cpu(), want) < 1e-4
        assert relative_error(outs[True][1][s].cpu(), rec[0]) < 1e-4


# -- SEED_DELTANET_PREFILL_CHUNKED: the chunked (WY) DeltaNet prefill ------------------------


@pytest.mark.parametrize("lengths", [(37, 0, 1, 16, 5), (64,)])
def test_deltanet_chunked_prefill_matches_recurrence(lengths) -> None:
    """`deltanet_prefill_chunked.chunked_prefill` against the per-position kernel: partial
    last chunks, a chunk-aligned sequence, empty and length-1 sequences, carried state."""
    import deltanet_fused  # noqa: PLC0415
    import deltanet_prefill_chunked  # noqa: PLC0415

    heads, dk, dv = 2, 16, 32
    total = sum(lengths)
    g = torch.Generator().manual_seed(100 + len(lengths))
    q = torch.randn(total, heads, dk, generator=g).to(DEVICE)
    k = torch.randn(total, heads, dk, generator=g).to(DEVICE)
    v = torch.randn(total, heads, dv, generator=g).to(DEVICE)
    gl = (-torch.rand(total, heads, generator=g) * 0.5).to(DEVICE)
    beta = torch.rand(total, heads, generator=g).to(DEVICE)
    state0 = torch.randn(len(lengths), heads, dk, dv, generator=g).to(DEVICE) * 0.1
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    cu = cu.to(DEVICE)

    st_ref = state0.clone()
    want, _ = deltanet_fused.fused_recurrent_prefill(
        q, k, v, gl, beta, st_ref, cu, prefetch=False, chunked=False
    )
    st = state0.clone()
    got, _ = deltanet_prefill_chunked.chunked_prefill(
        q, k, v, gl, beta, st, cu, chunk=16, block_v=16
    )
    assert relative_error(got, want) < 1e-4
    assert relative_error(st, st_ref) < 1e-4


@pytest.mark.parametrize(("rows", "width"), [(1, 32), (3, 16)])
def test_deltanet_chunked_prefill_uniform_table_matches(rows: int, width: int) -> None:
    """`uniform=W` (the capturable, device-built chunk table `graph_prefill` uses) against the
    host-built table and the per-position kernel, with padded columns (`beta = g = 0`, the
    graph layout's identity steps) at the end of a row."""
    import deltanet_fused  # noqa: PLC0415
    import deltanet_prefill_chunked  # noqa: PLC0415

    heads, dk, dv = 2, 16, 32
    total = rows * width
    g = torch.Generator().manual_seed(7 + rows)
    q = torch.randn(total, heads, dk, generator=g).to(DEVICE)
    k = torch.randn(total, heads, dk, generator=g).to(DEVICE)
    v = torch.randn(total, heads, dv, generator=g).to(DEVICE)
    gl = (-torch.rand(total, heads, generator=g) * 0.5).to(DEVICE)
    beta = torch.rand(total, heads, generator=g).to(DEVICE)
    pad = torch.zeros(total, dtype=torch.bool)
    pad[width - 5 : width] = True  # row 0 has 5 padded columns
    gl[pad.to(DEVICE)], beta[pad.to(DEVICE)] = 0.0, 0.0
    state0 = torch.randn(rows, heads, dk, dv, generator=g).to(DEVICE) * 0.1
    cu = torch.arange(0, total + 1, width, dtype=torch.int32).to(DEVICE)

    st_ref, st_host, st = state0.clone(), state0.clone(), state0.clone()
    want, _ = deltanet_fused.fused_recurrent_prefill(
        q, k, v, gl, beta, st_ref, cu, prefetch=False, chunked=False
    )
    host, _ = deltanet_prefill_chunked.chunked_prefill(
        q, k, v, gl, beta, st_host, cu, chunk=16, block_v=16
    )
    got, _ = deltanet_prefill_chunked.chunked_prefill(
        q, k, v, gl, beta, st, None, chunk=16, block_v=16, uniform=width
    )
    assert torch.equal(got, host) and torch.equal(st, st_host)
    assert relative_error(got, want) < 1e-4
    assert relative_error(st, st_ref) < 1e-4


@pytest.mark.skipif(INTERPRET, reason="real dimensions: GPU only")
def test_deltanet_chunked_prefill_real_dims() -> None:
    """16 local heads x 128 x 128, two packed sequences (1500 + 548 rows), bf16 q/k/v/beta as
    the model passes them: chunked vs per-position kernel, both fp32 inside."""
    import deltanet_fused  # noqa: PLC0415
    import deltanet_prefill_chunked  # noqa: PLC0415

    heads, d, lengths = 16, 128, (1500, 548)
    total = sum(lengths)
    g = torch.Generator().manual_seed(5)
    q, k, v = (torch.randn(total, heads, d, generator=g).to(DEVICE, torch.bfloat16) for _ in "qkv")
    gl = (-torch.rand(total, heads, generator=g) * 0.2).to(DEVICE)
    beta = torch.rand(total, heads, generator=g).to(DEVICE, torch.bfloat16)
    state0 = torch.randn(2, heads, d, d, generator=g).to(DEVICE) * 0.05
    cu = torch.tensor([0, lengths[0], total], dtype=torch.int32, device=DEVICE)
    st_ref, st = state0.clone(), state0.clone()
    want, _ = deltanet_fused.fused_recurrent_prefill(
        q, k, v, gl, beta, st_ref, cu, prefetch=False, chunked=False
    )
    got, _ = deltanet_prefill_chunked.chunked_prefill(q, k, v, gl, beta, st, cu)
    assert relative_error(got, want) < 1e-3
    assert relative_error(st, st_ref) < 1e-3

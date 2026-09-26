"""`SEED_PREFILL_PACK` kernels: several segments in one packed stream against the same
segments each at the start of its own row (the per-row layout). The DeltaNet glue and the
chunked recurrence must be bit-identical per segment, including the conv window and the
recurrent state written back to each lane. GPU only.

    <python-with-torch> -m pytest seed_tests/test_prefill_pack.py -q -o addopts=
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["SEED_DN_PREFILL_GLUE_FUSED"] = "1"

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (Triton)")

KH, KD, VH, VD, KC = 4, 128, 16, 128, 4
ALIGN = 16


def _align(n: int) -> int:
    return -(-n // ALIGN) * ALIGN


@pytest.mark.parametrize(
    "width,lengths,stream",
    [
        (64, [64, 1, 17, 40, 33], 256),
        (128, [100, 3, 128, 16], 320),
        (256, [256, 200, 5], 512),
    ],
)
def test_packed_glue_and_recurrence_match_per_row(width, lengths, stream) -> None:  # noqa: ANN001
    import deltanet_prefill_chunked
    import deltanet_prefill_glue

    torch.manual_seed(width + stream)
    dev = torch.device("cuda")
    n_seg, n_lanes = len(lengths), 12
    c = 2 * KH * KD + VH * VD
    p = c + VH * VD + 2 * VH
    conv_w = torch.randn(c, 1, KC, device=dev, dtype=torch.bfloat16) * 0.5
    conv_pool = torch.randn(n_lanes, c, KC - 1, device=dev, dtype=torch.bfloat16)
    rec_pool = torch.randn(n_lanes, VH, KD, VD, device=dev) * 0.1
    a_log = torch.randn(VH, device=dev)
    dt_bias = torch.randn(VH, device=dev)
    lanes = torch.tensor([9, 2, 7, 4, 11][:n_seg], device=dev)
    seg_proj = [torch.randn(n, p, device=dev, dtype=torch.bfloat16) for n in lengths]

    # Per-row reference: segment j at the start of row j, padding after it.
    rows_proj = torch.randn(n_seg, width, p, device=dev, dtype=torch.bfloat16)
    for j, x in enumerate(seg_proj):
        rows_proj[j, : len(x)] = x
    length = torch.tensor(lengths, device=dev)
    active = torch.ones(n_seg, dtype=torch.bool, device=dev)
    real = torch.arange(width, device=dev)[None, :] < length[:, None]
    ref_conv, ref_rec = conv_pool.clone(), rec_pool.clone()
    q, k, v, beta, g = deltanet_prefill_glue.prefill_glue(
        rows_proj, conv_w, ref_conv, lanes, length, active, real, a_log, dt_bias, KH * KD, VH
    )
    want, _ = deltanet_prefill_chunked.chunked_prefill(
        q.view(-1, KH, KD), k.view(-1, KH, KD), v.view(-1, VH, VD), g, beta, ref_rec, None,
        uniform=width, lanes=lanes, active=active, table_free=True,
    )
    want_q = q.view(n_seg, width, -1)
    want = want.view(n_seg, width, VH, VD)

    # Packed: segments at aligned offsets, garbage between and after, two unused slots.
    slots = n_seg + 2
    pk = torch.randn(stream, p, device=dev, dtype=torch.bfloat16)
    tok_seg = torch.zeros(stream, dtype=torch.int32)
    tok_off = torch.full((stream,), 1 << 30, dtype=torch.int32)
    tok_real = torch.zeros(stream, dtype=torch.bool)
    first, count = [0] * slots, [0] * slots
    start, seg_len = [0] * slots, [1] * slots
    seg_lanes, seg_active = [0] * slots, [False] * slots
    at, offsets = 0, []
    for j, x in enumerate(seg_proj):
        n = len(x)
        pk[at : at + n] = x
        tok_seg[at : at + _align(n)] = j
        tok_off[at : at + _align(n)] = torch.arange(_align(n), dtype=torch.int32)
        tok_real[at : at + n] = True
        first[j], count[j], start[j], seg_len[j] = at // ALIGN, _align(n) // ALIGN, at, n
        seg_lanes[j], seg_active[j] = int(lanes[j]), True
        offsets.append(at)
        at += _align(n)
    assert at <= stream
    i32 = {"dtype": torch.int32, "device": dev}
    got_conv, got_rec = conv_pool.clone(), rec_pool.clone()
    seg_lanes_d = torch.tensor(seg_lanes, device=dev)
    seg_active_d = torch.tensor(seg_active, device=dev)
    q2, k2, v2, beta2, g2 = deltanet_prefill_glue.prefill_glue_packed(
        pk, conv_w, got_conv, seg_lanes_d, torch.tensor(start, **i32),
        torch.tensor(seg_len, **i32), seg_active_d, tok_seg.to(dev), tok_off.to(dev),
        tok_real.to(dev), a_log, dt_bias, KH * KD, VH,
    )
    got, _ = deltanet_prefill_chunked.chunked_prefill_packed(
        q2.view(-1, KH, KD), k2.view(-1, KH, KD), v2.view(-1, VH, VD), g2, beta2, got_rec,
        torch.tensor(first, **i32), torch.tensor(count, **i32), seg_lanes_d, seg_active_d,
    )

    for j, n in enumerate(lengths):
        o = offsets[j]
        assert torch.equal(q2[o : o + n], want_q[j, :n]), j
        assert torch.equal(got[o : o + n], want[j, :n]), j
    assert torch.equal(got_conv, ref_conv)
    assert torch.equal(got_rec, ref_rec)
    assert torch.isfinite(got).all()
    assert (got[at:] == 0).all()

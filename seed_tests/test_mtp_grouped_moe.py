"""`mtp_dense_moe.grouped_moe` (expert-grouped BF16 MTP experts) against `fused_moe` and a torch
oracle. Needs an accelerator (Triton on ROCm/CUDA).

    python -m pytest seed_tests/test_mtp_grouped_moe.py -p no:cacheprovider --no-cov
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mtp_dense_moe  # noqa: E402

NEEDS_GPU = pytest.mark.skipif(
    not (mtp_dense_moe.HAVE_TRITON and torch.cuda.is_available()), reason="needs an accelerator"
)


def _case(seed: int, tokens: int, top_k: int, local: int, lo: int, total: int, hot: bool):
    gen = torch.Generator().manual_seed(seed)
    hidden, inter = 256, 128
    dev = torch.device("cuda")
    h = (torch.randn(tokens, hidden, generator=gen) * 0.5).to(torch.bfloat16).to(dev)
    experts = {
        "gate_up": (torch.randn(local, 2 * inter, hidden, generator=gen) * 0.05).to(torch.bfloat16).to(dev),
        "down": (torch.randn(local, hidden, inter, generator=gen) * 0.05).to(torch.bfloat16).to(dev),
    }
    if hot:  # one local expert on most tokens: chunks of more than BM rows
        ids = torch.stack([torch.randperm(total, generator=gen)[:top_k] for _ in range(tokens)])
        ids[:, 0] = lo
    else:
        ids = torch.stack([torch.randperm(total, generator=gen)[:top_k] for _ in range(tokens)])
    w = torch.rand(tokens, top_k, generator=gen)
    w = w / w.sum(-1, keepdim=True)
    w[0, -1] = 0.0  # zero-weight skip
    active = torch.rand(tokens, generator=gen) > 0.2
    return h, experts, ids.to(dev), w.to(dev), active.to(dev), (lo, lo + local), inter


def _oracle(h, experts, ids, w, active, rng):
    lo, hi = rng
    tokens, top_k = ids.shape
    out = torch.zeros(h.shape, dtype=torch.float32, device=h.device)
    own = (ids >= lo) & (ids < hi) & (w != 0) & active[:, None]
    local = (ids - lo).clamp(0, hi - lo - 1)
    gu = torch.einsum("th,tknh->tkn", h.float(), experts["gate_up"][local].float())
    gate, up = gu.chunk(2, -1)
    act = (F.silu(gate) * up).to(torch.bfloat16).float()
    y = torch.einsum("tkn,tkhn->tkh", act, experts["down"][local].float())
    return (y * (w * own)[..., None]).sum(1) + out


@NEEDS_GPU
@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize(
    ("tokens", "top_k", "local", "lo", "total", "hot"),
    [(96, 10, 16, 16, 64, False), (7, 4, 8, 0, 8, False), (48, 8, 4, 4, 16, True), (1, 2, 8, 8, 32, False)],
)
def test_grouped_matches_per_assignment(seed, tokens, top_k, local, lo, total, hot) -> None:
    h, experts, ids, w, active, rng, inter = _case(seed, tokens, top_k, local, lo, total, hot)
    hidden = h.shape[1]
    inter_s = torch.empty(tokens * top_k, inter, dtype=torch.bfloat16, device=h.device)
    out_a = torch.empty(tokens, hidden, dtype=torch.bfloat16, device=h.device)
    out_b = torch.empty_like(out_a)
    want = mtp_dense_moe.fused_moe(h, experts, (ids, w), top_k, rng, inter_s, out_a, active).clone()
    scratch = mtp_dense_moe.grouped_scratch(tokens, top_k, local, hidden, h.device)
    got = mtp_dense_moe.grouped_moe(h, experts, (ids, w), top_k, rng, inter_s, out_b, scratch, active)
    ref = _oracle(h, experts, ids, w, active, rng)
    assert torch.equal(got[~active], torch.zeros_like(got[~active]))
    # Same tiles and order as `fused_moe`; only its broadcast-column average can move a last bit.
    diff = (got.float() - want.float()).abs()
    assert float(diff.max()) <= 2 * float(want.float().abs().max()) * 2**-8
    assert float((got.float() == want.float()).float().mean()) > 0.99
    assert torch.allclose(got.float(), ref, atol=2e-2, rtol=2e-2)

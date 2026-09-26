"""`Model.full_attention_packed`'s `SEED_VARLEN_PREFILL_ATTN=1` dispatch against the real model,
end to end -- not just the kernel-level oracle `test_varlen_prefill_attn.py` already covers.

Reuses `test_batched_prefill.py`'s tiny real checkpoint and packed-vs-solo scenarios (fresh
sequences, resumed prefixes with mixed chunk lengths including a length-one chunk): the claim
here is that `_full_attention_packed_fused` (the `varlen_prefill_attn` dispatch) produces the
same logits as the v1 per-sequence loop (`prefill_attention`/SDPA + `_gather_cached_prefix`) it
replaces, on the exact same packed calls, not merely that the standalone kernel matches its own
synthetic oracle.

Runs on CPU: `varlen_prefill_attn.available` only ever returns `True` on `cuda` (same contract
as `paged_attn.available`/`decode_attn_v2.available`), so exercising the fused branch here
monkeypatches it to `True` regardless of device -- the same pattern `test_paged_attn.py`'s
`test_model_decode_attention_paged_dispatches_to_the_kernel` already uses -- and needs
`TRITON_INTERPRET=1` for the kernel itself to run.

    TRITON_INTERPRET=1 python -m pytest \\
        examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/seed_tests/test_varlen_prefill_attn_integration.py
"""

import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import varlen_prefill_attn  # noqa: E402
from test_batched_decode import build, prompt_of  # noqa: E402
from test_seed_parity import build_hf, write_checkpoint  # noqa: E402

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"

NEEDS_KERNEL = pytest.mark.skipif(
    not varlen_prefill_attn.HAVE_TRITON or not (torch.cuda.is_available() or INTERPRET),
    reason="needs triton plus either an accelerator or TRITON_INTERPRET=1",
)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("tiny")
    write_checkpoint(build_hf(), out, mxfp4=False)
    return out


@NEEDS_KERNEL
def test_fused_packed_prefill_matches_v1_loop_from_scratch(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(varlen_prefill_attn, "available", lambda _dev: True)
    monkeypatch.setattr(varlen_prefill_attn, "BLOCK_M", 4)  # exercise multi-tile sequences
    prompts = [prompt_of(301, 5), prompt_of(302, 3), prompt_of(303, 7)]

    v1 = build(checkpoint, max_batch=len(prompts))
    for slot in range(len(prompts)):
        v1.begin(slot)
    v1_logits = v1.prefill_batch([(slot, p, 0) for slot, p in enumerate(prompts)])

    fused = build(checkpoint, max_batch=len(prompts))
    for slot in range(len(prompts)):
        fused.begin(slot)
    fused_logits = fused.prefill_batch([(slot, p, 0) for slot, p in enumerate(prompts)])

    assert len(fused_logits) == len(v1_logits)
    for idx, (want, got) in enumerate(zip(v1_logits, fused_logits, strict=True)):
        assert got.shape == want.shape
        assert torch.allclose(got, want, atol=1e-2, rtol=1e-2), f"slot {idx}"
        assert int(got.argmax()) == int(want.argmax())


@NEEDS_KERNEL
def test_fused_packed_prefill_matches_v1_loop_when_resuming(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resumed prefixes at different, nonzero `start` positions, mixed chunk lengths including
    a length-one chunk -- the shape `full_attention`'s own `_gather_cached_prefix` branch
    (`start > 0`) exists for, now routed through the fused kernel's block-table read instead."""
    monkeypatch.setattr(varlen_prefill_attn, "available", lambda _dev: True)
    monkeypatch.setattr(varlen_prefill_attn, "BLOCK_M", 4)
    prompts = [prompt_of(401, 6), prompt_of(402, 9), prompt_of(403, 4)]
    firsts = [p[:3] for p in prompts]
    rests = [prompts[0][3:], prompts[1][3:], prompts[2][3:4]]  # lengths 3, 6, 1

    v1 = build(checkpoint, max_batch=len(prompts))
    for slot, first in enumerate(firsts):
        v1.begin(slot)
        v1.prefill(slot, first, 0)
    v1_logits = [
        v1.prefill(slot, rest, len(first))
        for slot, (first, rest) in enumerate(zip(firsts, rests, strict=True))
    ]

    fused = build(checkpoint, max_batch=len(prompts))
    for slot, first in enumerate(firsts):
        fused.begin(slot)
    fused.prefill_batch([(slot, first, 0) for slot, first in enumerate(firsts)])
    fused_logits = fused.prefill_batch(
        [(slot, rest, len(first)) for slot, (first, rest) in enumerate(zip(firsts, rests, strict=True))]
    )

    for idx, (want, got) in enumerate(zip(v1_logits, fused_logits, strict=True)):
        assert got.shape == want.shape
        assert torch.allclose(got, want, atol=1e-2, rtol=1e-2), f"slot {idx}"
        assert int(got.argmax()) == int(want.argmax())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

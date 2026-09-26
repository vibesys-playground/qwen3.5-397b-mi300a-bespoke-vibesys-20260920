"""Regression test: `Model.forward_packed` must run `lm_head` over only each sequence's own
last packed row, not every packed position.

Before the fix, `forward_packed` ran the final norm + `lm_head` over the full `[total_T,
hidden]` packed activations and let `prefill_batch` slice the resulting `[total_T, vocab]`
logits down to one row per sequence afterwards. At 2k packed tokens that intermediate logits
tensor is a ~3 GiB transient, discarded down to `len(seqs)` rows a line later. The fix slices
`x` to `meta.spans()`'s last row per sequence *before* the final norm + `lm_head`, so the
projection's own input is `[len(seqs), hidden]`.

Uses the same tiny random Qwen3.5-MoE checkpoint and `build`/`prompt_of` helpers as
`test_batched_prefill.py` (imported from `test_batched_decode`).

    /tmp/torchenv/bin/python -m pytest \\
        examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/seed_tests/test_forward_packed_lm_head_slicing.py \\
        -p no:cacheprovider --no-cov
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from test_batched_decode import build, prompt_of  # noqa: E402
from test_seed_parity import build_hf, write_checkpoint  # noqa: E402


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("tiny")
    write_checkpoint(build_hf(), out, mxfp4=False)
    return out


def lm_head_input_rows(
    monkeypatch: pytest.MonkeyPatch, model, calls: list[tuple[int, list[int], int]]
) -> tuple[list[torch.Tensor], int]:
    """Run `model.prefill_batch(calls)`, spying on every `F.linear` call whose weight is
    `model.lm_head` (identity, not value equality: `rmsnorm`/embedding weights are separate
    tensors of possibly-matching shape). Returns (results, captured input row count)."""
    real_linear = F.linear
    seen_rows: list[int] = []

    def spy_linear(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None):
        if weight is model.lm_head:
            seen_rows.append(input.shape[0])
        return real_linear(input, weight, bias)

    monkeypatch.setattr(F, "linear", spy_linear)
    results = model.prefill_batch(calls)
    return results, seen_rows


def test_lm_head_only_sees_one_row_per_sequence(
    monkeypatch: pytest.MonkeyPatch, checkpoint: Path
) -> None:
    prompts = [prompt_of(301, 5), prompt_of(302, 3), prompt_of(303, 7)]  # total_T = 15
    total_t = sum(len(p) for p in prompts)

    model = build(checkpoint, max_batch=len(prompts))
    for slot in range(len(prompts)):
        model.begin(slot)
    calls = [(slot, prompt, 0) for slot, prompt in enumerate(prompts)]

    results, seen_rows = lm_head_input_rows(monkeypatch, model, calls)

    assert len(results) == len(prompts)
    assert seen_rows, "lm_head (model.lm_head) was never called by prefill_batch"
    # forward_packed's own lm_head call must see exactly one row per sequence, never the
    # full packed width.
    assert seen_rows[-1] == len(prompts)
    assert seen_rows[-1] != total_t, "lm_head saw the full packed width, not per-sequence rows"


def test_forward_packed_matches_solo_prefill_after_slicing(checkpoint: Path) -> None:
    """The compacting change must not move which row each sequence's logits come from."""
    prompts = [prompt_of(401, 6), prompt_of(402, 4), prompt_of(403, 8)]

    solo = build(checkpoint, max_batch=len(prompts))
    solo_logits = []
    for slot, prompt in enumerate(prompts):
        solo.begin(slot)
        solo_logits.append(solo.prefill(slot, prompt, 0))

    packed = build(checkpoint, max_batch=len(prompts))
    for slot in range(len(prompts)):
        packed.begin(slot)
    calls = [(slot, prompt, 0) for slot, prompt in enumerate(prompts)]
    packed_logits = packed.prefill_batch(calls)

    for i, (want, got) in enumerate(zip(solo_logits, packed_logits, strict=True)):
        assert got.shape == want.shape == (1, model_vocab(packed))
        assert torch.allclose(got, want, atol=1e-4, rtol=1e-4), f"slot {i}"


def model_vocab(model) -> int:
    return model.lm_head.shape[0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

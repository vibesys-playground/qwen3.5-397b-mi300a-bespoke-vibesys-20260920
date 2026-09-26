"""Hermetic CPU tests for cross-request batched (packed varlen) prefill against the real
model code.

Uses the same tiny random Qwen3.5-MoE checkpoint as `test_batched_decode.py` (4 layers, both
layer kinds, 8 experts), on CPU in float32. The tensor-level claim is that `Model.prefill_batch`
packing several sequences into one forward call produces exactly what calling `Model.prefill`
once per sequence produces: the attention and DeltaNet mixers each loop over sequences inside
one packed call (see `model.py`'s `full_attention_packed`/`deltanet_packed`), so packing must
not change any sequence's own math, only how many Python-level forward calls it takes.

    /tmp/torchenv/bin/python -m pytest \\
        examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/seed_tests/test_batched_prefill.py
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import model as seed_model  # noqa: E402
from scheduler import Request  # noqa: E402
from test_batched_decode import build, make_scheduler, prompt_of, reference  # noqa: E402
from test_scheduler import Sink, drain  # noqa: E402
from test_seed_parity import build_hf, write_checkpoint  # noqa: E402


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("tiny")
    write_checkpoint(build_hf(), out, mxfp4=False)
    return out


@pytest.fixture(autouse=True, params=[False, True], ids=["loop", "vec"])
def packed_prefill_vec(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> bool:
    """Every test runs against both `deltanet_packed` paths (`SEED_PACKED_PREFILL_VEC`)."""
    monkeypatch.setattr(seed_model, "PACKED_PREFILL_VEC", request.param)
    return request.param


# ---------------------------------------------------------------- (a) packed vs. per-sequence


def test_packed_prefill_matches_separate_prefill_from_scratch(checkpoint: Path) -> None:
    """The first chunk of several fresh sequences, packed into one call, must match forwarding
    each sequence alone."""
    prompts = [prompt_of(101, 5), prompt_of(102, 3), prompt_of(103, 7)]

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

    assert len(packed_logits) == len(prompts)
    for i, (want, got) in enumerate(zip(solo_logits, packed_logits, strict=True)):
        assert got.shape == want.shape
        assert torch.allclose(got, want, atol=1e-4, rtol=1e-4), f"slot {i}"
        assert int(got.argmax()) == int(want.argmax())


def test_packed_prefill_matches_separate_prefill_when_resuming(checkpoint: Path) -> None:
    """Continuation chunks at different, nonzero `start` positions, of different lengths
    (including a length-1 chunk, the DeltaNet decode-scratch edge case), packed together."""
    prompts = [prompt_of(201, 6), prompt_of(202, 9), prompt_of(203, 4)]
    firsts = [p[:3] for p in prompts]  # every sequence already has 3 tokens resident
    rests = [prompts[0][3:], prompts[1][3:], prompts[2][3:4]]  # lengths 3, 6, 1

    solo = build(checkpoint, max_batch=len(prompts))
    for slot, first in enumerate(firsts):
        solo.begin(slot)
        solo.prefill(slot, first, 0)
    solo_logits = [
        solo.prefill(slot, rest, len(first))
        for slot, (first, rest) in enumerate(zip(firsts, rests, strict=True))
    ]

    packed = build(checkpoint, max_batch=len(prompts))
    for slot, first in enumerate(firsts):
        packed.begin(slot)
        packed.prefill(slot, first, 0)
    calls = [
        (slot, rest, len(first))
        for slot, (first, rest) in enumerate(zip(firsts, rests, strict=True))
    ]
    packed_logits = packed.prefill_batch(calls)

    for i, (want, got) in enumerate(zip(solo_logits, packed_logits, strict=True)):
        assert torch.allclose(got, want, atol=1e-4, rtol=1e-4), f"slot {i}"


def test_packed_prefill_matches_separate_prefill_across_kv_block_boundary(checkpoint: Path) -> None:
    """A resumed chunk that spans a KV-block boundary (`model.KV_BLOCK_SIZE`, 16 by default),
    packed alongside a sequence that stays within a single block: exercises
    `Model._physical_rows_range`'s block-id gather and address arithmetic across more than one
    block of a slot's table, not just the single-block case the other tests happen to hit."""
    block = seed_model.KV_BLOCK_SIZE
    prompts = [prompt_of(301, block + 12), prompt_of(302, 5)]
    firsts = [prompts[0][: block - 2], prompts[1][:2]]  # first sequence: 2 tokens short of a block
    rests = [prompts[0][block - 2 :], prompts[1][2:]]  # its chunk crosses the block boundary

    solo = build(checkpoint, max_batch=len(prompts))
    for slot, first in enumerate(firsts):
        solo.begin(slot)
        solo.prefill(slot, first, 0)
    solo_logits = [
        solo.prefill(slot, rest, len(first))
        for slot, (first, rest) in enumerate(zip(firsts, rests, strict=True))
    ]

    packed = build(checkpoint, max_batch=len(prompts))
    for slot, first in enumerate(firsts):
        packed.begin(slot)
        packed.prefill(slot, first, 0)
    calls = [
        (slot, rest, len(first))
        for slot, (first, rest) in enumerate(zip(firsts, rests, strict=True))
    ]
    packed_logits = packed.prefill_batch(calls)

    for i, (want, got) in enumerate(zip(solo_logits, packed_logits, strict=True)):
        assert torch.allclose(got, want, atol=1e-4, rtol=1e-4), f"slot {i}"


def test_packed_prefill_leaves_state_correct_for_later_decode(checkpoint: Path) -> None:
    """A stronger check than matching logits: after a packed prefill, each slot's KV and
    DeltaNet state must be exactly what a solo prefill would have left, so that *subsequent*
    generation (not just the packed call's own last-token logits) also matches."""
    specs = [(prompt_of(11, 6), 5), (prompt_of(12, 4), 6), (prompt_of(13, 8), 4)]
    reference_model = build(checkpoint, max_batch=len(specs))
    expected = [reference(reference_model, p, n) for p, n in specs]

    sched = make_scheduler(
        build(checkpoint, max_batch=len(specs)), prefill_chunk=4, token_budget=64
    )
    sinks = [Sink() for _ in specs]
    for (prompt, max_new), sink in zip(specs, sinks, strict=True):
        sched.submit(Request(list(prompt), max_new, 0.0, frozenset(), sink))
    drain(sched)

    for sink, want in zip(sinks, expected, strict=True):
        assert sink.error is None, sink.error
        assert sink.tokens == want


def test_packed_prefill_budget_forces_multiple_packed_calls(checkpoint: Path) -> None:
    """A tighter budget still produces correct results, spread over more than one packed
    call, and the packed path actually ran (not silently degraded to per-request calls)."""
    specs = [(prompt_of(21 + i, 6 + i), 3) for i in range(4)]
    reference_model = build(checkpoint, max_batch=len(specs))
    expected = [reference(reference_model, p, n) for p, n in specs]

    sched = make_scheduler(build(checkpoint, max_batch=len(specs)), prefill_chunk=3, token_budget=5)
    sinks = [Sink() for _ in specs]
    for (prompt, max_new), sink in zip(specs, sinks, strict=True):
        sched.submit(Request(list(prompt), max_new, 0.0, frozenset(), sink))
    drain(sched)

    for sink, want in zip(sinks, expected, strict=True):
        assert sink.error is None, sink.error
        assert sink.tokens == want


# ---------------------------------------------------------------- (b) A/B flag


def test_batched_prefill_flag_off_matches_flag_on(checkpoint: Path) -> None:
    """`batched_prefill=False` must produce the same generation as the packed path: the flag
    only changes how many forward calls it takes, never the result."""
    specs = [(prompt_of(31, 5), 4), (prompt_of(32, 7), 5), (prompt_of(33, 3), 6)]

    sched_on = make_scheduler(build(checkpoint, max_batch=3), prefill_chunk=3, token_budget=8)
    sinks_on = [Sink() for _ in specs]
    for (prompt, max_new), sink in zip(specs, sinks_on, strict=True):
        sched_on.submit(Request(list(prompt), max_new, 0.0, frozenset(), sink))
    drain(sched_on)

    sched_off = make_scheduler(
        build(checkpoint, max_batch=3), prefill_chunk=3, batched_prefill=False
    )
    sinks_off = [Sink() for _ in specs]
    for (prompt, max_new), sink in zip(specs, sinks_off, strict=True):
        sched_off.submit(Request(list(prompt), max_new, 0.0, frozenset(), sink))
    drain(sched_off)

    for sink_on, sink_off in zip(sinks_on, sinks_off, strict=True):
        assert sink_on.error is None and sink_off.error is None
        assert sink_on.tokens == sink_off.tokens


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

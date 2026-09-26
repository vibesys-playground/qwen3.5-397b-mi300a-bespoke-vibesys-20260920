"""Hermetic CPU tests for `SEED_MIXED_BATCH` against the real model code (Sarathi-Serve-style
stall-free batching, `model.py`'s `Model.layer_mixed`/`forward_mixed`/`decode_mixed`).

Uses the same tiny random Qwen3.5-MoE checkpoint as `test_batched_decode.py`/
`test_batched_prefill.py` (4 layers, both layer kinds, 8 experts), on CPU in float32. Two
tensor-level claims, mirroring how `test_batched_prefill.py` validates packed prefill against
the per-request loop it replaces:

(a) `Model.decode_mixed`'s decode-side logits and prefill-side logits each exactly match what
    calling `Model.decode` and `Model.prefill_batch` separately would produce -- mixing the two
    into one forward call must not change either side's math, only how many Python-level
    forward calls (and collectives) it takes, the same claim `layer_packed` already proves for
    several packed prefill sequences.
(b) end to end through the scheduler (`SEED_MIXED_BATCH` on), a session with turn-2 admission
    overlapping turn-1 decode -- the exact scenario the campaign brief measures the stall in --
    produces the same greedy tokens as one-sequence-at-a-time `Model.generate`.

    /tmp/torchenv/bin/python -m pytest \\
        examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/seed_tests/test_mixed_batch.py
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


# ---------------------------------------------------------------- (a) tensor-level parity


def test_decode_mixed_matches_separate_decode_and_prefill_batch_calls(checkpoint: Path) -> None:
    """Three lanes already decoding (turn 1), one fresh prompt's first chunk arriving as the
    mixed-in prefill work (turn 2's admission) -- exactly the scenario `SEED_MIXED_BATCH` is
    for. `decode_mixed`'s two return halves must match `decode` and `prefill_batch` run
    separately on an identically-seeded fresh model, bit-for-bit up to float reassociation."""
    decoding_prompts = [prompt_of(1, 5), prompt_of(2, 7), prompt_of(3, 4)]
    new_prompt = prompt_of(4, 6)

    def setup(m: seed_model.Model) -> tuple[list[int], list[int]]:
        for slot, prompt in enumerate(decoding_prompts):
            m.begin(slot)
            m.prefill(slot, prompt, 0)
        m.begin(3)  # the lane the new prefill chunk lands on
        tokens = [p[-1] for p in decoding_prompts]
        positions = [len(p) for p in decoding_prompts]
        return tokens, positions

    separate = build(checkpoint, max_batch=4)
    tokens, positions = setup(separate)
    decode_logits_ref = separate.decode([0, 1, 2], tokens, positions)
    (prefill_logits_ref,) = separate.prefill_batch([(3, new_prompt, 0)])

    mixed = build(checkpoint, max_batch=4)
    setup(mixed)
    decode_logits_got, prefill_logits_got = mixed.decode_mixed(
        [0, 1, 2], tokens, positions, [(3, new_prompt, 0)]
    )

    assert decode_logits_got.shape == decode_logits_ref.shape
    assert torch.allclose(decode_logits_got, decode_logits_ref, atol=1e-4, rtol=1e-4)
    assert torch.equal(decode_logits_got.argmax(-1), decode_logits_ref.argmax(-1))

    assert len(prefill_logits_got) == 1
    assert prefill_logits_got[0].shape == prefill_logits_ref.shape
    assert torch.allclose(prefill_logits_got[0], prefill_logits_ref, atol=1e-4, rtol=1e-4)
    assert int(prefill_logits_got[0].argmax()) == int(prefill_logits_ref.argmax())


def test_decode_mixed_matches_separate_calls_with_several_prefill_chunks_packed(
    checkpoint: Path,
) -> None:
    """The prefill side of one mixed step is not limited to a single request -- `_mixed_budget`
    can admit several requests' chunks in one call, same as ordinary batched prefill
    (`_select_prefill_batch`). Two decoding lanes, two fresh chunks packed into the mixed
    forward's prefill side."""
    decoding_prompts = [prompt_of(11, 5), prompt_of(12, 6)]
    new_prompts = [prompt_of(13, 4), prompt_of(14, 3)]

    def setup(m: seed_model.Model) -> tuple[list[int], list[int]]:
        for slot, prompt in enumerate(decoding_prompts):
            m.begin(slot)
            m.prefill(slot, prompt, 0)
        for slot in (2, 3):
            m.begin(slot)
        tokens = [p[-1] for p in decoding_prompts]
        positions = [len(p) for p in decoding_prompts]
        return tokens, positions

    separate = build(checkpoint, max_batch=4)
    tokens, positions = setup(separate)
    decode_ref = separate.decode([0, 1], tokens, positions)
    calls = [(2, new_prompts[0], 0), (3, new_prompts[1], 0)]
    prefill_ref = separate.prefill_batch(calls)

    mixed = build(checkpoint, max_batch=4)
    setup(mixed)
    decode_got, prefill_got = mixed.decode_mixed([0, 1], tokens, positions, calls)

    assert torch.allclose(decode_got, decode_ref, atol=1e-4, rtol=1e-4)
    for got, want in zip(prefill_got, prefill_ref, strict=True):
        assert torch.allclose(got, want, atol=1e-4, rtol=1e-4)
        assert int(got.argmax()) == int(want.argmax())


def test_decode_mixed_with_no_prefill_calls_matches_plain_decode(checkpoint: Path) -> None:
    """`decode_mixed`'s documented fallback: an empty `prefill_calls` must be indistinguishable
    from calling `decode` directly (`scheduler._mixed_step` relies on this when a mixed
    iteration's adaptive budget admits only exact-duplicate matches, or none at all)."""
    prompts = [prompt_of(21, 5), prompt_of(22, 6)]

    plain = build(checkpoint, max_batch=2)
    for slot, p in enumerate(prompts):
        plain.begin(slot)
        plain.prefill(slot, p, 0)
    tokens, positions = [p[-1] for p in prompts], [len(p) for p in prompts]
    plain_logits = plain.decode([0, 1], tokens, positions)

    mixed = build(checkpoint, max_batch=2)
    for slot, p in enumerate(prompts):
        mixed.begin(slot)
        mixed.prefill(slot, p, 0)
    mixed_logits, prefill_logits = mixed.decode_mixed([0, 1], tokens, positions, [])

    assert prefill_logits == []
    assert torch.equal(mixed_logits, plain_logits)


def test_decode_mixed_refuses_under_mtp(checkpoint: Path) -> None:
    """Design constraint (`model.py`'s `Model.decode_mixed` docstring, `scheduler.py`'s
    `SEED_MIXED_BATCH` docstring): mixing is out of scope for v1 under MTP. This tiny
    checkpoint carries no MTP weights (`self.mtp` is `None` already), so the guard is checked
    directly against a sentinel rather than standing up real MTP weights just to reach it."""
    m = build(checkpoint, max_batch=1)
    m.begin(0)
    m.mtp = object()  # sentinel: only `is not None` is ever checked
    with pytest.raises(ValueError, match="SEED_MTP"):
        m.decode_mixed([0], [1], [1], [(0, [2, 3], 0)])


# ---------------------------------------------------------------- (b) scheduler integration


def test_mixed_batch_scheduler_matches_serial_generation_across_turns(checkpoint: Path) -> None:
    """End to end: a long-lived turn-1 request already decoding, then a turn-2 admission burst
    (the exact "prefill stalls decode" scenario the campaign brief measures) through a
    scheduler with `SEED_MIXED_BATCH` on. Output must match one-sequence-at-a-time
    `Model.generate`, and turn 2 must still hit turn 1's published prefix."""
    session = [prompt_of(31, 5)]
    session.append(session[0] + prompt_of(32, 6))

    reference_model = build(checkpoint, max_batch=1)
    expected = [reference(reference_model, p, 5) for p in session]

    model = build(checkpoint, max_batch=3)
    sched = make_scheduler(model, prefill_chunk=3, token_budget=4, mixed_batch=True)

    long_sink = Sink()
    sched.submit(Request(list(session[0]), 5, 0.0, frozenset(), long_sink))
    for _ in range(50):  # run turn 1 partway into decode before turn 2 arrives
        if sched.decoding:
            break
        assert sched.step()
    assert sched.decoding, "setup: turn 1 never started decoding"

    burst_sink = Sink()
    sched.submit(Request(list(session[1]), 5, 0.0, frozenset(), burst_sink))
    drain(sched)

    assert long_sink.error is None and burst_sink.error is None
    assert long_sink.tokens == expected[0]
    assert burst_sink.tokens == expected[1], "turn 2 diverged from serial generation"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

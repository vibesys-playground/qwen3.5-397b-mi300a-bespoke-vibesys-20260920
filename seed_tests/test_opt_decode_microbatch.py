"""Hermetic CPU tests: the decode microbatch count no longer ignores the step's batch size.

`Model.decode` used to cut every step into `self.microbatches` microbatches
(`len(stages) * MICROBATCHES_PER_STAGE`, a ceiling fixed at construction) regardless of how
many slots the step actually held. At the module default (4 stages, 2 per stage: a ceiling
of 8) that ceiling is also the step's own slot count once the batch reaches 8, so every
microbatch got exactly 1 slot: nothing left to batch, only 4 full per-layer dispatch and
per-microbatch MoE weight-read passes stacked back to back instead of one. Measured on 4x
MI300A with the real checkpoint, at merge base 837fd861 merged with opt-mxfp4-fused-gemv and
opt-attn-decode-batch:

    batch   ms/tok   tok/s
        1   107.04        9
        8   844.77        9
       16   795.26       20
       48  1116.74       43

An 8x smaller step at batch 8 cost 7.9x more time per token than batch 1, tok/s barely
moved, and batch 16 (a less degenerate split: 16 slots over the same 8-way ceiling is 2 per
microbatch, not 1) was faster than batch 8. `_decode_group_count` keeps at least
`MIN_MICROBATCH_SLOTS` per microbatch, falling back toward fewer, larger microbatches (down
to one undivided batch) instead of hitting the ceiling on a step too small to spend it on.

A follow-up sweep of `MIN_MICROBATCH_SLOTS` at batch 48 (the deployed `max_batch`, real
hardware, mean of 3 runs each) found splitting was a net loss at every microbatch count tried:

    min_slots  groups   ms/tok   tok/s
            6       8  1233.77     39   (the split above)
            8       6   992.23     48
           12       4   677.15     71
           16       3   577.30     83
           24       2   428.58    112
           49       1   311.15    154   (undivided)

Monotonic to undivided, 4.0x faster than the split this module shipped with. See
`MIN_MICROBATCH_SLOTS`'s docstring in model.py for why (the fused MXFP4 GEMV, batched
attention and batched DeltaNet all cut the per-layer cost the pipeline's fill/drain bubble
existed to hide). The default is now 49: every batch size up to 48 is undivided.

    /tmp/torchenv/bin/python -m pytest \\
        examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/seed_tests/test_opt_decode_microbatch.py
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import model as seed_model  # noqa: E402
from test_batched_decode import MAX_SEQ, prompt_of  # noqa: E402
from test_seed_parity import build_hf, write_checkpoint  # noqa: E402

STAGES = 4
DEVICES = ["cpu"] * STAGES
"""Four stages on one CPU, same trick test_pipeline.py uses: `Model` keys stages on the
position in `devices`, not the device object, so this gets the real 4-stage control flow
without four GPUs."""


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("tiny-decode-microbatch")
    write_checkpoint(build_hf(), out, mxfp4=False)
    return out


def build(checkpoint: Path, max_batch: int) -> seed_model.Model:
    return seed_model.Model(checkpoint, DEVICES, torch.float32, MAX_SEQ, max_batch)


def prefilled(checkpoint: Path, prompts: list[list[int]]) -> seed_model.Model:
    model = build(checkpoint, max_batch=len(prompts))
    for slot, prompt in enumerate(prompts):
        model.begin(slot)
        model.prefill(slot, prompt, 0)
    return model


# ---------------------------------------------------------------- (1) the group-count math


def test_decode_group_count_keeps_a_slot_floor_per_microbatch() -> None:
    f = seed_model._decode_group_count
    # Below the floor: collapse toward fewer, larger microbatches rather than hit the ceiling.
    assert f(1, ceiling=8, min_slots=6) == 1
    assert f(8, ceiling=8, min_slots=6) == 1  # the reported regression: was 8 groups of 1
    assert f(11, ceiling=8, min_slots=6) == 1
    assert f(12, ceiling=8, min_slots=6) == 2
    # At and above a floor of 6, 48 slots reaches the ceiling: the split this module shipped
    # with, which the batch-48 sweep found a 4.0x regression against undivided (see
    # MIN_MICROBATCH_SLOTS's docstring) -- not this fix's concern, but worth pinning so a
    # future reader does not mistake "reaches the ceiling" for "is therefore fast."
    assert f(48, ceiling=8, min_slots=6) == 8
    assert f(100, ceiling=8, min_slots=6) == 8  # never exceeds the ceiling
    # A pipeline-less model (ceiling 1) or a disabled floor are unaffected.
    assert f(48, ceiling=1, min_slots=6) == 1
    assert f(48, ceiling=8, min_slots=0) == 8
    # At the current default (49), every batch size up to 48 is undivided.
    assert f(1, ceiling=8, min_slots=49) == 1
    assert f(48, ceiling=8, min_slots=49) == 1
    assert f(49, ceiling=8, min_slots=49) == 1
    assert f(50, ceiling=8, min_slots=49) == 1  # 50 // 49 == 1: still one group, not two


def test_decode_group_count_never_exceeds_the_batch_or_drops_below_one() -> None:
    for n in (0, 1, 2, 5, 7, 8, 16, 48):
        for ceiling in (1, 4, 8, 16):
            for min_slots in (0, 1, 6, 100):
                groups = seed_model._decode_group_count(n, ceiling, min_slots)
                assert groups >= 1
                assert groups <= max(1, ceiling)


# ---------------------------------------------------------------- (2) the model actually uses it


def test_a_batch_at_the_ceiling_is_no_longer_split_to_single_slot_microbatches(
    checkpoint: Path,
) -> None:
    """The regression itself: 8 slots at the default ceiling of 8 used to become 8 microbatches
    of 1 slot each. It is now one undivided batch, since 8 is below `MIN_MICROBATCH_SLOTS`
    (49 at the current default; still true at the earlier default of 6 as well).
    """
    model = build(checkpoint, max_batch=8)
    assert model.microbatches == STAGES * seed_model.MICROBATCHES_PER_STAGE == 8  # unchanged
    slots, positions = list(range(8)), [4] * 8
    x = torch.zeros(8, 1, model.cfg.hidden)
    micros = model._microbatches(slots, positions, x)
    sizes = [len(m.slots) for m in micros]
    assert sizes == [8], sizes  # one microbatch, not eight of size one
    model.close()


def test_the_deployed_max_batch_is_undivided_at_the_current_default(checkpoint: Path) -> None:
    """48 slots used to cut into 8 microbatches of 6 (the split `MICROBATCHES_PER_STAGE` was
    tuned against). The batch-48 sweep in this module's docstring found that split a 4.0x
    regression against undivided at the same batch size, so the current default
    (`MIN_MICROBATCH_SLOTS` 49) makes it one undivided microbatch instead."""
    model = build(checkpoint, max_batch=48)
    slots, positions = list(range(48)), [4] * 48
    x = torch.zeros(48, 1, model.cfg.hidden)
    micros = model._microbatches(slots, positions, x)
    sizes = [len(m.slots) for m in micros]
    assert sizes == [48], sizes
    model.close()


def test_a_batch_past_the_floor_splits_into_more_than_one_microbatch(checkpoint: Path) -> None:
    """The floor does not disable splitting outright: a step with enough slots to spare (twice
    the floor, here) gets more than one microbatch, each still at least the floor wide."""
    model = build(checkpoint, max_batch=100)
    slots, positions = list(range(100)), [4] * 100
    x = torch.zeros(100, 1, model.cfg.hidden)
    micros = model._microbatches(slots, positions, x)
    sizes = [len(m.slots) for m in micros]
    assert sum(sizes) == 100
    assert len(sizes) > 1, sizes
    assert min(sizes) >= seed_model.MIN_MICROBATCH_SLOTS, sizes
    model.close()


def test_a_batch_many_times_the_floor_reaches_the_pipeline_ceiling(checkpoint: Path) -> None:
    """With enough slots to spare `ceiling` times the floor, splitting reaches the same
    pipeline-depth ceiling the 48-slot case used to hit at the old, lower floor."""
    model = build(checkpoint, max_batch=400)
    slots, positions = list(range(400)), [4] * 400
    x = torch.zeros(400, 1, model.cfg.hidden)
    micros = model._microbatches(slots, positions, x)
    sizes = [len(m.slots) for m in micros]
    assert len(sizes) == model.microbatches == STAGES * seed_model.MICROBATCHES_PER_STAGE
    assert sum(sizes) == 400
    assert min(sizes) >= seed_model.MIN_MICROBATCH_SLOTS, sizes
    model.close()


@pytest.mark.parametrize("batch", [1, 4, 7, 8, 11, 12, 16, 48])
def test_no_microbatch_is_smaller_than_the_floor_unless_the_whole_step_is(
    checkpoint: Path, batch: int
) -> None:
    model = build(checkpoint, max_batch=batch)
    slots, positions = list(range(batch)), [4] * batch
    x = torch.zeros(batch, 1, model.cfg.hidden)
    micros = model._microbatches(slots, positions, x)
    sizes = [len(m.slots) for m in micros]
    assert sum(sizes) == batch
    if len(sizes) > 1:
        assert min(sizes) >= seed_model.MIN_MICROBATCH_SLOTS, sizes
    else:
        assert sizes == [batch]  # every batch size up to 48 is undivided at the current default
    model.close()


# ---------------------------------------------------------------- (3) correctness is unaffected


def test_an_undivided_small_batch_still_matches_each_slot_decoded_alone(checkpoint: Path) -> None:
    """Collapsing 8 slots to one microbatch (instead of 8 of size 1) must not change what the
    step computes: each slot's logits still match that slot decoded on its own."""
    prompts = [prompt_of(300 + i, 4 + i) for i in range(8)]
    positions = [len(p) for p in prompts]

    batched = prefilled(checkpoint, prompts)
    assert len(batched._microbatches(list(range(8)), positions, torch.zeros(8, 1, 1))) == 1
    together = batched.decode(list(range(8)), [p[-1] for p in prompts], positions)

    alone = prefilled(checkpoint, prompts)
    for slot, prompt in enumerate(prompts):
        alone.bind(slot)
        row = alone.forward(torch.tensor([[prompt[-1]]]), positions[slot])[-1]
        torch.testing.assert_close(together[slot], row, atol=1e-4, rtol=1e-4, msg=f"slot {slot}")
    batched.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

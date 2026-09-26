"""Hermetic CPU tests for the env-gated fault-injection hooks in `model.py`.

Each hook must be a no-op when its flag is unset (regression protection: default behavior is
unchanged) and must actually perturb output logits when set (so the accuracy-gate calibration
plan can rely on it producing a detectably wrong model). Tests set the module-level flags
directly (`model.FAULT_*`) rather than through `os.environ` plus a re-import: the flags are
plain globals read at call time in the hot functions, not import-time constants baked into
closures, so a direct `monkeypatch.setattr` is equivalent and far cheaper than a subprocess
or importlib reload.

    /tmp/torchenv/bin/python -m pytest \\
        examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/seed_tests/test_fault_injection.py
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import model as seed_model  # noqa: E402
from test_batched_decode import build, prompt_of  # noqa: E402
from test_seed_parity import build_hf, write_checkpoint  # noqa: E402

FAULT_FLAGS = (
    "FAULT_ROPE_BASE",
    "FAULT_DROP_EXPERT",
    "FAULT_SKIP_DELTANET_GATE",
    "FAULT_FP8_KV_NO_SCALE",
    "FAULT_FLIP_CAUSAL_MASK",
)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("tiny")
    write_checkpoint(build_hf(), out, mxfp4=False)
    return out


@pytest.fixture(autouse=True)
def _clean_flags() -> None:
    """Every fault flag defaults False before and after each test, whatever it sets."""
    for name in FAULT_FLAGS:
        assert getattr(seed_model, name) is False, f"{name} leaked on from a prior test"
    yield
    for name in FAULT_FLAGS:
        setattr(seed_model, name, False)
    seed_model.FAULT_FLIP_CAUSAL_MASK_LAYER = 0


def _prefill_logits(checkpoint: Path, prompt: list[int]) -> torch.Tensor:
    m = build(checkpoint, max_batch=1)
    m.begin(0)
    return m.prefill(0, prompt, 0)


def _baseline(checkpoint: Path) -> torch.Tensor:
    prompt = prompt_of(seed=7, length=6)
    return _prefill_logits(checkpoint, prompt)


def test_all_flags_off_is_unchanged(checkpoint: Path) -> None:
    """No flags set at all: two independent builds must agree exactly (sanity on the fixture,
    not the hooks -- establishes the baseline every other test diffs against)."""
    a = _baseline(checkpoint)
    b = _baseline(checkpoint)
    assert torch.equal(a, b)


@pytest.mark.parametrize("flag", FAULT_FLAGS)
def test_flag_changes_logits_when_on(checkpoint: Path, flag: str) -> None:
    baseline = _baseline(checkpoint)
    setattr(seed_model, flag, True)
    prompt = prompt_of(seed=7, length=6)
    faulty = _prefill_logits(checkpoint, prompt)
    assert not torch.equal(baseline, faulty), f"{flag}=True produced identical logits"


def test_rope_base_off_matches_baseline_bit_for_bit(checkpoint: Path) -> None:
    assert seed_model.FAULT_ROPE_BASE is False
    prompt = prompt_of(seed=7, length=6)
    assert torch.equal(_baseline(checkpoint), _prefill_logits(checkpoint, prompt))


def test_flip_causal_mask_targets_only_the_selected_full_attention_layer(
    checkpoint: Path,
) -> None:
    """Layer index is 0-based among full-attention layers, not the global layer index.
    `tiny_config` (test_seed_parity.py) has exactly one full-attention layer (index 2 of 4),
    so flipping "full-attention layer 0" must still perturb output (there is one to flip),
    while flipping "full-attention layer 1" (out of range: only one exists) must be a no-op.
    """
    prompt = prompt_of(seed=7, length=6)
    baseline = _prefill_logits(checkpoint, prompt)

    seed_model.FAULT_FLIP_CAUSAL_MASK = True
    seed_model.FAULT_FLIP_CAUSAL_MASK_LAYER = 0
    flipped = _prefill_logits(checkpoint, prompt)
    assert not torch.equal(baseline, flipped)

    seed_model.FAULT_FLIP_CAUSAL_MASK_LAYER = 1
    out_of_range = _prefill_logits(checkpoint, prompt)
    assert torch.equal(baseline, out_of_range)


def _score_logits(checkpoint: Path, prompt: list[int], continuation_start: int) -> torch.Tensor:
    """`Model.score`'s own call shape, not `prefill`'s: this is the function `/v1/score`
    actually calls (see server.score_continuation), returning every scored position's
    logits, not just the last token's."""
    m = build(checkpoint, max_batch=1)
    m.begin(0)
    return m.score(0, prompt, continuation_start)


@pytest.mark.parametrize("flag", FAULT_FLAGS)
def test_flag_changes_every_v1_score_position_when_on(checkpoint: Path, flag: str) -> None:
    """Regression coverage for a gap `test_flag_changes_logits_when_on` above does not close:
    that test diffs `prefill`'s last-token logits, but `/v1/score` calls `Model.score` and
    reports *every* continuation position, not just the last. A fault whose direct effect is
    concentrated on early positions (`FAULT_FLIP_CAUSAL_MASK`: the last token in a causal
    model already sees its whole past, so flipping one layer to bidirectional barely changes
    *that* token's own attention output -- see the calibration note below) could look broken
    through `prefill`'s lens while still being wired correctly, or vice versa. This asserts
    each fault perturbs the logits `/v1/score` actually returns, at every scored position, on
    a tiny model deep enough to have more than one layer after the fault site.

    Context: a 2026-09-23 GPU fault sweep (`bespoke/opt-score-endpoint`'s `/debug/fault`
    against the real 397B checkpoint) found `FAULT_FLIP_CAUSAL_MASK` and
    `FAULT_FP8_KV_NO_SCALE` produced ~0.02-0.03 mean |logprob diff| and zero top-1 flips
    across 383 high-margin positions, next to `FAULT_ROPE_BASE`/`FAULT_DROP_EXPERT`'s ~0.03-
    0.05 mean and one flip each -- all four looked roughly as weak as each other, next to
    `FAULT_SKIP_DELTANET_GATE`'s catastrophic ~10.7 mean. This test (and the manual probe
    behind it) confirms the mechanism is not silently disconnected from `/v1/score`'s path:
    on a tiny model it produces a clear, decaying-toward-the-last-position signal for both
    faults (largest near the first scored position, smallest at the very last one -- exactly
    what a single-layer, single-fact perturbation should look like once a few more layers and
    the residual stream have had a chance to mix it back in). The GPU run's weak numbers are
    better explained by (a) `FAULT_FLIP_CAUSAL_MASK_LAYER=0` landing on the *first* of ~15
    full-attention layers in the real 60-layer stack, 57 more layers of dilution than this
    4-layer tiny model has, and (b) that run's fallback reference reused the old 14-pin v1
    set, which has no long-context prompt -- `FAULT_FP8_KV_NO_SCALE`'s error is expected to
    grow with context length (see the fault table above), and the v1 pins top out around 32
    tokens. Neither is a wiring bug; both point at data coverage. Re-run against the v2 pin
    set (which has 4k/8k-token prompts) before concluding these two faults need a stronger
    hook rather than a longer prompt.
    """
    prompt = prompt_of(seed=7, length=20)
    continuation_start = 12
    baseline = _score_logits(checkpoint, prompt, continuation_start)
    setattr(seed_model, flag, True)
    faulty = _score_logits(checkpoint, prompt, continuation_start)
    diff = (faulty - baseline).abs()
    assert diff.max().item() > 1e-4, f"{flag}=True left every /v1/score position unchanged"
    per_position_changed = (diff.max(dim=-1).values > 1e-4).tolist()
    assert any(per_position_changed), f"{flag}=True changed no scored position at all"


def test_fp8_kv_no_scale_is_read_back_in_the_same_forward_that_writes_it(
    checkpoint: Path,
) -> None:
    """Answers the coordinator's specific question: does `/v1/score`'s one-shot forward
    (`Model.score`, `start=0`, the whole prompt in one call) actually read the truncated KV
    back, or does the write happen too late to affect the attention this same call computes?

    `full_attention` truncates `k`/`v` to fp8 *before* writing them into `st["k"]`/`st["v"]`
    (see the `FAULT_FP8_KV_NO_SCALE` branch, above the cache write two lines below it), and
    then immediately slices `keys, vals = st["k"][..., :start+t], st["v"][..., :start+t]` from
    that same buffer for the SDPA call -- so a one-shot `/v1/score` forward does read back
    what it just truncated, same call, no staleness window. Pinned here on the first scored
    position, which depends on every earlier key/value in the same forward and would be the
    first place a stale (unwritten-back) read would go undetected.
    """
    prompt = prompt_of(seed=7, length=20)
    continuation_start = 12
    baseline = _score_logits(checkpoint, prompt, continuation_start)
    seed_model.FAULT_FP8_KV_NO_SCALE = True
    faulty = _score_logits(checkpoint, prompt, continuation_start)
    assert not torch.equal(baseline[0], faulty[0]), (
        "the first scored position's logits are unaffected by FAULT_FP8_KV_NO_SCALE -- "
        "suggests the truncated KV is not being read back within /v1/score's own forward"
    )


def test_rope_base_takes_effect_on_the_next_forward_of_a_live_model(checkpoint: Path) -> None:
    """Regression test for a runtime-switchability bug: `/debug/fault` flips
    `FAULT_ROPE_BASE` on an already-running server, not a freshly constructed model, and
    `Model._rope_table` caches its cos/sin table per device. A cache keyed on device alone
    would keep serving whichever table got built first (off or on) regardless of later
    flips; the fix keys it on `(device, FAULT_ROPE_BASE)` instead. Unlike the other tests in
    this file (each of which builds a brand-new `Model` under one fixed flag value, so a
    stale per-device cache could never show up), this one flips the flag twice on a single
    live model, exactly as the calibration sweep does between configs.
    """
    prompt = prompt_of(seed=7, length=6)
    m = build(checkpoint, max_batch=1)

    m.begin(0)
    off_first = m.prefill(0, prompt, 0)

    seed_model.FAULT_ROPE_BASE = True
    m.begin(0)
    on = m.prefill(0, prompt, 0)
    assert not torch.equal(off_first, on), "flipping the flag on had no effect on a live model"

    seed_model.FAULT_ROPE_BASE = False
    m.begin(0)
    off_second = m.prefill(0, prompt, 0)
    assert torch.equal(off_first, off_second), "flipping the flag back off did not restore it"

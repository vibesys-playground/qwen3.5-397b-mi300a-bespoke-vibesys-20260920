"""Hermetic CPU tests: batched decode-time DeltaNet against the old per-slot-loop behavior.

`Model.deltanet_decode` used to loop over `slots` and call the single-sequence `deltanet`
once per slot (T=1 each time). It now gathers every active slot's conv/recurrent state out
of the pool, runs the same recurrent-form math with a real batch dimension, and scatters the
updated state back. `old_deltanet_decode` below reproduces the removed loop exactly (it is
just `bind` then `deltanet`, both still present and unchanged), so it is the ground truth for
"what the per-slot loop used to produce", including the updated pool state, not only the
output tensor.

    /tmp/torchenv/bin/python -m pytest \\
        examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/seed_tests/test_opt_deltanet_batch_slots.py
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import model as seed_model  # noqa: E402
from test_batched_decode import build, decode_activations, layer_index, prompt_of  # noqa: E402
from test_seed_parity import build_hf, write_checkpoint  # noqa: E402

MAX_SEQ = 96
ATOL, RTOL = 1e-5, 1e-5


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("tiny")
    write_checkpoint(build_hf(), out, mxfp4=False)
    return out


def old_deltanet_decode(
    model: seed_model.Model, i: int, x: torch.Tensor, slots: list[int]
) -> torch.Tensor:
    """The removed implementation: bind each slot and call the single-sequence `deltanet`.

    Both `bind` and `deltanet` are unchanged, so this is not a reimplementation of the old
    behavior, it is the old behavior, run against whichever model instance is passed in.
    """
    outs = []
    for j, slot in enumerate(slots):
        model.bind(slot)
        outs.append(model.deltanet(i, x[j : j + 1]))
    return torch.cat(outs, dim=0)


def pool_state(model: seed_model.Model, i: int, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
    pool = model.pool[i]
    return pool["conv"][slot].clone(), pool["rec"][slot].clone()


def assert_same_output_and_state(
    old: seed_model.Model, new: seed_model.Model, i: int, slots: list[int], label: str
) -> None:
    for slot in slots:
        old_conv, old_rec = pool_state(old, i, slot)
        new_conv, new_rec = pool_state(new, i, slot)
        torch.testing.assert_close(
            new_conv, old_conv, atol=ATOL, rtol=RTOL, msg=f"{label}: slot {slot} conv state"
        )
        torch.testing.assert_close(
            new_rec, old_rec, atol=ATOL, rtol=RTOL, msg=f"{label}: slot {slot} rec state"
        )


def run_scenario(
    checkpoint: Path, max_batch: int, prompts: dict[int, list[int]], steps: list[list[int]]
) -> None:
    """Prefill `prompts` (slot -> prompt) into fresh old/new models, then run `steps`.

    Each entry of `steps` is the list of active slots for one decode step, in the order the
    scheduler would pass them. Every step's output and every touched slot's updated pool
    state (conv, rec) are compared between the old per-slot loop and the new batched call.
    """
    old, new = build(checkpoint, max_batch), build(checkpoint, max_batch)
    for slot, prompt in prompts.items():
        for m in (old, new):
            m.begin(slot)
            m.prefill(slot, prompt, 0)

    i = layer_index(old, "linear_attention")
    idle = [s for s in range(max_batch) if s not in prompts]
    idle_before = {s: pool_state(old, i, s) for s in idle}

    for step, slots in enumerate(steps):
        x = decode_activations(old, len(slots), seed=1000 + step)
        old_out = old_deltanet_decode(old, i, x, slots)
        new_out = new.deltanet_decode(i, x, slots)

        assert new_out.shape == old_out.shape == x.shape
        torch.testing.assert_close(
            new_out, old_out, atol=ATOL, rtol=RTOL, msg=f"step {step} output"
        )
        assert_same_output_and_state(old, new, i, slots, f"step {step}")

    # Slots never named in `prompts` or any step must be exactly as `_new_pool` left them: the
    # batched call only reads/writes the rows it was given.
    for s in idle:
        conv, rec = pool_state(new, i, s)
        want_conv, want_rec = idle_before[s]
        assert torch.equal(conv, want_conv), f"idle slot {s} conv corrupted"
        assert torch.equal(rec, want_rec), f"idle slot {s} rec corrupted"
        assert torch.equal(rec, torch.zeros_like(rec)), f"idle slot {s} rec should still be zero"


# ---------------------------------------------------------------- scenarios


def test_one_active_slot(checkpoint: Path) -> None:
    prompts = {0: prompt_of(1, 5)}
    run_scenario(checkpoint, max_batch=4, prompts=prompts, steps=[[0], [0], [0]])


def test_a_few_active_slots_with_gaps(checkpoint: Path) -> None:
    """Active slots are not contiguous, matching how eviction leaves holes in the pool."""
    prompts = {0: prompt_of(2, 6), 2: prompt_of(3, 4), 4: prompt_of(4, 8)}
    run_scenario(checkpoint, max_batch=6, prompts=prompts, steps=[[0, 2, 4], [4, 0, 2], [0, 2, 4]])


def test_full_batch(checkpoint: Path) -> None:
    prompts = {s: prompt_of(10 + s, 3 + s) for s in range(5)}
    steps = [list(range(5)) for _ in range(2)]
    run_scenario(checkpoint, max_batch=5, prompts=prompts, steps=steps)


def test_mixed_active_inactive_and_newly_added_slots(checkpoint: Path) -> None:
    """A slot idles for the whole run, and one is evicted and reused mid-sequence.

    This is what the scheduler actually does: `_evict` and `begin` hand a freshly reset slot
    to a new session between decode steps, and unused slots simply never appear in `slots`.
    """
    max_batch = 4
    prompts = {0: prompt_of(20, 5), 1: prompt_of(21, 4), 2: prompt_of(22, 6)}
    # slot 3 is never begun or prefilled: it stays idle for the whole test.
    old, new = build(checkpoint, max_batch), build(checkpoint, max_batch)
    for slot, prompt in prompts.items():
        for m in (old, new):
            m.begin(slot)
            m.prefill(slot, prompt, 0)

    i = layer_index(old, "linear_attention")

    def step(slots: list[int], seed: int) -> None:
        x = decode_activations(old, len(slots), seed=seed)
        old_out = old_deltanet_decode(old, i, x, slots)
        new_out = new.deltanet_decode(i, x, slots)
        torch.testing.assert_close(new_out, old_out, atol=ATOL, rtol=RTOL, msg=f"slots={slots}")
        assert_same_output_and_state(old, new, i, slots, f"slots={slots}")

    step([0, 1, 2], seed=2001)  # all three sessions decode together

    # session in slot 1 finishes; the scheduler evicts it and starts a new session there.
    new_prompt = prompt_of(23, 7)
    for m in (old, new):
        m.begin(1)  # resets slot 1's conv/rec state before the new prefill
        m.prefill(1, new_prompt, 0)

    step([0, 1, 2], seed=2002)  # slot 1 is now mid-sequence for its *new* session
    step([2, 0], seed=2003)  # order need not match the previous step
    step([0], seed=2004)  # slot 2 goes idle without ever being touched again this test

    # Slot 2 was computed on (but not touched by this test's last step); the batched and
    # per-slot-loop matmuls reassociate float32 additions differently (same tolerance as the
    # MoE/attention batching checks), so state agreement here is float-close, not bit-exact.
    conv2, rec2 = pool_state(new, i, 2)
    old_conv2, old_rec2 = pool_state(old, i, 2)
    torch.testing.assert_close(
        conv2, old_conv2, atol=ATOL, rtol=RTOL, msg="slot 2 conv diverged while idle"
    )
    torch.testing.assert_close(
        rec2, old_rec2, atol=ATOL, rtol=RTOL, msg="slot 2 rec diverged while idle"
    )

    # Slot 3 was never begun, prefilled, or named in any step's `slots`: no float op ever
    # touched it, so it must still be exactly what `_new_pool` allocated.
    conv3, rec3 = pool_state(new, i, 3)
    assert torch.equal(conv3, torch.zeros_like(conv3)), "never-used slot 3 conv should stay zero"
    assert torch.equal(rec3, torch.zeros_like(rec3)), "never-used slot 3 rec should stay zero"


# ---------------------------------------------------------------- batched-vs-solo property


def test_batched_slot_matches_that_slot_decoded_alone(checkpoint: Path) -> None:
    """Same property style as the server-batching and micro-sweep checks: a batched call's
    per-slot output and state must equal running that slot alone (T=1, single-sequence)."""
    prompts = [prompt_of(30 + s, 4 + s) for s in range(4)]
    solo_model = build(checkpoint, max_batch=4)
    batched_model = build(checkpoint, max_batch=4)
    for slot, prompt in enumerate(prompts):
        for m in (solo_model, batched_model):
            m.begin(slot)
            m.prefill(slot, prompt, 0)

    i = layer_index(solo_model, "linear_attention")
    slots = list(range(4))
    x = decode_activations(solo_model, len(slots), seed=42)

    solo_outs = []
    for j, slot in enumerate(slots):
        solo_model.bind(slot)
        solo_outs.append(solo_model.deltanet(i, x[j : j + 1]))
    solo_out = torch.cat(solo_outs, dim=0)

    batched_out = batched_model.deltanet_decode(i, x, slots)

    torch.testing.assert_close(batched_out, solo_out, atol=ATOL, rtol=RTOL)
    for slot in slots:
        solo_conv, solo_rec = pool_state(solo_model, i, slot)
        batch_conv, batch_rec = pool_state(batched_model, i, slot)
        torch.testing.assert_close(
            batch_conv, solo_conv, atol=ATOL, rtol=RTOL, msg=f"slot {slot} conv"
        )
        torch.testing.assert_close(
            batch_rec, solo_rec, atol=ATOL, rtol=RTOL, msg=f"slot {slot} rec"
        )


def test_decode_must_not_take_the_chunked_path(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The batched T=1 call must still dispatch to the recurrent form, never the chunked one."""
    prompts = [prompt_of(40 + s, 4) for s in range(3)]
    model = build(checkpoint, max_batch=3)
    for slot, prompt in enumerate(prompts):
        model.begin(slot)
        model.prefill(slot, prompt, 0)

    i = layer_index(model, "linear_attention")
    slots = list(range(3))
    x = decode_activations(model, len(slots), seed=7)

    monkeypatch.setattr(
        seed_model,
        "delta_rule_chunked",
        lambda *a, **k: pytest.fail("batched decode must not take the chunked path"),
    )
    out = model.deltanet_decode(i, x, slots)
    assert out.shape == x.shape


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

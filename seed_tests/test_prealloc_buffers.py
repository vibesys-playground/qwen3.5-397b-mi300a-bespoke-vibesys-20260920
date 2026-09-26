"""Hermetic CPU tests for the decode-hot-path scratch buffers (`Model.delta_scratch`,
`Model.moe_scratch`): buffer reuse must match the old fresh-allocation-every-call behavior
exactly, and a step must never see data left over from a previous step or a different slot.

    /tmp/torchenv/bin/python -m pytest \\
        examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/seed_tests/test_prealloc_buffers.py \\
        -p no:cacheprovider --no-cov
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import model as seed_model  # noqa: E402
from test_delta_rule import HEADS, V_DIM, random_inputs  # noqa: E402
from test_moe_vectorize import build_layer, fake_model, moe_via_model  # noqa: E402
from test_seed_parity import VOCAB, build_hf, write_checkpoint  # noqa: E402

MAX_SEQ = 96
GARBAGE = 1e30  # loud sentinel: any position the code fails to (re)write stays visibly wrong


# ---------------------------------------------------------------- (a) delta_rule_recurrent


def test_delta_rule_recurrent_reused_buffer_matches_fresh_alloc() -> None:
    """A caller-supplied `out` gets the exact values a fresh `torch.empty_like` call would."""
    buf = torch.full((1, 1, HEADS, V_DIM), GARBAGE)
    for step in range(5):
        args = random_inputs(seed=step, t=1, heads=HEADS)
        want = seed_model.delta_rule_recurrent(
            args["q"], args["k"], args["v"], args["g"], args["beta"], args["rec"].clone()
        )
        buf.fill_(GARBAGE)  # simulate a buffer still holding an older step's/slot's contents
        rec = args["rec"].clone()
        got = seed_model.delta_rule_recurrent(
            args["q"], args["k"], args["v"], args["g"], args["beta"], rec, buf
        )
        assert got.data_ptr() == buf.data_ptr(), "must write into the given buffer, not allocate"
        torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-6, msg=f"step {step}")


def test_delta_rule_recurrent_reused_buffer_allocates_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = random_inputs(seed=0, t=1, heads=HEADS)
    buf = torch.empty(1, 1, HEADS, V_DIM)
    calls = []
    real_empty_like = torch.empty_like
    monkeypatch.setattr(
        seed_model.torch, "empty_like", lambda *a, **k: calls.append(1) or real_empty_like(*a, **k)
    )
    seed_model.delta_rule_recurrent(
        args["q"], args["k"], args["v"], args["g"], args["beta"], args["rec"].clone(), buf
    )
    assert not calls, "torch.empty_like must not be called when a buffer is supplied"


def test_delta_rule_dispatch_passes_the_buffer_through_only_for_the_recurrent_form() -> None:
    """`Model.delta_rule`'s `out` reaches `delta_rule_recurrent` (T=1) and is ignored for the
    chunked (T>1, prefill) form, which always allocates its own output."""
    buf = torch.full((1, 1, HEADS, V_DIM), GARBAGE)
    args = random_inputs(seed=1, t=1, heads=HEADS)
    got = seed_model.Model.delta_rule(
        args["q"], args["k"], args["v"], args["g"], args["beta"], args["rec"].clone(), buf
    )
    assert got.data_ptr() == buf.data_ptr()

    args = random_inputs(seed=1, t=5, heads=HEADS)
    long_buf = torch.full((1, 5, HEADS, V_DIM), GARBAGE)
    got = seed_model.Model.delta_rule(
        args["q"], args["k"], args["v"], args["g"], args["beta"], args["rec"].clone(), long_buf
    )
    assert got.data_ptr() != long_buf.data_ptr()  # chunked path never touches `out`


# ---------------------------------------------------------------- (b) moe out_buf


def test_moe_out_buf_reused_across_varying_batch_sizes_matches_fresh() -> None:
    """Consecutive calls at different, non-monotonic token counts, sharing one `max_batch`
    buffer, must each match a fresh `torch.zeros_like` call -- including a step whose count
    shrinks right after a step that used the whole buffer."""
    hidden, inter, experts, top_k, max_batch = 32, 32, 4, 2, 6
    dtype = torch.float32
    cfg = SimpleNamespace(hidden=hidden, top_k=top_k)
    layer = build_layer(1, hidden, inter, experts, mxfp4=False, dtype=dtype)
    scratch = torch.full((max_batch, hidden), GARBAGE, dtype=dtype)

    for step, t in enumerate([6, 2, 5, 1, 6, 3]):  # full, shrink, grow, singleton, full, mid
        x = torch.randn(1, t, hidden, generator=torch.Generator().manual_seed(100 + step))
        want = moe_via_model(cfg, layer, x, dtype)

        scratch.fill_(GARBAGE)  # older step's leftover contents, on purpose
        got = seed_model.Model.moe(fake_model(cfg, layer, dtype), 0, x, scratch[:t])

        assert got.shape == want.shape
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5, msg=f"step {step} t={t}")


def test_moe_out_buf_reused_allocates_no_fresh_accumulator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hidden, inter, experts, top_k = 32, 32, 4, 2
    dtype = torch.float32
    cfg = SimpleNamespace(hidden=hidden, top_k=top_k)
    layer = build_layer(2, hidden, inter, experts, mxfp4=False, dtype=dtype)
    x = torch.randn(1, 3, hidden, generator=torch.Generator().manual_seed(7))
    fake_self = fake_model(cfg, layer, dtype)
    scratch = torch.empty(3, hidden, dtype=dtype)

    calls = []
    real_zeros_like = torch.zeros_like
    monkeypatch.setattr(
        seed_model.torch, "zeros_like", lambda *a, **k: calls.append(1) or real_zeros_like(*a, **k)
    )
    seed_model.Model.moe(fake_self, 0, x, scratch)
    assert not calls, "torch.zeros_like must not be called when out_buf is given"

    seed_model.Model.moe(fake_self, 0, x)  # sanity: the None-branch still allocates
    assert calls


# ---------------------------------------------------------------- (c) end to end: Model.decode


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("tiny")
    write_checkpoint(build_hf(), out, mxfp4=False)
    return out


def build(checkpoint: Path, max_batch: int) -> seed_model.Model:
    return seed_model.Model(checkpoint, ["cpu"], torch.float32, MAX_SEQ, max_batch)


def prompt_of(seed: int, length: int) -> list[int]:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(2, VOCAB, (length,), generator=gen).tolist()


def test_decode_matches_solo_across_several_steps_with_varying_active_slots(
    checkpoint: Path,
) -> None:
    """Several consecutive batched `decode()` calls, with the active-slot set shrinking,
    growing, and shrinking to a singleton between calls (not monotonic), must match
    one-slot-at-a-time `forward()` at every step. `moe_scratch` is keyed by row 0..len(slots)-1
    and `delta_scratch` by the bound slot index, so a step with fewer or different active slots
    than the previous one is exactly the scenario where a stale row could leak through.
    """
    max_batch = 4
    prompts = [prompt_of(200 + i, 5 + i) for i in range(max_batch)]

    solo_model, batched_model = build(checkpoint, max_batch), build(checkpoint, max_batch)
    for slot, prompt in enumerate(prompts):
        for m in (solo_model, batched_model):
            m.begin(slot)
            m.prefill(slot, prompt, 0)

    positions = [len(p) for p in prompts]
    next_tok = [p[-1] for p in prompts]
    subsets = [[0, 1, 2, 3], [0, 2], [1, 3], [0, 1, 2, 3], [2], [0, 1, 2, 3]]

    for step, subset in enumerate(subsets):
        solo_rows = {}
        for slot in subset:
            solo_model.bind(slot)
            solo_rows[slot] = solo_model.forward(torch.tensor([[next_tok[slot]]]), positions[slot])[
                -1
            ]
        batched = batched_model.decode(
            subset, [next_tok[s] for s in subset], [positions[s] for s in subset]
        )
        for j, slot in enumerate(subset):
            torch.testing.assert_close(
                batched[j],
                solo_rows[slot],
                atol=1e-4,
                rtol=1e-4,
                msg=f"step {step} subset {subset} slot {slot}",
            )
        for j, slot in enumerate(subset):
            next_tok[slot] = int(batched[j].argmax())
            positions[slot] += 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

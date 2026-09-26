"""Hermetic CPU tests for batching and prefix reuse against the real model code.

Uses the tiny random Qwen3.5-MoE checkpoint that test_seed_parity.py builds (4 layers,
both layer kinds, 8 experts), on CPU in float32. The reference is the seed's own
one-sequence `Model.generate`, so these tests say whether the new batched decode and
the prefix cache change what the model produces, not whether the model is right.

    /tmp/torchenv/bin/python -m pytest \\
        examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/seed_tests/test_batched_decode.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import model as seed_model  # noqa: E402
from scheduler import Request, Scheduler  # noqa: E402
from session_cache import SessionCache  # noqa: E402
from test_moe_vectorize import build_layer, moe_via_model  # noqa: E402
from test_scheduler import Sink, drain  # noqa: E402
from test_seed_parity import VOCAB, build_hf, write_checkpoint  # noqa: E402

MAX_SEQ = 96


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("tiny")
    write_checkpoint(build_hf(), out, mxfp4=False)
    return out


def build(checkpoint: Path, max_batch: int) -> seed_model.Model:
    return seed_model.Model(checkpoint, ["cpu"], torch.float32, MAX_SEQ, max_batch)


def make_scheduler(model: seed_model.Model, **kw) -> Scheduler:
    """A `Scheduler` over a real `Model`, with its own `SessionCache` (Stage 2: the cache is
    sized from the model's own pools -- `block_allocator`, `block_size`, `num_snapshots` --
    the same way `seed_tests/test_scheduler.py`'s `make()` builds one for `FakeRunner`)."""
    cache = SessionCache(model.block_allocator, model.block_size, model.num_snapshots)
    return Scheduler(model, cache, **kw)


def prompt_of(seed: int, length: int) -> list[int]:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(2, VOCAB, (length,), generator=gen).tolist()


def reference(model: seed_model.Model, prompt: list[int], max_new: int) -> list[int]:
    """One-at-a-time greedy generation: the behavior batching has to preserve."""
    return list(model.generate(prompt, max_new, 0.0, frozenset()))


def run(sched: Scheduler, specs: list[tuple[list[int], int]]) -> list[Sink]:
    sinks = [Sink() for _ in specs]
    for (prompt, max_new), sink in zip(specs, sinks, strict=True):
        sched.submit(Request(list(prompt), max_new, 0.0, frozenset(), sink))
    drain(sched)
    for sink in sinks:
        assert sink.error is None, sink.error
    return sinks


# ---------------------------------------------------------------- (a) batching


def test_batched_decode_logits_match_one_sequence_at_a_time(checkpoint: Path) -> None:
    """The tensor-level claim under the scheduler: a shared step equals separate steps."""
    prompts = [prompt_of(1, 7), prompt_of(2, 5), prompt_of(3, 9)]
    model = build(checkpoint, max_batch=len(prompts))
    solo = []
    for slot, prompt in enumerate(prompts):
        model.begin(slot)
        model.prefill(slot, prompt, 0)
        solo.append(model.forward(torch.tensor([[prompt[-1]]]), len(prompt))[-1])

    fresh = build(checkpoint, max_batch=len(prompts))
    for slot, prompt in enumerate(prompts):
        fresh.begin(slot)
        fresh.prefill(slot, prompt, 0)
    batched = fresh.decode(
        list(range(len(prompts))), [p[-1] for p in prompts], [len(p) for p in prompts]
    )

    assert batched.shape == (len(prompts), fresh.cfg.vocab)
    for i, row in enumerate(solo):
        assert torch.allclose(batched[i], row, atol=1e-4, rtol=1e-4), f"slot {i}"
        assert int(batched[i].argmax()) == int(row.argmax())


def test_concurrent_requests_produce_the_serial_tokens(checkpoint: Path) -> None:
    specs = [(prompt_of(11, 6), 7), (prompt_of(12, 11), 5), (prompt_of(13, 4), 9)]
    model = build(checkpoint, max_batch=3)
    expected = [reference(model, p, n) for p, n in specs]

    sched = make_scheduler(build(checkpoint, max_batch=3), prefill_chunk=4)
    for sink, want in zip(run(sched, specs), expected, strict=True):
        assert sink.tokens == want
    assert sink.end == ("length", specs[-1][1], 0)


def test_more_requests_than_slots_still_match_serial(checkpoint: Path) -> None:
    specs = [(prompt_of(20 + i, 5 + i), 4) for i in range(5)]
    model = build(checkpoint, max_batch=2)
    expected = [reference(model, p, n) for p, n in specs]

    sched = make_scheduler(build(checkpoint, max_batch=2), prefill_chunk=3)
    for sink, want in zip(run(sched, specs), expected, strict=True):
        assert sink.tokens == want


# ---------------------------------------------------------------- (b) prefix reuse


def test_prefix_reuse_matches_a_run_with_no_cache(checkpoint: Path) -> None:
    """A turn served from a cached prefix must generate what a cold server generates."""
    turns = [prompt_of(31, 6)]
    turns.append(turns[0] + prompt_of(32, 7))
    turns.append(turns[1] + prompt_of(33, 5))
    reference_model = build(checkpoint, max_batch=1)
    expected = [reference(reference_model, p, 6) for p in turns]

    warm = make_scheduler(build(checkpoint, max_batch=2), prefill_chunk=4)
    for turn, (prompt, want) in enumerate(zip(turns, expected, strict=True)):
        (sink,) = run(warm, [(prompt, 6)])
        assert sink.tokens == want, f"turn {turn} diverged when served from the prefix cache"


def test_prefix_reuse_holds_with_sessions_interleaved(checkpoint: Path) -> None:
    sessions = []
    for s in range(3):
        first = prompt_of(40 + s, 5)
        sessions.append([first, first + prompt_of(50 + s, 6)])
    reference_model = build(checkpoint, max_batch=1)
    expected = [[reference(reference_model, p, 5) for p in turns] for turns in sessions]

    sched = make_scheduler(build(checkpoint, max_batch=3), prefill_chunk=4)
    for turn in range(2):
        sinks = run(sched, [(s[turn], 5) for s in sessions])
        for i, sink in enumerate(sinks):
            assert sink.tokens == expected[i][turn], f"session {i} turn {turn}"


# ---------------------------------------------------------------- (c) eviction


def test_evicted_session_recomputes_correctly(checkpoint: Path) -> None:
    sessions = []
    for s in range(3):
        first = prompt_of(60 + s, 5)
        sessions.append([first, first + prompt_of(70 + s, 6)])
    reference_model = build(checkpoint, max_batch=1)
    expected = [[reference(reference_model, p, 4) for p in turns] for turns in sessions]

    sched = make_scheduler(build(checkpoint, max_batch=2), prefill_chunk=4)
    for i in range(3):  # the third session evicts the first session's slot
        (sink,) = run(sched, [(sessions[i][0], 4)])
        assert sink.tokens == expected[i][0]
    for i in range(3):  # session 0 comes back with no cached prefix left
        (sink,) = run(sched, [(sessions[i][1], 4)])
        assert sink.tokens == expected[i][1], f"session {i} corrupted after eviction"


# ------------------------------------------- (d) batching against the other optimizations
#
# The batching work was written against the pre-optimization `Model.moe`, `full_attention`
# and `delta_rule`. These check the three merged pairs at a real batch dimension: a slot's
# output inside a batched step must equal that slot's output on its own.


def prefilled(checkpoint: Path, prompts: list[list[int]]) -> seed_model.Model:
    model = build(checkpoint, max_batch=len(prompts))
    for slot, prompt in enumerate(prompts):
        model.begin(slot)
        model.prefill(slot, prompt, 0)
    return model


def routed_experts(cfg: SimpleNamespace, layer: dict, x: torch.Tensor) -> list[frozenset[int]]:
    """The experts each token of a [B, T, hidden] batch routes to, one entry per token."""
    probs = F.linear(x.reshape(-1, cfg.hidden), layer["router"]).softmax(-1, dtype=torch.float)
    return [frozenset(row.tolist()) for row in probs.topk(cfg.top_k, dim=-1)[1]]


def with_chosen_routing(
    layer: dict, x: torch.Tensor, experts: int, top_k: int
) -> list[frozenset[int]]:
    """Make routing deterministic: router = the first `experts` coordinates of the hidden state.

    Token t of slot b is then steered onto experts (3b + t + j) mod `experts`, so slots route
    differently, per-expert token counts are uneven, and every expert group in a batched call
    mixes slots. Returns the intended per-token expert sets.
    """
    router = torch.zeros(experts, x.shape[-1], dtype=x.dtype)
    router[torch.arange(experts), torch.arange(experts)] = 1.0
    layer["router"] = router
    x[..., :experts] = -1.0
    want = []
    for b in range(x.shape[0]):
        for t in range(x.shape[1]):
            picks = [(3 * b + t + j) % experts for j in range(top_k)]
            for rank, e in enumerate(picks):
                x[b, t, e] = 1.0 - 0.1 * rank
            want.append(frozenset(picks))
    return want


@pytest.mark.parametrize("mxfp4", [False, True])
def test_batched_moe_matches_single_sequence(mxfp4: bool) -> None:
    """Vectorized MoE at batch > 1: each slot gets what it would get alone.

    `Model.moe` sorts the whole call's (token, expert) pairs into per-expert groups padded to
    the call's own largest group. At batch > 1 those groups mix tokens from several slots and
    the padding width is set by the busiest expert across the batch, so the batched call is
    not the same tensor program as the single-slot call; it only has to give the same answer.
    Slots are aimed at different experts so the groups really are mixed.
    """
    hidden, inter, experts, top_k, batch, tokens = 64, 32, 8, 3, 4, 3
    dtype = torch.float32
    cfg = SimpleNamespace(hidden=hidden, top_k=top_k)
    layer = build_layer(5, hidden, inter, experts, mxfp4=mxfp4, dtype=dtype)

    gen = torch.Generator().manual_seed(9)
    x = torch.randn(batch, tokens, hidden, generator=gen, dtype=dtype) * 0.2
    want = with_chosen_routing(layer, x, experts, top_k)
    assert routed_experts(cfg, layer, x) == want
    per_slot = [frozenset().union(*want[b * tokens : (b + 1) * tokens]) for b in range(batch)]
    assert len(set(per_slot)) == batch, f"slots must route differently, got {per_slot}"
    assert len(frozenset().union(*per_slot)) > max(len(s) for s in per_slot)

    batched = moe_via_model(cfg, layer, x, dtype)
    assert batched.shape == x.shape
    for b in range(batch):
        solo = moe_via_model(cfg, layer, x[b : b + 1], dtype)
        # Residual is float32 reassociation only: the two calls group the same products into
        # differently shaped bmms. Observed max abs error here is under 1e-5 on outputs of
        # magnitude ~90, for both the dense and the MXFP4 weight layout.
        torch.testing.assert_close(batched[b : b + 1], solo, atol=1e-4, rtol=1e-5, msg=f"slot {b}")


def test_decode_step_drives_the_moe_at_full_batch(checkpoint: Path) -> None:
    """End to end: a full `max_batch` decode step matches solo decode, and the MoE saw batch B."""
    prompts = [prompt_of(90 + i, 4 + 2 * i) for i in range(4)]
    positions = [len(p) for p in prompts]
    slots = list(range(len(prompts)))

    solo_model = prefilled(checkpoint, prompts)
    solo = []
    for slot, prompt in enumerate(prompts):
        solo_model.bind(slot)
        solo.append(solo_model.forward(torch.tensor([[prompt[-1]]]), positions[slot])[-1])

    batched_model = prefilled(checkpoint, prompts)
    seen: list[int] = []
    inner = batched_model.moe

    def spy(i: int, x: torch.Tensor, out_buf: torch.Tensor | None = None) -> torch.Tensor:
        seen.append(x.shape[0])
        return inner(i, x, out_buf)

    batched_model.moe = spy
    batched = batched_model.decode(slots, [p[-1] for p in prompts], positions)

    assert seen and set(seen) == {len(prompts)}, f"moe batch dims seen: {sorted(set(seen))}"
    for slot, row in enumerate(solo):
        torch.testing.assert_close(batched[slot], row, atol=1e-4, rtol=1e-4, msg=f"slot {slot}")


def layer_index(model: seed_model.Model, kind: str) -> int:
    return model.cfg.layer_types.index(kind)


def decode_activations(model: seed_model.Model, batch: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 1, model.cfg.hidden, generator=gen, dtype=torch.float32) * 0.1


def test_attn_decode_matches_single_sequence_attention(checkpoint: Path) -> None:
    """SDPA attention and the cached rope table through the batched decode path.

    Slot positions differ, so `rope_at` gathers non-contiguous rows of the cached table where
    `rope` would slice a contiguous range.
    """
    prompts = [prompt_of(100 + i, 5 + 3 * i) for i in range(4)]
    positions = [len(p) for p in prompts]
    slots = list(range(len(prompts)))
    assert len(set(positions)) == len(positions)

    solo_model, batched_model = prefilled(checkpoint, prompts), prefilled(checkpoint, prompts)
    i = layer_index(solo_model, "full_attention")
    x = decode_activations(solo_model, len(prompts), seed=17)

    solo = []
    for slot in slots:
        solo_model.bind(slot)
        solo.append(solo_model.full_attention(i, x[slot : slot + 1], positions[slot]))
    batched = batched_model.attn_decode(i, x, slots, positions)

    assert batched.shape == x.shape
    for slot in slots:
        torch.testing.assert_close(
            batched[slot : slot + 1], solo[slot], atol=1e-5, rtol=1e-5, msg=f"slot {slot}"
        )
    # Keyed by (device, FAULT_ROPE_BASE), not device alone (see model.py's `_rope_table`
    # docstring); this test never touches the fault flag, so it is always False.
    assert list(batched_model.rope_cache) == [
        (torch.device("cpu"), seed_model.FAULT_ROPE_BASE)
    ]  # one table, built once


def test_deltanet_decode_takes_the_recurrent_path_per_slot(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-slot DeltaNet decode: T=1 calls must dispatch to the recurrent form and match solo."""
    prompts = [prompt_of(110 + i, 4 + i) for i in range(4)]
    slots = list(range(len(prompts)))
    solo_model, batched_model = prefilled(checkpoint, prompts), prefilled(checkpoint, prompts)
    i = layer_index(solo_model, "linear_attention")
    x = decode_activations(solo_model, len(prompts), seed=23)

    solo = []
    for slot in slots:
        solo_model.bind(slot)
        solo.append(solo_model.deltanet(i, x[slot : slot + 1]))

    monkeypatch.setattr(
        seed_model,
        "delta_rule_chunked",
        lambda *a, **k: pytest.fail("decode must not take the chunked path"),
    )
    batched = batched_model.deltanet_decode(i, x, slots)

    assert batched.shape == x.shape
    for slot in slots:
        torch.testing.assert_close(
            batched[slot : slot + 1], solo[slot], atol=1e-5, rtol=1e-5, msg=f"slot {slot}"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

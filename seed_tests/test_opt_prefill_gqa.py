"""Hermetic CPU tests: `Model.full_attention`'s prefill-continuation (`start > 0`) path.

`full_attention`'s `start > 0` branch (resuming after a cached prefix, e.g. a turn-2-plus
request whose shared prefix was matched by the slot pool) built a causal-with-offset
`attn_mask` and passed `enable_gqa=True` alongside it. That combination is the same trap
`decode_attention` documents: on gfx942, an explicit `attn_mask` takes SDPA off its fused
backend, and the fallback broadcasts the KV up to `heads` before the matmul. This branch is
NOT the decode path -- the query here is a real multi-token sequence (`t > 1`), so the
decode fix's fold-the-group-into-the-query-length trick does not apply verbatim; it needs
`(group, t)` merged into one query-length axis instead of just `group`. `prefill_attention`
in model.py does that, and this module is the parity/regression coverage for it.

This branch feeds `p95_ttft_turn2plus_ms`, the primary scored metric.

    /tmp/torchenv/bin/python -m pytest \\
        examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/seed_tests/test_opt_prefill_gqa.py
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import model as seed_model  # noqa: E402
from test_batched_decode import MAX_SEQ, build  # noqa: E402
from test_seed_parity import VOCAB, build_hf, write_checkpoint  # noqa: E402

ATOL, RTOL = 1e-5, 1e-5


def capture_sdpa(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Record the q/keys/vals/attn_mask/enable_gqa of every SDPA call model.py makes.

    Formerly shared with `test_opt_attn_decode_batch.py` (the decode path used to call SDPA
    too); the paged-KV redesign moved decode off SDPA entirely (see `paged_attn.py`), so this
    helper now lives only here, next to its one remaining use: pinning that `full_attention`'s
    `start > 0` continuation, and `prefill_attention` underneath it, never ask SDPA for GQA.
    """
    calls: list[dict] = []
    real = F.scaled_dot_product_attention

    def spy(q, keys, vals, **kwargs):  # noqa: ANN001, ANN202
        calls.append({"q": q, "keys": keys, "vals": vals, **kwargs})
        return real(q, keys, vals, **kwargs)

    monkeypatch.setattr(seed_model.F, "scaled_dot_product_attention", spy)
    return calls


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("tiny")
    write_checkpoint(build_hf(), out, mxfp4=False)
    return out


# ---------------------------------------------------------------- prefill_attention, standalone


def causal_offset_mask(t: int, start: int) -> torch.Tensor:
    """The same mask `full_attention`'s `start > 0` branch builds: `j <= start + i`."""
    return torch.arange(start + t)[None, :] <= (start + torch.arange(t))[:, None]


# (label, heads, kv_heads, head_dim, batch, t, start)
SHAPE_CASES = [
    ("served shape, one new token", 32, 2, 256, 1, 1, 1),
    ("served shape, chunk boundary", 32, 2, 256, 1, 7, 0),
    ("served shape, short continuation", 32, 2, 256, 1, 7, 37),
    ("served shape, a full PREFILL_CHUNK continuation", 32, 2, 256, 1, 512, 4096),
    ("served shape, long resumed prefix", 32, 2, 256, 1, 3, 30000),
    ("tiny-fixture shape", 4, 2, 16, 1, 5, 11),
    ("group of 1 (MHA, no GQA at all)", 4, 4, 16, 1, 5, 11),
    ("batch > 1", 2, 2, 8, 3, 4, 6),
    ("group does not divide evenly into a small t", 8, 2, 16, 1, 3, 9),
]


@pytest.mark.parametrize(
    ("seed", "label", "heads", "kv_heads", "head_dim", "batch", "t", "start"),
    [(i, *case) for i, case in enumerate(SHAPE_CASES)],
)
def test_prefill_attention_matches_the_enable_gqa_reference(
    seed: int, label: str, heads: int, kv_heads: int, head_dim: int, batch: int, t: int, start: int
) -> None:
    """Bit-exact (fp32) and bf16-tolerance parity against the `enable_gqa=True` reference."""
    gen = torch.Generator().manual_seed(1000 + seed)
    q = torch.randn(batch, heads, t, head_dim, generator=gen, dtype=torch.float32)
    keys = torch.randn(batch, kv_heads, start + t, head_dim, generator=gen, dtype=torch.float32)
    vals = torch.randn(batch, kv_heads, start + t, head_dim, generator=gen, dtype=torch.float32)
    mask = causal_offset_mask(t, start)
    scale = head_dim**-0.5

    want = F.scaled_dot_product_attention(
        q, keys, vals, attn_mask=mask, scale=scale, enable_gqa=(heads != kv_heads)
    )
    got = seed_model.prefill_attention(q, keys, vals, mask, scale)

    assert got.shape == want.shape, label
    torch.testing.assert_close(got, want, atol=ATOL, rtol=RTOL, msg=label)
    assert torch.isfinite(got).all(), label

    q16, keys16, vals16 = q.bfloat16(), keys.bfloat16(), vals.bfloat16()
    want16 = F.scaled_dot_product_attention(
        q16, keys16, vals16, attn_mask=mask, scale=scale, enable_gqa=(heads != kv_heads)
    )
    got16 = seed_model.prefill_attention(q16, keys16, vals16, mask, scale)
    torch.testing.assert_close(got16, want16, atol=2e-2, rtol=2e-2, msg=f"{label} (bf16)")


def test_prefill_attention_never_asks_sdpa_for_gqa(monkeypatch: pytest.MonkeyPatch) -> None:
    """The spelling the decode path had to stop using, generalized: no `enable_gqa`, ever."""
    heads, kv_heads, head_dim = 32, 2, 256
    gen = torch.Generator().manual_seed(19)
    q = torch.randn(1, heads, 9, head_dim, generator=gen, dtype=torch.float32)
    keys = torch.randn(1, kv_heads, 40, head_dim, generator=gen, dtype=torch.float32)
    vals = torch.randn(1, kv_heads, 40, head_dim, generator=gen, dtype=torch.float32)
    mask = causal_offset_mask(9, 31)

    calls = capture_sdpa(monkeypatch)
    seed_model.prefill_attention(q, keys, vals, mask, head_dim**-0.5)

    assert len(calls) == 1
    assert not calls[0].get("enable_gqa", False)
    assert calls[0]["keys"].shape[1] == kv_heads, "the kv heads were broadcast to `heads`"
    assert calls[0]["q"].shape[1] == kv_heads


# ---------------------------------------------------------------- through the real model


def test_full_attention_never_asks_sdpa_for_gqa_when_resuming_a_cached_prefix(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins the spelling through the real `start > 0` code path, not just the helper."""
    model = build(checkpoint, max_batch=1)
    c = model.cfg
    assert c.heads > c.kv_heads, "the fixture model has to be a GQA model for this to mean anything"
    prefix = torch.randint(2, VOCAB, (13,), generator=torch.Generator().manual_seed(5)).tolist()
    tail = torch.randint(2, VOCAB, (6,), generator=torch.Generator().manual_seed(6)).tolist()

    model.begin(0)
    model.prefill(0, prefix, 0)
    calls = capture_sdpa(monkeypatch)
    model.prefill(0, tail, len(prefix))

    full_attention_calls = [call for call in calls if call["keys"].shape[1] == c.kv_heads]
    assert full_attention_calls, "expected at least one full-attention SDPA call"
    for call in full_attention_calls:
        assert not call.get("enable_gqa", False)
        assert call["attn_mask"] is not None, "start > 0 must still mask to the causal prefix"


# (label, prefix_len, tail_len)
CONTINUATION_CASES = [
    ("one new token, like a single decode step routed through prefill", 5, 1),
    ("short turn-2 tail", 9, 4),
    ("tail longer than the prefix", 3, 11),
    ("prefix and tail both small", 1, 1),
    ("prefix near the pool's max_seq", MAX_SEQ - 8, 6),
]


@pytest.mark.parametrize(("label", "prefix_len", "tail_len"), CONTINUATION_CASES)
def test_prefill_continuation_matches_one_shot_forward_over_the_full_prompt(
    checkpoint: Path, label: str, prefix_len: int, tail_len: int
) -> None:
    """Correctness, not just the SDPA spelling: resuming must equal processing it all at once."""
    gen = torch.Generator().manual_seed(100 + prefix_len + tail_len)
    prompt = torch.randint(2, VOCAB, (prefix_len + tail_len,), generator=gen).tolist()

    one_shot = build(checkpoint, max_batch=1)
    one_shot.begin(0)
    want = one_shot.forward(torch.tensor([prompt]), 0, all_logits=True)[prefix_len:]

    resumed = build(checkpoint, max_batch=1)
    resumed.begin(0)
    resumed.prefill(0, prompt[:prefix_len], 0)
    got = resumed.forward(torch.tensor([prompt[prefix_len:]]), prefix_len, all_logits=True)

    assert got.shape == want.shape, label
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4, msg=label)


def test_prefill_continuation_matches_one_shot_forward_across_several_slots(
    checkpoint: Path,
) -> None:
    """Same property, batched across slots: one slot's continuation must not leak into another's."""
    specs = [(4, 3), (9, 1), (2, 7)]  # (prefix_len, tail_len) per slot
    prompts = [
        torch.randint(2, VOCAB, (p + n,), generator=torch.Generator().manual_seed(200 + j)).tolist()
        for j, (p, n) in enumerate(specs)
    ]

    want = []
    for prompt, (prefix_len, _) in zip(prompts, specs, strict=True):
        solo = build(checkpoint, max_batch=1)
        solo.begin(0)
        want.append(solo.forward(torch.tensor([prompt]), 0, all_logits=True)[prefix_len:])

    model = build(checkpoint, max_batch=len(specs))
    got = []
    for slot, (prompt, (prefix_len, _)) in enumerate(zip(prompts, specs, strict=True)):
        model.begin(slot)
        model.prefill(slot, prompt[:prefix_len], 0)
        got.append(model.forward(torch.tensor([prompt[prefix_len:]]), prefix_len, all_logits=True))

    for slot, (g, w) in enumerate(zip(got, want, strict=True)):
        torch.testing.assert_close(g, w, atol=1e-4, rtol=1e-4, msg=f"slot {slot}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

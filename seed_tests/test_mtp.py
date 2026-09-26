"""Hermetic CPU tests for MTP speculative decoding (mtp.py), against the real model code.

The tiny random Qwen3.5-MoE checkpoint `test_seed_parity.build_hf`/`write_checkpoint` produce
has no `mtp.*` weights: the installed reference `transformers` modeling code does not
implement the MTP module at all (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` in
reference/modeling_qwen3_5_moe.py), so there is no HF model to source them from.
`write_mtp_shard` below adds them as one extra safetensors shard, at the shapes the real
checkpoint has (a full-attention self-attn identical to one of the target model's own, an MoE
identical to the target model's, an `fc` combining a normalized embedding and a normalized
hidden state -- see mtp.py's own docstring and `scratchpad/mtp-design.md`), scaled down to the
tiny config.

Run with:
    /tmp/torchenv/bin/python -m pytest \\
        examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/seed_tests/test_mtp.py
"""

import math
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import model as seed_model  # noqa: E402
import mtp  # noqa: E402
import mtp_dense_moe  # noqa: E402
from test_seed_parity import VOCAB, build_hf, tiny_config, write_checkpoint  # noqa: E402

MAX_SEQ = 64
ATOL, RTOL = 1e-5, 1e-5


def write_mtp_shard(cfg, out: Path, seed: int = 7) -> None:
    """Synthetic `mtp.*` tensors at `cfg`'s shapes, as one more safetensors shard next to
    `write_checkpoint`'s: `weights.Checkpoint` globs every `*.safetensors` file in a directory,
    so this needs no cooperation from `write_checkpoint` itself."""
    g = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g) * 0.1

    hidden, heads, kv_heads, head_dim = (
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
    )
    experts, inter, shared_inter = (
        cfg.num_experts,
        cfg.moe_intermediate_size,
        cfg.shared_expert_intermediate_size,
    )
    tensors = {
        "mtp.fc.weight": randn(hidden, 2 * hidden),
        "mtp.pre_fc_norm_embedding.weight": randn(hidden),
        "mtp.pre_fc_norm_hidden.weight": randn(hidden),
        "mtp.norm.weight": randn(hidden),
        "mtp.layers.0.input_layernorm.weight": randn(hidden),
        "mtp.layers.0.post_attention_layernorm.weight": randn(hidden),
        "mtp.layers.0.self_attn.q_proj.weight": randn(heads * 2 * head_dim, hidden),
        "mtp.layers.0.self_attn.k_proj.weight": randn(kv_heads * head_dim, hidden),
        "mtp.layers.0.self_attn.v_proj.weight": randn(kv_heads * head_dim, hidden),
        "mtp.layers.0.self_attn.o_proj.weight": randn(hidden, heads * head_dim),
        "mtp.layers.0.self_attn.q_norm.weight": randn(head_dim),
        "mtp.layers.0.self_attn.k_norm.weight": randn(head_dim),
        "mtp.layers.0.mlp.gate.weight": randn(experts, hidden),
        "mtp.layers.0.mlp.shared_expert_gate.weight": randn(1, hidden),
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight": randn(shared_inter, hidden),
        "mtp.layers.0.mlp.shared_expert.up_proj.weight": randn(shared_inter, hidden),
        "mtp.layers.0.mlp.shared_expert.down_proj.weight": randn(hidden, shared_inter),
    }
    for e in range(experts):
        tensors[f"mtp.layers.0.mlp.experts.{e}.gate_proj.weight"] = randn(inter, hidden)
        tensors[f"mtp.layers.0.mlp.experts.{e}.up_proj.weight"] = randn(inter, hidden)
        tensors[f"mtp.layers.0.mlp.experts.{e}.down_proj.weight"] = randn(hidden, inter)
    save_file(tensors, str(out / "model-mtp.safetensors"))


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("tiny-mtp")
    cfg = tiny_config()
    write_checkpoint(build_hf(cfg=cfg), out, mxfp4=False)
    write_mtp_shard(cfg, out)
    return out


def build(
    checkpoint: Path, max_batch: int, *, mtp_env: bool = True, k: int = 3
) -> seed_model.Model:
    """A real `Model` over `checkpoint`, with `SEED_MTP` forced on for the duration of the
    build (`mtp.load` reads the flag once, at `Model.__init__` time)."""
    old_enabled, old_k = mtp.MTP_ENABLED, mtp.MTP_K
    mtp.MTP_ENABLED, mtp.MTP_K = mtp_env, k
    try:
        m = seed_model.Model(checkpoint, ["cpu"], torch.float32, MAX_SEQ, max_batch)
    finally:
        mtp.MTP_ENABLED, mtp.MTP_K = old_enabled, old_k
    return m


def prompt_of(seed: int, length: int) -> list[int]:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(2, VOCAB, (length,), generator=gen).tolist()


def sequential_reference(
    model: seed_model.Model, slot: int, start_tok: int, pos: int, n: int
) -> list[int]:
    """`n` ordinary greedy decode steps on `slot`, starting from `start_tok` at `pos`. The
    ground truth `verify_and_commit`'s unrolled steps have to reproduce exactly."""
    tok, out = start_tok, []
    for i in range(n):
        logits = model.decode([slot], [tok], [pos + i])
        tok = model.sample_batch(logits, [0.0])[0]
        out.append(tok)
    return out


def assert_slot_state_matches(
    got: seed_model.Model, want: seed_model.Model, slot: int, consumed: int
) -> None:
    """Compare every persistent input to the next decode step at one logical lane."""
    state_atol, state_rtol = 5e-5, 2e-4
    for i, layer_type in enumerate(got.cfg.layer_types):
        if layer_type == "full_attention":
            got_rows = got.block_tables[slot].physical_rows(consumed, got.block_size)
            want_rows = want.block_tables[slot].physical_rows(consumed, want.block_size)
            torch.testing.assert_close(
                got.pool[i]["k"][got_rows], want.pool[i]["k"][want_rows],
                atol=state_atol, rtol=state_rtol,
            )
            torch.testing.assert_close(
                got.pool[i]["v"][got_rows], want.pool[i]["v"][want_rows],
                atol=state_atol, rtol=state_rtol,
            )
        else:
            torch.testing.assert_close(
                got.pool[i]["conv"][slot], want.pool[i]["conv"][slot],
                atol=state_atol, rtol=state_rtol,
            )
            torch.testing.assert_close(
                got.pool[i]["rec"][slot], want.pool[i]["rec"][slot],
                atol=state_atol, rtol=state_rtol,
            )
    torch.testing.assert_close(
        got.hidden_scratch[slot], want.hidden_scratch[slot], atol=state_atol, rtol=state_rtol
    )
    got_rows = got.block_tables[slot].physical_rows(consumed, got.block_size)
    want_rows = want.block_tables[slot].physical_rows(consumed, want.block_size)
    for name in ("k", "v"):
        torch.testing.assert_close(
            got.mtp.pool[name][got_rows], want.mtp.pool[name][want_rows],
            atol=state_atol, rtol=state_rtol,
        )


def test_prompt_builds_persistent_mtp_history_with_base_position(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = build(checkpoint, max_batch=1, k=3)
    model.begin(0)
    prompt = prompt_of(90, 6)
    reference = build(checkpoint, max_batch=1, k=3)
    reference.begin(0)
    expected_previous = []
    for position, token in enumerate(prompt):
        expected_previous.append(reference.hidden_scratch[0].clone())
        reference.prefill(0, [token], position)
    seen: list[torch.Tensor] = []
    seen_previous: list[torch.Tensor] = []
    original = mtp.cache_target_rows

    def record(model_, mtp_, tokens, previous, positions, rows, keep=None):  # noqa: ANN001, ANN202
        seen.append(positions.clone())
        seen_previous.append(previous.clone())
        return original(model_, mtp_, tokens, previous, positions, rows, keep)

    monkeypatch.setattr(mtp, "cache_target_rows", record)
    logits = model.prefill(0, prompt, 0)
    assert seen and seen[0].tolist() == list(range(len(prompt)))
    torch.testing.assert_close(
        seen_previous[0], torch.stack(expected_previous), atol=5e-5, rtol=2e-4
    )
    rows = model._physical_rows_range(0, 0, len(prompt), model.devices[-1])
    assert torch.count_nonzero(model.mtp.pool["k"][rows]) > 0
    assert torch.count_nonzero(model.mtp.pool["v"][rows]) > 0

    rope_positions: list[torch.Tensor] = []
    original_rope = model.rope_at

    def record_rope(positions, dtype):  # noqa: ANN001, ANN202
        rope_positions.append(positions.clone())
        return original_rope(positions, dtype)

    monkeypatch.setattr(model, "rope_at", record_rope)
    base = int(logits.argmax(-1))
    mtp.draft(model, model.mtp, model.cached_hidden(0), [base], [0], [len(prompt)])
    assert [int(pos[0]) for pos in rope_positions] == [len(prompt) + i for i in range(3)]


def test_persistent_history_changes_first_draft_state(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt = prompt_of(91, 7)
    models = [build(checkpoint, max_batch=1, k=3) for _ in range(2)]
    first_hiddens = []
    first_logits = []
    attention = []
    original_attention = mtp._m.verify_attention_paged

    def record_attention(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        out = original_attention(*args, **kwargs)
        attention.append(out.clone())
        return out

    monkeypatch.setattr(mtp._m, "verify_attention_paged", record_attention)
    for model in models:
        model.begin(0)
        logits = model.prefill(0, prompt, 0)
        base = int(logits.argmax(-1))
        model.grow_lane(0, len(prompt) + 1)
        pos = torch.tensor([len(prompt)])
        rows = model._physical_rows_range(0, len(prompt), len(prompt) + 1, model.devices[-1])
        table, valid = model._paged_read_buffers(model.devices[-1], [0], [len(prompt)])
        prefix = model._physical_rows_range(0, 0, len(prompt), model.devices[-1])
        model.mtp.pool["k"][prefix] = 0
        if model is models[0]:
            model.mtp.pool["v"][prefix] = 10
        else:
            model.mtp.pool["v"][prefix] = 0
        hidden, _ = mtp.draft_step(
            model, model.mtp, mtp.draft_seed(model, model.cached_hidden(0)),
            torch.tensor([[base]]), pos, rows, table, valid,
        )
        first_hiddens.append(hidden)
        first_logits.append(model.unembed(hidden[:, None])[:, 0].float())
    assert not torch.equal(attention[0], attention[1]), (attention[0] - attention[1]).abs().max()
    assert not torch.equal(first_logits[0], first_logits[1])


def test_snapshot_and_cow_preserve_mtp_state(checkpoint: Path) -> None:
    model = build(checkpoint, max_batch=1, k=3)
    model.begin(0)
    model.prefill(0, prompt_of(92, 6), 0)
    saved_hidden = model.hidden_scratch[0].clone()
    model.save_snapshot(0, 0)
    model.hidden_scratch[0].fill_(123)
    model.load_snapshot(0, 0)
    torch.testing.assert_close(model.hidden_scratch[0], saved_hidden)

    src, dst, filled = 1, 2, 5
    src_rows = slice(src * model.block_size, src * model.block_size + filled)
    dst_rows = slice(dst * model.block_size, dst * model.block_size + filled)
    model.mtp.pool["k"][src_rows].normal_()
    model.mtp.pool["v"][src_rows].normal_()
    model.mtp.pool["k"][dst_rows].zero_()
    model.mtp.pool["v"][dst_rows].zero_()
    model.copy_block(dst, src, filled)
    torch.testing.assert_close(model.mtp.pool["k"][dst_rows], model.mtp.pool["k"][src_rows])
    torch.testing.assert_close(model.mtp.pool["v"][dst_rows], model.mtp.pool["v"][src_rows])


# ---------------------------------------------------------------- (a) greedy_accept_length


def test_greedy_accept_length_full_match() -> None:
    assert mtp.greedy_accept_length([5, 6, 7], [5, 6, 7]) == 3


def test_greedy_accept_length_no_match() -> None:
    assert mtp.greedy_accept_length([5, 6, 7], [9, 6, 7]) == 0


def test_greedy_accept_length_partial_match() -> None:
    assert mtp.greedy_accept_length([5, 6, 7], [5, 6, 9]) == 2


# ---------------------------------------------------------------- (b) verify + rollback, forced acceptance


@pytest.mark.parametrize("accept", [0, 1, 2, 3])
def test_verify_and_commit_matches_sequential_decode(checkpoint: Path, accept: int) -> None:
    """Force acceptance to exactly `accept` (0..k) by feeding `verify_and_commit` the real
    greedy continuation for the first `accept` draft slots and a token that cannot be the
    target's argmax for the rest, then check both the committed tokens and every side effect
    (DeltaNet/conv state, KV, the cached hidden `mtp.draft` would seed off) against plain
    sequential `Model.decode`, by running one more ordinary decode step on each and comparing
    logits: any state divergence -- attention KV, DeltaNet recurrence, or the hidden-state
    seed -- would show up there.
    """
    k = 3
    prompt = prompt_of(1, 6)
    oracle = build(checkpoint, max_batch=1, k=k)
    oracle.begin(0)
    oracle_logits = oracle.prefill(0, prompt, 0)
    base = oracle.sample_batch(oracle_logits, [0.0])[0]
    ground_truth = sequential_reference(oracle, 0, base, len(prompt), k + 1)

    seq_model = build(checkpoint, max_batch=1, k=k)
    seq_model.begin(0)
    seq_model.prefill(0, prompt, 0)
    committed_seq = sequential_reference(seq_model, 0, base, len(prompt), accept + 1)
    assert committed_seq == ground_truth[: accept + 1]

    spec_model = build(checkpoint, max_batch=1, k=k)
    spec_model.begin(0)
    spec_model.prefill(0, prompt, 0)
    speculative_rows = spec_model._physical_rows_range(
        0, len(prompt), len(prompt) + k + 1, spec_model.devices[-1]
    )
    rejected_before = {
        name: spec_model.mtp.pool[name][speculative_rows].clone() for name in ("k", "v")
    }
    draft_tokens = [_forced_draft(ground_truth, accept, k)]
    committed = mtp.verify_and_commit(
        spec_model, spec_model.mtp, [0], [base], draft_tokens, [len(prompt)]
    )
    assert committed == [ground_truth[: accept + 1]]

    # Check the state directly before another forward can overwrite a stale rejected row.
    next_pos = len(prompt) + accept + 1
    assert_slot_state_matches(spec_model, seq_model, 0, next_pos)
    for name in ("k", "v"):
        torch.testing.assert_close(
            spec_model.mtp.pool[name][speculative_rows[accept + 1 :]],
            rejected_before[name][accept + 1 :],
        )

    # Rejected draft K/V may remain in physical storage, but the next draft overwrites its
    # base row before reading and cannot observe later rejected positions.
    seq_draft = mtp.draft(
        seq_model, seq_model.mtp, seq_model.cached_hidden(0),
        [committed_seq[-1]], [0], [next_pos],
    )
    spec_draft = mtp.draft(
        spec_model, spec_model.mtp, spec_model.cached_hidden(0),
        [committed[0][-1]], [0], [next_pos],
    )
    assert spec_draft == seq_draft

    # One more ordinary step also checks that the state produces the same logits.
    seq_logits = seq_model.decode([0], [committed_seq[-1]], [next_pos])
    spec_logits = spec_model.decode([0], [committed[0][-1]], [next_pos])
    torch.testing.assert_close(spec_logits, seq_logits, atol=ATOL, rtol=RTOL)


def _forced_draft(ground_truth: list[int], accept: int, k: int) -> list[int]:
    """`k` draft tokens matching `ground_truth` for the first `accept` of them, then a token
    that cannot be the target's argmax (so acceptance stops at exactly `accept`), then padding
    that is never read (acceptance already stopped)."""
    if accept >= k:
        return ground_truth[:k]
    wrong = (ground_truth[accept] + 1) % VOCAB
    return ground_truth[:accept] + [wrong] + [0] * (k - accept - 1)


def test_verify_and_commit_batched_matches_solo(checkpoint: Path) -> None:
    """Two slots verified together must not leak state into each other: run both slots
    through one batched `verify_and_commit` call, forcing a different accept length for each,
    and check each slot's result against that same slot verified alone."""
    k = 3
    prompts = [prompt_of(10, 5), prompt_of(11, 7)]
    accepts = [1, 3]
    bases, ground_truths = [], []
    for prompt in prompts:
        m = build(checkpoint, max_batch=1, k=k)
        m.begin(0)
        m_logits = m.prefill(0, prompt, 0)
        base = m.sample_batch(m_logits, [0.0])[0]
        bases.append(base)
        ground_truths.append(sequential_reference(m, 0, base, len(prompt), k + 1))
    drafts = [_forced_draft(gt, a, k) for gt, a in zip(ground_truths, accepts, strict=True)]
    positions = [len(p) for p in prompts]

    solo_committed, solo_next_logits = [], []
    for j in range(2):
        m = build(checkpoint, max_batch=1, k=k)
        m.begin(0)
        m.prefill(0, prompts[j], 0)
        committed = mtp.verify_and_commit(m, m.mtp, [0], [bases[j]], [drafts[j]], [positions[j]])
        solo_committed.append(committed[0])
        solo_next_logits.append(m.decode([0], [committed[0][-1]], [positions[j] + accepts[j] + 1]))

    batch = build(checkpoint, max_batch=2, k=k)
    for slot, prompt in enumerate(prompts):
        batch.begin(slot)
        batch.prefill(slot, prompt, 0)
    committed_batch = mtp.verify_and_commit(batch, batch.mtp, [0, 1], bases, drafts, positions)
    assert committed_batch == solo_committed
    for slot in range(2):
        got = batch.decode(
            [slot], [committed_batch[slot][-1]], [positions[slot] + accepts[slot] + 1]
        )
        torch.testing.assert_close(got, solo_next_logits[slot], atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------- (c) end-to-end equivalence


def test_mtp_moe_uses_capture_safe_fused_routing(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = build(checkpoint, max_batch=2, k=3)
    x = torch.linspace(-1, 1, 2 * model.cfg.hidden).reshape(2, 1, model.cfg.hidden)
    monkeypatch.setattr(mtp._m, "FUSE_GLUE", False)
    want = mtp._moe(model, model.mtp, x)

    called = []

    def route(logits, experts_out, top_k):  # noqa: ANN001, ANN202
        called.append(True)
        probs = logits[:, :experts_out].softmax(-1, dtype=torch.float)
        weights, indices = probs.topk(top_k, dim=-1)
        return weights / weights.sum(-1, keepdim=True), indices

    monkeypatch.setattr(mtp._m, "FUSE_GLUE", True)
    monkeypatch.setattr(mtp.router_fused, "available", lambda device: True)
    monkeypatch.setattr(mtp.router_fused, "route", route)
    got = mtp._moe(model, model.mtp, x)
    assert called
    torch.testing.assert_close(got, want)


def test_dense_mtp_expert_scratch_is_preallocated(checkpoint: Path) -> None:
    model = build(checkpoint, max_batch=4, k=3)
    module = model.mtp
    intermediate = module.weights["experts"]["gate_up"].shape[1] // 2
    assert module.embedding_scratch.shape == (4 * (module.k + 1), model.cfg.hidden)
    assert module.expert_inter_scratch.shape == (4 * model.cfg.top_k, intermediate)
    assert module.expert_out_scratch.shape == (4, model.cfg.hidden)
    assert mtp_dense_moe.scratch_bytes(4, model.cfg.top_k, intermediate, model.cfg.hidden) == (
        module.expert_inter_scratch.numel() + module.expert_out_scratch.numel()
    ) * 2


def test_capture_safe_row_helpers_cover_verify_tensor_ranks() -> None:
    rows = torch.tensor([3, 1, 3])
    for trailing in ((2, 5), (2, 3, 4)):
        src = torch.arange(6 * math.prod(trailing)).reshape(6, *trailing)
        got = mtp_dense_moe.gather_first_dim(src, rows)
        assert torch.equal(got, src[rows])

        dst = torch.full_like(src, -1)
        active = torch.tensor([True, False, True])
        mtp_dense_moe.scatter_active_first_dim(dst, rows, got, active)
        assert torch.equal(dst[1], torch.full_like(dst[1], -1))
        assert torch.equal(dst[3], got[2])


def test_capture_safe_row_scatter_repeats_lane_activity() -> None:
    dst = torch.full((8, 2, 3), -1.0)
    rows = torch.tensor([1, 2, 5, 6])
    src = torch.arange(24.0).reshape(4, 2, 3)
    mtp_dense_moe.scatter_active_first_dim(
        dst, rows, src, torch.tensor([True, False]), active_repeat=2
    )
    assert torch.equal(dst[1], src[0])
    assert torch.equal(dst[2], src[1])
    assert torch.equal(dst[5], torch.full_like(dst[5], -1))
    assert torch.equal(dst[6], torch.full_like(dst[6], -1))


def test_step_moe_stages_contiguous_time_rows() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.rows = []

        def moe(self, _layer: int, row: torch.Tensor) -> torch.Tensor:
            assert row.is_contiguous()
            self.rows.append(row.clone())
            return row + 1

    model = FakeModel()
    h = torch.arange(2 * 3 * 5).reshape(2, 3, 5)
    got = seed_model.mtp_step_moe(model, 7, h)

    assert torch.equal(got, h + 1)
    assert len(model.rows) == 3
    assert all(torch.equal(row, h[:, step]) for step, row in enumerate(model.rows))


def test_speculative_decode_matches_sequential_for_several_rounds(checkpoint: Path) -> None:
    """The correctness requirement itself: under greedy decoding, several rounds of
    `Model.speculative_decode` must commit exactly what the same number of ordinary
    `Model.decode` steps would have produced, whatever `mtp.draft` actually predicts (its
    prediction quality only changes how many rounds this loop takes, i.e. the acceptance
    rate, never the tokens committed)."""
    k = 3
    prompt = prompt_of(2, 6)

    ref = build(checkpoint, max_batch=1, k=k)
    ref.begin(0)
    ref_logits = ref.prefill(0, prompt, 0)
    base = ref.sample_batch(ref_logits, [0.0])[0]
    ground_truth = sequential_reference(ref, 0, base, len(prompt), 3 * (k + 1))

    spec = build(checkpoint, max_batch=1, k=k)
    spec.begin(0)
    spec.prefill(0, prompt, 0)
    tok, pos, got = base, len(prompt), []
    while len(got) < len(ground_truth) - k:  # stop before the tail a partial last round would need
        committed = spec.speculative_decode([0], [tok], [pos])[0]
        assert 1 <= len(committed) <= k + 1
        got.extend(committed)
        pos += len(committed)
        tok = committed[-1]
    assert got == ground_truth[: len(got)]


def test_speculative_decode_returns_valid_batch_shape(checkpoint: Path) -> None:
    """Shape/plumbing sanity at a real (>1) batch: every slot gets 1..k+1 committed tokens."""
    k = 3
    model = build(checkpoint, max_batch=3, k=k)
    bases = []
    for slot, prompt in enumerate([prompt_of(20 + s, 4 + s) for s in range(3)]):
        model.begin(slot)
        slot_logits = model.prefill(slot, prompt, 0)
        bases.append((model.sample_batch(slot_logits, [0.0])[0], len(prompt)))
    committed = model.speculative_decode([0, 1, 2], [b for b, _ in bases], [p for _, p in bases])
    assert len(committed) == 3
    for tokens in committed:
        assert 1 <= len(tokens) <= k + 1
        assert all(isinstance(t, int) for t in tokens)


def test_speculative_decode_grows_rows_before_draft_at_block_boundary(checkpoint: Path) -> None:
    model = build(checkpoint, max_batch=1, k=3)
    prompt = prompt_of(93, model.block_size)
    model.begin(0)
    base = int(model.prefill(0, prompt, 0).argmax(-1))
    committed = model.speculative_decode([0], [base], [len(prompt)])
    assert committed and committed[0]
    assert model.block_tables[0].token_capacity(model.block_size) >= len(prompt) + model.mtp.k + 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

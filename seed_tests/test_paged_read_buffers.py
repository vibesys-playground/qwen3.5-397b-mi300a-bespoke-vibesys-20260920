"""Eager decode builds the paged-attention block table once per step, not once per layer.

`Model._paged_read_buffers` used to rebuild the `[B, max_blocks]` table (a Python list of
`B * max_blocks` ints plus a synchronizing host-to-device copy) inside every full-attention
layer, 15 times per step on the real model. These tests use a tiny checkpoint with two
full-attention layers so a per-layer rebuild is observable, and check that the shared table
stays correct when a lane crosses a block boundary or a slot is reset between steps.
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import block_pool  # noqa: E402
import model as seed_model  # noqa: E402
from test_seed_parity import VOCAB, build_hf, tiny_config, write_checkpoint  # noqa: E402

MAX_SEQ = 64
TWO_FULL = ["full_attention", "linear_attention", "full_attention", "linear_attention"]


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("tiny2full")
    write_checkpoint(build_hf(cfg=tiny_config(layer_types=TWO_FULL)), out, mxfp4=False)
    return out


def build(checkpoint: Path, max_batch: int) -> seed_model.Model:
    return seed_model.Model(checkpoint, ["cpu"], torch.float32, MAX_SEQ, max_batch)


def prompt_of(seed: int, length: int) -> list[int]:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(2, VOCAB, (length,), generator=gen).tolist()


def test_table_is_built_once_per_step(checkpoint: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = build(checkpoint, max_batch=3)
    assert model.cfg.layer_types.count("full_attention") == 2
    prompts = [prompt_of(1, 7), prompt_of(2, 5), prompt_of(3, 9)]
    for slot, prompt in enumerate(prompts):
        model.begin(slot)
        model.prefill(slot, prompt, 0)

    calls = 0
    real = block_pool.BlockTable.padded_row

    def counting(self: block_pool.BlockTable, max_blocks: int) -> list[int]:
        nonlocal calls
        calls += 1
        return real(self, max_blocks)

    monkeypatch.setattr(block_pool.BlockTable, "padded_row", counting)
    positions = [len(p) for p in prompts]
    for step in range(3):
        calls = 0
        model.decode([0, 1, 2], [5, 6, 7], [p + step for p in positions])
        assert calls == 3  # one row per lane, shared by both full-attention layers


def test_shared_table_matches_serial_generation_across_block_boundary(checkpoint: Path) -> None:
    """Batched greedy decode that crosses `KV_BLOCK_SIZE` (a fresh block mid-decode) and
    reuses a slot after `begin` still equals one-sequence-at-a-time generation."""
    block = seed_model.KV_BLOCK_SIZE
    prompts = [prompt_of(4, block - 3), prompt_of(5, block - 1), prompt_of(6, 4)]
    steps = 6

    want = []
    ref = build(checkpoint, max_batch=1)
    for prompt in prompts:
        # No explicit `ref.begin(0)` here: `generate` already does `_release_lane_blocks(0)`
        # then `begin(0)` itself (see its docstring). Calling `begin` again first would clear
        # lane 0's block-table list without decref'ing it -- a leak stage 2's much tighter
        # pool sizing (blocks sized to need, not a generous fixed constant) no longer
        # tolerates across three prompts sharing one lane.
        want.append(list(ref.generate(prompt, steps, 0.0, frozenset())))

    model = build(checkpoint, max_batch=3)
    for _ in range(2):  # second round reuses every slot after `begin`, at the same positions
        next_tok = []
        for slot, prompt in enumerate(prompts):
            model.begin(slot)
            next_tok.append(int(model.prefill(slot, prompt, 0).argmax(-1)))
        got = [[t] for t in next_tok]
        pos = [len(p) for p in prompts]
        for _ in range(steps - 1):
            logits = model.decode([0, 1, 2], next_tok, pos)
            next_tok = logits.argmax(-1).tolist()
            pos = [p + 1 for p in pos]
            for seq, tok in zip(got, next_tok, strict=True):
                seq.append(tok)
        assert got == want


def test_memo_invalidated_when_a_lane_is_reassigned_via_attach_blocks(checkpoint: Path) -> None:
    """Stage 2's session-cache resume (`load_snapshot` + `attach_blocks`, `scheduler._acquire`'s
    sequence for a cache hit) reassigns a lane's block table with no `begin`/`reset` in between
    -- unlike every other case above, which always goes through `begin`. If the once-per-step
    memo this module tests stayed keyed on a stale `(epoch, ...)` across that reassignment, lane
    0 below would keep decoding off request A's KV blocks after being handed to B's session, and
    diverge from B generated straight through. `decode` bumps `_decode_epoch` unconditionally on
    every call (see `Model.__init__`'s docstring), so this passes today; it pins that contract
    against a future change that only bumps the epoch on `reset`.
    """
    block = seed_model.KV_BLOCK_SIZE
    prompt_a = prompt_of(20, block - 2)
    prompt_b = prompt_of(21, block + 3)  # crosses a block boundary on its own
    steps_before, steps_after = 4, 3

    ref = build(checkpoint, max_batch=1)
    ref.begin(0)
    ref_tokens = list(ref.generate(prompt_b, steps_before + steps_after, 0.0, frozenset()))

    model = build(checkpoint, max_batch=2)
    model.begin(0)
    model.begin(1)
    tok_a = int(model.prefill(0, prompt_a, 0).argmax(-1))
    tok_b = int(model.prefill(1, prompt_b, 0).argmax(-1))
    pos_a, pos_b = len(prompt_a), len(prompt_b)
    got_b = [tok_b]
    for _ in range(steps_before - 1):
        logits = model.decode([0, 1], [tok_a, tok_b], [pos_a, pos_b])
        tok_a, tok_b = logits.argmax(-1).tolist()
        got_b.append(tok_b)
        pos_a, pos_b = pos_a + 1, pos_b + 1
    assert got_b == ref_tokens[:steps_before]
    # This batched call over [0, 1] memoized _paged_read_buffers' table keyed to lane 0's
    # blocks at this point -- request A's, not B's.

    model.save_snapshot(1, 0)
    model.load_snapshot(0, 0)
    model.attach_blocks(0, model.lane_blocks(1))  # lane 0 now aliases B's own blocks
    tok0, pos0, got_after = tok_b, pos_b, []
    for _ in range(steps_after):
        logits = model.decode([0], [tok0], [pos0])  # lane 1 left idle; only 0 need be right
        tok0 = int(logits.argmax(-1)[0])
        got_after.append(tok0)
        pos0 += 1
    assert got_after == ref_tokens[steps_before:]

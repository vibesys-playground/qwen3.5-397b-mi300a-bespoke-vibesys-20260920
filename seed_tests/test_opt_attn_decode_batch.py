"""Hermetic CPU tests: batched decode-time full attention against the block-paged design.

`Model.attn_decode` now reads each slot's K/V through its own `block_pool.BlockTable` into
the shared `[num_blocks * block_size, kv_heads, head_dim]` pool (`block_pool.py`,
`paged_attn.py`, `model.py`'s "paged KV pool" section) instead of a per-slot dense window of
a `[max_batch, kv_heads, max_seq, head_dim]` tensor. `Model._KVWindow`/`_kv_window`,
`_attend_batch`, the free-standing `decode_attention`, and `noisy_pool`'s raw per-slot pool
writes are all gone with that dense pool: there is no contiguous per-slot view to build a
window over any more (see `paged_attn.py`'s module docstring for why an in-kernel block-table
read replaces it).

What this file covers, against the real model code:

1. The core batching property, restated for the paged read: `Model.attn_decode`'s batched
   call over several slots must equal doing the same step one slot at a time through
   `Model.full_attention`, across a table of slot/position combinations (`CASES`) that
   includes a slot at position 0, every slot at position 0, out-of-ascending-order slots, and
   a prefix as long as the pool allows.
2. No leakage across the shared pool: two slots' block tables never hold the same block id
   (Stage 1 has no cross-session sharing yet -- see `block_pool.py`'s module docstring -- so
   this holds trivially, and is checked against the real allocator under real prefill traffic
   rather than assumed), and writing garbage into rows that belong to a *different* slot's
   block table never changes a given slot's `attn_decode` output. This replaces the two old
   dense-window leakage tests (a row outside the batch, a position past a slot's own prefix):
   the paged design collapses both into one property, "a row this slot's own table does not
   list is dead weight."
3. Fragmentation: a slot's block ids do not have to be a contiguous run in the pool (sessions
   come and go, freeing and re-taking blocks out of order), and `attn_decode` must read a
   fragmented table exactly as if it were laid out contiguously. This is the paged design's
   central claim, and the old dense-window design had no equivalent case to test at all.
4. The torch fallback inside `decode_attention_paged` (what CPU always runs, since
   `paged_attn.available` is cuda-only) folds the grouped query heads against `kv_heads`-wide
   K/V the same way `Model.full_attention`'s docstring avoids `enable_gqa`: it must never
   broadcast K/V up to `heads` with a literal `.expand`/`.repeat_interleave` before the
   matmul. There is no SDPA call in this path to spy on any more (see `paged_attn.py`'s module
   docstring), so this is checked directly: the two forbidden methods are made to fail the
   test if called, and the result is checked against a reference that does the wasteful
   broadcast on purpose.
5. A full-attention (and, unchanged, a DeltaNet) decode layer's aten dispatch count stays
   constant as batch size grows. Still true under paging: the per-slot host loop that grows
   each slot's block table (`Model.attn_decode`'s `grow_to`/`physical_row` calls) is pure
   Python bookkeeping (`block_pool.py` touches no tensor), and the `torch.tensor` builds it
   feeds into (`Model._host_index`, `Model._paged_read_buffers`) are one call each per
   `attn_decode`, regardless of how many slots are in the step.
6. End to end: scattered, non-ascending slots decode one token each exactly like one sequence
   at a time (`begin`/`prefill`/`forward`/`decode`, no internals touched).

Not covered here: `block_pool.BlockAllocator`/`BlockTable`'s own bookkeeping (refcounts, the
free list, `grow_to`'s block-count math) -- that is `seed_tests/test_block_pool.py`'s job, on
the allocator directly, torch-free.

    /tmp/torchenv/bin/python -m pytest \\
        examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/seed_tests/test_opt_attn_decode_batch.py
"""

import sys
from collections import Counter
from pathlib import Path

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import block_pool  # noqa: E402
import model as seed_model  # noqa: E402
from test_batched_decode import (  # noqa: E402
    MAX_SEQ,
    build,
    decode_activations,
    layer_index,
    prompt_of,
)
from test_seed_parity import VOCAB, build_hf, write_checkpoint  # noqa: E402

ATOL, RTOL = 1e-5, 1e-5


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("tiny")
    write_checkpoint(build_hf(), out, mxfp4=False)
    return out


# ---------------------------------------------------------------- fixtures for the unit cases


def prefilled_at(
    checkpoint: Path, max_batch: int, slots: list[int], positions: list[int], seed: int
) -> seed_model.Model:
    """A model with each of `slots` prefilled to a real (if arbitrary) `positions[j]`-token prefix.

    `positions[j]` is the position `attn_decode`/`full_attention` will write and read next --
    the same convention `test_batched_decode.py`'s own fixtures use (`positions[slot] =
    len(prompt)`). A position of 0 means "just began, nothing prefilled yet," which skips the
    prefill call (there are no tokens to embed).
    """
    model = build(checkpoint, max_batch=max_batch)
    for slot, pos in zip(slots, positions, strict=True):
        model.begin(slot)
        if pos:
            model.prefill(slot, prompt_of(seed + 97 * slot + pos, pos), 0)
    return model


# ---------------------------------------------------------------- (1) batched vs. per-slot loop

# (label, max_batch, slots, positions)
CASES = [
    ("single slot", 4, [2], [7]),
    ("every slot, mixed lengths", 4, [0, 1, 2, 3], [9, 3, 14, 6]),
    ("every slot, uniform lengths", 4, [0, 1, 2, 3], [11, 11, 11, 11]),
    ("scattered slots", 8, [1, 6], [4, 19]),
    ("slots out of ascending order", 8, [5, 0, 3], [12, 2, 8]),
    ("a slot at position 0", 4, [0, 1], [0, 9]),
    ("every slot at position 0", 4, [0, 1, 2], [0, 0, 0]),
    ("prefix as long as the pool", 4, [0, 1], [MAX_SEQ - 1, 5]),
    ("one slot, high index", 8, [7], [21]),
]


@pytest.mark.parametrize(("label", "max_batch", "slots", "positions"), CASES)
def test_batched_attention_matches_the_per_slot_loop(
    checkpoint: Path, label: str, max_batch: int, slots: list[int], positions: list[int]
) -> None:
    solo_model = prefilled_at(checkpoint, max_batch, slots, positions, seed=1000 + max_batch)
    batched_model = prefilled_at(checkpoint, max_batch, slots, positions, seed=1000 + max_batch)
    i = layer_index(solo_model, "full_attention")
    x = decode_activations(solo_model, len(slots), seed=2000 + max_batch)

    want = []
    for j, slot in enumerate(slots):
        solo_model.bind(slot)
        want.append(solo_model.full_attention(i, x[j : j + 1], positions[j]))
    got = batched_model.attn_decode(i, x, slots, positions)

    assert got.shape == x.shape
    for j in range(len(slots)):
        torch.testing.assert_close(got[j : j + 1], want[j], atol=ATOL, rtol=RTOL, msg=label)
    assert torch.isfinite(got).all(), label


# ---------------------------------------------------------------- (2) no leakage across the shared pool


def test_block_tables_never_share_a_block_across_busy_slots(checkpoint: Path) -> None:
    """Stage 1's whole-session-per-lane keying: no two lanes' tables ever hold the same block.

    Trivial by construction today (`block_pool.py`'s module docstring: Stage 1 keys a
    `BlockTable` by lane, so every `decref` drops straight back to the free list; sharing only
    starts to matter once Stage 2's session-keyed prefix trie lands). Checked here anyway,
    against the real allocator running under real prefill traffic, not assumed from reading
    the source.
    """
    slots, positions = [0, 1, 2, 3], [5, 20, 1, 33]
    model = prefilled_at(checkpoint, max_batch=4, slots=slots, positions=positions, seed=500)

    owner: dict[int, int] = {}
    for slot in range(model.max_batch):
        for b in model.block_tables[slot].blocks:
            assert b not in owner, f"block {b} held by both slot {owner[b]} and slot {slot}"
            owner[b] = slot
    assert owner, "fixture allocated no blocks at all"


def test_corrupting_other_slots_pool_rows_never_changes_this_slots_output(
    checkpoint: Path,
) -> None:
    """The paged equivalent of the two removed dense-window leakage tests.

    There is no window to leak past any more: every read goes through `victim`'s own block
    table, so the property to check is simply "another slot's rows are dead weight to this
    call," whether or not that slot is in this step's batch and whatever position within it.
    """
    victim, bystander = 1, 2
    model = prefilled_at(
        checkpoint, max_batch=4, slots=[victim, bystander], positions=[9, 13], seed=510
    )
    i = layer_index(model, "full_attention")
    x = decode_activations(model, 1, seed=511)

    before = model.attn_decode(i, x, [victim], [9])

    gen = torch.Generator().manual_seed(512)
    pool = model.pool[i]
    corrupt_rows = model.block_tables[bystander].physical_rows(13, model.block_size)
    assert corrupt_rows, "fixture must give the bystander at least one resident row to corrupt"
    for row in corrupt_rows:
        for name in ("k", "v"):
            pool[name][row] = torch.randn(
                pool[name][row].shape, generator=gen, dtype=pool[name].dtype
            )

    after = model.attn_decode(i, x, [victim], [9])
    assert torch.equal(before, after)


# ---------------------------------------------------------------- (3) fragmentation


def test_attn_decode_is_correct_when_a_slots_blocks_are_fragmented_in_the_pool(
    checkpoint: Path,
) -> None:
    """The paged design's central claim: physical layout must not matter to a logical read.

    Interleaves other slots' begin/prefill/reset cycles between two chunks of `keep`'s own
    prefill, so by the time `keep` grows its second block the free list's top is not the block
    adjacent to its first: `keep`'s two block ids end up non-contiguous in the pool (asserted
    below, not assumed -- the exact ids depend on `block_pool.BlockAllocator`'s LIFO free list,
    which this drives directly rather than guessing at). `attn_decode`'s output for `keep` must
    be identical either way, since `physical_rows`/`physical_row` are the only thing standing
    between a scattered block table and a wrong read.
    """
    block_size = seed_model.KV_BLOCK_SIZE
    keep, spacers = 0, [1, 2, 3, 4]
    keep_prompt = prompt_of(900, block_size + 3)  # spans 2 blocks, not a whole number of them
    first_chunk = block_size // 2

    model = build(checkpoint, max_batch=1 + len(spacers))
    model.begin(keep)
    model.prefill(keep, keep_prompt[:first_chunk], 0)  # keep's first block
    for s in spacers:
        model.begin(s)
        model.prefill(s, prompt_of(910 + s, block_size), 0)  # each spacer takes one block, held
    model.begin(spacers[1])  # release just that one spacer's block, opening a gap
    model.prefill(keep, keep_prompt[first_chunk:], first_chunk)  # keep's 2nd block fills the gap

    keep_blocks = model.block_tables[keep].blocks
    assert len(keep_blocks) == 2, keep_blocks
    lo, hi = min(keep_blocks), max(keep_blocks)
    assert sorted(keep_blocks) != list(range(lo, hi + 1)), (
        f"blocks {keep_blocks} came out contiguous; this case exercises nothing"
    )

    i = layer_index(model, "full_attention")
    pos = len(keep_prompt)
    x = decode_activations(model, 1, seed=901)
    model.bind(keep)
    fragmented = model.attn_decode(i, x, [keep], [pos])

    solo = build(checkpoint, max_batch=1)
    solo.begin(0)
    solo.prefill(0, keep_prompt, 0)
    solo.bind(0)
    unfragmented = solo.attn_decode(i, x, [0], [pos])

    torch.testing.assert_close(fragmented, unfragmented, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------- (4) the torch fallback's head fold


def decode_attention_paged_reference(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_table: torch.Tensor,
    block_valid: torch.Tensor,
    positions: torch.Tensor,
    block_size: int,
    scale: float,
) -> torch.Tensor:
    """What `decode_attention_paged`'s torch fallback must never actually run: expand the K/V
    pool read up to `heads` (a `repeat_interleave`) before the matmul, the same trap
    `model.full_attention`'s docstring calls out for `enable_gqa`'s fallback. A good oracle
    for the same reason it was one there: obvious, wasteful, and unarguably correct.
    """
    b, heads, _, head_dim = q.shape
    kv_heads = k_pool.shape[1]
    group = heads // kv_heads
    max_blocks = block_table.shape[1]
    dev = q.device

    tok = torch.arange(block_size, device=dev)
    logical_pos = (
        torch.arange(max_blocks, device=dev)[:, None] * block_size + tok[None, :]
    ).reshape(-1)
    live = logical_pos[None, :] <= positions[:, None]
    rows = (block_table[:, :, None] * block_size + tok[None, None, :]).reshape(b, -1)
    safe_rows = torch.where(live, rows, torch.zeros_like(rows))

    k = k_pool[safe_rows].permute(0, 2, 1, 3).float()  # [B, kv_heads, S, head_dim]
    v = v_pool[safe_rows].permute(0, 2, 1, 3).float()
    k = k.repeat_interleave(group, dim=1)  # [B, heads, S, head_dim] -- the broadcast to avoid
    v = v.repeat_interleave(group, dim=1)

    scores = torch.matmul(q.float(), k.transpose(-1, -2)) * scale  # [B, heads, 1, S]
    probs = scores.masked_fill(~live[:, None, None, :], float("-inf")).softmax(-1)
    out = torch.matmul(probs, v)  # [B, heads, 1, head_dim]
    return out.to(q.dtype)


def test_the_torch_fallback_never_broadcasts_kv_heads_before_the_matmul(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The paged path's equivalent of `model.full_attention`'s `enable_gqa` avoidance.

    There is no SDPA call in `decode_attention_paged`'s torch fallback to spy on any more (see
    `paged_attn.py`'s module docstring: the fused kernel folds the query's group axis directly
    rather than reshaping into the query length, and the torch fallback below is this
    function's own correctness oracle, not a second SDPA call). The equivalent lesson is
    checked directly: `Tensor.expand` and `Tensor.repeat_interleave` are made to fail the test
    if the fallback calls either, and the result is checked against a reference
    (`decode_attention_paged_reference`, above) that does the broadcast on purpose.
    """
    heads, kv_heads, head_dim, block_size, b = 4, 2, 8, 4, 3
    gen = torch.Generator().manual_seed(81)
    num_blocks = 6
    k_pool = torch.randn(num_blocks * block_size, kv_heads, head_dim, generator=gen)
    v_pool = torch.randn(num_blocks * block_size, kv_heads, head_dim, generator=gen)
    q = torch.randn(b, heads, 1, head_dim, generator=gen)
    block_table = torch.tensor([[1, 2, 0], [3, 4, 0], [2, 0, 0]], dtype=torch.int32)
    positions = torch.tensor([6, 7, 1])
    block_valid = torch.tensor(
        [block_pool.valid_block_count(int(p) + 1, block_size) for p in positions],
        dtype=torch.int32,
    )
    scale = head_dim**-0.5

    want = decode_attention_paged_reference(
        q, k_pool, v_pool, block_table, block_valid, positions, block_size, scale
    )

    def forbidden(name: str):  # noqa: ANN202
        def _fail(*_a: object, **_k: object) -> None:
            pytest.fail(f"decode_attention_paged's torch fallback called Tensor.{name}")

        return _fail

    monkeypatch.setattr(torch.Tensor, "expand", forbidden("expand"))
    monkeypatch.setattr(torch.Tensor, "repeat_interleave", forbidden("repeat_interleave"))

    got = seed_model.decode_attention_paged(
        q, k_pool, v_pool, block_table, block_valid, positions, block_size, scale
    )

    torch.testing.assert_close(got, want, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------- (5) one call per layer, not per slot


class CountAten(TorchDispatchMode):
    """Count aten dispatches, which is what the per-slot loop used to scale with."""

    def __init__(self) -> None:
        self.ops: Counter[str] = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: ANN001, ANN204
        self.ops[str(func)] += 1
        return func(*args, **kwargs or {})

    @property
    def total(self) -> int:
        return sum(self.ops.values())


def prefilled_slots(checkpoint: Path, lengths: list[int]) -> seed_model.Model:
    """One model whose slot j holds a prefilled prompt of `lengths[j]` tokens."""
    model = build(checkpoint, max_batch=len(lengths))
    for slot, length in enumerate(lengths):
        gen = torch.Generator().manual_seed(700 + slot)
        model.begin(slot)
        model.prefill(slot, torch.randint(2, VOCAB, (length,), generator=gen).tolist(), 0)
    return model


def decode_layer_ops(checkpoint: Path, batch: int, kind: str) -> int:
    lengths = [4 + 2 * j for j in range(batch)]
    model = prefilled_slots(checkpoint, lengths)
    i = layer_index(model, kind)
    x = decode_activations(model, batch, seed=71)
    with CountAten() as counter, torch.no_grad():
        model.decode_layer(i, x, list(range(batch)), lengths)
    return counter.total


def test_a_full_attention_decode_layer_costs_the_same_dispatches_at_any_batch(
    checkpoint: Path,
) -> None:
    """The regression this still guards against, restated for the paged write path.

    `Model.attn_decode`'s per-slot `grow_to`/`physical_row` host loop and its `_host_index`/
    `_paged_read_buffers` calls are pure Python plus a fixed handful of `torch.tensor` builds
    (one call each, regardless of how many slots are in the step): none of that scales with
    batch size any more than the one batched `decode_attention_paged` call itself does.
    """
    counts = [decode_layer_ops(checkpoint, b, "full_attention") for b in (2, 4, 8)]
    assert len(set(counts)) == 1, f"full-attention decode dispatches vary with batch: {counts}"


def test_a_deltanet_decode_layer_stays_constant_too(checkpoint: Path) -> None:
    """Pins the other half of the decode step, which was batched in an earlier round."""
    counts = [decode_layer_ops(checkpoint, b, "linear_attention") for b in (2, 4, 8)]
    assert len(set(counts)) == 1, f"DeltaNet decode dispatches vary with batch: {counts}"


# ---------------------------------------------------------------- (6) end to end


def test_scattered_slots_still_generate_the_one_at_a_time_tokens(checkpoint: Path) -> None:
    """A batch on non-contiguous, non-ascending slots must decode like one sequence at a time.

    The steady state of the served workload is exactly this: sessions pause between turns, so
    the slots that are decoding are a scattered subset of a fully occupied pool.
    """
    max_batch, slots = 6, [4, 1, 5]
    lengths = [5, 11, 3]
    prompts = [
        torch.randint(2, VOCAB, (n,), generator=torch.Generator().manual_seed(800 + j)).tolist()
        for j, n in enumerate(lengths)
    ]

    solo = build(checkpoint, max_batch=max_batch)
    want = []
    for slot, prompt in zip(slots, prompts, strict=True):
        solo.begin(slot)
        solo.prefill(slot, prompt, 0)
        want.append(solo.forward(torch.tensor([[prompt[-1]]]), len(prompt))[-1])

    batched = build(checkpoint, max_batch=max_batch)
    for slot, prompt in zip(slots, prompts, strict=True):
        batched.begin(slot)
        batched.prefill(slot, prompt, 0)
    got = batched.decode(slots, [p[-1] for p in prompts], [len(p) for p in prompts])

    for j, row in enumerate(want):
        torch.testing.assert_close(got[j], row, atol=1e-4, rtol=1e-4, msg=f"slot {slots[j]}")
        assert int(got[j].argmax()) == int(row.argmax())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

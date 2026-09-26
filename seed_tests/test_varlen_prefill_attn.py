"""`varlen_prefill_attn`'s packed varlen paged-prefill kernel against an independent oracle,
same posture as `test_paged_attn.py`/`test_decode_attn_v2.py`: the reference here gathers each
sequence's real, logical KV range straight from a synthetic paged pool (its resumed prefix plus
this chunk's own freshly-written rows, exactly as `full_attention` leaves the pool before this
kernel would run) and does a plain causal masked-softmax attention -- independent of
`varlen_prefill_attn.py` and of `model.py`.

Run modes, same as the other kernel test files:

    python -m pytest .../seed_tests/test_varlen_prefill_attn.py -p no:cacheprovider --no-cov
    TRITON_INTERPRET=1 python -m pytest ... -p no:cacheprovider --no-cov
"""

import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import block_pool  # noqa: E402
import varlen_prefill_attn  # noqa: E402

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NEEDS_KERNEL = pytest.mark.skipif(
    not varlen_prefill_attn.HAVE_TRITON or not (torch.cuda.is_available() or INTERPRET),
    reason="needs triton plus either an accelerator or TRITON_INTERPRET=1",
)

FP32_TOL = 5e-3
"""Looser than `test_paged_attn.py`'s 1e-4: this kernel's `P @ V` matmul rounds `P` to bf16
before the dot (see `varlen_prefill_attn.py`'s module docstring for why that is the deliberate,
standard FlashAttention-2 rounding, not a bug), which the decode kernels' pure-fp32 elementwise
path does not do."""


def relative_error(got: torch.Tensor, want: torch.Tensor) -> float:
    scale = want.float().abs().max().item()
    return ((got.float() - want.float()).abs().max() / max(scale, 1e-30)).item()


# ---------------------------------------------------------------- fixtures


def make_case(
    *,
    head_dim: int,
    block_size: int,
    kv_heads: int,
    group: int,
    seqs: list[dict],
    seed: int,
    device: torch.device = DEVICE,
) -> dict:
    """Build one packed varlen prefill call.

    `seqs` is a list of `{"block_ids": [...], "prefix_len": int, "chunk_len": int}`:
    `block_ids` are this sequence's resident block ids (covering `prefix_len + chunk_len`
    tokens once padded, never `block_pool.RESERVED_BLOCK`); `prefix_len` is how many of those
    tokens were already resident before this call (`start` in `full_attention`'s terms);
    `chunk_len` is how many new query tokens this call adds (`t`). The K/V pool is filled with
    random data for every resident row up front (mirroring `full_attention`'s write-before-
    attend order: by the time the kernel runs, the pool already holds this chunk's own K/V, not
    just the prefix's), and `q` holds this chunk's own new queries only, one packed `[total_T,
    heads, head_dim]` tensor across every sequence.
    """
    heads = kv_heads * group
    max_block_id = max(max(spec["block_ids"]) for spec in seqs)
    num_blocks = max_block_id + 1
    max_blocks_row = max(len(spec["block_ids"]) for spec in seqs)

    gen = torch.Generator().manual_seed(seed)
    k_pool = torch.randn(num_blocks * block_size, kv_heads, head_dim, generator=gen).to(device)
    v_pool = torch.randn(num_blocks * block_size, kv_heads, head_dim, generator=gen).to(device)

    total_t = sum(spec["chunk_len"] for spec in seqs)
    q = torch.randn(total_t, heads, head_dim, generator=gen).to(device)

    block_table = torch.tensor(
        [
            block_pool.BlockTable(blocks=list(spec["block_ids"])).padded_row(max_blocks_row)
            for spec in seqs
        ],
        dtype=torch.int32,
        device=device,
    )
    seq_start, seq_len, seq_prefix = [], [], []
    lo = 0
    for spec in seqs:
        seq_start.append(lo)
        seq_len.append(spec["chunk_len"])
        seq_prefix.append(spec["prefix_len"])
        lo += spec["chunk_len"]
    seq_start_t = torch.tensor(seq_start, dtype=torch.int32, device=device)
    seq_len_t = torch.tensor(seq_len, dtype=torch.int32, device=device)
    seq_prefix_t = torch.tensor(seq_prefix, dtype=torch.int32, device=device)

    return {
        "q": q,
        "k_pool": k_pool,
        "v_pool": v_pool,
        "block_table": block_table,
        "seq_start": seq_start_t,
        "seq_len": seq_len_t,
        "seq_prefix": seq_prefix_t,
        "block_size": block_size,
        "scale": head_dim**-0.5,
        "seqs": seqs,
        "kv_heads": kv_heads,
        "group": group,
    }


def reference_prefill(case: dict) -> torch.Tensor:
    """Independent causal-attention oracle: gather each sequence's real, resident KV range
    (`prefix_len + chunk_len` tokens) via its block table, then a plain masked-softmax
    attention with `key_pos <= query_pos` (`query_pos = prefix_len + local_row`). Written from
    scratch here, not derived from `varlen_prefill_attn.py`.
    """
    k_pool, v_pool = case["k_pool"], case["v_pool"]
    kv_heads, group, block_size = case["kv_heads"], case["group"], case["block_size"]
    seqs = case["seqs"]
    dev = case["q"].device
    heads = kv_heads * group
    head_dim = k_pool.shape[-1]

    out = torch.zeros(case["q"].shape[0], heads, head_dim, dtype=torch.float32, device=dev)
    lo = 0
    for seq in seqs:
        prefix_len, chunk_len = seq["prefix_len"], seq["chunk_len"]
        total_len = prefix_len + chunk_len
        rows = []
        for tok in range(total_len):
            blk, off = divmod(tok, block_size)
            rows.append(seq["block_ids"][blk] * block_size + off)
        rows_t = torch.tensor(rows, dtype=torch.long, device=dev)
        k = k_pool[rows_t].float()  # [total_len, kv_heads, head_dim]
        v = v_pool[rows_t].float()
        for kvh in range(kv_heads):
            qh = case["q"][lo : lo + chunk_len, kvh * group : (kvh + 1) * group].float()
            qh = qh.transpose(0, 1).float()  # [group, chunk_len, head_dim]
            scores = torch.einsum("gtd,sd->gts", qh, k[:, kvh, :]) * case["scale"]
            query_pos = prefix_len + torch.arange(chunk_len, device=dev)
            key_pos = torch.arange(total_len, device=dev)
            causal = key_pos[None, :] <= query_pos[:, None]  # [chunk_len, total_len]
            scores = scores.masked_fill(~causal[None, :, :], float("-inf"))
            probs = scores.softmax(dim=-1)
            o = torch.einsum("gts,sd->gtd", probs, v[:, kvh, :])  # [group, chunk_len, head_dim]
            out[lo : lo + chunk_len, kvh * group : (kvh + 1) * group] = o.transpose(0, 1)
        lo += chunk_len
    return out


# ---------------------------------------------------------------- the cases


def case_single_seq_fresh_prefix() -> dict:
    """No resumed prefix (`prefix_len=0`, the `start == 0` shape) -- the plain causal case."""
    return make_case(
        head_dim=16, block_size=4, kv_heads=1, group=1,
        seqs=[{"block_ids": [3, 7], "prefix_len": 0, "chunk_len": 6}],
        seed=1,
    )


def case_single_seq_resumed_prefix_crossing_boundary() -> dict:
    """Resumed prefix that does not end on a block boundary -- the first new token shares its
    block with cached prefix tokens, and its own block table entry (not a fresh block) is what
    `full_attention`'s `table.grow_to` would have appended to."""
    return make_case(
        head_dim=16, block_size=4, kv_heads=1, group=1,
        # prefix_len=5 sits mid-block-1 (blocks are 4 tokens); chunk_len=7 spans the rest of
        # block 1 into blocks 2 and 3.
        seqs=[{"block_ids": [9, 2, 15, 4], "prefix_len": 5, "chunk_len": 7}],
        seed=2,
    )


def case_multi_seq_packed_mixed_shapes() -> dict:
    """Several sequences in one packed call, mixed prefix/chunk lengths, non-ascending
    fragmented block ids -- the real packed-prefill shape."""
    return make_case(
        head_dim=16, block_size=4, kv_heads=1, group=1,
        seqs=[
            {"block_ids": [3], "prefix_len": 0, "chunk_len": 4},  # fresh, exactly one block
            {"block_ids": [5, 6, 7, 1], "prefix_len": 10, "chunk_len": 5},  # resumed, boundary
            {"block_ids": [9, 2], "prefix_len": 0, "chunk_len": 1},  # length-one chunk
        ],
        seed=3,
    )


def case_length_one_chunk_resumed() -> dict:
    """A single new token against a long resumed prefix -- the degenerate `qblk == 0`,
    `t_i == 1` tile, and `BLOCK_M` far wider than the chunk."""
    return make_case(
        head_dim=16, block_size=4, kv_heads=1, group=1,
        seqs=[{"block_ids": [9, 2, 15, 4, 20, 1, 8, 13], "prefix_len": 30, "chunk_len": 1}],
        seed=4,
    )


def case_chunk_wider_than_block_m() -> dict:
    """Chunk longer than `varlen_prefill_attn.BLOCK_M` (monkeypatched down to 4 by the test),
    so a single sequence spans more than one query-tile program."""
    return make_case(
        head_dim=16, block_size=4, kv_heads=1, group=1,
        seqs=[{"block_ids": list(range(1, 9)), "prefix_len": 0, "chunk_len": 30}],
        seed=5,
    )


def case_gqa_kv_heads_two_group_four() -> dict:
    return make_case(
        head_dim=32, block_size=4, kv_heads=2, group=4,
        seqs=[
            {"block_ids": [3, 6, 8], "prefix_len": 4, "chunk_len": 8},
            {"block_ids": [11, 2, 9, 7, 14], "prefix_len": 0, "chunk_len": 9},
        ],
        seed=6,
    )


CASES = [
    pytest.param(case_single_seq_fresh_prefix, id="single-seq-fresh-prefix"),
    pytest.param(
        case_single_seq_resumed_prefix_crossing_boundary, id="resumed-prefix-crosses-boundary"
    ),
    pytest.param(case_multi_seq_packed_mixed_shapes, id="multi-seq-packed-mixed-shapes"),
    pytest.param(case_length_one_chunk_resumed, id="length-one-chunk-resumed"),
    pytest.param(case_chunk_wider_than_block_m, id="chunk-wider-than-block-m"),
    pytest.param(case_gqa_kv_heads_two_group_four, id="gqa-kv-heads-2-group-4"),
]


@NEEDS_KERNEL
@pytest.mark.parametrize("case_fn", CASES)
def test_varlen_prefill_attn_matches_independent_reference(
    case_fn, monkeypatch: pytest.MonkeyPatch  # noqa: ANN001
) -> None:
    monkeypatch.setattr(varlen_prefill_attn, "BLOCK_M", 4)  # exercise multi-tile sequences
    monkeypatch.setattr(varlen_prefill_attn, "BLOCKS_PER_ITER", 2)  # exercise multi-iter reads
    case = case_fn()
    got = varlen_prefill_attn.varlen_prefill_attention_paged(
        case["q"],
        case["k_pool"],
        case["v_pool"],
        case["block_table"],
        case["seq_start"],
        case["seq_len"],
        case["seq_prefix"],
        case["block_size"],
        case["scale"],
    )
    want = reference_prefill(case)
    assert got.shape == want.shape
    err = relative_error(got, want)
    assert err < FP32_TOL, f"rel_err {err:.3e} >= {FP32_TOL:.3e}"


@NEEDS_KERNEL
def test_available_respects_the_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEED_VARLEN_PREFILL_ATTN", "0")
    assert not varlen_prefill_attn.available(torch.device("cuda"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

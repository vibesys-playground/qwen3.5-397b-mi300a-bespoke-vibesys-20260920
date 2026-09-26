"""Hermetic CPU tests for the opt-micro-sweep changes to model.py's attention module:

1. `Model.full_attention` now calls `F.scaled_dot_product_attention` instead of a manual
   softmax(QK^T/sqrt(d))V sequence.
2. `Model.rope` now slices a per-device table built once (`_rope_table`) instead of
   recomputing sin/cos from scratch on every call.

Each test compares the new implementation against the old formula (kept here, inline, as
the pre-change reference) on small random synthetic inputs, no checkpoint or HF model
needed.

Run with:
    /tmp/torchenv/bin/python -m pytest \
        examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/seed_tests/test_opt_micro_sweep.py \
        -p no:cacheprovider --no-cov
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import model as seed_model  # noqa: E402
import tp  # noqa: E402


def tiny_cfg(**over: object) -> seed_model.Cfg:
    base: dict = dict(
        hidden=32, vocab=64, layer_types=("full_attention",), eps=1e-6,
        heads=4, kv_heads=2, head_dim=8, rot_dim=8, rope_theta=10000.0,
        k_heads=1, v_heads=1, k_dim=1, v_dim=1, conv_k=1,
        experts=1, top_k=1, shared_inter=1, eos=(),
    )  # fmt: skip
    base.update(over)
    return seed_model.Cfg(**base)


def make_bare_model(
    cfg: seed_model.Cfg, dtype: torch.dtype, max_seq: int, max_batch: int = 1
) -> seed_model.Model:
    """A Model with only the state `full_attention`/`rope`/`attn_decode` touch, no checkpoint."""
    m = seed_model.Model.__new__(seed_model.Model)
    m.cfg, m.dtype, m.max_seq, m.max_batch = cfg, dtype, max_seq, max_batch
    m.devices = m.layer_dev = [torch.device("cpu")]
    m.tp = tp.TP.single(cfg, "cpu")  # `_new_pool` and `full_attention` read the plan's heads
    m.rope_cache = {}
    torch.manual_seed(0)
    m.layers = [
        {
            "q_proj": torch.randn(cfg.heads * cfg.head_dim * 2, cfg.hidden, dtype=dtype) * 0.1,
            "k_proj": torch.randn(cfg.kv_heads * cfg.head_dim, cfg.hidden, dtype=dtype) * 0.1,
            "v_proj": torch.randn(cfg.kv_heads * cfg.head_dim, cfg.hidden, dtype=dtype) * 0.1,
            "o_proj": torch.randn(cfg.hidden, cfg.heads * cfg.head_dim, dtype=dtype) * 0.1,
            "q_norm": torch.randn(cfg.head_dim, dtype=dtype) * 0.1,
            "k_norm": torch.randn(cfg.head_dim, dtype=dtype) * 0.1,
        }
    ]
    # Same paged-KV wiring as Model.__init__ (Stage 1: `tiny_cfg`'s one layer is always
    # full_attention, so this mirrors the `_new_kv_block_pool`/`block_tables` branch, not
    # `_new_pool` -- `_new_pool` now asserts it is never called for a full-attention layer).
    m.block_size = seed_model.KV_BLOCK_SIZE
    m.num_kv_blocks, m.num_snapshots = m._size_pools(torch.device("cpu"))  # noqa: SLF001
    m.block_allocator = seed_model.block_pool.BlockAllocator(m.num_kv_blocks, m.block_size)
    m.max_blocks_per_lane = seed_model.block_pool.max_blocks_per_lane(max_seq, m.block_size)
    m.block_tables = [seed_model.block_pool.BlockTable() for _ in range(max_batch)]
    m._prefill_scratch = [[None] for _ in range(max_batch)]
    m.pool = [m._new_kv_block_pool(0, m.num_kv_blocks, torch.device("cpu"))]  # noqa: SLF001
    m.snapshot_pool = [m._new_snapshot_pool(0)]  # noqa: SLF001
    m.slot_state = [[{}] for _ in range(max_batch)]  # full-attention layers: no per-slot row
    m.state = m.slot_state[0]
    m.current_slot = 0
    return m


def old_full_attention(m: seed_model.Model, i: int, x: torch.Tensor, start: int) -> torch.Tensor:
    """Pre-opt-micro-sweep `full_attention`: manual softmax(QK^T/sqrt(d))V, rope recomputed
    from scratch every call. Kept here only as the "old" reference for parity tests.

    `m.state[i]` is `{}` for a full-attention layer now (paged-KV, Stage 1: there is no
    per-slot dense "k"/"v" row any more, see `Model.__init__`), so this reference keeps its
    own private dense per-slot KV buffer instead, lazily created on `m` and persisting across
    calls the same way `m.state[i]["k"/"v"]` used to -- this function's whole point is to be
    the frozen pre-paged-KV formula, so it should not read the new paged pool at all.
    """
    c, w = m.cfg, m.layers[i]
    if not hasattr(m, "_old_kv"):
        m._old_kv = {
            "k": torch.zeros(1, c.kv_heads, m.max_seq, c.head_dim, dtype=x.dtype),
            "v": torch.zeros(1, c.kv_heads, m.max_seq, c.head_dim, dtype=x.dtype),
        }
    st = m._old_kv
    t = x.shape[1]
    q, gate = F.linear(x, w["q_proj"]).view(1, t, c.heads, 2 * c.head_dim).chunk(2, dim=-1)
    q = seed_model.rmsnorm(q, w["q_norm"], c.eps).transpose(1, 2)
    k = seed_model.rmsnorm(
        F.linear(x, w["k_proj"]).view(1, t, c.kv_heads, c.head_dim), w["k_norm"], c.eps
    ).transpose(1, 2)
    v = F.linear(x, w["v_proj"]).view(1, t, c.kv_heads, c.head_dim).transpose(1, 2)
    inv = 1.0 / (
        c.rope_theta ** (torch.arange(0, c.rot_dim, 2, device=x.device).float() / c.rot_dim)
    )
    pos = torch.arange(start, start + t, device=x.device).float()
    emb = torch.cat([pos[:, None] * inv[None], pos[:, None] * inv[None]], dim=-1)
    cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)
    q, k = seed_model.apply_rope(q, cos, sin), seed_model.apply_rope(k, cos, sin)
    st["k"][:, :, start : start + t] = k
    st["v"][:, :, start : start + t] = v
    rep = c.heads // c.kv_heads
    keys = st["k"][:, :, : start + t].repeat_interleave(rep, dim=1)
    vals = st["v"][:, :, : start + t].repeat_interleave(rep, dim=1)
    scores = (q @ keys.transpose(2, 3)) * c.head_dim**-0.5
    causal = (
        torch.arange(start + t, device=x.device)[None, :]
        <= (start + torch.arange(t, device=x.device))[:, None]
    )
    scores = scores.masked_fill(~causal, float("-inf"))
    out = (scores.float().softmax(-1).to(x.dtype) @ vals).transpose(1, 2).reshape(1, t, -1)
    return F.linear(out * torch.sigmoid(gate.reshape(1, t, -1)), w["o_proj"])


def test_sdpa_full_attention_matches_manual_softmax_prefill_and_decode() -> None:
    cfg = tiny_cfg()
    new_m = make_bare_model(cfg, torch.float32, max_seq=32)
    old_m = make_bare_model(cfg, torch.float32, max_seq=32)
    old_m.layers[0] = {k: v.clone() for k, v in new_m.layers[0].items()}

    # chunked prefill (start=0, t=5), a second prefill chunk against the filled KV cache
    # (start=5, t=3), then single-token decode (start=8, t=1): exercises the offset causal
    # mask (kv-cache position <= query absolute position) in every regime.
    for start, t in [(0, 5), (5, 3), (8, 1)]:
        torch.manual_seed(start)
        x = torch.randn(1, t, cfg.hidden, dtype=torch.float32) * 0.1
        got = new_m.full_attention(0, x, start)
        want = old_full_attention(old_m, 0, x, start)
        assert torch.allclose(got, want, atol=1e-5, rtol=1e-5), f"start={start} t={t}"


def test_sdpa_full_attention_matches_manual_softmax_no_gqa() -> None:
    """heads == kv_heads: enable_gqa is False, exercises the plain-MHA branch too."""
    cfg = tiny_cfg(heads=2, kv_heads=2)
    new_m = make_bare_model(cfg, torch.float32, max_seq=16)
    old_m = make_bare_model(cfg, torch.float32, max_seq=16)
    old_m.layers[0] = {k: v.clone() for k, v in new_m.layers[0].items()}

    torch.manual_seed(2)
    x = torch.randn(1, 4, cfg.hidden, dtype=torch.float32) * 0.1
    got = new_m.full_attention(0, x, 0)
    want = old_full_attention(old_m, 0, x, 0)
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5)


def test_rope_cache_matches_recompute_and_is_built_once_per_device() -> None:
    cfg = tiny_cfg()
    m = make_bare_model(cfg, torch.float32, max_seq=64)
    x = torch.randn(1, 1, cfg.hidden)

    for start, t in [(0, 5), (5, 3), (8, 1), (20, 1)]:
        got_cos, got_sin = m.rope(start, t, x)
        inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.rot_dim, 2).float() / cfg.rot_dim))
        pos = torch.arange(start, start + t).float()
        emb = torch.cat([pos[:, None] * inv[None], pos[:, None] * inv[None]], dim=-1)
        want_cos, want_sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)
        assert torch.allclose(got_cos, want_cos)
        assert torch.allclose(got_sin, want_sin)

    # Keyed by (device, FAULT_ROPE_BASE), not device alone, since FAULT_ROPE_BASE can flip at
    # runtime (see model.py's `_rope_table` docstring); the tests here never touch it, so it
    # is always False.
    key = (torch.device("cpu"), seed_model.FAULT_ROPE_BASE)
    assert list(m.rope_cache.keys()) == [key]
    table = m.rope_cache[key]
    m.rope(20, 1, x)
    assert m.rope_cache[key] is table  # not rebuilt on a later call


def test_is_causal_prefill_matches_the_explicit_mask_spelling() -> None:
    """`start == 0` takes `is_causal=True`; it must equal the explicit-mask spelling exactly.

    `full_attention` passes `is_causal=True` when there is no prefix, to keep SDPA on its
    flash backend: an explicit boolean mask drops it onto the math path, which materializes
    the [heads, t, start+t] score tensor (measured on MI300A: 5280 MiB at T=4096 against
    144 MiB under `is_causal`). This pins the two spellings agreeing, so the optimization
    cannot silently change the arithmetic.
    """
    cfg = tiny_cfg()
    m = make_bare_model(cfg, torch.float32, max_seq=32)
    c = cfg
    for t in (1, 4, 7):
        torch.manual_seed(t)
        q = torch.randn(1, c.heads, t, c.head_dim)
        k = torch.randn(1, c.kv_heads, t, c.head_dim)
        v = torch.randn(1, c.kv_heads, t, c.head_dim)
        gqa = c.heads != c.kv_heads
        want = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=torch.arange(t)[None, :] <= torch.arange(t)[:, None],
            scale=c.head_dim**-0.5,
            enable_gqa=gqa,
        )
        got = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=c.head_dim**-0.5, enable_gqa=gqa
        )
        assert torch.allclose(got, want, atol=1e-6, rtol=1e-6), f"t={t}"
    assert m.cfg is cfg  # the bare model is what full_attention runs against


def test_offset_causal_prefill_keeps_the_explicit_mask() -> None:
    """`start > 0` must NOT use `is_causal`: it would hide the prefix.

    The offset mask is `j <= start + i` over the whole cache, so every prefix key is visible
    to every query. `is_causal` on a non-square q/k aligns to the top-left and would mask
    most of the prefix away, which is why the branch in `full_attention` is on `start`.
    """
    cfg = tiny_cfg()
    c, start, t = cfg, 5, 3
    torch.manual_seed(1)
    q = torch.randn(1, c.heads, t, c.head_dim)
    k = torch.randn(1, c.kv_heads, start + t, c.head_dim)
    v = torch.randn(1, c.kv_heads, start + t, c.head_dim)
    gqa = c.heads != c.kv_heads
    offset = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=torch.arange(start + t)[None, :] <= (start + torch.arange(t))[:, None],
        scale=c.head_dim**-0.5,
        enable_gqa=gqa,
    )
    wrong = F.scaled_dot_product_attention(
        q, k, v, is_causal=True, scale=c.head_dim**-0.5, enable_gqa=gqa
    )
    assert not torch.allclose(offset, wrong, atol=1e-4)

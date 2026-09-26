"""EAGLE-style multi-token-prediction (MTP) speculative decoding.

Behind `SEED_MTP=1` (`SEED_MTP_K` draft tokens, default 3). See
`scratchpad/mtp-design.md` for the design and cost estimates this implements.

The checkpoint's `mtp.*` weights are architecturally one more decoder layer -- the same
full-attention self-attn (32 query heads, 2 KV heads, head_dim 256) and the same MoE
(512 experts, MXFP4, a shared expert) as any of the target model's 15 full-attention layers
-- fed by an `fc` that combines a normalized token embedding with a normalized hidden state,
instead of by the previous layer's residual stream. There is no dedicated MTP embedding or LM
head (`mtp_use_dedicated_embeddings: false`, no `mtp.lm_head.weight` key): both are the target
model's own `embed_tokens`/`lm_head`.

Three things this module owns, matching the design doc's numbered problems:

1. **Draft** (`draft`/`draft_step`): `k` autoregressive steps of the MTP layer alone, seeded
   by the target model's last hidden state (final-normed by default, `SEED_MTP_SEED_HIDDEN`)
   and last committed token. Its self-attention owns a paged KV cache over the complete
   sequence. Prompt, ordinary decode, and accepted verify rows rebuild that cache from the
   matching target hidden stream. Draft rows extend it tentatively at their absolute
   positions; rejected rows remain past the next visible position and are overwritten.
2. **Verify** (`verify_and_commit`): one *wide* forward of the *target* model over all
   `B * (k+1)` fed tokens (last committed + `k` drafts, per slot) at once --
   `Model.decode_layer_verify`, one dispatch per layer instead of `k+1` sequential
   `Model.decode_layer` calls. Dense/MoE run on the flattened `B*(k+1)` tokens as-is; full
   attention is one `attn_verify` call per layer (`k+1` query rows per lane against the shared
   paged KV pool, causal among the new rows); DeltaNet is one `cu_seqlens`-packed
   `deltanet_fused.fused_recurrent_prefill` call per layer across all `B` slots
   (`Model.deltanet_verify`).
3. **Rollback**: replay-the-accepted-prefix (design doc's option (b)). A full-attention layer's
   KV rollback is free (the next round's position argument simply does not advance past the
   accepted length; the speculatively-written-but-rejected rows are never read and are
   overwritten before they could be misread again). A DeltaNet layer's recurrent state is not
   free: `Model.deltanet_verify` keeps the pre-verify `conv`/`rec` state and the raw per-token
   inputs, and `Model.deltanet_verify_rollback` corrects the state to each slot's real accept
   length in one more wide `cu_seqlens` dispatch (conv's short window is exact by slicing,
   no recompute). See the design doc for why this beats per-step state snapshots (option (a))
   or a custom kernel (option (c)) given `fused_recurrent_prefill` only returns the *final*
   per-sequence state today.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import model as _m
import mtp_dense_moe
import mxfp4_gemv
import router_fused
import torch
import torch.nn.functional as F
from weights import Checkpoint


def _flag(name: str) -> bool:
    return os.environ.get(name, "0") not in ("0", "", "false", "False")


MTP_ENABLED = _flag("SEED_MTP_SERVE") or _flag("SEED_MTP")

DRAFT_FAST = os.environ.get("SEED_MTP_DRAFT_FAST", "1") not in ("0", "", "false", "False")
"""`draft_step` on the decode graph's production kernels (`_draft_step_fast`): split-K paged
attention with the fused q/k norm + RoPE + KV write, the fused all-reduce + add + RMSNorm,
expert-grouped BF16 experts (`mtp_dense_moe.grouped_moe`), and a vocab-parallel argmax that
all-gathers one `(max, index)` pair per rank instead of the full logits. Each piece only
applies where the matching production flag and kernel are available; otherwise it falls back
to `draft_step`'s original spelling. Same routing and same greedy token definition; the
kernels differ from the original draft by rounding order only. `=0` is the A/B knob."""

DRAFT_FUSED_ROUTE = os.environ.get("SEED_MTP_DRAFT_FUSED_ROUTE", "0") not in ("0", "", "false", "False")
"""`_moe_fast` routes with `router_fused.route` instead of `_route`'s softmax/topk (A/B knob)."""
"""Whether the scheduler drives decode through `Model.speculative_decode` instead of
`Model.decode`. `SEED_MTP_SERVE=1` is the serving flag; `SEED_MTP=1` is its older alias and
turns on exactly the same path. `scheduler.py` reads the same variables (it cannot import this
module -- see its own docstring on staying torch-free), so one flag turns the feature on end
to end."""

MAX_STOP_IDS = 8
"""Widest per-lane stop-id set a speculative round accepts (`commit_limit`'s `stops`, the
captured round's `[capacity, MAX_STOP_IDS]` stop buffer, `tp_driver`'s payload). A lane with
more stop ids makes the scheduler take the ordinary decode path for that step. Mirrored as
`scheduler.MTP_MAX_STOP_IDS` (torch-free)."""

MTP_K_CHOICES = (1, 2, 3)
"""The only `SEED_MTP_K` values the wide verify step and its graph capture (`graph_mtp.py`)
are sized for. `graph_mtp.py`'s verify buffers are `[capacity, mtp.k + 1, ...]`, fixed for the
lifetime of the process (baked in at `prepare()` time), so `k` has to be a boot-time choice,
not a per-request one -- this is that choice's closed set. 3 is `SGLang`'s own EAGLE baseline
(not read, just matched as a target: 3 steps, topk=1, 4 draft tokens including the bonus); 1
and 2 are here for accuracy/latency trade-off sweeps at a cheaper verify width."""


def _read_mtp_k() -> int:
    raw = os.environ.get("SEED_MTP_K", "3")
    try:
        k = int(raw)
    except ValueError:
        raise ValueError(f"SEED_MTP_K must be an integer in {MTP_K_CHOICES}, got {raw!r}") from None
    if k not in MTP_K_CHOICES:
        raise ValueError(f"SEED_MTP_K must be one of {MTP_K_CHOICES}, got {k}")
    return k


MTP_K = _read_mtp_k()
"""Draft tokens per decode round, validated at import time (boot) against `MTP_K_CHOICES`
rather than left to fail wherever a wrongly-shaped `k` would first surface (a verify buffer
size mismatch deep in `graph_mtp.py`, or a lopsided draft loop). `SGLang`'s own EAGLE baseline
(not read, just matched as a target) uses 3 steps, topk=1, 4 draft tokens; `MTP_K=3` plus the
one bonus token is the same 4-tokens-per-round shape."""


@dataclass
class MTP:
    """One rank's MTP module and its paged, sequence-persistent KV cache."""

    k: int
    weights: dict  # self_attn + moe, keyed like a real layer's dict (see load_layer)
    fc: torch.Tensor
    pre_fc_norm_embedding: torch.Tensor
    pre_fc_norm_hidden: torch.Tensor
    norm: torch.Tensor
    pool: dict  # {"k": [physical_rows, kv_heads_local, head_dim], "v": same}
    embedding_scratch: torch.Tensor
    expert_inter_scratch: torch.Tensor
    expert_out_scratch: torch.Tensor
    grouped_scratch: dict | None = None  # `mtp_dense_moe.grouped_moe`'s, SEED_MTP_DRAFT_FAST


def load(
    ck: Checkpoint,
    full_cfg: _m.Cfg,
    tp: _m.TP,
    dev: torch.device,
    dtype: torch.dtype,
    max_batch: int,
    kv_rows: int,
) -> MTP | None:
    """Load the MTP module, or `None` when the checkpoint has none (or the flag is off).

    `full_cfg` is the *global* config, exactly what `load_attention`/`load_moe_dense` expect
    (see `load_layer`'s own docstring): the MTP self-attn and MoE are sharded the same way a
    real layer's are, by the same functions, from the same `tp.plan`. The top-level `mtp.*`
    tensors (`fc`, the two pre-fc norms, the final norm) are replicated on every rank, like
    `embed_tokens`/`lm_head`/the final norm are: none of them touch a sharded axis.
    """
    if not MTP_ENABLED or not ck.has("mtp.fc.weight"):
        return None
    top = _m._Reader(ck, "mtp", dev, dtype)
    layer = _m._Reader(ck, "mtp.layers.0", dev, dtype)
    weights = {
        "in_norm": layer.whole("input_layernorm.weight"),
        "post_norm": layer.whole("post_attention_layernorm.weight"),
        **_m.load_attention(layer, full_cfg, tp.plan),
        **_m.load_moe_dense(layer, tp.plan),
        "experts": _m.load_experts(ck, "mtp.layers.0.mlp", tp.plan.experts, dev, dtype),
    }
    k = MTP_K
    pool = {
        name: torch.zeros(
            kv_rows, tp.plan.kv.count, full_cfg.head_dim, dtype=dtype, device=dev
        )
        for name in ("k", "v")
    }
    inter = weights["experts"]["gate_up"].shape[1] // 2
    return MTP(
        k=k,
        weights=weights,
        fc=top.whole("fc.weight"),
        pre_fc_norm_embedding=top.whole("pre_fc_norm_embedding.weight"),
        pre_fc_norm_hidden=top.whole("pre_fc_norm_hidden.weight"),
        norm=top.whole("norm.weight"),
        pool=pool,
        embedding_scratch=torch.empty(
            max_batch * (k + 1), full_cfg.hidden, dtype=dtype, device=dev
        ),
        expert_inter_scratch=torch.empty(
            max_batch * full_cfg.top_k, inter, dtype=dtype, device=dev
        ),
        expert_out_scratch=torch.empty(max_batch, full_cfg.hidden, dtype=dtype, device=dev),
        grouped_scratch=mtp_dense_moe.grouped_scratch(
            max_batch, full_cfg.top_k, weights["experts"]["gate_up"].shape[0], full_cfg.hidden, dev
        ),
    )


# ---------------------------------------------------------------- draft


def _combine(
    model: _m.Model, mtp: MTP, hidden: torch.Tensor, tokens: list[int] | torch.Tensor
) -> torch.Tensor:
    """`fc(concat(norm(embed(tokens)), norm(hidden)))`, [B, 1, hidden].

    `hidden` is [B, hidden], raw (pre-final-norm): the target's own last hidden state at step
    0, this function's own previous output at every later draft step. Both sources feed the
    same two norms and the same `fc`, which is what lets one draft step serve both cases.

    `tokens` is a host `list[int]` (step 0: the target's own last committed token, already a
    Python value the caller holds) or a `[B, 1]` device `LongTensor` (every later step: `draft`
    keeps its own token feedback as a GPU tensor -- see that function's docstring -- so this
    accepts both instead of forcing a host round trip on the steps that do not need one).
    """
    c = model.cfg
    ids = tokens if torch.is_tensor(tokens) else torch.tensor([[t] for t in tokens], device=hidden.device)
    embedding = mtp.embedding_scratch[: ids.numel()].view(*ids.shape, c.hidden)
    mtp_dense_moe.gather_embeddings(model.embed, ids, embedding)
    emb = _m.rmsnorm(embedding, mtp.pre_fc_norm_embedding, c.eps)
    hid = _m.rmsnorm(hidden[:, None, :], mtp.pre_fc_norm_hidden, c.eps)
    return F.linear(torch.cat([emb, hid], dim=-1), mtp.fc)


def _combine_rows(
    model: _m.Model,
    mtp: MTP,
    hidden: torch.Tensor,
    tokens: torch.Tensor,
    embedding_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """The MTP fusion projection for matching arbitrary leading dimensions."""
    c = model.cfg
    if embedding_out is None:
        embedding = F.embedding(tokens, model.embed)
    else:
        mtp_dense_moe.gather_embeddings(model.embed, tokens, embedding_out)
        embedding = embedding_out
    emb = _m.rmsnorm(embedding, mtp.pre_fc_norm_embedding, c.eps)
    hid = _m.rmsnorm(hidden, mtp.pre_fc_norm_hidden, c.eps)
    return F.linear(torch.cat([emb, hid], dim=-1), mtp.fc)


def _attn_rows(
    model: _m.Model,
    mtp: MTP,
    x: torch.Tensor,
    rope_pos: torch.Tensor,
    write_rows: torch.Tensor,
    block_table: torch.Tensor,
    block_valid: torch.Tensor,
    active: torch.Tensor | None = None,
) -> torch.Tensor:
    """The MTP layer's own self-attention for one draft step. x is [B, 1, hidden], normed.

    The MTP cache uses the target's block table and physical rows. It therefore survives
    decode rounds, prefix snapshots and copy-on-write exactly as the target full-attention
    cache does. `active` makes graph padding writes no-ops.
    """
    c, w, pl = model.cfg, mtp.weights, model.tp.plan
    b, nq, nkv = x.shape[0], pl.q.count, pl.kv.count
    q, gate = F.linear(x, w["q_proj"]).view(b, 1, nq, 2 * c.head_dim).chunk(2, dim=-1)
    q = _m.rmsnorm(q, w["q_norm"], c.eps).transpose(1, 2)
    k = _m.rmsnorm(
        F.linear(x, w["k_proj"]).view(b, 1, nkv, c.head_dim), w["k_norm"], c.eps
    ).transpose(1, 2)
    v = F.linear(x, w["v_proj"]).view(b, 1, nkv, c.head_dim).transpose(1, 2)
    cos, sin = model.rope_at(rope_pos, x.dtype)
    q = _m.apply_rope(q, cos[:, None, None], sin[:, None, None])
    k = _m.apply_rope(k, cos[:, None, None], sin[:, None, None])
    new_k, new_v = k[:, :, 0], v[:, :, 0]
    live = active if active is not None else torch.ones_like(write_rows, dtype=torch.bool)
    mtp_dense_moe.scatter_active_first_dim(
        mtp.pool["k"], write_rows, new_k.contiguous(), live
    )
    mtp_dense_moe.scatter_active_first_dim(
        mtp.pool["v"], write_rows, new_v.contiguous(), live
    )
    out = _m.verify_attention_paged(
        q,
        mtp.pool["k"],
        mtp.pool["v"],
        block_table,
        block_valid,
        rope_pos,
        model.block_size,
        c.head_dim**-0.5,
    )
    out = out.transpose(1, 2).reshape(b, 1, -1)
    return F.linear(out * torch.sigmoid(gate.reshape(b, 1, -1)), w["o_proj"])


def _moe(
    model: _m.Model, mtp: MTP, x: torch.Tensor, active: torch.Tensor | None = None
) -> torch.Tensor:
    """The MTP layer's MoE. Routing and the dequant it depends on are `Model`'s own (shared);
    only the routed-expert dispatch loop is duplicated, because `Model._routed_grouped` and
    `Model._routed_fused` are indexed by a real layer number (`self.layers[i]`,
    `self.moe_inter_scratch[i]`) that several existing seed_tests bind directly by name
    (`test_moe_vectorize.py`, `test_prealloc_buffers.py`), so they cannot be reparameterized
    without breaking those tests. The MTP layer is not one of `self.layers` (appending it there
    would make every `range(len(self.layers))` loop in `Model.forward`/`forward_packed` walk
    off the end of `cfg.layer_types`), so it needs its own copy of just that dispatch."""
    c, w = model.cfg, mtp.weights
    h = x.reshape(-1, c.hidden)
    routing, experts_out, top_w, top_i = _route(model, mtp, h)
    ex = w["experts"]
    if "gate_up_scale" in ex and mxfp4_gemv.available(h.device):
        # The fused MXFP4 kernels: every shape is a function of `t` and `top_k` alone, no host
        # read, so the draft step is capturable (`graph_mtp.draft_round_step`). Same call
        # `Model._routed_fused` makes on its default (per-assignment) path; the MTP experts
        # are never `bw_shuffle`d, so the variant kernels do not apply to them.
        out = mxfp4_gemv.fused_moe(
            h,
            ex,
            (top_i.to(torch.int32).reshape(-1), top_w.reshape(-1)),
            c.top_k,
            model.expert_range,
            total_experts=c.experts,
        )
    elif mtp_dense_moe.available(h, ex):
        out = mtp_dense_moe.fused_moe(
            h,
            ex,
            (top_i, top_w),
            c.top_k,
            model.expert_range,
            mtp.expert_inter_scratch,
            mtp.expert_out_scratch,
            active,
        )
    else:
        out = _routed_grouped(model, w, h, top_i, top_w.to(h.dtype))
    shared = _m.swiglu_mlp(h, w["shared_expert.gate_up_proj"], w["shared_expert.down_proj"])
    out = out + torch.sigmoid(routing[:, experts_out:]) * shared
    return out.reshape(x.shape)


def _route(
    model: _m.Model, mtp: MTP, h: torch.Tensor
) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
    """The MTP router: `(router GEMM output, routed column count, top_w, top_i)`."""
    c, w = model.cfg, mtp.weights
    routing = F.linear(h, w["router_gate"])
    experts_out = w["router"].shape[0]
    if _m.FUSE_GLUE and router_fused.available(h.device):
        # `torch.topk` on ROCm dispatches through a runtime path that cannot be recorded by a
        # HIP graph. The target layers already use this fixed-shape Triton route kernel for
        # captured decode; the MTP layer needs the same route, for both capture and parity.
        top_w, top_i = router_fused.route(routing, experts_out, c.top_k)
    else:
        probs = routing[:, :experts_out].softmax(-1, dtype=torch.float)
        top_w, top_i = probs.topk(c.top_k, dim=-1)
        top_w = top_w / top_w.sum(-1, keepdim=True)
    return routing, experts_out, top_w, top_i


def _routed_grouped(
    model: _m.Model, w: dict, h: torch.Tensor, top_i: torch.Tensor, top_w: torch.Tensor
) -> torch.Tensor:
    """`Model._routed_grouped`'s algorithm, parameterized by `w` instead of `self.layers[i]`.

    Dense-checkpoint/CPU fallback of `_moe` (MXFP4 payloads on an accelerator take the fused
    kernels instead). Sizes its `bmm` by host reads, so it is not capturable; the CPU tests'
    capture backend re-runs the step eagerly and does not care.
    """
    lo, hi = model.expert_range
    flat_expert, flat_weight = top_i.reshape(-1), top_w.reshape(-1)
    flat_token = torch.arange(h.shape[0], device=h.device).repeat_interleave(model.cfg.top_k)
    mine = (flat_expert >= lo) & (flat_expert < hi)
    flat_expert = flat_expert[mine] - lo
    flat_weight, flat_token = flat_weight[mine], flat_token[mine]
    if flat_expert.numel() == 0:
        return _m._zeroed_accumulator(h, None)
    order = torch.argsort(flat_expert, stable=True)
    sorted_expert, sorted_token, sorted_weight = (
        flat_expert[order],
        flat_token[order],
        flat_weight[order],
    )
    active, counts = torch.unique_consecutive(sorted_expert, return_counts=True)
    num_active, max_count = active.numel(), int(counts.max())
    row = torch.repeat_interleave(torch.arange(num_active, device=h.device), counts)
    pos = torch.arange(sorted_expert.shape[0], device=h.device) - torch.repeat_interleave(
        counts.cumsum(0) - counts, counts
    )
    tok_idx = torch.zeros(num_active, max_count, dtype=torch.long, device=h.device)
    weight_pad = torch.zeros(num_active, max_count, dtype=h.dtype, device=h.device)
    mask = torch.zeros(num_active, max_count, dtype=torch.bool, device=h.device)
    tok_idx[row, pos] = sorted_token
    weight_pad[row, pos] = sorted_weight
    mask[row, pos] = True
    h_pad = h[tok_idx]
    gate_up_w, down_w = model.expert_weights_batch(w["experts"], active)
    gate, up = torch.bmm(h_pad, gate_up_w.transpose(1, 2)).chunk(2, dim=-1)
    y = torch.bmm(F.silu(gate) * up, down_w.transpose(1, 2))
    out = _m._zeroed_accumulator(h, None)
    out.index_add_(0, tok_idx[mask], (y[mask] * weight_pad[mask, None]).to(out.dtype))
    return out


def draft(
    model: _m.Model,
    mtp: MTP,
    hidden: torch.Tensor,
    tokens: list[int],
    slots: list[int],
    positions: list[int],
) -> list[list[int]]:
    """`mtp.k` autoregressive MTP steps. `hidden`/`tokens` are the target's last raw hidden
    state and last committed token id, one row per slot; `positions` is that token's absolute
    position. Returns each slot's `k` draft token ids, greedy (this path only ever runs with an
    all-greedy batch, see `Scheduler._speculative_ok`).

    The eager spelling of `graph_mtp.draft_round_step`: both loop over `draft_step`, whose
    token feedback stays on the device, so this reads the drafts back once per round.
    """
    b = len(slots)
    hid, tok = draft_seed(model, hidden), torch.tensor([[t] for t in tokens], device=hidden.device)
    out_tensor = torch.empty(b, mtp.k, dtype=torch.long, device=hidden.device)
    for step in range(mtp.k):
        step_positions = [p + step for p in positions]
        pos = model._host_index("mtp_rope_pos", hidden.device, tuple(step_positions))
        write_rows = torch.cat(
            [model._physical_rows_range(slot, p, p + 1, hidden.device) for slot, p in zip(slots, step_positions, strict=True)]
        )
        block_table, block_valid = model._paged_read_buffers(hidden.device, slots, step_positions)
        hid, tok = draft_step(model, mtp, hid, tok, pos, write_rows, block_table, block_valid)
        out_tensor[:, step] = tok[:, 0]
    return out_tensor.tolist()  # one host round trip for the whole round, not one per step


SEED_POST_NORM = os.environ.get("SEED_MTP_SEED_HIDDEN", "post") != "pre"
"""Which hidden state feeds the MTP head's `pre_fc_norm_hidden`: the target's (and, from the
second draft step on, the MTP layer's own) *final-normed* output, i.e. what the LM head reads
(`post`, default), or the raw residual stream before that norm (`pre`, the original
implementation). DeepSeek-V3's MTP (arXiv 2412.19437, section 2.2) defines the input as "the
output representation of the main model" before the output head, and vLLM's and SGLang's MTP
drafters pass the model's normed output; `pre` differs from it by the final norm's
per-channel scale, which lowers acceptance but never changes committed tokens (verify is
exact). `SEED_MTP_SEED_HIDDEN=pre` is the A/B knob."""


def draft_seed(model: _m.Model, raw_hidden: torch.Tensor) -> torch.Tensor:
    """Step 0's hidden input from the target's raw cached hidden (`Model.hidden_scratch`)."""
    if SEED_POST_NORM:
        return _m.rmsnorm(raw_hidden, model.final_norm, model.cfg.eps)
    return raw_hidden


def cache_target_rows(
    model: _m.Model,
    mtp: MTP,
    tokens: torch.Tensor,
    raw_previous_hidden: torch.Tensor,
    positions: torch.Tensor,
    write_rows: torch.Tensor,
    keep: torch.Tensor | None = None,
    embedding_out: torch.Tensor | None = None,
) -> None:
    """Write canonical MTP K/V derived from target hidden states.

    Draft K/V for accepted tokens must be rebuilt from the target hidden stream, matching
    draft-extend in NEXTN implementations. Rejected speculative rows remain unreachable.
    This fixed-shape spelling is also used for prompt extension and graph replay.
    """
    c, w, pl = model.cfg, mtp.weights, model.tp.plan
    hidden = draft_seed(model, raw_previous_hidden)
    x = _combine_rows(model, mtp, hidden, tokens, embedding_out)
    x = _m.rmsnorm(x, w["in_norm"], c.eps)
    nkv = pl.kv.count
    k = _m.rmsnorm(
        F.linear(x, w["k_proj"]).view(*x.shape[:-1], nkv, c.head_dim), w["k_norm"], c.eps
    )
    v = F.linear(x, w["v_proj"]).view(*x.shape[:-1], nkv, c.head_dim)
    cos, sin = model.rope_at(positions.reshape(-1), x.dtype)
    flat_n = positions.numel()
    k = _m.apply_rope(
        k.reshape(flat_n, nkv, c.head_dim).transpose(0, 1).unsqueeze(0),
        cos.reshape(1, 1, flat_n, c.rot_dim),
        sin.reshape(1, 1, flat_n, c.rot_dim),
    ).squeeze(0).transpose(0, 1).reshape(*positions.shape, nkv, c.head_dim)
    flat_rows = write_rows.reshape(-1)
    flat_k = k.reshape(-1, nkv, c.head_dim)
    flat_v = v.reshape(-1, nkv, c.head_dim)
    if keep is None:
        mtp.pool["k"][flat_rows] = flat_k
        mtp.pool["v"][flat_rows] = flat_v
    else:
        live = keep.reshape(-1)
        mtp_dense_moe.scatter_active_first_dim(
            mtp.pool["k"], flat_rows, flat_k.contiguous(), live
        )
        mtp_dense_moe.scatter_active_first_dim(
            mtp.pool["v"], flat_rows, flat_v.contiguous(), live
        )


def prefill_cache(
    model: _m.Model,
    mtp: MTP,
    slot: int,
    tokens: torch.Tensor,
    raw_hidden: torch.Tensor,
    start: int,
    raw_previous_hidden: torch.Tensor,
) -> None:
    """Extend one lane's persistent MTP K/V alongside a target prefill chunk."""
    if tokens.numel() == 0:
        return
    previous = torch.cat([raw_previous_hidden.reshape(1, -1), raw_hidden[:-1]], dim=0)
    pos = torch.arange(start, start + tokens.numel(), device=raw_hidden.device)
    rows = model._physical_rows_range(slot, start, start + tokens.numel(), raw_hidden.device)
    cache_target_rows(model, mtp, tokens.reshape(-1), previous, pos, rows)


def draft_step(
    model: _m.Model,
    mtp: MTP,
    hid: torch.Tensor,
    tok: torch.Tensor,
    rope_pos: torch.Tensor,
    write_rows: torch.Tensor,
    block_table: torch.Tensor,
    block_valid: torch.Tensor,
    active: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One MTP decoder-layer step over device tensors only: `(hid [B, hidden], tok [B, 1])` in,
    the next pair out. `rope_pos` is each row's absolute position, `rows` its lane (the
    `MTP.pool` row), and optional `active` masks graph-bucket padding before expert weight
    reads. No host read and no data-dependent shape once `_moe` takes the fused kernels, so
    `graph_mtp` captures this same function.

    The fc output is the layer's residual input and goes through `input_layernorm` before
    self-attention, like any pre-norm decoder layer (Qwen3-Next's MTP block is a standard
    decoder layer fed by `fc`).
    """
    if DRAFT_FAST and hid.is_cuda:
        return _draft_step_fast(
            model, mtp, hid, tok, rope_pos, write_rows, block_table, block_valid, active
        )
    c, w = model.cfg, mtp.weights
    x = _combine(model, mtp, hid, tok)
    mixer = _attn_rows(
        model,
        mtp,
        _m.rmsnorm(x, w["in_norm"], c.eps),
        rope_pos,
        write_rows,
        block_table,
        block_valid,
        active,
    )
    x = x + model.tp.all_reduce(mixer)
    moe_out = _moe(model, mtp, _m.rmsnorm(x, w["post_norm"], c.eps), active)
    x = x + model.tp.all_reduce(moe_out)
    normed = _m.rmsnorm(x, mtp.norm, c.eps)
    logits = model.unembed(normed)[:, 0].float()
    tok = logits.argmax(-1, keepdim=True)  # [B, 1] long, fed straight to the next _combine
    return (normed if SEED_POST_NORM else x)[:, 0], tok


def _attn_fast(
    model: _m.Model,
    mtp: MTP,
    h: torch.Tensor,
    rope_pos: torch.Tensor,
    write_rows: torch.Tensor,
    block_table: torch.Tensor,
    block_valid: torch.Tensor,
    active: torch.Tensor | None,
) -> torch.Tensor:
    """`_attn_rows` on `graph_decode.attn_decode_core`'s kernels: the fused q/k RMSNorm +
    RoPE + MTP-pool write (`SEED_ATTN_ROPE_KV_FUSED`), `model.decode_attention_paged`
    (split-K under `SEED_DECODE_ATTN_SPLITK`, one query row per lane, which the draft always
    has), and the fused sigmoid gate (`SEED_ELEMWISE_FUSED`). Returns the `o_proj` partial."""
    import attn_decode_fused  # noqa: PLC0415 -- accelerator-side module, import on use
    import decode_glue  # noqa: PLC0415
    import graph_decode  # noqa: PLC0415 -- imports this module's importers; avoid a cycle

    c, w, pl = model.cfg, mtp.weights, model.tp.plan
    b, nq, nkv, hd = h.shape[0], pl.q.count, pl.kv.count, c.head_dim
    q_raw, gate = F.linear(h, w["q_proj"]).view(b, 1, nq, 2 * hd).chunk(2, dim=-1)
    k_raw = F.linear(h, w["k_proj"]).view(b, 1, nkv, hd)
    v_raw = F.linear(h, w["v_proj"]).view(b, 1, nkv, hd)
    cos, sin = model.rope_at(rope_pos, h.dtype)
    if graph_decode.ATTN_ROPE_KV_FUSED and attn_decode_fused.available(h.device):
        q = attn_decode_fused.rmsnorm_rope_and_kv_write(
            q_raw, k_raw, v_raw, w["q_norm"], w["k_norm"], cos, sin, c.rot_dim, c.eps,
            mtp.pool["k"], mtp.pool["v"], write_rows.contiguous(),
            (active if active is not None else torch.ones_like(write_rows, dtype=torch.bool)).contiguous(),
        )
    else:
        q = _m.apply_rope(
            _m.rmsnorm(q_raw, w["q_norm"], c.eps).transpose(1, 2), cos[:, None, None], sin[:, None, None]
        )
        k = _m.apply_rope(
            _m.rmsnorm(k_raw, w["k_norm"], c.eps).transpose(1, 2), cos[:, None, None], sin[:, None, None]
        )
        live = active if active is not None else torch.ones_like(write_rows, dtype=torch.bool)
        mtp_dense_moe.scatter_active_first_dim(mtp.pool["k"], write_rows, k[:, :, 0].contiguous(), live)
        mtp_dense_moe.scatter_active_first_dim(
            mtp.pool["v"], write_rows, v_raw[:, 0].contiguous(), live
        )
    out = _m.decode_attention_paged(
        q, mtp.pool["k"], mtp.pool["v"], block_table, block_valid, rope_pos, model.block_size, hd**-0.5
    )
    out = out.transpose(1, 2).reshape(b, 1, -1)
    if decode_glue.available(out):
        out = decode_glue.sigmoid_gate_mul(out, gate)
    else:
        out = out * torch.sigmoid(gate.reshape(b, 1, -1))
    return F.linear(out, w["o_proj"])


def _moe_fast(
    model: _m.Model, mtp: MTP, x: torch.Tensor, active: torch.Tensor | None
) -> torch.Tensor:
    """`_moe` with the routed half on `mtp_dense_moe.grouped_moe` (each touched local expert
    read once per step instead of once per assignment). Same router, same shared expert."""
    c, w = model.cfg, mtp.weights
    h = x.reshape(-1, c.hidden)
    ex = w["experts"]
    if mtp.grouped_scratch is None or not mtp_dense_moe.available(h, ex):
        return _moe(model, mtp, x, active)
    if DRAFT_FUSED_ROUTE and router_fused.available(h.device):
        # The target decode graph's one-kernel route. Not the default: its fp32 reduction
        # order and `tl.argmax` tie-break can pick a different expert than `_route`'s
        # softmax/topk at a near tie, which moves draft tokens for ~0.01 ms a step.
        routing = F.linear(h, w["router_gate"])
        experts_out = w["router"].shape[0]
        top_w, top_i = router_fused.route(routing, experts_out, c.top_k)
    else:
        routing, experts_out, top_w, top_i = _route(model, mtp, h)
    out = mtp_dense_moe.grouped_moe(
        h,
        ex,
        (top_i.contiguous(), top_w.contiguous()),
        c.top_k,
        model.expert_range,
        mtp.expert_inter_scratch,
        mtp.expert_out_scratch,
        mtp.grouped_scratch,
        active,
    )
    shared = _m.swiglu_mlp(h, w["shared_expert.gate_up_proj"], w["shared_expert.down_proj"])
    out = out + torch.sigmoid(routing[:, experts_out:]) * shared
    return out.reshape(x.shape)


def greedy_token(model: _m.Model, normed: torch.Tensor) -> torch.Tensor:
    """`model.unembed(normed).float().argmax(-1, keepdim=True)` for `[B, hidden]` rows, `[B, 1]`
    long, without all-gathering the logits under `SEED_LMHEAD_VOCAB_TP`.

    Each rank takes the argmax of its own vocab shard (the first maximum, as `argmax` does),
    then one all-gather of `(max logit, global index)` per rank and row picks the largest
    value, lowest rank on a tie. Shards are contiguous and in rank order, so that is the first
    global maximum: the same id as the full-logits argmax, exactly (bf16 logits and indices
    below 2^24 are exact in fp32)."""
    if not getattr(model, "vocab_tp", False):
        return model.unembed(normed).float().argmax(-1, keepdim=True)
    local = F.linear(normed, model.lm_head)  # [B, vocab / world]
    idx = local.argmax(-1, keepdim=True)
    val = local.gather(-1, idx).float()
    offset = model.tp.plan.rank * model.lm_head.shape[0]
    pair = torch.cat([val, (idx + offset).float()], dim=-1)  # [B, 2]
    both = model.tp.all_gather_last(pair).view(pair.shape[0], -1, 2)  # [B, world, 2]
    best = both[..., 0].argmax(-1, keepdim=True)  # first max: lowest rank on a tie
    return both[..., 1].gather(-1, best).long()


def _draft_step_fast(
    model: _m.Model,
    mtp: MTP,
    hid: torch.Tensor,
    tok: torch.Tensor,
    rope_pos: torch.Tensor,
    write_rows: torch.Tensor,
    block_table: torch.Tensor,
    block_valid: torch.Tensor,
    active: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`draft_step` on production kernels (`SEED_MTP_DRAFT_FAST`, see the flag)."""
    import graph_decode  # noqa: PLC0415

    c, w = model.cfg, mtp.weights
    x = _combine(model, mtp, hid, tok)
    mixer = _attn_fast(
        model, mtp, _m.rmsnorm(x, w["in_norm"], c.eps), rope_pos, write_rows, block_table,
        block_valid, active,
    )
    x, h = graph_decode._residual_norm(model, mixer, x, w["post_norm"])
    moe_out = _moe_fast(model, mtp, h, active)
    x, normed = graph_decode._residual_norm(model, moe_out, x, mtp.norm)
    tok = greedy_token(model, normed[:, 0])
    return (normed if SEED_POST_NORM else x)[:, 0], tok


# ---------------------------------------------------------------- verify + rollback


def greedy_accept_length(draft_tokens: list[int], step_argmax: list[int]) -> int:
    """The longest prefix of `draft_tokens` that equals the target's own greedy prediction at
    that step (`step_argmax[i]` is the target's argmax after consuming the first `i` fed
    tokens, i.e. the prediction `draft_tokens[i]` has to match). Pure and batch-free so it is
    unit-testable on its own, without a model."""
    a = 0
    for d, t in zip(draft_tokens, step_argmax, strict=True):
        if d != t:
            break
        a += 1
    return a


def commit_limit(
    accept: int, step_argmax: list[int], budget: int | None, stops: Sequence[int]
) -> int:
    """Clamp one lane's greedy accept length so the round commits nothing the scheduler would
    discard.

    `step_argmax` is the lane's `k + 1` target argmaxes; the committed tokens are
    `step_argmax[: a + 1]` for accept length `a` (accepted drafts equal the target's own
    argmax, and the last one is the bonus token). The scheduler stops emitting at the first
    stop id and after `budget` tokens, so the round's persistent state (DeltaNet rollback, the
    MTP seed) must end there too: otherwise the lane's live state would be ahead of what it
    emitted, and the turn-close publish (which re-forwards only the last emitted token on top
    of the live state) would snapshot the wrong state. Returns `min(a, first stop index,
    budget - 1)`, at least 0: the round always commits at least one token, like plain decode.
    """
    limit = accept
    for s in range(accept + 1):
        if step_argmax[s] in stops:
            limit = s
            break
    if budget is not None:
        limit = min(limit, budget - 1)
    return max(limit, 0)


def verify_and_commit(
    model: _m.Model,
    mtp: MTP,
    slots: list[int],
    base_tokens: list[int],
    draft_tokens: list[list[int]],
    positions: list[int],
    budgets: Sequence[int] | None = None,
    stops: Sequence[Sequence[int]] | None = None,
) -> list[list[int]]:
    """One *wide* verify forward over all `B * (mtp.k + 1)` fed tokens at once, then greedy
    accept + rollback.

    `base_tokens`/`positions` are each slot's last committed token id and its absolute
    position (`Scheduler`'s `next_token`/`pos`, unchanged from the non-speculative contract).
    `draft_tokens[j]` is slot `j`'s `mtp.k` draft ids from `draft`. Returns each slot's newly
    committed token ids (accepted drafts plus one bonus token, length 1..`mtp.k + 1`).

    Every layer runs *once* over the whole `[B, t]` fed-token grid (`t = mtp.k + 1`) via
    `Model.decode_layer_verify`, not as `t` sequential per-token `decode_layer` steps: dense
    projections and MoE are already per-token (batch-shape-agnostic), full attention becomes
    one `attn_verify` dispatch per layer (`t` query rows per lane, causal among the new rows,
    against the shared paged KV pool), and DeltaNet becomes one `deltanet_verify` dispatch per
    layer (`t`-token `cu_seqlens`-packed prefill across all `B` slots at once, see its
    docstring). This is what the coordinator's wide-batched design asks for: dispatch count
    per verify round is `O(layers)`, not `O(layers * t)`.

    Rollback: full attention needs none (an unaccepted row is simply never read again, and its
    slot's running position only ever advances to `positions[j] + accept_len[j] + 1` -- see
    `attn_verify`'s docstring). DeltaNet's `conv`/`rec` state, written speculatively for all `t`
    tokens, is corrected per layer by `Model.deltanet_verify_rollback` once every slot's real
    accept length is known (design doc option (b): replay the accepted-length prefix from the
    pre-verify snapshot in one more wide `cu_seqlens` dispatch, not a kernel change and not `t`
    per-step state snapshots -- see that function's docstring for why).

    `budgets`/`stops` (optional, per slot): the scheduler's remaining `max_tokens` and stop-id
    set for each lane. Each lane's accept length is clamped by `commit_limit`, so a stop token
    or the token budget landing mid-accept ends the committed list (and the rolled-back state)
    exactly where the scheduler stops emitting.
    """
    b, k, t = len(slots), mtp.k, mtp.k + 1
    token_matrix = [[base_tokens[j], *draft_tokens[j]] for j in range(b)]
    dn_layers = [i for i, ty in enumerate(model.cfg.layer_types) if ty != "full_attention"]
    raw_seed = torch.cat([model.cached_hidden(slot) for slot in slots], dim=0)

    ids = torch.tensor(token_matrix, device=model.devices[0])  # [B, t]
    x = F.embedding(ids, model.embed)
    snapshots: dict[int, dict] = {}
    for i in range(len(model.layers)):
        x, snap = model.decode_layer_verify(i, x.to(model.layer_dev[i]), slots, positions, t)
        if snap is not None:
            snapshots[i] = snap
    x = x.to(model.devices[-1])
    # x[j, s] is the raw (pre-final-norm) hidden state after consuming fed tokens 0..s of slot
    # j (s + 1 real tokens), causal by construction (attn_verify/deltanet_verify never let
    # position s see position > s), matching the old step-by-step `snap_hidden[s][j]`.
    logits = model.unembed(_m.rmsnorm(x, model.final_norm, model.cfg.eps)).float()
    step_argmax = logits.argmax(-1)  # [B, t]: step_argmax[j, s] predicts fed token s + 1

    argmax_rows = step_argmax.tolist()
    accept_len = [
        commit_limit(
            greedy_accept_length(draft_tokens[j], argmax_rows[j][:k]),
            argmax_rows[j],
            None if budgets is None else budgets[j],
            () if stops is None else stops[j],
        )
        for j in range(b)
    ]
    committed = [argmax_rows[j][: accept_len[j] + 1] for j in range(b)]
    accept_len_plus1 = [a + 1 for a in accept_len]

    for i in dn_layers:
        model.deltanet_verify_rollback(i, slots, snapshots[i], accept_len_plus1)
    pos_grid = torch.tensor(positions, device=x.device)[:, None] + torch.arange(t, device=x.device)
    write_rows = torch.stack(
        [
            model._physical_rows_range(slot, pos, pos + t, x.device)
            for slot, pos in zip(slots, positions, strict=True)
        ]
    )
    previous = torch.cat([raw_seed[:, None], x[:, :-1]], dim=1)
    keep = torch.arange(t, device=x.device)[None, :] <= torch.tensor(
        accept_len, device=x.device
    )[:, None]
    cache_target_rows(model, mtp, ids.to(x.device), previous, pos_grid, write_rows, keep)
    for j, slot in enumerate(slots):
        model.cache_hidden(slot, x[j : j + 1, accept_len[j]])
    return committed

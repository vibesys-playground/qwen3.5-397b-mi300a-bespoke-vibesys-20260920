"""Opt-in HIP/CUDA graph capture for the MTP decode round (`SEED_MTP_SERVE=1` plus
`--enable-graph-capture`).

Shares `graph_decode.py`'s bucket ladder (`build_buckets`/`bucket_for`/`pad_slot_for`), its
capture backend protocol and, when served through `GraphMTPRunner`, its persistent device
block-table mirror and dirty flags, so one `--enable-graph-capture` flag turns both on together
and one `prepare()` failure model applies to both: never raises, disables itself, and logs why.

**Two graphs per bucket: draft, then verify.** `draft_round_step` is `mtp.draft`'s `k`
autoregressive MTP-layer steps over static buffers (`mtp.draft_step`, the same function the
eager path runs); it writes its argmax tokens into `token_matrix[:, 1:]`, which
`verify_round_step` then reads. The draft used to stay eager because its MoE sized a `bmm` by
host reads; `mtp._moe` now takes the fused MXFP4 kernels on an accelerator (fixed shape, no
host read), the same fix `graph_decode.moe_static` documents for the target MoE. The MTP
KV pool uses the target block tables and fixed physical write rows, so every graph address
and shape is stable while positions and block ids remain replay inputs.
`speculative_decode` replays both back to back with no host sync between them;
`verify_and_commit` (forced drafts, tests and validation) replays verify alone.

**Block reservation (TP).** Under serving (`model.scheduler_owns_blocks`), `fill` never
allocates: the scheduler reserved every active lane's blocks to `pos + k + 1` on rank 0 and
broadcast them as `EXTEND_BLOCKS` ahead of the round's own command (`Scheduler._decode_step`),
so `Model.grow_lane` is only a capacity check here, and idle/padding lanes are never touched.
Padding rows point at `block_pool.RESERVED_BLOCK` with `pos = 0`, exactly like
`GraphDecodeRunner.fill`: their masked KV write rewrites the reserved block's own bytes.

**Single segment only.** `graph_decode.plan_segments` can return more than one `Segment` under
the `--tp 1` pipeline layout; this module requires exactly one (what a TP rank always has,
since it owns one device -- see `graph_decode.py`'s own module docstring) and `prepare` below
refuses to enable itself otherwise, logging why, rather than build the cross-device hand-off a
split would need. That hand-off is not just more plumbing: accept-length and rollback depend
on the *whole* forward's own final logits, which only exist after the *last* segment runs, so
an earlier segment's DeltaNet rollback could not happen inside that segment's own graph -- it
would need a second capture bridged by explicit snapshot buffers per DeltaNet layer. Verify's
real deployment target (TP=4) never has more than one segment, so that plumbing is deferred
rather than built for a layout nothing here runs on.

**Verify: forward, accept and rollback in one graph.** Forward, greedy accept-length (clamped by
each row's stop ids and budget, `_commit_limit`), and DeltaNet rollback all run inside *one*
captured callable. This works because a captured region is just "whatever GPU ops
one Python call issues, in order" -- there is no rule against a function that runs the 60-layer
forward, computes accept-length from its own output, and then loops back over the DeltaNet
layers to roll them back, all as ordinary Python control flow with intermediate tensors kept
alive by ordinary Python references (`decode_layer_static` already threads `x` across layers
the same way). What capture forbids is a data-dependent *shape* or a host sync, neither of
which this needs: `mtp.greedy_accept_length`'s host loop is replaced here by a GPU tensor
reduction (`_accept_len`), so the accept length that feeds `cu_seqlens` into the rollback's
`fused_recurrent_prefill` call is a *value*, computed on-device, not a host int -- fixed shape,
varying contents, the same story every buffer in this module and in `graph_decode.py` already
tells. The only host round trip is the one every capture-adjacent call site keeps outside the
graph: reading back `step_argmax`/`accept_len` after a replay to build the ragged per-slot
`committed`-token lists the caller needs.

**Padding rows.** A bucket wider than the round's real batch gets padding rows exactly like
`graph_decode.Buffers`: `pad_slot_for` picks one filler lane, `buf.active` marks which rows are
real, and every persistent-state write (DeltaNet conv/rec, the cached-hidden seed for next
round's draft) is masked so a padding row reproduces the value already there rather than
writing something new; its paged-KV rows are `RESERVED_BLOCK`'s, rewritten with their own
bytes. The draft's MTP KV pool is persistent, so inactive rows preserve their existing
reserved-block bytes rather than writing a filler lane. Unlike decode, a padding row's rollback also needs a defined accept
length to build `cu_seqlens`: it is clamped to 1 (a real, already-tested case --
`accept_len_plus1 == 1` is what an ordinary row rejecting every draft produces) rather than 0,
so a padding row never exercises a zero-length-segment path nothing here can check without a
GPU. The final persistent-state write is masked on top of that regardless, so what a padding
row's rollback computes cannot matter even if that assumption is wrong.

    <python-with-torch> -m pytest seed_tests/test_graph_mtp.py -q -o addopts=
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import block_pool
import deltanet_fused
import graph_decode
import mtp as mtp_mod
import skinny_hip
import torch
import torch.nn.functional as F
from graph_decode import CaptureBackend, CudaGraphBackend, causal_conv_static, log
from model import (
    MTP_VERIFY_DN_EXACT,
    MTP_VERIFY_STEP_ROWS,
    Model,
    apply_rope,
    copy_from_host,
    delta_rule_recurrent,
    gated_rmsnorm,
    in_proj_sizes,
    mtp_verify_moe,
    rmsnorm,
    verify_attention_paged,
)

VERIFY_WIDE = __import__("os").environ.get("SEED_MTP_VERIFY_WIDE", "0") not in ("0", "", "false", "False")
"""`SEED_MTP_VERIFY_WIDE=1`: build each bucket's verify graph with `graph_verify_wide` (the
decode graph's kernels at `capacity * t` rows) instead of `verify_round_step`. Wide GEMMs and
MoE change rounding, so boot validation accepts `VERIFY_WIDE_MIN_AGREE` of rows agreeing with
the eager round rather than exact equality."""

VERIFY_WIDE_MIN_AGREE = float(__import__("os").environ.get("SEED_MTP_VERIFY_WIDE_MIN_AGREE", "0.9"))

_env = __import__("os").environ

VERIFY_WIDE_MIN_BUCKET = int(_env.get("SEED_MTP_MIN_BUCKET", "48" if VERIFY_WIDE else "1"))
"""Smallest captured MTP bucket. Under `SEED_MTP_VERIFY_WIDE` rounds of fewer lanes pad up to
it (`bucket_for`), so no batch size falls back to the eager round."""

REQUIRE_GRAPH = _env.get("SEED_MTP_REQUIRE_GRAPH", "1" if VERIFY_WIDE else "0") not in (
    "0", "", "false", "False"
)
"""Fail the boot when the captured MTP round cannot be enabled, instead of serving every
round eagerly (about 200 ms at B96). On by default with `SEED_MTP_VERIFY_WIDE`."""

TRACE = _env.get("SEED_MTP_TRACE", "0") not in ("0", "", "false", "False")
"""Diagnostic: synchronize and log after each validation stage and each draft/verify replay
(localizes an asynchronous device fault to one graph and bucket)."""

ROUND_TIMING = _env.get("SEED_MTP_ROUND_TIMING", "0") not in ("0", "", "false", "False")
"""Log per-round host fill, draft, verify and readback times every `ROUND_TIMING_EVERY` rounds
(device times from CUDA events; one extra host sync per timed round)."""

ROUND_TIMING_EVERY = int(_env.get("SEED_MTP_ROUND_TIMING_EVERY", "50"))

VERIFY_WIDE_MARGINS = __import__("os").environ.get("SEED_MTP_VERIFY_WIDE_MARGINS", "0") not in (
    "0", "", "false", "False"
)
"""Diagnostic: keep the wide verify's logits (`[max_batch * t, vocab]` fp32) so boot
validation can report the top-2 margin of any row that disagrees with the eager round."""

VALIDATE_REL_TOL = 2e-2
"""Same tolerance `graph_decode.py` validates decode buckets against; see that module."""


# ---------------------------------------------------------------- buffers


class MTPVerifyBuffers:
    """One bucket's static tensors for the verify+accept+rollback step, on the one segment's
    device (see module docstring: multi-segment is refused before this is ever built).

    `token_matrix` (column 0; columns 1.. too when drafts are forced), `pos`, `active`,
    `block_table`, `block_valid`, `write_rows`, `slot_rows`, `budget` and `stops` are written
    before every replay (`MTPVerifyRunner.fill`); the draft graph writes `token_matrix[:, 1:]`;
    `step_argmax` and `accept_len` are read after the verify graph. Row `j` is this round's `j`-th request, the same
    row-is-caller-order convention `graph_decode.Buffers` uses post-bucketing, not a lane id.
    """

    def __init__(self, model: Model, capacity: int, t: int, device: torch.device) -> None:
        self.capacity, self.t = capacity, t
        self.token_matrix = torch.zeros(capacity, t, dtype=torch.long, device=device)
        self.pos = torch.zeros(capacity, dtype=torch.long, device=device)
        self.active = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.block_table = torch.zeros(
            capacity, model.max_blocks_per_lane, dtype=torch.int32, device=device
        )
        self.block_valid = torch.zeros(capacity, dtype=torch.int32, device=device)
        self.write_rows = torch.zeros(capacity, t, dtype=torch.long, device=device)
        self.slot_rows = torch.zeros(capacity, dtype=torch.long, device=device)
        self.draft_hidden = torch.zeros(
            capacity, model.cfg.hidden, dtype=model.dtype, device=device
        )
        self.step_argmax = torch.zeros(capacity, t, dtype=torch.long, device=device)
        self.accept_len = torch.zeros(capacity, dtype=torch.long, device=device)
        # Per-row commit limits (`mtp.commit_limit`): remaining token budget and stop ids,
        # right-padded with -1 (never a token id).
        self.budget = torch.full((capacity,), t, dtype=torch.long, device=device)
        self.stops = torch.full(
            (capacity, mtp_mod.MAX_STOP_IDS), -1, dtype=torch.long, device=device
        )
        # Forced feed (`SEED_FOLD_TURN_SUFFIX` chat-suffix ids, `SEED_MTP_FORCED_DRAFTS`): row
        # `j`'s first `forced_n[j]` drafts are replaced by `forced_ids[j]` and accepted
        # unconditionally (wide verify only). Zero on every other row.
        self.forced_n = torch.zeros(capacity, dtype=torch.long, device=device)
        self.forced_ids = torch.zeros(capacity, max(t - 1, 1), dtype=torch.long, device=device)


# ---------------------------------------------------------------- static-shape verify step


def attn_verify_static(model: Model, i: int, x: torch.Tensor, buf: MTPVerifyBuffers, t: int) -> torch.Tensor:
    """`Model.attn_verify`, generalized to static buffers and padding-row safety.

    The eager version never masks a row (it is only ever called with a round's real slots);
    here a padding row's KV write is masked to reproduce whatever was already at its target
    physical rows, the same "no-op via `torch.where`" trick `graph_decode.attn_decode_static`
    uses, generalized from one new row per lane to `t`.
    """
    c, w, pool = model.cfg, model.layers[i], model.pool[i]
    b, pl = x.shape[0], model.tp.plan
    nq, nkv = pl.q.count, pl.kv.count
    q, gate = F.linear(x, w["q_proj"]).view(b, t, nq, 2 * c.head_dim).chunk(2, dim=-1)
    q = rmsnorm(q, w["q_norm"], c.eps).transpose(1, 2)
    k = rmsnorm(
        F.linear(x, w["k_proj"]).view(b, t, nkv, c.head_dim), w["k_norm"], c.eps
    ).transpose(1, 2)
    v = F.linear(x, w["v_proj"]).view(b, t, nkv, c.head_dim).transpose(1, 2)

    base = buf.pos
    pos_grid = base[:, None] + torch.arange(t, device=x.device)[None, :]
    cos, sin = model.rope_at(pos_grid.reshape(-1), x.dtype)
    cos, sin = cos.reshape(b, t, -1), sin.reshape(b, t, -1)
    q = apply_rope(q, cos[:, None], sin[:, None])
    k = apply_rope(k, cos[:, None], sin[:, None])

    new_k = k.transpose(1, 2).reshape(b, t, nkv, c.head_dim)
    new_v = v.transpose(1, 2).reshape(b, t, nkv, c.head_dim)
    write_rows = buf.write_rows.reshape(-1)
    mtp_mod.mtp_dense_moe.scatter_active_first_dim(
        pool["k"], write_rows, new_k.reshape(-1, nkv, c.head_dim).contiguous(), buf.active, t
    )
    mtp_mod.mtp_dense_moe.scatter_active_first_dim(
        pool["v"], write_rows, new_v.reshape(-1, nkv, c.head_dim).contiguous(), buf.active, t
    )

    out = verify_attention_paged(
        q, pool["k"], pool["v"], buf.block_table, buf.block_valid, base, model.block_size,
        c.head_dim**-0.5,
    )
    out = out.transpose(1, 2).reshape(b, t, -1)
    return F.linear(out * torch.sigmoid(gate.reshape(b, t, -1)), w["o_proj"])


def deltanet_verify_static(
    model: Model, i: int, x: torch.Tensor, buf: MTPVerifyBuffers, t: int
) -> tuple[torch.Tensor, dict]:
    """`Model.deltanet_verify`, generalized to static buffers, `buf.slot_rows` gather/scatter
    (see `graph_decode.deltanet_decode_static` for why a bucket needs this at all), and
    padding-row masking on the persistent-state write.

    Returns `(mixer_out, snapshot)`, consumed later in the *same* captured call by
    `deltanet_verify_rollback_static` -- no separate buffers needed for the snapshot, since it
    never has to survive past the end of this one Python call (see module docstring).
    """
    c, w, pool = model.cfg, model.layers[i], model.pool[i]
    b = x.shape[0]
    rows = buf.slot_rows
    keep = buf.active[:, None, None]
    key_dim, val_dim = c.k_heads * c.k_dim, c.v_heads * c.v_dim

    if MTP_VERIFY_STEP_ROWS:
        projected = torch.stack(
            [skinny_hip.linear(x[:, step], w["in_proj_all"]) for step in range(t)], dim=1
        )
    else:
        projected = F.linear(x, w["in_proj_all"])
    qkv, z, beta_raw, a_raw = projected.split(in_proj_sizes(c), dim=-1)
    raw_qkv = qkv.transpose(1, 2)  # [b, conv_dim, t], pre-conv: what rollback replays
    conv_state = mtp_mod.mtp_dense_moe.gather_first_dim(pool["conv"], rows)
    pre_conv = conv_state.clone()  # snapshot before causal_conv_static mutates it below
    if MTP_VERIFY_DN_EXACT and deltanet_fused.available(x.device):
        mixed = deltanet_fused.causal_conv_verify_exact(
            qkv, w["conv"], conv_state, active=buf.active
        ).transpose(1, 2)
    else:
        mixed = causal_conv_static(raw_qkv, w["conv"], conv_state, keep)
    mtp_mod.mtp_dense_moe.scatter_active_first_dim(
        pool["conv"], rows, conv_state, buf.active
    )

    q, k, v = mixed.transpose(1, 2).split([key_dim, key_dim, val_dim], dim=-1)
    q_small = q.reshape(b, t, c.k_heads, c.k_dim)
    k_small = k.reshape(b, t, c.k_heads, c.k_dim)
    v = v.reshape(b, t, c.v_heads, c.v_dim).contiguous()  # see Model.deltanet_verify's own note
    beta = beta_raw.sigmoid()
    g = -w["A_log"].exp() * F.softplus(a_raw.float() + w["dt_bias"])
    rep = c.v_heads // c.k_heads
    q = q_small.repeat_interleave(rep, dim=2)
    k = k_small.repeat_interleave(rep, dim=2)

    rec_state = mtp_mod.mtp_dense_moe.gather_first_dim(pool["rec"], rows)
    pre_rec = rec_state.clone()
    heads = c.v_heads
    if MTP_VERIFY_DN_EXACT and deltanet_fused.available(x.device):
        out = deltanet_fused.delta_rule_verify_exact(
            (q_small.contiguous(), k_small.contiguous()),
            v,
            (a_raw, beta_raw),
            rec_state,
            (w["A_log"], w["dt_bias"]),
            active=buf.active,
        ).to(x.dtype)
    elif deltanet_fused.available_prefill(x.device):
        cu = torch.arange(0, b * t + 1, t, dtype=torch.int32, device=x.device)
        out, _ = deltanet_fused.fused_recurrent_prefill(
            q.reshape(b * t, heads, c.k_dim),
            k.reshape(b * t, heads, c.k_dim),
            v.reshape(b * t, heads, c.v_dim),
            g.reshape(b * t, heads).float(),
            beta.reshape(b * t, heads),
            rec_state,
            cu,
        )
        out = out.reshape(b, t, heads, c.v_dim).to(x.dtype)
    else:
        out = delta_rule_recurrent(q, k, v, g, beta, rec_state).to(x.dtype)
    mtp_mod.mtp_dense_moe.scatter_active_first_dim(pool["rec"], rows, rec_state, buf.active)

    if MTP_VERIFY_STEP_ROWS and deltanet_fused.available(x.device):
        norm_out = torch.stack(
            [
                deltanet_fused.gated_rmsnorm(
                    out[:, step],
                    z[:, step].unflatten(-1, (c.v_heads, c.v_dim)),
                    w["dn_norm"],
                    c.eps,
                    x.dtype,
                )
                for step in range(t)
            ],
            dim=1,
        )
    else:
        norm_out = gated_rmsnorm(
            out.reshape(-1, c.v_dim), z.reshape(-1, c.v_dim), w["dn_norm"], c.eps
        ).reshape(b, t, c.v_heads, c.v_dim)
    if MTP_VERIFY_STEP_ROWS:
        mixer = torch.stack(
            [
                skinny_hip.linear(norm_out[:, step].reshape(b, -1), w["out_proj"])
                for step in range(t)
            ],
            dim=1,
        )
    else:
        mixer = F.linear(norm_out.reshape(b, t, -1), w["out_proj"])
    snapshot = {
        "pre_conv": pre_conv,
        "pre_rec": pre_rec,
        "raw_qkv": raw_qkv,
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "q_small": q_small,
        "k_small": k_small,
        "a_raw": a_raw,
        "beta_raw": beta_raw,
    }
    return mixer, snapshot


def deltanet_verify_rollback_static(
    model: Model,
    i: int,
    buf: MTPVerifyBuffers,
    accept_len_plus1: torch.Tensor,
    snapshot: dict,
    keep: torch.Tensor,
    packed_src: torch.Tensor,
    packed_cu: torch.Tensor,
) -> None:
    """`Model.deltanet_verify_rollback`, generalized to `buf.slot_rows` and GPU-tensor
    `accept_len_plus1` (a device value computed by `_accept_len` in the same captured call, not
    a host list): no `.item()`/`.tolist()` anywhere in this function, which is what lets it run
    inside the same graph as the forward pass that produced `accept_len_plus1`.

    `keep` masks the final persistent-state write so a padding row's rollback -- run on
    `accept_len_plus1` clamped to 1, see module docstring -- cannot leave a mark even if that
    clamp or the kernel's handling of a 1-token segment were somehow wrong; the mask makes this
    correct independent of what the unmasked computation produces.
    """
    c, pool = model.cfg, model.pool[i]
    rows = buf.slot_rows
    b, t = snapshot["q"].shape[0], snapshot["q"].shape[1]
    conv_k1 = pool["conv"].shape[-1]
    dev = pool["conv"].device

    full = torch.cat([snapshot["pre_conv"], snapshot["raw_qkv"]], dim=-1)
    win = torch.arange(conv_k1, device=dev)
    idx = (accept_len_plus1[:, None] + win[None, :]).clamp(max=full.shape[-1] - 1)
    idx = idx[:, None, :].expand(-1, full.shape[1], -1)
    rolled_conv = torch.gather(full, -1, idx)
    mtp_mod.mtp_dense_moe.scatter_active_first_dim(pool["conv"], rows, rolled_conv, keep)

    heads = c.v_heads
    q, k, v, g, beta = (snapshot[n] for n in ("q", "k", "v", "g", "beta"))
    rec_state = snapshot["pre_rec"].clone()
    if MTP_VERIFY_DN_EXACT and deltanet_fused.available(q.device):
        deltanet_fused.delta_rule_verify_exact(
            (snapshot["q_small"].contiguous(), snapshot["k_small"].contiguous()),
            v,
            (snapshot["a_raw"], snapshot["beta_raw"]),
            rec_state,
            (model.layers[i]["A_log"], model.layers[i]["dt_bias"]),
            active=buf.active,
            lengths=accept_len_plus1.to(torch.int32),
        )
    elif deltanet_fused.available_prefill(q.device):
        # Pack each lane's accepted prefix into a fixed B*T allocation. Boolean indexing here
        # used to allocate sum(accept_len_plus1) rows, a replay-dependent shape hidden by the
        # CPU capture shim. The fused kernel reads only [cu[j], cu[j+1]) for lane j, so tail
        # rows after cu[-1] may contain any in-bounds value; keeping the allocation fixed is
        # what makes the rollback replay-safe for every accept-length mixture.
        packed_q = mtp_mod.mtp_dense_moe.gather_first_dim(
            q.reshape(b * t, heads, c.k_dim), packed_src
        )
        packed_k = mtp_mod.mtp_dense_moe.gather_first_dim(
            k.reshape(b * t, heads, c.k_dim), packed_src
        )
        packed_v = mtp_mod.mtp_dense_moe.gather_first_dim(
            v.reshape(b * t, heads, c.v_dim), packed_src
        )
        packed_g = mtp_mod.mtp_dense_moe.gather_first_dim(
            g.reshape(b * t, heads).float(), packed_src
        )
        packed_beta = mtp_mod.mtp_dense_moe.gather_first_dim(
            beta.reshape(b * t, heads), packed_src
        )
        deltanet_fused.fused_recurrent_prefill(
            packed_q,
            packed_k,
            packed_v,
            packed_g,
            packed_beta,
            rec_state,
            packed_cu,
        )
    else:
        # CPU-test-only fallback (mirrors `Model.deltanet_verify_rollback`'s own non-fused
        # branch): `accept_len_plus1[j]` is read to the host to slice a per-row window, which
        # is fine here because `available_prefill` is only False off an accelerator or under
        # `SEED_FUSED_RECURRENT_PREFILL=0` -- neither is the configuration a real capture runs
        # under, so this branch never executes inside a graph that actually gets captured (the
        # Python-level `if` above is resolved once at capture/trace time, off `q.device`, which
        # cannot change between replays of one captured callable).
        for j in range(b):
            n = int(accept_len_plus1[j].item())
            delta_rule_recurrent(
                q[j : j + 1, :n], k[j : j + 1, :n], v[j : j + 1, :n],
                g[j : j + 1, :n], beta[j : j + 1, :n], rec_state[j : j + 1],
            )
    mtp_mod.mtp_dense_moe.scatter_active_first_dim(pool["rec"], rows, rec_state, keep)


def _fixed_prefix_pack_index(lengths: torch.Tensor, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed-size gather map for packing variable lane prefixes under graph capture."""
    batch = lengths.shape[0]
    cu = torch.zeros(batch + 1, dtype=torch.int32, device=lengths.device)
    cu[1:] = lengths.to(torch.int32).cumsum(0)
    dst = torch.arange(batch * width, device=lengths.device)
    lane = (dst[:, None] >= cu[1:][None, :]).sum(dim=1).clamp(max=batch - 1)
    local = (dst - torch.gather(cu, 0, lane).to(dst.dtype)).clamp(min=0, max=width - 1)
    return lane * width + local, cu


def _accept_len(draft_tokens: torch.Tensor, step_argmax: torch.Tensor) -> torch.Tensor:
    """GPU-only counterpart of `mtp.greedy_accept_length`, batched over rows.

    `draft_tokens` is `[B, k]`, `step_argmax[:, :k]` is the target's own prediction at each
    draft position; the longest prefix where they agree is the accept length, computed as a
    tensor reduction (no Python loop, no host read) so it stays inside a captured region. A
    mismatch at position `s` poisons every later position too, via the cumulative "seen a
    mismatch yet" flag -- the tensor equivalent of `greedy_accept_length`'s early `break`.
    """
    matches = draft_tokens == step_argmax[:, : draft_tokens.shape[1]]
    mismatch_seen = (~matches).cumsum(dim=-1) > 0
    return (~mismatch_seen).sum(dim=-1)


def _commit_limit(
    accept: torch.Tensor, step_argmax: torch.Tensor, budget: torch.Tensor, stops: torch.Tensor
) -> torch.Tensor:
    """`mtp.commit_limit`, batched over rows as tensor ops (no host read): the accept length
    clamped to the first stop id among the committed tokens `step_argmax[:, : a + 1]` and to
    `budget - 1`. `stops` is `[B, S]`, padded with -1."""
    is_stop = (step_argmax[:, :, None] == stops[:, None, :]).any(-1)  # [B, t]
    first_stop = (is_stop.cumsum(dim=-1) == 0).sum(dim=-1)  # t when there is none
    limit = torch.minimum(torch.minimum(accept, first_stop), budget - 1)
    return limit.clamp(min=0)


def draft_round_step(
    model: Model, mtp: mtp_mod.MTP, buf: MTPVerifyBuffers, t: int
) -> Callable[[], None]:
    """The captured draft: `mtp.draft`'s `t - 1` steps over static buffers, writing each
    step's argmax into `buf.token_matrix[:, 1 + step]`. Seeded from `model.hidden_scratch` at
    each row's lane and the row's base token (`token_matrix[:, 0]`)."""

    def step() -> None:
        mtp_mod.mtp_dense_moe.gather_rows(
            model.hidden_scratch, buf.slot_rows, buf.draft_hidden
        )
        hid = mtp_mod.draft_seed(model, buf.draft_hidden)
        tok = buf.token_matrix[:, :1]
        for s in range(t - 1):
            hid, tok = mtp_mod.draft_step(
                model,
                mtp,
                hid,
                tok,
                buf.pos + s,
                buf.write_rows[:, s],
                buf.block_table,
                buf.block_valid,
                buf.active,
            )
            buf.token_matrix[:, 1 + s].copy_(tok[:, 0])

    return step


build_draft_step = draft_round_step
"""`MTP_INTERFACE.md`'s draft entry point. `mtp.draft_step` takes the production-kernel path
(`mtp.DRAFT_FAST`, `SEED_MTP_DRAFT_FAST`, default on) whenever its inputs are on an
accelerator; capture bakes in whichever path was selected at capture time."""


def verify_round_step(
    model: Model, mtp: mtp_mod.MTP, buf: MTPVerifyBuffers, t: int
) -> Callable[[], None]:
    """The callable that gets captured: one wide forward, accept-length, and DeltaNet
    rollback, over every DeltaNet layer -- see module docstring for why all three fit in one
    captured region with no snapshot buffers of their own.
    """
    dn_layers = [i for i, ty in enumerate(model.cfg.layer_types) if ty != "full_attention"]

    def step() -> None:
        x = mtp.embedding_scratch[: buf.capacity * t].view(buf.capacity, t, model.cfg.hidden)
        mtp_mod.mtp_dense_moe.gather_embeddings(model.embed, buf.token_matrix, x)
        snapshots: dict[int, dict] = {}
        for i in range(len(model.layers)):
            w, c = model.layers[i], model.cfg
            h = rmsnorm(x, w["in_norm"], c.eps)
            if c.layer_types[i] == "full_attention":
                mixer, snap = attn_verify_static(model, i, h, buf, t), None
            else:
                mixer, snap = deltanet_verify_static(model, i, h, buf, t)
                snapshots[i] = snap
            x = x + model.tp.all_reduce(mixer)
            h_mid = rmsnorm(x, w["post_norm"], c.eps)
            moe_out = mtp_verify_moe(model, i, h_mid)
            x = x + model.tp.all_reduce(moe_out)

        logits = model.unembed(rmsnorm(x, model.final_norm, model.cfg.eps))
        step_argmax = logits.argmax(-1)  # [capacity, t] long
        buf.step_argmax.copy_(step_argmax)
        accept_len = _commit_limit(
            _accept_len(buf.token_matrix[:, 1:], step_argmax), step_argmax, buf.budget, buf.stops
        )
        buf.accept_len.copy_(accept_len)
        accept_len_plus1 = torch.where(
            buf.active, accept_len + 1, torch.ones_like(accept_len)
        )  # padding rows: clamp to 1, a real and already-tested value -- see module docstring
        packed_src, packed_cu = _fixed_prefix_pack_index(accept_len_plus1, t)

        for i in dn_layers:
            deltanet_verify_rollback_static(
                model,
                i,
                buf,
                accept_len_plus1,
                snapshots[i],
                buf.active,
                packed_src,
                packed_cu,
            )

        pos_grid = buf.pos[:, None] + torch.arange(t, device=x.device)[None, :]
        previous = torch.cat([buf.draft_hidden[:, None], x[:, :-1]], dim=1)
        accepted = torch.arange(t, device=x.device)[None, :] <= accept_len[:, None]
        mtp_mod.cache_target_rows(
            model,
            mtp,
            buf.token_matrix,
            previous,
            pos_grid,
            buf.write_rows,
            buf.active[:, None] & accepted,
            embedding_out=mtp.embedding_scratch[: buf.capacity * t].view(
                buf.capacity, t, model.cfg.hidden
            ),
        )
        mtp_mod.mtp_dense_moe.scatter_time_active_rows(
            model.hidden_scratch, buf.slot_rows, x, accept_len, buf.active
        )

    return step


# ---------------------------------------------------------------- the runner


@dataclass
class BucketVerifyGraph:
    capacity: int
    buf: MTPVerifyBuffers
    draft: Callable[[], None]
    replay: Callable[[], None]


def _limits_rows(
    capacity: int,
    n: int,
    t: int,
    budgets: Sequence[int] | None,
    stops: Sequence[Sequence[int]] | None,
) -> tuple[list[int], list[list[int]]]:
    """Host rows for `MTPVerifyBuffers.budget`/`stops`: a real row's own limits (no budget
    means `t`, i.e. no clamp), padding rows `t` and no stop ids."""
    width = mtp_mod.MAX_STOP_IDS
    row_budget = [t] * capacity
    row_stops = [[-1] * width for _ in range(capacity)]
    for j in range(n):
        if budgets is not None:
            row_budget[j] = budgets[j]
        if stops is not None:
            ids = list(stops[j])
            if len(ids) > width:
                raise ValueError(f"lane stop set of {len(ids)} ids exceeds MAX_STOP_IDS={width}")
            row_stops[j][: len(ids)] = ids
    return row_budget, row_stops


class MTPVerifyRunner:
    """Captures and replays the MTP decode round (draft graph + verify graph per bucket),
    bucketed the same way `graph_decode.GraphDecodeRunner` buckets ordinary decode.

    Standalone (tests), it owns its own device block-table mirror and dirty flags. Served
    through `GraphMTPRunner`, it shares the decode runner's (`lane_table`/`dirty`), so every
    host-table change the decode runner tracks (`begin`, `attach_blocks`, `extend_blocks`, a
    prefill, an eager step) also resyncs this runner's view, and there is one mirror per rank.
    """

    def __init__(
        self,
        model: Model,
        mtp: mtp_mod.MTP,
        backend: CaptureBackend | None = None,
        lane_table: graph_decode.LaneBlockTables | None = None,
        dirty: list[bool] | None = None,
    ) -> None:
        self.model, self.mtp = model, mtp
        self.backend: CaptureBackend = backend if backend is not None else CudaGraphBackend()
        self.t = mtp.k + 1
        self.buckets = [
            b for b in graph_decode.build_buckets(model.max_batch) if b >= VERIFY_WIDE_MIN_BUCKET
        ] or [model.max_batch]
        self.eager_rounds = 0  # rounds served by the eager fallback since boot
        self._timing: list[list[float]] = []
        self.graphs: dict[int, BucketVerifyGraph] = {}
        self.enabled = False
        self.lane_table = (
            lane_table
            if lane_table is not None
            else graph_decode.LaneBlockTables(model, model.devices[-1])
        )
        self._dirty = dirty if dirty is not None else [True] * model.max_batch
        self.rowbufs: dict[int, graph_decode.Buffers] = {}
        self._wide_scratch: list[torch.Tensor] | None = None
        self._wide_logits: torch.Tensor | None = None

    def _verify_step(self, buf: MTPVerifyBuffers, capacity: int) -> Callable[[], None]:
        if not VERIFY_WIDE:
            return verify_round_step(self.model, self.mtp, buf, self.t)
        import graph_verify_wide  # noqa: PLC0415 -- it imports this module

        model = self.model
        if self._wide_scratch is None:
            rows = model.max_batch * self.t
            self._wide_scratch = [
                torch.empty(rows, model.cfg.hidden, dtype=model.dtype, device=d)
                for d in model.layer_dev
            ]
        rowbuf = graph_verify_wide.row_buffers(model, capacity * self.t)
        self.rowbufs[capacity] = rowbuf
        if self._wide_logits is None and VERIFY_WIDE_MARGINS:
            self._wide_logits = torch.zeros(
                model.max_batch * self.t, model.cfg.vocab, dtype=torch.float32,
                device=model.devices[-1],
            )
        logits_out = (
            self._wide_logits[: capacity * self.t] if self._wide_logits is not None else None
        )
        return graph_verify_wide.build_verify_step(
            model, self.mtp, buf, rowbuf, self.t, self._wide_scratch, logits_out
        )

    def _mismatch(self, capacity: int, j: int, got: list[int], want: list[int]) -> str:
        """Classify one disagreeing row: `accept-length` when one committed list is a prefix
        of the other (the drafts differed, every committed token is still the target's), else
        `token` with the wide verify's top-2 logit margin at the first differing column."""
        m = min(len(got), len(want))
        c = next((i for i in range(m) if got[i] != want[i]), None)
        if c is None:
            return f"row {j}: accept-length {len(got)} vs {len(want)}"
        if self._wide_logits is None:
            return f"row {j} col {c}: token {got[c]} vs {want[c]}"
        row = self._wide_logits[j * self.t + c]
        top = row.topk(2)
        eager_gap = float(row[top.indices[0]] - row[want[c]])
        return (
            f"row {j} col {c}: token {got[c]} vs {want[c]}, wide top-2 margin "
            f"{float(top.values[0] - top.values[1]):.4f}, wide logit gap to eager token "
            f"{eager_gap:.4f}, |top1| {float(top.values[0].abs()):.2f}"
        )

    def begin(self, slot: int) -> None:
        """Forwards to the model, then marks `slot`'s device block-table mirror dirty, like
        `GraphDecodeRunner.begin` (same two effects, same order)."""
        self.model.begin(slot)
        self._dirty[slot] = True

    def supported(self) -> bool:
        """Whether this deployment's layer layout is one segment (see module docstring)."""
        return len(graph_decode.plan_segments(self.model.layer_dev)) == 1

    def prepare(self) -> bool:
        """Capture and validate every bucket. Never raises; mirrors `GraphDecodeRunner.
        prepare`'s contract (agree-across-ranks, warm before capture, validate after).
        Boot-only: warmup and validation grow and release lanes with the model's own
        allocator, before the scheduler owns blocks."""
        if not self.supported():
            log("MTP capture needs a single-segment layout (TP owns one device); staying eager")
            self.enabled = False
            return self.agree_or_default(False)
        try:
            self.graphs = {}
            device = self.model.devices[-1]
            for capacity in self.buckets:
                t0 = time.perf_counter()
                buf = MTPVerifyBuffers(self.model, capacity, self.t, device)
                draft = draft_round_step(self.model, self.mtp, buf, self.t)
                step = self._verify_step(buf, capacity)
                self._warm(buf, draft, step, capacity)
                try:
                    draft_replay = self.backend.capture(draft, device)
                except Exception as exc:
                    raise RuntimeError(
                        f"MTP draft bucket {capacity} capture failed: {exc!r}"
                    ) from exc
                try:
                    replay = self.backend.capture(step, device)
                except Exception as exc:
                    raise RuntimeError(
                        f"MTP verify bucket {capacity} capture failed: {exc!r}"
                    ) from exc
                self.graphs[capacity] = BucketVerifyGraph(capacity, buf, draft_replay, replay)
                log(f"captured MTP draft+verify bucket {capacity} (t={self.t}) in "
                    f"{(time.perf_counter() - t0) * 1e3:.0f} ms")
            self.enabled = True
        except Exception as exc:
            self.enabled = False
            log(f"MTP capture failed, MTP rounds stay eager: {exc!r}")
        ok = self.agree_or_default(self.enabled)
        if ok:
            ok = self.agree_or_default(self._validate())
        self._reset_lanes()
        if not ok:
            log("ERROR: captured MTP round is OFF; every MTP round would run eagerly")
            if REQUIRE_GRAPH:
                raise RuntimeError(
                    "SEED_MTP_REQUIRE_GRAPH: captured MTP round failed capture or validation"
                )
        return ok

    def agree_or_default(self, ok: bool) -> bool:
        """Reduce `ok` across the TP group the way `GraphDecodeRunner.agree` does (identity at
        world 1), so the ranks turn the captured round on together or not at all."""
        plan = self.model.tp.plan
        votes = torch.tensor([1.0 if ok else 0.0], device=self.model.tp.device)
        self.model.tp.all_reduce(votes)
        ok = int(votes.item()) == plan.world
        self.enabled = ok
        return ok

    def _reset_lanes(self) -> None:
        """Release every block warmup/validation grew and reset every lane (boot-only, like
        `GraphDecodeRunner.reset_slots`): `tp_driver.pool_handshake` requires a fully free
        allocator afterwards."""
        if self.model.scheduler_owns_blocks:
            raise RuntimeError("MTP capture is boot-only: the scheduler owns lane blocks now")
        for slot in range(self.model.max_batch):
            self.model._release_lane_blocks(slot)
            self.begin(slot)
        self.model.bind(0)

    def _warm(
        self,
        buf: MTPVerifyBuffers,
        draft: Callable[[], None],
        step: Callable[[], None],
        capacity: int,
    ) -> None:
        self.fill(buf, list(range(capacity)), [0] * capacity, None, [0] * capacity)
        for _ in range(graph_decode.WARMUP_STEPS):
            draft()
            step()
        self._reset_lanes()

    def fill(
        self,
        buf: MTPVerifyBuffers,
        slots: list[int],
        base_tokens: list[int],
        draft_tokens: list[list[int]] | None,
        positions: list[int],
        budgets: Sequence[int] | None = None,
        stops: Sequence[Sequence[int]] | None = None,
        forced: Sequence[Sequence[int]] | None = None,
    ) -> None:
        """Copy one round's inputs into `buf`, padding to `buf.capacity` rows -- the
        `MTPVerifyBuffers` counterpart of `GraphDecodeRunner.fill` (row-is-caller-order,
        `pad_slot_for` filler lane, incremental device block-table mirror).

        Only this round's lanes are sized, to `pos + t` tokens, through `Model.grow_lane`: a
        capacity check once the scheduler owns blocks (it reserved them on rank 0 and
        broadcast `EXTEND_BLOCKS` before this round's command), an allocation only at boot.
        Idle lanes are never grown, so no rank ever allocates a block id the others do not
        hold. Padding rows get `RESERVED_BLOCK` and `pos = 0` (see module docstring).

        `draft_tokens` is None for a full round (the draft graph fills columns 1..).
        """
        model = self.model
        for slot, pos in zip(slots, positions, strict=True):
            table = model.block_tables[slot]
            before = len(table.blocks)
            model.grow_lane(slot, pos + self.t)
            if self._dirty[slot] or len(table.blocks) != before:
                self.lane_table.sync_lane(slot, table.blocks)
                self._dirty[slot] = False

        n = len(slots)
        pad = graph_decode.pad_slot_for(slots, model.max_batch)
        row_slots = list(slots) + [pad] * (buf.capacity - n)
        row_pos = [0] * buf.capacity
        row_active = [False] * buf.capacity
        row_tokens = [[0] * self.t for _ in range(buf.capacity)]
        for j, (base, pos) in enumerate(zip(base_tokens, positions, strict=True)):
            drafts = draft_tokens[j] if draft_tokens is not None else [0] * (self.t - 1)
            row_tokens[j] = [base, *drafts]
            row_pos[j], row_active[j] = pos, True
        row_budget, row_stops = _limits_rows(buf.capacity, n, self.t, budgets, stops)

        copy_from_host(buf.token_matrix, torch.tensor(row_tokens, dtype=torch.long))
        copy_from_host(buf.pos, torch.tensor(row_pos, dtype=torch.long))
        copy_from_host(buf.active, torch.tensor(row_active, dtype=torch.bool))
        copy_from_host(buf.slot_rows, torch.tensor(row_slots, dtype=torch.long))
        copy_from_host(buf.budget, torch.tensor(row_budget, dtype=torch.long))
        copy_from_host(buf.stops, torch.tensor(row_stops, dtype=torch.long))
        buf.draft_hidden.copy_(model.hidden_scratch[buf.slot_rows])

        buf.block_table.copy_(
            torch.where(
                buf.active[:, None],
                self.lane_table.table[buf.slot_rows],
                block_pool.RESERVED_BLOCK,
            )
        )
        buf.block_valid.copy_(
            ((buf.pos + self.t + model.block_size - 1) // model.block_size).to(torch.int32)
        )
        pos_grid = buf.pos[:, None] + torch.arange(self.t, device=buf.pos.device)[None, :]
        block_idx = (pos_grid // model.block_size).long()
        block_id = torch.gather(buf.block_table.long(), 1, block_idx)
        buf.write_rows.copy_(block_id * model.block_size + pos_grid % model.block_size)
        k = self.t - 1
        f_n = [0] * buf.capacity
        f_ids = [[0] * max(k, 1) for _ in range(buf.capacity)]
        if forced is not None:
            for j, ids in enumerate(forced):
                ids = list(ids)[:k]
                f_n[j] = len(ids)
                f_ids[j][: len(ids)] = ids
        copy_from_host(buf.forced_n, torch.tensor(f_n, dtype=torch.long))
        copy_from_host(buf.forced_ids, torch.tensor(f_ids, dtype=torch.long))
        if buf.capacity in self.rowbufs:
            import graph_verify_wide  # noqa: PLC0415

            graph_verify_wide.fill_rows(model, buf, self.rowbufs[buf.capacity], self.t)

    def replayable(self, slots: Sequence[int], stops: Sequence[Sequence[int]] | None) -> bool:
        """Whether a captured bucket can serve this round. Reads only broadcast inputs, so
        every rank decides the same way."""
        b = self.model.max_batch
        return (
            self.enabled
            and 1 <= len(slots) <= b
            and len(set(slots)) == len(slots)
            and (stops is None or all(len(s) <= mtp_mod.MAX_STOP_IDS for s in stops))
        )

    def _committed(self, buf: MTPVerifyBuffers, n: int) -> list[list[int]]:
        """Read the round back: one host sync for the whole batch. Committed tokens are
        `step_argmax[j, : accept + 1]` (accepted drafts equal the target's own argmax)."""
        rows = torch.cat([buf.step_argmax[:n], buf.accept_len[:n, None]], dim=1).tolist()
        return [row[: row[-1] + 1] for row in rows]

    def speculative_decode(
        self,
        slots: list[int],
        tokens: list[int],
        positions: list[int],
        budgets: Sequence[int] | None = None,
        stops: Sequence[Sequence[int]] | None = None,
        forced: Sequence[Sequence[int]] | None = None,
    ) -> list[list[int]]:
        """`Model.speculative_decode`'s contract: draft and verify graphs back to back.
        `forced` (wide verify only): per-lane known next inputs, accepted unconditionally."""
        if forced is not None and any(forced) and not (VERIFY_WIDE and self.enabled):
            raise RuntimeError("forced MTP drafts need the captured wide verify")
        if not self.replayable(slots, stops):
            self.eager_rounds += 1
            if self.eager_rounds == 1 or self.eager_rounds % 100 == 0:
                log(f"WARNING: MTP round {self.eager_rounds} served eagerly "
                    f"(captured round {'on' if self.enabled else 'OFF'})")
            return self.model.speculative_decode(slots, tokens, positions, budgets, stops)
        graph = self.graphs[graph_decode.bucket_for(len(slots), self.buckets)]
        if not (ROUND_TIMING or TRACE):
            self.fill(graph.buf, slots, tokens, None, positions, budgets, stops, forced)
            graph.draft()
            graph.replay()
            return self._committed(graph.buf, len(slots))
        return self._timed_round(graph, slots, tokens, positions, budgets, stops, forced)

    def _timed_round(self, graph, slots, tokens, positions, budgets, stops, forced):  # noqa: ANN001, ANN202, PLR0913
        """`speculative_decode` with CUDA-event timing (`SEED_MTP_ROUND_TIMING`) and per-graph
        synchronization (`SEED_MTP_TRACE`)."""
        t0 = time.perf_counter()
        self.fill(graph.buf, slots, tokens, None, positions, budgets, stops, forced)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        t1 = time.perf_counter()
        ev[0].record()
        graph.draft()
        if TRACE:
            torch.cuda.synchronize()
            log(f"trace: draft bucket {graph.capacity} ok")
        ev[1].record()
        graph.replay()
        ev[2].record()
        if TRACE:
            torch.cuda.synchronize()
            log(f"trace: verify bucket {graph.capacity} ok")
        t2 = time.perf_counter()
        out = self._committed(graph.buf, len(slots))
        t3 = time.perf_counter()
        committed = sum(len(c) for c in out)
        self._timing.append([
            (t1 - t0) * 1e3, ev[0].elapsed_time(ev[1]), ev[1].elapsed_time(ev[2]),
            (t3 - t0) * 1e3, (t2 - t1) * 1e3, len(slots), committed,
        ])
        if len(self._timing) >= ROUND_TIMING_EVERY:
            cols = list(zip(*self._timing, strict=True))
            mean = [sum(c) / len(c) for c in cols]
            lanes = sum(cols[5])
            log(
                f"mtp-round n={len(self._timing)} fill_ms={mean[0]:.2f} draft_ms={mean[1]:.2f} "
                f"verify_ms={mean[2]:.2f} host_launch_ms={mean[4]:.2f} round_wall_ms={mean[3]:.2f} "
                f"lanes={mean[5]:.1f} tok_per_lane={sum(cols[6]) / max(lanes, 1):.3f} "
                f"eager_rounds={self.eager_rounds}"
            )
            self._timing = []
        return out

    def verify_and_commit(
        self,
        slots: list[int],
        base_tokens: list[int],
        draft_tokens: list[list[int]],
        positions: list[int],
        budgets: Sequence[int] | None = None,
        stops: Sequence[Sequence[int]] | None = None,
    ) -> list[list[int]]:
        """`mtp.verify_and_commit`'s contract (forced drafts), served from the verify graph."""
        if not self.replayable(slots, stops):
            return mtp_mod.verify_and_commit(
                self.model, self.mtp, slots, base_tokens, draft_tokens, positions, budgets, stops
            )
        graph = self.graphs[graph_decode.bucket_for(len(slots), self.buckets)]
        self.fill(graph.buf, slots, base_tokens, draft_tokens, positions, budgets, stops)
        graph.replay()
        return self._committed(graph.buf, len(slots))

    def _validate(self) -> bool:
        """Check every bucket's full round (draft graph + verify graph) against the eager
        `Model.speculative_decode` on one synthetic round per bucket at real weights, with a
        stop id and a budget that clamp some rows mid-accept.

        Both sides start from the same freshly reset lanes; the eager call's DeltaNet state,
        MTP seed and block tables are restored before the replay (KV rows are rewritten at the
        same positions by both, so they need no restore).
        """
        model = self.model
        wide_rows = [0, 0]
        try:
            for capacity in self.buckets:
                slots = list(range(capacity))
                base_tokens = [(7 * i + 3) % model.cfg.vocab for i in range(capacity)]
                # Capture warmup uses position zero. Validate beyond the first KV block so a
                # graph that accidentally froze warmup's host-computed gather width fails
                # closed before serving real contexts.
                positions = [model.block_size + 1 + i % 3 for i in range(capacity)]
                budgets = [1 + i % self.t for i in range(capacity)]
                stops = [[(5 * i) % model.cfg.vocab] for i in range(capacity)]
                self._reset_lanes()
                # The serving scheduler reserves these rows before either eager or captured
                # MTP. Validation calls the model directly, so reproduce that reservation
                # before eager draft addresses its persistent cache.
                for slot, pos in zip(slots, positions, strict=True):
                    model.grow_lane(slot, pos + self.t)
                    self.lane_table.sync_lane(slot, model.block_tables[slot].blocks)
                    self._dirty[slot] = False
                # A deterministic seed, identical on every rank (a random one would make the
                # ranks' drafts, and so their verdicts, diverge).
                grid = torch.arange(model.hidden_scratch.numel(), device=model.hidden_scratch.device)
                model.hidden_scratch.copy_(
                    (grid.reshape(model.hidden_scratch.shape) * 0.37).sin().to(model.hidden_scratch.dtype)
                )
                seed = model.hidden_scratch.clone()
                snap_conv = [pool["conv"].clone() for pool in model.pool if "conv" in pool]
                snap_rec = [pool["rec"].clone() for pool in model.pool if "rec" in pool]
                if TRACE:
                    log(f"trace: validate bucket {capacity}: eager round")
                want = model.speculative_decode(slots, base_tokens, positions, budgets, stops)
                if TRACE:
                    torch.cuda.synchronize()
                    log(f"trace: validate bucket {capacity}: eager ok")
                for pool, before in zip(
                    (p for p in model.pool if "conv" in p), snap_conv, strict=True
                ):
                    pool["conv"].copy_(before)
                for pool, before in zip(
                    (p for p in model.pool if "rec" in p), snap_rec, strict=True
                ):
                    pool["rec"].copy_(before)
                model.hidden_scratch.copy_(seed)
                got = self.speculative_decode(slots, base_tokens, positions, budgets, stops)
                if VERIFY_WIDE:
                    # Aggregate over every bucket: a near-tie flip in a 4-row bucket is 25%
                    # of that bucket, while a real indexing bug disagrees on most rows.
                    same = [g == w for g, w in zip(got, want, strict=True)]
                    wide_rows[0] += sum(same)
                    wide_rows[1] += len(same)
                    bad = [
                        self._mismatch(capacity, j, got[j], want[j])
                        for j, ok in enumerate(same)
                        if not ok
                    ][:4]
                    log(f"MTP wide bucket {capacity}: {sum(same)}/{len(same)} rows match eager"
                        + (f"; {bad}" if bad else ""))
                    continue
                if got != want:
                    log(f"MTP bucket {capacity} disagrees with eager speculative_decode; "
                        "MTP rounds stay eager")
                    return False
        except Exception as exc:
            log(f"MTP validation could not run, MTP rounds stay eager: {exc!r}")
            return False
        if VERIFY_WIDE:
            agree = wide_rows[0] / max(wide_rows[1], 1)
            log(f"MTP wide: {wide_rows[0]}/{wide_rows[1]} rows match eager (need "
                f"{VERIFY_WIDE_MIN_AGREE:.2f})")
            return agree >= VERIFY_WIDE_MIN_AGREE
        log(f"MTP: every bucket matches eager speculative_decode (t={self.t})")
        return True


class GraphMTPRunner(graph_decode.GraphDecodeRunner):
    """`GraphDecodeRunner` plus the captured MTP round (`server.build_runner` picks this when
    the model loaded an MTP head). Decode and MTP share one device block-table mirror and one
    set of dirty flags (see `MTPVerifyRunner`)."""

    def __init__(self, model: Model, backend: CaptureBackend | None = None) -> None:
        super().__init__(model, backend)
        if model.mtp is None:
            raise ValueError("GraphMTPRunner needs a model with an MTP head (SEED_MTP_SERVE=1)")
        lane_table = self.lane_tables.get(model.devices[-1])
        self.mtp_runner = MTPVerifyRunner(
            model, model.mtp, self.backend, lane_table=lane_table, dirty=self._dirty
        )

    def prepare(self) -> bool:
        decode_ok = super().prepare()
        mtp_ok = self.mtp_runner.prepare()
        log(f"MTP round capture {'enabled' if mtp_ok else 'off'} (k={self.model.mtp.k})")
        return decode_ok

    def decode(
        self, slots: Sequence[int], tokens: Sequence[int], positions: Sequence[int]
    ) -> torch.Tensor:
        """Keep MTP's hidden seed current across a temporarily ineligible ordinary step.

        The captured decode graph does not expose its final hidden states. MTP batches only
        take this path while speculation is temporarily ineligible, so use the eager model
        call whose `_decode_ids` refreshes `hidden_scratch`, then mark graph mirrors dirty.
        """
        slots, tokens, positions = list(slots), list(tokens), list(positions)
        self.decode_path = "mtp-fallback-eager"
        self._touch(slots)
        return self.model.decode(slots, tokens, positions)

    def speculative_decode(
        self,
        slots: Sequence[int],
        tokens: Sequence[int],
        positions: Sequence[int],
        budgets: Sequence[int] | None = None,
        stops: Sequence[Sequence[int]] | None = None,
        forced: Sequence[Sequence[int]] | None = None,
    ) -> list[list[int]]:
        slots, tokens, positions = list(slots), list(tokens), list(positions)
        eager = not self.mtp_runner.replayable(slots, stops)
        if eager:
            self._touch(slots)
        out = self.mtp_runner.speculative_decode(slots, tokens, positions, budgets, stops, forced)
        # `SEED_STEP_TIMING`'s `path=` carries the boot-cumulative eager-round count, so a
        # benchmark log shows any eager fallback without a separate channel.
        self.decode_path = f"mtp-eager{self.mtp_runner.eager_rounds}" if eager else (
            "mtp-graph" if not self.mtp_runner.eager_rounds
            else f"mtp-graph-eager{self.mtp_runner.eager_rounds}"
        )
        return out

    def forced_drafts(self) -> bool:
        """Whether rounds may carry forced feed (`scheduler`'s `SEED_MTP_FORCED_DRAFTS`)."""
        return VERIFY_WIDE and self.mtp_runner.enabled

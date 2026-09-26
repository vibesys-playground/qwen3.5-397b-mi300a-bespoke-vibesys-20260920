"""MTP verify on the decode graph's own kernels, at `B * t` rows (round 15, W6).

`graph_mtp.verify_round_step` runs its own wide forward (plain `F.linear`, per-step routed
MoE, unfused norms): 200.8 ms at B96, t=3. This step instead treats every `(lane, column)` of
the round as one decode row and reuses what the production decode graph replays:
`attn_decode_static` (fused RoPE/KV write, split-K paged attention at each row's own
position), `model.moe` at `B * t` rows, and the fused AR+add+RMSNorm. Only DeltaNet needs the
time axis, and runs `deltanet_fused.delta_rule_verify_exact` over each lane's `t` steps.

**Rows.** Row `r = j * t + c` is lane-row `j`'s column `c` at position `pos[j] + c`. The KV of
all `t` columns is written by one kernel before the attention kernel reads it, so column `c`
attends to positions `<= pos[j] + c`: causal within the round by the decode kernel's own `pos`
bound.

**DeltaNet without a snapshot copy.** The forward pass reads each lane's recurrent state in
place and never stores it (`active` all false, no per-step materialization). After the accept
length is known, the rollback replays the same kernel with `lengths = accept + 1` and stores
the state; the conv window is gathered from `[old window, raw inputs]` at that length. State
traffic is one read in the forward plus one read and one write in the rollback.

**Numerics.** Wide MoE and GEMMs are not bit-identical to sequential decode (different row
counts change tiling and reduction order), so this is a different-numbers change under the
campaign policy: it needs the teacher agreement test as well as the gate.
"""

from __future__ import annotations

from collections.abc import Callable

import deltanet_fused
import graph_decode
import graph_prefill
import mtp as mtp_mod
import skinny_hip
import torch
import torch.nn.functional as F
from graph_mtp import MTPVerifyBuffers, _accept_len, _commit_limit
from model import Model, copy_from_host, in_proj_sizes, rmsnorm


def row_buffers(model: Model, rows: int) -> graph_decode.Buffers:
    """Row-level decode buffers for `rows = B * t` (one segment: TP owns one device)."""
    segs = graph_decode.plan_segments(model.layer_dev)
    if len(segs) != 1:
        raise RuntimeError("wide verify needs a single-segment layout")
    return graph_decode.Buffers(model, segs[0], rows)


def fill_rows(model: Model, buf: MTPVerifyBuffers, rowbuf: graph_decode.Buffers, t: int) -> None:
    """Derive the row-level decode inputs from the lane-level verify buffer, on the device.
    Call after `MTPVerifyRunner.fill` (or an equivalent) has written `buf`."""
    bs = model.block_size
    col = torch.arange(t, device=buf.pos.device)
    rowbuf.pos.copy_((buf.pos[:, None] + col[None, :]).reshape(-1))
    rowbuf.active.copy_(buf.active.repeat_interleave(t))
    rowbuf.slot_rows.copy_(buf.slot_rows.repeat_interleave(t))
    rowbuf.block_table.copy_(buf.block_table.repeat_interleave(t, dim=0))
    rowbuf.block_valid.copy_(((rowbuf.pos + bs) // bs).to(torch.int32))
    rowbuf.write_rows.copy_(buf.write_rows.reshape(-1))


def _dn_forward(
    model: Model, i: int, h: torch.Tensor, buf: MTPVerifyBuffers, t: int, no_store: torch.Tensor
) -> tuple[torch.Tensor, dict]:
    c, w, pool = model.cfg, model.layers[i], model.pool[i]
    rows = h.shape[0]
    b = rows // t
    proj = skinny_hip.linear(h.reshape(rows, -1), w["in_proj_all"]).view(b, t, -1)
    qkv, z, beta_raw, a_raw = proj.split(in_proj_sizes(c), dim=-1)
    pre_conv = mtp_mod.mtp_dense_moe.gather_first_dim(pool["conv"], buf.slot_rows)
    conv_state = pre_conv.clone()
    mixed = deltanet_fused.causal_conv_verify_exact(qkv, w["conv"], conv_state, active=buf.active)
    key_dim = c.k_heads * c.k_dim
    q = mixed[..., :key_dim].unflatten(-1, (c.k_heads, c.k_dim))
    k = mixed[..., key_dim : 2 * key_dim].unflatten(-1, (c.k_heads, c.k_dim))
    v = mixed[..., 2 * key_dim :].unflatten(-1, (c.v_heads, c.v_dim))
    out = deltanet_fused.delta_rule_verify_exact(
        (q, k),
        v,
        (a_raw, beta_raw),
        pool["rec"],
        (w["A_log"], w["dt_bias"]),
        active=no_store,
        lanes=buf.slot_rows,
        materialize_steps=False,
    )
    gate = z.reshape(rows, c.v_heads, c.v_dim)
    normed = deltanet_fused.gated_rmsnorm(
        out.reshape(rows, c.v_heads, c.v_dim), gate, w["dn_norm"], c.eps, h.dtype
    )
    mixer = skinny_hip.linear(normed.reshape(rows, 1, -1), w["out_proj"])
    snap = {"pre_conv": pre_conv, "qkv": qkv, "q": q, "k": k, "v": v, "a": a_raw, "b": beta_raw}
    return mixer, snap


def _dn_rollback(
    model: Model, i: int, buf: MTPVerifyBuffers, lengths: torch.Tensor, snap: dict
) -> None:
    w, pool = model.layers[i], model.pool[i]
    conv_k1 = pool["conv"].shape[-1]
    full = torch.cat([snap["pre_conv"], snap["qkv"].transpose(1, 2)], dim=-1)
    win = torch.arange(conv_k1, device=full.device)
    idx = (lengths.long()[:, None] + win[None, :]).clamp(max=full.shape[-1] - 1)
    rolled = torch.gather(full, -1, idx[:, None, :].expand(-1, full.shape[1], -1))
    mtp_mod.mtp_dense_moe.scatter_active_first_dim(pool["conv"], buf.slot_rows, rolled, buf.active)
    deltanet_fused.delta_rule_verify_exact(
        (snap["q"], snap["k"]),
        snap["v"],
        (snap["a"], snap["b"]),
        pool["rec"],
        (w["A_log"], w["dt_bias"]),
        active=buf.active,
        lanes=buf.slot_rows,
        lengths=lengths,
        materialize_steps=False,
    )


def apply_forced(
    token_matrix: torch.Tensor, forced_n: torch.Tensor, forced_ids: torch.Tensor
) -> None:
    """Forced feed (MTP_INTERFACE.md): row `j`'s first `forced_n[j]` drafts
    (`token_matrix[j, 1:]`) become `forced_ids[j, :forced_n[j]]`, in place; column 0 and the
    remaining drafts are untouched."""
    k = token_matrix.shape[1] - 1
    col = torch.arange(k, device=token_matrix.device)[None, :]
    token_matrix[:, 1:].copy_(
        torch.where(col < forced_n[:, None], forced_ids[:, :k], token_matrix[:, 1:])
    )


def build_verify_step(
    model: Model,
    mtp: mtp_mod.MTP | None,
    buf: MTPVerifyBuffers,
    rowbuf: graph_decode.Buffers,
    t: int,
    moe_scratch: list[torch.Tensor],
    logits_out: torch.Tensor | None = None,
) -> Callable[[], None]:
    """The capturable verify step (see module docstring and MTP_INTERFACE.md). `moe_scratch`
    holds one `[>= B * t, hidden]` buffer per layer; `mtp` None skips the MTP-cache commit
    (target-only benchmarking)."""
    c = model.cfg
    n = len(model.layers)
    b = buf.capacity
    rows = b * t
    cr = getattr(model.tp, "custom_reduce", None)
    # Row-sharded residual (`SEED_AR_SP`, graph_prefill's proven contract): the fused
    # AR+add+RMSNorm kernel caps at 256 rows, so above that the plain path falls back to RCCL.
    sp = cr is not None and graph_prefill._sp_rows_ok(model, rows)
    shard = rows // model.tp.world if sp else rows
    lo = model.tp.rank * shard if sp else 0

    def step() -> None:
        # Constants are built inside the step, so they live in the graph's own pool: a tensor
        # allocated out here and only closed over is freed with the closure, and a replay
        # would read its reused address (see `graph_decode.GraphReplay`).
        no_store = torch.zeros(b, dtype=torch.bool, device=buf.pos.device)
        if t > 1:
            apply_forced(buf.token_matrix, buf.forced_n, buf.forced_ids)
        x = F.embedding(buf.token_matrix.reshape(rows, 1), model.embed)  # [rows, 1, hidden]
        h = rmsnorm(x, model.layers[0]["in_norm"], c.eps)
        if sp:
            x = x.reshape(rows, -1)[lo : lo + shard].contiguous()
        snaps: dict[int, dict] = {}
        for i in range(n):
            w = model.layers[i]
            if c.layer_types[i] == "full_attention":
                mixer = graph_decode.attn_decode_static(model, i, h, rowbuf)
            else:
                mixer, snaps[i] = _dn_forward(model, i, h, buf, t, no_store)
            nxt = model.layers[i + 1]["in_norm"] if i + 1 < n else model.final_norm
            if sp:
                x, h_mid = cr.sp_ar_add_rmsnorm(
                    mixer.reshape(rows, -1).contiguous(), x, w["post_norm"], c.eps
                )
                moe_out = model.moe(i, h_mid.view(rows, 1, -1), moe_scratch[i][:rows])
                x, h = cr.sp_ar_add_rmsnorm(moe_out.reshape(rows, -1).contiguous(), x, nxt, c.eps)
                h = h.view(rows, 1, -1)
                continue
            x, h_mid = graph_decode._residual_norm(model, mixer, x, w["post_norm"])
            moe_out = model.moe(i, h_mid, moe_scratch[i][:rows])
            x, h = graph_decode._residual_norm(model, moe_out, x, nxt)
        if sp:
            # Final residual rows, gathered once for the hidden seed and the MTP cache.
            full = torch.zeros(rows, c.hidden, dtype=x.dtype, device=x.device)
            full[lo : lo + shard] = x
            x = model.tp.all_reduce(full)
        logits = model.unembed(h)
        if logits_out is not None:  # agreement tests only: `[rows, vocab]` fp32
            logits_out.copy_(logits.reshape(rows, -1))
        step_argmax = logits.argmax(-1).view(b, t)
        buf.step_argmax.copy_(step_argmax)
        accept = _commit_limit(
            _accept_len(buf.token_matrix[:, 1:], step_argmax), step_argmax, buf.budget, buf.stops
        )
        # A forced lane commits exactly its forced ticks plus one target token; forced ticks
        # are not generated output, so stop ids and the budget do not apply to them.
        accept = torch.where(buf.forced_n > 0, buf.forced_n, accept)
        buf.accept_len.copy_(accept)
        lengths = torch.where(buf.active, accept + 1, torch.ones_like(accept)).to(torch.int32)
        for i, snap in snaps.items():
            _dn_rollback(model, i, buf, lengths, snap)
        xs = x.view(b, t, -1)
        if mtp is not None:
            pos_grid = buf.pos[:, None] + torch.arange(t, device=x.device)[None, :]
            previous = torch.cat([buf.draft_hidden[:, None], xs[:, :-1]], dim=1)
            accepted = torch.arange(t, device=x.device)[None, :] <= accept[:, None]
            mtp_mod.cache_target_rows(
                model,
                mtp,
                buf.token_matrix,
                previous,
                pos_grid,
                buf.write_rows,
                buf.active[:, None] & accepted,
                embedding_out=mtp.embedding_scratch[: b * t].view(b, t, c.hidden),
            )
        if model.hidden_scratch is not None:
            mtp_mod.mtp_dense_moe.scatter_time_active_rows(
                model.hidden_scratch, buf.slot_rows, xs, accept, buf.active
            )

    return step


def host_fill(
    model: Model,
    runner: graph_decode.GraphDecodeRunner,
    buf: MTPVerifyBuffers,
    rowbuf: graph_decode.Buffers,
    t: int,
    slots: list[int],
    base_tokens: list[int],
    drafts: list[list[int]] | None,
    positions: list[int],
) -> None:
    """Lane- and row-level fill sharing the decode runner's block-table mirror."""
    import block_pool  # noqa: PLC0415

    for slot, pos in zip(slots, positions, strict=True):
        table = model.block_tables[slot]
        before = len(table.blocks)
        model.grow_lane(slot, pos + t)
        if runner._dirty[slot] or len(table.blocks) != before:
            for lane_table in runner.lane_tables.values():
                lane_table.sync_lane(slot, table.blocks)
            runner._dirty[slot] = False
    cap, nreal = buf.capacity, len(slots)
    pad = graph_decode.pad_slot_for(slots, model.max_batch)
    row_tokens = [[0] * t for _ in range(cap)]
    for j, base in enumerate(base_tokens):
        row_tokens[j] = [base, *(drafts[j] if drafts is not None else [0] * (t - 1))]
    copy_from_host(buf.token_matrix, torch.tensor(row_tokens, dtype=torch.long))
    copy_from_host(buf.pos, torch.tensor(list(positions) + [0] * (cap - nreal), dtype=torch.long))
    copy_from_host(buf.active, torch.tensor([True] * nreal + [False] * (cap - nreal)))
    copy_from_host(buf.slot_rows, torch.tensor(list(slots) + [pad] * (cap - nreal), dtype=torch.long))
    buf.budget.fill_(t)
    buf.stops.fill_(-1)
    buf.forced_n.zero_()
    if model.hidden_scratch is not None:
        buf.draft_hidden.copy_(model.hidden_scratch[buf.slot_rows])
    table = runner.lane_tables[buf.pos.device].table
    buf.block_table.copy_(
        torch.where(buf.active[:, None], table[buf.slot_rows], block_pool.RESERVED_BLOCK)
    )
    bs = model.block_size
    buf.block_valid.copy_(((buf.pos + t + bs - 1) // bs).to(torch.int32))
    grid = buf.pos[:, None] + torch.arange(t, device=buf.pos.device)[None, :]
    block_id = torch.gather(buf.block_table.long(), 1, (grid // bs).long())
    buf.write_rows.copy_(block_id * bs + grid % bs)
    fill_rows(model, buf, rowbuf, t)

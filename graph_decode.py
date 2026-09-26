"""Opt-in HIP/CUDA graph capture for the batched decode step (`--enable-graph-capture`).

Off by default. When off, nothing in this module runs and the scheduler drives `Model`
directly, so the served path is exactly what it is without this file.

**Reconciled with tensor parallelism.** The static step is sharded the way `Model`'s own
decode is: full attention reads its head counts off `model.tp.plan` rather than `Model.cfg`
(whose `heads`/`kv_heads` stay global under TP, while `local_cfg` shards only the DeltaNet
head counts), and `decode_layer_static` holds the same two `tp.all_reduce` calls at the same
two points `Model.decode_layer` has them. Every rank captures its own graph over the whole
layer stack, so a replay issues that rank's collectives from inside the graph.

Two properties make that safe, and both are enforced here rather than assumed:

- **Ranks agree on whether the captured path is live.** `prepare` reduces its own verdict
  across the group, so a rank whose capture or validation failed cannot go on running eager
  collectives against three replaying ranks, which would deadlock at the first layer.
- **Ranks agree on which batches replay.** `replayable` reads only the `(slots, positions)`
  the driver already broadcasts, so every rank reaches the same decision from the same
  inputs without a second round trip.

The premise an earlier version of this file gave for *not* doing this ("capture only removes
host dispatch, and the decode step is 96% device time inside two Triton kernels") was true
under pipeline parallelism and is false under TP=4 plus the byte-LUT decode:
`TP_BYTELUT_BOTTLENECK_2026-09-22.md` section 1 measures the step at 186 ms of host issue
time against 122 ms of device kernel time, so the GPU idles about a third of every step
waiting on ~7,400 `aten` dispatches. Dispatch is the binding constraint, and DeltaNet fusion
plus dense-GEMM solution selection (see `deltanet_fused.py`, `blas_tune.py`) cut device time
enough to put the step back on the host-bound side of that crossover a round after it looked
settled from the device-time side; see this file's git history for the measurement trail.

Why decode and not prefill. A decode step is one token per slot, so once it is padded to
the full `max_batch` slot pool every tensor in it has a shape that is a compile-time
constant of the server (`max_batch`, `max_seq`, `top_k`, the layer dims). Prefill's token
count is the chunk length and its attention span is the prompt, neither of which is fixed,
so prefill stays eager: the shape that would have to be baked into a graph is the one thing
about prefill that varies. This is the same split the prior campaign on this model landed
on, where only a "breakable" (repeatedly broken and resumed) graph was ever accepted for
prefill while decode was captured whole.

What would break a captured graph here, and what this module does about it:

- **Host syncs.** Capture aborts with "operation not permitted when stream is capturing"
  on any device-to-host round trip inside the captured region. So the captured region
  cannot be `Model.decode`; `decode_step` below is a second, static-shape spelling of the
  same arithmetic with no `.item()`, no `int(tensor)`, no data-dependent shape and no
  Python branch on a device value.
- **Data-dependent shapes.** The MoE used to be the worst offender here, sizing its bmm by
  the number of distinct experts the batch activated and by the busiest expert's token
  count. `mxfp4_gemv` removed that: sparsity is a per-program early return inside the fused
  kernel rather than a shape, so `moe_static` is now just `Model.moe` and costs nothing
  extra to capture. See its docstring.
- **Varying attention span.** Each lane's valid context length changes step to step, and
  (paged-kv-design.md, Stage 1) its block table's *contents* grow as it crosses block
  boundaries -- both host ints/lists that a captured graph cannot replay at a different
  shape. The static form keeps `block_table` at its full `[max_batch, max_blocks_per_lane]`
  width always (the worst case, same cost model the dense pool's `max_seq`-wide read already
  had) and varies only the *values* `GraphDecodeRunner.fill` copies into it and into
  `block_valid`/`write_rows` every replay; see `paged_attn.py` for how the kernel turns those
  values into "skip this block-table slot" rather than a shape change.
- **Varying batch composition.** The captured step always runs all `max_batch` rows with
  row j bound to slot j, so which slots are in flight, and how many, changes no shape and
  no index, only the contents of the `active` mask. Rows that are not in this step's batch
  must leave their slot's state alone: their KV write is turned into a no-op by writing
  back what is already there. `deltanet_decode_static`'s masking works differently from the
  attention pool's and is explained on that function: the fused Triton kernel activates its
  gates internally, so there is no pre-activation input that is guaranteed to zero them the
  way an unfused, pre-activated `beta`/`g` could be.
- **Stale inputs.** Every input buffer is rewritten on every replay, unconditionally. The
  prior campaign's one shipped-and-rejected capture bug was a per-step refresh that was
  skipped because it was gated on `id(batch)`, which CPython reuses; there is no such gate
  here.
- **Un-warmed kernels.** Capturing the first call bakes in autotune probes and lazy
  allocations instead of the steady-state kernels, so `prepare` runs the static step
  eagerly `WARMUP_STEPS` times first (which also populates `Model.rope_cache`, whose table
  is built lazily on first use). GEMM solution selection is a separate startup step done
  before this one: `server.build_model` calls `blas_tune.tune` right after loading weights
  and before `build_runner` is reached, and `blas_tune.tune` itself turns its search off
  (`tunable.tuning_enable(False)`) before returning, which is what makes a frozen, already-
  selected GEMM solution safe to run inside a capture -- the search itself synchronizes and
  cannot be captured, but nothing here re-triggers it.
- **Multiple devices.** Layers are split contiguously over the devices, and stream capture
  covers one device's stream. One graph is captured per device over its own run of layers,
  and the hidden state is handed across device boundaries eagerly between replays. Under TP
  a rank owns one device, so `plan_segments` returns a single segment covering the whole
  layer stack and no hand-off runs; the machinery stays for the `--tp 1` pipeline layout.

Three things this module used to name as blockers, and what measuring them on gfx942 (ROCm
7.0, torch 2.9) actually showed, each checked as its own capture of the ingredient alone
against a control that does fail (a `float(t[0, 0])` inside the region, which raises
"operation not permitted when stream is capturing" on every rank):

- **RCCL collectives compose with capture.** `dist.all_reduce` at the step's shape captures
  and replays on all four ranks, alone and interleaved with compute. This is what makes
  holding the collectives *inside* the captured region possible; the alternative was 120
  graph breaks per step, which is no dispatch saving at all.
- **`tl.inline_asm_elementwise` is capture-neutral.** The byte-LUT MXFP4 decode captures on
  all four ranks, on both the per-assignment and the grouped kernel pair. Inline asm is
  compiled into the kernel binary, so a launch of it is an ordinary launch; the only thing
  that has to happen before capture is the JIT, which `warm_static` already forces.
- **The allocator hypothesis was wrong.** An earlier version of this file recorded an
  untested guess that `hipErrorStreamCaptureInvalidated`, which blocked an earlier attempt,
  came from a gigabyte-scale transient driving the caching allocator into a mid-capture
  `hipFree`. A 1 GiB transient captures fine on an idle device; so does a 3 GiB one on a
  device filled to within 4 GiB, one whose first allocation happens inside the capture, and
  one taken against a deliberately fragmented cache. The transient was never the mechanism.
  What actually blocked that attempt was a host sync the pre-`mxfp4_gemv` MoE did, and the
  `enable_gqa` broadcast it was blamed on was a performance bug the batched decode attention
  removed for unrelated reasons.

The cost of all this is that a replay always pays the full `max_seq` attention span (every
lane's block table is `max_blocks_per_lane` wide regardless of bucket) and the padding up to
whatever bucket a batch lands in. A single bucket at `max_batch` (this file's behavior through
integration-r1/r2) made that padding cost up to `max_batch / batch_size`x the real work for a
small batch, so anything under half of `max_batch` stayed eager rather than pay it. `decode`
below instead pads to the *smallest* captured bucket that holds the batch (see `build_buckets`,
`bucket_for`): the padding waste is bounded by the gap between adjacent bucket sizes (2x for
the default `1, 2, 4, 8, 16, 24, 32, 48` ladder) instead of by `max_batch` itself, so even a
batch of 1 replays from its own graph rather than falling back to the host-dispatch-bound eager
path this module exists to avoid.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import attn_decode_fused
import block_pool
import decode_glue
import decode_stamps
import deltanet_fused
import mem_timeline
import rmsnorm_fused
import skinny_hip
import step_timing
import torch
import torch.nn.functional as F
from model import (
    DN_NORM_F32_IN,
    Model,
    PendingTokens,
    apply_rope,
    copy_from_host,
    decode_attention_paged,
    delta_rule_recurrent,
    gated_rmsnorm,
    host_to_device,
    in_proj_sizes,
    rmsnorm,
)

WARMUP_STEPS = 3
"""Eager runs of the static step before capture, so autotuning and lazy allocation settle."""

FUSE_GLUE = os.environ.get("SEED_FUSE_GLUE", "0") not in ("0", "false", "False")
# `SEED_DN_STATE_INPLACE=1`: `deltanet_decode_static` advances the DeltaNet recurrent state in
# the lane pool through `slot_rows` inside the kernel, instead of gather -> (clone -> where) ->
# scatter. Removes 2 (FUSE_GLUE=1) or 4 (FUSE_GLUE=0) full-state HBM round trips per layer.
DN_STATE_INPLACE = os.environ.get("SEED_DN_STATE_INPLACE", "0") not in ("0", "false", "False")
"""Off by default. Collapses the decode step's copy/cast/where glue into the neighboring
Triton kernels: `attn_decode_fused.rope_and_kv_write` (RoPE + masked KV-cache write, replacing
`apply_rope` plus a gather/`torch.where`/scatter per tensor) and `deltanet_fused`'s
`active`-gated `rec` store (replacing a full-state `clone()` plus `torch.where`). See
`DECODE_BOTTLENECK_2026-09-22.md` and each kernel's own docstring for what each replaces and
why it is exact, not an approximation. Independent of `SEED_FUSED_AR_NORM` below: this flag is
the per-tensor glue, that one is the collective/residual/norm fusion.
"""

DN_CONV_INPLACE = os.environ.get("SEED_DN_CONV_INPLACE", "0") not in ("0", "false", "False")
"""Off by default. `deltanet_decode_static` runs `deltanet_fused.causal_conv_decode` in place
of `causal_conv_static`: one Triton kernel reads `qkv` directly (`[B, 1, C]`, no transpose),
reads/advances lane `slot_rows[row]`'s conv state directly in `st["conv"]` (no gather, no
scatter), and writes its `[B, C]` output straight in the layout `q`/`k`/`v` are sliced from
(views, not a `mixed[:, :, 0]` slice of a `[B, C, 1]` `F.conv1d` output). Independent of
`SEED_FUSE_GLUE`/`SEED_DN_STATE_INPLACE`: this collapses the *conv* step's gather/cat/where/
conv1d/silu/scatter chain, not the recurrence or the RoPE/KV-write glue. See
`deltanet_fused.causal_conv_decode`'s docstring for the exact math and its precision note.
"""

ATTN_ROPE_KV_FUSED = os.environ.get("SEED_ATTN_ROPE_KV_FUSED", "0") not in ("0", "false", "False")
"""Off by default. `attn_decode_static` runs `attn_decode_fused.rmsnorm_rope_and_kv_write` in
place of the separate `rmsnorm(q)`/`rmsnorm(k)` calls plus RoPE plus the KV-cache write: one
kernel fuses q/k RMSNorm, RoPE, and the masked paged-KV write for k/v. Independent of
`SEED_FUSE_GLUE`, which only fuses RoPE+KV-write (no norm) and is off in production; this flag
removes the two `rmsnorm_fused` launches on top of that. See
`attn_decode_fused.rmsnorm_rope_and_kv_write`'s docstring for the rounding points this
preserves.
"""

DN_DECODE_FUSED = os.environ.get("SEED_DN_DECODE_FUSED", "0") not in ("0", "false", "False")
"""Off by default; needs `SEED_DN_STATE_INPLACE=1`. The DeltaNet decode conv, recurrence and
gated RMSNorm run as one kernel (`deltanet_fused.dn_decode_fused`) instead of three: 90
launches a step fewer, and the conv output and fp32 recurrence output stay on chip or in L2.
Bit-exact against `SEED_DN_CONV_INPLACE=1` + the in-place recurrence + the norm. A win at
small batches only (see the kernel's docstring); slower per layer at B48."""

ADD_RMSNORM = os.environ.get("SEED_ADD_RMSNORM", "0") not in ("0", "false", "False")
"""Off by default. Each residual update and the RMSNorm that reads it (`post_norm` after the
mixer's all-reduce, the next layer's `in_norm` or the final norm after MoE's) run as one
`rmsnorm_fused.add_rmsnorm` kernel instead of an `add` plus an `rmsnorm`, through
`segment_step`'s chained loop (the same loop `SEED_FUSED_AR_NORM` uses, without touching the
all-reduce). Bit-exact against the unfused pair. Also drops the lm-head `.float()` copy:
`buf.out.copy_` casts the bf16 logits itself. 2 launches a layer, ~120 a step.
"""


AR_RMSNORM_FUSED = os.environ.get("SEED_AR_RMSNORM_FUSED", "0") not in ("0", "false", "False")
"""Off by default. Each all-reduce and the residual add + RMSNorm after it run as one Triton
launch (`allreduce_custom.CustomAllReduce.ar_add_rmsnorm`), graph-safe: it shares the
`SEED_AR_GRAPH_SAFE` device call counter, so it needs `SEED_CUSTOM_ALLREDUCE=1
SEED_AR_GRAPH_SAFE=1` and 4 ranks, and falls back per call to the `SEED_ADD_RMSNORM` pair
otherwise. Bit-exact against that pair. 120 launches a step fewer (2 a layer)."""


AR_SP_DECODE = os.environ.get("SEED_AR_SP_DECODE", "0") not in ("0", "false", "False")
"""Off by default; needs `SEED_AR_SP=1 SEED_AR_RMSNORM_FUSED=1`. Captured decode steps whose
bucket is divisible by 4 keep the residual sharded by rows and run every all-reduce as
`CustomAllReduce.sp_ar_add_rmsnorm` (reduce-scatter + add + norm + all-gather, bit-identical
per row to `ar_add_rmsnorm`)."""

AR_SP_DECODE_MIN = int(os.environ.get("SEED_AR_SP_DECODE_MIN", "48"))
"""Smallest decode bucket `SEED_AR_SP_DECODE` applies to. Captured-step A/B on node04
(pos 2048): B8 +0.56 ms, B16 +0.38 ms, B48 -0.78 ms, B96 -2.0 to -2.8 ms per step; SP's two
flag rounds cost more than the one-shot's single round below ~32 rows."""


def _residual_norm(
    model: Model, partial: torch.Tensor, x: torch.Tensor, norm_w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """`x + all_reduce(partial)` and its RMSNorm under `norm_w`: the fused AR+add+norm
    (`SEED_AR_RMSNORM_FUSED`), the HIP fused-AR kernel (`SEED_FUSED_AR_NORM`), the fused
    add+norm (`SEED_ADD_RMSNORM`), or the plain pair."""
    eps = model.cfg.eps
    cr = getattr(model.tp, "custom_reduce", None)
    if AR_RMSNORM_FUSED and cr is not None and cr.ar_add_rmsnorm_ok(partial, x):
        return cr.ar_add_rmsnorm(partial, x, norm_w, eps)
    if ADD_RMSNORM and not _fused_ar_norm() and rmsnorm_fused.available(x.device):
        return rmsnorm_fused.add_rmsnorm(x, model.tp.all_reduce(partial), norm_w, eps)
    return model.tp.all_reduce_residual_norm(partial, x, norm_w, eps, rmsnorm)


def _fused_ar_norm() -> bool:
    return os.environ.get("SEED_FUSED_AR_NORM", "0") not in ("0", "false", "False")


DEFAULT_BUCKET_SIZES: tuple[int, ...] = (
    1,
    2,
    4,
    8,
    16,
    24,
    32,
    48,
    64,
    80,
    96,
    112,
    128,
)
"""Canonical decode-graph bucket capacities, before capping to a deployment's `max_batch`.

Replaces the old single-bucket-at-`max_batch` capture (integration-r1/r2's `MIN_BATCH_
FRACTION` cutoff, which sent every batch under half of `max_batch` -- so anything below 24 at
the campaign's `max_batch=48` -- down the eager path). The 64 through 128 tail keeps deployments
with more lanes from padding every batch above 48 all the way to `max_batch`. A batch now pads
only to the smallest
bucket that holds it, so the worst-case padding waste is bounded by 2x (the gap between
adjacent canonical sizes) instead of by `max_batch / min_batch`, and a batch of 1 gets its own
graph instead of never being captured at all. See `build_buckets`.
"""

VALIDATE_REL_TOL = 2e-2
"""Max relative logit difference allowed between a replay and the eager step at startup."""


def log(message: str) -> None:
    print(f"[graph-capture] {message}", flush=True)


def all_ranks(tp, ok: bool) -> bool:  # noqa: ANN001 -- tp.TP, kept duck-typed for tests
    """`ok` on every rank of `tp`'s group: one fp32 all-reduce of a 0/1 vote (a MIN written as
    a sum against the world size). Every rank gets the same answer, so every rank takes the
    same branch after it. Identity at `--tp 1`.

    Boot-time validation must vote at every point where a rank could decide alone to stop
    issuing collectives: a rank that rejects a shape and returns while its peers go on to the
    next shape's replay leaves them waiting in that replay's all-reduce (`oneshot_ar_kernel`
    under `SEED_CUSTOM_ALLREDUCE`). The numbers compared are partly rank-local (each rank's
    DeltaNet state shard, its own GEMM solutions), so the ranks can disagree.
    """
    votes = torch.tensor([1.0 if ok else 0.0], device=tp.device)
    tp.all_reduce(votes)
    return int(votes.item()) == tp.plan.world


# ---------------------------------------------------------------- bucket bookkeeping (pure, CPU-testable)


def build_buckets(max_batch: int, canonical: Sequence[int] = DEFAULT_BUCKET_SIZES) -> list[int]:
    """Ascending capture capacities, every canonical size `<= max_batch` plus `max_batch` itself.

    `max_batch` is always included (even if no canonical size equals it) so every legal batch
    size, `1..max_batch`, has some bucket to pad up to; `replayable` below is what enforces
    that no batch is ever asked for outside `[1, max_batch]` in the first place. Pure host
    arithmetic -- no device, no model -- so this is exercised directly by the CPU tests without
    a GPU, per this campaign's no-GPU-until-15:00 constraint.
    """
    if max_batch < 1:
        raise ValueError(f"max_batch must be positive, got {max_batch}")
    return sorted({b for b in canonical if 1 <= b <= max_batch} | {max_batch})


def bucket_for(batch_size: int, buckets: Sequence[int]) -> int:
    """The smallest captured bucket that holds `batch_size` rows.

    Raises if `batch_size` exceeds every bucket, which should not happen when `buckets` came
    from `build_buckets(max_batch)` and the caller already checked `batch_size <= max_batch`
    (`GraphDecodeRunner.replayable` does, before this is ever called) -- kept as an explicit
    error rather than silently picking the largest bucket, so a scheduler bug that lets a
    batch grow past `max_batch` fails loudly here instead of quietly truncating replies.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    for b in buckets:
        if b >= batch_size:
            return b
    raise ValueError(f"batch of {batch_size} exceeds the largest captured bucket {buckets[-1]}")


def pad_slot_for(slots: Sequence[int], max_batch: int, avoid: Sequence[int] = ()) -> int:
    """A lane id in `[0, max_batch)` that is not in `slots`, to fill a bucket's padding rows.

    Every padding row of a replay is pointed at the *same* lane rather than each getting its
    own: the row's compute result is discarded (only `Buffers.active` rows are ever gathered
    back out, see `GraphDecodeRunner.gather`), so all that matters is that no two rows of one
    replay disagree about a lane's true state. Pointing every padding row at one lane not in
    this step's real batch, instead of at `RESERVED_BLOCK`-style dummy storage, reuses the
    invariant `fill` already keeps for every lane (a valid, if stale, block table and KV
    state) instead of adding a second one.

    A caller-visible batch is always `<= max_batch` (`replayable` checks this before any
    bucket is chosen), and the only bucket with room for `max_batch` real rows and zero padding
    rows is the `max_batch` bucket itself -- so whenever padding rows exist at all, `slots` is
    a strict subset of `[0, max_batch)` and some unused lane exists. The `slots[0]` fallback
    below only fires when there is no padding to place (bucket == len(slots)), so its value is
    never actually read as a pad target.

    `avoid` (`graph_mixed`) are the lanes of the *other* half of a mixed replay: skipped like
    `slots` while a free lane exists, and the fallback when none does. That fallback is safe
    where `slots[0]` would not be: two rows of one gather/scatter on the same lane would race
    (one writes the new state, the other the old), while a padding row on the other half's lane
    writes back that lane's state unchanged at a point where the other half is not mid-update
    (`graph_mixed`'s module docstring, "State semantics").
    """
    used = {*slots, *avoid}
    for lane in range(max_batch):
        if lane not in used:
            return lane
    if avoid:
        return avoid[0]
    return slots[0] if slots else 0


# ---------------------------------------------------------------- static-shape decode step
#
# These mirror `Model.attn_decode`, `Model.deltanet_decode` and `Model.moe` at a fixed
# shape. They read the slot pools directly rather than through `Model.bind`, so nothing
# here can leave the single-sequence path pointed at the wrong slot.


def attn_decode_static(model: Model, i: int, x: torch.Tensor, buf: Buffers) -> torch.Tensor:
    """Full attention for one token in every slot, at a fixed shape.

    One difference from `Model.attn_decode`, forced by capture: this always reads through
    `buf.block_table`/`buf.block_valid`, `max_batch` lanes wide (`model.max_blocks_per_lane`
    columns), rather than a per-step batch of just the active slots -- a graph cannot be
    replayed at a row count or a block-table width other than the one it was captured at, and
    both are host ints there (same "replay always pays the worst case" cost model
    `Buffers`/`fill` already hold `pos`/`active` to). `buf.write_rows` is this step's physical
    pool row per lane, refreshed every replay by `GraphDecodeRunner.fill` from `model.
    block_tables` -- host bookkeeping (growing a lane's block table, same as
    `Model.attn_decode`'s eager path does) that happens *before* this runs, not inside the
    captured region.

    The head counts come from `model.tp.plan`, not from `Model.cfg`. `local_cfg` shards only
    the DeltaNet head counts, so `cfg.heads` and `cfg.kv_heads` are still the global 32 and 2
    on every rank while `q_proj`/`k_proj`/`v_proj` have already been narrowed to this rank's
    shard. Reading the global counts here would reshape a `pl.q.count`-wide projection into
    `cfg.heads` heads: at TP=4 that raises, and at a world size where it happened to divide it
    would silently reinterpret the shard. `Model.attn_decode` reads `pl` for the same reason.
    """
    w = model.layers[i]
    qg = skinny_hip.linear(x, w["q_proj"])
    k, v = skinny_hip.linear(x, w["k_proj"]), skinny_hip.linear(x, w["v_proj"])
    decode_stamps.mark("dense_qkv")
    core = attn_decode_core(model, i, qg, k, v, buf)
    decode_stamps.mark("attn_core")
    out = skinny_hip.linear(core, w["o_proj"])
    decode_stamps.mark("dense_o")
    return out


def attn_decode_core(
    model: Model, i: int, qg: torch.Tensor, k: torch.Tensor, v: torch.Tensor, buf: Buffers
) -> torch.Tensor:
    """`attn_decode_static` from the three projections' outputs (`qg` = `q_proj`, query and
    gate interleaved per head; `k`, `v`) to the gated attention output `o_proj` reads,
    `[b, 1, nq * head_dim]`. Split out so `graph_mixed` can run the projections and `o_proj`
    once over decode and prefill rows together."""
    c, w, pool = model.cfg, model.layers[i], model.pool[i]
    b, pl = qg.shape[0], model.tp.plan
    nq, nkv = pl.q.count, pl.kv.count
    q_raw, gate = qg.view(b, 1, nq, 2 * c.head_dim).chunk(2, dim=-1)
    k_raw = k.view(b, 1, nkv, c.head_dim)
    v_raw = v.view(b, 1, nkv, c.head_dim)
    cos, sin = model.rope_at(buf.pos, qg.dtype)
    if ATTN_ROPE_KV_FUSED and attn_decode_fused.available(qg.device):
        # One kernel: RMSNorm q and k, RoPE both, then scatter-write roped-k and v into the
        # paged pool, gated on `buf.active`. Reads `q_raw`/`k_raw`/`v_raw` straight off the
        # projections (no `.transpose(1, 2)`, no separate `rmsnorm` calls) -- see the kernel's
        # docstring for the rounding points this preserves and what it replaces: the 2
        # `rmsnorm_fused` launches plus whatever `SEED_FUSE_GLUE` below leaves for RoPE and the
        # KV-cache write.
        q = attn_decode_fused.rmsnorm_rope_and_kv_write(
            q_raw,
            k_raw,
            v_raw,
            w["q_norm"],
            w["k_norm"],
            cos,
            sin,
            c.rot_dim,
            c.eps,
            pool["k"],
            pool["v"],
            buf.write_rows,
            buf.active,
        )
    else:
        q = rmsnorm(q_raw, w["q_norm"], c.eps).transpose(1, 2)
        k = rmsnorm(k_raw, w["k_norm"], c.eps).transpose(1, 2)
        v = v_raw.transpose(1, 2)
        if FUSE_GLUE and attn_decode_fused.available(qg.device):
            # One kernel: RoPE q and k, then scatter-write roped-k and v into the paged pool,
            # gated on `buf.active` so an inactive row's cache entry is untouched -- see the
            # kernel's docstring for why that replaces `apply_rope` (4 torch ops) plus a
            # gather/`torch.where`/scatter per tensor (6 more) with zero extra HBM traffic
            # beyond the write itself.
            q = attn_decode_fused.rope_and_kv_write(
                q, k, v, cos, sin, c.rot_dim, pool["k"], pool["v"], buf.write_rows, buf.active
            )
        else:
            q = apply_rope(q, cos[:, None, None], sin[:, None, None])
            k = apply_rope(k, cos[:, None, None], sin[:, None, None])
            # An inactive row still indexes some slot's (paged) KV, so its write is made a
            # no-op by writing back the value already there. Skipping the write is not an
            # option: the index is the same tensor op for every row.
            keep = buf.active[:, None, None]
            pool["k"][buf.write_rows] = torch.where(keep, k[:, :, 0], pool["k"][buf.write_rows])
            pool["v"][buf.write_rows] = torch.where(keep, v[:, :, 0], pool["v"][buf.write_rows])
    out = decode_attention_paged(
        q,
        pool["k"],
        pool["v"],
        buf.block_table,
        buf.block_valid,
        buf.pos,
        model.block_size,
        c.head_dim**-0.5,
    )
    out = out.transpose(1, 2).reshape(b, 1, -1)
    if decode_glue.available(out):  # `SEED_ELEMWISE_FUSED`: reshape copy + sigmoid + mul in one
        return decode_glue.sigmoid_gate_mul(out, gate)
    return out * torch.sigmoid(gate.reshape(b, 1, -1))


def causal_conv_static(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    keep: torch.Tensor,
    active: torch.Tensor | None = None,
) -> torch.Tensor:
    """`Model.causal_conv` over every slot at once, leaving inactive rows' state untouched.

    `keep` (`[B, 1, 1]`, broadcast-shaped) and `active` (`[B]`, one bool per row) are the same
    mask in two shapes: `keep` for the `SEED_FUSE_GLUE=0` `torch.where`, `active` for
    `deltanet_fused.masked_row_copy`'s per-row gated store, which needs a flat row axis to
    index by `tl.program_id(0)`. Callers without a flat mask (`graph_mtp`) pass only `keep`
    and always take the `torch.where` path.
    """
    full = torch.cat([state, x], dim=-1)
    new_state = full[:, :, -state.shape[-1] :]
    if active is not None and FUSE_GLUE and deltanet_fused.available(x.device):
        deltanet_fused.masked_row_copy(state, new_state, active)
    else:
        state.copy_(torch.where(keep, new_state, state))
    return F.silu(F.conv1d(full, weight, groups=full.shape[1]))


def deltanet_decode_static(model: Model, i: int, x: torch.Tensor, buf: Buffers) -> torch.Tensor:
    """DeltaNet for one token in every slot, batched over slots, via the fused decode path.

    Mirrors `Model.deltanet_decode`'s `t == 1` fused branch: one `in_proj_all` GEMM in place
    of four separate projections, then `deltanet_fused.delta_rule_decode`/`gated_rmsnorm`
    (one Triton kernel each) in place of the torch recurrence and gated-rmsnorm chain. Falls
    back to the plain torch path when `deltanet_fused.available` is false (CPU, interpreter
    tests, or `SEED_FUSED_DELTANET=0`), the same condition `Model.deltanet_decode` gates on,
    so this stays correct off the real accelerator without a second capture path to maintain.

    The head counts here (`c.k_heads`, `c.v_heads`, ...) are already this rank's local ones:
    unlike full attention, `local_cfg` shards DeltaNet's head counts into `Model.cfg` itself,
    so nothing here needs `model.tp.plan` the way `attn_decode_static` does.

    **Bucket rows vs. lanes.** `model.pool[i]["conv"]`/`["rec"]` are `[max_batch, ...]`,
    indexed by *lane*, but a bucket smaller than `max_batch` only computes `buf` rows
    `0..b-1`, whose row-to-lane mapping is `buf.slot_rows` (this step's real slots, then
    `pad_slot_for`'s filler; see `GraphDecodeRunner.fill`) -- not the identity a `max_batch`
    -sized buffer could once assume. So this gathers each row's state out of the lane pool by
    `buf.slot_rows` before computing and scatters it back after, the same `rows = ...; pool[
    rows]` shape `Model.deltanet_decode` already uses for its own (non-static, non-bucketed)
    gather. `buf.slot_rows` is itself a static buffer refreshed every replay, so the *indices*
    the gather/scatter read vary per replay while the op shapes stay fixed -- capturable for
    the same reason `write_rows`/`block_table` are.

    A padding row's `slot_rows` entry is one shared filler lane (see `pad_slot_for`), and every
    padding row computes with `keep = False`, so every duplicate write the scatter makes to
    that lane writes back the *same* unmodified value it gathered -- safe regardless of which
    duplicate write lands last, because they are identical, not because of write ordering.

    Masking an inactive row exactly is not the trick the pre-fusion version of this function
    used. That version pre-activated `beta`/`g` in Python and zeroed them *after*
    activation, which is exact because `delta_rule_recurrent` takes already-activated values:
    `beta = 0` makes the rank-1 update `k * 0` and `g = 0` makes the decay `exp(0) = 1`. The
    fused kernel does not expose that seam -- it takes the raw `in_proj_b`/`in_proj_a`
    projections and computes `sigmoid`/`softplus` internally, for every row it is given, and
    writes `rec` in place for every row. There is no raw input guaranteed to zero a sigmoid
    or drive a softplus to bit-exact zero (both only reach their limit asymptotically), so
    masking the inputs would leave inactive rows' state decaying by some not-quite-1.0 factor
    every step they are inactive -- silent, compounding drift on a slot that might be
    reactivated many steps later. `SEED_FUSE_GLUE=0` (default) runs the kernel on a scratch
    copy of `rec` and selects the update back into the real state with `torch.where`, exact by
    construction (a plain select) rather than by a floating-point identity holding at every
    intermediate step of a kernel this function does not control. `SEED_FUSE_GLUE=1` instead
    passes `active` straight into `delta_rule_decode`, which gates its own `rec` store the
    same exact way -- see that kernel's docstring -- and needs no scratch copy at all.

    `SEED_DN_CONV_INPLACE=1` (`DN_CONV_INPLACE`, independent of the two flags above): the conv
    step itself runs through `deltanet_fused.causal_conv_decode` instead of
    `causal_conv_static`, reading/advancing `st["conv"][rows]` inside the kernel rather than
    through the `st["conv"][rows] = ...; ...; st["conv"][rows] = conv_state` gather/scatter
    pair, and reading `qkv` directly with no `.transpose(1, 2)`. Off, this function's conv step
    is unchanged from before that flag existed.
    """
    w = model.layers[i]
    proj = skinny_hip.linear(x, w["in_proj_all"])
    decode_stamps.mark("dense_in_proj")
    core = deltanet_decode_core(model, i, proj, buf)
    decode_stamps.mark("dn_core")
    out = skinny_hip.linear(core, w["out_proj"])
    decode_stamps.mark("dense_out_proj")
    return out


def deltanet_decode_core(model: Model, i: int, proj: torch.Tensor, buf: Buffers) -> torch.Tensor:
    """`deltanet_decode_static` from `in_proj_all`'s output (`[b, 1, ...]`) to the gated-norm
    output `out_proj` reads, `[b, 1, v_heads * v_dim]`. Split out for `graph_mixed`, like
    `attn_decode_core`."""
    c, w, st = model.cfg, model.layers[i], model.pool[i]
    b = proj.shape[0]
    keep = buf.active[:, None, None]
    rows = buf.slot_rows

    qkv, z, beta_raw, a_raw = proj.split(in_proj_sizes(c), dim=-1)
    if DN_DECODE_FUSED and DN_STATE_INPLACE and deltanet_fused.available(proj.device):
        out = deltanet_fused.dn_decode_fused(
            qkv,
            (w["conv"], st["conv"]),
            (a_raw[:, 0], beta_raw[:, 0]),
            st["rec"],
            (w["A_log"], w["dt_bias"]),
            (buf.active, rows),
            (z.unflatten(-1, (c.v_heads, c.v_dim))[:, 0], w["dn_norm"], c.eps),
            (c.k_heads, c.v_heads, c.k_dim, c.v_dim),
        )
        return out.reshape(b, 1, -1)
    conv_in_place = DN_CONV_INPLACE and deltanet_fused.available(proj.device)
    if conv_in_place:
        # `SEED_DN_CONV_INPLACE=1`: one kernel reads `qkv` directly and advances lane `rows`'s
        # conv state in `st["conv"]` in place -- no gather, no scatter, no transpose. `flat` is
        # already `[B, C]`, so the `q`/`k`/`v` splits below are views of it, same as the old
        # path's `mixed[:, :, 0]` view of a `[B, C, 1]` `F.conv1d` output.
        flat = deltanet_fused.causal_conv_decode(qkv, w["conv"], st["conv"], rows, buf.active)
    else:
        conv_state = st["conv"][rows]
        mixed = causal_conv_static(qkv.transpose(1, 2), w["conv"], conv_state, keep, buf.active)
        st["conv"][rows] = conv_state

    if deltanet_fused.available(proj.device):
        if not conv_in_place:
            flat = mixed[:, :, 0]
        key_dim = c.k_heads * c.k_dim
        q = flat[:, :key_dim].unflatten(-1, (c.k_heads, c.k_dim))
        k = flat[:, key_dim : 2 * key_dim].unflatten(-1, (c.k_heads, c.k_dim))
        v = flat[:, 2 * key_dim :].unflatten(-1, (c.v_heads, c.v_dim))

        if DN_STATE_INPLACE:
            out = deltanet_fused.delta_rule_decode(
                (q, k),
                v,
                (a_raw[:, 0], beta_raw[:, 0]),
                st["rec"],
                (w["A_log"], w["dt_bias"]),
                active=buf.active,
                lanes=rows,
            )
            gate = z.unflatten(-1, (c.v_heads, c.v_dim))[:, 0]
            out = deltanet_fused.gated_rmsnorm(
                out if DN_NORM_F32_IN else out.to(proj.dtype), gate, w["dn_norm"], c.eps, proj.dtype
            )
            return out.reshape(b, 1, -1)
        rec_state = st["rec"][rows]
        if FUSE_GLUE:
            # Gated in-kernel store on the gathered rows; inactive rows keep what was gathered.
            out = deltanet_fused.delta_rule_decode(
                (q, k),
                v,
                (a_raw[:, 0], beta_raw[:, 0]),
                rec_state,
                (w["A_log"], w["dt_bias"]),
                active=buf.active,
            )
            st["rec"][rows] = rec_state
        else:
            rec_scratch = rec_state.clone()
            out = deltanet_fused.delta_rule_decode(
                (q, k), v, (a_raw[:, 0], beta_raw[:, 0]), rec_scratch, (w["A_log"], w["dt_bias"])
            )
            st["rec"][rows] = torch.where(buf.active[:, None, None, None], rec_scratch, rec_state)
        gate = z.unflatten(-1, (c.v_heads, c.v_dim))[:, 0]
        out = deltanet_fused.gated_rmsnorm(
            out if DN_NORM_F32_IN else out.to(proj.dtype), gate, w["dn_norm"], c.eps, proj.dtype
        )
        return out.reshape(b, 1, -1)

    key_dim, val_dim = c.k_heads * c.k_dim, c.v_heads * c.v_dim
    q, k, v = mixed.transpose(1, 2).split([key_dim, key_dim, val_dim], dim=-1)
    q, k = q.reshape(b, 1, c.k_heads, c.k_dim), k.reshape(b, 1, c.k_heads, c.k_dim)
    v = v.reshape(b, 1, c.v_heads, c.v_dim)
    beta = beta_raw.sigmoid()
    g = -w["A_log"].exp() * F.softplus(a_raw.float() + w["dt_bias"])
    beta, g = torch.where(keep, beta, 0.0), torch.where(keep, g, 0.0)
    rep = c.v_heads // c.k_heads
    q, k = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)
    rec_state = st["rec"][rows]
    out = delta_rule_recurrent(q, k, v, g, beta, rec_state).to(proj.dtype)
    st["rec"][rows] = rec_state
    out = gated_rmsnorm(out.reshape(-1, c.v_dim), z.reshape(-1, c.v_dim), w["dn_norm"], c.eps)
    return out.reshape(b, 1, -1)


def moe_static(model: Model, i: int, x: torch.Tensor, buf: Buffers) -> torch.Tensor:
    """MoE at a fixed shape, which is now what `Model.moe` already is on the fused path.

    This used to be a second spelling of the MoE, because the grouped torch form sized its
    `bmm` by the number of distinct experts the batch activated and by the busiest group, and
    both are host reads. It bought static shape by computing the worst case: one expert weight
    set per (token, expert) assignment with no de-duplication, and, under expert parallelism,
    a clamp into the rank's range instead of a drop, so every rank dequantized every
    assignment. With `mxfp4_gemv` that trade is gone. The fused kernel's launch shape is
    `max_batch * top_k` regardless of routing, and it drops the assignments this rank does not
    own inside the kernel, so the capturable form and the fast form are the same code.

    `buf` is unused and kept for the uniform `(model, i, x, buf)` signature the segment
    builder calls these through.
    """
    return model.moe(i, x, model.moe_scratch[i][: x.shape[0]])


def decode_layer_static(model: Model, i: int, x: torch.Tensor, buf: Buffers) -> torch.Tensor:
    """One layer, with `Model.decode_layer`'s two all-reduces at the same two points.

    Both mixers and the routed MoE produce row-parallel partial sums under TP, so the
    collectives are part of the arithmetic and belong inside the captured region: leaving
    them outside would mean breaking the graph twice per layer, which is 120 breaks and no
    dispatch saving. RCCL collectives do capture and replay on this target; that is measured,
    not assumed, and the module docstring says how.
    """
    w, c = model.layers[i], model.cfg
    h = rmsnorm(x, w["in_norm"], c.eps)
    mixer = (
        attn_decode_static(model, i, h, buf)
        if c.layer_types[i] == "full_attention"
        else deltanet_decode_static(model, i, h, buf)
    )
    x = x + model.tp.all_reduce(mixer)
    return x + model.tp.all_reduce(moe_static(model, i, rmsnorm(x, w["post_norm"], c.eps), buf))


def decode_layer_static_fused(
    model: Model, i: int, x: torch.Tensor, h: torch.Tensor, buf: Buffers, next_norm_w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """`decode_layer_static`, with both all-reduces fused with the RMSNorm that follows them
    (`SEED_FUSED_AR_NORM=1`; see `tp.TP.all_reduce_residual_norm`).

    `h` is `rmsnorm(x, in_norm, eps)` for layer `i`, already computed -- by the previous
    layer's call to this function (its second fused all-reduce produces exactly this layer's
    `in_norm`-normalized input as a side output), or by `segment_step` for the segment's first
    layer, where there is no previous fused all-reduce to have produced it. Returns
    `(x, h_next)`: `x` is this layer's residual output, `h_next` is `rmsnorm(x, next_norm_w,
    eps)` -- the next layer's `in_norm` input, or, when `next_norm_w` is `model.final_norm`
    (the segment's last layer), the LM head's input, ready to feed straight in without a
    second norm kernel.
    """
    w, c = model.layers[i], model.cfg
    mixer = (
        attn_decode_static(model, i, h, buf)
        if c.layer_types[i] == "full_attention"
        else deltanet_decode_static(model, i, h, buf)
    )
    x, h_mid = _residual_norm(model, mixer, x, w["post_norm"])
    decode_stamps.mark("ar_norm")
    moe_out = moe_static(model, i, h_mid, buf)
    out = _residual_norm(model, moe_out, x, next_norm_w)
    decode_stamps.mark("ar_norm")
    return out


# ---------------------------------------------------------------- segments and buffers


@dataclass(frozen=True)
class Segment:
    """Layers `[start, stop)`, all of which live on `device`. `last` carries the lm head."""

    start: int
    stop: int
    device: torch.device
    last: bool


def plan_segments(layer_dev: Sequence[torch.device]) -> list[Segment]:
    """Split the layer stack into one segment per contiguous run of layers on one device."""
    segments: list[Segment] = []
    start = 0
    for i in range(1, len(layer_dev) + 1):
        if i == len(layer_dev) or layer_dev[i] != layer_dev[start]:
            segments.append(Segment(start, i, layer_dev[start], i == len(layer_dev)))
            start = i
    return segments


class Buffers:
    """One segment's static tensors at one bucket capacity. A captured graph points at these
    addresses, so every one of them is allocated exactly once, here, and only ever refreshed
    in place afterward (`GraphDecodeRunner.fill`'s `.copy_()` calls) -- never resized, freed,
    or replaced; see that method and `paged-kv-design.md` Stage 1 for why a captured replay
    cannot tolerate a reallocation.

    `capacity` is this buffer set's bucket size (`<= model.max_batch`; the bucket list itself
    is `build_buckets(model.max_batch)`), not always `model.max_batch` the way a single
    full-batch capture's buffers were. Row `j` is this step's `j`-th request, *not* lane `j`
    (`GraphDecodeRunner.fill`'s `row_slots`/`slot_rows` carry the row-to-lane mapping; only the
    `capacity == model.max_batch` bucket happens to make that mapping the identity).

    `x_in`, `pos`, `active`, `block_table`, `block_valid`, `write_rows` and `slot_rows` are
    written before every replay; `out` is read after it. `block_table` (`[capacity, model.
    max_blocks_per_lane]`, this row's lane's block ids, right-padded with `block_pool.
    RESERVED_BLOCK`), `block_valid` (`[capacity]`, `block_pool.valid_block_count` per row) and
    `write_rows` (`[capacity]`, this row's lane's physical pool row) are the paged-KV
    counterparts of the old dense pool's `rows`/`span` constants (paged-kv-design.md, Stage 1):
    unlike those, these three are not constants -- a lane's block table grows every `block_size`
    steps -- so `GraphDecodeRunner.fill` rewrites them every replay exactly like `pos`/`active`,
    rather than building them once here. `slot_rows` (`[capacity]`, int64) is this row's lane id
    in `model.pool[i]`'s `[max_batch, ...]` DeltaNet state arrays; `deltanet_decode_static`
    gathers/scatters through it instead of assuming row-equals-lane.
    """

    def __init__(self, model: Model, segment: Segment, capacity: int) -> None:
        b, dev, c = capacity, segment.device, model.cfg
        self.capacity = capacity
        self.x_in = torch.zeros(b, 1, c.hidden, dtype=model.dtype, device=dev)
        self.pos = torch.zeros(b, dtype=torch.long, device=dev)
        self.active = torch.zeros(b, dtype=torch.bool, device=dev)
        self.block_table = torch.zeros(b, model.max_blocks_per_lane, dtype=torch.int32, device=dev)
        self.block_valid = torch.zeros(b, dtype=torch.int32, device=dev)
        self.write_rows = torch.zeros(b, dtype=torch.long, device=dev)
        self.slot_rows = torch.zeros(b, dtype=torch.long, device=dev)
        self.out = (
            torch.zeros(b, c.vocab, dtype=torch.float32, device=dev)
            if segment.last
            else torch.zeros(b, 1, c.hidden, dtype=model.dtype, device=dev)
        )


@dataclass
class BucketGraphs:
    """One bucket's captured state: its buffers and replay callables, one pair per segment."""

    capacity: int
    buffers: list[Buffers] = field(default_factory=list)
    replays: list[Callable[[], None]] = field(default_factory=list)


class LaneBlockTables:
    """Persistent, incrementally-updated device mirror of every lane's paged-KV block table.

    `GraphDecodeRunner.fill` used to rebuild a bucket's `block_table`/`block_valid`/
    `write_rows` from `model.block_tables[slot].blocks` (a host Python list) on *every*
    replay -- host round-tripping `capacity * max_blocks_per_lane` ints even on the steps
    where not one lane's block *list* actually changed. Measured on the cluster at ~4.6 ms of
    host time per step at b48, max_seq 16384 (`max_blocks_per_lane` = 1024 at the campaign's
    16-token block size: `torch.tensor` converting a `capacity`-long list of 1024-long Python
    lists dominates it), most of the host-side gap between replays.

    A lane's block *list* only changes on two host events, both already local to one lane, not
    the whole batch: `BlockTable.grow_to` appending fresh blocks (roughly once every
    `block_size` steps per lane) and `attach_blocks` replacing it wholesale (admission --
    prefix-cache reuse and copy-on-write). This tensor mirrors exactly those two events into a
    persistent `[max_batch, max_blocks_per_lane]` device buffer, one instance per segment
    device (a real TP deployment has one device and so one instance; only the legacy `--tp 1`
    pipeline layout's extra devices pay for more). Every other read of "this step's block
    table" is `GraphDecodeRunner.fill`'s device-side gather by `slot_rows` plus device
    arithmetic off `pos` -- see that method -- not a host rebuild.

    `grow_to` only ever appends (`BlockTable`'s own docstring: "append-only as its token count
    grows"), so a length change before/after calling it is an exact, sufficient signal that
    this lane needs a sync; `fill`'s grow loop uses exactly that. `attach_blocks` can replace a
    same-length list with different block ids (copy-on-write forking the tail of an otherwise
    shared prefix), where a length check would miss the change, so `GraphDecodeRunner.
    attach_blocks` syncs unconditionally instead of diffing.
    """

    def __init__(self, model: Model, device: torch.device) -> None:
        b, w = model.max_batch, model.max_blocks_per_lane
        self.table = torch.full((b, w), block_pool.RESERVED_BLOCK, dtype=torch.int32, device=device)

    def sync_lane(self, slot: int, blocks: Sequence[int]) -> None:
        """Push lane `slot`'s current block list to the device. Call only when it changed.

        Entries at or beyond `len(blocks)` are left as whatever they were -- stale, possibly
        holding a former occupant's now-freed block ids -- rather than re-zeroed to
        `RESERVED_BLOCK`. That stays memory-safe (every block id, current or stale, is an
        in-bounds index into the shared, block-paged KV pool tensor: reading a stale one costs
        nothing but returns some other lane's bytes) and those entries are never trusted for
        correctness: the reader is `block_valid`, computed fresh from `pos` every replay
        (`GraphDecodeRunner.fill`), and `fill`'s own grow loop always brings a lane's real
        length up to what `pos` needs, syncing as it does, before anything reads this lane's
        table for that step -- the same "out-of-range slot still addresses real storage, and
        the valid-count predicate is what actually gates reads" argument `block_pool.py`'s
        `RESERVED_BLOCK` docstring makes for the padding case.
        """
        if blocks:
            n = len(blocks)
            self.table[slot, :n].copy_(host_to_device(list(blocks), self.table.device, torch.int32))


def segment_step(model: Model, segment: Segment, buf: Buffers) -> Callable[[], None]:
    """The callable that gets captured: static buffers in, static buffer out, nothing returned.

    Writing the result into `buf.out` rather than returning it is what makes a replay and an
    eager call interchangeable to the caller: a replay cannot return anything, and a tensor
    a captured graph allocated is only valid until the next replay overwrites it.

    `SEED_FUSED_AR_NORM=1` runs `decode_layer_static_fused` instead, chaining each layer's
    second fused all-reduce's normalized output straight into the next layer's mixer input,
    with no separate `in_norm` kernel between them: `h` is computed once, before the loop, and
    every layer after that gets it as this call's `h_next`. This restructuring is worth doing
    (one fewer `rmsnorm` dispatch a layer) even on a rank whose `TP.all_reduce_residual_norm`
    calls fall back to plain torch ops internally (RCCL, CPU/gloo, or the custom all-reduce
    unavailable): the fallback returns the exact same `(x, normed)` pair the fused kernel
    would, so this loop shape is correct and the norm-count saving holds regardless of which
    branch each call takes -- see `TP.all_reduce_residual_norm`'s docstring. That is also what
    makes it testable without a GPU (`seed_tests/test_fuse_glue.py`).

    Only every layer *inside this segment* can be chained this way -- the next segment's
    `in_norm` lives on a different device under the pipeline (`--tp 1`) layout, where
    `plan_segments` can return more than one segment, so a non-final segment's last layer falls
    back to the unfused two-call sequence (its second all-reduce genuinely has no norm to fuse
    with: `segment.last is False` just copies `x` to `buf.out` below either way). TP's single-
    segment layout (`plan_segments` on one device's full layer stack) is the only shape this
    model deploys decode through, so that fallback is reachability-only, not a real trade.
    """
    fused_ar_norm = _fused_ar_norm() or ADD_RMSNORM or AR_RMSNORM_FUSED
    cr = getattr(model.tp, "custom_reduce", None)
    sp = (
        AR_SP_DECODE
        and buf.capacity >= AR_SP_DECODE_MIN
        and AR_RMSNORM_FUSED
        and segment.start == 0
        and segment.last
        and cr is not None
        and cr.sp_ok(buf.x_in, buf.x_in[: buf.capacity // model.tp.world])
    )

    def step() -> None:
        x = buf.x_in
        if not fused_ar_norm:
            for i in range(segment.start, segment.stop):
                x = decode_layer_static(model, i, x, buf)
            if segment.last:
                logits = model.unembed(rmsnorm(x, model.final_norm, model.cfg.eps))
                buf.out.copy_(logits[:, 0].float())
            else:
                buf.out.copy_(x)
            return

        decode_stamps.mark("start")
        h = rmsnorm(x, model.layers[segment.start]["in_norm"], model.cfg.eps)
        if sp:
            shard_rows = buf.capacity // model.tp.world
            lo = model.tp.rank * shard_rows
            x = x[lo : lo + shard_rows].contiguous()
        decode_stamps.mark("glue")
        for i in range(segment.start, segment.stop):
            last_in_segment = i == segment.stop - 1
            if last_in_segment and segment.last:
                next_norm_w = model.final_norm
            elif last_in_segment:  # non-final segment boundary: no norm to fuse into, see above
                w, c = model.layers[i], model.cfg
                mixer = (
                    attn_decode_static(model, i, h, buf)
                    if c.layer_types[i] == "full_attention"
                    else deltanet_decode_static(model, i, h, buf)
                )
                x, h_mid = _residual_norm(model, mixer, x, w["post_norm"])
                moe_out = moe_static(model, i, h_mid, buf)
                x = x + model.tp.all_reduce(moe_out)
                buf.out.copy_(x)
                return
            else:
                next_norm_w = model.layers[i + 1]["in_norm"]
            if sp:
                w, c = model.layers[i], model.cfg
                mixer = (
                    attn_decode_static(model, i, h, buf)
                    if c.layer_types[i] == "full_attention"
                    else deltanet_decode_static(model, i, h, buf)
                )
                x, h_mid = cr.sp_ar_add_rmsnorm(
                    mixer.contiguous(),
                    x,
                    w["post_norm"],
                    c.eps,
                )
                decode_stamps.mark("ar_norm")
                x = x.reshape(shard_rows, 1, -1)
                h_mid = h_mid.reshape(buf.capacity, 1, -1)
                x, h = cr.sp_ar_add_rmsnorm(
                    moe_static(model, i, h_mid, buf).contiguous(), x, next_norm_w, c.eps
                )
                decode_stamps.mark("ar_norm")
                x = x.reshape(shard_rows, 1, -1)
                h = h.reshape(buf.capacity, 1, -1)
            else:
                x, h = decode_layer_static_fused(model, i, x, h, buf, next_norm_w)

        if segment.last:
            logits = model.unembed(h)  # h == rmsnorm(x, final_norm, eps) already
            decode_stamps.mark("lm_head")
            # `copy_` casts bf16 -> fp32 exactly as `.float()` does, minus one full-logits copy.
            buf.out.copy_(logits[:, 0] if ADD_RMSNORM else logits[:, 0].float())
            decode_stamps.mark("glue")
        else:
            buf.out.copy_(x)

    return step


# ---------------------------------------------------------------- capture backends


class CaptureBackend(Protocol):
    """Turns a callable over static buffers into a callable that re-runs it.

    Injected rather than called directly so the plumbing around capture can be tested
    without a GPU: a backend that returns `step` itself is what a correct replay has to be
    equivalent to, since both must read every input from the static buffers.
    """

    def capture(self, step: Callable[[], None], device: torch.device) -> Callable[[], None]: ...


class CudaGraphBackend:
    """Captures with `torch.cuda.graph`, which is the HIP graph API on ROCm under its
    CUDA-shaped names.

    **Shared memory pool across buckets.** With eight-plus buckets times the segments per
    device, a device sees many captures back to back. Each `torch.cuda.graph()` call defaults
    to reserving its own private memory pool for whatever the captured region allocates
    internally (autograd is off here, so this is workspace/scratch, not activations kept for a
    backward pass); with one pool per capture that scratch is never reused *across* captures,
    so the device's reserved allocation grows with the number of buckets even though only one
    bucket's graph ever replays at a time. `torch.cuda.graph_pool_handle()` plus the `pool=`
    argument to `torch.cuda.graph()` is the documented mechanism for handing a set of captures
    the same pool on purpose (see the PyTorch CUDA graphs docs' "graph pools" section, written
    for exactly this multi-graph-on-one-device case); this backend allocates one pool per
    device on first use and threads it through every later `capture()` call on that device, so
    the eight (device, bucket) graphs one segment plans still stay live independently but only
    reference the same backing pool's memory in aggregate.
    """

    def __init__(self) -> None:
        self._pools: dict[torch.device, object] = {}

    def capture(self, step: Callable[[], None], device: torch.device) -> Callable[[], None]:
        with torch.cuda.device(device):
            torch.cuda.synchronize(device)
            pool = self._pools.setdefault(device, torch.cuda.graph_pool_handle())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                step()
            torch.cuda.synchronize(device)
        return GraphReplay(graph, step)


class GraphReplay:
    """A captured graph's replay, holding the step it was captured from.

    The graph reads every tensor `step` closed over by device address, but holds no Python
    reference to any of them. A builder that allocates a tensor outside the step and closes
    over it (a constant index, a mask) would otherwise free it once the step goes out of
    scope, and the caching allocator hands the address to the next allocation: the replay
    then reads whatever that allocation holds. Keeping `step` alive keeps its closure alive.
    """

    __slots__ = ("graph", "step")

    def __init__(self, graph: object, step: Callable[[], None]) -> None:
        self.graph, self.step = graph, step

    def __call__(self) -> None:
        self.graph.replay()


def _cuda_memory_bytes(devices: Iterable[torch.device]) -> int:
    """Total `memory_allocated` across `devices`, 0 for any non-CUDA device (CPU tests).

    Used to report each bucket's capture memory cost (`GraphDecodeRunner.prepare`): the delta
    across one bucket's `capture()` calls is what that bucket's graphs actually added to the
    device's live allocation, on top of whatever the shared pool already reserved for earlier
    buckets.
    """
    seen: set[torch.device] = set()
    total = 0
    for dev in devices:
        if dev.type == "cuda" and dev not in seen:
            seen.add(dev)
            total += torch.cuda.memory_allocated(dev)
    return total


# ---------------------------------------------------------------- the runner


class GraphDecodeRunner:
    """`Runner` adapter that replays captured decode graphs, falling back to `Model`.

    Everything except `decode` is the model's own method. `decode` is a dispatcher: a batch
    that does not fit what was captured, or a capture that never succeeded, takes the
    model's eager path unchanged.
    """

    def __init__(
        self,
        model: Model,
        backend: CaptureBackend | None = None,
        prefill_graphs: bool | None = None,
        mixed_graphs: bool | None = None,
    ) -> None:
        self.model = model
        self.backend: CaptureBackend = backend if backend is not None else CudaGraphBackend()
        # `SEED_PREFILL_GRAPHS` (graph_prefill.py) unless the caller says otherwise; captured
        # at the end of `prepare`, served by `prefill`/`prefill_batch`.
        self.want_prefill_graphs = prefill_graphs
        self.prefill_runner = None
        # `SEED_MIXED_GRAPH` (graph_mixed.py) unless the caller says otherwise; captured after
        # the prefill graphs, served by `decode_mixed`.
        self.want_mixed_graphs = mixed_graphs
        self.mixed_runner = None
        self.mixed_path = "eager"  # the same for the last mixed step ("graphD+RxW" or "eager")
        self.segments = plan_segments(model.layer_dev)
        self.buckets = build_buckets(model.max_batch)
        self.graphs: dict[int, BucketGraphs] = {}
        self.enabled = False
        self.tokens = torch.zeros(model.max_batch, 1, dtype=torch.long, device=model.devices[0])
        self.decode_path = "eager"  # which path the last `decode` took; SEED_STEP_TIMING reports it
        self.prefill_path = "eager"  # the same for the last prefill call ("graphRxW" or "eager")
        self.decode_bucket = 0  # the bucket the last replayed `decode` used, 0 when eager
        self._timing_replays = 0
        self.lane_tables = {
            dev: LaneBlockTables(model, dev) for dev in {s.device for s in self.segments}
        }
        self._dirty = [True] * model.max_batch  # forces every lane to sync on its first fill()

    @property
    def buffers(self) -> list[Buffers]:
        """The largest bucket's (`== model.max_batch`) segment buffers.

        Kept as the pre-bucketing name/shape for callers (tests, `SEED_STEP_TIMING` logging)
        that want "the" capture at full batch; `self.graphs` is the real, per-bucket state.
        """
        graphs = self.graphs.get(self.model.max_batch)
        return graphs.buffers if graphs else []

    @property
    def replays(self) -> list[Callable[[], None]]:
        graphs = self.graphs.get(self.model.max_batch)
        return graphs.replays if graphs else []

    # -- Runner protocol ------------------------------------------------------
    @property
    def max_batch(self) -> int:
        return self.model.max_batch

    def begin(self, slot: int) -> None:
        """Forwards to the model, then marks `slot`'s device block-table mirror dirty.

        `Model.reset` (`begin`'s own reset) sets `block_tables[slot].blocks = []` without
        touching `self.lane_tables` -- prefill, which grows a lane's table straight through
        `Model.full_attention`, never goes through `GraphDecodeRunner.fill`'s tracked grow loop
        either. So a lane reset then reprefilled between two `decode()` calls can end up at the
        *same* block count it had before (a freed block immediately reallocated, LIFO), which
        `fill`'s length-diff check alone would not notice, and the wrong (stale) block ids
        would replay against paged KV. The dirty flag set here forces `fill`'s next grow loop
        to resync this lane unconditionally, closing that gap regardless of what the lengths
        happen to say.
        """
        self.model.begin(slot)
        self._dirty[slot] = True

    def _touch(self, lanes: Iterable[int]) -> None:
        """Mark `lanes`' device block-table mirrors stale: a call that may have grown their
        host tables outside `fill` (a prefill, an eager step, `extend_blocks`) goes through
        here, so `fill` re-syncs them before the next replay reads them."""
        for lane in lanes:
            self._dirty[lane] = True

    def prefill(self, slot: int, ids: Sequence[int], start: int) -> torch.Tensor:
        shape = self._prefill_shape([(slot, ids, start)])
        self.prefill_path = shape.label if shape is not None else "eager"
        if shape is not None:
            return self.prefill_runner.prefill_batch(shape, [(slot, list(ids), start)])[0]
        self._touch((slot,))
        return self.model.prefill(slot, list(ids), start)

    def prefill_batch(self, calls: Sequence[tuple[int, Sequence[int], int]]) -> list[torch.Tensor]:
        """A call that fits a captured prefill shape (`SEED_PREFILL_GRAPHS`, graph_prefill.py)
        replays it; anything else is the model's own packed path unchanged."""
        calls = [(slot, list(ids), start) for slot, ids, start in calls]
        shape = self._prefill_shape(calls)
        self.prefill_path = shape.label if shape is not None else "eager"
        if shape is not None:
            return self.prefill_runner.prefill_batch(shape, calls)
        self._touch(slot for slot, _, _ in calls)
        return self.model.prefill_batch(calls)

    def prefill_fit(self) -> tuple[Callable[[Sequence[int]], bool], int] | None:
        """`(fits, max_width)` for the captured prefill shapes, or None when prefill replays
        are off. `fits(lengths)` says whether one call of those per-row chunk lengths replays.
        Rank 0's scheduler shapes its packed calls with it (`SEED_PREFILL_FIT_GRAPH`); every
        rank holds the same agreed shape set, so this is a local read, not a broadcast."""
        runner = self.prefill_runner
        if runner is None or not runner.enabled or not runner.shapes:
            return None
        import graph_prefill  # noqa: PLC0415 -- graph_prefill imports this module

        shapes = runner.shapes
        return (
            lambda lengths: graph_prefill.shape_for(lengths, shapes) is not None,
            max(s.area if s.packed else s.width for s in shapes),
        )

    def prefill_shapes(self) -> list[tuple[int, int]]:
        """The agreed captured prefill shapes as `(rows, width)`, empty when prefill replays
        are off. `SEED_PREFILL_ACCUM` picks the shape a fired step packs toward from these."""
        runner = self.prefill_runner
        if runner is None or not runner.enabled:
            return []
        return [(s.rows, s.width) for s in runner.shapes]

    def prefill_packing(self) -> dict[tuple[int, int], tuple[int, int]]:
        """`{(rows, width): (segment slots, alignment)}` for the agreed shapes captured packed
        (`SEED_PREFILL_PACK`): `SEED_PREFILL_ACCUM` fills those by aligned token area and slot
        count instead of by rows and width. Local read, like `prefill_fit`."""
        runner = self.prefill_runner
        if runner is None or not runner.enabled:
            return {}
        import graph_prefill  # noqa: PLC0415 -- graph_prefill imports this module

        align = graph_prefill.PACK_ALIGN
        return {(s.rows, s.width): (s.segs, align) for s in runner.shapes if s.packed}

    def mixed_fit(self, n_decode: int) -> tuple[Callable[[Sequence[int]], bool], int, int] | None:
        """`(fits(lengths), max_width, max_tokens)` for a mixed step with `n_decode` decode
        rows (`graph_mixed.MixedGraphRunner.fit`), or None when mixed replays are off or no
        captured shape has that many decode rows. Local read, like `prefill_fit`."""
        runner = self.mixed_runner
        return runner.fit(n_decode) if runner is not None else None

    def _prefill_shape(self, calls: Sequence[tuple[int, Sequence[int], int]]):  # noqa: ANN202
        """The captured prefill shape serving `calls`, or None (eager). Reads only `calls`."""
        if self.prefill_runner is None:
            return None
        return self.prefill_runner.replayable(calls)

    def save_snapshot(self, lane: int, snap: int) -> None:
        self.model.save_snapshot(lane, snap)

    def score(self, slot: int, ids: Sequence[int], continuation_start: int) -> torch.Tensor:
        """Teacher-forced scoring is never captured (only `decode` is): the model's own
        eager path, same as `prefill` above."""
        self._touch((slot,))
        return self.model.score(slot, list(ids), continuation_start)

    def load_snapshot(self, lane: int, snap: int) -> None:
        self.model.load_snapshot(lane, snap)

    def lane_blocks(self, lane: int) -> tuple[int, ...]:
        return self.model.lane_blocks(lane)

    def attach_blocks(self, lane: int, blocks: Sequence[int]) -> None:
        """Forwards to the model, then syncs `lane`'s device block-table mirror unconditionally.

        Unlike `fill`'s grow loop (a length check, exact for `grow_to` because it only ever
        appends), `attach_blocks` can swap in a same-length but different-content list
        (copy-on-write forking the tail of an otherwise shared prefix), where a length check
        would miss the change -- see `LaneBlockTables`'s docstring.
        """
        self.model.attach_blocks(lane, blocks)
        for lane_table in self.lane_tables.values():
            lane_table.sync_lane(lane, self.model.block_tables[lane].blocks)
        self._dirty[lane] = False

    def copy_block(self, dst_block: int, src_block: int, filled: int) -> None:
        self.model.copy_block(dst_block, src_block, filled)

    def lane_block_count(self, lane: int) -> int:
        return self.model.lane_block_count(lane)

    def extend_blocks(self, grants: Sequence[tuple[int, int]]) -> None:
        """Rank-0-reserved ids appended to lanes' host tables (every rank, via
        `EXTEND_BLOCKS`). The scheduler calls this before the step that writes into them,
        including before an `OVERLAP_SCHED` lookahead launch, so marking the lanes here is
        what makes that step's `fill` push the new ids to the device mirror."""
        self.model.extend_blocks(grants)
        self._touch(lane for lane, _ in grants)

    def decode_tokens_per_step(self) -> int:
        return self.model.decode_tokens_per_step()

    def decode_row(self, logits: torch.Tensor, row: int) -> torch.Tensor:
        return self.model.decode_row(logits, row)

    def sample_batch(self, logits: torch.Tensor, temperatures: Sequence[float]) -> list[int]:
        return self.model.sample_batch(logits, list(temperatures))

    def cache_node_logits(self, snap: int, logits: torch.Tensor) -> None:
        self.model.cache_node_logits(snap, logits)

    def cached_node_logits(self, snap: int) -> torch.Tensor:
        return self.model.cached_node_logits(snap)

    def warmup(self) -> None:
        """The model's own warmup. Named apart from `warm_static`, which warms the captured
        step and takes the callables to run; `server.py` and `tp_driver.apply` call this one
        through the `Runner` protocol, so the two must not share a name."""
        self.model.warmup()

    @property
    def max_seq(self) -> int:
        return self.model.max_seq

    def speculative_decode(
        self,
        slots: Sequence[int],
        tokens: Sequence[int],
        positions: Sequence[int],
        budgets: Sequence[int] | None = None,
        stops: Sequence[Sequence[int]] | None = None,
    ) -> list[list[int]]:
        """The model's own MTP round, uncaptured. `graph_mtp.GraphMTPRunner` overrides this
        with the captured draft+verify round; `server.build_runner` picks that subclass
        whenever the model loaded an MTP head."""
        self.decode_path = "mtp-eager"
        self._touch(slots)
        return self.model.speculative_decode(
            list(slots), list(tokens), list(positions), budgets, stops
        )

    def decode_mixed(
        self,
        slots: Sequence[int],
        tokens: Sequence[int],
        positions: Sequence[int],
        prefill_calls: Sequence[tuple[int, Sequence[int], int]],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """A step with a prefill chunk replays a captured mixed shape when one holds it
        (`SEED_MIXED_GRAPH`, graph_mixed.py) and otherwise runs eager (`Model.decode_mixed`,
        `SEED_MIXED_BATCH`); with no chunk it is an ordinary decode step and may replay."""
        if not prefill_calls:
            return self.decode(slots, tokens, positions), []
        mixed = self.mixed_runner
        shape = mixed.replayable(slots, positions, prefill_calls) if mixed is not None else None
        if shape is not None:
            self.decode_path = self.mixed_path = f"graph{shape.name}"
            return mixed.run(shape, slots, tokens, positions, prefill_calls)
        self.decode_path = self.mixed_path = "mixed"
        self._touch([*slots, *(slot for slot, _, _ in prefill_calls)])
        return self.model.decode_mixed(list(slots), list(tokens), list(positions), prefill_calls)

    def decode_launch(
        self,
        slots: Sequence[int],
        tokens: Sequence[int],
        positions: Sequence[int],
        temperatures: Sequence[float] | None,
    ) -> PendingTokens | None:
        """`SEED_OVERLAP_SCHED`'s decode: `Model.decode_launch`'s contract on the captured
        path. `fill` resolves `LOOKAHEAD_TOKEN` entries on the device straight into the
        graph's static `tokens` buffer, so a replay can be launched before the previous one's
        tokens reach the host; `Model.launch_tail` then samples, shares and stores this step's
        ids the same way the eager path does."""
        slots, tokens, positions = list(slots), list(tokens), list(positions)
        if not self.enabled or not self.replayable(slots, positions):
            self.decode_path = "eager"
            self._touch(slots)
            return self.model.decode_launch(slots, tokens, positions, temperatures)
        meter = self.model.gap_meter
        if meter is not None:
            meter.step_start()
        logits = self._replay(slots, tokens, positions)
        pending = self.model.launch_tail(slots, logits, temperatures)
        if meter is not None:
            meter.step_end()
        return pending

    def decode(
        self, slots: Sequence[int], tokens: Sequence[int], positions: Sequence[int]
    ) -> torch.Tensor:
        slots, tokens, positions = list(slots), list(tokens), list(positions)
        if not self.enabled or not self.replayable(slots, positions):
            self.decode_path, self.decode_bucket = "eager", 0
            self._touch(slots)
            return self.model.decode(slots, tokens, positions)
        meter = self.model.gap_meter
        if meter is not None:
            meter.step_start()
        logits = self._replay(slots, tokens, positions)
        if meter is not None:
            meter.step_end()
        return logits

    def _replay(self, slots: list[int], tokens: list[int], positions: list[int]) -> torch.Tensor:
        """Fill, replay and gather one replayable step; shared by `decode`/`decode_launch`."""
        capacity = bucket_for(len(slots), self.buckets)
        graphs = self.graphs[capacity]
        self.decode_path, self.decode_bucket = "graph", capacity
        timed = False
        if step_timing.ENABLED:
            self._timing_replays += 1
            timed = self._timing_replays % step_timing.EVERY == 0
            t0 = time.perf_counter()
        self.fill(graphs.buffers, capacity, slots, tokens, positions)
        if timed:
            t_fill = time.perf_counter()
        try:
            self.run(graphs.buffers, graphs.replays)
        except Exception as exc:
            # Unlike a failure before the first replay, this one may have advanced some of
            # the step's state already, so the step is not retried eagerly: re-running it
            # would step those slots twice. The scheduler fails this batch and drops the
            # slots' cached prefixes, and every later step takes the eager path.
            self.enabled = False
            log(f"replay failed, eager from here on: {exc!r}")
            raise
        if timed:
            for segment in self.segments:
                if segment.device.type == "cuda":
                    torch.cuda.synchronize(segment.device)
            t1 = time.perf_counter()
            step_timing.log(
                f"rank {self.model.tp.plan.rank} graph decode n={self._timing_replays} "
                f"batch={len(slots)} bucket={capacity} max_pos={max(positions, default=0)} "
                f"fill_host_ms={(t_fill - t0) * 1e3:.2f} replay_ms={(t1 - t_fill) * 1e3:.1f} "
                f"wall_ms={(t1 - t0) * 1e3:.1f}"
            )
        return self.gather(graphs.buffers, len(slots))

    # -- dispatch -------------------------------------------------------------
    def replayable(self, slots: Sequence[int], positions: Sequence[int]) -> bool:
        """Whether this batch is one some captured bucket can serve.

        The shape assumptions a replay makes are checked here rather than assumed from the
        scheduler's behavior, so a scheduler change shows up as a slower step and not as a
        corrupted slot. Every batch from 1 to `max_batch` has a bucket (`build_buckets`
        guarantees `max_batch` is always one of them), so unlike the old single-bucket runner
        there is no lower cutoff here beyond "the batch fits at all."
        """
        b = self.model.max_batch
        return (
            1 <= len(slots) <= b
            and len(set(slots)) == len(slots)
            and all(0 <= s < b for s in slots)
            and all(0 <= p < self.model.max_seq for p in positions)
        )

    def fill(
        self,
        buffers: list[Buffers],
        capacity: int,
        slots: list[int],
        tokens: list[int],
        positions: list[int],
        avoid: Sequence[int] = (),
    ) -> None:
        """Copy this step's inputs into `buffers` (one bucket's segment buffers), padding to
        `capacity` rows. `avoid` lists lanes the padding rows should not point at when a free
        one exists (`graph_mixed`: the same replay's prefill lanes).

        Row `j` is this call's `j`-th request (`slots[j]`), not lane `j`: `row_slots` below is
        the row-to-lane map, `slots` followed by one repeated filler lane from `pad_slot_for`
        for the padding rows. A row that is not in this batch is switched off in `active`
        rather than pointed at a real lane it would corrupt. Every row is written on every
        call, including the padding rows: a buffer left over from the previous step is the
        classic way a replay reads stale data.

        Paged-KV: each *active* lane's table is made to cover `pos + 1` tokens first
        (`Model.grow_lane`: a capacity check once the scheduler owns blocks and has already
        reserved them on rank 0 and broadcast `EXTEND_BLOCKS`; boot-time warmup and validation
        grow it here). Idle lanes are never grown: under TP a worker must not allocate.

        `block_table`/`block_valid`/`write_rows` are a device-side gather out of the persistent
        `self.lane_tables` mirror by `slot_rows`, plus device arithmetic off `pos`; see
        `LaneBlockTables`. A lane is re-synced into the mirror when `self._dirty` says its host
        table changed through some runner call (`begin`, `extend_blocks`, a prefill, an eager
        step), or when the `grow_lane` below appended. Padding rows are forced to
        `block_pool.RESERVED_BLOCK` (all-padding table row; `pos = 0`, so one valid block and
        write row 0 of block 0): they never read or write the filler lane's real KV, which may
        be mid-prefill for another request.
        """
        model = self.model
        for slot, pos in zip(slots, positions, strict=True):
            table = model.block_tables[slot]
            before = len(table.blocks)
            model.grow_lane(slot, pos + 1)
            if self._dirty[slot] or len(table.blocks) != before:
                for lane_table in self.lane_tables.values():
                    lane_table.sync_lane(slot, table.blocks)
                self._dirty[slot] = False

        pad = pad_slot_for(slots, model.max_batch, avoid)
        row_slots = list(slots) + [pad] * (capacity - len(slots))
        row_tokens = [0] * capacity
        row_pos = [0] * capacity
        row_active = [False] * capacity
        for j, (token, pos) in enumerate(zip(tokens, positions, strict=True)):
            row_tokens[j], row_pos[j], row_active[j] = token, pos, True

        # `LOOKAHEAD_TOKEN` rows (SEED_OVERLAP_SCHED) take their lane's last sampled id from
        # `Model.lane_tokens`, on the device, indexed by the row's lane.
        self.tokens[:capacity].copy_(model.resolve_tokens(row_tokens, row_slots)[:, None])
        first = buffers[0]
        first.x_in.copy_(F.embedding(self.tokens[:capacity], model.embed))
        pos_cpu = torch.tensor(row_pos, dtype=torch.long)
        active_cpu = torch.tensor(row_active, dtype=torch.bool)
        slot_rows_cpu = torch.tensor(row_slots, dtype=torch.long)

        for buf in buffers:
            copy_from_host(buf.pos, pos_cpu)
            copy_from_host(buf.active, active_cpu)
            copy_from_host(buf.slot_rows, slot_rows_cpu)
            lane_table = self.lane_tables[buf.block_table.device].table
            buf.block_table.copy_(
                torch.where(
                    buf.active[:, None], lane_table[buf.slot_rows], block_pool.RESERVED_BLOCK
                )
            )
            buf.block_valid.copy_(
                ((buf.pos + 1 + model.block_size - 1) // model.block_size).to(torch.int32)
            )
            block_idx = (buf.pos // model.block_size).long()
            block_id = buf.block_table.gather(1, block_idx[:, None]).squeeze(1).long()
            buf.write_rows.copy_(block_id * model.block_size + buf.pos % model.block_size)

    def run(self, buffers: list[Buffers], actions: Sequence[Callable[[], None]]) -> None:
        """Run one bucket's segments in order, handing the hidden state across device
        boundaries.

        The handoff stays outside the captured regions: it is one copy of `[capacity, 1,
        hidden]` per boundary, and a single graph cannot span two devices' streams anyway.
        """
        for index, action in enumerate(actions):
            if index:
                buffers[index].x_in.copy_(buffers[index - 1].out)
            action()

    def gather(self, buffers: list[Buffers], n: int) -> torch.Tensor:
        """This step's `n` real rows, in the caller's own order, copied out of `out`.

        Row `j` of a bucket's `out` is already the caller's `j`-th request (`fill`'s
        `row_slots` puts `slots` first, padding after), so this is a prefix slice, not a
        permutation -- unlike the pre-bucketing runner, which had to gather by lane because a
        capacity-`max_batch` buffer's row *was* the lane. `.clone()` still matters: `out` is
        overwritten by the next replay, and a bare slice would be a view into it.
        """
        return buffers[-1].out[:n].clone()

    # -- setup ----------------------------------------------------------------
    def prepare(self) -> bool:
        """Warm up, capture, and check a replay against the eager step. Never raises.

        Call once, before the scheduler starts: warmup and the check both run real decode
        steps, which advance whatever state the slots hold.

        Returns whether the captured path is live. On any failure the runner stays a
        pass-through to the eager model, which is the behavior with the flag off.

        Under TP every rank runs this, in step, because warmup and validation both drive real
        decode steps and so issue this rank's share of every collective. The verdict is agreed
        across the group before it is acted on (`agree`): one rank silently falling back to
        eager would deadlock the group at the next layer's all-reduce, where three replaying
        ranks wait for a collective the fourth issues from a different place.

        The one failure this cannot paper over is an exception raised *part way through*
        `warm_static`, which under TP leaves the group holding different numbers of issued
        collectives, so `agree` itself would hang. That is not the failure mode capture has:
        every rank runs the same code over the same shapes, so a capture failure is
        deterministic and symmetric, and `agree` exists for the case where the verdict
        differs while the collective count does not (a device that will not capture, a
        validation that only one rank's numerics fail). A genuinely asymmetric failure is a
        hang, and is meant to be: the alternative is three ranks serving from a graph while
        the fourth serves from somewhere else.
        """
        try:
            self.graphs = {}
            prepare_t0 = time.perf_counter()
            devices = [s.device for s in self.segments]
            mem_before_all = _cuda_memory_bytes(devices)
            for capacity in self.buckets:
                t0 = time.perf_counter()
                mem_before = _cuda_memory_bytes(devices)
                buffers = [Buffers(self.model, s, capacity) for s in self.segments]
                steps = [
                    segment_step(self.model, s, b)
                    for s, b in zip(self.segments, buffers, strict=True)
                ]
                self.warm_static(buffers, steps, capacity)
                replays = [
                    self.backend.capture(step, segment.device)
                    for step, segment in zip(steps, self.segments, strict=True)
                ]
                self.graphs[capacity] = BucketGraphs(capacity, buffers, replays)
                mem_timeline.mark(f"decode_graph_b{capacity}", self.model.devices[0])
                mem_after = _cuda_memory_bytes(devices)
                log(
                    f"captured bucket {capacity} ({len(replays)} graph(s)) in "
                    f"{(time.perf_counter() - t0) * 1e3:.0f} ms, "
                    f"+{(mem_after - mem_before) / 2**20:.1f} MiB live"
                )
            self.enabled = True
            log(
                f"captured {len(self.buckets)} bucket(s) {self.buckets} in "
                f"{(time.perf_counter() - prepare_t0):.1f} s total, "
                f"+{(_cuda_memory_bytes(devices) - mem_before_all) / 2**20:.1f} MiB live overall"
            )
        except Exception as exc:
            self.enabled = False
            log(f"capture failed, decode stays eager: {exc!r}")
        self.enabled = self.agree(self.enabled)
        if self.enabled:
            self.enabled = self.agree(self.validate())
        self.reset_slots()
        mem_timeline.mark("decode_validate", self.model.devices[0])
        if self.enabled:
            log(f"enabled for decode batches of 1 to {self.model.max_batch}")
        self._prepare_prefill_graphs()
        mem_timeline.mark("prefill_graphs_done", self.model.devices[0])
        self._prepare_mixed_graphs()
        mem_timeline.mark("mixed_graphs_done", self.model.devices[0])
        return self.enabled

    def _prepare_mixed_graphs(self) -> None:
        """`SEED_MIXED_GRAPH`: capture the mixed decode+prefill shapes (graph_mixed.py). Every
        rank reaches this with the same flag and the same agreed decode verdict, so capture and
        agreement collectives line up; a failure leaves `mixed_runner` disabled."""
        import graph_mixed  # noqa: PLC0415 -- graph_mixed imports this module

        want = self.want_mixed_graphs
        if not (graph_mixed.MIXED_GRAPH if want is None else want):
            return
        runner = graph_mixed.MixedGraphRunner(self, self.backend)
        runner.prepare()
        self.mixed_runner = runner

    def _prepare_prefill_graphs(self) -> None:
        """`SEED_PREFILL_GRAPHS`: capture the small-prefill shapes (graph_prefill.py). Every
        rank reaches this with the same flag, so the capture and its agreement collectives
        line up; a failure leaves `prefill_runner` disabled (eager prefill)."""
        import graph_prefill  # noqa: PLC0415 -- graph_prefill imports this module

        want = self.want_prefill_graphs
        if not (graph_prefill.PREFILL_GRAPHS if want is None else want):
            return
        lane_table = self.lane_tables.get(self.model.devices[-1])
        if lane_table is None:
            log("prefill capture off: needs a single-segment layout")
            return
        runner = graph_prefill.PrefillGraphRunner(self.model, self.backend, lane_table, self._dirty)
        graph_prefill._mem_note(runner.device, "before prefill capture")
        runner.prepare()
        graph_prefill._mem_note(runner.device, "after prefill validation")
        self.prefill_runner = runner

    def agree(self, ok: bool) -> bool:
        """`ok` on every rank of the group, as one collective. Identity at `--tp 1`.

        The reduce runs on the same process group the step's own all-reduces use, so a rank
        that reaches this point has issued exactly as many collectives as its peers, whatever
        it decided.
        """
        if all_ranks(self.model.tp, ok):
            return True
        if ok:
            log("another rank could not capture; decode stays eager on every rank")
        return False

    def warm_static(
        self, buffers: list[Buffers], steps: Sequence[Callable[[], None]], capacity: int
    ) -> None:
        """Run one bucket's static step eagerly a few times before capturing it.

        Capture records whatever the first call would launch. On a first call that is
        autotune probes and lazy allocations (`Model.rope_cache`'s table among them), not
        the kernels the steady state would use.
        """
        self.fill(buffers, capacity, list(range(capacity)), [0] * capacity, [0] * capacity)
        for _ in range(WARMUP_STEPS):
            self.run(buffers, steps)
        self.reset_slots()

    def reset_slots(self) -> None:
        """Clear every slot's recurrent state, release the blocks warmup/validation grew,
        and leave the model bound to slot 0.

        Warmup and validation step real slots, once per bucket now (`prepare` warms and
        validates every bucket in turn), so this runs several times per `prepare()` call
        rather than the twice a single-bucket runner needed. `_release_lane_blocks` before
        `begin` matters more here than it used to for exactly that reason: `Model.reset`
        (`begin`'s own reset) drops a lane's `BlockTable.blocks` list without decref-ing the
        allocator (see `Model._release_lane_blocks`'s docstring -- `Model.generate` has the
        same requirement, for the same reason), so skipping this leaks `capacity` blocks per
        bucket per reset instead of reusing them, and `len(self.buckets)` buckets times two
        resets per bucket (warm, then validate) is enough to exhaust a small pool. KV rows
        themselves do not need clearing because a slot's rows are always overwritten by its
        next prefill before anything reads them.

        Goes through `self.begin` (not `self.model.begin` directly) so every reset also marks
        `self._dirty`, same as a scheduler-driven reset would -- `fill`'s own length-diff check
        already catches the very next regrow this method's callers immediately trigger, so
        skipping the flag would not be an observed bug here, but nothing about this call site
        should rely on that coincidence when the correct, general path costs nothing extra.
        """
        if self.model.scheduler_owns_blocks:
            raise RuntimeError("reset_slots is boot-only: the scheduler owns lane blocks now")
        for slot in range(self.model.max_batch):
            self.model._release_lane_blocks(slot)
            self.begin(slot)
        self.model.bind(0)

    def validate(self) -> bool:
        """Check every bucket's replay against the eager step on one synthetic full step.

        This is the only check that can say a captured graph computes the right thing;
        everything before it only says capture did not raise. It runs on the real weights at
        startup, once per bucket (including padding rows, since a bucket at less than
        `max_batch` real rows is the common case at serve time, not the exception the old
        single-bucket runner's full-batch-only check covered) -- a mismatch in any bucket
        disables the feature entirely instead of serving a mix of trustworthy and untrustworthy
        bucket sizes.

        The verdict is agreed after every bucket (`agree`), so all ranks stop at the same
        bucket and issue the same collectives.
        """
        model = self.model
        worst_error, worst_bucket = 0.0, 0
        for capacity in self.buckets:
            ok = True
            try:
                slots = list(range(capacity))
                tokens = [(7 * i + 3) % model.cfg.vocab for i in range(capacity)]
                positions = [0] * capacity
                self.reset_slots()
                want = model.decode(slots, tokens, positions)
                self.reset_slots()
                graphs = self.graphs[capacity]
                self.fill(graphs.buffers, capacity, slots, tokens, positions)
                self.run(graphs.buffers, graphs.replays)
                got = self.gather(graphs.buffers, capacity)
                scale = want.abs().max().clamp(min=1.0)
                error = float((got - want).abs().max() / scale)
                mismatched = int((got.argmax(-1) != want.argmax(-1)).sum())
                if error > worst_error:
                    worst_error, worst_bucket = error, capacity
                if error >= VALIDATE_REL_TOL or mismatched:
                    log(
                        f"bucket {capacity} replay disagrees with the eager step (relative "
                        f"error {error:.3g}, {mismatched}/{capacity} rows pick a different "
                        "token); decode stays eager"
                    )
                    ok = False
            except Exception as exc:
                log(f"validation could not run, decode stays eager: {exc!r}")
                ok = False
            if not self.agree(ok):
                return False
        log(
            f"every bucket matches the eager step (worst relative error {worst_error:.3g} at bucket {worst_bucket})"
        )
        return True

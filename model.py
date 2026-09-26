"""Plain-PyTorch Qwen3.5-MoE (text only), written from the HF modeling code.

Design (see reference/modeling_qwen3_5_moe.py for the semantics this mirrors):
- `max_batch` decode lanes (paged-kv-design.md, Stage 2 calls them lanes, not slots -- a
  lane is a row index with no state of its own once released; see `session_cache.py`'s and
  `scheduler.py`'s module docstrings for the lane/cached-session split this file implements
  the `Runner` side of), each with its own static live state, allocated once.
- full-attention layers (paged-kv-design.md, Stage 1): KV storage is a block pool shared
  across lanes, `[num_blocks * block_size, kv_heads, head_dim]`, sized from measured free
  memory at startup (`_size_pools`) with a floor of `max_batch * max_seq` tokens' worth. Each
  lane owns a `block_pool.BlockTable` (an ordered list of block ids, grown as its sequence
  lengthens); Stage 2 moves *ownership* of those block ids to `scheduler.py`'s `SessionCache`
  (a lane's table is just a working copy the scheduler seeds via `attach_blocks` and forgets
  on `reset`, never itself incref/decref) -- see `block_pool.py` and `paged_attn.py`.
- Gated DeltaNet layers: conv state `[max_batch, conv_dim, K-1]` and fp32 recurrent state
  `[max_batch, v_heads, k_dim, v_dim]` for each lane's *live* state, plus a second pool, sized
  `[num_snapshots, ...]` independently of `max_batch` (Stage 2: a snapshot is a cached
  session's saved boundary, addressed by `SessionCache`-issued id via `save_snapshot`/
  `load_snapshot`, not a fixed one-per-lane register the way Stage 1's `save_prefix` was); the
  delta rule runs chunkwise (per-token loop only at decode).
- Every state tensor is allocated in `__init__` and written in place afterwards. Serving a
  turn allocates no persistent state, so the footprint is whatever startup reserved,
  whether the server has handled one session or a thousand.
- layers are split contiguously over the given devices (one process, activations hop devices).
  That split makes each device one *pipeline stage*; see `decode` and pipeline.py.
- tensor parallel instead, which is what the server deploys: pass a `tp.TP` and this process
  owns a shard of every layer's attention heads, routed experts and DeltaNet value heads, and
  runs *all* the layers. `tp.py` holds the plan, the collectives and the sharding convention;
  read its module docstring before adding a component. `tp.TP.single` is the single-process
  model above, unchanged, and is what the CPU tests and the accuracy path use. The two axes
  are exclusive: tensor parallelism is one process per device, so a tensor-parallel `Model`
  owns exactly one device, the layer split degenerates to one stage, and there is nothing to
  pipeline (`__init__` rejects the combination rather than letting stage threads issue
  collectives out of order).
- routed experts stay MXFP4 in memory and are never dequantized into HBM: `moe` runs the
  fused MXFP4 grouped-GEMV in mxfp4_gemv.py, which unpacks the nibbles in registers inside
  the matmul. `_routed_grouped` keeps the older dequantize-then-`bmm` spelling for dense
  checkpoints and for CPU, where Triton cannot compile. `SEED_HOT_EXPERTS=1` (default off) is
  the one exception: a bounded, per-rank bf16 copy of each layer's hottest local experts is
  kept standing (hot_experts.py), and their assignments are diverted to a dense batched GEMM
  instead of the MXFP4 kernels, which skip anything already served that way.

Two execution paths share the layer weights:
- Single sequence (`forward`, `generate`): one slot bound at a time, batch 1. This is the
  path the accuracy checker and the parity tests use, and it is unchanged.
- Batched decode (`decode`): one token for each of several slots in one pass. The MoE and
  every projection see a real batch dimension; full attention is one call to
  `decode_attention_paged` (`attn_decode`) reading every active slot's KV through its own
  block table (paged-kv-design.md, Stage 1 -- see `block_pool.py`/`paged_attn.py`), so nothing
  about the step scales per slot in Python;
  DeltaNet gathers each active slot's conv/recurrent state out of the pool, runs the
  recurrent form with that batch dimension, and scatters the updated state back (the
  conv/recurrent state is small enough per slot that the gather-scatter copy is cheap,
  unlike the KV cache). The step's slots are cut into microbatches and pipelined across
  the stages, so all four devices compute at once instead of one computing while three
  wait for control flow.

`self.state` is the state of the currently bound slot, so the single-sequence layer code
below never has to know about slots. Binding is not thread safe: one scheduler thread owns
it. The batched decode path never binds; it reads `slot_state` directly, because a pipelined
step has several microbatches, on different slots, inside the model at the same time.

Memory math for the real model (60 layers x 512 experts x 3 x 1024x4096 = 386.5 B params):
  bf16 experts = 773 GB > 4 x 128 GB = 512 GB, so they cannot be held dense.
  MXFP4 experts = 386.5 B x (4 + 8/32) bits = ~205 GB, plus ~20 GB bf16 for everything else.
  Per lane, max_seq 16384: 45 DeltaNet layers x (64 x 128 x 128 fp32 + conv) = 183 MB of
  *live* state, one dense allocation per lane (48 lanes is ~9 GB spread over the 4 devices,
  reserved at startup as before).
  The 15 full-attention layers' KV and the DeltaNet snapshot pool are Stage 2's two
  independently sized, LRU-evicted resources (`_size_pools`, paged-kv-design.md section 1.6).
  Both are sized to *need*, hard-capped, never as a fraction of measured free memory: MI300A is
  a unified-memory APU, so `torch.cuda.mem_get_info`'s "free" is shared with host RSS that grows
  after load during real serving, and a fractional-of-free split (an earlier version of this
  function) OOM'd a real boot even though it looked well within budget at load time. The KV pool
  is `max_batch * max_seq` tokens (`SEED_KV_POOL_MIN_CAPACITY_FACTOR=1.0` floor, unsharded that
  floor is 48 x 16384 x 15 x 2 x 2 x 256 x 2 B = 24.2 GB of blocks) times
  `SEED_KV_POOL_HEADROOM_FACTOR` (default 1.5) for cached-but-idle sessions, capped at
  `SEED_KV_POOL_CAP_GIB` (default 20 GiB/rank; ~15 KB/token/rank puts that at ~1.1M tokens) or
  pinned exactly via `SEED_KV_POOL_GIB`. The snapshot pool is a fixed slot count
  (`SEED_SNAPSHOT_POOL_COUNT`, default 256, ~11 GiB/rank at ~45 MiB/snapshot) or a GiB budget via
  `SEED_SNAPSHOT_POOL_GIB`, not a share of anything measured. Free memory is only consulted
  afterward to keep `SEED_MIN_UNALLOCATED_GIB` (default 25) unallocated for host RSS and
  transient activations (prefill at 2k-token packed batches, MoE workspaces), shrinking the KV
  pool toward its floor if needed. A configuration that cannot even match the floors fails while
  loading, not after N sessions; see the `[pool-plan]`/`[kv-pool]`/`[snapshot-pool]` log lines
  for the numbers an actual run chose.
  At TP=4 the same totals are per rank instead of per node: DeltaNet's dense live-state pool is
  a lane's 16 of 64 value heads (45 MB/lane, ~2.2 GB per device at 48 lanes) on top of ~51 GB
  of MXFP4 experts per device; the KV block pool's per-block cost is `_kv_block_bytes` and the
  snapshot pool's per-snapshot cost is `_snapshot_bytes` (both one rank's local head counts),
  and each pool's floor is a quarter of the unsharded floor above.

  Stage 2 (paged-kv-design.md, "decouple cache entries from batch lanes; prefix trie") splits
  what Stage 1 called a "slot" into a `max_batch`-sized pool of decode lanes (this file, the
  live-state pool above and `self.block_tables`) and an independently sized directory of
  cached sessions (`session_cache.SessionCache`, driven by `scheduler.py`), which is why the
  snapshot pool above is sized by `num_snapshots`, not `max_batch`: see those two modules'
  own docstrings for the full design and for why `Model.save_prefix`/`load_prefix` and
  `cache_logits`/`cached_logits` became `save_snapshot`/`load_snapshot` and
  `cache_node_logits`/`cached_node_logits`, addressed by snapshot id instead of by lane.
"""

import json
import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import block_pool
import decode_attn_splitk
import decode_attn_v2
import decode_glue
import decode_stamps
import deltanet_chunked
import deltanet_fused
import deltanet_prefill_chunked
import deltanet_tp
import hot_experts
import moe_hip
import mem_timeline
import mtp
import mxfp4_gemv
import mxfp4_moe_bw2
import paged_attn
import prefill_moe
import rmsnorm_fused
import router_fused
import skinny_gemm
import skinny_hip
import step_timing
import torch
import torch.nn.functional as F
import varlen_prefill_attn
from mxfp4 import dequant_mxfp4
from pipeline import StagePipeline
from tp import TP, Plan, Shard
from weights import Checkpoint

# `mtp` imports this module back (`import model as _m`) to reuse `Cfg`, `_Reader`,
# `load_attention`, `rmsnorm`, ...; safe because both sides only touch the other module's
# attributes from inside function bodies, never at import time (see mtp.py's own docstring).

PREFILL_CHUNK = 512
LOAD_THREADS = int(os.environ.get("SEED_LOAD_THREADS", "32"))
PREFIX = "model.language_model."

# ---------------------------------------------------------------- fault injection
# Env-gated, off by default, byte-identical to the unmodified path when unset. Each hook
# is a deliberately broken variant used only to calibrate the accuracy gate's thresholds
# (accuracy_checker/README.md "Calibration"): it must be caught, not tolerated.
FAULT_ROPE_BASE = os.environ.get("SEED_FAULT_ROPE_BASE", "0") != "0"
"""Wrong RoPE base: theta scaled 1e-3x the checkpoint's own value (1e4 instead of the real
checkpoint's 1e7), for every full-attention layer (DeltaNet layers never call `apply_rope`,
so nothing further scopes this)."""

FAULT_DROP_EXPERT = os.environ.get("SEED_FAULT_DROP_EXPERT", "0") != "0"
"""Silently drop routed expert 0 from every token's top-k assignment, every layer, with no
renormalization of the remaining weights (a missing contribution, not a rescaled one)."""

FAULT_SKIP_DELTANET_GATE = os.environ.get("SEED_FAULT_SKIP_DELTANET_GATE", "0") != "0"
"""Skip DeltaNet's output gate (`silu(z)` in `gated_rmsnorm`): pass the ungated value through."""

FAULT_FP8_KV_NO_SCALE = os.environ.get("SEED_FAULT_FP8_KV_NO_SCALE", "0") != "0"
"""Truncate full-attention K/V cache writes to fp8 (e4m3) with no scale factor, naive
round-trip truncation rather than proper quantization."""

FAULT_FLIP_CAUSAL_MASK = os.environ.get("SEED_FAULT_FLIP_CAUSAL_MASK", "0") != "0"
"""Make one full-attention layer's mask non-causal (queries also attend future keys)."""
FAULT_FLIP_CAUSAL_MASK_LAYER = int(os.environ.get("SEED_FAULT_FLIP_CAUSAL_MASK_LAYER", "0"))
"""Which full-attention layer to flip, 0-based among full-attention layers only (layer 0 of
the `full_attention_interval` pattern, not the global layer index)."""

FUSE_GLUE = os.environ.get("SEED_FUSE_GLUE", "0") not in ("0", "false", "False")
"""Off by default. Collapses per-tensor decode "glue" (RoPE, masked KV-cache/state writes,
the MoE router) into fewer, larger Triton kernels rather than the chain of small torch ops
each replaces. See `attn_decode_fused.py`, `deltanet_fused.py`'s `active`/`masked_row_copy`,
and `router_fused.py` for each fusion's own docstring and what it costs/saves. This is the
per-tensor-glue flag; `SEED_FUSED_AR_NORM` (`tp.py`) is the separate collective/residual/norm
fusion. `Model.moe`'s router fusion is the one call site in this file that reads it directly;
the decode-attention/DeltaNet glue lives in `graph_decode.py`'s own copy of this flag (the
eager batched-decode path, `Model.attn_decode`/`Model.deltanet_decode`, only ever runs real
rows and has no masking to fuse away, so it does not need this flag at all)."""

DN_CONV_INPLACE = os.environ.get("SEED_DN_CONV_INPLACE", "0") not in ("0", "false", "False")
"""Off by default. `graph_decode.DN_CONV_INPLACE` reads the same variable; this copy puts the
eager `Model.deltanet_decode` T=1 step on the same conv kernel so graph validation stays exact."""

MTP_VERIFY_SKINNY_PROJ = os.environ.get("SEED_MTP_VERIFY_SKINNY_PROJ", "0") not in (
    "0",
    "false",
    "False",
)
"""Diagnostic: use decode's `skinny_hip.linear` spelling for wide DeltaNet verify projections.

This isolates projection-kernel differences from the wide recurrent kernel. It is deliberately
limited to eager `deltanet_verify`; the captured MTP implementation has its own static path.
"""

MTP_VERIFY_DN_SEQUENTIAL = os.environ.get("SEED_MTP_VERIFY_DN_SEQUENTIAL", "0") not in (
    "0",
    "false",
    "False",
)
"""Diagnostic: advance wide verify's DeltaNet recurrence with `t` decode-kernel calls.

The surrounding projection, convolution, output norm, and output projection remain wide. This
therefore measures only how much `fused_recurrent_prefill` contributes to eager verify drift.
"""

MTP_VERIFY_DN_EXACT = os.environ.get("SEED_MTP_VERIFY_DN_EXACT", "0") not in (
    "0",
    "false",
    "False",
)
"""Use the fixed-T4 decode-order recurrence for verify forward and accepted-prefix rollback."""

MTP_VERIFY_STEP_ROWS = os.environ.get("SEED_MTP_VERIFY_STEP_ROWS", "0") not in (
    "0",
    "false",
    "False",
)
"""Diagnostic: run DeltaNet's rowwise projection/norm boundaries at ordinary M=B.

Wide verify normally presents B*T rows to each operation.  This arm presents each time slice
as B rows, matching four ordinary decode calls while leaving convolution and recurrence wide.
It deliberately adds three Python dispatches per boundary and is not production wiring.
"""

MTP_VERIFY_STEP_MOE = os.environ.get("SEED_MTP_VERIFY_STEP_MOE", "0") not in (
    "0",
    "false",
    "False",
)
"""Diagnostic: run target verify's MoE once per time slice at ordinary decode batch width."""

MTP_VERIFY_STEP_SHARED_MOE = os.environ.get("SEED_MTP_VERIFY_STEP_SHARED_MOE", "0") not in (
    "0", "false", "False"
)
MTP_VERIFY_STEP_ROUTED_MOE = os.environ.get("SEED_MTP_VERIFY_STEP_ROUTED_MOE", "0") not in (
    "0", "false", "False"
)

MTP_VERIFY_TRACE_DN = os.environ.get("SEED_MTP_VERIFY_TRACE_DN", "0") not in (
    "0",
    "false",
    "False",
)
"""Retain internal DeltaNet verify tensors in its rollback snapshot for diagnostic probes."""


def mtp_step_moe(model: "Model", i: int, h: torch.Tensor) -> torch.Tensor:
    """Run verify MoE at ordinary decode width, with contiguous per-time inputs."""
    time_major = h.transpose(0, 1).contiguous()
    assert time_major.is_contiguous()
    return torch.stack([model.moe(i, time_major[step]) for step in range(h.shape[1])], dim=1)


def mtp_verify_moe(model: "Model", i: int, h: torch.Tensor) -> torch.Tensor:
    """Select the full or component-only per-time MoE diagnostic."""
    if MTP_VERIFY_STEP_MOE:
        return mtp_step_moe(model, i, h)
    return model.moe(
        i,
        h,
        verify_width=h.shape[1],
        step_shared=MTP_VERIFY_STEP_SHARED_MOE,
        step_routed=MTP_VERIFY_STEP_ROUTED_MOE,
    )

ROUTE_FUSED = os.environ.get("SEED_MOE_ROUTE_FUSED", "0") not in ("0", "false", "False")
"""Off by default. On the `SEED_MOE_HIP` decode path only: collapses the router
(softmax/top-k/renormalize) and the `moe_hip`/`bw_prep` work-list build into one router-kernel
launch (`router_fused.route` with `index_dtype=torch.int32`, so its `top_i` needs no separate
cast before `mxfp4_gemv.bw_prep` reads it as `bw_prep`'s `a_expert`), and folds the
shared-expert sigmoid-gate multiply and the final add into the HIP kernels' combine launch
(`moe_hip.fused_moe_hip_glued` -> `mxfp4_gemv.bw_combine_glue`) instead of doing them as
separate elementwise ops in `moe`. See `_moe_hip_route_fused`'s docstring for the exact kernel
count before/after and the routing-equivalence caveat it inherits from `router_fused.py`
(`SEED_FUSE_GLUE`'s tie-break risk: `tl.argmax` need not break a softmax tie the same way
`torch.topk` does). Requires `SEED_MOE_HIP=1`; independent of `SEED_FUSE_GLUE` (this flag
never falls through to the torch softmax/topk/renorm chain `SEED_FUSE_GLUE` also replaces)."""

PREP_FORK = os.environ.get("SEED_MOE_PREP_FORK", "0") == "1"
"""Off by default. In `_moe_hip_route_fused` (`SEED_MOE_ROUTE_FUSED`'s decode path), runs
`router_fused.route` and `moe_hip.prep` (the work-list build) on a side stream, concurrently
with the shared expert's GEMMs (`swiglu_mlp`), which do not depend on routing: a graph
fork/join, decode steps only (`out_buf is not None`). Bit-exact: the two streams compute
disjoint outputs and are joined (`cur.wait_stream(side)`) before the combine launch reads
both. Saves the shared expert's dense-GEMM latency that the routing/prep launches would
otherwise sit in front of."""

PACKED_PREFILL_VEC = os.environ.get("SEED_PACKED_PREFILL_VEC", "0") not in ("0", "false", "False")
"""Off by default. `deltanet_packed` runs its projections, causal conv, and (on an
accelerator) delta rule once over the whole packed batch instead of once per sequence. Exact
math, but GEMMs over `total_T` rows instead of per-sequence rows can round differently on an
accelerator, so it stays an A/B knob rather than a silent default."""

BATCHED_PREFILL = os.environ.get("SEED_BATCHED_PREFILL", "1") not in ("0", "false", "False")
"""Whether `prefill_batch` actually packs several requests' chunks into one forward call.

Off (`SEED_BATCHED_PREFILL=0`) makes `prefill_batch` a plain per-call loop over `prefill`,
functionally identical to the packed path but with no cross-request amortization: the A/B
knob for measuring the packed path against the one-request-at-a-time path it replaces.
`scheduler.py` reads the same variable to decide whether to build packed batches at all, so
turning it off disables the feature at both layers with one flag."""

OVERLAP_SCHED = os.environ.get("SEED_OVERLAP_SCHED", "0") not in ("0", "", "false", "False")
"""Overlap the scheduler's host work with the device's decode step (`scheduler.py` reads the
same variable; see its `OVERLAP_SCHED` docstring for the design). On this side it does two
things: `decode_launch` keeps sampling on the device and feeds the sampled ids straight into
the next step (`lane_tokens`), and the decode path's host-to-device copies go through
`host_to_device`/`copy_from_host`, which stage through pinned memory with `non_blocking=True`.
A plain `torch.tensor(values, device=cuda)` is a pageable copy that synchronizes the stream,
which would put back the very bubble the overlap removes. Off, both helpers are the plain
synchronous spelling, so the default path is unchanged."""

PREFILL_FUSED_IN_PROJ = os.environ.get("SEED_PREFILL_FUSED_IN_PROJ", "0") not in (
    "0",
    "",
    "false",
    "False",
)
"""Eager `Model.deltanet` (per-request prefill) runs its four input projections as one GEMM
against `in_proj_all` plus a split, as decode and the captured prefill already do. Saves 3
launches per DeltaNet layer (135 per call) and, at M >= 128, the default-tile cost of the
N=16 `in_proj_b`/`in_proj_a` GEMMs (~34 us each regardless of M): measured on one MI300A at
M=128, 167 us for the four vs 35 us fused."""

LOOKAHEAD_TOKEN = -1
"""A `decode_launch` token id meaning "this lane's last sampled token", still on the device
in `Model.lane_tokens` and not yet read back by the host. Mirrors `scheduler.LOOKAHEAD`."""


def host_to_device(
    values: object, device: torch.device, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """`torch.tensor(values, dtype=dtype, device=device)`, without a stream sync when
    `OVERLAP_SCHED` is on and `device` is a GPU (see the flag's docstring). The pinned staging
    tensor stays alive until the copy completes: PyTorch's caching host allocator records the
    copy's stream event and does not reuse the block before it fires."""
    t = torch.tensor(values, dtype=dtype)
    if OVERLAP_SCHED and device.type == "cuda":
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)


def copy_from_host(dst: torch.Tensor, src: torch.Tensor) -> None:
    """`dst.copy_(src)` for a host `src`, non-blocking under `OVERLAP_SCHED` like
    `host_to_device`."""
    if OVERLAP_SCHED and dst.is_cuda:
        dst.copy_(src.pin_memory(), non_blocking=True)
    else:
        dst.copy_(src)


def sample_on_device(logits: torch.Tensor, temperatures: Sequence[float]) -> torch.Tensor:
    """One token id per row of [B, vocab], left on the device ([B] int64).

    `Model.sample_batch`'s arithmetic with the `.tolist()` taken off, so the overlap path and
    the synchronous path draw the same tokens (and consume the same RNG) for the same batch.
    Whether a row is greedy is known on the host, so no branch here reads the device."""
    if all(t <= 0 for t in temperatures):
        return logits.argmax(-1)
    scale = host_to_device([t if t > 0 else 1.0 for t in temperatures], logits.device, logits.dtype)
    drawn = torch.multinomial((logits / scale[:, None]).softmax(-1), 1)[:, 0]
    greedy = host_to_device([t <= 0 for t in temperatures], logits.device)
    return torch.where(greedy, logits.argmax(-1), drawn)


class PendingTokens:
    """One launched decode step's result as `scheduler.Scheduler` sees it under overlap.

    The sampled ids are copied device-to-host into pinned memory with `non_blocking=True` and
    an event recorded behind the copy, so `tokens()` waits only for *this* step, not for
    whatever the host has enqueued after it (the next step, already launched). `row(i)` is
    the step's [1, vocab] logits row for batch position `i`, for a turn-close publish; it is a
    device tensor that may still be being computed, which is fine for the stream-ordered
    `cache_node_logits` copy that consumes it.
    """

    def __init__(self, tokens: torch.Tensor, logits: torch.Tensor) -> None:
        self._logits = logits
        self._event = None
        if tokens.is_cuda:
            self._host = torch.empty(tokens.shape, dtype=tokens.dtype, pin_memory=True)
            self._host.copy_(tokens, non_blocking=True)
            self._event = torch.cuda.Event()
            self._event.record(torch.cuda.current_stream(tokens.device))
        else:
            self._host = tokens.clone()

    def tokens(self) -> list[int]:
        if self._event is not None:
            self._event.synchronize()
        return self._host.tolist()

    def row(self, i: int) -> torch.Tensor:
        return self._logits[i : i + 1]


KV_BLOCK_SIZE = int(os.environ.get("SEED_KV_BLOCK_SIZE", "16"))
"""Tokens per KV block for the paged full-attention pool (`paged-kv-design.md` section 1.3
recommends 16 as the default reuse-granularity/overhead trade-off). A flag, not a fixed
constant, so it can be swept; see `block_pool.py` for the block-table side of what a larger
or smaller value costs and `paged_attn.py` for the kernel-iteration side."""

KV_POOL_MIN_CAPACITY_FACTOR = float(os.environ.get("SEED_KV_POOL_MIN_CAPACITY_FACTOR", "1.0"))
"""Floor on block-pool capacity, as a multiple of today's `max_batch * max_seq` token ceiling --
this Stage 1's own requirement ("capacity equivalent to today's 48 x max_seq_len at minimum").
1.0 is that floor exactly; nothing below sizes the pool under this, only above it."""

SNAPSHOT_POOL_MIN_CAPACITY_FACTOR = float(
    os.environ.get("SEED_SNAPSHOT_POOL_MIN_CAPACITY_FACTOR", "1.0")
)
"""Floor on snapshot-pool capacity, as a multiple of `max_batch`. 1.0 (the default) is at least
one snapshot slot per lane -- Stage 1's old one-snapshot-per-lane capacity, kept as the floor a
configuration cannot fall under."""

KV_POOL_HEADROOM_FACTOR = float(os.environ.get("SEED_KV_POOL_HEADROOM_FACTOR", "1.5"))
"""Multiple of the `KV_POOL_MIN_CAPACITY_FACTOR` floor requested as the KV pool's "need",
before the hard cap (`KV_POOL_CAP_GIB`) clamps it. This is capacity for cached-but-idle
sessions sitting in the prefix trie beyond the `max_batch` lanes actively decoding, not a
free-memory dividend: MI300A is a unified-memory APU, so "free" HBM measured at load time
also belongs to host RSS that grows during serving (tokenizer, aiohttp buffers, checkpoint
page cache, pinned staging) -- sizing this pool as a fraction of *free* memory, as an earlier
version of this function did, already OOM'd a real boot even though the KV pool alone (60.5 of
72 GiB free) looked like it fit with an 11.6 GiB margin at load time. Size to need, capped
hard, never to a fraction of a measurement that keeps shrinking after boot."""

KV_POOL_CAP_GIB = float(os.environ.get("SEED_KV_POOL_CAP_GIB", "20"))
"""Hard ceiling on the KV pool, GiB/rank, applied after `KV_POOL_HEADROOM_FACTOR`. ~15 KB/token/
rank means 16 GiB is already ~1.1M tokens: far more concurrent context than `max_batch` lanes
can use at once. `SEED_KV_POOL_GIB` overrides the whole computation (need *and* cap) with an
exact size when a deployment wants to hand-pick it."""

LMHEAD_VOCAB_TP = os.environ.get("SEED_LMHEAD_VOCAB_TP", "0") not in ("0", "", "false", "False")
"""Under TP, each rank holds a quarter of `lm_head`'s vocabulary rows and `Model.unembed`
all-gathers the logit columns, instead of every rank running the whole head. The replicated
head is the step's largest single GEMM (0.92 ms per step at b48 on one MI300A, 2 GB of bf16
weight); the gather moves `B x vocab` bf16 logits. Every logit is the same dot product over
`hidden` as before, so the logits are bit-exact when the sharded GEMM's kernel reduces over
`hidden` in the same order (`scratchpad/lmhead_split_check.py` checks that on the GPU). Every
rank still ends with the full logits, so sampling, logprobs and `/v1/score` are unchanged."""

KV_POOL_GIB_OVERRIDE = os.environ.get("SEED_KV_POOL_GIB")
"""Exact KV pool size in GiB/rank, bypassing `KV_POOL_HEADROOM_FACTOR`/`KV_POOL_CAP_GIB`
entirely. Unset by default; still cannot go below the `KV_POOL_MIN_CAPACITY_FACTOR` floor."""

SNAPSHOT_POOL_COUNT = int(os.environ.get("SEED_SNAPSHOT_POOL_COUNT", "256"))
"""Fixed snapshot-pool capacity (slot count), independent of measured free memory. At ~45 MiB/
rank/snapshot (see `_snapshot_bytes`) 256 is ~11 GiB/rank -- "hundreds of cached sessions", the
actual design target. `SEED_SNAPSHOT_POOL_GIB` overrides this with a GiB budget instead of a
slot count (converted via `_snapshot_bytes`); either way the `SNAPSHOT_POOL_MIN_CAPACITY_FACTOR`
floor still applies."""

SNAPSHOT_POOL_GIB_OVERRIDE = os.environ.get("SEED_SNAPSHOT_POOL_GIB")
"""GiB/rank budget for the snapshot pool, converted to a slot count via `_snapshot_bytes`.
Takes precedence over `SNAPSHOT_POOL_COUNT` when set."""

MIN_UNALLOCATED_GIB = float(os.environ.get("SEED_MIN_UNALLOCATED_GIB", "25"))
"""Minimum GiB/rank of unified memory `_size_pools` tries to leave unallocated after the KV and
snapshot pools, on top of weights/DeltaNet live-state/scratch already resident: headroom for
host RSS (unified with HBM on MI300A) plus transient activations (prefill at 2k-token packed
batches, MoE workspaces). If the requested pools would not leave this much, the KV pool (never
the snapshot pool, which is a small, fixed, already-conservative budget) is shrunk toward its
floor to make room; if even the floors together do not leave this much, `_size_pools` logs a
warning and proceeds at the floors rather than refusing to boot."""

MICROBATCHES_PER_STAGE = int(os.environ.get("SEED_DECODE_MICROBATCHES", "2"))
"""Decode microbatches per pipeline stage; 0 runs the stages sequentially (no pipelining).

With S stages a step is cut into at most S * this many microbatches (see
`MIN_MICROBATCH_SLOTS` below for why "at most", and why that ceiling is not reached at any
batch size this currently deploys at). Raising it is not free: each microbatch replays the
whole per-layer Python and kernel launch sequence, and its MoE re-reads the weights of every
expert it activates, which a larger microbatch would have shared. This constant is a
pipeline-depth ceiling for whenever a deployment's `max_batch` grows past the point
`MIN_MICROBATCH_SLOTS` allows one undivided batch; it does nothing at the batch sizes
measured so far. `benchmark/` is where a future sweep at a larger `max_batch` belongs.
"""

MIN_MICROBATCH_SLOTS = int(os.environ.get("SEED_DECODE_MIN_MICROBATCH_SLOTS", "49"))
"""Floor on slots per decode microbatch; below it, `decode` uses fewer, larger microbatches.

`MICROBATCHES_PER_STAGE` sets a *ceiling* of `len(stages) * MICROBATCHES_PER_STAGE`
microbatches (8 at the default), applied regardless of the step's actual slot count. Once a
step's batch size reaches that ceiling, every microbatch gets exactly 1 slot: nothing left to
batch, only `stages` full per-layer dispatch and per-microbatch MoE weight-read passes
stacked back to back. Measured on 4x MI300A with the real checkpoint, that turned an 8x
smaller step into an 8.5x *slower* one (112ms/tok at batch 1 to 960ms/tok at batch 8, tok/s
flat at 8-9) and made batch 16 non-monotonic against batch 8, because 16 slots split 8 ways
is less degenerate than 8 slots split 8 ways.

`_decode_group_count` keeps at least this many slots in every microbatch, falling back toward
fewer, larger microbatches (down to one undivided batch) instead of hitting the ceiling on a
step too small to spend it on. That alone (floor 6, matched to the pre-existing 48-slot / 8
microbatch split) fixed the 8-slot regression, but a follow-up sweep at the deployed
`max_batch` of 48 -- the batch size actually measured, not assumed -- found the pipelining
this module was built around is a net loss at every microbatch count tried, not merely
expensive at the smallest one:

    min_slots  groups  ms/tok   tok/s
            6       8  1233.77     39   (the pre-existing split)
            8       6   992.23     48
           12       4   677.15     71
           16       3   577.30     83
           24       2   428.58    112
           49       1   311.15    154   (undivided)

Monotonic all the way to undivided: 4.0x faster and 4.0x higher tok/s than the split this
module was originally tuned to use, at the same batch size that tuning targeted. The fused
MXFP4 GEMV, batched attention and batched DeltaNet landed since that tuning all cut the
per-layer cost the pipeline's fill/drain bubble was built to hide, while a narrower MoE call
still re-reads more of the (now cheaper, but not free) expert weight traffic -- at this
model's shape, on this hardware, today, splitting no longer pays for itself at any batch size
up to 48. The default here (49) makes every currently-deployed batch size (1-48) undivided;
`MICROBATCHES_PER_STAGE` and the pipeline-thread machinery stay in place; either can be
lowered again if a future `max_batch` or workload shape reopens the case for overlap -- resweep
before assuming it still holds.
"""

DELTA_CHUNK = 64
"""Tokens per chunk in the parallel delta rule. Within a chunk the work is matmuls plus one
triangular solve; the recurrent state is carried sequentially only from chunk to chunk."""


@dataclass(frozen=True)
class Cfg:
    hidden: int
    vocab: int
    layer_types: tuple[str, ...]
    eps: float
    # full attention
    heads: int
    kv_heads: int
    head_dim: int
    rot_dim: int
    rope_theta: float
    # gated deltanet
    k_heads: int
    v_heads: int
    k_dim: int
    v_dim: int
    conv_k: int
    # moe
    experts: int
    top_k: int
    shared_inter: int
    eos: tuple[int, ...]


def load_cfg(model_dir: str | Path) -> Cfg:
    raw = json.loads((Path(model_dir) / "config.json").read_text())
    t = raw.get("text_config", raw)
    rope = t.get("rope_parameters", {})
    head_dim = t["head_dim"]
    eos = t.get("eos_token_id", ())
    return Cfg(
        hidden=t["hidden_size"],
        vocab=t["vocab_size"],
        layer_types=tuple(t["layer_types"]),
        eps=t["rms_norm_eps"],
        heads=t["num_attention_heads"],
        kv_heads=t["num_key_value_heads"],
        head_dim=head_dim,
        rot_dim=int(
            head_dim * rope.get("partial_rotary_factor", t.get("partial_rotary_factor", 1.0))
        ),
        rope_theta=rope.get("rope_theta", t.get("rope_theta", 1e7)),
        k_heads=t["linear_num_key_heads"],
        v_heads=t["linear_num_value_heads"],
        k_dim=t["linear_key_head_dim"],
        v_dim=t["linear_value_head_dim"],
        conv_k=t["linear_conv_kernel_dim"],
        experts=t["num_experts"],
        top_k=t["num_experts_per_tok"],
        shared_inter=t["shared_expert_intermediate_size"],
        eos=(eos,) if isinstance(eos, int) else tuple(eos),
    )


# ---------------------------------------------------------------- small math helpers


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3.5 RMSNorm: normalize in fp32, scale by (1 + w).

    Dispatches to `rmsnorm_fused` on an accelerator (one Triton kernel instead of six unfused
    ops; see that module's docstring for the measurement). This is the one place that decision
    is made: every call site -- eager single-sequence, eager batched decode, and the captured
    static path in `graph_decode.py`, which imports this function directly -- goes through it,
    so there is no second call site to keep in sync and no risk of the fused path landing in
    eager while the static path silently keeps the torch chain (the graph-capture port has been
    burned by exactly that shape of bug before, in `deltanet_decode_static` and
    `attn_decode_static`). The Python-level `available` check is a device-type read, not a
    device-value read, so it is capture-safe: it takes the same branch on every replay, the same
    way `deltanet_fused.available` already does inside `deltanet_decode_static`.
    """
    if rmsnorm_fused.available(x.device):
        return rmsnorm_fused.rmsnorm(x, w, eps)
    y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (y * (1.0 + w.float())).type_as(x)


DN_NORM_F32_IN = os.environ.get("SEED_DN_NORM_F32_IN", "0") not in ("0", "false", "False")
"""Off by default. The DeltaNet decode output (fp32 off `delta_rule_decode`) goes straight into
`deltanet_fused.gated_rmsnorm`, which rounds it to bf16 in-kernel, instead of through a separate
`.to(bf16)` launch first (`out_dtype`). Bit-exact; one launch per DeltaNet layer, 45 a step."""


def gated_rmsnorm(x: torch.Tensor, gate: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """DeltaNet output norm: plain weight (no +1), then silu(gate)."""
    y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    y = w * y.to(x.dtype)
    if FAULT_SKIP_DELTANET_GATE:  # pass the ungated value through
        return y.to(x.dtype)
    return (y * F.silu(gate.float())).to(x.dtype)


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the first rot_dim channels of x [1, heads, T, head_dim]; pass the rest through."""
    r = cos.shape[-1]
    rot, rest = x[..., :r], x[..., r:]
    return torch.cat([rot * cos + rotate_half(rot) * sin, rest], dim=-1)


def swiglu_mlp(x: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    """SwiGLU against `gate_up`, the gate and up projections concatenated (`GATE_UP_ORDER`).

    `SEED_SKINNY_SWIGLU`: the down projection reads `gate_up`'s output directly and applies
    `silu(gate) * up` in its input load (`skinny_gemm`), same two bf16 rounding points.
    """
    if skinny_gemm.SWIGLU and skinny_gemm.available(x, down, force=True):
        return skinny_gemm.skinny_linear(F.linear(x, gate_up), down, act="swiglu")
    if decode_glue.available(x):  # `SEED_ELEMWISE_FUSED`: silu and mul as one launch
        return skinny_hip.linear(decode_glue.silu_mul(skinny_hip.linear(x, gate_up)), down)
    gate, up = skinny_hip.linear(x, gate_up).chunk(2, dim=-1)
    return skinny_hip.linear(F.silu(gate) * up, down)


# ---------------------------------------------------------------- gated delta rule
#
# Both forms below implement the same recurrence over a sequence of tokens t, per head, with
# state S [k_dim, v_dim] (see reference/modeling_qwen3_5_moe.py):
#
#     S_t = exp(g_t) * S_{t-1}
#     S_t = S_t + k_t ((v_t - S_t^T k_t) * beta_t)^T      = S_t (I - beta_t k_t k_t^T) + beta_t k_t v_t^T
#     out_t = S_t^T q_t
#
# with q, k l2-normalized, q additionally scaled by k_dim ** -0.5, beta_t = sigmoid(...) in (0, 1),
# and the forget gate g_t = -exp(A_log) * softplus(a_t + dt_bias) <= 0. Everything runs in fp32.


def _delta_rule_inputs(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp32 delta-rule inputs: l2-normalized q (times the 1/sqrt(k_dim) scale) and k."""
    return (
        l2norm(q.float()) * q.shape[-1] ** -0.5,
        l2norm(k.float()),
        v.float(),
        beta.float(),
    )


def delta_rule_recurrent(q, k, v, g, beta, rec, out=None) -> torch.Tensor:  # noqa: ANN001
    """Gated delta rule, one token at a time. q,k,v: [1,T,H,d]; g,beta: [1,T,H].

    Returns fp32 outputs [1,T,H,v_dim] and advances `rec` [1,H,k_dim,v_dim] in place.

    `out`, when given, is a preallocated buffer matching v's post-scale shape/dtype that
    every position gets written into (decode's T=1 call writes the whole buffer, so there is
    nothing to leak from a previous call); omitting it allocates fresh, as before.
    """
    q, k, v, beta = _delta_rule_inputs(q, k, v, beta)
    if out is None:
        out = torch.empty_like(v)
    for s in range(q.shape[1]):
        rec.mul_(g[:, s].exp()[..., None, None])
        mem = (rec * k[:, s, :, :, None]).sum(-2)
        delta = (v[:, s] - mem) * beta[:, s, :, None]
        rec.add_(k[:, s, :, :, None] * delta[:, :, None, :])
        out[:, s] = (rec * q[:, s, :, :, None]).sum(-2)
    return out


def _delta_rule_chunks(q, k, v, g, beta, chunk: int) -> tuple[torch.Tensor, ...]:  # noqa: ANN001
    """Split [1,T,H,d] inputs into [1,H,NC,chunk,d], right-padding the last chunk.

    Padded positions get k=v=beta=0 and g=0, which makes their rows of the within-chunk
    system zero, so they contribute nothing to the outputs or to the carried state.
    `g` comes back as the within-chunk cumulative sum.
    """
    t = q.shape[1]
    pad = -t % chunk
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    g, beta = g.float().transpose(1, 2), beta.transpose(1, 2)
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        g, beta = F.pad(g, (0, pad)), F.pad(beta, (0, pad))
    nc = (t + pad) // chunk
    q, k, v = (x.unflatten(2, (nc, chunk)) for x in (q, k, v))
    return q, k, v, g.unflatten(2, (nc, chunk)).cumsum(-1), beta.unflatten(2, (nc, chunk))


def delta_rule_chunked(q, k, v, g, beta, rec) -> torch.Tensor:  # noqa: ANN001
    """Gated delta rule in the chunked parallel (WY-representation) form.

    Same contract as `delta_rule_recurrent`. Writing the `chunk` successive rank-1 updates of one
    chunk as a single linear system in the within-chunk outputs turns the per-token loop into one
    unit-lower-triangular solve plus matmuls; only the state hand-off between chunks stays serial.
    """
    t, dv = q.shape[1], v.shape[-1]
    chunk = min(DELTA_CHUNK, t)
    q, k, v, beta = _delta_rule_inputs(q, k, v, beta)
    q, k, v, g, beta = _delta_rule_chunks(q, k, v, g, beta, chunk)

    # decay[i,j] = exp(G_i - G_j) for j <= i else 0, G the within-chunk cumulative gate (<= 0).
    decay = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().tril()
    k_beta, v_beta = k * beta[..., None], v * beta[..., None]
    # (I + A) [U W] = [v_beta, k_beta*exp(G)] with A strictly lower triangular: U is the chunk's
    # own delta contribution, W maps the incoming state onto the same basis.
    a = ((k_beta @ k.transpose(-1, -2)) * decay).tril(-1)
    rhs = torch.cat([v_beta, k_beta * g.exp()[..., None]], dim=-1)
    u, w = torch.linalg.solve_triangular(a, rhs, upper=False, unitriangular=True).split(
        [dv, k.shape[-1]], dim=-1
    )

    state, out = rec.clone(), torch.empty_like(v)
    for i in range(q.shape[2]):
        q_i, k_i, g_i = q[:, :, i], k[:, :, i], g[:, :, i]
        v_new = u[:, :, i] - w[:, :, i] @ state
        intra = (q_i @ k_i.transpose(-1, -2)) * decay[:, :, i]
        out[:, :, i] = (q_i * g_i[..., None].exp()) @ state + intra @ v_new
        tail = g_i[..., -1:]  # total decay across the chunk
        state = (
            state * tail[..., None].exp()
            + (k_i * (tail - g_i)[..., None].exp()).transpose(-1, -2) @ v_new
        )
    rec.copy_(state)
    return out.flatten(2, 3)[:, :, :t].transpose(1, 2).contiguous()


# ---------------------------------------------------------------- weight loading


def load_experts(ck: Checkpoint, p: str, sh: Shard, dev: torch.device, dtype: torch.dtype) -> dict:
    """Stack this rank's experts into [sh.count, ...] tensors. MXFP4 keeps uint8 + scale.

    Expert parallel: rank r reads experts [sh.start, sh.stop) and never touches the rest, so
    the four ranks read the 205 GB of MXFP4 expert weights once between them instead of once
    each. Local expert index i is global expert `sh.start + i`, which is the translation
    `mxfp4_gemv.fused_moe` does GPU-side from `Model.expert_range` and `_routed_grouped` does
    on the torch path. Only the expert axis is sliced, so the MXFP4 blocking (along K, 32
    values a block) is untouched and neither `dequant_mxfp4` nor the fused kernels change.

    `sh` is the whole axis for the unsharded model, so that path is unchanged.
    """
    quant = ck.has(f"{p}.experts.0.gate_proj.weight_scale")

    def stack(read) -> torch.Tensor:
        first = read(sh.start)
        out = torch.empty((sh.count, *first.shape), dtype=first.dtype, device=dev)
        out[0] = first

        def fill(i: int) -> None:
            out[i] = read(sh.start + i)

        with ThreadPoolExecutor(LOAD_THREADS) as pool:  # parallel reads: one reader gets ~45 MB/s
            list(pool.map(fill, range(1, sh.count)))
        return out

    keep = (
        None if quant else dtype
    )  # uint8 payloads stay uint8; dense checkpoints cast to model dtype

    def ld(e: int, proj: str, suffix: str = "weight") -> torch.Tensor:
        return ck.load(
            f"{p}.experts.{e}.{proj}.{suffix}", dev, keep if suffix == "weight" else None
        )

    ex = {
        "gate_up": stack(lambda e: torch.cat([ld(e, "gate_proj"), ld(e, "up_proj")], 0)),
        "down": stack(lambda e: ld(e, "down_proj")),
    }
    if quant:
        ex["gate_up_scale"] = stack(
            lambda e: torch.cat(
                [ld(e, "gate_proj", "weight_scale"), ld(e, "up_proj", "weight_scale")], 0
            )
        )
        ex["down_scale"] = stack(lambda e: ld(e, "down_proj", "weight_scale"))
    return ex


class _Reader:
    """Reads one layer's tensors, keeping only this rank's shard of each.

    The three methods are the three cases of the convention in `tp.py`: `rows` for a
    column-parallel weight (keep the output rows this rank's heads or channels own), `cols`
    for a row-parallel one (keep the matching input columns), `whole` for a replicated one.
    A sharded read lands on the host and is sliced there, so the full tensor is never
    materialized on the device. `Checkpoint.load` reads into host memory either way, so at
    `world == 1`, where every slice is the whole axis, this costs one extra host copy per
    weight and no extra I/O.
    """

    def __init__(self, ck: Checkpoint, prefix: str, dev: torch.device, dtype: torch.dtype) -> None:
        self.ck, self.prefix, self.dev, self.dtype = ck, prefix, dev, dtype

    def whole(self, name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        return self.ck.load(f"{self.prefix}.{name}", self.dev, dtype or self.dtype)

    def rows(self, name: str, *parts: slice) -> torch.Tensor:
        """Keep `parts` of the output axis, concatenated in the order given."""
        t = self.ck.load(f"{self.prefix}.{name}", "cpu", self.dtype)
        return torch.cat([t[s] for s in parts]).to(self.dev)

    def cols(self, name: str, part: slice) -> torch.Tensor:
        """Keep `part` of the input axis (dim 1)."""
        t = self.ck.load(f"{self.prefix}.{name}", "cpu", self.dtype)
        return t[:, part].contiguous().to(self.dev)


def load_attention(rd: _Reader, cfg: Cfg, pl: Plan) -> dict:
    """Head-parallel full attention: q/k/v column-parallel by head, `o_proj` row-parallel.

    `q_proj` emits `2 * head_dim` rows per query head (the query itself and this model's
    output gate), so one head's rows are one contiguous block of that width and the gate
    follows its head. `k_proj`/`v_proj` follow `pl.kv`, which at TP=4 is the one KV head this
    rank's query block reads (see `tp.kv_shard`). Both norms are over `head_dim`, an axis
    nothing shards, so they stay whole.
    """
    d = cfg.head_dim
    return {
        "q_proj": rd.rows("self_attn.q_proj.weight", pl.q.rows(2 * d)),
        "k_proj": rd.rows("self_attn.k_proj.weight", pl.kv.rows(d)),
        "v_proj": rd.rows("self_attn.v_proj.weight", pl.kv.rows(d)),
        "o_proj": rd.cols("self_attn.o_proj.weight", pl.q.rows(d)),
        "q_norm": rd.whole("self_attn.q_norm.weight"),
        "k_norm": rd.whole("self_attn.k_norm.weight"),
    }


IN_PROJ_ORDER = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
"""Output-axis order of `in_proj_all`. `in_proj_sizes` derives the split from `Cfg`."""


def in_proj_sizes(cfg: Cfg) -> list[int]:
    """Output widths of `IN_PROJ_ORDER` for `cfg`, which must be this rank's local config."""
    return [
        2 * cfg.k_heads * cfg.k_dim + cfg.v_heads * cfg.v_dim,
        cfg.v_heads * cfg.v_dim,
        cfg.v_heads,
        cfg.v_heads,
    ]


def fuse_rows(layer: dict, fused_name: str, order: Sequence[str]) -> dict:
    """Concatenate `order`'s weights on the output axis and add them as `fused_name`.

    Projections that read the same activation can share one `F.linear` plus a split of the
    result. That is worth doing twice over: it removes dispatches, and it widens N, which is
    what lifts the GEMM off the per-call floor a skinny (M = batch) GEMM otherwise sits on
    (`blas_tune`).

    Each name in `order` stays in the dict as a *view* into the concatenation, so no weight is
    stored twice and readers of the individual projections -- the seed tests, `graph_decode`'s
    own attention spelling -- are unchanged. A layer that holds none of them passes through,
    which is how one call covers both mixer types.
    """
    if any(name not in layer for name in order):
        return layer
    parts = [layer[name] for name in order]
    fused = torch.cat(parts, dim=0)
    out = dict(layer, **{fused_name: fused})
    at = 0
    for name, part in zip(order, parts, strict=True):
        out[name] = fused[at : at + part.shape[0]]
        at += part.shape[0]
    return out


def fuse_in_proj(layer: dict) -> dict:
    """Add `in_proj_all`, the four DeltaNet input projections concatenated on the output axis.

    All four read the same `x`, so one `F.linear` plus a split of the result replaces four
    calls. That is three fewer dispatches per DeltaNet layer, 135 per decode step, which is
    what matters on a step limited by the rate the host can issue work rather than by device
    time (`TP_BYTELUT_BOTTLENECK_2026-09-22.md` section 1).
    """
    return fuse_rows(layer, "in_proj_all", IN_PROJ_ORDER)


ROUTER_GATE_ORDER = ("router", "shared_gate")
"""Output-axis order of `router_gate`: the `experts` routing logits, then the shared gate."""

GATE_UP_ORDER = ("shared_expert.gate_proj", "shared_expert.up_proj")
"""Output-axis order of `shared_expert.gate_up_proj`, matching `swiglu_mlp`'s chunk."""


def fuse_moe_dense(layer: dict) -> dict:
    """Concatenate the MoE block's two pairs of projections that read the same hidden state.

    Every layer runs four dense GEMMs against `h` -- the router, the shared expert's gate and
    up, and the shared gate -- and at decode all four are skinny (M = batch), where cost is
    set much more by the launch than by the weight bytes. Measured on one MI300A at the real
    TP=4 widths and batch 48, with solution selection already on: the router at N = 512 costs
    13.2 us, `gate_proj` and `up_proj` at N = 256 cost 21.1 and 20.0 us, and `shared_gate` at
    N = 1 costs 8.4 us for 8 KB of weights. Pairing them into one N = 513 and one N = 512 GEMM
    is 120 fewer calls per step and about 2.2 ms of the 8.09 ms the dense GEMMs cost.
    """
    return fuse_rows(
        fuse_rows(layer, "router_gate", ROUTER_GATE_ORDER),
        "shared_expert.gate_up_proj",
        GATE_UP_ORDER,
    )


def load_deltanet(rd: _Reader, cfg: Cfg, tp: TP) -> dict:
    """Gated DeltaNet, head-parallel over value heads. See `deltanet_tp` for the derivation.

    Unlike the attention and MoE weights above, which `_Reader` slices on the host while it
    reads them, the DeltaNet tensors are read whole and then sliced by
    `deltanet_tp.shard_deltanet_weights`. That costs one layer's worth of transient memory
    per rank (~200 MB in bf16 on the real model, freed as soon as this returns) and keeps
    the sharding rules in one place, next to their derivation. `cfg` must be the *global*
    config, because the tensors being sliced are the checkpoint's full-width ones.
    """
    layer = {
        n: rd.whole(f"linear_attn.{n}.weight")
        for n in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")
    }
    layer["conv"] = rd.whole("linear_attn.conv1d.weight")
    layer["A_log"] = rd.whole("linear_attn.A_log", torch.float32)
    layer["dt_bias"] = rd.whole("linear_attn.dt_bias", torch.float32)
    layer["dn_norm"] = rd.whole("linear_attn.norm.weight")
    return fuse_in_proj(deltanet_tp.shard_deltanet_weights(layer, cfg, tp))


def load_moe_dense(rd: _Reader, pl: Plan) -> dict:
    """Router and shared expert. The shared expert is plain TP over its intermediate axis.

    The router stays whole on every rank: that is what lets every rank reach the same routing
    decision from the same replicated hidden state without a collective.
    """
    cols = pl.inter.rows(1)
    return fuse_moe_dense(
        {
            "shared_expert.gate_proj": rd.rows("mlp.shared_expert.gate_proj.weight", cols),
            "shared_expert.up_proj": rd.rows("mlp.shared_expert.up_proj.weight", cols),
            "shared_expert.down_proj": rd.cols("mlp.shared_expert.down_proj.weight", cols),
            "router": rd.whole("mlp.gate.weight"),
            "shared_gate": rd.whole("mlp.shared_expert_gate.weight"),
        }
    )


def load_layer(
    ck: Checkpoint, i: int, cfg: Cfg, tp: TP, dev: torch.device, dtype: torch.dtype
) -> dict:
    """This rank's shard of layer `i`, on `dev`. `cfg` is the global (unsharded) config.

    `dev` is passed rather than taken from `tp` because the two parallelism axes are
    independent: under tensor parallelism it is always `tp.device`, but a single-process
    model splits its layers over several devices and `Model` picks the one for this layer.
    """
    p = f"{PREFIX}layers.{i}"
    rd = _Reader(ck, p, dev, dtype)
    full = cfg.layer_types[i] == "full_attention"
    return {
        "in_norm": rd.whole("input_layernorm.weight"),
        "post_norm": rd.whole("post_attention_layernorm.weight"),
        **(load_attention(rd, cfg, tp.plan) if full else load_deltanet(rd, cfg, tp)),
        **load_moe_dense(rd, tp.plan),
        "experts": load_experts(ck, f"{p}.mlp", tp.plan.experts, dev, dtype),
    }


# ---------------------------------------------------------------- the model


def _stage_ranges(layer_stage: list[int]) -> list[range]:
    """The contiguous layer runs of `layer_stage`, in execution order: one range per stage.

    Keyed on the stage number rather than the device object so that repeating a device in
    `devices` (which the CPU tests do, to exercise the pipeline without four GPUs) still
    produces one stage per entry.
    """
    ranges: list[range] = []
    previous: int | None = None
    for i, stage in enumerate(layer_stage):
        if stage == previous:
            ranges[-1] = range(ranges[-1].start, i + 1)
        else:
            ranges.append(range(i, i + 1))
            previous = stage
    return ranges


def _microbatch_cuts(n: int, groups: int) -> list[tuple[int, int]]:
    """Split `n` batch rows into `groups` contiguous [lo, hi) ranges, largest first."""
    groups = max(1, min(groups, n))
    cuts, lo = [], 0
    for j in range(groups):
        hi = lo + n // groups + (1 if j < n % groups else 0)
        cuts.append((lo, hi))
        lo = hi
    return cuts


def _decode_group_count(n: int, ceiling: int, min_slots: int) -> int:
    """How many microbatches an `n`-slot decode step is actually cut into.

    Bounded above by `ceiling` (`len(stages) * MICROBATCHES_PER_STAGE`, the pipeline-depth
    budget) and below by leaving at least `min_slots` slots in every microbatch. A step too
    small to give the ceiling's worth of microbatches `min_slots` each is cut into fewer,
    larger ones instead, down to one undivided batch; `_microbatch_cuts` still handles the
    remaining edge cases (n == 0, groups > n).
    """
    return max(1, min(ceiling, n // min_slots)) if min_slots > 0 else max(1, ceiling)


@dataclass(slots=True)
class _Micro:
    """One microbatch in flight: its activations and the slots/positions they belong to."""

    x: torch.Tensor
    slots: list[int]
    positions: list[int]


def decode_attention_paged(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_table: torch.Tensor,
    block_valid: torch.Tensor,
    positions: torch.Tensor,
    block_size: int,
    scale: float,
) -> torch.Tensor:
    """Decode attention over a block-paged KV pool (`paged-kv-design.md` section 3.1).

    One query token per slot, read through a per-slot block table instead of a per-slot
    contiguous pool row (`Model._attend_batch`'s old trick has no equivalent once storage is
    block-paged -- see `paged_attn.py`'s module docstring for why gathering every step is a
    bandwidth non-starter and an in-kernel block-table read is the only option left).

    Same one-dispatch-point shape `rmsnorm`/`gated_rmsnorm` already use: `paged_attn.available`
    picks the fused Triton kernel on an accelerator; the torch fallback below is what every
    other device (CPU, the seed tests) runs, and it is this kernel's own correctness oracle
    (`seed_tests/test_paged_attn.py`), the same role `model.rmsnorm`'s torch chain plays for
    `rmsnorm_fused.rmsnorm`. Precision matches `decode_attention`'s old masked ("written out")
    branch: fp32 product and softmax, one rounding at the final cast.

    `SEED_DECODE_ATTN_V2=1` (`decode_attn_v2.available`) swaps the fused branch for the
    split-KV ("flash-decoding") kernel instead of the base single-pass one -- same signature,
    same fallback, see `decode_attn_v2.py`'s module docstring for why and when that is a win.
    Checked before the base kernel so the two fused paths are mutually exclusive, never both,
    at one call site.

    `SEED_DECODE_ATTN_SPLITK=1` (`decode_attn_splitk.available`) takes precedence over both:
    split-K over each lane's own length with an MFMA inner loop, see that module's docstring.

    `q` is `[B, heads, 1, head_dim]`; `k_pool`/`v_pool` are `[num_blocks * block_size,
    kv_heads, head_dim]` (row `r` holds token `r % block_size` of block `r // block_size`);
    `block_table` is `[B, max_blocks]` int, this step's per-lane block ids, right-padded with
    `block_pool.RESERVED_BLOCK`; `block_valid` is `[B]` int, `block_pool.valid_block_count`
    per lane (read only by the fused kernel: see its module docstring for why a post-hoc mask
    is not enough on its own); `positions` is `[B]`, each lane's current (0-indexed) position.
    """
    if decode_attn_splitk.available(q.device):
        return decode_attn_splitk.decode_attention_paged_splitk(
            q, k_pool, v_pool, block_table, block_valid, positions, block_size, scale
        )
    if decode_attn_v2.available(q.device):
        return decode_attn_v2.decode_attention_paged_v2(
            q, k_pool, v_pool, block_table, block_valid, positions, block_size, scale
        )
    if paged_attn.available(q.device):
        return paged_attn.decode_attention_paged(
            q, k_pool, v_pool, block_table, block_valid, positions, block_size, scale
        )
    b, heads, _, head_dim = q.shape
    kv_heads = k_pool.shape[1]
    group = heads // kv_heads
    max_blocks = block_table.shape[1]
    dev = q.device

    tok = torch.arange(block_size, device=dev)
    logical_pos = (
        torch.arange(max_blocks, device=dev)[:, None] * block_size + tok[None, :]
    ).reshape(-1)  # [S], S = max_blocks * block_size
    live = logical_pos[None, :] <= positions[:, None]  # [B, S]
    rows = (block_table[:, :, None] * block_size + tok[None, None, :]).reshape(b, -1)  # [B, S]
    safe_rows = torch.where(live, rows, torch.zeros_like(rows))  # RESERVED_BLOCK is always valid

    k = k_pool[safe_rows].permute(0, 2, 1, 3).float()  # [B, kv_heads, S, head_dim]
    v = v_pool[safe_rows].permute(0, 2, 1, 3).float()
    grouped_q = q.reshape(b, kv_heads, group, head_dim).float()

    scores = torch.matmul(grouped_q, k.transpose(-1, -2)) * scale  # [B, kv_heads, group, S]
    probs = scores.masked_fill(~live[:, None, None, :], float("-inf")).softmax(-1)
    out = torch.matmul(probs, v)  # [B, kv_heads, group, head_dim]
    return out.reshape(b, heads, 1, head_dim).to(q.dtype)


def verify_attention_paged(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_table: torch.Tensor,
    block_valid: torch.Tensor,
    base_positions: torch.Tensor,
    block_size: int,
    scale: float,
) -> torch.Tensor:
    """`decode_attention_paged`, generalized to `T` new query rows per lane in one call
    (MTP verify: `T = mtp.k + 1`), instead of one call per row.

    `q` is `[B, heads, T, head_dim]`; `k_pool`/`v_pool`/`block_table`/`block_valid`/
    `block_size` are `decode_attention_paged`'s own paged-pool arguments -- the pool must
    already hold this round's `T` new tokens' own K/V (the caller writes them before calling
    this, the same order `attn_decode` already writes-then-reads in). `base_positions` is
    `[B]`, each lane's position *before* this round: lane `b`'s query row `t` sits at the
    absolute position `base_positions[b] + t`, so row `t`'s causal visibility is the cached
    prefix plus this round's own rows `0..t` (never rows `> t`, which have not been "spoken"
    yet at that row's point in the sequence -- the same causal-with-offset shape
    `prefill_attention` computes for a prefill chunk, here batched across every lane in one
    call instead of one `prefill_attention` call per sequence).

    On an accelerator this dispatches `paged_attn.verify_attention_paged`, whose fixed launch
    grid covers all `T` query rows while each program reads its current context length from
    `base_positions` on device. This is required for graph replay: the torch fallback's
    bounded gather below converts `block_valid.max()` to a Python integer, which is correct in
    eager mode but would freeze the warmup context width during capture.

    The gather below is bounded to `block_valid`'s max over the batch, not `block_table`'s
    fixed `max_blocks_per_lane` width: at b48/16k context, gathering the full width reads
    ~0.8 GB/layer of K/V that every lane short of the batch's longest has not written yet.
    `decode_attention_paged`'s docstring notes `block_valid` is normally read only by the
    fused kernel; this function has no fused counterpart (see above), so its torch path is
    the one place that needs to read it itself. Every lane's own valid block count is at
    most this batch max by construction, and `live`'s per-row position mask (not this bound)
    is what keeps a lane's *unwritten* blocks within that range from being attended to, so
    narrowing the gather changes memory traffic only, not the result.
    """
    if paged_attn.available(q.device):
        # Reuse the capture-tested paged decode kernel once per fixed verify row. Every
        # invocation reads its position on device at replay time, so unlike the bounded torch
        # gather below it cannot freeze warmup's one-block context in the captured graph.
        return torch.cat(
            [
                paged_attn.decode_attention_paged(
                    q[:, :, step : step + 1],
                    k_pool,
                    v_pool,
                    block_table,
                    block_valid,
                    base_positions + step,
                    block_size,
                    scale,
                )
                for step in range(q.shape[2])
            ],
            dim=2,
        )
    b, heads, t, head_dim = q.shape
    kv_heads = k_pool.shape[1]
    group = heads // kv_heads
    dev = q.device
    max_blocks = max(1, min(block_table.shape[1], int(block_valid.max().item())))
    block_table = block_table[:, :max_blocks]

    tok = torch.arange(block_size, device=dev)
    logical_pos = (
        torch.arange(max_blocks, device=dev)[:, None] * block_size + tok[None, :]
    ).reshape(-1)  # [S]
    query_pos = base_positions[:, None] + torch.arange(t, device=dev)[None, :]  # [B, T]
    live = logical_pos[None, None, :] <= query_pos[:, :, None]  # [B, T, S]
    rows = (block_table[:, :, None] * block_size + tok[None, None, :]).reshape(b, -1)  # [B, S]
    # A column live for any row is live for every later row (visibility only grows with the
    # query position), so the last row's mask is exactly "read this pool row at all".
    safe_rows = torch.where(live[:, -1, :], rows, torch.zeros_like(rows))

    k = k_pool[safe_rows].permute(0, 2, 1, 3).float()  # [B, kv_heads, S, head_dim]
    v = v_pool[safe_rows].permute(0, 2, 1, 3).float()
    grouped_q = q.reshape(b, kv_heads, group, t, head_dim).float()

    scores = torch.einsum("bkgtd,bksd->bkgts", grouped_q, k) * scale  # [B, kv_heads, group, T, S]
    probs = scores.masked_fill(~live[:, None, None, :, :], float("-inf")).softmax(-1)
    out = torch.einsum("bkgts,bksd->bkgtd", probs, v)  # [B, kv_heads, group, T, head_dim]
    return out.reshape(b, heads, t, head_dim).to(q.dtype)


def prefill_attention(
    q: torch.Tensor,
    keys: torch.Tensor,
    vals: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """A real (T > 1) query sequence against a cached prefix. q [B, heads, T, d], kv [B, kv_heads, S, d].

    The same trap as `decode_attention`, at a different shape. `full_attention`'s `start == 0`
    branch passes no `attn_mask` at all (`is_causal=True` covers a square, no-prefix causal
    mask), so `enable_gqa` there stays on SDPA's fused backend. Resuming after a cached prefix
    (`start > 0`, what feeds `p95_ttft_turn2plus_ms`) needs an explicit causal-with-offset
    mask, and on gfx942 `enable_gqa=True` combined with an explicit `attn_mask` falls off the
    fused backend the same way it does for decode: the fallback materializes the KV expanded
    to `heads` before the matmul, a `heads // kv_heads` read blowup of the tensor this call is
    bound by once the resumed prefix is long.

    The decode fix folds the grouped query heads into the query length, which is trivial when
    the query is one token (T=1, so the fold *is* the query length). Here T > 1: reshaping
    `[B, heads, T, d]` by splitting `heads` into `(kv_heads, group)` and merging `(group, T)`
    gives `[B, kv_heads, group * T, d]`. Row-major flattening makes that merge group-major,
    T-minor (row `j * T + t` is group `j`'s copy of real query position `t`), so
    `mask.repeat(group, 1)`, which repeats the whole `[T, S]` block `group` times, lines every
    repeated block back up with its group. Every kv head's keys are then read exactly once,
    same as the decode fix; the cost that used to fall on the KV read instead falls on
    building a `group`-times-larger boolean mask, which is orders of magnitude cheaper.

    Measured on MI300A (gfx942), one call at the served shape (32 heads, 2 kv heads, 256 head
    dim), bf16:

    | continuation | resumed prefix | `enable_gqa` | grouped |
    |-------------:|---------------:|-------------:|--------:|
    |     8 tokens |             64 |      0.23 ms | 0.06 ms |
    |     8 tokens |          2,048 |      0.56 ms | 0.50 ms |
    |     8 tokens |          8,192 |      2.08 ms | 3.45 ms |
    |     8 tokens |         32,768 |    711.17 ms | 13.66 ms |
    |   512 tokens |          8,192 |    710.35 ms |  4.04 ms |
    |   512 tokens |         32,768 |    721.77 ms | 20.97 ms |

    A short (8-token) continuation over a mid-length prefix is the one shape where the
    grouped call loses in absolute terms (both are sub-4ms, so it does not matter for
    `p95_ttft_turn2plus_ms`); everywhere else, and especially at a full `PREFILL_CHUNK`
    continuation, `enable_gqa` is the one that falls off a cliff once the resumed prefix is
    long.

    `group` comes from the tensors, so a tensor-parallel rank gets its own ratio (8 query
    heads over 1 KV head at TP=4) with no change here; see `decode_attention`.
    """
    b, heads, t, d = q.shape
    kv_heads = keys.shape[1]
    group = heads // kv_heads
    grouped_q = q.reshape(b, kv_heads, group * t, d)
    grouped_mask = mask.repeat(group, 1)
    out = F.scaled_dot_product_attention(grouped_q, keys, vals, attn_mask=grouped_mask, scale=scale)
    return out.reshape(b, heads, t, d)


def _zeroed_accumulator(h: torch.Tensor, out_buf: torch.Tensor | None) -> torch.Tensor:
    """The zeroed `[t, hidden]` accumulator `Model._routed_grouped` adds its experts into."""
    if out_buf is None:
        return torch.zeros_like(h)
    return out_buf[: h.shape[0]].zero_()


def _kv_block_bytes(cfg: Cfg, kv_heads_local: int, block_size: int, dtype: torch.dtype) -> int:
    """Bytes one KV block costs on one rank, across every full-attention layer at once.

    Matches this module's own memory-math docstring: `n_full_attention_layers *
    kv_heads_local * block_size * head_dim * 2 (K, V) * dtype size`. Block ids are shared
    across the full-attention layers -- one `block_pool.BlockAllocator`, one
    `block_pool.BlockTable` per lane -- so this is what one increment of a lane's block table
    actually costs, not a per-layer figure.
    """
    n_full = sum(1 for lt in cfg.layer_types if lt == "full_attention")
    return n_full * kv_heads_local * block_size * cfg.head_dim * 2 * dtype.itemsize


def _snapshot_bytes(cfg: Cfg, dtype: torch.dtype) -> int:
    """Bytes for one DeltaNet-snapshot slot: every non-full-attention layer's conv + recurrent
    state, plus one fp32 cached-logits row (see `Model.cache_node_logits`).

    Mirrors the shapes `Model._new_pool` builds for a layer's *live* state, so a snapshot slot
    always costs exactly what one lane's own live row costs, layer for layer -- `conv_dim = 2 *
    k_heads * k_dim + v_heads * v_dim`, a `[conv_dim, conv_k - 1]` conv tensor in `dtype`, and a
    `[v_heads, k_dim, v_dim]` fp32 recurrent tensor. `cfg` is this rank's local config (see
    `Model.cfg`), so this is already one rank's share under TP.
    """
    total = 0
    for lt in cfg.layer_types:
        if lt == "full_attention":
            continue
        conv_dim = 2 * cfg.k_heads * cfg.k_dim + cfg.v_heads * cfg.v_dim
        total += conv_dim * (cfg.conv_k - 1) * dtype.itemsize
        total += cfg.v_heads * cfg.k_dim * cfg.v_dim * 4  # fp32 recurrent state
    return total + cfg.vocab * 4  # fp32 cached-logits row


@dataclass(frozen=True)
class PackedSeq:
    """One sequence's slice of a packed prefill call: its slot, resume position, and length."""

    slot: int
    start: int  # position this chunk begins at (tokens already resident in the slot's state)
    length: int  # tokens in this chunk


@dataclass(frozen=True)
class PackedBatch:
    """Metadata for one packed varlen prefill call: several requests' chunks end to end.

    The packed token ids themselves are a plain [total_T] tensor built by the caller; this
    only carries what `layer_packed` needs to route each row back to its own sequence.
    `spans()` derives each sequence's [lo, hi) slice of that axis (cu_seqlens, as it's usually
    called) from `seqs` alone -- nothing here stores it separately, so there is exactly one
    place a slice boundary could be wrong.
    """

    seqs: tuple[PackedSeq, ...]

    def spans(self) -> list[tuple[int, int]]:
        out, lo = [], 0
        for seq in self.seqs:
            out.append((lo, lo + seq.length))
            lo += seq.length
        return out

    @property
    def total(self) -> int:
        return sum(seq.length for seq in self.seqs)


class Model:
    scheduler_owns_blocks: bool = False
    """See `__init__`. A class default too, so a bare `Model.__new__` test double has it."""

    def __init__(
        self,
        model_dir: str | Path,
        devices: list[str],
        dtype: torch.dtype,
        max_seq: int,
        max_batch: int = 1,
        tp: TP | None = None,
    ) -> None:
        # Two independent parallelism axes. `devices` is the intra-process layer split (this
        # process runs layer i on devices[layer_dev[i]]). `tp` is this process's slice of a
        # tensor-parallel group, one process per device, where every rank runs every layer on
        # a shard of its heads and experts. They are mutually exclusive, and combining them
        # is rejected below rather than silently allowed: under TP the collectives have to be
        # reached by every rank in the same order, and a layer split hands layers to separate
        # stage threads, which would put two ranks' all-reduces in different orders.
        full_cfg = load_cfg(model_dir)
        self.devices = [torch.device(d) for d in devices]
        self.tp = TP.single(full_cfg, self.devices[0]) if tp is None else tp
        if self.tp.enabled and len(self.devices) > 1:
            raise ValueError(
                f"tp: a rank owns one device, got {len(self.devices)} at world "
                f"{self.tp.world}; pass devices=[tp.device]"
            )
        # `self.cfg` holds the *local* DeltaNet head counts, so every shape derived from
        # them - conv_dim, the recurrent state, the projection views - is this rank's
        # share; `full_cfg` keeps the global counts the checkpoint is laid out in. The
        # full-attention head counts stay global in `cfg` and are read off `self.tp.plan`
        # where they are needed, because `kv` is replicated rather than split and does not
        # follow `heads / world`.
        self.cfg = cfg = deltanet_tp.local_cfg(full_cfg, self.tp)
        self.dtype, self.max_seq, self.max_batch = dtype, max_seq, max_batch
        n = len(cfg.layer_types)
        self.layer_stage = [i * len(self.devices) // n for i in range(n)]  # contiguous split
        self.layer_dev = [self.devices[s] for s in self.layer_stage]
        self.stages = _stage_ranges(self.layer_stage)
        ck = Checkpoint(model_dir)
        # Replicated across ranks: every rank embeds and unembeds for itself, so a decode step
        # needs no collective outside the two per layer, and every rank can sample in step.
        mem_timeline.mark("model_start", self.devices[0])
        self.embed = ck.load(f"{PREFIX}embed_tokens.weight", self.devices[0], dtype)
        self.layers = []
        for i in range(n):
            self.layers.append(load_layer(ck, i, full_cfg, self.tp, self.layer_dev[i], dtype))
            if i % 5 == 4 or i == n - 1:
                mem_timeline.mark(f"weights_l{i}", self.layer_dev[i])
        last = self.devices[-1]
        self.final_norm = ck.load(f"{PREFIX}norm.weight", last, dtype)
        self.lm_head = (
            ck.load("lm_head.weight", last, dtype)
            if ck.has("lm_head.weight")
            else self.embed.to(last)
        )
        self.vocab_tp = LMHEAD_VOCAB_TP and self.tp.world > 1
        if self.vocab_tp:
            self.lm_head = self.tp.shard(self.lm_head, 0, "vocab")
        mem_timeline.mark("weights_lm_head", last)
        # Paged KV pool for the full-attention layers (paged-kv-design.md, Stage 1). Block ids
        # are shared across every full-attention layer (one allocator, one BlockTable per
        # lane), which requires those layers to share a device: true of every real deployment
        # (TP owns one device per rank) and every existing test (the pipeline test's "several
        # devices" are all literally `torch.device("cpu")`, see `_stage_ranges`'s docstring),
        # but not a general multi-device pipeline split, which this raises on rather than
        # silently mis-sizing.
        full_attn_devices = {
            self.layer_dev[i] for i in range(n) if cfg.layer_types[i] == "full_attention"
        }
        if len(full_attn_devices) > 1:
            raise ValueError(
                "paged KV pool (Stage 1): every full-attention layer must live on the same "
                f"device, got {sorted(str(d) for d in full_attn_devices)}; TP (one device per "
                "rank) and every existing test satisfy this. A genuine multi-device pipeline "
                "split of the full-attention layers is not supported."
            )
        self.block_size = KV_BLOCK_SIZE
        kv_dev = next(iter(full_attn_devices))
        self.num_kv_blocks, self.num_snapshots = self._size_pools(kv_dev)
        self.block_allocator = block_pool.BlockAllocator(self.num_kv_blocks, self.block_size)
        self.max_blocks_per_lane = block_pool.max_blocks_per_lane(max_seq, self.block_size)
        self.block_tables = [block_pool.BlockTable() for _ in range(max_batch)]
        # False: a forward that outgrows a lane's table allocates from `block_allocator`
        # itself (the single-sequence `generate`/`warmup` path and boot-time graph capture,
        # which own every lane outright). True once serving starts (`tp_driver.
        # pool_handshake`): the scheduler on rank 0 is then the only allocator authority and
        # hands every rank the resolved ids via `extend_blocks`/`attach_blocks`, so a forward
        # that finds its table short raises instead of allocating a rank-local id that the
        # other ranks would not agree on (see `grow_lane`).
        self.scheduler_owns_blocks = False
        # Per lane, per full-attention layer: the growing (resumed prefix + chunks so far)
        # contiguous K/V scratch `full_attention` feeds to `prefill_attention` (paged-kv-
        # design.md section 3.2's "gather-once" path). `None` between prefill sequences --
        # `reset`/`save_snapshot` both clear it, so it never survives past the point where the
        # authoritative data is the paged pool instead. Indexed by full layer index for every
        # lane (not `full_attention` layers only) so `full_attention(i, ...)` can address it
        # directly by `i`; the DeltaNet-layer entries are simply never read.
        self._prefill_scratch: list[list[tuple[torch.Tensor, torch.Tensor] | None]] = [
            [None] * n for _ in range(max_batch)
        ]
        self.pool = [
            self._new_kv_block_pool(i, self.num_kv_blocks, kv_dev)
            if cfg.layer_types[i] == "full_attention"
            else self._new_pool(i)
            for i in range(n)
        ]
        mem_timeline.mark("kv_dn_pools", kv_dev)
        self.snapshot_pool = [self._new_snapshot_pool(i) for i in range(n)]
        mem_timeline.mark("snapshot_pool", kv_dev)
        # Per lane, a batch-1 view of every DeltaNet layer's pool row: the layer code below
        # reads `self.state` and writes through these views into the pool. Full-attention
        # layers get an empty dict -- there is no per-lane "row" of a shared block pool to
        # view, and `full_attention`/`attn_decode` read `self.pool[i]` and
        # `self.block_tables[slot]` directly instead of `self.state[i]`.
        self.slot_state = [
            [
                {} if lt == "full_attention" else {name: t[s : s + 1] for name, t in p.items()}
                for lt, p in zip(cfg.layer_types, self.pool, strict=True)
            ]
            for s in range(max_batch)
        ]
        self.state = self.slot_state[0]
        self.current_slot = 0
        # Keyed by (device, FAULT_ROPE_BASE) rather than device alone: FAULT_ROPE_BASE can now
        # flip at runtime (see /debug/fault), and a cache keyed only on device would keep
        # serving whichever table was built first regardless of the flag's current value.
        self.rope_cache: dict[tuple[torch.device, bool], tuple[torch.Tensor, torch.Tensor]] = {}
        # Per (role, device), the last `torch.tensor(...)` built from a host-side index list and
        # the list it was built from. See `_host_index`.
        self.index_cache: dict[str, dict[torch.device, tuple[tuple[int, ...], torch.Tensor]]] = {}
        # Decode-hot-path scratch, preallocated once and written in place every call instead of
        # fresh-allocated: per-slot DeltaNet decode output, and per-layer MoE output accumulator
        # sized for a full batched-decode step. Neither is part of the sequence's persisted state
        # (unlike `pool`/`slot_state`), so they stay out of save_snapshot/load_snapshot.
        #
        # Safe under the pipeline below because both are indexed per layer, and a layer belongs
        # to exactly one stage, which runs one microbatch at a time. Two microbatches in flight
        # are always in different stages, so they never touch the same scratch entry.
        self.delta_scratch = [
            torch.empty(max_batch, 1, cfg.v_heads, cfg.v_dim, dtype=torch.float32, device=dev)
            if lt != "full_attention"
            else None
            for lt, dev in zip(cfg.layer_types, self.layer_dev, strict=True)
        ]
        self.moe_scratch = [
            torch.empty(max_batch, cfg.hidden, dtype=dtype, device=dev) for dev in self.layer_dev
        ]
        # One row per snapshot id, holding that cached session's next-token logits for as long
        # as it stays in the trie (see `cache_node_logits`/`cached_node_logits`): an
        # exact-duplicate prompt, or one that repeats a finished request's own reply
        # verbatim, is answered from here with no forward call at all. Sized by `num_snapshots`,
        # not `max_batch` (Stage 2: a cached session outlives whatever lane computed it) -- part
        # of the static pool like everything else above, so holding a cached row costs nothing
        # beyond what startup already reserved, unlike a fresh `[1, vocab]` tensor per cache.
        self.node_logits_scratch = torch.zeros(
            self.num_snapshots, cfg.vocab, dtype=torch.float32, device=last
        )
        # Per-assignment gated intermediate handed from the fused MoE's gate_up kernel to its
        # down kernel. Sized for a full decode step; a prefill chunk is larger than `max_batch`
        # and allocates fresh (`_routed_fused`), which is fine off the decode hot path.
        self.moe_inter_scratch = [
            torch.empty(
                max_batch * cfg.top_k,
                layer["experts"]["gate_up"].shape[1] // 2,  # gate and up are stacked
                dtype=dtype,
                device=dev,
            )
            for layer, dev in zip(self.layers, self.layer_dev, strict=True)
        ]
        # The de-duplicating path's un-combined per-assignment down-projection output
        # (`mxfp4_gemv.fused_moe_dedup`'s `y`): same per-assignment sizing idea as
        # `moe_inter_scratch`, one hidden-width row per assignment instead of one
        # intermediate-width row.
        self.moe_dedup_y_scratch = [
            torch.empty(max_batch * cfg.top_k, cfg.hidden, dtype=dtype, device=dev)
            for dev in self.layer_dev
        ]
        # Per-expert atomic cursor `mxfp4_gemv.fused_expert_order` claims scatter slots from.
        # Zeroed by the kernel itself every call (see its docstring), so this is reused as-is
        # across layers and steps; sized by the *global* expert count, matching what
        # `expert_sorted_order`/`fused_expert_order` sort over.
        self.moe_route_cursor = [
            torch.empty(cfg.experts, dtype=torch.int32, device=dev) for dev in self.layer_dev
        ]
        # Global expert ids this rank owns: the whole axis unsharded, one quarter of it at
        # TP=4. `mxfp4_gemv` drops the out-of-range assignments GPU-side (so every rank
        # launches the same shape and the step stays sync-free) and `_routed_grouped` drops
        # them host-side on the torch path; `load_experts` loaded exactly this slice, so a
        # local expert index is the global id minus `expert_range[0]`.
        self.expert_range = (self.tp.plan.experts.start, self.tp.plan.experts.stop)
        # `SEED_MOE_BW=1`: the bandwidth-first MoE folds each e8m0 scale into its decode
        # table, which is exact only for scales in [2, 252] (see `mxfp4_gemv.bw_scales_ok`).
        # Checked once here, at load time, because the check syncs the host.
        self.moe_bw = all(
            mxfp4_gemv.bw_available(layer["experts"]["gate_up"].device)
            and mxfp4_gemv.bw_scales_ok(layer["experts"])
            for layer in self.layers
            if "experts" in layer
        ) and any("experts" in layer for layer in self.layers)
        # `SEED_HOT_EXPERTS=1` (hot_experts.py): `hot_cache[i]` starts empty (falls through to
        # the cold MXFP4 path entirely) and `hot_counts[i]` accumulates this rank's own routed
        # assignment counts, per local expert, until `finalize_hot_experts` freezes the cache.
        # `None` in either list at a given index means "no experts on this layer" (`load_experts`
        # never ran for it), matching `w["experts"]` itself being absent.
        n_local_experts = self.expert_range[1] - self.expert_range[0]
        self.hot_cache: list[hot_experts.LayerCache | None] = [
            None
            if "experts" not in layer
            else hot_experts.LayerCache(
                torch.zeros(0, dtype=torch.long), torch.empty(0), torch.empty(0)
            )
            for layer in self.layers
        ]
        self.hot_counts: list[torch.Tensor | None] = [
            None
            if "experts" not in layer
            else torch.zeros(
                n_local_experts, dtype=torch.long, device=layer["experts"]["gate_up"].device
            )
            for layer in self.layers
        ]
        self.hot_experts_finalized = False
        # `SEED_MOE_HIP=1`: hand-written gfx942 decode MoE (`moe_hip`), same scale precondition
        # plus its compiled shapes. The extension is built (or loaded from its content-hashed
        # cache) here, at boot, never inside graph capture. Decided before any reshuffle below:
        # the HIP kernels read the load-time (unshuffled) layout.
        self.moe_hip = all(
            moe_hip.available(layer["experts"]["gate_up"].device)
            and moe_hip.supports(layer["experts"])
            for layer in self.layers
            if "experts" in layer
        ) and any("experts" in layer for layer in self.layers)
        if self.moe_hip:
            moe_hip.ext()
        # `SEED_MOE_BW_VARIANT` (default v1) picks the kernel variant (`mxfp4_moe_bw2`). A
        # variant that reads the reshuffled weight layout gets it here, once, in place: 0
        # extra steady-state bytes, one tensor's copy as the transient peak. From then on
        # these experts are readable only by `fused_moe_bw2` (and `hot_experts`, which
        # unshuffles the few experts it caches). With `SEED_MOE_HIP` also on, the HIP decode
        # kernels need the unshuffled layout, so the variant falls back to v1 (no shuffle).
        self.moe_bw_variant = mxfp4_moe_bw2.selected_variant()
        if self.moe_bw and self.moe_bw_variant.shuffle:
            if self.moe_hip:
                print(
                    f"[moe] SEED_MOE_HIP needs the unshuffled layout: SEED_MOE_BW_VARIANT="
                    f"{self.moe_bw_variant.name} falls back to v1",
                    flush=True,
                )
                self.moe_bw_variant = mxfp4_moe_bw2.VARIANTS["v1"]
            else:
                for layer in self.layers:
                    if "experts" in layer:
                        mxfp4_moe_bw2.bw_shuffle(layer["experts"])
        self.pipeline = self._build_pipeline()
        # The *ceiling* on microbatches a decode step is cut into, held apart from `pipeline`
        # so that closing the pipeline runs the same partition sequentially: the parity
        # reference in the tests. `_microbatches` narrows this per call via
        # `_decode_group_count`; see `MIN_MICROBATCH_SLOTS` for why a step does not always get
        # the full ceiling.
        self.microbatches = len(self.stages) * MICROBATCHES_PER_STAGE if self.pipeline else 1
        # MTP speculative decoding (see mtp.py). `None` when `SEED_MTP` is off or the
        # checkpoint carries no `mtp.*` weights, in which case `speculative_decode` is never
        # called (the scheduler checks the same flag). `hidden_scratch` mirrors
        # `logits_scratch`'s one-row-per-slot pattern: the raw (pre-final-norm) hidden state
        # at each slot's last committed token, `mtp.draft`'s seed for its next round.
        mem_timeline.mark("scratch", last)
        self.mtp = mtp.load(
            ck,
            full_cfg,
            self.tp,
            last,
            dtype,
            max_batch,
            self.num_kv_blocks * self.block_size,
        )
        mem_timeline.mark("mtp", last)
        self.hidden_scratch = (
            torch.zeros(max_batch, cfg.hidden, dtype=dtype, device=last)
            if self.mtp is not None
            else None
        )
        self.mtp_hidden_snapshot = (
            torch.zeros(self.num_snapshots, cfg.hidden, dtype=dtype, device=last)
            if self.mtp is not None
            else None
        )
        # `SEED_STEP_TIMING` diagnostics (see step_timing.py). `decode_path` is what the
        # scheduler's timing line reports; `GraphDecodeRunner` overrides it with its own.
        self.decode_path = "eager"
        # One eager decode step's paged-attention inputs, shared by all full-attention layers
        # of that step (see `_paged_read_buffers`). `_decode_epoch` advances on every `decode`
        # call (so the memo is fresh for each step, shared only across that step's own layers)
        # and on every `reset`. `attach_blocks`/`copy_block` never run on their own: the
        # scheduler's only call site (`_acquire`) always pairs them with the `begin` -> `reset`
        # that admits the lane, before that lane's first `decode` of the new epoch -- so a
        # lane's blocks never change without an intervening epoch bump for the memo to catch.
        self._decode_epoch = 0
        self._paged_read_memo: tuple[tuple, torch.Tensor, torch.Tensor] | None = None
        self._timing_decodes = 0
        self._timing_sample = False
        self._attn_events: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._attn_host_ms = 0.0
        self._timing_prefills = 0
        # `SEED_OVERLAP_SCHED`: each lane's last sampled token id, written on the device by
        # `launch_tail` and read back into the next step by `resolve_tokens`, so step N+1 can
        # be launched before step N's tokens reach the host. Every rank keeps its own copy
        # (rank 0 samples, the others receive the ids by broadcast), so the next step's
        # embedding lookup is rank-local.
        self.lane_tokens = torch.zeros(max_batch, dtype=torch.long, device=self.devices[0])
        # `SEED_STEP_TIMING`: device idle time between decode steps (CUDA only).
        self.gap_meter: step_timing.GapMeter | None = None
        if step_timing.ENABLED and self.devices[0].type == "cuda":
            self.gap_meter = step_timing.GapMeter(
                lambda: torch.cuda.Event(enable_timing=True), f"rank {self.tp.plan.rank} decode"
            )

    def _build_pipeline(self) -> StagePipeline | None:
        """One pipeline stage per device, or None when there is nothing to overlap."""
        if len(self.stages) < 2 or MICROBATCHES_PER_STAGE < 1:
            return None
        for layers in self.stages:  # build the rope tables here, not from the stage threads
            self._rope_table(self.layer_dev[layers.start])
        return StagePipeline([self._stage_fn(layers) for layers in self.stages], name="decode")

    def _stage_fn(self, layers: range) -> Callable[[_Micro], _Micro]:
        return lambda micro: self._decode_stage(layers, micro)

    def close(self) -> None:
        """Release the stage threads. Idempotent; decode then runs the stages sequentially."""
        if self.pipeline is not None:
            self.pipeline.close()
            self.pipeline = None

    # -- static state ---------------------------------------------------------
    def _size_pools(self, dev: torch.device) -> tuple[int, int]:
        """Size the KV block pool and the DeltaNet snapshot pool to *need*, each with a hard
        cap, never as a fraction of measured free memory.

        MI300A is a unified-memory APU: `torch.cuda.mem_get_info`'s "free" is shared with host
        RSS (python, tokenizer, aiohttp buffers, checkpoint page cache, pinned staging), which
        keeps growing after model load as real traffic arrives. An earlier version of this
        function sized both pools as a fraction of free memory measured once at load time --
        a real boot OOM'd under a Slurm cgroup limit shortly after the load gate, because "free
        at load time" was never a safe stand-in for "free once serving is warmed up". The KV
        pool is now sized to `max_batch * max_seq` token capacity (Stage 1's own floor) times
        `KV_POOL_HEADROOM_FACTOR` for cached-but-idle sessions, hard-capped at `KV_POOL_CAP_GIB`
        (or pinned exactly via `SEED_KV_POOL_GIB`). The snapshot pool is a fixed slot count
        (`SNAPSHOT_POOL_COUNT`, or a GiB budget via `SEED_SNAPSHOT_POOL_GIB`), not a share of
        anything measured. Free memory is only consulted afterward, to check the chosen pools
        still leave `MIN_UNALLOCATED_GIB` unallocated for host RSS and transient activations
        (prefill at 2k-token packed batches, MoE workspaces) -- if not, the KV pool (never the
        snapshot pool) is shrunk toward its floor; if even the floors do not leave enough, this
        logs a warning and proceeds at the floors rather than refusing to boot.

        On CPU (the seed tests, and any device `torch.cuda.mem_get_info` cannot read) there is
        no free-memory signal for the headroom check, so both pools are sized to exactly their
        floors: `max_batch * max_seq` tokens for KV, `max_batch` slots for snapshots.

        Prints the plan (`[pool-plan]`) plus one `[kv-pool]`/`[snapshot-pool]` line each,
        before returning `(num_kv_blocks, num_snapshots)`.
        """
        kv_floor_tokens = int(self.max_batch * self.max_seq * KV_POOL_MIN_CAPACITY_FACTOR)
        kv_floor_blocks = -(-kv_floor_tokens // self.block_size)
        snap_floor = max(1, int(self.max_batch * SNAPSHOT_POOL_MIN_CAPACITY_FACTOR))
        bytes_per_block = _kv_block_bytes(
            self.cfg, self.tp.plan.kv.count, self.block_size, self.dtype
        )
        bytes_per_snap = _snapshot_bytes(self.cfg, self.dtype)

        # Snapshot pool: fixed count (or an explicit GiB budget converted to a count), never a
        # function of measured free memory.
        if SNAPSHOT_POOL_GIB_OVERRIDE is not None:
            snap_target = int(float(SNAPSHOT_POOL_GIB_OVERRIDE) * (1024**3)) // bytes_per_snap
        else:
            snap_target = SNAPSHOT_POOL_COUNT
        num_snapshots = max(snap_target, snap_floor)
        print(
            f"[snapshot-pool] {dev}: {bytes_per_snap} B/snapshot, target {snap_target} slots "
            f"({'SEED_SNAPSHOT_POOL_GIB override' if SNAPSHOT_POOL_GIB_OVERRIDE else 'fixed count'}"
            f"), floor {snap_floor} -> {num_snapshots} slots chosen "
            f"({num_snapshots * bytes_per_snap / 1024**3:.2f} GiB)",
            flush=True,
        )

        # KV pool: need-based (floor * headroom), hard-capped, never a fraction of free memory.
        if KV_POOL_GIB_OVERRIDE is not None:
            kv_target_bytes = int(float(KV_POOL_GIB_OVERRIDE) * (1024**3))
            kv_cap_note = "SEED_KV_POOL_GIB override"
        else:
            need_bytes = int(kv_floor_blocks * bytes_per_block * KV_POOL_HEADROOM_FACTOR)
            cap_bytes = int(KV_POOL_CAP_GIB * (1024**3))
            kv_target_bytes = min(need_bytes, cap_bytes)
            kv_cap_note = "capped" if need_bytes > cap_bytes else "under cap"
        num_blocks = max(kv_target_bytes // bytes_per_block, kv_floor_blocks)

        # Headroom check: shrink the KV pool toward its floor, never below it, if the chosen
        # pools would not leave MIN_UNALLOCATED_GIB free. Snapshot pool is left alone: it is
        # already a small, fixed, conservative budget.
        if dev.type == "cuda":
            free_bytes, total_bytes = torch.cuda.mem_get_info(dev)
            min_unallocated = int(MIN_UNALLOCATED_GIB * (1024**3))
            snap_bytes = num_snapshots * bytes_per_snap
            kv_budget_for_headroom = free_bytes - snap_bytes - min_unallocated
            affordable_blocks = max(0, kv_budget_for_headroom) // bytes_per_block
            if affordable_blocks < num_blocks:
                num_blocks = max(kv_floor_blocks, affordable_blocks)
            remaining = free_bytes - (num_blocks * bytes_per_block) - snap_bytes
            if remaining < min_unallocated:
                print(
                    f"[pool-plan] {dev}: WARNING even the pool floors leave only "
                    f"{remaining / 1024**3:.2f} GiB unallocated, below the "
                    f"{MIN_UNALLOCATED_GIB:.0f} GiB target; proceeding anyway",
                    flush=True,
                )
            print(
                f"[pool-plan] {dev}: free {free_bytes / 1024**3:.2f} GiB of "
                f"{total_bytes / 1024**3:.2f} GiB total (unified with host RSS on MI300A, not "
                f"re-measured after this point) -> KV {num_blocks * bytes_per_block / 1024**3:.2f} "
                f"GiB + snapshot {snap_bytes / 1024**3:.2f} GiB, leaving "
                f"{remaining / 1024**3:.2f} GiB unallocated (target {MIN_UNALLOCATED_GIB:.0f} GiB)",
                flush=True,
            )
        else:
            num_blocks = kv_floor_blocks
            num_snapshots = snap_floor
            print(
                f"[pool-plan] {dev}: no free-memory reading available (not an accelerator); "
                "sizing both pools to their floors",
                flush=True,
            )

        print(
            f"[kv-pool] {dev}: {bytes_per_block} B/block at {self.block_size} tok/block, "
            f"need {kv_floor_blocks} blocks ({kv_floor_tokens} tokens = {self.max_batch} x "
            f"{self.max_seq}) x {KV_POOL_HEADROOM_FACTOR:.1f} headroom, cap {KV_POOL_CAP_GIB:.0f} "
            f"GiB ({kv_cap_note if dev.type == 'cuda' else 'floor, no accelerator'}) -> "
            f"{num_blocks} blocks chosen ({num_blocks * self.block_size} token capacity, "
            f"{num_blocks * bytes_per_block / 1024**3:.2f} GiB)",
            flush=True,
        )
        return num_blocks + 1, num_snapshots  # +1 for block_pool.RESERVED_BLOCK

    def _new_kv_block_pool(self, i: int, num_blocks: int, dev: torch.device) -> dict:
        """Layer `i`'s (full-attention) block-paged K/V storage.

        `[num_blocks * block_size, kv_heads, head_dim]`: row `r` holds token `r % block_size`
        of block `r // block_size`. Block ids are shared across every full-attention layer
        (one `block_pool.BlockAllocator`, one `block_pool.BlockTable` per lane, both built in
        `__init__` before this is called) -- this tensor is *this layer's* data for those ids,
        not its own separate id space.
        """
        rows = num_blocks * self.block_size
        shape = (rows, self.tp.plan.kv.count, self.cfg.head_dim)
        return {
            "k": torch.zeros(shape, dtype=self.dtype, device=dev),
            "v": torch.zeros(shape, dtype=self.dtype, device=dev),
        }

    def _new_pool(self, i: int) -> dict:
        """One DeltaNet layer's per-slot state, holding only this rank's heads.

        Full-attention layers do not come through here any more: their storage is the shared
        block pool `_new_kv_block_pool` builds once, not a per-slot allocation keyed by `i`.
        """
        c, dev, b = self.cfg, self.layer_dev[i], self.max_batch
        assert c.layer_types[i] != "full_attention", "full-attention layers use _new_kv_block_pool"
        # `c` is the local config, so `k_heads`/`v_heads` are already this rank's share.
        conv_dim = 2 * c.k_heads * c.k_dim + c.v_heads * c.v_dim
        return {
            "conv": torch.zeros(b, conv_dim, c.conv_k - 1, dtype=self.dtype, device=dev),
            "rec": torch.zeros(b, c.v_heads, c.k_dim, c.v_dim, dtype=torch.float32, device=dev),
        }

    def _new_snapshot_pool(self, i: int) -> dict | None:
        """Snapshot rows for layer `i`, shaped like its live-state pool but with `num_snapshots`
        rows instead of `max_batch` (Stage 2: an independently sized pool of cached-session
        boundaries, not one register per lane -- see the module docstring and `_size_pools`).

        `None` for a full-attention layer: KV rows survive decode (they are masked by
        length), so only the recurrent state, which decode overwrites in place, is saved.
        This is allocated with the rest of the static pool rather than on a lane's first
        `save_snapshot`, so the whole serving footprint is reserved before the first request
        instead of growing per new cached session.
        """
        pool = self.pool[i]
        if "conv" not in pool:
            return None
        return {
            name: torch.empty(self.num_snapshots, *t.shape[1:], dtype=t.dtype, device=t.device)
            for name, t in pool.items()
        }

    def snapshot_row(self, snap: int) -> list[dict]:
        """Snapshot `snap`'s per-layer view into `self.snapshot_pool`, shaped like one entry of
        `self.slot_state` but built on demand: `num_snapshots` can be much larger than
        `max_batch`, and unlike `slot_state` (read on every layer call), this is only read by
        `save_snapshot`/`load_snapshot`, so materializing the view dict per call costs nothing
        that matters.
        """
        return [
            None if p is None else {name: t[snap : snap + 1] for name, t in p.items()}
            for p in self.snapshot_pool
        ]

    def reset(self) -> None:
        """Start a new sequence: zero DeltaNet state and forget this lane's block table.

        DeltaNet's per-lane dense rows are masked by length and just get zeroed. The paged
        full-attention pool is different, but Stage 2 moves ownership of a lane's block
        *refcounts* to the scheduler (`session_cache.SessionCache.release_lane_blocks`/
        `.adopt`/`.publish`, driven by `scheduler.py`'s `_release`): by the time this runs, the
        scheduler has already decref'd whatever blocks this lane held (or, for a lane that was
        never bound to a cache node, there was nothing to decref), so decref'ing here too would
        double-free every block a resumed lane still shares with a live cached session. This
        just forgets the lane's own list; the allocator is never touched here.

        A caller with no scheduler/`SessionCache` mediating it (`generate`, `warmup`: the
        single-sequence path, unchanged by Stage 2) is its own only owner of whatever the
        lane holds, so it must call `_release_lane_blocks` itself first -- see those methods.
        """
        slot = self.current_slot
        for st in self.state:
            if "conv" in st:
                st["conv"].zero_()
                st["rec"].zero_()
        self.block_tables[slot].blocks = []
        self._prefill_scratch[slot] = [None] * len(self.layers)
        if self.hidden_scratch is not None:
            self.hidden_scratch[slot].zero_()
        self._decode_epoch += 1

    def _release_lane_blocks(self, slot: int) -> None:
        """Decref whatever blocks `slot`'s table currently holds, then forget nothing else --
        `reset`/`begin` still do the actual clearing.

        For the scheduler-driven path this is `SessionCache.release_lane_blocks`, called by
        `scheduler.py` before `begin`/`reset` run again (see `reset`'s docstring). A bare
        `Model` used with no scheduler or cache (`generate`, `warmup`: the single-sequence
        path the accuracy checker and parity tests use) owns lane 0 end to end with nothing
        else ever incref'ing its blocks, so it must release them itself here before its next
        `begin`, or repeated calls would grow this lane's table forever and eventually exhaust
        the block pool.
        """
        blocks = self.block_tables[slot].blocks
        if blocks:
            self.block_allocator.decref(blocks)

    def unembed(self, h: torch.Tensor) -> torch.Tensor:
        """Logits `[..., vocab]` of final-normed hidden states `h`, in the activation dtype.
        Under `SEED_LMHEAD_VOCAB_TP` each rank multiplies its vocab shard and the columns are
        all-gathered (in rank order, which is vocab order), so every rank gets full logits."""
        logits = F.linear(h, self.lm_head)
        return self.tp.all_gather_last(logits) if self.vocab_tp else logits

    # -- lane API (the scheduler's Runner protocol) ---------------------------
    def bind(self, slot: int) -> None:
        """Point the single-sequence layer code at `slot`'s state."""
        self.state = self.slot_state[slot]
        self.current_slot = slot

    def begin(self, slot: int) -> None:
        self.bind(slot)
        self.reset()

    def lane_blocks(self, lane: int) -> tuple[int, ...]:
        """`lane`'s current full flat block-id table."""
        return tuple(self.block_tables[lane].blocks)

    def attach_blocks(self, lane: int, blocks: Sequence[int]) -> None:
        """Seed `lane`'s block table with an already-resolved id list (the caller,
        `scheduler.py`, has already applied copy-on-write via `copy_block` if
        `session_cache.SessionCache.needs_cow` said so)."""
        self.block_tables[lane].blocks = list(blocks)

    def lane_block_count(self, lane: int) -> int:
        """How many blocks `lane`'s table holds (what the scheduler sizes growth against)."""
        return len(self.block_tables[lane].blocks)

    def extend_blocks(self, grants: Sequence[tuple[int, int]]) -> None:
        """Append already-reserved block ids to lanes' tables, `(lane, block)` pairs in order.

        The scheduler reserves these on rank 0 (`SessionCache.reserve_blocks`, which evicts
        cache entries under pressure) and broadcasts them, so every rank's table holds the
        same ids. Validated before any table changes, so a rejected call leaves every table
        as it was and the caller can release the ids it reserved.
        """
        added = [0] * self.max_batch
        for lane, block in grants:
            if not 0 < block < self.num_kv_blocks:
                raise ValueError(f"extend_blocks: block id {block} outside [1, {self.num_kv_blocks})")
            added[lane] += 1
        for lane, n in enumerate(added):
            if n and len(self.block_tables[lane].blocks) + n > self.max_blocks_per_lane:
                raise MemoryError(
                    f"lane {lane} would hold {len(self.block_tables[lane].blocks) + n} blocks, "
                    f"lane cap is {self.max_blocks_per_lane}"
                )
        for lane, block in grants:
            self.block_tables[lane].blocks.append(block)

    def decode_tokens_per_step(self) -> int:
        """KV rows one decode round can write per lane: `mtp.k + 1` under MTP, else 1."""
        return self.mtp.k + 1 if self.mtp is not None else 1

    def grow_lane(self, slot: int, tokens: int) -> None:
        """Make `slot`'s table cover `tokens` tokens before a forward writes into it.

        With `scheduler_owns_blocks` the scheduler already extended the table (and under TP
        broadcast the ids), so a short table is a scheduler bug: raise rather than allocate an
        id only this rank knows about. Otherwise this rank owns the lane and grows it itself.
        """
        table = self.block_tables[slot]
        if self.scheduler_owns_blocks:
            if table.token_capacity(self.block_size) < tokens:
                raise RuntimeError(
                    f"lane {slot} holds {len(table.blocks)} block(s) for {tokens} tokens: the "
                    "scheduler must reserve and extend_blocks before the forward"
                )
            return
        table.grow_to(tokens, self.block_size, self.block_allocator, self.max_blocks_per_lane)

    def copy_block(self, dst_block: int, src_block: int, filled: int) -> None:
        """Copy-on-write's physical half: copy `filled` tokens' K/V rows from `src_block` into
        `dst_block`, every full-attention layer. `scheduler.py` decides whether/when this runs
        (`SessionCache.needs_cow`) and has already allocated `dst_block`; this only moves bytes.
        """
        if filled <= 0:
            return
        src_lo, dst_lo = src_block * self.block_size, dst_block * self.block_size
        src_rows, dst_rows = slice(src_lo, src_lo + filled), slice(dst_lo, dst_lo + filled)
        for i, lt in enumerate(self.cfg.layer_types):
            if lt != "full_attention":
                continue
            pool = self.pool[i]
            pool["k"][dst_rows] = pool["k"][src_rows]
            pool["v"][dst_rows] = pool["v"][src_rows]
        if self.mtp is not None:
            self.mtp.pool["k"][dst_rows] = self.mtp.pool["k"][src_rows]
            self.mtp.pool["v"][dst_rows] = self.mtp.pool["v"][src_rows]

    def save_snapshot(self, lane: int, snap: int) -> None:
        """Copy `lane`'s live DeltaNet/conv state into snapshot slot `snap`.

        Same copy Stage 1's `save_prefix` did, addressed by snapshot id into the independently
        sized `snapshot_pool` instead of by lane into a fixed one-per-lane register (see the
        module docstring). Also discards `lane`'s prefill scratch (paged-kv-design.md section
        3.2): this call marks the boundary the scheduler is recording (`len(prompt) -
        suffix_len`, which is this request's own prompt end when `suffix_len` is 0), so the
        scratch fed so far has done its job and would otherwise dangle, doubling this lane's
        resident KV bytes, until overwritten. When `suffix_len > 0` this request still prefills
        its own suffix on top of the snapshot afterwards, which repopulates the scratch past
        this boundary for the rest of this request; `full_attention` re-checks the cached
        length against `start` (not just `is None`) before trusting it, so a stale post-suffix
        scratch left resident into a *later* request's resume is re-gathered rather than
        reused. The paged pool stays the authoritative copy either way; a resumed sequence
        gathers it fresh from there (`_gather_cached_prefix`) whenever the cached scratch does
        not already cover it.
        """
        for st, snap_row in zip(self.slot_state[lane], self.snapshot_row(snap), strict=True):
            if snap_row is not None:
                for name, buf in snap_row.items():
                    buf.copy_(st[name])
        if self.mtp_hidden_snapshot is not None:
            self.mtp_hidden_snapshot[snap].copy_(self.hidden_scratch[lane])
        self._prefill_scratch[lane] = [None] * len(self.layers)

    def load_snapshot(self, lane: int, snap: int) -> None:
        """Copy the DeltaNet/conv state at snapshot slot `snap` into `lane`'s live state. KV
        rows stay valid."""
        for st, snap_row in zip(self.slot_state[lane], self.snapshot_row(snap), strict=True):
            if snap_row is not None:
                for name, buf in snap_row.items():
                    st[name].copy_(buf)
        if self.mtp_hidden_snapshot is not None:
            self.hidden_scratch[lane].copy_(self.mtp_hidden_snapshot[snap])

    @torch.no_grad()
    def prefill(self, slot: int, ids: list[int], start: int) -> torch.Tensor:
        """Forward one prompt chunk into `slot`; returns its last token's logits as [1, vocab]."""
        self.bind(slot)
        return self.forward(torch.tensor([list(ids)]), start)[-1:]

    @torch.no_grad()
    def score(self, slot: int, ids: list[int], continuation_start: int) -> torch.Tensor:
        """Teacher-forced logits for `ids[continuation_start:]`, one forward over the whole
        sequence. Mirrors `prefill`'s bind-then-forward shape but returns every scored
        position's logits (via `forward`'s `logits_from`), not just the last token's. Used by
        `/v1/score`; see server.py's `score_continuation`. Under tensor parallelism this must
        be called through the `Broadcaster` (like `prefill`), never on a bare rank-0 `Model`
        directly -- a direct call here skips broadcasting the op to the other ranks, which
        then never join this forward's collectives and hang forever.
        """
        self.bind(slot)
        return self.forward(torch.tensor([list(ids)]), 0, logits_from=continuation_start - 1)

    @torch.no_grad()
    def prefill_batch(self, calls: Sequence[tuple[int, Sequence[int], int]]) -> list[torch.Tensor]:
        """Packed cross-request prefill: one forward call for every (slot, ids, start) in `calls`.

        Each result is that call's own last-token logits, [1, vocab] -- the same contract
        `prefill` has for one call -- so a caller (`scheduler._prefill_step`) can use this in
        place of a `prefill` loop with no change to how it interprets the result. Falls back to
        exactly that loop when there is only one call (nothing to pack) or when
        `BATCHED_PREFILL` is off, which is the A/B knob against the per-request path this
        replaces; see the flag's own docstring.
        """
        if len(calls) == 1 or not BATCHED_PREFILL:
            return [self.prefill(slot, list(ids), start) for slot, ids, start in calls]
        seqs = tuple(PackedSeq(slot, start, len(ids)) for slot, ids, start in calls)
        meta = PackedBatch(seqs)
        flat_ids = torch.tensor([tok for _, ids, _ in calls for tok in ids])
        sample = False
        if step_timing.ENABLED:
            self._timing_prefills += 1
            sample = self._timing_prefills % step_timing.EVERY == 0
        t0 = time.perf_counter() if sample else 0.0
        # forward_packed already narrows to each sequence's last row: [len(seqs), vocab],
        # not [total_T, vocab], so this indexes by sequence position, not packed-span offset.
        logits = self.forward_packed(flat_ids, meta)
        if sample:
            self._log_prefill_timing(len(meta.seqs), meta.total, t0)
        return [logits[k : k + 1] for k in range(len(seqs))]

    def _log_prefill_timing(self, sequences: int, tokens: int, t0: float) -> None:
        """`SEED_STEP_TIMING` only, on the sampled call: this rank's wall time for one packed
        `forward_packed` call, with the sequence count and token count it actually packed --
        `scheduler._log_step`'s own `sched prefill` line already reports the scheduler-level
        wall time and token budget for the step this call is part of, but not this rank's own
        device time or a per-token rate for the packed forward itself. Syncs the device first
        (like `_log_decode_timing`) so `wall_ms` includes the kernels this call launched, not
        just how long the host took to issue them.
        """
        for dev in {d for d in self.devices if d.type == "cuda"}:
            torch.cuda.synchronize(dev)
        wall_ms = (time.perf_counter() - t0) * 1e3
        step_timing.log(
            f"rank {self.tp.plan.rank} eager prefill n={self._timing_prefills} "
            f"sequences={sequences} tokens={tokens} wall_ms={wall_ms:.1f} "
            f"ms_per_token={wall_ms / tokens:.3f}"
        )

    def decode_row(self, logits: torch.Tensor, row: int) -> torch.Tensor:
        """The [1, vocab]-shaped `row` of a batched `decode` result."""
        return logits[row : row + 1]

    def cache_node_logits(self, snap: int, logits: torch.Tensor) -> None:
        """Copy `logits` (a [1, vocab] result, from `prefill` or `decode_row`) into snapshot
        `snap`'s row of the static `node_logits_scratch` pool. A cached session wants to hold
        onto "the next-token logits for this recorded prefix" for as long as it stays in the
        trie, addressed by snapshot id rather than by lane since the session can long outlive
        whatever lane originally computed it (Stage 1's `cache_logits`, re-keyed). Copying into
        an already-reserved row, rather than keeping the fresh tensor the call returned, is
        what keeps that free (see `test_state_memory.py`)."""
        self.node_logits_scratch[snap : snap + 1].copy_(logits)

    def cached_node_logits(self, snap: int) -> torch.Tensor:
        """The [1, vocab] logits last passed to `cache_node_logits` for snapshot `snap`."""
        return self.node_logits_scratch[snap : snap + 1]

    def cache_hidden(self, slot: int, hidden: torch.Tensor) -> None:
        """Copy `hidden` (this slot's own raw, pre-final-norm [1, hidden] state at its last
        committed token) into its row of `hidden_scratch`. Only written to when `self.mtp is
        not None`: `mtp.draft` reads it back as the seed for that slot's next draft round."""
        self.hidden_scratch[slot : slot + 1].copy_(hidden)

    def cached_hidden(self, slot: int) -> torch.Tensor:
        """The [1, hidden] state last passed to `cache_hidden` for `slot`."""
        return self.hidden_scratch[slot : slot + 1]

    def _full_attention_index(self, i: int) -> int:
        """0-based index of layer `i` among full-attention layers only (fault injection)."""
        return sum(1 for j in range(i) if self.cfg.layer_types[j] == "full_attention")

    # -- layers ---------------------------------------------------------------
    def _physical_rows_range(self, slot: int, lo: int, hi: int, dev: torch.device) -> torch.Tensor:
        """Pool rows for `slot`'s tokens `[lo, hi)`, computed with tensor ops, not a Python list.

        `block_pool.BlockTable.physical_rows` builds one Python `int` per token of `[0, hi)`
        via a list comprehension, even when a caller wants only the last few of them --
        `full_attention`'s KV-write index used to call it for `[0, start + t)` and then slice
        `[start : start + t]`, throwing away everything but the last `t`, on *every*
        full-attention layer of *every* prefill chunk: O(context) host-side work per layer,
        not O(chunk). The address arithmetic itself (`block_id * block_size + offset`) is
        elementwise, so it runs as vectorized tensor ops instead: one short (`O(blocks)`, not
        `O(context)`) Python list for the table's own block ids -- the one piece that is
        genuinely host state -- turned into a tensor, then `arange`/`div`/`mod`/gather do the
        rest on-device. Freshly built every call, like the index tensor this replaces (see
        `full_attention`'s note on why that one is not `_host_index`-memoized): the table's own
        block list already changes when a chunk grows the table, so memoizing this by slot
        would just be a second cache with the same staleness problem, for no shrink in what
        stays live. Reads only ids already in the table (under TP, the rank-0-reserved ones
        `grow_lane` checked for); it never allocates.
        """
        table = self.block_tables[slot]
        blocks = torch.tensor(table.blocks, dtype=torch.long, device=dev)
        pos = torch.arange(lo, hi, device=dev)
        block_idx = torch.div(pos, self.block_size, rounding_mode="floor")
        return blocks[block_idx] * self.block_size + pos % self.block_size

    def _gather_cached_prefix(
        self, i: int, slot: int, length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`[1, kv_heads, length, head_dim]` K/V for `slot`'s first `length` paged tokens.

        `paged-kv-design.md` section 3.2's "gather-once" path: a single strided-index copy
        out of the block pool, sized to the resumed prefix -- not a per-decode-step tax
        (`_attend_batch`'s old per-step contiguous-view trick has no equivalent once storage
        is paged, see `paged_attn.py`'s module docstring), and not repeated per
        `PREFILL_CHUNK` either. `full_attention` only calls this once per prefill sequence (at
        the first chunk of a resumed request); every following chunk extends the resulting
        scratch tensor with freshly computed K/V instead of gathering anything.
        """
        pool = self.pool[i]
        idx = self._physical_rows_range(slot, 0, length, pool["k"].device)
        return pool["k"][idx].transpose(0, 1)[None], pool["v"][idx].transpose(0, 1)[None]

    def full_attention(
        self, i: int, x: torch.Tensor, start: int, slot: int | None = None
    ) -> torch.Tensor:
        """This rank's query heads' attention. The result is a row-parallel partial sum.

        Only the head counts change under TP: a head's whole computation, up to `o_proj`,
        touches no other head, so running `pl.q.count` of the 32 heads here and summing
        `o_proj`'s outputs across ranks is the unsharded result exactly. `pl.kv.count` is 1 at
        TP=4 and every local query head reads it (`tp.kv_shard` is what guarantees a rank's
        query block does not straddle two KV groups). Both SDPA spellings below survive that
        unchanged, which is why the sharding costs no new attention code: the `is_causal`
        branch reads the head counts off the tensors it is given, and `prefill_attention`
        derives its `group` as `q.shape[1] // keys.shape[1]`, which is 8 // 1 at TP=4 exactly
        as it is 32 // 2 unsharded.

        Paged-KV (Stage 1): the KV pool is now block-paged and shared across sessions, so this
        writes the chunk's K/V into `self.pool[i]` at the physical rows its block table maps
        `[start, start + t)` to (growing the table first, allocating fresh blocks if needed),
        instead of writing into a per-slot dense view. Prefill still needs a *contiguous*
        `[1, kv_heads, start + t, head_dim]` view for `prefill_attention`/`is_causal` SDPA
        (paging buys nothing here since prefill is not graph-captured either way -- see
        `paged-kv-design.md` section 3.2), which `self._prefill_scratch[slot][i]` provides:
        gathered once from the pool at the first chunk of a resumed prefix
        (`_gather_cached_prefix`), empty at the first chunk of a fresh one, and simply extended
        by `cat`-ing each following chunk's freshly computed K/V, which needs no gather at all.
        `Model.reset`/`save_snapshot` are what clear it back to `None` between prefill
        sequences, so it never persists into decode (see their docstrings).

        `slot` defaults to the bound slot, like `deltanet`'s own `st` parameter defaults to the
        bound slot's state; the packed-prefill path (`full_attention_packed`) passes a specific
        slot directly, because one packed call loops over several slots with no single "current
        slot" to bind -- each sequence's chunk still reads and writes only its own slot's block
        table and paged KV rows, keyed by this `slot`, not by any shared "current" one.
        """
        c, w, pl = self.cfg, self.layers[i], self.tp.plan
        slot = self.current_slot if slot is None else slot
        t = x.shape[1]
        nq, nkv = pl.q.count, pl.kv.count
        q, gate = F.linear(x, w["q_proj"]).view(1, t, nq, 2 * c.head_dim).chunk(2, dim=-1)
        q = rmsnorm(q, w["q_norm"], c.eps).transpose(1, 2)
        k = rmsnorm(
            F.linear(x, w["k_proj"]).view(1, t, nkv, c.head_dim), w["k_norm"], c.eps
        ).transpose(1, 2)
        v = F.linear(x, w["v_proj"]).view(1, t, nkv, c.head_dim).transpose(1, 2)
        cos, sin = self.rope(start, t, x)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if FAULT_FP8_KV_NO_SCALE:  # naive fp8 round-trip, no scale factor: truncation, not quant
            k = k.to(torch.float8_e4m3fn).to(k.dtype)
            v = v.to(torch.float8_e4m3fn).to(v.dtype)

        self.grow_lane(slot, start + t)
        # `_physical_rows_range` computes only the `t` rows this chunk needs, via tensor ops,
        # not `block_pool.physical_rows(start + t, ...)`'s O(start + t) Python list sliced down
        # to `t` afterward. Not `_host_index`-memoized, unlike `attn_decode`'s per-step index: a
        # prefill chunk's length (and so this tensor's length) varies with `t`, which is not the
        # fixed-batch hot path `_host_index`'s persistent per-(role, device) cache exists to
        # serve, and `full_attention` already allocates its `causal` mask fresh per chunk for
        # the same reason. Memoizing this made `Model`-level footprint depend on the last
        # prefill chunk's length seen (test_state_memory.py's steady-state byte count caught it).
        idx = self._physical_rows_range(slot, start, start + t, x.device)
        pool = self.pool[i]
        pool["k"][idx] = k[0].transpose(0, 1)
        pool["v"][idx] = v[0].transpose(0, 1)

        # Reusable only when it continues from exactly `start`: a request whose `save_snapshot`
        # boundary sits before its own prompt's end (`suffix_len > 0`, see scheduler.py's
        # module docstring) still prefills its suffix chunk-by-chunk *after* that snapshot,
        # so this slot's own scratch keeps extending past the saved boundary for the rest of
        # this request -- decode never touches or clears it. The next external request that
        # resumes this slot at the saved boundary then sees a stale, too-long scratch left
        # over from that suffix, not `None` (`load_snapshot`, called once per resume, rewinds
        # only the DeltaNet recurrent state, not this). Checking the cached length against
        # `start` (not just `is None`) makes this self-correcting regardless of exactly when
        # `save_snapshot`/`load_snapshot` last ran, at the cost of one extra gather on the rare
        # chunk where it actually was stale.
        scratch = self._prefill_scratch[slot][i]
        if scratch is None or scratch[0].shape[2] != start:
            scratch = (
                self._gather_cached_prefix(i, slot, start)
                if start
                else (k.new_zeros(1, nkv, 0, c.head_dim), v.new_zeros(1, nkv, 0, c.head_dim))
            )
        keys, vals = torch.cat([scratch[0], k], dim=2), torch.cat([scratch[1], v], dim=2)
        self._prefill_scratch[slot][i] = (keys, vals)

        # fused kernel: same causal mask (True = attend) and scale as the manual
        # softmax(QK^T/sqrt(d))V it replaces; enable_gqa broadcasts kv heads the
        # same way the old repeat_interleave(rep, dim=1) did.
        #
        # An explicit boolean `attn_mask` is what the mask below builds, but passing one
        # takes SDPA off its flash backend on gfx942 and onto the math path, which
        # materializes the full [heads, t, start+t] score tensor. Measured on MI300A, peak
        # allocated for one call: 447 MiB at T=1024, 1476 at 2048, 5280 at 4096 (about
        # quadratic), against 36/72/144 MiB for the same call under `is_causal` (linear), a
        # 36x gap at T=4096. With no prefix (`start == 0`) the mask is exactly
        # `j <= i`, which is what `is_causal=True` means, so the flag is a drop-in and the
        # score tensor never has to exist.
        #
        # It is only a drop-in there. For `start > 0` the mask is causal with an offset
        # (`j <= start + i`, every key in the prefix is visible to every query), while
        # `is_causal` on a non-square q/k aligns to the top-left and would hide most of the
        # prefix. Those calls keep the mask; their score tensor is O(t * (start + t)) with
        # t bounded by PREFILL_CHUNK, so it grows linearly in the prefix, not quadratically.
        #
        # `enable_gqa` has to go too once there is a mask: combined with an explicit
        # `attn_mask`, it takes SDPA off its fused backend on gfx942 the same way it does for
        # decode, and the fallback broadcasts the KV up to `heads` before the matmul. This
        # branch feeds `p95_ttft_turn2plus_ms` (resuming after a cached prefix), so
        # `prefill_attention` folds the grouped heads into the query length instead; see its
        # docstring for the measured decode-side cost of the spelling it avoids.
        flip_mask = FAULT_FLIP_CAUSAL_MASK and self._full_attention_index(i) == (
            FAULT_FLIP_CAUSAL_MASK_LAYER
        )
        if start == 0:
            out = F.scaled_dot_product_attention(
                q,
                keys,
                vals,
                is_causal=not flip_mask,  # flipped: bidirectional, queries see future keys too
                scale=c.head_dim**-0.5,
                enable_gqa=nq != nkv,
            )
        else:
            causal = (
                torch.arange(start + t, device=x.device)[None, :]
                <= (start + torch.arange(t, device=x.device))[:, None]
            )
            if flip_mask:
                causal = torch.ones_like(causal)  # every key visible, causal order ignored
            out = prefill_attention(q, keys, vals, causal, c.head_dim**-0.5)
        out = out.transpose(1, 2).reshape(1, t, -1)
        return F.linear(out * torch.sigmoid(gate.reshape(1, t, -1)), w["o_proj"])

    def full_attention_packed(self, i: int, x: torch.Tensor, meta: PackedBatch) -> torch.Tensor:
        """`full_attention`, once per sequence in `meta`, over `x`'s packed [total_T, hidden] rows.

        Each sequence attends only its own slot's paged KV (block table and pool rows) and
        writes only its own slot's rows (`full_attention` already does both, keyed by the
        `slot` it is given), so looping over `meta.seqs` and concatenating the per-sequence
        results back in order is exact: it is the same computation `_prefill_step` used to get
        by calling `prefill` once per request, just gathered into one Python-level call now
        that the token embeddings for every request arrive already packed. v1 keeps the loop
        here (see the module docstring): the GEMMs inside `full_attention`, not this loop,
        dominate cost.

        `SEED_VARLEN_PREFILL_ATTN=1` (`varlen_prefill_attn.available`) replaces the loop's
        attention step (each iteration's `_gather_cached_prefix` gather plus
        `prefill_attention`/SDPA call) with one packed kernel launch across every sequence in
        `meta`, reading each one's cached prefix straight out of the paged pool through its
        block table instead of gathering it into a contiguous scratch tensor first -- see
        `varlen_prefill_attn.py`. `_full_attention_packed_fused` still runs the projections,
        RoPE, and paged-KV write per sequence (each needs its own `start` offset and block
        table), unchanged from `full_attention`'s own math; only the attention computation and
        `o_proj` move to one packed call each.
        """
        if varlen_prefill_attn.available(x.device):
            return self._full_attention_packed_fused(i, x, meta)
        outs = []
        for seq, (lo, hi) in zip(meta.seqs, meta.spans(), strict=True):
            xi = x[lo:hi].unsqueeze(0)
            outs.append(self.full_attention(i, xi, seq.start, slot=seq.slot)[0])
        return torch.cat(outs, dim=0)

    def _full_attention_packed_fused(
        self, i: int, x: torch.Tensor, meta: PackedBatch
    ) -> torch.Tensor:
        """`SEED_VARLEN_PREFILL_ATTN=1` path for `full_attention_packed`: projections, RoPE,
        and the paged-KV write per sequence (identical math to the corresponding lines of
        `full_attention`, just not routed through it, since `full_attention` also runs the
        gather + SDPA this path replaces), then one `varlen_prefill_attn` kernel call across
        the whole packed batch, then one packed `o_proj` over every sequence's rows at once
        (`full_attention`'s own `o_proj` is already per-sequence-shape-invariant, so batching
        it here is exact and removes `len(meta.seqs) - 1` more dispatches on top of the
        attention kernel's own savings).

        Does not thread `FAULT_FLIP_CAUSAL_MASK`/`SEED_FAULT_FP8_KV_NO_SCALE`'s mask-flip
        branch (`FAULT_FP8_KV_NO_SCALE` still applies to the KV write below; the causal-flip
        fault is `full_attention`-only): this path is a performance path, not a debug-fault
        target, and `full_attention_packed`'s v1 loop remains the correctness oracle for both
        faults and for this path itself (`test_varlen_prefill_attn.py`'s independent oracle is
        the kernel-level check; `SEED_VARLEN_PREFILL_ATTN=0`, the default, is the fallback A/B).
        """
        c, w, pl = self.cfg, self.layers[i], self.tp.plan
        nq, nkv = pl.q.count, pl.kv.count
        total_t, dev = x.shape[0], x.device
        pool = self.pool[i]

        q_all = x.new_empty(total_t, nq, c.head_dim)
        gate_all = x.new_empty(total_t, nq, c.head_dim)
        for seq, (lo, hi) in zip(meta.seqs, meta.spans(), strict=True):
            t = hi - lo
            xi = x[lo:hi].unsqueeze(0)
            q, gate = F.linear(xi, w["q_proj"]).view(1, t, nq, 2 * c.head_dim).chunk(2, dim=-1)
            q = rmsnorm(q, w["q_norm"], c.eps).transpose(1, 2)
            k = rmsnorm(
                F.linear(xi, w["k_proj"]).view(1, t, nkv, c.head_dim), w["k_norm"], c.eps
            ).transpose(1, 2)
            v = F.linear(xi, w["v_proj"]).view(1, t, nkv, c.head_dim).transpose(1, 2)
            cos, sin = self.rope(seq.start, t, x)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            if FAULT_FP8_KV_NO_SCALE:
                k = k.to(torch.float8_e4m3fn).to(k.dtype)
                v = v.to(torch.float8_e4m3fn).to(v.dtype)

            # Rank-0-reserved blocks under TP (see `grow_lane`): never allocate here.
            self.grow_lane(seq.slot, seq.start + t)
            idx = self._physical_rows_range(seq.slot, seq.start, seq.start + t, dev)
            pool["k"][idx] = k[0].transpose(0, 1)
            pool["v"][idx] = v[0].transpose(0, 1)

            q_all[lo:hi] = q[0].transpose(0, 1)
            gate_all[lo:hi] = gate[0]

        max_blocks_row = max(len(self.block_tables[seq.slot].blocks) for seq in meta.seqs)
        block_table = torch.tensor(
            [self.block_tables[seq.slot].padded_row(max_blocks_row) for seq in meta.seqs],
            dtype=torch.int32,
            device=dev,
        )
        spans = meta.spans()
        seq_start = torch.tensor([lo for lo, _ in spans], dtype=torch.int32, device=dev)
        seq_len = torch.tensor([seq.length for seq in meta.seqs], dtype=torch.int32, device=dev)
        seq_prefix = torch.tensor([seq.start for seq in meta.seqs], dtype=torch.int32, device=dev)

        out = varlen_prefill_attn.varlen_prefill_attention_paged(
            q_all,
            pool["k"],
            pool["v"],
            block_table,
            seq_start,
            seq_len,
            seq_prefix,
            self.block_size,
            c.head_dim**-0.5,
        )
        out = out.reshape(total_t, -1)
        return F.linear(out * torch.sigmoid(gate_all.reshape(total_t, -1)), w["o_proj"])

    def _rope_table(self, dev: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin for every position, built once per (device, fault-flag state) and cached
        (see `rope`). The cache key includes `FAULT_ROPE_BASE`'s current value so flipping it
        at runtime (`/debug/fault`) takes effect on the next forward instead of continuing to
        serve a table built under the flag's old value.
        """
        key = (dev, FAULT_ROPE_BASE)
        cached = self.rope_cache.get(key)
        if cached is not None:
            return cached
        c = self.cfg
        # 1e-3x the checkpoint's own base: wrong by three orders of magnitude regardless of
        # the checkpoint (the real model's base is 1e7; scaling relative to `c.rope_theta`
        # keeps this a "wrong base" fault on any checkpoint, including tiny test configs,
        # instead of a hardcoded constant that could coincidentally match one).
        theta = c.rope_theta * 1e-3 if FAULT_ROPE_BASE else c.rope_theta
        inv = 1.0 / (theta ** (torch.arange(0, c.rot_dim, 2, device=dev).float() / c.rot_dim))
        pos = torch.arange(self.max_seq, device=dev).float()
        emb = torch.cat([pos[:, None] * inv[None], pos[:, None] * inv[None]], dim=-1)
        table = (emb.cos().to(self.dtype), emb.sin().to(self.dtype))
        self.rope_cache[key] = table
        return table

    def rope(self, start: int, t: int, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice the per-device rope table instead of recomputing sin/cos every call."""
        cos, sin = self._rope_table(x.device)
        return cos[start : start + t].to(x.dtype), sin[start : start + t].to(x.dtype)

    def rope_at(self, pos: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin [len(pos), rot_dim] for arbitrary positions (one per batch row).

        Same cached table as `rope`, gathered instead of sliced: a batched decode step holds
        one position per slot, and those are not contiguous.
        """
        cos, sin = self._rope_table(pos.device)
        return cos[pos].to(dtype), sin[pos].to(dtype)

    def deltanet(self, i: int, x: torch.Tensor, st: dict | None = None) -> torch.Tensor:
        """Gated DeltaNet layer `i` over one sequence, on this rank's value heads.

        `st` defaults to the bound slot's state; the batched decode path passes it
        explicitly, because several microbatches on different slots are inside the model at
        once and there is no single "current slot" to bind.

        The result is a row-parallel partial sum. The recurrence is block diagonal over value
        heads, so nothing here needs the other ranks' heads and nothing here communicates:
        only `out_proj` leaves a partial, which `layer` all-reduces exactly as it does full
        attention's `o_proj`. `self.cfg` already carries the local head counts, so every
        shape below is this rank's share with no further change. See `deltanet_tp` for the
        derivation.
        """
        c, w = self.cfg, self.layers[i]
        st = self.state[i] if st is None else st
        t = x.shape[1]
        key_dim, val_dim = c.k_heads * c.k_dim, c.v_heads * c.v_dim
        if PREFILL_FUSED_IN_PROJ and "in_proj_all" in w:
            qkv, z, b_raw, a_raw = F.linear(x, w["in_proj_all"]).split(in_proj_sizes(c), dim=-1)
        else:
            qkv, z = F.linear(x, w["in_proj_qkv"]), F.linear(x, w["in_proj_z"])
            b_raw, a_raw = F.linear(x, w["in_proj_b"]), F.linear(x, w["in_proj_a"])
        z = z.reshape(1, t, c.v_heads, c.v_dim)
        mixed = self.causal_conv(qkv.transpose(1, 2), w["conv"], st["conv"])
        q, k, v = mixed.transpose(1, 2).split([key_dim, key_dim, val_dim], dim=-1)
        q, k = q.reshape(1, t, c.k_heads, c.k_dim), k.reshape(1, t, c.k_heads, c.k_dim)
        v = v.reshape(1, t, c.v_heads, c.v_dim)
        beta = b_raw.sigmoid()
        g = -w["A_log"].exp() * F.softplus(a_raw.float() + w["dt_bias"])
        rep = c.v_heads // c.k_heads
        q, k = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)
        # Decode-only (T=1) write buffer for the currently bound slot; delta_rule ignores it
        # when T > 1 (prefill), whose length it does not match.
        scratch = self.delta_scratch[i][self.current_slot : self.current_slot + 1]
        out = self.delta_rule(q, k, v, g, beta, st["rec"], scratch).to(x.dtype)
        out = gated_rmsnorm(out.reshape(-1, c.v_dim), z.reshape(-1, c.v_dim), w["dn_norm"], c.eps)
        return F.linear(out.reshape(1, t, -1), w["out_proj"])

    def deltanet_packed(self, i: int, x: torch.Tensor, meta: PackedBatch) -> torch.Tensor:
        """`deltanet` per sequence, or `_deltanet_packed_vec` under `SEED_PACKED_PREFILL_VEC=1`.

        The per-sequence loop runs each sequence against its own slot's carried conv/recurrent
        state, exactly where a one-request-at-a-time `prefill` loop would have left it.
        """
        if PACKED_PREFILL_VEC:
            return self._deltanet_packed_vec(i, x, meta)
        outs = []
        for seq, (lo, hi) in zip(meta.seqs, meta.spans(), strict=True):
            xi = x[lo:hi].unsqueeze(0)
            outs.append(self.deltanet(i, xi, self.slot_state[seq.slot][i])[0])
        return torch.cat(outs, dim=0)

    def _deltanet_packed_vec(self, i: int, x: torch.Tensor, meta: PackedBatch) -> torch.Tensor:
        """`deltanet` over a whole packed prefill batch: one set of projections, one causal
        conv, one delta-rule call, instead of the full per-sequence stack repeated
        `len(meta.seqs)` times.

        `x` is `[total_T, hidden]`, packed end to end. Every projection (`in_proj_z`,
        `in_proj_qkv`, `in_proj_b`, `in_proj_a`) is already batch-shape-invariant over its
        leading axis, so running each once over all `total_T` rows -- rather than once per
        sequence inside a Python loop, as calling `deltanet` per sequence did -- is exact and is
        what actually removes the per-sequence cost: the GEMMs, not the loop itself, dominate.
        Only the two stateful pieces need to know sequence boundaries at all:
        `_causal_conv_packed` (each sequence's own carried conv state, no cross-sequence
        leakage) and `_delta_rule_packed` (each sequence's own carried recurrent state, via
        `cu_seqlens` on an accelerator). Both gather their slot's rows out of this layer's state
        pool, advance them in place, and scatter back -- the same gather/compute/scatter shape
        `deltanet_decode` already uses for batched decode, just with each sequence contributing
        its own chunk length instead of always one token.
        """
        c, w, pool = self.cfg, self.layers[i], self.pool[i]
        total_t = x.shape[0]
        key_dim, val_dim = c.k_heads * c.k_dim, c.v_heads * c.v_dim
        xb = x.unsqueeze(0)
        rows = self._host_index("packed_slots", x.device, tuple(seq.slot for seq in meta.seqs))

        z = F.linear(xb, w["in_proj_z"]).view(1, total_t, c.v_heads, c.v_dim)
        qkv_in = F.linear(xb, w["in_proj_qkv"]).transpose(1, 2)  # [1, conv_dim, total_T]
        conv_state = pool["conv"][rows]  # fancy indexing copies: safe to mutate below
        mixed, conv_state = self._causal_conv_packed(qkv_in, w["conv"], conv_state, meta)
        pool["conv"][rows] = conv_state

        q, k, v = mixed.transpose(1, 2).split([key_dim, key_dim, val_dim], dim=-1)
        q, k = q.reshape(1, total_t, c.k_heads, c.k_dim), k.reshape(1, total_t, c.k_heads, c.k_dim)
        v = v.reshape(1, total_t, c.v_heads, c.v_dim)
        beta = F.linear(xb, w["in_proj_b"]).sigmoid()
        g = -w["A_log"].exp() * F.softplus(F.linear(xb, w["in_proj_a"]).float() + w["dt_bias"])
        rep = c.v_heads // c.k_heads
        q, k = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)

        rec_state = pool["rec"][rows]  # same copy-then-scatter-back pattern as conv_state
        out = self._delta_rule_packed(q, k, v, g, beta, rec_state, meta).to(x.dtype)
        pool["rec"][rows] = rec_state

        out = gated_rmsnorm(out.reshape(-1, c.v_dim), z.reshape(-1, c.v_dim), w["dn_norm"], c.eps)
        return F.linear(out.reshape(total_t, -1), w["out_proj"])

    @staticmethod
    def _causal_conv_packed(
        x: torch.Tensor, weight: torch.Tensor, conv_state: torch.Tensor, meta: PackedBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One `F.conv1d` call for a whole packed prefill batch, each sequence keeping its own
        causal window and carried state -- no padding to a common length, and no cross-sequence
        mixing.

        `x` is `[1, C, total_T]` (channel-major, packed across `meta.seqs`); `conv_state` is
        `[n_seqs, C, K-1]`, each sequence's own carried state (already gathered by the caller,
        in `meta.seqs` order). `causal_conv`'s own math for sequence `i` needs only
        `[state_i, x_i]` (length `K - 1 + T_i`): building that block for every sequence and
        concatenating them end to end, then running *one* `conv1d` over the whole thing, is
        exact -- a `K`-wide window ending anywhere inside `x_i` reaches back at most `K - 1`
        positions, i.e. never earlier than `state_i`'s own start, so it never crosses into a
        different sequence's block. Sequence `i`'s output is exactly
        `conv_out[start_i : start_i + T_i]`, `start_i` being where its own `[state_i, x_i]`
        block begins in the concatenation; its new carried state is the last `K - 1` columns of
        that same block, sliced out of the padded input (`padded`), the same slice
        `causal_conv` itself would take.
        """
        k_minus_1 = weight.shape[-1] - 1
        pieces: list[torch.Tensor] = []
        starts: list[int] = []
        lengths: list[int] = []
        offset = 0
        for row, (_, (lo, hi)) in enumerate(zip(meta.seqs, meta.spans(), strict=True)):
            length = hi - lo
            pieces.append(conv_state[row : row + 1])
            pieces.append(x[:, :, lo:hi])
            starts.append(offset)
            lengths.append(length)
            offset += k_minus_1 + length
        padded = torch.cat(pieces, dim=-1)  # [1, C, total_T + n_seqs * (K - 1)]
        conv_out = F.silu(F.conv1d(padded, weight, groups=padded.shape[1]))

        out_pieces, state_pieces = [], []
        for start, length in zip(starts, lengths, strict=True):
            out_pieces.append(conv_out[:, :, start : start + length])
            state_pieces.append(padded[:, :, start + length : start + length + k_minus_1])
        mixed = torch.cat(out_pieces, dim=-1)
        new_state = torch.cat(state_pieces, dim=0)
        return mixed, new_state

    def _delta_rule_packed(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        rec_state: torch.Tensor,
        meta: PackedBatch,
    ) -> torch.Tensor:
        """`delta_rule`'s prefill dispatch, run once for the whole packed batch on an
        accelerator instead of once per sequence.

        `fused_recurrent_prefill`'s own `cu_seqlens` and stacked `initial_state` parameters
        exist for exactly this (see its docstring): its grid already has a sequence axis, so one
        launch over the packed `[total_T, ...]` tensors does what a per-sequence loop of
        launches did, with `rec_state` (this layer's gathered `[n_seqs, heads, k_dim, v_dim]`
        state) advanced in place for every sequence in that one call.

        CPU (no Triton, what the hermetic tests exercise) has no packed kernel to call, and the
        recurrence is inherently sequential in time -- not something a packed layout removes the
        loop from -- so this falls back to `delta_rule`'s own single-sequence dispatch, once per
        sequence, same as before the projections and the conv were vectorized. That keeps the
        CPU path correct without a second, packed-aware reimplementation of the recurrence.
        """
        if deltanet_fused.available_prefill(q.device):
            spans = meta.spans()
            cu_seqlens = torch.tensor(
                [0, *(hi for _, hi in spans)], dtype=torch.int32, device=q.device
            )
            out, _ = deltanet_fused.fused_recurrent_prefill(
                q[0],
                k[0],
                v[0].contiguous(),
                g[0].float(),
                beta[0],
                rec_state,
                cu_seqlens,
                chunked=deltanet_prefill_chunked.use_chunked(q.shape[1]),
            )
            return out.unsqueeze(0)
        outs = []
        for row, (lo, hi) in enumerate(meta.spans()):
            outs.append(
                self.delta_rule(
                    q[:, lo:hi],
                    k[:, lo:hi],
                    v[:, lo:hi],
                    g[:, lo:hi],
                    beta[:, lo:hi],
                    rec_state[row : row + 1],
                )
            )
        return torch.cat(outs, dim=1)

    @staticmethod
    def causal_conv(x: torch.Tensor, weight: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Depthwise causal conv over [1, C, T] with the previous K-1 inputs in `state` (updated in place)."""
        full = torch.cat([state, x], dim=-1)
        state.copy_(full[:, :, -state.shape[-1] :])
        return F.silu(F.conv1d(full, weight, groups=full.shape[1]))

    @staticmethod
    def delta_rule(q, k, v, g, beta, rec, out=None) -> torch.Tensor:  # noqa: ANN001
        """Gated delta rule over q,k,v [1,T,H,d] and g,beta [1,T,H]; `rec` advances in place.

        Decode (`T == 1`) always takes the per-token recurrence: it is already the fast path
        there (`out` is the preallocated T=1 write buffer, only usable at this length).

        Prefill (`T > 1`) dispatches, in order:

        1. `deltanet_fused.fused_recurrent_prefill`, one Triton kernel that loops over every
           position internally (one program per `(sequence, value head, v-block)`, the state
           tile held in registers for the whole sequence instead of round-tripping through
           `rec` once per token), when `deltanet_fused.available_prefill` says an accelerator
           is present. This is the fast path: same math as the recurrence below, at roughly
           the decode kernel's per-token cost instead of a Python-level dispatch per token.
        2. `deltanet_chunked.delta_rule_chunked`, the WY-representation parallel form with the
           intra-chunk solve as a Triton kernel (`deltanet_chunked.lower_tri_solve`), when
           `deltanet_chunked.available` says so. Off by default (`SEED_CHUNKED_DELTANET=0`):
           the previous chunked form (this module's `delta_rule_chunked`, kept below for the
           CPU parity tests) used `torch.linalg.solve_triangular` for that solve, which
           measured about 10x *slower* than the recurrence on 4x MI300A (gfx942); rocBLAS's
           batched-matmul heuristics make even the Triton-solve version of this path ~100x
           slower than the recurrence at the real per-rank shape, so it is not the default and
           exists for future rocBLAS-heuristic work. See `deltanet_chunked.py`'s module
           docstring for the kernel and `DECODE_BOTTLENECK`-style perf notes for detail.
        3. The recurrence, unconditionally correct and the only path with no Triton
           dependency (CPU, or either fused/chunked flag forced off).

        `out` is an optional preallocated write buffer, sized for the T=1 decode step; ignored
        for T > 1 (a prefill would otherwise index a one-token buffer at position T-1).
        """
        t = q.shape[1]
        if t != 1:
            out = None
        if t > 1 and deltanet_fused.available_prefill(q.device):
            # v (unlike q/k, which repeat_interleave already makes contiguous) reaches here as
            # a transpose-then-split view of `causal_conv`'s [1, C, T] output, so its last axis
            # has stride T, not 1 -- `fused_recurrent_prefill` requires stride 1 there.
            fused_out, _ = deltanet_fused.fused_recurrent_prefill(
                q[0],
                k[0],
                v[0].contiguous(),
                g[0].float(),
                beta[0],
                rec,
                chunked=deltanet_prefill_chunked.use_chunked(t),
            )
            return fused_out.unsqueeze(0)
        if t > 1 and deltanet_chunked.available(q.device):
            return deltanet_chunked.delta_rule_chunked(q, k, v, g, beta, rec, chunk=DELTA_CHUNK)
        return delta_rule_recurrent(q, k, v, g, beta, rec, out)

    # -- MoE ------------------------------------------------------------------
    def expert_weights_batch(
        self, ex: dict, idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dense (gate_up, down) for a batch of expert indices idx [Ea].

        One dequant call for all Ea experts at once (not one call per expert): `dequant_mxfp4`
        is already elementwise/broadcasting over leading dims, so indexing the whole batch of
        activated experts before calling it does the batching.
        """
        if "gate_up_scale" not in ex:
            return ex["gate_up"][idx], ex["down"][idx]
        return (
            dequant_mxfp4(ex["gate_up"][idx], ex["gate_up_scale"][idx], self.dtype),
            dequant_mxfp4(ex["down"][idx], ex["down_scale"][idx], self.dtype),
        )

    def observe_routing(self, i: int, top_i: torch.Tensor) -> None:
        """Accumulate this decode step's local-expert assignment counts for layer `i`.

        Called from `moe` while `SEED_HOT_EXPERTS=1` and the cache has not yet been frozen
        (`hot_experts_finalized` is False); a no-op once frozen or when layer `i` has no
        experts, so steady-state decode never pays this host-syncing bincount. `top_i` is the
        router's raw global ids (pre-rank-filter, same convention as `_routed_grouped`'s
        `flat_expert`).
        """
        counts = self.hot_counts[i]
        if counts is None:
            return
        lo, hi = self.expert_range
        flat = top_i.reshape(-1)
        local = flat[(flat >= lo) & (flat < hi)] - lo
        if local.numel() == 0:
            return
        counts += torch.bincount(local, minlength=counts.numel())

    def finalize_hot_experts(self) -> None:
        """Freeze each layer's hot-expert bf16 cache from accumulated `hot_counts`.

        Call once, after warmup traffic and before graph capture (`Model.warmup` does this
        when `SEED_HOT_EXPERTS=1`): every decode step after this is a fixed hot/cold split, so
        nothing device-side depends on a host read for it. Idempotent -- a second call rebuilds
        from whatever counts have accumulated since the last one, which is the shape a
        production periodic EPLB-style rebalance would take, though nothing here schedules
        that automatically; see hot_experts.py's module docstring.

        `Model.warmup`'s own traffic (a handful of dummy tokens) is not representative real
        traffic, so the resulting hot set is only as good as whatever called `observe_routing`
        before this ran; a deployment wanting the coverage `HOT_EXPERTS_ANALYSIS.md` reports
        should extend warmup with real-shaped prompts, not change anything in this module.
        """
        if not hot_experts.ENABLED:
            return
        n_local_experts = self.expert_range[1] - self.expert_range[0]
        experts_layers = [layer["experts"] for layer in self.layers if "experts" in layer]
        if not experts_layers:
            return
        bytes_per_expert = hot_experts.per_expert_bf16_bytes(experts_layers[0])
        h = hot_experts.budget_h_per_layer(
            hot_experts.BUDGET_GIB, len(experts_layers), bytes_per_expert, n_local_experts
        )
        for i, layer in enumerate(self.layers):
            counts = self.hot_counts[i]
            if counts is None:
                continue
            ids = hot_experts.select_hot_local(counts, h)
            self.hot_cache[i] = hot_experts.build_layer_cache(layer["experts"], ids, self.dtype)
        self.hot_experts_finalized = True

    def moe(
        self,
        i: int,
        x: torch.Tensor,
        out_buf: torch.Tensor | None = None,
        *,
        verify_width: int = 0,
        step_shared: bool = False,
        step_routed: bool = False,
    ) -> torch.Tensor:
        """Route tokens to experts, run this rank's share, and add the shared expert.

        Under TP the result is a row-parallel partial sum, which `layer`/`decode_layer`
        all-reduce. Expert parallel: rank r owns experts `[e0, e0 + 512/world)` whole, rather
        than a column shard of all 512. Activations stay replicated, so a rank simply drops
        the (token, expert) pairs it does not own; there is no all-to-all, and the combine is
        the same all-reduce the row-parallel projections already need. The routed output is a
        sum over the selected experts, so summing the ranks' partials reproduces the
        unsharded sum. Routing is replicated, not communicated: every rank holds the whole
        router and the same hidden state, so all ranks select the same experts.

        Chosen over sharding each expert's intermediate axis (the other standard option)
        because each rank then reads a quarter of the 205 GB of expert weights instead of all
        of it, which is the term the decode step is built around. The cost is load imbalance:
        a rank holding more of the step's hot experts holds up the all-reduce. It is bounded
        at prefill, where ~512x10 assignments spread over 512 experts leave each quarter with
        a near-equal share, and largest at small decode batches.

        The shared expert is column/row-parallel over its intermediate axis, so its output is
        a partial sum too. `shared_gate` and `h` are replicated, so scaling each rank's
        partial by the same sigmoid and summing gives `gate * (sum of partials)`, the whole
        term.

        Two spellings of the routed half, chosen by `mxfp4_gemv.available`:

        - **Fused (MXFP4 payloads on an accelerator).** Two Triton kernels read the packed
          weights directly and dequantize in registers, so no dense weight copy is written to
          HBM, and the expert-parallel drop and the zero-weight skip are GPU-side branches.
          Nothing in it reads a device value back to the host and every shape is a function of
          `t` and `top_k` alone, so the whole MoE is sync-free and capturable.
        - **Grouped torch (`_routed_grouped`).** The pre-kernel path, kept for dense
          checkpoints and for CPU, where Triton cannot compile. It sizes its `bmm` by the
          number of distinct experts the batch activated and by the busiest group, both of
          which are host reads.

        `out_buf`, when given, is a preallocated `[>=t, hidden]` buffer for the token-sized
        accumulator, which decode sizes to `max_batch` and callers slice to this call's `t`;
        omitting it allocates fresh (prefill's call site does not pass one).
        """
        c, w = self.cfg, self.layers[i]
        h = x.reshape(-1, c.hidden)
        # One GEMM for the router and the shared gate (`fuse_moe_dense`); the split is a view,
        # and the softmax is over the router's columns alone, as it was when they were two.
        routing = skinny_hip.linear(h, w["router_gate"])
        decode_stamps.mark("dense_router")
        experts_out = w["router"].shape[0]
        if (
            ROUTE_FUSED
            and self.moe_hip
            and router_fused.available(h.device)
            and h.shape[0] <= moe_hip.MAX_TOKENS
            and not FAULT_DROP_EXPERT
            and not (hot_experts.ENABLED and self.hot_cache[i] is not None)
        ):
            return self._moe_hip_route_fused(
                i,
                h,
                routing,
                experts_out,
                out_buf,
                verify_width=verify_width,
                step_shared=step_shared,
                step_routed=step_routed,
            ).reshape(x.shape)
        if step_shared or step_routed:
            raise RuntimeError("component-step MoE diagnostics require the fused HIP route path")
        if FUSE_GLUE and router_fused.available(h.device):
            # One kernel: softmax -> top-k -> renormalize, in place of four torch ops (see
            # `router_fused.py`'s module docstring for what it costs and its one real
            # semantic risk -- top-k tie-breaking -- next to the fp32-rounding-order-only
            # divergence this codebase's other fused kernels carry).
            top_w, top_i = router_fused.route(routing, experts_out, c.top_k)
        else:
            probs = routing[:, :experts_out].softmax(-1, dtype=torch.float)
            top_w, top_i = probs.topk(c.top_k, dim=-1)
            top_w = top_w / top_w.sum(-1, keepdim=True)
        if FAULT_DROP_EXPERT:  # zero expert 0's weight post-renormalization: missing mass
            top_w = top_w.masked_fill(top_i == 0, 0.0)

        # `SEED_HOT_EXPERTS=1`: before the cache exists, accumulate this step's routing into
        # `hot_counts` (see `observe_routing`) and run the cold path alone, same as the flag
        # being off. Once frozen, `hot_expert_forward` computes the hot experts' dense bf16
        # contribution and reports which flat `(token, slot)` assignments it consumed; zeroing
        # `top_w` there before the cold call is the same "zero-weight skip" `FAULT_DROP_EXPERT`
        # above already relies on (`mxfp4_gemv.py`'s `a_weight != 0` gate), so a hot assignment
        # is computed exactly once and the cold path needs no changes.
        hot_out = None
        if hot_experts.ENABLED and self.hot_cache[i] is not None:
            if not self.hot_experts_finalized:
                self.observe_routing(i, top_i)
            elif self.hot_cache[i].ids.numel() > 0:
                hot_out, consumed = hot_experts.hot_expert_forward(
                    h, top_i, top_w, self.expert_range[0], self.hot_cache[i]
                )
                top_w = top_w.masked_fill(consumed, 0.0)

        ex = w["experts"]
        if "gate_up_scale" in ex and mxfp4_gemv.available(h.device):
            out = self._routed_fused(i, h, top_i, top_w, out_buf)
        else:
            out = self._routed_grouped(i, h, top_i, top_w.to(h.dtype), out_buf)
        if hot_out is not None:
            out = out + hot_out
        shared = swiglu_mlp(h, w["shared_expert.gate_up_proj"], w["shared_expert.down_proj"])
        out = out + torch.sigmoid(routing[:, experts_out:]) * shared
        return out.reshape(x.shape)

    def _moe_hip_route_fused(
        self,
        i: int,
        h: torch.Tensor,
        routing: torch.Tensor,
        experts_out: int,
        out_buf: torch.Tensor | None,
        *,
        verify_width: int = 0,
        step_shared: bool = False,
        step_routed: bool = False,
    ) -> torch.Tensor:
        """`SEED_MOE_ROUTE_FUSED`'s decode path: `moe`'s router GEMM output straight through to
        the final MoE output, in as few launches as the router's per-token parallelism and
        `bw_prep`'s single-program counting sort allow.

        Kernels this replaces (`SEED_MOE_HIP=1`, `SEED_FUSE_GLUE=0`, the production default
        this flag is meant to run under): `softmax`, `topk` (itself more than one launch on
        CUDA), `sum`, `div`, and the `top_i.to(torch.int32)` cast that `_routed_fused` needs
        before `mxfp4_gemv.bw_prep` -- six-plus torch launches -- become two: `router_fused.route`
        (softmax/top-k/renormalize, one launch, `index_dtype=torch.int32` so its `top_i` is
        already `bw_prep`'s `a_expert` dtype) and `mxfp4_gemv.bw_prep` (the counting sort/work
        list, unchanged). These stay two launches rather than one: `route` parallelizes one
        program per token (up to `t` blocks across CUs), while `bw_prep`'s counting sort needs
        every assignment visible to a single program's atomics for a global scan, so serializing
        the per-token softmax into that single-program shape would trade the router's CU-level
        parallelism for one fewer launch -- not attempted here (documented as a follow-up, not
        implemented in this pass). On the combine side, `bw_combine` + `sigmoid` + `mul` + `add`
        (four launches) become one: `moe_hip.fused_moe_hip_glued`'s combine
        (`mxfp4_gemv.bw_combine_glue`). Net: 10 routing/glue launches per layer become 3
        (route, bw_prep, fused combine); the two `moe_hip` GEMV launches (gate_up, down) and the
        three dense GEMM launches (router+gate, shared gate_up, shared down) are unchanged.

        Routing-equivalence caveat: `router_fused.route` selects top-k by `tl.argmax`, which
        need not break an exact softmax tie the same way `torch.topk`'s CUDA implementation
        does (undocumented/implementation-defined there). This is the same, already-documented
        risk `SEED_FUSE_GLUE` carries (see `router_fused.py`'s module docstring) -- inherited,
        not introduced, by reusing that kernel here -- and is a measure-zero event over
        continuous router logits, not a rounding-order difference. `bw_combine_glue` accumulates
        the routed sum and the shared-expert term in fp32 before one final cast, instead of
        `bw_combine` rounding to `out`'s bf16 dtype first and adding the shared term in bf16
        after; that is the ordinary fp32-reduction-order divergence this codebase's other fused
        kernels already carry, not a routing difference.

        Callers must have already checked `self.moe_hip`, `router_fused.available`,
        `h.shape[0] <= moe_hip.MAX_TOKENS`, and that hot experts / `SEED_FAULT_DROP_EXPERT` are
        both off (this path does not implement either); see `moe`'s call site.
        """
        if step_shared and step_routed:
            raise ValueError("step_shared and step_routed are mutually exclusive")
        if (step_shared or step_routed) and (verify_width < 2 or h.shape[0] % verify_width):
            raise ValueError("component-step MoE requires a valid verify_width")

        c, w = self.cfg, self.layers[i]
        t = h.shape[0]

        def time_major(value: torch.Tensor) -> torch.Tensor:
            batch = t // verify_width
            return value.view(batch, verify_width, *value.shape[1:]).transpose(0, 1).contiguous()

        if step_routed:
            h_time = time_major(h)
            shared_time = time_major(
                swiglu_mlp(h, w["shared_expert.gate_up_proj"], w["shared_expert.down_proj"])
            )
            outputs = []
            for step in range(verify_width):
                # The router GEMM is also width-sensitive: at B48, its wide M=B*T spelling
                # moved later time rows by up to 0.03125 and their normalized top-k weights
                # by 5.96e-8. Reusing that wide routing here therefore preserved the exact
                # drift this diagnostic is meant to remove. Route each contiguous M=B row
                # together with its routed experts; the shared expert remains wide because
                # its output is bit exact across these widths.
                routing_step = skinny_hip.linear(h_time[step], w["router_gate"])
                top_w, top_i = router_fused.route(
                    routing_step, experts_out, c.top_k, index_dtype=torch.int32
                )
                routed = (top_i.reshape(-1), top_w.reshape(-1))
                outputs.append(
                    moe_hip.fused_moe_hip_glued(
                        h_time[step],
                        w["experts"],
                        routed,
                        c.top_k,
                        self.expert_range,
                        shared_time[step],
                        routing_step[:, experts_out:],
                    )
                )
            return torch.stack(outputs, dim=1).reshape(t, c.hidden)

        def route_and_prep():
            top_w, top_i = router_fused.route(routing, experts_out, c.top_k, index_dtype=torch.int32)
            flat = (top_i.reshape(-1), top_w.reshape(-1))
            return flat, moe_hip.prep(h, flat, c.top_k, self.expert_range)

        # `SEED_MOE_PREP_FORK`: routing and the work list run on a side stream while the shared
        # expert (which does not depend on routing) runs here; decode steps only (`out_buf is
        # not None`). See the flag's docstring for the fork/join correctness argument.
        fork = PREP_FORK and out_buf is not None and h.is_cuda
        if fork:
            cur = torch.cuda.current_stream(h.device)
            side = self._moe_side_stream(h.device)
            side.wait_stream(cur)
            with torch.cuda.stream(side):
                routed, prepared = route_and_prep()
        else:
            routed, prepared = route_and_prep()
        assignments = t * c.top_k
        scratch = self.moe_inter_scratch[i]
        inter = scratch[:assignments] if assignments <= scratch.shape[0] else None
        y_scratch = self.moe_dedup_y_scratch[i]
        y = y_scratch[:assignments] if assignments <= y_scratch.shape[0] else None
        out = None if out_buf is None else out_buf[:t]
        if step_shared:
            h_time = time_major(h)
            shared = torch.stack(
                [
                    swiglu_mlp(
                        h_time[step],
                        w["shared_expert.gate_up_proj"],
                        w["shared_expert.down_proj"],
                    )
                    for step in range(verify_width)
                ],
                dim=1,
            ).reshape(t, c.hidden)
        else:
            shared = swiglu_mlp(h, w["shared_expert.gate_up_proj"], w["shared_expert.down_proj"])
        decode_stamps.mark("dense_shared")
        gate = routing[:, experts_out:]
        if fork:
            cur.wait_stream(side)
        result = moe_hip.fused_moe_hip_glued(
            h,
            w["experts"],
            routed,
            c.top_k,
            self.expert_range,
            shared,
            gate,
            out=out,
            inter=inter,
            y=y,
            prepared=prepared,
        )
        decode_stamps.mark("moe_routed")
        return result

    def _moe_side_stream(self, device: torch.device) -> torch.cuda.Stream:
        """One side stream per device for `_moe_hip_route_fused`'s `SEED_MOE_PREP_FORK` fork."""
        streams = self.__dict__.setdefault("_moe_side_streams", {})
        if device not in streams:
            streams[device] = torch.cuda.Stream(device)
        return streams[device]

    def _routed_fused(
        self,
        i: int,
        h: torch.Tensor,
        top_i: torch.Tensor,
        top_w: torch.Tensor,
        out_buf: torch.Tensor | None,
    ) -> torch.Tensor:
        """Routed-expert output through the fused MXFP4 grouped-GEMV kernels.

        The routing handed to the kernel is the raw flattened `[t * top_k]` assignment list.
        There is deliberately no de-duplication and no filtering here: both were the source of
        the data-dependent shapes, and the kernel expresses the same sparsity as a per-program
        early return instead.

        `SEED_FUSED_MOE_ROUTING=1` (default) additionally computes `_gate_up_silu_kernel`'s
        `order` argument -- the permutation that groups assignments sharing an expert, so
        consecutive programs hit the same weight tile in L2 -- with one Triton kernel
        (`mxfp4_gemv.fused_expert_order`) instead of leaving it unset. `=0` restores the old
        behavior (`order=None`, programs run in arrival order) for A/B and rollback.

        `fused_expert_order` only applies at `t <= max_batch` (decode shapes). Its
        `ASSIGNMENTS` grid size is a `tl.constexpr`, so a prefill call (`t` varies chunk to
        chunk) would recompile the kernel per shape and run one program over up to
        `max_seq * top_k` assignments; decode's `t` is bounded by `max_batch` and stable
        across steps, so it stays a small, reused compilation. Prefill always takes
        `order=None`.

        `SEED_PREFILL_GROUPED_MOE=1` (off by default, unmeasured) takes precedence over
        everything below for calls of at least `prefill_moe.MIN_TOKENS` tokens (prefill
        shapes): an expert-grouped MXFP4 GEMM on MFMA with `BLOCK_M`-row blocks and a
        weighted, atomics-free combine; see `prefill_moe`'s module docstring. It skips
        zero-weight assignments like every path below, so it composes with hot experts.

        `SEED_MOE_HIP_WIDE=1` (with `SEED_MOE_HIP=1`) puts the HIP kernels ahead of
        `prefill_moe` for calls up to `moe_hip.MAX_TOKENS` (4096 under that flag): measured
        faster at every width, see `moe_hip.WIDE`.

        `SEED_MOE_HIP=1` (off by default, unmeasured) takes precedence over everything below for
        batches of at most `moe_hip.MAX_TOKENS` tokens: hand-written gfx942 kernels with an
        explicit ring of in-flight weight loads, see `moe_hip`'s module docstring.

        `SEED_MOE_BW=1` (off by default, unmeasured) takes precedence over both and routes
        through `mxfp4_gemv.fused_moe_bw`, the bandwidth-first design in `mxfp4_gemv`'s module
        docstring; `self.moe_bw` is false when its load-time scale precondition fails.
        `SEED_MOE_BW_VARIANT` selects among its refinements (`mxfp4_moe_bw2`).

        `SEED_MOE_DEDUP=1` (off by default as of integration build #2, a no-win pending
        GPU revalidation on the merged tree -- see `dedup_available`) instead routes through
        `mxfp4_gemv.fused_moe_dedup`: an `align_blocks` grid that reads each distinct local
        expert's weight tile once per (assignment-block, N-tile) instead of once per (token,
        expert) assignment; see `mxfp4_gemv`'s de-duplicating-MoE module docstring for the
        design and what still caps it short of the distinct-expert weight-reuse bound.
        `order`/`SEED_FUSED_MOE_ROUTING` do not apply to it: the L2-locality problem they solve
        for the per-assignment kernels does not exist here, because assignments are already
        grouped by expert via `align_blocks`.

        Composition with `SEED_HOT_EXPERTS` (the rule for every path above): hot experts run
        in bf16 (`hot_experts.hot_expert_forward`, in `moe`), cold experts go through whichever
        MXFP4 kernel the precedence above selects (hip, then bw/variant, then dedup, then
        per-assignment). `moe` zeroes a hot assignment's routing weight before calling here,
        and every one of these kernels drops zero-weight assignments: the bw/hip prep kernel's
        `live` mask and `bw_combine`, and the per-assignment/dedup kernels' `a_weight != 0`
        gate. So each assignment is computed exactly once, by exactly one of the two paths.
        """
        c, t = self.cfg, h.shape[0]
        assignments = t * c.top_k
        scratch = self.moe_inter_scratch[i]
        flat_expert = top_i.to(torch.int32).reshape(-1)
        flat_weight = top_w.reshape(-1)
        hip_wide = self.moe_hip and moe_hip.WIDE and t <= moe_hip.MAX_TOKENS
        if prefill_moe.available(h.device, t) and not hip_wide:
            return prefill_moe.prefill_moe(
                h,
                self.layers[i]["experts"],
                (flat_expert, flat_weight),
                c.top_k,
                self.expert_range,
                out=None if out_buf is None else out_buf[:t],
            )
        inter = scratch[:assignments] if assignments <= scratch.shape[0] else None
        out = None if out_buf is None else out_buf[:t]
        if self.moe_hip and t <= moe_hip.MAX_TOKENS:
            y_scratch = self.moe_dedup_y_scratch[i]
            return moe_hip.fused_moe_hip(
                h,
                self.layers[i]["experts"],
                (flat_expert, flat_weight),
                c.top_k,
                self.expert_range,
                out=out,
                inter=inter,
                y=y_scratch[:assignments] if assignments <= y_scratch.shape[0] else None,
            )
        if self.moe_bw:
            y_scratch = self.moe_dedup_y_scratch[i]
            return mxfp4_moe_bw2.fused_moe_variant(
                h,
                self.layers[i]["experts"],
                (flat_expert, flat_weight),
                c.top_k,
                self.expert_range,
                self.moe_bw_variant,
                out=out,
                inter=inter,
                y=y_scratch[:assignments] if assignments <= y_scratch.shape[0] else None,
            )
        if mxfp4_gemv.dedup_available(h.device):
            y_scratch = self.moe_dedup_y_scratch[i]
            return mxfp4_gemv.fused_moe_dedup(
                h,
                self.layers[i]["experts"],
                (flat_expert, flat_weight),
                c.top_k,
                self.expert_range,
                out=out,
                inter=inter,
                y=y_scratch[:assignments] if assignments <= y_scratch.shape[0] else None,
            )
        order = None
        if t <= self.max_batch and os.environ.get("SEED_FUSED_MOE_ROUTING", "1") != "0":
            order = mxfp4_gemv.fused_expert_order(
                flat_expert, c.experts, cursor=self.moe_route_cursor[i]
            )
        return mxfp4_gemv.fused_moe(
            h,
            self.layers[i]["experts"],
            (flat_expert, flat_weight),
            c.top_k,
            self.expert_range,
            out=out,
            inter=inter,
            order=order,
            # The whole expert axis, not this rank's slice of it: `c.experts` is what these
            # `assignments` are spread over, and it is what picks the grouped kernels over
            # the per-assignment ones. See `TP_BYTELUT_BOTTLENECK_2026-09-22.md` section 3.
            total_experts=c.experts,
        )

    def _routed_grouped(
        self,
        i: int,
        h: torch.Tensor,
        top_i: torch.Tensor,
        top_w: torch.Tensor,
        out_buf: torch.Tensor | None,
    ) -> torch.Tensor:
        """Routed-expert output with no Python-level loop over experts, in plain torch.

        Tokens are sorted by assigned expert so each activated expert's tokens land in one
        contiguous, padded group; the padded groups become the batch dimension of two `bmm`
        calls (gate_up projection, down projection) run once for every activated expert at
        once, instead of once per expert in a Python loop.

        The routing/gather buffers below (`tok_idx`, `weight_pad`, `mask`, and everything they
        feed) are sized by `num_active`/`max_count`, which depend on this call's actual routing
        and are not predictable ahead of time, so they always allocate fresh.

        The expert-parallel drop that `mxfp4_gemv` expresses as a GPU-side early return is
        host-side boolean indexing here. That is the difference the fused path exists to
        avoid, and it is acceptable on this one because every buffer below is already sized
        by a host read of the routing (`int(counts.max())`), so this path is not sync-free
        with or without it.
        """
        c, w = self.cfg, self.layers[i]
        t = h.shape[0]
        lo, hi = self.expert_range
        flat_expert = top_i.reshape(-1)
        flat_weight = top_w.reshape(-1)
        flat_token = torch.arange(t, device=h.device).repeat_interleave(c.top_k)

        mine = (flat_expert >= lo) & (flat_expert < hi)
        flat_expert = flat_expert[mine] - lo  # local expert index; `load_experts` loaded [lo, hi)
        flat_weight, flat_token = flat_weight[mine], flat_token[mine]
        if flat_expert.numel() == 0:  # no expert of this rank was selected this step
            return _zeroed_accumulator(h, out_buf)

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

        h_pad = h[tok_idx]  # [num_active, max_count, hidden], gathered via one vectorized index
        gate_up_w, down_w = self.expert_weights_batch(w["experts"], active)
        gate, up = torch.bmm(h_pad, gate_up_w.transpose(1, 2)).chunk(2, dim=-1)
        y = torch.bmm(F.silu(gate) * up, down_w.transpose(1, 2))

        out = _zeroed_accumulator(h, out_buf)
        out.index_add_(0, tok_idx[mask], (y[mask] * weight_pad[mask, None]).to(out.dtype))
        return out

    def layer(self, i: int, x: torch.Tensor, start: int) -> torch.Tensor:
        """One layer. `x` is replicated on entry and on exit.

        Both of the layer's collectives are here, per the convention in `tp.py`: the mixer
        and the MoE each return a row-parallel partial sum and this is the only place either
        is summed. Both mixers are sharded (attention over query heads, DeltaNet over value
        heads), so the reduction is unconditional; there is no mixer whose output is already
        whole, and reducing one that was would multiply it by the world size. Unsharded,
        `all_reduce` is the identity.
        """
        w, c = self.layers[i], self.cfg
        h = rmsnorm(x, w["in_norm"], c.eps)
        mixer = (
            self.full_attention(i, h, start)
            if c.layer_types[i] == "full_attention"
            else self.deltanet(i, h)
        )
        x = x + self.tp.all_reduce(mixer)
        return x + self.tp.all_reduce(self.moe(i, rmsnorm(x, w["post_norm"], c.eps)))

    def layer_packed(self, i: int, x: torch.Tensor, meta: PackedBatch) -> torch.Tensor:
        """One layer over a packed varlen prefill batch. `x` is [total_T, hidden] end to end.

        Same shape as `layer`'s per-request call, only cut differently: `rmsnorm` normalizes
        the last axis regardless of what the leading axis holds, and `moe` already flattens its
        input to `[-1, hidden]` (`Model.moe`), so neither needs to know several requests are
        concatenated here. Only the two mixers do (attention and DeltaNet each need their own
        sequence's own KV/recurrent state and must not attend or recur across a sequence
        boundary), which is what `full_attention_packed`/`deltanet_packed` handle.
        """
        w, c = self.layers[i], self.cfg
        h = rmsnorm(x, w["in_norm"], c.eps)
        mixer = (
            self.full_attention_packed(i, h, meta)
            if c.layer_types[i] == "full_attention"
            else self.deltanet_packed(i, h, meta)
        )
        x = x + self.tp.all_reduce(mixer)
        return x + self.tp.all_reduce(self.moe(i, rmsnorm(x, w["post_norm"], c.eps)))

    def layer_mixed(
        self,
        i: int,
        x: torch.Tensor,
        decode_lanes: list[int],
        decode_positions: list[int],
        meta: PackedBatch,
    ) -> torch.Tensor:
        """One layer over a *mixed* batch: `len(decode_lanes)` decode rows (this step's next
        token, one per lane) concatenated with `meta.total` packed prefill rows (one chunk,
        of one or more requests, each continuing at its own `start`). `x` is
        `[len(decode_lanes) + meta.total, hidden]`, decode rows first.

        The idea (Sarathi-Serve, Agrawal et al., OSDI'24, "stall-free batching"; vLLM's
        chunked-prefill scheduling is the same idea): rather than alternating whole iterations
        -- a whole prefill chunk, which stalls every running decode lane for that chunk's
        entire cost, then a whole decode step -- one iteration always carries every running
        decode lane's next token *and* admits only as much prefill work as an adaptive token
        budget allows (`scheduler._mixed_budget`), so the iteration's own added cost over a
        decode-only step stays bounded. See `scheduler.py`'s `SEED_MIXED_BATCH` docstring for
        the measured stall this replaces.

        Same split-before-mixer, concat-after-mixer, single-collective shape `layer_packed`
        already uses for several packed prefill sequences in one call: `rmsnorm` and `moe` are
        per-token and batch-shape-agnostic, so they run once over the whole concatenated `x`
        with the layer's usual two all-reduces (unconditional, same call count and order as
        `layer`/`layer_packed`/`decode_layer` -- every TP rank reaches them the same way
        regardless of how the batch happens to split), not one pair for the decode rows and a
        separate pair for the prefill rows. Only the two mixers need to know the rows are two
        kinds: decode rows go through `attn_decode`/`deltanet_decode` (batched, paged KV,
        per-lane recurrent state -- the graph-capturable decode kernels), prefill rows go
        through `full_attention_packed`/`deltanet_packed` (varlen, chunked, continuing each
        sequence's own `start`/initial DeltaNet state) exactly as an ordinary packed-prefill
        call already would. Concatenating the two mixers' outputs back in row order, then
        running the shared collective and `moe` over the concatenation, is exact for the same
        reason `layer_packed` looping over several sequences in one call is exact: neither
        mixer reads or writes another row's KV/state, and summing a partial sum over the whole
        row set equals summing it over each disjoint subset and concatenating the sums.

        Not (yet) graph-captured: `attn_decode`/`deltanet_decode` are the same kernels the
        decode-only captured graph already calls, but the *shapes* here change chunk to chunk
        (`meta.total` varies with the adaptive budget), so replaying one fixed graph is not a
        drop-in the way it is for the decode-only step (fixed `max_batch`, always 1 new token).
        A mixed step therefore runs eager. See `scheduler.py`'s module docstring for the
        measured eager-vs-graph gap this costs, and `Model.decode_mixed` for the shape-bucket
        capture option this leaves for later.
        """
        w, c = self.layers[i], self.cfg
        b = len(decode_lanes)
        h = rmsnorm(x, w["in_norm"], c.eps)
        h_decode, h_prefill = h[:b], h[b:]
        full = c.layer_types[i] == "full_attention"
        if full:
            m_decode = self.attn_decode(i, h_decode.unsqueeze(1), decode_lanes, decode_positions)
            m_decode = m_decode[:, 0]
            m_prefill = (
                self.full_attention_packed(i, h_prefill, meta) if meta.seqs else h_prefill
            )
        else:
            m_decode = self.deltanet_decode(i, h_decode.unsqueeze(1), decode_lanes)[:, 0]
            m_prefill = self.deltanet_packed(i, h_prefill, meta) if meta.seqs else h_prefill
        mixer = torch.cat([m_decode, m_prefill], dim=0)
        x = x + self.tp.all_reduce(mixer)
        moe_out = self.moe(i, rmsnorm(x, w["post_norm"], c.eps))
        return x + self.tp.all_reduce(moe_out)

    @torch.no_grad()
    def forward_mixed(
        self,
        decode_lanes: list[int],
        decode_tokens: list[int],
        decode_positions: list[int],
        prefill_ids: torch.Tensor,
        meta: PackedBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One forward: `len(decode_lanes)` decode rows (1 token each) concatenated with
        `meta.total` packed prefill rows. Returns `(decode_logits [B, vocab], prefill_logits
        [len(meta.seqs), vocab])` -- the same per-row contracts `decode`/`forward_packed` already
        have (each prefill sequence's last row only), computed in the same pass. `lm_head` needs
        no split: it is a per-row projection, run once over the narrowed rows like any dense op
        (see `layer_mixed`).

        Not pipelined, unlike `decode`: `forward_packed` (ordinary packed prefill) is not
        pipelined either, for the same reason -- the pipeline's microbatch cuts assume a fixed
        per-slot shape (`_microbatches`), which a variable-length prefill chunk does not have.
        """
        b = len(decode_lanes)
        dec_ids = torch.tensor(decode_tokens, device=self.devices[0])
        ids = torch.cat([dec_ids, prefill_ids.to(self.devices[0])])
        x = F.embedding(ids, self.embed)
        for i in range(len(self.layers)):
            x = self.layer_mixed(i, x.to(self.layer_dev[i]), decode_lanes, decode_positions, meta)
        x = x.to(self.devices[-1])
        # Narrow before `lm_head`, as `forward_packed` does: every decode row, then only each
        # prefill sequence's last row (a `[meta.total, vocab]` transient is GiBs at real vocab).
        x = x[list(range(b)) + [b + hi - 1 for _, hi in meta.spans()]]
        logits = self.unembed(rmsnorm(x, self.final_norm, self.cfg.eps)).float()
        return logits[:b], logits[b:]

    @torch.no_grad()
    def decode_mixed(
        self,
        decode_lanes: Sequence[int],
        decode_tokens: Sequence[int],
        decode_positions: Sequence[int],
        prefill_calls: Sequence[tuple[int, Sequence[int], int]],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """`SEED_MIXED_BATCH`'s one-forward step: every lane in `decode_lanes` advances its own
        next token (`decode`'s own contract: one row per lane, in order) in the *same* forward
        call as one packed prefill chunk (`prefill_calls`, `prefill_batch`'s own `(slot, ids,
        start)` contract). Returns `(decode_logits, prefill_logits)`: `decode_logits` indexable
        exactly like `decode`'s own result (`decode_row`), `prefill_logits` one `[1, vocab]` row
        per `prefill_calls` entry, in order, like `prefill_batch`'s own result -- so
        `scheduler._mixed_step` applies each half exactly as `_decode_step`/
        `_finish_prefill_chunk` already do, unchanged.

        Falls back to a plain `decode` call, with no prefill side at all, when `prefill_calls`
        is empty (an ordinary decode-only step, e.g. nothing left in `prefill_q` this
        iteration) -- callers need not special-case that.

        MTP (`SEED_MTP`) is out of scope for v1: `speculative_decode`'s draft/verify/accept
        round has its own wide-attention (`attn_verify`) and rollback-capable DeltaNet
        (`deltanet_verify`/`deltanet_verify_rollback`) machinery that a mixed forward does not
        (yet) share, and this call does not seed `mtp.draft`'s per-slot hidden-state cache
        (`cache_hidden`) the way `forward`/`forward_packed` do. `scheduler.Scheduler` never
        calls this while `spec_decode` is on (see its own `mixed_batch` construction); this
        assertion is the model-side backstop for a caller that bypasses the scheduler.
        """
        if self.mtp is not None:
            raise ValueError("decode_mixed: SEED_MTP and SEED_MIXED_BATCH cannot combine (v1)")
        if not prefill_calls:
            return self.decode(list(decode_lanes), list(decode_tokens), list(decode_positions)), []
        self._decode_epoch += 1
        b = len(decode_lanes)
        seqs = tuple(PackedSeq(lane, start, len(ids)) for lane, ids, start in prefill_calls)
        meta = PackedBatch(seqs)
        prefill_ids = torch.tensor([tok for _, ids, _ in prefill_calls for tok in ids])
        decode_logits, prefill_logits = self.forward_mixed(
            list(decode_lanes), list(decode_tokens), list(decode_positions), prefill_ids, meta
        )
        assert decode_logits.shape[0] == b
        return decode_logits, [prefill_logits[k : k + 1] for k in range(len(seqs))]

    # -- batched decode -------------------------------------------------------
    def _paged_read_buffers(
        self, dev: torch.device, slots: list[int], positions: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """This step's `[B, max_blocks]` block table and `[B]` valid-block-count.

        Eager decode is not the captured path, so these are ordinary tensors, not persistent
        buffers a replay mutates in place -- `graph_decode.Buffers` is where the persistent
        versions of these two live. Built once per decode step and reused by every
        full-attention layer of that step: every layer's `grow_to` after the first is a no-op,
        so the tables are identical across layers. Rebuilding per layer cost a `[B,
        max_blocks]` Python list (48 x 1024 ints, ~4 ms of host time) and a synchronizing
        host-to-device copy per layer, 15 of each per step on the real model. Keyed on
        `_decode_epoch` as well as the batch, since a `reset` between steps can hand a lane
        different blocks at the same position.
        """
        key = (self._decode_epoch, dev, tuple(slots), tuple(positions))
        memo = self._paged_read_memo
        if memo is not None and memo[0] == key:
            return memo[1], memo[2]
        rows = [self.block_tables[slot].padded_row(self.max_blocks_per_lane) for slot in slots]
        valid = [block_pool.valid_block_count(p + 1, self.block_size) for p in positions]
        block_table = host_to_device(rows, dev, torch.int32)
        block_valid = host_to_device(valid, dev, torch.int32)
        self._paged_read_memo = (key, block_table, block_valid)
        return block_table, block_valid

    def attn_decode(
        self, i: int, x: torch.Tensor, slots: list[int], positions: list[int]
    ) -> torch.Tensor:
        """Full attention for one token per slot. x is [B, 1, hidden]. A partial sum.

        The decode counterpart of `full_attention`, and sharded the same way: this rank's
        `pl.q.count` query heads against its `pl.kv.count` KV heads, with `o_proj` row-parallel
        so `decode_layer` reduces the result. `decode_attention_paged` reads both head counts
        off `q` and the pool, so its group fold is `[B, 1, 8, d]` at TP=4 where it is
        `[B, 2, 16, d]` unsharded, exact for the same reason as always: `tp.kv_shard` keeps a
        rank's query block inside one KV group.

        Paged-KV (Stage 1): each active slot's new token is written at the physical row its
        own block table maps `positions[j]` to (growing the table first, exactly as
        `full_attention` does for a prefill chunk), and the read is a block-table lookup
        through `decode_attention_paged`/`paged_attn.py` instead of a contiguous window slice
        of a per-slot dense pool -- there is no such window once storage is shared and paged
        (see `paged_attn.py`'s module docstring for why gathering every step instead is a
        bandwidth non-starter at long context).
        """
        c, w, pool = self.cfg, self.layers[i], self.pool[i]
        b, pl = x.shape[0], self.tp.plan
        nq, nkv = pl.q.count, pl.kv.count
        q, gate = skinny_hip.linear(x, w["q_proj"]).view(b, 1, nq, 2 * c.head_dim).chunk(2, dim=-1)
        q = rmsnorm(q, w["q_norm"], c.eps).transpose(1, 2)
        k = rmsnorm(
            skinny_hip.linear(x, w["k_proj"]).view(b, 1, nkv, c.head_dim), w["k_norm"], c.eps
        ).transpose(1, 2)
        v = skinny_hip.linear(x, w["v_proj"]).view(b, 1, nkv, c.head_dim).transpose(1, 2)
        pos = self._host_index("positions", x.device, tuple(positions))
        cos, sin = self.rope_at(pos, x.dtype)
        q = apply_rope(q, cos[:, None, None], sin[:, None, None])
        k = apply_rope(k, cos[:, None, None], sin[:, None, None])

        for slot, p in zip(slots, positions, strict=True):
            self.grow_lane(slot, p + 1)
        write_rows = self._host_index(
            "decode_write_rows",
            x.device,
            tuple(
                self.block_tables[slot].physical_row(p, self.block_size)
                for slot, p in zip(slots, positions, strict=True)
            ),
        )
        pool["k"][write_rows] = k[:, :, 0]
        pool["v"][write_rows] = v[:, :, 0]

        timed = self._timing_sample and x.device.type == "cuda"
        if timed:
            t0 = time.perf_counter()
        block_table, block_valid = self._paged_read_buffers(x.device, slots, positions)
        if timed:
            self._attn_host_ms += (time.perf_counter() - t0) * 1e3
            ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            ev[0].record()
        out = decode_attention_paged(
            q,
            pool["k"],
            pool["v"],
            block_table,
            block_valid,
            pos,
            self.block_size,
            c.head_dim**-0.5,
        )
        if timed:
            ev[1].record()
            self._attn_events.append(ev)
        out = out.transpose(1, 2).reshape(b, 1, -1)
        return skinny_hip.linear(out * torch.sigmoid(gate.reshape(b, 1, -1)), w["o_proj"])

    def attn_verify(
        self, i: int, x: torch.Tensor, slots: list[int], base_positions: list[int], t: int
    ) -> torch.Tensor:
        """Wide verify-step attention: `t` new tokens per slot (MTP's `mtp.k + 1`) in one
        dispatch, instead of `t` calls to `attn_decode`. `x` is `[B, t, hidden]`: this round's
        fed tokens (the last committed token, then the `t - 1` draft tokens), embedded and
        normed, in causal order per slot.

        Exactly `attn_decode`, generalized from one new row per slot to `t`: project, RoPE at
        each row's own absolute position, grow every slot's block table once to cover all `t`
        new rows (`grow_to` is idempotent past what it already holds), write all `t` rows' K/V,
        then one `verify_attention_paged` call for the whole batch. Only `mtp.verify_and_commit`
        calls this -- `decode_layer`/`layer` still call `attn_decode` for ordinary decode.
        """
        c, w, pool = self.cfg, self.layers[i], self.pool[i]
        b, pl = x.shape[0], self.tp.plan
        nq, nkv = pl.q.count, pl.kv.count
        q, gate = F.linear(x, w["q_proj"]).view(b, t, nq, 2 * c.head_dim).chunk(2, dim=-1)
        q = rmsnorm(q, w["q_norm"], c.eps).transpose(1, 2)
        k = rmsnorm(
            F.linear(x, w["k_proj"]).view(b, t, nkv, c.head_dim), w["k_norm"], c.eps
        ).transpose(1, 2)
        v = F.linear(x, w["v_proj"]).view(b, t, nkv, c.head_dim).transpose(1, 2)

        base = self._host_index("verify_base", x.device, tuple(base_positions))
        pos_grid = base[:, None] + torch.arange(t, device=x.device)[None, :]  # [B, T]
        cos, sin = self.rope_at(pos_grid.reshape(-1), x.dtype)
        cos, sin = cos.reshape(b, t, -1), sin.reshape(b, t, -1)
        q = apply_rope(q, cos[:, None], sin[:, None])
        k = apply_rope(k, cos[:, None], sin[:, None])

        for slot, p in zip(slots, base_positions, strict=True):
            self.grow_lane(slot, p + t)
        write_rows = self._host_index(
            "verify_write_rows",
            x.device,
            tuple(
                self.block_tables[slot].physical_row(p + j, self.block_size)
                for slot, p in zip(slots, base_positions, strict=True)
                for j in range(t)
            ),
        )
        pool["k"][write_rows] = k.transpose(1, 2).reshape(-1, nkv, c.head_dim)
        pool["v"][write_rows] = v.transpose(1, 2).reshape(-1, nkv, c.head_dim)

        end_positions = [p + t - 1 for p in base_positions]
        block_table, block_valid = self._paged_read_buffers(x.device, slots, end_positions)
        out = verify_attention_paged(
            q,
            pool["k"],
            pool["v"],
            block_table,
            block_valid,
            base,
            self.block_size,
            c.head_dim**-0.5,
        )
        out = out.transpose(1, 2).reshape(b, t, -1)
        return F.linear(out * torch.sigmoid(gate.reshape(b, t, -1)), w["o_proj"])

    def _host_index(self, role: str, dev: torch.device, values: tuple[int, ...]) -> torch.Tensor:
        """`torch.tensor(values, device=dev)` memoized on its own value, per role and device.

        A decode step's slot list and position list are the same for all 60 layers, so the
        plain spelling costs 60 host-to-device copies per step where one would do. The memo
        holds one entry per `(role, device)` and is keyed on the values themselves, so a
        microbatch with a different slot list simply rebuilds rather than reading a stale
        tensor. The returned tensor is only ever read (it indexes pools), never written.
        """
        cache = self.index_cache.setdefault(role, {})
        hit = cache.get(dev)
        if hit is not None and hit[0] == values:
            return hit[1]
        built = host_to_device(values, dev)
        cache[dev] = (values, built)
        return built

    def deltanet_decode(self, i: int, x: torch.Tensor, slots: list[int]) -> torch.Tensor:
        """DeltaNet for one token per slot, batched across every active slot in one call.

        `delta_rule_recurrent`'s math (and `causal_conv`'s) is already batch-generic: the
        leading dim of every tensor is just "batch", not hardcoded to 1. So this gathers the
        conv and recurrent state for exactly the given `slots` out of the layer's pool (one
        row per active slot), runs the same per-token recurrence as `deltanet` with that real
        batch dimension instead of a Python loop calling it once per slot, and scatters the
        updated state back into just those pool rows. A slot not in `slots` is never read or
        indexed into, so it cannot be corrupted by, or corrupt, this call; that mirrors how
        `attn_decode` reads and writes only `rows`-indexed positions of the KV pool.

        Indexing the pool directly is also what makes this safe to run from the stage threads
        of a pipelined step: nothing here reads `self.state` or moves the `bind` cursor, which
        is shared mutable state that several microbatches in flight would race on. It passes no
        `out` buffer to `delta_rule` either, so it shares no scratch between stages.

        Under tensor parallelism the pool rows hold only this rank's value heads and
        `out_proj` is column-sharded, so the result is a row-parallel partial sum like
        `deltanet`'s, which `decode_layer` reduces once for the whole batch.

        The four input projections are one `F.linear` against `in_proj_all` (see
        `fuse_in_proj`), and at one token per slot the recurrence and the output norm are one
        Triton kernel each (see `deltanet_fused`). Both are dispatch-count changes: this mixer
        measured 64.12 ms of a 188 ms step across ~1,575 `aten` calls, on a step that spends
        35% of its time with the device idle waiting for the host to issue the next op.
        """
        c, w, pool = self.cfg, self.layers[i], self.pool[i]
        b, t = x.shape[0], x.shape[1]
        key_dim, val_dim = c.k_heads * c.k_dim, c.v_heads * c.v_dim
        rows = self._host_index("slots", x.device, tuple(slots))

        projected = skinny_hip.linear(x, w["in_proj_all"])
        qkv, z, beta_raw, a_raw = projected.split(in_proj_sizes(c), dim=-1)
        if t == 1 and DN_CONV_INPLACE and deltanet_fused.available(x.device):
            # Same kernel the captured step runs (`graph_decode.deltanet_decode_static`), so
            # `GraphDecodeRunner.validate`'s eager-vs-replay check compares like with like;
            # every row here is a real slot, hence `active` all true.
            active = torch.ones(b, dtype=torch.bool, device=x.device)
            flat = deltanet_fused.causal_conv_decode(qkv, w["conv"], pool["conv"], rows, active)
            mixed = flat[:, :, None]
        else:
            conv_state = pool["conv"][rows]  # fancy indexing copies: safe to mutate in place
            mixed = self.causal_conv(qkv.transpose(1, 2), w["conv"], conv_state)
            pool["conv"][rows] = conv_state
        rec_state = pool["rec"][rows]  # same copy-then-scatter-back pattern as conv_state

        if t == 1 and deltanet_fused.available(x.device):
            rec_out = self._delta_rule_fused(c, w, mixed, (beta_raw, a_raw), rec_state)
            pool["rec"][rows] = rec_state
            gate = z.unflatten(-1, (c.v_heads, c.v_dim))[:, 0]
            norm_out = deltanet_fused.gated_rmsnorm(
                rec_out if DN_NORM_F32_IN else rec_out.to(x.dtype),
                gate,
                w["dn_norm"],
                c.eps,
                x.dtype,
            )
            mixer = skinny_hip.linear(norm_out.reshape(b, t, -1), w["out_proj"])
            if MTP_VERIFY_TRACE_DN:
                self._mtp_verify_decode_trace = {
                    "layer": i,
                    "projected": projected,
                    "raw_qkv": qkv.transpose(1, 2),
                    "mixed": mixed,
                    "a_raw": a_raw,
                    "beta_raw": beta_raw,
                    "rec_out": rec_out,
                    "norm_out": norm_out,
                    "mixer": mixer,
                }
            return mixer

        q, k, v = mixed.transpose(1, 2).split([key_dim, key_dim, val_dim], dim=-1)
        q, k = q.reshape(b, t, c.k_heads, c.k_dim), k.reshape(b, t, c.k_heads, c.k_dim)
        v = v.reshape(b, t, c.v_heads, c.v_dim)
        beta = beta_raw.sigmoid()
        g = -w["A_log"].exp() * F.softplus(a_raw.float() + w["dt_bias"])
        rep = c.v_heads // c.k_heads
        q, k = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)

        out = self.delta_rule(q, k, v, g, beta, rec_state).to(x.dtype)
        pool["rec"][rows] = rec_state

        out = gated_rmsnorm(out.reshape(-1, c.v_dim), z.reshape(-1, c.v_dim), w["dn_norm"], c.eps)
        return skinny_hip.linear(out.reshape(b, t, -1), w["out_proj"])

    @staticmethod
    def _delta_rule_fused(
        c: Cfg, w: dict, mixed: torch.Tensor, gate: tuple[torch.Tensor, torch.Tensor], rec
    ) -> torch.Tensor:
        """`deltanet_fused.delta_rule_decode` off the T=1 conv output. Advances `rec` in place.

        `mixed` is `[B, 2 * key_dim + val_dim, 1]`, so dropping its length axis gives a `[B, C]`
        tensor whose three segments unflatten into `[B, heads, dim]` views with no copy. The
        gate pair is `in_proj_b` and `in_proj_a` straight off the fused projection; the kernel
        applies their activations itself.
        """
        flat = mixed[:, :, 0]
        key_dim = c.k_heads * c.k_dim
        q = flat[:, :key_dim].unflatten(-1, (c.k_heads, c.k_dim))
        k = flat[:, key_dim : 2 * key_dim].unflatten(-1, (c.k_heads, c.k_dim))
        v = flat[:, 2 * key_dim :].unflatten(-1, (c.v_heads, c.v_dim))
        beta_raw, a_raw = gate
        return deltanet_fused.delta_rule_decode(
            (q, k),
            v,
            (a_raw[:, 0], beta_raw[:, 0]),
            rec,
            (w["A_log"], w["dt_bias"]),
        )

    def decode_layer(
        self, i: int, x: torch.Tensor, slots: list[int], positions: list[int]
    ) -> torch.Tensor:
        """One decode layer. The two all-reduces of `layer`, once for the whole batch.

        Batching the slots is what makes that possible: one collective per layer per step
        rather than `len(slots)` of them, which is the same win the batched attention and
        DeltaNet calls take on the kernel side.

        The first all-reduce goes through `TP.all_reduce_residual_norm` (`SEED_FUSED_AR_NORM`),
        which folds `x = x + all_reduce(mixer)` and the `post_norm` that immediately follows
        it into one kernel on the fused path -- see that method's docstring. The second is left
        as a plain `all_reduce` here: this call's caller (`_decode_stage`, driven per layer
        range from a pipeline stage) does not know the next layer's `in_norm` weight, so fusing
        it with *its* following norm needs a call site that owns the layer loop, which is what
        `graph_decode.segment_step` does for the captured path this model actually ships
        decode through under TP. `decode_layer` still gets one of the two fusions the graph
        path gets both of.
        """
        w, c = self.layers[i], self.cfg
        h = rmsnorm(x, w["in_norm"], c.eps)
        mixer = (
            self.attn_decode(i, h, slots, positions)
            if c.layer_types[i] == "full_attention"
            else self.deltanet_decode(i, h, slots)
        )
        x, h_mid = self.tp.all_reduce_residual_norm(mixer, x, w["post_norm"], c.eps, rmsnorm)
        moe_out = self.moe(i, h_mid, self.moe_scratch[i][: len(slots)])
        return x + self.tp.all_reduce(moe_out)

    def deltanet_verify(
        self, i: int, x: torch.Tensor, slots: list[int], t: int
    ) -> tuple[torch.Tensor, dict]:
        """Wide DeltaNet verify: `t` new tokens per slot (MTP's `mtp.k + 1`), one dispatch
        across every slot via `deltanet_fused.fused_recurrent_prefill`'s `cu_seqlens` packing
        (`delta_rule_recurrent` off-GPU, already batch- *and* length-generic). Mutates
        `pool[i]["conv"]`/`["rec"]` to the state after all `t` fed tokens, which is only
        correct for a slot that accepts every draft -- speculative, like `attn_verify`'s KV
        writes, but unlike KV this state cannot simply be left unread past the real accept
        length (the recurrence is path-dependent), so it needs correcting.

        Returns `(mixer_out, snapshot)`. `snapshot` is what `deltanet_verify_rollback` needs to
        correct that state once the caller knows each slot's real accept length: the
        pre-verify `conv`/`rec` state and the raw per-token conv input and mixer inputs
        (activations, not `t` copies of the state itself -- see the design doc's cost
        comparison against per-step state snapshots).
        """
        c, w, pool = self.cfg, self.layers[i], self.pool[i]
        b = x.shape[0]
        key_dim, val_dim = c.k_heads * c.k_dim, c.v_heads * c.v_dim
        rows = self._host_index("slots", x.device, tuple(slots))

        linear = skinny_hip.linear if MTP_VERIFY_SKINNY_PROJ else F.linear
        if MTP_VERIFY_STEP_ROWS:
            projected = torch.stack(
                [skinny_hip.linear(x[:, step], w["in_proj_all"]) for step in range(t)],
                dim=1,
            )
        else:
            projected = linear(x, w["in_proj_all"])
        qkv, z, beta_raw, a_raw = projected.split(in_proj_sizes(c), dim=-1)
        raw_qkv = qkv.transpose(1, 2)  # [B, conv_dim, t], pre-conv: what rollback replays
        conv_state = pool["conv"][rows].clone()  # copy-then-scatter-back, like deltanet_decode
        pre_conv = conv_state.clone()  # causal_conv mutates conv_state in place below
        if MTP_VERIFY_DN_EXACT and deltanet_fused.available(x.device):
            # MIOpen's wide convolution reduces taps differently from the ordinary decode
            # kernel. Preserve decode's per-token tap order so later recurrent layers see the
            # same q/k/v values as sequential target decoding.
            mixed = deltanet_fused.causal_conv_verify_exact(
                qkv, w["conv"], conv_state
            ).transpose(1, 2)
        else:
            mixed = self.causal_conv(raw_qkv, w["conv"], conv_state)
        pool["conv"][rows] = conv_state

        q, k, v = mixed.transpose(1, 2).split([key_dim, key_dim, val_dim], dim=-1)
        q_small = q.reshape(b, t, c.k_heads, c.k_dim)
        k_small = k.reshape(b, t, c.k_heads, c.k_dim)
        # .contiguous(): v (unlike q/k, which repeat_interleave below makes contiguous) stays a
        # transpose-then-split view of causal_conv's [B, C, t] output, whose last axis has
        # stride t, not 1 -- fused_recurrent_prefill requires stride 1 there (see delta_rule's
        # single-sequence branch, which has the same bug for the same reason).
        v = v.reshape(b, t, c.v_heads, c.v_dim).contiguous()
        beta = beta_raw.sigmoid()
        g = -w["A_log"].exp() * F.softplus(a_raw.float() + w["dt_bias"])
        rep = c.v_heads // c.k_heads

        rec_state = pool["rec"][rows]  # copy-then-scatter-back, like deltanet_decode
        pre_rec = rec_state.clone()
        heads = c.v_heads
        exact = MTP_VERIFY_DN_EXACT and deltanet_fused.available(x.device)
        sequential = MTP_VERIFY_DN_SEQUENTIAL and deltanet_fused.available(x.device)
        if exact:
            q_small, k_small = q_small.contiguous(), k_small.contiguous()
            out = deltanet_fused.delta_rule_verify_exact(
                (q_small, k_small),
                v,
                (a_raw, beta_raw),
                rec_state,
                (w["A_log"], w["dt_bias"]),
            ).to(x.dtype)
        elif sequential:
            q_small, k_small = q_small.contiguous(), k_small.contiguous()
            out = torch.stack(
                [
                    deltanet_fused.delta_rule_decode(
                        (q_small[:, s], k_small[:, s]),
                        v[:, s],
                        (a_raw[:, s], beta_raw[:, s]),
                        rec_state,
                        (w["A_log"], w["dt_bias"]),
                    )
                    for s in range(t)
                ],
                dim=1,
            ).to(x.dtype)
        else:
            q = q_small.repeat_interleave(rep, dim=2)
            k = k_small.repeat_interleave(rep, dim=2)
        if not exact and not sequential and t > 1 and deltanet_fused.available_prefill(x.device):
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
        elif not exact and not sequential:
            out = delta_rule_recurrent(q, k, v, g, beta, rec_state).to(x.dtype)
        q = q_small.repeat_interleave(rep, dim=2)
        k = k_small.repeat_interleave(rep, dim=2)
        pool["rec"][rows] = rec_state

        rec_out = out
        if MTP_VERIFY_STEP_ROWS and deltanet_fused.available(x.device):
            norm_out = torch.stack(
                [
                    deltanet_fused.gated_rmsnorm(
                        rec_out[:, step],
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
                rec_out.reshape(-1, c.v_dim),
                z.reshape(-1, c.v_dim),
                w["dn_norm"],
                c.eps,
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
            mixer = linear(norm_out.reshape(b, t, -1), w["out_proj"])
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
        if MTP_VERIFY_TRACE_DN:
            snapshot["trace"] = {
                "projected": projected,
                "raw_qkv": raw_qkv,
                "mixed": mixed,
                "q": q,
                "k": k,
                "v": v,
                "a_raw": a_raw,
                "beta_raw": beta_raw,
                "rec_out": rec_out,
                "norm_out": norm_out,
                "mixer": mixer,
            }
        return mixer, snapshot

    def deltanet_verify_rollback(
        self, i: int, slots: list[int], snapshot: dict, accept_len_plus1: list[int]
    ) -> None:
        """Corrects layer `i`'s `conv`/`rec` state, written speculatively for all `t` fed
        tokens by `deltanet_verify`, down to each slot's real accept length
        (`accept_len_plus1[j]` tokens, `1..t`).

        Conv state is a fixed-width window of raw (pre-conv) inputs, so it is exact by
        slicing `deltanet_verify`'s own `raw_qkv` at the right length -- no recompute, no
        kernel call. Recurrent state has no such shortcut (it depends on the whole path, not a
        window), so it is replayed from the pre-verify snapshot over just the accepted-length
        prefix, packed into one `cu_seqlens` batch across every slot at once (design doc option
        (b): one extra wide dispatch, not a kernel change and not `t` per-step snapshots).
        """
        c, pool = self.cfg, self.pool[i]
        rows = self._host_index("slots", pool["conv"].device, tuple(slots))
        b, t = snapshot["q"].shape[0], snapshot["q"].shape[1]
        conv_k1 = pool["conv"].shape[-1]
        dev = pool["conv"].device

        full = torch.cat([snapshot["pre_conv"], snapshot["raw_qkv"]], dim=-1)
        starts = torch.tensor(accept_len_plus1, device=dev)
        win = torch.arange(conv_k1, device=dev)
        idx = (starts[:, None] + win[None, :]).clamp(max=full.shape[-1] - 1)
        idx = idx[:, None, :].expand(-1, full.shape[1], -1)
        pool["conv"][rows] = torch.gather(full, -1, idx)

        heads = c.v_heads
        q, k, v, g, beta = (snapshot[n] for n in ("q", "k", "v", "g", "beta"))
        rec_state = snapshot["pre_rec"].clone()
        if MTP_VERIFY_DN_EXACT and deltanet_fused.available(q.device):
            lengths = torch.tensor(accept_len_plus1, device=q.device, dtype=torch.int32)
            deltanet_fused.delta_rule_verify_exact(
                (snapshot["q_small"], snapshot["k_small"]),
                v,
                (snapshot["a_raw"], snapshot["beta_raw"]),
                rec_state,
                (self.layers[i]["A_log"], self.layers[i]["dt_bias"]),
                lengths=lengths,
            )
        elif t > 1 and deltanet_fused.available_prefill(q.device):
            mask = torch.arange(t, device=q.device)[None, :] < starts[:, None]
            flat_mask = mask.reshape(-1)
            cu = torch.zeros(b + 1, dtype=torch.int32, device=q.device)
            cu[1:] = torch.tensor(accept_len_plus1, device=q.device, dtype=torch.int32).cumsum(0)
            deltanet_fused.fused_recurrent_prefill(
                q.reshape(b * t, heads, c.k_dim)[flat_mask],
                k.reshape(b * t, heads, c.k_dim)[flat_mask],
                v.reshape(b * t, heads, c.v_dim)[flat_mask],
                g.reshape(b * t, heads).float()[flat_mask],
                beta.reshape(b * t, heads)[flat_mask],
                rec_state,
                cu,
            )
        else:
            for j in range(b):
                n = accept_len_plus1[j]
                delta_rule_recurrent(
                    q[j : j + 1, :n],
                    k[j : j + 1, :n],
                    v[j : j + 1, :n],
                    g[j : j + 1, :n],
                    beta[j : j + 1, :n],
                    rec_state[j : j + 1],
                )
        pool["rec"][rows] = rec_state

    def decode_layer_verify(
        self, i: int, x: torch.Tensor, slots: list[int], base_positions: list[int], t: int
    ) -> tuple[torch.Tensor, dict | None]:
        """`decode_layer`, generalized to `t` new tokens per slot in one wide dispatch (MTP's
        `verify_and_commit`): one `attn_verify`/`deltanet_verify` dispatch and one `moe()` call
        over all `B * t` tokens, instead of `t` sequential `decode_layer` calls.

        Returns `(x, snapshot)`: `snapshot` is `None` for a full-attention layer (its paged KV
        needs no rollback -- an unaccepted row is simply never read again, see
        `attn_verify`'s docstring) and, for a DeltaNet layer, what `deltanet_verify_rollback`
        needs once the caller knows every slot's real accept length.
        """
        w, c = self.layers[i], self.cfg
        h = rmsnorm(x, w["in_norm"], c.eps)
        if c.layer_types[i] == "full_attention":
            mixer, snapshot = self.attn_verify(i, h, slots, base_positions, t), None
        else:
            mixer, snapshot = self.deltanet_verify(i, h, slots, t)
        x = x + self.tp.all_reduce(mixer)
        h_mid = rmsnorm(x, w["post_norm"], c.eps)
        moe_out = mtp_verify_moe(self, i, h_mid)
        return x + self.tp.all_reduce(moe_out), snapshot

    @torch.no_grad()
    def _decode_stage(self, layers: range, micro: _Micro) -> _Micro:
        """Run one device's layer run over one microbatch. The only cross-device copy is the
        `.to` below: a stage hands the next stage an activation, never a collective."""
        x = micro.x.to(self.layer_dev[layers.start])
        for i in layers:
            x = self.decode_layer(i, x, micro.slots, micro.positions)
        return _Micro(x, micro.slots, micro.positions)

    def _microbatches(
        self, slots: list[int], positions: list[int], x: torch.Tensor
    ) -> list[_Micro]:
        """Cut a decode step into the microbatches the pipeline staggers across the stages.

        `self.microbatches` is the pipeline-depth ceiling (see its assignment in `__init__`);
        `_decode_group_count` narrows it toward fewer, larger microbatches when this step does
        not have `MIN_MICROBATCH_SLOTS` per microbatch to spare at that ceiling.
        """
        groups = _decode_group_count(len(slots), self.microbatches, MIN_MICROBATCH_SLOTS)
        return [
            _Micro(x[lo:hi], list(slots[lo:hi]), list(positions[lo:hi]))
            for lo, hi in _microbatch_cuts(len(slots), groups)
        ]

    @torch.no_grad()
    def decode(self, slots: list[int], tokens: list[int], positions: list[int]) -> torch.Tensor:
        """One decode step for several slots at once. Returns fp32 logits [B, vocab].

        Pipelined: the step is cut into microbatches and each stage thread pulls the next one
        as soon as it is free, so stage s runs microbatch m while stage s-1 runs m+1. The
        arithmetic a microbatch sees does not depend on the schedule. Microbatches hold
        disjoint slots, so their KV rows and DeltaNet states never overlap; a layer's pool
        belongs to exactly one stage, so exactly one thread touches it; and the weights are
        read only. Running the same microbatches one after another gives the same tensors,
        bit for bit (test_pipeline.py asserts it with `torch.equal`).

        The cut itself is not bit-preserving against one undivided batch, for the one reason
        `moe` is not batch-shape-invariant: it pads each activated expert's tokens to the
        call's own largest group, so a narrower call groups the same products into differently
        shaped `bmm`s. Only float32 reassociation moves (1.4e-5 on logits of magnitude 8 in
        the CPU test), the same effect batching itself already has; see
        test_the_microbatch_cut_only_moves_the_logits_by_float_reassociation.
        """
        if self.gap_meter is not None:
            self.gap_meter.step_start()
        logits = self._decode_ids(
            slots, host_to_device([[t] for t in tokens], self.devices[0]), positions
        )
        if self.gap_meter is not None:
            self.gap_meter.step_end()
        return logits

    def _decode_ids(
        self, slots: list[int], ids: torch.Tensor, positions: list[int]
    ) -> torch.Tensor:
        """`decode`'s body, with the token ids already on the device as [B, 1]: the shared
        tail of `decode` (ids from the host) and `decode_launch` (ids resolved on the device,
        see `resolve_tokens`)."""
        self._decode_epoch += 1
        if step_timing.ENABLED:
            self._timing_decodes += 1
            self._timing_sample = self._timing_decodes % step_timing.EVERY == 0
            t0 = time.perf_counter()
        mtp_previous = None
        if self.mtp is not None:
            mtp_previous = self.hidden_scratch[torch.tensor(slots, device=self.devices[-1])].clone()
        micros = self._microbatches(slots, positions, F.embedding(ids, self.embed))
        if len(micros) > 1 and self.pipeline is not None:
            micros = self.pipeline.run(micros)
        else:
            for layers in self.stages:
                micros = [self._decode_stage(layers, m) for m in micros]
        x = torch.cat([m.x.to(self.devices[-1]) for m in micros], dim=0)
        if self.mtp is not None:
            # A later step may return to MTP after the whole batch temporarily fell back to
            # ordinary decode (for example, while one sampled request shared the batch).
            # Keep every lane's draft seed aligned with the token this call just consumed.
            pos = torch.tensor(positions, dtype=torch.long, device=x.device)
            write_rows = torch.cat(
                [
                    self._physical_rows_range(slot, position, position + 1, x.device)
                    for slot, position in zip(slots, positions, strict=True)
                ]
            )
            mtp.cache_target_rows(
                self, self.mtp, ids[:, 0].to(x.device), mtp_previous, pos, write_rows
            )
            for row, slot in enumerate(slots):
                self.cache_hidden(slot, x[row, -1:])
        logits = self.unembed(rmsnorm(x, self.final_norm, self.cfg.eps))[:, 0].float()
        if self._timing_sample:
            self._log_decode_timing(len(slots), max(positions, default=0), t0)
        return logits

    def _log_decode_timing(self, batch: int, max_pos: int, t0: float) -> None:
        """`SEED_STEP_TIMING` only, on the sampled eager step: sync, then log this rank's
        step wall time, paged-attention kernel device time summed over the full-attention
        layers, and the host time spent building that kernel's block-table inputs."""
        host_issue_ms = (time.perf_counter() - t0) * 1e3
        for dev in {d for d in self.devices if d.type == "cuda"}:
            torch.cuda.synchronize(dev)
        wall_ms = (time.perf_counter() - t0) * 1e3
        attn_ms = sum(a.elapsed_time(b) for a, b in self._attn_events)
        step_timing.log(
            f"rank {self.tp.plan.rank} eager decode n={self._timing_decodes} batch={batch} "
            f"max_pos={max_pos} wall_ms={wall_ms:.1f} host_issue_ms={host_issue_ms:.1f} "
            f"attn_kernel_ms={attn_ms:.2f} over {len(self._attn_events)} layers "
            f"attn_table_build_host_ms={self._attn_host_ms:.2f}"
        )
        self._attn_events.clear()
        self._attn_host_ms = 0.0
        self._timing_sample = False

    # -- SEED_OVERLAP_SCHED -------------------------------------------------------
    @torch.no_grad()
    def decode_launch(
        self,
        slots: list[int],
        tokens: list[int],
        positions: list[int],
        temperatures: Sequence[float] | None,
    ) -> PendingTokens | None:
        """One decode step that samples on the device and does not wait for the result.

        `decode`'s contract, except that a `tokens` entry may be `LOOKAHEAD_TOKEN`: that lane's
        input is its own last sampled id, still in `lane_tokens` (one-step lookahead; the idea
        is SGLang's overlap scheduler and vLLM's async scheduling, re-derived here, nothing
        imported). The step's sampled ids are written back into `lane_tokens` on the device.
        Returns rank 0's `PendingTokens`; on the other ranks (which receive the ids and never
        read them) `None`. `temperatures` is only read on rank 0, the only rank that samples.
        """
        if self.gap_meter is not None:
            self.gap_meter.step_start()
        ids = self.resolve_tokens(tokens, slots)
        logits = self._decode_ids(slots, ids[:, None], positions)
        pending = self.launch_tail(slots, logits, temperatures)
        if self.gap_meter is not None:
            self.gap_meter.step_end()
        return pending

    def resolve_tokens(self, tokens: Sequence[int], slots: Sequence[int] | None) -> torch.Tensor:
        """[B] device token ids: `tokens[i]`, or lane `slots[i]`'s `lane_tokens` entry where
        `tokens[i]` is `LOOKAHEAD_TOKEN`. `slots=None` means row i is lane i over every lane
        (the captured step's layout, see `graph_decode.GraphDecodeRunner.fill`)."""
        dev = self.devices[0]
        host = host_to_device(list(tokens), dev, torch.long)
        if all(t != LOOKAHEAD_TOKEN for t in tokens):
            return host
        lane = (
            self.lane_tokens
            if slots is None
            else self.lane_tokens[host_to_device(list(slots), dev, torch.long)]
        )
        return torch.where(host == LOOKAHEAD_TOKEN, lane, host)

    def launch_tail(
        self, slots: Sequence[int], logits: torch.Tensor, temperatures: Sequence[float] | None
    ) -> PendingTokens | None:
        """Sample (rank 0), share the ids with every rank, store them per lane, and start the
        async readback. Shared by the eager `decode_launch` and the captured
        `GraphDecodeRunner.decode_launch`.

        Under TP only rank 0 samples, same as the synchronous path (a nonzero temperature must
        not draw different tokens on different ranks). Where the synchronous path relays the
        id to the others in the *next* command's payload, which needs it on rank 0's host
        first, here it goes over one device-side `broadcast` on the model's own process group,
        stream-ordered after the step and before the next step's embedding lookup on every
        rank, with no host round trip. Every rank calls this at the same point of the same
        command (`tp_driver.Op.DECODE_LAUNCH`), so the collective order stays identical.
        """
        rank = self.tp.plan.rank
        if rank == 0:
            assert temperatures is not None, "rank 0 samples, so it needs the temperatures"
            sampled = sample_on_device(logits, temperatures)
        else:
            sampled = torch.empty(len(slots), dtype=torch.long, device=logits.device)
        if self.tp.plan.world > 1:
            import torch.distributed as dist

            dist.broadcast(sampled, src=0)
        dev = self.devices[0]
        lanes = host_to_device(list(slots), dev, torch.long)
        self.lane_tokens[lanes] = sampled.to(dev, non_blocking=True)
        return PendingTokens(sampled, logits) if rank == 0 else None

    @torch.no_grad()
    def speculative_decode(
        self,
        slots: list[int],
        tokens: list[int],
        positions: list[int],
        budgets: Sequence[int] | None = None,
        stops: Sequence[Sequence[int]] | None = None,
    ) -> list[list[int]]:
        """One MTP decode round: draft `self.mtp.k` tokens off each slot's cached hidden
        state, verify all `self.mtp.k + 1` fed tokens in one unrolled forward, and commit the
        greedy-accepted prefix plus a bonus token. `tokens`/`positions` are each slot's last
        committed token id and its position -- `decode`'s own contract, unchanged -- and the
        return is each slot's newly committed token ids, length 1..`self.mtp.k + 1`.

        Only ever called with `self.mtp is not None` and every request in `slots` decoding
        greedy: `mtp.draft`'s own argmax has no sampling temperature to branch on, so a
        request with `temperature > 0` has to take the ordinary `decode` path instead (the
        scheduler's `speculative_decode`/`decode` choice is per step, over the whole active
        batch, not per request; see `Scheduler._decode_step`).

        `budgets`/`stops`: each lane's remaining token budget and stop ids; the committed list
        and the lane's state end at the first stop id or at the budget (`mtp.commit_limit`).
        """
        # Draft writes start at `position`, before verify's target attention gets a chance to
        # grow the lane. Under serving this validates the scheduler's prior reservation; bare
        # Model callers allocate the same `k + 1` rows here.
        for slot, position in zip(slots, positions, strict=True):
            self.grow_lane(slot, position + self.mtp.k + 1)
        hidden = torch.cat([self.cached_hidden(s) for s in slots], dim=0)
        draft_tokens = mtp.draft(self, self.mtp, hidden, list(tokens), slots, list(positions))
        return mtp.verify_and_commit(
            self, self.mtp, slots, list(tokens), draft_tokens, list(positions), budgets, stops
        )

    # -- entry points ---------------------------------------------------------
    @torch.no_grad()
    def forward(
        self,
        ids: torch.Tensor,
        start: int,
        *,
        all_logits: bool = False,
        logits_from: int | None = None,
    ) -> torch.Tensor:
        """Run tokens ids [1, T] at positions start..start+T. Returns fp32 logits [T or 1, vocab].

        `all_logits=True` returns logits for every position (equivalent to `logits_from=0`).
        `logits_from=k` returns logits for positions k..T-1 only: teacher-forced scoring
        (`/v1/score`) wants logits for its continuation tail, not the whole prompt, and this
        keeps the `[T, vocab]` projection bounded by the tail length instead of prompt length.
        Omitting both returns just the last position's logits, as every decode step does.
        """
        if self.gap_meter is not None:
            self.gap_meter.interrupt()  # device-busy between two decode steps, not idle
        mtp_previous = self.cached_hidden(self.current_slot).clone() if self.mtp is not None else None
        x = F.embedding(ids.to(self.devices[0]), self.embed)
        for i in range(len(self.layers)):
            x = self.layer(i, x.to(self.layer_dev[i]), start)
        if self.mtp is not None and logits_from is None:
            mtp.prefill_cache(
                self,
                self.mtp,
                self.current_slot,
                ids[0].to(x.device),
                x[0],
                start,
                mtp_previous,
            )
        if logits_from is not None:
            x = x[:, logits_from:].to(self.devices[-1])
        elif all_logits:
            x = x.to(self.devices[-1])
        else:
            x = x[:, -1:].to(self.devices[-1])
        if self.mtp is not None and logits_from is None:
            # The bound slot's raw hidden at its own last position, either way: `x` is
            # already sliced to one row when `all_logits` is off, and `[:, -1]` is that same
            # row's position when it is on. `mtp.draft`'s seed for this slot's next round.
            # Skipped for a `logits_from` (teacher-forced scoring) call: that path is never
            # followed by a decode step for this slot, so there is nothing to seed.
            self.cache_hidden(self.current_slot, x[:, -1])
        return self.unembed(rmsnorm(x, self.final_norm, self.cfg.eps))[0].float()

    @torch.no_grad()
    def forward_packed(self, ids: torch.Tensor, meta: PackedBatch) -> torch.Tensor:
        """Packed varlen prefill: `ids` [total_T] concatenated across `meta.seqs`.

        `forward`'s counterpart for several sequences at once: every layer runs
        `layer_packed` instead of `layer`. Unlike `forward`'s default, only the final
        norm + `lm_head` are narrowed here, to each sequence's own last packed row --
        every caller (`prefill_batch`) only ever wants that row per sequence, and running
        `lm_head` over the full `[total_T, hidden]` instead (every packed position, most of
        which nothing reads) is a `[total_T, vocab]` transient, ~3 GiB at 2k packed tokens.
        Returns `[len(meta.seqs), vocab]`, row-for-row with `meta.seqs`; `prefill_batch`
        indexes it by sequence position, not by packed-span offset.
        """
        if self.gap_meter is not None:
            self.gap_meter.interrupt()
        mtp_previous = (
            [self.cached_hidden(seq.slot).clone() for seq in meta.seqs]
            if self.mtp is not None
            else None
        )
        x = F.embedding(ids.to(self.devices[0]), self.embed)
        for i in range(len(self.layers)):
            x = self.layer_packed(i, x.to(self.layer_dev[i]), meta)
        x = x.to(self.devices[-1])
        if self.mtp is not None:
            for seq, (lo, hi), previous in zip(
                meta.seqs, meta.spans(), mtp_previous, strict=True
            ):
                mtp.prefill_cache(
                    self, self.mtp, seq.slot, ids[lo:hi].to(x.device), x[lo:hi], seq.start, previous
                )
        last_rows = [hi - 1 for _, hi in meta.spans()]
        if self.mtp is not None:
            # Each sequence's own last packed row, same rows `last_rows` picks out below.
            # `mtp.draft`'s seed for that slot's first decode round.
            for seq, row in zip(meta.seqs, last_rows, strict=True):
                self.cache_hidden(seq.slot, x[row : row + 1])
        x = x[last_rows]
        return self.unembed(rmsnorm(x, self.final_norm, self.cfg.eps)).float()

    @torch.no_grad()
    def generate(self, prompt: list[int], max_new: int, temperature: float, stop: frozenset[int]):
        """Yield generated token ids (a stop token is yielded last, then generation ends).

        No scheduler or `SessionCache` mediates this path (single-sequence, always slot 0),
        so this call is its own only owner of whatever slot 0's block table held from a prior
        `generate` call and must release it before `begin` clears the table -- see `reset`'s
        and `_release_lane_blocks`'s docstrings.
        """
        self._release_lane_blocks(0)
        self.begin(0)  # single-sequence path: always slot 0
        logits = None
        for s in range(0, len(prompt), PREFILL_CHUNK):
            chunk = torch.tensor([prompt[s : s + PREFILL_CHUNK]])
            logits = self.forward(chunk, s)
        pos = len(prompt)
        for _ in range(max_new):
            tok = self.sample(logits[-1], temperature)
            yield tok
            if tok in stop:
                return
            logits = self.forward(torch.tensor([[tok]]), pos)
            pos += 1

    @staticmethod
    def sample(logits: torch.Tensor, temperature: float) -> int:
        """One row, one token. The single-sequence reference path (`generate`) uses this."""
        if temperature <= 0:
            return int(logits.argmax())
        return int(torch.multinomial((logits / temperature).softmax(-1), 1))

    @staticmethod
    def sample_batch(logits: torch.Tensor, temperatures: list[float]) -> list[int]:
        """One token per row of [B, vocab], with a single host sync for the whole batch.

        The per-row `sample` above needs one device-to-host sync per token, which is what a
        batched decode step must not do B times. Whether a row is greedy is known on the
        host (it is the request's temperature), so the branch below needs no sync either.
        """
        return sample_on_device(logits, temperatures).tolist()

    def warmup(self) -> None:
        for _ in self.generate([1, 2, 3, 4], 2, 0.0, frozenset()):
            pass
        self._release_lane_blocks(self.current_slot)
        self.reset()
        # Freeze the hot-expert cache (no-op unless `SEED_HOT_EXPERTS=1`) before whatever calls
        # `warmup` next enables graph capture: see `finalize_hot_experts`'s docstring for why
        # this traffic alone is a weak source of hot-expert statistics.
        self.finalize_hot_experts()

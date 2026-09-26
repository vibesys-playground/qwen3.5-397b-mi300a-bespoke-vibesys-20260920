"""Tensor parallelism: which slice of each axis a rank owns, and the collectives.

TP replaces the seed's layer-by-layer device split (the *pipeline* stages in pipeline.py).
Under that split one decode step ran one layer at a time on one device, so three of the four
devices were idle at every instant and the whole node's expert-weight traffic went through
one device's HBM sequentially. Measured at batch 48 that is 192.5 GB per step on one device:
36.3 ms at MI300A's *spec* 5.3 TB/s, more than the 30.9 ms the objective's gate allows for
the entire step, before any kernel runs. Under TP every rank runs every layer on a shard of
its heads and experts, so the same traffic is 48.1 GB per device, concurrently. The price is
the collectives below; see DECODE_BOTTLENECK_2026-09-22.md sections 3 and 4 for both sides
of that trade, measured.

# Convention

This is the contract any further sharding work has to hold to. Read it before adding a
component.

**Ranks and devices.** One process per rank, `world` ranks, rank `r` owns `cuda:r` and
nothing else. There is no second axis: no pipeline stage, no data-parallel replica. Every
rank runs the whole layer stack on every token, so the ranks are in lockstep and every
collective is reached by all of them in the same order. `Plan` is a pure value: it is the
shard arithmetic alone and holds no tensors, no device and no process group, so it can be
constructed and tested without torch.distributed. `TP` binds a `Plan` to a device and an
all-reduce.

**Column- and row-parallel (the Megatron-LM convention).** A linear whose output axis is a
count of heads or of intermediate channels is *column-parallel*: rank `r` keeps the output
rows its heads own, its output is that shard of the real output, and no collective is
needed. A linear that consumes such a shard and produces hidden-size output is
*row-parallel*: rank `r` keeps the input columns its heads own, and its output is a
**partial sum**, correct only once summed across ranks. Two column-parallel projections
feeding a pointwise nonlinearity feeding one row-parallel projection is the unit; all the
sharding here is an instance of it.

**Where the collectives are.** One all-reduce per row-parallel output, placed in
`Model.layer` / `Model.decode_layer`, not inside the component. A component returns its
partial sum and the caller reduces it. Two per layer once every component is sharded (one
after the mixer's output projection, one after the MoE), so 120 per forward pass at 60
layers. Nothing else communicates: routing, sampling and the residual stream are replicated,
not exchanged.

**Replicated tensors.** Embeddings, the LM head, every RMSNorm weight (they normalize over
head_dim or hidden, never over a sharded axis), the router, and the shared-expert gate. Any
tensor whose input and output axes are both hidden-size stays whole on every rank. This is
what keeps routing collective-free: every rank holds the same router and the same hidden
state, so all ranks pick the same experts without talking. That in turn assumes the
all-reduce is bit-identical *across ranks*, which RCCL's ring/tree reduction is, and that the
ranks run identical code on identical hardware.

**Numerics, and why the collective is fp32.** A row-parallel projection sums `world`
separately rounded partials instead of accumulating one dot product, so a sharded forward is
equal to the unsharded one only up to that rounding. That much is intrinsic to the sharding
and cannot be removed. What *can* be removed is a second rounding on top of it: reducing in
the activation dtype rounds every intermediate sum of the ring to bf16 as well.
`all_reduce` therefore upcasts to fp32, reduces, and rounds once on the way back, so the
result is the correctly rounded bf16 of the exact sum of the four partials.

This matters because the objective's accuracy gate is greedy-token pins (90% of pins must
match the first 32 generated tokens *exactly*), and greedy decoding turns any logit
perturbation into a token flip at the first near-tie. Landing TP and a reduced-precision
collective at once would make a pin regression impossible to attribute to either. The cost
is 786 KiB per collective instead of 393 KiB at batch 48, 120 times per step; these are
latency-dominated at this size, so it is well under a millisecond of an ~85 ms step. Do not
switch this to bf16 without measuring it as a separate change, against the pins.

**The custom all-reduce (`allreduce_custom.py`).** RCCL's ring/tree collective pays several
kernel launches and inter-rank round trips no matter how little data moves, so at these sizes
it is latency- not bandwidth-bound (see `rccl-bench/size_bf16_results.jsonl`: ~40-100us flat
from 16 KiB to 2 MiB). On one node with a full XGMI mesh, a direct one-shot all-reduce over
IPC-mapped peer buffers removes that per-call floor. `TP.all_reduce` routes a bf16 partial at
or under `allreduce_custom.MAX_BYTES` (1.5 MiB) through it instead of the fp32-upcast RCCL
path above -- transmitting bf16 (the value already is bf16 at that point) rather than
upcasting to fp32 for the wire, but still reducing in fp32 and rounding once, so the result is
bit-identical to the path above, not a separate, less-accurate one. `SEED_CUSTOM_ALLREDUCE=0`
disables it (falls back to RCCL); it also disables itself, silently, if the IPC exchange or
the extension build fails. See `allreduce_custom.py`'s module docstring for the design and the
capture-safety argument.

# Sharding, per component

| Component | Axis | Split | Collective |
|:--|:--|:--|:--|
| Full attention | query heads (32) | even, `heads / world` per rank | all-reduce after `o_proj` |
| Full attention | KV heads (2) | replicated at `world > kv_heads`, see `kv_shard` | (same) |
| MoE routed experts (512) | expert index | whole experts per rank, `experts / world` | all-reduce after the combine |
| MoE shared expert | intermediate (1024) | even | (same all-reduce) |
| Gated DeltaNet | v heads (64), k heads (16) | even on both, see `deltanet_tp` | all-reduce after `out_proj` |

## Gated DeltaNet

`deltanet_tp.py` shards the 45 DeltaNet layers over value heads and carries the derivation:
the gated delta rule's state is block diagonal over value heads, so the recurrence itself
communicates nothing at any timestep or chunk boundary, and only `out_proj` is row-parallel.
The key-head split is forced by the value-head one (value head `j` reads key head `j // rep`
with `rep = v_heads // k_heads`, so a rank's value-head block has to line up with its
key-head block, which `even` on both axes gives exactly). `dn_k` / `dn_v` below name those
two cuts; `deltanet_tp` reaches the same slices through the generic `split_size` / `shard`
helpers, and `seed_tests/test_tensor_parallel.py` pins the two against each other.

Because the DeltaNet `out_proj` is row-parallel like attention's `o_proj`, `Model.layer` and
`Model.decode_layer` all-reduce their mixer unconditionally: no mixer's output is already
whole, so there is no case where reducing would multiply it by the world size. Two
collectives per layer, 120 per forward pass at 60 layers.

Per device this leaves a quarter of the DeltaNet weights (about 2.6 GB of 10.6 GB) and a
quarter of the per-slot recurrent state (about 90 MB per slot including the prefix
snapshot), which is what the memory math in `model.py` counts.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import torch

Reducer = Callable[[torch.Tensor], None]
"""Sums a tensor across the ranks of the group, in place.

The collective is injected rather than called directly anywhere in the model, so swapping
the backend is a different factory, not an edit to the forward pass. `init` supplies the
c10d RCCL one, and `seed_tests/test_tensor_parallel.py` supplies a thread-barrier one to
drive four sharded models in one process without a group at all.
"""

DEFAULT_PORT = 29500


@dataclass(frozen=True)
class Shard:
    """`count` consecutive items of some axis, the first of them at index `start`."""

    start: int
    count: int

    @property
    def stop(self) -> int:
        return self.start + self.count

    def rows(self, width: int) -> slice:
        """The rows of a weight laid out as `width` output rows per item of this axis."""
        return slice(self.start * width, self.stop * width)


def even(total: int, rank: int, world: int, axis: str) -> Shard:
    """Split `total` items evenly. `axis` names the config field in the error."""
    if total % world:
        raise ValueError(f"tp: {axis}={total} does not divide over {world} ranks")
    per = total // world
    return Shard(rank * per, per)


def kv_shard(heads: int, kv_heads: int, rank: int, world: int) -> Shard:
    """The KV heads rank `rank` needs, given it owns `heads // world` query heads.

    With at least one KV head per rank this is an even split. With fewer, which is this
    model at TP=4 (32 query heads, 2 KV heads, so GQA 16:1), the KV heads are *replicated*:
    query head `j` reads KV head `j // rep` with `rep = heads // kv_heads`, so a rank whose
    query block sits inside one KV group needs exactly that one KV head, and `world //
    kv_heads` ranks each keep a copy of it. Replicating is what makes the split legal at
    all; the alternative, splitting 2 KV heads over 4 ranks, has no meaning. It costs
    `world // kv_heads` copies of the KV cache node-wide, but per device the cache still
    shrinks by `kv_heads / world` (here 2 heads to 1), which is what the memory budget sees.

    The query block must not straddle two KV groups, hence the check: it would need two KV
    heads on one rank while its neighbour needs none, and neither `enable_gqa` nor the
    group-into-query-length fold `decode_attention`/`prefill_attention` use can express that.
    """
    if kv_heads >= world:
        return even(kv_heads, rank, world, "num_key_value_heads")
    rep, per_rank = heads // kv_heads, heads // world
    if heads % kv_heads or rep % per_rank:
        raise ValueError(
            f"tp: {heads} query heads over {world} ranks puts a rank's query block across "
            f"two of the {kv_heads} KV groups"
        )
    return Shard(rank // (world // kv_heads), 1)


@dataclass(frozen=True)
class Plan:
    """One rank's slice of every sharded axis. Pure arithmetic: no tensors, no device."""

    rank: int
    world: int
    q: Shard
    """Full-attention query heads. `q_proj` is column-parallel, `o_proj` row-parallel."""
    kv: Shard
    """Full-attention KV heads, replicated when they are fewer than the ranks."""
    dn_k: Shard
    """Gated DeltaNet key heads. Forced to match `dn_v`; see module doc and `deltanet_tp`."""
    dn_v: Shard
    """Gated DeltaNet value heads. `in_proj_*` column-parallel, `out_proj` row-parallel."""
    experts: Shard
    """Routed experts owned whole by this rank (expert parallel)."""
    inter: Shard
    """Shared-expert intermediate channels. gate/up column-parallel, down row-parallel."""


def plan(cfg, rank: int, world: int) -> Plan:  # noqa: ANN001 -- model.Cfg, avoiding a cycle
    """Build `rank`'s plan, or raise naming the config field that does not divide."""
    if world < 1 or not 0 <= rank < world:
        raise ValueError(f"tp: rank {rank} is not a rank of a {world}-rank group")
    return Plan(
        rank=rank,
        world=world,
        q=even(cfg.heads, rank, world, "num_attention_heads"),
        kv=kv_shard(cfg.heads, cfg.kv_heads, rank, world),
        dn_k=even(cfg.k_heads, rank, world, "linear_num_key_heads"),
        dn_v=even(cfg.v_heads, rank, world, "linear_num_value_heads"),
        experts=even(cfg.experts, rank, world, "num_experts"),
        inter=even(cfg.shared_inter, rank, world, "shared_expert_intermediate_size"),
    )


class TP:
    """One rank's plan, its device, and its all-reduce.

    `reduce` is injected rather than reached for directly so that the sharded forward can be
    exercised without a process group at all: `seed_tests/test_tensor_parallel.py` drives
    four ranks as threads over a barrier, and `world=1` needs no reduce. `init` builds the
    real RCCL-backed one.
    """

    def __init__(self, plan_: Plan, device: str | torch.device, reduce: Reducer | None) -> None:
        if plan_.world > 1 and reduce is None:
            raise ValueError(f"tp: world size {plan_.world} needs an all-reduce")
        self.plan, self.device, self._reduce = plan_, torch.device(device), reduce
        self.custom_reduce = None
        if plan_.world > 1 and torch.cuda.is_available():
            import allreduce_custom

            self.custom_reduce = allreduce_custom.build(plan_.rank, plan_.world, self.device)

    @classmethod
    def single(cls, cfg, device: str | torch.device) -> TP:  # noqa: ANN001 -- model.Cfg
        """The unsharded model: one rank, no collectives. The accuracy and parity path."""
        return cls(plan(cfg, 0, 1), device, None)

    @classmethod
    def from_torch_distributed(cls, cfg, device: str | torch.device) -> TP:  # noqa: ANN001
        """Wrap the already-initialized default process group, or fall back to `single`.

        The seam for code that did not bring the group up itself. `init` is the other
        direction: it brings the group up and hands back just the reducer.
        """
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return cls.single(cfg, device)
        return cls(plan(cfg, dist.get_rank(), dist.get_world_size()), device, _reduce_default)

    @property
    def rank(self) -> int:
        return self.plan.rank

    @property
    def world(self) -> int:
        return self.plan.world

    @property
    def world_size(self) -> int:
        """`world` under the name `deltanet_tp` uses."""
        return self.plan.world

    @property
    def enabled(self) -> bool:
        """Whether there is anything to communicate with."""
        return self.plan.world > 1

    def split_size(self, total: int, what: str) -> int:
        """`total / world`, rejecting an axis that does not divide evenly."""
        return even(total, self.plan.rank, self.plan.world, what).count

    def shard(self, t: torch.Tensor, dim: int, what: str) -> torch.Tensor:
        """This rank's contiguous slice of `t` along `dim`."""
        n = self.split_size(t.shape[dim], what)
        return t.narrow(dim, self.plan.rank * n, n).contiguous()

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Sum a row-parallel partial across ranks. Returns `x`, which is summed in place.

        A bf16 (or fp16) partial is reduced in fp32 and rounded once on the way back, so the
        result is the correctly rounded activation-dtype value of the exact sum, not a sum of
        separately rounded ring intermediates. See "Numerics" in the module docstring for why
        that is worth the doubled wire bytes and what it would take to change it.
        """
        if self._reduce is None:
            return x
        if (
            self.custom_reduce is not None
            and x.dtype == torch.bfloat16
            and x.numel() <= self.custom_reduce.max_elems
        ):
            return self.custom_reduce(x)
        if x.dtype in (torch.float32, torch.float64):
            self._reduce(x)
            return x
        wide = x.float()
        self._reduce(wide)
        return x.copy_(wide)

    def all_gather_last(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate every rank's `x` along the last dim, in rank order (`[..., n]` ->
        `[..., world * n]`, identical on every rank). One all-gather on the default group,
        stream-ordered and capturable like the all-reduce; `SEED_LMHEAD_VOCAB_TP`'s logits."""
        if self.plan.world == 1:
            return x
        import torch.distributed as dist

        lead, n = x.shape[:-1], x.shape[-1]
        flat = x.reshape(-1, n).contiguous()
        if flat.is_cuda:
            out = torch.empty((self.plan.world, *flat.shape), dtype=x.dtype, device=x.device)
            dist.all_gather_into_tensor(out, flat)
        else:  # gloo (the CPU tests)
            parts = [torch.empty_like(flat) for _ in range(self.plan.world)]
            dist.all_gather(parts, flat)
            out = torch.stack(parts)
        return out.permute(1, 0, 2).reshape(*lead, self.plan.world * n)

    def all_reduce_residual_norm(
        self,
        mixer: torch.Tensor,
        residual: torch.Tensor,
        norm_w: torch.Tensor,
        eps: float,
        rmsnorm_fn: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`residual + self.all_reduce(mixer)`, then `rmsnorm_fn` of the result -- one kernel.

        `model.decode_layer`/`graph_decode.decode_layer_static` call this once after the
        mixer's all-reduce (normalizing into `post_norm`) and once after MoE's (normalizing
        into the next layer's `in_norm`, or the final norm on the last layer): the same
        all-reduce-then-residual-then-norm sequence, twice a layer. Idea: TensorRT-LLM's and
        vLLM's fused-all-reduce-residual-RMSNorm custom kernels (cited for the shape of the
        fusion in `allreduce_custom.CustomAllReduce.residual_norm`'s docstring, which has the
        numerics argument for why the fused kernel's residual add reproduces the unfused
        double-rounding exactly).

        `rmsnorm_fn` is injected (the `Reducer` this file already takes is the same shape of
        seam) rather than imported, so `tp.py` does not import `model.py` -- `model.py`
        already imports `tp.py`, and a reverse edge would be a cycle the coding-best-practices
        upward-import rule forbids. Every real call site passes `model.rmsnorm`.

        Falls back to the exact unfused sequence -- `x = residual + self.all_reduce(mixer);
        return x, rmsnorm_fn(x, norm_w, eps)` -- whenever the fused kernel cannot run:
        `SEED_FUSED_AR_NORM` off (default), `world == 1`, the custom one-shot all-reduce is
        unavailable (RCCL path, CPU/gloo, or IPC/build failed), the payload exceeds its cap,
        the dtype is not bf16, or `hidden` is not a multiple of 8. This is not a second, less-
        accurate path a caller has to reason about separately: it is bit-for-bit the same
        computation as the fallback, restated as one kernel instead of three (see the
        `residual_norm` docstring for the numerics argument).
        """
        use_fused = (
            os.environ.get("SEED_FUSED_AR_NORM", "0") not in ("0", "false", "False")
            and self.custom_reduce is not None
            and mixer.dtype == torch.bfloat16
            and residual.dtype == torch.bfloat16
            and mixer.numel() <= self.custom_reduce.max_elems
            and mixer.shape[-1] % 8 == 0
        )
        if use_fused:
            return self.custom_reduce.residual_norm(mixer, residual, norm_w, eps)
        x = residual + self.all_reduce(mixer)
        return x, rmsnorm_fn(x, norm_w, eps)


def default_world() -> int:
    """Ranks to run: one per visible GPU, or 1 on CPU. ROCm reports through `torch.cuda`."""
    return torch.cuda.device_count() or 1


def device_for(rank: int) -> str:
    """Rank `r` owns `cuda:r`. The single convention; nothing else picks devices."""
    return f"cuda:{rank}" if torch.cuda.is_available() else "cpu"


def _reduce_default(x: torch.Tensor) -> None:
    """In-place sum over torch.distributed's default group."""
    import torch.distributed as dist

    dist.all_reduce(x, op=dist.ReduceOp.SUM)


def init(rank: int, world: int, *, port: int = DEFAULT_PORT, backend: str = "") -> Reducer:
    """Join the process group and return its in-place all-reduce.

    `backend="nccl"` is RCCL in a ROCm build of torch: the two share the API and the backend
    name, and `torch.distributed.is_nccl_available()` reports the ROCm one. The default here
    picks nccl when there are GPUs and gloo otherwise, which is what lets the CPU test in
    `seed_tests/` exercise this same function.

    The rendezvous binds `127.0.0.1:port`, which assumes all `world` ranks are processes of
    one node-local container, and it does not set `ROCR_VISIBLE_DEVICES`, so it assumes each
    rank can see all four devices and selects its own by index. That is how `server.py`
    starts them and how the Slurm/pyxis container runs.
    """
    import torch.distributed as dist

    backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))
    dist.init_process_group(backend=backend, rank=rank, world_size=world)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    return _reduce_default

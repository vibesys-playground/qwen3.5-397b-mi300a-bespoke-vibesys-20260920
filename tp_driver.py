"""Driving the ranks from one scheduler: the SPMD command protocol.

`scheduler.py` is written against a single `Runner` and does not know about ranks. Under
tensor parallelism all `world` ranks have to make the identical sequence of model calls, in
the same order, because each holds a shard of every layer and every all-reduce has to be
reached by all of them. So rank 0 runs the scheduler and broadcasts each `Runner` call before
making it; ranks 1.. sit in `serve_worker`, receive the call, make it, and drop the result.

The protocol is two broadcasts per command: a fixed four-int64 header (op, two scalars, and
the payload length) and then the payload. Commands are tiny next to the 120 all-reduces a
forward pass already costs (a prefill chunk is at most `PREFILL_CHUNK` ids, a packed prefill
batch at most `PREFILL_TOKEN_BUDGET` ids plus three ints per request, a decode step
`3 * max_batch` ints), so there is nothing to gain from packing them harder.

Only rank 0 samples. Every rank computes the same logits, since the LM head is replicated,
but sampling on one rank keeps a nonzero temperature from drawing different tokens on
different ranks. The chosen token reaches the others as the payload of the next command.

Overlap (`SEED_OVERLAP_SCHED`, see `scheduler.OVERLAP_SCHED`). Two things change. Decode
steps go out as `DECODE_LAUNCH`, whose token list may say "this lane's last sampled id"
(`model.LOOKAHEAD_TOKEN`) instead of an id: rank 0 no longer has that id on its host when it
launches the next step, so the ranks exchange it themselves, device-side, inside the step
(`Model.launch_tail`). And the command channel moves to a host-side gloo group
(`command_channel`). Over the device group, `send` builds its header with a synchronizing
host-to-device copy, and a worker's `recv` reads it back with `.tolist()`, which waits for
that rank's previous step to finish before it can even see the next command. Either one puts
the per-step bubble back. Over gloo the commands travel host to host, so every rank enqueues
step N+1 while its device still runs step N. Ordering is unchanged: the command sequence is
identical on every rank, so the device collectives each command issues stay in the same order.

Failure mode to know about: this is a lockstep protocol with no timeout. If one rank raises
inside a model call, it stops broadcasting or stops receiving and the others block on their
next collective. `Broadcaster` therefore does not swallow exceptions, and `server.py` watches
the worker processes so a dead rank surfaces on `/health` instead of hanging silently.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import torch

HEADER = 4
"""int64s in a command header: op, two scalars, payload length."""

NO_BUDGET = 2**31 - 1
"""A SPECULATIVE_DECODE budget that never clamps (`Broadcaster.speculative_decode` called
without budgets)."""


class Op(IntEnum):
    """One `Runner` call, or a lifecycle step the ranks take together."""

    BEGIN = 0
    PREFILL = 1
    PREFILL_BATCH = 2
    SAVE_SNAPSHOT = 3
    LOAD_SNAPSHOT = 4
    DECODE = 5
    WARMUP = 6
    STOP = 7
    ATTACH_BLOCKS = 8
    COPY_BLOCK = 9
    SPECULATIVE_DECODE = 10
    SCORE = 11
    EXTEND_BLOCKS = 12
    POOL_HANDSHAKE = 13
    DECODE_LAUNCH = 14
    DECODE_MIXED = 15
    SPECULATIVE_LAUNCH = 16


@dataclass(frozen=True)
class Command:
    """One broadcast unit. `a`/`b` are the scalar arguments, `payload` the variable part."""

    op: Op
    a: int = 0
    b: int = 0
    payload: tuple[int, ...] = field(default=())

    def header(self) -> list[int]:
        return [int(self.op), self.a, self.b, len(self.payload)]

    @classmethod
    def decode_step(
        cls, slots: Sequence[int], tokens: Sequence[int], pos: Sequence[int], op: Op = Op.DECODE
    ) -> Command:
        """A DECODE (or DECODE_LAUNCH) command. `a` is the batch size; the three lists are
        concatenated. SPECULATIVE_DECODE extends this payload (`speculative_cmd`)."""
        return cls(op, a=len(slots), payload=(*slots, *tokens, *pos))

    def batch(self) -> tuple[list[int], list[int], list[int]]:
        """Split a DECODE payload back into (slots, tokens, positions)."""
        b = self.a
        p = self.payload
        return list(p[:b]), list(p[b : 2 * b]), list(p[2 * b :])

    @classmethod
    def speculative_cmd(
        cls,
        slots: Sequence[int],
        tokens: Sequence[int],
        pos: Sequence[int],
        budgets: Sequence[int],
        stops: Sequence[Sequence[int]],
        forced: Sequence[Sequence[int]] | None = None,
        op: Op = Op.SPECULATIVE_DECODE,
    ) -> Command:
        """A SPECULATIVE_DECODE (or SPECULATIVE_LAUNCH) command (one MTP round). `a` is the batch size, `b` the stop-id
        width: the payload is `decode_step`'s three lists, then each lane's remaining token
        budget, then each lane's stop ids right-padded with -1 to `b` (row-major). Every rank
        needs the limits, not just rank 0: they decide each lane's committed length and so the
        DeltaNet state every rank rolls back to (`mtp.commit_limit`)."""
        width = max((len(s) for s in stops), default=0)
        flat = [x for s in stops for x in (*s, *([-1] * (width - len(s))))]
        return cls(
            op,
            a=len(slots),
            b=width,
            payload=(*slots, *tokens, *pos, *budgets, *flat, *_forced_payload(forced)),
        )

    def unbatch_speculative(
        self,
    ) -> tuple[list[int], list[int], list[int], list[int], list[list[int]]]:
        """Split a SPECULATIVE_DECODE payload into (slots, tokens, positions, budgets, stops)."""
        n, w, p = self.a, self.b, self.payload
        slots, tokens, pos = Command(Op.DECODE, a=n, payload=p[: 3 * n]).batch()
        budgets = list(p[3 * n : 4 * n])
        flat = p[4 * n :]
        stops = [[x for x in flat[j * w : (j + 1) * w] if x >= 0] for j in range(n)]
        return slots, tokens, pos, budgets, stops

    def unbatch_forced(self) -> list[list[int]] | None:
        """Forced feed appended by `speculative_cmd(forced=...)`: per lane a count, then
        `k` ids (right-padded). None when the command carries none."""
        n, w, p = self.a, self.b, self.payload
        rest = p[4 * n + n * w :]
        if not rest:
            return None
        counts, flat = rest[:n], rest[n:]
        k = len(flat) // n
        return [list(flat[j * k : j * k + counts[j]]) for j in range(n)]

    @classmethod
    def prefill_batch_cmd(cls, calls: Sequence[tuple[int, Sequence[int], int]]) -> Command:
        """A PREFILL_BATCH command: several (slot, ids, start) calls in one broadcast.

        `a` is the number of calls. Each call's `ids` is a different length, so the
        payload cannot just concatenate three same-length arrays the way `decode_step`
        does: it holds `slots`, `starts`, and each call's own `len(ids)` (one int per
        call, in that order) followed by every call's ids concatenated, in call order.
        `unbatch_prefill` is the inverse.
        """
        slots = [slot for slot, _, _ in calls]
        starts = [start for _, _, start in calls]
        lengths = [len(ids) for _, ids, _ in calls]
        flat_ids = [tok for _, ids, _ in calls for tok in ids]
        return cls(Op.PREFILL_BATCH, a=len(calls), payload=(*slots, *starts, *lengths, *flat_ids))

    def unbatch_prefill(self) -> list[tuple[int, list[int], int]]:
        """Split a PREFILL_BATCH payload back into (slot, ids, start) calls, in call order."""
        n = self.a
        p = self.payload
        slots, starts, lengths = p[:n], p[n : 2 * n], p[2 * n : 3 * n]
        ids = p[3 * n :]
        calls, lo = [], 0
        for slot, start, length in zip(slots, starts, lengths, strict=True):
            calls.append((slot, list(ids[lo : lo + length]), start))
            lo += length
        return calls

    @classmethod
    def mixed_cmd(
        cls,
        slots: Sequence[int],
        tokens: Sequence[int],
        pos: Sequence[int],
        calls: Sequence[tuple[int, Sequence[int], int]],
    ) -> Command:
        """A DECODE_MIXED command: `decode_step`'s payload followed by `prefill_batch_cmd`'s.
        `a` is the decode batch size, `b` the number of prefill calls."""
        prefill = cls.prefill_batch_cmd(calls)
        return cls(
            Op.DECODE_MIXED,
            a=len(slots),
            b=len(calls),
            payload=(*slots, *tokens, *pos, *prefill.payload),
        )

    def unbatch_mixed(
        self,
    ) -> tuple[list[int], list[int], list[int], list[tuple[int, list[int], int]]]:
        """Split a DECODE_MIXED payload into (slots, tokens, positions, prefill calls)."""
        n = 3 * self.a
        slots, tokens, pos = Command(Op.DECODE, a=self.a, payload=self.payload[:n]).batch()
        calls = Command(Op.PREFILL_BATCH, a=self.b, payload=self.payload[n:]).unbatch_prefill()
        return slots, tokens, pos, calls


class PoolMismatchError(RuntimeError):
    """The ranks' KV pools disagree at boot; serving would corrupt KV or hang."""


def pool_config(model: Any) -> list[int]:
    """What every rank's paged-KV bookkeeping must agree on before serving starts."""
    return [
        model.num_kv_blocks,
        model.block_size,
        model.max_blocks_per_lane,
        model.num_snapshots,
        model.max_batch,
        model.block_allocator.free_count,
    ]


def pool_handshake(model: Any) -> None:
    """Boot self-check, then hand block allocation to the scheduler. Every rank calls this.

    All-gathers `pool_config` and raises `PoolMismatchError` on every rank if any rank
    differs, or if any block is still allocated (boot-time warmup/capture must release what
    it grew). Then sets `model.scheduler_owns_blocks`: from here on rank 0's scheduler is the
    single allocator authority, and a forward that would need a rank-local allocation raises
    instead (see `Model.grow_lane`).
    """
    mine = pool_config(model)
    world = model.tp.plan.world
    if world > 1:
        import torch.distributed as dist

        t = torch.tensor(mine, dtype=torch.int64, device=model.tp.device)
        gathered = [torch.zeros_like(t) for _ in range(world)]
        dist.all_gather(gathered, t)
        rows = [g.tolist() for g in gathered]
    else:
        rows = [mine]
    fields = "num_kv_blocks, block_size, max_blocks_per_lane, num_snapshots, max_batch, free"
    if any(row != rows[0] for row in rows):
        raise PoolMismatchError(f"KV pool config differs across ranks ({fields}): {rows}")
    if model.block_allocator.free_count != model.block_allocator.usable_blocks:
        raise PoolMismatchError(
            f"{model.block_allocator.usable_blocks - model.block_allocator.free_count} KV "
            "block(s) still allocated after boot warmup; they would leak"
        )
    model.scheduler_owns_blocks = True


class Channel:
    """Broadcasts commands from `src` to the rest of the group.

    Tensors are allocated on `device` because RCCL collectives need device memory; the gloo
    backend the CPU test uses takes host tensors, and `device` is `cpu` there.
    """

    def __init__(self, device: torch.device, src: int = 0, group: Any = None) -> None:
        self.device, self.src, self.group = device, src, group

    def _bcast(self, t: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist

        dist.broadcast(t, src=self.src, group=self.group)
        return t

    def send(self, cmd: Command) -> None:
        self._bcast(torch.tensor(cmd.header(), dtype=torch.int64, device=self.device))
        if cmd.payload:
            self._bcast(torch.tensor(cmd.payload, dtype=torch.int64, device=self.device))

    def recv(self) -> Command:
        head = self._bcast(torch.zeros(HEADER, dtype=torch.int64, device=self.device)).tolist()
        op, a, b, n = head
        if not n:
            return Command(Op(op), a, b)
        body = self._bcast(torch.zeros(n, dtype=torch.int64, device=self.device))
        return Command(Op(op), a, b, tuple(body.tolist()))


def command_channel(device: torch.device, overlap: bool) -> Channel:
    """The command channel every rank must build the same way, after the default group is up.

    `overlap` (`SEED_OVERLAP_SCHED`) puts it on a separate gloo group of host tensors; see the
    module docstring for why the device group would reintroduce the per-step bubble.
    `new_group` is itself collective, so every rank calls this at the same point of startup.
    """
    if not overlap:
        return Channel(device)
    import torch.distributed as dist

    return Channel(torch.device("cpu"), group=dist.new_group(backend="gloo"))


def _forced_payload(forced: Sequence[Sequence[int]] | None) -> tuple[int, ...]:
    """`(counts..., ids...)` with ids right-padded to the longest list; empty when no lane has
    forced feed, so the payload is unchanged for every round without it."""
    if not forced or not any(forced):
        return ()
    k = max(len(f) for f in forced)
    ids = [x for f in forced for x in (*f, *([0] * (k - len(f))))]
    return (*(len(f) for f in forced), *ids)


class Broadcaster:
    """The `Runner` rank 0 hands the scheduler: announce the call, then make it locally.

    Every method broadcasts before it touches the model, so a worker is never behind on a
    call that has already changed rank 0's state.
    """

    def __init__(self, model: Any, send: Callable[[Command], None]) -> None:
        self.model, self.send = model, send
        self.max_batch: int = model.max_batch

    @property
    def decode_path(self) -> str:
        """The wrapped runner's last decode path (`SEED_STEP_TIMING` reporting only)."""
        return getattr(self.model, "decode_path", "eager")

    @property
    def prefill_path(self) -> str:
        """The wrapped runner's last prefill path (`SEED_STEP_TIMING` reporting only)."""
        return getattr(self.model, "prefill_path", "eager")

    @property
    def mixed_path(self) -> str:
        """The wrapped runner's last mixed-step path (`SEED_STEP_TIMING` reporting only)."""
        return getattr(self.model, "mixed_path", "eager")

    def mixed_fit(self, n_decode: int) -> Any:
        """The wrapped runner's captured mixed shapes for `n_decode` decode rows
        (`GraphDecodeRunner.mixed_fit`), or None. Local read, like `prefill_fit`."""
        fit = getattr(self.model, "mixed_fit", None)
        return fit(n_decode) if fit is not None else None

    def prefill_fit(self) -> Any:
        """The wrapped runner's captured prefill shapes (`GraphDecodeRunner.prefill_fit`), or
        None. Local read: every rank agreed on the same shapes at boot."""
        fit = getattr(self.model, "prefill_fit", None)
        return fit() if fit is not None else None

    def prefill_shapes(self) -> list[tuple[int, int]]:
        """The wrapped runner's captured prefill shapes (`GraphDecodeRunner.prefill_shapes`).
        Local read, like `prefill_fit`."""
        shapes = getattr(self.model, "prefill_shapes", None)
        return shapes() if shapes is not None else []

    def prefill_packing(self) -> dict[tuple[int, int], tuple[int, int]]:
        """The wrapped runner's packed prefill shapes (`GraphDecodeRunner.prefill_packing`).
        Local read, like `prefill_fit`."""
        packing = getattr(self.model, "prefill_packing", None)
        return packing() if packing is not None else {}

    def begin(self, slot: int) -> None:
        self.send(Command(Op.BEGIN, slot))
        self.model.begin(slot)

    def prefill(self, slot: int, ids: Sequence[int], start: int) -> Any:
        self.send(Command(Op.PREFILL, slot, start, tuple(ids)))
        return self.model.prefill(slot, list(ids), start)

    def prefill_batch(self, calls: Sequence[tuple[int, Sequence[int], int]]) -> list[Any]:
        """The packed-prefill counterpart of `prefill`: one broadcast for `scheduler.py`'s
        whole packed batch, then one local `model.prefill_batch` call covering all of it,
        same as `decode` covers a whole decode step in one broadcast."""
        self.send(Command.prefill_batch_cmd(calls))
        return self.model.prefill_batch([(slot, list(ids), start) for slot, ids, start in calls])

    def save_snapshot(self, lane: int, snap: int) -> None:
        self.send(Command(Op.SAVE_SNAPSHOT, lane, snap))
        self.model.save_snapshot(lane, snap)

    def score(self, slot: int, ids: Sequence[int], continuation_start: int) -> Any:
        """Teacher-forced scoring (`/v1/score`, calibration only), broadcast like `prefill`:
        every rank must join this forward's collectives, or the ones that didn't hang on the
        next one. See `Model.score`."""
        self.send(Command(Op.SCORE, slot, continuation_start, tuple(ids)))
        return self.model.score(slot, list(ids), continuation_start)

    def load_snapshot(self, lane: int, snap: int) -> None:
        self.send(Command(Op.LOAD_SNAPSHOT, lane, snap))
        self.model.load_snapshot(lane, snap)

    def attach_blocks(self, lane: int, blocks: Sequence[int]) -> None:
        """Broadcast, unlike `lane_blocks`/`decode_row`/`cache_node_logits` below: a lane's
        block-table *contents* are a `SessionCache` decision made only on rank 0 (the scheduler
        never runs on the other ranks), so every rank has to be told the resolved ids, not just
        the call -- every full-attention layer reads `self.block_tables[lane]` for its physical
        row addresses, and a rank whose table disagreed would read or write the wrong rows of
        the (otherwise identically sharded) paged KV pool."""
        self.send(Command(Op.ATTACH_BLOCKS, lane, payload=tuple(blocks)))
        self.model.attach_blocks(lane, blocks)

    def copy_block(self, dst_block: int, src_block: int, filled: int) -> None:
        """Broadcast: this mutates every rank's own paged KV pool at the same physical rows,
        same reasoning as `attach_blocks`."""
        self.send(Command(Op.COPY_BLOCK, dst_block, src_block, payload=(filled,)))
        self.model.copy_block(dst_block, src_block, filled)

    def extend_blocks(self, grants: Sequence[tuple[int, int]]) -> None:
        """Broadcast, like `attach_blocks`: the scheduler reserved these ids from rank 0's
        allocator, the only one in use once serving starts, so every rank must append the
        same ids. Workers never allocate (see `pool_handshake`)."""
        self.send(Command(Op.EXTEND_BLOCKS, payload=tuple(x for pair in grants for x in pair)))
        self.model.extend_blocks(grants)

    def pool_handshake(self) -> None:
        """Broadcast, then run the boot self-check collective with every worker."""
        self.send(Command(Op.POOL_HANDSHAKE))
        pool_handshake(self._model())

    def _model(self) -> Any:
        """The `Model` under this rank's runner (a `GraphDecodeRunner` wraps one)."""
        return getattr(self.model, "model", self.model)

    def lane_blocks(self, lane: int) -> tuple[int, ...]:
        """Rank 0 only, like `sample_batch`: every rank's table already agrees (`begin`/
        `attach_blocks`/`extend_blocks`/`copy_block` are all broadcast), so there is nothing
        to tell the others."""
        return self.model.lane_blocks(lane)

    def lane_block_count(self, lane: int) -> int:
        """Rank 0 only, like `lane_blocks`."""
        return self.model.lane_block_count(lane)

    def decode_tokens_per_step(self) -> int:
        return self.model.decode_tokens_per_step()

    def decode_row(self, logits: Any, row: int) -> Any:
        """Rank 0 only: every rank computed the same batched `decode` result locally, so
        slicing a row out of it needs no collective, same as `sample_batch`."""
        return self.model.decode_row(logits, row)

    def decode(self, slots: Sequence[int], tokens: Sequence[int], positions: Sequence[int]) -> Any:
        self.send(Command.decode_step(slots, tokens, positions))
        return self.model.decode(list(slots), list(tokens), list(positions))

    def decode_mixed(
        self,
        slots: Sequence[int],
        tokens: Sequence[int],
        positions: Sequence[int],
        prefill_calls: Sequence[tuple[int, Sequence[int], int]],
    ) -> tuple[Any, list[Any]]:
        """`SEED_MIXED_BATCH`'s decode-plus-prefill step, one broadcast like `decode`."""
        self.send(Command.mixed_cmd(slots, tokens, positions, prefill_calls))
        return self.model.decode_mixed(
            list(slots),
            list(tokens),
            list(positions),
            [(slot, list(ids), start) for slot, ids, start in prefill_calls],
        )

    def decode_launch(
        self,
        slots: Sequence[int],
        tokens: Sequence[int],
        positions: Sequence[int],
        temperatures: Sequence[float],
    ) -> Any:
        """`SEED_OVERLAP_SCHED`'s decode step, broadcast once like `decode`. Temperatures stay
        on rank 0 (only rank 0 samples); the sampled ids reach the other ranks inside the step
        (`Model.launch_tail`), not in a later payload."""
        self.send(Command.decode_step(slots, tokens, positions, op=Op.DECODE_LAUNCH))
        return self.model.decode_launch(list(slots), list(tokens), list(positions), temperatures)

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
        forced: Sequence[Sequence[int]] | None = None,
    ) -> list[list[int]]:
        """One MTP round, broadcast once like `decode`. Every rank computes the same draft
        (greedy argmax, no sampling) and the same verify/accept/rollback off the same
        broadcast inputs, so -- unlike ordinary per-token sampling, which needs rank 0's drawn
        token relayed before the next step -- nothing mid-round needs its own broadcast. The
        lanes' KV blocks to `pos + k + 1` were reserved on rank 0 and sent as `EXTEND_BLOCKS`
        on this same ordered channel before this command (`Scheduler._decode_step`)."""
        n = len(slots)
        budgets = list(budgets) if budgets is not None else [NO_BUDGET] * n
        stops = [list(s) for s in stops] if stops is not None else [[] for _ in range(n)]
        self.send(Command.speculative_cmd(slots, tokens, positions, budgets, stops, forced))
        extra = {"forced": [list(f) for f in forced]} if forced and any(forced) else {}
        return self.model.speculative_decode(
            list(slots), list(tokens), list(positions), budgets, stops, **extra
        )

    def forced_drafts(self) -> bool:
        probe = getattr(self.model, "forced_drafts", None)
        return bool(probe()) if probe is not None else False

    def speculative_launch_ok(self, slots: Sequence[int], stops: Sequence[Sequence[int]]) -> bool:
        """Local read of the wrapped runner's verdict (it reads only broadcast inputs)."""
        ok = getattr(self.model, "speculative_launch_ok", None)
        return bool(ok is not None and ok(list(slots), stops))

    def speculative_launch(
        self,
        slots: Sequence[int],
        tokens: Sequence[int],
        positions: Sequence[int],
        budgets: Sequence[int],
        stops: Sequence[Sequence[int]],
        forced: Sequence[Sequence[int]] | None = None,
    ) -> Any:
        """`SEED_OVERLAP_SCHED` under MTP: one captured round launched without waiting
        (`mtp_overlap.OverlapMTPRunner.speculative_launch`), broadcast once like
        `speculative_decode`. Callers check `speculative_launch_ok` first, so every rank
        launches."""
        stops = [list(s) for s in stops]
        self.send(
            Command.speculative_cmd(
                slots, tokens, positions, budgets, stops, forced, op=Op.SPECULATIVE_LAUNCH
            )
        )
        extra = {"forced": [list(f) for f in forced]} if forced and any(forced) else {}
        return self.model.speculative_launch(
            list(slots), list(tokens), list(positions), list(budgets), stops, **extra
        )

    def warmup(self) -> None:
        self.send(Command(Op.WARMUP))
        self.model.warmup()

    def stop(self) -> None:
        """Release the workers from `serve_worker` so they can tear the group down cleanly.

        `server.py` never reaches a clean shutdown (the harness kills the process group, and
        `stop_workers` signals them), so this is for callers that do finish, which today is
        the multi-process check in `seed_tests/`.
        """
        self.send(Command(Op.STOP))

    def sample_batch(self, logits: Any, temperatures: Sequence[float]) -> list[int]:
        """Sampling is rank 0 only: it needs no collective and must not diverge."""
        return self.model.sample_batch(logits, list(temperatures))

    def cache_node_logits(self, snap: int, logits: Any) -> None:
        """Rank 0 only, like `sample_batch`: every rank already computed the same logits
        locally (the LM head is replicated), and this just remembers rank 0's own copy in
        its own static storage for a later `_prefill_step` fast path, which itself makes
        no model call at all and so needs nothing broadcast to the other ranks either."""
        self.model.cache_node_logits(snap, logits)

    def cached_node_logits(self, snap: int) -> Any:
        return self.model.cached_node_logits(snap)


def apply(model: Any, cmd: Command) -> None:
    """Make on this rank the call `cmd` names. Results are dropped: only rank 0 needs them."""
    if cmd.op is Op.BEGIN:
        model.begin(cmd.a)
    elif cmd.op is Op.PREFILL:
        model.prefill(cmd.a, list(cmd.payload), cmd.b)
    elif cmd.op is Op.PREFILL_BATCH:
        model.prefill_batch(cmd.unbatch_prefill())
    elif cmd.op is Op.SAVE_SNAPSHOT:
        model.save_snapshot(cmd.a, cmd.b)
    elif cmd.op is Op.LOAD_SNAPSHOT:
        model.load_snapshot(cmd.a, cmd.b)
    elif cmd.op is Op.ATTACH_BLOCKS:
        model.attach_blocks(cmd.a, list(cmd.payload))
    elif cmd.op is Op.COPY_BLOCK:
        model.copy_block(cmd.a, cmd.b, cmd.payload[0])
    elif cmd.op is Op.EXTEND_BLOCKS:
        p = cmd.payload
        model.extend_blocks([(p[i], p[i + 1]) for i in range(0, len(p), 2)])
    elif cmd.op is Op.POOL_HANDSHAKE:
        pool_handshake(getattr(model, "model", model))
    elif cmd.op is Op.DECODE:
        model.decode(*cmd.batch())
    elif cmd.op is Op.DECODE_MIXED:
        model.decode_mixed(*cmd.unbatch_mixed())
    elif cmd.op is Op.DECODE_LAUNCH:
        model.decode_launch(*cmd.batch(), None)
    elif cmd.op is Op.SPECULATIVE_DECODE:
        forced = cmd.unbatch_forced()
        extra = {"forced": forced} if forced else {}
        model.speculative_decode(*cmd.unbatch_speculative(), **extra)
    elif cmd.op is Op.SPECULATIVE_LAUNCH:
        forced = cmd.unbatch_forced()
        extra = {"forced": forced} if forced else {}
        model.speculative_launch(*cmd.unbatch_speculative(), **extra)
    elif cmd.op is Op.WARMUP:
        model.warmup()
    elif cmd.op is Op.SCORE:
        model.score(cmd.a, list(cmd.payload), cmd.b)
    else:
        raise ValueError(f"tp_driver: {cmd.op!r} is not a model call")


def serve_worker(model: Any, recv: Callable[[], Command]) -> None:
    """Ranks 1..: apply broadcast commands until STOP. Runs on the process's main thread."""
    while True:
        cmd = recv()
        if cmd.op is Op.STOP:
            return
        apply(model, cmd)

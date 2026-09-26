"""Overlap scheduling for captured MTP rounds (`SEED_OVERLAP_SCHED` with `SEED_MTP_SERVE`).

The scheduler launches round N+1 before it reads round N's committed tokens back
(`Scheduler._overlap_speculative_step`). A lane still in flight from round N is sent as
`LOOKAHEAD`: its base token, position and remaining budget exist only on the device, so each
round ends by writing them into per-lane tables (`LaneTables.update`), and the next round's
fill reads them there (`LaneTables.resolve`). A lane whose round N ended on a stop id or on its
budget is marked not `live`, and round N+1 runs it as an inactive row: the verify and draft
steps mask every persistent write by `active` (MTP_INTERFACE.md), so its state is exactly the
serial path's and the scheduler's `REFORWARD` turn-close publish stays correct. The host
discards that row's result (the flight is `done` by then).

Accept counts feed the lookahead only through these tables; the host learns them at readback
and advances the flight's position by the committed count.

TP: every rank runs the same fill, round and table update from the same broadcast command
(`tp_driver.Op.SPECULATIVE_LAUNCH`); `step_argmax`/`accept_len` are identical on every rank,
so the tables are too. Only rank 0 reads the round back.
"""

from __future__ import annotations

from collections.abc import Sequence

import block_pool
import graph_decode
import torch
from graph_mtp import GraphMTPRunner, MTPVerifyBuffers
from model import LOOKAHEAD_TOKEN, copy_from_host


class LaneTables:
    """Per-lane device state of the last round each lane ran (see the module docstring)."""

    def __init__(self, max_batch: int, device: torch.device) -> None:
        self.next_tok = torch.zeros(max_batch, dtype=torch.long, device=device)
        self.next_pos = torch.zeros(max_batch, dtype=torch.long, device=device)
        self.budget_left = torch.zeros(max_batch, dtype=torch.long, device=device)
        self.live = torch.zeros(max_batch, dtype=torch.bool, device=device)

    def resolve(self, buf: MTPVerifyBuffers, look: torch.Tensor) -> None:
        """Replace the lookahead rows' host placeholders with the tables' values, and drop
        rows whose lane finished in its previous round. `look` is `[capacity]` bool (False on
        padding rows)."""
        lanes = buf.slot_rows
        buf.token_matrix[:, 0].copy_(
            torch.where(look, self.next_tok[lanes], buf.token_matrix[:, 0])
        )
        buf.pos.copy_(torch.where(look, self.next_pos[lanes], buf.pos))
        buf.budget.copy_(torch.where(look, self.budget_left[lanes], buf.budget))
        buf.active.copy_(buf.active & (~look | self.live[lanes]))

    def update(self, buf: MTPVerifyBuffers) -> None:
        """Record each active row's round result for its lane; other lanes keep theirs."""
        lanes, active = buf.slot_rows, buf.active
        accept = buf.accept_len
        rows = torch.arange(buf.capacity, device=accept.device)
        last = buf.step_argmax[rows, accept]
        # A forced lane (`buf.forced_n`, folded chat suffix) whose round drained its suffix
        # generated one real token; the scheduler only sends it as lookahead in that case.
        forced = getattr(buf, "forced_n", None)
        used = accept + 1 if forced is None else torch.where(forced > 0, 1, accept + 1)
        left = buf.budget - used
        stopped = (buf.stops == last[:, None]).any(dim=1)
        for table, value in (
            (self.next_tok, last),
            (self.next_pos, buf.pos + accept + 1),
            (self.budget_left, left),
            (self.live, ~stopped & (left > 0)),
        ):
            table[lanes] = torch.where(active, value, table[lanes])


def rederive(runner: GraphMTPRunner, buf: MTPVerifyBuffers) -> None:
    """The device-side tail of `MTPVerifyRunner.fill`, rerun after `LaneTables.resolve`
    changed `pos` and `active`: block table (reserved for inactive rows), valid-block count,
    KV write rows, and the wide-verify row buffers."""
    mr, model = runner.mtp_runner, runner.model
    bs = model.block_size
    buf.block_table.copy_(
        torch.where(
            buf.active[:, None], mr.lane_table.table[buf.slot_rows], block_pool.RESERVED_BLOCK
        )
    )
    buf.block_valid.copy_(((buf.pos + mr.t + bs - 1) // bs).to(torch.int32))
    grid = buf.pos[:, None] + torch.arange(mr.t, device=buf.pos.device)[None, :]
    block_id = torch.gather(buf.block_table.long(), 1, (grid // bs).long())
    buf.write_rows.copy_(block_id * bs + grid % bs)
    if buf.capacity in mr.rowbufs:
        import graph_verify_wide  # noqa: PLC0415

        graph_verify_wide.fill_rows(model, buf, mr.rowbufs[buf.capacity], mr.t)


class PendingRound:
    """One launched round's committed tokens, read back asynchronously (pinned copy plus an
    event, like `model.PendingTokens`): `committed()` waits for this round only."""

    def __init__(self, buf: MTPVerifyBuffers, n: int) -> None:
        both = torch.cat([buf.step_argmax[:n], buf.accept_len[:n, None]], dim=1)
        self._event = None
        if both.is_cuda:
            self._host = torch.empty(both.shape, dtype=both.dtype, pin_memory=True)
            self._host.copy_(both, non_blocking=True)
            self._event = torch.cuda.Event()
            self._event.record(torch.cuda.current_stream(both.device))
        else:
            self._host = both.clone()

    def committed(self) -> list[list[int]]:
        if self._event is not None:
            self._event.synchronize()
        return [row[: row[-1] + 1] for row in self._host.tolist()]


class OverlapMTPRunner(GraphMTPRunner):
    """`GraphMTPRunner` plus `speculative_launch` (`server.build_runner` picks it whenever
    the model has an MTP head; the scheduler uses `speculative_launch` only under
    `SEED_OVERLAP_SCHED`)."""

    def __init__(self, model, backend=None) -> None:  # noqa: ANN001
        super().__init__(model, backend)
        self.lane_state = LaneTables(model.max_batch, model.devices[-1])

    def speculative_decode(self, *args, **kwargs) -> list[list[int]]:  # noqa: ANN002, ANN003
        """`GraphMTPRunner.speculative_decode` with `SEED_STEP_TIMING`'s device gap meter
        around the round, so serial and overlapped rounds report the same host gap."""
        meter = self.model.gap_meter
        if meter is not None:
            meter.step_start()
        out = super().speculative_decode(*args, **kwargs)
        if meter is not None:
            meter.step_end()
        return out

    def speculative_launch_ok(self, slots: Sequence[int], stops: Sequence[Sequence[int]]) -> bool:
        """Whether a captured round serves these lanes (reads only broadcast inputs)."""
        return self.mtp_runner.replayable(list(slots), stops)

    def speculative_launch(
        self,
        slots: Sequence[int],
        tokens: Sequence[int],
        positions: Sequence[int],
        budgets: Sequence[int],
        stops: Sequence[Sequence[int]],
        forced: Sequence[Sequence[int]] | None = None,
    ) -> PendingRound | None:
        """One captured MTP round, not waited for. A `tokens` entry `LOOKAHEAD_TOKEN` means the
        lane's previous round is still in flight: `positions`/`budgets` then hold that
        round's launch values (only used for the capacity check), and the device tables
        supply the real ones. None (nothing launched, on every rank) when no captured bucket
        serves the lanes. `forced`: per-lane forced drafts (`SEED_MTP_FORCED_DRAFTS`), never
        on a lookahead row. Returns rank 0's handle; other ranks get None too."""
        slots, tokens, positions = list(slots), list(tokens), list(positions)
        mr = self.mtp_runner
        if not self.speculative_launch_ok(slots, stops):
            return None
        self.decode_path = "mtp-graph-overlap"
        graph = mr.graphs[graph_decode.bucket_for(len(slots), mr.buckets)]
        buf = graph.buf
        look_host = [t == LOOKAHEAD_TOKEN for t in tokens]
        base = [0 if look else t for t, look in zip(tokens, look_host, strict=True)]
        meter = self.model.gap_meter
        if meter is not None:
            meter.step_start()
        mr.fill(buf, slots, base, None, positions, budgets, stops, forced)
        if any(look_host):
            look = torch.zeros(buf.capacity, dtype=torch.bool, device=buf.pos.device)
            copy_from_host(look[: len(slots)], torch.tensor(look_host, dtype=torch.bool))
            self.lane_state.resolve(buf, look)
            rederive(self, buf)
        graph.draft()
        graph.replay()
        self.lane_state.update(buf)
        if meter is not None:
            meter.step_end()
        return PendingRound(buf, len(slots)) if self.model.tp.plan.rank == 0 else None

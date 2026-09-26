"""Opt-in per-step timing log (`SEED_STEP_TIMING=1`), a diagnostic, off by default.

Torch-free so `scheduler.py` can import it. Every `SEED_STEP_TIMING_EVERY` (default 20)
steps of a given kind, one `[step-timing]` line is printed:

- scheduler (rank 0): step kind (prefill/decode), batch size, tokens, wall ms of the last
  step and the mean over the window, and which decode path ran (`graph`/`eager`). Wall time
  is accurate without an extra sync because both step kinds end in `sample_batch`, which
  reads the logits back to the host.
- model/graph runner (every rank): device time of the paged decode attention kernel summed
  over the full-attention layers, the host time spent building its block-table inputs, and
  the replay wall time on the captured path. Those lines sync the device, but only on the
  sampled step.
- `GapMeter` (every rank, CUDA only): device idle time between consecutive decode steps, the
  host gap `SEED_OVERLAP_SCHED` removes. No sync; see the class docstring.
"""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Callable
from typing import Any

ENABLED = os.environ.get("SEED_STEP_TIMING", "0") not in ("0", "", "false", "False")
EVERY = max(1, int(os.environ.get("SEED_STEP_TIMING_EVERY", "20")))


def log(message: str) -> None:
    print(f"[step-timing] {message}", flush=True)


class Window:
    """Rolling per-kind step statistics between two log lines."""

    def __init__(self) -> None:
        self.count: dict[str, int] = {}
        self.total_ms: dict[str, float] = {}
        self.total_tokens: dict[str, int] = {}

    def add(self, kind: str, ms: float, tokens: int) -> bool:
        """Record one step; True when this step should be logged (every `EVERY`th of `kind`)."""
        self.count[kind] = self.count.get(kind, 0) + 1
        self.total_ms[kind] = self.total_ms.get(kind, 0.0) + ms
        self.total_tokens[kind] = self.total_tokens.get(kind, 0) + tokens
        return self.count[kind] % EVERY == 0

    def drain(self, kind: str) -> tuple[float, float]:
        """(mean ms per step, tokens per second) over the window, then reset `kind`'s window."""
        n = EVERY
        ms, tok = self.total_ms[kind], self.total_tokens[kind]
        self.total_ms[kind], self.total_tokens[kind] = 0.0, 0
        return ms / n, (tok / (ms / 1e3) if ms else 0.0)


class GapMeter:
    """Device idle time between consecutive decode steps (`SEED_STEP_TIMING` only).

    The host gap is what `SEED_OVERLAP_SCHED` exists to remove: the time the device sits
    idle between the last kernel of decode step N and the first kernel of step N+1 while the
    host reads back tokens, runs the scheduler and builds step N+1's inputs. It is measured
    on the device, with a timing event recorded on the stream at the start and end of every
    decode step, so it does not depend on where the host happens to block.

    Nothing here synchronizes: `end` only folds in pairs whose events have both completed
    (`query()`), so the meter adds no bubble of its own. A `prefill` between two decode steps
    `interrupt`s the pairing, since that gap is device-busy, not idle. Every `EVERY`
    completed gaps one line is logged with the mean and median in ms.

    Torch-free on purpose (like the rest of this module): the caller passes a factory for
    `torch.cuda.Event(enable_timing=True)`.
    """

    def __init__(self, new_event: Callable[[], Any], label: str) -> None:
        self._new_event = new_event
        self._label = label
        self._prev_end: Any = None
        self._pending: deque[tuple[Any, Any]] = deque(maxlen=4 * EVERY)
        self._window: list[float] = []

    def step_start(self) -> None:
        ev = self._new_event()
        ev.record()
        if self._prev_end is not None:
            self._pending.append((self._prev_end, ev))
        self._prev_end = None

    def step_end(self) -> None:
        ev = self._new_event()
        ev.record()
        self._prev_end = ev
        while self._pending and self._pending[0][1].query():
            a, b = self._pending.popleft()
            self._window.append(a.elapsed_time(b))
        if len(self._window) >= EVERY:
            gaps = sorted(self._window)
            self._window.clear()
            log(
                f"{self._label} host_gap_ms mean={sum(gaps) / len(gaps):.3f} "
                f"p50={gaps[len(gaps) // 2]:.3f} max={gaps[-1]:.3f} over {len(gaps)} gaps"
            )

    def interrupt(self) -> None:
        self._prev_end = None


SERVICE_TRACE = os.environ.get("SEED_SERVICE_TRACE", "")
"""`SEED_SERVICE_TRACE=<path>`: append one JSON line per scheduler step to `<path>` (rank 0),
off by default. Each line carries the step kind, its rows/tokens/path, host timestamps, and
device timestamps of four markers (step start, forward start, forward end, step end) on the
scheduler's stream. See `ServiceTrace`."""


class ServiceTrace:
    """Per-step device timeline of the scheduler's stream (`SEED_SERVICE_TRACE` only).

    Every step records up to four timing events on the current stream: `step` (top of
    `Scheduler.step`), `fwd0`/`fwd1` (around the model call) and `end`. Device times are
    `elapsed_time` from one base event, so consecutive steps give device busy time per kind
    and the device idle gap between steps (host scheduling, sampling readback). Nothing here
    synchronizes: a step is written out only once its `end` event has completed (`query()`),
    so the trace adds no bubble beyond the event records themselves.

    Torch-free: the caller passes a factory for `torch.cuda.Event(enable_timing=True)`.
    """

    def __init__(self, new_event: Callable[[], Any], path: str) -> None:
        import json
        import time

        self._json, self._time = json, time
        self._new_event = new_event
        self._file = open(path, "a", buffering=1)  # noqa: SIM115 -- line-buffered, lives for the process
        self._base: Any = None
        self._cur: dict[str, Any] | None = None
        self._pending: deque[dict[str, Any]] = deque()
        # `<path>.pause` present: record nothing (checked every 32 steps), so one server can
        # serve a traced and an untraced benchmark run back to back.
        self._pause_path = path + ".pause"
        self._paused = os.path.exists(self._pause_path)
        self._steps = 0

    def _record(self) -> Any:
        ev = self._new_event()
        ev.record()
        return ev

    def begin(self) -> None:
        self._steps += 1
        if self._steps % 32 == 0:
            self._paused = os.path.exists(self._pause_path)
            self._drain()
        if self._paused:
            self._cur = None
            return
        if self._base is None:
            self._base = self._record()
            self._file.write(
                self._json.dumps({"base_wall": self._time.time(), "base_perf": self._time.perf_counter()})
                + "\n"
            )
        self._cur = {"ev": {"step": self._record()}, "h0": self._time.perf_counter()}

    def mark(self, name: str) -> None:
        if self._cur is not None and name not in self._cur["ev"]:
            self._cur["ev"][name] = self._record()
            self._cur["h_" + name] = self._time.perf_counter()

    def end(self, kind: str, **meta: Any) -> None:
        cur, self._cur = self._cur, None
        if cur is None:
            return
        cur["ev"]["end"] = self._record()
        cur["h1"] = self._time.perf_counter()
        cur["kind"] = kind
        cur.update(meta)
        self._pending.append(cur)
        self._drain()

    def drop(self) -> None:
        self._cur = None

    def _drain(self) -> None:
        while self._pending and self._pending[0]["ev"]["end"].query():
            cur = self._pending.popleft()
            events = cur.pop("ev")
            cur["d"] = {name: self._base.elapsed_time(ev) for name, ev in events.items()}
            self._file.write(self._json.dumps(cur, separators=(",", ":")) + "\n")

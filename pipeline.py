"""Stage-parallel dataflow pipeline over the model's layer-to-device split.

`Model` gives each device a contiguous run of layers, so a step that walks the layers in
index order leaves three of four devices idle at every instant. This module runs the same
stages as a pipeline instead: the step's work is cut into microbatches, and while stage s
computes microbatch m, stage s-1 is already computing microbatch m+1. With S stages and M
microbatches each stage has work in M of the M+S-1 waves, so the fill and drain bubbles
cost (S-1)/(M+S-1) of the wall time and everything else overlaps.

One OS thread per stage is what makes the overlap real, rather than a single driver thread
issuing all stages in wave order. A stage body blocks its thread whenever it reads a
device-resident value back to the host, which `Model.moe` does once per layer to size its
per-expert groups, and on one thread every later stage would queue behind that read. With a
thread per stage the read blocks only its own device's stage.

The module deliberately knows nothing about tensors or devices: a stage is a callable, an
item is whatever the caller threads through them. `seed_tests/test_pipeline.py` drives it
with plain Python stages that record their own interleaving.

Ordering and isolation. Each stage is one thread consuming one FIFO queue, so a stage sees
microbatches in submission order and `run` returns results in that order. Two microbatches
are never inside the same stage at once, so per-stage state (a layer's KV pool) is touched
by exactly one thread, and microbatches at different stages touch disjoint per-slot state.

`run` is not reentrant: one caller thread (the scheduler's) drives the pipeline.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

Stage = Callable[[Any], Any]


@dataclass(frozen=True, slots=True)
class _Work:
    index: int
    payload: Any


@dataclass(frozen=True, slots=True)
class _Failed:
    index: int
    error: BaseException


_SHUTDOWN = object()


class StagePipeline:
    """Runs `items` through `stages` in order, with every stage busy on a different item."""

    def __init__(self, stages: Sequence[Stage], name: str = "stage") -> None:
        if not stages:
            raise ValueError("StagePipeline needs at least one stage")
        self._stages = list(stages)
        self._inboxes: list[queue.Queue] = [queue.Queue() for _ in self._stages]
        self._out: queue.Queue = queue.Queue()
        self._closed = False
        self._threads = [
            threading.Thread(target=self._serve, args=(s,), name=f"{name}-{s}", daemon=True)
            for s in range(len(self._stages))
        ]
        for thread in self._threads:
            thread.start()

    def __enter__(self) -> StagePipeline:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def depth(self) -> int:
        return len(self._stages)

    def run(self, items: Sequence[Any]) -> list[Any]:
        """Push every item through every stage and return the results in item order.

        Raises the first stage exception, but only after every item has come back, so a
        failed call leaves no work in flight and the pipeline stays usable.
        """
        if self._closed:
            raise RuntimeError("StagePipeline is closed")
        for index, payload in enumerate(items):
            self._inboxes[0].put(_Work(index, payload))
        results: list[Any] = [None] * len(items)
        failure: BaseException | None = None
        for _ in range(len(items)):
            done = self._out.get()
            if isinstance(done, _Failed):
                failure = done.error if failure is None else failure
            else:
                results[done.index] = done.payload
        if failure is not None:
            raise failure
        return results

    def close(self) -> None:
        """Stop the stage threads. Idempotent; call only when no `run` is in flight."""
        if self._closed:
            return
        self._closed = True
        for inbox in self._inboxes:
            inbox.put(_SHUTDOWN)
        for thread in self._threads:
            thread.join(timeout=5.0)

    def _serve(self, index: int) -> None:
        """One stage's thread: apply the stage to each item and hand it to the next stage."""
        stage = self._stages[index]
        inbox = self._inboxes[index]
        outbox = self._inboxes[index + 1] if index + 1 < len(self._stages) else self._out
        while True:
            item = inbox.get()
            if item is _SHUTDOWN:
                return
            if isinstance(item, _Failed):  # an earlier stage failed; carry it to the caller
                outbox.put(item)
                continue
            try:
                outbox.put(_Work(item.index, stage(item.payload)))
            except BaseException as exc:  # noqa: BLE001 -- re-raised by `run`, on its thread
                # Nothing may escape and end this thread: `run` waits for one result per item,
                # so a dropped item would hang the scheduler thread instead of failing it.
                outbox.put(_Failed(item.index, exc))

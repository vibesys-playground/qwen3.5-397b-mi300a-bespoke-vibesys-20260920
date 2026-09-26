"""One rank of the gloo-backed `SEED_OVERLAP_SCHED` check. Started by test_overlap_sched.py.

Not a test: like `tp_gloo_rank.py`, the program a test starts once per rank, through the
production entry points (`tp.init`, `model.Model`, `tp_driver.command_channel`, `Broadcaster`,
`serve_worker`), but rank 0 drives a real `Scheduler` rather than a hand-written greedy loop,
so `Op.DECODE_LAUNCH`, the device-side token broadcast in `Model.launch_tail`, the gloo
command group and every publish/release call the scheduler makes all cross the process
boundary in the order the server issues them.

    python overlap_gloo_rank.py --checkpoint DIR --rank R --world 4 --port P \
        --specs FILE --overlap 1 [--mixed 1] [--prefill-graphs 1] [--mixed-graph 1] [--defer 1] \
        [--out FILE]

`--prefill-graphs 1` wraps every rank's model in a `GraphDecodeRunner` (eager stand-in
capture backend, as in test_graph_capture.py) with `SEED_PREFILL_GRAPHS` on, prepared on every
rank before serving, the way `server.build_runner` does. `--mixed-graph 1` does the same with
`SEED_MIXED_GRAPH` (shapes from `SEED_MIXED_GRAPH_DECODE`/`_PREFILL`) and turns the scheduler's
mixed-graph policy on. `--defer 1` sets the scheduler's `SEED_DEFER_TURN_CLOSE`.

`--specs` is a JSON list of [prompt, max_new, stop_ids, arrival_step]. Rank 0 writes
[[tokens, end], ...] to `--out`.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tp  # noqa: E402
from scheduler import Request, Scheduler  # noqa: E402
from session_cache import SessionCache  # noqa: E402
from tp_driver import Broadcaster, command_channel, serve_worker  # noqa: E402
from tp_gloo_rank import build  # noqa: E402


class Sink:
    def __init__(self) -> None:
        self.tokens: list[int] = []
        self.end = None

    def __call__(self, event: tuple) -> None:
        kind, val = event
        if kind == "tok":
            self.tokens.append(val)
        else:
            self.end = [kind, val if kind == "error" else list(val)]


def drive(
    runner: Broadcaster,
    model,  # noqa: ANN001
    specs: list,
    overlap: bool,
    mixed: bool = False,
    defer: bool = False,
    mixed_graph: bool = False,
) -> list:
    cache = SessionCache(model.block_allocator, model.block_size, model.num_snapshots)
    sched = Scheduler(
        runner,
        cache,
        prefill_chunk=4,
        overlap=overlap,
        mixed_batch=mixed,
        defer_turn_close=defer,
        mixed_graph=mixed_graph,
    )
    sinks = [Sink() for _ in specs]
    pending = sorted(range(len(specs)), key=lambda i: specs[i][3])
    step = 0
    while True:
        while pending and specs[pending[0]][3] <= step:
            i = pending.pop(0)
            prompt, max_new, stop, _ = specs[i]
            sched.submit(Request(list(prompt), max_new, 0.0, frozenset(stop), sinks[i]))
        if not sched.step() and not pending:
            return [[s.tokens, s.end] for s in sinks]
        step += 1


def main(argv: list[str] | None = None) -> None:
    import torch.distributed as dist

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--world", type=int, required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--specs", required=True)
    p.add_argument("--overlap", type=int, required=True)
    p.add_argument("--mixed", type=int, default=0)
    p.add_argument("--prefill-graphs", type=int, default=0)
    p.add_argument("--defer", type=int, default=0)
    p.add_argument("--mixed-graph", type=int, default=0)
    p.add_argument("--out", default="")
    a = p.parse_args(argv)

    reduce = tp.init(a.rank, a.world, port=a.port, backend="gloo")
    model = build(a.checkpoint, a.rank, a.world, reduce)
    runner = model
    if a.prefill_graphs or a.mixed_graph:
        from graph_decode import GraphDecodeRunner
        from test_graph_capture import EagerBackend

        runner = GraphDecodeRunner(
            model,
            backend=EagerBackend(),
            prefill_graphs=bool(a.prefill_graphs),
            mixed_graphs=bool(a.mixed_graph),
        )
        runner.prepare()
        if a.prefill_graphs:
            assert runner.prefill_runner is not None and runner.prefill_runner.enabled
        if a.mixed_graph:
            assert runner.mixed_runner is not None and runner.mixed_runner.enabled
    channel = command_channel(torch.device("cpu"), bool(a.overlap))
    try:
        if a.rank:
            serve_worker(runner, channel.recv)
        else:
            driver = Broadcaster(runner, channel.send)
            # As `server.py` does: from here on rank 0's scheduler is the only block allocator
            # and every lane's growth reaches the workers as `EXTEND_BLOCKS`.
            driver.pool_handshake()
            specs = json.loads(Path(a.specs).read_text())
            result = drive(
                driver,
                model,
                specs,
                bool(a.overlap),
                bool(a.mixed),
                bool(a.defer),
                bool(a.mixed_graph),
            )
            if a.prefill_graphs:
                result.append(runner.prefill_runner.replays)
            if a.mixed_graph:
                result.append(runner.mixed_runner.replays)
            driver.stop()
            Path(a.out).write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

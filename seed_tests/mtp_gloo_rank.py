"""One rank of the gloo-backed `SEED_MTP_SERVE` check. Started by test_mtp_serve_tp.py.

Not a test: like `overlap_gloo_rank.py`, the program a test starts once per rank, through the
production entry points (`tp.init`, `model.Model`, `tp_driver.command_channel`, `Broadcaster`,
`serve_worker`), with rank 0 driving a real `Scheduler` with MTP rounds on, so every
`EXTEND_BLOCKS` reservation and `SPECULATIVE_DECODE` round crosses the process boundary in the
order the server issues them.

    python mtp_gloo_rank.py --checkpoint DIR --rank R --world 4 --port P --k K \
        --specs FILE --oracle FILE --graph 0|1 --trace FILE [--out FILE]

`--specs` is a JSON list of [prompt, max_new, stop_ids, arrival_step]. `--oracle` is a JSON
list of [pos, token, drafts]: the drafter every rank uses instead of the random tiny MTP head
(a pure function of the broadcast `(token, position)`, so all ranks draft identically).
Every rank writes its applied op sequence and its final block tables to `--trace`; rank 0
writes [[tokens, end], ...] to `--out`.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mtp  # noqa: E402
import tp  # noqa: E402
import tp_driver  # noqa: E402
from model import Model, load_cfg  # noqa: E402
from overlap_gloo_rank import Sink  # noqa: E402
from scheduler import Request, Scheduler  # noqa: E402
from session_cache import SessionCache  # noqa: E402
from tp_driver import Broadcaster, Command, command_channel, serve_worker  # noqa: E402
from tp_gloo_rank import MAX_SEQ  # noqa: E402


class EagerBackend:
    def capture(self, step, device):  # noqa: ANN001, ANN201
        return step


def oracle_drafts(table: dict, k: int, tokens: list[int], positions: list[int]) -> list[list[int]]:
    return [list(table.get((p, t), [0] * k)) for t, p in zip(tokens, positions, strict=True)]


def main(argv: list[str] | None = None) -> None:
    import torch.distributed as dist

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--world", type=int, required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--specs", required=True)
    p.add_argument("--oracle", required=True)
    p.add_argument("--graph", type=int, default=0)
    p.add_argument("--overlap", type=int, default=0)
    p.add_argument("--trace", required=True)
    p.add_argument("--out", default="")
    a = p.parse_args(argv)

    table = {(pos, tok): d for pos, tok, d in json.loads(Path(a.oracle).read_text())}
    mtp.MTP_ENABLED, mtp.MTP_K = True, a.k
    mtp.draft = lambda model, m, hidden, tokens, slots, positions: oracle_drafts(  # noqa: ARG005
        table, a.k, list(tokens), list(positions)
    )

    reduce = tp.init(a.rank, a.world, port=a.port, backend="gloo")
    plan = tp.plan(load_cfg(a.checkpoint), a.rank, a.world)
    model = Model(
        a.checkpoint, ["cpu"], torch.float32, MAX_SEQ, max_batch=2, tp=tp.TP(plan, "cpu", reduce)
    )
    runner = model
    if a.graph:
        import graph_mtp

        if a.overlap:
            import mtp_overlap

            runner = mtp_overlap.OverlapMTPRunner(model, backend=EagerBackend())
        else:
            runner = graph_mtp.GraphMTPRunner(model, backend=EagerBackend())
        runner.prepare()
        assert runner.mtp_runner.enabled, "MTP capture did not enable"
        for graph in runner.mtp_runner.graphs.values():
            buf = graph.buf

            def draft(buf: graph_mtp.MTPVerifyBuffers = buf) -> None:
                drafts = oracle_drafts(
                    table, a.k, buf.token_matrix[:, 0].tolist(), buf.pos.tolist()
                )
                buf.token_matrix[:, 1:].copy_(torch.tensor(drafts))

            graph.draft = draft

    ops: list[int] = []
    channel = command_channel(torch.device("cpu"), False)
    try:
        if a.rank:
            real_apply = tp_driver.apply

            def recording_apply(model_: object, cmd: Command) -> None:
                ops.append(int(cmd.op))
                real_apply(model_, cmd)

            tp_driver.apply = recording_apply  # `serve_worker` looks `apply` up per call
            serve_worker(runner, channel.recv)
        else:

            def send(cmd: Command) -> None:
                if cmd.op is not tp_driver.Op.STOP:
                    ops.append(int(cmd.op))
                channel.send(cmd)

            driver = Broadcaster(runner, send)
            driver.pool_handshake()
            specs = json.loads(Path(a.specs).read_text())
            cache = SessionCache(model.block_allocator, model.block_size, model.num_snapshots)
            sched = Scheduler(
                driver, cache, prefill_chunk=4, spec_decode=True, overlap=bool(a.overlap)
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
                    break
                step += 1
            driver.stop()
            Path(a.out).write_text(json.dumps([[s.tokens, s.end] for s in sinks]))
        trace = {
            "ops": ops,
            "tables": [list(t.blocks) for t in model.block_tables],
            "local_free": model.block_allocator.free_count,
            "usable": model.block_allocator.usable_blocks,
        }
        Path(a.trace).write_text(json.dumps(trace))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

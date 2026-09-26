"""One rank of the gloo-backed scheduler check. Started by test_tensor_parallel.py.

Like `tp_gloo_rank.py`, but rank 0 drives the real `Scheduler` + `SessionCache` through
`Broadcaster`, over a workload that forces copy-on-write, eviction (a 2-slot snapshot pool
and an 8-block KV pool) and reuse of freed blocks. Every rank records its block tables at
each command boundary and, at the end, its paged K/V pool; the test checks that all ranks
agree. That is the property rank 0's single block-allocator authority exists for: a rank
that allocated on its own would attach different ids after the first COW or eviction.

    python tp_gloo_sched_rank.py --checkpoint DIR --rank R --world 4 --port P --out-dir DIR
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tp  # noqa: E402
from model import Model, load_cfg  # noqa: E402
from scheduler import Request, Scheduler  # noqa: E402
from session_cache import SessionCache  # noqa: E402
from tp_driver import Broadcaster, Channel, Op, apply  # noqa: E402

MAX_SEQ = 64
MAX_BATCH = 2


def build(checkpoint: str, rank: int, world: int, reduce: tp.Reducer | None) -> Model:
    plan = tp.plan(load_cfg(checkpoint), rank, world)
    handle = tp.TP(plan, "cpu", reduce)
    return Model(checkpoint, ["cpu"], torch.float32, MAX_SEQ, max_batch=MAX_BATCH, tp=handle)


def tables(model: Model) -> list[list[int]]:
    return [list(t.blocks) for t in model.block_tables]


def run_workload(sched: Scheduler, vocab: int) -> list[list[int]]:
    """Greedy requests in rounds; returns every request's generated tokens in order.

    Round structure (block size 16, 8-block pool, 2 snapshot slots):
    - session A turn 1 (11 tokens) publishes nodes at 11 and 15 (turn-close);
    - A turn 2 resumes the depth-15 node: copy-on-write of its partial block;
    - sessions B..E run two at a time, evicting A's nodes and reusing their blocks;
    - A turn 3 and a long request finish the run, allocating from recycled blocks.
    """
    gen = torch.Generator().manual_seed(7)

    def fresh(n: int) -> list[int]:
        return torch.randint(2, vocab, (n,), generator=gen).tolist()

    outputs: list[list[int]] = []

    def run(prompts: list[tuple[list[int], int]]) -> list[list[int]]:
        sinks = []
        for prompt, max_new in prompts:
            got: list[int] = []
            err: list[Any] = []

            def emit(event: tuple, got: list[int] = got, err: list[Any] = err) -> None:
                if event[0] == "tok":
                    got.append(event[1])
                elif event[0] == "error":
                    err.append(event[1])

            sched.submit(Request(list(prompt), max_new, 0.0, frozenset(), emit))
            sinks.append((got, err))
        for _ in range(10_000):
            if not sched.step():
                break
        for got, err in sinks:
            assert not err, err
            outputs.append(got)
        return [got for got, _ in sinks]

    a1 = fresh(11)
    (r1,) = run([(a1, 4)])
    a2 = [*a1, *r1, *fresh(3)]
    (r2,) = run([(a2, 4)])
    run([(fresh(20), 5), (fresh(13), 6)])
    run([(fresh(9), 7), (fresh(17), 3)])
    a3 = [*a2, *r2, *fresh(2)]
    run([(a3, 5), (fresh(30), 20)])
    return outputs


def main(argv: list[str] | None = None) -> None:
    import torch.distributed as dist

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--world", type=int, required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args(argv)

    reduce = tp.init(a.rank, a.world, port=a.port, backend="gloo")
    model = build(a.checkpoint, a.rank, a.world, reduce)
    channel = Channel(torch.device("cpu"))
    log: list[list[list[int]]] = []  # this rank's tables at every command boundary
    out = Path(a.out_dir)
    try:
        if a.rank:
            while True:
                cmd = channel.recv()
                log.append(tables(model))
                if cmd.op is Op.STOP:
                    break
                apply(model, cmd)
        else:

            def send(cmd: Any) -> None:
                log.append(tables(model))
                channel.send(cmd)

            driver = Broadcaster(model, send)
            driver.warmup()
            driver.pool_handshake()
            cache = SessionCache(model.block_allocator, model.block_size, model.num_snapshots)
            outputs = run_workload(Scheduler(driver, cache), model.cfg.vocab)
            live = [
                b for b in range(1, model.num_kv_blocks) if model.block_allocator.refcount(b) > 0
            ]
            driver.stop()
            (out / "result.json").write_text(json.dumps({"outputs": outputs, "live": live}))
        (out / f"tables{a.rank}.json").write_text(json.dumps(log))
        pools = {
            i: (layer["k"].clone(), layer["v"].clone())
            for i, layer in enumerate(model.pool)
            if model.cfg.layer_types[i] == "full_attention"
        }
        torch.save({"pools": pools, "kv_start": model.tp.plan.kv.start}, out / f"pool{a.rank}.pt")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""Paired captured ordinary-decode A/B for ``SEED_AR_SP`` on TP=4.

The process loads and tunes one model, captures only B48/B80/B96 first with SP disabled and
then enabled, checks exact logits and representative state on every rank, and times alternating
replay rounds. Run with the production flags plus ``SEED_AR_SP=1`` so SP IPC storage is
allocated at model construction.

    srun ... env $FLAGS SEED_AR_SP=1 python3 -u ab_graph_decode_sp.py --tp 4
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import allreduce_custom
import torch
from graph_decode import GraphDecodeRunner

import server


def _reset_and_fill(runner: GraphDecodeRunner, graphs, capacity: int) -> None:  # noqa: ANN001
    runner.reset_slots()
    slots = list(range(capacity))
    tokens = [1000 + i for i in slots]
    runner.fill(graphs.buffers, capacity, slots, tokens, [0] * capacity)


def _state_snapshot(model, buf, capacity: int) -> list[torch.Tensor]:  # noqa: ANN001
    """Small exact sample spanning every layer's mutable state kind."""
    lanes = torch.tensor(sorted({0, capacity // 2, capacity - 1}), device=buf.x_in.device)
    write_rows = buf.write_rows[lanes].long()
    out = []
    for pool in model.pool:
        for name, value in sorted(pool.items()):
            if name in ("conv", "rec"):
                out.append(value[lanes].cpu())
            elif name in ("k", "v"):
                out.append(value[write_rows].cpu())
    return out


def _run_once(runner: GraphDecodeRunner, graphs, capacity: int):  # noqa: ANN001
    _reset_and_fill(runner, graphs, capacity)
    runner.run(graphs.buffers, graphs.replays)
    torch.cuda.synchronize()
    return runner.gather(graphs.buffers, capacity).cpu(), _state_snapshot(
        runner.model, graphs.buffers[0], capacity
    )


def run_rank(a: argparse.Namespace) -> None:
    assert allreduce_custom.SP, "run with SEED_AR_SP=1"
    args = argparse.Namespace(
        model_path=a.model_path,
        dtype="bfloat16",
        tp=a.tp,
        tp_port=a.port,
        max_seq_len=a.max_seq,
        max_batch=max(a.batches),
        devices="",
        rank=a.rank,
    )
    model = server.build_model(args, a.rank)
    runner = GraphDecodeRunner(model, prefill_graphs=False, mixed_graphs=False)
    runner.buckets = list(a.batches)

    import graph_decode

    graph_decode.AR_SP_DECODE = False
    assert runner.prepare(), "baseline capture/validation failed"
    graphs = {"base": dict(runner.graphs)}
    graph_decode.AR_SP_DECODE = True
    assert runner.prepare(), "SP capture/validation failed"
    graphs["sp"] = dict(runner.graphs)

    parity = {}
    for capacity in a.batches:
        base_logits, base_state = _run_once(runner, graphs["base"][capacity], capacity)
        sp_logits, sp_state = _run_once(runner, graphs["sp"][capacity], capacity)
        logits_exact = torch.equal(base_logits, sp_logits)
        state_exact = len(base_state) == len(sp_state) and all(
            torch.equal(x, y) for x, y in zip(base_state, sp_state, strict=True)
        )
        argmax_exact = torch.equal(base_logits.argmax(-1), sp_logits.argmax(-1))
        maxdiff = float((base_logits - sp_logits).abs().max())
        parity[capacity] = (logits_exact, state_exact, argmax_exact, maxdiff)
        assert logits_exact and state_exact and argmax_exact, parity[capacity]

    samples = {(v, b, k): [] for v in graphs for b in a.batches for k in ("loop", "step")}
    for _ in range(a.rounds):
        for variant in ("base", "sp"):
            for capacity in a.batches:
                graph = graphs[variant][capacity]
                _reset_and_fill(runner, graph, capacity)
                runner.run(graph.buffers, graph.replays)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(a.reps):
                    runner.run(graph.buffers, graph.replays)
                torch.cuda.synchronize()
                samples[(variant, capacity, "loop")].append(
                    (time.perf_counter() - t0) * 1e3 / a.reps
                )
                step = []
                for _ in range(a.reps):
                    t0 = time.perf_counter()
                    runner.fill(
                        graph.buffers,
                        capacity,
                        list(range(capacity)),
                        [1000 + i for i in range(capacity)],
                        [0] * capacity,
                    )
                    runner.run(graph.buffers, graph.replays)
                    torch.cuda.synchronize()
                    step.append((time.perf_counter() - t0) * 1e3)
                samples[(variant, capacity, "step")].append(statistics.median(step))

    if a.rank == 0:
        for capacity in a.batches:
            exact = parity[capacity]
            parts = []
            for kind in ("loop", "step"):
                base = statistics.median(samples[("base", capacity, kind)])
                sp = statistics.median(samples[("sp", capacity, kind)])
                parts.append(f"{kind} base {base:.2f} sp {sp:.2f} delta {sp - base:+.2f} ms")
            print(
                f"AB B{capacity}: logits_exact={exact[0]} state_exact={exact[1]} "
                f"argmax_exact={exact[2]} maxdiff={exact[3]:.3g}; " + "; ".join(parts),
                flush=True,
            )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--port", type=int, default=29643)
    p.add_argument("--batches", type=lambda s: tuple(map(int, s.split(","))), default=(48, 80, 96))
    p.add_argument("--max-seq", type=int, default=4096)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument(
        "--model-path",
        default=os.environ.get(
            "MODEL_PATH", "/path/to/scratch/models/Qwen3.5-397B-A17B-MXFP4"
        ),
    )
    a = p.parse_args()
    if a.rank:
        run_rank(a)
        return
    argv = [sys.executable, "-u", os.path.abspath(__file__)] + sys.argv[1:]
    workers = [
        subprocess.Popen([*argv, "--rank", str(rank)], env=os.environ.copy())
        for rank in range(1, a.tp)
    ]
    try:
        run_rank(a)
    finally:
        for proc in workers:
            proc.wait(timeout=1200)


if __name__ == "__main__":
    main()

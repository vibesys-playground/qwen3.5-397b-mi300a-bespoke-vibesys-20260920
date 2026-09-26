"""Correctness, HIP graph capture, and latency probe for dense BF16 MTP experts.

Run on one MI300A after staging this directory and the checkpoint locally::

    MODEL_PATH=/path/to/Qwen3.5-397B-A17B-MXFP4 \
      python bench_mtp_dense_moe.py --batches 1,4,48,96 --rank 0 --out result.json

Each batch captures one graph, then changes the routing buffers in place and replays cases
covering uniform random routing, all-remote assignments, ownership boundaries, duplicate
experts, zero weights, and inactive bucket rows. Results are compared with a plain PyTorch
per-assignment oracle. ``--all-ranks`` additionally loads the four shards in turn and checks
that their summed partial output matches an unsharded oracle.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mtp_dense_moe  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from model import load_experts  # noqa: E402
from tp import Shard  # noqa: E402
from weights import Checkpoint  # noqa: E402

MODEL_PATH = os.environ.get(
    "MODEL_PATH", "/path/to/scratch/models/Qwen3.5-397B-A17B-MXFP4"
)
HIDDEN, INTER, EXPERTS, TOP_K, WORLD = 4096, 1024, 512, 10, 4


def load_rank(ck: Checkpoint, rank: int, dev: torch.device) -> dict:
    local = EXPERTS // WORLD
    return load_experts(ck, "mtp.layers.0.mlp", Shard(rank * local, local), dev, torch.bfloat16)


def oracle(
    x: torch.Tensor,
    experts: dict,
    expert: torch.Tensor,
    weight: torch.Tensor,
    span: tuple[int, int],
    active: torch.Tensor,
) -> torch.Tensor:
    """PyTorch BF16 spelling of one rank's routed partial, including duplicate assignments."""
    lo, hi = span
    out = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    for token in range(x.shape[0]):
        if not bool(active[token]):
            continue
        for j in range(TOP_K):
            e = int(expert[token, j])
            if not lo <= e < hi or float(weight[token, j]) == 0.0:
                continue
            local = e - lo
            gate, up = F.linear(x[token], experts["gate_up"][local]).chunk(2)
            y = F.linear(F.silu(gate) * up, experts["down"][local])
            out[token] += y * weight[token, j].to(x.dtype)
    return out


def cases(
    batch: int, rank: int, dev: torch.device
) -> list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]]:
    lo, hi = rank * (EXPERTS // WORLD), (rank + 1) * (EXPERTS // WORLD)
    gen = torch.Generator(device=dev).manual_seed(1000 + batch + rank)
    random_i = torch.randint(EXPERTS, (batch, TOP_K), generator=gen, device=dev)
    random_w = torch.rand(batch, TOP_K, generator=gen, device=dev)
    random_w /= random_w.sum(-1, keepdim=True)
    remote = (hi + torch.arange(TOP_K, device=dev)) % EXPERTS
    remote = remote.repeat(batch, 1)
    boundary = (
        torch.tensor(
            [lo - 1, lo, lo + 1, hi - 1, hi, hi + 1, lo, lo, hi - 1, hi - 1],
            device=dev,
        )
        .remainder(EXPERTS)
        .repeat(batch, 1)
    )
    duplicate = torch.full((batch, TOP_K), lo, dtype=torch.long, device=dev)
    nonuniform = torch.arange(1, TOP_K + 1, device=dev, dtype=torch.float32).repeat(batch, 1)
    nonuniform /= nonuniform.sum(-1, keepdim=True)
    zero = nonuniform.clone()
    zero[:, 1::2] = 0
    active = torch.ones(batch, dtype=torch.bool, device=dev)
    padded = active.clone()
    padded[batch // 2 :] = False
    return [
        ("random", random_i, random_w, active),
        ("all_remote", remote, nonuniform, active),
        ("boundaries_duplicates", boundary, nonuniform, active),
        ("duplicate_zero_weight", duplicate, zero, active),
        ("inactive_tail", boundary, nonuniform, padded),
    ]


def error(got: torch.Tensor, want: torch.Tensor) -> dict[str, float]:
    diff = (got.float() - want.float()).abs()
    return {
        "max_abs": diff.max().item(),
        "max_rel_to_output": (diff.max() / want.float().abs().max().clamp_min(1e-30)).item(),
    }


def run_batch(
    batch: int, rank: int, experts: dict, reps: int, dev: torch.device
) -> dict[str, object]:
    lo, hi = rank * (EXPERTS // WORLD), (rank + 1) * (EXPERTS // WORLD)
    x = torch.randn(batch, HIDDEN, device=dev, dtype=torch.bfloat16)
    expert = torch.empty(batch, TOP_K, dtype=torch.long, device=dev)
    weight = torch.empty(batch, TOP_K, dtype=torch.float32, device=dev)
    active = torch.empty(batch, dtype=torch.bool, device=dev)
    inter = torch.empty(batch * TOP_K, INTER, dtype=torch.bfloat16, device=dev)
    out = torch.empty(batch, HIDDEN, dtype=torch.bfloat16, device=dev)

    first = cases(batch, rank, dev)[0]
    expert.copy_(first[1])
    weight.copy_(first[2])
    active.copy_(first[3])
    for _ in range(3):
        mtp_dense_moe.fused_moe(x, experts, (expert, weight), TOP_K, (lo, hi), inter, out, active)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mtp_dense_moe.fused_moe(x, experts, (expert, weight), TOP_K, (lo, hi), inter, out, active)

    checks = []
    for name, case_i, case_w, case_active in cases(batch, rank, dev):
        expert.copy_(case_i)
        weight.copy_(case_w)
        active.copy_(case_active)
        graph.replay()
        torch.cuda.synchronize()
        want = oracle(x, experts, expert, weight, (lo, hi), active)
        checks.append({"case": name, **error(out, want)})

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    expert.copy_(first[1])
    weight.copy_(first[2])
    active.copy_(first[3])
    start.record()
    for _ in range(reps):
        graph.replay()
    end.record()
    end.synchronize()
    return {"batch": batch, "graph_us": start.elapsed_time(end) * 1e3 / reps, "checks": checks}


def all_rank_check(batch: int, ck: Checkpoint, dev: torch.device) -> dict[str, float]:
    x = torch.randn(batch, HIDDEN, device=dev, dtype=torch.bfloat16)
    gen = torch.Generator(device=dev).manual_seed(2026 + batch)
    expert = torch.randint(EXPERTS, (batch, TOP_K), generator=gen, device=dev)
    weight = torch.rand(batch, TOP_K, generator=gen, device=dev)
    weight /= weight.sum(-1, keepdim=True)
    active = torch.ones(batch, dtype=torch.bool, device=dev)
    partials, full = [], torch.zeros_like(x)
    for rank in range(WORLD):
        ex = load_rank(ck, rank, dev)
        span = (rank * (EXPERTS // WORLD), (rank + 1) * (EXPERTS // WORLD))
        inter = torch.empty(batch * TOP_K, INTER, dtype=torch.bfloat16, device=dev)
        out = torch.empty_like(x)
        partials.append(
            mtp_dense_moe.fused_moe(
                x, ex, (expert, weight), TOP_K, span, inter, out, active
            ).clone()
        )
        full += oracle(x, ex, expert, weight, span, active)
        del ex
    return error(torch.stack(partials).float().sum(0).to(torch.bfloat16), full)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="1,4,48,96")
    parser.add_argument("--rank", type=int, default=0, choices=range(WORLD))
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--all-ranks", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    dev = torch.device("cuda:0")
    ck = Checkpoint(MODEL_PATH)
    experts = load_rank(ck, args.rank, dev)
    rows = [
        run_batch(b, args.rank, experts, args.reps, dev) for b in map(int, args.batches.split(","))
    ]
    bad = [
        (row["batch"], check)
        for row in rows
        for check in row["checks"]
        if check["max_abs"] > 0.02 or check["max_rel_to_output"] > 0.02
    ]
    if bad:
        raise AssertionError(f"dense MTP kernel parity failed: {bad}")
    result: dict[str, object] = {"rank": args.rank, "rows": rows}
    if args.all_ranks:
        del experts
        result["all_rank"] = all_rank_check(max(map(int, args.batches.split(","))), ck, dev)
    print(json.dumps(result, indent=2), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

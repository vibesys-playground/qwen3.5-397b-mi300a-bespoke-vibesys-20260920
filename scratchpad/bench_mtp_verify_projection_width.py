"""Measure exact-width DeltaNet verify projection cost on one TP=4 MI300A rank.

Compares the current flattened ``M=B*T`` input/output GEMMs with four ordinary-shaped
``M=B`` calls captured into one graph.  The latter is an upper bound for a grouped kernel
that preserves ordinary decode's reduction order.  Multiply the per-layer delta by 45 for
one Qwen3.5 target verify round.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F


def capture(step):  # noqa: ANN001, ANN202
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    return graph


def milliseconds(graph: torch.cuda.CUDAGraph, repeats: int) -> float:
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    dev, dtype = torch.device("cuda", 0), torch.bfloat16
    batch, width = args.batch, args.width
    # TP=4 local Qwen3.5 DeltaNet shapes: in_proj_all 5152x4096, out_proj 4096x2048.
    in_w = torch.randn(5152, 4096, device=dev, dtype=dtype)
    out_w = torch.randn(4096, 2048, device=dev, dtype=dtype)
    wide_in = torch.randn(batch * width, 4096, device=dev, dtype=dtype)
    wide_out = torch.randn(batch * width, 2048, device=dev, dtype=dtype)
    step_in = wide_in.view(batch, width, 4096).transpose(0, 1).contiguous()
    step_out = wide_out.view(batch, width, 2048).transpose(0, 1).contiguous()

    def wide() -> None:
        F.linear(wide_in, in_w)
        F.linear(wide_out, out_w)

    def step_rows() -> None:
        for step in range(width):
            F.linear(step_in[step], in_w)
            F.linear(step_out[step], out_w)

    in_w_batched = in_w.t().unsqueeze(0).expand(width, -1, -1)
    out_w_batched = out_w.t().unsqueeze(0).expand(width, -1, -1)

    def batched() -> None:
        torch.bmm(step_in, in_w_batched)
        torch.bmm(step_out, out_w_batched)

    reference_in = torch.stack([F.linear(step_in[s], in_w) for s in range(width)])
    reference_out = torch.stack([F.linear(step_out[s], out_w) for s in range(width)])
    batched_in = torch.bmm(step_in, in_w_batched)
    batched_out = torch.bmm(step_out, out_w_batched)
    batched_max_diff = max(
        float((batched_in.float() - reference_in.float()).abs().max()),
        float((batched_out.float() - reference_out.float()).abs().max()),
    )

    wide_graph, step_graph, batched_graph = capture(wide), capture(step_rows), capture(batched)
    wide_ms = milliseconds(wide_graph, args.repeats)
    step_ms = milliseconds(step_graph, args.repeats)
    batched_ms = milliseconds(batched_graph, args.repeats)
    weights_bytes = (in_w.numel() + out_w.numel()) * in_w.element_size()
    extra_bytes = weights_bytes * (width - 1) * 45
    print(
        json.dumps(
            {
                "batch": batch,
                "width": width,
                "wide_ms_per_layer": wide_ms,
                "step_rows_ms_per_layer": step_ms,
                "delta_ms_per_layer": step_ms - wide_ms,
                "delta_ms_per_45_layer_round": (step_ms - wide_ms) * 45,
                "batched_ms_per_layer": batched_ms,
                "batched_delta_ms_per_45_layer_round": (batched_ms - wide_ms) * 45,
                "batched_max_abs_diff_vs_four_linears": batched_max_diff,
                "extra_weight_gib_per_round": extra_bytes / 2**30,
                "effective_extra_tb_s": extra_bytes / ((step_ms - wide_ms) * 45 / 1e3) / 1e12
                if step_ms > wide_ms
                else None,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

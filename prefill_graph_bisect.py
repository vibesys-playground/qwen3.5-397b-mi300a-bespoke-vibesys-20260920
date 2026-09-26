"""Per-layer eager vs graph_prefill static-path comparison (uncaptured), SPMD TP. Diagnoses the
startup "prefill shape 1x16 logits disagree with eager" that disables SEED_PREFILL_GRAPHS."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_decode
import graph_prefill
import torch
import torch.nn.functional as F
from model import rmsnorm
from prefill_profile import build


def rel(a, b):
    return float((a.float() - b.float()).abs().max() / b.float().abs().max().clamp(min=1e-6))


def run_rank(a, rank):
    model = build(a, rank)
    c = model.cfg
    width = a.width
    slot, start = 0, 3
    ids = [(5 * t + 1) % c.vocab for t in range(width)]
    model.begin(slot)
    model.prefill(slot, [2, 3, 4], 0)
    pools = [p for p in model.pool if "conv" in p]
    saved = [(p["conv"].clone(), p["rec"].clone()) for p in pools]
    with torch.no_grad():
        model.bind(slot)
        x = F.embedding(torch.tensor([ids], device=model.devices[0]), model.embed)
        eager = []
        for i in range(len(model.layers)):
            w = model.layers[i]
            h = rmsnorm(x, w["in_norm"], c.eps)
            mix = (
                model.full_attention(i, h, start)
                if c.layer_types[i] == "full_attention"
                else model.deltanet(i, h)
            )
            mix = model.tp.all_reduce(mix)
            x = x + mix
            moe = model.tp.all_reduce(model.moe(i, rmsnorm(x, w["post_norm"], c.eps)))
            x = x + moe
            eager.append((mix[0].clone(), moe[0].clone(), x[0].clone()))
        eager_state = [(p["conv"].clone(), p["rec"].clone()) for p in pools]
        for p, (cv, rc) in zip(pools, saved, strict=True):
            p["conv"].copy_(cv)
            p["rec"].copy_(rc)
        dev = model.devices[-1]
        lane_table = graph_decode.LaneBlockTables(model, dev)
        dirty = [True] * model.max_batch
        runner = graph_prefill.PrefillGraphRunner(model, None, lane_table, dirty)
        shape = graph_prefill.Shape(1, width)
        buf = graph_prefill.PrefillBuffers(model, shape, dev)
        runner.fill(buf, [(slot, ids, start)])
        x = F.embedding(buf.tokens, model.embed)
        rows = []
        for i in range(len(model.layers)):
            w = model.layers[i]
            h = rmsnorm(x, w["in_norm"], c.eps)
            mix = (
                graph_prefill.attn_prefill_static(model, i, h, buf)
                if c.layer_types[i] == "full_attention"
                else graph_prefill.deltanet_prefill_static(model, i, h, buf)
            )
            mix = model.tp.all_reduce(mix)
            # isolate: feed the EAGER residual into this layer's moe too, so errors do not compound
            em, emoe, ex = eager[i]
            xin = x + mix
            moe = model.tp.all_reduce(model.moe(i, rmsnorm(xin, w["post_norm"], c.eps)))
            x = xin + moe
            rows.append(
                {
                    "i": i,
                    "type": c.layer_types[i][:4],
                    "mix": round(rel(mix[0], em), 4),
                    "moe": round(rel(moe[0], emoe), 4),
                    "x": round(rel(x[0], ex), 4),
                }
            )
        st = [
            (round(rel(p["conv"], ec), 4), round(rel(p["rec"], er), 4))
            for p, (ec, er) in zip(pools, eager_state, strict=True)
        ]
    if rank == 0:
        for r in rows:
            print("LAYER " + json.dumps(r), flush=True)
        print("STATE worst conv/rec", max(s[0] for s in st), max(s[1] for s in st), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--port", type=int, default=29537)
    p.add_argument("--batch", type=int, default=48)
    p.add_argument("--max-seq", type=int, default=4608)
    p.add_argument("--width", type=int, default=16)
    a = p.parse_args()
    if a.rank:
        return run_rank(a, a.rank)
    base = [
        sys.executable,
        "-u",
        os.path.abspath(__file__),
        "--tp",
        str(a.tp),
        "--port",
        str(a.port),
        "--batch",
        str(a.batch),
        "--max-seq",
        str(a.max_seq),
        "--width",
        str(a.width),
    ]
    ws = [
        subprocess.Popen([*base, "--rank", str(r)], env=os.environ.copy()) for r in range(1, a.tp)
    ]
    try:
        run_rank(a, 0)
    finally:
        for w in ws:
            w.wait(timeout=300)


if __name__ == "__main__":
    main()

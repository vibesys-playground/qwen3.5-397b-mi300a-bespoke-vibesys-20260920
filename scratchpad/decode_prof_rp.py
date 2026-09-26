"""Per-phase decode-step A/B for fusion work (run from the bundle root, spawns --tp ranks).

Phases: "name:mod.ATTR=1,mod.ATTR2=0;name2:..." -- module globals flipped before each
recapture. Per phase and batch: wall ms/step (median of --reps x --steps). Under rocprofv3
(--kernel-trace), each profiled window is bracketed by torch.cuda._sleep marker kernels so
parse_rp.py can cut exactly --psteps steps out of the trace. Logit check: first phase is the
reference; CHECK lines report bit-exactness / maxdiff / argmax agreement per later phase.
"""

import argparse, importlib, os, subprocess, sys, time

sys.path.insert(0, os.getcwd())
import torch
import tp
from graph_decode import GraphDecodeRunner
from model import Model, load_cfg

MODEL_PATH = os.environ["MODEL_PATH"]


def parse_phases(s):
    out = []
    for spec in s.split(";"):
        name, _, sets = spec.partition(":")
        kv = []
        for item in filter(None, sets.split(",")):
            k, v = item.split("=")
            mod, attr = k.rsplit(".", 1)
            kv.append((mod, attr, v not in ("0", "false", "False")))
        out.append((name, kv))
    return out


def run_rank(a, rank):
    reduce = tp.init(rank, a.tp, port=a.port)
    handle = tp.TP(tp.plan(load_cfg(MODEL_PATH), rank, a.tp), tp.device_for(rank), reduce)
    model = Model(MODEL_PATH, [handle.device], torch.bfloat16, a.max_seq, a.batch, tp=handle)
    try:
        import blas_tune

        blas_tune.tune(model)
    except Exception as e:  # noqa
        print("blas_tune skipped", e, flush=True)
    prompt = list(range(10, 10 + a.prompt))
    for s in range(model.max_batch):
        model.begin(s)
        model.prefill(s, prompt, 0)
    ref = None
    for name, kv in parse_phases(a.phases):
        for mod, attr, val in kv:
            setattr(importlib.import_module(mod), attr, val)
        gr = GraphDecodeRunner(model)
        ok = gr.prepare()
        print(f"rank{rank} phase={name} captured={ok} buckets={gr.buckets}", flush=True)
        slots = list(range(48))
        toks = [(7 * i + 3) % model.cfg.vocab for i in slots]
        gr.reset_slots()
        cur = [gr.decode(slots, toks, [p] * 48).clone() for p in range(4)]
        cur += [gr.decode(slots[:16], toks[:16], [4] * 16).clone()]
        gr.reset_slots()
        if rank == 0 and os.environ.get("PROF_DUMP"):
            torch.save([x.cpu() for x in cur], os.environ["PROF_DUMP"])
        if ref is None:
            ref = cur
        else:
            diff = max((x.float() - y.float()).abs().max().item() for x, y in zip(ref, cur))
            agree = sum(int((x.argmax(-1) == y.argmax(-1)).sum()) for x, y in zip(ref, cur))
            tot = sum(x.shape[0] for x in cur)
            same = all(torch.equal(x, y) for x, y in zip(ref, cur))
            print(
                f"CHECK rank{rank} {name} bitexact={same} maxdiff={diff:.4g} argmax_agree={agree}/{tot}",
                flush=True,
            )
        for b in [int(x) for x in a.batches.split(",")]:
            sl = list(range(b))
            tk = [1000 + s for s in sl]
            pos = [a.prompt] * b
            for _ in range(5):
                gr.decode(sl, tk, pos)
            torch.cuda.synchronize()
            ts = []
            for _ in range(a.reps):
                t0 = time.perf_counter()
                for _ in range(a.steps):
                    gr.decode(sl, tk, pos)
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) / a.steps * 1e3)
            ts.sort()
            print(
                f"RESULT {name} rank{rank} b{b} wall_ms={ts[len(ts) // 2]:.3f} min={ts[0]:.3f}",
                flush=True,
            )
            torch.cuda._sleep(200000)
            torch.cuda.synchronize()
            for _ in range(a.psteps):
                gr.decode(sl, tk, pos)
            torch.cuda.synchronize()
            torch.cuda._sleep(200000)
            torch.cuda.synchronize()
            print(f"MARK {name} rank{rank} b{b}", flush=True)
        del gr
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--port", type=int, default=29581)
    p.add_argument("--batch", type=int, default=48)
    p.add_argument("--max-seq", type=int, default=4096)
    p.add_argument("--prompt", type=int, default=1024)
    p.add_argument("--batches", default="16,48")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--psteps", type=int, default=3)
    p.add_argument("--phases", default="base:")
    a = p.parse_args()
    if a.rank:
        run_rank(a, a.rank)
        return
    base = [sys.executable, "-u", os.path.abspath(__file__)] + [
        x
        for k in (
            "tp",
            "port",
            "batch",
            "max_seq",
            "prompt",
            "batches",
            "steps",
            "reps",
            "psteps",
            "phases",
        )
        for x in (f"--{k.replace('_', '-')}", str(getattr(a, k)))
    ]
    ws = [
        subprocess.Popen(base + ["--rank", str(r)], env=os.environ.copy()) for r in range(1, a.tp)
    ]
    try:
        run_rank(a, 0)
    finally:
        for w in ws:
            w.wait(timeout=600)


if __name__ == "__main__":
    main()

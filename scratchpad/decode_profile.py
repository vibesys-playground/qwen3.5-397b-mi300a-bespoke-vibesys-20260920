"""SPMD per-rank torch.profiler breakdown of graph decode replay at several batch sizes, run from
the bundle root. `--phases base,inplace` recaptures with SEED_DN_STATE_INPLACE flipped and checks
logits are bit-exact. Each rank writes decprof_{phase}_r{rank}_b{batch}.json and a chrome trace;
`decode_profile_segments.py <out> <phase>_` groups rank 0's trace by layer segment."""
import argparse, json, os, subprocess, sys, time
sys.path.insert(0, os.getcwd())
import torch
import tp
from graph_decode import GraphDecodeRunner
from model import Model, load_cfg
from torch.profiler import ProfilerActivity, profile

MODEL_PATH = os.environ["MODEL_PATH"]


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
    import graph_decode
    ref = None
    for phase in a.phases.split(","):
        graph_decode.DN_STATE_INPLACE = phase.startswith("inplace")
        gr = GraphDecodeRunner(model)
        ok = gr.prepare()
        print(f"rank{rank} phase={phase} captured={ok} buckets={gr.buckets}", flush=True)
        slots = list(range(48)); toks = [(7 * i + 3) % model.cfg.vocab for i in slots]
        gr.reset_slots()
        # Positions 0,1,2 then 3: every KV row attention reads was written in this sequence
        # (reset_slots hands out fresh blocks whose rows hold stale data).
        lg = [gr.decode(slots, toks, [p] * 48).clone() for p in range(3)]
        lg1 = [gr.decode(slots[:5], toks[:5], [3] * 5).clone()]
        gr.reset_slots()
        cur = lg + lg1
        if rank == 0 and a.save_logits:
            torch.save([t.cpu() for t in cur], a.save_logits)
        if rank == 0 and a.ref_logits:
            other = [t.to(cur[0].device) for t in torch.load(a.ref_logits, map_location="cpu")]
            diff = max((x.float() - y.float()).abs().max().item() for x, y in zip(other, cur))
            agree = all(torch.equal(x.argmax(-1), y.argmax(-1)) for x, y in zip(other, cur))
            same = all(torch.equal(x, y) for x, y in zip(other, cur))
            print(f"REFCHECK {phase} bitexact={same} maxdiff={diff:.4g} argmax_agree={agree}", flush=True)
        if ref is None:
            ref = cur
        else:
            diff = max((x.float() - y.float()).abs().max().item() for x, y in zip(ref, cur))
            agree = all(torch.equal(x.argmax(-1), y.argmax(-1)) for x, y in zip(ref, cur))
            print(f"CHECK rank{rank} {phase} bitexact=" + str(all(torch.equal(x, y) for x, y in zip(ref, cur))) + f" maxdiff={diff:.4g} argmax_agree={agree}", flush=True)
        run_phase(a, rank, gr, phase)
        del gr
        torch.cuda.synchronize(); torch.cuda.empty_cache()


def run_phase(a, rank, gr, phase):
    for b in [int(x) for x in a.batches.split(",")]:
        slots = list(range(b))
        toks = [1000 + s for s in slots]
        pos = [a.prompt] * b
        for _ in range(5):
            gr.decode(slots, toks, pos)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.steps):
            gr.decode(slots, toks, pos)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / a.steps * 1e3
        if not a.profile or not phase.startswith(a.profile_phase):
            print(f"RESULT {phase} rank{rank} b{b} wall_ms={wall:.3f}", flush=True)
            continue
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(a.psteps):
                gr.decode(slots, toks, pos)
            torch.cuda.synchronize()
        ev = {}
        for e in prof.key_averages():
            d = getattr(e, "self_device_time_total", 0.0) or 0.0
            if d > 0:
                ev[e.key] = [d / a.psteps, e.count / a.psteps]
        out = {"rank": rank, "batch": b, "bucket": min(x for x in gr.buckets if x >= b),
               "wall_ms": wall, "kernels": ev}
        with open(os.path.join(a.out, f"decprof_{phase}_r{rank}_b{b}.json"), "w") as f:
            json.dump(out, f)
        prof.export_chrome_trace(os.path.join(a.out, f"trace_{phase}_r{rank}_b{b}.json"))
        print(f"RESULT {phase} rank{rank} b{b} wall_ms={wall:.3f} dev_ms={sum(v[0] for v in ev.values())/1e3:.3f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--port", type=int, default=29531)
    p.add_argument("--batch", type=int, default=48)
    p.add_argument("--max-seq", type=int, default=4096)
    p.add_argument("--prompt", type=int, default=32)
    p.add_argument("--batches", default="1,16,48")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--psteps", type=int, default=3)
    p.add_argument("--out", default=".")
    p.add_argument("--phases", default="base")
    p.add_argument("--profile", type=int, default=1)
    p.add_argument("--profile-phase", default="")
    p.add_argument("--save-logits", default="")
    p.add_argument("--ref-logits", default="")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.rank:
        run_rank(a, a.rank)
        return
    base = [sys.executable, "-u", os.path.abspath(__file__)] + [x for k in
            ("tp", "port", "batch", "max_seq", "prompt", "batches", "steps", "psteps", "out", "phases", "profile", "profile_phase", "save_logits", "ref_logits")
            for x in (f"--{k.replace('_','-')}", str(getattr(a, k)))]
    ws = [subprocess.Popen(base + ["--rank", str(r)], env=os.environ.copy()) for r in range(1, a.tp)]
    try:
        run_rank(a, 0)
    finally:
        for w in ws:
            w.wait(timeout=300)


if __name__ == "__main__":
    main()

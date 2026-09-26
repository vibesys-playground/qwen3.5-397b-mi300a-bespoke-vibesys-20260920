"""`SEED_LMHEAD_VOCAB_TP` bit-exactness and cost on one GPU: the real `lm_head`, full vs each
of its `world` vocab shards, at the decode widths, under the given TunableOp cache.

    python3 scratchpad/lmhead_split_check.py --model-path DIR [--cache CSV] [--world 4]

Prints per width: whether every shard's logits equal the full GEMM's columns bit for bit,
and the mean time of the full GEMM vs one shard.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from weights import Checkpoint  # noqa: E402


def timed(fn, reps: int = 50) -> float:  # noqa: ANN001
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--cache", default="")
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--widths", default="1,4,8,16,32,48,64")
    ap.add_argument(
        "--pin", action="store_true", help="give each shard shape its full shape's solution"
    )
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    if args.cache and args.pin:
        import blas_tune

        validators, table = blas_tune.read_table(Path(args.cache))
        for (op, params), (_sol, ms) in list(table.items()):
            f = params.split("_")  # tn_N_M_K_ld_...
            if len(f) > 3 and f[1] == str(248320 // args.world):
                full = (op, "_".join([f[0], "248320", f[2], f[3], "ld", f[3], f[3], "248320"]))
                if full in table:
                    table[(op, params)] = (table[full][0], ms)
        args.cache = str(Path(args.cache).with_suffix(".pinned.csv"))
        blas_tune.write_table(Path(args.cache), validators, table)
    if args.cache:
        t = torch.cuda.tunable
        t.set_filename(args.cache, False)
        t.enable(True)
        t.tuning_enable(False)
        t.read_file(args.cache)
    w = Checkpoint(args.model_path).load("lm_head.weight", dev, torch.bfloat16)
    n = w.shape[0] // args.world
    shards = [w[r * n : (r + 1) * n].contiguous() for r in range(args.world)]
    gen = torch.Generator(device=dev).manual_seed(0)
    for m in [int(x) for x in args.widths.split(",")]:
        h = torch.randn(m, w.shape[1], generator=gen, device=dev, dtype=torch.bfloat16)
        full = F.linear(h, w)
        exact = all(
            torch.equal(F.linear(h, s), full[:, r * n : (r + 1) * n]) for r, s in enumerate(shards)
        )
        ms_full = timed(lambda h=h: F.linear(h, w))
        ms_shard = timed(lambda h=h: F.linear(h, shards[0]))
        print(
            f"b{m}: bit-exact {exact}  full {ms_full * 1e3:.0f} us  shard {ms_shard * 1e3:.0f} us",
            flush=True,
        )


if __name__ == "__main__":
    main()

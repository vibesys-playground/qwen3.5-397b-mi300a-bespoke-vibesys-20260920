#!/bin/bash
# GPU A/B for SEED_OVERLAP_SCHED (one-step-lookahead decode). Same build, two server boots:
# SEED_OVERLAP_SCHED=0 then =1, both with SEED_STEP_TIMING=1 and --enable-graph-capture, TP=4.
#
# Reports per mode: decode tok/s at C=48 (ignore_eos, full batch), the per-rank device idle
# time between decode steps ("host_gap_ms", step_timing.GapMeter), and whether every greedy
# output text is identical across the two modes (full-length and natural-EOS waves).
#
# Prediction (see the branch's commit message): overlap off, host_gap_ms p50 ~5 ms per step
# at b48 (GraphDecodeRunner.fill's per-step block-table build ~4.6 ms + readback, scheduler
# and RCCL command broadcast ~0.5-1 ms); on, p50 < 0.5 ms. Step 79 -> ~74 ms, tok/s +6-7%.
#
#   srun --jobid=<jobid> --overlap --environment=/path/to/scratch/runtime.toml \
#     bash -lc 'cd <this worktree>/examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke &&
#       export MODEL_PATH=/path/to/scratch/models/Qwen3.5-397B-A17B-MXFP4 &&
#       export SEED_BLAS_CACHE_DIR=/path/to/scratch/blas-cache/job-<jobid> &&
#       timeout 5400 ./scratchpad/gpu_ab_overlap_sched.sh'
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${MODEL_PATH:?set MODEL_PATH}"
: "${SEED_BLAS_CACHE_DIR:?set a job-private SEED_BLAS_CACHE_DIR}"
PORT=${PORT:-30000}
CONCURRENCY=${CONCURRENCY:-48}
MAX_TOKENS=${MAX_TOKENS:-256}
OUT=${OUT_DIR:-$(mktemp -d)}
echo "results in $OUT"

run_one() {
  local label=$1 flag=$2
  pkill -f '[s]erver.py' || true
  sleep 3
  SEED_OVERLAP_SCHED=$flag SEED_STEP_TIMING=1 \
    timeout 2400 python3 -u server.py --model-path "$MODEL_PATH" --host 127.0.0.1 \
      --port "$PORT" --tp 4 --max-batch 48 --enable-graph-capture \
      >"$OUT/$label.server.log" 2>&1 &
  local pid=$!
  timeout 1500 python3 - "$PORT" <<'PY'
import sys, time, urllib.request
url = f"http://127.0.0.1:{sys.argv[1]}/health"
deadline = time.time() + 1480
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            if r.status == 200:
                sys.exit(0)
    except Exception:
        pass
    time.sleep(10)
sys.exit(1)
PY
  timeout 1500 python3 scratchpad/overlap_probe.py --port "$PORT" --concurrency "$CONCURRENCY" \
    --max-tokens "$MAX_TOKENS" --out "$OUT/$label.json"
  pkill -f '[s]erver.py' || true
  wait "$pid" 2>/dev/null || true
}

run_one off 0
run_one on 1

timeout 60 python3 - "$OUT" <<'PY'
import json, re, statistics, sys
out = sys.argv[1]
for label in ("off", "on"):
    r = json.load(open(f"{out}/{label}.json"))
    log = open(f"{out}/{label}.server.log").read()
    gaps = [float(x) for x in re.findall(r"rank 0 decode host_gap_ms mean=[0-9.]+ p50=([0-9.]+)", log)]
    paths = re.findall(r"sched decode .* path=(\w+)", log)
    print(f"{label:>3}: tok/s={r['decode_tok_s']:.1f}  rank0 host_gap p50 (median of windows)="
          f"{statistics.median(gaps) if gaps else float('nan'):.3f} ms over {len(gaps)} windows  "
          f"graph-path windows={paths.count('graph')}/{len(paths)}")
off, on = (json.load(open(f"{out}/{l}.json")) for l in ("off", "on"))
for k in ("texts_full", "texts_natural"):
    same = sum(a == b for a, b in zip(off[k], on[k]))
    print(f"{k}: {same}/{len(off[k])} identical")
print(f"speedup: {on['decode_tok_s'] / off['decode_tok_s']:.3f}x")
PY

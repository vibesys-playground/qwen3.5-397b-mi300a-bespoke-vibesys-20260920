#!/bin/bash
# GPU A/B validation for SEED_MIXED_BATCH, at C=48 (the campaign brief's own regime).
#
# Launches the candidate server twice (SEED_MIXED_BATCH=0 baseline, then =1), runs the
# existing load-ramp benchmark (benchmark/run.py, --concurrency 48) against each, and prints
# both JSON results side by side: p95_ttft_turn2plus_ms, mean_tpot_ms, throughput. Confirms or
# refutes scratchpad/mixed_batch_sim.py's prediction (p95 TPOT stays within the 250 ms guardrail
# under mixed batching, collapses under plain alternation).
#
# Usage (from the bundle's own worktree root, e.g. inside the srun shell below):
#   SEED_MIXED_TOKEN_BUDGET=512 ./scratchpad/gpu_validate_mixed_batch.sh
#
# Full job invocation (fill in <jobid>; matches COMMON_BRIEF.md's operational constraints:
# job-private BLAS cache, everything under timeout, no bare curl):
#
#   srun --jobid=<jobid> --overlap \
#     --environment=/path/to/scratch/runtime.toml \
#     bash -lc '
#       cd /path/to/wt-mixed-batch/examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke &&
#       export MODEL_PATH=/path/to/scratch/models/Qwen3.5-397B-A17B-MXFP4 &&
#       export SEED_BLAS_CACHE_DIR=/path/to/scratch/blas-cache/job-<jobid> &&
#       timeout 1800 ./scratchpad/gpu_validate_mixed_batch.sh
#     '
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."  # bundle root (server.py lives here)

: "${MODEL_PATH:?set MODEL_PATH to the checkpoint dir}"
HOST=127.0.0.1
PORT=${PORT:-30000}
CONCURRENCY=${CONCURRENCY:-48}
OUT_DIR=$(mktemp -d)
echo "results in $OUT_DIR"

run_one() {
  local label="$1" mixed_flag="$2"
  echo "=== $label (SEED_MIXED_BATCH=$mixed_flag) ==="
  pkill -f '[s]erver.py' || true
  sleep 2
  SEED_MIXED_BATCH="$mixed_flag" \
    timeout 3000 python3 server.py \
      --model-path "$MODEL_PATH" --host "$HOST" --port "$PORT" \
      --enable-graph-capture \
      >"$OUT_DIR/$label.server.log" 2>&1 &
  local server_pid=$!

  # Poll /health with urllib (no curl, per the campaign brief), under an explicit timeout.
  timeout 300 python3 - "$HOST" "$PORT" <<'PY'
import sys, time, urllib.request
host, port = sys.argv[1], sys.argv[2]
url = f"http://{host}:{port}/health"
deadline = time.time() + 290
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            if r.status == 200:
                sys.exit(0)
    except Exception:
        pass
    time.sleep(5)
sys.exit(1)
PY

  timeout 1800 python3 benchmark/run.py \
    --base-url "http://$HOST:$PORT" \
    --concurrency "$CONCURRENCY" \
    --output-json "$OUT_DIR/$label.json"

  pkill -f '[s]erver.py' || true
  wait "$server_pid" 2>/dev/null || true
  sleep 2
}

run_one baseline 0
run_one mixed 1

echo
echo "=== summary ==="
python3 - "$OUT_DIR/baseline.json" "$OUT_DIR/mixed.json" <<'PY'
import json, sys
base, mixed = (json.load(open(p)) for p in sys.argv[1:3])
for k in ("p95_ttft_turn2plus_ms", "mean_tpot_ms"):
    print(f"{k:>28}: baseline={base.get(k):>10.1f}  mixed={mixed.get(k):>10.1f}")
PY

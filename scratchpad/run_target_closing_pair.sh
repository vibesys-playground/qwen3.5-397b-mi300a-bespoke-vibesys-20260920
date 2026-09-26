#!/bin/bash
# Same-node C96 pair: the 2f012388 suffix incumbent versus this target-closing bundle.
# Both arms use the incumbent production flags. The candidate differs only in source plus the
# validated 2x128 and 1x384 prefill containers. SEED_AR_SP is already an incumbent flag; the
# bundle extends its exact row-sharded residual path to ordinary decode and captured prefill.
set -euo pipefail

CONTROL_WORKSPACE=${CONTROL_WORKSPACE:?set CONTROL_WORKSPACE to a clean 2f012388 checkout}
CANDIDATE_WORKSPACE=${CANDIDATE_WORKSPACE:?set CANDIDATE_WORKSPACE to the bundle checkout}
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the Qwen3.5 checkpoint}
CACHE_SEED=${CACHE_SEED:?set CACHE_SEED to a frozen hipcache/blascache directory}
OUT_ROOT=${OUT_ROOT:-/tmp/target-closing-pair-${SLURM_JOB_ID:-local}}
PORT=${PORT:-30000}

test "$(git -C "$CONTROL_WORKSPACE" rev-parse HEAD)" = \
  "2f012388c37534876175f77189059b273593a300"
git -C "$CONTROL_WORKSPACE" diff --quiet
git -C "$CANDIDATE_WORKSPACE" diff --quiet
mkdir -p "$OUT_ROOT"
printf 'CONTROL_COMMIT %s\n' "$(git -C "$CONTROL_WORKSPACE" rev-parse HEAD)" | tee "$OUT_ROOT/source.txt"
printf 'CANDIDATE_COMMIT %s\n' "$(git -C "$CANDIDATE_WORKSPACE" rev-parse HEAD)" | tee -a "$OUT_ROOT/source.txt"

BASE_FLAGS=(
  SEED_MOE_HIP=1 SEED_OVERLAP_SCHED=1 SEED_PREFILL_GROUPED_MOE=1
  SEED_DELTANET_PREFILL_CHUNKED=1 SEED_DECODE_ATTN_SPLITK=1 SEED_DEFER_TURN_CLOSE=1
  SEED_PREFILL_GRAPHS=1 SEED_PREFILL_FUSED_IN_PROJ=1 SEED_BLAS_TUNE_BUCKETS=1
  SEED_DN_STATE_INPLACE=1 SEED_MOE_ROUTE_FUSED=1 SEED_DN_CONV_INPLACE=1
  SEED_ATTN_ROPE_KV_FUSED=1 SEED_PREFILL_FIT_GRAPH=1 SEED_CUSTOM_ALLREDUCE=1
  SEED_AR_GRAPH_SAFE=1 SEED_ADD_RMSNORM=1 SEED_ELEMWISE_FUSED=1
  SEED_DN_NORM_F32_IN=1 SEED_PREFILL_GRAPH_DN_CHUNKED=1 SEED_AR_RMSNORM_FUSED=1
  SEED_AR_SP=1 SEED_MOE_HIP_WIDE=1 SEED_LMHEAD_VOCAB_TP=1 SEED_BLAS_TUNE_MERGE=1
  SEED_SCORE_ENDPOINT=1 SEED_MIXED_FINE_TOTALS=1 SEED_DN_PREFILL_GLUE_FUSED=1
  SEED_MOE_PREP_FORK=1 SEED_FOLD_TURN_SUFFIX=1
  SEED_MOE_HIP_WIDE2=0 SEED_PREFILL_BURST=0 SEED_MIXED_GRAPH=0 SEED_MIXED_BATCH=0
  SEED_CHAT_HISTORY_REUSE=0 SEED_SNAPSHOT_POOL_COUNT=384 SEED_KV_POOL_HEADROOM_FACTOR=1.0
  SEED_KV_POOL_CAP_GIB=24 SEED_MIN_UNALLOCATED_GIB=25 PYTHONUNBUFFERED=1
)
CONTROL_SHAPES=1x16,2x16,4x16,8x16,1x64,2x64,4x64,1x256,2x256,1x512
CANDIDATE_SHAPES=1x16,2x16,4x16,8x16,1x64,2x64,4x64,2x128,1x256,1x384,2x256,1x512
SERVER_PID=
CACHE_TMP=

stop_server() {
  if [[ -n ${SERVER_PID:-} ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=
  fi
}

cleanup() {
  stop_server
  [[ -z ${CACHE_TMP:-} ]] || find "$CACHE_TMP" -depth -delete 2>/dev/null || true
}
trap cleanup EXIT

start_server() {
  local label=$1 workspace=$2 shapes=$3
  CACHE_TMP=$(mktemp -d "/tmp/target-closing-${label}-${SLURM_JOB_ID:-local}.XXXXXX")
  mkdir "$CACHE_TMP/hipcache" "$CACHE_TMP/blascache"
  cp -R "$CACHE_SEED/hipcache/." "$CACHE_TMP/hipcache/"
  cp -R "$CACHE_SEED/blascache/." "$CACHE_TMP/blascache/"
  chmod -R u+rwX "$CACHE_TMP"
  find "$CACHE_TMP" -maxdepth 3 -type f \( -name lock -o -name '*.lock' \) -delete

  cd "$workspace"
  env "${BASE_FLAGS[@]}" SEED_PREFILL_GRAPH_SHAPES="$shapes" \
    AITER_JIT_DIR=/path/to/scratch/aiter-jit-cache \
    SEED_HIP_CACHE_DIR="$CACHE_TMP/hipcache" SEED_BLAS_CACHE_DIR="$CACHE_TMP/blascache" \
    python3 -u server.py --port "$PORT" --max-batch 96 --max-seq-len 8192 \
      --enable-graph-capture >"$OUT_ROOT/server_${label}.log" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 540); do
    if python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/health', timeout=4)" \
      >/dev/null 2>&1; then
      echo "HEALTH_READY $label $(date -Is)"
      return
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      tail -100 "$OUT_ROOT/server_${label}.log"
      exit 1
    fi
    [[ $i -lt 540 ]] || { tail -100 "$OUT_ROOT/server_${label}.log"; exit 1; }
    [[ $((i % 6)) -ne 0 ]] || echo "BOOT_WAIT $label seconds=$((i * 10))"
    sleep 10
  done
}

run_arm() {
  local label=$1 workspace=$2 shapes=$3
  start_server "$label" "$workspace" "$shapes"
  python3 "$workspace/benchmark/run.py" --base-url "http://127.0.0.1:$PORT" \
    --quick --ramp 96 --window-s 60 --no-early-stop \
    --output-json "$OUT_ROOT/${label}_c96.json" | tee "$OUT_ROOT/${label}_c96.log"
  stop_server
  find "$CACHE_TMP" -depth -delete
  CACHE_TMP=
}

run_arm control "$CONTROL_WORKSPACE" "$CONTROL_SHAPES"
run_arm candidate "$CANDIDATE_WORKSPACE" "$CANDIDATE_SHAPES"

python3 - "$OUT_ROOT/control_c96.json" "$OUT_ROOT/candidate_c96.json" <<'PY'
import json
import sys

control = json.load(open(sys.argv[1]))["peak_goodput_tok_s"]
candidate = json.load(open(sys.argv[2]))["peak_goodput_tok_s"]
print(
    f"PAIRED control={control:.6f} candidate={candidate:.6f} "
    f"ratio={candidate / control:.6f} gain_pct={(candidate / control - 1) * 100:.3f}"
)
PY
grep -iE 'SessionCacheError|Traceback|out of memory|failed turn' \
  "$OUT_ROOT"/server_*.log "$OUT_ROOT"/*_c96.log || true

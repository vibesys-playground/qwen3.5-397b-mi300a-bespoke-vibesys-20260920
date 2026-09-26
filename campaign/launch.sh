#!/bin/bash
# Boot the final engine with the production flag set and run one quick C96 measurement.
# Run from the repository root on one 4x MI300A node.
set -euo pipefail

MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the Qwen3.5-397B-A17B-MXFP4 checkpoint}
CACHE_SEED=${CACHE_SEED:?set CACHE_SEED to a hipcache/blascache seed directory (see campaign/README.md)}
AITER_JIT_DIR=${AITER_JIT_DIR:?set AITER_JIT_DIR to a writable AITER JIT cache}
OUT_ROOT=${OUT_ROOT:-/tmp/bespoke-c96-${SLURM_JOB_ID:-local}}
PORT=${PORT:-30000}
export MODEL_PATH AITER_JIT_DIR

PROD_FLAGS=(
  # Base
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
  SEED_CHAT_HISTORY_REUSE=0 SEED_SNAPSHOT_POOL_COUNT=192 SEED_KV_POOL_HEADROOM_FACTOR=1.0
  SEED_KV_POOL_CAP_GIB=24 SEED_MIN_UNALLOCATED_GIB=25
  # Accumulated wide prefill
  SEED_PREFILL_ACCUM=1 SEED_PREFILL_ACCUM_TOKENS=1536 SEED_PREFILL_ACCUM_MAX_WAIT_MS=800
  SEED_PREFILL_GRAPH_TUNE=0
  # Decode split-K attention
  SEED_DECODE_ATTN_SPLITK_PER_CU=8 SEED_DECODE_ATTN_SPLITK_STAGES=1
  # Fused prefill kernels
  SEED_PREFILL_DN_STATE_INPLACE=1 SEED_PREFILL_DN_NORM_FUSED=1 SEED_PREFILL_ATTN_FUSED=1
  # Packed prefill, 32-row MoE tiles, early prefill launch, boot memory checks
  SEED_PREFILL_PACK=1 SEED_MOE_HIP_BT32=1 SEED_PREFILL_EARLY_LAUNCH=1
  SEED_PREFILL_VALIDATE_LANES=1 SEED_BOOT_MEM_PREFLIGHT_GIB=100
  # Sequence-parallel all-reduce
  SEED_AR_SP_DECODE=1 SEED_AR_SP_MAX_TOKENS=2048 SEED_AR_SP_ONE_RELEASE=1 SEED_AR_SP_ROWS_PER_PROG=4
  # MTP speculative decoding
  SEED_MTP_SERVE=1 SEED_MTP_K=2 SEED_MTP_VERIFY_WIDE=1 SEED_MTP_FORCED_DRAFTS=1
  SEED_PREFILL_GRAPH_SHAPES=1x16,2x16,4x16,8x16,1x64,2x64,4x64,2x128,1x256,1x384,2x256,1x512,3x256,4x256,6x256,8x256,2x384,3x384,4x384,2x512,4x512
  PYTHONUNBUFFERED=1
)

# Leaked device masks from single-GPU tools boot TP=1 and run out of memory.
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
# GPU crash dumps are ~20 GB per rank and consume unified memory.
ulimit -c 0

mkdir -p "$OUT_ROOT"
# Never point the server at the seed itself: it writes into the cache directories.
CACHE_TMP=$(mktemp -d "/tmp/bespoke-cache-${SLURM_JOB_ID:-local}.XXXXXX")
cp -R "$CACHE_SEED/hipcache" "$CACHE_SEED/blascache" "$CACHE_TMP/"
find "$CACHE_TMP" -type f \( -name lock -o -name '*.lock' \) -delete

SERVER_PID=
cleanup() {
  if [[ -n $SERVER_PID ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -rf "$CACHE_TMP"
}
trap cleanup EXIT

env "${PROD_FLAGS[@]}" SEED_HIP_CACHE_DIR="$CACHE_TMP/hipcache" SEED_BLAS_CACHE_DIR="$CACHE_TMP/blascache" \
  python3 -u server.py --port "$PORT" --max-batch 96 --max-seq-len 8192 --enable-graph-capture \
  >"$OUT_ROOT/server.log" 2>&1 &
SERVER_PID=$!

# Boot (weights, BLAS, graph capture) takes 10 to 19 minutes.
for _ in $(seq 1 180); do
  python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/health', timeout=4)" \
    >/dev/null 2>&1 && break
  kill -0 "$SERVER_PID" 2>/dev/null || { tail -100 "$OUT_ROOT/server.log"; exit 1; }
  sleep 10
done

python3 benchmark/run.py --base-url "http://127.0.0.1:$PORT" \
  --quick --ramp 96 --window-s 60 --no-early-stop --output-json "$OUT_ROOT/c96.json"

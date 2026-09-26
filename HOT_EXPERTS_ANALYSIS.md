# Hot-expert bf16 cache + cross-rank redundant experts: trace-driven analysis

Branch `bespoke/opt-moe-hotexperts`, base `bespoke/integration-r2` (`e8d25883`). No GPU access
during this work (pre-15:00 UTC maintenance window); everything below is CPU-validated plus a
derived prediction for the GPU run to confirm or refute.

## Data

1,320 real-traffic routing traces (60 MoE layers x 22 decode steps, batch 48, top_k 10, 512
global experts, 128 local/rank at TP=4 expert-parallel) copied from
`cluster:/path/to/scratch/bespoke-opt-round2/routing_traces/` to
`scratchpad/routing_traces/` (11 MB). Traffic: 48 distinct synthetic prompts built from a fixed
word bank (`real_traffic_routing.py`'s generator, per that directory's `README.txt`), greedy
decode. Analysis script: `scratchpad/analyze_hot_experts.py`; raw output:
`scratchpad/hot_experts_analysis.json`.

Caveat worth flagging: this traffic is far more repetitive than open-domain chat (a handful of
global experts dominate almost every token's top-10, dedup factor up to ~17x in a companion
log seen alongside these traces), which likely *overstates* achievable coverage relative to
production traffic. The held-out numbers below are the more honest estimate for that reason
(see "Coverage" methodology).

## 1. Hot-expert bf16 cache -- implemented, `SEED_HOT_EXPERTS=1`

**Cited ideas:** DeepSeek-V3's EPLB (placement/caching driven by monitored router load, not a
static assignment) and dequantized-hot-weight caching (a bf16 standing copy of the experts a
rank's own traffic hits hardest, dense-computed instead of re-dequantized per assignment).

**Design** (`hot_experts.py`, spliced into `model.Model.moe`/`_routed_fused`/`_routed_grouped`
in `model.py`): `Model` accumulates per-layer local-expert assignment counts
(`observe_routing`) until `finalize_hot_experts` (called from `Model.warmup`) freezes, per
layer, the `H` local experts this rank saw most and builds their dense bf16 copy
(`build_layer_cache`, via the existing `mxfp4.dequant_mxfp4`). Every decode step after that:
`hot_expert_forward` runs one **dense, fixed-shape (`H` x batch) batched `torch.matmul`** for
the cached experts against every token (graph-capturable: never a function of that step's
routing) and returns which `(token, slot)` assignments it consumed; `moe` zeros `top_w` there
before calling the unmodified MXFP4 cold path, which already skips zero-weight assignments
(`mxfp4_gemv.py`'s `a_weight != 0` gate -- the same mechanism `FAULT_DROP_EXPERT` uses). No
kernel changes; a hot assignment is computed exactly once.

**Budget** (`SEED_HOT_EXPERT_GIB`, default 16, clamped to <=20 per COMMON_BRIEF, split evenly
over the model's 60 MoE layers): one expert's dense bf16 `gate_up`+`down` is
`(2*1024*4096 + 4096*1024) * 2 bytes = 25.17 MB` (no MXFP4 scale tensors in bf16; ~6.68 MB
packed -> ~25.2 MB dense, close to the ~26.7 MB estimate in COMMON_BRIEF). At the default 16
GiB/rank this affords **H=11 experts/layer** (14.06 GiB); the 20 GiB cap affords **H=14**
(18.28 GiB) -- both comfortably inside the stage-2 allocator's `SEED_MIN_UNALLOCATED_GIB=25`
headroom on top of everything else already reserved.

### Coverage (fraction of assignments hitting the cached experts)

Per (layer, rank): pick the `H` hottest local experts from steps 0-10 ("train"), measure the
hit rate on steps 11-21 ("held-out"); "in-sample" (all 22 steps both pick and measure) is the
optimistic upper bound.

| H | in-sample | held-out (mean) | held-out p10 |
|---:|---:|---:|---:|
| 2 | 27.4% | 12.1% | 0.3% |
| 4 | 42.8% | 18.8% | 2.1% |
| 8 | 62.0% | 28.9% | 7.6% |
| 11 (default budget) | 71.3% | 36.2% | 10.4% |
| 14 (20 GiB cap) | 78.0% | 43.2% | 13.6% |
| 16 | 81.4% | 46.8% | 15.4% |

### Predicted GPU numbers

Cold-path savings model: the MXFP4 kernel is compute-bound on INT32 dequant ops (measured
~3.3x over its own HBM roofline, not just weight-load latency), uniform per assignment, so
removing a `coverage` fraction of assignments removes ~`coverage` fraction of its time. Dense
hot-path added cost: memory-bound floor for `H` experts/layer x 60 layers, at the measured
one-MI300A HBM rate 3.03 TB/s (`torch` d2d copy probe). Baseline: 38 ms/step measured MoE
expert GEMMs (b48, graph replay).

| H | coverage (held-out) | cold-path savings | hot dense add | **net** | % of 38 ms MoE bucket |
|---:|---:|---:|---:|---:|---:|
| 8 | 28.9% | 10.98 ms | 3.99 ms | 7.00 ms | 18.4% |
| **11 (default)** | **36.2%** | **13.76 ms** | **5.48 ms** | **8.27 ms** | **21.8%** |
| 14 (20 GiB) | 43.2% | 16.42 ms | 6.98 ms | 9.44 ms | 24.8% |

**Predicted: ~8.3 ms/step off the MoE bucket at the default `H=11` (~10-11% off the measured
79 ms/step b48 decode step), rising to ~9.4 ms at the 20 GiB cap.** The dominant uncertainty is
the dense hot path's real achieved rate (memory-bound floor assumed; a compute-bound estimate
at 10% MFU of an assumed ~975 TFLOPS bf16 matrix peak, CU-scaled 228/304 from the ~1.3 PFLOP/s
MI300X figure in `resources/skills/serving-systems/references/platforms/rocm/hardware.md`,
gives a similar order of magnitude, 3-12 ms depending on H) -- the GPU run should measure this
directly rather than trust either roofline.

### Validation (CPU, no GPU)

`seed_tests/test_hot_experts.py`, 11 cases, all passing on `venv-cpu`:

- Pure logic: `select_hot_local` tie-breaking/determinism, `budget_h_per_layer` clipping,
  `per_expert_bf16_bytes` shape math.
- **Numerics**: tiny synthetic MXFP4 weights (`test_moe_vectorize.py`'s `build_layer`/
  `quantize_mxfp4`), `hot_expert_forward` + cold reference (zero-weight-skipped) summed vs. a
  full per-assignment MXFP4 reference (`dequant_mxfp4` + `F.linear`), float32, `atol=rtol=1e-4`.
  Edge cases: 0 hot experts (hot path contributes exactly zero) and all experts hot (cold path
  contributes exactly zero).
- **Routing-split correctness**: `consumed` exactly matches "assignment's expert is in the hot
  set"; the zeroed/nonzero partition of `top_w` after splicing is exactly consumed/not-consumed,
  no double count, no dropped weight.
- Full-model regression: `seed_tests/test_seed_parity.py` (5 cases, real tiny HF-parity model
  through the actual server/warmup path) passes both with the flag off and with
  `SEED_HOT_EXPERTS=1 SEED_HOT_EXPERT_GIB=0.001`, confirming `Model.__init__`/`warmup`/`moe`
  wiring doesn't break construction or generation.

No new Triton kernel (the hot path is a plain `torch.matmul`, expected to lower to
hipBLASLt/rocBLAS batched GEMM), so the offline gfx942 ISA/occupancy check
(`scratchpad/isa.py`/`loopstat.py`) doesn't apply here.

### Known limitation, by design

`Model.warmup`'s own traffic (2 dummy decode tokens) is not representative; the mechanism
(accumulate -> freeze -> cache) is correct and tested, but a real deployment wanting the
coverage above needs `finalize_hot_experts` called after real-shaped warmup traffic, which is a
server-startup wiring choice, not a `hot_experts.py` concern. Left as a follow-up rather than
touching `server.py`'s startup sequence in this narrowly-scoped change.

## 2. Cross-rank redundant experts -- NOT implemented; analysis only

COMMON_BRIEF's stop condition: implement only if the trace analysis predicts >=10% MoE-time
savings. It does not.

**Skew measured**: median max/median distinct-owned-expert ratio across ranks is **1.28x**
(mean 1.38x, p90 1.78x) over all 1,320 (layer, step) samples -- in the same range as, if a bit
lower than, COMMON_BRIEF's 1.47x (different trace, same idea).

**Simulation** (`redundant_expert_simulation`/`_by_load` in `analyze_hot_experts.py`), replicating
the `R` globally-hottest experts (by total assignment count, R in {1,2,4,8}) onto whichever rank
is least-loaded that step:

- Distinct-expert-count proxy (a replica lets the idle rank skip a *load* it would otherwise
  make): **0.0% to -0.02%** predicted reduction. The globally hottest experts are already so
  dominant (dedup factor up to ~17x) that essentially every rank already has them in its
  distinct set most steps -- a replica adds nothing to "have I already paid for this expert."
- Optimistic assignment-count proxy (best case: a replica perfectly halves that expert's *own*
  token count between owner and replica, no dispatch cost modeled): **0.2% (R=1) to 1.6%
  (R=8)** predicted reduction.

Both are far under the 10% bar, so idea 2 is not implemented. It also does not fit the
existing expert-parallel design without adding a partial all-to-all (a rank only ever computes
experts it owns; splitting one expert's tokens across two ranks needs a second rank to receive
some of that layer's activations, which the current "replicated routing / all-reduce combine,
no all-to-all" design (`model.py`'s `moe` docstring) does not have) -- a second, larger reason
not to build it for a <10% predicted payoff.

## GPU validation (once the maintenance window ends)

Launch the server directly (so `--enable-graph-capture`/env reach it; `benchmark/run.py`'s own
launcher does not forward flags to a boot it does itself) and drive it with
`benchmark/run.py --base-url`:

```bash
JOBID=<id>
MODEL=/path/to/scratch/models/Qwen3.5-397B-A17B-MXFP4
RUN() { timeout 3600 srun --jobid=$JOBID --overlap --environment=/path/to/scratch/runtime.toml "$@"; }

for hot in 0 1; do
  CACHE=$SCRATCH/blas-cache-$JOBID-$hot
  mkdir -p "$CACHE"
  RUN env MODEL_PATH=$MODEL SEED_BLAS_CACHE_DIR=$CACHE \
      SEED_HOT_EXPERTS=$hot SEED_HOT_EXPERT_GIB=16 SEED_STEP_TIMING=1 \
      python3 server.py --model-path $MODEL --tp 4 --enable-graph-capture \
      --host 0.0.0.0 --port 30000 > server_hot$hot.log 2>&1 &
  RUN python3 -c "
import urllib.request, time
for _ in range(600):
    try:
        if urllib.request.urlopen('http://127.0.0.1:30000/health', timeout=2).status == 200:
            break
    except Exception:
        pass
    time.sleep(5)
"
  RUN python3 benchmark/run.py --base-url http://127.0.0.1:30000 --concurrency 48 \
      --output-json $SCRATCH/hotexp_${hot}.json
  pkill -f '[s]erver.py'
done
grep -a '\[step-timing\].*decode' server_hot0.log server_hot1.log | tail -20
```

Compare `[step-timing]` decode mean ms and `hotexp_0.json` vs `hotexp_1.json`'s throughput/TTFT
against the ~8.3 ms/step (~10-11%) prediction above.

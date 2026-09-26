# Prefill and mixed-step cost by component (TP=4, 4x MI300A), 2026-09-24

Measured with `prof_widths.py` under `rocprofv3 --kernel-trace`, split by `prof_split.py`:
eager steps (GEMMs BLAS-tuned at each width, as the server does), rank 0, flags_r6s. `pT` is
one fresh T-token prefill; `m48+W` is 48 decode lanes plus one W-token chunk. Numbers are
summed kernel ms per step. All-reduce time in an eager trace includes waiting for the slowest
rank (launch skew), so it is an upper bound, not the collective's own cost.

## Decomposition (kernel ms per step, rank 0)

Base = flags_r6s. Wide = + `SEED_MOE_HIP_WIDE=1`.

| case | MoE base / wide | dense GEMM | DeltaNet | attention | glue base / wide | all-reduce (bound) |
|---|---|---|---|---|---|---|
| p64 | 17.2 / 13.7 | 5.8 | 5.2 | 0.4 | 10.2 / 9.1 | 31 |
| p256 | 32.7 / 25.7 | 8.2 | 4.0 | 0.7 | 12.3 / 11.7 | 21 |
| p512 | 60.5 / 36.4 | 12.1 | 7.1 | 1.1 | 37.9 / 16.2 | 39 |
| p1024 | 78.7 / 57.2 | 19.9 | 13.3 | 2.0 | 45.3 / 21.3 | 27 |
| m48+16 | 16.4 / 11.8 | 9.5 | 3.8 | 0.2 | 16.4 / 15.2 | 35 |
| m48+208 | 49.1 / 25.8 | 24.1 | 5.2 | 0.7 | 19.6 / 17.6 | 20 |
| m48+464 | 59.9 / 36.4 | 24.7 | 8.9 | 1.0 | 43.7 / 22.5 | 37 |

Above 256 rows the base build leaves the HIP MoE for `prefill_moe` (Triton, 64-row blocks)
and the torch routing path (`topk`, scatter/gather, combine): that is the MoE and glue jump
at p512/p1024/m48+464. `SEED_MOE_HIP_WIDE` removes both.

## Floors per GPU

MI300A per GPU: 5.3 TB/s HBM; 980 TFLOP/s dense bf16 MFMA. gfx942 has no FP4 MFMA (FP8 is
1961 TFLOP/s but needs FP8 activations), so MXFP4 weights are decoded to bf16 and the matrix
peak is the bf16 one. Per rank per token: MoE 2.5 local assignments x 25.2 MFLOP x 60 layers
= 3.8 GFLOP; dense (2.05 G params per rank incl. shared expert and router) 4.1 GFLOP.

| component | bytes per step | FLOPs per step | floor at 256 / 512 / 1024 rows | measured (wide) |
|---|---|---|---|---|
| MoE | D(T) x 6.68 MB x 60; D = 115 / 122 / 128 distinct local experts | 3.8 GF x T | 8.7 / 9.2 / 9.7 ms (HBM) | 25.7 / 36.4 / 57.2 |
| dense GEMM | 4.1 GB | 4.1 GF x T | 1.1 / 2.1 / 4.3 ms (MFMA) | 8.2 / 12.1 / 19.9 |
| DeltaNet (45) | state 1 MB/layer | ~0.07 GF x T | < 0.2 ms | 4.0 / 7.1 / 13.3 |
| attention (15) | KV | small at these contexts | < 0.2 ms | 0.7 / 1.1 / 2.0 |
| all-reduce (120) | T x 8 KB each; one-shot reads 3 peers | - | ~3.8 / 7.5 / 15 ms (xGMI ~64 GB/s per peer) | eager upper bound only |
| glue | ~15 passes over T x 8 KB per layer | - | ~0.4 / 0.7 / 1.4 ms (HBM) | 11.7 / 16.2 / 21.3 |

MoE is HBM-bound up to ~2400 rows (compute floor 1.9 ms at 512 vs 9.2 ms of weight bytes), so
its cost per prefill token falls almost as 1/T once most local experts are touched. The HIP
kernel runs at ~2.1 TB/s effective per unit (~3.2 us per 16-token unit of 6.68 MB): its
per-unit decode (2.06 VALU per weight) plus 16x16x16 MFMAs at 2 waves per SIMD, not HBM, is
the limit. What remains after this change, largest first: MoE (2.5-6x its floor), all-reduce
(not overlapped with compute), glue and dense GEMMs (~5x), DeltaNet.

## MoE microbenchmark (one rank, layer 30, real weights and router)

`bench_moe_widths.py`, us per layer (60 layers = x60), rel. error vs an fp32 oracle:

| T | HIP (split) | HIP (flat, WIDE) | prefill_moe (grouped) | us/token base -> wide |
|---|---|---|---|---|
| 64 | 310 | 307 | 1009 | 4.8 -> 4.8 |
| 256 | 644 | 465 | 1364 | 2.5 -> 1.8 |
| 512 | 681 | 562 | 1430 | 2.8 -> 1.1 |
| 1024 | 768 | 758 | 1542 | 1.5 -> 0.74 |
| 2048 | 1337 | 1268 | 1779 | 0.87 -> 0.62 |

"base" per token is what flags_r6s runs at that width: HIP at <= 256 rows, `prefill_moe`
above. Max relative error: 0.21-0.48% for every kernel (the bf16 rounding of `inter` and
`y`); the flat split changes no arithmetic, only which program computes an item, and the host
emulation (`seed_tests/moe_hip_host_test.cpp`, built with `-DMOE_FLAT=1`) matches its
reference with 0 unwritten and 0 stray outputs at grids 228, 5 and 1000.

## Served ramp (same node, one lease, `lane_sub.sh.v6`, MAX_BATCH=64, levels 16/48/64)

Base = flags_r6s without `SEED_HIP_SKINNY_GEMM`, plus `SEED_MIXED_GRAPH=1
SEED_PREFILL_GRAPH_DN_CHUNKED=1` (integration-r7). Throughput tok/s (turn2+ TTFT p95 ms):

| config | C16 | C48 | C64 | peak |
|---|---|---|---|---|
| base | 483.5 (476) | 628.8 (1008) | 614.3 (1210) | 628.8 |
| + `SEED_MOE_HIP_WIDE=1` | 484.7 (375) | 644.3 (629) | 651.8 (897) | **651.8** (+3.7%) |
| + WIDE kernel only (base totals, budget 512) | 476.3 (449) | 644.6 (963) | 609.1 (2584) | 644.6 |

The kernel alone helps at C48; the wider mixed shapes add the C64 gain. Captured mixed
replays under WIDE (`SEED_MIXED_GRAPH_BENCH`): 48+1x208 67.8 ms and 64+1x192 68.2 ms against
decode b48 24.4 / b64 28.9 ms, i.e. ~0.2 ms per prefill token. In the ramp the 512-row shapes
ran ~120 ms with 240-330 live prefill tokens of 464 (padding): the next scheduler lever is
fitting chunks to shapes (or more totals between 256 and 512), since at these widths
GEMMs, glue, DeltaNet and all-reduce, not the MoE, set the per-row cost.

## Round 2 (lease 649845)

Paired ramp on one node (node06; nodes differ a lot: the same WIDE build peaked 651.8,
688.3 and 602.1 on three nodes), `lane_sub.sh.v6`, MAX_BATCH=64, levels 16/48/64:

| config | C16 | C48 | C64 | peak |
|---|---|---|---|---|
| r7 + `SEED_MOE_HIP_WIDE=1` | 441.1 | 598.5 | 602.1 | 602.1 |
| + `SEED_MIXED_FINE_TOTALS=1 SEED_DN_PREFILL_GLUE_FUSED=1` | 456.0 | 634.7 | 666.1 | **666.1 (+10.6%)** |

Mixed steps (`mixed_fill.py`, rank 0 step-timing samples of the same two ramps):

| config | real / replayed prefill rows | (wall - decode step) per real prefill token |
|---|---|---|
| WIDE | 0.55 | 0.462 ms |
| + FINE + GLUE | 0.71 (384-row shapes 0.64, 512-row 0.89) | 0.361 ms (-22%) |

Both flags were measured together only: the FINE-alone boot died with the node twice (see
below), so there is no separate FINE vs GLUE split in serving. Offline, the fused glue alone
saves 3.3 ms per step at 16-32 prefill rows and 6.5 ms at 480 (device time, 45 layers,
`bench_dn_glue.py`).

**Padding (`SEED_MIXED_FINE_TOTALS`).** A first set of 13 totals (51 mixed graphs per rank,
up to 1024 rows) exhausted MI300A's unified memory during capture: NODE_FAIL on node07
and node08. The committed set is 9 totals, max 512 rows, 35 graphs. The 64/128-row
steps stay at 11-19% fill: they are the 5-token chat-suffix chunk every turn ends with
(published as its own chunk for the session-cache boundary), and cost ~30 ms over a decode
step each. Removing that step (a boundary state snapshot inside a longer chunk) is the next
padding lever.

**Glue (`SEED_DN_PREFILL_GLUE_FUSED`).** One Triton kernel for the DeltaNet prefill glue
(conv window, conv + silu, q/k/v split, beta, g, conv-state write-back); the chunked kernels
read key-head q/k (no `repeat_interleave`). Per layer device time: 1x32 137.5 -> 63.9 us,
1x480 323.6 -> 179.1 us.

**Dense GEMMs** (`bench_dense_widths.py`, one GPU; bf16 weights, not MXFP4). hipBLASLt
tuned solutions reach 190-350 TFLOP/s on the big projections at 256-1024 rows (20-35% of the
980 peak); a 10x longer TunableOp budget finds nothing better. The default heuristic is
2-9x slower on the narrow-N shapes (kv_proj, router, shared expert), so every served width
must be tuned (FINE tunes all mixed widths; cold tuning of 4 new widths took ~12 min at
boot, cached after). Sum per layer set: 256 rows 219 us tuned vs 32 floor, 512: 305 vs 63,
1024: 451 vs 125. Closing that needs a different GEMM kernel, not tuning.

**All-reduce** (`bench_custom_allreduce.py --rows`, graph replay, 4 ranks, per call):

| rows | one-shot | RCCL (fp32 upcast, today above 256 rows) |
|---|---|---|
| 256 | 58 us | 69 us |
| 512 | 100 us | 110 us |
| 1024 | 177 us | 177 us |

So transfer is ~7 / 12 / 21 ms per step (120 calls) at 256 / 512 / 1024 rows; the eager
trace's 20-90 ms is mostly waiting for the slowest rank. `SEED_AR_WIDE` (one-shot up to 1024
rows) saves ~1 ms per 512-row step; not ramped: the one boot with it (plus FINE's 51 graphs)
hung in mixed-graph validation before the first NODE_FAIL, so it stays off and unproven.
Overlapping the all-reduce with the next GEMM was not attempted.

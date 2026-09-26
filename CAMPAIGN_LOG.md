# Bespoke Qwen3.5-397B-A17B on 4x MI300A: campaign log

Last updated: 2026-09-25 ~00:45 UTC.

## Round 14: target-closing bundle and final high-upside gates

- **The target-closing bundle is not eligible for promotion at C96.** Candidate
  `c4b793d6` starts from the measured suffix incumbent `2f012388` and adds ordinary
  captured-decode row sharding,
  captured-prefill row sharding, and the measured `2x128`/`1x384` prefill containers. The
  pre-run component model sums to a 7.614% service-time reduction, or 1,249.623 tok/s from
  the 1,154.477 tok/s incumbent. That is only 0.306 tok/s above the 1,249.317 target, so it
  remains a prioritization estimate rather than an acceptance result. The same-node control
  arm measured 1,078.4 tok/s with 435 completed turns and zero failures. Its sequential
  candidate arm captured and passed eager parity, then remained in uninterruptible work for
  more than five minutes after Gloo initialization and never opened its health endpoint.
  A fresh candidate-only retry again passed all decode buckets and all 12 prefill shapes on
  every rank; it was stopped at the campaign wrap-up cutoff two minutes after Gloo, before
  either health or a second confirmed stall. Therefore no candidate throughput or target
  claim is valid, and the bundle must not be promoted. The preserved artifact is
  `/path/to/home/target_closing_wrap_651079_results.tgz`, SHA-256
  `5b2d594009eb2cd0b42e23b56c07e2c46001d736b3a43d5f725e530139812366`.
- **Captured-prefill row sharding passes its GPU gate.** Commit `bda9ef52` keeps the residual
  row-sharded between the two tensor-parallel collectives per layer and restores full
  normalized rows for projections. All 12 production shapes, including `2x128` and
  `1x384`, passed captured-versus-eager checks on all four ranks. Same-process captured
  replay changed `1x512` from 95.875 to 90.161 ms and `2x256` from 94.175 to 88.092 ms,
  saving 5.714 and 6.083 ms respectively. The log is
  `/path/to/home/prefill_sp_651079.log`, SHA-256
  `447149a1ed2bf97cc4433641d8cc7c3882b24fcbff3ff4efdb043862cf1262fa`.
- **Speculative decoding is closed.** Exact per-time routed execution in `5470c9af`
  restored B48 k=2 parity over 128 tokens and achieved 2.5819568 accepted tokens per round.
  Its eager round took 202.0635 ms, or 78.26 ms per accepted token, versus 80.088 ms for
  ordinary eager decode. The captured path completed B1/B2/B4 capture but reproducibly hit
  a HIP illegal memory access on its first replay/validation, including under serialized
  launch. Even before the exact-router cost, the historical captured round was 54.6-55.9
  ms versus a 42.85 ms break-even. Further capture debugging has no credible target-closing
  payoff.
- **The hot-expert BF16 cache is rejected.** In the held-out B96 layer-30/rank-1 gate, the
  current routed kernel took 0.07120 ms/layer. H8, H11, and H14 took 0.36306, 0.42840, and
  0.48110 ms/layer, a 5.1-6.8x regression, despite 79.55-86.36% held-out local coverage and
  100% row-argmax agreement. The caches cost 11.25, 15.47, and 19.69 GiB per rank. The log
  is `/path/to/home/hot_gate_dd254574_651005.log`, SHA-256
  `f0deb1c7407f3173b215248675c29573c892918ca14bd76125de45c05d4c54e5`; JSON SHA-256 is
  `353ca79d4610badf7bb4809c8a5522710cf568f46e32da82d7b70f50bb0e00bd`.

## Round 13: suffix incumbent and rejected next vectors

- **The incumbent is source `2f012388` with folded turn suffix at C96:**
  **1,154.477 tok/s**, versus the **1,249.317 tok/s** terminal target. The
  remaining gap is 8.21%.
- **C112 is rejected by measured throughput.** Both the sparse decode ladder
  and full ladder captured and ran safely, but produced 999.923 and
  998.853 tok/s respectively. Extra concurrency did not offset its longer
  decode cycle.
- **Current speculative decoding is not competitive.** The exact full-step MoE
  path reached 2.58196 committed tokens per lane per round, but making target
  MoE row-semantics exact added as much as 10.365 ms per round, and the eager
  end-to-end probe was 5.8% slower than ordinary decoding. The first B48 target
  divergence is the routed expert output; router, top-k, shared expert, and
  pre-MoE hidden rows agree. The routed-only `cdd3d9af` helper remains nonexact,
  so no speculative branch is eligible for a service ramp.
- **Turn-delta ingestion is rejected.** Across 282 eligible cached follow-up
  turns, the new-token delta was at least 151 tokens, median 368 tokens, and no
  turn was at most 128 tokens. A short-delta path cannot activate enough to
  close the service gap.
- **The refreshed captured-prefill profile is usable; its decode trace is
  not.** Same-process observer ratios were 1.004-1.008 for `1x512`,
  `2x256`, and `4x64`. Rank-0 calibrated exclusive phase times in milliseconds
  were respectively: MoE 38.950/38.388/27.810, collectives
  15.971/16.618/13.422, dense 15.167/14.746/9.222, DeltaNet
  4.850/5.428/2.431, attention 5.725/4.156/1.545, and glue
  18.496/18.136/13.504. ROCTracer expands one graph replay even when the timed
  scope submits two; the corrected analyzer accounts for that deduplication.
  The remaining unattributed rank-0 time is 0.776/0.184/0 ms. The B96 decode
  observer ratio was 10.42x and its phase shares are rejected. The corrected
  artifact is `/path/to/home/c96_profile_2f_651044_results_corrected.tgz`,
  SHA-256
  `8bd5e49f1a0d21ea4cc9d1f1844d7ec0019790b28b5bff815b0ec8e3e26d64a4`.
- **Another paged-attention pass is rejected by ceiling.** Attention is only
  5.73 ms in the slowest measured `1x512` prefill, 5.7% of that phase and about
  2.6% of service at the measured 45% prefill share. Perfect removal cannot
  provide the required greater-than-8% end-to-end gain.
- **Wide MoE schedule and parallel-prep changes are rejected.** At T512 with
  122 distinct local experts, the production HIP schedule took 37.51 ms over
  60 layers. The best depth/grid arm took 36.73 ms, only 0.78 ms faster. The
  parallel work-list prep was bit exact, but saved only 0.027 ms over 60 layers
  at T512 and regressed prep at T256 and T1024. The schedule log is
  `/path/to/home/moe_t512_sweep_651044.log`, SHA-256
  `be552921ab6e4ebc06b8c4db5b46fa23755a9081d8d39edc9e66aadc43e6d6bc`;
  the prep log is `/path/to/home/moe_prep_651044.log`, SHA-256
  `014abf837f2c915e0448b902e381e1c234485d84360e4e3db3c9bacf1170a558`.
- **Wide one-shot all-reduce is rejected alone; row-sharded prefill remains a
  bundle component.** In exact 120-call captured graphs, plain wide one-shot
  changed T512 from 112.94 to 111.60 us/call. Row-sharded fused all-reduce,
  residual add, norm, and all-gather took 62.13 us/call, saving 6.10 ms over
  120 calls. It was bit exact on every rank before and after 30 graph replays,
  but misses the 8 ms standalone gate. The log is
  `/path/to/home/ar_wide_prefill_651044.log`, SHA-256
  `af3d3bf6edce15b32273abfc9bba3408a3ebfb7eba4dcfefc46dc8996bf905b5`.

## Round 12 takeover: corrected baseline and active evidence

- **The current terminal target is 1,249.317 tok/s.** Load-ramp v6 recomputes
  each concurrency level after all workers drain. Applied to the retained SGLang
  turns, it moves the peak from the right-censored v5 value of 883.204 tok/s to
  **961.013 tok/s at C48**. The terminal threshold is 30% above that corrected
  peak. All 883/1,148 tok/s targets below are preserved historical decisions and
  are superseded by this target.
- **The valid incumbent is 1,055.537 tok/s at C96**, leaving an 18.36% gain to
  the target. It passed 1,074/1,088 teacher top-1 and 5/5 legacy greedy checks,
  with zero failed turns. Turn-2+ TTFT p95 was 501.64 ms and TPOT p95 was
  102.815 ms. Sampled timing, calibrated by a +1.27% instrumented versus
  uninstrumented delta, attributes about 45% of the critical path to prefill and
  55% to decode. A single-phase solution therefore needs about 1.525x faster
  prefill or 1.392x faster decode.
- **C112 and C128 are rejected at the capacity gate.** C112 with sequence length
  8,192 became unresponsive during graph initialization. C128 with sequence
  length 4,096 loaded all shards and completed BLAS tuning, then reached 85-88%
  memory during graph capture and stalled all workers. Neither attempt reached
  correctness or a throughput ramp. MI300A host and device allocations share
  one pool, so nominal free VRAM did not make either configuration safe.
- **Vectorized packed DeltaNet prefill is correct but loses to the captured
  path.** Scalar to vectorized eager timings were 282.67 to 179.21 ms at 4x128
  (1.58x), 410.57 to 226.28 ms at 8x64 (1.81x), and 750.40 to 316.96 ms at
  16x32 (2.37x). It passed top-1 parity for prefill and the next four tokens;
  logit correlations were 0.999393 and 0.998877, and state correlations were
  0.998766 recurrent, 0.996117 convolution, 0.997949 K, and 0.996476 V. The
  production captured 4x256 path takes 134.03 ms for 1,024 tokens, so the best
  eager vectorized result remains 2.67x slower per token. Do not enable
  `SEED_PACKED_PREFILL_VEC` on this evidence.
- **The B96 decode trace is directional, not an exhaustive service profile.**
  Its token positions and resulting routing come from fixed synthetic inputs.
  Observer ratios of 1.070-1.081 pass the 10% calibration gate, but
  `SEED_MOE_PREP_FORK` launches work on a child stream that the trace missed,
  including 120 of 330 dense launches. The calibrated rank-local trace
  attributed 37-42% of decode to MoE, but the corrected production estimate is
  13-16 ms per B96 decode step and 15-18% of service time.
- **Another decode MoE kernel pass is rejected by the corrected roofline.** A
  B96 step reads about 7.80 GB per rank, for a 2.23-2.36 ms HBM floor at the
  measured bandwidth and arithmetic intensity 66.9 FLOP/byte. Closing the full
  terminal gap through MoE alone would require about a 6.8x phase speedup. The
  prior V2 kernel produced only about 6%, so this family does not have a credible
  target-closing path under the current architecture.
- **Folding the turn-open suffix into overlapped decode is a paired 9.60% win.**
  `SEED_FOLD_TURN_SUFFIX=1` feeds the roughly five-token chat-template suffix
  through already scheduled decode steps instead of separate `1x16` prefill
  replays. On job 650888/node09, exact source `7afdc3fd`, the flag-off C96
  control measured **1,053.391 tok/s** and the flag-on repeat measured
  **1,154.477 tok/s**. Both had zero failed turns. The candidate completed 471
  turns; turn-2+ TTFT p95 was 1,047.04 ms and TPOT p95 was 94.12 ms. It passed
  the campaign gates at 1,074/1,088 teacher top-1 and 5/5 legacy greedy. The
  expanded record-only check moved from the inherited 10/14 to 9/14 because
  `pin09` first diverged at token 6; this is recorded numerical drift between
  packed prefill and sequential decode, not a campaign gate failure. The feature
  remains default-off and is enabled only by the experiment environment until
  the deployment flags are updated. The paired
  artifact is `/path/to/home/fold_suffix_pair_650888_results.tgz`, SHA-256
  `e399e134d354fd0e502a3cb354196c9d8c1543d4191b2c1138767a2b92fd5542`.
  This result is 20.13% above corrected SGLang and leaves 8.21% to the terminal
  target.
- **Speculative decoding remains under diagnosis.** Fixes have removed frozen
  verify context width, stale hidden scratch, graph-unsafe rollback, and
  graph-unsafe routing. The minimal capture bisection now reaches the real MTP
  draft MoE and still fails HIP graph capture there. Job 650710 is splitting
  that path into smaller components before another full-model run.
- **Captured prefill profiling is the other active path.** A calibrated harness
  on `bespoke/graph-prefill-profile` is starting on job 650888 after recovery
  from the C128 stall. It will measure the production graph path by phase before
  selecting a prefill kernel or scheduling change.
- **Row-sharded residuals for ordinary captured decode are rejected.** The
  `SEED_AR_SP` microgate was bit exact on all four ranks for every row count from
  8 through 96, before and after 120-call graph replay. At B48/B80/B96 it saved
  only 2.45/9.67/13.76 us per collective against the current fused
  all-reduce, add, and norm kernel. Across the decode step's 120 collectives,
  the B96 ceiling is 1.65 ms, about 3.9% of the 41.95 ms decode phase and 2.1%
  of service time. B8/B16/B24 regressed and B32 was neutral. No full-model A/B
  was run. The diagnostic source is `61233b8a`; the microgate log is
  `/path/to/home/decode_ar_sp_micro_650931.log`, SHA-256
  `2271f3ca7094a3f190e20ecd904958cbc99a62ac27d17df5f9c81bbb1c5d55c6`.
- **Adding `2x128` and `1x384` captured prefill shapes is rejected at the 3%
  service gate.** All four tested shapes matched their uncaptured replay and
  eager prefill contracts on every rank. In the exact 60-second C96 window,
  117 calls with 257-384 new tokens moved from `1x512` (84.53 ms median) to
  `1x384` (71.24 ms), projecting 1,599.27 ms saved. Five arrival-order short
  pairs moved from `2x256` (79.36 ms) to `2x128` (61.58 ms), projecting another
  88.93 ms. The combined 1,688.21 ms is 2.814% of service time; the short-pair
  grouping is modeled rather than an authoritative scheduler trace, so it does
  not justify a paired ramp. The diagnostic source is `39303d89`; the log is
  `/path/to/home/prefill_shapes_gate_650931.log`, SHA-256
  `fe6ad567ef091f6285395bf462ad463b0c89c84c5a1f8ac58d2c59f613626dbb`.
- **The C112 BLAS cache seed was mutated during the prefill-shape diagnostic.**
  The run pointed `SEED_BLAS_CACHE_DIR` at the seed itself and rewrote the four
  rank caches while adding M=384 tuning. That directory is no longer a pristine
  C112 seed. Its frozen diagnostic-only copy is
  `/path/to/home/c112_cache_seed_diag_650931_20260924.tar.gz`, SHA-256
  `7c10abe68911794158d10d6bd2739edef04632470e430512ae52b4998ee98708`;
  its internal manifest has SHA-256
  `b9df0577deaf6eb96b2a46c0e91031afde4c4a9b2813da7a7b58d880be4facc1`.
  Future runs must copy a named seed into an isolated per-run directory before
  enabling tuning.

The sections below are chronological history. Their targets and active-job
statements describe the state at the time and are not current guidance.

## Round 11 evidence in progress

- **No throughput win is established yet.** The terminal target remains at least
  **1,148 tok/s** with the full correctness contract.
- **Higher-row prefill bursts are rejected.** Tiering equal-area bursts through
  `8x128` and `16x64` reduced burst count by 6.2% and raised mean fill from 450.2
  to 479.8 tokens, but burst wall per token regressed 4.2%. The tiered activation
  measured **637.1 diagnostic tok/s** and improved total scheduler wall by only
  **0.5%**, below the 10% continuation gate. Commit `0a71d216` records the result.
- **Chat-history reuse v1 was inactive.** Its unpaired activation measured
  **814.2 tok/s**, only **+3.7%**, and cache counters showed that the next turn
  reused prior prompt tokens but none of the preceding generated tokens. Treat
  this as diagnosis, not a performance win.
- **Exact-token history root causes are now fixed in source.** Commit `5d25672c`
  added generation-template normalization, but the production template's empty
  transient path disabled it. Commit `9a49890c` retained exact generated token
  ids because streamed text need not re-encode to those ids, but registry lookup
  still depended on successful normalization and stop-finished turns were not
  registered. Commit `073c7eef` accepts the empty-transient case, tries the raw
  template under strict prefix checks, records the visible ids from deferred
  stop completions, and adds sparse activation counters. These commits are not a
  throughput result until the live gates below finish.
- **C96 is inconclusive after node failures.** The decode microbenchmark captured
  batch 96 and matched eager, but both server attempts on node10 ended in a
  Slurm node failure before an end-to-end result. No C96 throughput or stability
  claim is valid.
- **The bounded C80 memory plan is safe at activation.** With max batch 80,
  max sequence 8,192, 320 snapshots, and KV headroom 1.0, each rank allocated
  9.38 GiB of KV and 14.61 GiB of snapshots while leaving 46.52-46.64 GiB
  unallocated. Decode buckets through 80 captured and matched eager. The 14-pin
  check reproduced the inherited campaign baseline, 10/14 exact with `pin14`
  diverging at token 1; this is not evidence that the batch-80 change regressed
  batch-1 output. The runner stopped before its C64/C80 ramp, so it provides no
  throughput result.
- **Validation is active.** The isolated C64 exact-history source `3f6bf0d2` is
  running as step `650294.27` from node-local
  `/dev/shm/history_exact2_650294`. Terminal integration `073c7eef` is running
  on job 650291/node11 from node-local storage with max batch 80, sequence
  8,192, 320 snapshots, exact history and `SEED_AR_SP=1`; mixed graphs, mixed
  batching, prefill bursts, and WIDE2 are disabled. Its fail-closed sequence is
  the three-turn history preflight, the expanded 14-pin record, the campaign
  gate (at least 1,074/1,088 teacher top-1 and 5/5 legacy greedy matches), then
  the C80 quick. None of those live gates had completed when this entry was
  written. Replacement jobs 650307 and 650525 remain pending behind cancelled
  predecessors.

## Round 10 takeover

- **Terminal target:** at least **1,148 tok/s**, 30% above the recorded SGLang
  load-ramp v5 result of 883 tok/s, with the full correctness contract intact.
- **Provisional incumbent:** `bespoke/integration-r10`, based on
  `bespoke/opt-ar-overlap` at `838afb4b`. `SEED_AR_SP=1` passed the recorded
  accuracy checks and won two same-node pairs by 5.6% and 6.1%.
- **Measurement rule:** rank hypotheses from calibrated production-path
  profiles and separate hardware, architecture, and hypothesis ceilings. A
  profiled result perturbed by more than 10% is activation evidence only.
- **Round decision:** reject the combined phase-separated prefill plus NG2
  candidate. On job 650294/node12, the incumbent measured **784.9166 tok/s**
  and the candidate measured **790.0208 tok/s**, only **+0.65%**. The candidate
  passed the manual probe, quick check, full 16/48/64 ramp, latency guardrails,
  and full accuracy: 98.7132% teacher top-1 and 5/5 identical greedy outputs.
- **Latency at the candidate C64 peak:** TTFT p95 1,994.37 ms, turn-2+ TTFT p95
  2,066.44 ms, TPOT p95 67.658 ms, and zero failed turns.
- **Rejected this round:** selective M384 dense dispatch. The current merged
  TunableOp cache already covers M384. Its perfect dense-to-hardware-floor
  ceiling is 3.91% overall, below the predeclared 5% experiment gate.
- **Queued replacement leases:** Slurm jobs 650306-650309, four GPUs each on
  `mi300`, are pending on dependencies after the 650291-650294 evidence runs.
- **Integration state:** no Round 10 candidate is promoted. The phase-separated
  prefill, NG2, and MTP source branches remain separate and unmerged.

### Round 10 measurements and decisions

#### Phase-separated prefill

- A shape-faithful microbenchmark supported the mechanism: seven mixed replays
  carrying 1,024 real prefill tokens took 408.43 ms, while seven decode-only
  replays plus one captured 4x256 prefill replay took 335.31 ms. This is an
  ideal equal-work throughput gain of **21.81%**. The 4x256 prefill replay was
  134.03 ms, below the predeclared 223 ms latency gate, and matched the eager
  reference (logit correlation 0.99880, state correlation 0.99647-0.99891).
- Production activation did not create the microbenchmark's full bursts. Across
  193 bursts it processed 84,382 useful tokens, **437.2 tokens per burst** on
  average (median 428, p95 779, maximum 1,023). No burst reached the 1,024-token
  target. Average fill was 3.77 rows and 165/193 bursts used all four rows, but
  row chunks were usually shorter than 256 tokens. The scheduler age limit
  forced 122/193 bursts (63.2%); burst latency was 134.5 ms median, 143.6 ms p95,
  and 170.9 ms maximum.
- Raising the age limit from seven to 18 decode cycles was rejected before a
  ramp. Of the 122 age-forced bursts, 94 already had four rows. The remaining
  28 partial bursts accounted for only 2.062 s of 68.185 s scheduler wall time.
  Even eliminating all of that time gives an impossible upper bound of
  **+3.1%** throughput. Waiting cannot lengthen an existing capped row or add a
  fifth row, and would mainly postpone admission. Keep target 1,024 and age 7.

#### Wide-MoE reuse

- NG2 groups at most 32 assignments per expert. At p512 it reduced the slowest
  rank's MoE time from 423.6 to 314.2 us per layer, saving 6.56 ms over 60
  layers, a 5.9% step ceiling. At p1024 it reduced 786.4 to 552.1 us per layer,
  saving 14.06 ms, a 10.0% step ceiling. Both shapes had zero bitwise
  mismatches. The small +0.65% combined ramp result shows these isolated wide
  shape gains do not translate into enough production wall-time reduction.
- NG4 was rejected at the kernel gate. Four groups increased p512 from 428.7 to
  554.8 us and p1024 from 766.0 to 972.4 us, despite zero mismatches. Added VGPR
  pressure and LDS barriers outweighed further weight reuse.

#### MTP feasibility

- At C48, the captured decode control was about 18.74 ms. Forced-acceptance MTP
  replay times were 39.495 ms for k=1, 53.307 ms for k=2, and 63.944 ms for k=3;
  including host fill and readback, round times were about 40.82-41.91 ms,
  54.60-55.92 ms, and 65.39-66.54 ms respectively.
- Those costs require about 1.18-1.24 accepted drafts per lane for k=1, which is
  impossible, 1.91-1.98 for k=2 (95.7-99.2% of draft positions), and 2.49-2.55
  for k=3 (83.0-85.0%). The harness forced oracle drafts and therefore measured
  100% positional acceptance; that result is a timing ceiling, not live model
  acceptance evidence.
- The live-acceptance server path produced output corruption and was parked.
  Do not use its output for accuracy or throughput claims. Resume MTP only after
  fixing the live state/commit path and re-establishing clean output before a
  load ramp.

The original status and speed-of-light model below describe the r9 handoff and
remain the comparison history for this round.

## Status

- **Target.** Beat SGLang's load-ramp v5 peak goodput of **883 tok/s**. That run used the tuned SGLang stack with MTP speculative decoding, k=3. SGLang is a score to beat only: we do not profile it and do not copy its code.
- **Best build: `bespoke/integration-r9` (de69eeda).**
  - Paired on one node, r9 beats r8 by 8.8% (632.5 → 688.4 tok/s).
  - On the faster node where r8 reached 751, that ratio projects to about 815 tok/s, roughly 92% of SGLang.
  - Accuracy is unchanged: teacher-forced top-1 98.71%, and 5/5 greedy prompts are identical.
- **Estimated speed of light (SOL).** About 8,000 tok/s ideal and about 5,000-6,000 tok/s practical at C48 (derivation below).
  - r9 is roughly 13-16% of practical SOL.
  - SGLang is about 16%.

## Speed-of-light estimate

**Method.** Per-rank bytes and FLOPs per step, using sizes from our own measured shape inventory.
- **Ideal:** HBM 5.3 TB/s, bf16 MFMA 980 TFLOP/s, all-reduce latency ~10 µs.
- **Practical:**
  - bandwidth: 3.5 TB/s, the best plain cold streaming read measured on this part;
  - compute: MFMA at ~60%;
  - all-reduce: 17 µs per call, our fused one-shot kernel at 48 rows.

**Per-rank sizes (TP=4, 128 local experts per rank):**

| Item | Size | Source |
|---|---|---|
| Dense bf16 weights (in_proj, out_proj, q/k/v/o, router, shared expert) | 4.1 GB | dense shape inventory, opt-dense-gemm-hip |
| lm_head shard | 0.5 GB | 62080 x 4096 bf16 |
| One MXFP4 expert | ~6.7 MB | MoE floor measurement, opt-moe-bw2 |
| All local experts, 60 layers | ~51 GB | 128 x 6.7 MB x 60 |
| DeltaNet state, one request, 45 layers | 45 MB fp32 | 16 v-heads x 128 x 128 x 4 B per layer |
| KV, one token, 15 attention layers | 15 KB | 1 KV head x 256 dim, K+V bf16 |

**Workload.** Sessions have 3-6 turns. Measured at C48, prefill work is about 2.3 new prompt tokens per generated token. That ratio comes from TPOT (~65 ms) minus the decode step (~24 ms), divided by the marginal mixed-step cost per prefill token.

**Decode step at b48 (48 tokens out):**

| Component | Bytes / work | Ideal ms | Practical ms | Measured ms (r8, rocprof) |
|---|---|---|---|---|
| Dense + lm_head | 4.6 GB | 0.9 | 1.3 | 4.2 + 0.9 |
| MoE, ~16.5 experts/rank/layer touched | 6.6 GB | 1.25 | 1.9 | 10.4 |
| DeltaNet state read + write | 4.3 GB | 0.8 | 1.2 | 2.8 |
| KV reads, ~2k avg context | ~1.4 GB | 0.3 | 0.4 | 0.9 |
| All-reduce, 120 calls | latency | 1.2 | 2.0 | 4.9 |
| Glue, routing, launch | 0 | 0 | 0 | 1.8 |
| **Total** | | **4.4** | **6.8** | **~24** |

**Prefill.** It is about 8 GFLOP per token per rank: 4.1 GFLOP dense plus 3.8 GFLOP of active MoE.
- A wide step that touches every local expert reads about 56 GB/rank.
- So a 1024-row prefill step is balanced between bandwidth and compute: ~10.6 ms ideal, ~14.3 ms practical.
- That works out to ~10 µs/token ideal and ~14 µs/token practical.
- We measure 0.2-0.36 ms per real prefill token in mixed steps, which is 15-25x practical.

**Workload SOL at C48.**
- Each 48-token decode cycle needs ~110 prefill tokens.
- Ideal: 4.4 + 110 x 0.0107 = 5.6 ms per 48 tokens, about 8,600 tok/s.
- Practical: 6.8 + 110 x 0.014 = 8.3 ms, about 5,800 tok/s.

**Structural consequence.** This SOL assumes large prefill batches of ≥1024 rows, run separately from cheap decode steps. A mixed step of 48 decode + ~110 prefill rows touches ~96% of the experts. So it re-reads almost the whole ~56 GB model every step, a practical floor of ~20 ms per 48 tokens (~2,400 tok/s).
- Mixed steps (Sarathi-style) won today because our wide kernels are far from their floors.
- Once the wide MoE is near its floor, fewer and fatter prefill steps should beat mixing.
- The guardrails leave room for this: TPOT p95 ≤ 250 ms (we run at ~65-90 ms) and turn-2+ TTFT p95 ≤ 10 s.

## Campaign history

All numbers are load-ramp v5 peak goodput. Pairs marked "paired" ran back to back on the same lease, because node-to-node variance is ±10%.

| Build | Peak tok/s | % SGLang | What it added |
|---|---|---|---|
| Initial bespoke (2026-09-23) | 12.8 | 1% | Serialized prefill; full-history re-prefill starved decode |
| Prefix-cache fix d08fbec2 | 55.6 (C16) | 6% | Turn-2+ prefix hits |
| integration-r3 43f96ce8 | 337.1 | 38% | HIP decode graphs, HIP MoE, overlap scheduler, grouped prefill MoE, chunked DeltaNet prefill, split-K decode attention, deferred turn close |
| integration-r4 4023d4e0 | 448.6 | 51% | Prefill graphs (validation fix), fused in_proj, BLAS tuning per bucket, in-place DeltaNet state |
| integration-r5 9f4fc1d7 | 548.1 | 62% | Custom one-shot all-reduce, fused routing, fused RoPE+KV write, prefill shape fitting, max-seq 8192 |
| integration-r6 | 564 | 64% | Bit-exact glue fusions (add+RMSNorm, silu*mul, gate, DN norm) |
| integration-r7 10fa347d | 659 | 75% | Graph-captured mixed decode+prefill step, chunked DeltaNet inside graphs; skinny GEMM off |
| + opt-decode-fusion3 | 681.7 | 77% | All-reduce + residual add + RMSNorm in one graph-safe kernel (bit-exact) |
| integration-r8 27b26b30 | 751.0 (paired: 689.7 → 751.0) | 85% | Wide-width HIP MoE, rank-agreed graph validation, min-merged per-rank GEMM tuning (rank 0 b16 dense 6.5 → 4.3 ms), vocab-parallel lm_head |
| integration-r9 de69eeda | 688.4 on a slow node (paired: 632.5 → 688.4, +8.8%); ~815 projected | ~92% (proj.) | Finer mixed totals (fill 0.55 → 0.71), fused DeltaNet prefill glue, MoE routing prep on a side stream |

### Tried and parked

| Item | Result | Why parked |
|---|---|---|
| HIP skinny GEMM for decode | -5% peak with mixed graphs | Loses on mixed widths |
| MoE decode kernel V2 / fused persistent launch | Bit-exact, ramp-neutral | Streams at ~2.8 of ~3.5 TB/s per wave; the rest is 6-13 µs fixed cost per phase |
| Expert load balancing (EPLB) v1/v2 | ~0.2 ms/step, mostly from the side-stream fork | The assignment kernel costs about what balancing saves; 9 GiB of replicas crashed nodes because the pool planner ignores them |
| One-kernel DeltaNet decode | Faster at b1, slower at b48 | Serializes on cross-program hand-offs |
| Custom HIP dense GEMM | 0.6-0.9x tuned hipBLASLt | Stuck at ~2 TB/s weight stream; root cause not found (see open levers) |
| MTP speculative decoding (opt-mtp2) | Captures and validates; verify round is 4.3-8.4x a b48 decode step | Wide MoE cost grows with rows; also a 17 ms accept+commit and a concurrency crash. Resume after the wide MoE lands |
| max-batch 96, hot-expert dense path, eager mixed batch | 393 / 58 / 50 tok/s | Superseded |

### Process lessons

- Only paired runs on the same lease count. The same build peaked at 602, 652 and 688 on three nodes.
- Boot-time GEMM tuning can hang, so use a pre-merged tuning cache (`blascache_merged`).
- Large captured-graph sets (51 mixed graphs up to 1024 rows) crashed nodes. MI300A memory is unified, so host RSS and HBM share one pool.
- Every per-rank decision (graph validation, GEMM algorithm) must be agreed across ranks. Otherwise ranks diverge and hang in the all-reduce.
- Flush caches in microbenchmarks by reading, not with `fill_`. Dirty lines halve measured bandwidth.
- Pass flag sets to launch scripts as files, not through shell string substitution.

## Work in flight

Payoffs are estimates relative to r9, and are not measured yet.

| Workstream | Branch | Lever | Expected payoff |
|---|---|---|---|
| Wide-width MoE | `bespoke/opt-moe-wide2` | Dequant each expert's weight tile once and reuse it for all its tokens. MoE is ~26 ms of a 48+208 mixed step against a ~9 ms floor. | +10-20%. Also the precondition for MTP and for separate large-batch prefill |
| All-reduce / compute overlap | `bespoke/opt-ar-overlap` | Two micro-batches per mixed step, so one half's all-reduce hides behind the other half's compute. AR transfer is ~7-12 ms per 256-512-row step. | +4-7% |
| Fold the turn-close suffix | `bespoke/opt-turn-suffix` | The ~5-token end-of-turn chunk rides a nearly empty wide step (11-19% fill, ~30 ms over decode) once per turn. Snapshot the state inside the previous chunk instead. | +3-8% |
| Boot time | `bespoke/opt-boot-time` | Boot takes 11-19 min. Round 1 overlapped the HIP compile with weight load (1-2%). Round 2 targets a no-op cached BLAS phase (131-286 s today), cheaper graph warmup, and faster weight load. | No throughput change; faster iteration |

If the three throughput items land, r9 × ~1.2-1.4 gives ~980-1,140 tok/s, past SGLang's 883 but still under ~20% of practical SOL.

## Open levers toward SOL, largest first

1. **Prefill scheduling.** With a near-floor wide MoE, run large prefill batches (≥1024 rows) apart from cheap decode steps instead of re-reading the model in every mixed step. The estimated gap is 2-2.5x on the prefill share of time.
2. **Decode step: 24 ms measured vs 6.8 ms practical.**
   - MoE fixed costs (10.4 vs 1.9 ms).
   - All-reduce latency (4.9 vs 2.0 ms).
   - Dense GEMMs at ~1.7 TB/s against ~3.5 TB/s achievable streaming. hipBLASLt is not at the practical limit; our custom kernel failed, but the gap is real.
   - DeltaNet fp32 state traffic.
3. **MTP speculative decoding**, once the verify-width MoE cost is near flat from 48 to 192 rows.
4. **Higher concurrency**, if the benchmark exposes it. Decode rows cost ~0.2 ms each, and TPOT has ~3x headroom under the guardrail.

## Appendix: flag sets

- **r9.** `r1700/integ9/flags_r9.txt` on the test cluster. That is the r8 set plus `SEED_MIXED_FINE_TOTALS=1 SEED_DN_PREFILL_GLUE_FUSED=1 SEED_MOE_PREP_FORK=1`.
- **r8.** `flags_r6s.txt` minus `SEED_HIP_SKINNY_GEMM`, plus `SEED_MIXED_GRAPH=1 SEED_PREFILL_GRAPH_DN_CHUNKED=1 SEED_AR_RMSNORM_FUSED=1 SEED_MOE_HIP_WIDE=1 SEED_LMHEAD_VOCAB_TP=1 SEED_BLAS_TUNE_MERGE=1`.
- **Server args.** `--enable-graph-capture --max-batch 64 --max-seq-len 8192`.
- **Keep these off:** `SEED_AR_WIDE` (unproven), `SEED_HIP_SKINNY_GEMM`, `SEED_DN_DECODE_FUSED`, `SEED_MOE_EPLB`, `SEED_DENSE_HIP`, `SEED_MTP2`.

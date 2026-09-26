# Qwen3.5-397B-A17B Multi-Turn, From Scratch, 4x MI300A

Bespoke-system bundle: the agent builds its own serving engine (no SGLang, vLLM,
or TensorRT-LLM) for `amd/Qwen3.5-397B-A17B-MXFP4` and minimizes
`p95_ttft_turn2plus_ms` on a multi-turn chat workload. The workload is the v4
benchmark of the SGLang multi-turn task (PR #421), unchanged. See
`OBJECTIVE.md` for the task, the disallowed-engine-code rule, and the
candidate contract.

## Run

```bash
export MODEL_PATH=/path/to/Qwen3.5-397B-A17B-MXFP4
vibesys --input examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke \
  --runs-dir /work/vibesys-runs \
  --run-environment skypilot --backend rocm --cluster-profile <profile>
```

The execution image is described in
`examples/model-serving/images/rocm-mi30x-engine-free/`.

## Layout

| Path | Role |
| --- | --- |
| `vibesys.input.toml`, `objectives.toml` | Input manifest; metric `p95_ttft_turn2plus_ms` (min) |
| `OBJECTIVE.md` | Task, validity rules, candidate contract |
| `benchmark/run.py` | Workload driver; writes `--output-json`; hard-fails on any dropped, truncated, or errored turn |
| `benchmark/launcher.py` | Starts and stops `python3 server.py`, polls `/health` |
| `benchmark/test_run.py` | Hermetic tests (fake clock and server) |
| `accuracy_checker/checker.py` | 13 probes plus greedy-token pins; see its README |
| `accuracy_checker/make_pins.py` | Generates `reference/pins.json` from a transformers forward |
| `tp.py`, `tp_driver.py`, `deltanet_tp.py` | Tensor parallelism: the shard plan and collectives, the SPMD command protocol rank 0 broadcasts, and the DeltaNet value-head split with its derivation |
| `blas_tune.py` | Per-shape GEMM solution selection for the decode step's skinny (M = batch) dense projections; `SEED_BLAS_TUNE=0` to disable |
| `seed_tests/` | CPU tests for the seed itself: HF parity, MoE/delta-rule/attention parity, batching and prefix reuse, the sharding arithmetic and a real four-process gloo group (needs torch, not in the repo env) |
| `reference/` | Reference modeling code and `pins.json` |
| `requirements.txt` | Pure-python extras; torch, triton, AITER come from the image |

## Checks without a GPU

```bash
D=examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke
uv run pytest $D/benchmark/test_run.py -q --no-cov -p no:tach
uv run pytest $D/accuracy_checker/test_checker.py -q --no-cov -p no:tach
uv run vibesys validate $D
```

`seed_tests/` additionally needs torch, transformers and safetensors, and covers
the seed's own properties on CPU (HF parity on a tiny model, the vectorized MoE
and chunked delta rule against the implementations they replaced, batched decode
against one-sequence decode, batched decode-time DeltaNet against the old
per-slot loop, a cached prefix against a full recompute, eviction, the HTTP
contract under concurrency, and the tensor-parallel sharding both by hand and
under a real four-process `torch.distributed` group on the gloo backend):

```bash
<python-with-torch> -m pytest $D/seed_tests -q -o addopts= -p no:cacheprovider
```

Before the first real run, generate `reference/pins.json` once on the cluster
(see `accuracy_checker/README.md`); the accuracy gate fails without it.

## Seed status

The seed (`server.py`, `model.py`, `scheduler.py`, `prompt_cache.py`,
`mxfp4.py`, `weights.py`) is a plain PyTorch implementation written from the
reference modeling code: static per-slot caches, the model sharded across the
GPUs, experts kept in MXFP4. It matches the transformers reference on a tiny random
model on CPU (`seed_tests/`). Loading the real checkpoint, the 4-GPU split, and
its speed on ROCm have not been run. Expect the first measurement to be very
slow; the accuracy gate is the first thing to confirm on the cluster.

Changes on top of the original seed. Each is verified on CPU against the
implementation it replaced; none is measured on real hardware.

- **Fused MXFP4 grouped-GEMV (`mxfp4_gemv.py`), measured on 4x MI300A.** Two
  Triton kernels read the packed uint8 nibbles and e8m0 scales straight from
  the layout `load_experts` produces and dequantize in registers, so no
  dequantized weight ever reaches HBM. That was the dominant term in the decode
  roofline: the torch dequant chain moved ~37 bytes of traffic per logical
  weight value against the MXFP4 payload's 0.53 (see
  `ROOFLINE_BOTTLENECK_ANALYSIS.md`). Measured on one MI300A, one MoE layer at
  the real dimensions, against the path it replaces: **17x at 1 token, 27x at
  8, 21x at 48**, and device-to-host round trips per MoE call **12 to 0**.

  The second property matters as much as the speed. Both ways an assignment can
  be a no-op (this rank does not own the expert, or the routing weight is zero)
  are GPU-side early returns, so the MoE's launch shape is a function of
  `max_batch` and `top_k` alone. The routing is now the raw `[t * top_k]`
  assignment list: no de-duplication, no `unique_consecutive`, no boolean-mask
  indexing, no `int(counts.max())`. That is what made the step capturable, and
  `graph_decode.moe_static` is no longer a separate spelling of the MoE.

  Correctness, on gfx942: the in-register dequant is bit-exact against
  `mxfp4.dequant_mxfp4` over all 256 byte values and every live e8m0 scale, and
  at the real dimensions the fused output is closer to an fp32 oracle than the
  bf16 `bmm` path it replaces. One inert difference is pinned rather than
  papered over: e8m0 scale byte 0 gives `+0.0` instead of the subnormal
  `2**-127` (`seed_tests/test_mxfp4_fused_gemv.py`, which also runs on CPU
  under `TRITON_INTERPRET=1`). `SEED_FUSED_MOE=0` forces the torch path back at
  runtime, which is the A/B switch.

  The kernel was bound by its own dequantization arithmetic rather than by
  bandwidth or by tiling, which is what `DECODE_BOTTLENECK_2026-09-22.md`
  establishes and what the ~705 GB/s this bullet used to report was measuring.
  A sweep of `SEED_MOE_BLOCK_N`/`SEED_MOE_BLOCK_K`/`SEED_MOE_WARPS` at the real
  dimensions found the shipped tile already near-optimal and every larger tile
  worse (`BLOCK_N=128` or `BLOCK_K=256` regress by 30-80%, almost certainly
  register pressure), so the fix was to the decode, twice: first as bit-field
  arithmetic instead of a four-case magnitude select, then as a pair of
  `v_perm_b32` byte-table lookups (`_fp4_words_to_values`, gfx9 only,
  `SEED_MOE_PERM_LUT=0` to fall back). Measured on one MI300A at the real
  dimensions, 48 tokens, one MoE layer: **4.33 -> 3.30 -> 1.92-2.07 ms**, the
  `gate_up` inner loop **411 -> 183 vector instructions** (12.8 -> 5.7 per
  decoded weight value), and **~990 -> ~1,600 GB/s**. Deleting the decode
  outright while keeping every load is no faster than keeping it, so the kernel
  is no longer ALU-bound.
- **De-duplicating grouped MoE for prefill (`mxfp4_gemv.py`).** The kernels
  above give every (token, expert) assignment its own program, which is right
  at decode (batch 48 at top-10 touches almost 480 distinct experts, so there
  is nothing to amortize) and badly wrong at prefill (a 512-token chunk puts
  5,120 assignments on 512 experts, so each expert's weights are read about ten
  times). `_grouped_gate_up_kernel` and `_grouped_down_kernel` sort assignments
  by expert into `BLOCK_M`-row blocks, so a block reads its expert's tile once
  and applies it to up to `BLOCK_M` rows with one `tl.dot`. Measured on one
  MI300A at the real dimensions, against the per-assignment kernels:

  | tokens | assignments | per-assignment | grouped | speedup |
  | -----: | ----------: | -------------: | ------: | ------: |
  |    128 |       1,280 |       11.68 ms | 8.47 ms |   1.38x |
  |    256 |       2,560 |       22.68 ms | 9.37 ms |   2.42x |
  |    512 |       5,120 |       44.62 ms | 9.73 ms |   4.58x |
  |  1,024 |      10,240 |       87.38 ms | 16.36 ms |  5.34x |

  The grouped cost barely moves from 128 to 512 tokens because it is bounded by
  reading all 512 experts once, which is a constant. `PREFILL_CHUNK` is 512, so
  prefill hits the 4.58x row. `use_grouped` picks between the two paths on
  assignment density alone (Python ints, no device read); the measured crossover
  is ~2.5 assignments per expert and the threshold sits just above it.

  `align_blocks` builds the block table in fixed-shape torch with no host sync,
  which is the same constraint the per-assignment path meets: assignments this
  rank does not own and zero-weight rows go to a sentinel bucket instead of
  being filtered out, and `grouped_block_count` bounds the grid at
  `ceil(A / BLOCK_M) + E + 1`.

  This does not reach the ~67x the roofline report's TTFT projection assumes
  from de-duplication alone. It reads 10x fewer bytes but sustains only
  ~355 GB/s against the per-assignment path's ~755, because at ten assignments
  per expert a 16-row block is 62% live. `BLOCK_M` was tuned by measurement and
  the curve is not monotonic (32 is worse than both 16 and 64, an MFMA
  tile-selection artifact), so retune it per architecture rather than deriving
  it. Likewise `SEED_MOE_GROUPED_WARPS` is 4 while `SEED_MOE_WARPS` is 8: 8
  warps is 5x slower here and faster there, because `tl.dot` over a 16-row block
  leaves most of each MFMA tile idle.
- **Vectorized MoE (`model.py`, now the fallback).** `Model._routed_grouped`
  dequantizes and matmuls every activated expert in a layer as one batched op.
  Tokens are sorted by assigned expert into padded groups; the padded groups
  are the batch dimension of two `bmm` calls (gate_up, down), and
  `dequant_mxfp4` runs once over all activated experts' weights, not once per
  expert. No AITER MXFP4 or grouped-gemm kernel applies on gfx942 (MI300A); see
  `resources/skills/serving-systems/references/platforms/rocm/aiter.md`. This
  removed the original profiled bottleneck's per-expert Python loop, but padding
  to the batch's own largest per-expert token count wastes compute under skewed
  routing, and the dequantized copy it writes to HBM is what the fused kernel
  above exists to delete. It now runs only for dense checkpoints and on CPU.
  Checked against a per-expert-loop oracle on synthetic tensors
  (`seed_tests/test_moe_vectorize.py`).
- **SDPA attention and a cached rope table (`model.py`).**
  `Model.full_attention` calls `F.scaled_dot_product_attention` (same causal
  mask and scale, `enable_gqa` in place of `repeat_interleave`) instead of a
  manual `softmax(QK^T/sqrt(d))V`, and `Model.rope` slices a per-device cos/sin
  table built once (`_rope_table`) instead of recomputing it in every layer.
  Checked against the manual formulas in `seed_tests/test_opt_micro_sweep.py`.
- **Chunked DeltaNet (`model.py`).** The gated delta rule ran as a strict
  per-token Python loop, profiled at about 73 ms per layer at 512 tokens
  against 2 ms for a full-attention layer. Multi-token calls now use the
  chunked parallel (WY-representation) form: `DELTA_CHUNK` = 64 tokens per
  chunk, the within-chunk delta updates solved as one unit-lower-triangular
  system and applied as matmuls, with the recurrent state carried serially only
  from chunk to chunk. That is about 8x fewer sequential steps and a
  correspondingly smaller number of kernel launches and re-reads of the 4 MB
  per-layer state at a 512-token prefill. Single-token decode still takes the
  per-token path, which is the same arithmetic without the padding and the
  solve. `Model.delta_rule` keeps its signature and its in-place state update.
  `seed_tests/test_delta_rule.py` checks the two forms agree within
  `atol=1e-5, rtol=1e-4` across seeds and lengths (worst observed relative
  error about 1e-5), including sub-chunk lengths, chunk-boundary lengths, and
  prefill-then-decode state hand-off. **On real hardware (bespoke-opt-validate
  run, 4x MI300A), the chunked path is disabled.** It does not crash, but a
  chunked-prefill call of a 13-token prompt measured about 130s versus 13.6s
  for the same call forced through the recurrent path, reproduced across
  several distinct prompts and not explained by one-time kernel warmup.
  `Model.delta_rule` currently always calls `delta_rule_recurrent`;
  `delta_rule_chunked` and `torch.linalg.solve_triangular` are unused on the
  serving path until the ROCm root cause is found. The recurrent path is
  itself O(prompt length) per DeltaNet layer, so this trades the chunked
  path's pathological cost for the older, much-more-sequential prefill cost
  on long prompts.
- **Request batching and prefix reuse (`scheduler.py`, `prompt_cache.py`,
  `model.py`, `server.py`).** The server ran one request at a time.
  `scheduler.py` now admits many requests at once and runs one prefill chunk or
  one batched decode step per iteration over a pool of `max_batch` sequence
  slots; `Model.decode(slots, tokens, positions)` steps every resident slot in
  one pass, and `Model.sample_batch` turns the whole step's logits into tokens
  with a single device-to-host sync instead of one per request. Each finished
  turn keeps its state in its slot, so the session's next turn prefills only the
  new suffix; `prompt_cache.py` does the same for tokenization. Checked against
  the one-sequence path and a full recompute in
  `seed_tests/test_batched_decode.py`, `test_prompt_cache.py`,
  `test_scheduler.py` and `test_server_concurrency.py`.
  A slot's prefix snapshot (`Model.save_prefix`) is a row of a second state pool
  allocated in `Model.__init__`, not a buffer created on that slot's first save.
  The lazy version grew the resident footprint by one slot's recurrent state for
  every new session until all `max_batch` slots had been used, which on 4x
  MI300A showed up as GPU 0 climbing across an accuracy gate and then a HIP
  out-of-memory kill. Reserving it up front makes the footprint constant from
  the first request; the cost is that a `--max-batch` that does not fit now
  fails while loading instead of several sessions in.
  `seed_tests/test_state_memory.py` holds live tensor storage constant across
  250 turns and 12 sessions.

- **Batched decode-time DeltaNet (`model.py`).** `Model.deltanet_decode` used
  to loop over `slots` and call the single-sequence `deltanet` once per slot
  (T=1 each time), the last per-slot Python loop left in the batched decode
  step. It now gathers each active slot's conv and recurrent state out of the
  layer's pool (one row per slot in `slots`), runs `causal_conv` and
  `delta_rule_recurrent` once with that real batch dimension (both were
  already batch-generic over their leading dim, not hardcoded to 1), and
  scatters the updated state back into just those pool rows, the same
  gather-then-index-write pattern `attn_decode` uses for the KV pool. A slot
  not named in `slots` is never read or written, so it cannot be corrupted by,
  or corrupt, a batched call. Fewer Python-loop iterations and kernel launches
  per decode step; not measured on real hardware. Checked against the old
  per-slot loop (output and updated state) across several active-slot-count
  scenarios, including gaps and a slot evicted and reused mid-sequence, and
  against that slot decoded alone, in
  `seed_tests/test_opt_deltanet_batch_slots.py`.
- **Pipelined decode across the four devices (`pipeline.py`, `model.py`).** The
  layer split gives each device a contiguous run of layers, and a step that
  walked the layers in index order left three of the four idle at every
  instant: a hard 4x loss on work that is memory-bandwidth-bound on this
  hardware, which at decode is all of it. Each device is now a pipeline stage
  with its own thread, and a decode step is cut into
  `len(stages) * MICROBATCHES_PER_STAGE` microbatches of slots, so stage s
  computes microbatch m while stage s-1 computes m+1 and every device has work
  in flight. The only cross-device traffic is one activation copy per stage
  boundary (`[microbatch, 1, hidden]`), not a collective. A thread per stage,
  rather than one driver issuing stages in wave order, is what makes the
  overlap real: at the time it was written `Model.moe` read a device-resident
  group size back to the host once per layer, and on one thread every later
  stage would queue behind that read. The fused MoE kernel has since removed
  that read, so the thread-per-stage argument now rests on launch overhead
  alone. `SEED_DECODE_MICROBATCHES` tunes the count (0 restores the sequential
  walk); the default of 2 per stage leaves the fill/drain bubble at 27% of the
  wall time while limiting the per-microbatch launch overhead and the expert
  weights a narrower MoE call cannot share. `seed_tests/test_pipeline.py`
  checks the schedule with fake stages (stages really do hold different
  microbatches at once) and the model with four stages on one CPU, where a
  pipelined step is bit-identical (`torch.equal`) to the same microbatches run
  one after another. Verified on 4x MI300A: instrumenting `_decode_stage` with
  `torch.cuda.synchronize` on each side of the call and comparing wall-clock
  intervals across the four stage threads shows real cross-thread overlap
  (busy-time-summed / wall-clock ratio of 2.85-3.40x out of a 4x ceiling), not
  the fully serialized schedule a broken pipeline would produce.

  That overlap being real is what exposed a second bug, now fixed
  (`MIN_MICROBATCH_SLOTS`, `_decode_group_count`, both in `model.py`): the
  microbatch count above is a *ceiling*, `len(stages) * MICROBATCHES_PER_STAGE`
  (8 at the default), applied regardless of the step's actual slot count. A
  step at or below that ceiling got cut into that many microbatches anyway, so
  an 8-slot step -- exactly the ceiling -- became 8 microbatches of 1 slot
  each: nothing left to batch, only 4 full per-layer dispatch and
  per-microbatch MoE weight-read passes stacked back to back, which the ~3x
  overlap only partly hid. Measured on 4x MI300A, real checkpoint, max_batch
  48, prompt 32, mean of 3 runs each:

  ```
  batch   before (ms/tok, tok/s)   after (ms/tok, tok/s)   speedup
      1        107.04,  9                107.19,  9         1.0x
      8        844.77,  9                204.14, 39         4.1x
     16        795.26, 20                331.29, 48         2.4x
     48       1116.74, 43               1171.89, 41         1.0x (unchanged, by design)
  ```

  `_decode_group_count(n, ceiling, min_slots)` keeps at least
  `MIN_MICROBATCH_SLOTS` (env `SEED_DECODE_MIN_MICROBATCH_SLOTS`) slots in
  every microbatch, collapsing toward fewer, larger microbatches -- down to
  one undivided batch, as batch 8 now does -- instead of hitting the ceiling
  on a step too small to spend it on. Batch 16's row above also confirms the
  original curve's non-monotonicity (16 was faster than 8 pre-fix) was the
  split misbehaving, not a tuning curve: 16 slots over the same 8-way ceiling
  was 2 per microbatch, a less degenerate split than 8 slots at 1 per
  microbatch.

  The table above used the floor (6) this module had been tuned to since the
  pipeline was written -- 48 slots over 8 microbatches -- on the theory that
  it was "unaffected." A follow-up sweep of `MIN_MICROBATCH_SLOTS` at that
  same batch size, real hardware, mean of 3 runs each, found otherwise:

  ```
  min_slots  groups   ms/tok   tok/s
          6       8  1233.77     39   (the pre-existing split)
          8       6   992.23     48
         12       4   677.15     71
         16       3   577.30     83
         24       2   428.58    112
         49       1   311.15    154   (undivided)
  ```

  Monotonic all the way to undivided: 4.0x faster and 4.0x higher tok/s than
  the split this module shipped with, at the exact batch size that split was
  tuned for. The fused MXFP4 GEMV, batched attention and batched DeltaNet
  above all cut the per-layer cost the pipeline's fill/drain bubble existed to
  hide, while a narrower MoE call still re-reads more of the (now cheaper, but
  not free) expert weight traffic; at this model's shape, on this hardware,
  splitting no longer pays for itself at any batch size up to 48. The default
  is now 49, making every currently-deployed batch size (1-48) undivided.
  `MICROBATCHES_PER_STAGE` and the pipeline-thread machinery are unchanged and
  still built at startup, so lowering `MIN_MICROBATCH_SLOTS` (or raising
  `max_batch` past it) reactivates them without a code change; resweep before
  doing either, since this conclusion is a measurement of today's kernels, not
  a property of the architecture. `seed_tests/test_opt_decode_microbatch.py`
  covers the group-count math directly and pins the current
  always-undivided-so-far behavior. Not verified: the cross-device copies'
  cost, since every `.to` on CPU is a no-op; the pipeline's overlap value at a
  `max_batch` large enough for microbatches to make sense again.

- **Tensor parallelism at TP=4, replacing the pipeline (`tp.py`, `tp_driver.py`,
  `deltanet_tp.py`, `model.py`, `server.py`), measured on 4x MI300A.** The
  pipeline above overlaps the four devices, but it cannot make them share the
  *weight traffic*: under it the node's whole 192.5 GB/step of expert weights
  flows through one device's HBM at a time. At MI300A's spec 5.3 TB/s that is
  36.3 ms, more than the 30.9 ms the 25% gate allows for an entire decode step,
  before a single kernel runs. That is a property of the topology, not of the
  kernels, so no MoE kernel lets the pipeline reach the gate
  (`DECODE_BOTTLENECK_2026-09-22.md` section 3). Under TP every rank runs every
  layer on a shard of its heads and experts, so the same traffic is 48.1 GB per
  device, concurrently. One OS process per rank; rank 0 serves HTTP, runs the
  scheduler and broadcasts each `Runner` call to the others (`tp_driver.py`),
  which also removes the need for the pipeline's stage threads. `--tp 1` keeps
  the pipelined layer split, and is what the CPU tests use.

  Sharding, per `tp.py`'s convention: query heads for full attention (KV heads
  are *replicated*, since 2 of them do not split over 4 ranks -- see
  `tp.kv_shard`), value heads for DeltaNet (the gated delta rule's state is
  block diagonal over them, so the recurrence itself communicates nothing;
  `deltanet_tp.py` carries the derivation), whole experts for the MoE, and the
  intermediate axis for the shared expert. Two all-reduces per layer, 120 per
  step. Nothing else communicates: the router is replicated, so every rank
  reaches the same routing from the same hidden state without a collective.

  Almost none of the attention code changed, which was not expected. `model.py`'s
  `decode_attention` and `prefill_attention` read both head counts off the
  tensors they are given rather than off `Cfg`, so the group-into-query-length
  fold, the mask repeat and the `is_causal` fast path (all measured fixes for
  real gfx942 SDPA backend behavior, see their docstrings) are exact at 8 query
  heads over 1 KV head just as they are at 32 over 2. What changed is the
  projection views, the KV pool width, and `tp.kv_shard` keeping a rank's query
  block inside one KV group so the fold stays faithful. The MoE was nearly free
  for the same reason: `mxfp4_gemv`'s expert-range early return already existed.

  **Measured on 4x MI300A**, real checkpoint, `max_batch` 48, prompt 32, the
  same probe and the same node for both rows, driven through `tp_driver` exactly
  as the server drives it:

  ```
  topology                        ms/step   tok/s
  --tp 1 (pipelined layer split)   317.03   151.4
  --tp 4 (this change)             199.49   240.6
  ```

  **1.59x**, against the 2.8x `DECODE_BOTTLENECK_2026-09-22.md` projected from a
  standalone-kernel probe. The gap is measured, and it is expert-routing skew.
  That document assumed uniform routing and explicitly listed the assumption as
  one to challenge if a real run disagreed. It disagrees: the step's 480
  assignments land 94 / 158 / 119 / 119 on the four expert shards, so the
  busiest rank carries 1.32x the uniform share, and every rank waits for it at
  each of the 120 collectives. Balancing the expert-to-rank assignment is the
  obvious follow-up and is not done here.

  **The collectives are not the cost, which settles an open question.** Swapping
  the reducer for a no-op between timed phases on all four ranks at once (this
  probe calls `Model.decode` directly rather than through the broadcaster, hence
  the ~11 ms offset from the table above):

  ```
  phase                       ms/step
  fp32 all-reduce (shipped)    188.14
  bf16 all-reduce              188.51
  no reducer at all            187.79
  ```

  and standalone at the step's shape, 120 collectives of `[48, 4096]`: fp32
  **44.4 us** each (5.33 ms/step), bf16 **55.2 us** (6.63 ms/step). fp32 is not
  slower here; RCCL's bf16 reduction path is. So `tp.all_reduce`'s upcast costs
  nothing, and fusing the mixer and MoE reductions into one per layer -- which
  that document suggested as a 10-18% follow-up -- would buy under 3 ms of a
  199 ms step. Neither is worth doing.

  Verified on CPU in three layers before any of the above
  (`seed_tests/test_tensor_parallel.py`, `seed_tests/test_deltanet_tp*.py`): the
  plan arithmetic including the GQA replication case; four real sharded `Model`s
  in one process with their partials summed by hand, per component (attention
  prefill and resumed-prefix, batched decode at uniform and ragged positions,
  MoE dense and MXFP4, DeltaNet, and each weight shard against the unsharded
  tensor); and the whole prefill-then-decode sequence under a real four-process
  `torch.distributed` group on gloo, driven through `tp_driver` exactly as
  `server.py` drives it. Those run in fp32, where the row-parallel residual is
  ~1e-6, so they do not bound the bf16 case; the accuracy gate below is what
  does.

  **Accuracy gate, the thing this change was most at risk of moving.** TP sums
  four separately rounded partials per row-parallel projection instead of
  accumulating one dot product, so it is not bit-identical to the single-device
  sum, and the gate is exact-match greedy pins. `tp.all_reduce` therefore
  reduces in fp32 and rounds once on the way back, landing the topology change
  without also changing collective precision. Run through the real server, same
  node, same checkpoint, same pins file, `--max-batch` 48:

  ```
  topology   gates    greedy pins (need >= 90% exact on 32 tokens)
  --tp 4     13/14    10/14 exact, early divergence {pin14: 1}
  --tp 1      0/14    server died mid-gate; see below
  ```

  At TP=4 the gate still fails, as it has all campaign, and it fails on the pins
  rather than on any of the 13 behavioural probes.

  **The `--tp 1` row is not a numerics result.** That server loaded and warmed
  up, then aborted partway through the holdout sessions with
  `Memory access fault by GPU node-4 ... Reason: Unknown`, and every later probe
  failed to connect. That is the unexplained fault
  `DECODE_BOTTLENECK_2026-09-22.md` section 6 priced and could not reproduce,
  and it is the same signature as the earlier pipelined accuracy run on record
  (3/14, all connection failures after the server disappeared). So the
  comparison this change most wanted -- the same pins under the pipeline --
  cannot be taken today, because the pipelined server does not survive the gate.

  What can be said is stronger than a numerics comparison anyway: **TP=4
  completed the gate that the pipelined path crashes in**, on the same node, the
  same checkpoint and the same batch size. Whether tensor parallelism fixes that
  fault or merely no longer forms the shape that triggers it is not established
  here and is worth its own investigation.

  Do not switch the collective to bf16 without re-running the TP=4 row; the pins
  are the only thing that would catch it.

  Not verified: any batch size other than 48, any prompt other than 32, prefill
  latency (`p95_ttft_turn2plus_ms`, the objective's actual metric, which should
  gain more than decode since prefill is the same MXFP4 decode over ~10x the
  assignments and was equally serialized under the pipeline), and RCCL under
  anything but this one node's four devices. A second oddity seen and not
  chased, pointing the same way as the fault above: the synthetic decode probe's
  logits come back NaN under `--tp 1` and finite (and run-to-run identical)
  under `--tp 4` on the same garbage-token prompt.

The changes compose: a batched decode step drives the vectorized MoE, the SDPA
attention and now DeltaNet at a real batch dimension, checked slot-by-slot
against the same slot running alone in
`seed_tests/test_batched_decode.py::test_batched_moe_matches_single_sequence`,
`seed_tests/test_opt_deltanet_batch_slots.py` and the attention/rope cases
beside them. The pipeline narrows that batch dimension to a microbatch, which is
the one part of the change that is not bit-preserving: `Model.moe` pads each
activated expert's tokens to the call's own largest group, so a narrower call
groups the same products into differently shaped `bmm`s and float32
reassociation moves by about 1.4e-5 on logits of magnitude 8, the same effect
batching itself already has (pinned by
`test_the_microbatch_cut_only_moves_the_logits_by_float_reassociation`).

Deferred, not done here: prefill is neither batched across requests nor
pipelined. One `prefill` call is one chunk of one request, and a chunk's tokens
are causally dependent, so there is no independent work inside it to stagger;
overlapping different requests' chunks would need the scheduler to run several
prefills per iteration and is untouched here. With `delta_rule_chunked`
disabled (see the chunked-DeltaNet bullet above), prefill's DeltaNet cost is
also the per-token recurrence over the whole prompt, so finding a parallel
prefill form that is actually fast on gfx942 remains open.

**TP=4 combined with the byte-permute MXFP4 decode, measured together
(`bespoke/opt-integrated-v4`).** The two changes above were developed on
separate branches and each measured only against its own base: TP=4 without
the byte-LUT decode, and the byte-LUT decode PP-shaped (480 assignments over
512 local experts), not under TP's per-rank sharded count (~120-158 of 480).
The branches touch disjoint files except this README (TP: `tp.py`,
`tp_driver.py`, `deltanet_tp.py`, `model.py`, `server.py`; the decode change:
`mxfp4_gemv.py` only), so the merge is a clean union with no conflicts.

**Measured on 4x MI300A**, real checkpoint, `--tp 4`, batch 48, prompt 32,
same probe (`tp_decode_bench.py`) driven through `tp_driver` as the server
drives it, mean of 3 runs:

```
stack                                    ms/step   tok/s
--tp 1, original decode (pipeline)        317.03   151.4
--tp 4 alone                              199.49   240.6
--tp 4 + byte-LUT decode (this, mean of
  178.08 / 192.49 / 198.74)               189.77   253.47
```

**1.67x over the pipelined baseline, 1.05x over TP alone.** The byte-LUT
decode adds only about 5% on top of TP, far short of its own standalone
1.6-1.7x: TP already shrank what that kernel change had left to speed up.
Under TP each rank dequantizes only its own ~120-158 of the 480 assignments
(not the full 480 the isolated microbenchmark and the pipelined path both
did), so the MoE decode is a smaller share of a TP rank's step time, and
there is proportionally less of it for a faster decode to remove. Run-to-run
spread was wide for this probe, 178-199 ms across the 3 runs (about 11% of
the mean), not investigated further here.

Against the 25%-of-roofline gate (30.9 ms/step, 1,554 tok/s at batch 48):
**253.47 tok/s is 16.3% of gate**, up from TP alone's 15.5% and the pipelined
baseline's 9.7%. Still 6.1x short.

**Accuracy gate**, real server, same node and checkpoint, `--max-batch` 48:
13/14 checks passed, the same signature as TP alone (10/14 greedy pins exact,
early divergence at `pin14`) — consistent with the byte-LUT decode's own
bit-identity claim at model dimensions, which predicts no accuracy change
from TP alone. The gate still fails on the pins, as it has all campaign.

CPU: 400 passed, 18 skipped (`seed_tests`, `TRITON_INTERPRET=1`), 0 failed.

**Two claims in this section did not survive the next measurement pass and are
corrected below.** The 1.59x-versus-2.8x gap is *not* expert-routing skew: the
94 / 158 / 119 / 119 split is one layer's routing, and over all 60 layers the
four shards take 1.009 / 1.008 / 0.944 / 1.037 of the uniform share. And the
byte-LUT decode's 1.05x is not Amdahl on a smaller MoE share: the step had
66 ms of device idle, so a change that removes device time fell into the slack.
Both are re-derived from measurement in `TP_BYTELUT_BOTTLENECK_2026-09-22.md`.
- **Token-id `/v1/completions` and a `/reset_prefix_cache` hook (`server.py`,
  `scheduler.py`, `model.py`, `graph_decode.py`, `tp_driver.py`), not measured
  on real hardware.** A benchmarking client that drives the engine by token id
  (rather than chat text) needs a raw completions endpoint: `prompt` is a
  token-id array, not text; streamed chunks carry `token_ids` instead of a
  text delta; usage reports `prompt_tokens_details.cached_tokens`. It shares
  `scheduler.Scheduler`'s admission path with chat, so the same prefix
  matching, batching, and graph-capture decode apply; chat's own requests are
  untouched, since every new parameter (`Request.extend_prefix`,
  `DetokWorker`'s `include_token_id`) defaults off and chat's call sites pass
  neither.

  A completions request may also opt its slot into reusing its own reply: on
  a later request whose prompt is exactly prompt+reply, the slot resumes past
  the reply instead of re-prefilling it. This is recorded lazily
  (`_Slot.reply`, applied by `_try_apply_reply` only when a later prompt
  actually needs it) rather than eagerly at request-finish, which is what
  keeps an exact-duplicate prompt matchable too: an eager scheme extends the
  recorded prefix immediately, so a plain repeat of the original prompt no
  longer has a prefix to match at all. `_match_prefix` tries the extended
  candidate first, then the un-extended one, so a slot is never made *less*
  matchable than before this change; matching is still exact-equality, so a
  match is never *incorrect*, only occasionally a missed reuse (e.g. two
  requests share a prefix that isn't recorded on any idle slot). The
  identical-prompt case (two probes with the same prompt back to back, as a
  cache-warming preflight does) reuses the slot's cached last-position logits
  and makes no forward call at all, which is also where the reported
  `cached_tokens` for that case comes from: every prompt token, none
  reprefilled.

  Caching the slot's own last-position logits without holding a fresh tensor
  per slot needed a small static buffer (`Model.logits_scratch`,
  `cache_logits`/`cached_logits`), copied into and read as a view rather than
  a newly allocated tensor per call. Without it, `test_state_memory.py`'s
  "serving a turn allocates no persistent state" invariant fails (a slot's
  logits are exactly the kind of persistent per-turn allocation that test
  catches). `/reset_prefix_cache` clears every idle slot's recorded prefix
  (and any unresumed reply) without touching in-flight requests, mirroring
  vLLM's endpoint of the same name.

  Capacity is unchanged by this: `--max-seq-len` 16384, `--max-batch` 48
  slots by default, one prompt+session per slot. A completions prompt at or
  over `max_seq_len` gets a clean 400 (same check chat already had), not a
  crash or a silent truncation.

  Verified on CPU only (`seed_tests/test_completions_endpoint.py`, extended
  `seed_tests/test_scheduler.py`): request validation (rejects text prompts,
  batched prompts, non-int ids), the SSE chunk and final-usage shape, cached
  reuse across a second request extending the first, the identical-prompt
  case, `/reset_prefix_cache` clearing idle slots only, and over-capacity
  rejection.

## Host dispatch: the `use_grouped` denominator and the fused DeltaNet decode

`TP_BYTELUT_BOTTLENECK_2026-09-22.md` measured the TP=4 + byte-LUT step as
**host-issue-bound**: 186 ms of host time to issue a step the device finished
2.16 ms later, against 122 ms of device kernel time, so the GPU idled 35% of
every step. In that regime an op costs what it costs to *issue*, and the two
changes here both remove issues rather than work.

**1. `use_grouped` compared the global assignment count against the local
expert count.** Expert parallelism hands every rank all 480 decode assignments
and drops the unowned ones GPU-side, so the pile-up depth on a local expert is
`480 / 512`. Reading it against the rank's 128-expert shard reports four times
that, clears the threshold, and selects the de-duplicating prefill kernels for
a decode step they lose on — about 900 extra dispatches per step from
`align_blocks`. `fused_moe` now takes `total_experts`, defaulting to the width
of `expert_range` so the single-rank case is unchanged, and `_routed_fused`
passes `c.experts`, which `local_cfg` deliberately leaves global.

**2. DeltaNet was 64.12 ms of the 188 ms step across ~1,575 `aten` calls.** Its
whole weight and state traffic is ~10 GB/step, about 3 ms of bandwidth, so it
was 20x above its own memory roofline purely on dispatch count. Three changes,
all in `deltanet_fused.py` and `Model.deltanet_decode`:

- **One input projection.** `in_proj_qkv`, `in_proj_z`, `in_proj_b` and
  `in_proj_a` all read the same `x`, so `fuse_in_proj` concatenates them on the
  output axis at load time and the four names stay as views into the result. One
  `F.linear` plus a split replaces four: 135 fewer dispatches per step, and no
  weight stored twice.
- **A Triton `gated_rmsnorm`.** Twelve torch ops on a few-hundred-KB tensor
  become one kernel.
- **A Triton decode recurrence.** The `T = 1` gated delta rule, its
  `l2norm`/cast preamble, the `repeat_interleave` onto the value heads and the
  gate activations become one kernel. It is also the only spelling that reads
  the recurrent state once where the torch form touches it five times, which is
  ~15 GB/step of avoidable traffic on top of the dispatches.
- `_host_index` memoizes the step's slot and position index tensors, which were
  rebuilt (and re-copied to the device) once per layer for all 60 layers.

Prefill keeps the torch path throughout: its calls are large enough to be worth
their own launch.

**Measured on 4x MI300A** (node02), four arms alternating step by step in one
process so drift cancels, 12 steps per arm, mean of all 4 ranks:

```
arm                                        ms/step   host issue   wall - host
grouped kernels + torch DeltaNet (v4)       179.52      177.28          2.2
  + use_grouped fix                         159.05      156.42          2.6
  + one input projection, memoized index    135.17      105.24         30.1
  + Triton gated_rmsnorm and recurrence      118.91       94.35         24.8
```

**1.51x, and the step is device-bound again.** Host issue time falls 177 -> 94
ms while wall clock only falls to 119, so the 2 ms host/device gap that defined
the previous pass becomes 25 ms: the queue no longer runs dry. 119 ms is the
122 ms device kernel time that document measured, which is the strongest
confirmation of its model available — and it means the remaining dispatch items
it ranked (graph capture) now have almost nothing left to buy. The next binding
term is device time, where `aten::mm` is 586 calls and 48.9 ms per step at
`M = 48`, an order of magnitude above roofline and not root-caused.

The four ranks agree to 0.4 ms on every arm (rank spread: v4 179.41-179.74,
v5 118.80-119.16), so this is not one rank's artifact.

**End to end**, `tp_decode_bench.py` against the patched source, batch 48,
prompt 32, 20 steps after 20 warmup (a kernel-path change needs the long warmup;
at the default 3 the per-assignment kernels are still settling):

```
stack                                       ms/step   tok/s   node        when
--tp 1, original decode (pipeline)           317.03   151.4   --          earlier
--tp 4 alone                                 199.49   240.6   --          earlier
--tp 4 + byte-LUT                            189.77   253.5   node02   earlier
--tp 4 + byte-LUT + use_grouped fix          166.05   289.1   node02   earlier
--tp 4 + byte-LUT + use_grouped fix          175.39   273.7   node01   pair A
--tp 4 + byte-LUT + both (this)              124.08   386.9   node01   pair A
--tp 4 + byte-LUT + both (this)              123.04   390.1   node01   pair A
--tp 4 + byte-LUT + use_grouped fix          194.49   246.8   node02   pair B
--tp 4 + byte-LUT + both (this)              119.48   401.7   node02   pair B
```

**Only same-node, same-hour pairs are comparable, and this run makes the reason
plain.** The `use_grouped` arm measured 159.05 (interleaved A/B), 166.05 (on
record), 175.39 and 194.49 ms across four measurements of identical source, a
22% spread; the fused arm measured 118.91, 119.48, 123.04 and 124.08 ms, a 4%
spread. That asymmetry is the mechanism: a host-issue-bound step's time is set
by how fast Python can issue, which varies with node load and time-in-run, while
a device-bound step's time is set by the hardware. The fused arm is stable
because it is no longer waiting on the host.

So the honest statement of the gain is the interleaved A/B's **1.34x** over the
`use_grouped` fix and **1.51x** over the branch on record, corroborated by
1.42x (pair A) and 1.63x (pair B) end to end. Do not read 189.77 -> 124.08 as a
single measurement; it crosses nodes and hours.

Against the 25%-of-roofline gate (30.9 ms/step, 1,554 tok/s at batch 48),
**386.9-401.7 tok/s is 24.9-25.9% of gate**, up from 16.3%. Roughly 4x short.

**Accuracy gate**, real server, same node and checkpoint, `--max-batch` 48:

```
stack                                  gates   greedy pins (need >= 90% exact of 32 tokens)
--tp 4 + byte-LUT (on record)          13/14   10/14 exact, early divergence {pin14: 1}
--tp 4 + byte-LUT + use_grouped fix    13/14    8/14 exact, early divergence {pin14: 1}
--tp 4 + byte-LUT + both (this)        13/14   10/14 exact, early divergence
                                                 {pin09: 6, pin14: 1}
```

**The verdict and the failing gate are unchanged, and so is the 4.6-pin distance
from the threshold.** All 13 behavioural probes (six holdout sessions, four
history probes, three arithmetic) pass on every row.

The v4 row on record was taken on the *grouped* kernels, which the section above
did not know; its "consistent with the byte-LUT decode's bit-identity claim"
inference was therefore about a path that did not run. That claim has still not
been checked on a path that did.

Read the pin column as noise around a number the gate fails by a wide margin,
not as a trend: it goes 10 -> 8 -> 10 across three numerically distinct paths,
all far below the 12.6/14 needed. The one new detail is `pin09`, which now
diverges at token 6 where it previously diverged past token 8. It was a
non-exact pin on every row; what changed is when. That is what a recurrence
whose reduction order moved should do, because the DeltaNet state carries fp32
differences forward across all 32 generated tokens, and it is why 10 pins
staying bit-exact over those 32 tokens is the load-bearing evidence that this is
reassociation and not a bug — alongside `test_deltanet_fused.py` holding both
kernels to the torch chain on hardware, advanced state included.

Neither change is numerics-neutral: both move reduction order. Both are
reversible without a revert for a bisect — `SEED_MOE_GROUPED_MIN` sets the
kernel-selection threshold and `SEED_FUSED_DELTANET=0` forces the torch DeltaNet.

Nothing here is aimed at `pin14`, which has diverged early under every topology
and every kernel this campaign has run, and nothing here explains it.

CPU: 422 passed, 18 skipped (`seed_tests`, `TRITON_INTERPRET=1`), 0 failed.
On the accelerator, `test_deltanet_fused.py` is 21 passed: the Triton kernels
against the torch chain they replace, output and advanced state both, at fp32
and bf16.

## Dense GEMM: rocBLAS picks a large-M tile for an M = batch GEMM

`TP_BYTELUT_BOTTLENECK_2026-09-22.md` section 7 item 4 left this measured and
not root-caused: `aten::mm` was the largest device-time term in the step, and
"roughly an order of magnitude above roofline". It is a tile-selection problem,
and the mechanism is visible in the launch grid.

Every dense projection in a decode step is `[B, hidden] @ [hidden, out]`, so at
`max_batch = 48` every one of them is M = 48 against K and N in the thousands.
rocBLAS chooses a macro tile as if M were large. At the real per-rank TP=4
shapes on one MI300A (gfx942, 228 CUs), the shared expert's `gate_proj`
(K = 4096, N = 256, 2.1 MB of weights) gets `MT256x224x64`. With M = 48 there is
one tile on the M axis, so the grid is `ceil(256 / 224) = 2` workgroups: **two
CUs out of 228 run the whole GEMM**, for 126 us where the bytes say 0.6 us. Not
an anomaly of that shape — `q_proj` at N = 4096 gets `MT128x96x128`, a 43-workgroup
grid, and the whole step's dense GEMMs came to 34.87 ms of device time.

Two changes, measured separately.

**1. Select a solution per shape (`blas_tune.py`).** PyTorch's TunableOp
benchmarks every rocBLAS and hipBLASLt solution for a shape once and keeps the
fastest. Given the choice it takes tiles that are small on M and N and deep on
K — `MT16x16x256` for that `gate_proj`, `MT32x64x256` for `q_proj` — which is
the shape of a grid that fills the machine. `build_model` tunes the decode
shapes at startup, freezes the choices, and leaves the search off, so no request
and no graph capture ever hits a synchronizing tuning call; results are cached
per GPU architecture under `~/.cache/qwen35-bespoke`, so only the first start on
a machine pays the ~20 s per shape. `SEED_BLAS_TUNE=0` is the ablation.

Two alternatives were measured and rejected. Forcing hipBLASLt over rocBLAS
(`TORCH_BLAS_PREFER_HIPBLASLT`) changes nothing, kernel names included: both
routes reach the same Tensile solution. A hand-written Triton skinny-M GEMM with
split-K, swept over tile and split configurations at each shape, reaches 17.66 ms
against solution selection's 8.09 ms, so the custom kernel is half the win for
an order of magnitude more code.

**2. Two fewer dense GEMMs per layer (`fuse_moe_dense`).** With selection on,
what is left is a per-call floor: the shared gate costs 8.4 us to multiply 8 KB
of weights. The MoE block runs four GEMMs against the same `h`, so `fuse_rows`
(the generalization of `fuse_in_proj`) concatenates the router with the shared
gate and `gate_proj` with `up_proj` at load time, and the individual names stay
as views. That is 120 fewer calls per step, and it widens N, which is what lifts
those GEMMs off the floor.

Device time per decode step, summed over the dense GEMM calls at their real
per-rank shapes and per-step counts, one MI300A, batch 48:

| | step ms | calls |
| --- | ---: | ---: |
| default heuristic | 34.87 | 451 |
| + `fuse_moe_dense` | 20.15 | 331 |
| + solution selection | **5.74** | 331 |

**6.1x, and it holds across the batch widths the engine serves.** Untuned to
tuned at the post-fusion shapes: 20.09 -> 5.17 ms at batch 1, 19.94 -> 5.48 at 8,
19.99 -> 5.38 at 16, 20.15 -> 5.74 at 48 (3.5-3.9x each). The worst individual
shapes are the ones with the worst grids: `shared_expert.gate_up_proj` 112.3 ->
16.1 us, `in_proj_all` 146.5 -> 18.6 us.

End to end on the real checkpoint, 4x MI300A, TP=4, batch 48, prompt 32, against
this branch's base:

| | ms/step | tok/s |
| --- | ---: | ---: |
| base (fused DeltaNet decode) | 122.58 | 391.6 |
| + `fuse_moe_dense` | 110.67 | 433.7 |
| + solution selection | **107.37** | **447.1** |

1.142x. Note the shape of it: the concatenation is worth 11% and the solution
selection only 3% more, although selection is by far the larger *device*-time
saving. That is the dispatch ceiling again — at 110 ms the step is back to
waiting on host issue, so most of the 14.4 ms of device time selection frees is
landing in idle. It is banked, not lost: it converts to wall clock as soon as
graph capture removes the issue floor.

Correctness. Against an fp32 oracle at every shape and at batches 1/8/16/48, the
selected solution's error equals the default's to four decimals; where the two
bf16 results differ at all they differ by one ulp, from the different reduction
order. Full-model greedy argmax over the first eight decode steps is unchanged
across all three configurations above.

The routing is bit-exact under the concatenation — the router's own columns are a
whole macro tile either way — so expert selection cannot change. The shared
gate's single column does move by a few 1e-6 relative, because BLAS reduces a
one-column GEMV differently from a wide GEMM, which is why
`test_vectorized_moe_matches_old_per_expert_loop` now allows 1e-4 in its float32
spelling; in bf16 that is far below one ulp.

Leaving TunableOp enabled costs nothing at the shapes it has no entry for, which
is what protects prefill and so TTFT: at M = 512 the six prefill projections take
469.0 us enabled against 468.9 us disabled, and at M = 2048, 723.8 against 734.2.

## Decode graph capture, off by default (`--enable-graph-capture`)

`graph_decode.py` captures the batched decode step as a HIP graph (one per
device, over that device's own run of layers) and replays it, so a step costs
one replay instead of a few thousand Python-dispatched launches. **It is off by
default and inert unless the server is started with `--enable-graph-capture`;
with the flag off nothing in that file runs and the scheduler drives `Model`
directly.**

It has now been executed on gfx942, and **capture fails**: `prepare()` spends
115s and then returns False with `hipErrorStreamCaptureInvalidated` (4x MI300A,
real checkpoint, `max_batch` 16). The designed fallback works, so the flag is
inert rather than harmful, but the feature does not do anything today.

That is a *different* blocker from the one this section used to describe. The
host syncs are genuinely gone; something in the captured region now invalidates
the HIP capture stream instead. The likeliest suspect is `attn_decode_static`'s
SDPA call, for two reasons measured separately on gfx942:

- It passes `enable_gqa=True`. Under a mask on this target that falls off the
  fused backend and materializes the kv broadcast up to all 32 query heads
  before the matmul, a 16x read of the tensor a decode step is bound by. At
  `max_batch` 16 and `max_seq` 4096 that is a 1.07 GB transient per call.
- It passes `attn_mask` at all. Flash is not available under a mask here; only
  the efficient and math backends are, and efficient charges about 3x for
  taking the mask.

PyTorch's allocator tolerates a fresh `hipMalloc` inside a capture (it uses a
graph-private pool) but not the `hipDeviceSynchronize` + `hipFree` it does when
it has to release cached blocks, and gigabyte transients per layer are a good
way to reach that path. **Untested**, and stated as a hypothesis rather than a
finding: the cheap experiment is to write the masked case out as matmul, masked
softmax, matmul (at one query token the score tensor is `[B, kv_heads, group,
S]`, a few MB, so there is nothing for a fused kernel to avoid materializing),
then re-run capture.

Capture needs every shape in the step to be a constant of the server. The MoE
used to be the worst offender, sizing its expert groups from the batch's own
routing and reading `int(counts.max())` back to the host, which is illegal
inside a capture; the fused kernel removed that, and `moe_static` is now just
`Model.moe` (measured: zero device-to-host round trips per MoE call). What is
left is `Model._attend`, which slices each slot's KV to its own length.
So the captured region is a second, static-shape spelling of the same
arithmetic in `graph_decode.py`: the step always runs all `max_batch` rows with
row j bound to slot j, attention runs over the full `max_seq` pool under a mask
instead of a slice, the MoE runs one expert weight set per (token, expert)
assignment with no de-duplication, and DeltaNet steps every slot at once. Rows
that are not in the batch are switched off by a mask and leave their slot's KV
and recurrent state bit-identical. `Model` itself is untouched.

A replay therefore always pays the worst case, which is why batches smaller
than half of `max_batch` go back to the eager path rather than being padded up
to it. The prior campaign on this model measured exactly this effect when it
folded static-shape DeltaNet kernels into its prefill graph (worst-case padding
ate about half the saving, and a stale per-step refresh then corrupted output
under concurrency); see
`resources/skills/serving-systems/references/platforms/rocm/gated-delta-net.md`.
Bucketing the batch dimension, so a batch of 12 replays a graph captured at 16
instead of at `max_batch`, is the obvious follow-up and is not done here.

Every failure path ends in eager execution: capture raising, the captured step
disagreeing with the eager step on a full-batch check at startup, or a batch
whose slots or positions are not what was captured. `seed_tests/test_graph_capture.py`
covers those paths, the flag being a no-op when off, and the static step against
the eager step, using an injected capture backend whose "replay" re-runs the
step against the same static buffers. What it cannot cover is capture itself.

## Double weight load between accuracy and benchmark (known, deferred)

`accuracy_checker/checker.py` and `benchmark/run.py` each independently boot
`python3 server.py` through `benchmark/launcher.py`, so a full evaluation
round loads the checkpoint twice (about 6 minutes extra). Both scripts
already accept `--base-url` to run against an already-booted server instead,
so a same-bundle fix is possible in principle: have the accuracy phase leave
its server running (a small workspace handoff file with the base URL) and
have the benchmark phase detect and reuse it before falling back to booting
its own. The evaluator's execution model permits this for a Docker/local run
(the sandbox container is long-lived across gate invocations, and benchmark
only runs after an accuracy pass, so ordering is guaranteed), so this is not
a change to vibesys's core evaluator protocol. It is deferred here anyway:
this bundle's actual deployment path is SkyPilot
(`--run-environment skypilot --backend rocm`), and whether a background
server process survives between separate `sky exec` invocations, and whether
the same persistent cluster session can be reused across candidate rounds
and retries without stale-server or port-conflict risk, is unverified and
cannot be checked without cluster access. Landing an unverified handoff risks
a candidate benchmark silently scoring a stale server (wrong code, wrong
weights) or leaking a GPU-memory-resident process, both worse than the known,
bounded double-load cost. Revisit once cluster access can confirm SkyPilot's
process-survival semantics across separate job invocations.

## Baseline protocol

Success is judged against the optimized SGLang tree from the #421 campaign,
not against the seed. Compare paired runs on the same node:

1. Per node, run the SGLang tree and the candidate back to back on the same
   allocation. Alternate which goes first across nodes.
2. Each side runs 5 repetitions of `benchmark/run.py`. The first repetition is
   a warmup and is discarded.
3. Flush caches before every repetition. For SGLang that is its flush-cache
   endpoint; for a candidate, restart the server or otherwise start each
   repetition with no cross-repetition prefix state. Without this, later
   repetitions inherit prior-turn state that the workload does not offer.
4. Compare the median of the 4 measured repetitions of
   `p95_ttft_turn2plus_ms` per node.

The candidate passes when, on every paired node, it first matches the SGLang
tree within noise, then exceeds it (lower p95 TTFT). Run-to-run spread on one
node was about 6 percent for identical trees, and identical trees differed by
tens of percent across nodes, so unpaired or cross-node comparisons are not
evidence.

Also report `mean_tpot_ms` and `total_token_throughput` for both sides. They are
not the objective, but a TTFT gain that costs decode speed or throughput is
reported as a tradeoff.

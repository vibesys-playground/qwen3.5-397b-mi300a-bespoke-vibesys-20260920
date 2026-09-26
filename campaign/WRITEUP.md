# Bespoke Qwen3.5 Serving Optimization on AMD MI300A

## Experiment overview

The experiment built and optimized a serving engine from scratch for amd/Qwen3.5-397B-A17B-MXFP4. The engine could use PyTorch,
Triton, and individual AITER kernels, but could not import or copy serving code from SGLang, vLLM, or TensorRT-LLM.

The workload was a deterministic multi-turn chat benchmark:

- Concurrent chat sessions, ramped from 1 to 192 in the load-ramp benchmark
- 3 to 6 turns per session
- 80 to 300 generated tokens per turn
- 1 to 8 seconds of think time between turns
- Full accumulated conversation resent on each turn
- Greedy decoding with fixed completion lengths
- Hard failure on dropped, errored, empty, or truncated responses

Although the original task emphasizes turn-2+ TTFT, this optimization campaign used peak goodput in tokens per second as its main
comparative performance measure, subject to correctness and latency gates. From round 11 onward the headline number is goodput at
96 concurrent sessions (C96).

Every accepted change had to pass a numerics gate: at least 1,074 of 1,088 teacher-forced top-1 tokens matching the reference, and
5 of 5 greedy reference completions.

## Model

The model is Qwen3.5-397B-A17B-MXFP4:

- Approximately 397 billion total parameters
- Approximately 17 billion active parameters per token
- Sparse mixture of experts model
- 512 routed experts
- MXFP4 expert weights
- 60 layers, a mixture of:
    - 45 Gated DeltaNet linear-attention layers with recurrent state
    - 15 full-attention layers with paged KV state
    - Shared and routed MoE computation
- A built-in multi-token prediction (MTP) head

This combination is difficult to serve efficiently because prefix reuse must preserve both conventional attention KV and DeltaNet
recurrent state. A DeltaNet state snapshot for one point in a conversation costs about 47 MiB, so the prefix cache holds far fewer
conversations than a pure-attention model would. The large expert set also makes expert-weight traffic the dominant cost for many
decode and prefill shapes.

The final architecture used four ranks across the node. Dense projections, attention, and DeltaNet were tensor sharded. MoE experts
were partitioned across the same ranks, with each rank owning a subset of the experts and the partial outputs combined collectively.
Pipeline parallelism was not used.

## Hardware

The target system was one node containing:

- 4× AMD MI300A
- ROCm
- gfx942
- Unified host and accelerator memory

Unified memory materially affected the experiment. Large snapshot pools, KV pools, captured graph allocations, and process RSS all
competed for the same physical capacity. Several otherwise correct configurations became unresponsive during graph initialization or
immediately before serving. Device memory allocated on a nearly full NUMA node was also measurably slower, so the final engine checks
free memory at boot and leaves 15 to 24 GiB free per node.

Node-to-node variation was roughly ±10%, so every accepted change was measured as same-node pairs (control and candidate on one node,
at least two pairs).

## SGLang baseline

The comparison baseline is SGLang v0.5.18 with MTP speculative decoding (k=3), on the same hardware and benchmark:

- 961.013 tok/s
- Peak at concurrency 48

SGLang as released does not run this model on this hardware. The baseline number comes from a VibeSys-modified SGLang; see
[SGLang on MI300A](#sglang-on-mi300a) for why stock SGLang fails and what it took to make it run and run well.

## Performance target

The target was 1,249.317 tok/s at C96, 30% above SGLang.

The best verified bespoke results were:

| Build | C96 goodput | vs SGLang (961.013) |
|---|---|---|
| Without speculative decoding | 1,889.9 / 1,892.9 tok/s | +97% |
| With MTP speculative decoding (final) | 2,242.4 / 2,201.4 tok/s | +133% / +129% |

Both results passed the numerics gate with zero failed turns. The target was met early in round 15 and the campaign goal was raised to
2,000 tok/s, which the final build also exceeds.

## Performance trajectory

The engine began as a simple PyTorch implementation. It gradually acquired custom kernels, graph capture, scheduling, state caching,
four-GPU parallel execution, and finally speculative decoding.

Rows before round 15 are single measurements. Round-15 rows are same-node pairs on different nodes, so the paired gain is the
comparable quantity; absolute numbers across rows differ partly because of the node.

| Stage | Peak throughput | Main change |
|---|---|---|
| Initial bespoke engine | 12.8 tok/s | Serialized prefill and full-history recomputation starved decode |
| Prefix-cache fix | 55.6 tok/s | Reused prior-turn model state |
| Integration r3 | 337.1 tok/s | HIP decode graphs, fused MoE, overlap scheduler, grouped prefill MoE, chunked DeltaNet, split-K attention |
| Integration r4 | 448.6 tok/s | Working prefill graphs, fused projections, per-shape BLAS tuning, in-place DeltaNet state |
| Integration r5 | 548.1 tok/s | Custom all-reduce, fused routing, fused RoPE and KV writes, graph-aware prefill shaping |
| Integration r6 | 564 tok/s | Elementwise and normalization fusions |
| Integration r7 | 659 tok/s | Captured mixed decode and prefill graphs |
| Decode collective fusion | 681.7 tok/s | Fused all-reduce, residual add, and RMSNorm |
| Integration r8 | 751.0 tok/s | Wide HIP MoE, rank-agreed validation, improved GEMM tuning, vocab-parallel LM head |
| Integration r9 | 688.4 tok/s on a slower node | Paired improvement was 8.8%; cross-node projection was approximately 815 tok/s |
| Round 10 incumbent | 784.9 tok/s | Later integrated scheduling and kernel work |
| Later C96 incumbent | 1,055.537 tok/s | Higher concurrency and accumulated integration work |
| Folded turn suffix | 1,154.477 tok/s | Eliminated separate small prefill calls for the chat-template suffix |
| R15: reduced target-closing bundle | +2.8% / +3.9% | Measured subset of the round-14 bundle |
| R15: accumulated wide prefill | +9.3% / +6.4% | Accumulate waiting prefills into chunks of up to 1,536 tokens (800 ms max wait) |
| R15: decode split-K attention knob | +1.1% | Tuned split-K partitioning for C96 decode attention |
| R15: fused prefill kernels | 1,167.3 / 1,164.8 tok/s (+3.8% / +4.8%) | In-place DeltaNet state, fused DeltaNet norm, fused prefill attention |
| R15: packed prefill | 1,266.8 / 1,219.6 / 1,266.8 tok/s (+5.6% / +3.5% / +6.5%) | Pack prefill sequences without padding to container shapes |
| R15: 32-row MoE blocks | 1,327.0 / 1,323.7 tok/s (+4.8% / +3.7%) | 32-row block tiles in the HIP MXFP4 MoE kernel |
| R15: prefix-cache snapshot reclaim | 1,610.1 / 1,621.2 tok/s (+26.8% / +27.8%) | Evict stale DeltaNet snapshots before live conversations |
| R15: early prefill launch | 1,351.0 / 1,362.7 tok/s (+2.3% / +4.7%) | Launch waiting prefills earlier in the scheduler step |
| R15: tuned BLAS cache | +5.2% / +4.8% | Tuned hipBLASLt solutions for the new prefill and verify widths |
| R15: sequence-parallel all-reduce + lookup fix | 1,889.9 / 1,892.9 tok/s (+9.3% / +8.1%) | New all-reduce protocol for decode and prefill up to 2,048 rows |
| R15: MTP speculative decoding | 2,242.4 / 2,201.4 tok/s (+20.3% / +16.0%) | MTP k=2 with captured verify graphs |

On a fixed reference node, the round-15 build at the packed-prefill stage measured 1,312.0 / 1,319.5 tok/s against 1,023.2 / 1,032.8
for the round-14 incumbent (+28%), which is the measurement that first cleared the target.

## Main successful optimizations

### Folded turn suffix (round 12-13)

Each later conversation turn previously incurred a separate small prefill replay for roughly five chat-template suffix tokens. The
optimization fed those known suffix tokens through decode steps already scheduled for the active batch.

In the same-node C96 pair:

- Control: 1,053.391 tok/s
- Candidate: 1,154.477 tok/s
- Gain: 9.60%
- Completed turns: 471
- Failed turns: 0

### Prefix-cache snapshot reclaim (round 15, +27%)

The prefix cache is a tree of past conversations. It stored a 47 MiB DeltaNet snapshot at every branch point, including points no
request would resume from because a later turn had already extended past them. Those stale snapshots used the memory budget, so the
cache evicted whole conversations that were still active, and their next turns recomputed the full history.

The fix evicts stale internal snapshots first and keeps the snapshot at the current end of each live conversation. Prefilled tokens
dropped about 40%. A follow-up fix made lookup find the snapshot re-saved at a conversation's end instead of stopping at the entry
whose snapshot was dropped; this recovered 15% at C16 after a C96 run.

### MTP speculative decoding (round 15, +16-20%)

Round 14 closed speculative decoding as noncompetitive because the captured implementation crashed with a HIP illegal-memory access on
its first replay. Round 15 found the root cause: captured graphs referenced tensors that were owned only by a Python closure and freed
after capture, so replay read freed memory. Keeping the step object alive with its graph fixed the crash.

The final configuration:

- k=2 draft tokens, verified in one wide captured graph
- Forced draft tokens so every lane has a fixed verify shape
- Tuned BLAS entries for the verify widths
- A fix for the packed-prefill tail, without which C96 halved to 958.6 tok/s
- 2.68 tokens emitted per lane per verify round; 84.1% of drafts accepted

In the same-node C96 pairs against the non-speculative build: 2,242.4 vs 1,905.4 and 2,201.4 vs 1,898.3 tok/s. The cost is latency at
low load: C16 turn-2+ p95 TTFT rises 11-15% (about 336-342 ms vs 299-303 ms).

### Sequence-parallel all-reduce (round 15, +8-9%)

The all-reduce between layers was replaced by reduce-scatter, residual add, RMSNorm, and all-gather, with one system-scope release per
phase and 4 rows per program. For a 2,048-row prefill, one call dropped from 332 µs to 152 µs. It is used for decode and for prefill
up to 2,048 tokens.

## Other paths investigated

### Higher concurrency

In round 13, C112 and C128 did not help:

- C112 full ladder: 998.853 tok/s
- C112 sparse ladder: 999.923 tok/s
- C128 encountered memory and graph-capture stalls

After the round-15 memory and prefix-cache work, full load ramps of the non-speculative build no longer stall and peak slightly above
C96: 1,888.6 tok/s at C128 on one node and 1,972.6 tok/s at C192 on another. C96 remains the headline point.

### Prefill profiling and row sharding

A calibrated captured-prefill profile showed the largest prefill components at the 1×512 shape:

| Component | Time |
|---|---|
| MoE | 38.95 ms |
| Collectives | 15.97 ms |
| Dense projections | 15.17 ms |
| Attention | 5.73 ms |
| DeltaNet | 4.85 ms |
| Glue and other work | 18.50 ms |

A row-sharded residual implementation reduced collective overhead by 5.7 to 6.1 ms per captured prefill and passed eager and captured
parity on all four ranks across all 12 production shapes. This work led to the round-15 sequence-parallel all-reduce above.

### Combined target-closing bundle

Round 14 projected that a bundle of row sharding, tighter prefill containers (1×384, 2×128), and the folded-suffix incumbent would
reach 1,249.623 tok/s, a 0.306 tok/s margin over the target, but the candidate never reached a serving-ready state.

In round 15 the bundle was measured directly. A reduced subset gained +2.8% / +3.9% and was merged. The full bundle measured +5.7%
unpaired. The target was ultimately reached through the round-15 prefill and prefix-cache work rather than this bundle.

### Hot-expert cache

A BF16 cache for frequently selected local experts covered 79.55% to 86.36% of held-out local assignments but was 5.1 to 6.8 times
slower than the routed kernel (0.363 to 0.481 vs 0.0712 ms/layer at B96) and consumed 11.25 to 19.69 GiB per rank. It was rejected.

### Reusing the generated reply (parked)

The benchmark client re-renders prior turns through the chat template, which drops the model's `<think>` block. The server therefore
cannot reuse the state it computed while generating the reply and re-prefills the assistant turn. Splicing the generated tokens back in
would save that prefill but would make the model condition on text the client did not send. This was parked as a policy decision: the
server must condition on the prompt the client sends.

### Other rejected paths

These were measured and found too small, negative, or gate-failing:

- Wide one-shot all-reduce alone
- Wide MoE launch-grid and pipeline-depth tuning
- Parallel MoE routing preparation
- Additional paged-attention work
- Short turn-delta ingestion
- Packed vectorized DeltaNet prefill
- Hot expert replication
- Persistent decode MoE kernel revisions, including a flat-grid variant (-0.5% to -0.7%)
- Custom dense GEMM replacing tuned hipBLASLt
- Larger 2,048-token prefill chunks (+7.7%, not adopted because of the TTFT cost)
- Moving host memory off the fuller NUMA node (-1.6% / -0.3%)
- Tiled prefill attention: the BF16 variant failed the numerics gate; the FP32 variant was +0.7%, within noise

## Same-node SGLang comparison

Both engines ran the same load-ramp benchmark on the same node.

| Concurrency | Tuned SGLang fork | Bespoke (no speculative decoding) |
|---|---|---|
| C1 | 130.8 | 91.5 |
| C4 | 310.1 | 302.4 |
| C16 | 724.5 | 651.5 |
| C32 | 899.4 | 900.0 |
| C48 | **938.2** (peak) | 1,142.4 |
| C64 | 822.0 | 1,559.0 |
| C96 | 830.9 (15.5 s TTFT guardrail breach) | 1,857.7 |
| C128 | | **1,888.6** (peak) |

- The bespoke peak is 2.01× the tuned SGLang peak.
- SGLang leads at C1 to C16, where its MTP helps and the bespoke build in this comparison had none. C16 p95 TTFT: SGLang 236.8 ms,
  bespoke 320.2 ms.
- This comparison ran on a different node from the 961.013 tok/s baseline measurement; nodes vary by about ±10%.

## Final status

The campaign improved the bespoke engine from 12.8 tok/s to 2,242.4 tok/s at C96, approximately a 175× increase.

Against the SGLang baseline:

- SGLang: 961.013 tok/s
- Best bespoke engine: 2,242.4 tok/s (2,201.4 in the second pair)
- Relative gain: +133% (+129%)
- Target: 1,249.317 tok/s, exceeded by 79%
- Without speculative decoding: 1,889.9 tok/s, +97%

Open items:

- C16 latency: MTP makes turn-2+ p95 TTFT 11-15% worse, and SGLang is faster at low concurrency.
- Estimated speed of light for this hardware and workload is 2,500 to 3,000 tok/s at C96 without speculative decoding.
- The next measured lever is overlapping MoE weight loads with matrix math, estimated at +1.5% to 2%.

## SGLang on MI300A

### Stock SGLang does not run

SGLang v0.5.18 as released cannot serve Qwen3.5-397B-A17B-MXFP4 on 4× MI300A. Three problems block it.

**No usable MXFP4 MoE kernel on gfx942.** SGLang's AITER MoE path for MXFP4 quantizes activations to FP4 before the expert GEMM.
Every MXFP4 kernel in the AITER version SGLang pins, including FP4 activation quantization, is gated to gfx950 (MI350-class) GPUs.
On gfx942 the server aborts during graph warmup with `fused_dynamic_mx_quant_moe_sort_hip: not support output type: fp4x2`. The
only other path is a generic Triton W4A16 fallback. It has no tuned configuration for 512 experts on gfx942 and runs at 1-3% of HBM
bandwidth, so MoE dominates every decode step.

**Memory sizing assumes discrete GPU memory.** On MI300A the weights, the OS page cache, host processes, and all GPU pools share
the same HBM. SGLang sizes its KV and Mamba-state pools from the free memory it sees after loading weights. Loading weights through
the page cache leaves the four ranks with uneven and shrinking free memory, so boots fail in different ways: out-of-memory, a negative
Mamba-state budget, or a node that stalls after over-allocating. When a boot does succeed, the KV pool size changes from run to run.

**Boot does not finish.** With default loading, weights took about 60 minutes to load. AITER then compiles one prefill attention
variant just in time on the first real request (about 105 s), and SGLang's 20 s health check kills the server while it compiles.

### Changes needed to make it run

| Problem | Change |
|---|---|
| No gfx942 MXFP4 MoE kernel | A custom fused HIP MoE kernel that reads MXFP4 weights directly and dequantizes them in registers (BF16 activations, no activation quantization, same numbers as the reference). Loaded through a new HIP extension loader with a content-keyed build cache. |
| Memory sizing on unified memory | Pinned the KV pool size (`--max-total-tokens 787936`) and set `--mem-fraction-static 0.72` (SGLang scales this by 0.85 internally for AITER at long context). Replaced page-cache-heavy loading with bounded reads: no mmap, threaded per-expert weight dispatch. |
| Weight load time | Pre-sharded per-rank weight artifact on striped storage, read without mmap. Weight load fell from about 60 min to about 1 min; server ready in about 200 s. |
| Health check during JIT compile | Raised the health-check timeout to 1,800 s and shipped a prebuilt AITER JIT cache containing every attention variant. |

### Changes needed to make it run well

| Area | Optimization | Observed effect |
|---|---|---|
| MoE | Replaced the gfx942 Triton fallback with the custom fused HIP MXFP4 MoE kernel | Largest initial serving improvement |
| MoE kernel refinement | Added register byte permutation, wider packed-weight loads, and a lower stage-1 threshold | Successive C48 TPOT reductions of 12.4%, 30.0%, and 8.9% in controlled comparisons |
| Dense layers | Added a custom skinny BF16 GEMM for small batch sizes. AITER's shipped GEMM tuning tables are keyed by CU count and never match MI300A's 228 CUs, so dense GEMMs fell back to untuned defaults | Reduced decode cost at small batch sizes |
| Speculative decoding | Enabled the model's NEXTN/MTP head with 3 speculative steps and 4 draft tokens | Median TPOT at C48 fell from 70.86 to 38.27 ms; average accepted length was about 2.85 tokens |
| Decode GEMMs | Generated PyTorch TunableOp tables for all captured draft, verify, and decode shapes | C48 TPOT fell from 38.05 to 22.86 ms |
| Prefill GEMMs | Padded the M dimension to 64 and tuned buckets from 256 through 1024 tokens | Turn-2+ p95 TTFT at C48 fell from 342.1 to 206.1 ms |
| Prefill execution | Enabled breakable prefill HIP graphs | C48 p95 TTFT fell from 363.6 to 321.1 ms in the isolated comparison |
| Scheduling | Disabled overlap scheduling for this latency-sensitive, multi-turn workload | Reduced p95 TTFT by 24-28%, with a small TPOT cost |
| Tokenization | Reused the cached prefix and sent precomputed token IDs for the new suffix | Saved roughly 4-6 ms of median TTFT |

Speculative decoding and mixed chunked prefill could not be used together, so the MTP path took precedence. A pre-MTP study found a
chunk size of 1024 best among 512, 1024, and 2048.

### Final SGLang configuration

Four-way tensor parallelism with:

```
SGLANG_USE_AITER=1
SGLANG_USE_AITER_UNIFIED_ATTN=1
SGLANG_MXFP4_MOE_HIP=1
SGLANG_SKINNY_GEMM=1
SGLANG_AITER_GEMM_PAD_M=64
SGLANG_MAMBA_SSM_DTYPE=bfloat16
SGLANG_HEALTH_CHECK_TIMEOUT=1800
PYTORCH_TUNABLEOP_ENABLED=1
PYTORCH_TUNABLEOP_TUNING=0
```

```
--tp 4
--attention-backend aiter
--page-size 16
--mem-fraction-static 0.72
--max-total-tokens 787936
--max-running-requests 48
--speculative-algorithm NEXTN
--speculative-eagle-topk 1
--speculative-num-steps 3
--speculative-num-draft-tokens 4
--enable-linear-replayssm-spec
--disable-overlap-schedule
```

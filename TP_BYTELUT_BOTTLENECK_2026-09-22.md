# Decode bottleneck, third pass: the step stopped being device-bound

Supersedes the ranking in `DECODE_BOTTLENECK_2026-09-22.md`, which was written on the
pipeline-parallel topology before TP=4 or the byte-LUT MXFP4 decode landed. That document's
roofline arithmetic and its two recommendations were right, and both shipped. Its model of
*what the step is made of* no longer holds, and neither does the explanation the README
attached to the combined result.

Scope: `bespoke/opt-integrated-v4` at `2495461`, batched decode at 48 concurrent slots,
prompt 32, real checkpoint, driven through `tp_driver` exactly as `server.py` drives it.
Everything below is measured on 4x MI300A (the test cluster, node01 / node02) in this pass,
except the rows marked projected.

## Verdict

**The decode step is no longer limited by the device. It is limited by the rate at which
Python can issue work to it.**

Three measurements carry the analysis.

1. **The host takes 186.06 ms to issue a step that the device finishes 2.16 ms later.**
   Device kernel time is 122.0 ms. The GPU is idle about 66 ms of every 188 ms step, 35% of
   the time, waiting for the next op. This is not the Python layer loop: replacing the whole
   layer body with a passthrough leaves 1.91 ms, so the cost is the 7,400-plus `aten` ops the
   60 layers issue, not the loop around them.

2. **Under TP=4 the MoE silently runs the wrong kernel pair.** `use_grouped` compares the
   *global* 480 assignments against the *local* 128 experts, so expert-parallel sharding
   quadruples the apparent per-expert density and selects the de-duplicating prefill
   kernels for a decode step they lose on. Correcting the comparison takes the step from
   **189.77 ms to 166.05 ms**, 253.5 to 289.1 tok/s, measured end to end on the same probe
   that produced the number on record.

3. **Expert-routing skew is 1.04x, not 1.32x, and is not the reason TP=4 returned 1.59x.**
   Measured over all 60 layers the four shards take 7267 / 7262 / 6800 / 7471 of 28,800
   assignments. The 94 / 158 / 119 / 119 on record is one layer's routing, not the step's.
   The residual rank-to-rank imbalance that does exist is *anti-correlated* with assignment
   count, so it is device variation, not routing.

**The question that prompted this pass has a different answer than the working hypothesis.**
The byte-LUT decode did not lose its 1.6-1.7x because the MoE became a small share of the
step. By ablation the routed MoE is still 31% of the step, the largest term after DeltaNet.
It lost it because the step has 66 ms of device slack: a change that removes *device time*
falls into the slack and does not shorten the step, while a change that removes *dispatch*
shortens it one-for-one. The `use_grouped` fix demonstrates exactly this, cutting wall clock
by 23.53 ms and host issue time by 23.54 ms, the same number.

This also re-prices the one item the previous pass ruled out by name. It deprioritized
graph capture on the premise that "two Triton kernels are 96% of device time and account for
only 120 of the step's launches. There is no longer a dispatch problem to solve." That
premise was true under PP and is false now. Dispatch is the binding constraint.

## 0. What is measured today

| | ceiling | 25% gate | today | gap |
| --- | ---: | ---: | ---: | ---: |
| aggregate tok/s @ c48 | 6,216 | **1,554** | **253.5** | **6.1x** |
| ms per decode step | 7.72 | **30.9** | **189.77** | **6.1x** |

Ceiling and gate are unchanged from `ROOFLINE_BOTTLENECK_ANALYSIS.md` section 0 and are kept
as the contract. The 189.77 ms is the number on record (mean of 178.08 / 192.49 / 198.74).

Step time varies with the node and with time-in-run. Four independent runs in this pass
measured a base step of 188.22, 190.98, 194.5 and 198.53 ms, and a control re-measured eight
slots later in the same process came back 9.74 ms slower than its own base. **Any
before/after smaller than about 10 ms has to be taken as an interleaved A/B, not as two
runs.** Section 3's headline number is; section 5's sub-breakdown is not, and says so.

## 1. The step is host-issue-bound

Timing a step three ways in one process: sync, issue the whole step, read the clock before
synchronizing again (`t_issue`), then sync and read it again (`t_wall`).

| | rank 0, base step |
| --- | ---: |
| host time to issue the step | **186.06 ms** |
| wall time for the step | 188.22 ms |
| device finishes after the host stops issuing | **2.16 ms** |
| device kernel time, `torch.profiler`, collectives excluded | **122.02 ms** |
| device kernel time, collectives included | ~128 ms |
| **GPU idle** | **~66 ms, 35% of the step** |

The device is never more than about 2 ms of work ahead of the host. The queue runs dry and
stays dry.

Two controls say this is dispatch and not something else:

- **It is not the layer loop.** Replacing `decode_layer`'s body with `return x` leaves a
  **1.91 ms** step. The Python around the ops is free; the ops are not.
- **It is not the collectives.** Removing all 120 all-reduces leaves host issue time at
  170.65 ms against a 172.84 ms wall, the same 2 ms gap. A collective-free step is still
  host-bound.

This regime has a floor and a consequence. The floor is device time: no amount of dispatch
removal takes the step below ~122 ms. The consequence is that below the host issue time,
**device-side optimizations are invisible**, which is section 6.

## 2. Where the 188 ms goes

By ablation, on all four ranks at once, each component replaced by a shape-correct stand-in.
This is the method the previous pass used for the reducer, and it is the only attribution
that is trustworthy here: it prices a component's real in-model contribution, dispatch
included. Per-phase CUDA events were also collected and are *not* used for the table below,
because an event pair around a small op brackets the device's queue backlog rather than the
op, which inflates whatever follows a busy stretch (it put the MoE router at 16.0 ms against
the ablation's 2.10 ms).

Mean of 4 ranks, base step 188.22 ms:

| removed | step ms | cost ms | share |
| --- | ---: | ---: | ---: |
| nothing (base) | 188.22 | | |
| DeltaNet mixers, 45 layers | 124.10 | **64.12** | 34.1% |
| MoE routed kernels + block prep, 60 layers | 129.45 | **58.77** | 31.2% |
| MoE shared expert, dense bf16, 60 layers | 160.74 | **27.49** | 14.6% |
| attention mixers, 15 layers | 177.30 | 10.92 | 5.8% |
| all 120 collectives | 178.32 | 9.91 | 5.3% |
| MoE router (by difference) | | 2.10 | 1.1% |
| layer body entirely (skeleton) | 1.91 | 186.31 | 99.0% |
| residual: 2 rmsnorm/layer, residual adds, embedding, LM head | | 13.00 | 6.9% |

The model is 60 layers, **15 `full_attention` and 45 `linear_attention`** (Gated DeltaNet),
read from `reference/config.json`.

**These do not add up, and the direction is informative.** Removing both mixers saves
85.97 ms where removing them separately saves 64.12 + 10.92 = 75.04. Near the base step the
device has 66 ms of slack, so the first removals give back only what they cost the host; as
enough work is removed the step hits the device floor and behaves differently. In the
`no_mixer` case it already has: host issue 82.95 ms against a 102.33 ms wall, a 19.4 ms gap
where every other row has 2 ms. **That is the transition to device-bound, measured.**

Compare with the PP-era breakdown, which put the two MXFP4 kernels at 86-96% of the step and
bounded everything else, DeltaNet included, at "4-14% residual". The MoE is now 31% and
DeltaNet alone is 34%. The previous pass flagged DeltaNet as the item it had most likely
mis-ranked; it had.

## 3. Under TP=4 the MoE runs the prefill kernels

`torch.profiler` on the real step shows `_grouped_gate_up_kernel` and `_grouped_down_kernel`
at 60 calls each per step, and `_gate_up_silu_kernel` / `_down_combine_kernel` not at all.
The decode path is running the de-duplicating kernels the previous pass measured as *slower*
at decode density and recorded `use_grouped` as "correctly declining".

The comparison is the bug:

```python
if use_grouped(assignments, hi - lo) if grouped is None else grouped:   # hi - lo is LOCAL
```

`assignments` is the full `tokens * top_k` list, because expert parallelism deliberately
hands every rank every assignment and drops the out-of-range ones GPU-side. `hi - lo` is the
rank's shard. So the ratio the heuristic reads is not a density at all:

| topology | assignments | experts passed | threshold `3 x experts` | selected |
| --- | ---: | ---: | ---: | --- |
| PP, 1 rank | 480 | 512 (global) | 1536 | per-assignment, correct |
| **TP=4** | 480 | **128 (local)** | **384** | **grouped, wrong** |
| prefill, either | 5,120 | 512 / 128 | 1536 / 384 | grouped, correct both ways |

The depth that the grouped form actually wins on is assignments per *local* expert, which is
`480 * (128/512) / 128 = 0.94`, unchanged by sharding and nowhere near 3. Sharding the expert
axis must not move this decision, and comparing against the global count is what makes it
not move.

### What it costs, measured

Two arms alternating step by step in one process, so any drift cancels, 14 steps per arm per
rank, run independently on two nodes:

| | base (grouped) | fixed (per-assignment) | delta | |
| --- | ---: | ---: | ---: | ---: |
| node01, rank 0 | 198.65 | 175.19 | 23.46 | |
| node01, rank 1 | 198.58 | 175.11 | 23.47 | |
| node01, rank 2 | 198.57 | 175.02 | 23.55 | |
| node01, rank 3 | 198.34 | 174.68 | 23.66 | |
| **node01 mean** | **198.53** | **175.00** | **23.53** | **1.134x** |
| **node02 mean** | **191.29** | **168.22** | **23.08** | **1.137x** |
| host issue time, node01 rank 0 | 196.41 | 172.87 | 23.54 | |
| host issue time, node02 rank 0 | 189.35 | 166.09 | 23.26 | |

Per-step spreads do not overlap on either node (node02: base 189.15-192.70, fixed
167.00-170.29). Timing the same two arms the way `tp_decode_bench.py` does, ten steps with
one synchronize at the end, agrees on both: 198.82 / 202.49 against 175.91 / 175.53, and
191.58 / 198.51 against 173.52 / 169.13.

**The wall-clock saving and the host-issue saving are the same 23 ms, on both nodes.** The
grouped path costs what it costs because `align_blocks` runs about 15 extra torch ops per
layer (`argsort`, `cumsum` twice, `scatter_add_`, `scatter_`, `scatter_reduce_`, `cummax`,
the gathers between them) and `_grouped_moe` allocates and zero-fills a `[480, hidden]`
combine buffer, roughly 900 extra dispatches per step. Its device time is not the problem and
removing it is not why this is faster.

### End to end, against the patched source

`tp_decode_bench.py` run against the patched source rather than a monkey-patched arm, same
probe and same node as the 189.77 ms on record:

```
stack                                       ms/step   tok/s
--tp 4 + byte-LUT (on record)                189.77   253.5
--tp 4 + byte-LUT + use_grouped fix          166.05   289.1
```

**1.143x, 35.6 more tok/s, 16.3% to 18.6% of the gate.**

That run needed 20 warmup steps. The first attempt, at `tp_decode_bench.py`'s usual 3-4,
returned 188.35 ms and appeared to refute the A/B entirely. It did not: the per-assignment
kernels are not resident after a handful of steps the way the grouped ones are, so a short
warmup charges their JIT and allocator settling to the timed window. **Anything that changes
which kernel the decode path selects needs its warmup raised before it is timed**, which is
worth knowing independently of this fix, because the 189.77 ms on record was taken at the
default warmup of 3.

A note on a check that does *not* work here: the probe's reported logits change between any
two runs with different step counts, because the DeltaNet recurrent state advances in place
on every decode call while the probe feeds the same tokens and positions. Logit movement is
therefore not evidence that a kernel path changed. The timing is.

### The fix

`fused_moe` takes the expert count the assignments are spread over, defaulting to the width
of `expert_range` so the single-rank case is unchanged, and `Model._routed_fused` passes
`c.experts`, which `local_cfg` deliberately leaves global. Applied on this branch.

## 4. Expert-routing skew is 1.04x

The README attributes TP=4's 1.59x (against a projected 2.8x) to routing skew, citing
94 / 158 / 119 / 119 assignments across the four shards and a busiest rank at 1.32x, and
calls balancing it "the obvious follow-up". Measured per rank over all 60 layers of a real
step:

| rank | expert range | assignments (60 layers) | per layer | vs uniform 120 |
| ---: | --- | ---: | ---: | ---: |
| 0 | [0, 128) | 7,267 | 121.1 | 1.009x |
| 1 | [128, 256) | 7,262 | 121.0 | 1.008x |
| 2 | [256, 384) | 6,800 | 113.3 | 0.944x |
| 3 | [384, 512) | 7,471 | **124.5** | **1.037x** |
| | | 28,800 | 480.0 | |

The 94 / 158 split is real but it is *one layer*. Across 60 layers it averages out to 1.04x.

Per-layer skew still costs something, because there is a collective at every layer and each
one waits for that layer's slowest rank. To price it, remove the collectives and let each
rank free-run:

| | rank 0 | rank 1 | rank 2 | rank 3 | mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| free-running, no collectives | 172.84 | 183.83 | 188.77 | 167.83 | 178.32 |
| with collectives | 188.22 | 188.22 | 188.23 | 188.22 | 188.22 |

Every rank runs at the slowest rank's free-run time, as expected. But **the slowest rank
(2, 188.77 ms) is the one with the fewest assignments (6,800) and the fastest (3, 167.83 ms)
is the one with the most (7,471)**. The imbalance is anti-correlated with routing, so
routing is not what causes it; device-to-device variation on the node is the remaining
explanation.

Ceiling on balancing, therefore: perfect balance puts every rank at the 178.32 ms mean, plus
the true collective cost of 6.15 ms (120 x 51.3 us measured standalone at `[48, 4096]` fp32),
so **~184.5 ms against today's 188.22: 3.7 ms, 2.0%, about 5 tok/s.** Not worth doing.

Collectives at the step's shape, measured standalone on all four ranks: fp32 **51.3 us**,
bf16 **53.4 us**. Their whole in-model cost including every layer's wait is 9.91 ms. Fusing
the two per-layer reductions into one would buy under 3 ms, as the README already concluded.

## 5. DeltaNet is the largest single term

64.12 ms over 45 layers is 1.42 ms per layer for a mixer whose decode step carries one token
per slot. Its HBM traffic does not explain that: the recurrent state is a few tens of MB per
rank per layer and the whole mixer's weight and state traffic is ~10 GB/step, about 3 ms at
the 3.03 TB/s this part delivers. It is 20x above its memory roofline and it is made of
small ops: roughly 35 `aten` calls per layer, about 1,575 per step.

Sub-ablations, run in one process in slot order with a `base` control at slot 1 and repeated
at slot 9. That control drifted +9.74 ms, so the raw column understates every later row and
the corrected column removes a linear 1.22 ms/slot. The correction is validated
independently: it puts the `use_grouped` fix at 22.0 ms where section 3's drift-free A/B
measured 23.53 ms.

| part of `deltanet_decode` | raw delta ms | drift-corrected ms |
| --- | ---: | ---: |
| `gated_rmsnorm` | 18.02 | **~24** |
| four input projections | 9.06 | ~16 |
| `delta_rule` recurrence | 12.95 | ~15 |
| `out_proj` | -0.13 | ~8 |
| `causal_conv` | 0.41 | ~5 |
| conv/rec pool gather and scatter-back | -4.46 | ~0 |

Corrected, these sum to ~69 ms against the 64.12 ms whole-mixer ablation, 8% non-additive.
**Treat the ordering and the magnitudes as indicative, not as the 0.2 ms numbers of
section 3.**

`gated_rmsnorm` leading is the clearest single illustration of the regime. It is twelve torch
ops (`float`, `pow`, `mean`, `add`, `rsqrt`, `mul`, two casts, `silu`, two more `mul`) on a
`[48 * v_heads_local, 128]` tensor, a few hundred KB. It costs ~530 us per call because it is
twelve dispatches, not because of anything it computes.

## 6. Why the byte-LUT decode bought 5%

The record is TP=4 at 199.49 ms and TP=4 + byte-LUT at 189.77 ms, a 1.05x where the kernel
change measures 1.6-1.7x standalone. The README explains this as the MoE being a smaller
share of a TP rank's step, an Amdahl argument. **That explanation does not survive the
ablation.** The routed MoE is 58.77 ms of the 188.22 ms step, 31%; a 1.65x on 31% of the step
would be 1.24x overall, not 1.05x.

The measured explanation is section 1. The byte-LUT removes device time from a step that has
66 ms of device slack, so it removes idle rather than latency. The two changes of this
campaign are a controlled pair:

| change | removes | measured effect on the step |
| --- | --- | ---: |
| byte-LUT MXFP4 decode | device time | 9.72 ms, inside the 178-199 ms run-to-run spread |
| `use_grouped` fix | ~900 dispatches | **23.1-23.5 ms, drift-free, 2 nodes, 4/4 ranks** |

The same argument re-reads TP=4's own result. TP=4 cut MoE device time roughly fourfold and
returned 1.59x rather than 2.8x not because of routing skew (section 4) but because it drove
device time under the host issue floor at ~190 ms and stopped there. **TP=4's real
achievement is that it took the step from device-bound at 317 ms to host-bound at ~190 ms.**
Everything since has been pushing on a term that is no longer binding.

## 7. Ranked recommendations

Ordered by measured gap closed per unit of work. Only row 1 is measured end to end; rows 2-4
are projected from the measured components named beside them, and the projection method is
"host issue time falls by the dispatch removed, until it reaches the ~122 ms device floor".

| # | change | step ms | tok/s | % of gate | basis |
| ---: | --- | ---: | ---: | ---: | --- |
| 0 | today | 189.77 | 253.5 | 16.3% | measured |
| 1 | `use_grouped` fix | **166.05** | **289.1** | **18.6%** | **measured end to end, 1.143x** |
| 2 | + fuse DeltaNet's small ops | ~130 | ~370 | ~24% | projected from 64.12 ms measured |
| 3 | + eliminate dispatch (graph capture) | ~122 | ~393 | ~25% | projected; equals measured device time |
| 4 | + dense GEMM efficiency | open | | | 48.9 ms measured, not root-caused |
| -- | MoE HBM floor under TP=4 | 15.9 | | | prior pass, measured bandwidth |

### 1. Fix `use_grouped` (measured 1.143x, about 6 lines)

Section 3. Applied on this branch. It is the only item here measured end to end on hardware,
it is the cheapest, and it does not interact with anything else. Do it first. Its own
accuracy-gate run is in section 8: 13/14, unchanged verdict and unchanged failure mode.

### 2. Cut DeltaNet's op count (projected ~35-45 ms)

DeltaNet is 64.12 ms of measured in-model cost across ~1,575 dispatches. Three pieces, in
order of measured size:

- **A Triton `gated_rmsnorm`.** Twelve ops to one, ~24 ms measured today. The largest single
  win per line of code in this document after item 1, and it is a leaf function with an
  obvious contract, so it is testable in isolation.
- **Concatenate the four input projections.** `in_proj_z`, `in_proj_qkv`, `in_proj_b` and
  `in_proj_a` all read the same `x`. One GEMM plus a split replaces four, removing 135
  `aten::mm` dispatches per step against ~16 ms measured.
- **A Triton kernel for the decode recurrence.** At `T = 1` `delta_rule_recurrent` is about
  ten elementwise and reduction ops over the recurrent state, ~15 ms measured.

The same treatment applies to `rmsnorm` (120 calls/step, eight ops each) and to the shared
expert's `gate_proj`/`up_proj` pair, which are inside the 27.49 ms shared-expert term.

Note the ceiling: these are dispatch removals and they stop paying once host issue time
reaches device time. From 172.87 ms (after item 1) there is about 50 ms of headroom before
the ~122 ms floor, so item 2 can be taken nearly in full but item 2 plus item 3 do not add.

### 3. Eliminate dispatch: revisit graph capture (projected 1.4x, structural)

This is the item the previous pass ruled out by name, on a premise section 1 falsifies. A
captured step costs one replay instead of ~7,400 dispatches, which takes the step directly to
the device floor whatever the op count is, and, more importantly, **restores the property
that device-side work is worth doing.** Until dispatch is removed, every kernel improvement
lands in the 66 ms of idle.

Capture currently fails with `hipErrorStreamCaptureInvalidated` after 115 s in `prepare()`.
The README states an untested hypothesis with a cheap experiment attached: `attn_decode_static`
passes `enable_gqa=True` and an `attn_mask` to SDPA, which on gfx942 falls off the fused
backend and materializes the KV broadcast, giving gigabyte transients per layer and so a
likely allocator `hipFree` inside the capture. Writing the masked case out as matmul, masked
softmax, matmul and re-running capture is the experiment. That hypothesis is unchanged by this
pass, but its priority is not: it moves from "revisit only if the step ever gets under 20 ms"
to the second structural item.

### 4. Dense GEMM efficiency (measured, not root-caused)

`aten::mm` is **586 calls and 48.9 ms of device time per step**, averaging 83.5 us, which is
the largest device-time term in the step, larger than the two MXFP4 kernels at 28.8 ms
together. Every one of these is `M = 48`, GEMV-shaped, and rocBLAS is selecting large-M tiles
for them: one kernel, `MT256x224x64`, accounts for 150 calls and 25.6 ms at 171 us each. At
these shapes the cost should be the weight bytes, a few microseconds. This is roughly an
order of magnitude above roofline and it was not chased in this pass. It becomes the binding
term once items 1-3 land, and it should get its own pass then, not before: it is device time,
and device time is currently free.

### 5. Not recommended, with the number

- **Balancing expert-to-rank assignment.** Ceiling 3.7 ms, 2.0%, ~5 tok/s (section 4). The
  premise on record, a 1.32x busiest rank, is a single layer's routing.
- **Fusing the two per-layer collectives.** The entire collective term including all waiting
  is 9.91 ms; the true collective cost is 6.15 ms. Halving the count buys under 3 ms.
- **Further MXFP4 kernel work, including tile tuning.** 28.8 ms of a 122 ms device budget in
  a step that is not device-bound. Revisit after item 3.
- **`bf16` collectives.** 53.4 us against fp32's 51.3 us, so the fp32 upcast that protects
  the pins costs nothing. Unchanged from the README.

## 8. Effect on the accuracy gate

The gate has failed all campaign at 13/14, with 10/14 greedy pins exact and `pin14` diverging
before token 8. Nothing here is aimed at it and nothing here explains it. One finding does
bear on it.

**The accuracy run on record for v4 did not exercise the kernels the campaign believes it
validated.** Section 3 shows the decode MoE under TP=4 runs `_grouped_gate_up_kernel` and
`_grouped_down_kernel`. The byte-LUT decode's bit-identity argument, and the README's
inference that v4's gate result is "consistent with the byte-LUT decode's own bit-identity
claim at model dimensions", were established on the per-assignment kernels and their
`_dequant_tile` path. The grouped kernels dequantize through `_dequant_rows` and reduce in a
different order. The claim may still hold, but it has not been checked on the path that
actually ran.

Consequently the `use_grouped` fix was gated on its own rather than inheriting that result.
Real server, same node and checkpoint, `--max-batch` 48:

```
stack                                gates    greedy pins (need >= 90% exact on 32 tokens)
--tp 4 + byte-LUT (on record)        13/14    10/14 exact, early divergence {pin14: 1}
--tp 4 + byte-LUT + use_grouped fix  13/14     8/14 exact, early divergence {pin14: 1}
```

**The fix does not change the gate's verdict, its failure mode, or `pin14`.** All 13
behavioural probes pass either way and the same single pin diverges before token 8. It does
move the exact-pin count by two, which is the float-reassociation change the kernel swap
predicts: both numbers are far below the 12.6/14 the gate needs, so this is two near-ties
flipping, not a regression with a direction. Do not read 10 to 8 as the fix making accuracy
worse; read it as confirmation that the fix is not numerics-neutral and that the pin count is
not a stable statistic at this distance from the threshold.

Ranked against the gate, none of section 7's items is numerics-neutral except item 3, and
item 3 is the one that is blocked. Items 1 and 2 both move reduction order. Item 1 is at
least cheaply reversible without a revert: `SEED_MOE_GROUPED_MIN` sets the threshold, so the
old behaviour is one environment variable away for a bisect.

Nothing found in this pass explains `pin14`, which has diverged early under every topology
and every kernel the campaign has run.

## Appendix: method

Four probes, all on 4x MI300A via `srun --overlap` on existing allocations, none added to the
repo. Each runs one process per rank, re-execs itself for ranks 1.. exactly as `server.py`
does, and drives every rank through `tp_driver.Broadcaster`, so the collectives and the
command broadcast are inside every measured step. Each keys its behaviour on a *decode-call
index* so that rank 0 and the workers switch mode together without adding a command to the
protocol, and each rank writes its own JSON.

1. **Phase profile.** Per-rank CUDA-event accumulators around each phase of `decode_layer`,
   `moe` and `tp.all_reduce`; `torch.profiler` over three steps for the kernel taxonomy;
   per-rank expert-assignment counts over one step; and a standalone all-reduce benchmark at
   `[48, 4096]` run before the weights load. The event phases are reported here only where
   the ablations corroborate them, for the reason given in section 2.
2. **Component ablation.** Each component replaced by a shape-correct stand-in on all four
   ranks at once, eight steps per arm, plus `t_issue` on every step for the host/device
   split. This is the primary attribution.
3. **DeltaNet sub-ablation.** The same, inside `deltanet_decode`, with a repeated `base`
   control to measure drift.
4. **Interleaved A/B.** The two `use_grouped` arms alternating step by step, 14 steps per arm
   per rank, plus the saturated ten-step-one-synchronize form for comparability with
   `tp_decode_bench.py`.

Three things cost this pass a wrong answer before they were caught, and will cost the next
one the same:

- **`tp_decode_bench.py`'s default warmup is too short for a kernel-path change.** At 3-4
  steps the per-assignment kernels are still settling and the fix measured 188.35 ms; at 20
  it measures 166.05 ms. Raise warmup whenever the change alters which kernel is selected.
- **The probe's logits are not a path fingerprint.** They move between any two runs with
  different step counts, because the DeltaNet recurrent state advances in place while the
  probe replays the same tokens and positions.
- **`curl` is present but broken in this container** (missing `libldap_r-2.4.so.2`), so a
  `curl`-based server-readiness loop never succeeds. A loop that then idles for its full
  timeout leaves the TP workers blocked on the command broadcast and trips RCCL's 600 s
  watchdog, which looks exactly like the server instability on record and is not.

Assumptions worth challenging if a later run disagrees:

- **Device kernel time of 122.0 ms** comes from `torch.profiler` over three steps, summing
  leaf kernels and excluding the RCCL kernel, whose reported residency (314 ms/step) is
  spin-wait and not work. The profiled steps carry profiling overhead, which lands on the
  host side, so the device figure should be close; it is the one number here that a second
  method has not confirmed, and the whole of section 7's floor rests on it. The cheap
  confirmation is a `rocprof` kernel-time sum on an unprofiled step.
- **35% GPU idle** follows from that figure and from the directly measured 186.06 ms host
  issue time. The host measurement is robust and reproduced in every probe; the idle
  percentage inherits the device figure's uncertainty.
- **Ablations are not additive** (section 2), because the step crosses from host-bound to
  device-bound as work is removed. Every "cost ms" is therefore a cost *at today's operating
  point*, and the section 7 projections, which stack several of them, are correspondingly
  soft. The ~122 ms floor is the part of those projections to trust.
- **Routing counts** are one step's. Routing is input-dependent and this is the synthetic
  probe's garbage-token prompt at position 32. The 1.04x aggregate could differ on real text;
  the structural point, that a per-layer 94/158 split is not a per-step 94/158 split, does
  not depend on the input.

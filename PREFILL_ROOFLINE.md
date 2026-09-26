# Prefill roofline: one packed 2048-token prefill, TP=4, per rank

Status: source analysis and offline gfx942 ISA only (no GPU, 2026-09-23). Every "now"
number below is an estimate with its basis stated; `bench_prefill_moe.py` measures the
two largest terms (MoE per layer, DeltaNet recurrence per token) and should replace them.

Per-rank budget: 980 TFLOP/s bf16 dense MFMA, 5.3 TB/s HBM (shared with the host).
Shapes per rank: hidden 4096; DeltaNet 16 value heads (4 key heads) x 128; GQA 8 query
heads + 1 KV head x 256; MoE 128 local experts of 512, top-10, intermediate 1024, MXFP4.

At T=2048, top-10 gives 20,480 assignments over 512 experts, i.e. **~40 rows per expert**
(not 160: each rank owns a quarter of the experts *and* a quarter of the assignments), so
~5,120 local assignments per rank per layer, and every local expert is touched.

## Ranking by gap (T=2048, whole model)

| # | Op (layers) | Kernel today | FLOP, bytes | SOL | Now (est.) | Gap |
|---|---|---|---|---|---|---|
| 1 | DeltaNet recurrence (45) | `fused_recurrent_prefill`: one program per (seq, head, 32 v-cols) = 64 programs per sequence, 2048 serial steps, 4 warps; per step (gfx942 ISA) 36 scalar `global_load_ushort`, 12 `s_barrier` (4 cross-warp reductions), and step t's loads head step t's math | 3.2 GF, <10 MB per layer | ~1 ms total (chunked form on MFMA) | 2-15 us/step: **180-1400 ms** | largest, most uncertain |
| 2 | Routed MoE (60) | `mxfp4_gemv._grouped_moe`: BLOCK_M=16 `tl.dot` blocks, per-element scale fetch+mul, ~3 re-dequants of each expert, un-weighted `y[20480, 4096]` zero-filled + 4 torch combine ops | 129 GF, 855 MB weights per layer | 0.16 ms/layer, **10 ms** | 3-4.5 ms/layer (scaled from the measured 9.68 ms at 5,120 assignments / 833 blocks): **180-270 ms** | 170-260 ms |
| 3 | All-reduce (120) | RCCL / custom AR, 16.8 MB bf16 each, not overlapped | 2 GB moved | ~20 ms (xGMI) | ~30-36 ms | 10-15 ms |
| 4 | Elementwise glue (60) | rmsnorm x2, residual, conv1d, softplus/sigmoid, `repeat_interleave` q/k to 16 heads, gated rmsnorm, rope | ~15 passes x 8-16 MB per layer | ~3 ms | ~8-12 ms | 5-9 ms |
| 5 | Dense bf16 GEMMs (60) | hipBLASLt `F.linear`: DeltaNet in/out (29.5 M params), GQA q/k/v/o (27.3 M), shared expert (3.1 M), router (2.1 M) | ~8.4 TF total | 8.6 ms | ~13 ms (65% of peak) | ~5 ms |
| 6 | MoE routing/combine glue (60) | argsort-based `align_blocks` (~13 torch launches) + zero-fill and combine of `y` | ~0.6 GB per layer | ~7 ms | ~12 ms | ~5 ms |
| 7 | GQA attention core (15) | varlen Triton / SDPA, causal | 17 GF total | 0.3 ms | 2-5 ms | 2-5 ms |

Sum of SOL terms: ~50 ms per 2048 tokens (~0.025 ms/token, the "0.03 ms/token" floor).

Where the 2.2 s goes. With `SEED_PACKED_PREFILL_VEC=0` (the default) a packed call still
runs DeltaNet one sequence at a time, so its recurrence steps are serial across the whole
pack; the vectorized path measured 0.3 ms/token against 1.0. That 0.7 ms/token difference
(1.4 s per 2048 tokens) is the per-sequence serialization, dominated by the recurrence's
serial steps (and per-sequence small GEMMs). It implies up to ~15 us per recurrence step,
well above the 2-3 us that the ISA suggests (load latency plus 12 barriers). The bench's
`deltanet_recurrence_us_per_token` separates the two: x 45 layers x 2048 gives its share.

## Conclusions

1. **Routed MoE: built.** `prefill_moe.py` (`SEED_PREFILL_GROUPED_MOE=1`, T >= 256): grouped
   GEMM with 64-row expert blocks (MegaBlocks-style), MXFP4 decoded in registers, `tl.dot`
   on `v_mfma_f32_32x32x8_bf16`, two-trip ping-pong prefetch, routing weight in the down
   epilogue, atomics-free combine over live rows only. Offline ISA per K trip per wave:
   8,192 weight values per workgroup, ~170 VALU and 16 MFMA per wave (~680 VALU cycles vs
   ~512 MFMA cycles: decode-bound by ~30%), 6 global loads
   in flight ahead of the decode, 184/196 VGPRs (occupancy 2), no spills.
   Predicted: gate_up ~0.3 ms + down ~0.15 ms + align/combine ~0.1 ms = **~0.5-0.7 ms per
   layer, 30-42 ms per 2048-token prefill** (from 180-270 ms), i.e. 140-230 ms saved.
2. **DeltaNet recurrence: chunked form built.** `deltanet_prefill_chunked.py`
   (`SEED_DELTANET_PREFILL_CHUNKED=1`, >= 128 rows; both model prefill call sites opt in,
   the captured MTP verify path does not): the WY/chunkwise form (Yang et al. 2024, Gated
   DeltaNet 2025), same math as `model.delta_rule_chunked`. Prep kernel, parallel over
   (chunk, head): norms, cumulative decay, `A` and `intra` by `tl.dot`, forward
   substitution for `U`, `W`. State kernel, serial over chunks only: three fp32 `tl.dot`s
   per chunk (`v_mfma_f32_32x32x2_f32`). Serial depth 32 + T/32 = 96 at T=2048 instead of
   2048. Offline ISA (chunk 32, 4 warps): prep 457 / state 483 VGPRs, no spills; state loop
   160 MFMA + 321 LDS per chunk step (~1.5-2 us), prep ~570 VALU per substitution row.
   Predicted: prep ~170 us + state ~120 us = **~0.3 ms per layer, ~13 ms per 2048-token
   prefill** (from 180-1400 ms). Cheaper knobs on the old kernel, for comparison in the
   bench: `SEED_DELTANET_PREFILL_PREFETCH=1` (load step t+1 first) and
   `SEED_DELTANET_PREFILL_WARPS` (1/2 turn the cross-warp reductions intra-wave).
3. `SEED_PACKED_PREFILL_VEC=1` should be on for any packed prefill: it removes the serial
   per-sequence recurrence outright (measured 1.0 -> 0.3 ms/token).
   Predicted with 1-3: MoE 30-42 + DeltaNet ~13 + all-reduce 30-36 + dense GEMMs ~13 + glue
   ~20 + attention 2-5 = **~110-130 ms per 2048 tokens (~0.06 ms/token)**, 2-2.5x the SOL sum.
4. After 1-3, the remaining gap is all-reduce (not overlapped with compute) and glue, each
   ~10 ms: overlap the MoE all-reduce with the next layer's norm/projection, fuse glue.

## Appendix: derivations

- MoE FLOP: 2 x 5120 x 4096 x 2048 (gate_up) + 2 x 5120 x 1024 x 4096 (down) = 129 GF.
  Bytes: 128 x (4.46 + 2.23) MB = 855 MB. With 64-row blocks and ~40 rows per expert, MFMA
  work is ~1.5x the live rows (padding); dequant work is 1 pass per block (~1.2 per expert).
- New kernel, per wave per trip (BLOCK_K 64, 4 waves): 64 x 64 x 2 weights for gate_up /
  4 = 2,048 values decoded (~5.3 VALU lane-ops per value incl. scale and bf16 pack; a
  bf16-direct decode like `_BW_ASM` would halve it), 16
  `v_mfma_f32_32x32x8_bf16` at 32 cycles. gate_up: ~170 blocks x 16 N-tiles x 64 trips on
  228 CUs x 2 workgroups; down: ~170 x 32 x 16 trips.
- Old kernel basis: `mxfp4_gemv.BLOCK_M`'s docstring table (512-token chunk, 512 experts,
  5,120 assignments, 833 blocks: 9.68 ms); at T=2048 per rank the same assignment count
  lands in ~400 16-row blocks.
- DeltaNet step: per program per step, 3 vector loads (q, k: 128; v: 32) + 2 scalars,
  4 reductions over 128 (2 l2-norms, `k^T S`, `q^T S`), all on the serial path.

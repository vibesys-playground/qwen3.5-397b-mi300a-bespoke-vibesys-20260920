# MTP speculative decoding for the Qwen3.5-397B-A17B bespoke engine

Branch `bespoke/opt-mtp-spec`, based on `bespoke/opt-batched-prefill` (`1cf24df8`). Behind
`SEED_MTP=1` / `SEED_MTP_K=3` (default). Implementation: `mtp.py`, plus hooks in `model.py`,
`scheduler.py`, `tp_driver.py`, `graph_decode.py`. Tests: `seed_tests/test_mtp.py`.

## Checkpoint facts (read from config/index metadata on the login node, no weights loaded)

`mtp_num_hidden_layers: 1`, `mtp_use_dedicated_embeddings: false`. The `mtp.*` tensors are:

- `mtp.layers.0.self_attn.*`: **full attention**, identical shapes to the target's own
  full-attention layers (`q_proj` `[16384,4096]` = 32 heads x (2x256), `k_proj`/`v_proj`
  `[512,4096]` = 2 KV heads x 256, `q_norm`/`k_norm` `[256]`). **Not** a DeltaNet layer: it
  carries a KV cache, not recurrent state.
- `mtp.layers.0.mlp.*`: the same MoE as a target layer (512 experts, MXFP4, a shared expert,
  `mlp.gate` router). `mtp.layers.0.{input,post_attention}_layernorm`: the same pre-norm
  transformer block shape as a real layer.
- `mtp.fc.weight` `[4096,8192]`, `mtp.pre_fc_norm_embedding.weight`,
  `mtp.pre_fc_norm_hidden.weight`, `mtp.norm.weight`: `fc(concat(norm(embed), norm(hidden)))`
  feeds the block above in place of a previous layer's residual stream.
- No `mtp.lm_head.weight` / `mtp.embed_tokens.weight`: both are the target model's own
  (confirmed by `mtp_use_dedicated_embeddings: false` and their absence from the index).

So the MTP module is architecturally "one more decoder layer, fed differently", not a
distinct model. This drives most of the design below: its self-attention and MoE reuse the
target model's own loading and math verbatim; only the `fc` combine step and its own paged KV
state are new.

## 1. Draft

`mtp.draft` runs `k` autoregressive steps of the MTP layer alone: at step 0 the seed is the
target's own last raw (pre-final-norm) hidden state and last committed token; at step `s > 0`
it is the MTP layer's own previous output and its own previous argmax. One `fc`/attention/MoE
call serves both, since both feed the same two norms and the same block.

Its self-attention keeps a sequence-persistent paged KV pool (`MTP.pool`) using the target's
block tables and absolute positions. Prompt and ordinary target forwards populate each row
from token `p` paired with target hidden state `p-1`. A speculative round writes its base at
`p` and later drafts at `p+s`; after verify, accepted rows are rebuilt from the true target
hidden stream. Rejected rows lie past the visible position and are overwritten before a
later draft can read them. Copy on write copies MTP rows with target attention rows.

Because there is no dedicated embedding/LM head, greedy drafting is exact argmax against the
target's own `lm_head`, no separate small vocabulary or extra weight to keep resident.

## 2. Verify

`mtp.verify_and_commit` feeds `base_token` (the last committed token) plus the `k` draft
tokens, `k+1` tokens total, through the *target* model, batched across every active slot, and
computes each slot's greedy accept length: the longest prefix where the draft token equals
the target's own argmax at that step, plus one bonus token (the target's argmax one past the
accepted prefix, always genuine model output, not a draft guess).

**Wide-batched implementation** (this PR): the `k+1` tokens for every active slot are fed as
one `[B, k+1]` grid through `Model.decode_layer_verify`, one dispatch per layer instead of
`k+1` sequential `Model.decode_layer` calls:

- Dense projections and MoE are already per-token-independent, so they run on the flattened
  `B*(k+1)` tokens exactly as `forward_packed`'s prefill path already does; MoE's expert dedup
  only improves at the wider batch.
- Full attention is one `attn_verify` call per layer: `k+1` query rows per lane against the
  shared paged KV pool, causal with an offset among the new rows (the same shape
  `prefill_attention` already computes for a prefill chunk, here read through the paged pool
  instead of a dense window and batched across every lane in one call).
- DeltaNet is one `cu_seqlens`-packed `deltanet_fused.fused_recurrent_prefill` call per layer
  across all `B` slots at once (`Model.deltanet_verify`), not a per-sequence loop
  (`deltanet_packed`'s existing loop would cost `B` kernel launches per layer, which is worse
  dispatch-count than the old unrolled-by-step design, not better).

Each layer's causal structure (row `s` only ever sees rows `0..s`) means row `s`'s output is
*by construction* what `s+1` sequential real decode steps on these tokens would have computed
-- the same correctness argument the previous unrolled design made, now argued per-row instead
of per-step. This is exactly the property the equivalence tests check, and they pass unchanged
against this rewrite (the tests are black-box against `verify_and_commit`'s contract). See
"Cost model" for the dispatch-count argument this rewrite is for.

## 3. Recurrent-state rollback

After verify, attention-layer KV needs no rollback at all: every fed position's KV row was
written at its real absolute position regardless of acceptance, and the *next* round's
position argument only ever advances to `positions[j] + accept_len[j] + 1`, so an
unaccepted row is simply never read, and is overwritten (at that same absolute position) by a
later round before it could be misread. This is the existing `reset()`/prefix-snapshot
philosophy this codebase already uses for KV ("masked by length"), unchanged.

DeltaNet's recurrent `conv`/`rec` state is not free: the state after processing all `k+1` fed
tokens is contaminated by whichever draft tokens were rejected, and there is no
"masked by length" escape hatch for a matrix that has been genuinely mutated by the wrong
inputs. Three options, per the task:

- **(a) Per-step state snapshots (rejected).** Have the DeltaNet kernel emit every
  intermediate per-token state, not just the final one, then select the accepted length's
  snapshot. This is the cheapest option *in principle* (a `[B, heads, k_dim, v_dim]` fp32
  snapshot per step is ~1 MB per slot per layer per step, ~2 GB/rank at `B=48`, `k+1=4`, 45
  layers -- affordable), but `deltanet_fused.fused_recurrent_prefill` as it exists today
  returns only the *final* per-sequence state (`initial_state` mutated in place; there is no
  intermediate-state output). Getting it would mean modifying the Triton kernel itself, which
  is GPU-only work this phase cannot validate ("don't use a GPU until told", and never run
  Triton under `TRITON_INTERPRET`). Revisit once there is GPU time to validate a kernel change.
- **(b) Replay the accepted prefix (chosen).** Keep the pre-verify `conv`/`rec` snapshot (one
  clone per layer, taken before the wide verify call mutates it) plus the raw per-token conv
  input and mixer inputs (`q`, `k`, `v`, `g`, `beta`) the verify call already computed. Once
  `accept_len` is known, correct the state in one more wide dispatch per layer instead of `B`
  sequential ones: conv's state is a fixed-width window of *raw* inputs, so it is exact by
  slicing the already-computed `raw_qkv` at the right per-slot length -- no recompute, no
  kernel call (`Model.deltanet_verify_rollback`'s conv branch). The recurrent state has no such
  window shortcut, so it is replayed via a *second* `cu_seqlens`-packed
  `fused_recurrent_prefill` call, seeded from the pre-verify snapshot, with each slot
  contributing only its own `accept_len[j] + 1` tokens (`cu_seqlens` packs variable per-slot
  lengths natively, so this is one wide kernel launch across all `B` slots, not `B` of them,
  and its cost is bounded by `k+1 <= 4` tokens per slot). **This is what makes (b) viable
  without a kernel change**: a literal reading of "recompute" as `B` sequential per-slot
  replays would cost as much as re-decoding the whole accepted prefix one token at a time,
  which is what MTP exists to avoid -- packing the replay into the same wide-dispatch shape
  the verify pass itself uses avoids that cost entirely.
- **(c) Custom kernel (rejected for now).** Modify `fused_recurrent_prefill` to expose
  intermediate states directly, turning (a) into a real option. Same asymptotic cost as (a),
  same GPU-only-work objection: this phase's checks are CPU-only, so a kernel change here is
  unvalidatable until GPU time is available. Revisit once (b)'s wide-dispatch replay is
  measured and, if it turns out to cost more than expected at high rejection rates, compare
  against actually modifying the kernel.

(b) is the only one that needs no Triton kernel change, which the CPU-only phase of this task
requires; its wide-dispatch replay keeps it at the same `O(layers)` dispatch count per round as
the rest of the wide-batched verify, not `O(layers x B)`.

## 4. Graph capture

Both shapes are fixed per `SEED_MTP_K` and so are capturable in principle, following the
existing `Buffers`/`CaptureBackend`/`Segment` pattern in `graph_decode.py`:

- **Draft**: `batch x 1` through the MTP layer alone, `k` times, the same static-buffer shape
  `decode_layer_static` already uses, just against `MTP.pool` instead of a real layer's pool.
- **Verify**: `batch x (k+1)`, a new fixed shape next to the existing `batch x 1` decode
  capture.

`graph_decode.GraphDecodeRunner.speculative_decode` is a plain passthrough to
`Model.speculative_decode` in this PR: there is no GPU here to capture against or validate a
capture on, and shipping an unexercised capture path would be exactly the kind of untested
graph-capture change this bundle's own history (`graph_decode.py`'s docstring) warns about.
Capturing it is GPU-phase follow-up work, not a design gap: the shapes do not change between
now and then.

## 5. Scheduler / TP driver integration

One new `Runner.speculative_decode(slots, tokens, positions) -> list[list[int]]`, gated by
`SPECULATIVE_DECODE` (`SEED_MTP`, read independently in `scheduler.py` since it stays
torch-free) and only taken when every request in the step is greedy (`temperature <= 0`):
MTP's own drafting is argmax-only, so a step with any sampling request falls back to the
ordinary `decode` + `sample_batch` path, same as today. `_decode_step` applies each slot's
returned token batch (1..k+1 tokens) through the existing `_emit` one token at a time, so stop
tokens and `max_new` budgets need no new logic; `_InFlight.done` (set by `_release`) lets
that inner loop stop early without re-scanning `self.decoding`.

Compatible with prefix snapshots and batched prefill by construction, not by special-casing:
`save_prefix`/`load_prefix` are unchanged (MTP's own KV pool never persists across rounds, so
there is nothing of the MTP module's own to snapshot), and prefill (`forward`/`forward_packed`)
only gains one addition, stashing the raw last-position hidden state per slot
(`cache_hidden`/`hidden_scratch`, the same one-row-per-slot pattern `cache_logits` already
uses) so the first decode round after prefill has a seed for `mtp.draft`.

TP driver: one new op, `SPECULATIVE_DECODE`, broadcast once per round with the same payload
shape `DECODE` already uses (`slots`, `tokens`, `positions`). Unlike ordinary per-token
sampling, which needs rank 0's drawn token relayed before the next token can be fed, MTP's
draft is deterministic argmax with no rank-0-only step, so every rank computes the identical
draft, verify, accept, and rollback from the one broadcast input with no further round trips.

## 6. Correctness

`seed_tests/test_mtp.py`, against a real (if tiny) checkpoint with a synthetic `mtp.*` shard
(the installed reference `transformers` code does not implement the MTP module at all, so
there is no HF model to source real tiny MTP weights from):

- `greedy_accept_length` unit tests (full/no/partial match).
- `test_verify_and_commit_matches_sequential_decode`, parametrized over `accept in 0..k`:
  forces each accept length by constructing draft tokens that match the real greedy
  continuation for exactly `accept` positions, then a token that cannot be the argmax. Checks
  both the committed tokens *and* every stateful side effect (DeltaNet state, KV, the cached
  hidden `mtp.draft` reads next) by running one further ordinary decode step on the
  speculative and the sequential model and comparing logits: any rollback bug changes them.
- `test_verify_and_commit_batched_matches_solo`: two slots, two different forced accept
  lengths, one batched call, checked against each slot run alone (no cross-slot leakage in
  the vectorized snapshot-select).
- `test_speculative_decode_matches_sequential_for_several_rounds`: the actual correctness
  requirement, end to end, real `mtp.draft` predictions (whatever they are) rather than
  forced ones -- several rounds of `Model.speculative_decode` must commit exactly what the
  same number of ordinary `Model.decode` steps would have, which the task states holds
  "modulo bf16 nondeterminism from the batch shape" (the CPU tests run in fp32, so they check
  it bit-for-bit/close-tolerance; the bf16 caveat is a GPU-phase concern).
- `test_speculative_decode_returns_valid_batch_shape`: batch-3 plumbing sanity.

All ten pass; the full pre-existing `seed_tests/` suite (403 tests) and `ruff check` are
unaffected (see PR).

## Cost model

Baseline (given): 77 ms/step at batch 48, ~620 tok/s decode capacity. SGLang's own EAGLE
baseline on the same load-ramp workload: 883 tok/s peak at 48 concurrent requests.

**This PR's verify is wide-batched, which is the performance target this cost model is
against.** Every layer runs once over the flattened `[48*(k+1), hidden]` grid instead of `k+1`
decode-shaped calls (`Model.decode_layer_verify`, see "Verify" above), so the per-layer host
dispatch cost -- what `TP_BYTELUT_BOTTLENECK_2026-09-22.md` measures as this step's actual
bottleneck, not GEMM flops -- is paid once per layer per round, not `k+1` times. The rollback
path adds one extra wide `cu_seqlens` dispatch per DeltaNet layer (45 of them), which is small
next to the layer stack's own dispatch count (attention + MoE + DeltaNet dispatches across 60
layers).

One round costs roughly `T_draft + T_verify_wide`. `T_draft` is cheap: `k` rounds through one
layer (not 60) plus the `fc` combine, on the order of a few ms each. `T_verify_wide` is closer
to one ordinary decode step's cost than to `k+1` of them: the extra work relative to an
ordinary batch-48 step is `(k+1)x` the GEMM *flops* plus one extra DeltaNet dispatch per layer,
both of which are far from the dispatch-bound floor a batch-48 step already sits at, so
widening the batch is closer to free than to `4x`. Call that
`T_round ~= T_draft + 1.3 x T_decode ~= 10 ms + 100 ms = 110 ms` (a deliberately conservative
multiplier; GPU measurement replaces this estimate -- see "What GPU validation should
measure"). Tokens per round average `1 + E[accept]`. Solving for the acceptance rate needed to
reach SGLang's 883 tok/s at batch 48 (`48 x (1 + E[accept]) / (T_round / 1000) = 883` tok/s):

| E[accept] (of 3) | tok/s @ T_round=110ms | tok/s @ T_round=90ms |
| --- | --- | --- |
| 0.5  | 655  | 800  |
| 1.0  | 873  | 1067 |
| 1.5  | 1091 | 1333 |
| 2.0  | 1309 | 1600 |

So an average acceptance around 1 of the 3 draft tokens (a 33% hit rate) is roughly break-even
against the 883 tok/s target at a plausible `T_round`, and the design does not need anywhere
near full 3-token acceptance to close most of the gap to SGLang. This is the number GPU
validation should measure directly, not assume.

## GPU measurement (job 648527, real Qwen3.5-397B-A17B-MXFP4 checkpoint, TP=4 on 4x MI300A)

**Correctness**: `speculative_decode`'s committed tokens matched sequential `Model.decode`'s
greedy output exactly over several rounds, real `mtp.*` weights, real prompt. This also
surfaced and fixed a real bug the CPU tests structurally cannot reach: `deltanet_fused.
fused_recurrent_prefill` requires its `v` argument contiguous along the last axis, but `v` (
unlike `q`/`k`, which `repeat_interleave` already materializes contiguous) reaches the kernel
as a transpose-then-split view of `causal_conv`'s output and is not. Fixed in both the
pre-existing single-sequence `Model.delta_rule` dispatch and the new `Model.deltanet_verify`.

**Timing**, batch 48, `k=3` (`t=4`), eager (graph capture for verify is not implemented --
see "Graph capture" above): wide-batched `draft` + `verify_and_commit` measured 241.15 ms
against this same run's own ordinary decode step at 120.95 ms -- a 1.99x ratio, not the
~1.3x this doc estimated. Scaled onto the 77 ms reference decode step, that is ~153.5 ms per
round. This is worse than assumed, though still far better than the rejected unrolled
design's `4 x 77 ms ~= 308 ms` floor (dispatch-count amortization did work, just not as much
as hoped -- likely MoE/attention dispatch cost genuinely growing with the 4x wider batch more
than the "closer to free" assumption below, plus the two extra DeltaNet rollback dispatches
per round). Re-solving the acceptance-rate table at `T_round ~= 165 ms` (using the projected
153.5 ms plus a small `T_draft`): `48 x (1 + E[accept]) / 0.165 = 883` needs `E[accept] ~= 2.0`
of 3 -- roughly 2 of 3 draft tokens accepted, not the ~1 of 3 this doc previously estimated.
At the run's own absolute decode time (120.95 ms, i.e. this workload/hardware state without
rescaling to the 77 ms reference), even 100% acceptance (`E=3`) does not reach 883 tok/s,
which argues for shrinking `T_verify_wide` further (a fused paged-attention kernel for
`verify_attention_paged`, and/or reducing the DeltaNet rollback pass's cost) before this is
competitive, not just for measuring acceptance rate on the real workload.

## What GPU validation should measure

1. **Acceptance rate on the load-ramp workload**: per-round accepted draft count (0..3),
   histogrammed, and its mean, at the actual serving concurrency (48 sessions). This is the
   one input the cost model above cannot get from CPU work at all.
2. **Verify step time at batch 48 x 4 tokens**, as now implemented (wide-batched, target near
   1.3x a single ordinary 77 ms decode step, i.e. roughly 100 ms), in both eager and captured
   (once graph capture lands, see problem 4) mode, and acceptance-independent (a fixed forced
   accept length isolates the kernel/dispatch cost from the workload's actual acceptance
   rate). Also useful as a sanity check against the rejected unrolled design's
   `4 x 77 ms ~= 308 ms` floor, to confirm the rewrite actually bought the dispatch-count win
   problem 2 argues for.
3. End-to-end `p95_ttft_turn2plus_ms` and `total_token_throughput` on the real benchmark, to
   confirm the decode-throughput win (this doc's actual target) does not regress TTFT (this
   bundle's scored objective): MTP only changes the decode loop, but a verify step is heavier
   per call than an ordinary decode step, and prefill scheduling (`MAX_PREFILL_BURST`) is
   tuned against the old per-step cost.

## Serving: `SEED_MTP_SERVE=1` (branch `bespoke/opt-mtp-serve`)

`SEED_MTP_SERVE=1` (alias `SEED_MTP=1`) serves decode through MTP rounds, `SEED_MTP_K` in
{1, 2, 3}. With `--enable-graph-capture`, `server.build_runner` uses `graph_mtp.GraphMTPRunner`:
per bucket, one captured draft graph (`mtp.draft_step` x k; `mtp._moe` now takes the fused MXFP4
kernels, so the draft has no host read or data-dependent shape) and one captured
verify+accept+rollback graph, replayed back to back with one host readback per round.

**Multi-token advance.** A round returns 1..k+1 tokens per lane. The scheduler passes each
lane's remaining `max_tokens` budget and stop ids; the round clamps its accept length to the
first stop id and to `budget - 1` (`mtp.commit_limit`, `graph_mtp._commit_limit` on device)
before the DeltaNet rollback and the MTP seed write. So after every step a lane's live state
has consumed exactly `prompt + output_ids[:-1]`, the invariant plain decode keeps; turn-close
publish (re-forward the last token, snapshot), the session cache and snapshot pools need no
MTP-specific code. Without the clamp, a stop landing inside an accepted draft leaves the state
ahead of the reply, and the turn-close snapshot resumes the next turn from the wrong state
(the multi-turn test fails exactly that way with the clamp disabled).

**Blocks under TP.** `Scheduler._decode_step` reserves every lane to `pos + k + 1` on rank 0 and
broadcasts `EXTEND_BLOCKS` before the round's `SPECULATIVE_DECODE` on the same ordered channel
(the ordering `_launch_overlap` uses). `MTPVerifyRunner.fill` no longer grows every lane with
the model's own allocator (rank-local ids under TP): it only capacity-checks the round's lanes
(`Model.grow_lane`), and padding rows use `RESERVED_BLOCK` like `GraphDecodeRunner.fill`.
`SPECULATIVE_DECODE`'s payload carries budgets and stop ids (`Command.speculative_cmd`).

**Eligibility.** A step is an MTP round when every lane is greedy, has at most 8 stop ids,
and has `pos + k + 1 <= max_seq`; otherwise it is a plain decode step. `SEED_OVERLAP_SCHED`
and `SEED_MIXED_BATCH` are turned off while MTP is on (logged at startup).

**Draft quality fixes.** The MTP layer now applies its `input_layernorm` to the fc output
before self-attention (it was loaded but unused), and the seed hidden state is the final-normed
output (DeepSeek-V3 MTP's definition; `SEED_MTP_SEED_HIDDEN=pre` restores the raw residual).
Neither changes committed tokens; both change acceptance.

**Counters.** With `SEED_STEP_TIMING=1` the scheduler's decode line adds `mtp_rounds=`,
`mtp_accept=` (accepted drafts / drafted) and `mtp_tok_per_lane_step=`; `tokens=`/`tok_s=`
count emitted tokens.

**Predicted cost (b48, graph).** From the eager measurement (t=1: 121 ms, t=4: 241 ms), device
work grows ~40 ms per extra token of verify width at b48, dominated by the per-assignment MoE
kernels (latency-bound, cost follows assignments, not distinct experts). On the 75 ms graph
decode step: k=1 round ~= 75 + 40 + ~3 (draft) + ~3 (rollback) ~= 120 ms; k=2 ~= 160 ms. At
per-draft acceptance 0.7: k=1 gives 1.7 tokens/round -> ~71 ms/token (~1.06x); k=2 gives
1 + 0.7 + 0.49 = 2.19 -> ~73 ms/token (~1.03x). The 1.5-2x lever needs the verify width to be
nearly free, i.e. MoE cost set by distinct experts (weight bytes), not assignments: with a
dedup/bandwidth MoE kernel (verify ~85 ms at k=1), k=1 -> ~51 ms/token (1.5x). The GPU run's
`mtp_accept` and the round time decide which regime holds.

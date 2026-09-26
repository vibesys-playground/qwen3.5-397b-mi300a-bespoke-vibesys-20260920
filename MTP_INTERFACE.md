# MTP round interface (round 15: W6 verify, W7 draft)

This file defines the contract between the MTP draft (W7, branch `bespoke/r15-mtp-draft`) and
the wide verify plus scheduler (W6, branch `bespoke/r15-mtp96`). One MTP round at bucket
capacity `B` (96 in production) with `k` drafts (`t = k + 1` fed tokens per lane) is two
captured graphs replayed back to back, with no host sync between them:

```
fill(buf, ...)   host: lanes, positions, budgets, stops, block tables
draft()          W7: writes buf.token_matrix[:, 1:t]
verify()         W6: wide forward, accept, rollback, MTP-cache and hidden-seed commit
read back        host: buf.step_argmax[:n], buf.accept_len[:n] (one sync)
```

## Shared buffer

Both graphs read and write one `graph_mtp.MTPVerifyBuffers(model, capacity=B, t, device)`.
Row `j` is the round's `j`-th request (caller order); rows `>= n` are padding. Fields keep
their current names, shapes, and dtypes:

| Field | Shape, dtype | Written by | Meaning |
|:--|:--|:--|:--|
| `token_matrix[:, 0]` | `[B]` long | fill | last committed token of the lane (not yet fed to the target) |
| `pos` | `[B]` long | fill | absolute position of `token_matrix[:, 0]`; 0 on padding rows |
| `active` | `[B]` bool | fill | real row |
| `slot_rows` | `[B]` long | fill | lane id (index into `model.pool[i]` state and `model.hidden_scratch`) |
| `block_table`, `block_valid` | `[B, max_blocks]` int32, `[B]` int32 | fill | target lane block table; `RESERVED_BLOCK` on padding rows. Valid through `pos + t` |
| `write_rows` | `[B, t]` long | fill | physical KV row of positions `pos .. pos + k` (target pool and MTP pool share row ids) |
| `draft_hidden` | `[B, hidden]` | fill | copy of `model.hidden_scratch[slot_rows]` taken at fill time |
| `token_matrix[:, 1:t]` | `[B, k]` long | **draft** | greedy draft ids; draft `s` predicts the token at `pos + 1 + s` |
| `step_argmax` | `[B, t]` long | verify | target argmax at each fed position |
| `accept_len` | `[B]` long | verify | accepted drafts after stop/budget clamp; committed = `step_argmax[j, :accept_len[j] + 1]` |
| `budget`, `stops` | `[B]`, `[B, MAX_STOP_IDS]` long | fill | commit limits |

## Draft contract (W7)

`build_draft_step(model, mtp, buf, t) -> Callable[[], None]`, capturable, same signature as
today's `graph_mtp.draft_round_step`, so it is a drop-in replacement.

- **Inputs:** the seed hidden is `model.hidden_scratch[slot_rows]` (equivalently
  `buf.draft_hidden`): the target's raw residual (pre-final-norm) at position `pos - 1`. Pass
  it through `mtp.draft_seed` (final norm) before `pre_fc_norm_hidden`, as now. Step 0 embeds
  `token_matrix[:, 0]` at position `pos`; step `s` runs at position `pos + s`, reading MTP KV
  rows `<= pos + s` through `block_table`.
- **Outputs:** `token_matrix[:, 1 + s]` for `s < k`, greedy argmax over the full vocab (every
  rank gets the same ids). The draft may write MTP KV rows `write_rows[:, s]` for `s < k`.
  Rows at or beyond the accepted length are never read again: verify rewrites the canonical
  rows for the accepted positions.
- **Must not** write `model.hidden_scratch`, target KV, or DeltaNet state, and must mask
  padding rows (`active`) for any persistent write.
- Must be identical on every rank (same collective sequence, no per-rank decisions).

## Verify contract (W6)

`build_verify_step(model, mtp, buf, t, ...) -> Callable[[], None]`, capturable, reads
`token_matrix`, writes `step_argmax` and `accept_len`, and leaves the lane state as if the
target had decoded exactly the committed prefix:

- Target KV for positions `pos .. pos + accept_len` (rows past that are stale and
  unreachable); DeltaNet conv and recurrent state advanced by `accept_len + 1` tokens.
- MTP KV canonical rows for positions `pos .. pos + accept_len`, rebuilt from target hidden
  states (`mtp.cache_target_rows`).
- `model.hidden_scratch[lane]` set to the target raw hidden at position `pos + accept_len`,
  which is the next round's draft seed.

After the round the host sets the lane's next `token_matrix[:, 0]` to
`step_argmax[j, accept_len[j]]` at position `pos + accept_len + 1`.

## Ownership (round 15)

To avoid merge conflicts, each workstream edits only its own hooks. Add new functions rather
than editing a shared one. `model.py` is shared: add, do not rewrite.

| Owner | Files and hooks |
|:--|:--|
| W6 (`bespoke/r15-mtp96`) | `graph_verify_wide.py`; `graph_mtp.py`: `MTPVerifyRunner` (verify build, `fill`, `_validate`, `speculative_decode`), `GraphMTPRunner.decode` and `.speculative_decode`; `scheduler.py`: `_speculative_ok`, `_speculative_decode_step`, and forced-suffix ids fed as always-accepted drafts; `mtp96_*` harnesses |
| W7 (`bespoke/r15-mtp-draft`) | `mtp.py` draft path (`draft_step`, `_moe`, `_attn_rows`, `draft`), `graph_mtp.draft_round_step` (or a new module it delegates to), per the draft contract above |
| W10 (`bespoke/r15-mtp-serve`) | `graph_prefill.py` (`prefill_step` MTP tail, `supported()`), `graph_mixed.py` if needed, `graph_decode.GraphDecodeRunner._prepare_prefill_graphs`; `scheduler.py`: `Scheduler.__init__` gating of `overlap`/`prefill_accum`/mixed under `spec_decode`, and the overlap and prefill-accum code paths |

With `SEED_MTP_SERVE`/`SEED_MTP` off, every path must behave exactly as on the integration
incumbent.

## Serve path (W10)

**Captured prefill under MTP.** `graph_prefill.prefill_step` ends with `_mtp_tail` when
`model.mtp` is set, the eager `Model.forward`/`forward_packed` MTP epilogue on a `[rows, width]`
call: the final raw residual (all-gathered first under the row-sharded residual,
`SEED_AR_SP`), `mtp.cache_target_rows` over the real columns (the target's own KV write rows;
column `c` pairs token `c` with the hidden at `pos + c - 1`, `hidden_scratch[lane]` for
`c = 0`), then `hidden_scratch[lane]` = residual at the row's last real column. Padded columns
and padding rows write nothing. Boot validation (`PrefillGraphRunner._check`) checks the tail
against the uncaptured step (fidelity) and the eager MTP prefill (correlation), zeroes the
calls' MTP KV rows between runs, and requires other lanes' `hidden_scratch` rows unchanged.
`SEED_PREFILL_ACCUM` runs under MTP. Mixed steps stay off under MTP.

**Overlap scheduling under MTP (`SEED_OVERLAP_SCHED`).** Round N+1 launches before
round N's `step_argmax`/`accept_len` reach the host. Per-lane device tables, written right
after each verify replay (stream-ordered, active rows only):

| Table | Value after round N for lane `slot_rows[j]` |
|:--|:--|
| `next_tok` | `step_argmax[j, accept_len[j]]` |
| `next_pos` | `pos[j] + accept_len[j] + 1` |
| `budget_left` | `budget[j] - accept_len[j] - 1` |
| `live` | no stop id in the committed prefix and `budget_left > 0` |

A lane still outstanding at launch is sent as `LOOKAHEAD`; fill takes `token_matrix[:, 0]`,
`pos`, `budget` from the tables and ANDs `active` with `live`, then derives `block_table`,
`write_rows` and the wide-verify row buffers on the device from those values. The host
reserves blocks to `launch_pos + 2t` (round N commits at most `t`). A lane whose round N ended
on a stop or its budget is therefore an inactive row in N+1: verify and draft mask every
persistent write by `active`, so its state is exactly the serial path's and the `REFORWARD`
turn-close publish stays correct; the host discards N+1's row for it (`fl.done`). Accept
counts feed the lookahead only through these device tables; the host learns them at N's
readback and advances `fl.pos` by `len(committed)`. Prefill steps, eager fallbacks and
non-speculative steps flush the in-flight round first, as today. Forced drafts
(`SEED_MTP_FORCED_DRAFTS`): a round with `f` forced drafts whose suffix outlasts them commits
exactly `f + 1` forced ticks, so the next round sends that lane explicitly (next suffix id at
`pos + f + 1`); a round that drains the suffix charges one generated token to `budget_left`.
Implemented in `mtp_overlap.py` (`OverlapMTPRunner`, `LaneTables`) and
`Scheduler._overlap_speculative_step`; TP op `SPECULATIVE_LAUNCH`.

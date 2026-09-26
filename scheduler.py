"""Continuous-batching scheduler, decoupled batch lanes, and the cross-session prefix cache.

`paged-kv-design.md` Stage 2 ("decouple cache entries from batch lanes; prefix trie"). Read
`session_cache.py`'s module docstring first: this file is the orchestrator, `session_cache.py`
is the torch-free trie/refcount bookkeeping it drives, and `block_pool.py` is the shared
allocator both ultimately sit on.

Iteration model. Unchanged from Stage 1: one `step()` is either one prefill iteration or one
decode step batched over every request past prefill, prefill winning ties up to
`MAX_PREFILL_BURST` consecutive iterations. See `_prefill_step`/`_decode_step` below; nothing in
this section changed.

Lanes vs. cached sessions -- the actual Stage 2 change. A **lane** is a row index into the
decode/graph-capture buffers with no state of its own: once a request releases it, the lane
costs nothing and needs no LRU (contrast Stage 1's `_Slot`, whose idle occupant was itself the
only copy of that session's reusable state, so evicting one was a real loss). A **cached
session** is a `session_cache.Node`: a snapshot boundary with its own refcounted KV blocks and
DeltaNet state, independent of whether any lane currently references it. Admission
(`_acquire`) now does two independent things: pick any free lane (no LRU -- see
`session_cache.py`'s eviction section for why), and separately resolve the longest matching
node in the trie, evicting *cache entries* (not lanes) under pressure via
`SessionCache.reserve_blocks`/`reserve_snapshot`.

Publishing. Where Stage 1 recorded a whole-prompt token tuple directly on the slot, Stage 2
inserts a new trie node at each snapshot boundary (`_publish_boundary`): the chat-suffix
turn-open boundary (`Request.suffix_len`, unchanged semantics from Stage 1 -- see below) during
prefill, and the turn-close boundary (`_publish_turn_close`) when a request finishes, using the
model's live end-of-generation state directly when there is no positional gap
(`suffix_len == 0`), or a short position-shifted re-forward of the reply when there is one (the
same trick Stage 1's `_try_apply_reply` used, ported to Stage 2's node model -- see
`_publish_turn_close`'s docstring for why the gap needs it).

A chat template can make the recorded prefix stop *before* the prompt's own end
(`Request.suffix_len`): Qwen3.5's thinking-disabled template appends `<think>\n\n</think>\n\n`
after `<|im_start|>assistant\n` only to the prompt it generates from, and renders that same turn
in the *next* prompt's history without it. `suffix_len` tells `_finish_prefill_chunk` to publish
a turn-open node at `len(prompt) - suffix_len`, before the suffix, then keep prefilling the
suffix on top of that snapshot for this request's own decode without moving the node. A later
request that only shares the boundary, not the suffix, still matches; one that also repeats the
suffix verbatim is the ordinary exact-match case (`node.depth == len(prompt)`), handled by the
same trie lookup, not separate machinery (paged-kv-design.md section 2.4).

An exact repeat of a cached node's full token span answers from that node's cached logits
(`cache_node_logits`/`cached_node_logits`, indexed by the node's own snapshot id rather than a
lane, since the node can long outlive whatever lane originally computed it) with no forward call
at all -- see `_prefill_step`'s "exact_dup" handling, unchanged in spirit from Stage 1's
`cached_logits`, just re-keyed.

Dropped from Stage 1: `Request.extend_prefix`/the lazy "unapplied reply" mechanism. Stage 1
needed it because a slot had exactly one snapshot register, so recording the turn-close boundary
too meant either paying for a second dense per-slot buffer or reapplying the reply lazily on
demand. Stage 2's snapshot pool is sized independently and can hold many boundaries at once, so
turn-close is published unconditionally and eagerly, the same way turn-open already was --
strictly more capable (also serves the raw-completions exact-duplicate case Stage 1's opt-in flag
existed for) at the cost of always doing the suffix_len>0 re-forward eagerly rather than only
when a follow-up turns up. `SEED_DEFER_TURN_CLOSE` (`DEFER_TURN_CLOSE`) takes that re-forward off
the decode path again: it measured +47 ms per decode step at C48, for a node chat traffic never
matches.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import step_timing
from session_cache import Node, SessionCache, SessionCacheError

PREFILL_CHUNK = 512
# Consecutive prefill chunks allowed before a decode step is forced.
MAX_PREFILL_BURST = 4

PREFILL_TOKEN_BUDGET = 2048
"""Default token budget for one packed prefill call (see `_select_prefill_batch`)."""

BATCHED_PREFILL = os.environ.get("SEED_BATCHED_PREFILL", "1") not in ("0", "false", "False")
"""Whether `_prefill_step` packs several requests' chunks into one `runner.prefill_batch`
call, or falls back to the original one-request-at-a-time `runner.prefill` loop. The A/B
knob for measuring the packed path against the path it replaces; `model.py` reads the same
variable for its own side of the same choice (see `Model.prefill_batch`'s docstring), so one
flag turns the feature off end to end."""

PREFILL_FIT_GRAPH = os.environ.get("SEED_PREFILL_FIT_GRAPH", "0") not in ("0", "", "false", "False")
"""`SEED_PREFILL_FIT_GRAPH=1`: shape each packed prefill call so it replays a captured prefill
graph (`SEED_PREFILL_GRAPHS`, graph_prefill.py) whenever the runner has any. Each chunk is
capped at the widest captured width, and a request's chunk joins the call only while the call
still fits some captured `rows x width` shape; the rest wait for the next prefill step.

Why: an eager prefill call pays ~110 ms of host dispatch on top of its device time (r4 at C16:
263-470-token eager steps took 180-240 ms), and every prefill step stalls all decoding lanes.
A call that misses the ladder by one row (two lanes of ~300 tokens against no `2x256`) went
eager. This is chunked prefill (Sarathi-Serve's fixed-size chunks) with the chunk size and
lane count chosen to match the captured shapes instead of a flat token budget.

Measured (4x MI300A, r4 flags, with `SEED_PREFILL_GRAPH_SHAPES=` the default ladder plus
`2x256,1x512`): idle turn-2 TTFT 147 ms at 142 new tokens, 195 ms at 507, 658 ms at 1966
(four 512-token replays); load-ramp TTFT turn2+ p95 at C16 460 ms vs 570 ms for r4, peak
440 vs 449 tok/s (within run-to-run noise). No effect when prefill graphs are off; with the
default ladder alone (max width 256) it caps chunks at 256."""

PREFILL_ACCUM = os.environ.get("SEED_PREFILL_ACCUM", "0") not in ("0", "", "false", "False")
"""`SEED_PREFILL_ACCUM=1`: while lanes decode, hold queued prefill work until one captured
prefill replay can carry `SEED_PREFILL_ACCUM_TOKENS` real tokens, or the oldest queued flight
has waited `SEED_PREFILL_ACCUM_MAX_WAIT_MS`; decode keeps running meanwhile. A fired step packs
toward the captured shape with the best modeled real tokens per ms
(`SEED_PREFILL_ACCUM_COST`, affine in area), capping each row's chunk at that shape's width.

Why: each captured prefill step streams nearly the whole local expert set, so its cost grows
far slower than its token count, while the incumbent fires a step as soon as any prefill is
queued (one ~300-token row per step at C96). Round 10's burst variant counted rows capped at
256 tokens and filled only 437 tokens per burst; this counts packable real tokens across the
wider captured shapes. Needs captured batched prefill (`SEED_PREFILL_FIT_GRAPH`); off under
the mixed-step paths. Works under MTP (`SEED_MTP_SERVE`): captured prefill seeds the MTP cache
and draft hidden itself (graph_prefill's MTP tail), and a held step only lets MTP rounds run."""

PREFILL_ACCUM_TOKENS = int(os.environ.get("SEED_PREFILL_ACCUM_TOKENS", "1024"))
PREFILL_ACCUM_MAX_WAIT_MS = float(os.environ.get("SEED_PREFILL_ACCUM_MAX_WAIT_MS", "400"))
PREFILL_ACCUM_MIN_DECODE = int(os.environ.get("SEED_PREFILL_ACCUM_MIN_DECODE", "48"))
"""Accumulate only while at least this many lanes decode. Below it (low concurrency, the
C16 TTFT reference level) prefill fires at once, as without the flag: few turns arrive per
decode step there, so holding would add most of the wait bound to TTFT for little fill."""
PREFILL_ACCUM_COST = tuple(
    float(x) for x in os.environ.get("SEED_PREFILL_ACCUM_COST", "33,0.104").split(",")
)
"""`(fixed_ms, ms_per_area_token)`: the replay cost model the shape choice ranks by. Default
fit to captured replays on 4x MI300A (round 15, `bench_big_prefill_shapes.py`, incumbent
flags): 1x512 86.8 ms, 4x256 136.3, 4x384 194.9, 4x512 246.9 (full rows)."""

PREFILL_EARLY_LAUNCH = os.environ.get("SEED_PREFILL_EARLY_LAUNCH", "0") not in (
    "0",
    "",
    "false",
    "False",
)
"""`SEED_PREFILL_EARLY_LAUNCH=1` (off by default; `OVERLAP_SCHED` only): a prefill step enqueues
its replay right behind the in-flight decode step and completes that decode step (reads its ids,
emits, releases, publishes) afterwards, instead of completing it first. Why: completing first
leaves the device idle while the host emits ~96 tokens and builds the prefill call; measured at
C96 (3be127a3, node03) as 15.9 ms of device idle per prefill step, 3.5% of wall time.
Correctness: the prefill selects, grows and forwards only `prefill_q` lanes, which are disjoint
from the in-flight step's decoding lanes; the in-flight step's reads, releases and publishes are
host bookkeeping plus stream-ordered device copies, so enqueueing them after the prefill changes
no kernel's inputs. Blocks the in-flight step would free become available one step later."""

PREFILL_FILL_LOG = os.environ.get("SEED_PREFILL_FILL_LOG", "0") not in ("0", "", "false", "False")
"""`SEED_PREFILL_FILL_LOG=1`: one `[prefill-fill]` line per packed prefill step (rows, real
tokens, replayed shape, host-synced wall, and under `SEED_PREFILL_ACCUM` the wait and whether
the age bound forced it). Works with the flag on or off, so both arms of an A/B log fill."""

SPECULATIVE_DECODE = any(
    os.environ.get(name, "0") not in ("0", "", "false", "False")
    for name in ("SEED_MTP_SERVE", "SEED_MTP")
)
"""`SEED_MTP_SERVE=1` (or its older alias `SEED_MTP=1`): `_decode_step` drives an all-greedy
batch through `runner.speculative_decode` (MTP draft/verify/accept, 1..k+1 tokens per lane
per step) instead of `runner.decode` + `sample_batch`. `mtp.py` reads the same variables for
the model side of the same choice; this module cannot import `mtp.py` (it stays torch-free,
see the module docstring), so the flag is read here too, the same way `BATCHED_PREFILL`
already is in both this module and `model.py`.

Multi-token advance. A round commits `c` tokens per lane; `_speculative_decode_step` emits
them through the ordinary `_emit`, so streaming, stop tokens and `max_tokens` are the
single-token rules applied `c` times. The round is told each lane's remaining budget and stop
ids, and clamps its own accept length to them (`mtp.commit_limit`), so a stop token or the
budget landing mid-accept ends both the committed list and the lane's DeltaNet state there.
That keeps the invariant every other path relies on: after a step, the lane's live state has
consumed exactly `prompt + output_ids[:-1]` (the last emitted token is the next step's input).
Turn-close publish (`_publish_turn_close`) therefore works unchanged (it re-forwards the last
token on top of the live state), and so do the prefix/session cache and snapshot pools, which
only ever see states at those boundaries. KV rows written for rejected drafts sit past the
lane's position and are never read (masked by length), like any lane's unused tail.

Blocks. `_decode_step` reserves every lane to `pos + k + 1` on rank 0 (`_grow_lanes`, one
`EXTEND_BLOCKS` broadcast under TP) before the round's own command, the same ordering
`_launch_overlap` uses; the runner never allocates.

Eligibility, per step: every lane greedy, at most `MTP_MAX_STOP_IDS` stop ids per lane, and
`pos + k + 1 <= max_seq` for every lane (a round writes `k + 1` KV rows). Otherwise the step
is an ordinary decode step. `SEED_OVERLAP_SCHED` and `SEED_MIXED_BATCH` are turned off while
MTP is on (`Scheduler.__init__`)."""

MTP_FORCED_DRAFTS = os.environ.get("SEED_MTP_FORCED_DRAFTS", "0") not in ("0", "", "false", "False")
"""`SEED_MTP_FORCED_DRAFTS=1` (with `SEED_MTP_SERVE` and `SEED_MTP_VERIFY_WIDE`): a lane still
draining its folded chat suffix (`fl.forced`) joins the MTP round instead of forcing the whole
step to plain decode. Its known suffix ids replace its first drafts and are accepted
unconditionally (`graph_verify_wide`); `_emit` pops them exactly as it does for plain decode
ticks. Takes effect only when the runner reports `forced_drafts()`."""

MTP_ACCEPT_LOG = os.environ.get("SEED_MTP_ACCEPT_LOG", "0") not in ("0", "", "false", "False")
"""`SEED_MTP_ACCEPT_LOG=1` (diagnostic): every 50 MTP rounds, one `[mtp-accept]` line with the
mean accepted drafts per lane-round, split by the lane's round index within its turn
(0 = first round after the turn's prefill) and by first vs follow-up turn (`reused > 0`).
Rows carrying forced drafts are excluded (their accept length is forced)."""

MTP_ACCEPT_BUCKETS = ((0, 0), (1, 1), (2, 2), (3, 7), (8, 31), (32, 1 << 30))

MTP_MAX_STOP_IDS = 8
"""Mirror of `mtp.MAX_STOP_IDS` (this module stays torch-free)."""

REFORWARD = object()
"""`_emit`'s `logits` for a token that has no logits row of its own (an MTP round returns
token ids only). Non-None so a finishing request still publishes its turn-close boundary;
`_publish_turn_close` re-forwards the last token in that case and never reads this value."""

OVERLAP_SCHED = os.environ.get("SEED_OVERLAP_SCHED", "0") not in ("0", "", "false", "False")
"""Overlap the scheduler's host work with the device's decode step (one-step lookahead).

Off, a decode step is strictly serial: launch, read the sampled ids back (`sample_batch`'s
`.tolist()`), do the bookkeeping, build and launch the next step; the device idles through
everything after the readback. On, `_overlap_decode_step` launches step N+1 *before* it reads
step N's ids: sampling stays on the device and the next step's input ids are taken from the
device-side per-lane table the previous step wrote (`Runner.decode_launch` with `LOOKAHEAD`
tokens), while step N's ids come back through a pinned, event-tracked async copy. The host
then does step N's bookkeeping (stop/length checks, streaming, publishing, releasing) while
the device runs step N+1. Idea from SGLang's overlap scheduler and vLLM's async scheduling
(re-derived here; no code from either).

What the one-step lag costs, and how each case keeps the output identical to the flag off:

- **Stop token (EOS/`stop`).** Unknowable until the id is read back, so a request whose step-N
  token is a stop token has already been launched into step N+1. That extra token is
  discarded (`fl.done` when step N+1 is completed). Its forward is not wasted for turn-close:
  step N+1's input *was* the final token, which is exactly the forward `_publish_turn_close`
  would otherwise re-run as a one-token prefill, so it publishes step N+1's state and logits
  row directly (`forwarded=True`) instead.
- **`max_tokens`.** Known on the host, so no extra step: a request whose in-flight token will
  be its last is not launched again (`_launch_overlap`), and it finishes on the ordinary path.
- **Prefill.** A prefill iteration samples on the host, so the in-flight step is completed
  first (`_flush`); prefill stays serial. Admission itself (`_admit`) only enqueues device work
  and is stream-ordered behind the in-flight step.
- **Lane release.** A lane released while step N+1 still runs on it has its `begin`, block
  release and any re-admission prefill enqueued after step N+1 on the same stream, so nothing
  touches its state early.
- **MTP.** With a runner that has `speculative_launch` (`mtp_overlap.OverlapMTPRunner`), round
  N+1 launches before round N's committed tokens are read (`_overlap_speculative_step`). A lane
  still in flight is sent as `LOOKAHEAD`: its next token, position and budget come from
  per-lane device tables round N wrote, and a lane whose round N ended on a stop or its budget
  runs as an inactive row, so its state matches the serial path (see mtp_overlap.py). The host
  reserves `pos + 2t` for an in-flight lane and advances `fl.pos` at readback. Steps that are
  not captured MTP rounds flush and run serially. Without such a runner overlap is off.
- **TP.** `tp_driver.Op.DECODE_LAUNCH`; see `tp_driver`'s module docstring.

Detokenization and streaming already run off this thread (`server.py`'s per-request detok
worker; `_emit` only queues), so the host work left on the critical path here is the
bookkeeping itself plus the next step's input build.
"""

FOLD_TURN_SUFFIX = os.environ.get("SEED_FOLD_TURN_SUFFIX", "0") not in ("0", "", "false", "False")
"""`SEED_FOLD_TURN_SUFFIX=1`: fold the chat-suffix tail (`Request.suffix_len`, the turn-open
boundary's own ~5-token `<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`-shaped priming
text) into the ordinary decode path instead of giving it a dedicated prefill/mixed step.

Off (today's behavior): `_chunk_limit` stops the chunk that reaches the boundary exactly
there, publishes the boundary node, and leaves the suffix tokens in `fl.pending` for a *later*
prefill/mixed step of their own. That step is real work (a full forward over every layer:
dense, MoE, attention, DeltaNet, all-reduce) for as few as 5 real rows, padded up to whatever
captured mixed-graph shape is available (64 or 128 rows): 11-19% fill, ~30 ms over a plain
decode step each turn (`PREFILL_COST.md`'s padding section).

On: once the boundary-hitting chunk finishes (`_finish_prefill_chunk`), the leftover suffix
tokens never become a `pending` prefill chunk at all. Instead the flight moves straight into
`self.decoding` with those tokens queued as forced feed (`_InFlight.forced`): each subsequent
decode/mixed step's real sampled token for this lane is discarded and overridden with the next
queued id (`_emit`'s guard), exactly the teacher-forcing an eager suffix chunk would have done
one row at a time, riding steps that are already scheduled instead of a dedicated one. Forced
ticks are never streamed, never counted against `max_tokens`, and never checked against stop
ids (`_emit` returns before any of that bookkeeping) -- they are not generated output, just the
model catching up to where real generation starts. Position/KV/DeltaNet state advance exactly
as an ordinary decode step's forward already does, so the state once `forced` drains is bit for
bit what the removed suffix chunk used to produce.

The boundary node itself (`_publish_boundary` at `boundary = len(prompt) - suffix_len`) is
unchanged: this flag only changes what happens to the suffix *after* that node is published,
never where it is published, so cache hit lengths for future turns are unaffected.

Not folded into an MTP round: `_speculative_ok` excludes any flight with a nonempty `forced`
queue, so a lane still draining its suffix forces that step to plain per-token decode instead
(brief, `suffix_len` steps at most, and MTP is auto-disabled together with `SEED_MIXED_BATCH`
whenever `SEED_MTP_SERVE` is on regardless -- see `SPECULATIVE_DECODE`)."""

DEFER_TURN_CLOSE = os.environ.get("SEED_DEFER_TURN_CLOSE", "0") not in ("0", "", "false", "False")
"""Take `_publish_turn_close`'s eager re-forward off the decode critical path.

Off, every finishing request runs one eager `runner.prefill` on its lane before the lane is
released: the last generated token (`suffix_len == 0`), or the whole reply at shifted positions
(`suffix_len > 0`). That call sits between two decode steps, so every decoding lane waits for
it: ~120 ms fixed eager cost plus ~1 ms per re-forwarded token (measured at C48: +47 ms per
decode step on average, 10-15% of steps at 270-370 ms).

On:

- **`suffix_len > 0` (every chat request): publish nothing.** The node the re-forward builds
  has edge `reply` directly after the turn-open boundary, but a chat template renders a prior
  assistant turn in the next prompt as `<|im_start|>assistant\\n` + content + close, so the next
  turn's prompt diverges from that edge at its first token and the node is never matched
  (`test_chat_suffix_reuse.py` asserts reuse stops exactly at the boundary). The next turn
  resumes from the turn-open node and prefills the rendered reply as part of its own prompt,
  which it already does today. Nothing reusable is lost; the re-forward was dead work.
- **`suffix_len == 0` (raw completions): publish at `prompt + reply[:-1]`, no forward.** After
  a non-lookahead step the lane's live state has consumed exactly `prompt + output_ids[:-1]`
  (the last token was sampled but never forwarded), and the `logits` that sampled the last
  token are that state's next-token logits. So the lane's current state *is* a valid node one
  token short of the full reply. A follow-up whose prompt extends the reply matches it and
  prefills the one missing token together with its own new tokens (lazy turn close: the
  closing token is prepended to the next turn's prefill, which runs anyway). An exact repeat of
  `prompt + reply` is one token past the node and prefills that one token instead of hitting
  the cached logits; that is the only reuse given up.
- **Lookahead stop (`forwarded`) and MTP (`REFORWARD`) are unchanged.** The lookahead step
  already forwarded the last token, so publishing the full reply costs nothing. An MTP round
  returns no logits row to cache, so it keeps the eager one-token re-forward.
"""

LOOKAHEAD = -1
"""`Runner.decode_launch` token id meaning "the id this lane sampled in the previous step,
still on the device". Mirrors `model.LOOKAHEAD_TOKEN` (this module stays torch-free)."""

MIXED_BATCH = os.environ.get("SEED_MIXED_BATCH", "0") not in ("0", "false", "False")
"""Whether `step()` replaces plain alternation (a whole `_prefill_step`, `MAX_PREFILL_BURST`
times at most, then a whole `_decode_step`) with `_mixed_step`: one iteration that always
carries every running decode lane's next token *and* one packed prefill chunk, in the same
forward call (`Runner.decode_mixed`, `model.py`'s `Model.layer_mixed`).

Why. Idea: Sarathi-Serve's stall-free batching (Agrawal et al., "Taming Throughput-Latency
Tradeoff in LLM Inference with Sarathi-Serve", OSDI'24) and vLLM's chunked-prefill scheduling
-- an admitted prefill request is chunked and interleaved with running decodes token budget by
token budget, instead of ever running a whole prefill iteration on its own. Plain alternation
stalls *every* decoding lane for a whole prefill chunk's own cost each time `_prefill_step`
wins a tie: measured at C=48, a packed prefill call costs ~2.2 s per 2,048 tokens (~1 ms/token)
against a decode step of 80-120 ms, so turn 2+ admission (a fresh burst of prefill work
arriving while other lanes are already decoding) can push several seconds of stall onto every
other lane in the batch before its own decode step runs again -- p95 TPOT (time-per-output-
token) breaches typical SLOs (e.g. 250 ms) by an order of magnitude, and measured throughput
collapses with it (a stalled decode lane is a lane not producing tokens). Mixed batching
removes the stall by construction: once decoding has started, every scheduler iteration
contains every decoding lane's next token, so no lane ever waits more than one iteration for
its own next token regardless of how much prefill work is queued behind it; the prefill side
is capped by an adaptive token budget (`_mixed_budget`) sized so the extra cost stays a
bounded fraction of a decode-only step, not the whole chunk unconditionally
(`SEED_MIXED_TOKEN_BUDGET` is only the outer ceiling on that adaptive budget, not its normal
operating point -- see `_mixed_budget`'s docstring).

Auto-disabled under MTP (`SEED_MTP`/`spec_decode`, see `Scheduler.__init__`): speculative
decode's draft/verify/accept round has its own wide-attention and rollback machinery
(`model.py`'s `attn_verify`/`deltanet_verify`/`deltanet_verify_rollback`) that a mixed forward
does not share in v1 -- `Model.decode_mixed` also refuses to run under MTP as a model-side
backstop. Mixed batching also runs eager, not graph-captured, when it actually mixes in a
prefill chunk (variable chunk length each step is not a fixed captured shape); see
`model.py`'s `Model.layer_mixed` docstring for that cost and the shape-bucket option it leaves
open. A pure decode-only step (no `prefill_q` this iteration) is unaffected either way --
`step()` only takes the mixed path when both `self.decoding` and `self.prefill_q` are
nonempty, so it never adds eager-step cost when there is nothing to mix in."""

MIXED_GRAPH = os.environ.get("SEED_MIXED_GRAPH", "0") not in ("0", "", "false", "False")
"""`SEED_MIXED_GRAPH=1`: the graph-captured form of `SEED_MIXED_BATCH` (`graph_mixed.py`).
While any lane decodes and prefill work is queued, every step carries all decoding lanes plus
prefill chunks shaped to a captured mixed shape (`Runner.mixed_fit`), so decode never waits for
a separate prefill forward (Sarathi-Serve's stall-free batching). The prefill side's budget is
the largest captured prefill area for this decode count, capped by `SEED_MIXED_TOKEN_BUDGET`:
the shape set is the budget, chosen from measured replay cost (`SEED_MIXED_GRAPH_BENCH`). When
no captured shape holds the step (too many decode rows, mixed graphs off), the scheduler runs
separate prefill and decode steps as before, never the eager mixed forward. Independent of
`SEED_MIXED_BATCH`; off under MTP like it."""

_MOE_HIP_WIDE = os.environ.get("SEED_MOE_HIP_WIDE", "0") not in ("0", "", "false", "False")

MIXED_TOKEN_BUDGET = int(
    os.environ.get("SEED_MIXED_TOKEN_BUDGET", "1024" if _MOE_HIP_WIDE else "512")
)
"""Outer ceiling on one mixed step's prefill-side token budget; see `_mixed_budget`. 1024
under `SEED_MOE_HIP_WIDE`, whose mixed shapes go up to 1024 rows (`graph_mixed.WIDE_TOTALS`)."""

MIXED_BUDGET_SLACK = 0.5
"""`_mixed_budget` sizes the prefill side so it adds at most this fraction of a decode-only
step's own cost -- 0.5 means a mixed step costs at most 1.5x a decode-only step, the design
target (see `SEED_MIXED_BATCH`'s docstring)."""

MIXED_DECODE_STEP_MS = float(os.environ.get("SEED_MIXED_DECODE_STEP_MS", "100"))
"""Assumed decode-only step cost the adaptive budget is sized against, milliseconds. Default
is the midpoint of the measured 80-120 ms/step range (b48, graph replay); override once
`SEED_STEP_TIMING` has measured this build's own number."""

MIXED_PREFILL_MS_PER_TOKEN = float(os.environ.get("SEED_MIXED_PREFILL_MS_PER_TOKEN", "1.0"))
"""Assumed packed-prefill cost per token, milliseconds. Default is the measured pre-
vectorization figure (~1 ms/token, packed 2,048-token call ~2.2 s); drop to ~0.3 once the
vectorized kernel lands (see the campaign brief) to let the adaptive budget admit more tokens
per mixed step at the same 1.5x cost ceiling."""

Event = tuple[str, Any]

POOL_EXHAUSTED = "KV block pool exhausted: no evictable cache entry left"


class Runner(Protocol):
    """The model surface the scheduler drives. `Model` in model.py implements it."""

    max_batch: int  # lane count (paged-kv-design.md section 1.5); unrelated to cache size now

    def begin(self, lane: int) -> None:
        """Zero `lane`'s live DeltaNet/conv state and empty its block table. Does not touch
        the block allocator's refcounts -- the caller already released whatever the lane held
        (see `SessionCache.release_lane_blocks`) before calling this."""

    def lane_blocks(self, lane: int) -> tuple[int, ...]:
        """`lane`'s current full flat block-id table, for the caller to publish or release."""

    def attach_blocks(self, lane: int, blocks: Sequence[int]) -> None:
        """Seed `lane`'s block table with an already-resolved id list (adopted from a cache
        node, copy-on-write already applied by the caller if `SessionCache.needs_cow` said so)."""

    def copy_block(self, dst_block: int, src_block: int, filled: int) -> None:
        """Copy-on-write's physical half: copy `filled` tokens' K/V rows from `src_block` into
        `dst_block`, every full-attention layer. The caller decides whether/when this runs
        (`SessionCache.needs_cow`) and has already allocated `dst_block`; this call only moves
        bytes."""

    def lane_block_count(self, lane: int) -> int:
        """How many blocks `lane`'s table holds."""

    def extend_blocks(self, grants: Sequence[tuple[int, int]]) -> None:
        """Append already-reserved ids to lanes' tables, `(lane, block)` pairs in order.

        The scheduler is the only block allocator while serving (see `_grow_lanes`): every
        forward's KV rows are reserved here first, so the runner never allocates on its own
        and, under TP, every rank holds the same ids."""

    def decode_tokens_per_step(self) -> int:
        """KV rows one `speculative_decode` round may write per lane (`mtp.k + 1`)."""

    def score(self, lane: int, ids: Sequence[int], continuation_start: int) -> Any:
        """Teacher-forced logits for `ids[continuation_start:]` (`/v1/score`)."""

    def load_snapshot(self, lane: int, snap: int) -> None:
        """Copy the DeltaNet/conv state at snapshot slot `snap` into `lane`'s live state."""

    def save_snapshot(self, lane: int, snap: int) -> None:
        """Copy `lane`'s live DeltaNet/conv state into snapshot slot `snap`."""

    def prefill(self, lane: int, ids: Sequence[int], start: int) -> Any:
        """Forward `ids` at positions `start..start+len(ids)`; return [1, vocab] logits."""

    def prefill_batch(self, calls: Sequence[tuple[int, Sequence[int], int]]) -> list[Any]:
        """One packed forward covering several `prefill` calls; one [1, vocab] result per
        call, in the same order. `Model.prefill_batch` implements this by concatenating the
        chunks into a single varlen forward; only used when `BATCHED_PREFILL` is on."""

    def decode(self, lanes: Sequence[int], tokens: Sequence[int], positions: Sequence[int]) -> Any:
        """One batched decode step; returns logits indexable by batch position."""

    def decode_mixed(
        self,
        lanes: Sequence[int],
        tokens: Sequence[int],
        positions: Sequence[int],
        prefill_calls: Sequence[tuple[int, Sequence[int], int]],
    ) -> tuple[Any, list[Any]]:
        """`SEED_MIXED_BATCH`'s one-forward step (Sarathi-Serve / vLLM chunked-prefill idea --
        see this module's `SEED_MIXED_BATCH` docstring, and `model.py`'s `Model.layer_mixed`
        for the tensor-level argument). `lanes`/`tokens`/`positions` are `decode`'s own
        per-lane contract; `prefill_calls` is `prefill_batch`'s own `(slot, ids, start)`
        contract, sized by `_mixed_budget`. Returns `(decode_logits, prefill_logits)`:
        `decode_logits` indexable exactly like `decode`'s own result (via `decode_row`),
        `prefill_logits` one `[1, vocab]` row per `prefill_calls` entry, in the same order,
        like `prefill_batch`'s own result. `prefill_calls` may be empty (nothing to mix in this
        step); implementations fall back to a plain `decode` call in that case, same contract
        either way."""

    def decode_row(self, logits: Any, row: int) -> Any:
        """The [1, vocab]-shaped `row` of a batched `decode` result, addressable on its own --
        what a turn-close publish caches (`cache_node_logits`)."""

    max_seq: int  # context length; an MTP round needs `pos + k + 1 <= max_seq` per lane

    def speculative_decode(
        self,
        lanes: Sequence[int],
        tokens: Sequence[int],
        positions: Sequence[int],
        budgets: Sequence[int],
        stops: Sequence[Sequence[int]],
    ) -> list[list[int]]:
        """One MTP decode round for every lane: draft, verify, greedy-accept, and commit.
        `tokens`/`positions` are `decode`'s own per-lane last-token/position contract;
        `budgets`/`stops` are each lane's remaining token budget and stop ids. Returns each
        lane's newly committed token ids, in order, length 1..k+1 (the accepted draft prefix
        plus one bonus token), cut at the first stop id and at the budget, with the lane's
        state cut at the same point -- never called unless `SPECULATIVE_DECODE` is on and the
        step is eligible (see `_speculative_ok`). There is no per-lane logits row: turn-close
        publish re-forwards the last token instead (`REFORWARD`)."""

    def sample_batch(self, logits: Any, temperatures: Sequence[float]) -> list[int]:
        """One token id per row of `logits`, costing one device-to-host sync for the batch."""

    def decode_launch(
        self,
        lanes: Sequence[int],
        tokens: Sequence[int],
        positions: Sequence[int],
        temperatures: Sequence[float],
    ) -> PendingDecode:
        """`OVERLAP_SCHED` only: `decode` plus sampling, without waiting for either. A
        `tokens` entry may be `LOOKAHEAD` (that lane's previous sampled id, still on the
        device). Returns a handle to read the ids from later."""

    def cache_node_logits(self, snap: int, logits: Any) -> None:
        """Remember `logits` as snapshot slot `snap`'s cached next-token logits, so a future
        exact-duplicate match of that node can answer with no forward call at all."""

    def cached_node_logits(self, snap: int) -> Any:
        """The logits last passed to `cache_node_logits` for snapshot slot `snap`."""


class PendingDecode(Protocol):
    """A launched `decode_launch` step whose ids have not been read yet."""

    def tokens(self) -> list[int]:
        """This step's sampled ids, in batch order; waits for this step only."""

    def row(self, i: int) -> Any:
        """Batch position `i`'s [1, vocab] logits, for a turn-close publish."""


@dataclass(slots=True)
class Request:
    """One chat completion.

    `emit` receives ('tok', id) per streamed token, then either
    ('end', (reason, generated, reused)) or ('error', message). `reused` is how many prompt
    tokens came from a cached prefix instead of being prefilled.
    """

    prompt: list[int]
    max_new: int
    temperature: float
    stop: frozenset[int]
    emit: Callable[[Event], None]
    suffix_len: int = 0  # trailing tokens to prefill but not record as part of the reusable
    # prefix (see the module docstring's chat-template-suffix paragraph); 0 records at the
    # prompt's own end, as ever
    has_history: bool = False  # a follow-up turn (chat with a prior assistant message); only
    # splits the `[cache-stats]` admission counters


@dataclass(slots=True)
class _ResetPrefixCache:
    """A control message on `Scheduler.incoming`, not a `Request`: see `reset_prefix_cache`'s
    docstring for why `SessionCache.evict_all()` must run on the scheduler's own thread rather
    than being called into directly from whatever thread the admin endpoint handles."""

    done: threading.Event = field(default_factory=threading.Event)
    result: int = 0


@dataclass(slots=True)
class ScoreJob:
    """A `/v1/score` call routed through the scheduler thread (see `Scheduler.score`).

    Runs on a free lane between iterations, so it never races the scheduler's own runner
    calls (under TP, two threads interleaving `Channel.send` deadlock the ranks) and never
    clobbers a live lane. `finish` turns the runner's logits into the caller's result on the
    scheduler thread, before the lane is reused."""

    ids: list[int]
    continuation_start: int
    finish: Callable[[Any], Any]
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: str | None = None


@dataclass(slots=True)
class _InFlight:
    req: Request
    lane: int
    pos: int  # tokens already forwarded into the lane
    pending: list[int]  # prompt tail not yet prefilled
    reused: int  # prompt tokens the lane already held at admission
    parent: Node  # deepest published node this flight's state currently extends
    next_token: int | None = None  # emitted but not yet forwarded
    forced: deque[int] = field(default_factory=deque)  # SEED_FOLD_TURN_SUFFIX: remaining
    # chat-suffix ids to feed as this lane's next input, one per decode/mixed step, before it
    # resumes ordinary sampled generation -- see FOLD_TURN_SUFFIX's docstring.
    generated: int = 0
    output_ids: list[int] = field(default_factory=list)  # every token emitted, in order
    done: bool = False  # set by `_release`; lets a multi-token speculative commit stop early
    # OVERLAP_SCHED only: the last launched step that included this flight (0: none), and its
    # batch row there. `last_step > Scheduler._done_step` means its newest token is still on
    # the device (the next launch feeds it as `LOOKAHEAD`).
    last_step: int = 0
    row: int = 0
    inflight_forced: int = 0  # OVERLAP_SCHED under MTP: forced drafts in the in-flight round
    mtp_rounds: int = 0  # SEED_MTP_ACCEPT_LOG: MTP rounds this turn has run
    queued_at: float = 0.0  # SEED_PREFILL_ACCUM: clock time this flight entered `prefill_q`
    own_parent: bool = False  # `parent` was published by this flight, not resumed at admission


@dataclass(slots=True)
class _Launched:
    """OVERLAP_SCHED: one launched decode step whose ids the host has not read yet."""

    step: int
    flights: list[_InFlight]
    handle: Any  # PendingDecode, or a speculative round's handle (`committed()`)
    speculative: bool = False


class Scheduler:
    """Admits requests into a shared batch and runs them to completion."""

    def __init__(
        self,
        runner: Runner,
        cache: SessionCache,
        prefill_chunk: int = PREFILL_CHUNK,
        token_budget: int = PREFILL_TOKEN_BUDGET,
        batched_prefill: bool = BATCHED_PREFILL,
        spec_decode: bool = SPECULATIVE_DECODE,
        overlap: bool = OVERLAP_SCHED,
        mixed_batch: bool = MIXED_BATCH,
        mixed_token_budget: int = MIXED_TOKEN_BUDGET,
        mixed_decode_step_ms: float = MIXED_DECODE_STEP_MS,
        mixed_prefill_ms_per_token: float = MIXED_PREFILL_MS_PER_TOKEN,
        defer_turn_close: bool = DEFER_TURN_CLOSE,
        prefill_fit_graph: bool = PREFILL_FIT_GRAPH,
        mixed_graph: bool = MIXED_GRAPH,
        fold_turn_suffix: bool = FOLD_TURN_SUFFIX,
        prefill_accum: bool = PREFILL_ACCUM,
        prefill_accum_tokens: int = PREFILL_ACCUM_TOKENS,
        prefill_accum_max_wait_ms: float = PREFILL_ACCUM_MAX_WAIT_MS,
        prefill_accum_cost: tuple[float, ...] = PREFILL_ACCUM_COST,
        prefill_accum_min_decode: int = PREFILL_ACCUM_MIN_DECODE,
        clock: Callable[[], float] = time.perf_counter,
        prefill_early_launch: bool = PREFILL_EARLY_LAUNCH,
    ) -> None:
        self.runner = runner
        self.prefill_fit_graph = prefill_fit_graph
        self.defer_turn_close = defer_turn_close
        self.fold_turn_suffix = fold_turn_suffix
        self.cache = cache
        self.prefill_chunk = prefill_chunk
        self.token_budget = token_budget
        self.batched_prefill = batched_prefill
        self.spec_decode = spec_decode
        probe = getattr(runner, "forced_drafts", None)
        self._forced_drafts = bool(
            spec_decode and MTP_FORCED_DRAFTS and probe is not None and probe()
        )
        # Under MTP, overlap needs a runner that launches rounds without waiting
        # (`speculative_launch`, mtp_overlap.py); see OVERLAP_SCHED's docstring.
        self.overlap = overlap and (not spec_decode or hasattr(runner, "speculative_launch"))
        # Early prefill launch hooks the plain overlapped decode step only.
        self.prefill_early_launch = prefill_early_launch and self.overlap and not spec_decode
        if overlap and not self.overlap:
            print(
                "[scheduler] SEED_OVERLAP_SCHED is ignored: this runner cannot launch MTP "
                "rounds asynchronously",
                flush=True,
            )
        if mixed_batch and spec_decode:
            print("[scheduler] SEED_MIXED_BATCH is ignored while SEED_MTP_SERVE is on", flush=True)
        self._launched: _Launched | None = None
        self._launch_count = 0  # step id of the last `_launch_overlap`
        self._done_step = 0  # step id of the last step whose ids were read back
        # Auto-disabled under MTP -- see SEED_MIXED_BATCH's docstring's "Auto-disabled" note;
        # `Model.decode_mixed` also refuses to run under MTP as a model-side backstop.
        self.mixed_batch = mixed_batch and not spec_decode
        self.mixed_graph = mixed_graph and not spec_decode
        self.mixed_token_budget = mixed_token_budget
        self.mixed_decode_step_ms = mixed_decode_step_ms
        self.mixed_prefill_ms_per_token = mixed_prefill_ms_per_token
        # Under MTP too: captured prefill carries the MTP tail (graph_prefill `_mtp_tail`),
        # and accumulation only decides when a prefill step fires, never what a round does.
        self.prefill_accum = (
            prefill_accum
            and batched_prefill
            and prefill_fit_graph
            and not mixed_batch
            and not mixed_graph
        )
        if prefill_accum and not self.prefill_accum:
            print(
                "[scheduler] SEED_PREFILL_ACCUM needs captured batched prefill and no mixed "
                "steps; ignored",
                flush=True,
            )
        self.prefill_accum_tokens = max(1, prefill_accum_tokens)
        self.prefill_accum_max_wait_s = max(0.0, prefill_accum_max_wait_ms) / 1e3
        self.prefill_accum_cost = (prefill_accum_cost[0], prefill_accum_cost[1])
        self.prefill_accum_min_decode = max(1, prefill_accum_min_decode)
        self._clock = clock
        self._fill_log = PREFILL_FILL_LOG
        self.free_lanes: deque[int] = deque(range(runner.max_batch))
        self.incoming: queue.Queue[Request | _ResetPrefixCache | ScoreJob] = queue.Queue()
        self.waiting: deque[Request] = deque()
        self.score_jobs: deque[ScoreJob] = deque()
        self.prefill_q: deque[_InFlight] = deque()
        self.decoding: list[_InFlight] = []
        self._burst = 0
        self._owner_thread: threading.Thread | None = None
        # Set by `run()`, not `__init__`: a caller that drives the scheduler directly via
        # `step()` (every CPU seed test) never starts that thread, and `reset_prefix_cache`
        # below must still work synchronously in that mode -- see its docstring.
        self._timing = step_timing.Window() if step_timing.ENABLED else None
        self._trace: step_timing.ServiceTrace | None = None
        if step_timing.SERVICE_TRACE:
            import torch  # noqa: PLC0415 -- only under the diagnostic flag; module stays torch-free

            self._trace = step_timing.ServiceTrace(
                lambda: torch.cuda.Event(enable_timing=True), step_timing.SERVICE_TRACE
            )
        self._last_prefill = (0, 0)  # (requests, tokens) of the last prefill step, for timing
        self._last_decode_tokens = 0  # tokens the last decode step emitted, for timing
        self._trace_rows = 0  # SEED_SERVICE_TRACE: rows of the last launched decode step
        self._trace_forced = 0  # ... of which fed a non-final folded suffix token
        # SEED_STEP_TIMING: MTP counters since the last decode log line: lanes, drafted
        # tokens, accepted drafts, committed tokens, rounds.
        self._mtp_window = [0, 0, 0, 0, 0]
        # SEED_MTP_ACCEPT_LOG: (turn kind, bucket) -> [lane-rounds, accepted drafts]
        self._accept_diag: dict[tuple[str, int], list[int]] = {}
        self._accept_rounds = 0

    # -- public -------------------------------------------------------------
    def submit(self, req: Request) -> None:
        """Hand a request to the scheduler thread. Safe to call from any thread."""
        self.incoming.put(req)

    def reset_prefix_cache(self) -> int:
        """Drop every cache entry. Safe to call from any thread: unlike `submit`, whose target
        (`queue.Queue`) is itself thread-safe and needs nothing more, `SessionCache` is not --
        its trie/free-list mutations (`reserve_blocks`, `reserve_snapshot`, `publish`,
        `_evict_one_leaf`, ...) all assume the scheduler's own thread is the only writer. Calling
        `SessionCache.evict_all()` directly from here, on the admin endpoint's thread, would race
        with exactly those mutations while `run()` is mid-step on its own thread, corrupting the
        trie or `_free_snapshots` (silently: the exception, if any, lands on the *scheduler*
        thread's `step()`, gets swallowed by `run()`'s abort-and-continue below, and can leak a
        snapshot slot forever with no trie entry left to account for it -- observed exactly this
        way against a real server). So a call from any *other* thread enqueues a control message
        and blocks until the scheduler thread itself performs the evict, the same single-writer
        discipline `submit` already gives `Request`s. A call from the scheduler's *own* thread
        (or when `run()` was never started at all, e.g. every CPU seed test driving `step()`
        directly -- `_owner_thread` is `None` then) runs `evict_all()` immediately instead:
        enqueuing and waiting on `self.incoming` in that case would deadlock, since nothing ever
        drains that queue except `run()`/`step()` itself, which is exactly the caller blocked
        waiting. Either way, returns the number of entries removed; safe with lanes in flight
        (see `SessionCache.evict_all`'s docstring): only the trie's own references go away, a
        lane actively resuming from a node keeps the physical blocks/state alive through its own
        reference.
        """
        if self._owner_thread is None or threading.current_thread() is self._owner_thread:
            return self.cache.evict_all(protect=self._live_parents())
        msg = _ResetPrefixCache()
        self.incoming.put(msg)
        msg.done.wait()
        return msg.result

    def score(self, ids: Sequence[int], continuation_start: int, finish: Callable[[Any], Any]):
        """Run one teacher-forced scoring forward on a free lane; returns `finish(logits)`.

        Blocks until done; safe from any thread, same single-writer discipline as
        `reset_prefix_cache` (a call from the owner thread, or with `run()` never started,
        runs inline). Raises `RuntimeError` with the scheduler-side error message on failure.
        """
        job = ScoreJob(list(ids), continuation_start, finish)
        if self._owner_thread is None or threading.current_thread() is self._owner_thread:
            self.score_jobs.append(job)
            while not job.done.is_set():
                if not self._run_score_jobs():
                    self.step()
        else:
            self.incoming.put(job)
            job.done.wait()
        if job.error is not None:
            raise RuntimeError(job.error)
        return job.result

    def run(self) -> None:
        """Loop forever, blocking on `incoming` whenever there is no work."""
        self._owner_thread = threading.current_thread()
        while True:
            try:
                progressed = self.step()
                if not progressed:
                    # Something still queued but not admissible right now (only possible
                    # transiently): poll instead of blocking, so it is retried.
                    busy = bool(self.waiting or self.score_jobs)
                    try:
                        item = self.incoming.get(timeout=0.05) if busy else self.incoming.get()
                    except queue.Empty:
                        continue
                    self._intake(item)
            except Exception as exc:  # noqa: BLE001 -- an iteration that cannot run must not hang clients
                self.abort(repr(exc))

    def _intake(self, item: Request | _ResetPrefixCache | ScoreJob) -> None:
        """Route one item off `incoming`: a `_ResetPrefixCache` control message is handled right
        here, on the scheduler's own thread (see `reset_prefix_cache`'s docstring); a
        `ScoreJob` waits for a free lane; anything else is a `Request`, queued as before."""
        if isinstance(item, _ResetPrefixCache):
            try:
                item.result = self.cache.evict_all(protect=self._live_parents())
            finally:
                item.done.set()
        elif isinstance(item, ScoreJob):
            self.score_jobs.append(item)
        else:
            self.waiting.append(item)

    def abort(self, message: str) -> None:
        """Fail every request the scheduler is holding. Nothing here can raise again: each
        client's terminal event is sent even if releasing another one's lane fails.

        Under `OVERLAP_SCHED` the in-flight step is dropped unread: its flights are all in
        `self.decoding` and are failed below, and their lanes' release/`begin` is stream-ordered
        behind it, so nothing touches their state early."""
        self._launched = None
        self._done_step = self._launch_count
        flights = [*self.prefill_q, *self.decoding]
        self.prefill_q.clear()
        for fl in flights:
            try:
                self._fail(fl, message)
            except Exception:  # noqa: BLE001, S110 -- `_release` already emitted in `finally`
                pass
        self.decoding.clear()
        pending = [*self.waiting, *self.score_jobs]
        self.waiting.clear()
        self.score_jobs.clear()
        for item in pending:
            try:
                if isinstance(item, ScoreJob):
                    item.error = message
                    item.done.set()
                else:
                    item.emit(("error", message))
            except Exception:  # noqa: BLE001, S110 -- one client's emit must not strand the rest
                pass

    def step(self) -> bool:
        """Run one iteration. Returns False when there is nothing left to do."""
        if self._trace is not None:
            self._trace.begin()
        progressed = self._step()
        if self._trace is not None:
            self._trace.drop()
        return progressed

    def _trace_end(self, kind: str, **meta: Any) -> None:
        """`SEED_SERVICE_TRACE`: close the current step's record (see `step_timing.ServiceTrace`)."""
        if self._trace is None:
            return
        self._trace.end(
            kind,
            path=getattr(self.runner, f"{kind}_path", "-"),
            decoding=len(self.decoding),
            prefill_q=len(self.prefill_q),
            waiting=len(self.waiting),
            **meta,
        )

    def _step(self) -> bool:
        self._drain()
        if self._run_score_jobs():
            return True
        self._admit()
        fit = self._mixed_fit() if self.prefill_q else None
        if fit is not None:
            # SEED_MIXED_GRAPH: all decoding lanes plus a captured-shape prefill chunk in one
            # replay. Sampled on the host like prefill, so the in-flight step completes first.
            # The fit was taken for the pre-flush decode count, which the flush only shrinks.
            self._flush()
            if self.decoding and self.prefill_q:
                self._burst = 0
                t0 = time.perf_counter()
                self._mixed_step(fit)
                if self._timing is not None:
                    self._log_step("mixed", t0, len(self.decoding), self._last_prefill[1])
                self._trace_end("mixed", rows=self._last_prefill[0], tokens=self._last_prefill[1])
                return True
        if self.mixed_batch and self.prefill_q and (self.decoding or self._launched is not None):
            # A mixed step samples its decode rows on the host, like prefill, so under
            # OVERLAP_SCHED the in-flight step is completed first and the mixed step itself runs
            # serially (overlap resumes on the next decode-only step). Completing it can finish
            # every decoding flight, so the check below is repeated after the flush.
            self._flush()
        if self.mixed_batch and self.decoding and self.prefill_q:
            # SEED_MIXED_BATCH: both sides have work, so mix them into one iteration instead of
            # alternating -- see the flag's own docstring. A step with only one side's work
            # (nothing yet decoding, or the prefill queue momentarily empty) falls through to
            # the ordinary single-purpose steps below unchanged; there is nothing to mix in.
            self._burst = 0
            t0 = time.perf_counter()
            self._mixed_step()
            if self._timing is not None:
                self._log_step("mixed", t0, len(self.decoding), self._last_prefill[1])
            self._trace_end("mixed", rows=self._last_prefill[0], tokens=self._last_prefill[1])
            return True
        prefill_first = not self.decoding or self._burst < MAX_PREFILL_BURST
        plan = None
        if self.prefill_accum and self.prefill_q and prefill_first:
            if self._start_accum_exact_duplicates():
                return True
            plan = self._accum_plan()
            prefill_first = plan is not None
        if self.prefill_q and prefill_first:
            previous = None
            if self.prefill_early_launch:  # see PREFILL_EARLY_LAUNCH: complete it after the launch
                previous, self._launched = self._launched, None
            else:
                self._flush()  # prefill samples on the host; see OVERLAP_SCHED's docstring
            self._burst += 1
            t0 = time.perf_counter()
            self._prefill_step(plan)
            if previous is not None:
                self._complete(previous, None)
            if self._timing is not None:
                self._log_step("prefill", t0, *self._last_prefill)
            self._trace_end(
                "prefill",
                rows=self._last_prefill[0],
                tokens=self._last_prefill[1],
                plan=list(plan) if plan is not None else None,
            )
            return True
        if self.decoding or self._launched is not None:
            self._burst = 0
            batch = len(self.decoding)
            self._last_decode_tokens = batch
            self._trace_rows = self._trace_forced = 0
            t0 = time.perf_counter()
            if self.overlap:
                self._overlap_decode_step()
            else:
                self._decode_step()
            if self._timing is not None:
                self._log_step("decode", t0, batch, self._last_decode_tokens)
            self._trace_end(
                "decode",
                rows=self._trace_rows,
                forced=self._trace_forced,
                bucket=getattr(getattr(self.runner, "model", None), "decode_bucket", -1),
            )
            return True
        return False

    def _log_step(self, kind: str, t0: float, batch: int, tokens: int) -> None:
        """`SEED_STEP_TIMING` only: record one step and log every `step_timing.EVERY`th.

        Under `OVERLAP_SCHED` a decode step's wall time is the host's launch-plus-complete
        time, not the device's step time; the device-side idle time between steps is what
        `step_timing.GapMeter` reports, per rank."""
        ms = (time.perf_counter() - t0) * 1e3
        assert self._timing is not None
        if not self._timing.add(kind, ms, tokens):
            return
        mean_ms, tok_s = self._timing.drain(kind)
        path = (
            getattr(self.runner, f"{kind}_path", "-")
            if kind in ("decode", "prefill", "mixed")
            else "-"
        )
        mtp = ""
        if kind == "decode" and self._mtp_window[4]:
            lanes, drafted, accepted, committed, rounds = self._mtp_window
            mtp = (
                f" mtp_rounds={rounds} mtp_accept={accepted / max(drafted, 1):.3f} "
                f"mtp_tok_per_lane_step={committed / max(lanes, 1):.2f}"
            )
            self._mtp_window = [0, 0, 0, 0, 0]
        snapshots = self.cache.snapshot_stats()
        step_timing.log(
            f"sched {kind} n={self._timing.count[kind]} batch={batch} tokens={tokens} "
            f"wall_ms={ms:.1f} mean_ms={mean_ms:.1f} tok_s={tok_s:.1f} path={path} "
            f"decoding={len(self.decoding)} prefill_q={len(self.prefill_q)} burst={self._burst} "
            f"snap_used={snapshots.used}/{snapshots.capacity} "
            f"snap_peak={snapshots.high_watermark} "
            f"snap_evict={snapshots.pressure_evictions} "
            f"snap_reclaim={snapshots.reclaims} "
            f"snap_exhaust={snapshots.reservation_failures} "
            f"cache_nodes={self.cache.node_count()}"
            f"{mtp}"
        )

    # -- admission ----------------------------------------------------------
    def _drain(self) -> None:
        while True:
            try:
                item = self.incoming.get_nowait()
            except queue.Empty:
                return
            self._intake(item)

    def _admit(self) -> None:
        while self.waiting:
            req = self.waiting[0]
            if req.max_new <= 0:  # nothing to generate; never touches a lane
                self.waiting.popleft()
                req.emit(("end", ("length", 0, 0)))
                continue
            try:
                placed = self._acquire(req.prompt, req.has_history)
            except Exception as exc:  # noqa: BLE001 -- `_acquire` rolled back; fail just this one
                self.waiting.popleft()
                req.emit(("error", repr(exc)))
                continue
            if placed is None:
                return
            self.waiting.popleft()
            lane, node = placed
            start = node.depth
            self.prefill_q.append(
                _InFlight(
                    req,
                    lane,
                    start,
                    list(req.prompt[start:]),
                    start,
                    node,
                    queued_at=self._clock(),
                )
            )

    def _live_parents(self) -> list[Node]:
        """Every node some other in-flight request currently depends on as its `fl.parent`.

        Eviction (`SessionCache._evict_one_leaf`) picks the LRU-oldest *leaf* over the whole
        trie with no idea which leaves are someone's `fl.parent` right now -- a node can be a
        perfectly ordinary childless leaf and still be exactly what a different flight is about
        to extend or publish under. `protect=` on `reserve_blocks`/`reserve_snapshot` stops a
        reservation from picking its *own* caller's node, but a concurrent flight's reservation
        knows nothing about that caller's node unless it is *also* in the protected set -- see
        `SessionCache.reserve_blocks`'s docstring for the orphaned-subtree leak this closes.
        Every caller below passes its own node explicitly *in addition* to this, so this need not
        (and structurally cannot, since `_release` already dropped `fl` from `self.decoding`
        before calling `_publish_turn_close`) include every caller's own case itself.
        """
        return [fl.parent for fl in self.prefill_q] + [fl.parent for fl in self.decoding]

    def _acquire(self, prompt: list[int], has_history: bool = False) -> tuple[int, Node] | None:
        """Reserve a lane for `prompt`: the deepest matching cache node (if any), adopted onto
        any free lane. `None` means no free lane; the request stays queued. When the cache
        cannot free a block for copy-on-write even after evicting everything evictable, this
        admits from scratch (the root) instead of refusing: refusing with nothing in flight
        would leave the request waiting for a completion that never comes.

        On any runner failure the lane goes back to `free_lanes` and every block reference
        taken here is dropped before the exception propagates (`_admit` fails the request).
        """
        if not self.free_lanes:
            return None
        node, matched = self.cache.lookup_detail(prompt)
        self._count_admission(prompt, node, matched, has_history)
        blocks: tuple[int, ...] = ()
        new_block: int | None = None
        try:
            if not node.is_root():
                try:
                    if self.cache.needs_cow(node):
                        # protect=[node, *live]: reserving may need to evict, and node -- about
                        # to be adopted -- may itself still be a childless leaf right now, so it
                        # must not be its own victim; every other flight's own `fl.parent` must
                        # survive this too (see `_live_parents`).
                        protect = [node, *self._live_parents()]
                        new_block = self.cache.reserve_blocks(1, protect=protect)[0]
                        filled = node.depth % self.cache.block_size
                        self.runner.copy_block(new_block, node.blocks[-1], filled)
                        blocks = self.cache.adopt(node, replacement_last_block=new_block)
                    else:
                        blocks = self.cache.adopt(node)
                    new_block = None  # owned by `blocks` now
                except SessionCacheError:
                    node, blocks = self.cache.root, ()
        except BaseException:
            if new_block is not None:
                self.cache.release_lane_blocks([new_block])
            raise
        lane = self.free_lanes.popleft()
        try:
            self.runner.begin(lane)
            if not node.is_root():
                self.runner.load_snapshot(lane, node.snapshot)
                self.runner.attach_blocks(lane, blocks)
        except BaseException:
            self.cache.release_lane_blocks(blocks)
            self.free_lanes.appendleft(lane)
            raise
        return lane, node

    def _count_admission(
        self, prompt: list[int], node: Node, matched: int, has_history: bool
    ) -> None:
        """Classify one admission (`SessionCache.classify`) and log `[cache-stats]` sparsely:
        the first 4 admissions, then every 256th. Always on; costs one dict update per
        admission plus, on a miss, a hash over the prompt."""
        stats = self.cache.stats
        cls = self.cache.classify(prompt, node, matched)
        turn = "t2" if has_history else "t1"
        stats[f"{turn}_{cls}"] += 1
        stats[f"{turn}_admitted"] += 1
        stats[f"{turn}_reused_tokens"] += node.depth
        stats[f"{turn}_prompt_tokens"] += len(prompt)
        stats["admitted"] += 1
        count = stats["admitted"]
        if count <= 4 or count % 256 == 0:
            snaps = self.cache.snapshot_stats()
            fields = " ".join(f"{k}={v}" for k, v in sorted(stats.items()))
            print(
                f"[cache-stats] {fields} snap_used={snaps.used}/{snaps.capacity} "
                f"snap_reclaim={snaps.reclaims} snap_evict={snaps.pressure_evictions} "
                f"snap_exhaust={snaps.reservation_failures} cache_nodes={self.cache.node_count()}",
                flush=True,
            )

    def _grow_lanes(
        self, targets: Sequence[tuple[int, int]], protect: Sequence[Node] = ()
    ) -> set[int]:
        """Reserve and attach KV blocks so each `(lane, tokens)` target can hold `tokens`
        tokens, before the forward that writes them. Returns the lanes that could not be grown.

        This is the only place blocks are allocated while serving. `SessionCache.
        reserve_blocks` evicts LRU cache entries under pressure (a direct allocator call would
        just raise mid-decode with the pool full of evictable cached sessions), and the
        resolved ids reach every rank through `runner.extend_blocks` (under TP a rank-local
        allocation would diverge from rank 0's, see `tp_driver.Broadcaster.extend_blocks`).
        A lane fails only if the pool is still short after evicting every unprotected leaf.
        """
        bs = self.cache.block_size
        guard = [*protect, *self._live_parents()]
        grants: list[tuple[int, int]] = []
        failed: set[int] = set()
        for lane, tokens in targets:
            need = -(-tokens // bs) - self.runner.lane_block_count(lane)
            if need <= 0:
                continue
            try:
                got = self.cache.reserve_blocks(need, protect=guard)
            except SessionCacheError:
                failed.add(lane)
                continue
            grants.extend((lane, b) for b in got)
        if grants:
            try:
                self.runner.extend_blocks(grants)
            except BaseException:
                self.cache.release_lane_blocks([b for _, b in grants])
                raise
        return failed

    def _run_score_jobs(self) -> bool:
        """Run queued `ScoreJob`s while a lane is free. Returns whether any ran."""
        ran = False
        while self.score_jobs and self.free_lanes:
            job = self.score_jobs.popleft()
            lane = self.free_lanes.popleft()
            ran = True
            try:
                self.runner.begin(lane)
                if self._grow_lanes([(lane, len(job.ids))]):
                    raise SessionCacheError("KV block pool exhausted for /v1/score")
                logits = self.runner.score(lane, job.ids, job.continuation_start)
                job.result = job.finish(logits)
            except Exception as exc:  # noqa: BLE001 -- reported to the caller
                job.error = repr(exc)
            finally:
                try:
                    self.cache.release_lane_blocks(self.runner.lane_blocks(lane))
                    self.runner.begin(lane)
                finally:
                    self.free_lanes.append(lane)
                    job.done.set()
        return ran

    # -- publishing -----------------------------------------------------------
    def _publish_boundary(
        self, fl: _InFlight, boundary_pos: int, edge: tuple[int, ...], logits: Any
    ) -> None:
        """Insert a new node at `boundary_pos`, extending `fl.parent`, and advance `fl.parent`
        to it. A no-op if `edge` is empty (the boundary coincides with `fl.parent` already, e.g.
        an admission that matched exactly at this boundary), if the cache cannot free a
        snapshot slot even after evicting everything evictable, or if anything else in here
        fails unexpectedly -- publishing a cache entry is always best-effort, and every caller
        of this method (`_finish_prefill_chunk`, `_publish_turn_close`) is on a path that must
        still deliver a token or a terminal event to the client regardless, so nothing in here
        is allowed to propagate and abort the request over a caching failure.
        """
        if not edge:
            return
        snap: int | None = None
        try:
            # protect=[fl.parent, *live]: reserving may need to evict, and fl.parent -- about
            # to gain the child this call is building -- may itself still be a childless leaf
            # right now. Evicting it here would orphan the child about to be published under it
            # (its blocks/snapshot already decref'd/freed by the eviction) and leak this
            # reservation, since `publish` below would attach to a node no longer reachable
            # from the trie root; every other flight's own `fl.parent` needs the same
            # protection against *this* reservation (see `_live_parents`).
            snap = self.cache.reserve_snapshot(protect=[fl.parent, *self._live_parents()])
            self.runner.save_snapshot(fl.lane, snap)
            self.runner.cache_node_logits(snap, logits)
            blocks = self.runner.lane_blocks(fl.lane)
            nblocks = -(-boundary_pos // self.cache.block_size)
            fl.parent = self.cache.publish(
                fl.parent, edge, tuple(blocks[:nblocks]), snap, own=fl.own_parent
            )
            fl.own_parent = True
            snap = None  # the new node owns it
        except Exception:  # noqa: BLE001 -- see the docstring: never fail the request over this
            self.cache.stats["publish_failed"] += 1
        finally:
            if snap is not None:  # reserved but never published: give the slot back
                self.cache.release_snapshot(snap)

    def _publish_turn_close(self, fl: _InFlight, logits: Any, forwarded: bool = False) -> None:
        """Publish the turn-close boundary (paged-kv-design.md section 2.3) when a request
        finishes: prompt plus the tokens actually generated, the resume point the *next* chat
        turn's re-serialized history needs.

        Always re-forwards at least the *last* generated token before publishing, regardless of
        `suffix_len`: decode only ever forwards token K as the *input* to the step that produces
        token K+1, so the very last token of a finished reply is sampled but never itself
        forwarded ("the last sampled token is never forwarded again: nothing follows it" -- true
        of every finished generation, Stage 1 included). Its KV rows/DeltaNet update therefore
        do not exist yet anywhere, and turn-close's whole point is a resumable state that
        includes them.

        When `suffix_len == 0` there is no positional gap between `fl.parent` (where generation
        started) and the live state, so only that one missing token needs forwarding -- cheap,
        and strictly less work than Stage 1's `_try_apply_reply`, which always re-forwarded the
        whole reply. When `suffix_len > 0`, generation actually ran at positions past the suffix
        (`fl.parent.depth + suffix_len`), but the *next* turn's history renders the reply
        directly after the boundary with no suffix in between (the same reason `suffix_len`
        exists at all -- see the module docstring), so the *entire* reply must be re-forwarded
        at `fl.parent.depth` to land at the positions a real recompute of the next prompt would
        use -- exactly the position-shifted forward Stage 1's `_try_apply_reply` used to do
        lazily; here it runs eagerly, on `fl.lane` before it is released.

        The `suffix_len > 0` branch only rewinds *state* (`load_snapshot`), never the block
        table: `fl.lane`'s own table already holds a valid, continuously-owned prefix covering
        `[0, fl.parent.depth)` -- established at admission (`_acquire`'s `adopt`, copy-on-write
        already applied there if `fl.parent`'s last block was partial) and never released since,
        since it never has to shrink. Re-attaching `fl.parent.blocks` directly here would be
        wrong: if admission needed copy-on-write, `fl.parent.blocks` still ends in the
        *original* (un-cow'd) block, which this lane holds no reference to at all -- writing
        into it would both corrupt whatever else still shares it and be unpinned against a
        concurrent eviction. Leaving the lane's own table alone and letting the re-forward's
        `prefill` call grow/overwrite it in place (its abandoned suffix+decode tail, if any, is
        harmless leftover data the lane already owns and `_release` still decrefs wholesale)
        is both simpler and avoids a double-incref that routing this through `SessionCache.
        adopt` again would cause, since the lane already holds this share.

        `forwarded` (OVERLAP_SCHED only): the lane already forwarded the last generated token,
        as the input of the lookahead step launched before this token was known to be the
        last, and `logits` is that step's row. With `suffix_len == 0` that forward *is* the
        one-token re-forward above, so it is published as is. With `suffix_len > 0` the whole
        reply is re-forwarded from the snapshot anyway, which does not care what the live
        state holds.

        `SEED_DEFER_TURN_CLOSE`: no forward at all; see `DEFER_TURN_CLOSE` for what is
        published instead and why that is exact.
        """
        if not fl.output_ids:
            return
        # Both branches assume `fl.parent` is the prompt's turn-open boundary: the publish
        # there (or the admission match) must have succeeded. If it failed, `fl.parent` is
        # some shallower node (possibly the root, snapshot -1, which LOAD_SNAPSHOT would
        # index out of range on every rank), the suffix_len>0 rewind would resume from the
        # wrong state, and the suffix_len==0 forward would land at the wrong position and
        # publish a node whose KV does not match its tokens. Skip instead.
        if fl.parent.is_root() or fl.parent.depth != len(fl.req.prompt) - max(fl.req.suffix_len, 0):
            return
        edge = tuple(fl.output_ids)
        if self.defer_turn_close:
            # SEED_DEFER_TURN_CLOSE: see its docstring for why each case is exact.
            if fl.req.suffix_len > 0:
                return
            if not forwarded and logits is not REFORWARD:
                self._publish_boundary(fl, fl.parent.depth + len(edge) - 1, edge[:-1], logits)
                return
        if fl.req.suffix_len > 0:
            forward, start = edge, fl.parent.depth
        elif forwarded:
            self._publish_boundary(fl, fl.parent.depth + len(edge), edge, logits)
            return
        else:
            forward, start = edge[-1:], fl.parent.depth + len(edge) - 1
        try:
            if self._grow_lanes([(fl.lane, start + len(forward))], protect=[fl.parent]):
                return
            if fl.req.suffix_len > 0:
                self.runner.load_snapshot(fl.lane, fl.parent.snapshot)
            logits = self.runner.prefill(fl.lane, list(forward), start)
        except Exception:  # noqa: BLE001 -- turn-close reuse is an optimization, not required
            return
        self._publish_boundary(fl, fl.parent.depth + len(edge), edge, logits)

    # -- iterations ---------------------------------------------------------
    def _prefill_step(self, plan: tuple[int, int, bool, float, int, int] | None = None) -> None:
        if self.batched_prefill:
            self._prefill_step_batched(plan)
        else:
            self._prefill_step_single()

    def _prefill_step_single(self) -> None:
        """The original one-request-at-a-time path: only ever touches `prefill_q[0]`.

        Kept byte-for-byte in shape so `batched_prefill=False` (or `SEED_BATCHED_PREFILL=0`) is
        a real A/B baseline, not just a differently-shaped version of the batched path at width 1.
        """
        fl = self.prefill_q[0]
        self._last_prefill = (0, 0)
        if not fl.pending:  # exact-duplicate match: nothing left to forward, see _acquire
            self.prefill_q.popleft()
            self._start_exact_dup(fl)
            return
        chunk = fl.pending[: self._chunk_limit(fl, self.prefill_chunk)]
        del fl.pending[: len(chunk)]
        self._last_prefill = (1, len(chunk))
        try:
            if self._grow_lanes([(fl.lane, fl.pos + len(chunk))]):
                raise SessionCacheError(POOL_EXHAUSTED)
            logits = self.runner.prefill(fl.lane, chunk, fl.pos)
        except Exception as exc:  # noqa: BLE001 -- report to the client, keep serving
            self.prefill_q.popleft()
            self._fail(fl, repr(exc))
            return
        self._finish_prefill_chunk(fl, chunk, logits)

    def _chunk_limit(self, fl: _InFlight, limit: int) -> int:
        """`limit`, shrunk so this chunk stops exactly at the chat-suffix boundary (see the
        module docstring) when that boundary falls inside it; otherwise `limit` unchanged."""
        boundary = len(fl.req.prompt) - fl.req.suffix_len
        if fl.req.suffix_len > 0 and fl.pos < boundary <= fl.pos + limit:
            return boundary - fl.pos  # stop exactly at the boundary this step
        return limit

    def _select_prefill_batch(
        self,
        budget: int | None = None,
        mixed_fit: Any = None,
        max_rows: int | None = None,
        max_width: int | None = None,
        pack: tuple[int, int] | None = None,
    ) -> tuple[list[_InFlight], list[tuple[_InFlight, list[int]]]]:
        """Walk `prefill_q` in FIFO order, filling `budget` (`self.token_budget` by default)
        with each request's next chunk (each capped at `prefill_chunk` too, same as the
        single-request path).

        A request whose whole prompt is already resident (`fl.pending` empty, an exact-
        duplicate match) costs no tokens and is always collected, regardless of budget: it
        needs no forward call at all (see `_prefill_step_batched`). Once the budget is spent,
        later requests needing real chunks are left for the next call; earlier ones in FIFO
        order always get first claim on the budget, which is the fairness property -- true
        whether `budget` is the ordinary `token_budget` or `_mixed_budget`'s tighter figure
        (`_select_mixed_prefill_batch` is this same walk, just parameterized). Chunks already
        taken here mutate `fl.pending` (and are cut short of the true budget limit if the
        chat-suffix boundary falls inside them), matching the single-request path's own
        bookkeeping order.

        `pack=(area, align)` (`SEED_PREFILL_PACK` under `SEED_PREFILL_ACCUM`): the call targets a
        packed shape, so each chunk costs its length rounded up to `align` out of `area` (rows
        are not a limit, only `max_rows` segment slots), and the flight that meets the end of
        the area is cut to what is left.
        """
        exact_dup: list[_InFlight] = []
        batch: list[tuple[_InFlight, list[int]]] = []
        # `SEED_PREFILL_FIT_GRAPH` shapes packed calls; `mixed_fit` (`SEED_MIXED_GRAPH`)
        # shapes a mixed step's chunks to a captured mixed shape. Eager mixed steps are unshaped.
        if mixed_fit is not None:
            fit = mixed_fit[:2]
        else:
            fit = self._prefill_fit() if budget is None else None
        budget = self.token_budget if budget is None else budget
        area_left, align = pack if pack is not None else (0, 1)
        for fl in self.prefill_q:
            if not fl.pending:
                exact_dup.append(fl)
                continue
            if budget <= 0 or (max_rows is not None and len(batch) >= max_rows):
                continue  # no budget left for a real chunk; keep scanning for exact-dups only
            if pack is not None:
                if area_left < align:
                    continue
                chunk = fl.pending[: self._chunk_limit(fl, min(self.prefill_chunk, budget, area_left))]
                del fl.pending[: len(chunk)]
                batch.append((fl, chunk))
                budget -= len(chunk)
                area_left -= -(-len(chunk) // align) * align
                continue
            limit = min(self.prefill_chunk, budget)
            if max_width is not None:
                limit = min(limit, max_width)
            if fit is not None:
                fits, fit_width = fit
                limit = min(limit, fit_width)
                n = min(self._chunk_limit(fl, limit), len(fl.pending))
                lengths = [len(c) for _, c in batch]
                if batch and mixed_fit is not None:
                    # A mixed step's multi-row shapes are narrower than its one-row shapes:
                    # shrink this row's chunk to the widest that still fits, if any.
                    while n > 0 and not fits([*lengths, n]):
                        n -= 1
                    limit = n
                if batch and (n <= 0 or not fits([*lengths, n])):
                    budget = 0  # the call is full; later requests wait for the next step
                    continue
            chunk = fl.pending[: self._chunk_limit(fl, limit)]
            del fl.pending[: len(chunk)]
            batch.append((fl, chunk))
            budget -= len(chunk)
        return exact_dup, batch

    def _prefill_fit(self) -> Any:
        """The runner's `(fits, max_width)` for captured prefill shapes, or None (see
        `PREFILL_FIT_GRAPH`). Runners without `prefill_fit` (CPU tests, eager) give None."""
        if not self.prefill_fit_graph:
            return None
        fit = getattr(self.runner, "prefill_fit", None)
        return fit() if fit is not None else None

    def _prefill_step_batched(self, plan: tuple[int, int, bool, float, int, int] | None = None) -> None:
        """Pack every request `_select_prefill_batch` picked into one `runner.prefill_batch`
        call, then apply each result exactly as `_prefill_step_single` would have applied its
        own single-request result (boundary-publish bookkeeping unchanged, just run once per
        request in the packed batch instead of once per call). `plan` (`SEED_PREFILL_ACCUM`,
        `_accum_plan`) caps the call at one shape's rows and width."""
        t0 = time.perf_counter()
        if plan is None:
            exact_dup, batch = self._select_prefill_batch()
        elif plan[4]:  # a packed shape: `plan[4]` segment slots, `plan[5]` alignment
            exact_dup, batch = self._select_prefill_batch(
                max_rows=plan[4], pack=(plan[0] * plan[1], plan[5])
            )
        else:
            exact_dup, batch = self._select_prefill_batch(max_rows=plan[0], max_width=plan[1])
        self._last_prefill = (len(batch), sum(len(chunk) for _, chunk in batch))
        try:
            self._run_prefill_batch(exact_dup, batch)
        finally:
            if self._fill_log and batch:
                forced, wait = (plan[2], plan[3]) if plan is not None else (False, 0.0)
                print(
                    f"[prefill-fill] rows={len(batch)} tokens={self._last_prefill[1]} "
                    f"max_chunk={max(len(c) for _, c in batch)} "
                    f"path={getattr(self.runner, 'prefill_path', '-')} "
                    f"wall_ms={(time.perf_counter() - t0) * 1e3:.1f} "
                    f"decoding={len(self.decoding)} q={len(self.prefill_q)} "
                    f"accum={int(plan is not None)} forced={int(forced)} "
                    f"wait_ms={wait * 1e3:.0f}",
                    flush=True,
                )

    def _run_prefill_batch(
        self, exact_dup: list[_InFlight], batch: list[tuple[_InFlight, list[int]]]
    ) -> None:
        for fl in exact_dup:
            self.prefill_q.remove(fl)
            self._start_exact_dup(fl)
        if batch:
            failed = self._grow_lanes([(fl.lane, fl.pos + len(chunk)) for fl, chunk in batch])
            for fl, _ in [item for item in batch if item[0].lane in failed]:
                self.prefill_q.remove(fl)
                self._fail(fl, POOL_EXHAUSTED)
            batch = [item for item in batch if item[0].lane not in failed]
        if not batch:
            return
        calls = [(fl.lane, chunk, fl.pos) for fl, chunk in batch]
        try:
            if self._trace is not None:
                self._trace.mark("fwd0")
            results = self.runner.prefill_batch(calls)
            if self._trace is not None:
                self._trace.mark("fwd1")
        except Exception as exc:  # noqa: BLE001 -- report to the clients, keep serving
            for fl, _ in batch:
                self.prefill_q.remove(fl)
                self._fail(fl, repr(exc))
            return
        for (fl, chunk), logits in zip(batch, results, strict=True):
            self._finish_prefill_chunk(fl, chunk, logits)

    def _accum_plan(self) -> tuple[int, int, bool, float, int, int] | None:
        """`SEED_PREFILL_ACCUM`: `(rows, width, age_forced, oldest_wait_s, segs, align)` of the
        captured shape to fire now, or None to keep accumulating (decode runs instead). `segs`
        is 0 for a per-row shape, else the packed shape's segment slots (`SEED_PREFILL_PACK`),
        which is filled by aligned area (`_packed_real`) instead of rows capped at the width.

        Fires at once when fewer than `prefill_accum_min_decode` lanes decode, when the best shape
        already carries `prefill_accum_tokens` real tokens, or when the oldest queued flight
        has waited `prefill_accum_max_wait_s`. The shape maximizes modeled real tokens per ms
        over the FIFO head: each shape takes the first `rows` flights, each chunk capped at
        its width (and at the chat-suffix boundary, as the packed path does)."""
        flights = [fl for fl in self.prefill_q if fl.pending]
        if not flights:
            return None
        shapes = self._prefill_shapes()
        if not shapes:
            return (len(flights), self.prefill_chunk, False, 0.0, 0, 1)
        packing = self._prefill_packing()
        fixed, per_tok = self.prefill_accum_cost
        best: tuple[float, int, int, int, int, int] | None = None  # (score, real, rows, width, segs, align)
        for rows, width in shapes:
            segs, align = packing.get((rows, width), (0, 1))
            if segs:
                w = width
                real = self._packed_real(flights, rows * width, segs, align)
            else:
                w = min(width, self.prefill_chunk)
                real = sum(
                    min(len(fl.pending), self._chunk_limit(fl, w)) for fl in flights[:rows]
                )
            score = real / (fixed + per_tok * rows * width)
            if best is None or (score, real) > (best[0], best[1]):
                best = (score, real, rows, w, segs, align)
        assert best is not None
        _, real, rows, width, segs, align = best
        wait = self._clock() - min(fl.queued_at for fl in flights)
        if len(self.decoding) < self.prefill_accum_min_decode or real >= self.prefill_accum_tokens:
            return (rows, width, False, wait, segs, align)
        if wait >= self.prefill_accum_max_wait_s:
            return (rows, width, True, wait, segs, align)
        return None

    def _packed_real(self, flights: list[_InFlight], area: int, segs: int, align: int) -> int:
        """Real tokens `_select_prefill_batch(pack=(area, align), max_rows=segs)` would take
        from `flights` (FIFO, pending work only), without taking them."""
        real, budget = 0, self.token_budget
        for fl in flights[:segs]:
            if area < align or budget <= 0:
                break
            n = min(len(fl.pending), self._chunk_limit(fl, min(self.prefill_chunk, budget, area)))
            real += n
            budget -= n
            area -= -(-n // align) * align
        return real

    def _prefill_packing(self) -> dict[tuple[int, int], tuple[int, int]]:
        packing = getattr(self.runner, "prefill_packing", None)
        return dict(packing()) if packing is not None else {}

    def _prefill_shapes(self) -> list[tuple[int, int]]:
        shapes = getattr(self.runner, "prefill_shapes", None)
        return list(shapes()) if shapes is not None else []

    def _start_accum_exact_duplicates(self) -> bool:
        """Answer zero-work exact-duplicate admissions now rather than holding them for the
        next accumulated step (they need no forward call)."""
        exact = [fl for fl in self.prefill_q if not fl.pending]
        for fl in exact:
            self.prefill_q.remove(fl)
            self._start_exact_dup(fl)
        return bool(exact)

    def _start_exact_dup(self, fl: _InFlight) -> None:
        """Answer an exact-duplicate admission from its node's cached logits, no forward.

        `fl.parent` is protected from eviction while `fl` is queued (`_live_parents`), so its
        snapshot slot, which indexes the cached logits, cannot have been freed and reused by
        another node. Checked anyway: reading a reused slot would silently emit another
        prompt's next token."""
        if not self.cache.is_live(fl.parent):
            self._fail(fl, "cached prefix was evicted before its first token")
            return
        logits = self.runner.cached_node_logits(fl.parent.snapshot)
        self.decoding.append(fl)
        self._emit(fl, self.runner.sample_batch(logits, [fl.req.temperature])[0], logits)

    def _finish_prefill_chunk(self, fl: _InFlight, chunk: list[int], logits: Any) -> None:
        """Bookkeeping for one flight's one successfully-forwarded chunk: advance its
        position, publish the chat-suffix boundary or the prompt's own end (whichever applies),
        and either leave it queued for its next chunk or move it to decode. Shared by both the
        single-request and the packed prefill paths, since neither differs in what a completed
        chunk means for one request."""
        fl.pos += len(chunk)
        boundary = len(fl.req.prompt) - fl.req.suffix_len
        if fl.req.suffix_len > 0 and fl.pos == boundary:
            # Publish here, before the suffix a chat template only adds to the prompt it
            # generates from (see the module docstring); the suffix itself still gets
            # prefilled below (or fed via decode ticks, if folded), on top of this node, for
            # this request's own decode.
            edge = tuple(fl.req.prompt[fl.parent.depth : boundary])
            self._publish_boundary(fl, boundary, edge, logits)
            if self.fold_turn_suffix and fl.pending:
                # SEED_FOLD_TURN_SUFFIX: don't give the suffix a prefill chunk of its own.
                # `fl.pending` is exactly the suffix (`_chunk_limit` stopped `chunk` at the
                # boundary), so queue it as forced decode input and join `self.decoding`
                # directly -- no forward call here, and no sample from the boundary state's
                # own logits (they predict the token after the boundary, not after the
                # suffix, so they are irrelevant to what gets fed next).
                suffix_ids = fl.pending
                fl.pending = []
                fl.forced = deque(suffix_ids[1:])
                fl.next_token = suffix_ids[0]
                self.prefill_q.remove(fl)
                self.decoding.append(fl)
                return
        if fl.pending:
            return
        self.prefill_q.remove(fl)
        if fl.req.suffix_len <= 0:
            edge = tuple(fl.req.prompt[fl.parent.depth : fl.pos])
            self._publish_boundary(fl, fl.pos, edge, logits)
        self.decoding.append(fl)
        self._emit(fl, self.runner.sample_batch(logits, [fl.req.temperature])[0], logits)

    def _mixed_budget(self) -> int:
        """Adaptive token budget for one `_mixed_step`'s prefill side (Sarathi-Serve's
        fixed-per-iteration "token budget" idea, sized dynamically here instead of fixed):
        chosen so the prefill side adds at most `MIXED_BUDGET_SLACK` (0.5, i.e. 1.5x total) of
        `mixed_decode_step_ms`'s own cost, at `mixed_prefill_ms_per_token` cost per token,
        capped by the outer `mixed_token_budget` ceiling (`SEED_MIXED_TOKEN_BUDGET`).

        `max(1, ...)`: the FIFO head of `prefill_q` must always make *some* progress, even
        under a budget so tight the formula would otherwise round to 0 tokens -- a real chunk
        of 0 would stall that request forever behind an ever-growing decode batch that keeps
        making the 1.5x ceiling tighter still. One token over budget is a bounded, one-time
        cost; zero progress is unbounded.
        """
        if self.mixed_prefill_ms_per_token <= 0:
            return self.mixed_token_budget
        by_cost = int(
            MIXED_BUDGET_SLACK * self.mixed_decode_step_ms / self.mixed_prefill_ms_per_token
        )
        return max(1, min(self.mixed_token_budget, by_cost))

    def _select_mixed_prefill_batch(
        self, fit: Any = None
    ) -> tuple[list[_InFlight], list[tuple[_InFlight, list[int]]]]:
        if fit is not None:
            return self._select_prefill_batch(min(self.mixed_token_budget, fit[2]), fit)
        return self._select_prefill_batch(self._mixed_budget())

    def _mixed_fit(self) -> Any:
        """`SEED_MIXED_GRAPH`: the runner's `(fits, max_width, max_tokens)` for a mixed step
        over the current decoding lanes, or None (flag off, nothing decoding, no captured
        shape with that many decode rows, or a runner without `mixed_fit`)."""
        if not self.mixed_graph or not self.decoding:
            return None
        fit = getattr(self.runner, "mixed_fit", None)
        return fit(len(self.decoding)) if fit is not None else None

    def _mixed_step(self, fit: Any = None) -> None:
        """One `SEED_MIXED_BATCH` iteration: every decoding lane's next token, in the same
        forward call as one prefill chunk sized by `_mixed_budget` -- see the flag's own
        docstring for the idea (Sarathi-Serve / vLLM chunked prefill) and the stall it removes.
        Only called from `step()` when both `self.decoding` and `self.prefill_q` are nonempty,
        and never with a step in flight (`step()` flushes one first under `OVERLAP_SCHED`).

        Exact-duplicate matches (`fl.pending` already empty -- see `_select_prefill_batch`)
        are resolved exactly as `_prefill_step_batched` resolves them (`_start_exact_dup`): no
        forward call, cached logits, straight to `self.decoding`.

        Blocks for both halves are reserved on rank 0 before the forward, like `_decode_step`
        and `_prefill_step_batched` do separately: `pos + 1` per decode row, `pos + len(chunk)`
        per prefill chunk, in one `_grow_lanes` call (one `EXTEND_BLOCKS` broadcast under TP).
        A lane that cannot be grown is failed with `POOL_EXHAUSTED` and left out; if every
        decode row is left out, the surviving chunks run as a plain `prefill_batch`.
        """
        exact_dup, batch = self._select_mixed_prefill_batch(fit)
        for fl in exact_dup:
            self.prefill_q.remove(fl)
            self._start_exact_dup(fl)

        # Same per-lane list-building as `_decode_step`: `flights` doubles as the pre-mutation
        # snapshot of `self.decoding` the final loop needs, since `_emit`/`_release` mutate
        # `self.decoding` as requests finish below.
        flights: list[_InFlight] = []
        lanes: list[int] = []
        next_tokens: list[int] = []
        positions: list[int] = []
        temperatures: list[float] = []
        for fl in self.decoding:
            flights.append(fl)
            lanes.append(fl.lane)
            next_tokens.append(fl.next_token)
            positions.append(fl.pos)
            temperatures.append(fl.req.temperature)

        targets = [(lane, pos + 1) for lane, pos in zip(lanes, positions, strict=True)]
        targets += [(fl.lane, fl.pos + len(chunk)) for fl, chunk in batch]
        failed = self._grow_lanes(targets)
        if failed:
            keep = [i for i, fl in enumerate(flights) if fl.lane not in failed]
            for fl in [fl for fl in flights if fl.lane in failed]:
                self._fail(fl, POOL_EXHAUSTED)
            flights = [flights[i] for i in keep]
            lanes = [lanes[i] for i in keep]
            next_tokens = [next_tokens[i] for i in keep]
            positions = [positions[i] for i in keep]
            temperatures = [temperatures[i] for i in keep]
            for fl, _ in [item for item in batch if item[0].lane in failed]:
                self.prefill_q.remove(fl)
                self._fail(fl, POOL_EXHAUSTED)
            batch = [item for item in batch if item[0].lane not in failed]

        self._last_prefill = (len(batch), sum(len(chunk) for _, chunk in batch))
        if not flights and not batch:
            return
        calls = [(fl.lane, chunk, fl.pos) for fl, chunk in batch]
        try:
            if flights:
                decode_logits, prefill_logits = self.runner.decode_mixed(
                    lanes, next_tokens, positions, calls
                )
            else:
                decode_logits, prefill_logits = None, self.runner.prefill_batch(calls)
        except Exception as exc:  # noqa: BLE001 -- report to the clients, keep serving
            for fl in flights:
                self._fail(fl, repr(exc))
            for fl, _ in batch:
                self.prefill_q.remove(fl)
                self._fail(fl, repr(exc))
            return

        if flights:
            tokens = self.runner.sample_batch(decode_logits, temperatures)
            for i, (fl, token) in enumerate(zip(flights, tokens, strict=True)):
                fl.pos += 1
                fl.next_token = None
                self._emit(fl, token, self.runner.decode_row(decode_logits, i))

        for (fl, chunk), logits in zip(batch, prefill_logits, strict=True):
            self._finish_prefill_chunk(fl, chunk, logits)

    def _decode_step(self) -> None:
        # One pass over the active lanes builds every per-lane list `runner.decode` and
        # `sample_batch` need, instead of a separate comprehension (and attribute lookup)
        # per field; `flights` doubles as the `list(self.decoding)` snapshot the final loop
        # needs, since `_release` mutates `self.decoding` as requests finish below.
        flights: list[_InFlight] = []
        lanes: list[int] = []
        next_tokens: list[int] = []
        positions: list[int] = []
        temperatures: list[float] = []
        for fl in self.decoding:
            flights.append(fl)
            lanes.append(fl.lane)
            next_tokens.append(fl.next_token)
            positions.append(fl.pos)
            temperatures.append(fl.req.temperature)
        speculative = self._speculative_ok(flights)
        width = self.runner.decode_tokens_per_step() if speculative else 1
        failed = self._grow_lanes(
            [(lane, pos + width) for lane, pos in zip(lanes, positions, strict=True)]
        )
        if failed:
            for fl in [fl for fl in flights if fl.lane in failed]:
                self._fail(fl, POOL_EXHAUSTED)
            keep = [i for i, fl in enumerate(flights) if fl.lane not in failed]
            flights = [flights[i] for i in keep]
            lanes = [lanes[i] for i in keep]
            next_tokens = [next_tokens[i] for i in keep]
            positions = [positions[i] for i in keep]
            temperatures = [temperatures[i] for i in keep]
            if not flights:
                return
        if speculative:
            self._speculative_decode_step(flights, lanes, next_tokens, positions)
            return
        try:
            logits = self.runner.decode(lanes, next_tokens, positions)
        except Exception as exc:  # noqa: BLE001 -- report to the clients, keep serving
            for fl in flights:
                self._fail(fl, repr(exc))
            return
        tokens = self.runner.sample_batch(logits, temperatures)
        for i, (fl, token) in enumerate(zip(flights, tokens, strict=True)):
            fl.pos += 1
            fl.next_token = None
            self._emit(fl, token, self.runner.decode_row(logits, i))

    def _speculative_ok(self, flights: list[_InFlight]) -> bool:
        """Whether this decode step can be an MTP round; see `SPECULATIVE_DECODE`.

        A flight with a nonempty `forced` queue (`SEED_FOLD_TURN_SUFFIX`) is excluded: it
        needs its exact next input to be the queued chat-suffix id, not a drafted/accepted
        token, so any step it is part of falls back to plain per-token decode until its
        suffix drains (`_finish_prefill_chunk`, `_emit`)."""
        if not self.spec_decode:
            return False
        width = self.runner.decode_tokens_per_step()
        max_seq = self.runner.max_seq
        forced_ok = self._forced_drafts
        return all(
            (forced_ok or not fl.forced)
            and fl.req.temperature <= 0
            and len(fl.req.stop) <= MTP_MAX_STOP_IDS
            and fl.pos + width <= max_seq
            for fl in flights
        )

    def _speculative_decode_step(
        self, flights: list[_InFlight], lanes: list[int], tokens: list[int], positions: list[int]
    ) -> None:
        """The MTP counterpart of the tail of `_decode_step`: one `runner.speculative_decode`
        call in place of `decode` + `sample_batch`, then each lane's whole accepted batch
        (1..`mtp.k + 1` tokens) applied through the ordinary `_emit` one token at a time, same
        bookkeeping (`fl.pos`, stop tokens, `max_new`) as a single-token step repeated. A
        request that stops or hits its budget partway through its own accepted batch (`fl.done`
        after `_release`) simply does not see its later tokens emitted; the model has already
        advanced that lane's state past them, which is wasted work, not a correctness problem
        (see `mtp.verify_and_commit`'s docstring on why an unread KV/state row is harmless).

        The round is given each lane's remaining budget and stop ids and ends each lane's
        committed list (and state) at the first stop id or at the budget (`mtp.commit_limit`),
        so every returned token is emitted; `fl.done` stays as a guard only.

        `logits=REFORWARD` on every `_emit` here: `speculative_decode` returns committed token
        ids, not a batched logits tensor, so a finishing request's turn-close publish
        re-forwards its last token (see `REFORWARD` and `SPECULATIVE_DECODE`)."""
        budgets = [fl.req.max_new - fl.generated for fl in flights]
        stops = [sorted(fl.req.stop) for fl in flights]
        extra = {}
        if self._forced_drafts and any(fl.forced for fl in flights):
            k = self.runner.decode_tokens_per_step() - 1
            extra["forced"] = [list(fl.forced)[:k] for fl in flights]
        try:
            committed = self.runner.speculative_decode(
                lanes, tokens, positions, budgets, stops, **extra
            )
        except Exception as exc:  # noqa: BLE001 -- report to the clients, keep serving
            for fl in flights:
                self._fail(fl, repr(exc))
            return
        forced_n = [len(f) for f in extra["forced"]] if extra else [0] * len(flights)
        self._accept_note(flights, committed, forced_n)  # W10 diagnostic
        emitted = 0
        for fl, batch in zip(flights, committed, strict=True):
            for token in batch:
                if fl.done:
                    break
                fl.pos += 1
                fl.next_token = None
                emitted += 1
                self._emit(fl, token, REFORWARD)
        self._last_decode_tokens = emitted
        if self._timing is not None:
            k = self.runner.decode_tokens_per_step() - 1
            w = self._mtp_window
            w[0] += len(flights)
            w[1] += k * len(flights)
            w[2] += sum(len(batch) - 1 for batch in committed)
            w[3] += sum(len(batch) for batch in committed)
            w[4] += 1

    # -- OVERLAP_SCHED ------------------------------------------------------------
    def _overlap_decode_step(self) -> None:
        """Launch step N+1, then complete step N while the device runs N+1.

        A flight left out of N+1 because its KV blocks could not be reserved is failed only
        after N completes, so the token it already has in flight still reaches the client
        first (the serial path emits step N's token before step N+1's reservation fails).
        """
        if self.spec_decode:
            self._overlap_speculative_step()
            return
        current, starved = self._launch_overlap()
        previous, self._launched = self._launched, current
        if previous is not None:
            self._complete(previous, current)
        for fl in starved:
            if not fl.done:
                self._fail(fl, POOL_EXHAUSTED)

    def _flush(self) -> None:
        """Complete the in-flight step, if any, with nothing launched behind it."""
        previous, self._launched = self._launched, None
        if previous is not None:
            self._complete(previous, None)

    def _launch_overlap(self) -> tuple[_Launched | None, list[_InFlight]]:
        """Launch one decode step over every flight that still needs a token. Returns the
        launched step (None if nothing launched) and the flights whose KV blocks could not be
        reserved (left out of the launch; `_overlap_decode_step` fails them).

        A flight whose newest token is still on the device (`last_step > _done_step`) is fed
        `LOOKAHEAD` and is one token further along than `generated` says; if that in-flight
        token is its last under `max_new`, it is left out, since the host already knows it
        will finish (see OVERLAP_SCHED's docstring). Positions are advanced here, at launch,
        because the host knows them without the ids.

        Block reservation happens here, before the launch, exactly as `_decode_step` does it:
        step N+1 writes KV row `fl.pos` for every lane it runs, and `fl.pos` is known on the
        host without step N's ids, so `_grow_lanes` sizes each lane to `fl.pos + 1` and (under
        TP) broadcasts `EXTEND_BLOCKS` ahead of `DECODE_LAUNCH` on the same ordered command
        channel. Every rank's table therefore holds the ids before the launch reads it. Step
        N, still running, captured its own block-table snapshot at its launch, and extending
        only appends ids, so it is unaffected. Blocks freed while N+1 is in flight (a release
        or an eviction) can be handed to another lane only for work enqueued after N+1 on the
        same stream, so N+1's writes land first.
        """
        flights: list[_InFlight] = []
        lanes: list[int] = []
        tokens: list[int] = []
        positions: list[int] = []
        temperatures: list[float] = []
        for fl in self.decoding:
            outstanding = fl.last_step > self._done_step
            if fl.generated + outstanding >= fl.req.max_new:
                continue
            flights.append(fl)
            lanes.append(fl.lane)
            # A folded suffix token is known on the host even while the previous step is still
            # outstanding. Peek at the next queued id here. Completing the previous step below
            # pops that same id into `next_token`, keeping host bookkeeping aligned without
            # putting the model's discarded suffix prediction on the next launch. Once the
            # queue is empty, LOOKAHEAD is correct again: that in-flight suffix-final step's
            # prediction is the first generated token.
            token = fl.forced[0] if fl.forced else LOOKAHEAD
            tokens.append(token if outstanding else fl.next_token)
            positions.append(fl.pos)
            temperatures.append(fl.req.temperature)
        if not flights:
            return None, []
        failed = self._grow_lanes(
            [(lane, pos + 1) for lane, pos in zip(lanes, positions, strict=True)]
        )
        starved: list[_InFlight] = []
        if failed:
            keep = [i for i, fl in enumerate(flights) if fl.lane not in failed]
            starved = [fl for fl in flights if fl.lane in failed]
            flights = [flights[i] for i in keep]
            lanes = [lanes[i] for i in keep]
            tokens = [tokens[i] for i in keep]
            positions = [positions[i] for i in keep]
            temperatures = [temperatures[i] for i in keep]
            if not flights:
                return None, starved
        if self._trace is not None:
            self._trace_rows = len(lanes)
            self._trace_forced = sum(1 for fl in flights if len(fl.forced) >= 2)
            self._trace.mark("fwd0")
        try:
            handle = self.runner.decode_launch(lanes, tokens, positions, temperatures)
            if self._trace is not None:
                self._trace.mark("fwd1")
        except Exception as exc:  # noqa: BLE001 -- report to the clients, keep serving
            for fl in flights:
                self._fail(fl, repr(exc))
            return None, starved
        self._launch_count += 1
        for row, fl in enumerate(flights):
            fl.last_step, fl.row = self._launch_count, row
            fl.pos += 1
        return _Launched(self._launch_count, flights, handle), starved

    def _overlap_speculative_step(self) -> None:
        """The MTP form of `_overlap_decode_step`: launch round N+1, then read round N. A
        step that cannot be a captured round (a greedy/stop/`max_seq` rule, a forced suffix
        without `SEED_MTP_FORCED_DRAFTS`, no bucket) completes the in-flight round and runs
        `_decode_step` serially.

        A lane in round N is fed as `LOOKAHEAD` (device tables) unless round N carried `f`
        forced drafts and its suffix outlasts them: such a round commits exactly `f + 1`
        forced ticks and no generated token, so the host knows the next input
        (`fl.forced[f]`) and position (`fl.pos + f + 1`) and sends them explicitly."""
        t = self.runner.decode_tokens_per_step()
        k = t - 1
        flights: list[_InFlight] = []
        lanes: list[int] = []
        tokens: list[int] = []
        positions: list[int] = []
        budgets: list[int] = []
        forced: list[list[int]] = []
        need: list[tuple[int, int]] = []
        for fl in self.decoding:
            queue = list(fl.forced)
            if fl.last_step > self._done_step:
                f = fl.inflight_forced
                if f and len(queue) > f:
                    token, pos, queue = queue[f], fl.pos + f + 1, queue[f + 1 :]
                    end = pos + t
                else:
                    if fl.generated + 1 >= fl.req.max_new:
                        continue  # the in-flight round commits a generated token: its last
                    token, pos, queue, end = LOOKAHEAD, fl.pos, [], fl.pos + 2 * t
            else:
                token, pos, end = fl.next_token, fl.pos, fl.pos + t
            flights.append(fl)
            lanes.append(fl.lane)
            tokens.append(token)
            positions.append(pos)
            budgets.append(fl.req.max_new - fl.generated)
            forced.append(queue[:k])
            need.append((fl.lane, end))
        stops = [sorted(fl.req.stop) for fl in flights]
        max_seq = self.runner.max_seq
        ok = (
            bool(flights)
            and self._speculative_ok(flights)
            and (self._forced_drafts or not any(forced))
            and all(end <= max_seq for _, end in need)
            and self.runner.speculative_launch_ok(lanes, stops)
        )
        if not ok:
            self._flush()
            if self.decoding:
                self._decode_step()
            return
        failed = self._grow_lanes(need)
        starved = [fl for fl in flights if fl.lane in failed]
        if failed:
            keep = [i for i, fl in enumerate(flights) if fl.lane not in failed]
            flights = [flights[i] for i in keep]
            lanes, tokens = [lanes[i] for i in keep], [tokens[i] for i in keep]
            positions, budgets = [positions[i] for i in keep], [budgets[i] for i in keep]
            stops, forced = [stops[i] for i in keep], [forced[i] for i in keep]
        current = None
        if flights:
            extra = {"forced": forced} if any(forced) else {}
            try:
                handle = self.runner.speculative_launch(
                    lanes, tokens, positions, budgets, stops, **extra
                )
            except Exception as exc:  # noqa: BLE001 -- report to the clients, keep serving
                for fl in flights:
                    self._fail(fl, repr(exc))
                handle = None
            if handle is not None:
                self._launch_count += 1
                for row, fl in enumerate(flights):
                    fl.last_step, fl.row = self._launch_count, row
                    fl.inflight_forced = len(forced[row])
                current = _Launched(self._launch_count, flights, handle, speculative=True)
        previous, self._launched = self._launched, current
        if previous is not None:
            self._complete(previous, current)
        for fl in starved:
            if not fl.done:
                self._fail(fl, POOL_EXHAUSTED)

    def _accept_note(self, flights: list[_InFlight], committed: list[list[int]], forced: list[int]) -> None:
        """`SEED_MTP_ACCEPT_LOG`: fold one round's accept lengths into the diagnostic split."""
        if not MTP_ACCEPT_LOG:
            return
        for fl, batch, f in zip(flights, committed, forced, strict=True):
            n = fl.mtp_rounds
            fl.mtp_rounds += 1
            if f:
                continue
            b = next(i for i, (lo, hi) in enumerate(MTP_ACCEPT_BUCKETS) if lo <= n <= hi)
            cell = self._accept_diag.setdefault(("follow" if fl.reused else "first", b), [0, 0])
            cell[0] += 1
            cell[1] += len(batch) - 1
        self._accept_rounds += 1
        if self._accept_rounds % 50:
            return
        parts = []
        for (kind, b), (lanes, acc) in sorted(self._accept_diag.items()):
            lo, hi = MTP_ACCEPT_BUCKETS[b]
            span = f"{lo}" if lo == hi else f"{lo}-{hi if hi < 1 << 30 else 'inf'}"
            parts.append(f"{kind}/r{span}={acc / max(lanes, 1):.3f}(n={lanes})")
        print(f"[mtp-accept] rounds={self._accept_rounds} " + " ".join(parts), flush=True)
        self._accept_diag.clear()

    def _complete_speculative(self, done: _Launched) -> None:
        """Read one launched MTP round back and emit each lane's committed tokens, like
        `_speculative_decode_step`'s tail. A flight that finished in an earlier round (its
        row ran inactive) is skipped."""
        try:
            committed = done.handle.committed()
        except Exception as exc:  # noqa: BLE001 -- report to the clients, keep serving
            self._done_step = done.step
            for fl in done.flights:
                if not fl.done:
                    self._fail(fl, repr(exc))
            return
        self._done_step = done.step
        self._accept_note(done.flights, committed, [fl.inflight_forced for fl in done.flights])
        emitted = 0
        for fl, batch in zip(done.flights, committed, strict=True):
            for token in batch:
                if fl.done:
                    break
                fl.pos += 1
                fl.next_token = None
                emitted += 1
                self._emit(fl, token, REFORWARD)
        self._last_decode_tokens = emitted
        if self._timing is not None:
            k = self.runner.decode_tokens_per_step() - 1
            live = [(fl, b) for fl, b in zip(done.flights, committed, strict=True)]
            w = self._mtp_window
            w[0] += len(live)
            w[1] += k * len(live)
            w[2] += sum(len(b) - 1 for _, b in live)
            w[3] += sum(len(b) for _, b in live)
            w[4] += 1

    def _complete(self, done: _Launched, current: _Launched | None) -> None:
        """Read `done`'s ids back and apply them, `_decode_step`'s tail one step late.

        A flight that finished while `done` was in flight (a stop token read from the step
        before) is skipped: this is its discarded extra token. A flight that is also in
        `current` has already forwarded this token, which `_emit` passes on to a turn-close
        publish as `forwarded` with `current`'s logits row.
        """
        if done.speculative:
            self._complete_speculative(done)
            return
        try:
            ids = done.handle.tokens()
        except Exception as exc:  # noqa: BLE001 -- report to the clients, keep serving
            self._done_step = done.step
            for fl in done.flights:
                if not fl.done:
                    self._fail(fl, repr(exc))
            return
        self._done_step = done.step
        for i, (fl, token) in enumerate(zip(done.flights, ids, strict=True)):
            if fl.done:
                continue
            fl.next_token = None
            forwarded = current is not None and fl.last_step == current.step
            logits = current.handle.row(fl.row) if forwarded else done.handle.row(i)
            self._emit(fl, token, logits, forwarded)

    def _emit(self, fl: _InFlight, token: int, logits: Any, forwarded: bool = False) -> None:
        """Report one generated token, mirroring Model.generate's stop/budget rules.

        `logits` is this step's own next-token logits for `fl` (a prefill chunk's result, the
        exact-duplicate node's cached logits, or one row of a batched decode result) -- carried
        through only in case this call also finishes the request, so `_release` can publish the
        turn-close boundary without an extra forward call in the common (`suffix_len == 0`) case.
        `forwarded`: see `_publish_turn_close`.

        A stop token counts towards `completion_tokens` but its text is not streamed, which is
        what the one-at-a-time server did. `output_ids` records every emitted token, including a
        stop token, so `_release`'s turn-close publish sees the whole reply.

        `fl.forced` (`SEED_FOLD_TURN_SUFFIX`): a nonempty queue means this step's real `token`
        is a chat-suffix tick, not generated output -- discard it, feed the next queued id, and
        return before any of the stop/budget/streaming bookkeeping below runs. See
        `FOLD_TURN_SUFFIX`'s docstring.
        """
        if fl.forced:
            fl.next_token = fl.forced.popleft()
            return
        fl.generated += 1
        fl.output_ids.append(token)
        if token in fl.req.stop:
            self._release(fl, ("end", ("stop", fl.generated, fl.reused)), logits, forwarded)
            return
        fl.req.emit(("tok", token))
        if fl.generated >= fl.req.max_new:
            self._release(fl, ("end", ("length", fl.generated, fl.reused)), logits, forwarded)
        else:
            fl.next_token = token

    def _fail(self, fl: _InFlight, message: str) -> None:
        self._release(fl, ("error", message), None)

    def _release(self, fl: _InFlight, event: Event, logits: Any, forwarded: bool = False) -> None:
        """Publish turn-close (best effort), free the lane, and send the terminal event.

        The event is sent in `finally`: `fl` has already left `self.decoding`, so nothing else
        will ever fail or release it, and a client with no terminal event hangs forever. An
        exception from the lane cleanup still propagates (to `run()`'s abort) after that.
        `forwarded`: see `_publish_turn_close`."""
        fl.done = True
        if event[0] == "error" or (MTP_ACCEPT_LOG and event[0] == "end" and event[1][0] == "stop"):
            print(
                f"[scheduler] turn {event[0]} lane={fl.lane} generated={fl.generated} "
                f"max_new={fl.req.max_new} pos={fl.pos}: {event[1]!r}",
                flush=True,
            )
        try:
            try:
                self.decoding.remove(fl)  # one scan instead of an `in` check plus a second scan
            except ValueError:
                pass  # fl never reached self.decoding (a prefill-time failure)
            if event[0] != "error" and logits is not None:
                try:
                    self._publish_turn_close(fl, logits, forwarded)
                except Exception:  # noqa: BLE001, S110 -- publishing is an optimization
                    pass
            try:
                self.cache.release_lane_blocks(self.runner.lane_blocks(fl.lane))
                self.runner.begin(fl.lane)  # published or abandoned: start the lane clean
            finally:
                self.free_lanes.append(fl.lane)
        finally:
            fl.req.emit(event)

"""OpenAI-style chat server around the seed model.

    python3 server.py --model-path <dir> --host 0.0.0.0 --port 8000

One worker thread owns the model and runs `scheduler.Scheduler`, which admits
many requests at once: a new request's prefill is interleaved with the decode
steps of the requests already in flight instead of queueing behind them, and
one decode step covers the whole batch. The scheduler also keeps each finished
request's state in its slot, so the next turn of the same session prefills only
the tokens the previous turn did not already have (see scheduler.py). A chat
template renders a generation prompt (role marker plus, in Qwen3.5's case, a
`<think>` block, open or already closed depending on `chat_template_kwargs`)
that is not part of any later turn's history, whether or not thinking is on:
history only ever replays the messages themselves. `chat_suffix_len` derives
that per request, fresh from its own messages and kwargs (see its docstring),
so the scheduler records the reusable prefix before it, not after (see
scheduler.py's `Request.suffix_len`), which is what keeps this reuse working
regardless of thinking mode or how a client represents a prior turn.

Processes. The model is tensor-parallel over the node's GPUs (see tp.py), one
process per rank. This process is rank 0: it serves HTTP, runs the scheduler,
and broadcasts every model call to the others, which it starts by re-executing
this file with `--rank` and which sit in `tp_driver.serve_worker`. Rank 0 owns
their lifetime and stops them when the server exits, and the launcher starts
rank 0 in its own process group, so killing that group takes the workers with
it. At `--tp 1` there are no other processes and no collectives: one process
holds the whole model, with its layers split contiguously over `--devices` and
the decode step pipelined across them (see pipeline.py). That is the path the
CPU tests take, and the two are mutually exclusive; see model.py.

HTTP surface: POST /v1/chat/completions takes text messages, same SSE shape as
before, one chunk per generated token. POST /v1/completions is the token-id
path benchmarking tools drive directly: `prompt` is a list of token ids (never
text), streamed chunks carry `token_ids` instead of text deltas, and the
trailing usage chunk reports `prompt_tokens_details.cached_tokens` for prefix
reuse. Every finished request, on either endpoint, publishes a turn-close cache
entry unconditionally (scheduler.py's Stage 2 prefix cache; see its module
docstring's "Dropped from Stage 1" paragraph), so a later exact repeat of a
request's own prompt+reply is always a full cache hit -- no opt-in flag needed.
POST /reset_prefix_cache clears every idle cached prefix (vLLM's name for the
same admin hook) without touching in-flight requests. GET /health is 503 until
the weights are loaded and a warmup forward succeeded, and reports an error if
a worker rank has died.
"""

import argparse
import asyncio
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter, OrderedDict
from pathlib import Path

import blas_tune
import graph_mixed
import graph_prefill
import host_numa
import mem_timeline
import model as model_module
import torch
import tp
from aiohttp import web
from graph_decode import GraphDecodeRunner
from model import Model, load_cfg
from prompt_cache import PromptCache
from scheduler import DEFER_TURN_CLOSE, OVERLAP_SCHED, Request, Scheduler
from session_cache import SessionCache
from tp_driver import (
    Broadcaster,
    PoolMismatchError,
    command_channel,
    pool_handshake,
    serve_worker,
)
from transformers import AutoTokenizer

RANK_BW_PROBE = os.environ.get("SEED_RANK_BW_PROBE", "0") == "1"
"""Boot diagnostic: every rank prints `[rank-bw]`, its device read bandwidth; see host_numa."""

CHAT_HISTORY_REUSE = os.environ.get("SEED_CHAT_HISTORY_REUSE", "0") not in (
    "0",
    "",
    "false",
    "False",
)
CHAT_HISTORY_REUSE_ENTRIES = int(os.environ.get("SEED_CHAT_HISTORY_REUSE_ENTRIES", "1024"))
"""Preserve the generation-prompt prefix inside rendered assistant history.

Qwen3.5's generation prompt starts an assistant message and then adds thinking-mode
boilerplate. The ordinary history render keeps the assistant marker but drops that
boilerplate, so the state that decoded a reply is at different token positions from the
same reply in the next request. With this flag, prior assistant content retains that text
and a bounded registry splices in the exact generated token ids, which need not equal an
encoding of concatenated streamed text. This makes the next prompt's token positions and
ids match the state that decoded the reply.
"""


class Engine:
    """Loads the model in the background, then runs the scheduler on that worker thread."""

    def __init__(
        self, args: argparse.Namespace, workers: list[subprocess.Popen] | None = None
    ) -> None:
        self.args = args
        self.workers = workers or []  # ranks 1.., for liveness reporting only
        self.tok = AutoTokenizer.from_pretrained(args.model_path)
        self.stop_ids = self._stop_ids(args.model_path)
        self.prompts = PromptCache(self.tok, capacity=max(args.max_batch, 8))
        self.chat_history: OrderedDict[str, tuple[str, tuple[int, ...], tuple[int, ...]]] = (
            OrderedDict()
        )
        self.chat_history_stats: Counter[str] = Counter()
        self.prompts.selfcheck(probe_prompts(self.tok))
        self.model: Model | None = None
        self.sched: Scheduler | None = None
        self.error: str | None = None
        threading.Thread(target=self._run, daemon=True).start()

    def _stop_ids(self, path: str) -> frozenset[int]:
        ids = set(load_cfg(path).eos)
        gen = Path(path) / "generation_config.json"
        if gen.exists():
            eos = json.loads(gen.read_text()).get("eos_token_id", [])
            ids |= {eos} if isinstance(eos, int) else set(eos)
        if self.tok.eos_token_id is not None:
            ids.add(self.tok.eos_token_id)
        return frozenset(ids)

    @property
    def ready(self) -> bool:
        return self.model is not None

    def dead_worker(self) -> str | None:
        """The first worker rank that has exited, if any.

        A dead rank hangs the next collective, because `tp_driver`'s protocol has no timeout,
        so `/health` reports it rather than leaving clients waiting on a step that will never
        come back.
        """
        for i, proc in enumerate(self.workers, start=1):
            if proc.poll() is not None:
                return f"rank {i} exited with {proc.returncode}"
        return None

    def _run(self) -> None:
        try:
            a = self.args
            t0 = time.time()
            model = build_model(a, rank=0)
            t1 = time.time()
            runner = build_runner(a, model)
            if a.tp > 1:  # outermost, so every call the scheduler makes is broadcast
                runner = Broadcaster(runner, command_channel(model.tp.device, OVERLAP_SCHED).send)
            runner.warmup()
            # Boot self-check on every rank, then the scheduler becomes the only block
            # allocator (see `tp_driver.pool_handshake`).
            if a.tp > 1:
                runner.pool_handshake()
            else:
                pool_handshake(model)
            cache = SessionCache(model.block_allocator, model.block_size, model.num_snapshots)
            self.sched = Scheduler(runner, cache)
            self.model = model  # last: this is what `ready` reports
            print(f"weights loaded in {t1 - t0:.0f}s, warmup {time.time() - t1:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)
            if isinstance(e, PoolMismatchError):
                # Fail fast: ranks that disagree on the KV pool would corrupt output or hang.
                # SIGTERM lets `web.run_app` exit through `main`'s `stop_workers`.
                print(f"[pool-handshake] {e}; shutting down", flush=True)
                os.kill(os.getpid(), signal.SIGTERM)
            raise
        self.sched.run()

    def submit(self, req: Request) -> None:
        self.sched.submit(req)


def build_model(args: argparse.Namespace, rank: int) -> Model:
    """Join the process group if there is one, then load this rank's shards.

    The two parallelism axes are exclusive (see model.py). At `--tp 1` this is the seed's
    single process holding a contiguous slice of the layers on each of `--devices`. Above
    that, this process is one rank owning `cuda:rank` and a shard of every layer, and
    `--devices` is not consulted.
    """
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    if args.tp == 1:
        devices = args.devices.split(",") if args.devices else _default_devices()
        model = Model(args.model_path, devices, dtype, args.max_seq_len, args.max_batch)
    else:
        reduce = tp.init(rank, args.tp, port=args.tp_port)
        mem_timeline.set_rank(rank)
        mem_timeline.mark("tp_init", tp.device_for(rank))
        plan = tp.plan(load_cfg(args.model_path), rank, args.tp)
        handle = tp.TP(plan, tp.device_for(rank), reduce)
        model = Model(
            args.model_path, [handle.device], dtype, args.max_seq_len, args.max_batch, tp=handle
        )
    # Before anything is served and before any capture: the search synchronizes, and every
    # rank needs it, since the step waits on the slowest. See `blas_tune`. Captured prefill
    # shapes (`SEED_PREFILL_GRAPHS`) add their GEMM widths; see `graph_prefill.TUNE`.
    mtp_t = None if model.mtp is None else model.mtp.k + 1
    blas_tune.tune(model, batches=tune_batches(model.max_batch, model.max_seq, mtp_t))
    mem_timeline.mark("blas_tune", model.devices[0])
    return model


def build_runner(args: argparse.Namespace, model: Model):
    """What drives the model on this rank: the model itself, or a graph-replaying wrapper.

    Without `--enable-graph-capture` this is the model, so the served path is the one that
    exists without graph_decode.py at all. With it, capture is attempted once here, before
    any request is served (it steps every slot), and a failure leaves a wrapper that forwards
    straight to the model.

    Every rank builds this, not just rank 0. A captured step holds that rank's all-reduces,
    so a rank still running eager would meet a replaying peer inside a collective; and even
    where that happened to line up, a step costs what its slowest rank costs, so capturing on
    rank 0 alone would remove no wall clock at all. `GraphDecodeRunner.prepare` reduces the
    verdict across the group so the ranks turn the captured path on together or not at all.

    Called after `build_model`, which is what guarantees `blas_tune.tune` has already run and
    turned its search off: the GEMM solutions this rank's decode shapes replay under capture
    are the frozen ones `tune` selected, not a fresh, capture-illegal search.
    """
    runner = model
    if args.enable_graph_capture:
        # With an MTP head (`SEED_MTP_SERVE=1`), also capture the draft+verify round.
        if model.mtp is not None:
            from mtp_overlap import OverlapMTPRunner  # noqa: PLC0415

            runner = OverlapMTPRunner(model)
        else:
            runner = GraphDecodeRunner(model)
        runner.prepare()
    if RANK_BW_PROBE and model.tp.device.type == "cuda":
        host_numa.probe(args.rank, model.tp.device)
    mem_timeline.report(model, model.devices[0])
    return runner


def tune_batches(max_batch: int, max_seq: int, mtp_t: int | None) -> list[int]:
    """Every GEMM M `build_model` has `blas_tune` tune: the decode buckets, the captured
    prefill and mixed-step widths, and (with an MTP head, `mtp_t = k + 1`) the wide-verify
    widths."""
    widths = [
        *blas_tune.tuned_batches(max_batch),
        *graph_prefill.gemm_widths(max_batch, max_seq),
        *graph_mixed.gemm_widths(max_batch, max_seq),
    ]
    if mtp_t is not None:
        widths += _verify_gemm_widths(max_batch, mtp_t)
    return sorted(set(widths))


def _verify_gemm_widths(max_batch: int, t: int) -> tuple[int, ...]:
    """GEMM M values the captured wide verify runs at (`bucket * t` per MTP bucket at or above
    `SEED_MTP_MIN_BUCKET`), for `blas_tune.tune`. Untuned widths take hipBLASLt's default
    heuristic, several times slower per call at M=144/240 (W13, round 15). Same list as W13's
    `graph_mtp.verify_gemm_widths`; kept here until that lands, since graph_mtp.py is W6's."""
    import graph_decode  # noqa: PLC0415
    import graph_mtp  # noqa: PLC0415

    if not graph_mtp.VERIFY_WIDE:
        return ()
    buckets = [
        b for b in graph_decode.build_buckets(max_batch) if b >= graph_mtp.VERIFY_WIDE_MIN_BUCKET
    ] or [max_batch]
    return tuple(sorted({b * t for b in buckets}))


def _default_devices() -> list[str]:
    n = torch.cuda.device_count()  # ROCm reports through torch.cuda too
    return [f"cuda:{i}" for i in range(n)] if n else ["cpu"]


def run_worker(args: argparse.Namespace) -> None:
    """Rank 1..: load this rank's shards, then apply rank 0's commands until it stops."""
    model = build_model(args, args.rank)
    runner = build_runner(args, model)
    serve_worker(runner, command_channel(model.tp.device, OVERLAP_SCHED).recv)


def start_workers(args: argparse.Namespace) -> list[subprocess.Popen]:
    """Re-exec this file once per non-zero rank. The caller owns stopping them.

    `--enable-graph-capture` has to be forwarded: a rank that did not capture would meet a
    replaying peer inside a layer's all-reduce. Every other flag here is one the rank's own
    shard depends on, which is why the list is explicit rather than a replay of `sys.argv`.
    """
    base = [
        sys.executable,
        os.path.abspath(__file__),
        *("--model-path", args.model_path),
        *("--dtype", args.dtype),
        *("--tp", str(args.tp)),
        *("--tp-port", str(args.tp_port)),
        *("--max-seq-len", str(args.max_seq_len)),
        *("--max-batch", str(args.max_batch)),
        *(["--enable-graph-capture"] if args.enable_graph_capture else []),
    ]
    return [subprocess.Popen([*base, "--rank", str(r)]) for r in range(1, args.tp)]  # noqa: S603


def stop_workers(workers: list[subprocess.Popen]) -> None:
    """Terminate every worker and wait for it. Safe to call more than once."""
    for proc in workers:
        if proc.poll() is None:
            proc.terminate()
    for proc in workers:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


class Detok:
    """Incremental detokenizer: text delta per token, holding back incomplete UTF-8.

    Decodes only the window of ids since the last resolved boundary, not the whole
    reply, so cost is O(1) amortized per push instead of O(reply length). A push whose
    window decode ends in the UTF-8 replacement char withholds output (the window may
    span a multi-byte codepoint, or a BPE piece, split across tokens); once a decode is
    unambiguous the window is closed and the next push starts a fresh one. Because
    `tok.decode` concatenates each token's own bytes and only ever extends an already
    UTF-8-valid prefix with further, independently valid bytes, closing a window at an
    unambiguous decode is exactly the point a from-scratch decode of the whole reply
    would also stop revising past text, so the emitted deltas match a full redecode of
    `ids` on every token.
    """

    def __init__(self, tok) -> None:  # noqa: ANN001
        self.tok, self.ids, self.start = tok, [], 0

    def push(self, token: int) -> str:
        self.ids.append(token)
        text = self.tok.decode(self.ids[self.start :], skip_special_tokens=True)
        if text.endswith("�"):
            return ""
        self.start = len(self.ids)
        return text


class DetokWorker:
    """Runs one request's `Detok` on its own thread, off the scheduler thread.

    `submit` is the `Request.emit` callback the scheduler calls inline from its
    iteration loop (see scheduler.py `_emit`); it only queues the event, an O(1)
    handoff with no tokenizer work, so the scheduler thread returns immediately and
    moves on to the next prefill/decode step. A dedicated worker thread drains the
    queue in submission order, decodes each token, and posts the result to the
    asyncio loop, so decode work for token k overlaps with the GPU step that produces
    token k+1 instead of running inline between them. One worker thread per request
    (bounded by concurrent connections) keeps ordering trivial: a single consumer
    processes its queue strictly FIFO, so output is never dropped, duplicated, or
    reordered relative to `submit` calls. The thread exits after forwarding the
    request's terminal ("end" or "error") event, which the scheduler always
    eventually emits (including on abort), so no thread outlives its request.

    `include_token_id` is for `/v1/completions` (`return_token_ids`): request-factory's
    `openai` backend carries the real generated ids forward as the next round's prompt,
    so it needs the raw id alongside the text delta, not just the detokenized text chat
    streams. Off by default, so `/v1/chat/completions`'s emitted event shape (`("tok",
    text)`) is unchanged.
    """

    def __init__(
        self,
        tok,  # noqa: ANN001
        loop: asyncio.AbstractEventLoop,
        out: asyncio.Queue,
        *,
        include_token_id: bool = False,
    ) -> None:
        self._detok = Detok(tok)
        self._loop = loop
        self._out = out
        self._include_token_id = include_token_id
        self._events: queue.SimpleQueue[tuple] = queue.SimpleQueue()
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, event: tuple) -> None:
        self._events.put(event)

    def _run(self) -> None:
        while True:
            kind, val = self._events.get()
            if kind == "tok":
                text = self._detok.push(val)
                item = ("tok", (val, text)) if self._include_token_id else ("tok", text)
            else:
                item = (kind, val)
            self._loop.call_soon_threadsafe(self._out.put_nowait, item)
            if kind != "tok":
                return


PROBE_TURNS = (
    {"role": "user", "content": "first probe message"},
    {"role": "assistant", "content": "first probe reply"},
    {"role": "user", "content": "second probe message"},
    {"role": "assistant", "content": "second probe reply"},
    {"role": "user", "content": "third probe message"},
)


def probe_prompts(tok) -> list[str]:  # noqa: ANN001
    """Growing multi-turn renders, shortest first, for PromptCache.selfcheck."""
    try:
        return [
            tok.apply_chat_template(
                list(PROBE_TURNS[:k]), tokenize=False, add_generation_prompt=True
            )
            for k in (1, 3, 5)
        ]
    except Exception:  # noqa: BLE001 -- no usable chat template: nothing to check against
        return []


def _history_reuse_messages(engine: Engine, body: dict) -> list[dict] | None:
    """Return messages whose assistant history retains generation-only boilerplate.

    Derive the boilerplate from this tokenizer and this request's template arguments. A
    template whose open render, closed render, or non-empty assistant probe does not have the
    required literal-prefix relationship returns ``None`` and uses the ordinary safe
    suffix-boundary path. Exact cache lookup remains the final correctness check if a
    client's re-tokenized reply differs from the tokens the model generated.
    """
    messages = body["messages"]
    if not CHAT_HISTORY_REUSE or not any(m.get("role") == "assistant" for m in messages):
        return messages if CHAT_HISTORY_REUSE else None
    if any(not isinstance(m.get("content"), str) for m in messages):
        return None
    kwargs = body.get("chat_template_kwargs") or {}
    try:
        closed = engine.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, **kwargs
        )
        opened = engine.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )
        probe = "__SEED_ASSISTANT_CONTENT_PROBE_7f5e2c__"
        with_probe_assistant = engine.tok.apply_chat_template(
            [*messages, {"role": "assistant", "content": probe}],
            tokenize=False,
            add_generation_prompt=False,
            **kwargs,
        )
    except Exception:  # noqa: BLE001 -- fall back to the existing boundary-only cache path
        return None
    if not opened.startswith(closed) or not with_probe_assistant.startswith(closed):
        return None
    generation_suffix = opened[len(closed) :]
    probe_tail = with_probe_assistant[len(closed) :]
    if probe_tail.count(probe) != 1:
        return None
    assistant_suffix = probe_tail[: probe_tail.index(probe)]
    common = 0
    for left, right in zip(generation_suffix, assistant_suffix, strict=False):
        if left != right:
            break
        common += 1
    transient = generation_suffix[common:]
    if common == 0:
        _chat_history_stat(engine, "normalize_no_common_prefix")
        return None
    _chat_history_stat(engine, "normalize_empty_transient" if not transient else "normalize_ok")
    return [
        (
            {**message, "content": transient + message["content"]}
            if message.get("role") == "assistant"
            else message
        )
        for message in messages
    ]


def build_prompt(engine: Engine, body: dict, messages: list[dict] | None = None) -> list[int]:
    kwargs = body.get("chat_template_kwargs") or {}
    text = engine.tok.apply_chat_template(
        messages or body["messages"], tokenize=False, add_generation_prompt=True, **kwargs
    )
    return engine.prompts.encode(text)


def _chat_history_key(messages: list[dict], assistant_text: str, kwargs: dict) -> str:
    """Full, collision-safe identity for one client-visible completed chat turn."""
    return json.dumps(
        [messages, assistant_text, kwargs],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _chat_history_stat(engine: Engine, name: str) -> None:
    """Sparse activation logging for the opt-in path and its safety fallbacks."""
    stats = getattr(engine, "chat_history_stats", None)
    if stats is None:
        stats = engine.chat_history_stats = Counter()
    stats[name] += 1
    count = stats[name]
    if count <= 4 or count & (count - 1) == 0:
        entries = len(getattr(engine, "chat_history", ()))
        print(
            f"[chat-history] {name}={count} entries={entries}",
            flush=True,
        )


def _history_registry_prompt(
    engine: Engine, body: dict, messages: list[dict]
) -> tuple[list[int], str] | None:
    """Splice exact prior generated ids before the next turn's template tail.

    Incremental detokenization is client-correct but concatenating its text and encoding it
    again need not recover the generated token ids. Also, Qwen3.5 renders the assistant being
    generated with a think opener which it deliberately removes when that assistant becomes
    history. Consequently the ordinary next-turn render cannot have ``prior_text + reply`` as
    a prefix. The registry supplies the exact ids through the reply, while a unique marker in
    the real template identifies the tail beginning at the assistant's closing special token.
    Strict render and special-token checks make an unfamiliar template fall back to ordinary
    prompt caching.
    """
    source = body["messages"]
    if len(source) < 3 or source[-2].get("role") != "assistant":
        return None
    assistant_text = source[-2].get("content")
    if not isinstance(assistant_text, str):
        return None
    kwargs = body.get("chat_template_kwargs") or {}
    key = _chat_history_key(source[:-2], assistant_text, kwargs)
    record = engine.chat_history.get(key)
    if record is None:
        _chat_history_stat(engine, "lookup_key_miss")
        return None
    _, prior_prompt, generated = record
    marker = "__SEED_ASSISTANT_CONTENT_PROBE_7f5e2c__"
    if marker in assistant_text or any(
        isinstance(message.get("content"), str) and marker in message["content"]
        for message in messages
    ):
        _chat_history_stat(engine, "tail_marker_collision")
        return None
    marked_messages = [*messages]
    marked_messages[-2] = {**marked_messages[-2], "content": marker}
    try:
        text = engine.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )
        marked = engine.tok.apply_chat_template(
            marked_messages, tokenize=False, add_generation_prompt=True, **kwargs
        )
    except Exception:  # noqa: BLE001 -- fall back to ordinary prompt construction
        _chat_history_stat(engine, "tail_render_error")
        return None
    if marked.count(marker) != 1:
        _chat_history_stat(engine, "tail_marker_miss")
        return None
    rendered_prefix, tail = marked.split(marker)
    # Qwen applies ``trim`` to message content. This equality also rejects content which the
    # template transforms specially (reasoning/tool syntax), rather than splicing the wrong
    # tail after it.
    if text != rendered_prefix + assistant_text.strip() + tail:
        _chat_history_stat(engine, "string_prefix_miss")
        return None
    if not tail or not any(tail.startswith(token) for token in engine.tok.all_special_tokens):
        _chat_history_stat(engine, "tail_not_special")
        return None
    engine.chat_history.move_to_end(key)
    _chat_history_stat(engine, "splice_ok")
    return [*prior_prompt, *generated, *engine.prompts.tokenize(tail)], text


def _find_history_registry_prompt(
    engine: Engine, body: dict, normalized_messages: list[dict] | None
) -> tuple[list[int], str] | None:
    """Try the derived normalization, then the raw template under the same strict checks."""
    candidates = [normalized_messages, body["messages"]]
    for index, messages in enumerate(candidates):
        if messages is None or (index and messages == normalized_messages):
            continue
        result = _history_registry_prompt(engine, body, messages)
        if result is not None:
            if index:
                _chat_history_stat(engine, "raw_template_splice_ok")
            return result
    return None


def _record_chat_history(
    engine: Engine,
    body: dict,
    prompt_text: str,
    prompt: list[int],
    assistant_text: str,
    output_ids: list[int],
    finish_reason: str,
) -> None:
    """Bound the exact-token registry while retaining full keys for collision checks."""
    # With deferred close, a stop finish publishes edge[:-1]: exactly the ids emitted to the
    # client, excluding the hidden sampled stop token. Without it, the close contains the stop
    # id and cannot prefix-match client-visible history.
    reusable_finish = finish_reason == "length" or (finish_reason == "stop" and DEFER_TURN_CLOSE)
    if not reusable_finish:
        _chat_history_stat(engine, f"record_skip_finish_{finish_reason}")
        return
    if not output_ids:
        _chat_history_stat(engine, "record_skip_empty")
        return
    if CHAT_HISTORY_REUSE_ENTRIES <= 0:
        _chat_history_stat(engine, "record_skip_disabled")
        return
    kwargs = body.get("chat_template_kwargs") or {}
    key = _chat_history_key(body["messages"], assistant_text, kwargs)
    engine.chat_history[key] = (prompt_text, tuple(prompt), tuple(output_ids))
    engine.chat_history.move_to_end(key)
    while len(engine.chat_history) > CHAT_HISTORY_REUSE_ENTRIES:
        engine.chat_history.popitem(last=False)
    _chat_history_stat(engine, "record_ok")


def chat_suffix_len(engine: Engine, prompt: list[int], body: dict) -> int:
    """How many of `prompt`'s trailing tokens are this turn's generation-prompt opening
    (role marker plus, in a thinking-capable template, whatever `<think>` boilerplate that
    mode adds) and so should be prefilled but not recorded as part of the reusable prefix
    (see scheduler.py's `Request.suffix_len`).

    Derived fresh from this request's own `messages` and `chat_template_kwargs`, not
    assumed from a fixed template mode: rendering the same messages with the generation
    prompt left off gives exactly the text up through the last message's own close (e.g.
    `<|im_end|>\\n`), and that is the boundary the *next* turn's history reproduces exactly,
    whatever `chat_template_kwargs` that next turn uses and however its `messages` chooses
    to represent this turn's reply (a client's own history never has to match what this
    turn actually generated token-for-token for the boundary itself to still line up,
    because the boundary sits *before* this turn's reply, not after it). One prior mistake
    here: hardcoding an expected mode (e.g. always `enable_thinking=False`) instead of
    reading it from the request, which silently returns 0 the moment a client's actual mode
    disagrees. Verified as a strict token-level prefix of `prompt` before being trusted;
    returns 0 -- the plain end-of-prompt snapshot -- when it is not (a template with no
    add_generation_prompt branch, or one whose closed render is not literally a prefix of
    its open render), which only ever costs a missed reuse, never a wrong one.
    """
    kwargs = body.get("chat_template_kwargs") or {}
    try:
        closed_text = engine.tok.apply_chat_template(
            body["messages"], tokenize=False, add_generation_prompt=False, **kwargs
        )
    except Exception:  # noqa: BLE001 -- no usable chat template: nothing to correct for
        return 0
    closed_ids = engine.prompts.encode(closed_text)
    n = len(closed_ids)
    if not 0 < n < len(prompt) or prompt[:n] != closed_ids:
        return 0
    return len(prompt) - n


def make_request(
    engine: Engine,
    prompt: list[int],
    body: dict,
    out: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    *,
    include_token_id: bool = False,
    suffix_len: int = 0,
) -> Request:
    """Build the scheduler request; its `emit` hands events to a per-request detok worker.

    `include_token_id` is off by default, so a chat request (which never passes it) builds
    almost exactly the `Request` it always has; only `suffix_len` (also 0 by default) is set
    for chat, since it is what fixes chat's own cross-turn reuse (see scheduler.py's module
    docstring). Stage 2 always publishes a turn-close cache entry for every finished request
    (scheduler.py's `Request` no longer takes an opt-in flag for this), so the raw-completions
    exact-repeat reuse Stage 1's `extend_prefix` opted into now happens unconditionally.
    """
    worker = DetokWorker(engine.tok, loop, out, include_token_id=include_token_id)
    return Request(
        prompt=prompt,
        max_new=min(int(body.get("max_tokens") or 256), engine.args.max_seq_len - len(prompt)),
        temperature=float(body.get("temperature") or 0.0),
        stop=frozenset() if body.get("ignore_eos") else engine.stop_ids,
        emit=worker.submit,
        suffix_len=suffix_len,
        has_history=any(m.get("role") == "assistant" for m in body.get("messages") or ()),
    )


def parse_token_prompt(body: dict) -> list[int]:
    """Validate `/v1/completions`'s `prompt` as a token-id array.

    Raises `ValueError` with a client-facing message for anything else: decoded text
    (this endpoint feeds ids straight into the model, skipping the tokenizer/chat
    template so its prefix matches request-factory's own token-id bookkeeping exactly),
    a batch of prompts (we only ever admit one sequence per request), or a missing/empty
    prompt.
    """
    prompt = body.get("prompt")
    if isinstance(prompt, str):
        raise ValueError(
            "prompt must be a token-id array; text prompts are not supported on "
            "/v1/completions (use /v1/chat/completions for text)"
        )
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("prompt must be a non-empty array of token ids")
    if not all(isinstance(t, int) and not isinstance(t, bool) for t in prompt):
        raise ValueError("prompt must be an array of integer token ids, not a batch or floats")
    return prompt


SCORE_ENDPOINT_ENV = "SEED_SCORE_ENDPOINT"
"""`/v1/score` is registered only when this is set to a non-zero value (calibration runs)."""


def parse_score_request(body: dict) -> tuple[list[int], int, int]:
    """Validate `/v1/score`'s body; returns (prompt, continuation_start, top_k).

    `prompt` uses the same token-id-array convention as `/v1/completions`
    (`parse_token_prompt`). `continuation_start` marks where the reference continuation
    begins: positions `[continuation_start, len(prompt))` are the tokens to be scored, so it
    must leave at least one token of left context and at least one token to score.
    """
    prompt = parse_token_prompt(body)
    continuation_start = body.get("continuation_start")
    if not isinstance(continuation_start, int) or isinstance(continuation_start, bool):
        raise ValueError("continuation_start must be an integer")
    if not 0 < continuation_start < len(prompt):
        raise ValueError(
            "continuation_start must satisfy 0 < continuation_start < len(prompt) "
            f"({len(prompt)}), got {continuation_start}"
        )
    top_k = body.get("top_k", 5)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    return prompt, continuation_start, top_k


def score_continuation(
    sched: Scheduler, prompt: list[int], continuation_start: int, top_k: int
) -> list[dict]:
    """Teacher-forced per-position logprobs for `prompt[continuation_start:]`, one forward call.

    Routed through the scheduler thread (`Scheduler.score`): it runs on a free lane between
    iterations, so it never races the scheduler's own runner calls (under TP two threads
    interleaving broadcasts deadlock the ranks), never touches a live lane, and releases the
    lane's KV blocks afterwards. Blocks the calling thread; `score` below calls it from an
    executor. Reuses the prefill path with `logits_from=continuation_start - 1` (see
    `Model.score`), so only the continuation's positions are projected to vocab size.
    """

    def rows(logits: torch.Tensor) -> list[dict]:
        logprobs = torch.log_softmax(logits, dim=-1)
        scores = []
        for offset, token_id in enumerate(prompt[continuation_start:]):
            row = logprobs[offset]
            top_vals, top_ids = row.topk(top_k)
            scores.append(
                {
                    "token_id": token_id,
                    "logprob": row[token_id].item(),
                    "top_logprobs": [
                        {"token_id": tid, "logprob": val}
                        for val, tid in zip(top_vals.tolist(), top_ids.tolist(), strict=True)
                    ],
                }
            )
        return scores

    return sched.score(prompt, continuation_start, rows)


FAULT_FLAGS: dict[str, tuple[str, type]] = {
    "rope_base": ("FAULT_ROPE_BASE", bool),
    "drop_expert": ("FAULT_DROP_EXPERT", bool),
    "skip_deltanet_gate": ("FAULT_SKIP_DELTANET_GATE", bool),
    "fp8_kv_no_scale": ("FAULT_FP8_KV_NO_SCALE", bool),
    "flip_causal_mask": ("FAULT_FLIP_CAUSAL_MASK", bool),
    "flip_causal_mask_layer": ("FAULT_FLIP_CAUSAL_MASK_LAYER", int),
}
"""Maps a `/debug/fault` request name to the `model.py` global it flips, and that global's
type. Kept in server.py rather than model.py: this is calibration-tooling wiring, not a
model-behavior concern (see `debug_fault`)."""


def fault_state() -> dict[str, bool | int]:
    return {name: getattr(model_module, attr) for name, (attr, _) in FAULT_FLAGS.items()}


async def debug_fault(request: web.Request) -> web.Response:
    """Flip a `SEED_FAULT_*` hook at runtime, for a one-boot accuracy-gate calibration sweep
    (clean plus every fault, without restarting the server between configs).

    Only registered when `SEED_FAULT_ENDPOINT=1` is set at server startup (see `make_app`):
    a production launch that omits the env var never exposes the route at all, rather than
    exposing it and trusting every caller to leave it alone.

    Every `SEED_FAULT_*` flag in `model.py` is a module global read fresh at its point of use
    (the RoPE table is the one exception, and its cache is now keyed on the flag's value too;
    see `Model._rope_table`), so a flip here takes effect on the next forward. `/v1/score`
    always runs a fresh forward against a slot `Model.begin` just reset, so there is no
    stale-state window between flipping a fault and the next `/v1/score` call observing it.

    Body: ``{"name": "<fault name>", "value": <bool or int>}`` (``name`` is a key of
    `FAULT_FLAGS`), or ``{"name": "all", "value": false}`` to reset every boolean fault flag
    to its off default in one call (the causal-mask layer index is left as-is; set it again
    explicitly if a non-default layer is wanted).
    """
    body = await request.json()
    name = body.get("name")
    value = body.get("value")
    if name == "all":
        if value is not False:
            return web.json_response(
                {"error": "'all' only supports value=false (reset every boolean flag)"},
                status=400,
            )
        for attr, kind in FAULT_FLAGS.values():
            if kind is bool:
                setattr(model_module, attr, False)
        return web.json_response({"status": "ok", "faults": fault_state()})
    if name not in FAULT_FLAGS:
        return web.json_response(
            {"error": f"unknown fault {name!r}; known: {sorted(FAULT_FLAGS)}, or 'all'"},
            status=400,
        )
    attr, kind = FAULT_FLAGS[name]
    if kind is bool and not isinstance(value, bool):
        return web.json_response({"error": f"{name!r} takes a bool value"}, status=400)
    if kind is int and (not isinstance(value, int) or isinstance(value, bool)):
        return web.json_response({"error": f"{name!r} takes an integer value"}, status=400)
    setattr(model_module, attr, value)
    return web.json_response({"status": "ok", "faults": fault_state()})


async def score(request: web.Request) -> web.Response:
    """`/v1/score`: teacher-forced logprobs for a reference continuation (calibration only).

    Registered only with `SEED_SCORE_ENDPOINT=1`. Safe alongside live traffic: see
    `score_continuation`.
    """
    engine: Engine = request.app["engine"]
    if not engine.ready:
        return web.json_response({"error": "model not ready"}, status=503)
    body = await request.json()
    try:
        prompt, continuation_start, top_k = parse_score_request(body)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    try:
        scores = await asyncio.get_running_loop().run_in_executor(
            None, score_continuation, engine.sched, prompt, continuation_start, top_k
        )
    except RuntimeError as exc:
        return web.json_response({"error": str(exc)}, status=500)
    return web.json_response(
        {
            "id": f"score-{uuid.uuid4().hex}",
            "object": "token_scores",
            "model": body.get("model") or "qwen3.5-397b-a17b-mxfp4",
            "prompt_tokens": len(prompt),
            "continuation_start": continuation_start,
            "scores": scores,
        }
    )


def usage(prompt: list[int], n: int, reused: int = 0) -> dict:
    """OpenAI usage. `cached_tokens` is the prompt prefix a cached slot already held."""
    return {
        "prompt_tokens": len(prompt),
        "completion_tokens": n,
        "total_tokens": len(prompt) + n,
        "prompt_tokens_details": {"cached_tokens": reused},
    }


def chunk(
    rid: str, model: str, delta: dict, finish: str | None = None, usage_: dict | None = None
) -> str:
    obj = {
        "id": rid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
    }
    obj["choices"] = [] if usage_ else [{"index": 0, "delta": delta, "finish_reason": finish}]
    if usage_:
        obj["usage"] = usage_
    return f"data: {json.dumps(obj)}\n\n"


async def chat(request: web.Request) -> web.StreamResponse:
    engine: Engine = request.app["engine"]
    if not engine.ready:
        return web.json_response({"error": "model not ready"}, status=503)
    body = await request.json()
    history_messages = _history_reuse_messages(engine, body)
    normalized = None
    if any(message.get("role") == "assistant" for message in body["messages"]):
        normalized = _find_history_registry_prompt(engine, body, history_messages)
    if normalized is not None:
        prompt, prompt_text = normalized
        history_capture = True
    elif CHAT_HISTORY_REUSE and not any(
        message.get("role") == "assistant" for message in body["messages"]
    ):
        kwargs = body.get("chat_template_kwargs") or {}
        prompt_text = engine.tok.apply_chat_template(
            body["messages"], tokenize=False, add_generation_prompt=True, **kwargs
        )
        prompt = engine.prompts.encode(prompt_text)
        history_capture = True
    else:
        prompt = build_prompt(engine, body)
        prompt_text = ""
        history_capture = False
    if len(prompt) >= engine.args.max_seq_len:
        return web.json_response(
            {"error": f"prompt has {len(prompt)} tokens, max_seq_len is {engine.args.max_seq_len}"},
            status=400,
        )
    out: asyncio.Queue = asyncio.Queue()
    req = make_request(
        engine, prompt, body, out, asyncio.get_running_loop(),
        include_token_id=history_capture,
        suffix_len=0 if history_capture else chat_suffix_len(engine, prompt, body),
    )  # fmt: skip
    engine.submit(req)
    rid, name = f"chatcmpl-{uuid.uuid4().hex}", body.get("model") or "qwen3.5-397b-a17b-mxfp4"
    if body.get("stream"):
        return await stream_reply(
            request,
            out,
            prompt,
            rid,
            name,
            (body.get("stream_options") or {}).get("include_usage", False),
            history=(engine, body, prompt_text) if history_capture else None,
        )
    text, output_ids, kind, val = "", [], "", None
    while kind not in ("end", "error"):
        kind, val = await out.get()
        if kind == "tok":
            token_id, delta = val if history_capture else (None, val)
            text += delta
            if token_id is not None:
                output_ids.append(token_id)
    if kind == "error":
        return web.json_response({"error": val}, status=500)
    reason, n, reused = val
    if history_capture:
        _record_chat_history(engine, body, prompt_text, prompt, text, output_ids, reason)
    msg = {"role": "assistant", "content": text}
    resp = {"id": rid, "object": "chat.completion", "created": int(time.time()), "model": name}
    resp["choices"] = [{"index": 0, "message": msg, "finish_reason": reason}]
    resp["usage"] = usage(prompt, n, reused)
    return web.json_response(resp)


async def stream_reply(  # noqa: ANN001, FBT001
    request, out, prompt, rid, name, include_usage, *, history=None
) -> web.StreamResponse:
    resp = web.StreamResponse(
        headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
    )
    await resp.prepare(request)
    first = True
    text, output_ids = "", []
    while True:
        kind, val = await out.get()
        if kind == "tok":  # one SSE chunk per generated token
            token_id, piece = val if history is not None else (None, val)
            text += piece
            if token_id is not None:
                output_ids.append(token_id)
            delta = {"content": piece} | ({"role": "assistant"} if first else {})
            first = False
            await resp.write(chunk(rid, name, delta).encode())
            continue
        if kind == "end":
            reason, n, reused = val
            if history is not None:
                engine, body, prompt_text = history
                _record_chat_history(engine, body, prompt_text, prompt, text, output_ids, reason)
            await resp.write(chunk(rid, name, {}, finish=reason).encode())
            if include_usage:
                await resp.write(chunk(rid, name, {}, usage_=usage(prompt, n, reused)).encode())
        else:
            await resp.write(f"data: {json.dumps({'error': val})}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        return resp


async def reset_prefix_cache(request: web.Request) -> web.Response:
    """Clear every idle slot's recorded prefix (vLLM's admin-endpoint name).

    For a benchmark that sweeps several independent points against one running server:
    without this, an earlier point's sessions stay in the slot pool as reusable prefixes,
    so a later point can spuriously "reuse" tokens it never actually sent, contaminating
    both TTFT and `cached_tokens`. Never disturbs an in-flight request; see
    `Scheduler.reset_prefix_cache`.
    """
    engine: Engine = request.app["engine"]
    if not engine.ready:
        return web.json_response({"error": "model not ready"}, status=503)
    cleared = engine.sched.reset_prefix_cache()
    engine.chat_history.clear()
    return web.json_response({"status": "ok", "cleared_slots": cleared})


def completion_chunk(
    rid: str,
    model: str,
    *,
    text: str = "",
    token_ids: list[int] | None = None,
    finish: str | None = None,
    usage_: dict | None = None,
) -> str:
    """One `/v1/completions` SSE object: a text-completion chunk, or a trailing usage-only
    one (`choices: []`), matching `chunk()`'s chat-completion shaping below."""
    obj = {"id": rid, "object": "text_completion", "created": int(time.time()), "model": model}
    if usage_ is not None:
        obj["choices"], obj["usage"] = [], usage_
    else:
        choice = {"index": 0, "text": text, "finish_reason": finish}
        if token_ids is not None:
            choice["token_ids"] = token_ids
        obj["choices"] = [choice]
    return f"data: {json.dumps(obj)}\n\n"


async def completions(request: web.Request) -> web.StreamResponse:
    """OpenAI-style `/v1/completions`: a raw token-id prompt, fed into the same
    scheduler/model path `/v1/chat/completions` uses after its own tokenization, so
    prefix matching, batching, graph capture, and `cached_tokens` accounting are shared.
    Built for request-factory's `openai` backend, whose rounds resend a prior round's prompt
    plus its own real output ids (see scheduler.py's module docstring, "Dropped from Stage 1":
    every finished request, on any endpoint, now publishes a turn-close cache entry
    unconditionally, so this exact-repeat reuse needs no opt-in here any more).
    """
    engine: Engine = request.app["engine"]
    if not engine.ready:
        return web.json_response({"error": "model not ready"}, status=503)
    body = await request.json()
    try:
        prompt = parse_token_prompt(body)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    if len(prompt) >= engine.args.max_seq_len:
        return web.json_response(
            {"error": f"prompt has {len(prompt)} tokens, max_seq_len is {engine.args.max_seq_len}"},
            status=400,
        )
    out: asyncio.Queue = asyncio.Queue()
    include_ids = bool(body.get("return_token_ids"))
    engine.submit(
        make_request(
            engine,
            prompt,
            body,
            out,
            asyncio.get_running_loop(),
            include_token_id=include_ids,
        )
    )
    rid, name = f"cmpl-{uuid.uuid4().hex}", body.get("model") or "qwen3.5-397b-a17b-mxfp4"
    if body.get("stream"):
        return await stream_completions_reply(
            request,
            out,
            prompt,
            rid,
            name,
            (body.get("stream_options") or {}).get("include_usage", False),
            include_ids=include_ids,
        )
    text, ids, kind, val = "", [], "", None
    while kind not in ("end", "error"):
        kind, val = await out.get()
        if kind != "tok":
            continue
        if include_ids:
            token_id, delta = val
            ids.append(token_id)
        else:
            delta = val
        text += delta
    if kind == "error":
        return web.json_response({"error": val}, status=500)
    reason, n, reused = val
    choice = {"index": 0, "text": text, "finish_reason": reason}
    if include_ids:
        choice["token_ids"] = ids
    resp = {
        "id": rid,
        "object": "text_completion",
        "created": int(time.time()),
        "model": name,
        "choices": [choice],
        "usage": usage(prompt, n, reused),
    }
    return web.json_response(resp)


async def stream_completions_reply(  # noqa: ANN001
    request,
    out,
    prompt,
    rid,
    name,
    include_usage,
    *,
    include_ids: bool,  # noqa: FBT001
) -> web.StreamResponse:
    resp = web.StreamResponse(
        headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
    )
    await resp.prepare(request)
    while True:
        kind, val = await out.get()
        if kind == "tok":  # one SSE chunk per generated token, never a resend of prior ids
            token_id, delta = val if include_ids else (None, val)
            ids = [token_id] if include_ids else None
            await resp.write(completion_chunk(rid, name, text=delta, token_ids=ids).encode())
            continue
        if kind == "end":
            reason, n, reused = val
            await resp.write(completion_chunk(rid, name, finish=reason).encode())
            if include_usage:
                await resp.write(
                    completion_chunk(rid, name, usage_=usage(prompt, n, reused)).encode()
                )
        else:
            await resp.write(f"data: {json.dumps({'error': val})}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        return resp


async def health(request: web.Request) -> web.Response:
    engine: Engine = request.app["engine"]
    dead = engine.dead_worker()
    if engine.ready and not dead:
        return web.json_response({"status": "ok"})
    return web.json_response({"status": "loading", "error": engine.error or dead}, status=503)


async def models(_request: web.Request) -> web.Response:
    return web.json_response(
        {"object": "list", "data": [{"id": "qwen3.5-397b-a17b-mxfp4", "object": "model"}]}
    )


def make_app(
    args: argparse.Namespace, workers: list[subprocess.Popen] | None = None
) -> web.Application:
    app = web.Application()
    app["engine"] = Engine(args, workers)
    app.add_routes(
        [
            web.get("/health", health),
            web.get("/v1/models", models),
            web.post("/v1/chat/completions", chat),
            web.post("/v1/completions", completions),
            web.post("/reset_prefix_cache", reset_prefix_cache),
        ]
    )
    if os.environ.get(SCORE_ENDPOINT_ENV, "0") != "0":
        app.add_routes([web.post("/v1/score", score)])
    # Gated by env var rather than always registered: this route mutates global model
    # behavior for every subsequent request, so it must not exist on a server unless
    # whoever launched it explicitly asked for it (calibration runs only).
    if os.environ.get("SEED_FAULT_ENDPOINT", "0") != "0":
        app.add_routes([web.post("/debug/fault", debug_fault)])
    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=os.environ.get("MODEL_PATH"))
    p.add_argument("--host", default="0.0.0.0")  # noqa: S104
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--tp", type=int, default=0, help="tensor-parallel ranks (default: one per GPU, else 1)"
    )
    p.add_argument("--rank", type=int, default=0, help="internal: this process's rank")
    p.add_argument("--tp-port", type=int, default=tp.DEFAULT_PORT, help="rendezvous port")
    p.add_argument(
        "--devices",
        default="",
        help="comma list, e.g. cuda:0,cuda:1 (default: all GPUs, else cpu). Only read at "
        "--tp 1, where it is the single process's layer split; a tensor-parallel rank owns "
        "cuda:RANK",
    )
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--max-seq-len", type=int, default=16384)
    p.add_argument(
        "--max-batch",
        type=int,
        default=48,
        help="sequence slots: the in-flight batch cap and the prefix-cache size "
        "(one per workload session); see the memory math in model.py",
    )
    p.add_argument(
        "--enable-graph-capture",
        action="store_true",
        help="capture a HIP/CUDA graph for the fixed-shape batched decode step and replay "
        "it instead of dispatching the step from Python (default off; falls back to eager "
        "execution if capture fails or the captured step disagrees with it)",
    )
    args = p.parse_args(argv)
    if not args.model_path:
        p.error("--model-path or MODEL_PATH is required")
    if args.max_batch < 1:
        p.error(f"--max-batch must be at least 1, got {args.max_batch}")
    args.tp = args.tp or tp.default_world()
    if not 0 <= args.rank < args.tp:
        p.error(f"--rank {args.rank} is not a rank of a {args.tp}-rank group")
    return args


def main(argv: list[str] | None = None) -> None:
    # Before any host allocation, so weight-read buffers and page cache follow the policy.
    placement = host_numa.apply()
    a = parse_args(argv)
    if placement:
        print(f"[host-numa] rank {a.rank}: {placement}", flush=True)
    if a.rank:
        run_worker(a)
        return
    mem_timeline.preflight()  # before any rank allocates; SEED_BOOT_MEM_PREFLIGHT_GIB
    workers = start_workers(a)
    try:
        web.run_app(make_app(a, workers), host=a.host, port=a.port)
    finally:
        stop_workers(workers)


if __name__ == "__main__":
    main()

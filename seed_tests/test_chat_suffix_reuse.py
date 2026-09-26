"""CPU end-to-end test for the chat-suffix prefix-reuse fix, against the real `server.py`.

Qwen3.5's chat template appends a generation-prompt opening after the last message's own
close: a role marker plus, in a thinking-capable template, a `<think>` block that is either
left open (the default, thinking-on mode: the model closes it itself while generating) or
immediately closed (`chat_template_kwargs.enable_thinking = False`). Neither ever appears in
a *later* prompt's history: history only ever replays the messages themselves, however a
client chooses to represent a finished turn's reply (including, in thinking-on mode, with
the model's own reasoning stripped out and only the final content kept). A slot's recorded
prefix has to stop before that opening, or it is never a prefix of the next turn's prompt at
all and every turn re-prefills the whole conversation from scratch.

`chat_suffix_len` derives that boundary fresh from each request's own `messages` and
`chat_template_kwargs` (see its docstring in server.py) rather than assuming a fixed mode,
which is what an earlier version of this fix got wrong: it hardcoded `enable_thinking:
False` at startup, so real traffic -- which sends no `chat_template_kwargs` and so gets the
template's default thinking-on branch -- never matched and reuse silently stayed at zero.

This test's fake chat template models both branches (an `enable_thinking` conditional, like
Qwen3.5's own) on the tiny random checkpoint, so the fix is checked against the real
tokenizer/template/scheduler path in both modes, not just the scheduler in isolation (see
test_scheduler.py's own suffix-boundary section for that).

    /tmp/torchenv/bin/python -m pytest \\
        examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/seed_tests/test_chat_suffix_reuse.py
"""

import asyncio
import http.client
import json
import os
import socket
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prompt_cache import PromptCache  # noqa: E402
from test_seed_parity import VOCAB, build_hf, post, wait_healthy, write_checkpoint  # noqa: E402

import server as server_module  # noqa: E402

WORDS = (
    "hello", "world", "next",  # test 1
    "greet", "planet", "onward",  # test 2
    "aa1", "aa2", "aa3",  # test 3, session A
    "bb1", "bb2", "bb3",  # test 3, session B
    "cc1", "cc2",  # test 3, throwaway session used only to harvest a real reply
    "openthink", "closethink", "answer",
    "reuse1", "reuse2", "reuse3",
)  # fmt: skip
# Each test (and, in test 3, each of its two independent sessions) uses its own words, so a
# slot the module-scoped server (only 2 slots) keeps from an earlier session can never
# validly prefix-match a later, unrelated one.
MAX_TOKENS = 5


def build_tokenizer(path: Path, *, canonical_stream: bool = False) -> None:
    """A template with an `enable_thinking` branch, like Qwen3.5's own: the generation
    prompt appends a role marker plus "openthink" (thinking on, the default when a client
    sends no `chat_template_kwargs` -- real traffic's actual shape) or "openthink closethink"
    (thinking off). Either way, a completed assistant turn renders in later history as a
    plain message: the for-loop branch below never emits either think token, mirroring how
    Qwen3.5 strips reasoning from history regardless of mode.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    plain = ["<unk>", "<|im_end|>", "<|im_start|>", "user", "assistant", "system", *WORDS]
    if canonical_stream:
        special, words = plain[:3], plain[3:]
        names = [*special, *(f"▁{word}" for word in words)]
        filler = [f"▁w{i}" for i in range(VOCAB - len(names))]
    else:
        names = plain
        filler = [f"w{i}" for i in range(VOCAB - len(names))]
    vocab = {word: i for i, word in enumerate([*names, *filler])}
    core = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))  # noqa: S106
    # Every ordinary token decodes with its own leading space, so concatenating the server's
    # one-token SSE deltas re-encodes to the generated ids. This is the history-reuse contract
    # the real Qwen byte-level tokenizer normally provides, made explicit in the tiny fixture.
    if canonical_stream:
        core.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
        core.decoder = decoders.Replace("▁", " ")
    else:
        core.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    template = (
        "{% for m in messages %}<|im_start|> {{ m.role }} {{ m.content }} <|im_end|> {% endfor %}"
        "{% if add_generation_prompt %}<|im_start|> assistant "
        "{% if enable_thinking is not defined or enable_thinking %}"
        "openthink "
        "{% else %}"
        "openthink closethink "
        "{% endif %}"
        "{% endif %}"
    )
    PreTrainedTokenizerFast(
        tokenizer_object=core,
        unk_token="<unk>",  # noqa: S106
        eos_token="<|im_end|>",  # noqa: S106
        additional_special_tokens=["<|im_start|>"] if canonical_stream else None,
        chat_template=template,
    ).save_pretrained(path)


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory):
    out = tmp_path_factory.mktemp("srv")
    write_checkpoint(build_hf(), out, mxfp4=False)
    build_tokenizer(out)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    cmd = [sys.executable, str(ROOT / "server.py"), "--model-path", str(out),
           "--host", "127.0.0.1", "--port", str(port), "--tp", "1", "--devices", "cpu",
           "--dtype", "float32", "--max-seq-len", "96", "--max-batch", "2"]  # fmt: skip
    proc = subprocess.Popen(cmd, cwd=ROOT)  # noqa: S603
    try:
        wait_healthy(port)
        yield port
    finally:
        proc.terminate()
        proc.wait(timeout=30)


@pytest.fixture(scope="module")
def history_server(tmp_path_factory: pytest.TempPathFactory):
    """Production-shaped full-history reuse: deferred close plus the opt-in flag."""
    out = tmp_path_factory.mktemp("history-srv")
    write_checkpoint(build_hf(), out, mxfp4=False)
    build_tokenizer(out, canonical_stream=True)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    cmd = [sys.executable, str(ROOT / "server.py"), "--model-path", str(out),
           "--host", "127.0.0.1", "--port", str(port), "--tp", "1", "--devices", "cpu",
           "--dtype", "float32", "--max-seq-len", "96", "--max-batch", "8"]  # fmt: skip
    env = os.environ | {"SEED_CHAT_HISTORY_REUSE": "1", "SEED_DEFER_TURN_CLOSE": "1"}
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env)  # noqa: S603
    try:
        wait_healthy(port)
        from transformers import AutoTokenizer

        yield port, AutoTokenizer.from_pretrained(out)
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def turn(user: str, prior: list[dict] | None = None) -> list[dict]:
    return [*(prior or []), {"role": "user", "content": user}]


def body(messages: list[dict], chat_template_kwargs: dict | None = None) -> dict:
    b = {
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if chat_template_kwargs is not None:
        b["chat_template_kwargs"] = chat_template_kwargs
    return b


def stream(port: int, messages: list[dict], kwargs: dict | None = None) -> tuple[str, dict]:
    """Send one streamed turn; returns (text, usage)."""
    status, data = post(port, body(messages, kwargs))
    assert status == 200, data[:300]
    events = [line[6:] for line in data.decode().splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    text = "".join(
        c["choices"][0]["delta"].get("content") or ""
        for c in chunks
        if c.get("choices") and "delta" in c["choices"][0]
    )
    return text, chunks[-1]["usage"]


def reset(port: int) -> None:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    conn.request("POST", "/reset_prefix_cache")
    response = conn.getresponse()
    data = response.read()
    assert response.status == 200, data[:300]


def raw_stream(port: int, prompt: list[int]) -> str:
    request_body = {
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
    }
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    conn.request(
        "POST",
        "/v1/completions",
        json.dumps(request_body),
        {"Content-Type": "application/json"},
    )
    response = conn.getresponse()
    data = response.read()
    assert response.status == 200, data[:300]
    events = [line[6:] for line in data.decode().splitlines() if line.startswith("data: ")]
    chunks = [json.loads(event) for event in events[:-1]]
    return "".join(chunk["choices"][0].get("text") or "" for chunk in chunks)


@pytest.mark.parametrize("kwargs", [None, {"enable_thinking": False}])
def test_history_normalization_preserves_the_generation_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kwargs: dict | None
) -> None:
    """The normalized history begins with the exact prior generation prompt in both of
    Qwen's thinking modes. This token-position identity is what makes decode state reusable.
    """
    from transformers import AutoTokenizer

    build_tokenizer(tmp_path)
    tok = AutoTokenizer.from_pretrained(tmp_path)
    engine = type("Engine", (), {"tok": tok})()
    monkeypatch.setattr(server_module, "CHAT_HISTORY_REUSE", True)
    template_kwargs = kwargs or {}
    first = turn("hello")
    opened = tok.apply_chat_template(
        first, tokenize=False, add_generation_prompt=True, **template_kwargs
    )
    second = turn("next", [*first, {"role": "assistant", "content": "answer"}])
    body_ = {"messages": second}
    if kwargs is not None:
        body_["chat_template_kwargs"] = kwargs
    normalized = server_module._history_reuse_messages(engine, body_)
    assert normalized is not None
    rendered = tok.apply_chat_template(
        normalized, tokenize=False, add_generation_prompt=True, **template_kwargs
    )
    assert rendered.startswith(opened + "answer")


def test_default_kwargs_thinking_on_reuses_the_boundary(server: int) -> None:
    """The bug this second fix closes: real traffic sends no `chat_template_kwargs` at all,
    which is the template's thinking-on default (`<think>` left open, "openthink" here) --
    a different generation-prompt shape than the hardcoded `enable_thinking: False` the
    first fix assumed, which is exactly why that fix was a no-op on real traffic."""
    first = turn("hello world")
    reply, first_use = stream(server, first)
    assert first_use["prompt_tokens_details"]["cached_tokens"] == 0, "nothing to reuse yet"

    second = turn("next", [*first, {"role": "assistant", "content": reply}])
    _, use = stream(server, second)

    boundary_len = first_use["prompt_tokens"] - 3  # minus "<|im_start|>", "assistant", "openthink"
    assert use["prompt_tokens_details"]["cached_tokens"] == boundary_len, (
        "did not reuse exactly up to the boundary in thinking-on (default) mode"
    )


def test_explicit_thinking_off_kwargs_reuses_the_boundary(server: int) -> None:
    kwargs = {"enable_thinking": False}
    first = turn("greet planet")
    reply, first_use = stream(server, first, kwargs)
    assert first_use["prompt_tokens_details"]["cached_tokens"] == 0

    second = turn("onward", [*first, {"role": "assistant", "content": reply}])
    _, use = stream(server, second, kwargs)

    # minus "<|im_start|>", "assistant", "openthink", "closethink"
    boundary_len = first_use["prompt_tokens"] - 4
    assert use["prompt_tokens_details"]["cached_tokens"] == boundary_len, (
        "did not reuse exactly up to the boundary in thinking-off mode"
    )


def test_reuse_holds_even_when_history_drops_the_prior_reasoning(server: int) -> None:
    """The boundary sits *before* the assistant's reply entirely (right after the last
    user turn's own close), so it must not care what a client puts there: a client that
    keeps only a summary of the model's real output -- dropping reasoning content, the
    thinking-on norm -- gets the same reuse as one that stores the reply verbatim. Run as
    two independent sessions (a single shared slot only ever remembers one prefix at a
    time, unrelated to this fix) rather than two continuations of the same first turn, so
    the second session's own admission cannot evict the first's recorded boundary first."""

    def second_turn(user1: str, user2: str, second_content: str) -> tuple[int, int]:
        first = turn(user1)
        _, first_use = stream(server, first)
        second = turn(user2, [*first, {"role": "assistant", "content": second_content}])
        _, use = stream(server, second)
        boundary_len = first_use["prompt_tokens"] - 3  # "<|im_start|>", "assistant", "openthink"
        return use["prompt_tokens_details"]["cached_tokens"], boundary_len

    reply_a, _ = stream(server, turn("cc1 cc2"))  # a real reply, discarded here, reused below
    verbatim_cached, verbatim_boundary = second_turn("aa1 aa2", "aa3", reply_a)
    # A stand-in for "reasoning discarded, only the final answer kept": necessarily a
    # different reply than session A's own, real (random-model) output.
    stripped_cached, stripped_boundary = second_turn("bb1 bb2", "bb3", "answer")
    assert "answer" != reply_a, "test is meaningless if the two replies happen to coincide"

    assert verbatim_boundary == stripped_boundary, "the two sessions' prompts were not comparable"
    assert verbatim_cached == verbatim_boundary
    assert stripped_cached == stripped_boundary, "reuse depended on the reply's own content"


def test_opt_in_reuses_generated_history_and_matches_cold_prefill(
    history_server: tuple[int, object], monkeypatch
) -> None:
    """Each cached turn reuses the prior prompt and generated reply state. Replaying the same
    normalized prompt after a cache reset must produce the same greedy output, which checks
    the loaded recurrent state and KV history against a full prefill over two continuations.
    """
    port, tok = history_server
    monkeypatch.setattr(server_module, "CHAT_HISTORY_REUSE", True)
    kwargs = {"enable_thinking": False}
    first = turn("reuse1")
    reply1, use1 = stream(port, first, kwargs)
    assert use1["prompt_tokens_details"]["cached_tokens"] == 0

    second = turn("reuse2", [*first, {"role": "assistant", "content": reply1}])
    reply2, use2 = stream(port, second, kwargs)
    assert use2["prompt_tokens_details"]["cached_tokens"] >= (
        use1["prompt_tokens"] + MAX_TOKENS - 1
    ), "turn 2 did not reuse the first generated reply"

    third = turn("reuse3", [*second, {"role": "assistant", "content": reply2}])
    reply3, use3 = stream(port, third, kwargs)
    assert use3["prompt_tokens_details"]["cached_tokens"] >= (
        use2["prompt_tokens"] + MAX_TOKENS - 1
    ), "turn 3 did not reuse the second generated reply"

    fake_engine = SimpleNamespace(tok=tok)
    body2 = body(second, kwargs)
    normalized2 = server_module._history_reuse_messages(fake_engine, body2)
    prompt2 = tok(
        tok.apply_chat_template(normalized2, tokenize=False, add_generation_prompt=True, **kwargs),
        add_special_tokens=False,
    )["input_ids"]
    reset(port)
    cold_reply2 = raw_stream(port, prompt2)
    assert cold_reply2 == reply2

    body3 = body(third, kwargs)
    normalized3 = server_module._history_reuse_messages(fake_engine, body3)
    prompt3 = tok(
        tok.apply_chat_template(normalized3, tokenize=False, add_generation_prompt=True, **kwargs),
        add_special_tokens=False,
    )["input_ids"]
    reset(port)
    cold_reply3 = raw_stream(port, prompt3)
    assert cold_reply3 == reply3


def test_exact_generated_ids_are_spliced_recursively(tmp_path: Path, monkeypatch) -> None:
    """Turn 2 and turn 3 retain model ids even when streamed text does not re-encode."""
    from transformers import AutoTokenizer

    build_tokenizer(tmp_path)  # isolated WordLevel decode is intentionally non-canonical
    tok = AutoTokenizer.from_pretrained(tmp_path)
    engine = SimpleNamespace(
        tok=tok, prompts=PromptCache(tok, capacity=8), chat_history=OrderedDict()
    )
    monkeypatch.setattr(server_module, "CHAT_HISTORY_REUSE", True)

    first = turn("reuse1")
    body1 = body(first, {"enable_thinking": False})
    kwargs = body1["chat_template_kwargs"]
    text1 = tok.apply_chat_template(first, tokenize=False, add_generation_prompt=True, **kwargs)
    prompt1 = engine.prompts.encode(text1)
    ids1 = [tok.convert_tokens_to_ids("w1"), tok.convert_tokens_to_ids("w2")]
    reply1 = "".join(tok.decode([token], skip_special_tokens=True) for token in ids1)
    assert engine.prompts.encode(reply1) != ids1
    server_module._record_chat_history(engine, body1, text1, prompt1, reply1, ids1, "length")

    second = turn("reuse2", [*first, {"role": "assistant", "content": reply1}])
    body2 = body(second, kwargs)
    normalized2 = server_module._history_reuse_messages(engine, body2)
    prompt2, text2 = server_module._history_registry_prompt(engine, body2, normalized2)
    assert prompt2[: len(prompt1) + len(ids1)] == [*prompt1, *ids1]

    ids2 = [tok.convert_tokens_to_ids("w3"), tok.convert_tokens_to_ids("w4")]
    reply2 = "".join(tok.decode([token], skip_special_tokens=True) for token in ids2)
    server_module._record_chat_history(engine, body2, text2, prompt2, reply2, ids2, "length")
    third = turn("reuse3", [*second, {"role": "assistant", "content": reply2}])
    body3 = body(third, kwargs)
    normalized3 = server_module._history_reuse_messages(engine, body3)
    prompt3, _ = server_module._history_registry_prompt(engine, body3, normalized3)
    assert prompt3[: len(prompt2) + len(ids2)] == [*prompt2, *ids2]


def test_qwen_history_branch_splices_after_exact_generated_ids(monkeypatch) -> None:
    """Qwen strips the live think opener from history, so extend at the close marker."""

    class QwenTemplate:
        all_special_tokens = ("</a>", "</u>")

        @staticmethod
        def apply_chat_template(messages, *, tokenize, add_generation_prompt, **_kwargs):
            assert not tokenize
            last_user = max(i for i, message in enumerate(messages) if message["role"] == "user")
            rendered = ""
            for i, message in enumerate(messages):
                content = message["content"].strip()
                if message["role"] == "user":
                    rendered += f"<u>{content}</u>"
                    continue
                # This models Qwen3.5's loop.index0 > last_query_index branch. It also
                # models the reasoning parser which defeats attempts to inject the think
                # block by prepending it to a historical assistant's content.
                if "</think>" in content:
                    content = content.split("</think>")[-1].lstrip("\n")
                opener = "<a><think>\n\n</think>\n\n" if i > last_user else "<a>"
                rendered += opener + content + "</a>"
            if add_generation_prompt:
                rendered += "<a><think>\n\n</think>\n\n"
            return rendered

    class Prompts:
        @staticmethod
        def encode(text):
            return [ord(char) for char in text]

        tokenize = encode

    tok = QwenTemplate()
    engine = SimpleNamespace(tok=tok, prompts=Prompts(), chat_history=OrderedDict())
    monkeypatch.setattr(server_module, "CHAT_HISTORY_REUSE", True)

    first = turn("reuse1")
    body1 = body(first, {"enable_thinking": False})
    text1 = tok.apply_chat_template(first, tokenize=False, add_generation_prompt=True)
    prompt1 = engine.prompts.encode(text1)
    ids1 = [301, 302]
    reply1 = "answer1"
    server_module._record_chat_history(engine, body1, text1, prompt1, reply1, ids1, "length")

    second = turn("reuse2", [*first, {"role": "assistant", "content": reply1}])
    body2 = body(second, {"enable_thinking": False})
    normalized2 = server_module._history_reuse_messages(engine, body2)
    assert normalized2 == second, "the final-assistant probe falsely sees no missing opener"
    prompt2, text2 = server_module._history_registry_prompt(engine, body2, normalized2)
    tail2 = "</a><u>reuse2</u><a><think>\n\n</think>\n\n"
    assert prompt2 == [*prompt1, *ids1, *engine.prompts.encode(tail2)]

    ids2 = [401, 402]
    reply2 = "answer2"
    server_module._record_chat_history(engine, body2, text2, prompt2, reply2, ids2, "length")
    third = turn("reuse3", [*second, {"role": "assistant", "content": reply2}])
    body3 = body(third, {"enable_thinking": False})
    normalized3 = server_module._history_reuse_messages(engine, body3)
    prompt3, _ = server_module._history_registry_prompt(engine, body3, normalized3)
    assert prompt3[: len(prompt2) + len(ids2)] == [*prompt2, *ids2]


def test_deferred_stop_completion_splices_visible_ids(tmp_path: Path, monkeypatch) -> None:
    """Deferred close omits the hidden stop id, leaving the visible reply reusable."""
    from transformers import AutoTokenizer

    build_tokenizer(tmp_path)
    tok = AutoTokenizer.from_pretrained(tmp_path)
    engine = SimpleNamespace(
        tok=tok, prompts=PromptCache(tok, capacity=8), chat_history=OrderedDict()
    )
    monkeypatch.setattr(server_module, "CHAT_HISTORY_REUSE", True)
    monkeypatch.setattr(server_module, "DEFER_TURN_CLOSE", True)
    first = turn("reuse1")
    body1 = body(first)
    text = tok.apply_chat_template(first, tokenize=False, add_generation_prompt=True)
    prompt = engine.prompts.encode(text)
    visible_ids = [tok.convert_tokens_to_ids("w7"), tok.convert_tokens_to_ids("w8")]
    reply = "".join(tok.decode([token], skip_special_tokens=True) for token in visible_ids)
    server_module._record_chat_history(engine, body1, text, prompt, reply, visible_ids, "stop")
    second = turn("reuse2", [*first, {"role": "assistant", "content": reply}])
    body2 = body(second)
    normalized = server_module._history_reuse_messages(engine, body2)
    prompt2, _ = server_module._history_registry_prompt(engine, body2, normalized)
    assert prompt2[: len(prompt) + len(visible_ids)] == [*prompt, *visible_ids]


def test_empty_transient_is_a_valid_normalization(monkeypatch) -> None:
    """Some templates already render history with the complete generation opening."""

    class Tokenizer:
        @staticmethod
        def apply_chat_template(messages, *, tokenize, add_generation_prompt, **_kwargs):
            assert not tokenize
            text = "".join(f"<{message['role']}>{message['content']}<end>" for message in messages)
            return text + ("<assistant>" if add_generation_prompt else "")

    engine = SimpleNamespace(tok=Tokenizer(), chat_history=OrderedDict())
    monkeypatch.setattr(server_module, "CHAT_HISTORY_REUSE", True)
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "next"},
    ]
    assert server_module._history_reuse_messages(engine, {"messages": messages}) == messages
    assert engine.chat_history_stats["normalize_empty_transient"] == 1


def test_registry_lookup_tries_raw_messages_when_normalization_fails(monkeypatch) -> None:
    messages = [{"role": "assistant", "content": "reply"}]
    body_ = {"messages": messages}
    seen = []

    def lookup(_engine, _body, candidate):
        seen.append(candidate)
        return ([1, 2], "rendered")

    monkeypatch.setattr(server_module, "_history_registry_prompt", lookup)
    assert server_module._find_history_registry_prompt(SimpleNamespace(), body_, None) == (
        [1, 2],
        "rendered",
    )
    assert seen == [messages]


def test_reset_clears_generated_history_registry() -> None:
    class Sched:
        def reset_prefix_cache(self):
            return 3

    engine = SimpleNamespace(ready=True, sched=Sched(), chat_history=OrderedDict([("key", ())]))
    response = asyncio.run(
        server_module.reset_prefix_cache(SimpleNamespace(app={"engine": engine}))
    )
    assert response.status == 200
    assert not engine.chat_history


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

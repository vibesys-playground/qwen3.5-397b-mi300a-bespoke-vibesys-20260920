"""CPU tests for `SEED_DEFER_TURN_CLOSE` (scheduler.py `DEFER_TURN_CLOSE`): a finishing
request publishes its turn-close state without an eager forward on the decode critical path,
and streamed tokens, finish events and the published cache state stay exact.

    <python-with-torch> -m pytest seed_tests/test_defer_turn_close.py -q -o addopts=
"""

import functools
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scheduler as scheduler_mod  # noqa: E402
import test_overlap_sched as overlap_tests  # noqa: E402
from scheduler import Request, Scheduler  # noqa: E402
from session_cache import SessionCache  # noqa: E402
from test_scheduler import Sink, drain, serial  # noqa: E402


def test_flag_defaults_to_off() -> None:
    assert scheduler_mod.DEFER_TURN_CLOSE is False


# ---------------------------------------------------------------- fake runner


@pytest.mark.parametrize("seed", range(150))
@pytest.mark.parametrize("overlap", [False, True], ids=["serial", "overlap"])
def test_randomized_interleavings_stay_exact_with_deferral(
    seed: int, overlap: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """test_overlap_sched's randomized workloads (follow-up turns, stop tokens, chat suffixes,
    lane and pool pressure) with deferral on: same tokens as the serial reference, every
    published node holds exactly its own path's state and logits, nothing leaks."""
    w = overlap_tests.workload(seed)
    monkeypatch.setattr(
        overlap_tests, "Scheduler", functools.partial(Scheduler, defer_turn_close=True)
    )
    sched, runner, cache, allocator, sinks = overlap_tests.play(w, overlap=overlap)
    assert sched.defer_turn_close
    for i, (req, sink) in enumerate(zip(w["reqs"], sinks, strict=True)):
        prompt, max_new, stop = req[0], req[1], req[2]
        if sink.error is not None:
            assert "block pool exhausted" in sink.error, sink.error
            continue
        assert sink.tokens == [t for t in serial(prompt, max_new, stop) if t not in stop], i
    overlap_tests.assert_cache_is_consistent(cache, runner)
    assert sorted(sched.free_lanes) == list(range(w["max_batch"]))
    cache.evict_all()
    assert allocator.free_count == allocator.usable_blocks


def make(defer: bool, overlap: bool = False):
    sched, runner, cache, _ = overlap_tests.make(
        overlap, max_batch=2, prefill_chunk=8, defer_turn_close=defer
    )
    return sched, runner, cache


def run(sched: Scheduler, prompt: list[int], max_new: int, stop=frozenset(), suffix_len=0):  # noqa: ANN001, ANN201
    sink = Sink()
    sched.submit(Request(list(prompt), max_new, 0.0, frozenset(stop), sink, suffix_len))
    drain(sched)
    assert sink.error is None, sink.error
    return sink


@pytest.mark.parametrize("overlap", [False, True], ids=["serial", "overlap"])
@pytest.mark.parametrize("suffix_len", [0, 3])
def test_turn_close_costs_no_forward(overlap: bool, suffix_len: int) -> None:
    """The point of the flag: a finished request forwards nothing after its last decode step
    (off, it re-forwards one token, or the whole reply when `suffix_len > 0`)."""
    prompt = [1, 2, 3, 4, 990, 991, 992]
    for defer in (False, True):
        sched, runner, _ = make(defer, overlap)
        sink = run(sched, prompt, 5, suffix_len=suffix_len)
        assert sink.tokens == serial(prompt, 5)
        extra = runner.prefilled - len(prompt)
        if defer:
            assert extra == 0
        else:
            assert extra == (5 if suffix_len else 1)


def test_zero_suffix_follow_up_resumes_one_token_short_of_the_reply() -> None:
    """Raw completions: the node sits at prompt + reply[:-1]; a follow-up that extends the
    reply prefills the missing closing token together with its own new tokens."""
    for stop in (frozenset(), None):
        sched, runner, cache = make(True)
        prompt = [5, 6, 7]
        if stop is None:
            stop = frozenset({serial(prompt, 6)[3]})
        first = run(sched, prompt, 6, stop)
        reply = serial(prompt, 6, stop)
        node = cache.lookup(prompt + reply)
        assert node.depth == len(prompt) + len(reply) - 1
        follow = prompt + reply + [11, 12]
        before = runner.prefilled
        second = run(sched, follow, 4)
        assert second.tokens == serial(follow, 4)
        assert second.end[2] == len(prompt) + len(reply) - 1
        assert runner.prefilled - before == 3  # closing token + two new ones
        assert first.end[0] == ("stop" if stop else "length")


def test_chat_follow_up_resumes_from_the_turn_open_boundary_either_way() -> None:
    """A chat template renders the prior reply behind a role marker, so the next prompt leaves
    the reply-after-boundary edge at its first token: the eager turn-close node is never
    matched, and both modes resume from the turn-open boundary with the same output."""
    boundary, suffix, marker = [1, 2, 3, 4], [990, 991, 992], [980, 981]
    results = {}
    for defer in (False, True):
        sched, runner, _ = make(defer)
        first = run(sched, boundary + suffix, 4, suffix_len=len(suffix))
        turn2 = boundary + marker + first.tokens + [50, 51] + suffix
        second = run(sched, turn2, 4, suffix_len=len(suffix))
        assert second.tokens == serial(turn2, 4)
        results[defer] = second.end
    assert results[True] == results[False]
    assert results[True][2] == len(boundary)


# ---------------------------------------------------------------- the tiny real model

torch = pytest.importorskip("torch")

from test_batched_decode import build, prompt_of  # noqa: E402
from test_seed_parity import build_hf, write_checkpoint  # noqa: E402


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("tiny-defer")
    write_checkpoint(build_hf(), out, mxfp4=False)
    return out


def sessions(ref, n: int, turns: int, rng: random.Random):  # noqa: ANN001, ANN201
    """Multi-turn specs per session: (prompt, max_new, stop, suffix_len). Raw-completion
    sessions extend prompt + reply; chat sessions put a role marker before the reply."""
    out = []
    for s in range(n):
        chat = s % 2 == 1
        suffix = [7, 8] if chat else []
        history = prompt_of(400 + s, 5 + s)
        specs = []
        for _ in range(turns):
            prompt = history + suffix
            greedy = list(ref.generate(prompt, 6, 0.0, frozenset()))
            stop = frozenset({greedy[rng.randint(1, 4)]}) if rng.random() < 0.6 else frozenset()
            reply = list(ref.generate(prompt, 6, 0.0, stop))
            specs.append((prompt, 6, stop, len(suffix), reply))
            history = history + ([3] if chat else []) + reply + prompt_of(rng.randint(1, 99), 2)
        out.append(specs)
    return out


def test_real_model_multi_turn_matches_generate_with_and_without_deferral(checkpoint: Path) -> None:
    ref = build(checkpoint, max_batch=1)
    specs = sessions(ref, 4, 3, random.Random(0))
    reused = {}
    for defer in (False, True):
        for overlap in (False, True):
            model = build(checkpoint, max_batch=3)
            cache = SessionCache(model.block_allocator, model.block_size, model.num_snapshots)
            sched = Scheduler(
                model, cache, prefill_chunk=4, overlap=overlap, defer_turn_close=defer
            )
            got = []
            for turn in range(3):  # sessions interleave within a turn, turns are sequential
                sinks = []
                for sess in specs:
                    prompt, max_new, stop, suffix_len, _ = sess[turn]
                    sink = Sink()
                    sched.submit(Request(list(prompt), max_new, 0.0, stop, sink, suffix_len))
                    sinks.append(sink)
                drain(sched)
                for sess, sink in zip(specs, sinks, strict=True):
                    _, _, stop, _, reply = sess[turn]
                    assert sink.error is None, sink.error
                    assert sink.tokens == [t for t in reply if t not in stop]
                    got.append(sink.end[2])
            reused[(defer, overlap)] = got
    # Reuse counts are not compared across modes: the tiny CPU KV pool (18 blocks) evicts, and
    # the eager turn-close nodes change what gets evicted. Output equality above is the check;
    # the fake-runner tests pin the exact reuse depths.
    assert any(r > 0 for r in reused[(True, False)][4:]), "no follow-up turn resumed a cache node"
    assert any(r > 0 for r in reused[(True, True)][4:]), "no follow-up turn resumed a cache node"

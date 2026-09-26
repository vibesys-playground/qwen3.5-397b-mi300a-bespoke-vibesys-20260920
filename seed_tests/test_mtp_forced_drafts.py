"""`SEED_MTP_FORCED_DRAFTS`: folded chat-suffix ids ride an MTP round as forced drafts.

CPU only, against a fake speculative runner whose "model" is `test_scheduler.next_token`.
Forced ticks must be fed in order, accepted unconditionally, and never streamed; the output
must equal one-request-at-a-time generation; and with the flag on, a lane draining its suffix
must not push the step back to plain decode.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scheduler as sched_mod  # noqa: E402
import tp_driver  # noqa: E402
from test_scheduler import FakeRunner, drain, make, next_token, request, serial  # noqa: E402

K = 2


class FakeSpecRunner(FakeRunner):
    """Perfect drafts for free lanes; forced lanes feed their suffix ids first."""

    max_seq = 1 << 20

    def __init__(self, *a, **kw) -> None:  # noqa: ANN002, ANN003
        super().__init__(*a, **kw)
        self.rounds: list[list[list[int]] | None] = []

    def decode_tokens_per_step(self) -> int:
        return K + 1

    def forced_drafts(self) -> bool:
        return True

    def speculative_decode(self, lanes, tokens, positions, budgets, stops, forced=None):  # noqa: ANN001, ANN201
        self.rounds.append(forced)
        out = []
        for j, (lane, tok, pos) in enumerate(zip(lanes, tokens, positions, strict=True)):
            assert pos == len(self.hist[lane])
            f = list(forced[j]) if forced is not None else []
            assert len(f) <= K
            fed = [tok, *f]
            committed = []
            if f:
                for x in fed:
                    self._write(lane, [x], len(self.hist[lane]))
                    self.hist[lane].append(x)
                    committed.append(next_token(self.hist[lane]))
            else:
                x = tok
                for c in range(K + 1):
                    self._write(lane, [x], len(self.hist[lane]))
                    self.hist[lane].append(x)
                    x = next_token(self.hist[lane])
                    committed.append(x)
                    if x in stops[j] or len(committed) >= budgets[j]:
                        break
            out.append(committed)
        return out


def make_spec(monkeypatch: pytest.MonkeyPatch, forced_on: bool, **kw):  # noqa: ANN003, ANN201
    monkeypatch.setattr(sched_mod, "MTP_FORCED_DRAFTS", forced_on)
    monkeypatch.setattr("test_scheduler.FakeRunner", FakeSpecRunner)
    return make(spec_decode=True, fold_turn_suffix=True, **kw)


@pytest.mark.parametrize("suffix", [[990], [990, 991], [990, 991, 992], [990, 991, 992, 993, 994]])
def test_forced_suffix_rides_the_round_and_matches_serial(
    monkeypatch: pytest.MonkeyPatch, suffix: list[int]
) -> None:
    prompt = [1, 2, 3, 4, *suffix]
    sched, runner, _ = make_spec(monkeypatch, True, max_batch=2, prefill_chunk=8)
    req, sink = request(prompt, 7, suffix_len=len(suffix))
    sched.submit(req)
    drain(sched)
    assert sink.tokens == serial(prompt, 7)
    if len(suffix) > 1:
        assert any(r is not None and any(r) for r in runner.rounds), "suffix never rode a round"
        assert not runner.batches, "a forced lane pushed the step back to plain decode"


def test_flag_off_keeps_plain_decode_for_forced_lanes(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt = [1, 2, 3, 4, 990, 991, 992]
    sched, runner, _ = make_spec(monkeypatch, False, max_batch=2, prefill_chunk=8)
    req, sink = request(prompt, 5, suffix_len=3)
    sched.submit(req)
    drain(sched)
    assert sink.tokens == serial(prompt, 5)
    assert runner.batches, "without the flag the forced lane must decode per token"
    assert all(r is None for r in runner.rounds)


@settings(max_examples=40, deadline=None)
@given(
    suffixes=st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=3),
    max_new=st.integers(min_value=1, max_value=9),
    stop_tok=st.one_of(st.none(), st.integers(min_value=1, max_value=997)),
)
# The first generated token is the stop id: it counts but is not streamed.
@example(suffixes=[4], max_new=1, stop_tok=258)
def test_mixed_forced_and_free_lanes_match_serial(suffixes, max_new, stop_tok) -> None:  # noqa: ANN001
    """Several concurrent requests, some still draining suffixes while others generate. A stop
    id ends the turn without being streamed (`test_stop_token_ends_the_turn_without_streaming_
    it`), so the expected stream is `serial` minus a final stop id."""
    mp = pytest.MonkeyPatch()
    try:
        sched, _, _ = make_spec(mp, True, max_batch=4, prefill_chunk=8)
        stop = frozenset() if stop_tok is None else frozenset({stop_tok})
        reqs = []
        for i, n in enumerate(suffixes):
            prompt = [10 * i + 1, 10 * i + 2, *[900 + 10 * i + j for j in range(n)]]
            req, sink = request(prompt, max_new, stop, suffix_len=n)
            sched.submit(req)
            reqs.append((prompt, sink))
        drain(sched)
        for prompt, sink in reqs:
            want = serial(prompt, max_new, stop)
            stopped = bool(want) and want[-1] in stop
            assert sink.tokens == (want[:-1] if stopped else want)
            assert sink.end is not None and sink.end[0] == ("stop" if stopped else "length")
            assert sink.end[1] == len(want)
    finally:
        mp.undo()


def test_speculative_command_roundtrips_forced_feed() -> None:
    slots, toks, pos = [3, 5, 7], [11, 12, 13], [100, 200, 300]
    budgets, stops = [4, 5, 6], [[1], [], [2, 3]]
    plain = tp_driver.Command.speculative_cmd(slots, toks, pos, budgets, stops)
    assert plain.unbatch_forced() is None
    forced = [[], [42, 43], [44]]
    cmd = tp_driver.Command.speculative_cmd(slots, toks, pos, budgets, stops, forced)
    assert cmd.unbatch_speculative() == (slots, toks, pos, budgets, stops)
    assert cmd.unbatch_forced() == forced
    none_forced = tp_driver.Command.speculative_cmd(slots, toks, pos, budgets, stops, [[], [], []])
    assert none_forced.payload == plain.payload


@settings(max_examples=200, deadline=None)
@given(
    rows=st.lists(
        st.tuples(
            st.integers(0, K),
            st.lists(st.integers(0, 1 << 17), min_size=K + 1, max_size=K + 1),
            st.lists(st.integers(0, 1 << 17), min_size=K, max_size=K),
        ),
        min_size=1,
        max_size=8,
    )
)
def test_apply_forced_replaces_exactly_the_first_forced_n_drafts(rows) -> None:  # noqa: ANN001
    """The wide verify's input columns after the forced feed: forced ids where
    `col < forced_n`, the draft otherwise, column 0 never touched."""
    import graph_verify_wide  # noqa: PLC0415
    import torch  # noqa: PLC0415

    tm = torch.tensor([r[1] for r in rows], dtype=torch.long)
    n = torch.tensor([r[0] for r in rows], dtype=torch.long)
    ids = torch.tensor([r[2] for r in rows], dtype=torch.long)
    graph_verify_wide.apply_forced(tm, n, ids)
    for j, (f_n, row, f_ids) in enumerate(rows):
        assert tm[j].tolist() == [row[0], *f_ids[:f_n], *row[1 + f_n :]]

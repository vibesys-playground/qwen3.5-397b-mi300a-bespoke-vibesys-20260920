"""CPU tests for the MTP wide-verify graph capture (`graph_mtp.py`).

Same posture as `test_graph_capture.py`: no GPU here, so this cannot check that `torch.cuda.
graph` actually records the verify step, only that the static-shape step computes what
`mtp.verify_and_commit` computes -- at a full bucket and a padded one, for every accept length,
and that a padding row never corrupts a real lane's persistent state. The capture backend is
`EagerBackend`, standing in for a real replay the same way `test_graph_capture.py`'s does.

    <python-with-torch> -m pytest seed_tests/test_graph_mtp.py -q -o addopts=
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import graph_mtp  # noqa: E402
import mtp  # noqa: E402
from graph_mtp import MTPVerifyRunner  # noqa: E402
from test_mtp import assert_slot_state_matches, build, prompt_of, write_mtp_shard  # noqa: E402
from test_seed_parity import build_hf, tiny_config, write_checkpoint  # noqa: E402

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("tiny-graph-mtp")
    cfg = tiny_config()
    write_checkpoint(build_hf(cfg=cfg), out, mxfp4=False)
    write_mtp_shard(cfg, out)
    return out


class EagerBackend:
    """Stands in for capture, exactly like `test_graph_capture.py`'s own: "replay" re-runs the
    step against the same static buffers, which is the equivalence a real replay has to
    satisfy without needing a device that can record it."""

    def __init__(self) -> None:
        self.devices: list[torch.device] = []

    def capture(self, step, device):
        self.devices.append(device)
        return step


class FailingBackend:
    def capture(self, step, device):
        raise RuntimeError("HIP error: operation not permitted when stream is capturing")


def prefilled_with_mtp(checkpoint: Path, prompts: list[list[int]], k: int = 3):
    model = build(checkpoint, max_batch=len(prompts), k=k)
    for slot, prompt in enumerate(prompts):
        model.begin(slot)
        model.prefill(slot, prompt, 0)
    return model


def round_of(model, mtp_mod_instance, prompt: list[int], slot: int = 0):
    """One draft+verify round's inputs, the way `Model.speculative_decode` builds them."""
    hidden = model.cached_hidden(slot)
    logits = model.decode([slot], [prompt[-1]], [len(prompt) - 1])
    base = int(logits.argmax(-1))
    return hidden, base


# ---------------------------------------------------------------- (a) capture + validate


def test_prepare_succeeds_on_cpu_with_the_eager_backend(checkpoint: Path) -> None:
    prompts = [prompt_of(300 + i, 5 + i) for i in range(3)]
    model = prefilled_with_mtp(checkpoint, prompts)
    runner = MTPVerifyRunner(model, model.mtp, backend=EagerBackend())
    assert runner.prepare() is True
    assert runner.buckets == [1, 2, 3]
    assert set(runner.graphs) == {1, 2, 3}


def test_ordinary_decode_refreshes_hidden_before_return_to_mtp(checkpoint: Path) -> None:
    """A temporarily ineligible batch may take one ordinary step, then resume MTP."""
    prompt = prompt_of(333, 7)

    eager = build(checkpoint, max_batch=1, k=3)
    eager.begin(0)
    eager_prefill = eager.prefill(0, prompt, 0)

    captured = build(checkpoint, max_batch=1, k=3)
    runner = graph_mtp.GraphMTPRunner(captured, backend=EagerBackend())
    assert runner.prepare() is True
    assert runner.mtp_runner.enabled
    runner.begin(0)
    captured_prefill = captured.prefill(0, prompt, 0)
    torch.testing.assert_close(captured_prefill, eager_prefill)

    token = int(eager_prefill.argmax(-1)[0])
    eager_logits = eager.decode([0], [token], [len(prompt)])
    captured_logits = runner.decode([0], [token], [len(prompt)])
    assert runner.decode_path == "mtp-fallback-eager"
    torch.testing.assert_close(captured_logits, eager_logits)
    torch.testing.assert_close(captured.cached_hidden(0), eager.cached_hidden(0))

    next_token = int(eager_logits.argmax(-1)[0])
    want = eager.speculative_decode([0], [next_token], [len(prompt) + 1])
    got = runner.speculative_decode([0], [next_token], [len(prompt) + 1])
    assert got == want
    assert_slot_state_matches(captured, eager, 0, len(prompt) + 1 + len(want[0]))


def test_prepare_fails_closed_with_a_failing_backend(checkpoint: Path) -> None:
    prompts = [prompt_of(310 + i, 5) for i in range(2)]
    model = prefilled_with_mtp(checkpoint, prompts)
    runner = MTPVerifyRunner(model, model.mtp, backend=FailingBackend())
    assert runner.prepare() is False
    assert runner.enabled is False


def test_multi_segment_layouts_are_refused(checkpoint: Path) -> None:
    """The one restriction this module's docstring names: verify capture needs a single
    segment (what a TP rank always has), not the `--tp 1` pipeline layout's split."""
    prompts = [prompt_of(320 + i, 5) for i in range(2)]
    model = prefilled_with_mtp(checkpoint, prompts)
    model.layer_dev = [CPU, CPU, torch.device("cpu:0"), torch.device("cpu:0")]
    runner = MTPVerifyRunner(model, model.mtp, backend=EagerBackend())
    assert runner.supported() is False
    assert runner.prepare() is False


# ---------------------------------------------------------------- (b) numerics match eager, every accept length


@pytest.mark.parametrize("accept", [0, 1, 2, 3])
def test_replay_matches_eager_verify_and_commit_at_every_accept_length(
    checkpoint: Path, accept: int
) -> None:
    """Mirrors `test_mtp.py::test_verify_and_commit_matches_sequential_decode`: force
    acceptance to exactly `accept` and check both the committed tokens and the state side
    effects (DeltaNet, KV, the cached-hidden seed) against one more ordinary decode step."""
    k = 3
    prompt = prompt_of(1, 6)

    oracle = build(checkpoint, max_batch=1, k=k)
    oracle.begin(0)
    oracle_logits = oracle.prefill(0, prompt, 0)
    base = oracle.sample_batch(oracle_logits, [0.0])[0]
    tok, ground_truth = base, []
    for i in range(k + 1):
        logits = oracle.decode([0], [tok], [len(prompt) + i])
        tok = oracle.sample_batch(logits, [0.0])[0]
        ground_truth.append(tok)

    def forced_draft() -> list[int]:
        if accept >= k:
            return ground_truth[:k]
        wrong = (ground_truth[accept] + 1) % oracle.cfg.vocab
        return ground_truth[:accept] + [wrong] + [0] * (k - accept - 1)

    seq_model = build(checkpoint, max_batch=1, k=k)
    seq_model.begin(0)
    seq_model.prefill(0, prompt, 0)
    tok, committed_seq = base, []
    for _ in range(accept + 1):
        logits = seq_model.decode([0], [tok], [len(prompt) + len(committed_seq)])
        tok = seq_model.sample_batch(logits, [0.0])[0]
        committed_seq.append(tok)
    assert committed_seq == ground_truth[: accept + 1]

    # `prepare()` (via `_validate`) resets every slot 0..capacity-1, so it must run *before*
    # anything is prefilled -- the same ordering `test_graph_capture.py`'s `graph_runner`
    # documents for decode capture.
    graph_model = build(checkpoint, max_batch=1, k=k)
    runner = MTPVerifyRunner(graph_model, graph_model.mtp, backend=EagerBackend())
    assert runner.prepare() is True
    runner.begin(0)  # not graph_model.begin: this also marks the runner's block-table mirror
    # dirty, which a bare `graph_model.begin` would not -- see `MTPVerifyRunner.begin`'s
    # docstring for why a prefill after a direct `model.begin` can leave `fill` reading a
    # stale lane_table entry from `prepare`'s own validation round.
    graph_model.prefill(0, prompt, 0)

    draft_tokens = [forced_draft()]
    committed = runner.verify_and_commit([0], [base], draft_tokens, [len(prompt)])
    assert committed == [ground_truth[: accept + 1]]

    next_pos = len(prompt) + accept + 1
    assert_slot_state_matches(graph_model, seq_model, 0, next_pos)
    seq_logits = seq_model.decode([0], [committed_seq[-1]], [next_pos])
    graph_logits = graph_model.decode([0], [committed[0][-1]], [next_pos])
    torch.testing.assert_close(graph_logits, seq_logits, atol=1e-5, rtol=1e-5)


def test_replay_matches_eager_at_a_real_batch_with_mixed_accept_lengths(checkpoint: Path) -> None:
    """Several slots verified together, each forced to a different accept length, replayed
    through the captured bucket and checked against plain (uncaptured) `mtp.verify_and_commit`
    on an identically-seeded second model."""
    k = 3
    prompts = [prompt_of(400 + i, 5 + i) for i in range(3)]
    accepts = [0, 2, 3]

    def build_and_prime():
        """Prime `m` (prefill, then the one priming decode at `len(prompt) - 1`) and stop --
        `m`'s state past that point is exactly what `want`'s own `verify_and_commit` call
        below has to start from, the same primed-but-not-further-advanced state `graph_model`
        reaches too. The ground-truth continuation used to build `drafts` is rolled out `k + 1`
        steps deeper than that, so it cannot be computed by stepping `m` itself (DeltaNet's
        recurrent state is path-dependent -- rolling `m` forward for `truth` would leave `m`
        primed `k + 1` tokens past where `graph_model` is, and `want` would stop matching `got`
        for a reason that has nothing to do with graph capture). A second, identically primed
        probe model carries that rollout instead, so `m` itself never advances past priming.
        """
        m = prefilled_with_mtp(checkpoint, prompts, k=k)
        probe = prefilled_with_mtp(checkpoint, prompts, k=k)
        bases, drafts = [], []
        for slot, prompt in enumerate(prompts):
            logits = m.decode([slot], [prompt[-1]], [len(prompt) - 1])
            base = int(logits.argmax(-1))
            probe.decode([slot], [prompt[-1]], [len(prompt) - 1])
            tok, truth = base, []
            for i in range(k + 1):
                logits = probe.decode([slot], [tok], [len(prompt) + i])
                tok = int(logits.argmax(-1))
                truth.append(tok)
            bases.append(base)
            a = accepts[slot]
            if a >= k:
                drafts.append(truth[:k])
            else:
                wrong = (truth[a] + 1) % m.cfg.vocab
                drafts.append(truth[:a] + [wrong] + [0] * (k - a - 1))
        return m, bases, drafts

    # Reset and re-run from the same prefilled state for a clean eager reference.
    eager_model, bases, draft_tokens = build_and_prime()
    positions = [len(p) for p in prompts]
    want = mtp.verify_and_commit(eager_model, eager_model.mtp, [0, 1, 2], bases, draft_tokens, positions)

    # As above: `prepare()` resets every slot, so build the (unprefilled) model, prepare the
    # runner, and only then prefill each prompt.
    graph_model = build(checkpoint, max_batch=len(prompts), k=k)
    runner = MTPVerifyRunner(graph_model, graph_model.mtp, backend=EagerBackend())
    assert runner.prepare() is True
    for slot, prompt in enumerate(prompts):
        runner.begin(slot)  # not graph_model.begin: see the note in the single-slot test above
        graph_model.prefill(slot, prompt, 0)
    for slot, prompt in enumerate(prompts):
        logits = graph_model.decode([slot], [prompt[-1]], [len(prompt) - 1])
        assert int(logits.argmax(-1)) == bases[slot]

    got = runner.verify_and_commit([0, 1, 2], bases, draft_tokens, positions)
    assert got == want


# ---------------------------------------------------------------- (c) bucketing and padding


def test_a_narrower_batch_pads_to_the_smallest_sufficient_bucket(checkpoint: Path) -> None:
    prompts = [prompt_of(500 + i, 5) for i in range(4)]
    model = prefilled_with_mtp(checkpoint, prompts, k=2)
    runner = MTPVerifyRunner(model, model.mtp, backend=EagerBackend())
    assert runner.prepare() is True
    assert runner.buckets == [1, 2, 4]

    base = [int(model.decode([s], [prompts[s][-1]], [len(prompts[s]) - 1]).argmax(-1)) for s in range(3)]
    drafts = [[0, 0] for _ in range(3)]
    positions = [len(prompts[s]) for s in range(3)]
    committed = runner.verify_and_commit([0, 1, 2], base, drafts, positions)
    assert len(committed) == 3
    for tokens in committed:
        assert 1 <= len(tokens) <= 3


def test_padding_row_does_not_corrupt_an_unrelated_lanes_state(checkpoint: Path) -> None:
    """The lane `pad_slot_for` picks as filler must come out of a replay bit-identical to
    before it, the wide-verify counterpart of `test_graph_capture.py`'s own padding-safety
    check."""
    prompts = [prompt_of(600 + i, 5 + i) for i in range(3)]
    model = prefilled_with_mtp(checkpoint, prompts, k=2)  # max_batch=3, buckets [1, 2, 3]
    runner = MTPVerifyRunner(model, model.mtp, backend=EagerBackend())
    assert runner.prepare() is True

    # A 2-request round on a 3-lane model pads to bucket 3; pad_slot_for([0, 1], 3) == 2.
    before_conv = [pool["conv"][2].clone() for pool in model.pool if "conv" in pool]
    before_rec = [pool["rec"][2].clone() for pool in model.pool if "rec" in pool]
    before_hidden = model.hidden_scratch[2].clone()

    base = [int(model.decode([s], [prompts[s][-1]], [len(prompts[s]) - 1]).argmax(-1)) for s in range(2)]
    drafts = [[0, 0], [0, 0]]
    positions = [len(prompts[0]), len(prompts[1])]
    runner.verify_and_commit([0, 1], base, drafts, positions)

    for pool, before in zip((p for p in model.pool if "conv" in p), before_conv, strict=True):
        assert torch.equal(pool["conv"][2], before)
    for pool, before in zip((p for p in model.pool if "rec" in p), before_rec, strict=True):
        assert torch.equal(pool["rec"][2], before)
    assert torch.equal(model.hidden_scratch[2], before_hidden)


# ---------------------------------------------------------------- (d) pure accept-length math


def test_accept_len_matches_greedy_accept_length_batched() -> None:
    # step_argmax has t = k + 1 = 4 columns: index i in [0, k) predicts draft[i], and the last
    # column (index k, here a "99" that is never read by acceptance) is the bonus token used
    # only when every draft is accepted -- see `verify_round_step`'s own docstring/comment.
    draft = torch.tensor([[5, 6, 7], [5, 6, 7], [5, 9, 7], [1, 2, 3]])
    step_argmax = torch.tensor(
        [[5, 6, 7, 99], [5, 6, 9, 99], [5, 2, 3, 99], [9, 2, 3, 99]]
    )
    got = graph_mtp._accept_len(draft, step_argmax)
    want = [
        mtp.greedy_accept_length(d.tolist(), step_argmax[i, :3].tolist())
        for i, d in enumerate(draft)
    ]
    assert got.tolist() == want == [3, 2, 1, 0]


def test_fixed_prefix_pack_index_has_static_shape_and_packed_live_prefix() -> None:
    cases = [
        ([1, 3, 2], [0, 1, 4, 6], [0, 4, 5, 6, 8, 9]),
        ([1, 1, 1], [0, 1, 2, 3], [0, 4, 8]),
        ([4, 4, 4], [0, 4, 8, 12], list(range(12))),
    ]
    for lengths, want_cu, want_live in cases:
        src, cu = graph_mtp._fixed_prefix_pack_index(torch.tensor(lengths), 4)
        assert src.shape == (12,)
        assert cu.tolist() == want_cu
        assert src[: int(cu[-1])].tolist() == want_live
        assert bool(((0 <= src) & (src < 12)).all())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

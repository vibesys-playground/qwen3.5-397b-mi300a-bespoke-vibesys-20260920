"""CPU tests for the opt-in decode graph capture (`--enable-graph-capture`).

What a machine without a GPU can and cannot check here.

Cannot: that `torch.cuda.graph` records the decode step correctly, that a replay is faster
than dispatching the step from Python, or that the step is free of the capture-time
illegalities (host syncs, data-dependent shapes) it was written to avoid. Stream capture
needs a device. Those are the questions the next cluster round answers, and it only answers
them if the server is started with the flag: without it none of this code runs.

Can, and does below: that the flag off changes nothing; that the static-shape step computes
what the eager step computes, at a full batch and a partial one; that padding rows leave
their slots' state alone; that the static buffers carry this step's inputs and not the
previous step's; and that every failure path (capture raises, the captured step disagrees
with eager, a replay raises later) ends with decode running eagerly rather than serving
wrong tokens. The capture backend is injected, so a "replay" that simply re-runs the step
against the same static buffers stands in for the real one: that is the equivalence a real
replay has to satisfy, minus the recording.

    <python-with-torch> -m pytest seed_tests/test_graph_capture.py -q -o addopts=
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import graph_decode  # noqa: E402
from graph_decode import GraphDecodeRunner, Segment, plan_segments  # noqa: E402
from scheduler import Request, Scheduler  # noqa: E402
from session_cache import SessionCache  # noqa: E402
from test_batched_decode import MAX_SEQ, build, prompt_of  # noqa: E402
from test_scheduler import Sink, drain  # noqa: E402
from test_seed_parity import build_hf, write_checkpoint  # noqa: E402

import server  # noqa: E402

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("tiny-graph")
    write_checkpoint(build_hf(), out, mxfp4=False)
    return out


class EagerBackend:
    """Stands in for capture: "replay" re-runs the step against the same static buffers.

    A real replay re-executes the recorded kernels against the addresses they were captured
    at, and reads every input from those addresses. Re-running the step reads the same
    buffers the same way, so anything the static step gets wrong about them (a stale input,
    an output read from the wrong place, a padding row that writes where it should not)
    shows up here. What it cannot show is whether the step is capturable at all.
    """

    def __init__(self) -> None:
        self.devices: list[torch.device] = []

    def capture(self, step, device):
        self.devices.append(device)
        return step


class FailingBackend:
    """Capture raises, the way an illegal op inside a captured region would."""

    def capture(self, step, device):
        raise RuntimeError("HIP error: operation not permitted when stream is capturing")


class CorruptingBackend:
    """Capture succeeds but the replay computes nothing, leaving the output buffer stale."""

    def capture(self, step, device):
        return lambda: None


class LateFailureBackend:
    """Replays work until `after` calls, then raise: a graph that goes bad mid-run."""

    def __init__(self, after: int) -> None:
        self.left = after

    def capture(self, step, device):
        def replay() -> None:
            if self.left <= 0:
                raise RuntimeError("replay blew up")
            self.left -= 1
            step()

        return replay


def prefilled(checkpoint: Path, prompts: list[list[int]], max_batch: int | None = None):
    model = build(checkpoint, max_batch=max_batch or len(prompts))
    for slot, prompt in enumerate(prompts):
        model.begin(slot)
        model.prefill(slot, prompt, 0)
    return model


def graph_runner(
    checkpoint: Path, prompts: list[list[int]], max_batch: int | None = None, backend=None
) -> GraphDecodeRunner:
    """A prepared runner with `prompts` prefilled, in the order the server does it.

    Capture comes first and prefill second, because `prepare` runs real decode steps
    (warmup, then its own check against the eager step) over every slot and then clears
    them. A slot prefilled before `prepare` would not survive it. The server calls
    `prepare` while it is still reporting 503, so no slot holds anything yet.
    """
    model = build(checkpoint, max_batch=max_batch or len(prompts))
    runner = GraphDecodeRunner(model, backend=backend or EagerBackend())
    assert runner.prepare(), "the static step must agree with the eager step on CPU"
    for slot, prompt in enumerate(prompts):
        runner.begin(slot)
        runner.prefill(slot, prompt, 0)
    return runner


# ---------------------------------------------------------------- (a) the flag is a no-op when off


def test_graph_capture_defaults_to_off() -> None:
    args = server.parse_args(["--model-path", "/nonexistent"])
    assert args.enable_graph_capture is False


def test_flag_turns_it_on() -> None:
    args = server.parse_args(
        ["--model-path", "/nonexistent", "--tp", "1", "--enable-graph-capture"]
    )
    assert args.enable_graph_capture is True


def test_the_flag_is_accepted_under_tensor_parallelism() -> None:
    """It used to be rejected above `--tp 1`, on the premise that the captured step was a
    second, unsharded spelling of decode. It is sharded now (`test_graph_capture_tp.py`)."""
    args = server.parse_args(
        ["--model-path", "/nonexistent", "--tp", "4", "--enable-graph-capture"]
    )
    assert args.enable_graph_capture is True


def test_workers_are_started_with_the_capture_flag() -> None:
    """A rank that did not capture would meet a replaying peer inside a layer's all-reduce.

    `start_workers` builds each worker's argv explicitly rather than replaying `sys.argv`,
    so a flag that is not listed there is silently dropped for ranks 1...
    """
    args = server.parse_args(
        ["--model-path", "/nonexistent", "--tp", "4", "--enable-graph-capture"]
    )
    started: list[list[str]] = []
    real_popen = server.subprocess.Popen
    try:
        server.subprocess.Popen = lambda argv, **kw: (
            started.append(argv) or SimpleNamespace(poll=lambda: None)
        )
        server.start_workers(args)
    finally:
        server.subprocess.Popen = real_popen
    assert len(started) == 3
    for argv in started:
        assert "--enable-graph-capture" in argv


def test_the_flag_is_not_passed_to_workers_when_it_is_off() -> None:
    args = server.parse_args(["--model-path", "/nonexistent", "--tp", "4"])
    started: list[list[str]] = []
    real_popen = server.subprocess.Popen
    try:
        server.subprocess.Popen = lambda argv, **kw: (
            started.append(argv) or SimpleNamespace(poll=lambda: None)
        )
        server.start_workers(args)
    finally:
        server.subprocess.Popen = real_popen
    assert started and all("--enable-graph-capture" not in argv for argv in started)


def test_scheduler_drives_the_model_itself_when_the_flag_is_off(checkpoint: Path) -> None:
    """The point of the flag: with it off, nothing wraps the model, so nothing can change."""
    model = build(checkpoint, max_batch=2)
    args = SimpleNamespace(enable_graph_capture=False)
    assert server.build_runner(args, model) is model


def test_capture_is_attempted_when_the_flag_is_on(checkpoint: Path) -> None:
    """With the real backend and no GPU, capture fails and the wrapper forwards to eager.

    This is the fallback that matters most: it is the one that runs if capture turns out to
    be impossible on the cluster too.
    """
    model = build(checkpoint, max_batch=2)
    args = SimpleNamespace(enable_graph_capture=True)
    runner = server.build_runner(args, model)
    assert isinstance(runner, GraphDecodeRunner)
    assert runner.enabled is False
    assert runner.max_batch == model.max_batch

    prompts = [prompt_of(1, 5), prompt_of(2, 6)]
    for slot, prompt in enumerate(prompts):
        runner.begin(slot)
        runner.prefill(slot, prompt, 0)
    want = prefilled(checkpoint, prompts).decode([0, 1], [p[-1] for p in prompts], [5, 6])
    got = runner.decode([0, 1], [p[-1] for p in prompts], [5, 6])
    torch.testing.assert_close(got, want, atol=0, rtol=0)


# ---------------------------------------------------------------- (a2) bucket bookkeeping (pure)


def test_build_buckets_caps_the_canonical_ladder_at_max_batch() -> None:
    assert graph_decode.build_buckets(48) == [1, 2, 4, 8, 16, 24, 32, 48]
    assert graph_decode.build_buckets(96) == [1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96]
    assert graph_decode.build_buckets(128) == [
        1,
        2,
        4,
        8,
        16,
        24,
        32,
        48,
        64,
        80,
        96,
        112,
        128,
    ]
    assert graph_decode.build_buckets(20) == [1, 2, 4, 8, 16, 20]
    assert graph_decode.build_buckets(1) == [1]


def test_build_buckets_always_includes_max_batch_even_off_the_canonical_ladder() -> None:
    assert graph_decode.build_buckets(30) == [1, 2, 4, 8, 16, 24, 30]


def test_build_buckets_rejects_nonpositive_max_batch() -> None:
    with pytest.raises(ValueError):
        graph_decode.build_buckets(0)


def test_bucket_for_picks_the_smallest_sufficient_bucket() -> None:
    buckets = [1, 2, 4, 8, 16, 24, 32, 48]
    assert graph_decode.bucket_for(1, buckets) == 1
    assert graph_decode.bucket_for(2, buckets) == 2
    assert graph_decode.bucket_for(3, buckets) == 4
    assert graph_decode.bucket_for(24, buckets) == 24
    assert graph_decode.bucket_for(25, buckets) == 32
    assert graph_decode.bucket_for(48, buckets) == 48


def test_bucket_for_rejects_a_batch_past_every_bucket() -> None:
    with pytest.raises(ValueError):
        graph_decode.bucket_for(49, [1, 2, 4, 8, 16, 24, 32, 48])


def test_bucket_for_rejects_nonpositive_batch_size() -> None:
    with pytest.raises(ValueError):
        graph_decode.bucket_for(0, [1, 2])


def test_pad_slot_for_avoids_every_slot_already_in_the_batch() -> None:
    assert graph_decode.pad_slot_for([0, 2], 4) == 1
    assert graph_decode.pad_slot_for([1, 2, 3], 4) == 0
    assert graph_decode.pad_slot_for([], 4) == 0


def test_pad_slot_for_falls_back_when_the_batch_covers_every_lane() -> None:
    """Only reachable when bucket == max_batch, where `fill` places no padding rows at all,
    so this value is never actually used as a pad target; see the function's docstring."""
    assert graph_decode.pad_slot_for([0, 1, 2], 3) == 0


# ---------------------------------------------------------------- (b) segment planning


def test_segments_are_contiguous_runs_of_one_device() -> None:
    a, b = torch.device("cuda:0"), torch.device("cuda:1")
    assert plan_segments([a, a, a]) == [Segment(0, 3, a, last=True)]
    assert plan_segments([a, a, b, b]) == [
        Segment(0, 2, a, last=False),
        Segment(2, 4, b, last=True),
    ]
    assert len(plan_segments([a, b, a, b])) == 4


def test_one_graph_per_device(checkpoint: Path) -> None:
    backend = EagerBackend()
    runner = graph_runner(checkpoint, [prompt_of(5, 4), prompt_of(6, 4)], backend=backend)
    # all four layers of the tiny model are on one device, and there is one graph per
    # (bucket, segment); at max_batch=2 there are two buckets (1 and 2) and one segment each.
    assert set(backend.devices) == {CPU}
    assert len(backend.devices) == len(runner.buckets) == 2
    assert len(runner.replays) == len(runner.buffers) == 1  # the max_batch bucket's own segment
    assert all(len(g.buffers) == len(g.replays) == 1 for g in runner.graphs.values())


def test_layers_split_over_two_devices_match_the_eager_step(checkpoint: Path) -> None:
    """The deployed layout is four devices, a CPU run is one, so the split needs its own test.

    One graph is captured per device, and the hidden state crosses between them outside the
    captured regions. `torch.device("cpu:0")` is a device object distinct from
    `torch.device("cpu")` but the same place to allocate, so this really does plan two
    segments and really does run the hand-off copy between them, without a second device.
    """
    prompts = [prompt_of(150 + i, 5 + i) for i in range(3)]
    positions = [len(p) for p in prompts]
    tokens = [p[-1] for p in prompts]
    want = prefilled(checkpoint, prompts).decode([0, 1, 2], tokens, positions)

    model = build(checkpoint, max_batch=3)
    second = torch.device("cpu:0")
    model.layer_dev = [CPU, CPU, second, second]
    runner = GraphDecodeRunner(model, backend=EagerBackend())
    assert runner.prepare() is True
    assert [s.device for s in runner.segments] == [CPU, second]
    assert len(runner.replays) == len(runner.buffers) == 2

    for slot, prompt in enumerate(prompts):
        runner.begin(slot)
        runner.prefill(slot, prompt, 0)
    got = runner.decode([0, 1, 2], tokens, positions)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------- (c) the static step is the eager step


def test_static_step_matches_the_eager_step_at_full_batch(checkpoint: Path) -> None:
    """`prepare`'s own startup check, which is what guards the cluster round.

    It runs the eager step and a replay from the same state and compares, so a green
    `prepare()` here is the assertion.
    """
    prompts = [prompt_of(10 + i, 4 + i) for i in range(3)]
    assert graph_runner(checkpoint, prompts).enabled is True


def test_graph_decode_matches_eager_decode_slot_by_slot(checkpoint: Path) -> None:
    """A real batch, against a second model in the same state driven the eager way."""
    prompts = [prompt_of(20 + i, 5 + 2 * i) for i in range(3)]
    positions = [len(p) for p in prompts]
    tokens = [p[-1] for p in prompts]
    slots = [0, 1, 2]

    want = prefilled(checkpoint, prompts).decode(slots, tokens, positions)
    got = graph_runner(checkpoint, prompts).decode(slots, tokens, positions)

    assert got.shape == want.shape
    for slot in slots:
        torch.testing.assert_close(got[slot], want[slot], atol=1e-4, rtol=1e-4, msg=f"slot {slot}")
        assert int(got[slot].argmax()) == int(want[slot].argmax())


def test_graph_decode_matches_eager_on_a_partial_batch(checkpoint: Path) -> None:
    """Two of three slots in flight: the third row is padding and must not change the answer."""
    prompts = [prompt_of(30 + i, 6 + i) for i in range(3)]
    batch = [0, 2]
    tokens = [prompts[s][-1] for s in batch]
    positions = [len(prompts[s]) for s in batch]

    want = prefilled(checkpoint, prompts).decode(batch, tokens, positions)
    got = graph_runner(checkpoint, prompts).decode(batch, tokens, positions)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


def test_rows_are_returned_in_the_callers_order(checkpoint: Path) -> None:
    """Row j is slot j inside the step, but the caller gets its own order back.

    The scheduler zips the returned rows against its own batch list, so a permutation here
    would hand every request another request's token.
    """
    prompts = [prompt_of(40 + i, 5 + i) for i in range(3)]
    batch = [2, 0, 1]
    tokens = [prompts[s][-1] for s in batch]
    positions = [len(prompts[s]) for s in batch]

    want = prefilled(checkpoint, prompts).decode(batch, tokens, positions)
    got = graph_runner(checkpoint, prompts).decode(batch, tokens, positions)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------- (d) padding rows and stale buffers


def state_of(model, slot: int) -> list[torch.Tensor]:
    """Every tensor that is this slot's state: DeltaNet conv and recurrent rows, and KV rows."""
    return [t[slot].clone() for pool in model.pool for t in pool.values()]


def test_padding_rows_leave_their_slots_state_alone(checkpoint: Path) -> None:
    """The step computes all `max_batch` rows, including slots that are not in the batch.

    Those rows index a real slot's KV and recurrent state, so "inactive" has to mean the
    state comes out bit-identical, not just that the logits are discarded.
    """
    prompts = [prompt_of(50 + i, 5 + i) for i in range(3)]
    runner = graph_runner(checkpoint, prompts)
    before = state_of(runner.model, 2)

    runner.decode([0, 1], [prompts[0][-1], prompts[1][-1]], [len(prompts[0]), len(prompts[1])])

    for old, new in zip(before, state_of(runner.model, 2), strict=True):
        assert torch.equal(old, new), "an inactive row wrote into its slot's state"


def test_active_rows_do_advance_their_state(checkpoint: Path) -> None:
    """The negative control for the test above: masking must not switch everything off."""
    prompts = [prompt_of(60 + i, 5 + i) for i in range(3)]
    runner = graph_runner(checkpoint, prompts)
    before = state_of(runner.model, 1)

    runner.decode([0, 1], [prompts[0][-1], prompts[1][-1]], [len(prompts[0]), len(prompts[1])])

    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, state_of(runner.model, 1), strict=True)
    )


def test_every_input_buffer_is_rewritten_each_step(checkpoint: Path) -> None:
    """No input may survive from the previous step into the next replay.

    A refresh that is skipped or partial is the classic silent graph bug: the graph reads
    whatever the buffer still held. The two steps below deliberately land in *different*
    buckets (3 slots, then 1) so this also exercises that each bucket's buffers are refreshed
    independently -- the 3-bucket's buffers must not be touched by the 1-bucket's replay.
    """
    prompts = [prompt_of(70 + i, 5 + i) for i in range(3)]
    runner = graph_runner(checkpoint, prompts)

    runner.decode([0, 1, 2], [p[-1] for p in prompts], [len(p) for p in prompts])
    assert runner.decode_bucket == 3
    buf3 = runner.graphs[3].buffers[0]
    assert buf3.active.tolist() == [True, True, True]

    runner.decode([1], [prompts[1][-1]], [len(prompts[1])])
    assert runner.decode_bucket == 1
    buf1 = runner.graphs[1].buffers[0]
    # Row 0 of the 1-bucket is slot 1 (this step's only real request), always active: there
    # is no padding row to go stale here (capacity == batch size), so what this checks is that
    # the row's own pos/token are this step's, not some earlier step's.
    assert buf1.active.tolist() == [True]
    assert buf1.pos.tolist() == [len(prompts[1])]
    assert runner.tokens[:1, 0].tolist() == [prompts[1][-1]]
    # The 3-bucket's own buffers are untouched by the 1-bucket replay.
    assert buf3.active.tolist() == [True, True, True]


def test_padding_row_of_a_bucket_is_refreshed_inactive_each_step(checkpoint: Path) -> None:
    """A batch that does not exactly fill its bucket still gets a fully rewritten padding row.

    `max_batch=4` with the default bucket ladder gives buckets `[1, 2, 4]`, so a 3-request
    batch pads to the 4-bucket with one padding row -- unlike the `3`-bucket test above, where
    every bucket the suite otherwise exercises happens to have batch size == capacity.
    """
    prompts = [prompt_of(200 + i, 5 + i) for i in range(3)]
    runner = graph_runner(checkpoint, prompts, max_batch=4)

    runner.decode([0, 1, 2], [p[-1] for p in prompts], [len(p) for p in prompts])
    assert runner.decode_bucket == 4
    buf = runner.graphs[4].buffers[0]
    assert buf.active.tolist() == [True, True, True, False]
    assert buf.pos.tolist()[:3] == [len(p) for p in prompts]
    assert buf.pos.tolist()[3] == 0


# ---------------------------------------------------------------- (e) dispatch and fallbacks


def test_small_batches_use_the_smallest_sufficient_bucket(checkpoint: Path) -> None:
    """Bucketing replaces the old "small batches stay eager" cutoff: even a batch of 1 has its
    own bucket (`build_buckets` always includes 1), so it replays instead of falling back."""
    prompts = [prompt_of(80 + i, 5) for i in range(4)]
    runner = graph_runner(checkpoint, prompts, max_batch=4)
    assert runner.buckets == [1, 2, 4]

    seen: list[int] = []
    inner = runner.model.decode
    runner.model.decode = lambda s, t, p: (seen.append(len(s)), inner(s, t, p))[1]

    runner.decode([0], [1], [5])
    assert seen == [], "a one-request batch now has its own bucket, not the eager path"
    assert runner.decode_path == "graph" and runner.decode_bucket == 1

    runner.decode([0, 1], [1, 1], [5, 5])
    assert seen == [] and runner.decode_bucket == 2

    runner.decode([0, 1, 2], [1, 1, 1], [5, 5, 5])
    assert seen == [] and runner.decode_bucket == 4, "3 requests pad up to the 4-bucket"


def test_a_batch_the_graph_cannot_serve_stays_eager(checkpoint: Path) -> None:
    runner = graph_runner(checkpoint, [prompt_of(90 + i, 5) for i in range(3)])
    assert runner.replayable([0, 1, 2], [1, 2, 3]) is True
    assert runner.replayable([0, 0, 1], [1, 2, 3]) is False, "duplicate slots"
    assert runner.replayable([0, 1, 9], [1, 2, 3]) is False, "slot outside the pool"
    assert runner.replayable([0, 1, 2], [1, 2, MAX_SEQ]) is False, "position past the captured span"
    assert runner.replayable([0, 1, 2], [1, 2, -1]) is False, "negative position"


def test_capture_failure_falls_back_to_eager(checkpoint: Path) -> None:
    """A failed capture leaves a wrapper whose decode is the model's decode, bit for bit."""
    prompts = [prompt_of(100 + i, 5 + i) for i in range(3)]
    runner = GraphDecodeRunner(build(checkpoint, max_batch=3), backend=FailingBackend())

    assert runner.prepare() is False
    assert runner.enabled is False

    for slot, prompt in enumerate(prompts):
        runner.begin(slot)
        runner.prefill(slot, prompt, 0)
    positions = [len(p) for p in prompts]
    want = prefilled(checkpoint, prompts).decode([0, 1, 2], [p[-1] for p in prompts], positions)
    got = runner.decode([0, 1, 2], [p[-1] for p in prompts], positions)
    torch.testing.assert_close(got, want, atol=0, rtol=0)


def test_a_replay_that_disagrees_with_eager_is_rejected(checkpoint: Path) -> None:
    """Capture succeeding is not evidence that the graph computes the right thing."""
    prompts = [prompt_of(110 + i, 5 + i) for i in range(3)]
    runner = GraphDecodeRunner(prefilled(checkpoint, prompts), backend=CorruptingBackend())

    assert runner.prepare() is False
    assert runner.enabled is False


def test_a_replay_failure_disables_the_graph_and_does_not_retry_eagerly(checkpoint: Path) -> None:
    """A replay may have advanced some slots before it failed, so the step is not re-run.

    Re-running it eagerly would step those slots twice. The scheduler turns the raised
    error into a failed batch and drops those slots' cached prefixes; every later step
    takes the eager path.
    """
    prompts = [prompt_of(120 + i, 5 + i) for i in range(3)]
    buckets = graph_decode.build_buckets(len(prompts))
    # `warm_static` runs the raw (uncaptured) step functions, not `backend`'s replay callables,
    # so only `validate()`'s one replay per bucket draws on this backend's budget before the
    # first real `decode()` -- not `len(buckets) * WARMUP_STEPS` more besides.
    backend = LateFailureBackend(after=len(buckets))
    runner = graph_runner(checkpoint, prompts, backend=backend)

    seen: list[int] = []
    inner = runner.model.decode
    runner.model.decode = lambda s, t, p: (seen.append(len(s)), inner(s, t, p))[1]
    positions = [len(p) for p in prompts]

    with pytest.raises(RuntimeError, match="replay blew up"):
        runner.decode([0, 1, 2], [p[-1] for p in prompts], positions)
    assert seen == [], "the failed step must not be retried eagerly"
    assert runner.enabled is False

    runner.decode([0, 1, 2], [p[-1] for p in prompts], positions)
    assert seen == [3], "later steps take the eager path"


def test_warmup_forwards_to_the_model(checkpoint: Path) -> None:
    """`Engine._run` and `tp_driver.apply` both call `runner.warmup()` through the `Runner`
    protocol, while the runner's other warmup takes the static steps to run. Without this
    forwarding the server raises a TypeError on startup before it serves anything."""
    model = build(checkpoint, max_batch=2)
    calls: list[str] = []
    model.warmup = lambda: calls.append("warmup")
    GraphDecodeRunner(model, backend=EagerBackend()).warmup()
    assert calls == ["warmup"]


# ---------------------------------------------------------------- (e2) agreeing across ranks


class Peers:
    """A `tp.TP` stand-in whose `all_reduce` adds `others` votes to this rank's."""

    def __init__(self, world: int, others: float) -> None:
        self.plan, self.device, self.others = SimpleNamespace(world=world), CPU, others
        self.seen: list[float] = []

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        self.seen.append(float(x))
        return x.add_(self.others)


def test_the_group_turns_the_captured_path_on_only_when_every_rank_can(checkpoint: Path) -> None:
    runner = GraphDecodeRunner(build(checkpoint, max_batch=2), backend=EagerBackend())
    runner.model.tp = Peers(world=4, others=3.0)  # three peers vote yes
    assert runner.agree(True) is True
    assert runner.model.tp.seen == [1.0]


def test_a_rank_stays_eager_when_a_peer_could_not_capture(checkpoint: Path) -> None:
    """Without this, three replaying ranks wait forever on a fourth issuing its collectives
    from the eager path instead of from inside a graph."""
    runner = GraphDecodeRunner(build(checkpoint, max_batch=2), backend=EagerBackend())
    runner.model.tp = Peers(world=4, others=2.0)  # one of the three peers voted no
    assert runner.agree(True) is False


def test_a_rank_that_could_not_capture_votes_no(checkpoint: Path) -> None:
    runner = GraphDecodeRunner(build(checkpoint, max_batch=2), backend=EagerBackend())
    runner.model.tp = Peers(world=4, others=3.0)
    assert runner.agree(False) is False
    assert runner.model.tp.seen == [0.0]


# ---------------------------------------------------------------- (f) end to end through the scheduler


def test_scheduler_over_the_graph_runner_generates_the_serial_tokens(checkpoint: Path) -> None:
    """The adapter is a `Runner`: the whole server loop on top of it must not change.

    Prefill, prefix save/load and sampling all go to the model untouched; only decode is
    replayed. The reference is the seed's own one-sequence generation.
    """
    specs = [(prompt_of(130 + i, 5 + i), 6) for i in range(3)]
    reference_model = build(checkpoint, max_batch=1)
    expected = [list(reference_model.generate(p, n, 0.0, frozenset())) for p, n in specs]

    runner = GraphDecodeRunner(build(checkpoint, max_batch=3), backend=EagerBackend())
    assert runner.prepare() is True
    model = runner.model
    cache = SessionCache(model.block_allocator, model.block_size, model.num_snapshots)
    sched = Scheduler(runner, cache, prefill_chunk=4)

    sinks = [Sink() for _ in specs]
    for (prompt, max_new), sink in zip(specs, sinks, strict=True):
        sched.submit(Request(list(prompt), max_new, 0.0, frozenset(), sink))
    drain(sched)

    for i, (sink, want) in enumerate(zip(sinks, expected, strict=True)):
        assert sink.error is None, sink.error
        assert sink.tokens == want, f"request {i} diverged under the replayed decode step"


def test_warmup_runs_before_capture(checkpoint: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Capturing a first call records autotune probes and lazy allocations, not the steady state."""
    model = prefilled(checkpoint, [prompt_of(140 + i, 5) for i in range(3)])
    calls: list[str] = []

    class Recorder:
        def capture(self, step, device):
            calls.append("capture")
            return step

    monkeypatch.setattr(graph_decode, "WARMUP_STEPS", 3)
    real_step = graph_decode.segment_step

    def counting_step(m, segment, buf):
        inner = real_step(m, segment, buf)

        def step() -> None:
            calls.append("step")
            inner()

        return step

    monkeypatch.setattr(graph_decode, "segment_step", counting_step)
    GraphDecodeRunner(model, backend=Recorder()).prepare()
    assert calls[:4] == ["step", "step", "step", "capture"]


# ---------------------------------------------------------------- (g) incremental block-table sync


def test_lane_block_tables_sync_writes_only_the_given_prefix(checkpoint: Path) -> None:
    """`sync_lane` writes exactly `len(blocks)` entries and leaves the rest of the row alone
    (see its own docstring for why leaving stale entries beyond that length is still safe)."""
    model = build(checkpoint, max_batch=3)
    lt = graph_decode.LaneBlockTables(model, CPU)
    lt.table[2].fill_(99)  # simulate a stale row from a former occupant

    lt.sync_lane(2, [5, 6])
    assert lt.table[2, :2].tolist() == [5, 6]
    assert lt.table[2, 2:].tolist() == [99] * (model.max_blocks_per_lane - 2)


def test_fill_only_syncs_a_lane_when_its_block_list_actually_changed(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of `LaneBlockTables`: a step that does not cross any lane's block
    boundary must not touch the device mirror at all, unlike the pre-optimization `fill` that
    rebuilt every row from host Python lists every single call."""
    prompts = [prompt_of(700 + i, 5) for i in range(3)]
    runner = graph_runner(
        checkpoint, prompts
    )  # block_size is large enough that 5 tokens is 1 block

    calls: list[int] = []
    for lane_table in runner.lane_tables.values():
        real_sync = lane_table.sync_lane
        monkeypatch.setattr(
            lane_table,
            "sync_lane",
            lambda slot, blocks, real=real_sync: (calls.append(slot), real(slot, blocks))[1],
        )

    runner.decode([0, 1, 2], [p[-1] for p in prompts], [len(p) for p in prompts])
    first_step_syncs = len(calls)
    assert first_step_syncs > 0, "the first step after prefill must grow (and sync) every lane"

    calls.clear()
    runner.decode(
        [0, 1, 2], [1, 1, 1], [len(prompts[0]) + 1, len(prompts[1]) + 1, len(prompts[2]) + 1]
    )
    assert calls == [], "no lane crossed a block boundary, so no sync should have run"


def test_device_block_table_matches_a_host_rebuild_including_padding(checkpoint: Path) -> None:
    """The new device-gather path (`GraphDecodeRunner.fill`'s `block_table`/`block_valid`/
    `write_rows`) must compute exactly what the old host rebuild (`BlockTable.padded_row`/
    `physical_row`, `block_pool.valid_block_count`) did, for a batch that both pads (max_batch
    4, 3 real requests) and has grown past one block boundary. Padding rows read and write
    only `RESERVED_BLOCK` (never the filler lane's own table)."""
    import block_pool

    prompts = [prompt_of(710 + i, 20) for i in range(3)]  # long enough to cross a block boundary
    runner = graph_runner(checkpoint, prompts, max_batch=4)
    slots = [0, 1, 2]
    tokens = [p[-1] for p in prompts]
    positions = [len(p) for p in prompts]

    runner.decode(slots, tokens, positions)  # advance state, grow block tables past one boundary
    capacity = graph_decode.bucket_for(len(slots), runner.buckets)
    runner.fill(runner.graphs[capacity].buffers, capacity, slots, tokens, positions)

    model = runner.model
    pad = graph_decode.pad_slot_for(slots, model.max_batch)
    row_slots = slots + [pad] * (capacity - len(slots))
    row_pos = positions + [0] * (capacity - len(slots))
    n_pad = capacity - len(slots)
    reserved_row = [block_pool.RESERVED_BLOCK] * model.max_blocks_per_lane
    want_block_table = [
        model.block_tables[lane].padded_row(model.max_blocks_per_lane) for lane in slots
    ] + [reserved_row] * n_pad
    want_block_valid = [block_pool.valid_block_count(p + 1, model.block_size) for p in row_pos]
    want_write_rows = [
        model.block_tables[lane].physical_row(p, model.block_size)
        for lane, p in zip(slots, positions, strict=True)
    ] + [block_pool.RESERVED_BLOCK * model.block_size] * n_pad
    assert row_slots[len(slots) :] == [pad] * n_pad

    buf = runner.graphs[capacity].buffers[0]
    assert buf.block_table.tolist() == want_block_table
    assert buf.block_valid.tolist() == want_block_valid
    assert buf.write_rows.tolist() == want_write_rows


def test_attach_blocks_syncs_the_device_mirror_even_at_the_same_length(checkpoint: Path) -> None:
    """A length-only diff (what `fill`'s grow loop uses) would miss a copy-on-write swap that
    keeps the same block *count* but different ids; `attach_blocks` syncs unconditionally."""
    prompts = [prompt_of(720 + i, 5) for i in range(2)]
    runner = graph_runner(checkpoint, prompts)
    lane = 0
    original = runner.model.block_tables[lane].blocks
    assert len(original) >= 1

    swapped = [b + 1 for b in original]  # same length, different (fake) ids -- sync_lane only
    # ever copies plain ints into a tensor, so these need not be real allocator-issued ids.
    runner.attach_blocks(lane, swapped)

    for lane_table in runner.lane_tables.values():
        got = lane_table.table[lane, : len(swapped)].tolist()
        assert got == swapped, "attach_blocks must sync even when the length did not change"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# ---------------------------------------------------------------- (g) block accounting


def test_prepare_releases_every_block_its_warmup_and_validation_grew(checkpoint: Path) -> None:
    """Warmup and validation step every lane; `reset_slots` used to only forget the tables,
    leaking one block per lane per call (and failing the boot pool self-check)."""
    model = build(checkpoint, max_batch=3)
    runner = GraphDecodeRunner(model, backend=EagerBackend())
    assert runner.prepare() is True
    assert model.block_allocator.free_count == model.block_allocator.usable_blocks


def test_inactive_rows_use_the_reserved_block_and_allocate_nothing(checkpoint: Path) -> None:
    """A padding row reads and writes only `RESERVED_BLOCK`: it used to grow an idle lane's
    table to a dummy block that the lane's next `begin` forgot (one leaked block per
    admission), and to write into row 0 of whatever a mid-prefill lane held."""
    import block_pool

    prompts = [prompt_of(20, 5), prompt_of(40, 7), prompt_of(60, 6)]
    runner = graph_runner(checkpoint, prompts, max_batch=4)
    model = runner.model
    free = model.block_allocator.free_count
    runner.decode([0, 1, 2], [3, 4, 5], [5, 7, 6])  # 3 rows pad to the 4-row bucket
    assert runner.decode_path == "graph" and runner.decode_bucket == 4
    assert model.block_allocator.free_count == free
    assert model.block_tables[3].blocks == []
    buf = runner.graphs[4].buffers[0]
    assert buf.block_table[3:].eq(block_pool.RESERVED_BLOCK).all()
    assert buf.write_rows[3:].eq(block_pool.RESERVED_BLOCK * model.block_size).all()


def test_extend_blocks_reaches_the_device_mirror_before_the_next_replay(checkpoint: Path) -> None:
    """Serving path: the scheduler reserves ids on rank 0 and hands them in through
    `extend_blocks` before the step that writes into them (`OVERLAP_SCHED` does this before its
    lookahead launch too). `fill`'s own length check sees no change then, since the table grew
    before `fill` ran, so `extend_blocks` must mark the lane for re-sync. Two lanes cross a
    block boundary in the same step; replay must match the eager model over several steps."""
    import model as seed_model

    block = seed_model.KV_BLOCK_SIZE
    prompts = [prompt_of(740 + i, block - 1) for i in range(2)]
    slots = [0, 1]
    runner = graph_runner(checkpoint, prompts)
    eager = prefilled(checkpoint, prompts)
    for m in (runner.model, eager):
        m.scheduler_owns_blocks = True
    tokens = [p[-1] for p in prompts]
    for pos in range(block - 1, block + 2):
        for m, extend in ((runner.model, runner.extend_blocks), (eager, eager.extend_blocks)):
            grants = []
            for slot in slots:
                if m.block_tables[slot].token_capacity(m.block_size) < pos + 1:
                    grants.append((slot, m.block_allocator.alloc(1)[0]))
            if grants:
                extend(grants)
        positions = [pos] * len(slots)
        got = runner.decode(slots, tokens, positions)
        assert runner.decode_path == "graph"
        want = eager.decode(slots, tokens, positions)
        torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4, msg=f"pos {pos}")
        tokens = [int(t) for t in want.argmax(-1).reshape(-1)]
    for lane_table in runner.lane_tables.values():
        for slot in slots:
            blocks = runner.model.block_tables[slot].blocks
            assert lane_table.table[slot, : len(blocks)].tolist() == blocks


def test_cuda_graph_replay_keeps_the_captured_steps_closure_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression (round 15): a captured graph reads the tensors its step closed over by
    address only. `capture` returned the bare `graph.replay`, so a constant a builder allocated
    outside the step was freed with the step and its address reused (seen as the MTP wide
    verify's forced feed never applying). The replay must keep the step, and so its closure,
    alive."""
    import contextlib  # noqa: PLC0415
    import gc  # noqa: PLC0415
    import weakref  # noqa: PLC0415

    class FakeGraph:
        def replay(self) -> None:
            pass

    cuda = torch.cuda
    monkeypatch.setattr(cuda, "device", lambda _d: contextlib.nullcontext())
    monkeypatch.setattr(cuda, "synchronize", lambda _d=None: None)
    monkeypatch.setattr(cuda, "graph_pool_handle", lambda: object())
    monkeypatch.setattr(cuda, "CUDAGraph", FakeGraph)
    monkeypatch.setattr(cuda, "graph", lambda _g, pool=None: contextlib.nullcontext())

    def build():  # noqa: ANN202
        const = torch.arange(4)
        return (lambda: const.sum()), weakref.ref(const)

    step, ref = build()
    replay = graph_decode.CudaGraphBackend().capture(step, torch.device("cpu"))
    del step
    gc.collect()
    assert ref() is not None
    replay()

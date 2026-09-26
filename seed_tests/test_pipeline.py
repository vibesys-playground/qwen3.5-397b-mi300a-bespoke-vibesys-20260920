"""Hermetic CPU tests for the pipelined decode path.

Two layers of checking, because this machine has no GPUs:

1. `StagePipeline` on its own, with plain Python stages. Those cover the scheduling: that
   stages really do run different items at the same time, that results come back in item
   order, and that a stage failure reaches the caller instead of hanging it.
2. `Model` with four stages, which the tests get by passing the same device four times
   (`["cpu"] * 4`). The layer-to-stage split, the microbatch cut, the per-stage `.to`
   hand-off, the stage threads and all the per-slot state handling are the real code; only
   the devices are fake. The claim these assert is the one that must hold on four GPUs: the
   pipelined step is bit-identical to running the same microbatches one after another.

What none of this can check: that the four stages overlap on real hardware, or that the
cross-device copies are correct, since every `.to` here is a no-op between CPU tensors.
Those need the real machine.

    /tmp/torchenv/bin/python -m pytest \\
        examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/seed_tests/test_pipeline.py
"""

import sys
import threading
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import model as seed_model  # noqa: E402
from pipeline import StagePipeline  # noqa: E402
from scheduler import Request, Scheduler  # noqa: E402
from test_batched_decode import MAX_SEQ, make_scheduler, prompt_of  # noqa: E402
from test_scheduler import Sink, drain  # noqa: E402
from test_seed_parity import build_hf, write_checkpoint  # noqa: E402

STAGES = 4
DEVICES = ["cpu"] * STAGES
"""Four stages on one CPU. `Model` keys its stages on the position in `devices`, not on the
device object, so repeating a device gives the four-stage control flow without four GPUs."""


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("tiny-pipeline")
    write_checkpoint(build_hf(), out, mxfp4=False)
    return out


@pytest.fixture(scope="module")
def serial_model(checkpoint: Path) -> seed_model.Model:
    """One-device, one-slot model. `generate` resets its state, so turns can share it."""
    return seed_model.Model(checkpoint, ["cpu"], torch.float32, MAX_SEQ, 1)


# ---------------------------------------------------------------- (1) the schedule alone


def test_results_come_back_in_item_order() -> None:
    with StagePipeline([lambda v: v + 1, lambda v: v * 10]) as pipe:
        assert pipe.run([1, 2, 3]) == [20, 30, 40]
        assert pipe.run([]) == []
        assert pipe.run([7]) == [80]  # the pipeline is reusable across calls


def test_stages_hold_different_items_at_the_same_time() -> None:
    """The point of the whole change: stage s runs item m while stage s-1 runs item m+1.

    The barrier is only satisfied if stage 0 is inside item 1 while stage 1 is inside item 0.
    A driver that pushed one item through every stage before starting the next would block
    here, and the `BrokenBarrierError` would surface as a failed `run`.
    """
    both_in = threading.Barrier(2, timeout=10)

    def first(item: int) -> int:
        if item == 1:
            both_in.wait()
        return item

    def second(item: int) -> int:
        if item == 0:
            both_in.wait()
        return item

    with StagePipeline([first, second]) as pipe:
        assert pipe.run([0, 1]) == [0, 1]


def test_every_stage_runs_concurrently_at_depth_four() -> None:
    """All four stages busy at once, which is what four idle GPUs are being traded for."""
    all_four = threading.Barrier(STAGES, timeout=10)

    def stage(index: int):
        def run(item: int) -> int:
            if item == STAGES - 1 - index:  # item m reaches stage S-1-m on wave S-1
                all_four.wait()
            return item

        return run

    with StagePipeline([stage(s) for s in range(STAGES)]) as pipe:
        assert pipe.run(list(range(STAGES))) == list(range(STAGES))


def test_a_stage_failure_reaches_the_caller_and_leaves_the_pipeline_usable() -> None:
    def boom(item: int) -> int:
        if item == 1:
            raise ValueError("stage 0 failed")
        return item

    seen: list[int] = []
    with StagePipeline([boom, lambda v: seen.append(v) or v]) as pipe:
        with pytest.raises(ValueError, match="stage 0 failed"):
            pipe.run([0, 1, 2])
        assert seen == [0, 2]  # the failed item is carried past later stages, not run by them
        assert pipe.run([5]) == [5]  # nothing was left in flight


def test_close_is_idempotent_and_refuses_later_runs() -> None:
    pipe = StagePipeline([lambda v: v])
    pipe.close()
    pipe.close()
    with pytest.raises(RuntimeError, match="closed"):
        pipe.run([1])


def test_empty_stage_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one stage"):
        StagePipeline([])


# ---------------------------------------------------------------- (2) splitting the work


def test_stage_ranges_group_the_contiguous_layer_runs() -> None:
    assert seed_model._stage_ranges([0, 0, 0, 1, 1, 2]) == [range(0, 3), range(3, 5), range(5, 6)]
    assert seed_model._stage_ranges([0] * 4) == [range(0, 4)]
    assert seed_model._stage_ranges([]) == []


def test_microbatch_cuts_cover_the_batch_exactly() -> None:
    assert seed_model._microbatch_cuts(8, 4) == [(0, 2), (2, 4), (4, 6), (6, 8)]
    assert seed_model._microbatch_cuts(7, 4) == [(0, 2), (2, 4), (4, 6), (6, 7)]
    assert seed_model._microbatch_cuts(3, 8) == [(0, 1), (1, 2), (2, 3)]  # never empty groups
    assert seed_model._microbatch_cuts(5, 1) == [(0, 5)]
    assert seed_model._microbatch_cuts(0, 4) == [(0, 0)]
    for n, groups in ((48, 8), (13, 5), (1, 4)):
        cuts = seed_model._microbatch_cuts(n, groups)
        assert [lo for lo, _ in cuts[1:]] == [hi for _, hi in cuts[:-1]]
        assert cuts[0][0] == 0 and cuts[-1][1] == n


def test_the_model_splits_its_layers_into_one_stage_per_device(checkpoint: Path) -> None:
    model = build(checkpoint, max_batch=2)
    assert len(model.stages) == STAGES
    assert [i for r in model.stages for i in r] == list(range(len(model.layers)))
    assert model.pipeline is not None and model.pipeline.depth == STAGES
    assert model.microbatches == STAGES * seed_model.MICROBATCHES_PER_STAGE
    model.close()
    assert model.pipeline is None
    model.close()  # idempotent


def test_a_single_device_model_does_not_start_stage_threads(checkpoint: Path) -> None:
    model = seed_model.Model(checkpoint, ["cpu"], torch.float32, MAX_SEQ, 2)
    assert len(model.stages) == 1
    assert model.pipeline is None
    assert model.microbatches == 1


# ---------------------------------------------------------------- (3) identical output


def build(checkpoint: Path, max_batch: int) -> seed_model.Model:
    return seed_model.Model(checkpoint, DEVICES, torch.float32, MAX_SEQ, max_batch)


def prefilled(checkpoint: Path, prompts: list[list[int]]) -> seed_model.Model:
    model = build(checkpoint, max_batch=len(prompts))
    for slot, prompt in enumerate(prompts):
        model.begin(slot)
        model.prefill(slot, prompt, 0)
    return model


def without_pipeline(model: seed_model.Model) -> seed_model.Model:
    """Drop the stage threads but keep the microbatch cut: the same work, one stage at a time."""
    model.close()
    return model


def test_pipelined_decode_is_bit_identical_to_the_sequential_stages(checkpoint: Path) -> None:
    """The correctness claim. Same microbatches, same arithmetic, only the schedule differs.

    Two models because a decode step advances the DeltaNet state in place, so the comparison
    needs two identically prefilled copies rather than two calls on one.

    100 slots, not 8: at the current `MIN_MICROBATCH_SLOTS` default (49), every batch size up
    to 48 -- the deployed `max_batch` -- is undivided (see test_opt_decode_microbatch.py and
    MIN_MICROBATCH_SLOTS's docstring for why), which would make this a no-op schedule
    comparison at a smaller batch. 100 slots keep a real 2-microbatch split in play.
    """
    prompts = [prompt_of(200 + i, 4 + i % 8) for i in range(100)]
    slots, positions = list(range(len(prompts))), [len(p) for p in prompts]
    tokens = [p[-1] for p in prompts]

    piped = prefilled(checkpoint, prompts)
    serial = without_pipeline(prefilled(checkpoint, prompts))
    assert piped.microbatches > 1 and piped.pipeline is not None
    assert serial.microbatches == piped.microbatches
    assert len(piped._microbatches(slots, positions, torch.zeros(len(prompts), 1, 1))) > 1

    for step in range(3):  # several steps: state carried between them must match too
        want = serial.decode(slots, tokens, positions)
        got = piped.decode(slots, tokens, positions)
        assert torch.equal(got, want), f"step {step} diverged from the sequential schedule"
        tokens = got.argmax(-1).tolist()
        positions = [p + 1 for p in positions]
    piped.close()


def test_the_microbatch_cut_only_moves_the_logits_by_float_reassociation(checkpoint: Path) -> None:
    """The one thing pipelining does not preserve bitwise, stated and bounded.

    Cutting a step into microbatches is not a schedule change: `Model.moe` sorts the call's
    (token, expert) pairs into per-expert groups padded to the call's own largest group, so a
    narrower call is a differently shaped pair of `bmm`s over the same products. The reduction
    is still over `hidden` in both, so only float32 reassociation moves, by the same mechanism
    and the same magnitude that test_batched_decode.py already records for batch composition.
    Observed here: 1.4e-5 on logits of magnitude 8, with every row's argmax unchanged.

    100 slots, not 8, for the same reason as the pipelined-vs-sequential test above: below
    `MIN_MICROBATCH_SLOTS` (49) a step no longer splits, which would make `cut` and `whole`
    identical rather than exercising the reassociation this test is about.
    """
    prompts = [prompt_of(205 + i, 4 + i % 8) for i in range(100)]
    slots, positions = list(range(len(prompts))), [len(p) for p in prompts]
    tokens = [p[-1] for p in prompts]

    cut = without_pipeline(prefilled(checkpoint, prompts))
    whole = without_pipeline(prefilled(checkpoint, prompts))
    whole.microbatches = 1
    assert cut.microbatches > 1
    assert len(cut._microbatches(slots, positions, torch.zeros(len(prompts), 1, 1))) > 1

    split, undivided = cut.decode(slots, tokens, positions), whole.decode(slots, tokens, positions)
    torch.testing.assert_close(split, undivided, atol=1e-4, rtol=1e-4)
    assert torch.equal(split.argmax(-1), undivided.argmax(-1))


def test_pipelined_decode_leaves_identical_slot_state(checkpoint: Path) -> None:
    """Slot isolation: no microbatch may write another's KV rows or DeltaNet state."""
    prompts = [prompt_of(210 + i, 3 + 2 * i) for i in range(6)]
    slots, positions = list(range(len(prompts))), [len(p) for p in prompts]

    piped = prefilled(checkpoint, prompts)
    serial = without_pipeline(prefilled(checkpoint, prompts))
    piped.decode(slots, [p[-1] for p in prompts], positions)
    serial.decode(slots, [p[-1] for p in prompts], positions)

    for i, (a, b) in enumerate(zip(piped.pool, serial.pool, strict=True)):
        for name in a:
            assert torch.equal(a[name], b[name]), f"layer {i} state {name}"
    piped.close()


def test_a_slot_is_unaffected_by_the_other_slots_in_its_step(checkpoint: Path) -> None:
    """One slot's decode through the pipeline matches that slot decoded on its own."""
    prompts = [prompt_of(220 + i, 4 + i) for i in range(8)]
    positions = [len(p) for p in prompts]

    batched = prefilled(checkpoint, prompts)
    together = batched.decode(list(range(len(prompts))), [p[-1] for p in prompts], positions)

    alone = prefilled(checkpoint, prompts)
    for slot, prompt in enumerate(prompts):
        alone.bind(slot)
        row = alone.forward(torch.tensor([[prompt[-1]]]), positions[slot])[-1]
        torch.testing.assert_close(together[slot], row, atol=1e-4, rtol=1e-4, msg=f"slot {slot}")
    batched.close()


def test_batched_decode_does_not_touch_the_bind_cursor(checkpoint: Path) -> None:
    """`self.state` is one shared cursor. Several microbatches are inside the model at once,
    on different threads, so the batched path must read `slot_state` and leave `state` alone.
    """
    prompts = [prompt_of(230 + i, 5 + i) for i in range(4)]
    model = prefilled(checkpoint, prompts)
    model.bind(0)
    before = model.state
    model.decode(list(range(len(prompts))), [p[-1] for p in prompts], [len(p) for p in prompts])
    assert model.state is before, "batched decode moved the single-sequence bind cursor"
    model.close()


# ---------------------------------------------------------------- (4) end to end


def reference_tokens(model: seed_model.Model, prompt: list[int], max_new: int) -> list[int]:
    """One sequence at a time on a single-device model: the behavior the server must keep."""
    return list(model.generate(prompt, max_new, 0.0, frozenset()))


def serve(sched: Scheduler, specs: list[tuple[list[int], int]]) -> list[Sink]:
    sinks = [Sink() for _ in specs]
    for (prompt, max_new), sink in zip(specs, sinks, strict=True):
        sched.submit(Request(list(prompt), max_new, 0.0, frozenset(), sink))
    drain(sched)
    for sink in sinks:
        assert sink.error is None, sink.error
    return sinks


def test_the_scheduler_over_a_pipelined_model_matches_serial_generation(
    checkpoint: Path, serial_model: seed_model.Model
) -> None:
    specs = [(prompt_of(240 + i, 5 + i), 6) for i in range(5)]
    expected = [reference_tokens(serial_model, p, n) for p, n in specs]

    model = build(checkpoint, max_batch=3)  # fewer slots than requests: reuse and eviction too
    sched = make_scheduler(model, prefill_chunk=4)
    for i, (sink, want) in enumerate(zip(serve(sched, specs), expected, strict=True)):
        assert sink.tokens == want, f"request {i} diverged under the pipelined model"
    model.close()


def test_prefix_reuse_still_holds_across_pipelined_steps(
    checkpoint: Path, serial_model: seed_model.Model
) -> None:
    """The prefix snapshot is taken between steps, so pipelining must not disturb it."""
    turns = [prompt_of(250, 5)]
    turns.append(turns[0] + prompt_of(251, 6))
    turns.append(turns[1] + prompt_of(252, 4))
    expected = [reference_tokens(serial_model, p, 5) for p in turns]

    model = build(checkpoint, max_batch=2)
    sched = make_scheduler(model, prefill_chunk=4)
    for turn, (prompt, want) in enumerate(zip(turns, expected, strict=True)):
        (sink,) = serve(sched, [(prompt, 5)])
        assert sink.tokens == want, f"turn {turn} diverged when served from the prefix cache"
    model.close()


def test_a_stage_failure_is_reported_to_every_request_in_the_step(checkpoint: Path) -> None:
    """A raising layer must fail the batch through the scheduler, not hang the stage threads."""
    model = build(checkpoint, max_batch=2)
    sched = make_scheduler(model, prefill_chunk=4)
    sinks = [Sink() for _ in range(2)]
    for i, sink in enumerate(sinks):
        sched.submit(Request(prompt_of(260 + i, 4), 20, 0.0, frozenset(), sink))
    while len(sched.decoding) < len(sinks):  # get both requests past prefill and into decode
        assert sched.step()

    failing = model.moe

    def boom(i: int, x: torch.Tensor, out_buf: torch.Tensor | None = None) -> torch.Tensor:
        if i == len(model.layers) - 1:  # a layer on the last stage
            raise RuntimeError("stage blew up")
        return failing(i, x, out_buf)

    model.moe = boom
    sched.step()
    assert all("stage blew up" in (s.error or "") for s in sinks), [s.error for s in sinks]
    model.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

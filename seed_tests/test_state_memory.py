"""Steady-state memory: serving a turn must not allocate persistent state.

The slot pool is fixed size and decode writes into it in place, so a server that has
answered a thousand turns should hold exactly what it held after the first one. Two levels
check that:

(a) torch-free, over the real `Scheduler` and `PromptCache`: after 250 turns across more
    sessions than there are slots, no per-request object is still referenced, the slot pool
    is still `max_batch` entries, and the tokenization cache is still within capacity.
(b) with torch, over the real `Model`: the bytes of live tensor storage after 250 turns
    equal the bytes after the first turn. This is the one that fails on the pre-fix code,
    where `Model` allocated a slot's prefix snapshot on that slot's first `save_prefix`
    instead of with the rest of the state pool, so the resident footprint grew by one
    slot's recurrent state for every session admitted until all `max_batch` slots had
    been used.

Neither test says anything about real HIP bytes: they pin the object lifecycle, not the
device allocator. Only a cluster run can confirm the device footprint is actually flat.

    python3 -m pytest .../seed_tests/test_state_memory.py         # (a) only
    <python-with-torch> -m pytest .../seed_tests/test_state_memory.py
"""

import gc
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import block_pool  # noqa: E402
from prompt_cache import PromptCache  # noqa: E402
from scheduler import Request, Scheduler  # noqa: E402
from session_cache import SessionCache  # noqa: E402
from test_scheduler import FakeRunner, Sink, drain, serial  # noqa: E402

NUM_BLOCKS = 4096
BLOCK_SIZE = 4
NUM_SNAPSHOTS = 256

TURNS = 250
SESSIONS = 12
MAX_NEW = 3


def conversation(base: int, turn: int) -> list[int]:
    """Turn `turn` of session `base`: what the session said before, plus a new tail.

    Ids stay under 100 so the same prompts drive the tiny real model in (b); the
    per-session head keeps one session's prompts from being a prefix of another's.
    """
    prompt = [base, base + 1, base + 2]
    for k in range(turn + 1):
        prompt += [(base * 7 + k * 11 + j) % 70 + 25 for j in range(3)]
    return prompt


def turns(count: int = TURNS):
    """(session base, prompt) for `count` turns, round-robin over `SESSIONS` sessions."""
    for i in range(count):
        base = 2 + i % SESSIONS
        yield base, conversation(base, i // SESSIONS)


def live_instances(name: str) -> int:
    """How many objects of class `name` are still referenced anywhere."""
    gc.collect()
    return sum(1 for obj in gc.get_objects() if type(obj).__name__ == name)


def serve(sched: Scheduler, prompt: list[int]) -> Sink:
    sink = Sink()
    sched.submit(Request(list(prompt), MAX_NEW, 0.0, frozenset(), sink))
    drain(sched)
    assert sink.error is None, sink.error
    return sink


# ---------------------------------------------------------------- (a) torch-free


class SlotAudit(FakeRunner):
    """Records which lanes the scheduler ever addressed."""

    def __init__(
        self, max_batch: int, allocator: block_pool.BlockAllocator, block_size: int
    ) -> None:
        super().__init__(max_batch, allocator, block_size)
        self.touched: set[int] = set()

    def prefill(self, slot: int, ids: list[int], start: int) -> list[int]:
        self.touched.add(slot)
        return super().prefill(slot, ids, start)


def test_many_turns_leave_no_per_request_state_referenced() -> None:
    """250 turns over 12 sessions and 4 lanes, checked for accumulation after every turn."""
    max_batch = 4
    allocator = block_pool.BlockAllocator(NUM_BLOCKS, BLOCK_SIZE)
    cache = SessionCache(allocator, BLOCK_SIZE, NUM_SNAPSHOTS)
    runner = SlotAudit(max_batch, allocator, BLOCK_SIZE)
    sched = Scheduler(runner, cache, prefill_chunk=4)
    in_flight = []

    for _, prompt in turns():
        assert serve(sched, prompt).tokens == serial(prompt, MAX_NEW)
        in_flight.append(live_instances("_InFlight"))

    assert runner.touched <= set(range(max_batch)), "the scheduler addressed a lane it has not"
    assert len(sched.free_lanes) == max_batch, "the lane pool changed size"
    assert not sched.decoding and not sched.prefill_q and not sched.waiting
    assert max(in_flight) == 0, f"in-flight requests outlived their turns: {in_flight}"
    # A cached session holds one conversation's prefix, not every conversation ever served
    # (Stage 2: this is now a property of the cache, not of a lane -- see
    # session_cache.py's module docstring).
    longest = len(conversation(2, TURNS // SESSIONS))
    assert cache.lookup(conversation(2, TURNS // SESSIONS)).depth <= longest


def test_tokenization_cache_stays_within_capacity() -> None:
    """The prompt cache is the other cross-turn cache; 250 growing prompts must not grow it."""

    class Tok:
        all_special_tokens = ("<|im_end|>",)

        def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
            return {"input_ids": [ord(ch) for ch in text]}

    cache = PromptCache(Tok(), capacity=8)
    assert cache.selfcheck(["a<|im_end|>b", "a<|im_end|>b<|im_end|>c"])
    for _, prompt in turns():
        text = "".join(f"{tok}<|im_end|>" for tok in prompt)
        assert cache.encode(text) == cache.tokenize(text)
        assert len(cache.entries) <= cache.capacity, "the prompt cache grew past its bound"


# ---------------------------------------------------------------- (b) the real model


def live_storage_bytes(torch) -> int:  # noqa: ANN001
    """Bytes of distinct tensor storage still referenced anywhere in the process.

    Keyed by storage address, so the many views of one pool row count once, which is
    what device memory does too. Sweeping every live object rather than the model's
    own attributes is what makes this catch state held somewhere unexpected.
    """
    gc.collect()
    seen: dict[int, int] = {}
    with warnings.catch_warnings():  # touching deprecated torch attributes while sweeping
        warnings.simplefilter("ignore")
        for obj in gc.get_objects():
            if isinstance(obj, torch.Tensor):
                try:
                    storage = obj.untyped_storage()
                except (RuntimeError, NotImplementedError):
                    continue
                seen[storage.data_ptr()] = storage.nbytes()
    return sum(seen.values())


@pytest.fixture(scope="module")
def served(tmp_path_factory: pytest.TempPathFactory):
    """A tiny real `Model` behind a `Scheduler`, with one session's turns already served."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import model as seed_model
    import torch
    from test_seed_parity import build_hf, write_checkpoint

    out = tmp_path_factory.mktemp("tiny-state")
    write_checkpoint(build_hf(), out, mxfp4=False)
    model = seed_model.Model(out, ["cpu"], torch.float32, 512, 8)
    cache = SessionCache(model.block_allocator, model.block_size, model.num_snapshots)
    sched = Scheduler(model, cache, prefill_chunk=8)
    for turn in range(3):  # one session only: lanes 1..7 are still untouched
        serve(sched, conversation(base=99, turn=turn))
    return torch, sched


def test_serving_turns_allocates_no_persistent_tensor(served) -> None:  # noqa: ANN001
    """The footprint after 250 turns over 12 sessions equals the footprint after 3."""
    torch, sched = served
    before = live_storage_bytes(torch)
    for _, prompt in turns():
        serve(sched, prompt)
    after = live_storage_bytes(torch)
    assert after == before, f"serving allocated {after - before} bytes of persistent state"


def test_snapshot_pool_is_part_of_the_static_allocation_before_the_first_request(served) -> None:  # noqa: ANN001
    """Stage 2: the snapshot pool is sized by `num_snapshots` (independent of lane count,
    unlike Stage 1's one-snapshot-per-slot `prefix_snap`) and allocated once with the rest of
    the static state, not grown on a lane's first `save_snapshot` (see model.py's module
    docstring and `Model._new_snapshot_pool`)."""
    torch, sched = served
    model = sched.runner
    assert model.num_snapshots >= model.max_batch  # SNAPSHOT_POOL_MIN_CAPACITY_FACTOR floor
    for layer, (state, pool) in enumerate(
        zip(model.slot_state[0], model.snapshot_pool, strict=True)
    ):
        if "conv" not in state:
            assert pool is None, f"layer {layer}: KV must not be snapshotted"
            continue
        assert pool is not None, f"layer {layer}: snapshot pool allocated lazily"
        for name, t in pool.items():
            assert t.shape[0] == model.num_snapshots
            assert t.shape[1:] == state[name].shape[1:]

    # A snapshot row is a view into the pool, not a private per-call allocation.
    row = model.snapshot_row(0)
    for layer, (state, snap) in enumerate(zip(model.slot_state[0], row, strict=True)):
        if snap is None:
            continue
        for name, buf in snap.items():
            assert buf.shape == state[name].shape
            assert buf.data_ptr() != state[name].data_ptr()
            assert buf._base is model.snapshot_pool[layer][name]  # noqa: SLF001


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

"""Snapshot reclaim before leaf eviction (`session_cache.py`'s module docstring).

Round 15 bug: each chat turn published its boundary as a child of the previous turn's, and every
internal node kept its DeltaNet snapshot although only leaves are ever resumed. A k-turn
conversation held k slots, the pool overflowed at C96, and LRU evicted the leaves of
conversations resting between turns: 51% of turn-2+ requests reused nothing.

    cd seed_tests && PYTHONPATH=. python3 -m pytest test_snapshot_reclaim.py
"""

import sys
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import block_pool  # noqa: E402
from scheduler import Scheduler  # noqa: E402
from session_cache import Node, SessionCache  # noqa: E402
from test_scheduler import BLOCK_SIZE, FakeRunner, drain, request, serial  # noqa: E402

SUFFIX = [900, 901, 902]  # generation-prompt opening, prefilled but not part of history
ASSISTANT, CLOSE = 950, 951  # how the next prompt renders the reply around its tokens


def make_cache(num_snapshots: int, num_blocks: int = 512) -> SessionCache:
    return SessionCache(
        block_pool.BlockAllocator(num_blocks, BLOCK_SIZE), BLOCK_SIZE, num_snapshots
    )


def publish(
    cache: SessionCache, parent: Node, edge: tuple[int, ...], protect=(), resumed: bool = True
) -> Node:
    """Publish `edge` under `parent` as the scheduler does. `resumed`: a new request resumed
    from `parent` (`adopt`), as opposed to the request that published `parent` itself
    extending it (a turn-close node under its own prompt-end node)."""
    if resumed and not parent.is_root():
        cache.release_lane_blocks(cache.adopt(parent))
    snap = cache.reserve_snapshot(protect=[parent, *protect])
    depth = parent.depth + len(edge)
    need = -(-depth // BLOCK_SIZE) - len(parent.blocks)
    blocks = (*parent.blocks, *cache.reserve_blocks(max(need, 0), protect=[parent, *protect]))
    node = cache.publish(parent, edge, blocks, snap, own=not resumed)
    cache.release_lane_blocks(blocks[len(parent.blocks) :])
    return node


def span(node: Node) -> list[int]:
    out: list[int] = []
    while not node.is_root():
        out[:0] = node.edge
        node = node.parent
    return out


def test_resting_leaf_survives_when_an_internal_snapshot_is_reclaimable() -> None:
    """Regression: the old cache evicted conversation A's newest boundary (a resting leaf)
    while A's superseded first-turn boundary still held a snapshot."""
    cache = make_cache(num_snapshots=3)
    a1 = publish(cache, cache.root, (1, 2, 3, 4))
    a2 = publish(cache, cache.lookup([1, 2, 3, 4, 5]), (5, 6, 7, 8))
    b1 = publish(cache, cache.root, (9, 10, 11, 12))
    assert cache.free_snapshot_count() == 0

    publish(cache, b1, (13, 14))  # B's next turn needs a slot while b1 is protected

    assert cache.lookup(span(a2) + [99]) is a2, "A's resting leaf was evicted"
    assert a1.snapshot == -1, "the superseded internal snapshot should have been reclaimed"
    assert cache.lookup([1, 2, 3, 4, 99]) is cache.root, "a stateless node is never resumed"
    assert cache.snapshot_stats().reclaims == 1
    assert cache.snapshot_stats().pressure_evictions == 0


def test_leaf_eviction_removes_stateless_ancestors_it_exposes() -> None:
    cache = make_cache(num_snapshots=2)
    a1 = publish(cache, cache.root, (1, 2, 3, 4))
    a2 = publish(cache, a1, (5, 6, 7, 8))
    publish(cache, cache.root, (9, 10, 11, 12), protect=[a2])  # reclaims a1's snapshot
    assert a1.snapshot == -1
    free_blocks = cache.allocator.free_count
    publish(cache, cache.root, (20, 21, 22, 23), protect=[])  # evicts a2, then a1 with it
    assert cache.node_count() == 3, "root + two single-node conversations"
    assert cache.allocator.free_count >= free_blocks + 1, "a1's blocks were not released"


def classify(cache: SessionCache, prompt: list[int]) -> str:
    node, matched = cache.lookup_detail(prompt)
    return cache.classify(prompt, node, matched)


def test_classify_separates_evicted_from_cold_misses() -> None:
    cache = make_cache(num_snapshots=1)
    prompt = list(range(100, 140))
    publish(cache, cache.root, tuple(prompt))
    assert classify(cache, prompt + [1]) == "hit"
    publish(cache, cache.root, tuple(range(200, 240)))  # evicts the first node
    assert classify(cache, prompt + [1]) == "miss_evicted"
    assert classify(cache, list(range(300, 340))) == "miss_cold"


# ------------------------------------------------------------------ scheduler-level chat flow


def chat_turn(sched, runner, history: list[int], user: list[int], max_new: int):
    prompt = history + user + SUFFIX
    req, sink = request(prompt, max_new, suffix_len=len(SUFFIX))
    sched.submit(req)
    return prompt, req, sink


class ChatDriver:
    """Drive chat turns through the production path (deferred turn close, folded suffix) on
    one long-lived scheduler and cache, like a server that serves several benchmark levels.

    `run(schedule)`: `schedule` is (conversation, user tokens, reply tokens) in arrival order;
    up to `batch` turns of distinct conversations run concurrently, the rest wait for the
    next wave. Conversations in `drop` are raw-token clients that drop the reply: the next
    prompt is the previous prompt plus new tokens (no suffix), so they resume the prompt-end
    node, which has the request's own turn-close node as its child. `fresh=True` starts every
    conversation over from its first turn (a new benchmark level replaying the same session
    ids); `salt` changes every user message after the first, as batch drift changes replies.
    Returns (conversation, reused, previous boundary) per turn-2+ turn."""

    def __init__(self, num_snapshots: int, batch: int) -> None:
        self.cache = make_cache(num_snapshots, num_blocks=8192)
        self.runner = FakeRunner(batch, self.cache.allocator, BLOCK_SIZE)
        self.sched = Scheduler(
            self.runner, self.cache, prefill_chunk=16, defer_turn_close=True, fold_turn_suffix=True
        )
        self.batch = batch
        self.history: dict[int, list[int]] = {}
        self.boundary: dict[int, int] = {}

    def run(self, schedule, drop=frozenset(), *, fresh: bool = False, salt: int = 0):
        if fresh:
            self.history, self.boundary = {}, {}
        reuse: list[tuple[int, int, int]] = []
        pending = list(schedule)
        while pending:
            wave, seen = [], set()
            for item in list(pending):
                if item[0] not in seen and len(wave) < self.batch:
                    wave.append(item)
                    seen.add(item[0])
                    pending.remove(item)
            started = []
            for conv, n_user, n_reply in wave:
                first = conv not in self.history
                hist = self.history.get(conv, [conv * 10_000 + j for j in range(1, 6)])
                offset = 0 if first else salt * 3_000
                user = [conv * 10_000 + 100 + offset + len(hist) + j for j in range(n_user)]
                if conv in drop:
                    prompt = hist + user
                    req, sink = request(prompt, n_reply)
                    self.sched.submit(req)
                    started.append((conv, prompt, req, sink))
                else:
                    started.append((conv, *chat_turn(self.sched, self.runner, hist, user, n_reply)))
            drain(self.sched)
            for conv, prompt, req, sink in started:
                assert sink.error is None, sink.error
                assert sink.tokens == serial(prompt, req.max_new), "reuse changed the output"
                reused = sink.end[2]
                if conv in self.boundary:
                    reuse.append((conv, reused, self.boundary[conv]))
                if conv in drop:
                    self.boundary[conv] = len(prompt)
                    self.history[conv] = prompt
                else:
                    self.boundary[conv] = len(prompt) - len(SUFFIX)
                    self.history[conv] = prompt[: -len(SUFFIX)] + [ASSISTANT, *sink.tokens, CLOSE]
        return reuse


def run_conversations(schedule, num_snapshots: int, batch: int, drop: frozenset = frozenset()):
    return ChatDriver(num_snapshots, batch).run(schedule, drop)


def test_replayed_first_turn_is_not_shadowed_by_its_reclaimed_twin() -> None:
    """Regression (stacked run, C96 then C16 on one server): the next level replays the same
    first turns. Its turn-1 prompt matched the old, reclaimed turn-1 node, fell back to the
    root, and published an identical sibling edge; lookup took the first matching sibling,
    the stateless old one, so every later turn of that conversation missed its boundary."""
    driver = ChatDriver(num_snapshots=4, batch=2)
    driver.run([(conv, 6, 4) for _ in range(4) for conv in range(3)])  # high-load level
    assert driver.cache.snapshot_stats().reclaims > 0, "setup: turn-1 nodes must be reclaimed"
    reuse = driver.run([(conv, 6, 4) for _ in range(3) for conv in range(2)], fresh=True, salt=1)
    assert [r for r in reuse if r[1] < r[2]] == [], "a replayed conversation missed its boundary"


def test_pool_full_of_stale_sessions_serves_a_new_low_concurrency_level() -> None:
    """A pool left full by an earlier high-concurrency level: the new level's conversations
    (new session ids) must evict the stale sessions, never each other."""
    driver = ChatDriver(num_snapshots=6, batch=4)
    driver.run([(conv, 6, 4) for _ in range(3) for conv in range(8)])
    reuse = driver.run([(conv, 5, 3) for _ in range(4) for conv in range(20, 23)], fresh=True)
    assert [r for r in reuse if r[1] != r[2]] == []


@settings(max_examples=100, deadline=None)
@given(
    data=st.data(),
    first=st.integers(2, 6),
    second=st.integers(1, 3),
    replay=st.booleans(),
    salt=st.integers(0, 2),
)
def test_second_level_on_a_stale_pool_keeps_every_boundary(
    data, first: int, second: int, replay: bool, salt: int
) -> None:
    """Level 1 fills the pool with any number of conversations; level 2 (same session ids
    replayed, or new ones) runs `second` conversations with `second + 1` slots: every level-2
    turn-2+ resumes at least its previous boundary (a replay may match deeper stale nodes)."""
    driver = ChatDriver(num_snapshots=second + 1, batch=2)
    level1 = data.draw(
        st.lists(
            st.tuples(st.integers(0, first - 1), st.integers(1, 8), st.integers(1, 4)),
            min_size=1,
            max_size=20,
        )
    )
    driver.run(level1)
    base = 0 if replay else 100
    level2 = data.draw(
        st.lists(
            st.tuples(st.integers(base, base + second - 1), st.integers(1, 8), st.integers(1, 4)),
            min_size=1,
            max_size=16,
        )
    )
    reuse = driver.run(level2, fresh=True, salt=salt)
    assert [r for r in reuse if r[1] < r[2]] == [], "a level-2 turn missed its boundary"


def test_chat_conversations_keep_their_boundary_with_one_spare_slot() -> None:
    """Regression for the C96 misses: 4 conversations, 5 turns each, 5 slots. The old cache
    kept every turn's snapshot and evicted resting conversations' boundaries."""
    schedule = [(conv, 7, 5) for _ in range(5) for conv in range(4)]
    reuse = run_conversations(schedule, num_snapshots=5, batch=2)
    assert [r for r in reuse if r[1] != r[2]] == [], "a turn missed its previous boundary"


def test_client_that_drops_the_reply_keeps_its_prompt_end_node() -> None:
    """A raw client resends its previous prompt plus new tokens. Its prompt-end node is
    internal (the request's own turn-close node is its child) but not superseded, so chat
    conversations' superseded snapshots must be reclaimed before it."""
    schedule = [(conv, 6, 4) for _ in range(5) for conv in range(3)]
    reuse = run_conversations(schedule, num_snapshots=5, batch=2, drop=frozenset({0}))
    assert [r for r in reuse if r[1] != r[2]] == [], "a turn missed its previous boundary"


@settings(max_examples=150, deadline=None)
@given(
    data=st.data(),
    conversations=st.integers(1, 5),
    spare=st.integers(1, 3),
    batch=st.integers(1, 4),
)
def test_every_turn_resumes_its_previous_boundary_when_slots_cover_conversations(
    data, conversations: int, spare: int, batch: int
) -> None:
    """Any arrival order (idle times are the gaps between a conversation's turns), any user
    and reply lengths, chat clients and reply-dropping raw clients mixed: with one slot per
    chat conversation, two per raw one (prompt-end plus its turn-close child), plus one,
    every turn-2+ reuses exactly its previous boundary, and outputs match serial
    generation."""
    drop = frozenset(data.draw(st.sets(st.integers(0, conversations - 1))))
    schedule = data.draw(
        st.lists(
            st.tuples(st.integers(0, conversations - 1), st.integers(1, 12), st.integers(1, 6)),
            min_size=1,
            max_size=24,
        )
    )
    slots = conversations + len(drop) + spare
    reuse = run_conversations(schedule, num_snapshots=slots, batch=batch, drop=drop)
    assert [r for r in reuse if r[1] != r[2]] == []


@settings(max_examples=300, deadline=None)
@given(
    ops=st.lists(
        st.tuples(
            st.integers(0, 4),  # conversation
            st.integers(1, 6),  # edge length
            st.sampled_from(["follow", "close", "drop"]),
            st.booleans(),  # also protect another conversation's tip
        ),
        max_size=40,
    )
)
def test_superseded_snapshots_go_first(ops) -> None:
    """Random chain operations over a small pool: `follow` resumes a conversation's newest
    node and publishes past it, `close` publishes a turn-close child from the same request,
    `drop` resumes the prompt-end node under the newest turn-close (a client that dropped the
    reply). Whenever a reservation evicts a leaf or reclaims a non-superseded snapshot, no
    unprotected superseded internal node may still hold one. Slot accounting stays exact and
    no stateless leaf is left behind."""
    cache = make_cache(num_snapshots=4, num_blocks=1024)
    tips: dict[int, Node] = {}
    for conv, n, kind, protect_other in ops:
        tip = tips.get(conv)
        if tip is not None and not cache.is_live(tip):
            tip = None
        parent = tip or cache.root
        if kind == "drop" and tip is not None and tip.parent is not None and tip.parent.has_state():
            parent = tip.parent
        others = [t for c, t in tips.items() if c != conv and cache.is_live(t)]
        protect = [parent, *others[:1]] if protect_other else [parent]
        first_tier = [
            node
            for node in cache._iter_nodes(cache.root)
            if not node.is_root()
            and (
                (node.superseded and not node.is_leaf() and node.snapshot >= 0)
                or (node.bypassed and node.is_leaf())
            )
            and all(node is not p for p in protect)
        ]
        nodes_before = {id(node): node for node in cache._iter_nodes(cache.root)}
        state_before = {id(node): node.snapshot >= 0 for node in nodes_before.values()}
        try:
            edge = tuple(conv * 1000 + (parent.depth + j) for j in range(n))
            tips[conv] = publish(cache, parent, edge, protect=protect[1:], resumed=kind != "close")
        except Exception:  # noqa: BLE001 -- exhaustion is allowed when everything is protected
            continue
        after = {id(node): node for node in cache._iter_nodes(cache.root)}
        lost = [
            nodes_before[i]
            for i, had in state_before.items()
            if had
            and not nodes_before[i].is_root()
            and (i not in after or after[i].snapshot < 0)
            and not nodes_before[i].superseded
            and not nodes_before[i].bypassed
        ]
        if lost:
            assert not first_tier, "lost a live boundary while a first-tier victim remained"
        live = [node for node in after.values() if not node.is_root()]
        held = [node.snapshot for node in live if node.snapshot >= 0]
        assert len(set(held)) == len(held), "two nodes share a snapshot slot"
        assert sorted(held + cache._free_snapshots) == list(range(cache.num_snapshots))
        assert all(node.has_state() for node in live if node.is_leaf()), "stateless leaf left"

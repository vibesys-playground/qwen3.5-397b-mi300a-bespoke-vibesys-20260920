"""Prefix trie + session directory, decoupled from batch lanes (paged-kv-design.md, Stage 2:
section 1.5 "Batch lanes decoupled from cached sessions", section 2 "Prefix cache", section
1.7 "Eviction policy"). Torch-free, host-side bookkeeping only -- the same posture
`block_pool.py` already takes (see its own module docstring's "Stage 1 vs Stage 2" section,
which this module is the other half of): exercised directly by CPU-hermetic tests
(`seed_tests/test_session_cache.py`), driven for real by `scheduler.py`, which is the only
caller. `model.py` never imports this module; it only implements the snapshot-pool and
block-copy primitives `scheduler.py` calls while orchestrating a `SessionCache`.

What a "session" is here. Not a scheduler request or a batch lane -- a **snapshot boundary**:
a token prefix at which some earlier request's turn-open or turn-close point (paged-kv-design.md
section 2.3) was recorded, together with the paged KV blocks and DeltaNet/conv state needed to
resume generation from exactly that point. A `Node` is one such boundary. The trie is a tree of
`Node`s branching at boundary points, not at arbitrary tokens or block boundaries (the design
doc's own "a trie or hash over snapshot-boundary prefixes is enough" simplification, since a
DeltaNet snapshot only exists at these points anyway -- there is nothing resumable in between to
branch on).

Block ownership. `Node.blocks` is the *complete* flat block-id list covering tokens
`[0, node.depth)`, mirroring `block_pool.BlockTable`'s own flat-list convention (a lane that
adopts a node can use `blocks` directly as its working table, no ancestor walk needed at decode
time). Every id in `blocks` holds one `BlockAllocator` reference on this node's behalf; two nodes
that share an ancestor's tokens each hold their own reference on the shared ids, which is exactly
what makes `incref`/`decref` on `BlockAllocator` meaningful in Stage 2 (see `block_pool.py`).
A lane that merely *borrows* a node's blocks while resuming (before anything new is published)
takes its own temporary reference too (`SessionCache.adopt`), so an unrelated eviction elsewhere
in the trie can never free blocks a live lane is still reading or extending -- refcounting alone
guarantees this, no separate "pinned" bookkeeping is needed (see `adopt`'s docstring).

Copy-on-write. A node's last block is "partial" when `node.depth` is not a multiple of
`block_size`: decode only ever appends new tokens into a block's first still-empty rows, so two
independent continuations from the same partial-last-block node would corrupt each other's
writes if they shared that physical block. `needs_cow` reports this; the caller (`scheduler.py`)
performs the actual device-side copy (a `Runner` call) and passes the resulting fresh block id
back into `adopt`/`publish` in place of the old one. This module never touches tensors -- it only
decides *whether* a copy is needed and does the id-level bookkeeping once the caller supplies the
new id.

Eviction. Two coupled LRU-managed resources per paged-kv-design.md section 1.6/1.7: KV blocks
(`BlockAllocator`, cheap and plentiful) and DeltaNet snapshot slots (`num_snapshots`, the scarce
one). Only a *leaf* node (no children) is ever evicted: an internal node's blocks are still
needed by its descendants' attention context even if its own snapshot is stale, so evicting a
node with children would corrupt every session still resolving through it. Evicting a leaf drops
its own block references and frees its snapshot slot; its parent, now possibly childless, simply
becomes eligible in a later pass.

Snapshot reclaim before eviction. Only a node's *own* snapshot is ever loaded (admission resumes
from the deepest match), so an internal node's snapshot is dead weight once a child extends it:
the child already carries the same blocks and a deeper state. A multi-turn chat publishes one
boundary per turn, each a child of the previous one, so a k-turn conversation held k snapshot
slots while only its leaf was resumable. Under C96 load 192 slots overflowed and LRU evicted the
leaves of conversations resting between turns, then their ancestors one by one: 51% of turn-2+
requests found nothing (round 15, W16/W18; the miss rate rose from 7% at 0-2 s idle to 87% at
10-12 s). `reserve_snapshot` therefore first *reclaims* the snapshot of the LRU
*superseded* internal node (`snapshot = -1`, blocks and trie position kept), and only then falls
back to plain LRU over leaves and the remaining internal snapshots. Superseded means a request
that resumed from the node (`adopt`) later published a deeper node under it: that request's
conversation has provably moved past it. A child published by the same request that published
the node (a turn-close node under its own prompt-end node) is not such evidence: a client may
drop the reply and resend the previous prompt plus new text, and that client needs the
prompt-end node. That turn-close child is instead marked *bypassed* once a later request resumes
from its parent and publishes elsewhere, and a bypassed leaf is evicted in the first tier too. `lookup` skips stateless nodes (it returns the deepest match that still has a state),
and a stateless node that becomes a leaf is removed with the child whose eviction exposed it,
since nothing can resume from it and it only pins blocks. The effect is about one live state per
conversation: the lane's state at the end of its last published boundary.
"""

from __future__ import annotations

import os
from collections import Counter, OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field

import block_pool


class SessionCacheError(Exception):
    """No admission is possible even after evicting everything evictable."""


@dataclass(frozen=True, slots=True)
class SnapshotStats:
    """Snapshot-pool occupancy and cumulative pressure counters for diagnostics."""

    capacity: int
    used: int
    high_watermark: int
    pressure_evictions: int
    reservation_failures: int
    reclaims: int = 0  # internal-node snapshots freed before any leaf is evicted


SNAPSHOT_RECLAIM = os.environ.get("SEED_SNAPSHOT_RECLAIM", "1") not in ("0", "", "false", "False")
"""`SEED_SNAPSHOT_RECLAIM=0` restores leaf-only eviction (internal nodes keep their snapshots),
the pre-round-15 behavior, for A/B runs. See the module docstring."""

EVICTED_INDEX_ENTRIES = 8192
"""Evicted/reclaimed spans remembered for miss classification (`classify`)."""
EVICTED_KEY_TOKENS = 32
"""Spans are indexed by their first this many tokens (shorter spans are not indexed)."""


@dataclass(slots=True)
class Node:
    """One snapshot boundary. See the module docstring for what each field owns."""

    parent: Node | None
    edge: tuple[int, ...]  # tokens from parent.depth to this node's own depth; () only for root
    depth: int
    blocks: tuple[int, ...]  # flat block-id list covering [0, depth); () only for depth 0
    snapshot: (
        int  # slot id into the runner's DeltaNet-snapshot and cached-logits pools; -1 for the
        # root and for an internal node whose snapshot was reclaimed (see the module docstring)
    )
    children: dict[int, list[Node]] = field(default_factory=dict)  # keyed by child.edge[0]
    used: int = 0  # LRU stamp, see SessionCache._touch
    adopted: bool = False  # some admission resumed from this node (`SessionCache.adopt`)
    superseded: bool = False  # a request that resumed here published a deeper node under it
    own_child: bool = False  # published by the request that also published its parent
    bypassed: bool = False  # an own child whose parent a later request extended elsewhere

    def is_leaf(self) -> bool:
        return not self.children

    def is_root(self) -> bool:
        return self.parent is None

    def has_state(self) -> bool:
        """Whether a request can resume from this node (the root resumes from zero state)."""
        return self.parent is None or self.snapshot >= 0


class SessionCache:
    """Owns the trie root, the shared `BlockAllocator`, and the snapshot-slot free list.

    `block_allocator` is shared with `model.py`'s paged full-attention pool (the same instance
    Stage 1 already builds); this class only ever calls `alloc`/`incref`/`decref` on it, never
    constructs its own. `num_snapshots` is Stage 2's new, independently-sized budget (paged-kv-
    design.md section 1.6): the DeltaNet-state and cached-logits pools `model.py` builds are both
    `[num_snapshots, ...]`, addressed by the same id this class hands out.
    """

    def __init__(
        self,
        block_allocator: block_pool.BlockAllocator,
        block_size: int,
        num_snapshots: int,
        reclaim: bool = SNAPSHOT_RECLAIM,
    ) -> None:
        if num_snapshots < 1:
            raise ValueError(f"session cache needs at least 1 snapshot slot, got {num_snapshots}")
        self.allocator = block_allocator
        self.block_size = block_size
        self.num_snapshots = num_snapshots
        self.reclaim = reclaim
        self.root = Node(parent=None, edge=(), depth=0, blocks=(), snapshot=-1)
        self._free_snapshots: list[int] = list(range(num_snapshots))
        self._snapshot_high_watermark = 0
        self._snapshot_pressure_evictions = 0
        self._snapshot_reservation_failures = 0
        self._snapshot_reclaims = 0
        self.stats: Counter[str] = Counter()  # diagnostic counters, see `classify`
        # first-EVICTED_KEY_TOKENS tokens -> [(depth, hash of the full span)], LRU-bounded
        self._evicted: OrderedDict[tuple[int, ...], list[tuple[int, int]]] = OrderedDict()
        self._stamp = 0
        self._touch(self.root)

    # -- lookup ---------------------------------------------------------------
    def lookup(self, prompt: Sequence[int]) -> Node:
        """The deepest node with a resumable state whose full token span (root to node) is a
        prefix of `prompt`.

        Never returns a node deeper than `len(prompt)`; returns `self.root` (depth 0, no
        snapshot) when nothing matches, exactly the "start from scratch" case Stage 1's
        `_match_prefix` returning `None` already meant. Always a full token-exact compare on
        every candidate edge (paged-kv-design.md section 2.1: "keep it token-exact including a
        full compare on hash hit"); the `edge[0]`-keyed bucket only narrows which candidates are
        worth comparing at all. A matched node whose snapshot was reclaimed is walked through
        but not returned; see `lookup_detail`.
        """
        return self.lookup_detail(prompt)[0]

    def lookup_detail(self, prompt: Sequence[int]) -> tuple[Node, int]:
        """`(lookup(prompt), depth of the deepest matching node, stateless or not)`.

        Searches every matching sibling, not just the first: two siblings can carry the same
        edge (an old node whose snapshot was reclaimed, and the same boundary published again
        by a request that therefore resumed from further up), and taking the first would
        return the stateless one's subtree and hide the live twin."""
        best = self.root
        deepest = 0
        n = len(prompt)
        stack = [(self.root, 0)]
        while stack:
            node, pos = stack.pop()
            deepest = max(deepest, node.depth)
            if node.has_state() and node.depth > best.depth:
                best = node
            if pos >= n:
                continue
            for child in node.children.get(prompt[pos], ()):
                span = len(child.edge)
                if pos + span <= n and tuple(prompt[pos : pos + span]) == child.edge:
                    stack.append((child, pos + span))
        self._touch(best)
        return best, deepest

    def classify(self, prompt: Sequence[int], node: Node, matched_depth: int) -> str:
        """Why an admission resumed at `node` (from `lookup_detail`), for diagnostics.

        `hit`: nothing deeper was ever cached, as far as this cache remembers. `short_reclaimed`
        / `miss_reclaimed`: a deeper node matched but its snapshot had been reclaimed.
        `short_evicted` / `miss_evicted`: a deeper boundary was published and later evicted
        (remembered in a bounded index of evicted spans). `miss_cold`: no match and nothing
        evicted matches either (first turn, a boundary never published, or a prompt that
        diverges from every cached span)."""
        kind = "miss" if node.is_root() else "short"
        if matched_depth > node.depth:
            return f"{kind}_reclaimed"
        if len(prompt) >= EVICTED_KEY_TOKENS:
            for depth, digest in self._evicted.get(tuple(prompt[:EVICTED_KEY_TOKENS]), ()):
                if node.depth < depth <= len(prompt) and hash(tuple(prompt[:depth])) == digest:
                    return f"{kind}_evicted"
        return "hit" if not node.is_root() else "miss_cold"

    # -- admission (borrowing a node's blocks onto a lane) ---------------------
    def needs_cow(self, node: Node) -> bool:
        """Whether resuming from `node` requires copy-on-write of its last block first.

        True exactly when `node.depth` is not a multiple of `block_size`: the last physical
        block still has empty rows a new continuation would write into, and that block may
        already be shared with `node`'s own ancestor chain or a sibling continuation.
        """
        return bool(node.blocks) and node.depth % self.block_size != 0

    def adopt(self, node: Node, replacement_last_block: int | None = None) -> tuple[int, ...]:
        """Borrow `node`'s blocks onto a lane's own working table, incref'd for the lane's own
        (temporary) ownership share, independent of `node`'s own share and of whatever share a
        later `publish` from this same admission takes.

        This is what makes an unrelated eviction elsewhere in the trie safe: once this returns,
        the returned ids are resident at a refcount that accounts for the lane's use, so nothing
        else can free them out from under it, even if `node` itself is later evicted (the trie
        entry disappears; the physical blocks the lane already incref'd do not).

        `replacement_last_block` is the fresh id `needs_cow(node)` required the caller to
        obtain (a real device copy, done by the runner) in place of `node.blocks[-1]`; passing it
        skips incref'ing the old partial block (the fresh one already holds its own reference
        from `reserve_blocks`) while still incref'ing everything before it.
        """
        if replacement_last_block is not None:
            inherited = node.blocks[:-1]
            if inherited:
                self.allocator.incref(inherited)
            lane_blocks = (*inherited, replacement_last_block)
        else:
            if node.blocks:
                self.allocator.incref(node.blocks)
            lane_blocks = node.blocks
        node.adopted = True
        self._touch(node)
        return lane_blocks

    def release_lane_blocks(self, blocks: Sequence[int]) -> None:
        """Drop a lane's own ownership share of `blocks` (its full working table, inherited plus
        whatever it grew) when the lane is freed. Idempotent with `publish`: a node created from
        (part of) this same table already took its own separate share, unaffected by this call.
        """
        if blocks:
            self.allocator.decref(blocks)

    # -- reservation (with eviction fallback) ----------------------------------
    def reserve_blocks(self, n: int, protect: Node | Sequence[Node] | None = None) -> list[int]:
        """`n` fresh block ids, evicting least-recently-used leaves if the pool is full.

        `protect` (a node, or a sequence of them) is never chosen as the eviction victim. Pass
        every node this reservation must not be able to pull out from under something still
        using it: the node a caller is about to extend (admit onto a lane, or publish a child
        under) -- at the moment this runs, that node may itself still be a childless leaf, since
        the child/lane share that will protect it doesn't exist yet -- *and* every other node
        currently in flight elsewhere in the scheduler (`Scheduler._live_parents`), since
        eviction here is a global, LRU-over-every-leaf operation that has no other way to know
        those nodes are still someone's `_InFlight.parent`.

        Without the in-flight part, a still-live `fl.parent` could be picked as this call's
        victim (it is, after all, an ordinary leaf as far as this cache can see): the physical
        KV blocks stay safe (`adopt`'s incref protects those independently), but the *trie node
        itself* becomes unreachable from `self.root` while `fl` still holds a Python reference to
        it. A later `publish(fl.parent, ...)` from that flight then attaches a new child under a
        node the trie can no longer find: an orphaned subtree, invisible to `node_count`/
        `_iter_leaves`/eviction forever after, permanently holding whatever snapshot slot and
        blocks it and its descendants hold -- a real leak, observed exactly this way against a
        real server under concurrent load before `Scheduler._live_parents` existed. See
        `scheduler.py`'s `_acquire` and `_publish_boundary` for the two call sites.

        Raises `SessionCacheError` if `n` blocks are still unavailable after every evictable
        leaf is gone -- the "fails admission, not mid-decode" contract (paged-kv-design.md
        section 1.2 point 5, section 4).
        """
        if n == 0:
            return []
        while True:
            try:
                return self.allocator.alloc(n)
            except MemoryError:
                if self._evict_one_leaf(protect) is not None:
                    self.stats["block_pressure_evictions"] += 1
                else:
                    raise SessionCacheError(
                        f"need {n} block(s), {self.allocator.free_count} free, "
                        "nothing left to evict"
                    ) from None

    def reserve_snapshot(self, protect: Node | Sequence[Node] | None = None) -> int:
        """A free snapshot slot id. If none is free: first reclaim the least-recently-used
        *superseded* internal node's snapshot (see the module docstring); otherwise take the
        least-recently-used unprotected node overall, reclaiming it if it is internal and
        evicting it if it is a leaf.

        See `reserve_blocks`'s `protect` for why this parameter exists. A protected node keeps
        its snapshot too: a caller may still load it (turn-close rewind, exact-duplicate
        logits).
        """
        while True:
            if self._free_snapshots:
                snapshot = self._free_snapshots.pop()
                self._snapshot_high_watermark = max(
                    self._snapshot_high_watermark,
                    self.num_snapshots - len(self._free_snapshots),
                )
                return snapshot
            if self.reclaim and self._reclaim_one_internal(protect):
                continue
            if self.reclaim and self._reclaim_or_evict_lru(protect):
                continue
            if self._evict_one_leaf(protect) is None:
                self._snapshot_reservation_failures += 1
                raise SessionCacheError(
                    f"need 1 snapshot slot of {self.num_snapshots}, none free, "
                    "nothing left to evict"
                )
            self._snapshot_pressure_evictions += 1

    # -- publish ----------------------------------------------------------------
    def publish(
        self,
        parent: Node,
        edge: tuple[int, ...],
        blocks: tuple[int, ...],
        snapshot: int,
        own: bool = False,
    ) -> Node:
        """Insert a new boundary node under `parent`.

        `own`: the publishing request also published `parent` (a turn-close node under its own
        prompt-end node). Otherwise the request resumed from `parent`, which marks `parent`
        superseded and any never-resumed own child of it bypassed (see the module docstring).

        `blocks` is the complete flat block-id list up to the new node's depth (parent's own
        inherited portion, already possibly cow'd by the caller, plus whatever new blocks this
        edge's tokens needed) -- always incref'd here as this node's own independent ownership
        share, on top of whatever share a lane currently also holds on the same ids (see
        `adopt`'s docstring). `snapshot` must already hold this boundary's DeltaNet state and
        cached logits (the caller writes those via the runner before calling this).

        Raises `SessionCacheError` if `parent` is no longer reachable from `self.root` (its link
        to *its own* parent is gone): a cheap, defense-in-depth check for the exact bug
        `reserve_blocks`'s `protect` docstring describes -- `parent` evicted out from under a
        caller that still held a reference to it. Correct callers (passing every in-flight
        node as `protect` on every reservation in between, as `scheduler.py` does) never trip
        this; it exists so a future gap in that discipline fails loudly, as a request-scoped
        `SessionCacheError` the caller already handles, instead of silently orphaning a subtree
        that leaks its snapshot slot and blocks forever.
        """
        if not edge:
            raise ValueError("cannot publish an empty edge")
        if not self.is_live(parent):
            raise SessionCacheError("cannot publish under a parent that was already evicted")
        twin = next(
            (
                node
                for node in parent.children.get(edge[0], ())
                if node.edge == edge and node.snapshot < 0
            ),
            None,
        )
        if blocks:
            self.allocator.incref(blocks)
        if twin is not None:
            # The same boundary again, under a node whose snapshot was reclaimed: give that
            # node the new state instead of adding a sibling with an identical edge.
            if twin.blocks:
                self.allocator.decref(twin.blocks)
            twin.blocks, twin.snapshot = blocks, snapshot
            twin.adopted = twin.superseded = twin.bypassed = False
            twin.own_child = own
            self.stats["rehydrated"] += 1
            self._mark_resumed_past(parent, own)
            self._touch(twin)
            return twin
        child = Node(
            parent=parent,
            edge=edge,
            depth=parent.depth + len(edge),
            blocks=blocks,
            snapshot=snapshot,
        )
        self._mark_resumed_past(parent, own)
        child.own_child = own
        parent.children.setdefault(edge[0], []).append(child)
        self._touch(child)
        return child

    def _mark_resumed_past(self, parent: Node, own: bool) -> None:
        if own or parent.is_root():
            return
        parent.superseded = True
        for bucket in parent.children.values():
            for sibling in bucket:
                if sibling.own_child and not sibling.adopted:
                    sibling.bypassed = True

    def release_snapshot(self, snapshot: int) -> None:
        """Return a slot from `reserve_snapshot` that was never published (the publish failed).

        Without this, a failure between reserving and publishing leaks the slot for good: no
        node holds it, so no eviction can ever free it."""
        if not 0 <= snapshot < self.num_snapshots or snapshot in self._free_snapshots:
            raise ValueError(f"snapshot {snapshot} is not a reserved slot")
        self._free_snapshots.append(snapshot)

    def is_live(self, node: Node) -> bool:
        """Whether `node` is still in the trie: the root, or linked from a live parent.

        An evicted node's snapshot slot may already belong to another node, so a caller
        holding a stale reference must not read that slot's state or cached logits."""
        while not node.is_root():
            parent = node.parent
            if not any(node is sibling for sibling in parent.children.get(node.edge[0], ())):
                return False
            node = parent
        return True

    # -- eviction -----------------------------------------------------------
    @staticmethod
    def _excluded(protect: Node | Sequence[Node] | None) -> tuple[Node, ...]:
        if protect is None:
            return ()
        if isinstance(protect, Node):
            return (protect,)
        return tuple(protect)

    def _evict_one_leaf(self, protect: Node | Sequence[Node] | None = None) -> Node | None:
        excluded = self._excluded(protect)
        leaves = [n for n in self._iter_leaves(self.root) if not any(n is p for p in excluded)]
        victim = min(leaves, key=lambda node: node.used, default=None)
        if victim is None or victim.is_root():
            return None
        self._remove(victim)
        return victim

    def _reclaim_one_internal(self, protect: Node | Sequence[Node] | None = None) -> bool:
        """First tier: the least-recently-used unprotected node that the traffic has provably
        moved past. A *superseded* internal node loses its snapshot and keeps its blocks (they
        back its descendants' spans); a *bypassed* leaf is evicted. False if there is none."""
        excluded = self._excluded(protect)
        candidates = [
            n
            for n in self._iter_nodes(self.root)
            if not n.is_root()
            and (
                (n.superseded and not n.is_leaf() and n.snapshot >= 0)
                or (n.bypassed and n.is_leaf())
            )
            and not any(n is p for p in excluded)
        ]
        victim = min(candidates, key=lambda node: node.used, default=None)
        if victim is None:
            return False
        if victim.is_leaf():
            self._remove(victim)
            self.stats["evict_bypassed"] += 1
        else:
            self._drop_snapshot(victim)
            self.stats["reclaim_superseded"] += 1
            self._snapshot_reclaims += 1
        return True

    def _reclaim_or_evict_lru(self, protect: Node | Sequence[Node] | None = None) -> bool:
        """Second tier: the least-recently-used of every unprotected leaf and every
        unprotected internal node that still has a snapshot. An internal victim only loses its
        snapshot; a leaf is evicted. False if there is no candidate."""
        excluded = self._excluded(protect)
        candidates = [
            n
            for n in self._iter_nodes(self.root)
            if not n.is_root()
            and (n.is_leaf() or n.snapshot >= 0)
            and not any(n is p for p in excluded)
        ]
        victim = min(candidates, key=lambda node: node.used, default=None)
        if victim is None:
            return False
        if victim.is_leaf():
            self._remove(victim)
            self._snapshot_pressure_evictions += 1
        else:
            self._drop_snapshot(victim)
            self.stats["reclaim_lru"] += 1
            self._snapshot_reclaims += 1
        return True

    def _drop_snapshot(self, node: Node) -> None:
        self._remember_evicted(node)
        self._free_snapshots.append(node.snapshot)
        node.snapshot = -1

    def _iter_nodes(self, node: Node) -> list[Node]:
        out = [node]
        for bucket in node.children.values():
            for child in bucket:
                out.extend(self._iter_nodes(child))
        return out

    def _iter_leaves(self, node: Node) -> list[Node]:
        out: list[Node] = []
        if node.is_leaf():
            if not node.is_root():
                out.append(node)
            return out
        for bucket in node.children.values():
            for child in bucket:
                out.extend(self._iter_leaves(child))
        return out

    def _span(self, node: Node) -> tuple[int, ...]:
        edges = []
        while not node.is_root():
            edges.append(node.edge)
            node = node.parent
        return tuple(token for edge in reversed(edges) for token in edge)

    def _remember_evicted(self, node: Node) -> None:
        """Index `node`'s span so `classify` can tell an evicted boundary from a cold miss."""
        if node.snapshot < 0 or node.depth < EVICTED_KEY_TOKENS:
            return
        span = self._span(node)
        key = span[:EVICTED_KEY_TOKENS]
        self._evicted.setdefault(key, []).append((node.depth, hash(span)))
        self._evicted.move_to_end(key)
        while len(self._evicted) > EVICTED_INDEX_ENTRIES:
            self._evicted.popitem(last=False)

    def _remove(self, node: Node) -> int:
        """Unlink leaf `node`, then any stateless ancestor it leaves childless (nothing can
        resume from one, it only pins blocks). Returns how many nodes were removed."""
        removed = 0
        while True:
            assert node.parent is not None, "cannot evict the root"
            self._remember_evicted(node)
            if node.blocks:
                self.allocator.decref(node.blocks)
            if node.snapshot >= 0:
                self._free_snapshots.append(node.snapshot)
            parent = node.parent
            bucket = parent.children[node.edge[0]]
            bucket.remove(node)
            if not bucket:
                del parent.children[node.edge[0]]
            removed += 1
            if parent.is_root() or parent.has_state() or not parent.is_leaf():
                return removed
            node = parent

    def _touch(self, node: Node) -> None:
        self._stamp += 1
        node.used = self._stamp

    def evict_all(self, protect: Node | Sequence[Node] | None = None) -> int:
        """Drop every cache entry except `protect` and its ancestors (an admin reset hook, e.g.
        between benchmark sweep points). Returns the number of entries removed.

        Pass every in-flight request's node (`Scheduler._live_parents`), as `reserve_blocks`
        callers do: a lane's own `adopt` incref keeps the physical blocks alive, but an evicted
        node's snapshot slot goes back to the free list and can be handed to another node
        while a request still reads it (turn-close `load_snapshot`, exact-duplicate cached
        logits) or publishes under it.
        """
        before = self.node_count()
        while self._evict_one_leaf(protect) is not None:
            pass
        return before - self.node_count()

    # -- introspection (tests, logging) --------------------------------------
    def node_count(self) -> int:
        return self._count(self.root)

    def _count(self, node: Node) -> int:
        return 1 + sum(self._count(child) for bucket in node.children.values() for child in bucket)

    def free_snapshot_count(self) -> int:
        return len(self._free_snapshots)

    def snapshot_stats(self) -> SnapshotStats:
        """Current occupancy plus cumulative pressure since cache construction."""
        return SnapshotStats(
            capacity=self.num_snapshots,
            used=self.num_snapshots - len(self._free_snapshots),
            high_watermark=self._snapshot_high_watermark,
            pressure_evictions=self._snapshot_pressure_evictions,
            reservation_failures=self._snapshot_reservation_failures,
            reclaims=self._snapshot_reclaims,
        )

"""
Tree Index Framework for CSC 502
================================

This single-file framework provides:

1. B+ Tree implementation
2. R-Tree implementation
3. Synthetic dataset generators
4. Experiment harness for build / query benchmarks
5. Example demo runner

Why a single file?
------------------
For a course project, a single file is often easier to:
- read
- submit
- expand
- debug

You can later split it into modules if you want.

Important note
--------------
This code is designed for **education and experimentation**, not production use.
In particular:
- the B+ Tree supports insertion, point search, and range search
- the R-Tree supports insertion and rectangle intersection queries
- deletion and nearest-neighbor search are not implemented yet
- the R-Tree uses a simple linear split heuristic for clarity

These choices make the code much easier to explain in a report and extend later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field, asdict
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)
import random
import time


# =====================================================================
# 1. Shared types and abstract interfaces
# =====================================================================

K = TypeVar("K")
V = TypeVar("V")
Q = TypeVar("Q")


@dataclass
class QueryStats:
    """Summary of a batch of queries."""

    total_seconds: float
    avg_seconds: float
    num_queries: int
    total_results: int
    avg_results: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BuildStats:
    """Summary of index construction time."""

    total_seconds: float
    items_indexed: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentResult:
    """
    One result row for your experiments.

    This is intentionally report-friendly: you can later export this to CSV,
    JSON, pandas, or LaTeX tables.
    """

    # Which index structure is being tested. Eg: BPlusTree, RTree
    algorithm: str
    # The type of query workload executed. Eg: bptree-point = point lookup queries, bptree-range: range queries
    workload: str
    # Number of records inserted into the index
    n_items: int
    # Total time to build the index with n_items, Unit: seconds. This measures the cost of insert N elements into index
    build_seconds: float
    # Total time for executing all queries
    query_total_seconds: float
    # Average time per query = query_total_s / num_queries
    query_avg_seconds: float
    # How many queries were executed
    num_queries: int
    # Total number of records returned
    total_results: int
    # Average number of records returned per query
    avg_results: float
    # Extra parameters of the experiment.
    # Eg: {'order': 16, 'query_type': 'point'} = B+ tree node order = 16, query type = point lookup
    # Eg: {'max_entries': 12, 'query_type': 'rect-intersection'} = each node can store up to 12 rectangles
    extra: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseIndex(ABC, Generic[K, V, Q]):
    """
    Small common interface for all indexes.

    The purpose of this abstraction is to make the framework expandable.
    If you later implement:
    - QuadTree
    - KD-Tree
    - packed R-Tree
    - disk-aware or distributed variants

    they can reuse the same experiment harness if they implement these methods.
    """

    @abstractmethod
    def build(self, items: Iterable[Tuple[K, V]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def insert(self, key: K, value: V) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(self, q: Q) -> List[V]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


# =====================================================================
# 2. Geometry helpers for spatial indexes
# =====================================================================


@dataclass(frozen=True)
class Rect:
    """
    Axis-aligned rectangle in 2D.

    Representation:
        (xmin, ymin, xmax, ymax)

    This is the standard minimal rectangle representation used in R-Trees.
    """

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def area(self) -> float:
        return max(0.0, self.xmax - self.xmin) * max(0.0, self.ymax - self.ymin)

    def intersects(self, other: "Rect") -> bool:
        """
        True if two rectangles intersect or touch.

        This is the key geometric predicate used by R-Tree range queries.
        """
        return not (
            self.xmax < other.xmin
            or self.xmin > other.xmax
            or self.ymax < other.ymin
            or self.ymin > other.ymax
        )

    def union(self, other: "Rect") -> "Rect":
        """Return the minimal bounding rectangle (MBR) covering both rectangles."""
        return Rect(
            xmin=min(self.xmin, other.xmin),
            ymin=min(self.ymin, other.ymin),
            xmax=max(self.xmax, other.xmax),
            ymax=max(self.ymax, other.ymax),
        )

    def enlargement_needed(self, other: "Rect") -> float:
        """
        Return how much this rectangle's area must grow to include `other`.

        In R-Tree insertion, this is the standard rule for choosing a subtree.
        """
        return self.union(other).area() - self.area()

    @staticmethod
    def enclosing(rects: Iterable["Rect"]) -> "Rect":
        rects = list(rects)
        if not rects:
            raise ValueError("Cannot compute enclosing rectangle of an empty list.")
        result = rects[0]
        for r in rects[1:]:
            result = result.union(r)
        return result


# =====================================================================
# 3. B+ Tree implementation
# =====================================================================


@dataclass
class BPlusNode:
    """
    One B+ Tree node.

    Internal node:
        - keys: separator keys
        - children: child pointers

    Leaf node:
        - keys: actual data keys
        - values: aligned payload lists
        - next_leaf: pointer to next leaf for efficient range scanning
    """

    is_leaf: bool
    keys: List[Any] = field(default_factory=list)
    children: List["BPlusNode"] = field(default_factory=list)
    values: List[List[Any]] = field(default_factory=list)
    next_leaf: Optional["BPlusNode"] = None


class BPlusTree(BaseIndex[Any, Any, Tuple[Any, Any] | Any]):
    """
    Educational B+ Tree.

    If order = m:
    - internal nodes have at most m children
    - so they have at most m-1 separator keys
    - leaf nodes also store at most m-1 keys

    Supported operations:
    - insertion
    - point search
    - range search

    Not yet included:
    - deletion
    - persistence to disk
    - bulk-loading
    """

    def __init__(self, order: int = 8):
        if order < 3:
            raise ValueError("B+ Tree order must be at least 3.")
        self.order = order
        self.root = BPlusNode(is_leaf=True)
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def build(self, items: Iterable[Tuple[Any, Any]]) -> None:
        """
        Build the tree by repeated insertion.

        This is slower than a bulk-loader, but it is ideal for a course project
        because it clearly demonstrates the true insertion algorithm.
        """
        for key, value in items:
            self.insert(key, value)

    def insert(self, key: Any, value: Any) -> None:
        """
        Insert one (key, value) pair.

        High-level process:
        1. descend to the appropriate leaf
        2. insert key there
        3. if leaf overflows, split leaf
        4. propagate promoted separator upward
        5. if root splits, create a new root
        """
        promoted = self._insert_recursive(self.root, key, value)

        # If the current root split, create a new root one level higher.
        if promoted is not None:
            promoted_key, right_node = promoted
            new_root = BPlusNode(is_leaf=False)
            new_root.keys = [promoted_key]
            new_root.children = [self.root, right_node]
            self.root = new_root

        self._size += 1

    def search(self, key: Any) -> List[Any]:
        """
        Exact match search.

        We return a list because duplicate keys are supported.
        """
        leaf = self._find_leaf(key)
        pos = bisect_left(leaf.keys, key)
        if pos < len(leaf.keys) and leaf.keys[pos] == key:
            return list(leaf.values[pos])
        return []

    def range_search(self, low: Any, high: Any) -> List[Any]:
        """
        Range query over [low, high].

        This is where B+ Trees shine: once we find the starting leaf,
        the linked-leaf structure allows efficient sequential scanning.
        """
        if low > high:
            return []

        leaf = self._find_leaf(low)
        results: List[Any] = []

        while leaf is not None:
            for i, key in enumerate(leaf.keys):
                if key < low:
                    continue
                if key > high:
                    return results
                results.extend(leaf.values[i])
            leaf = leaf.next_leaf

        return results

    def query(self, q: Tuple[Any, Any] | Any) -> List[Any]:
        """
        Unified query interface used by the experiment harness.

        Supported forms:
        - query(42)       -> point query
        - query((10,50))  -> range query
        """
        if isinstance(q, tuple) and len(q) == 2:
            return self.range_search(q[0], q[1])
        return self.search(q)

    def _find_leaf(self, key: Any) -> BPlusNode:
        """Descend from root to the leaf where `key` belongs."""
        node = self.root
        while not node.is_leaf:
            idx = bisect_right(node.keys, key)
            node = node.children[idx]
        return node

    def _insert_recursive(self, node: BPlusNode, key: Any, value: Any):
        """
        Recursive insertion helper.

        Return value:
            None
                if the subtree did not split

            (promoted_key, right_node)
                if the subtree split and the parent must insert a separator key
        """
        if node.is_leaf:
            return self._insert_into_leaf(node, key, value)

        # Find which child interval the key belongs to.
        child_idx = bisect_right(node.keys, key)
        promoted = self._insert_recursive(node.children[child_idx], key, value)

        if promoted is None:
            return None

        # Child split: insert promoted separator and new right child.
        promoted_key, right_child = promoted
        node.keys.insert(child_idx, promoted_key)
        node.children.insert(child_idx + 1, right_child)

        # If this internal node overflows, split it too.
        if len(node.children) > self.order:
            return self._split_internal(node)
        return None

    def _insert_into_leaf(self, leaf: BPlusNode, key: Any, value: Any):
        """
        Insert key into one leaf.

        Duplicate keys are stored in the same logical bucket.
        """
        pos = bisect_left(leaf.keys, key)

        if pos < len(leaf.keys) and leaf.keys[pos] == key:
            leaf.values[pos].append(value)
        else:
            leaf.keys.insert(pos, key)
            leaf.values.insert(pos, [value])

        if len(leaf.keys) <= self.order - 1:
            return None

        return self._split_leaf(leaf)

    def _split_leaf(self, leaf: BPlusNode):
        """
        Split one overflowing leaf.

        Standard B+ Tree rule:
        - roughly half stays left
        - half moves right
        - promote the smallest key in the right leaf
        - preserve leaf linked list
        """
        split_index = len(leaf.keys) // 2

        right = BPlusNode(is_leaf=True)
        right.keys = leaf.keys[split_index:]
        right.values = leaf.values[split_index:]

        leaf.keys = leaf.keys[:split_index]
        leaf.values = leaf.values[:split_index]

        right.next_leaf = leaf.next_leaf
        leaf.next_leaf = right

        promoted_key = right.keys[0]
        return promoted_key, right

    def _split_internal(self, node: BPlusNode):
        """
        Split one overflowing internal node.

        Internal-node split rule:
        - middle separator moves up to parent
        - left side remains here
        - right side moves to new node
        """
        mid = len(node.keys) // 2
        promoted_key = node.keys[mid]

        right = BPlusNode(is_leaf=False)
        right.keys = node.keys[mid + 1 :]
        right.children = node.children[mid + 1 :]

        node.keys = node.keys[:mid]
        node.children = node.children[: mid + 1]

        return promoted_key, right

    def height(self) -> int:
        """Return tree height in nodes."""
        h = 1
        node = self.root
        while not node.is_leaf:
            h += 1
            node = node.children[0]
        return h

    def dump_structure(self) -> str:
        """Readable multi-line tree printout for debugging or report examples."""
        lines = []
        level = [self.root]
        depth = 0

        while level:
            parts = []
            next_level = []
            for node in level:
                parts.append(f"{'L' if node.is_leaf else 'I'}:{node.keys}")
                if not node.is_leaf:
                    next_level.extend(node.children)
            lines.append(f"depth={depth}  " + " | ".join(parts))
            level = next_level
            depth += 1

        return "\n".join(lines)


# =====================================================================
# 4. R-Tree implementation
# =====================================================================


@dataclass
class RTreeEntry:
    """
    One entry in an R-Tree node.

    Internal entry:
        rect  = MBR of child subtree
        child = child node
        value = None

    Leaf entry:
        rect  = object's rectangle
        child = None
        value = payload
    """

    rect: Rect
    child: Optional["RTreeNode"] = None
    value: Any = None


@dataclass
class RTreeNode:
    is_leaf: bool
    entries: List[RTreeEntry] = field(default_factory=list)

    def mbr(self) -> Rect:
        """Compute the minimal bounding rectangle of this node."""
        if not self.entries:
            raise ValueError("Cannot compute MBR of an empty R-Tree node.")
        result = self.entries[0].rect
        for e in self.entries[1:]:
            result = result.union(e.rect)
        return result


class RTree(BaseIndex[Rect, Any, Rect]):
    """
    Simplified educational R-Tree.

    Supported operations:
    - insertion
    - rectangle intersection query

    Not yet included:
    - deletion
    - nearest-neighbor query
    - STR bulk loading
    - R*-Tree reinsertion heuristics
    """

    def __init__(self, max_entries: int = 8):
        if max_entries < 3:
            raise ValueError("R-Tree max_entries must be at least 3.")
        self.max_entries = max_entries
        self.min_entries = max(2, max_entries // 2)
        self.root = RTreeNode(is_leaf=True)
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def build(self, items: Iterable[Tuple[Rect, Any]]) -> None:
        for rect, value in items:
            self.insert(rect, value)

    def insert(self, key: Rect, value: Any) -> None:
        """
        Insert one rectangle + payload pair.

        Standard high-level idea:
        1. choose leaf needing the smallest enlargement
        2. insert new entry there
        3. split on overflow
        4. propagate split upward
        5. create new root if root splits
        """
        promoted = self._insert_recursive(self.root, RTreeEntry(rect=key, value=value))

        if promoted is not None:
            left_node, right_node = promoted
            new_root = RTreeNode(is_leaf=False)
            new_root.entries = [
                RTreeEntry(rect=left_node.mbr(), child=left_node),
                RTreeEntry(rect=right_node.mbr(), child=right_node),
            ]
            self.root = new_root

        self._size += 1

    def query(self, q: Rect) -> List[Any]:
        """Rectangle intersection query."""
        results: List[Any] = []
        self._query_recursive(self.root, q, results)
        return results

    def _query_recursive(
        self, node: RTreeNode, query_rect: Rect, out: List[Any]
    ) -> None:
        """
        Depth-first traversal with pruning.

        Core R-Tree rule:
        only descend into children whose MBR intersects the query rectangle.
        """
        if node.is_leaf:
            for entry in node.entries:
                if entry.rect.intersects(query_rect):
                    out.append(entry.value)
            return

        for entry in node.entries:
            if entry.rect.intersects(query_rect):
                self._query_recursive(entry.child, query_rect, out)

    def _insert_recursive(self, node: RTreeNode, new_entry: RTreeEntry):
        if node.is_leaf:
            node.entries.append(new_entry)
            if len(node.entries) > self.max_entries:
                return self._split_node(node)
            return None

        # Choose the child that needs the least area enlargement.
        best_idx = self._choose_subtree(node, new_entry.rect)
        child = node.entries[best_idx].child
        promoted = self._insert_recursive(child, new_entry)

        if promoted is None:
            # Child MBR may have expanded after insertion.
            node.entries[best_idx].rect = child.mbr()
            return None

        # Child split into two nodes: replace one entry by two.
        left_child, right_child = promoted
        node.entries.pop(best_idx)
        node.entries.insert(
            best_idx, RTreeEntry(rect=right_child.mbr(), child=right_child)
        )
        node.entries.insert(
            best_idx, RTreeEntry(rect=left_child.mbr(), child=left_child)
        )

        if len(node.entries) > self.max_entries:
            return self._split_node(node)

        return None

    def _choose_subtree(self, node: RTreeNode, rect: Rect) -> int:
        """
        Choose child using classic R-Tree heuristic:
        - smallest enlargement needed
        - tie-break by smaller current area
        """
        best_idx = 0
        best_enlargement = None
        best_area = None

        for i, entry in enumerate(node.entries):
            enlargement = entry.rect.enlargement_needed(rect)
            area = entry.rect.area()

            if (
                best_enlargement is None
                or enlargement < best_enlargement
                or (enlargement == best_enlargement and area < best_area)
            ):
                best_idx = i
                best_enlargement = enlargement
                best_area = area

        return best_idx

    def _split_node(self, node: RTreeNode):
        """
        Split one overflowing node.

        For clarity, this framework uses a simple linear split rather than a
        highly optimized industrial split policy.
        """
        left_entries, right_entries = self._linear_split(node.entries)
        left = RTreeNode(is_leaf=node.is_leaf, entries=left_entries)
        right = RTreeNode(is_leaf=node.is_leaf, entries=right_entries)
        return left, right

    def _linear_split(
        self, entries: List[RTreeEntry]
    ) -> Tuple[List[RTreeEntry], List[RTreeEntry]]:
        """
        Educational linear split heuristic.

        Procedure:
        1. pick two seed entries far apart in space
        2. place one seed in each group
        3. greedily assign remaining entries
        4. respect minimum occupancy constraints
        """
        if len(entries) < 2:
            raise ValueError("Cannot split a node with fewer than 2 entries.")

        seed_a, seed_b = self._pick_seeds(entries)
        group_a = [entries[seed_a]]
        group_b = [entries[seed_b]]

        remaining = [e for i, e in enumerate(entries) if i not in (seed_a, seed_b)]

        while remaining:
            # If one group must take all remaining entries in order to satisfy
            # the minimum fill requirement, do that immediately.
            if len(group_a) + len(remaining) == self.min_entries:
                group_a.extend(remaining)
                break
            if len(group_b) + len(remaining) == self.min_entries:
                group_b.extend(remaining)
                break

            entry = remaining.pop(0)

            mbr_a = Rect.enclosing(e.rect for e in group_a)
            mbr_b = Rect.enclosing(e.rect for e in group_b)

            enlarge_a = mbr_a.enlargement_needed(entry.rect)
            enlarge_b = mbr_b.enlargement_needed(entry.rect)

            if enlarge_a < enlarge_b:
                group_a.append(entry)
            elif enlarge_b < enlarge_a:
                group_b.append(entry)
            else:
                # Tie-break 1: smaller area
                if mbr_a.area() < mbr_b.area():
                    group_a.append(entry)
                elif mbr_b.area() < mbr_a.area():
                    group_b.append(entry)
                else:
                    # Tie-break 2: smaller group size
                    if len(group_a) <= len(group_b):
                        group_a.append(entry)
                    else:
                        group_b.append(entry)

        return group_a, group_b

    def _pick_seeds(self, entries: List[RTreeEntry]) -> Tuple[int, int]:
        """
        Pick the pair of entries whose centers are farthest apart.

        This is simple, understandable, and works well enough for a project.
        """

        def center(rect: Rect):
            return ((rect.xmin + rect.xmax) / 2.0, (rect.ymin + rect.ymax) / 2.0)

        best_pair = (0, 1)
        best_dist_sq = -1.0

        for i in range(len(entries)):
            cx1, cy1 = center(entries[i].rect)
            for j in range(i + 1, len(entries)):
                cx2, cy2 = center(entries[j].rect)
                dist_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
                if dist_sq > best_dist_sq:
                    best_dist_sq = dist_sq
                    best_pair = (i, j)

        return best_pair

    def height(self) -> int:
        """Return tree height in nodes."""
        h = 1
        node = self.root
        while not node.is_leaf:
            h += 1
            node = node.entries[0].child
        return h

    def dump_structure(self) -> str:
        """Readable structure dump for debugging or report examples."""
        lines = []
        level = [self.root]
        depth = 0

        while level:
            parts = []
            next_level = []
            for node in level:
                parts.append(
                    f"{'L' if node.is_leaf else 'I'}:"
                    f"{[(e.rect.xmin, e.rect.ymin, e.rect.xmax, e.rect.ymax) for e in node.entries]}"
                )
                if not node.is_leaf:
                    next_level.extend(e.child for e in node.entries)
            lines.append(f"depth={depth}  " + " | ".join(parts))
            level = next_level
            depth += 1

        return "\n".join(lines)


# =====================================================================
# 5. Synthetic dataset generators
# =====================================================================


def generate_key_value_pairs(
    n: int,
    *,
    key_min: int = 0,
    key_max: int = 1_000_000,
    allow_duplicates: bool = True,
    seed: int = 0,
) -> List[Tuple[int, int]]:
    """
    Generate synthetic (key, value) pairs for B+ Tree experiments.

    value = record id in this toy framework.
    """
    rng = random.Random(seed)
    pairs = []

    if allow_duplicates:
        for record_id in range(n):
            key = rng.randint(key_min, key_max)
            pairs.append((key, record_id))
    else:
        if key_max - key_min + 1 < n:
            raise ValueError("Key range too small for unique keys.")
        keys = rng.sample(range(key_min, key_max + 1), n)
        pairs = [(key, record_id) for record_id, key in enumerate(keys)]

    return pairs


def sample_point_queries(
    keys: Sequence[int], num_queries: int, *, seed: int = 0
) -> List[int]:
    """Sample exact-match queries from existing keys."""
    rng = random.Random(seed)
    return [rng.choice(keys) for _ in range(num_queries)]


def sample_range_queries(
    num_queries: int,
    *,
    low_min: int = 0,
    low_max: int = 900_000,
    max_width: int = 50_000,
    seed: int = 0,
) -> List[Tuple[int, int]]:
    """Generate random 1D ranges for B+ Tree range-search experiments."""
    rng = random.Random(seed)
    queries = []
    for _ in range(num_queries):
        low = rng.randint(low_min, low_max)
        width = rng.randint(1, max_width)
        queries.append((low, low + width))
    return queries


def generate_rectangles(
    n: int,
    *,
    space_width: float = 10_000.0,
    space_height: float = 10_000.0,
    max_rect_width: float = 100.0,
    max_rect_height: float = 100.0,
    seed: int = 0,
) -> List[Tuple[Rect, int]]:
    """Generate synthetic rectangles for R-Tree experiments."""
    rng = random.Random(seed)
    rects: List[Tuple[Rect, int]] = []

    for object_id in range(n):
        xmin = rng.uniform(0, space_width)
        ymin = rng.uniform(0, space_height)
        width = rng.uniform(1, max_rect_width)
        height = rng.uniform(1, max_rect_height)
        xmax = min(space_width, xmin + width)
        ymax = min(space_height, ymin + height)
        rects.append((Rect(xmin, ymin, xmax, ymax), object_id))

    return rects


def sample_rect_queries(
    num_queries: int,
    *,
    space_width: float = 10_000.0,
    space_height: float = 10_000.0,
    max_query_width: float = 500.0,
    max_query_height: float = 500.0,
    seed: int = 0,
) -> List[Rect]:
    """Generate rectangle intersection queries for R-Tree experiments."""
    rng = random.Random(seed)
    queries: List[Rect] = []

    for _ in range(num_queries):
        xmin = rng.uniform(0, space_width)
        ymin = rng.uniform(0, space_height)
        width = rng.uniform(1, max_query_width)
        height = rng.uniform(1, max_query_height)
        xmax = min(space_width, xmin + width)
        ymax = min(space_height, ymin + height)
        queries.append(Rect(xmin, ymin, xmax, ymax))

    return queries


# =====================================================================
# 6. Experiment harness
# =====================================================================


def benchmark_build(index, items: Sequence[Tuple[Any, Any]]) -> BuildStats:
    """Measure index construction time."""
    start = time.perf_counter()
    index.build(items)
    end = time.perf_counter()
    return BuildStats(total_seconds=end - start, items_indexed=len(items))


def benchmark_queries(index, queries: Sequence[Any]) -> QueryStats:
    """Measure total and average time for a batch of queries."""
    start = time.perf_counter()
    total_results = 0

    for q in queries:
        total_results += len(index.query(q))

    end = time.perf_counter()
    total_seconds = end - start
    num_queries = len(queries)

    return QueryStats(
        total_seconds=total_seconds,
        avg_seconds=(total_seconds / num_queries if num_queries else 0.0),
        num_queries=num_queries,
        total_results=total_results,
        avg_results=(total_results / num_queries if num_queries else 0.0),
    )


def run_single_experiment(
    *,
    index_factory,
    items,
    queries,
    workload_name: str,
    extra: Dict[str, Any] | None = None,
) -> ExperimentResult:
    """
    Run one algorithm on one dataset/workload.

    This function is ideal for your Option A and Option B experiments.
    """
    index = index_factory()
    build_stats = benchmark_build(index, items)
    query_stats = benchmark_queries(index, queries)

    return ExperimentResult(
        algorithm=index.name,
        workload=workload_name,
        n_items=len(items),
        build_seconds=build_stats.total_seconds,
        query_total_seconds=query_stats.total_seconds,
        query_avg_seconds=query_stats.avg_seconds,
        num_queries=query_stats.num_queries,
        total_results=query_stats.total_results,
        avg_results=query_stats.avg_results,
        extra=extra or {},
    )


def compare_indexes(
    *,
    index_factories,
    items,
    queries,
    workload_name: str,
    extra_builder=None,
) -> List[ExperimentResult]:
    """
    Compare several indexes or several configurations of one index.

    This function is the core of your Option C experiments.
    """
    results: List[ExperimentResult] = []

    for factory in index_factories:
        index = factory()
        build_stats = benchmark_build(index, items)
        query_stats = benchmark_queries(index, queries)
        extra = extra_builder(index) if extra_builder else {}

        results.append(
            ExperimentResult(
                algorithm=index.name,
                workload=workload_name,
                n_items=len(items),
                build_seconds=build_stats.total_seconds,
                query_total_seconds=query_stats.total_seconds,
                query_avg_seconds=query_stats.avg_seconds,
                num_queries=query_stats.num_queries,
                total_results=query_stats.total_results,
                avg_results=query_stats.avg_results,
                extra=extra,
            )
        )

    return results


def pretty_print_results(results: Sequence[ExperimentResult]) -> None:
    """Print results in a simple table without external dependencies."""
    if not results:
        print("No results.")
        return

    headers = [
        "algorithm",
        "workload",
        "n_items",
        "build_s",
        "query_total_s",
        "query_avg_s",
        "num_queries",
        "avg_results",
        "extra",
    ]

    rows = []
    for r in results:
        rows.append(
            [
                r.algorithm,
                r.workload,
                r.n_items,
                f"{r.build_seconds:.6f}",
                f"{r.query_total_seconds:.6f}",
                f"{r.query_avg_seconds:.8f}",
                r.num_queries,
                f"{r.avg_results:.3f}",
                str(r.extra),
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt_row(row):
        return " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


# =====================================================================
# 7. Demo runner
# =====================================================================


def demo_bplustree() -> None:
    """
    Option A from the project discussion:
    implement B+ Tree and benchmark it.
    """
    print("\n=== Option A: B+ Tree index ===")

    items = generate_key_value_pairs(2000000, allow_duplicates=True, seed=1)
    keys = [k for k, _ in items]
    point_queries = sample_point_queries(keys, num_queries=2000, seed=2)
    range_queries = sample_range_queries(num_queries=2000, seed=3)

    result_point = run_single_experiment(
        index_factory=lambda: BPlusTree(order=16),
        items=items,
        queries=point_queries,
        workload_name="bptree-point",
        extra={"order": 16, "query_type": "point"},
    )

    result_range = run_single_experiment(
        index_factory=lambda: BPlusTree(order=16),
        items=items,
        queries=range_queries,
        workload_name="bptree-range",
        extra={"order": 16, "query_type": "range"},
    )

    pretty_print_results([result_point, result_range])


def demo_rtree() -> None:
    """
    Option B from the project discussion:
    implement R-Tree and benchmark it.
    """
    print("\n=== Option B: R-Tree index ===")

    items = generate_rectangles(
        15_000,
        max_rect_width=120.0,
        max_rect_height=120.0,
        seed=10,
    )

    queries = sample_rect_queries(
        num_queries=300,
        max_query_width=600.0,
        max_query_height=600.0,
        seed=11,
    )

    result = run_single_experiment(
        index_factory=lambda: RTree(max_entries=12),
        items=items,
        queries=queries,
        workload_name="rtree-intersection",
        extra={"max_entries": 12, "query_type": "rect-intersection"},
    )

    pretty_print_results([result])


def demo_comparison() -> None:
    """
    Option C from the project discussion:
    compare multiple index configurations using one shared framework.

    Note:
    B+ Trees and R-Trees answer different query types, so the fairest comparison
    is usually:
    - B+ Tree vs B+ Tree with different orders
    - R-Tree vs R-Tree with different capacities

    That is what this demo does.
    """
    print("\n=== Option C1: Compare B+ Tree configurations ===")

    items = generate_key_value_pairs(30_000, allow_duplicates=True, seed=21)
    keys = [k for k, _ in items]
    point_queries = sample_point_queries(keys, num_queries=800, seed=22)

    bpt_results = compare_indexes(
        index_factories=[
            lambda: BPlusTree(order=8),
            lambda: BPlusTree(order=16),
            lambda: BPlusTree(order=32),
        ],
        items=items,
        queries=point_queries,
        workload_name="compare-bpt-orders",
        extra_builder=lambda idx: {"order": idx.order, "height": idx.height()},
    )
    pretty_print_results(bpt_results)

    print("\n=== Option C2: Compare R-Tree configurations ===")

    rect_items = generate_rectangles(20_000, seed=31)
    rect_queries = sample_rect_queries(num_queries=500, seed=32)

    rtree_results = compare_indexes(
        index_factories=[
            lambda: RTree(max_entries=6),
            lambda: RTree(max_entries=12),
            lambda: RTree(max_entries=24),
        ],
        items=rect_items,
        queries=rect_queries,
        workload_name="compare-rtree-capacity",
        extra_builder=lambda idx: {
            "max_entries": idx.max_entries,
            "height": idx.height(),
        },
    )
    pretty_print_results(rtree_results)


if __name__ == "__main__":
    demo_bplustree()
    demo_rtree()
    demo_comparison()

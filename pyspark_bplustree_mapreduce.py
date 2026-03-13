#!/usr/bin/env python3
"""
pyspark_bplustree_mapreduce.py
==============================

Educational implementation of a B+ Tree with a MapReduce-style / PySpark-style
build pipeline and a small test experiment harness.

This script is designed for a CSC 502 style project where you want to show:

1. The local B+ Tree algorithm
2. A distributed build idea inspired by MapReduce / PySpark
3. A repeatable experiment for build and query performance

Important notes
---------------
- This code is intentionally written for clarity and extensibility, not for
  production deployment.
- The "distributed B+ Tree" here follows the core idea often described in papers:
    * split data into key ranges
    * build local B+ Trees in parallel
    * collect small metadata from each partition
    * build a lightweight global routing directory
- Query execution is implemented in a way that is easy to explain in a report.
- Deletion is omitted to keep the script focused on build / search / experiments.

How to run
----------
Example:

    spark-submit pyspark_bplustree_mapreduce.py \
        --n-records 200000 \
        --n-partitions 8 \
        --tree-order 32 \
        --n-point-queries 200 \
        --n-range-queries 100
"""

from __future__ import annotations

import argparse
import math
import random
import time
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:
    from pyspark.sql import SparkSession
    from pyspark import StorageLevel
except Exception:
    SparkSession = None
    StorageLevel = None


# =====================================================================
# Result containers
# =====================================================================


@dataclass
class ExperimentRow:
    """One row of benchmark output."""

    algorithm: str
    workload: str
    n_items: int
    build_s: float
    query_total_s: float
    query_avg_s: float
    num_queries: int
    avg_results: float
    extra: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================================
# Local B+ Tree implementation
# =====================================================================


@dataclass
class BPlusNode:
    """
    One B+ Tree node.

    Internal node:
        - keys: separator keys
        - children: child pointers

    Leaf node:
        - keys: actual indexed keys
        - values: aligned list of payload buckets
        - next_leaf: linked-list pointer for efficient range scans

    Duplicate keys are supported by storing a *list* of values for each key.
    """

    is_leaf: bool
    keys: List[Any] = field(default_factory=list)
    children: List["BPlusNode"] = field(default_factory=list)
    values: List[List[Any]] = field(default_factory=list)
    next_leaf: Optional["BPlusNode"] = None


class BPlusTree:
    """
    Educational B+ Tree.

    If order = m:
    - internal nodes have at most m children
    - therefore at most m-1 separator keys
    - leaf nodes also store at most m-1 keys

    Supported operations:
    - insertion
    - exact search
    - range search

    Repeated insertion is slower than bulk loading, but it is ideal for a
    project because it clearly demonstrates node splitting behavior.
    """

    def __init__(self, order: int = 32):
        if order < 3:
            raise ValueError("B+ Tree order must be at least 3.")
        self.order = order
        self.root = BPlusNode(is_leaf=True)
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def build(self, items: Iterable[Tuple[Any, Any]]) -> None:
        """Build the tree by repeated insertion."""
        for key, value in items:
            self.insert(key, value)

    def insert(self, key: Any, value: Any) -> None:
        """
        Insert one (key, value) pair.

        High-level procedure:
        1. Recursively descend to the correct leaf.
        2. Insert the key there.
        3. If the leaf overflows, split it.
        4. Propagate the promoted separator upward.
        5. If the root splits, create a new root.
        """
        promoted = self._insert_recursive(self.root, key, value)

        if promoted is not None:
            promoted_key, right_node = promoted
            new_root = BPlusNode(is_leaf=False)
            new_root.keys = [promoted_key]
            new_root.children = [self.root, right_node]
            self.root = new_root

        self._size += 1

    def search(self, key: Any) -> List[Any]:
        """Exact lookup for one key."""
        leaf = self._find_leaf(key)
        pos = bisect_left(leaf.keys, key)
        if pos < len(leaf.keys) and leaf.keys[pos] == key:
            return list(leaf.values[pos])
        return []

    def range_search(self, low: Any, high: Any) -> List[Any]:
        """
        Range query over [low, high].

        Once the starting leaf is found, the linked leaves are scanned to the
        right until the upper bound is passed.
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

    def height(self) -> int:
        """Return the height of the tree in node levels."""
        h = 1
        node = self.root
        while not node.is_leaf:
            h += 1
            node = node.children[0]
        return h

    def leftmost_key(self) -> Optional[Any]:
        node = self.root
        while not node.is_leaf:
            node = node.children[0]
        return node.keys[0] if node.keys else None

    def rightmost_key(self) -> Optional[Any]:
        node = self.root
        while not node.is_leaf:
            node = node.children[-1]
        return node.keys[-1] if node.keys else None

    def dump_structure(self) -> str:
        """Readable multi-line representation of the tree."""
        lines: List[str] = []
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

    def _find_leaf(self, key: Any) -> BPlusNode:
        """Descend from the root to the leaf where `key` belongs."""
        node = self.root
        while not node.is_leaf:
            idx = bisect_right(node.keys, key)
            node = node.children[idx]
        return node

    def _insert_recursive(self, node: BPlusNode, key: Any, value: Any):
        """
        Recursive insertion helper.

        Returns:
            None
                if the subtree did not split
            (promoted_key, right_node)
                if the subtree split and the parent must insert a separator
        """
        if node.is_leaf:
            return self._insert_into_leaf(node, key, value)

        child_idx = bisect_right(node.keys, key)
        promoted = self._insert_recursive(node.children[child_idx], key, value)

        if promoted is None:
            return None

        promoted_key, right_child = promoted
        node.keys.insert(child_idx, promoted_key)
        node.children.insert(child_idx + 1, right_child)

        if len(node.children) > self.order:
            return self._split_internal(node)
        return None

    def _insert_into_leaf(self, leaf: BPlusNode, key: Any, value: Any):
        """Insert key into a leaf; duplicates are stored in the same bucket."""
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
        Split an overflowing leaf.

        Standard leaf split:
        - half remains on the left
        - half moves to a new right leaf
        - the first key of the right leaf is promoted upward
        - the leaf linked list is preserved
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
        Split an overflowing internal node.

        The middle separator is promoted upward and is not kept in either child.
        """
        mid = len(node.keys) // 2
        promoted_key = node.keys[mid]

        right = BPlusNode(is_leaf=False)
        right.keys = node.keys[mid + 1 :]
        right.children = node.children[mid + 1 :]

        node.keys = node.keys[:mid]
        node.children = node.children[: mid + 1]

        return promoted_key, right


# =====================================================================
# Synthetic data generation
# =====================================================================


def generate_records(
    n_records: int,
    *,
    key_min: int = 0,
    key_max: int = 1_000_000,
    allow_duplicates: bool = True,
    seed: int = 1,
) -> List[Tuple[int, int]]:
    """Generate synthetic (key, payload) records."""
    rng = random.Random(seed)
    items: List[Tuple[int, int]] = []

    if allow_duplicates:
        for record_id in range(n_records):
            key = rng.randint(key_min, key_max)
            items.append((key, record_id))
    else:
        if key_max - key_min + 1 < n_records:
            raise ValueError("Key range too small for unique keys.")
        keys = rng.sample(range(key_min, key_max + 1), n_records)
        items = [(key, record_id) for record_id, key in enumerate(keys)]

    return items


def sample_point_queries(
    keys: Sequence[int], n_queries: int, *, seed: int = 123
) -> List[int]:
    """Sample exact-match queries from keys that actually exist."""
    rng = random.Random(seed)
    return [rng.choice(keys) for _ in range(n_queries)]


def sample_range_queries(
    n_queries: int,
    *,
    low_min: int = 0,
    low_max: int = 900_000,
    max_width: int = 50_000,
    seed: int = 456,
) -> List[Tuple[int, int]]:
    """Generate random 1D range queries [low, high]."""
    rng = random.Random(seed)
    queries = []
    for _ in range(n_queries):
        low = rng.randint(low_min, low_max)
        width = rng.randint(1, max_width)
        queries.append((low, low + width))
    return queries


# =====================================================================
# Centralized experiment helpers
# =====================================================================


def run_centralized_build(
    items: Sequence[Tuple[int, int]], tree_order: int
) -> Tuple[BPlusTree, float]:
    """Build one centralized B+ Tree on the driver."""
    tree = BPlusTree(order=tree_order)
    start = time.perf_counter()
    tree.build(items)
    end = time.perf_counter()
    return tree, end - start


def run_centralized_point_queries(
    tree: BPlusTree, queries: Sequence[int]
) -> Tuple[float, float]:
    """Run exact-match queries on a centralized tree."""
    start = time.perf_counter()
    total_results = 0
    for q in queries:
        total_results += len(tree.search(q))
    end = time.perf_counter()
    total_time = end - start
    avg_results = total_results / len(queries) if queries else 0.0
    return total_time, avg_results


def run_centralized_range_queries(
    tree: BPlusTree, queries: Sequence[Tuple[int, int]]
) -> Tuple[float, float]:
    """Run range queries on a centralized tree."""
    start = time.perf_counter()
    total_results = 0
    for low, high in queries:
        total_results += len(tree.range_search(low, high))
    end = time.perf_counter()
    total_time = end - start
    avg_results = total_results / len(queries) if queries else 0.0
    return total_time, avg_results


# =====================================================================
# Distributed MapReduce-style B+ Tree builder in PySpark
# =====================================================================


class DistributedBPlusTree:
    """
    Lightweight distributed B+ Tree directory built with Spark.

    This is not one fully materialized global node graph. Instead, it consists of:
    1. one local B+ Tree per Spark partition,
    2. small partition metadata collected to the driver,
    3. a global separator list on the driver for query routing.

    This matches the key idea of many distributed B+ Tree papers:
    - parallel local construction
    - small global routing layer
    - key-range based query routing
    """

    def __init__(self, spark, tree_order: int = 32, n_partitions: int = 8):
        if SparkSession is None:
            raise RuntimeError("PySpark is not available in this environment.")

        self.spark = spark
        self.sc = spark.sparkContext
        self.tree_order = tree_order
        self.n_partitions = n_partitions

        # self.partition_index_rdd = None
        self.partition_data_rdd = None
        self.partition_bounds: List[Dict[str, Any]] = []
        self.global_separators: List[int] = []

        self._global_min_key: Optional[int] = None
        self._global_max_key: Optional[int] = None
        self._chunk_width: Optional[int] = None

    def build(self, records_rdd) -> float:
        """
        Build the distributed B+ Tree.

        New robust design:
        - persist sorted partition rows, not recursive B+ Tree objects
        - collect only small metadata to the driver
        """
        build_start = time.perf_counter()

        min_key = records_rdd.keys().min()
        max_key = records_rdd.keys().max()
        self._global_min_key = int(min_key)
        self._global_max_key = int(max_key)

        self._chunk_width = max(
            1,
            math.ceil(
                (self._global_max_key - self._global_min_key + 1) / self.n_partitions
            ),
        )

        global_min_key = self._global_min_key
        chunk_width = self._chunk_width
        n_partitions = self.n_partitions
        tree_order = self.tree_order

        def partition_func(key: int) -> int:
            idx = (int(key) - global_min_key) // chunk_width
            return min(max(idx, 0), n_partitions - 1)

        partitioned = records_rdd.partitionBy(
            n_partitions, partitionFunc=partition_func
        )

        if StorageLevel is not None:
            partitioned.persist(StorageLevel.MEMORY_AND_DISK)

        def prepare_partition(partition_id: int, iterator: Iterator[Tuple[int, int]]):
            rows = list(iterator)
            rows.sort(key=lambda x: x[0])

            # Build a temporary local tree only to compute metadata such as height.
            tmp_tree = BPlusTree(order=tree_order)
            tmp_tree.build(rows)

            metadata = {
                "partition_id": partition_id,
                "n_items": len(rows),
                "min_key": rows[0][0] if rows else None,
                "max_key": rows[-1][0] if rows else None,
                "height": tmp_tree.height() if rows else 0,
            }

            # Store rows, not the recursive tree object.
            yield (partition_id, rows, metadata)

        self.partition_data_rdd = partitioned.mapPartitionsWithIndex(prepare_partition)

        if StorageLevel is not None:
            self.partition_data_rdd.persist(StorageLevel.MEMORY_AND_DISK)

        # Materialize once so build time includes the actual work.
        _ = self.partition_data_rdd.count()

        self.partition_bounds = [
            row[1]
            for row in sorted(
                self.partition_data_rdd.map(lambda x: (x[0], x[2])).collect(),
                key=lambda z: z[0],
            )
        ]

        self.global_separators = [
            bound["min_key"]
            for bound in self.partition_bounds[1:]
            if bound["min_key"] is not None
        ]

        build_end = time.perf_counter()
        return build_end - build_start

    def _route_partition_for_key(self, key: int) -> int:
        """
        Route one key to the partition that should contain it.

        The driver-side separators behave like the internal separators of a
        higher-level B+ Tree directory.
        """
        if not self.partition_bounds:
            raise RuntimeError("Distributed index has not been built yet.")
        return bisect_right(self.global_separators, key)

    def _candidate_partitions_for_range(self, low: int, high: int) -> List[int]:
        """Return all partitions whose key intervals overlap [low, high]."""
        if low > high:
            return []
        p_low = self._route_partition_for_key(low)
        p_high = self._route_partition_for_key(high)
        return list(range(p_low, p_high + 1))

    def point_query(self, key: int) -> List[int]:
        """
        Distributed exact-match query.

        We rebuild the local B+ tree only inside the target partition task.
        This avoids pickling/storing recursive tree objects in Spark.
        """
        target_partition = self._route_partition_for_key(key)
        tree_order = self.tree_order

        def search_partition(
            iterator: Iterator[Tuple[int, List[Tuple[int, int]], Dict[str, Any]]],
        ):
            for partition_id, rows, _metadata in iterator:
                if partition_id == target_partition:
                    local_tree = BPlusTree(order=tree_order)
                    local_tree.build(rows)
                    yield from local_tree.search(key)

        return self.partition_data_rdd.mapPartitions(search_partition).collect()

    def range_query(self, low: int, high: int) -> List[int]:
        """
        Distributed range query.

        Rebuild local B+ trees only for the overlapping partitions.
        """
        candidate_partitions = set(self._candidate_partitions_for_range(low, high))
        tree_order = self.tree_order

        def search_partitions(
            iterator: Iterator[Tuple[int, List[Tuple[int, int]], Dict[str, Any]]],
        ):
            for partition_id, rows, _metadata in iterator:
                if partition_id in candidate_partitions:
                    local_tree = BPlusTree(order=tree_order)
                    local_tree.build(rows)
                    yield from local_tree.range_search(low, high)

        return self.partition_data_rdd.mapPartitions(search_partitions).collect()

    def directory_height_estimate(self) -> int:
        """
        Estimate the height of the global directory if it were itself stored as a
        B+ Tree over partitions. This is useful for theory discussion.
        """
        p = max(1, len(self.partition_bounds))
        fanout = self.tree_order
        h = 1
        while p > fanout:
            p = math.ceil(p / fanout)
            h += 1
        return h


# =====================================================================
# Distributed experiment helpers
# =====================================================================


def run_distributed_experiment(
    spark,
    items: Sequence[Tuple[int, int]],
    *,
    tree_order: int,
    n_partitions: int,
    point_queries: Sequence[int],
    range_queries: Sequence[Tuple[int, int]],
) -> List[ExperimentRow]:
    """Build and benchmark the distributed B+ Tree."""
    sc = spark.sparkContext
    records_rdd = sc.parallelize(items, n_partitions)

    distributed_tree = DistributedBPlusTree(
        spark=spark,
        tree_order=tree_order,
        n_partitions=n_partitions,
    )

    build_s = distributed_tree.build(records_rdd)

    point_start = time.perf_counter()
    point_total_results = 0
    for q in point_queries:
        point_total_results += len(distributed_tree.point_query(q))
    point_end = time.perf_counter()
    point_total_s = point_end - point_start
    point_avg_results = (
        point_total_results / len(point_queries) if point_queries else 0.0
    )

    range_start = time.perf_counter()
    range_total_results = 0
    for low, high in range_queries:
        range_total_results += len(distributed_tree.range_query(low, high))
    range_end = time.perf_counter()
    range_total_s = range_end - range_start
    range_avg_results = (
        range_total_results / len(range_queries) if range_queries else 0.0
    )

    return [
        ExperimentRow(
            algorithm="DistributedBPlusTree",
            workload="dist-bptree-point",
            n_items=len(items),
            build_s=build_s,
            query_total_s=point_total_s,
            query_avg_s=point_total_s / len(point_queries) if point_queries else 0.0,
            num_queries=len(point_queries),
            avg_results=point_avg_results,
            extra={
                "order": tree_order,
                "n_partitions": n_partitions,
                "directory_height_est": distributed_tree.directory_height_estimate(),
                "partition_heights": [
                    b["height"] for b in distributed_tree.partition_bounds
                ],
            },
        ),
        ExperimentRow(
            algorithm="DistributedBPlusTree",
            workload="dist-bptree-range",
            n_items=len(items),
            build_s=build_s,
            query_total_s=range_total_s,
            query_avg_s=range_total_s / len(range_queries) if range_queries else 0.0,
            num_queries=len(range_queries),
            avg_results=range_avg_results,
            extra={
                "order": tree_order,
                "n_partitions": n_partitions,
                "directory_height_est": distributed_tree.directory_height_estimate(),
                "partition_heights": [
                    b["height"] for b in distributed_tree.partition_bounds
                ],
            },
        ),
    ]


# =====================================================================
# Pretty-print helper
# =====================================================================


def pretty_print_table(rows: Sequence[ExperimentRow]) -> None:
    """Print rows in a fixed-width text table."""
    if not rows:
        print("No experiment rows to display.")
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

    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row.algorithm,
                row.workload,
                row.n_items,
                f"{row.build_s:.6f}",
                f"{row.query_total_s:.6f}",
                f"{row.query_avg_s:.8f}",
                row.num_queries,
                f"{row.avg_results:.3f}",
                str(row.extra),
            ]
        )

    widths = [len(h) for h in headers]
    for r in table_rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt(r):
        return " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(r))

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for r in table_rows:
        print(fmt(r))


# =====================================================================
# Main runner
# =====================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Educational PySpark MapReduce-style B+ Tree experiment."
    )
    parser.add_argument(
        "--n-records",
        type=int,
        default=200000,
        help="Number of synthetic records to generate.",
    )
    parser.add_argument(
        "--n-partitions",
        type=int,
        default=8,
        help="Number of Spark partitions / reducers.",
    )
    parser.add_argument("--tree-order", type=int, default=32, help="B+ Tree order.")
    parser.add_argument(
        "--n-point-queries", type=int, default=2000, help="Number of point queries."
    )
    parser.add_argument(
        "--n-range-queries", type=int, default=2000, help="Number of range queries."
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--key-min", type=int, default=0, help="Minimum generated key.")
    parser.add_argument(
        "--key-max", type=int, default=1_000_000, help="Maximum generated key."
    )
    args = parser.parse_args()

    if SparkSession is None:
        raise RuntimeError(
            "PySpark is not available. Please run this script with spark-submit "
            "or in an environment where pyspark is installed."
        )

    spark = SparkSession.builder.appName(
        "EducationalDistributedBPlusTree"
    ).getOrCreate()

    try:
        items = generate_records(
            args.n_records,
            key_min=args.key_min,
            key_max=args.key_max,
            allow_duplicates=True,
            seed=args.seed,
        )

        keys = [k for k, _ in items]
        point_queries = sample_point_queries(
            keys, args.n_point_queries, seed=args.seed + 1
        )
        range_queries = sample_range_queries(args.n_range_queries, seed=args.seed + 2)

        central_tree, central_build_s = run_centralized_build(
            items, tree_order=args.tree_order
        )
        central_point_total_s, central_point_avg_results = (
            run_centralized_point_queries(central_tree, point_queries)
        )
        central_range_total_s, central_range_avg_results = (
            run_centralized_range_queries(central_tree, range_queries)
        )

        centralized_rows = [
            ExperimentRow(
                algorithm="CentralizedBPlusTree",
                workload="central-bptree-point",
                n_items=len(items),
                build_s=central_build_s,
                query_total_s=central_point_total_s,
                query_avg_s=(
                    central_point_total_s / len(point_queries) if point_queries else 0.0
                ),
                num_queries=len(point_queries),
                avg_results=central_point_avg_results,
                extra={"order": args.tree_order, "height": central_tree.height()},
            ),
            ExperimentRow(
                algorithm="CentralizedBPlusTree",
                workload="central-bptree-range",
                n_items=len(items),
                build_s=central_build_s,
                query_total_s=central_range_total_s,
                query_avg_s=(
                    central_range_total_s / len(range_queries) if range_queries else 0.0
                ),
                num_queries=len(range_queries),
                avg_results=central_range_avg_results,
                extra={"order": args.tree_order, "height": central_tree.height()},
            ),
        ]

        distributed_rows = run_distributed_experiment(
            spark,
            items,
            tree_order=args.tree_order,
            n_partitions=args.n_partitions,
            point_queries=point_queries,
            range_queries=range_queries,
        )

        print("\n=== Centralized B+ Tree experiment ===")
        pretty_print_table(centralized_rows)

        print("\n=== Distributed MapReduce-style / PySpark B+ Tree experiment ===")
        pretty_print_table(distributed_rows)

        print("\nInterpretation notes:")
        print("1. build_s measures total index build time.")
        print("2. query_total_s is the total time for the whole query batch.")
        print("3. query_avg_s is the average time per query.")
        print("4. Centralized and distributed rows do not have the same constants.")
        print("5. Distributed query costs include Spark job overhead.")
        print("6. For larger data, the distributed build becomes more attractive.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()

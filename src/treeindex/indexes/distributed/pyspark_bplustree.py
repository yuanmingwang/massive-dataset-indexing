from __future__ import annotations

import math
import time
from bisect import bisect_right
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from treeindex.core.interfaces import DistributedIndex
from treeindex.indexes.distributed.base import DistributedExperimentRow
from treeindex.indexes.local.bplustree import BPlusTree

try:
    from pyspark import StorageLevel
    from pyspark.sql import SparkSession
except Exception:
    SparkSession = None
    StorageLevel = None


class DistributedBPlusTree(DistributedIndex):
    def __init__(self, spark, tree_order: int = 32, n_partitions: int = 8):
        if SparkSession is None:
            raise RuntimeError("PySpark is not available in this environment.")
        self.spark = spark
        self.sc = spark.sparkContext
        self.tree_order = tree_order
        self.n_partitions = n_partitions
        self.partition_data_rdd = None
        self.partition_bounds: List[Dict[str, int | None]] = []
        self.global_separators: List[int] = []
        self._global_min_key: Optional[int] = None
        self._global_max_key: Optional[int] = None
        self._chunk_width: Optional[int] = None

    def build(self, records_rdd) -> float:
        build_start = time.perf_counter()
        min_key = records_rdd.keys().min()
        max_key = records_rdd.keys().max()
        self._global_min_key = int(min_key)
        self._global_max_key = int(max_key)
        self._chunk_width = max(1, math.ceil((self._global_max_key - self._global_min_key + 1) / self.n_partitions))
        global_min_key = self._global_min_key
        chunk_width = self._chunk_width
        n_partitions = self.n_partitions
        tree_order = self.tree_order

        def partition_func(key: int) -> int:
            idx = (int(key) - global_min_key) // chunk_width
            return min(max(idx, 0), n_partitions - 1)

        # MapReduce build phase:
        # map each key/value row to a key-range partition, then let Spark shuffle rows
        # with the same partition id to the same reducer partition via partitionBy(...).
        partitioned = records_rdd.partitionBy(n_partitions, partitionFunc=partition_func)
        if StorageLevel is not None:
            partitioned.persist(StorageLevel.MEMORY_AND_DISK)

        def prepare_partition(partition_id: int, iterator: Iterator[Tuple[int, Any]]):
            # Reduce/build phase:
            # this function runs once per Spark partition, rebuilds one local B+ tree
            # from that partition's rows, and emits compact metadata for driver routing.
            rows = list(iterator)
            rows.sort(key=lambda x: x[0])
            local_tree = BPlusTree(order=tree_order)
            local_tree.build(rows)
            metadata = {
                "partition_id": partition_id,
                "n_items": len(rows),
                "min_key": rows[0][0] if rows else None,
                "max_key": rows[-1][0] if rows else None,
                "height": local_tree.height() if rows else 0,
            }
            yield (partition_id, local_tree, metadata)

        self.partition_data_rdd = partitioned.mapPartitionsWithIndex(prepare_partition)
        if StorageLevel is not None:
            self.partition_data_rdd.persist(StorageLevel.MEMORY_AND_DISK)
        _ = self.partition_data_rdd.count()
        self.partition_bounds = [
            row[1]
            for row in sorted(self.partition_data_rdd.map(lambda x: (x[0], x[2])).collect(), key=lambda z: z[0])
        ]
        self.global_separators = [bound["min_key"] for bound in self.partition_bounds[1:] if bound["min_key"] is not None]
        return time.perf_counter() - build_start

    def _route_partition_for_key(self, key: int) -> int:
        if not self.partition_bounds:
            raise RuntimeError("Distributed index has not been built yet.")
        return bisect_right(self.global_separators, key)

    def _candidate_partitions_for_range(self, low: int, high: int) -> List[int]:
        if low > high:
            return []
        p_low = self._route_partition_for_key(low)
        p_high = self._route_partition_for_key(high)
        return list(range(p_low, p_high + 1))

    def point_query(self, key: int):
        target_partition = self._route_partition_for_key(key)

        def search_partition(iterator):
            # Distributed point-search phase:
            # Spark scans reducer partitions, but only the routed partition rebuilds and
            # searches its local B+ tree for the target key.
            for partition_id, local_tree, _metadata in iterator:
                if partition_id == target_partition:
                    yield from local_tree.search(key)

        return self.partition_data_rdd.mapPartitions(search_partition).collect()

    def range_query(self, low: int, high: int):
        candidate_partitions = set(self._candidate_partitions_for_range(low, high))

        def search_partitions(iterator):
            # Distributed range-search phase:
            # only partitions whose key interval overlaps [low, high] rebuild a local
            # B+ tree and contribute matches back to the driver.
            for partition_id, local_tree, _metadata in iterator:
                if partition_id in candidate_partitions:
                    yield from local_tree.range_search(low, high)

        return self.partition_data_rdd.mapPartitions(search_partitions).collect()

    def directory_height_estimate(self) -> int:
        p = max(1, len(self.partition_bounds))
        fanout = self.tree_order
        h = 1
        while p > fanout:
            p = math.ceil(p / fanout)
            h += 1
        return h


def run_distributed_experiment(
    spark,
    items: Sequence[Tuple[int, Any]],
    *,
    tree_order: int,
    n_partitions: int,
    point_queries: Sequence[int],
    range_queries: Sequence[Tuple[int, int]],
    attribute_name: str = "key",
) -> List[DistributedExperimentRow]:
    records_rdd = spark.sparkContext.parallelize(items, n_partitions)
    distributed_tree = DistributedBPlusTree(spark=spark, tree_order=tree_order, n_partitions=n_partitions)
    build_s = distributed_tree.build(records_rdd)

    point_start = time.perf_counter()
    point_total_results = 0
    for q in point_queries:
        point_total_results += len(distributed_tree.point_query(q))
    point_total_s = time.perf_counter() - point_start

    range_start = time.perf_counter()
    range_total_results = 0
    for low, high in range_queries:
        range_total_results += len(distributed_tree.range_query(low, high))
    range_total_s = time.perf_counter() - range_start

    return [
        DistributedExperimentRow(
            algorithm="DistributedBPlusTree",
            workload=f"dist-bptree-{attribute_name}-point",
            n_items=len(items),
            build_s=build_s,
            query_total_s=point_total_s,
            query_avg_s=point_total_s / len(point_queries) if point_queries else 0.0,
            num_queries=len(point_queries),
            avg_results=point_total_results / len(point_queries) if point_queries else 0.0,
            extra={
                "order": tree_order,
                "n_partitions": n_partitions,
                "indexed_attribute": attribute_name,
                "directory_height_est": distributed_tree.directory_height_estimate(),
                "partition_heights": [b["height"] for b in distributed_tree.partition_bounds],
            },
        ),
        DistributedExperimentRow(
            algorithm="DistributedBPlusTree",
            workload=f"dist-bptree-{attribute_name}-range",
            n_items=len(items),
            build_s=build_s,
            query_total_s=range_total_s,
            query_avg_s=range_total_s / len(range_queries) if range_queries else 0.0,
            num_queries=len(range_queries),
            avg_results=range_total_results / len(range_queries) if range_queries else 0.0,
            extra={
                "order": tree_order,
                "n_partitions": n_partitions,
                "indexed_attribute": attribute_name,
                "directory_height_est": distributed_tree.directory_height_estimate(),
                "partition_heights": [b["height"] for b in distributed_tree.partition_bounds],
            },
        ),
    ]

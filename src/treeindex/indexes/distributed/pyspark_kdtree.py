from __future__ import annotations

import math
import time
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from treeindex.geometry.point import Point
from treeindex.geometry.rect import Rect
from treeindex.indexes.distributed.base import DistributedExperimentRow
from treeindex.indexes.local.kdtree import KDTree

try:
    from pyspark import StorageLevel
    from pyspark.sql import SparkSession
except Exception:
    SparkSession = None
    StorageLevel = None


def _grid_dimensions(n_partitions: int) -> Tuple[int, int]:
    cols = max(1, math.ceil(math.sqrt(n_partitions)))
    rows = max(1, math.ceil(n_partitions / cols))
    return rows, cols


def _grid_partition_id_for_point(
    point: Point,
    global_bounds: Rect,
    *,
    grid_rows: int,
    grid_cols: int,
    n_partitions: int,
) -> int:
    width = max(global_bounds.xmax - global_bounds.xmin, 1.0)
    height = max(global_bounds.ymax - global_bounds.ymin, 1.0)
    cell_width = max(width / grid_cols, 1.0)
    cell_height = max(height / grid_rows, 1.0)
    col = int((point.x - global_bounds.xmin) / cell_width)
    row = int((point.y - global_bounds.ymin) / cell_height)
    col = min(max(col, 0), grid_cols - 1)
    row = min(max(row, 0), grid_rows - 1)
    return min(row * grid_cols + col, n_partitions - 1)


class DistributedKDTree:
    def __init__(self, spark, n_partitions: int = 4, leaf_capacity: int = 1):
        if SparkSession is None:
            raise RuntimeError("PySpark is not available in this environment.")
        self.spark = spark
        self.sc = spark.sparkContext
        self.n_partitions = n_partitions
        self.leaf_capacity = leaf_capacity
        self.partition_data_rdd = None
        self.partition_bounds: List[Dict[str, object]] = []
        self.global_bounds: Optional[Rect] = None
        self.grid_rows, self.grid_cols = _grid_dimensions(n_partitions)

    def build(self, records_rdd) -> float:
        build_start = time.perf_counter()
        point_bounds = records_rdd.keys().map(lambda p: (p.x, p.y, p.x, p.y)).reduce(
            lambda a, b: (
                min(a[0], b[0]),
                min(a[1], b[1]),
                max(a[2], b[2]),
                max(a[3], b[3]),
            )
        )
        self.global_bounds = Rect(*point_bounds)

        grid_rows = self.grid_rows
        grid_cols = self.grid_cols
        n_partitions = self.n_partitions
        global_bounds = self.global_bounds
        leaf_capacity = self.leaf_capacity

        def partition_id_for_point(point: Point) -> int:
            return _grid_partition_id_for_point(
                point,
                global_bounds,
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                n_partitions=n_partitions,
            )

        # MapReduce build phase:
        # map each point to a grid-cell partition id, then partitionBy(...) shuffles
        # colocated points so each reducer partition can build one local KD-tree.
        partitioned = (
            records_rdd.map(lambda row: (partition_id_for_point(row[0]), row))
            .partitionBy(n_partitions, partitionFunc=lambda partition_id: partition_id)
        )
        if StorageLevel is not None:
            partitioned.persist(StorageLevel.MEMORY_AND_DISK)

        def prepare_partition(partition_id: int, iterator: Iterator[Tuple[int, Tuple[Point, int]]]):
            # Reduce/build phase:
            # rebuild one local KD-tree from this Spark partition and emit partition
            # bounds metadata used later for driver-side query routing.
            rows = [row for _partition_key, row in iterator]
            local_tree = KDTree(leaf_capacity=leaf_capacity)
            if rows:
                local_tree.build(rows)
                xs = [point.x for point, _ in rows]
                ys = [point.y for point, _ in rows]
                metadata = {
                    "partition_id": partition_id,
                    "n_items": len(rows),
                    "mbr": Rect(min(xs), min(ys), max(xs), max(ys)),
                    "height": local_tree.height(),
                }
            else:
                metadata = {
                    "partition_id": partition_id,
                    "n_items": 0,
                    "mbr": None,
                    "height": 0,
                }
            yield (partition_id, local_tree, metadata)

        self.partition_data_rdd = partitioned.mapPartitionsWithIndex(prepare_partition)
        if StorageLevel is not None:
            self.partition_data_rdd.persist(StorageLevel.MEMORY_AND_DISK)
        _ = self.partition_data_rdd.count()
        self.partition_bounds = [
            row[1]
            for row in sorted(self.partition_data_rdd.map(lambda x: (x[0], x[2])).collect(), key=lambda item: item[0])
        ]
        return time.perf_counter() - build_start

    def _candidate_partitions_for_query(self, query_rect: Rect) -> List[int]:
        if not self.partition_bounds:
            raise RuntimeError("Distributed index has not been built yet.")
        return [
            int(bound["partition_id"])
            for bound in self.partition_bounds
            if bound["mbr"] is not None and bound["mbr"].intersects(query_rect)
        ]

    def point_query(self, point: Point):
        candidate_partitions = set(
            int(bound["partition_id"])
            for bound in self.partition_bounds
            if bound["mbr"] is not None and bound["mbr"].contains_point(point)
        )

        def search_partitions(iterator: Iterator[Tuple[int, KDTree, Dict[str, object]]]):
            # Distributed point-search phase:
            # only partitions selected by the driver from partition MBRs rebuild and
            # search their local KD-trees for the exact point.
            for partition_id, local_tree, _metadata in iterator:
                if partition_id in candidate_partitions and len(local_tree):
                    yield from local_tree.query(point)

        return self.partition_data_rdd.mapPartitions(search_partitions).collect()

    def range_query(self, query_rect: Rect):
        candidate_partitions = set(self._candidate_partitions_for_query(query_rect))

        def search_partitions(iterator: Iterator[Tuple[int, KDTree, Dict[str, object]]]):
            # Distributed range-search phase:
            # only partitions whose partition MBR intersects the query rectangle rebuild
            # and search their local KD-trees before Spark collects the answers.
            for partition_id, local_tree, _metadata in iterator:
                if partition_id in candidate_partitions and len(local_tree):
                    yield from local_tree.query(query_rect)

        return self.partition_data_rdd.mapPartitions(search_partitions).collect()

    def query(self, query_rect: Rect):
        return self.range_query(query_rect)

    def directory_height_estimate(self) -> int:
        p = max(1, len([bound for bound in self.partition_bounds if bound["n_items"]]))
        fanout = 2
        h = 1
        while p > fanout:
            p = math.ceil(p / fanout)
            h += 1
        return h


def run_distributed_kdtree_experiment(
    spark,
    items: Sequence[Tuple[Point, int]],
    *,
    n_partitions: int,
    leaf_capacity: int,
    point_queries: Sequence[Point],
    queries: Sequence[Rect],
) -> List[DistributedExperimentRow]:
    records_rdd = spark.sparkContext.parallelize(items, n_partitions)
    distributed_tree = DistributedKDTree(spark=spark, n_partitions=n_partitions, leaf_capacity=leaf_capacity)
    build_s = distributed_tree.build(records_rdd)

    point_start = time.perf_counter()
    point_total_results = 0
    for query in point_queries:
        point_total_results += len(distributed_tree.point_query(query))
    point_total_s = time.perf_counter() - point_start

    query_start = time.perf_counter()
    total_results = 0
    for query in queries:
        total_results += len(distributed_tree.range_query(query))
    query_total_s = time.perf_counter() - query_start

    return [
        DistributedExperimentRow(
            algorithm="DistributedKDTree",
            workload="dist-kdtree-point",
            n_items=len(items),
            build_s=build_s,
            query_total_s=point_total_s,
            query_avg_s=point_total_s / len(point_queries) if point_queries else 0.0,
            num_queries=len(point_queries),
            avg_results=point_total_results / len(point_queries) if point_queries else 0.0,
            extra={
                "dims": 2,
                "leaf_capacity": leaf_capacity,
                "n_partitions": n_partitions,
                "grid_rows": distributed_tree.grid_rows,
                "grid_cols": distributed_tree.grid_cols,
                "directory_height_est": distributed_tree.directory_height_estimate(),
                "partition_heights": [bound["height"] for bound in distributed_tree.partition_bounds],
                "query_type": "point",
            },
        ),
        DistributedExperimentRow(
            algorithm="DistributedKDTree",
            workload="dist-kdtree-range",
            n_items=len(items),
            build_s=build_s,
            query_total_s=query_total_s,
            query_avg_s=query_total_s / len(queries) if queries else 0.0,
            num_queries=len(queries),
            avg_results=total_results / len(queries) if queries else 0.0,
            extra={
                "dims": 2,
                "leaf_capacity": leaf_capacity,
                "n_partitions": n_partitions,
                "grid_rows": distributed_tree.grid_rows,
                "grid_cols": distributed_tree.grid_cols,
                "directory_height_est": distributed_tree.directory_height_estimate(),
                "partition_heights": [bound["height"] for bound in distributed_tree.partition_bounds],
                "query_type": "rect-range",
            },
        )
    ]

from __future__ import annotations

import math
import time
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from treeindex.geometry.point import Point
from treeindex.geometry.rect import Rect
from treeindex.indexes.distributed.base import DistributedExperimentRow
from treeindex.indexes.local.rtree import RTree

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


def _grid_partition_id(
    rect: Rect,
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
    center_x = (rect.xmin + rect.xmax) / 2.0
    center_y = (rect.ymin + rect.ymax) / 2.0
    col = int((center_x - global_bounds.xmin) / cell_width)
    row = int((center_y - global_bounds.ymin) / cell_height)
    col = min(max(col, 0), grid_cols - 1)
    row = min(max(row, 0), grid_rows - 1)
    return min(row * grid_cols + col, n_partitions - 1)


class DistributedRTree:
    def __init__(self, spark, max_entries: int = 12, n_partitions: int = 4):
        if SparkSession is None:
            raise RuntimeError("PySpark is not available in this environment.")
        self.spark = spark
        self.sc = spark.sparkContext
        self.max_entries = max_entries
        self.n_partitions = n_partitions
        self.partition_data_rdd = None
        self.partition_bounds: List[Dict[str, object]] = []
        self.global_bounds: Optional[Rect] = None
        self.grid_rows, self.grid_cols = _grid_dimensions(n_partitions)
        self._cell_width = 1.0
        self._cell_height = 1.0

    def build(self, records_rdd) -> float:
        build_start = time.perf_counter()
        rects = records_rdd.keys()
        self.global_bounds = rects.reduce(lambda a, b: a.union(b))
        width = max(self.global_bounds.xmax - self.global_bounds.xmin, 1.0)
        height = max(self.global_bounds.ymax - self.global_bounds.ymin, 1.0)
        self._cell_width = max(width / self.grid_cols, 1.0)
        self._cell_height = max(height / self.grid_rows, 1.0)

        grid_rows = self.grid_rows
        grid_cols = self.grid_cols
        n_partitions = self.n_partitions
        global_bounds = self.global_bounds
        max_entries = self.max_entries

        def partition_id_for_rect(rect: Rect) -> int:
            return _grid_partition_id(
                rect,
                global_bounds,
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                n_partitions=n_partitions,
            )

        # MapReduce build phase:
        # map each rectangle to a spatial partition id, then partitionBy(...) shuffles
        # rectangles so one reducer partition can build one local R-tree.
        partitioned = (
            records_rdd.map(lambda row: (partition_id_for_rect(row[0]), row))
            .partitionBy(n_partitions, partitionFunc=lambda partition_id: partition_id)
        )
        if StorageLevel is not None:
            partitioned.persist(StorageLevel.MEMORY_AND_DISK)

        def prepare_partition(partition_id: int, iterator: Iterator[Tuple[int, Tuple[Rect, int]]]):
            # Reduce/build phase:
            # rebuild one local R-tree from this Spark partition and emit partition MBR
            # metadata so the driver can route later queries.
            rows = [row for _partition_key, row in iterator]
            local_tree = RTree(max_entries=max_entries)
            if rows:
                local_tree.build(rows)
                partition_mbr = Rect.enclosing(rect for rect, _ in rows)
                metadata = {
                    "partition_id": partition_id,
                    "n_items": len(rows),
                    "mbr": partition_mbr,
                    "height": local_tree.height(),
                }
            else:
                metadata = {
                    "partition_id": partition_id,
                    "n_items": 0,
                    "mbr": None,
                    "height": 0,
                }
            yield (partition_id, rows, metadata)

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
        max_entries = self.max_entries

        def search_partitions(iterator: Iterator[Tuple[int, List[Tuple[Rect, int]], Dict[str, object]]]):
            # Distributed point-search phase:
            # after driver-side routing by partition MBR, only candidate partitions
            # rebuild/search their local R-trees for rectangles containing the point.
            for partition_id, rows, _metadata in iterator:
                if partition_id in candidate_partitions and rows:
                    local_tree = RTree(max_entries=max_entries)
                    local_tree.build(rows)
                    yield from local_tree.query(point)

        return self.partition_data_rdd.mapPartitions(search_partitions).collect()

    def intersection_query(self, query_rect: Rect):
        candidate_partitions = set(self._candidate_partitions_for_query(query_rect))
        max_entries = self.max_entries

        def search_partitions(iterator: Iterator[Tuple[int, List[Tuple[Rect, int]], Dict[str, object]]]):
            # Distributed range/intersection phase:
            # only partitions whose partition MBR intersects the query rectangle rebuild
            # and search their local R-trees before Spark collects the partial results.
            for partition_id, rows, _metadata in iterator:
                if partition_id in candidate_partitions and rows:
                    local_tree = RTree(max_entries=max_entries)
                    local_tree.build(rows)
                    yield from local_tree.query(query_rect)

        return self.partition_data_rdd.mapPartitions(search_partitions).collect()

    def query(self, query_rect: Rect):
        return self.intersection_query(query_rect)

    def directory_height_estimate(self) -> int:
        p = max(1, len([bound for bound in self.partition_bounds if bound["n_items"]]))
        fanout = self.max_entries
        h = 1
        while p > fanout:
            p = math.ceil(p / fanout)
            h += 1
        return h


def run_distributed_rtree_experiment(
    spark,
    items: Sequence[Tuple[Rect, int]],
    *,
    max_entries: int,
    n_partitions: int,
    point_queries: Sequence[Point],
    queries: Sequence[Rect],
) -> List[DistributedExperimentRow]:
    records_rdd = spark.sparkContext.parallelize(items, n_partitions)
    distributed_tree = DistributedRTree(spark=spark, max_entries=max_entries, n_partitions=n_partitions)
    build_s = distributed_tree.build(records_rdd)

    point_start = time.perf_counter()
    point_total_results = 0
    for query in point_queries:
        point_total_results += len(distributed_tree.point_query(query))
    point_total_s = time.perf_counter() - point_start

    query_start = time.perf_counter()
    total_results = 0
    for query in queries:
        total_results += len(distributed_tree.intersection_query(query))
    query_total_s = time.perf_counter() - query_start

    return [
        DistributedExperimentRow(
            algorithm="DistributedRTree",
            workload="dist-rtree-point",
            n_items=len(items),
            build_s=build_s,
            query_total_s=point_total_s,
            query_avg_s=point_total_s / len(point_queries) if point_queries else 0.0,
            num_queries=len(point_queries),
            avg_results=point_total_results / len(point_queries) if point_queries else 0.0,
            extra={
                "max_entries": max_entries,
                "n_partitions": n_partitions,
                "grid_rows": distributed_tree.grid_rows,
                "grid_cols": distributed_tree.grid_cols,
                "directory_height_est": distributed_tree.directory_height_estimate(),
                "partition_heights": [bound["height"] for bound in distributed_tree.partition_bounds],
                "query_type": "point-location",
            },
        ),
        DistributedExperimentRow(
            algorithm="DistributedRTree",
            workload="dist-rtree-intersection",
            n_items=len(items),
            build_s=build_s,
            query_total_s=query_total_s,
            query_avg_s=query_total_s / len(queries) if queries else 0.0,
            num_queries=len(queries),
            avg_results=total_results / len(queries) if queries else 0.0,
            extra={
                "max_entries": max_entries,
                "n_partitions": n_partitions,
                "grid_rows": distributed_tree.grid_rows,
                "grid_cols": distributed_tree.grid_cols,
                "directory_height_est": distributed_tree.directory_height_estimate(),
                "partition_heights": [bound["height"] for bound in distributed_tree.partition_bounds],
                "query_type": "rect-intersection",
            },
        )
    ]

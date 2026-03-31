from __future__ import annotations

import math
import time
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from treeindex.geometry.point import Point
from treeindex.geometry.rect import Rect
from treeindex.indexes.distributed.base import DistributedExperimentRow
from treeindex.indexes.local.quadtree import QuadTree

try:
    from pyspark import StorageLevel
    from pyspark.sql import SparkSession
except Exception:
    SparkSession = None
    StorageLevel = None


def _quadtree_depth_for_partitions(n_partitions: int) -> int:
    depth = 0
    cells = 1
    while cells < max(1, n_partitions):
        depth += 1
        cells *= 4
    return depth


def _quadtree_partition_id_for_point(
    point: Point,
    global_bounds: Rect,
    *,
    depth: int,
    n_partitions: int,
) -> int:
    bounds = global_bounds
    partition_id = 0
    for _ in range(depth):
        mid_x = (bounds.xmin + bounds.xmax) / 2.0
        mid_y = (bounds.ymin + bounds.ymax) / 2.0
        east = point.x > mid_x
        north = point.y > mid_y
        quadrant = 0
        if east and not north:
            quadrant = 1
            bounds = Rect(mid_x, bounds.ymin, bounds.xmax, mid_y)
        elif not east and north:
            quadrant = 2
            bounds = Rect(bounds.xmin, mid_y, mid_x, bounds.ymax)
        elif east and north:
            quadrant = 3
            bounds = Rect(mid_x, mid_y, bounds.xmax, bounds.ymax)
        else:
            bounds = Rect(bounds.xmin, bounds.ymin, mid_x, mid_y)
        partition_id = partition_id * 4 + quadrant
    return min(partition_id, n_partitions - 1)


class DistributedQuadTree:
    def __init__(self, spark, bucket_capacity: int = 8, max_depth: int = 12, n_partitions: int = 4):
        if SparkSession is None:
            raise RuntimeError("PySpark is not available in this environment.")
        self.spark = spark
        self.sc = spark.sparkContext
        self.bucket_capacity = bucket_capacity
        self.max_depth = max_depth
        self.n_partitions = n_partitions
        self.partition_depth = _quadtree_depth_for_partitions(n_partitions)
        self.partition_data_rdd = None
        self.partition_bounds: List[Dict[str, object]] = []
        self.global_bounds: Optional[Rect] = None

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
        global_bounds = self.global_bounds
        partition_depth = self.partition_depth
        n_partitions = self.n_partitions
        bucket_capacity = self.bucket_capacity
        max_depth = self.max_depth

        def partition_id_for_point(point: Point) -> int:
            return _quadtree_partition_id_for_point(
                point,
                global_bounds,
                depth=partition_depth,
                n_partitions=n_partitions,
            )

        # MapReduce build phase:
        # map each point to a quadtree-style partition id, then partitionBy(...) shuffles
        # points so each reducer partition can build one local Quadtree.
        partitioned = (
            records_rdd.map(lambda row: (partition_id_for_point(row[0]), row))
            .partitionBy(n_partitions, partitionFunc=lambda partition_id: partition_id)
        )
        if StorageLevel is not None:
            partitioned.persist(StorageLevel.MEMORY_AND_DISK)

        def prepare_partition(partition_id: int, iterator: Iterator[Tuple[int, Tuple[Point, int]]]):
            # Reduce/build phase:
            # rebuild one local Quadtree from this Spark partition and emit partition
            # bounds metadata so the driver can route point and range queries.
            rows = [row for _partition_key, row in iterator]
            local_tree = QuadTree(bucket_capacity=bucket_capacity, max_depth=max_depth)
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
        def search_partitions(iterator: Iterator[Tuple[int, QuadTree, Dict[str, object]]]):
            # Distributed point-search phase:
            # after driver-side routing with partition MBRs, only candidate partitions
            # rebuild and search their local Quadtrees for the target point.
            for partition_id, local_tree, _metadata in iterator:
                if partition_id in candidate_partitions and len(local_tree):
                    yield from local_tree.query(point)

        return self.partition_data_rdd.mapPartitions(search_partitions).collect()

    def range_query(self, query_rect: Rect):
        candidate_partitions = set(self._candidate_partitions_for_query(query_rect))
        def search_partitions(iterator: Iterator[Tuple[int, QuadTree, Dict[str, object]]]):
            # Distributed range-search phase:
            # only partitions whose partition MBR intersects the query rectangle rebuild
            # and search their local Quadtrees before Spark collects the matches.
            for partition_id, local_tree, _metadata in iterator:
                if partition_id in candidate_partitions and len(local_tree):
                    yield from local_tree.query(query_rect)

        return self.partition_data_rdd.mapPartitions(search_partitions).collect()

    def query(self, query_rect: Rect):
        return self.range_query(query_rect)

    def directory_height_estimate(self) -> int:
        p = max(1, len([bound for bound in self.partition_bounds if bound["n_items"]]))
        fanout = 4
        h = 1
        while p > fanout:
            p = math.ceil(p / fanout)
            h += 1
        return h


def run_distributed_quadtree_experiment(
    spark,
    items: Sequence[Tuple[Point, int]],
    *,
    bucket_capacity: int,
    max_depth: int,
    n_partitions: int,
    point_queries: Sequence[Point],
    queries: Sequence[Rect],
) -> List[DistributedExperimentRow]:
    records_rdd = spark.sparkContext.parallelize(items, n_partitions)
    distributed_tree = DistributedQuadTree(
        spark=spark,
        bucket_capacity=bucket_capacity,
        max_depth=max_depth,
        n_partitions=n_partitions,
    )
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
            algorithm="DistributedQuadTree",
            workload="dist-quadtree-point",
            n_items=len(items),
            build_s=build_s,
            query_total_s=point_total_s,
            query_avg_s=point_total_s / len(point_queries) if point_queries else 0.0,
            num_queries=len(point_queries),
            avg_results=point_total_results / len(point_queries) if point_queries else 0.0,
            extra={
                "bucket_capacity": bucket_capacity,
                "max_depth": max_depth,
                "n_partitions": n_partitions,
                "partition_depth": distributed_tree.partition_depth,
                "directory_height_est": distributed_tree.directory_height_estimate(),
                "partition_heights": [bound["height"] for bound in distributed_tree.partition_bounds],
                "query_type": "point",
            },
        ),
        DistributedExperimentRow(
            algorithm="DistributedQuadTree",
            workload="dist-quadtree-range",
            n_items=len(items),
            build_s=build_s,
            query_total_s=query_total_s,
            query_avg_s=query_total_s / len(queries) if queries else 0.0,
            num_queries=len(queries),
            avg_results=total_results / len(queries) if queries else 0.0,
            extra={
                "bucket_capacity": bucket_capacity,
                "max_depth": max_depth,
                "n_partitions": n_partitions,
                "partition_depth": distributed_tree.partition_depth,
                "directory_height_est": distributed_tree.directory_height_estimate(),
                "partition_heights": [bound["height"] for bound in distributed_tree.partition_bounds],
                "query_type": "rect-range",
            },
        )
    ]

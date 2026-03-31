from __future__ import annotations

import unittest

from treeindex.data.generators import generate_points, sample_point_rect_queries
from treeindex.geometry.point import Point
from treeindex.geometry.rect import Rect
from treeindex.indexes.distributed.pyspark_quadtree import (
    DistributedQuadTree,
    SparkSession,
    _quadtree_depth_for_partitions,
    _quadtree_partition_id_for_point,
    run_distributed_quadtree_experiment,
)
from treeindex.indexes.local.quadtree import QuadTree


class QuadTreeLocalTest(unittest.TestCase):
    def test_generated_points_include_two_attributes(self) -> None:
        items = generate_points(10, seed=6)
        self.assertTrue(all(hasattr(record, "attr1") and hasattr(record, "attr2") for _point, record in items))

    def test_point_query_matches_exact_values(self) -> None:
        items = [
            (Point(1.0, 1.0), 1),
            (Point(2.0, 4.0), 2),
            (Point(5.0, 3.0), 3),
        ]
        tree = QuadTree(bucket_capacity=2, max_depth=6)
        tree.build(items)
        self.assertEqual(tree.query(Point(2.0, 4.0)), [2])
        self.assertEqual(tree.query(Point(9.0, 9.0)), [])

    def test_range_query_matches_bruteforce(self) -> None:
        items = [
            (Point(1.0, 1.0), 1),
            (Point(2.0, 4.0), 2),
            (Point(5.0, 3.0), 3),
            (Point(7.0, 8.0), 4),
            (Point(9.0, 2.0), 5),
        ]
        tree = QuadTree(bucket_capacity=2, max_depth=6)
        tree.build(items)

        query = Rect(0.0, 0.0, 6.0, 4.5)
        expected = sorted(
            value
            for point, value in items
            if query.xmin <= point.x <= query.xmax and query.ymin <= point.y <= query.ymax
        )
        self.assertEqual(sorted(tree.query(query)), expected)
        self.assertGreater(tree.height(), 0)

    def test_insert_preserves_query_results(self) -> None:
        tree = QuadTree(bucket_capacity=1, max_depth=6)
        tree.build([(Point(2.0, 2.0), 10), (Point(8.0, 8.0), 20)])
        tree.insert(Point(3.0, 3.0), 30)
        self.assertEqual(sorted(tree.query(Rect(0.0, 0.0, 5.0, 5.0))), [10, 30])


class QuadTreePartitioningTest(unittest.TestCase):
    def test_partition_depth_covers_requested_partitions(self) -> None:
        depth = _quadtree_depth_for_partitions(7)
        self.assertGreaterEqual(4**depth, 7)

    def test_partition_id_stays_in_range(self) -> None:
        global_bounds = Rect(0.0, 0.0, 100.0, 100.0)
        partition_id = _quadtree_partition_id_for_point(
            Point(74.0, 74.0),
            global_bounds,
            depth=_quadtree_depth_for_partitions(5),
            n_partitions=5,
        )
        self.assertGreaterEqual(partition_id, 0)
        self.assertLess(partition_id, 5)


class QuadTreeSparkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if SparkSession is None:
            raise unittest.SkipTest("PySpark is not available.")
        try:
            cls.spark = (
                SparkSession.builder.master("local[2]")
                .appName("DistributedQuadTreeTest")
                .config("spark.ui.enabled", "false")
                .getOrCreate()
            )
            cls.spark.sparkContext.setLogLevel("ERROR")
        except Exception as exc:
            raise unittest.SkipTest(f"Unable to create local Spark session: {exc}") from exc

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "spark"):
            cls.spark.stop()

    def test_distributed_queries_match_local_quadtree(self) -> None:
        items = generate_points(80, seed=18)
        point_queries = [point for point, _ in items[:6]]
        queries = sample_point_rect_queries(6, seed=19)

        local_tree = QuadTree(bucket_capacity=4, max_depth=8)
        local_tree.build(items)

        distributed_tree = DistributedQuadTree(self.spark, bucket_capacity=4, max_depth=8, n_partitions=4)
        build_s = distributed_tree.build(self.spark.sparkContext.parallelize(items, 4))

        self.assertGreater(build_s, 0.0)
        self.assertEqual(len(distributed_tree.partition_bounds), 4)
        self.assertTrue(any(bound["n_items"] for bound in distributed_tree.partition_bounds))

        for query in point_queries:
            self.assertEqual(
                sorted(local_tree.query(query)),
                sorted(distributed_tree.point_query(query)),
            )
            matches = distributed_tree.point_query(query)
            if matches:
                self.assertTrue(all(hasattr(record, "attr1") and hasattr(record, "attr2") for record in matches))

        for query in queries:
            self.assertEqual(
                sorted(local_tree.query(query)),
                sorted(distributed_tree.range_query(query)),
            )

    def test_experiment_row_shape(self) -> None:
        items = generate_points(60, seed=22)
        queries = [
            Rect(0.0, 0.0, 5000.0, 5000.0),
            Rect(2500.0, 2500.0, 7500.0, 7500.0),
            Rect(5000.0, 5000.0, 10000.0, 10000.0),
        ]

        rows = run_distributed_quadtree_experiment(
            self.spark,
            items,
            bucket_capacity=4,
            max_depth=8,
            n_partitions=4,
            point_queries=[point for point, _ in items[:3]],
            queries=queries,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].workload, "dist-quadtree-point")
        self.assertEqual(rows[1].workload, "dist-quadtree-range")
        for row in rows:
            self.assertEqual(row.algorithm, "DistributedQuadTree")
            self.assertEqual(row.n_items, len(items))
            self.assertIn("bucket_capacity", row.extra)
            self.assertIn("partition_heights", row.extra)


if __name__ == "__main__":
    unittest.main()

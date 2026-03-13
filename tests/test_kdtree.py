from __future__ import annotations

import unittest

from treeindex.data.generators import generate_points, sample_point_rect_queries
from treeindex.geometry.point import Point
from treeindex.geometry.rect import Rect
from treeindex.indexes.distributed.pyspark_kdtree import (
    DistributedKDTree,
    SparkSession,
    _grid_dimensions,
    _grid_partition_id_for_point,
    run_distributed_kdtree_experiment,
)
from treeindex.indexes.local.kdtree import KDTree


class KDTreeLocalTest(unittest.TestCase):
    def test_point_query_matches_exact_values(self) -> None:
        items = [
            (Point(1.0, 1.0), 1),
            (Point(2.0, 4.0), 2),
            (Point(5.0, 3.0), 3),
        ]
        tree = KDTree()
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
        tree = KDTree()
        tree.build(items)

        query = Rect(0.0, 0.0, 6.0, 4.5)
        expected = sorted(
            value
            for point, value in items
            if query.xmin <= point.x <= query.xmax and query.ymin <= point.y <= query.ymax
        )
        self.assertEqual(sorted(tree.query(query)), expected)
        self.assertGreater(tree.height(), 0)

    def test_insert_rebuild_preserves_query_results(self) -> None:
        tree = KDTree()
        tree.build([(Point(2.0, 2.0), 10), (Point(8.0, 8.0), 20)])
        tree.insert(Point(3.0, 3.0), 30)
        self.assertEqual(sorted(tree.query(Rect(0.0, 0.0, 5.0, 5.0))), [10, 30])


class KDTreePartitioningTest(unittest.TestCase):
    def test_grid_dimensions_cover_requested_partitions(self) -> None:
        rows, cols = _grid_dimensions(7)
        self.assertGreaterEqual(rows * cols, 7)
        self.assertGreaterEqual(rows, 1)
        self.assertGreaterEqual(cols, 1)

    def test_grid_partition_id_stays_in_range(self) -> None:
        global_bounds = Rect(0.0, 0.0, 100.0, 100.0)
        rows, cols = _grid_dimensions(5)
        partition_id = _grid_partition_id_for_point(
            Point(74.0, 74.0),
            global_bounds,
            grid_rows=rows,
            grid_cols=cols,
            n_partitions=5,
        )
        self.assertGreaterEqual(partition_id, 0)
        self.assertLess(partition_id, 5)


class KDTreeSparkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if SparkSession is None:
            raise unittest.SkipTest("PySpark is not available.")
        try:
            cls.spark = (
                SparkSession.builder.master("local[2]")
                .appName("DistributedKDTreeTest")
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

    def test_distributed_queries_match_local_kdtree(self) -> None:
        items = generate_points(80, seed=12)
        point_queries = [point for point, _ in items[:6]]
        queries = sample_point_rect_queries(6, seed=13)

        local_tree = KDTree()
        local_tree.build(items)

        distributed_tree = DistributedKDTree(self.spark, n_partitions=4)
        build_s = distributed_tree.build(self.spark.sparkContext.parallelize(items, 4))

        self.assertGreater(build_s, 0.0)
        self.assertEqual(len(distributed_tree.partition_bounds), 4)
        self.assertTrue(any(bound["n_items"] for bound in distributed_tree.partition_bounds))

        for query in point_queries:
            self.assertEqual(
                sorted(local_tree.query(query)),
                sorted(distributed_tree.point_query(query)),
            )

        for query in queries:
            self.assertEqual(
                sorted(local_tree.query(query)),
                sorted(distributed_tree.range_query(query)),
            )

    def test_experiment_row_shape(self) -> None:
        items = generate_points(60, seed=21)
        queries = [
            Rect(0.0, 0.0, 5000.0, 5000.0),
            Rect(2500.0, 2500.0, 7500.0, 7500.0),
            Rect(5000.0, 5000.0, 10000.0, 10000.0),
        ]

        rows = run_distributed_kdtree_experiment(
            self.spark,
            items,
            n_partitions=4,
            point_queries=[point for point, _ in items[:3]],
            queries=queries,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].workload, "dist-kdtree-point")
        self.assertEqual(rows[1].workload, "dist-kdtree-range")
        for row in rows:
            self.assertEqual(row.algorithm, "DistributedKDTree")
            self.assertEqual(row.n_items, len(items))
            self.assertIn("n_partitions", row.extra)
            self.assertIn("partition_heights", row.extra)


if __name__ == "__main__":
    unittest.main()

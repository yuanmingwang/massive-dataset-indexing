from __future__ import annotations

import unittest

from treeindex.data.generators import generate_rectangles, sample_rect_queries
from treeindex.geometry.rect import Rect
from treeindex.geometry.point import Point
from treeindex.indexes.distributed.pyspark_rtree import (
    DistributedRTree,
    SparkSession,
    _grid_dimensions,
    _grid_partition_id,
    run_distributed_rtree_experiment,
)
from treeindex.indexes.local.rtree import RTree


class DistributedRTreePartitioningTest(unittest.TestCase):
    def test_grid_dimensions_cover_requested_partitions(self) -> None:
        rows, cols = _grid_dimensions(8)
        self.assertGreaterEqual(rows * cols, 8)
        self.assertGreaterEqual(rows, 1)
        self.assertGreaterEqual(cols, 1)

    def test_grid_partition_id_stays_in_range(self) -> None:
        global_bounds = Rect(0.0, 0.0, 100.0, 100.0)
        rows, cols = _grid_dimensions(5)
        rect = Rect(74.0, 74.0, 80.0, 80.0)
        partition_id = _grid_partition_id(
            rect,
            global_bounds,
            grid_rows=rows,
            grid_cols=cols,
            n_partitions=5,
        )
        self.assertGreaterEqual(partition_id, 0)
        self.assertLess(partition_id, 5)


class RTreeLocalPointQueryTest(unittest.TestCase):
    def test_generated_rectangles_include_two_attributes(self) -> None:
        items = generate_rectangles(10, seed=9)
        self.assertTrue(all(hasattr(record, "attr1") and hasattr(record, "attr2") for _rect, record in items))

    def test_point_query_matches_containment(self) -> None:
        items = [
            (Rect(0.0, 0.0, 2.0, 2.0), 1),
            (Rect(1.0, 1.0, 4.0, 4.0), 2),
            (Rect(5.0, 5.0, 6.0, 6.0), 3),
        ]
        tree = RTree(max_entries=4)
        tree.build(items)

        self.assertEqual(sorted(tree.query(Rect(1.5, 1.5, 1.5, 1.5))), [1, 2])
        self.assertEqual(sorted(tree.query(Point(1.5, 1.5))), [1, 2])
        self.assertEqual(tree.query(Point(9.0, 9.0)), [])


class DistributedRTreeSparkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if SparkSession is None:
            raise unittest.SkipTest("PySpark is not available.")
        try:
            cls.spark = (
                SparkSession.builder.master("local[2]")
                .appName("DistributedRTreeTest")
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

    def test_intersection_queries_match_local_rtree(self) -> None:
        items = generate_rectangles(80, max_rect_width=120.0, max_rect_height=120.0, seed=7)
        point_queries = [
            Point((rect.xmin + rect.xmax) / 2.0, (rect.ymin + rect.ymax) / 2.0)
            for rect, _ in items[:6]
        ]
        queries = sample_rect_queries(6, max_query_width=500.0, max_query_height=500.0, seed=8)

        local_tree = RTree(max_entries=6)
        local_tree.build(items)

        distributed_tree = DistributedRTree(self.spark, max_entries=6, n_partitions=4)
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
                sorted(distributed_tree.intersection_query(query)),
            )

    def test_experiment_row_shape(self) -> None:
        items = generate_rectangles(60, max_rect_width=100.0, max_rect_height=100.0, seed=11)
        queries = [
            Rect(0.0, 0.0, 5000.0, 5000.0),
            Rect(2500.0, 2500.0, 7500.0, 7500.0),
            Rect(5000.0, 5000.0, 10000.0, 10000.0),
        ]

        rows = run_distributed_rtree_experiment(
            self.spark,
            items,
            max_entries=8,
            n_partitions=4,
            point_queries=[
                Point((rect.xmin + rect.xmax) / 2.0, (rect.ymin + rect.ymax) / 2.0)
                for rect, _ in items[:3]
            ],
            queries=queries,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].workload, "dist-rtree-point")
        self.assertEqual(rows[1].workload, "dist-rtree-intersection")
        for row in rows:
            self.assertEqual(row.algorithm, "DistributedRTree")
            self.assertEqual(row.n_items, len(items))
            self.assertIn("max_entries", row.extra)
            self.assertIn("partition_heights", row.extra)


if __name__ == "__main__":
    unittest.main()

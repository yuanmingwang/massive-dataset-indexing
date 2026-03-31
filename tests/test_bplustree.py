from __future__ import annotations

import unittest

from treeindex.data.generators import (
    generate_key_value_pairs,
    project_records_to_attribute,
    sample_attribute_point_queries,
    sample_attribute_range_queries,
)
from treeindex.indexes.distributed.pyspark_bplustree import SparkSession, DistributedBPlusTree, run_distributed_experiment
from treeindex.indexes.local.bplustree import BPlusTree


class BPlusTreeAttributeLocalTest(unittest.TestCase):
    def test_generated_records_include_two_non_key_attributes(self) -> None:
        items = generate_key_value_pairs(10, seed=3)

        self.assertEqual(len(items), 10)
        self.assertTrue(all(hasattr(record, "attr1") and hasattr(record, "attr2") for _key, record in items))

    def test_local_attribute_indexes_match_bruteforce_queries(self) -> None:
        items = generate_key_value_pairs(80, seed=7)

        for attribute in ("attr1", "attr2"):
            indexed_items = project_records_to_attribute(items, attribute)
            tree = BPlusTree(order=6)
            tree.build(indexed_items)

            point_queries = sample_attribute_point_queries(items, attribute, 5, seed=11)
            for query in point_queries:
                expected = sorted(record for value, record in indexed_items if value == query)
                self.assertEqual(sorted(tree.search(query)), expected)

            range_queries = sample_attribute_range_queries(items, attribute, 5, max_width=1000, seed=13)
            for low, high in range_queries:
                expected = sorted(record for value, record in indexed_items if low <= value <= high)
                self.assertEqual(sorted(tree.range_search(low, high)), expected)


class BPlusTreeAttributeSparkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if SparkSession is None:
            raise unittest.SkipTest("PySpark is not available in this environment.")
        try:
            cls.spark = (
                SparkSession.builder.master("local[2]")
                .appName("DistributedBPlusTreeAttributeTest")
                .getOrCreate()
            )
        except Exception as exc:
            raise unittest.SkipTest(f"Unable to create local Spark session: {exc}") from exc

    @classmethod
    def tearDownClass(cls) -> None:
        if getattr(cls, "spark", None) is not None:
            cls.spark.stop()

    def test_distributed_attribute_queries_match_local_bplustree(self) -> None:
        items = generate_key_value_pairs(80, seed=17)
        attribute = "attr1"
        indexed_items = project_records_to_attribute(items, attribute)

        local_tree = BPlusTree(order=6)
        local_tree.build(indexed_items)

        distributed_tree = DistributedBPlusTree(self.spark, tree_order=6, n_partitions=4)
        build_s = distributed_tree.build(self.spark.sparkContext.parallelize(indexed_items, 4))
        self.assertGreaterEqual(build_s, 0.0)

        point_queries = sample_attribute_point_queries(items, attribute, 5, seed=19)
        for query in point_queries:
            self.assertEqual(sorted(local_tree.search(query)), sorted(distributed_tree.point_query(query)))

        range_queries = sample_attribute_range_queries(items, attribute, 5, max_width=1000, seed=23)
        for low, high in range_queries:
            self.assertEqual(sorted(local_tree.range_search(low, high)), sorted(distributed_tree.range_query(low, high)))

    def test_distributed_experiment_reports_all_three_indexed_attributes(self) -> None:
        items = generate_key_value_pairs(60, seed=29)

        rows = []
        for attribute in ("key", "attr1", "attr2"):
            rows.extend(
                run_distributed_experiment(
                    self.spark,
                    project_records_to_attribute(items, attribute),
                    tree_order=6,
                    n_partitions=4,
                    point_queries=sample_attribute_point_queries(items, attribute, 3, seed=31),
                    range_queries=sample_attribute_range_queries(items, attribute, 3, max_width=1000, seed=37),
                    attribute_name=attribute,
                )
            )

        self.assertEqual(len(rows), 6)
        self.assertEqual(
            [row.workload for row in rows],
            [
                "dist-bptree-key-point",
                "dist-bptree-key-range",
                "dist-bptree-attr1-point",
                "dist-bptree-attr1-range",
                "dist-bptree-attr2-point",
                "dist-bptree-attr2-range",
            ],
        )


if __name__ == "__main__":
    unittest.main()

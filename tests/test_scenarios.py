from __future__ import annotations

import unittest

from treeindex.experiments.scenarios import local_bplustree_demo, local_comparison_demo


class LocalBPlusTreeScenarioTest(unittest.TestCase):
    def test_local_bplustree_demo_indexes_key_and_two_attributes(self) -> None:
        rows = local_bplustree_demo(
            {
                "n_items": 40,
                "key_min": 0,
                "key_max": 400,
                "attr1_min": 1000,
                "attr1_max": 1400,
                "attr2_min": 2000,
                "attr2_max": 2400,
                "allow_duplicates": True,
                "seed": 1,
                "tree_order": 16,
                "n_point_queries": 5,
                "n_range_queries": 5,
                "range_low_min": 0,
                "range_low_max": 300,
                "range_max_width": 40,
            }
        )

        self.assertEqual(len(rows), 6)
        self.assertEqual(
            [row.workload for row in rows],
            [
                "bptree-key-point",
                "bptree-key-range",
                "bptree-attr1-point",
                "bptree-attr1-range",
                "bptree-attr2-point",
                "bptree-attr2-range",
            ],
        )
        self.assertEqual(
            [row.extra["indexed_attribute"] for row in rows],
            ["key", "key", "attr1", "attr1", "attr2", "attr2"],
        )
        self.assertEqual(rows[0].build_seconds, rows[1].build_seconds)
        self.assertEqual(rows[2].build_seconds, rows[3].build_seconds)
        self.assertEqual(rows[4].build_seconds, rows[5].build_seconds)


class LocalComparisonScenarioTest(unittest.TestCase):
    def test_local_comparison_demo_returns_all_four_tree_sections(self) -> None:
        sections = local_comparison_demo(
            {
                "n_items": 40,
                "key_min": 0,
                "key_max": 400,
                "allow_duplicates": True,
                "seed": 1,
                "orders": [16, 32, 64],
                "n_point_queries": 5,
                "n_range_queries": 5,
                "range_low_min": 0,
                "range_low_max": 300,
                "range_max_width": 40,
            },
            {
                "n_items": 40,
                "space_width": 100.0,
                "space_height": 100.0,
                "max_rect_width": 8.0,
                "max_rect_height": 8.0,
                "seed": 1,
                "capacities": [16, 32, 64],
                "n_point_queries": 5,
                "n_queries": 5,
                "max_query_width": 20.0,
                "max_query_height": 20.0,
            },
            {
                "n_items": 40,
                "space_width": 100.0,
                "space_height": 100.0,
                "seed": 1,
                "leaf_capacities": [16, 32, 64],
                "n_point_queries": 5,
                "n_queries": 5,
                "max_query_width": 20.0,
                "max_query_height": 20.0,
            },
            {
                "n_items": 40,
                "space_width": 100.0,
                "space_height": 100.0,
                "seed": 1,
                "bucket_capacities": [16, 32, 64],
                "n_point_queries": 5,
                "n_queries": 5,
                "max_query_width": 20.0,
                "max_query_height": 20.0,
                "max_depth": 6,
            },
        )

        self.assertEqual(len(sections), 4)
        self.assertEqual(
            [title for title, _rows in sections],
            [
                "=== Local comparison: B+ Tree orders ===",
                "=== Local comparison: R-Tree capacities ===",
                "=== Local comparison: KD-Tree leaf capacities ===",
                "=== Local comparison: Quadtree bucket capacities ===",
            ],
        )
        for _title, rows in sections:
            if "B+ Tree" in _title:
                self.assertEqual(len(rows), 18)
                self.assertEqual(
                    {row.extra["indexed_attribute"] for row in rows},
                    {"key", "attr1", "attr2"},
                )
                for offset in range(0, len(rows), 2):
                    self.assertEqual(rows[offset].build_seconds, rows[offset + 1].build_seconds)
            else:
                self.assertEqual(len(rows), 6)
                for offset in range(0, len(rows), 2):
                    self.assertEqual(rows[offset].build_seconds, rows[offset + 1].build_seconds)


if __name__ == "__main__":
    unittest.main()

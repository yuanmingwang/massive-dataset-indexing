from __future__ import annotations

import unittest

from treeindex.experiments.scenarios import local_comparison_demo


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
            self.assertEqual(len(rows), 6)


if __name__ == "__main__":
    unittest.main()

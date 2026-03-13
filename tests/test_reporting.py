from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from treeindex.core.results import ExperimentResult
from treeindex.utils.reporting import write_experiment_report, write_text_report


class ReportingTest(unittest.TestCase):
    def test_write_experiment_report_creates_text_and_json(self) -> None:
        row = ExperimentResult(
            algorithm="KDTree",
            workload="kdtree-range",
            n_items=100,
            build_seconds=0.1,
            query_total_seconds=0.2,
            query_avg_seconds=0.02,
            num_queries=10,
            total_results=25,
            avg_results=2.5,
            extra={"dims": 2},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            txt_path, json_path = write_experiment_report(
                title="=== Test Experiment ===",
                rows=[row],
                filename_prefix="reporting-test",
                output_dir=tmpdir,
            )

            self.assertTrue(Path(txt_path).exists())
            self.assertTrue(Path(json_path).exists())
            self.assertIn("=== Test Experiment ===", Path(txt_path).read_text(encoding="utf-8"))
            self.assertIn("KDTree", Path(txt_path).read_text(encoding="utf-8"))
            self.assertIn('"algorithm": "KDTree"', Path(json_path).read_text(encoding="utf-8"))

    def test_write_experiment_report_can_skip_json(self) -> None:
        row = ExperimentResult(
            algorithm="QuadTree",
            workload="quadtree-range",
            n_items=50,
            build_seconds=0.1,
            query_total_seconds=0.2,
            query_avg_seconds=0.02,
            num_queries=10,
            total_results=20,
            avg_results=2.0,
            extra={"max_depth": 4},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            txt_path, json_path = write_experiment_report(
                title="=== Text Only ===",
                rows=[row],
                filename_prefix="reporting-text-only",
                output_dir=tmpdir,
                write_json=False,
            )

            self.assertTrue(Path(txt_path).exists())
            self.assertIsNone(json_path)
            self.assertEqual(len(list(Path(tmpdir).glob("*.json"))), 0)

    def test_write_text_report_creates_text_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            txt_path = write_text_report(
                text="=== Local demo ===\nKDTree results",
                filename_prefix="local-demo",
                output_dir=tmpdir,
            )

            self.assertTrue(Path(txt_path).exists())
            self.assertEqual(txt_path.suffix, ".txt")
            self.assertIn("KDTree results", Path(txt_path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

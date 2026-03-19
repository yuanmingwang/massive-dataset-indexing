from __future__ import annotations

import argparse

from treeindex.experiments.scenarios import (
    local_bplustree_demo,
    local_comparison_demo,
    local_kdtree_demo,
    local_quadtree_demo,
    local_rtree_demo,
)
from treeindex.utils.config import default_config_path, load_json_config
from treeindex.utils.reporting import write_text_report
from treeindex.utils.table import pretty_print_results, render_results_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local tree index demos from a JSON config file.")
    parser.add_argument(
        "--config",
        default=str(default_config_path("local_demo.json")),
        help="Path to the local demo JSON config.",
    )
    parser.add_argument("--output-dir", help="Directory to save the local demo text report.")
    args = parser.parse_args()
    config = load_json_config(args.config)
    report_sections: list[tuple[str, list[object]]] = []

    title = "=== Local B+ Tree demo ==="
    rows = local_bplustree_demo(config.get("local_bplustree"))
    print(f"\n{title}")
    pretty_print_results(rows)
    report_sections.append((title, rows))

    title = "=== Local R-Tree demo ==="
    rows = local_rtree_demo(config.get("local_rtree"))
    print(f"\n{title}")
    pretty_print_results(rows)
    report_sections.append((title, rows))

    title = "=== Local KD-Tree demo ==="
    rows = local_kdtree_demo(config.get("local_kdtree"))
    print(f"\n{title}")
    pretty_print_results(rows)
    report_sections.append((title, rows))

    title = "=== Local Quadtree demo ==="
    rows = local_quadtree_demo(config.get("local_quadtree"))
    print(f"\n{title}")
    pretty_print_results(rows)
    report_sections.append((title, rows))

    comparison_config = config.get("local_comparison", {})
    comparison_sections = local_comparison_demo(
        comparison_config.get("bplustree"),
        comparison_config.get("rtree"),
        comparison_config.get("kdtree"),
        comparison_config.get("quadtree"),
    )
    for title, rows in comparison_sections:
        print(f"\n{title}")
        pretty_print_results(rows)
        report_sections.append((title, rows))

    report_text = "\n\n".join(f"{section_title}\n{render_results_table(section_rows)}" for section_title, section_rows in report_sections)
    txt_path = write_text_report(
        text=report_text,
        filename_prefix="local-demo",
        output_dir=args.output_dir,
    )
    print(f"\nSaved results to {txt_path}")


if __name__ == "__main__":
    main()

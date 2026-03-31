from __future__ import annotations

import argparse

from treeindex.data.generators import generate_rectangles, sample_rect_point_queries, sample_rect_queries
from treeindex.indexes.distributed.pyspark_rtree import SparkSession, run_distributed_rtree_experiment
from treeindex.utils.config import default_config_path, load_json_config, merge_overrides
from treeindex.utils.reporting import write_experiment_report
from treeindex.utils.table import pretty_print_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Educational PySpark distributed R-Tree experiment.")
    parser.add_argument(
        "--config",
        default=str(default_config_path("distributed_rtree.json")),
        help="Path to the distributed R-Tree JSON config.",
    )
    parser.add_argument("--n-items", type=int)
    parser.add_argument("--n-partitions", type=int)
    parser.add_argument("--max-entries", type=int)
    parser.add_argument("--n-point-queries", type=int)
    parser.add_argument("--n-queries", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--space-width", type=float)
    parser.add_argument("--space-height", type=float)
    parser.add_argument("--attr1-min", type=int)
    parser.add_argument("--attr1-max", type=int)
    parser.add_argument("--attr2-min", type=int)
    parser.add_argument("--attr2-max", type=int)
    parser.add_argument("--max-rect-width", type=float)
    parser.add_argument("--max-rect-height", type=float)
    parser.add_argument("--max-query-width", type=float)
    parser.add_argument("--max-query-height", type=float)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    config = merge_overrides(
        load_json_config(args.config),
        {
            "n_items": args.n_items,
            "n_partitions": args.n_partitions,
            "max_entries": args.max_entries,
            "n_point_queries": args.n_point_queries,
            "n_queries": args.n_queries,
            "seed": args.seed,
            "space_width": args.space_width,
            "space_height": args.space_height,
            "attr1_min": args.attr1_min,
            "attr1_max": args.attr1_max,
            "attr2_min": args.attr2_min,
            "attr2_max": args.attr2_max,
            "max_rect_width": args.max_rect_width,
            "max_rect_height": args.max_rect_height,
            "max_query_width": args.max_query_width,
            "max_query_height": args.max_query_height,
        },
    )

    if SparkSession is None:
        raise RuntimeError("PySpark is not available. Please run with spark-submit or install pyspark.")

    spark = SparkSession.builder.appName("EducationalDistributedRTree").getOrCreate()
    try:
        items = generate_rectangles(
            config["n_items"],
            space_width=config["space_width"],
            space_height=config["space_height"],
            max_rect_width=config["max_rect_width"],
            max_rect_height=config["max_rect_height"],
            attr1_min=config["attr1_min"],
            attr1_max=config["attr1_max"],
            attr2_min=config["attr2_min"],
            attr2_max=config["attr2_max"],
            seed=config["seed"],
        )
        point_queries = sample_rect_point_queries(
            [rect for rect, _ in items],
            config["n_point_queries"],
            seed=config["seed"] + 1,
        )
        queries = sample_rect_queries(
            config["n_queries"],
            space_width=config["space_width"],
            space_height=config["space_height"],
            max_query_width=config["max_query_width"],
            max_query_height=config["max_query_height"],
            seed=config["seed"] + 2,
        )
        rows = run_distributed_rtree_experiment(
            spark,
            items,
            max_entries=config["max_entries"],
            n_partitions=config["n_partitions"],
            point_queries=point_queries,
            queries=queries,
        )
        title = "=== Distributed MapReduce-style / PySpark R-Tree experiment ==="
        print(f"\n{title}")
        pretty_print_results(rows)
        txt_path, json_path = write_experiment_report(
            title=title,
            rows=rows,
            filename_prefix="distributed-rtree",
            output_dir=args.output_dir,
        )
        print(f"\nSaved results to {txt_path}")
        print(f"Saved JSON to {json_path}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

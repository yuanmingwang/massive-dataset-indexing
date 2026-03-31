from __future__ import annotations

import argparse

from treeindex.data.generators import generate_points, sample_exact_point_queries, sample_point_rect_queries
from treeindex.indexes.distributed.pyspark_quadtree import SparkSession, run_distributed_quadtree_experiment
from treeindex.utils.config import default_config_path, load_json_config, merge_overrides
from treeindex.utils.reporting import write_experiment_report
from treeindex.utils.table import pretty_print_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Educational PySpark distributed Quadtree experiment.")
    parser.add_argument(
        "--config",
        default=str(default_config_path("distributed_quadtree.json")),
        help="Path to the distributed Quadtree JSON config.",
    )
    parser.add_argument("--n-items", type=int)
    parser.add_argument("--n-partitions", type=int)
    parser.add_argument("--n-point-queries", type=int)
    parser.add_argument("--n-queries", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--space-width", type=float)
    parser.add_argument("--space-height", type=float)
    parser.add_argument("--attr1-min", type=int)
    parser.add_argument("--attr1-max", type=int)
    parser.add_argument("--attr2-min", type=int)
    parser.add_argument("--attr2-max", type=int)
    parser.add_argument("--max-query-width", type=float)
    parser.add_argument("--max-query-height", type=float)
    parser.add_argument("--bucket-capacity", type=int)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    config = merge_overrides(
        load_json_config(args.config),
        {
            "n_items": args.n_items,
            "n_partitions": args.n_partitions,
            "n_point_queries": args.n_point_queries,
            "n_queries": args.n_queries,
            "seed": args.seed,
            "space_width": args.space_width,
            "space_height": args.space_height,
            "attr1_min": args.attr1_min,
            "attr1_max": args.attr1_max,
            "attr2_min": args.attr2_min,
            "attr2_max": args.attr2_max,
            "max_query_width": args.max_query_width,
            "max_query_height": args.max_query_height,
            "bucket_capacity": args.bucket_capacity,
            "max_depth": args.max_depth,
        },
    )

    if SparkSession is None:
        raise RuntimeError("PySpark is not available. Please run with spark-submit or install pyspark.")

    spark = SparkSession.builder.appName("EducationalDistributedQuadTree").getOrCreate()
    try:
        items = generate_points(
            config["n_items"],
            space_width=config["space_width"],
            space_height=config["space_height"],
            attr1_min=config["attr1_min"],
            attr1_max=config["attr1_max"],
            attr2_min=config["attr2_min"],
            attr2_max=config["attr2_max"],
            seed=config["seed"],
        )
        point_queries = sample_exact_point_queries(
            [point for point, _ in items],
            config["n_point_queries"],
            seed=config["seed"] + 1,
        )
        queries = sample_point_rect_queries(
            config["n_queries"],
            space_width=config["space_width"],
            space_height=config["space_height"],
            max_query_width=config["max_query_width"],
            max_query_height=config["max_query_height"],
            seed=config["seed"] + 2,
        )
        rows = run_distributed_quadtree_experiment(
            spark,
            items,
            bucket_capacity=config["bucket_capacity"],
            max_depth=config["max_depth"],
            n_partitions=config["n_partitions"],
            point_queries=point_queries,
            queries=queries,
        )
        title = "=== Distributed MapReduce-style / PySpark Quadtree experiment ==="
        print(f"\n{title}")
        pretty_print_results(rows)
        txt_path, json_path = write_experiment_report(
            title=title,
            rows=rows,
            filename_prefix="distributed-quadtree",
            output_dir=args.output_dir,
        )
        print(f"\nSaved results to {txt_path}")
        print(f"Saved JSON to {json_path}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

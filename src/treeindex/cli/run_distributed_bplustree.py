from __future__ import annotations

import argparse

from treeindex.data.generators import (
    generate_key_value_pairs,
    project_records_to_attribute,
    sample_attribute_point_queries,
    sample_attribute_range_queries,
    sample_range_queries,
)
from treeindex.indexes.distributed.pyspark_bplustree import SparkSession, run_distributed_experiment
from treeindex.utils.config import default_config_path, load_json_config, merge_overrides
from treeindex.utils.reporting import write_experiment_report
from treeindex.utils.table import pretty_print_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Educational PySpark MapReduce-style B+ Tree experiment.")
    parser.add_argument(
        "--config",
        default=str(default_config_path("distributed_bplustree.json")),
        help="Path to the distributed demo JSON config.",
    )
    parser.add_argument("--n-records", type=int)
    parser.add_argument("--n-partitions", type=int)
    parser.add_argument("--tree-order", type=int)
    parser.add_argument("--n-point-queries", type=int)
    parser.add_argument("--n-range-queries", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--key-min", type=int)
    parser.add_argument("--key-max", type=int)
    parser.add_argument("--attr1-min", type=int)
    parser.add_argument("--attr1-max", type=int)
    parser.add_argument("--attr2-min", type=int)
    parser.add_argument("--attr2-max", type=int)
    parser.add_argument("--range-low-min", type=int)
    parser.add_argument("--range-low-max", type=int)
    parser.add_argument("--range-max-width", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = merge_overrides(
        load_json_config(args.config),
        {
            "n_records": args.n_records,
            "n_partitions": args.n_partitions,
            "tree_order": args.tree_order,
            "n_point_queries": args.n_point_queries,
            "n_range_queries": args.n_range_queries,
            "seed": args.seed,
            "key_min": args.key_min,
            "key_max": args.key_max,
            "attr1_min": args.attr1_min,
            "attr1_max": args.attr1_max,
            "attr2_min": args.attr2_min,
            "attr2_max": args.attr2_max,
            "range_low_min": args.range_low_min,
            "range_low_max": args.range_low_max,
            "range_max_width": args.range_max_width,
        },
    )

    if SparkSession is None:
        raise RuntimeError("PySpark is not available. Please run with spark-submit or install pyspark.")

    spark = SparkSession.builder.appName("EducationalDistributedBPlusTree").getOrCreate()
    try:
        items = generate_key_value_pairs(
            config["n_records"],
            key_min=config["key_min"],
            key_max=config["key_max"],
            attr1_min=config["attr1_min"],
            attr1_max=config["attr1_max"],
            attr2_min=config["attr2_min"],
            attr2_max=config["attr2_max"],
            allow_duplicates=True,
            seed=config["seed"],
        )
        rows = []
        for attribute in ("key", "attr1", "attr2"):
            attribute_items = project_records_to_attribute(items, attribute)
            point_queries = sample_attribute_point_queries(
                items,
                attribute,
                config["n_point_queries"],
                seed=config["seed"] + 1,
            )
            if attribute == "key":
                range_queries = sample_range_queries(
                    config["n_range_queries"],
                    low_min=config["range_low_min"],
                    low_max=config["range_low_max"],
                    max_width=config["range_max_width"],
                    seed=config["seed"] + 2,
                )
            else:
                range_queries = sample_attribute_range_queries(
                    items,
                    attribute,
                    config["n_range_queries"],
                    max_width=config["range_max_width"],
                    seed=config["seed"] + 2,
                )
            rows.extend(
                run_distributed_experiment(
                    spark,
                    attribute_items,
                    tree_order=config["tree_order"],
                    n_partitions=config["n_partitions"],
                    point_queries=point_queries,
                    range_queries=range_queries,
                    attribute_name=attribute,
                )
            )
        title = "=== Distributed MapReduce-style / PySpark B+ Tree experiment ==="
        print(f"\n{title}")
        pretty_print_results(rows)
        txt_path, json_path = write_experiment_report(
            title=title,
            rows=rows,
            filename_prefix="distributed-bplustree",
            output_dir=args.output_dir,
        )
        print(f"\nSaved results to {txt_path}")
        print(f"Saved JSON to {json_path}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

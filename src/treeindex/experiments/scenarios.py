from __future__ import annotations

from treeindex.data.generators import (
    generate_key_value_pairs,
    generate_points,
    generate_rectangles,
    sample_exact_point_queries,
    sample_point_rect_queries,
    sample_point_queries,
    sample_range_queries,
    sample_rect_point_queries,
    sample_rect_queries,
)
from treeindex.experiments.benchmark import compare_indexes, run_single_experiment
from treeindex.indexes.local.bplustree import BPlusTree
from treeindex.indexes.local.kdtree import KDTree
from treeindex.indexes.local.quadtree import QuadTree
from treeindex.indexes.local.rtree import RTree


DEFAULT_LOCAL_BPLUSTREE_CONFIG = {
    "n_items": 200000,
    "key_min": 0,
    "key_max": 1_000_000,
    "allow_duplicates": True,
    "seed": 1,
    "tree_order": 16,
    "n_point_queries": 500,
    "n_range_queries": 500,
    "range_low_min": 0,
    "range_low_max": 900_000,
    "range_max_width": 50_000,
}

DEFAULT_LOCAL_RTREE_CONFIG = {
    "n_items": 15000,
    "space_width": 10_000.0,
    "space_height": 10_000.0,
    "max_rect_width": 120.0,
    "max_rect_height": 120.0,
    "seed": 10,
    "max_entries": 12,
    "n_point_queries": 300,
    "n_queries": 300,
    "max_query_width": 600.0,
    "max_query_height": 600.0,
}

DEFAULT_LOCAL_KDTREE_CONFIG = {
    "n_items": 20000,
    "space_width": 10_000.0,
    "space_height": 10_000.0,
    "seed": 40,
    "n_point_queries": 400,
    "n_queries": 400,
    "max_query_width": 500.0,
    "max_query_height": 500.0,
}

DEFAULT_LOCAL_QUADTREE_CONFIG = {
    "n_items": 20000,
    "space_width": 10_000.0,
    "space_height": 10_000.0,
    "seed": 50,
    "n_point_queries": 400,
    "n_queries": 400,
    "max_query_width": 500.0,
    "max_query_height": 500.0,
    "bucket_capacity": 8,
    "max_depth": 12,
}

DEFAULT_LOCAL_COMPARISON_BPT_CONFIG = {
    "n_items": 30000,
    "key_min": 0,
    "key_max": 1_000_000,
    "allow_duplicates": True,
    "seed": 21,
    "orders": [8, 16, 32],
    "n_queries": 800,
}

DEFAULT_LOCAL_COMPARISON_RTREE_CONFIG = {
    "n_items": 20000,
    "space_width": 10_000.0,
    "space_height": 10_000.0,
    "max_rect_width": 100.0,
    "max_rect_height": 100.0,
    "seed": 31,
    "capacities": [6, 12, 24],
    "n_queries": 500,
    "max_query_width": 500.0,
    "max_query_height": 500.0,
}


def _merge_config(defaults: dict, overrides: dict | None) -> dict:
    config = dict(defaults)
    if overrides:
        config.update(overrides)
    return config


def local_bplustree_demo(config: dict | None = None):
    cfg = _merge_config(DEFAULT_LOCAL_BPLUSTREE_CONFIG, config)
    items = generate_key_value_pairs(
        cfg["n_items"],
        key_min=cfg["key_min"],
        key_max=cfg["key_max"],
        allow_duplicates=cfg["allow_duplicates"],
        seed=cfg["seed"],
    )
    keys = [k for k, _ in items]
    point_queries = sample_point_queries(keys, num_queries=cfg["n_point_queries"], seed=cfg["seed"] + 1)
    range_queries = sample_range_queries(
        num_queries=cfg["n_range_queries"],
        low_min=cfg["range_low_min"],
        low_max=cfg["range_low_max"],
        max_width=cfg["range_max_width"],
        seed=cfg["seed"] + 2,
    )
    tree_order = cfg["tree_order"]
    return [
        run_single_experiment(
            index_factory=lambda: BPlusTree(order=tree_order),
            items=items,
            queries=point_queries,
            workload_name="bptree-point",
            extra={"order": tree_order, "query_type": "point"},
        ),
        run_single_experiment(
            index_factory=lambda: BPlusTree(order=tree_order),
            items=items,
            queries=range_queries,
            workload_name="bptree-range",
            extra={"order": tree_order, "query_type": "range"},
        ),
    ]


def local_rtree_demo(config: dict | None = None):
    cfg = _merge_config(DEFAULT_LOCAL_RTREE_CONFIG, config)
    items = generate_rectangles(
        cfg["n_items"],
        space_width=cfg["space_width"],
        space_height=cfg["space_height"],
        max_rect_width=cfg["max_rect_width"],
        max_rect_height=cfg["max_rect_height"],
        seed=cfg["seed"],
    )
    point_queries = sample_rect_point_queries([rect for rect, _ in items], num_queries=cfg["n_point_queries"], seed=cfg["seed"] + 1)
    queries = sample_rect_queries(
        num_queries=cfg["n_queries"],
        space_width=cfg["space_width"],
        space_height=cfg["space_height"],
        max_query_width=cfg["max_query_width"],
        max_query_height=cfg["max_query_height"],
        seed=cfg["seed"] + 2,
    )
    max_entries = cfg["max_entries"]
    return [
        run_single_experiment(
            index_factory=lambda: RTree(max_entries=max_entries),
            items=items,
            queries=point_queries,
            workload_name="rtree-point",
            extra={"max_entries": max_entries, "query_type": "point-location"},
        ),
        run_single_experiment(
            index_factory=lambda: RTree(max_entries=max_entries),
            items=items,
            queries=queries,
            workload_name="rtree-intersection",
            extra={"max_entries": max_entries, "query_type": "rect-intersection"},
        )
    ]


def local_kdtree_demo(config: dict | None = None):
    cfg = _merge_config(DEFAULT_LOCAL_KDTREE_CONFIG, config)
    items = generate_points(
        cfg["n_items"],
        space_width=cfg["space_width"],
        space_height=cfg["space_height"],
        seed=cfg["seed"],
    )
    points = [point for point, _ in items]
    point_queries = sample_exact_point_queries(points, num_queries=cfg["n_point_queries"], seed=cfg["seed"] + 1)
    queries = sample_point_rect_queries(
        num_queries=cfg["n_queries"],
        space_width=cfg["space_width"],
        space_height=cfg["space_height"],
        max_query_width=cfg["max_query_width"],
        max_query_height=cfg["max_query_height"],
        seed=cfg["seed"] + 2,
    )
    return [
        run_single_experiment(
            index_factory=KDTree,
            items=items,
            queries=point_queries,
            workload_name="kdtree-point",
            extra={"dims": 2, "query_type": "point"},
        ),
        run_single_experiment(
            index_factory=KDTree,
            items=items,
            queries=queries,
            workload_name="kdtree-range",
            extra={"dims": 2, "query_type": "rect-range"},
        )
    ]


def local_quadtree_demo(config: dict | None = None):
    cfg = _merge_config(DEFAULT_LOCAL_QUADTREE_CONFIG, config)
    items = generate_points(
        cfg["n_items"],
        space_width=cfg["space_width"],
        space_height=cfg["space_height"],
        seed=cfg["seed"],
    )
    points = [point for point, _ in items]
    point_queries = sample_exact_point_queries(points, num_queries=cfg["n_point_queries"], seed=cfg["seed"] + 1)
    queries = sample_point_rect_queries(
        num_queries=cfg["n_queries"],
        space_width=cfg["space_width"],
        space_height=cfg["space_height"],
        max_query_width=cfg["max_query_width"],
        max_query_height=cfg["max_query_height"],
        seed=cfg["seed"] + 2,
    )
    bucket_capacity = cfg["bucket_capacity"]
    max_depth = cfg["max_depth"]
    return [
        run_single_experiment(
            index_factory=lambda: QuadTree(bucket_capacity=bucket_capacity, max_depth=max_depth),
            items=items,
            queries=point_queries,
            workload_name="quadtree-point",
            extra={"bucket_capacity": bucket_capacity, "max_depth": max_depth, "query_type": "point"},
        ),
        run_single_experiment(
            index_factory=lambda: QuadTree(bucket_capacity=bucket_capacity, max_depth=max_depth),
            items=items,
            queries=queries,
            workload_name="quadtree-range",
            extra={"bucket_capacity": bucket_capacity, "max_depth": max_depth, "query_type": "rect-range"},
        )
    ]


def local_comparison_demo(
    bpt_config: dict | None = None,
    rtree_config: dict | None = None,
):
    bpt_cfg = _merge_config(DEFAULT_LOCAL_COMPARISON_BPT_CONFIG, bpt_config)
    items = generate_key_value_pairs(
        bpt_cfg["n_items"],
        key_min=bpt_cfg["key_min"],
        key_max=bpt_cfg["key_max"],
        allow_duplicates=bpt_cfg["allow_duplicates"],
        seed=bpt_cfg["seed"],
    )
    keys = [k for k, _ in items]
    point_queries = sample_point_queries(keys, num_queries=bpt_cfg["n_queries"], seed=bpt_cfg["seed"] + 1)
    bpt_results = compare_indexes(
        index_factories=[lambda order=order: BPlusTree(order=order) for order in bpt_cfg["orders"]],
        items=items,
        queries=point_queries,
        workload_name="compare-bpt-orders",
        extra_builder=lambda idx: {"order": idx.order, "height": idx.height()},
    )

    rtree_cfg = _merge_config(DEFAULT_LOCAL_COMPARISON_RTREE_CONFIG, rtree_config)
    rect_items = generate_rectangles(
        rtree_cfg["n_items"],
        space_width=rtree_cfg["space_width"],
        space_height=rtree_cfg["space_height"],
        max_rect_width=rtree_cfg["max_rect_width"],
        max_rect_height=rtree_cfg["max_rect_height"],
        seed=rtree_cfg["seed"],
    )
    rect_queries = sample_rect_queries(
        num_queries=rtree_cfg["n_queries"],
        space_width=rtree_cfg["space_width"],
        space_height=rtree_cfg["space_height"],
        max_query_width=rtree_cfg["max_query_width"],
        max_query_height=rtree_cfg["max_query_height"],
        seed=rtree_cfg["seed"] + 1,
    )
    rtree_results = compare_indexes(
        index_factories=[lambda max_entries=max_entries: RTree(max_entries=max_entries) for max_entries in rtree_cfg["capacities"]],
        items=rect_items,
        queries=rect_queries,
        workload_name="compare-rtree-capacity",
        extra_builder=lambda idx: {"max_entries": idx.max_entries, "height": idx.height()},
    )
    return bpt_results, rtree_results

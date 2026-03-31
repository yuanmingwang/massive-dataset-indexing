from __future__ import annotations

from treeindex.data.generators import (
    generate_key_value_pairs,
    generate_points,
    generate_rectangles,
    project_records_to_attribute,
    sample_attribute_point_queries,
    sample_attribute_range_queries,
    sample_exact_point_queries,
    sample_point_rect_queries,
    sample_point_queries,
    sample_range_queries,
    sample_rect_point_queries,
    sample_rect_queries,
)
from treeindex.experiments.benchmark import compare_indexes_with_shared_build, run_experiments_with_shared_build
from treeindex.indexes.local.bplustree import BPlusTree
from treeindex.indexes.local.kdtree import KDTree
from treeindex.indexes.local.quadtree import QuadTree
from treeindex.indexes.local.rtree import RTree


DEFAULT_LOCAL_BPLUSTREE_CONFIG = {
    "n_items": 200000,
    "key_min": 0,
    "key_max": 1_000_000,
    "attr1_min": 0,
    "attr1_max": 1_000_000,
    "attr2_min": 0,
    "attr2_max": 1_000_000,
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
    "attr1_min": 0,
    "attr1_max": 1_000_000,
    "attr2_min": 0,
    "attr2_max": 1_000_000,
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
    "attr1_min": 0,
    "attr1_max": 1_000_000,
    "attr2_min": 0,
    "attr2_max": 1_000_000,
    "seed": 40,
    "n_point_queries": 400,
    "n_queries": 400,
    "max_query_width": 500.0,
    "max_query_height": 500.0,
    "leaf_capacity": 1,
}

DEFAULT_LOCAL_QUADTREE_CONFIG = {
    "n_items": 20000,
    "space_width": 10_000.0,
    "space_height": 10_000.0,
    "attr1_min": 0,
    "attr1_max": 1_000_000,
    "attr2_min": 0,
    "attr2_max": 1_000_000,
    "seed": 50,
    "n_point_queries": 400,
    "n_queries": 400,
    "max_query_width": 500.0,
    "max_query_height": 500.0,
    "bucket_capacity": 8,
    "max_depth": 12,
}

DEFAULT_LOCAL_COMPARISON_BPT_CONFIG = {
    "n_items": 2_000_000,
    "key_min": 0,
    "key_max": 10_000_000,
    "attr1_min": 0,
    "attr1_max": 10_000_000,
    "attr2_min": 0,
    "attr2_max": 10_000_000,
    "allow_duplicates": True,
    "seed": 1,
    "orders": [16, 32, 64],
    "n_point_queries": 500,
    "n_range_queries": 500,
    "range_low_min": 0,
    "range_low_max": 9_000_000,
    "range_max_width": 500_000,
}

DEFAULT_LOCAL_COMPARISON_RTREE_CONFIG = {
    "n_items": 2_000_000,
    "space_width": 100_000.0,
    "space_height": 100_000.0,
    "max_rect_width": 1_200.0,
    "max_rect_height": 1_200.0,
    "attr1_min": 0,
    "attr1_max": 1_000_000,
    "attr2_min": 0,
    "attr2_max": 1_000_000,
    "seed": 1,
    "capacities": [16, 32, 64],
    "n_point_queries": 500,
    "n_queries": 500,
    "max_query_width": 6_000.0,
    "max_query_height": 6_000.0,
}

DEFAULT_LOCAL_COMPARISON_KDTREE_CONFIG = {
    "n_items": 2_000_000,
    "space_width": 100_000.0,
    "space_height": 100_000.0,
    "attr1_min": 0,
    "attr1_max": 1_000_000,
    "attr2_min": 0,
    "attr2_max": 1_000_000,
    "seed": 1,
    "leaf_capacities": [16, 32, 64],
    "n_point_queries": 500,
    "n_queries": 500,
    "max_query_width": 5_000.0,
    "max_query_height": 5_000.0,
}

DEFAULT_LOCAL_COMPARISON_QUADTREE_CONFIG = {
    "n_items": 2_000_000,
    "space_width": 100_000.0,
    "space_height": 100_000.0,
    "attr1_min": 0,
    "attr1_max": 1_000_000,
    "attr2_min": 0,
    "attr2_max": 1_000_000,
    "seed": 1,
    "bucket_capacities": [16, 32, 64],
    "n_point_queries": 500,
    "n_queries": 500,
    "max_query_width": 5_000.0,
    "max_query_height": 5_000.0,
    "max_depth": 20,
}


def _merge_config(defaults: dict, overrides: dict | None) -> dict:
    config = dict(defaults)
    if overrides:
        config.update(overrides)
    return config


def _bplustree_attribute_queries(cfg: dict, items, attribute: str):
    point_queries = sample_attribute_point_queries(
        items,
        attribute,
        num_queries=cfg["n_point_queries"],
        seed=cfg["seed"] + 1,
    )
    if attribute == "key":
        range_queries = sample_range_queries(
            num_queries=cfg["n_range_queries"],
            low_min=cfg["range_low_min"],
            low_max=cfg["range_low_max"],
            max_width=cfg["range_max_width"],
            seed=cfg["seed"] + 2,
        )
    else:
        range_queries = sample_attribute_range_queries(
            items,
            attribute,
            num_queries=cfg["n_range_queries"],
            max_width=cfg["range_max_width"],
            seed=cfg["seed"] + 2,
        )
    return point_queries, range_queries


def local_bplustree_demo(config: dict | None = None):
    cfg = _merge_config(DEFAULT_LOCAL_BPLUSTREE_CONFIG, config)
    items = generate_key_value_pairs(
        cfg["n_items"],
        key_min=cfg["key_min"],
        key_max=cfg["key_max"],
        attr1_min=cfg["attr1_min"],
        attr1_max=cfg["attr1_max"],
        attr2_min=cfg["attr2_min"],
        attr2_max=cfg["attr2_max"],
        allow_duplicates=cfg["allow_duplicates"],
        seed=cfg["seed"],
    )
    tree_order = cfg["tree_order"]
    rows = []
    for attribute in ("key", "attr1", "attr2"):
        attribute_items = project_records_to_attribute(items, attribute)
        point_queries, range_queries = _bplustree_attribute_queries(cfg, items, attribute)
        rows.extend(
            run_experiments_with_shared_build(
                index_factory=lambda: BPlusTree(order=tree_order),
                items=attribute_items,
                query_batches=[
                    {
                        "queries": point_queries,
                        "workload_name": f"bptree-{attribute}-point",
                        "extra": {"order": tree_order, "indexed_attribute": attribute, "query_type": "point"},
                    },
                    {
                        "queries": range_queries,
                        "workload_name": f"bptree-{attribute}-range",
                        "extra": {"order": tree_order, "indexed_attribute": attribute, "query_type": "range"},
                    },
                ],
            )
        )
    return rows


def local_rtree_demo(config: dict | None = None):
    cfg = _merge_config(DEFAULT_LOCAL_RTREE_CONFIG, config)
    items = generate_rectangles(
        cfg["n_items"],
        space_width=cfg["space_width"],
        space_height=cfg["space_height"],
        max_rect_width=cfg["max_rect_width"],
        max_rect_height=cfg["max_rect_height"],
        attr1_min=cfg["attr1_min"],
        attr1_max=cfg["attr1_max"],
        attr2_min=cfg["attr2_min"],
        attr2_max=cfg["attr2_max"],
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
    return run_experiments_with_shared_build(
        index_factory=lambda: RTree(max_entries=max_entries),
        items=items,
        query_batches=[
            {
                "queries": point_queries,
                "workload_name": "rtree-point",
                "extra": {"max_entries": max_entries, "query_type": "point-location"},
            },
            {
                "queries": queries,
                "workload_name": "rtree-intersection",
                "extra": {"max_entries": max_entries, "query_type": "rect-intersection"},
            },
        ],
    )


def local_kdtree_demo(config: dict | None = None):
    cfg = _merge_config(DEFAULT_LOCAL_KDTREE_CONFIG, config)
    items = generate_points(
        cfg["n_items"],
        space_width=cfg["space_width"],
        space_height=cfg["space_height"],
        attr1_min=cfg["attr1_min"],
        attr1_max=cfg["attr1_max"],
        attr2_min=cfg["attr2_min"],
        attr2_max=cfg["attr2_max"],
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
    leaf_capacity = cfg["leaf_capacity"]
    return run_experiments_with_shared_build(
        index_factory=lambda: KDTree(leaf_capacity=leaf_capacity),
        items=items,
        query_batches=[
            {
                "queries": point_queries,
                "workload_name": "kdtree-point",
                "extra": {"dims": 2, "leaf_capacity": leaf_capacity, "query_type": "point"},
            },
            {
                "queries": queries,
                "workload_name": "kdtree-range",
                "extra": {"dims": 2, "leaf_capacity": leaf_capacity, "query_type": "rect-range"},
            },
        ],
    )


def local_quadtree_demo(config: dict | None = None):
    cfg = _merge_config(DEFAULT_LOCAL_QUADTREE_CONFIG, config)
    items = generate_points(
        cfg["n_items"],
        space_width=cfg["space_width"],
        space_height=cfg["space_height"],
        attr1_min=cfg["attr1_min"],
        attr1_max=cfg["attr1_max"],
        attr2_min=cfg["attr2_min"],
        attr2_max=cfg["attr2_max"],
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
    return run_experiments_with_shared_build(
        index_factory=lambda: QuadTree(bucket_capacity=bucket_capacity, max_depth=max_depth),
        items=items,
        query_batches=[
            {
                "queries": point_queries,
                "workload_name": "quadtree-point",
                "extra": {"bucket_capacity": bucket_capacity, "max_depth": max_depth, "query_type": "point"},
            },
            {
                "queries": queries,
                "workload_name": "quadtree-range",
                "extra": {"bucket_capacity": bucket_capacity, "max_depth": max_depth, "query_type": "rect-range"},
            },
        ],
    )


def local_comparison_demo(
    bpt_config: dict | None = None,
    rtree_config: dict | None = None,
    kdtree_config: dict | None = None,
    quadtree_config: dict | None = None,
):
    bpt_cfg = _merge_config(DEFAULT_LOCAL_COMPARISON_BPT_CONFIG, bpt_config)
    bpt_items = generate_key_value_pairs(
        bpt_cfg["n_items"],
        key_min=bpt_cfg["key_min"],
        key_max=bpt_cfg["key_max"],
        attr1_min=bpt_cfg["attr1_min"],
        attr1_max=bpt_cfg["attr1_max"],
        attr2_min=bpt_cfg["attr2_min"],
        attr2_max=bpt_cfg["attr2_max"],
        allow_duplicates=bpt_cfg["allow_duplicates"],
        seed=bpt_cfg["seed"],
    )
    bpt_results = []
    for attribute in ("key", "attr1", "attr2"):
        attribute_items = project_records_to_attribute(bpt_items, attribute)
        bpt_point_queries, bpt_range_queries = _bplustree_attribute_queries(bpt_cfg, bpt_items, attribute)
        bpt_results.extend(
            compare_indexes_with_shared_build(
                index_factories=[lambda order=order: BPlusTree(order=order) for order in bpt_cfg["orders"]],
                items=attribute_items,
                query_batches=[
                    {
                        "queries": bpt_point_queries,
                        "workload_name": f"compare-bpt-{attribute}-point",
                        "query_type": "point",
                    },
                    {
                        "queries": bpt_range_queries,
                        "workload_name": f"compare-bpt-{attribute}-range",
                        "query_type": "range",
                    },
                ],
                extra_builder=lambda idx, batch, attribute=attribute: {
                    "order": idx.order,
                    "height": idx.height(),
                    "indexed_attribute": attribute,
                    "query_type": batch["query_type"],
                },
            )
        )

    rtree_cfg = _merge_config(DEFAULT_LOCAL_COMPARISON_RTREE_CONFIG, rtree_config)
    rect_items = generate_rectangles(
        rtree_cfg["n_items"],
        space_width=rtree_cfg["space_width"],
        space_height=rtree_cfg["space_height"],
        max_rect_width=rtree_cfg["max_rect_width"],
        max_rect_height=rtree_cfg["max_rect_height"],
        attr1_min=rtree_cfg["attr1_min"],
        attr1_max=rtree_cfg["attr1_max"],
        attr2_min=rtree_cfg["attr2_min"],
        attr2_max=rtree_cfg["attr2_max"],
        seed=rtree_cfg["seed"],
    )
    rect_point_queries = sample_rect_point_queries(
        [rect for rect, _ in rect_items],
        num_queries=rtree_cfg["n_point_queries"],
        seed=rtree_cfg["seed"] + 1,
    )
    rect_queries = sample_rect_queries(
        num_queries=rtree_cfg["n_queries"],
        space_width=rtree_cfg["space_width"],
        space_height=rtree_cfg["space_height"],
        max_query_width=rtree_cfg["max_query_width"],
        max_query_height=rtree_cfg["max_query_height"],
        seed=rtree_cfg["seed"] + 2,
    )
    rtree_factories = [lambda max_entries=max_entries: RTree(max_entries=max_entries) for max_entries in rtree_cfg["capacities"]]
    rtree_results = compare_indexes_with_shared_build(
        index_factories=rtree_factories,
        items=rect_items,
        query_batches=[
            {
                "queries": rect_point_queries,
                "workload_name": "compare-rtree-point",
                "query_type": "point-location",
            },
            {
                "queries": rect_queries,
                "workload_name": "compare-rtree-capacity",
                "query_type": "rect-intersection",
            },
        ],
        extra_builder=lambda idx, batch: {
            "max_entries": idx.max_entries,
            "height": idx.height(),
            "query_type": batch["query_type"],
        },
    )

    kdtree_cfg = _merge_config(DEFAULT_LOCAL_COMPARISON_KDTREE_CONFIG, kdtree_config)
    kd_items = generate_points(
        kdtree_cfg["n_items"],
        space_width=kdtree_cfg["space_width"],
        space_height=kdtree_cfg["space_height"],
        attr1_min=kdtree_cfg["attr1_min"],
        attr1_max=kdtree_cfg["attr1_max"],
        attr2_min=kdtree_cfg["attr2_min"],
        attr2_max=kdtree_cfg["attr2_max"],
        seed=kdtree_cfg["seed"],
    )
    kd_points = [point for point, _ in kd_items]
    kd_point_queries = sample_exact_point_queries(
        kd_points,
        num_queries=kdtree_cfg["n_point_queries"],
        seed=kdtree_cfg["seed"] + 1,
    )
    kd_range_queries = sample_point_rect_queries(
        num_queries=kdtree_cfg["n_queries"],
        space_width=kdtree_cfg["space_width"],
        space_height=kdtree_cfg["space_height"],
        max_query_width=kdtree_cfg["max_query_width"],
        max_query_height=kdtree_cfg["max_query_height"],
        seed=kdtree_cfg["seed"] + 2,
    )
    kd_factories = [
        lambda leaf_capacity=leaf_capacity: KDTree(leaf_capacity=leaf_capacity)
        for leaf_capacity in kdtree_cfg["leaf_capacities"]
    ]
    kdtree_results = compare_indexes_with_shared_build(
        index_factories=kd_factories,
        items=kd_items,
        query_batches=[
            {
                "queries": kd_point_queries,
                "workload_name": "compare-kdtree-point",
                "query_type": "point",
            },
            {
                "queries": kd_range_queries,
                "workload_name": "compare-kdtree-range",
                "query_type": "rect-range",
            },
        ],
        extra_builder=lambda idx, batch: {
            "leaf_capacity": idx.leaf_capacity,
            "height": idx.height(),
            "query_type": batch["query_type"],
        },
    )

    quadtree_cfg = _merge_config(DEFAULT_LOCAL_COMPARISON_QUADTREE_CONFIG, quadtree_config)
    quad_items = generate_points(
        quadtree_cfg["n_items"],
        space_width=quadtree_cfg["space_width"],
        space_height=quadtree_cfg["space_height"],
        attr1_min=quadtree_cfg["attr1_min"],
        attr1_max=quadtree_cfg["attr1_max"],
        attr2_min=quadtree_cfg["attr2_min"],
        attr2_max=quadtree_cfg["attr2_max"],
        seed=quadtree_cfg["seed"],
    )
    quad_points = [point for point, _ in quad_items]
    quad_point_queries = sample_exact_point_queries(
        quad_points,
        num_queries=quadtree_cfg["n_point_queries"],
        seed=quadtree_cfg["seed"] + 1,
    )
    quad_range_queries = sample_point_rect_queries(
        num_queries=quadtree_cfg["n_queries"],
        space_width=quadtree_cfg["space_width"],
        space_height=quadtree_cfg["space_height"],
        max_query_width=quadtree_cfg["max_query_width"],
        max_query_height=quadtree_cfg["max_query_height"],
        seed=quadtree_cfg["seed"] + 2,
    )
    quadtree_results = compare_indexes_with_shared_build(
        index_factories=[
            lambda bucket_capacity=bucket_capacity: QuadTree(
                bucket_capacity=bucket_capacity,
                max_depth=quadtree_cfg["max_depth"],
            )
            for bucket_capacity in quadtree_cfg["bucket_capacities"]
        ],
        items=quad_items,
        query_batches=[
            {
                "queries": quad_point_queries,
                "workload_name": "compare-quadtree-point",
                "query_type": "point",
            },
            {
                "queries": quad_range_queries,
                "workload_name": "compare-quadtree-range",
                "query_type": "rect-range",
            },
        ],
        extra_builder=lambda idx, batch: {
            "bucket_capacity": idx.bucket_capacity,
            "max_depth": idx.max_depth,
            "height": idx.height(),
            "query_type": batch["query_type"],
        },
    )
    return [
        ("=== Local comparison: B+ Tree orders ===", bpt_results),
        ("=== Local comparison: R-Tree capacities ===", rtree_results),
        ("=== Local comparison: KD-Tree leaf capacities ===", kdtree_results),
        ("=== Local comparison: Quadtree bucket capacities ===", quadtree_results),
    ]

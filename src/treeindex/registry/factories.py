from __future__ import annotations

from treeindex.indexes.local.bplustree import BPlusTree
from treeindex.indexes.local.kdtree import KDTree
from treeindex.indexes.local.quadtree import QuadTree
from treeindex.indexes.local.rtree import RTree


LOCAL_INDEX_FACTORIES = {
    "bplustree": lambda **kwargs: BPlusTree(order=kwargs.get("order", 16)),
    "kdtree": lambda **kwargs: KDTree(),
    "quadtree": lambda **kwargs: QuadTree(
        bucket_capacity=kwargs.get("bucket_capacity", 8),
        max_depth=kwargs.get("max_depth", 12),
    ),
    "rtree": lambda **kwargs: RTree(max_entries=kwargs.get("max_entries", 12)),
}

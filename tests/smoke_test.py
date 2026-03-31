from treeindex.data.generators import generate_key_value_pairs, generate_points, generate_rectangles
from treeindex.geometry.rect import Rect
from treeindex.indexes.local.bplustree import BPlusTree
from treeindex.indexes.local.kdtree import KDTree
from treeindex.indexes.local.quadtree import QuadTree
from treeindex.indexes.local.rtree import RTree


def main() -> None:
    items = generate_key_value_pairs(100, seed=1)
    bpt = BPlusTree(order=8)
    bpt.build(items)
    result = bpt.search(items[0][0])
    assert len(result) >= 1
    assert hasattr(result[0], "attr1")
    assert hasattr(result[0], "attr2")

    rects = generate_rectangles(25, seed=1)
    rtree = RTree(max_entries=4)
    rtree.build(rects)
    q = Rect(0, 0, 5000, 5000)
    rtree_results = rtree.query(q)
    assert isinstance(rtree_results, list)
    if rtree_results:
        assert hasattr(rtree_results[0], "attr1")
        assert hasattr(rtree_results[0], "attr2")

    points = generate_points(25, seed=1)
    kdtree = KDTree(leaf_capacity=4)
    kdtree.build(points)
    point_query = Rect(0, 0, 5000, 5000)
    kd_results = kdtree.query(point_query)
    assert isinstance(kd_results, list)
    if kd_results:
        assert hasattr(kd_results[0], "attr1")
        assert hasattr(kd_results[0], "attr2")

    quadtree = QuadTree(bucket_capacity=4, max_depth=8)
    quadtree.build(points)
    quad_results = quadtree.query(point_query)
    assert isinstance(quad_results, list)
    if quad_results:
        assert hasattr(quad_results[0], "attr1")
        assert hasattr(quad_results[0], "attr2")
    print("Smoke tests passed.")


if __name__ == "__main__":
    main()

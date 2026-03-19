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
    assert len(bpt.search(items[0][0])) >= 1

    rects = generate_rectangles(25, seed=1)
    rtree = RTree(max_entries=4)
    rtree.build(rects)
    q = Rect(0, 0, 5000, 5000)
    assert isinstance(rtree.query(q), list)

    points = generate_points(25, seed=1)
    kdtree = KDTree(leaf_capacity=4)
    kdtree.build(points)
    point_query = Rect(0, 0, 5000, 5000)
    assert isinstance(kdtree.query(point_query), list)

    quadtree = QuadTree(bucket_capacity=4, max_depth=8)
    quadtree.build(points)
    assert isinstance(quadtree.query(point_query), list)
    print("Smoke tests passed.")


if __name__ == "__main__":
    main()

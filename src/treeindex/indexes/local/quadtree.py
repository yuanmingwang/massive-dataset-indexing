from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Tuple

from treeindex.core.interfaces import BaseIndex
from treeindex.geometry.point import Point
from treeindex.geometry.rect import Rect


@dataclass
class QuadNode:
    bounds: Rect
    depth: int
    items: List[Tuple[Point, Any]] = field(default_factory=list)
    children: Optional[List["QuadNode"]] = None

    @property
    def is_leaf(self) -> bool:
        return self.children is None


class QuadTree(BaseIndex[Point, Any, Rect | Point]):
    def __init__(self, bucket_capacity: int = 8, max_depth: int = 12):
        if bucket_capacity < 1:
            raise ValueError("QuadTree bucket_capacity must be at least 1.")
        if max_depth < 0:
            raise ValueError("QuadTree max_depth must be non-negative.")
        self.bucket_capacity = bucket_capacity
        self.max_depth = max_depth
        self.root: Optional[QuadNode] = None
        self._items: List[Tuple[Point, Any]] = []

    def __len__(self) -> int:
        return len(self._items)

    def build(self, items: Iterable[Tuple[Point, Any]]) -> None:
        self._items = list(items)
        if not self._items:
            self.root = None
            return
        self.root = QuadNode(bounds=_bounds_for_points(point for point, _ in self._items), depth=0)
        for point, value in self._items:
            self._insert_into_node(self.root, point, value)

    def insert(self, key: Point, value: Any) -> None:
        self._items.append((key, value))
        if self.root is None:
            self.root = QuadNode(bounds=Rect(key.x, key.y, key.x, key.y), depth=0)
        else:
            self.root.bounds = self.root.bounds.union(Rect(key.x, key.y, key.x, key.y))
            # Rebuild if the root bounds expanded, to keep child boundaries coherent.
            self.build(self._items)
            return
        self._insert_into_node(self.root, key, value)

    def query(self, q: Rect | Point) -> List[Any]:
        if isinstance(q, Point):
            results: List[Any] = []
            self._point_query_node(self.root, q, results)
            return results
        results: List[Any] = []
        self._query_node(self.root, q, results)
        return results

    def height(self) -> int:
        return self._height(self.root)

    def dump_structure(self) -> str:
        lines: List[str] = []

        def visit(node: Optional[QuadNode]) -> None:
            if node is None:
                return
            lines.append(
                f"depth={node.depth} leaf={node.is_leaf} items={len(node.items)} "
                f"bounds=({node.bounds.xmin:.3f}, {node.bounds.ymin:.3f}, {node.bounds.xmax:.3f}, {node.bounds.ymax:.3f})"
            )
            if node.children:
                for child in node.children:
                    visit(child)

        visit(self.root)
        return "\n".join(lines)

    def _insert_into_node(self, node: QuadNode, point: Point, value: Any) -> None:
        if node.is_leaf:
            if (
                len(node.items) < self.bucket_capacity
                or node.depth >= self.max_depth
                or _is_degenerate(node.bounds)
            ):
                node.items.append((point, value))
                return
            self._subdivide(node)

        child = node.children[_child_index_for_point(point, node.bounds)]
        self._insert_into_node(child, point, value)

    def _subdivide(self, node: QuadNode) -> None:
        child_bounds = _subdivide_bounds(node.bounds)
        node.children = [QuadNode(bounds=bounds, depth=node.depth + 1) for bounds in child_bounds]
        existing_items = list(node.items)
        node.items.clear()
        for point, value in existing_items:
            child = node.children[_child_index_for_point(point, node.bounds)]
            self._insert_into_node(child, point, value)

    def _query_node(self, node: Optional[QuadNode], query_rect: Rect, out: List[Any]) -> None:
        if node is None or not node.bounds.intersects(query_rect):
            return

        for point, value in node.items:
            if _point_in_rect(point, query_rect):
                out.append(value)

        if node.children:
            for child in node.children:
                self._query_node(child, query_rect, out)

    def _point_query_node(self, node: Optional[QuadNode], point: Point, out: List[Any]) -> None:
        if node is None or not node.bounds.contains_point(point):
            return

        for candidate, value in node.items:
            if candidate == point:
                out.append(value)

        if node.children:
            child = node.children[_child_index_for_point(point, node.bounds)]
            self._point_query_node(child, point, out)

    def _height(self, node: Optional[QuadNode]) -> int:
        if node is None:
            return 0
        if node.children is None:
            return 1
        return 1 + max(self._height(child) for child in node.children)


def _bounds_for_points(points: Iterable[Point]) -> Rect:
    points = list(points)
    if not points:
        raise ValueError("Cannot compute QuadTree bounds for an empty point set.")
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    return Rect(min(xs), min(ys), max(xs), max(ys))


def _point_in_rect(point: Point, rect: Rect) -> bool:
    return rect.contains_point(point)


def _is_degenerate(bounds: Rect) -> bool:
    return bounds.xmin == bounds.xmax and bounds.ymin == bounds.ymax


def _child_index_for_point(point: Point, bounds: Rect) -> int:
    mid_x = (bounds.xmin + bounds.xmax) / 2.0
    mid_y = (bounds.ymin + bounds.ymax) / 2.0
    east = point.x > mid_x
    north = point.y > mid_y
    if not east and not north:
        return 0  # SW
    if east and not north:
        return 1  # SE
    if not east and north:
        return 2  # NW
    return 3  # NE


def _subdivide_bounds(bounds: Rect) -> List[Rect]:
    mid_x = (bounds.xmin + bounds.xmax) / 2.0
    mid_y = (bounds.ymin + bounds.ymax) / 2.0
    return [
        Rect(bounds.xmin, bounds.ymin, mid_x, mid_y),
        Rect(mid_x, bounds.ymin, bounds.xmax, mid_y),
        Rect(bounds.xmin, mid_y, mid_x, bounds.ymax),
        Rect(mid_x, mid_y, bounds.xmax, bounds.ymax),
    ]

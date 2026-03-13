from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple

from treeindex.core.interfaces import BaseIndex
from treeindex.geometry.point import Point
from treeindex.geometry.rect import Rect


@dataclass
class KDNode:
    point: Point
    value: Any
    axis: int
    left: Optional["KDNode"] = None
    right: Optional["KDNode"] = None


class KDTree(BaseIndex[Point, Any, Rect | Point]):
    def __init__(self) -> None:
        self.root: Optional[KDNode] = None
        self._items: List[Tuple[Point, Any]] = []

    def __len__(self) -> int:
        return len(self._items)

    def build(self, items: Iterable[Tuple[Point, Any]]) -> None:
        self._items = list(items)
        self.root = self._build_recursive(self._items, depth=0)

    def insert(self, key: Point, value: Any) -> None:
        # Rebuild-on-insert keeps the implementation compact and deterministic.
        self._items.append((key, value))
        self.root = self._build_recursive(self._items, depth=0)

    def query(self, q: Rect | Point) -> List[Any]:
        if isinstance(q, Point):
            results: List[Any] = []
            self._point_query_recursive(self.root, q, results)
            return results
        results: List[Any] = []
        self._query_recursive(self.root, q, results)
        return results

    def height(self) -> int:
        return self._height(self.root)

    def dump_structure(self) -> str:
        lines: List[str] = []

        def visit(node: Optional[KDNode], depth: int) -> None:
            if node is None:
                return
            lines.append(
                f"depth={depth} axis={node.axis} point=({node.point.x:.3f}, {node.point.y:.3f})"
            )
            visit(node.left, depth + 1)
            visit(node.right, depth + 1)

        visit(self.root, 0)
        return "\n".join(lines)

    def _build_recursive(self, items: List[Tuple[Point, Any]], depth: int) -> Optional[KDNode]:
        if not items:
            return None
        axis = depth % 2
        sorted_items = sorted(items, key=lambda item: item[0].coord(axis))
        mid = len(sorted_items) // 2
        point, value = sorted_items[mid]
        return KDNode(
            point=point,
            value=value,
            axis=axis,
            left=self._build_recursive(sorted_items[:mid], depth + 1),
            right=self._build_recursive(sorted_items[mid + 1 :], depth + 1),
        )

    def _query_recursive(self, node: Optional[KDNode], query_rect: Rect, out: List[Any]) -> None:
        if node is None:
            return

        if _point_in_rect(node.point, query_rect):
            out.append(node.value)

        split_value = node.point.coord(node.axis)
        lower = query_rect.xmin if node.axis == 0 else query_rect.ymin
        upper = query_rect.xmax if node.axis == 0 else query_rect.ymax

        if lower <= split_value:
            self._query_recursive(node.left, query_rect, out)
        if upper >= split_value:
            self._query_recursive(node.right, query_rect, out)

    def _point_query_recursive(self, node: Optional[KDNode], point: Point, out: List[Any]) -> None:
        if node is None:
            return

        if node.point == point:
            out.append(node.value)

        split_value = node.point.coord(node.axis)
        target_value = point.coord(node.axis)
        if target_value <= split_value:
            self._point_query_recursive(node.left, point, out)
        if target_value >= split_value:
            self._point_query_recursive(node.right, point, out)

    def _height(self, node: Optional[KDNode]) -> int:
        if node is None:
            return 0
        return 1 + max(self._height(node.left), self._height(node.right))


def _point_in_rect(point: Point, rect: Rect) -> bool:
    return rect.contains_point(point)

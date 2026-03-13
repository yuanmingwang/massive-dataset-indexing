from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Tuple

from treeindex.core.interfaces import BaseIndex
from treeindex.geometry.point import Point
from treeindex.geometry.rect import Rect


@dataclass
class RTreeEntry:
    rect: Rect
    child: Optional["RTreeNode"] = None
    value: Any = None


@dataclass
class RTreeNode:
    is_leaf: bool
    entries: List[RTreeEntry] = field(default_factory=list)

    def mbr(self) -> Rect:
        if not self.entries:
            raise ValueError("Cannot compute MBR of an empty R-Tree node.")
        result = self.entries[0].rect
        for e in self.entries[1:]:
            result = result.union(e.rect)
        return result


class RTree(BaseIndex[Rect, Any, Rect | Point]):
    def __init__(self, max_entries: int = 8):
        if max_entries < 3:
            raise ValueError("R-Tree max_entries must be at least 3.")
        self.max_entries = max_entries
        self.min_entries = max(2, max_entries // 2)
        self.root = RTreeNode(is_leaf=True)
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def build(self, items: Iterable[Tuple[Rect, Any]]) -> None:
        for rect, value in items:
            self.insert(rect, value)

    def insert(self, key: Rect, value: Any) -> None:
        promoted = self._insert_recursive(self.root, RTreeEntry(rect=key, value=value))
        if promoted is not None:
            left_node, right_node = promoted
            new_root = RTreeNode(is_leaf=False)
            new_root.entries = [
                RTreeEntry(rect=left_node.mbr(), child=left_node),
                RTreeEntry(rect=right_node.mbr(), child=right_node),
            ]
            self.root = new_root
        self._size += 1

    def query(self, q: Rect | Point) -> List[Any]:
        results: List[Any] = []
        if isinstance(q, Point):
            self._point_query_recursive(self.root, q, results)
        else:
            self._query_recursive(self.root, q, results)
        return results

    def height(self) -> int:
        h = 1
        node = self.root
        while not node.is_leaf:
            h += 1
            node = node.entries[0].child
        return h

    def dump_structure(self) -> str:
        lines = []
        level = [self.root]
        depth = 0
        while level:
            parts = []
            next_level = []
            for node in level:
                parts.append(
                    f"{'L' if node.is_leaf else 'I'}:"
                    f"{[(e.rect.xmin, e.rect.ymin, e.rect.xmax, e.rect.ymax) for e in node.entries]}"
                )
                if not node.is_leaf:
                    next_level.extend(e.child for e in node.entries)
            lines.append(f"depth={depth}  " + " | ".join(parts))
            level = next_level
            depth += 1
        return "\n".join(lines)

    def _query_recursive(self, node: RTreeNode, query_rect: Rect, out: List[Any]) -> None:
        if node.is_leaf:
            for entry in node.entries:
                if entry.rect.intersects(query_rect):
                    out.append(entry.value)
            return
        for entry in node.entries:
            if entry.rect.intersects(query_rect):
                self._query_recursive(entry.child, query_rect, out)

    def _point_query_recursive(self, node: RTreeNode, point: Point, out: List[Any]) -> None:
        if node.is_leaf:
            for entry in node.entries:
                if entry.rect.contains_point(point):
                    out.append(entry.value)
            return
        for entry in node.entries:
            if entry.rect.contains_point(point):
                self._point_query_recursive(entry.child, point, out)

    def _insert_recursive(self, node: RTreeNode, new_entry: RTreeEntry):
        if node.is_leaf:
            node.entries.append(new_entry)
            if len(node.entries) > self.max_entries:
                return self._split_node(node)
            return None
        best_idx = self._choose_subtree(node, new_entry.rect)
        child = node.entries[best_idx].child
        promoted = self._insert_recursive(child, new_entry)
        if promoted is None:
            node.entries[best_idx].rect = child.mbr()
            return None
        left_child, right_child = promoted
        node.entries.pop(best_idx)
        node.entries.insert(best_idx, RTreeEntry(rect=right_child.mbr(), child=right_child))
        node.entries.insert(best_idx, RTreeEntry(rect=left_child.mbr(), child=left_child))
        if len(node.entries) > self.max_entries:
            return self._split_node(node)
        return None

    def _choose_subtree(self, node: RTreeNode, rect: Rect) -> int:
        best_idx = 0
        best_enlargement = None
        best_area = None
        for i, entry in enumerate(node.entries):
            enlargement = entry.rect.enlargement_needed(rect)
            area = entry.rect.area()
            if (
                best_enlargement is None
                or enlargement < best_enlargement
                or (enlargement == best_enlargement and area < best_area)
            ):
                best_idx = i
                best_enlargement = enlargement
                best_area = area
        return best_idx

    def _split_node(self, node: RTreeNode):
        left_entries, right_entries = self._linear_split(node.entries)
        left = RTreeNode(is_leaf=node.is_leaf, entries=left_entries)
        right = RTreeNode(is_leaf=node.is_leaf, entries=right_entries)
        return left, right

    def _linear_split(self, entries: List[RTreeEntry]) -> Tuple[List[RTreeEntry], List[RTreeEntry]]:
        if len(entries) < 2:
            raise ValueError("Cannot split a node with fewer than 2 entries.")
        seed_a, seed_b = self._pick_seeds(entries)
        group_a = [entries[seed_a]]
        group_b = [entries[seed_b]]
        remaining = [e for i, e in enumerate(entries) if i not in (seed_a, seed_b)]
        while remaining:
            if len(group_a) + len(remaining) == self.min_entries:
                group_a.extend(remaining)
                break
            if len(group_b) + len(remaining) == self.min_entries:
                group_b.extend(remaining)
                break
            entry = remaining.pop(0)
            mbr_a = Rect.enclosing(e.rect for e in group_a)
            mbr_b = Rect.enclosing(e.rect for e in group_b)
            enlarge_a = mbr_a.enlargement_needed(entry.rect)
            enlarge_b = mbr_b.enlargement_needed(entry.rect)
            if enlarge_a < enlarge_b:
                group_a.append(entry)
            elif enlarge_b < enlarge_a:
                group_b.append(entry)
            else:
                if mbr_a.area() < mbr_b.area():
                    group_a.append(entry)
                elif mbr_b.area() < mbr_a.area():
                    group_b.append(entry)
                else:
                    if len(group_a) <= len(group_b):
                        group_a.append(entry)
                    else:
                        group_b.append(entry)
        return group_a, group_b

    def _pick_seeds(self, entries: List[RTreeEntry]) -> Tuple[int, int]:
        def center(rect: Rect):
            return ((rect.xmin + rect.xmax) / 2.0, (rect.ymin + rect.ymax) / 2.0)
        best_pair = (0, 1)
        best_dist_sq = -1.0
        for i in range(len(entries)):
            cx1, cy1 = center(entries[i].rect)
            for j in range(i + 1, len(entries)):
                cx2, cy2 = center(entries[j].rect)
                dist_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
                if dist_sq > best_dist_sq:
                    best_dist_sq = dist_sq
                    best_pair = (i, j)
        return best_pair

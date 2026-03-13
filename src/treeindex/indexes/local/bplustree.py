from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Tuple

from treeindex.core.interfaces import BaseIndex


@dataclass
class BPlusNode:
    is_leaf: bool
    keys: List[Any] = field(default_factory=list)
    children: List["BPlusNode"] = field(default_factory=list)
    values: List[List[Any]] = field(default_factory=list)
    next_leaf: Optional["BPlusNode"] = None


class BPlusTree(BaseIndex[Any, Any, Tuple[Any, Any] | Any]):
    def __init__(self, order: int = 8):
        if order < 3:
            raise ValueError("B+ Tree order must be at least 3.")
        self.order = order
        self.root = BPlusNode(is_leaf=True)
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def build(self, items: Iterable[Tuple[Any, Any]]) -> None:
        for key, value in items:
            self.insert(key, value)

    def insert(self, key: Any, value: Any) -> None:
        promoted = self._insert_recursive(self.root, key, value)
        if promoted is not None:
            promoted_key, right_node = promoted
            new_root = BPlusNode(is_leaf=False)
            new_root.keys = [promoted_key]
            new_root.children = [self.root, right_node]
            self.root = new_root
        self._size += 1

    def search(self, key: Any) -> List[Any]:
        leaf = self._find_leaf(key)
        pos = bisect_left(leaf.keys, key)
        if pos < len(leaf.keys) and leaf.keys[pos] == key:
            return list(leaf.values[pos])
        return []

    def range_search(self, low: Any, high: Any) -> List[Any]:
        if low > high:
            return []
        leaf = self._find_leaf(low)
        results: List[Any] = []
        while leaf is not None:
            for i, key in enumerate(leaf.keys):
                if key < low:
                    continue
                if key > high:
                    return results
                results.extend(leaf.values[i])
            leaf = leaf.next_leaf
        return results

    def query(self, q: Tuple[Any, Any] | Any) -> List[Any]:
        if isinstance(q, tuple) and len(q) == 2:
            return self.range_search(q[0], q[1])
        return self.search(q)

    def height(self) -> int:
        h = 1
        node = self.root
        while not node.is_leaf:
            h += 1
            node = node.children[0]
        return h

    def dump_structure(self) -> str:
        lines = []
        level = [self.root]
        depth = 0
        while level:
            parts = []
            next_level = []
            for node in level:
                parts.append(f"{'L' if node.is_leaf else 'I'}:{node.keys}")
                if not node.is_leaf:
                    next_level.extend(node.children)
            lines.append(f"depth={depth}  " + " | ".join(parts))
            level = next_level
            depth += 1
        return "\n".join(lines)

    def _find_leaf(self, key: Any) -> BPlusNode:
        node = self.root
        while not node.is_leaf:
            idx = bisect_right(node.keys, key)
            node = node.children[idx]
        return node

    def _insert_recursive(self, node: BPlusNode, key: Any, value: Any):
        if node.is_leaf:
            return self._insert_into_leaf(node, key, value)
        child_idx = bisect_right(node.keys, key)
        promoted = self._insert_recursive(node.children[child_idx], key, value)
        if promoted is None:
            return None
        promoted_key, right_child = promoted
        node.keys.insert(child_idx, promoted_key)
        node.children.insert(child_idx + 1, right_child)
        if len(node.children) > self.order:
            return self._split_internal(node)
        return None

    def _insert_into_leaf(self, leaf: BPlusNode, key: Any, value: Any):
        pos = bisect_left(leaf.keys, key)
        if pos < len(leaf.keys) and leaf.keys[pos] == key:
            leaf.values[pos].append(value)
        else:
            leaf.keys.insert(pos, key)
            leaf.values.insert(pos, [value])
        if len(leaf.keys) <= self.order - 1:
            return None
        return self._split_leaf(leaf)

    def _split_leaf(self, leaf: BPlusNode):
        split_index = len(leaf.keys) // 2
        right = BPlusNode(is_leaf=True)
        right.keys = leaf.keys[split_index:]
        right.values = leaf.values[split_index:]
        leaf.keys = leaf.keys[:split_index]
        leaf.values = leaf.values[:split_index]
        right.next_leaf = leaf.next_leaf
        leaf.next_leaf = right
        promoted_key = right.keys[0]
        return promoted_key, right

    def _split_internal(self, node: BPlusNode):
        mid = len(node.keys) // 2
        promoted_key = node.keys[mid]
        right = BPlusNode(is_leaf=False)
        right.keys = node.keys[mid + 1 :]
        right.children = node.children[mid + 1 :]
        node.keys = node.keys[:mid]
        node.children = node.children[: mid + 1]
        return promoted_key, right

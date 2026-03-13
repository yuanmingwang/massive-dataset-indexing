from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from treeindex.geometry.point import Point


@dataclass(frozen=True)
class Rect:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def area(self) -> float:
        return max(0.0, self.xmax - self.xmin) * max(0.0, self.ymax - self.ymin)

    def intersects(self, other: "Rect") -> bool:
        return not (
            self.xmax < other.xmin
            or self.xmin > other.xmax
            or self.ymax < other.ymin
            or self.ymin > other.ymax
        )

    def contains_point(self, point: Point) -> bool:
        return self.xmin <= point.x <= self.xmax and self.ymin <= point.y <= self.ymax

    def union(self, other: "Rect") -> "Rect":
        return Rect(
            xmin=min(self.xmin, other.xmin),
            ymin=min(self.ymin, other.ymin),
            xmax=max(self.xmax, other.xmax),
            ymax=max(self.ymax, other.ymax),
        )

    def enlargement_needed(self, other: "Rect") -> float:
        return self.union(other).area() - self.area()

    @staticmethod
    def enclosing(rects: Iterable["Rect"]) -> "Rect":
        rects = list(rects)
        if not rects:
            raise ValueError("Cannot compute enclosing rectangle of an empty list.")
        result = rects[0]
        for r in rects[1:]:
            result = result.union(r)
        return result

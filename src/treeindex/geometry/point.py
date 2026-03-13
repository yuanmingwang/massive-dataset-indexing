from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def coord(self, axis: int) -> float:
        if axis == 0:
            return self.x
        if axis == 1:
            return self.y
        raise ValueError(f"Unsupported axis for Point: {axis}")

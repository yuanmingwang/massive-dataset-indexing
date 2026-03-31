from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from treeindex.geometry.point import Point
from treeindex.geometry.rect import Rect


@dataclass(frozen=True, order=True)
class GeneratedRecord:
    record_id: int
    attr1: int
    attr2: int


def generate_key_value_pairs(
    n: int,
    *,
    key_min: int = 0,
    key_max: int = 1_000_000,
    attr1_min: int = 0,
    attr1_max: int = 1_000_000,
    attr2_min: int = 0,
    attr2_max: int = 1_000_000,
    allow_duplicates: bool = True,
    seed: int = 0,
) -> List[Tuple[int, GeneratedRecord]]:
    rng = random.Random(seed)
    pairs = []

    if allow_duplicates:
        for record_id in range(n):
            key = rng.randint(key_min, key_max)
            pairs.append(
                (
                    key,
                    GeneratedRecord(
                        record_id=record_id,
                        attr1=rng.randint(attr1_min, attr1_max),
                        attr2=rng.randint(attr2_min, attr2_max),
                    ),
                )
            )
    else:
        if key_max - key_min + 1 < n:
            raise ValueError("Key range too small for unique keys.")
        keys = rng.sample(range(key_min, key_max + 1), n)
        pairs = [
            (
                key,
                GeneratedRecord(
                    record_id=record_id,
                    attr1=rng.randint(attr1_min, attr1_max),
                    attr2=rng.randint(attr2_min, attr2_max),
                ),
            )
            for record_id, key in enumerate(keys)
        ]

    return pairs


def project_records_to_attribute(
    items: Sequence[Tuple[int, GeneratedRecord]],
    attribute: str,
) -> List[Tuple[int, GeneratedRecord]]:
    if attribute == "key":
        return list(items)
    if attribute == "attr1":
        return [(record.attr1, record) for _key, record in items]
    if attribute == "attr2":
        return [(record.attr2, record) for _key, record in items]
    raise ValueError(f"Unsupported attribute '{attribute}'.")


def sample_attribute_point_queries(
    items: Sequence[Tuple[int, GeneratedRecord]],
    attribute: str,
    num_queries: int,
    *,
    seed: int = 0,
) -> List[int]:
    values = [value for value, _record in project_records_to_attribute(items, attribute)]
    return sample_point_queries(values, num_queries=num_queries, seed=seed)


def sample_attribute_range_queries(
    items: Sequence[Tuple[int, GeneratedRecord]],
    attribute: str,
    num_queries: int,
    *,
    max_width: int,
    seed: int = 0,
) -> List[Tuple[int, int]]:
    values = [value for value, _record in project_records_to_attribute(items, attribute)]
    if not values:
        return []
    return sample_range_queries(
        num_queries=num_queries,
        low_min=min(values),
        low_max=max(values),
        max_width=max_width,
        seed=seed,
    )


def sample_point_queries(keys: Sequence[int], num_queries: int, *, seed: int = 0) -> List[int]:
    rng = random.Random(seed)
    return [rng.choice(keys) for _ in range(num_queries)]


def sample_range_queries(
    num_queries: int,
    *,
    low_min: int = 0,
    low_max: int = 900_000,
    max_width: int = 50_000,
    seed: int = 0,
) -> List[Tuple[int, int]]:
    rng = random.Random(seed)
    queries = []
    for _ in range(num_queries):
        low = rng.randint(low_min, low_max)
        width = rng.randint(1, max_width)
        queries.append((low, low + width))
    return queries


def generate_rectangles(
    n: int,
    *,
    space_width: float = 10_000.0,
    space_height: float = 10_000.0,
    max_rect_width: float = 100.0,
    max_rect_height: float = 100.0,
    attr1_min: int = 0,
    attr1_max: int = 1_000_000,
    attr2_min: int = 0,
    attr2_max: int = 1_000_000,
    seed: int = 0,
) -> List[Tuple[Rect, GeneratedRecord]]:
    rng = random.Random(seed)
    rects: List[Tuple[Rect, GeneratedRecord]] = []

    for object_id in range(n):
        xmin = rng.uniform(0, space_width)
        ymin = rng.uniform(0, space_height)
        width = rng.uniform(1, max_rect_width)
        height = rng.uniform(1, max_rect_height)
        xmax = min(space_width, xmin + width)
        ymax = min(space_height, ymin + height)
        rects.append(
            (
                Rect(xmin, ymin, xmax, ymax),
                GeneratedRecord(
                    record_id=object_id,
                    attr1=rng.randint(attr1_min, attr1_max),
                    attr2=rng.randint(attr2_min, attr2_max),
                ),
            )
        )

    return rects


def sample_rect_queries(
    num_queries: int,
    *,
    space_width: float = 10_000.0,
    space_height: float = 10_000.0,
    max_query_width: float = 500.0,
    max_query_height: float = 500.0,
    seed: int = 0,
) -> List[Rect]:
    rng = random.Random(seed)
    queries: List[Rect] = []

    for _ in range(num_queries):
        xmin = rng.uniform(0, space_width)
        ymin = rng.uniform(0, space_height)
        width = rng.uniform(1, max_query_width)
        height = rng.uniform(1, max_query_height)
        xmax = min(space_width, xmin + width)
        ymax = min(space_height, ymin + height)
        queries.append(Rect(xmin, ymin, xmax, ymax))

    return queries


def generate_points(
    n: int,
    *,
    space_width: float = 10_000.0,
    space_height: float = 10_000.0,
    attr1_min: int = 0,
    attr1_max: int = 1_000_000,
    attr2_min: int = 0,
    attr2_max: int = 1_000_000,
    seed: int = 0,
) -> List[Tuple[Point, GeneratedRecord]]:
    rng = random.Random(seed)
    points: List[Tuple[Point, GeneratedRecord]] = []

    for object_id in range(n):
        points.append(
            (
                Point(
                    x=rng.uniform(0, space_width),
                    y=rng.uniform(0, space_height),
                ),
                GeneratedRecord(
                    record_id=object_id,
                    attr1=rng.randint(attr1_min, attr1_max),
                    attr2=rng.randint(attr2_min, attr2_max),
                ),
            )
        )

    return points


def sample_point_rect_queries(
    num_queries: int,
    *,
    space_width: float = 10_000.0,
    space_height: float = 10_000.0,
    max_query_width: float = 500.0,
    max_query_height: float = 500.0,
    seed: int = 0,
) -> List[Rect]:
    return sample_rect_queries(
        num_queries,
        space_width=space_width,
        space_height=space_height,
        max_query_width=max_query_width,
        max_query_height=max_query_height,
        seed=seed,
    )


def sample_exact_point_queries(points: Sequence[Point], num_queries: int, *, seed: int = 0) -> List[Point]:
    rng = random.Random(seed)
    return [rng.choice(points) for _ in range(num_queries)]


def sample_rect_point_queries(rects: Sequence[Rect], num_queries: int, *, seed: int = 0) -> List[Point]:
    rng = random.Random(seed)
    queries: List[Point] = []
    for _ in range(num_queries):
        rect = rng.choice(rects)
        queries.append(
            Point(
                x=rng.uniform(rect.xmin, rect.xmax),
                y=rng.uniform(rect.ymin, rect.ymax),
            )
        )
    return queries

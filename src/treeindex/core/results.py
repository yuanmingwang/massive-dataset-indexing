from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class QueryStats:
    total_seconds: float
    avg_seconds: float
    num_queries: int
    total_results: int
    avg_results: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BuildStats:
    total_seconds: float
    items_indexed: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentResult:
    algorithm: str
    workload: str
    n_items: int
    build_seconds: float
    query_total_seconds: float
    query_avg_seconds: float
    num_queries: int
    total_results: int
    avg_results: float
    extra: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

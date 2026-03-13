from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class DistributedExperimentRow:
    algorithm: str
    workload: str
    n_items: int
    build_s: float
    query_total_s: float
    query_avg_s: float
    num_queries: int
    avg_results: float
    extra: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

from __future__ import annotations

from io import StringIO
from typing import Any, Protocol, Sequence


class ResultRow(Protocol):
    algorithm: str
    workload: str
    n_items: int
    num_queries: int
    avg_results: float
    extra: dict[str, Any]


def _build_seconds(row: Any) -> float:
    value = getattr(row, "build_seconds", None)
    if value is None:
        value = getattr(row, "build_s")
    return float(value)


def _query_total_seconds(row: Any) -> float:
    value = getattr(row, "query_total_seconds", None)
    if value is None:
        value = getattr(row, "query_total_s")
    return float(value)


def _query_avg_seconds(row: Any) -> float:
    value = getattr(row, "query_avg_seconds", None)
    if value is None:
        value = getattr(row, "query_avg_s")
    return float(value)


def pretty_print_results(results: Sequence[ResultRow]) -> None:
    print(render_results_table(results))


def render_results_table(results: Sequence[ResultRow]) -> str:
    if not results:
        return "No results."

    headers = [
        "algorithm",
        "workload",
        "n_items",
        "build_s",
        "query_total_s",
        "query_avg_s",
        "num_queries",
        "avg_results",
        "extra",
    ]

    rows = []
    for r in results:
        rows.append(
            [
                r.algorithm,
                r.workload,
                r.n_items,
                f"{_build_seconds(r):.6f}",
                f"{_query_total_seconds(r):.6f}",
                f"{_query_avg_seconds(r):.8f}",
                r.num_queries,
                f"{r.avg_results:.3f}",
                str(r.extra),
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt_row(row):
        return " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))

    out = StringIO()
    print(fmt_row(headers), file=out)
    print("-+-".join("-" * w for w in widths), file=out)
    for row in rows:
        print(fmt_row(row), file=out)
    return out.getvalue().rstrip()

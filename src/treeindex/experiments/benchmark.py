from __future__ import annotations

import time
from typing import Any, Dict, Sequence

from treeindex.core.results import BuildStats, ExperimentResult, QueryStats


def benchmark_build(index, items: Sequence[tuple[Any, Any]]) -> BuildStats:
    start = time.perf_counter()
    index.build(items)
    end = time.perf_counter()
    return BuildStats(total_seconds=end - start, items_indexed=len(items))


def benchmark_queries(index, queries: Sequence[Any]) -> QueryStats:
    start = time.perf_counter()
    total_results = 0
    for q in queries:
        total_results += len(index.query(q))
    end = time.perf_counter()
    total_seconds = end - start
    num_queries = len(queries)
    return QueryStats(
        total_seconds=total_seconds,
        avg_seconds=(total_seconds / num_queries if num_queries else 0.0),
        num_queries=num_queries,
        total_results=total_results,
        avg_results=(total_results / num_queries if num_queries else 0.0),
    )


def run_single_experiment(*, index_factory, items, queries, workload_name: str, extra: Dict[str, Any] | None = None) -> ExperimentResult:
    index = index_factory()
    build_stats = benchmark_build(index, items)
    query_stats = benchmark_queries(index, queries)
    return ExperimentResult(
        algorithm=index.name,
        workload=workload_name,
        n_items=len(items),
        build_seconds=build_stats.total_seconds,
        query_total_seconds=query_stats.total_seconds,
        query_avg_seconds=query_stats.avg_seconds,
        num_queries=query_stats.num_queries,
        total_results=query_stats.total_results,
        avg_results=query_stats.avg_results,
        extra=extra or {},
    )


def run_experiments_with_shared_build(*, index_factory, items, query_batches):
    index = index_factory()
    build_stats = benchmark_build(index, items)
    results = []
    for batch in query_batches:
        query_stats = benchmark_queries(index, batch["queries"])
        results.append(
            ExperimentResult(
                algorithm=index.name,
                workload=batch["workload_name"],
                n_items=len(items),
                build_seconds=build_stats.total_seconds,
                query_total_seconds=query_stats.total_seconds,
                query_avg_seconds=query_stats.avg_seconds,
                num_queries=query_stats.num_queries,
                total_results=query_stats.total_results,
                avg_results=query_stats.avg_results,
                extra=batch.get("extra", {}),
            )
        )
    return results


def compare_indexes(*, index_factories, items, queries, workload_name: str, extra_builder=None):
    results = []
    for factory in index_factories:
        index = factory()
        build_stats = benchmark_build(index, items)
        query_stats = benchmark_queries(index, queries)
        extra = extra_builder(index) if extra_builder else {}
        results.append(
            ExperimentResult(
                algorithm=index.name,
                workload=workload_name,
                n_items=len(items),
                build_seconds=build_stats.total_seconds,
                query_total_seconds=query_stats.total_seconds,
                query_avg_seconds=query_stats.avg_seconds,
                num_queries=query_stats.num_queries,
                total_results=query_stats.total_results,
                avg_results=query_stats.avg_results,
                extra=extra,
            )
        )
    return results


def compare_indexes_with_shared_build(*, index_factories, items, query_batches, extra_builder=None):
    results = []
    for factory in index_factories:
        index = factory()
        build_stats = benchmark_build(index, items)
        for batch in query_batches:
            query_stats = benchmark_queries(index, batch["queries"])
            extra = extra_builder(index, batch) if extra_builder else {}
            results.append(
                ExperimentResult(
                    algorithm=index.name,
                    workload=batch["workload_name"],
                    n_items=len(items),
                    build_seconds=build_stats.total_seconds,
                    query_total_seconds=query_stats.total_seconds,
                    query_avg_seconds=query_stats.avg_seconds,
                    num_queries=query_stats.num_queries,
                    total_results=query_stats.total_results,
                    avg_results=query_stats.avg_results,
                    extra=extra,
                )
            )
    return results

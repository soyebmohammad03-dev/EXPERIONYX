"""Benchmark/leaderboard: protocol identity, submissions, results and protocol-constrained
comparisons. Never declares a universal "best model" -- see leaderboard/__init__.py."""

from typing import Any

from fastapi import APIRouter

import experionyx.validation as v
from experionyx.api.common import clamp_pagination, page, record
from experionyx.api.deps import RegistryDep, StoreDep
from experionyx.benchmark.entities import Benchmark, BenchmarkResult
from experionyx.benchmark.registry import BenchmarkRegistry
from experionyx.leaderboard.compare import compare_submissions
from experionyx.leaderboard.entities import (
    BenchmarkProtocol,
    BenchmarkSubmission,
    LeaderboardEntry,
    LeaderboardSnapshot,
)

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])


@router.get("/benchmarks")
def search_benchmarks(
    registry: RegistryDep,
    model: str | None = None,
    dataset: str | None = None,
    protocol: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> dict[str, object]:
    found = BenchmarkRegistry(registry).search(model=model, dataset=dataset, protocol=protocol)
    lim, off = clamp_pagination(limit, offset)
    return {
        "items": [record(b) for b in found[off : off + lim]],
        "total": len(found),
        "limit": lim,
        "offset": off,
    }


@router.get("/benchmarks/{benchmark_id}")
def get_benchmark(benchmark_id: str, registry: RegistryDep) -> dict[str, object]:
    v.ref("benchmark_id", benchmark_id, Benchmark.PREFIX)
    b = BenchmarkRegistry(registry).get(benchmark_id)
    results = registry.find(BenchmarkResult, benchmark_id=benchmark_id)
    return {**record(b), "result_ids": sorted(r.id for r in results)}


@router.get("/results/{result_id}")
def get_result(result_id: str, registry: RegistryDep) -> dict[str, object]:
    v.ref("result_id", result_id, BenchmarkResult.PREFIX)
    reg = BenchmarkRegistry(registry)
    result = reg.result(result_id)
    units = reg.units(result_id)
    return {**record(result), "units": [record(u) for u in units]}


@router.get("/results/compare")
def compare_results(a: str, b: str, registry: RegistryDep) -> Any:
    v.ref("a", a, BenchmarkResult.PREFIX)
    v.ref("b", b, BenchmarkResult.PREFIX)
    return BenchmarkRegistry(registry).compare(a, b)


@router.get("/protocols")
def list_protocols(
    registry: RegistryDep, limit: int | None = None, offset: int | None = None
) -> dict[str, object]:
    return page(registry.find(BenchmarkProtocol), limit, offset)


@router.get("/protocols/{protocol_id}/leaderboard")
def get_leaderboard(protocol_id: str, registry: RegistryDep) -> dict[str, object]:
    v.ref("protocol_id", protocol_id, BenchmarkProtocol.PREFIX)
    registry.get(BenchmarkProtocol, protocol_id)
    snapshots = sorted(
        registry.find(LeaderboardSnapshot, protocol_id=protocol_id), key=lambda s: s.id
    )
    latest = snapshots[-1] if snapshots else None
    entries = registry.find(LeaderboardEntry, snapshot_id=latest.id) if latest else []
    return {
        "snapshot": record(latest) if latest else "unavailable",
        "entries": [record(e) for e in sorted(entries, key=lambda e: e.id)],
    }


@router.get("/submissions/{submission_id}")
def get_submission(submission_id: str, registry: RegistryDep) -> dict[str, object]:
    v.ref("submission_id", submission_id, BenchmarkSubmission.PREFIX)
    return record(registry.get(BenchmarkSubmission, submission_id))


@router.get("/submissions/compare")
def compare_submissions_view(a: str, b: str, registry: RegistryDep, store: StoreDep) -> Any:
    """Rejects a comparison across incompatible protocols (raises ValidationError -> 422); does
    not compute a new comparison rule beyond the existing `leaderboard.compare` logic."""
    v.ref("a", a, BenchmarkSubmission.PREFIX)
    v.ref("b", b, BenchmarkSubmission.PREFIX)
    return compare_submissions(registry, store, a, b)


__all__ = ["router"]

"""The benchmark registry: register, get, list, search, and retrieve results, units, artifacts,
provenance and the digest-verified result documents. A logical definition (spec) is registered once."""

from typing import Any

from experionyx.artifacts import ArtifactStore
from experionyx.benchmark.compare import compare_results
from experionyx.benchmark.entities import Benchmark, BenchmarkResult, BenchmarkUnit
from experionyx.domain import Artifact
from experionyx.errors import DuplicateError
from experionyx.faults.report import read_artifact
from experionyx.registry import Registry


class BenchmarkRegistry:
    def __init__(self, registry: Registry, store: ArtifactStore | None = None) -> None:
        self.reg, self.store = registry, store

    def register(self, benchmark: Benchmark) -> Benchmark:
        if self.reg.exists(Benchmark, benchmark.id):
            raise DuplicateError(
                f"benchmark {benchmark.id} is already registered (same definition)"
            )
        self.reg.add(benchmark)
        return benchmark

    def get(self, benchmark_id: str) -> Benchmark:
        return self.reg.get(Benchmark, benchmark_id)

    def result(self, result_id: str) -> BenchmarkResult:
        return self.reg.get(BenchmarkResult, result_id)

    def results(self, benchmark_id: str) -> list[BenchmarkResult]:
        return sorted(self.reg.find(BenchmarkResult, benchmark_id=benchmark_id), key=lambda r: r.id)

    def resolve_result(self, any_id: str) -> BenchmarkResult:
        """A result ID, or a benchmark ID that has exactly one result."""
        if any_id.startswith("brs_"):
            return self.result(any_id)
        rs = self.results(self.get(any_id).id)
        if len(rs) != 1:
            raise ValueError(f"benchmark {any_id} has {len(rs)} results; name one by its brs_ ID")
        return rs[0]

    def find(self, **columns: str) -> list[Benchmark]:
        return self.reg.find(Benchmark, **columns)

    def search(
        self,
        *,
        name: str | None = None,
        version: str | None = None,
        model: str | None = None,
        dataset: str | None = None,
        protocol: str | None = None,
        engine: str | None = None,
        fault: str | None = None,
        coverage: str | None = None,
    ) -> list[Benchmark]:
        cols = {
            k: x
            for k, x in (
                ("name", name),
                ("version", version),
                ("model_record_id", model),
                ("dataset_record_id", dataset),
                ("protocol_hash", protocol),
                ("engine_version", engine),
            )
            if x
        }
        found = self.reg.find(Benchmark, **cols)
        if fault:  # benchmarks that include a fault family of this type
            found = [
                b
                for b in found
                if any(g["fault"] == fault or g["name"] == fault for g in _grids(b))
            ]
        if coverage:
            found = [
                b
                for b in found
                if any(
                    r.coverage_status.value == coverage
                    for r in self.reg.find(BenchmarkResult, benchmark_id=b.id)
                )
            ]
        return sorted(found, key=lambda b: b.id)

    def units(self, result_id: str, **columns: str) -> list[BenchmarkUnit]:
        return sorted(
            self.reg.find(BenchmarkUnit, result_id=result_id, **columns), key=lambda u: u.unit_key
        )

    def artifacts(self, result_id: str) -> list[Artifact]:
        return [
            a
            for a in self.reg.find(Artifact, run_id=self.result(result_id).run_id)
            if a.path.startswith("benchmark/")
        ]

    def document(self, result_id: str, name: str) -> Any:
        """One digest-verified artifact of a result: spec, units, coverage, results or summary."""
        if self.store is None:
            raise ValueError("an artifact store is needed to read benchmark artifacts")
        return read_artifact(
            self.reg, self.store, self.result(result_id).run_id, f"benchmark/{name}.json"
        )

    def bundle(self, result_id: str) -> dict[str, Any]:
        return {
            n: self.document(result_id, n)
            for n in ("spec", "units", "coverage", "results", "summary")
        }

    def provenance(self, result_id: str) -> dict[str, object]:
        r = self.result(result_id)
        b = self.get(r.benchmark_id)
        return {
            "result_id": r.id,
            "benchmark_id": b.id,
            "run_id": r.run_id,
            "spec_id": b.spec_id,
            "protocol_hash": r.protocol_hash,
            "engine_version": b.engine_version,
            "provenance_fingerprint": r.provenance_fingerprint,
            "model_record_id": b.model_record_id,
            "dataset_record_id": b.dataset_record_id,
        }

    def compare(self, a_id: str, b_id: str) -> dict[str, Any]:
        a, b = self.resolve_result(a_id), self.resolve_result(b_id)
        return compare_results(self.bundle(a.id), self.bundle(b.id))


def _grids(b: Benchmark) -> list[Any]:
    grids: Any = b.spec["faults"]
    return list(grids)

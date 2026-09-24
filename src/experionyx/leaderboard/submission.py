"""Submitting one model's real evidence to a registered `BenchmarkProtocol`. Never duplicates
Phase 9 benchmark execution: it calls `benchmark.engine.run_benchmark` directly (the exact same
entry point `experionyx benchmark run` uses) and refuses, before anything is recorded as a
submission, if the executed benchmark's own `protocol_hash` does not match the target protocol
(see docs/leaderboard.md)."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from experionyx.artifacts import ArtifactStore
from experionyx.benchmark.engine import run_benchmark
from experionyx.benchmark.entities import BenchmarkResult
from experionyx.benchmark.spec import BenchmarkSpec
from experionyx.domain import RunStatus
from experionyx.errors import ValidationError
from experionyx.execution import Executor
from experionyx.leaderboard.entities import BenchmarkProtocol, BenchmarkSubmission
from experionyx.leaderboard.protocol import protocol_hash_of
from experionyx.registry import Registry


@dataclass(frozen=True)
class SubmissionResult:
    submission_id: str | None
    benchmark_id: str | None
    result_id: str | None
    status: RunStatus | None
    already_submitted: bool


def submit(
    registry: Registry,
    store: ArtifactStore,
    executor: Executor,
    protocol: BenchmarkProtocol,
    spec: BenchmarkSpec,
    *,
    source_root: Path,
) -> SubmissionResult:
    """`spec` MUST carry a model; its protocol-relevant fields must match `protocol` exactly, or
    this refuses before executing anything (`BenchmarkRefusal`-style: a clean `ValidationError`,
    not a crash, and nothing is recorded)."""
    if protocol_hash_of(registry, spec) != protocol.protocol_hash:
        raise ValidationError(
            f"spec does not match protocol {protocol.id}: its protocol-relevant fields differ "
            "(dataset, metrics, faults, seeds, evaluation, or another protocol-identity input)"
        )
    run = run_benchmark(registry, store, executor, spec, source_root=source_root)
    if run.result_id is None or run.benchmark_id is None:
        return SubmissionResult(None, run.benchmark_id, run.result_id, run.status, False)
    result = registry.get(BenchmarkResult, run.result_id)
    if result.protocol_hash != protocol.protocol_hash:
        raise ValidationError(
            f"benchmark {run.benchmark_id} expanded to a different protocol than {protocol.id} "
            "expected; nothing was submitted"
        )
    existing = registry.find(BenchmarkSubmission, protocol_id=protocol.id, result_id=result.id)
    if existing:
        return SubmissionResult(existing[0].id, run.benchmark_id, run.result_id, run.status, True)
    submission = BenchmarkSubmission(
        protocol.id, result.investigation_id, spec.model, run.benchmark_id, result.id,
        datetime.now(UTC),
    )  # fmt: skip
    registry.add(submission)
    return SubmissionResult(submission.id, run.benchmark_id, run.result_id, run.status, False)


def resolve_submission(registry: Registry, ref: str) -> BenchmarkSubmission:
    if ref.startswith(BenchmarkSubmission.PREFIX + "_"):
        return registry.get(BenchmarkSubmission, ref)
    matches = [s for s in registry.find(BenchmarkSubmission) if s.id.startswith(ref)]
    if len(matches) != 1:
        raise ValidationError(f"{'no' if not matches else 'ambiguous'} submission matches {ref!r}")
    return matches[0]


__all__ = ["SubmissionResult", "resolve_submission", "submit"]

"""Building one immutable `LeaderboardSnapshot`. Every metric value an entry carries is read
straight from its submission's own `BenchmarkResult` document (`BenchmarkRegistry.bundle`) --
nothing here recomputes a metric or a benchmark. Resource measurements stay in their own
`resource_context` field, never folded into the metric values (see docs/leaderboard.md)."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from experionyx.artifacts import ArtifactStore
from experionyx.benchmark.registry import BenchmarkRegistry
from experionyx.errors import ValidationError
from experionyx.hashing import content_hash
from experionyx.leaderboard.entities import (
    BenchmarkProtocol,
    BenchmarkSubmission,
    LeaderboardEntry,
    LeaderboardSnapshot,
)
from experionyx.leaderboard.taxonomy import ReproducibilityState
from experionyx.registry import Registry
from experionyx.reproducibility.entities import ReproductionAttempt
from experionyx.reproducibility.taxonomy import ComparisonOutcome, TargetKind

ENGINE_VERSION = "1.0.0"  # the snapshot-building methodology; bump when filtering/selection semantics change  # fmt: skip
_OUTCOME_TO_STATE = {
    ComparisonOutcome.EQUAL: ReproducibilityState.REPRODUCED,
    ComparisonOutcome.APPROXIMATELY_EQUAL: ReproducibilityState.PARTIALLY_REPRODUCED,
    ComparisonOutcome.DIFFERENT: ReproducibilityState.NOT_REPRODUCED,
    ComparisonOutcome.INCOMPARABLE: ReproducibilityState.NOT_REPRODUCED,
    ComparisonOutcome.UNAVAILABLE: ReproducibilityState.UNAVAILABLE,
}


def reproducibility_state(registry: Registry, result_id: str) -> ReproducibilityState:
    """The most recent `ReproductionAttempt` targeting this result, if any. Never fabricated:
    no attempt means UNAVAILABLE, not an assumed pass."""
    found = registry.find(ReproductionAttempt, target_kind=TargetKind.BENCHMARK_RESULT.value, target_id=result_id)  # fmt: skip
    attempts = sorted(found, key=lambda a: a.attempt)
    if not attempts:
        return ReproducibilityState.UNAVAILABLE
    return _OUTCOME_TO_STATE[attempts[-1].outcome]


@dataclass(frozen=True)
class _Included:
    submission: BenchmarkSubmission
    metrics: dict[str, Any]
    resource_context: dict[str, Any]
    content_fingerprint: str


def _metrics_for(bundle: dict[str, Any], metric_ids: tuple[str, ...]) -> dict[str, Any]:
    raw = {m["metric_id"]: m for m in bundle["results"]["baseline"].get("metrics", [])}
    chosen = metric_ids or tuple(sorted(raw))
    return {
        mid: {
            "value": raw[mid]["value"], "status": raw[mid]["status"],
            "higher_is_better": raw[mid]["higher_is_better"], "interval": raw[mid].get("interval"),
        }
        for mid in chosen
        if mid in raw
    }  # fmt: skip


def build_snapshot(
    registry: Registry,
    store: ArtifactStore,
    protocol: BenchmarkProtocol,
    *,
    metric_ids: tuple[str, ...] = (),
    require_complete_coverage: bool = False,
    require_reproduction: bool = False,
) -> LeaderboardSnapshot:
    """Deterministic per evidence state: rebuilding with the same protocol, the same filtering
    rules and unchanged submissions returns the existing snapshot, never a duplicate."""
    breg = BenchmarkRegistry(registry, store)
    submissions = sorted(registry.find(BenchmarkSubmission, protocol_id=protocol.id), key=lambda s: s.id)  # fmt: skip
    if not submissions:
        raise ValidationError(f"protocol {protocol.id} has no submissions yet; nothing to snapshot")  # fmt: skip
    included: list[_Included] = []
    excluded: dict[str, str] = {}
    for sub in submissions:
        result = breg.result(sub.result_id)
        if require_complete_coverage and result.coverage_status.value != "COMPLETE":
            excluded[sub.id] = f"coverage is {result.coverage_status.value}, not COMPLETE"
            continue
        state = reproducibility_state(registry, result.id)
        if require_reproduction and state not in (ReproducibilityState.REPRODUCED, ReproducibilityState.PARTIALLY_REPRODUCED):  # fmt: skip
            excluded[sub.id] = f"reproducibility state is {state.value}"
            continue
        bundle = breg.bundle(result.id)
        metrics = _metrics_for(bundle, metric_ids)
        resource_context = dict(bundle["summary"].get("resource_analysis", {"status": "UNAVAILABLE", "reason": "no resource measurement was requested by this benchmark"}))  # fmt: skip
        included.append(_Included(sub, metrics, resource_context, result.content_hash()))

    if not included and submissions:
        raise ValidationError("every submission to this protocol was excluded by the filtering rules/evidence requirements; nothing to snapshot")  # fmt: skip

    filtering_rules = {"require_complete_coverage": require_complete_coverage}
    evidence_requirements = {"require_reproduction": require_reproduction}
    chosen_metric_ids = metric_ids or tuple(sorted({mid for i in included for mid in i.metrics}))
    source_fingerprint = content_hash(
        {
            "engine": ENGINE_VERSION,
            "submissions": sorted((i.submission.id, i.content_fingerprint) for i in included),
            "excluded": sorted(excluded),
            "metric_ids": list(chosen_metric_ids),
            "filtering_rules": filtering_rules,
            "evidence_requirements": evidence_requirements,
        }
    )
    existing = [
        s for s in registry.find(LeaderboardSnapshot, protocol_id=protocol.id)
        if s.source_fingerprint == source_fingerprint
    ]  # fmt: skip
    if existing:
        return existing[0]

    investigation_id = included[0].submission.investigation_id
    snapshot = LeaderboardSnapshot(
        protocol.id, investigation_id, ENGINE_VERSION, chosen_metric_ids, filtering_rules,
        evidence_requirements, tuple(i.submission.id for i in included), excluded,
        source_fingerprint, datetime.now(UTC),
    )  # fmt: skip
    registry.add(snapshot)
    for i in included:
        state = reproducibility_state(registry, breg.result(i.submission.result_id).id)
        entry = LeaderboardEntry(
            snapshot.id, i.submission.id, i.submission.model_record_id, i.metrics,
            i.resource_context, state, datetime.now(UTC),
        )  # fmt: skip
        if not registry.exists(LeaderboardEntry, entry.id):
            registry.add(entry)
    return snapshot


__all__ = ["ENGINE_VERSION", "build_snapshot", "reproducibility_state"]

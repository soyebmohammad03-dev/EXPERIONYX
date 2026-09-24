"""Assembles evidence for a report from the registry's OWN persisted content. Never executes or
re-executes anything (see docs/reporting.md): every item here already exists because some earlier
engine (faults, failures, reliability, benchmark, ...) persisted it.

Most analysis entities carry `investigation_id` as an indexed column and are collected generically
through it, mirroring the same lookup the graph builder and CLI already use. `StatisticalAnalysis`
is the one evidence source with no investigation scope of its own (a statistic is computed over
values, not owned by an investigation), so it is only included when its ID is named explicitly in
the `ReportSpec` -- deliberately, rather than guessed from its opaque `sources` blob.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass

from experionyx.benchmark.entities import BenchmarkResult
from experionyx.calibration.entities import CalibrationAnalysis
from experionyx.data_quality.entities import QualityAnalysis
from experionyx.domain import Entity, Experiment, Run, to_jsonable
from experionyx.drift.entities import DriftAnalysis
from experionyx.failures.entities import FailureMode
from experionyx.faults.entities import FaultExperiment
from experionyx.graph.entities import GraphSnapshot
from experionyx.interactions.entities import InteractionAnalysis
from experionyx.leaderboard.entities import BenchmarkSubmission, LeaderboardSnapshot
from experionyx.registry import Registry
from experionyx.reliability.entities import ReliabilityProfile
from experionyx.reproducibility.entities import ReproductionAttempt
from experionyx.resources.entities import ResourceAnalysis
from experionyx.scheduler.entities import ScheduleRun
from experionyx.slices.entities import SliceAnalysis
from experionyx.stats.entities import StatisticalAnalysis
from experionyx.stress.entities import StressAnalysis

# source_kind -> the investigation-scoped entity type it is collected from.
EVIDENCE_SOURCES: tuple[tuple[str, type[Entity]], ...] = (
    ("fault_experiment", FaultExperiment),
    ("failure_mode", FailureMode),
    ("interaction_analysis", InteractionAnalysis),
    ("reliability_profile", ReliabilityProfile),
    ("benchmark_result", BenchmarkResult),
    ("benchmark_submission", BenchmarkSubmission),
    ("leaderboard_snapshot", LeaderboardSnapshot),
    ("slice_analysis", SliceAnalysis),
    ("drift_analysis", DriftAnalysis),
    ("quality_analysis", QualityAnalysis),
    ("stress_analysis", StressAnalysis),
    ("calibration_analysis", CalibrationAnalysis),
    ("resource_analysis", ResourceAnalysis),
    ("schedule_run", ScheduleRun),
    ("graph_snapshot", GraphSnapshot),
    ("reproduction_attempt", ReproductionAttempt),
)


@dataclass(frozen=True)
class EvidenceItem:
    source_kind: str
    entity: Entity

    def summary(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in _scalar_fields(self.entity).items())


EvidenceBundle = Mapping[str, tuple[EvidenceItem, ...]]


def _scalar_fields(entity: Entity) -> dict[str, object]:
    """The entity's own scalar fields, verbatim -- no nested blobs, no computation. This is how
    a report describes evidence without ever reinterpreting it."""
    out: dict[str, object] = {}
    if not is_dataclass(entity):  # every Entity subclass is a frozen dataclass; unreachable
        return out
    for f in fields(entity):
        if f.name in ("id", "created_at"):
            continue
        val = to_jsonable(getattr(entity, f.name))
        if isinstance(val, dict | list):
            continue
        out[f.name] = val
    return out


def collect_evidence(
    registry: Registry,
    investigation_id: str,
    *,
    run_ids: Sequence[str] = (),
    statistical_analysis_ids: Sequence[str] = (),
) -> EvidenceBundle:
    """Every persisted evidence item in scope for `investigation_id`, deterministically ordered
    by entity ID within each bucket."""
    bundle: dict[str, tuple[EvidenceItem, ...]] = {}

    experiments = sorted(
        registry.find(Experiment, investigation_id=investigation_id), key=lambda e: e.id
    )
    bundle["experiment"] = tuple(EvidenceItem("experiment", e) for e in experiments)

    runs = sorted(
        (r for e in experiments for r in registry.find(Run, experiment_id=e.id)), key=lambda r: r.id
    )
    if run_ids:
        wanted = set(run_ids)
        runs = [r for r in runs if r.id in wanted]
    bundle["run"] = tuple(EvidenceItem("run", r) for r in runs)

    for source_kind, cls in EVIDENCE_SOURCES:
        items = sorted(registry.find(cls, investigation_id=investigation_id), key=lambda e: e.id)
        bundle[source_kind] = tuple(EvidenceItem(source_kind, e) for e in items)

    stats = sorted(
        (registry.get(StatisticalAnalysis, sid) for sid in dict.fromkeys(statistical_analysis_ids)),
        key=lambda e: e.id,
    )
    bundle["statistical_analysis"] = tuple(EvidenceItem("statistical_analysis", s) for s in stats)

    return bundle


def evidence_digest(bundle: EvidenceBundle) -> str:
    """A content hash over exactly which evidence records (by ID and content) went into a
    report, so two generations against unchanged evidence always agree, and any change to any
    included record's content changes the digest."""
    from experionyx.hashing import content_hash

    payload = {
        kind: [[item.entity.id, item.entity.content_hash()] for item in items]
        for kind, items in sorted(bundle.items())
    }
    return content_hash(payload)


__all__ = [
    "EVIDENCE_SOURCES",
    "EvidenceBundle",
    "EvidenceItem",
    "collect_evidence",
    "evidence_digest",
]

"""Deterministic dossier construction and sufficiency analysis (see docs/dossier.md). Never
executes or reruns anything: it assembles and analyzes evidence the registry already holds,
reusing the reporting layer's evidence collector and `EvidenceReference` rather than duplicating
them. Weaknesses (gaps, conflicts, staleness, missing provenance) are recorded as first-class
`DossierFinding`s, never hidden."""

from dataclasses import dataclass
from datetime import UTC, datetime

from experionyx.domain import ClaimStatus, Entity, Experiment, Investigation, Run
from experionyx.dossier.entities import (
    DossierFinding,
    DossierItem,
    DossierSnapshot,
    EvidenceDossier,
)
from experionyx.dossier.spec import DossierSpec
from experionyx.dossier.taxonomy import DOSSIER_ENGINE_VERSION, DossierItemKind, SufficiencyStatus
from experionyx.errors import NotFoundError
from experionyx.provenance import Provenance
from experionyx.registry import Registry
from experionyx.reporting.collector import (
    EVIDENCE_SOURCES,
    EvidenceItem,
    collect_evidence,
    evidence_digest,
)
from experionyx.reporting.entities import EvidenceReference, Report, ReportFinding
from experionyx.stats.entities import StatisticalAnalysis

STANDARD_LIMITATIONS: tuple[str, ...] = (
    "This dossier reflects the registry's persisted state at construction time; nothing was "
    "re-executed to produce it.",
    "Conflicts are flagged when two statistical analyses of the same kind disagree; neither "
    "result is discarded or automatically resolved.",
    "Provenance and reproducibility gaps note the ABSENCE of a record, not a judgment that the "
    "underlying run is unreliable.",
)

# source_kind -> Entity type, for resolving an explicit (source_kind, source_id) pair. Extends
# the reporting collector's own table with the kinds it does not collect by default.
_RESOLVABLE_KINDS: dict[str, type[Entity]] = {
    **dict(EVIDENCE_SOURCES),
    "experiment": Experiment,
    "run": Run,
    "statistical_analysis": StatisticalAnalysis,
}


def _now() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


def _item_kind(source_kind: str) -> DossierItemKind:
    return {"experiment": DossierItemKind.EXPERIMENT, "run": DossierItemKind.RUN}.get(
        source_kind, DossierItemKind.EVIDENCE_SOURCE
    )


def _provenance_fingerprint(registry: Registry, source_kind: str, source_id: str) -> str | None:
    if source_kind != "run":
        return None
    matches = registry.find(Provenance, run_id=source_id)
    return matches[0].fingerprint if matches else None


@dataclass(frozen=True)
class DossierBuildResult:
    dossier_id: str
    item_ids: tuple[str, ...]
    finding_ids: tuple[str, ...]
    warnings: tuple[str, ...]


def build_dossier(registry: Registry, spec: DossierSpec) -> DossierBuildResult:
    """Builds and persists an `EvidenceDossier`, its `DossierItem`s and `DossierFinding`s from
    evidence already in `registry`. Deterministic: an unchanged registry and the same spec always
    yield the identical dossier ID."""
    if not registry.exists(Investigation, spec.investigation_id):
        raise NotFoundError(f"investigation {spec.investigation_id} not found")

    explicit_stats = tuple(
        sid for sk, sid in spec.explicit_evidence if sk == "statistical_analysis"
    )
    bundle = collect_evidence(
        registry, spec.investigation_id, statistical_analysis_ids=explicit_stats
    )
    digest = evidence_digest(bundle)

    all_items: list[EvidenceItem] = [i for items in bundle.values() for i in items]
    seen = {(i.source_kind, i.entity.id) for i in all_items}
    for source_kind, source_id in spec.explicit_evidence:
        if source_kind == "statistical_analysis" or (source_kind, source_id) in seen:
            continue
        cls = _RESOLVABLE_KINDS.get(source_kind)
        if cls is None or not registry.exists(cls, source_id):
            raise NotFoundError(f"explicit evidence {source_kind}:{source_id} does not resolve")
        all_items.append(EvidenceItem(source_kind, registry.get(cls, source_id)))
        seen.add((source_kind, source_id))
    all_items.sort(key=lambda i: (i.source_kind, i.entity.id))

    reports = sorted(
        registry.find(Report, investigation_id=spec.investigation_id), key=lambda r: r.id
    )
    report_ids = tuple(r.id for r in reports)
    has_runs = any(i.source_kind == "run" for i in all_items)
    reproduction_ids = tuple(i.entity.id for i in bundle.get("reproduction_attempt", ()))

    # Every gap statement the sufficiency analysis will report, gathered up front (report
    # carry-over, then missing evidence categories) so the dossier's own identity -- computed
    # below from `evidence_gaps` -- is known before any DossierFinding is built.
    seen_gaps: set[str] = set()
    gap_notes: dict[str, str] = {}
    for report in reports:
        for gap in report.evidence_gaps:
            if gap not in seen_gaps:
                seen_gaps.add(gap)
                gap_notes[gap] = f"carried over from report {report.id}"
    for source_kind, _cls in EVIDENCE_SOURCES:
        if bundle.get(source_kind):
            continue
        gap = f"no {source_kind} evidence recorded for investigation {spec.investigation_id}"
        if gap not in seen_gaps:
            seen_gaps.add(gap)
            gap_notes[gap] = "expected evidence category has no persisted records"
    evidence_gaps = tuple(sorted(seen_gaps))

    now = _now()
    dossier = EvidenceDossier(
        spec_id=spec.spec_id,
        investigation_id=spec.investigation_id,
        research_question=spec.research_question,
        source_kind=spec.source_kind,
        source_id=spec.source_id,
        report_ids=report_ids,
        reproduction_attempt_ids=reproduction_ids,
        limitations=STANDARD_LIMITATIONS,
        evidence_gaps=evidence_gaps,
        source_evidence_digest=digest,
        engine_version=DOSSIER_ENGINE_VERSION,
        created_at=now,
    )

    items = [
        DossierItem(
            dossier_id=dossier.id,
            item_kind=_item_kind(i.source_kind),
            source_kind=i.source_kind,
            source_id=i.entity.id,
            reason=f"collected as {i.source_kind} evidence for investigation {spec.investigation_id}",
            captured_content_hash=i.entity.content_hash(),
            provenance_fingerprint=_provenance_fingerprint(registry, i.source_kind, i.entity.id),
            created_at=now,
        )
        for i in all_items
    ]

    findings: list[DossierFinding] = []
    for report in reports:
        for f in sorted(registry.find(ReportFinding, report_id=report.id), key=lambda f: f.id):
            status = (
                SufficiencyStatus.SUPPORTED
                if f.status is ClaimStatus.SUPPORTED
                else SufficiencyStatus.UNSUPPORTED
            )
            findings.append(
                DossierFinding(
                    dossier_id=dossier.id,
                    statement=f.statement,
                    status=status,
                    evidence=f.evidence,
                    report_finding_id=f.id,
                    conflicting_with=(),
                    note=f"reused from report {report.id}",
                    created_at=now,
                )
            )

    for gap in evidence_gaps:
        findings.append(
            DossierFinding(
                dossier_id=dossier.id,
                statement=gap,
                status=SufficiencyStatus.MISSING_EXPECTED_EVIDENCE,
                evidence=(),
                report_finding_id=None,
                conflicting_with=(),
                note=gap_notes[gap],
                created_at=now,
            )
        )

    for item in items:
        if item.provenance_fingerprint is None and item.item_kind is DossierItemKind.RUN:
            findings.append(
                DossierFinding(
                    dossier_id=dossier.id,
                    statement=f"run {item.source_id} has no recorded provenance",
                    status=SufficiencyStatus.PROVENANCE_GAP,
                    evidence=(EvidenceReference("run", item.source_id, "run with no provenance"),),
                    report_finding_id=None,
                    conflicting_with=(),
                    note="absence of a record, not a reliability judgment",
                    created_at=now,
                )
            )

    if has_runs and not reproduction_ids:
        findings.append(
            DossierFinding(
                dossier_id=dossier.id,
                statement="no reproduction attempts have been recorded for this investigation",
                status=SufficiencyStatus.REPRODUCIBILITY_GAP,
                evidence=(),
                report_finding_id=None,
                conflicting_with=(),
                note="applies to every run in scope",
                created_at=now,
            )
        )

    by_kind: dict[str, list[EvidenceItem]] = {}
    for i in bundle.get("statistical_analysis", ()):
        by_kind.setdefault(str(getattr(i.entity, "analysis_kind", "")), []).append(i)
    for kind, group in sorted(by_kind.items()):
        distinct_hashes = {getattr(i.entity, "result_hash", None) for i in group}
        if len(distinct_hashes) <= 1:
            continue
        ordered = sorted(group, key=lambda i: i.entity.id)
        conflict_ids = tuple(
            DossierFinding(
                dossier_id=dossier.id,
                statement=f"conflicting {kind} results: {i.entity.id}",
                status=SufficiencyStatus.CONFLICTING_EVIDENCE,
                evidence=(EvidenceReference("statistical_analysis", i.entity.id, i.summary()),),
                report_finding_id=None,
                conflicting_with=(),
                note="disagreement between statistical analyses of the same kind",
                created_at=now,
            ).id
            for i in ordered
        )
        findings.extend(
            DossierFinding(
                dossier_id=dossier.id,
                statement=f"conflicting {kind} results: {i.entity.id}",
                status=SufficiencyStatus.CONFLICTING_EVIDENCE,
                evidence=(EvidenceReference("statistical_analysis", i.entity.id, i.summary()),),
                report_finding_id=None,
                conflicting_with=tuple(sorted(x for x in conflict_ids if x != this_id)),
                note="disagreement between statistical analyses of the same kind",
                created_at=now,
            )
            for i, this_id in zip(ordered, conflict_ids, strict=True)
        )

    warnings: list[str] = []
    with registry.transaction():
        if not registry.exists(EvidenceDossier, dossier.id):
            registry.add(dossier)
        else:
            warnings.append(f"dossier {dossier.id} already exists for this spec and evidence state")
        for item in items:
            if not registry.exists(DossierItem, item.id):
                registry.add(item)
        for finding in findings:
            if not registry.exists(DossierFinding, finding.id):
                registry.add(finding)

    return DossierBuildResult(
        dossier.id,
        tuple(sorted(i.id for i in items)),
        tuple(sorted(f.id for f in findings)),
        tuple(warnings),
    )


def build_snapshot(registry: Registry, dossier_id: str) -> str:
    """Freezes the CURRENT items and findings of `dossier_id` into an immutable, independently
    addressable `DossierSnapshot`. Idempotent: snapshotting an unchanged dossier again returns
    the same snapshot ID."""
    dossier = registry.get(EvidenceDossier, dossier_id)
    item_ids = tuple(sorted(i.id for i in registry.find(DossierItem, dossier_id=dossier_id)))
    finding_ids = tuple(sorted(f.id for f in registry.find(DossierFinding, dossier_id=dossier_id)))
    snapshot = DossierSnapshot(
        dossier.id, dossier.source_evidence_digest, item_ids, finding_ids, _now()
    )
    if not registry.exists(DossierSnapshot, snapshot.id):
        registry.add(snapshot)
    return snapshot.id


__all__ = ["STANDARD_LIMITATIONS", "DossierBuildResult", "build_dossier", "build_snapshot"]

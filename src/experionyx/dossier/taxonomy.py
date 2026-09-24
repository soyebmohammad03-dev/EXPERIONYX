"""Controlled vocabulary for evidence dossiers (see docs/dossier.md). A dossier never hides a
weakness: every status here is a first-class, reportable outcome, not an implicit default."""

from enum import StrEnum


class DossierItemKind(StrEnum):
    """What a `DossierItem` points at. Mirrors the reporting layer's `EvidenceReference` source
    kinds plus the two dossier-specific additions (a whole report, or another dossier item's
    claim). Kept as its own small vocabulary rather than a copy of every registry PREFIX, because
    a dossier item also carries an inclusion reason a generic reference does not."""

    INVESTIGATION = "INVESTIGATION"
    EXPERIMENT = "EXPERIMENT"
    RUN = "RUN"
    REPORT = "REPORT"
    REPORT_FINDING = "REPORT_FINDING"
    EVIDENCE_SOURCE = (
        "EVIDENCE_SOURCE"  # any collector.EvidenceItem (fault/failure/reliability/...)
    )


class SufficiencyStatus(StrEnum):
    """The evidentiary status of one claim examined by a dossier's sufficiency analysis. Never
    silently omitted: every claim in scope gets exactly one of these (see docs/dossier.md)."""

    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    MISSING_EXPECTED_EVIDENCE = "MISSING_EXPECTED_EVIDENCE"
    UNAVAILABLE_EVIDENCE = "UNAVAILABLE_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    STALE_EVIDENCE = "STALE_EVIDENCE"
    PROVENANCE_GAP = "PROVENANCE_GAP"
    REPRODUCIBILITY_GAP = "REPRODUCIBILITY_GAP"


class DossierSourceKind(StrEnum):
    """What a `DossierSpec` was built from. Determines how `builder.build_dossier` resolves the
    initial scope of runs/experiments/evidence before sufficiency analysis runs."""

    INVESTIGATION = "INVESTIGATION"
    REPORT = "REPORT"
    RUN = "RUN"
    FAILURE_MODE = "FAILURE_MODE"
    BENCHMARK_RESULT = "BENCHMARK_RESULT"
    RELIABILITY_PROFILE = "RELIABILITY_PROFILE"
    GRAPH_SNAPSHOT = "GRAPH_SNAPSHOT"
    EXPLICIT_EVIDENCE_IDS = "EXPLICIT_EVIDENCE_IDS"


DOSSIER_ENGINE_VERSION = "1.0.0"  # the construction/sufficiency-analysis methodology


__all__ = [
    "DOSSIER_ENGINE_VERSION",
    "DossierItemKind",
    "DossierSourceKind",
    "SufficiencyStatus",
]

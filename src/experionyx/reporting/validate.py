"""Research-quality validation for a generated report (see docs/reporting.md). Never mutates
anything; a report and its findings are immutable regardless of what validation finds."""

from collections.abc import Mapping
from dataclasses import dataclass

from experionyx.domain import ClaimStatus, Entity, Investigation
from experionyx.errors import NotFoundError
from experionyx.registry import Registry
from experionyx.reporting.collector import EVIDENCE_SOURCES
from experionyx.reporting.entities import Report, ReportFinding, ReportTemplate

# source_kind -> the Entity class an EvidenceReference of that kind is expected to resolve
# against. Kept alongside EVIDENCE_SOURCES plus the two additional kinds the collector emits.
_SOURCE_TYPES: Mapping[str, type[Entity]] = dict(EVIDENCE_SOURCES)


def _resolvable(registry: Registry, source_kind: str, source_id: str) -> bool | None:
    """True/False if resolvable, None if `source_kind` is not one this validator can check
    (an evidence source added since; treated as a warning, not a hard failure)."""
    from experionyx.domain import Experiment, Run
    from experionyx.stats.entities import StatisticalAnalysis

    cls = _SOURCE_TYPES.get(source_kind) or {
        "experiment": Experiment,
        "run": Run,
        "statistical_analysis": StatisticalAnalysis,
    }.get(source_kind)
    if cls is None:
        return None
    return registry.exists(cls, source_id)


@dataclass(frozen=True)
class ValidationIssue:
    severity: str  # "ERROR" | "WARNING"
    code: str
    message: str


def validate_report(registry: Registry, report_id: str) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    try:
        report = registry.get(Report, report_id)
    except NotFoundError:
        return (ValidationIssue("ERROR", "report_not_found", f"report {report_id} not found"),)

    if not registry.exists(Investigation, report.investigation_id):
        issues.append(
            ValidationIssue(
                "ERROR",
                "dangling_investigation",
                f"investigation {report.investigation_id} does not resolve",
            )
        )
    if not registry.exists(ReportTemplate, report.template_id):
        issues.append(
            ValidationIssue(
                "ERROR", "dangling_template", f"template {report.template_id} does not resolve"
            )
        )

    for section in report.sections:
        for ev in section.evidence:
            ok = _resolvable(registry, ev.source_kind, ev.source_id)
            if ok is False:
                issues.append(
                    ValidationIssue(
                        "ERROR",
                        "dangling_evidence",
                        f"section {section.kind.value}: evidence {ev.source_id} ({ev.source_kind}) does not resolve",
                    )
                )
            elif ok is None:
                issues.append(
                    ValidationIssue(
                        "WARNING",
                        "unknown_evidence_kind",
                        f"section {section.kind.value}: cannot verify evidence kind {ev.source_kind!r}",
                    )
                )

    findings = registry.find(ReportFinding, report_id=report_id)
    for finding in findings:
        if finding.status is ClaimStatus.SUPPORTED and not finding.evidence:
            issues.append(
                ValidationIssue(
                    "ERROR",
                    "unsupported_claim",
                    f"finding {finding.id} is SUPPORTED with no evidence",
                )
            )
        for ev in finding.evidence:
            ok = _resolvable(registry, ev.source_kind, ev.source_id)
            if ok is False:
                issues.append(
                    ValidationIssue(
                        "ERROR",
                        "dangling_evidence",
                        f"finding {finding.id}: evidence {ev.source_id} ({ev.source_kind}) does not resolve",
                    )
                )
            elif ok is None:
                issues.append(
                    ValidationIssue(
                        "WARNING",
                        "unknown_evidence_kind",
                        f"finding {finding.id}: cannot verify evidence kind {ev.source_kind!r}",
                    )
                )
        if finding.statistical_analysis_id is not None:
            from experionyx.stats.entities import StatisticalAnalysis

            if not registry.exists(StatisticalAnalysis, finding.statistical_analysis_id):
                issues.append(
                    ValidationIssue(
                        "ERROR",
                        "dangling_statistical_analysis",
                        f"finding {finding.id}: statistical analysis {finding.statistical_analysis_id} does not resolve",
                    )
                )

    return tuple(issues)


__all__ = ["ValidationIssue", "validate_report"]

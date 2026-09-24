"""Deterministic report generation from persisted evidence (see docs/reporting.md). Generation
never executes anything; it only reads, counts and cites what the registry already holds. A
finding is always a descriptive statement about the evidence found ("N fault experiments were
recorded"), never an interpretive one ("the model is reliable") -- interpretation is left to the
reader, who is given the evidence and its provenance to judge it with.
"""

from dataclasses import dataclass
from datetime import UTC, datetime

from experionyx.domain import ClaimStatus, Investigation
from experionyx.errors import NotFoundError, ValidationError
from experionyx.registry import Registry
from experionyx.reporting.collector import (
    EvidenceBundle,
    EvidenceItem,
    collect_evidence,
    evidence_digest,
)
from experionyx.reporting.entities import (
    EvidenceReference,
    Report,
    ReportFinding,
    ReportSection,
    ReportTemplate,
)
from experionyx.reporting.spec import REPORT_ENGINE_VERSION, ReportSpec
from experionyx.reporting.taxonomy import (
    ALL_SOURCE_KINDS,
    BUILTIN_TEMPLATE_SECTIONS,
    SECTION_SOURCES,
    SYNTHESIZED_SECTIONS,
    TEMPLATE_ENGINE_VERSION,
    ReportStatus,
    ReportType,
    SectionKind,
)

STANDARD_LIMITATIONS: tuple[str, ...] = (
    "This report reflects the registry's persisted state at generation time; nothing was "
    "re-executed to produce it.",
    "Findings are descriptive counts and citations of persisted evidence, not new statistical "
    "conclusions; statistical claims are reported only where a StatisticalAnalysis record exists.",
    "The absence of a section's evidence means no matching analysis has been persisted for this "
    "investigation, not that the underlying property is absent.",
)


def builtin_template(report_type: ReportType) -> ReportTemplate:
    """The current built-in template for `report_type`. `created_at` is fixed so the template's
    content-addressed ID is stable across processes."""
    return ReportTemplate(
        name=f"builtin-{report_type.value.lower()}",
        version=TEMPLATE_ENGINE_VERSION,
        report_type=report_type,
        section_kinds=BUILTIN_TEMPLATE_SECTIONS[report_type],
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def ensure_template(registry: Registry, template: ReportTemplate) -> str:
    """Persists `template` if it is not already there (content-addressed, so this is idempotent)
    and returns its ID."""
    if not registry.exists(ReportTemplate, template.id):
        registry.add(template)
    return template.id


def _ref(item: EvidenceItem) -> EvidenceReference:
    return EvidenceReference(item.source_kind, item.entity.id, item.summary())


def _section_content(kind: SectionKind, items: tuple[EvidenceItem, ...]) -> str:
    if not items:
        return f"No {kind.value.replace('_', ' ').lower()} evidence is available for this investigation."
    lines = [f"- `{i.entity.id}` ({i.source_kind}): {i.summary()}" for i in items]
    return "\n".join(lines)


def _build_section(kind: SectionKind, bundle: EvidenceBundle) -> tuple[ReportSection, list[str]]:
    """Returns the section and any evidence-gap notes it produced."""
    source_kinds = SECTION_SOURCES.get(kind, ())
    items = tuple(i for sk in source_kinds for i in bundle.get(sk, ()))
    available = bool(items) if source_kinds else True
    gaps = [f"no {sk} evidence found" for sk in source_kinds if not bundle.get(sk)]
    section = ReportSection(
        kind=kind,
        title=kind.value.replace("_", " ").title(),
        content=_section_content(kind, items),
        evidence=tuple(_ref(i) for i in items),
        available=available,
    )
    return section, gaps


def _findings(report_id: str, bundle: EvidenceBundle) -> tuple[ReportFinding, ...]:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    findings: list[ReportFinding] = []
    for source_kind in sorted(ALL_SOURCE_KINDS | {"statistical_analysis"}):
        items = bundle.get(source_kind, ())
        if not items:
            continue
        statement = f"{len(items)} {source_kind.replace('_', ' ')} record(s) were found for this investigation."
        sa_id = items[0].entity.id if source_kind == "statistical_analysis" else None
        findings.append(
            ReportFinding(
                report_id=report_id,
                statement=statement,
                status=ClaimStatus.SUPPORTED,
                evidence=tuple(_ref(i) for i in items),
                statistical_analysis_id=sa_id,
                limitations=(),
                created_at=now,
            )
        )
    return tuple(findings)


@dataclass(frozen=True)
class ReportGenerationResult:
    report_id: str
    finding_ids: tuple[str, ...]
    warnings: tuple[str, ...]


def generate_report(registry: Registry, spec: ReportSpec) -> ReportGenerationResult:
    """Builds and persists a `Report` (and its `ReportFinding`s) from evidence already in
    `registry`. Deterministic: calling this twice with an unchanged registry and the same spec
    yields the identical report ID (`add` is then a no-op duplicate, safe to call again)."""
    if not registry.exists(Investigation, spec.investigation_id):
        raise NotFoundError(f"investigation {spec.investigation_id} not found")
    template = registry.get(ReportTemplate, spec.template_id)
    if template.report_type is not spec.report_type:
        raise ValidationError(
            f"template {template.id} is for {template.report_type.value}, "
            f"spec asks for {spec.report_type.value}"
        )

    bundle = collect_evidence(
        registry,
        spec.investigation_id,
        run_ids=spec.run_ids,
        statistical_analysis_ids=spec.statistical_analysis_ids,
    )
    digest = evidence_digest(bundle)

    sections: list[ReportSection] = []
    gaps: list[str] = []
    for kind in template.section_kinds:
        if kind in SYNTHESIZED_SECTIONS:
            continue
        section, section_gaps = _build_section(kind, bundle)
        sections.append(section)
        gaps.extend(section_gaps)
    gaps = sorted(set(gaps))

    investigation = registry.get(Investigation, spec.investigation_id)
    total_items = sum(len(v) for k, v in bundle.items())
    summary = (
        f"{total_items} evidence record(s) across {sum(1 for v in bundle.values() if v)} "
        f"source(s) were collected for investigation `{investigation.id}` "
        f'("{investigation.name}").'
    )
    if SectionKind.EXECUTIVE_SUMMARY in template.section_kinds:
        sections.insert(
            0, ReportSection(SectionKind.EXECUTIVE_SUMMARY, "Executive Summary", summary, (), True)
        )
    if SectionKind.RESEARCH_QUESTION in template.section_kinds:
        sections.insert(
            1,
            ReportSection(
                SectionKind.RESEARCH_QUESTION, "Research Question", investigation.question, (), True
            ),
        )
    if SectionKind.METHODS in template.section_kinds:
        methods = (
            f"Evidence was assembled deterministically from persisted registry records by the "
            f"reporting engine (v{REPORT_ENGINE_VERSION}) using template `{template.id}`. "
            "No experiment was executed or re-executed to produce this report."
        )
        sections.append(ReportSection(SectionKind.METHODS, "Methods", methods, (), True))
    if SectionKind.LIMITATIONS in template.section_kinds:
        sections.append(
            ReportSection(
                SectionKind.LIMITATIONS,
                "Limitations",
                "\n".join(f"- {x}" for x in STANDARD_LIMITATIONS),
                (),
                True,
            )
        )
    if SectionKind.EVIDENCE_GAPS in template.section_kinds:
        gap_text = (
            "\n".join(f"- {g}" for g in gaps)
            if gaps
            else "No evidence gaps were identified for the sections in this template."
        )
        sections.append(
            ReportSection(SectionKind.EVIDENCE_GAPS, "Evidence Gaps", gap_text, (), True)
        )
    if SectionKind.CONCLUSIONS in template.section_kinds:
        conclusions = (
            "This report presents evidence as found; it draws no ranking, composite score, or "
            "universal capability conclusion. Readers should weigh each finding against its "
            "cited evidence and limitations."
        )
        sections.append(
            ReportSection(SectionKind.CONCLUSIONS, "Conclusions", conclusions, (), True)
        )

    status = ReportStatus.INCOMPLETE if gaps else ReportStatus.GENERATED
    report = Report(
        spec_id=spec.spec_id,
        report_type=spec.report_type,
        investigation_id=spec.investigation_id,
        template_id=template.id,
        title=f"{spec.report_type.value.replace('_', ' ').title()}: {investigation.name}",
        status=status,
        sections=tuple(sections),
        limitations=STANDARD_LIMITATIONS,
        evidence_gaps=tuple(gaps),
        reproducibility={
            "engine_version": REPORT_ENGINE_VERSION,
            "spec": spec.to_dict(),
            "evidence_digest": digest,
        },
        source_evidence_digest=digest,
        engine_version=REPORT_ENGINE_VERSION,
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    warnings: list[str] = []
    with registry.transaction():
        if not registry.exists(Report, report.id):
            registry.add(report)
        else:
            warnings.append(f"report {report.id} already exists for this spec and evidence state")
        finding_ids: list[str] = []
        for finding in _findings(report.id, bundle):
            if not registry.exists(ReportFinding, finding.id):
                registry.add(finding)
            finding_ids.append(finding.id)
    return ReportGenerationResult(report.id, tuple(finding_ids), tuple(warnings))


__all__ = [
    "STANDARD_LIMITATIONS",
    "ReportGenerationResult",
    "builtin_template",
    "ensure_template",
    "generate_report",
]

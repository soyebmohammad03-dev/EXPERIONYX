"""Domain-model tests for the reporting layer: identity, immutability and evidence discipline
(see docs/reporting.md)."""

from datetime import UTC, datetime

import pytest

from experionyx.domain import ClaimStatus
from experionyx.errors import ValidationError
from experionyx.reporting.entities import (
    EvidenceReference,
    Report,
    ReportFinding,
    ReportSection,
    ReportTemplate,
)
from experionyx.reporting.spec import ReportSpec
from experionyx.reporting.taxonomy import (
    BUILTIN_TEMPLATE_SECTIONS,
    ReportStatus,
    ReportType,
    SectionKind,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _template() -> ReportTemplate:
    return ReportTemplate(
        "builtin", "1.0.0", ReportType.INVESTIGATION_DOSSIER, (SectionKind.EXECUTIVE_SUMMARY,), T0
    )


def test_template_identity_changes_with_version() -> None:
    a = _template()
    b = ReportTemplate(
        "builtin", "2.0.0", ReportType.INVESTIGATION_DOSSIER, (SectionKind.EXECUTIVE_SUMMARY,), T0
    )
    assert a.id != b.id


def test_template_identity_changes_with_section_list() -> None:
    a = _template()
    b = ReportTemplate(
        "builtin",
        "1.0.0",
        ReportType.INVESTIGATION_DOSSIER,
        (SectionKind.EXECUTIVE_SUMMARY, SectionKind.CONCLUSIONS),
        T0,
    )
    assert a.id != b.id


def test_template_section_kinds_must_not_repeat() -> None:
    with pytest.raises(ValidationError):
        ReportTemplate(
            "x",
            "1.0.0",
            ReportType.FAILURE_DOSSIER,
            (SectionKind.CONCLUSIONS, SectionKind.CONCLUSIONS),
            T0,
        )


def test_every_report_type_has_a_non_empty_builtin_template() -> None:
    for report_type in ReportType:
        assert BUILTIN_TEMPLATE_SECTIONS[report_type]  # every type is covered, none is empty


def _report(template: ReportTemplate, investigation_id: str, digest: str) -> Report:
    return Report(
        spec_id="rsp_" + "1" * 32,
        report_type=ReportType.INVESTIGATION_DOSSIER,
        investigation_id=investigation_id,
        template_id=template.id,
        title="t",
        status=ReportStatus.GENERATED,
        sections=(),
        limitations=(),
        evidence_gaps=(),
        reproducibility={},
        source_evidence_digest=digest,
        engine_version="1.0.0",
        generated_at=T0,
    )


def test_report_identity_is_spec_and_evidence_digest() -> None:
    tpl = _template()
    inv = "inv_" + "1" * 32
    a = _report(tpl, inv, "sha256:" + "aa" * 32)
    b = _report(tpl, inv, "sha256:" + "aa" * 32)
    c = _report(tpl, inv, "sha256:" + "bb" * 32)
    assert a.id == b.id  # same spec + same evidence => same report
    assert a.id != c.id  # changed evidence => a distinct report, not a mutation


def test_a_supported_finding_must_cite_evidence() -> None:
    with pytest.raises(ValidationError, match="SUPPORTED"):
        ReportFinding("rpt_" + "1" * 32, "claim", ClaimStatus.SUPPORTED, (), None, (), T0)


def test_an_inconclusive_finding_may_have_no_evidence() -> None:
    finding = ReportFinding("rpt_" + "1" * 32, "claim", ClaimStatus.INCONCLUSIVE, (), None, (), T0)
    assert finding.status is ClaimStatus.INCONCLUSIVE


def test_finding_round_trips_through_dict() -> None:
    ref = EvidenceReference("run", "run_" + "1" * 32, "a run")
    finding = ReportFinding(
        "rpt_" + "1" * 32, "claim", ClaimStatus.SUPPORTED, (ref,), None, ("caveat",), T0
    )
    again = ReportFinding.from_dict(finding.to_dict())
    assert again == finding
    assert again.id == finding.id


def test_section_round_trips_and_records_unavailability_explicitly() -> None:
    section = ReportSection(
        SectionKind.RELIABILITY_EVIDENCE, "Reliability Evidence", "no evidence", (), False
    )
    again = ReportSection.from_dict(section.to_dict())
    assert again == section
    assert again.available is False


def test_report_spec_identity_is_deterministic_and_scope_sensitive() -> None:
    inv = "inv_" + "1" * 32
    tpl = "rtp_" + "1" * 32
    a = ReportSpec(ReportType.BENCHMARK_REPORT, inv, tpl)
    b = ReportSpec(ReportType.BENCHMARK_REPORT, inv, tpl)
    c = ReportSpec(ReportType.BENCHMARK_REPORT, inv, tpl, run_ids=("run_" + "2" * 32,))
    assert a.spec_id == b.spec_id
    assert a.spec_id != c.spec_id
    assert ReportSpec.from_dict(a.to_dict()).spec_id == a.spec_id

"""Real report generation against a real registry: evidence collection, section/claim assembly,
determinism, evidence-gap recording, validation and rendering (see docs/reporting.md)."""

import procedures
from conftest import Lab
from experionyx.domain import ClaimStatus, Experiment, Run
from experionyx.errors import NotFoundError
from experionyx.reporting.collector import collect_evidence, evidence_digest
from experionyx.reporting.entities import Report, ReportFinding
from experionyx.reporting.generator import builtin_template, ensure_template, generate_report
from experionyx.reporting.render import render_html, render_markdown
from experionyx.reporting.spec import ReportSpec
from experionyx.reporting.taxonomy import ReportStatus, ReportType
from experionyx.reporting.validate import validate_report


def _demo_run(lab: Lab) -> None:
    """A real, tiny executed run under `lab`'s experiment, so the investigation has baseline
    evidence beyond just the Investigation/Experiment records."""
    lab.executor.execute(lab.experiment.id, procedures.ok, seed=0)


def test_evidence_collection_is_empty_for_an_untouched_investigation(lab: Lab) -> None:
    bundle = collect_evidence(lab.registry, lab.investigation.id)
    assert bundle["experiment"][0].entity.id == lab.experiment.id
    assert bundle["fault_experiment"] == ()  # no fault evidence has been recorded


def test_evidence_digest_changes_when_a_new_run_is_added(lab: Lab) -> None:
    before = evidence_digest(collect_evidence(lab.registry, lab.investigation.id))
    _demo_run(lab)
    after = evidence_digest(collect_evidence(lab.registry, lab.investigation.id))
    assert before != after


def test_generate_report_is_deterministic_against_unchanged_evidence(lab: Lab) -> None:
    template_id = ensure_template(lab.registry, builtin_template(ReportType.INVESTIGATION_DOSSIER))
    spec = ReportSpec(ReportType.INVESTIGATION_DOSSIER, lab.investigation.id, template_id)
    first = generate_report(lab.registry, spec)
    second = generate_report(lab.registry, spec)
    assert first.report_id == second.report_id
    assert second.warnings  # the second call recognized the duplicate rather than erroring


def test_generate_report_produces_a_new_report_when_evidence_changes(lab: Lab) -> None:
    template_id = ensure_template(lab.registry, builtin_template(ReportType.INVESTIGATION_DOSSIER))
    spec = ReportSpec(ReportType.INVESTIGATION_DOSSIER, lab.investigation.id, template_id)
    before = generate_report(lab.registry, spec)
    _demo_run(lab)
    after = generate_report(lab.registry, spec)
    assert before.report_id != after.report_id
    assert lab.registry.exists(Report, before.report_id)  # the old report is untouched, not gone


def test_report_with_no_evidence_beyond_the_experiment_is_incomplete_with_recorded_gaps(
    lab: Lab,
) -> None:
    template_id = ensure_template(lab.registry, builtin_template(ReportType.INVESTIGATION_DOSSIER))
    spec = ReportSpec(ReportType.INVESTIGATION_DOSSIER, lab.investigation.id, template_id)
    result = generate_report(lab.registry, spec)
    report = lab.registry.get(Report, result.report_id)
    assert report.status is ReportStatus.INCOMPLETE
    assert any("fault_experiment" in g for g in report.evidence_gaps)


def test_every_supported_finding_cites_resolvable_evidence(lab: Lab) -> None:
    _demo_run(lab)
    template_id = ensure_template(lab.registry, builtin_template(ReportType.INVESTIGATION_DOSSIER))
    spec = ReportSpec(ReportType.INVESTIGATION_DOSSIER, lab.investigation.id, template_id)
    result = generate_report(lab.registry, spec)
    findings = lab.registry.find(ReportFinding, report_id=result.report_id)
    assert findings
    for f in findings:
        assert f.status is ClaimStatus.SUPPORTED
        assert f.evidence
        for ref in f.evidence:
            assert lab.registry.exists(
                Run if ref.source_kind == "run" else Experiment, ref.source_id
            ) or ref.source_kind not in ("run", "experiment")


def test_validate_report_finds_no_issues_on_a_well_formed_report(lab: Lab) -> None:
    _demo_run(lab)
    template_id = ensure_template(lab.registry, builtin_template(ReportType.INVESTIGATION_DOSSIER))
    spec = ReportSpec(ReportType.INVESTIGATION_DOSSIER, lab.investigation.id, template_id)
    result = generate_report(lab.registry, spec)
    issues = validate_report(lab.registry, result.report_id)
    assert not any(i.severity == "ERROR" for i in issues)


def test_validate_report_raises_not_found_information_for_an_unknown_report(lab: Lab) -> None:
    issues = validate_report(lab.registry, "rpt_" + "9" * 32)
    assert issues and issues[0].code == "report_not_found"


def test_generate_report_rejects_an_unknown_investigation(lab: Lab) -> None:
    template_id = ensure_template(lab.registry, builtin_template(ReportType.INVESTIGATION_DOSSIER))
    spec = ReportSpec(ReportType.INVESTIGATION_DOSSIER, "inv_" + "9" * 32, template_id)
    try:
        generate_report(lab.registry, spec)
        raise AssertionError("expected NotFoundError")
    except NotFoundError:
        pass


def test_markdown_export_is_deterministic_and_includes_the_claim_evidence_matrix(lab: Lab) -> None:
    _demo_run(lab)
    template_id = ensure_template(lab.registry, builtin_template(ReportType.INVESTIGATION_DOSSIER))
    spec = ReportSpec(ReportType.INVESTIGATION_DOSSIER, lab.investigation.id, template_id)
    result = generate_report(lab.registry, spec)
    report = lab.registry.get(Report, result.report_id)
    a = render_markdown(lab.registry, report)
    b = render_markdown(lab.registry, report)
    assert a == b
    assert "Claim / Evidence Matrix" in a
    assert report.id in a
    html = render_html(lab.registry, report)
    assert "<pre>" in html and report.title in html

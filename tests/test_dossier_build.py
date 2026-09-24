"""Real dossier construction against a real registry: evidence/report reuse, sufficiency
analysis, provenance and reproducibility gaps, conflicting evidence, and snapshot immutability
(see docs/dossier.md)."""

from datetime import UTC, datetime

import procedures
from conftest import Lab
from experionyx.dossier.builder import build_dossier, build_snapshot
from experionyx.dossier.entities import (
    DossierFinding,
    DossierItem,
    DossierSnapshot,
    EvidenceDossier,
)
from experionyx.dossier.spec import DossierSpec
from experionyx.dossier.taxonomy import DossierSourceKind, SufficiencyStatus
from experionyx.errors import NotFoundError
from experionyx.reporting.generator import builtin_template, ensure_template, generate_report
from experionyx.reporting.spec import ReportSpec
from experionyx.reporting.taxonomy import ReportType
from experionyx.stats.entities import StatisticalAnalysis


def _spec(lab: Lab, **kw: object) -> DossierSpec:
    return DossierSpec(
        lab.investigation.id,
        "does it reproduce?",
        DossierSourceKind.INVESTIGATION,
        lab.investigation.id,
        **kw,
    )  # type: ignore[arg-type]


def test_build_dossier_is_deterministic_against_unchanged_evidence(lab: Lab) -> None:
    spec = _spec(lab)
    first = build_dossier(lab.registry, spec)
    second = build_dossier(lab.registry, spec)
    assert first.dossier_id == second.dossier_id
    assert second.warnings


def test_build_dossier_produces_a_new_dossier_when_evidence_changes(lab: Lab) -> None:
    spec = _spec(lab)
    before = build_dossier(lab.registry, spec)
    lab.executor.execute(lab.experiment.id, procedures.ok, seed=0)
    after = build_dossier(lab.registry, spec)
    assert before.dossier_id != after.dossier_id
    assert lab.registry.exists(EvidenceDossier, before.dossier_id)  # untouched, not gone


def test_dossier_reuses_report_findings_rather_than_duplicating_claims(lab: Lab) -> None:
    lab.executor.execute(lab.experiment.id, procedures.ok, seed=0)
    template_id = ensure_template(lab.registry, builtin_template(ReportType.INVESTIGATION_DOSSIER))
    rspec = ReportSpec(ReportType.INVESTIGATION_DOSSIER, lab.investigation.id, template_id)
    rresult = generate_report(lab.registry, rspec)

    result = build_dossier(lab.registry, _spec(lab))
    dossier = lab.registry.get(EvidenceDossier, result.dossier_id)
    assert dossier.report_ids == (rresult.report_id,)
    findings = lab.registry.find(DossierFinding, dossier_id=dossier.id)
    reused = [f for f in findings if f.report_finding_id is not None]
    assert reused
    for f in reused:
        assert lab.registry.exists(
            type(lab.registry.find(DossierFinding, dossier_id=dossier.id)[0]), f.id
        )
        assert f.status is SufficiencyStatus.SUPPORTED


def test_missing_evidence_categories_are_recorded_as_explicit_gaps(lab: Lab) -> None:
    result = build_dossier(lab.registry, _spec(lab))
    findings = lab.registry.find(DossierFinding, dossier_id=result.dossier_id)
    gaps = [f for f in findings if f.status is SufficiencyStatus.MISSING_EXPECTED_EVIDENCE]
    assert gaps  # nothing beyond the experiment exists yet, so every evidence category is a gap
    assert any("fault_experiment" in f.statement for f in gaps)


def test_a_run_with_no_provenance_is_flagged_but_this_run_has_provenance(lab: Lab) -> None:
    lab.executor.execute(lab.experiment.id, procedures.ok, seed=0)
    result = build_dossier(lab.registry, _spec(lab))
    findings = lab.registry.find(DossierFinding, dossier_id=result.dossier_id)
    assert not any(
        f.status is SufficiencyStatus.PROVENANCE_GAP for f in findings
    )  # the executor always records provenance


def test_a_run_with_no_reproduction_attempts_gets_a_reproducibility_gap(lab: Lab) -> None:
    lab.executor.execute(lab.experiment.id, procedures.ok, seed=0)
    result = build_dossier(lab.registry, _spec(lab))
    findings = lab.registry.find(DossierFinding, dossier_id=result.dossier_id)
    assert any(f.status is SufficiencyStatus.REPRODUCIBILITY_GAP for f in findings)


def test_conflicting_statistical_analyses_are_both_preserved_and_cross_linked(lab: Lab) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    a = StatisticalAnalysis(
        "COMPARE",
        "sha256:" + "11" * 32,
        "1.0.0",
        "OBSERVED",
        {},
        {},
        {"value": 1},
        "sha256:" + "22" * 32,
        now,
    )
    b = StatisticalAnalysis(
        "COMPARE",
        "sha256:" + "33" * 32,
        "1.0.0",
        "OBSERVED",
        {},
        {},
        {"value": 2},
        "sha256:" + "44" * 32,
        now,
    )
    lab.registry.add(a)
    lab.registry.add(b)
    spec = _spec(
        lab, explicit_evidence=(("statistical_analysis", a.id), ("statistical_analysis", b.id))
    )
    result = build_dossier(lab.registry, spec)
    findings = lab.registry.find(DossierFinding, dossier_id=result.dossier_id)
    conflicts = [f for f in findings if f.status is SufficiencyStatus.CONFLICTING_EVIDENCE]
    assert len(conflicts) == 2
    ids = {f.id for f in conflicts}
    for f in conflicts:
        assert set(f.conflicting_with) == ids - {
            f.id
        }  # each preserves and cross-references the other


def test_build_dossier_rejects_an_unresolvable_explicit_evidence_reference(lab: Lab) -> None:
    spec = _spec(lab, explicit_evidence=(("fault_experiment", "fxp_" + "9" * 32),))
    try:
        build_dossier(lab.registry, spec)
        raise AssertionError("expected NotFoundError")
    except NotFoundError:
        pass


def test_snapshot_is_immutable_and_idempotent(lab: Lab) -> None:
    result = build_dossier(lab.registry, _spec(lab))
    snap_a = build_snapshot(lab.registry, result.dossier_id)
    snap_b = build_snapshot(lab.registry, result.dossier_id)
    assert snap_a == snap_b
    snapshot = lab.registry.get(DossierSnapshot, snap_a)
    assert snapshot.item_ids == tuple(
        sorted(i.id for i in lab.registry.find(DossierItem, dossier_id=result.dossier_id))
    )


def test_a_later_experiment_does_not_alter_an_existing_snapshot(lab: Lab) -> None:
    result = build_dossier(lab.registry, _spec(lab))
    snapshot_id = build_snapshot(lab.registry, result.dossier_id)
    before = lab.registry.get(DossierSnapshot, snapshot_id)

    lab.executor.execute(lab.experiment.id, procedures.ok, seed=0)
    new_result = build_dossier(lab.registry, _spec(lab))
    assert new_result.dossier_id != result.dossier_id  # new evidence produced a new dossier

    after = lab.registry.get(DossierSnapshot, snapshot_id)
    assert before == after  # the original snapshot is byte-identical, untouched

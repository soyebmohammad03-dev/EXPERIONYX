"""Domain-model tests for the evidence dossier layer: identity, immutability and sorted-ID
invariants that make snapshot identity deterministic (see docs/dossier.md)."""

from datetime import UTC, datetime

import pytest

from experionyx.dossier.entities import (
    DossierFinding,
    DossierItem,
    DossierSnapshot,
    EvidenceDossier,
)
from experionyx.dossier.spec import DossierSpec
from experionyx.dossier.taxonomy import DossierItemKind, DossierSourceKind, SufficiencyStatus
from experionyx.errors import ValidationError
from experionyx.reporting.entities import EvidenceReference

T0 = datetime(2026, 1, 1, tzinfo=UTC)
INV = "inv_" + "1" * 32
DSR = "dsr_" + "1" * 32


def _dossier(digest: str = "sha256:" + "aa" * 32) -> EvidenceDossier:
    return EvidenceDossier(
        spec_id="dsp_" + "1" * 32,
        investigation_id=INV,
        research_question="does it reproduce?",
        source_kind=DossierSourceKind.INVESTIGATION,
        source_id=INV,
        report_ids=(),
        reproduction_attempt_ids=(),
        limitations=(),
        evidence_gaps=(),
        source_evidence_digest=digest,
        engine_version="1.0.0",
        created_at=T0,
    )


def test_dossier_identity_is_spec_and_evidence_digest() -> None:
    a = _dossier()
    b = _dossier()
    c = _dossier("sha256:" + "bb" * 32)
    assert a.id == b.id
    assert a.id != c.id


def test_dossier_item_identity_is_scoped_by_dossier_and_source() -> None:
    item = DossierItem(
        DSR,
        DossierItemKind.RUN,
        "run",
        "run_" + "2" * 32,
        "collected",
        "sha256:" + "cc" * 32,
        None,
        T0,
    )
    again = DossierItem.from_dict(item.to_dict())
    assert again == item and again.id == item.id


def test_dossier_finding_reuses_a_report_finding_reference_without_duplicating_it() -> None:
    ref = EvidenceReference("run", "run_" + "2" * 32, "a run")
    finding = DossierFinding(
        DSR,
        "1 run record(s) were found",
        SufficiencyStatus.SUPPORTED,
        (ref,),
        "rfd_" + "1" * 32,
        (),
        "reused",
        T0,
    )
    again = DossierFinding.from_dict(finding.to_dict())
    assert again == finding
    assert again.report_finding_id == "rfd_" + "1" * 32


def test_snapshot_requires_sorted_item_and_finding_ids() -> None:
    with pytest.raises(ValidationError, match="sorted"):
        DossierSnapshot(DSR, "sha256:" + "aa" * 32, ("dsi_2" * 8, "dsi_1" * 8), (), T0)


def test_snapshot_identity_is_dossier_digest_and_exact_id_sets() -> None:
    a = DossierSnapshot(DSR, "sha256:" + "aa" * 32, ("dsi_" + "1" * 32,), (), T0)
    b = DossierSnapshot(DSR, "sha256:" + "aa" * 32, ("dsi_" + "1" * 32,), (), T0)
    c = DossierSnapshot(DSR, "sha256:" + "aa" * 32, ("dsi_" + "2" * 32,), (), T0)
    assert a.id == b.id
    assert a.id != c.id  # a different evidence set is a different snapshot, never a mutation


def test_dossier_spec_identity_is_deterministic_and_source_sensitive() -> None:
    a = DossierSpec(INV, "q", DossierSourceKind.INVESTIGATION, INV)
    b = DossierSpec(INV, "q", DossierSourceKind.INVESTIGATION, INV)
    c = DossierSpec(INV, "q", DossierSourceKind.RUN, "run_" + "3" * 32)
    assert a.spec_id == b.spec_id
    assert a.spec_id != c.spec_id
    assert DossierSpec.from_dict(a.to_dict()).spec_id == a.spec_id


def test_dossier_spec_rejects_duplicate_explicit_evidence() -> None:
    with pytest.raises(ValidationError):
        DossierSpec(
            INV,
            "q",
            DossierSourceKind.INVESTIGATION,
            INV,
            explicit_evidence=(("run", "run_" + "4" * 32), ("run", "run_" + "4" * 32)),
        )

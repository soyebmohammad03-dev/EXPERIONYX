"""Registry entities of the evidence dossier layer (see docs/dossier.md). A dossier is
structured, persisted research evidence -- not a document. It reuses the reporting layer's
`EvidenceReference` value object rather than inventing a second one, and reuses an existing
`ReportFinding` by reference wherever a claim originated from a generated report."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Self

import experionyx.validation as v
from experionyx.domain import Entity, Investigation, open_payload
from experionyx.dossier.taxonomy import DossierItemKind, DossierSourceKind, SufficiencyStatus
from experionyx.errors import ValidationError
from experionyx.reporting.entities import EvidenceReference


@dataclass(frozen=True)
class EvidenceDossier(Entity):
    """One structured research package: a scope, the evidence assembled for it, and the reports
    it draws on. Identity is (spec, evidence digest) -- the same pattern as `Report`: regenerating
    against unchanged evidence yields the identical dossier; changed evidence yields a new,
    distinct dossier rather than mutating this one."""

    KIND: ClassVar[str] = "evidence_dossier"
    PREFIX: ClassVar[str] = "dsr"
    spec_id: str
    investigation_id: str
    research_question: str
    source_kind: DossierSourceKind
    source_id: str
    report_ids: tuple[str, ...]
    reproduction_attempt_ids: tuple[str, ...]
    limitations: tuple[str, ...]
    evidence_gaps: tuple[str, ...]
    source_evidence_digest: str
    engine_version: str
    created_at: datetime

    def __post_init__(self) -> None:
        v.text("spec_id", self.spec_id)
        v.ref("investigation_id", self.investigation_id, Investigation.PREFIX)
        v.text("research_question", self.research_question)
        v.member("source_kind", self.source_kind, DossierSourceKind)
        v.text("source_id", self.source_id)
        for name in ("report_ids", "reproduction_attempt_ids", "limitations", "evidence_gaps"):
            val = getattr(self, name)
            if not isinstance(val, tuple) or not all(isinstance(x, str) for x in val):
                raise ValidationError(f"{name} must be a tuple of strings")
        for i, rid in enumerate(self.report_ids):
            v.ref(f"report_ids[{i}]", rid, "rpt")
        for i, aid in enumerate(self.reproduction_attempt_ids):
            v.ref(f"reproduction_attempt_ids[{i}]", aid, "rpa")
        v.digest("source_evidence_digest", self.source_evidence_digest)
        v.text("engine_version", self.engine_version)
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {"spec_id": self.spec_id, "source_evidence_digest": self.source_evidence_digest}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d,
            cls.KIND,
            (
                "spec_id",
                "investigation_id",
                "research_question",
                "source_kind",
                "source_id",
                "report_ids",
                "reproduction_attempt_ids",
                "limitations",
                "evidence_gaps",
                "source_evidence_digest",
                "engine_version",
                "created_at",
            ),
        )
        lists: dict[str, list[object]] = {}
        for name in ("report_ids", "reproduction_attempt_ids", "limitations", "evidence_gaps"):
            raw = v.get_raw(d, name)
            if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
                raise ValidationError(f"{name} must be a list of strings")
            lists[name] = raw
        return cls(
            v.get_str(d, "spec_id"),
            v.get_str(d, "investigation_id"),
            v.get_str(d, "research_question"),
            v.get_enum(d, "source_kind", DossierSourceKind),
            v.get_str(d, "source_id"),
            tuple(lists["report_ids"]),  # type: ignore[arg-type]
            tuple(lists["reproduction_attempt_ids"]),  # type: ignore[arg-type]
            tuple(lists["limitations"]),  # type: ignore[arg-type]
            tuple(lists["evidence_gaps"]),  # type: ignore[arg-type]
            v.get_str(d, "source_evidence_digest"),
            v.get_str(d, "engine_version"),
            v.get_time(d, "created_at"),
        )


@dataclass(frozen=True)
class DossierItem(Entity):
    """One piece of evidence included in a dossier, with its lineage. `captured_content_hash` is
    the evidence entity's own `content_hash()` at the moment it was included, so a later change to
    the underlying record is detectable as stale evidence without re-deriving the whole dossier."""

    KIND: ClassVar[str] = "dossier_item"
    PREFIX: ClassVar[str] = "dsi"
    dossier_id: str
    item_kind: DossierItemKind
    source_kind: str
    source_id: str
    reason: str
    captured_content_hash: str
    provenance_fingerprint: str | None
    created_at: datetime

    def __post_init__(self) -> None:
        v.ref("dossier_id", self.dossier_id, EvidenceDossier.PREFIX)
        v.member("item_kind", self.item_kind, DossierItemKind)
        v.text("source_kind", self.source_kind)
        v.text("source_id", self.source_id)
        v.text("reason", self.reason)
        v.digest("captured_content_hash", self.captured_content_hash)
        if self.provenance_fingerprint is not None:
            v.digest("provenance_fingerprint", self.provenance_fingerprint)
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {
            "dossier_id": self.dossier_id,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d,
            cls.KIND,
            (
                "dossier_id",
                "item_kind",
                "source_kind",
                "source_id",
                "reason",
                "captured_content_hash",
                "provenance_fingerprint",
                "created_at",
            ),
        )
        fp = v.get_raw(d, "provenance_fingerprint")
        return cls(
            v.get_str(d, "dossier_id"),
            v.get_enum(d, "item_kind", DossierItemKind),
            v.get_str(d, "source_kind"),
            v.get_str(d, "source_id"),
            v.get_str(d, "reason"),
            v.get_str(d, "captured_content_hash"),
            None if fp is None else str(fp),
            v.get_time(d, "created_at"),
        )


@dataclass(frozen=True)
class DossierFinding(Entity):
    """One claim examined by a dossier's sufficiency analysis. Reuses an existing `ReportFinding`
    by reference (`report_finding_id`) rather than duplicating its statement whenever the claim
    came from an included report; structural findings (a gap, a conflict, a provenance hole) have
    no such backing finding and stand on their own evidence references."""

    KIND: ClassVar[str] = "dossier_finding"
    PREFIX: ClassVar[str] = "dsf"
    dossier_id: str
    statement: str
    status: SufficiencyStatus
    evidence: tuple[EvidenceReference, ...]
    report_finding_id: str | None
    conflicting_with: tuple[str, ...]
    note: str
    created_at: datetime

    def __post_init__(self) -> None:
        v.ref("dossier_id", self.dossier_id, EvidenceDossier.PREFIX)
        v.text("statement", self.statement)
        v.member("status", self.status, SufficiencyStatus)
        if not isinstance(self.evidence, tuple) or not all(
            isinstance(x, EvidenceReference) for x in self.evidence
        ):
            raise ValidationError("evidence must be a tuple of EvidenceReference")
        if self.report_finding_id is not None:
            v.ref("report_finding_id", self.report_finding_id, "rfd")
        if not isinstance(self.conflicting_with, tuple) or not all(
            isinstance(x, str) for x in self.conflicting_with
        ):
            raise ValidationError("conflicting_with must be a tuple of strings")
        for i, x in enumerate(self.conflicting_with):
            v.ref(f"conflicting_with[{i}]", x, "dsf")
        v.text("note", self.note)
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {"dossier_id": self.dossier_id, "statement": self.statement}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d,
            cls.KIND,
            (
                "dossier_id",
                "statement",
                "status",
                "evidence",
                "report_finding_id",
                "conflicting_with",
                "note",
                "created_at",
            ),
        )
        evidence = v.get_raw(d, "evidence")
        if not isinstance(evidence, list):
            raise ValidationError("evidence must be a list")
        conflicting = v.get_raw(d, "conflicting_with")
        if not isinstance(conflicting, list) or not all(isinstance(x, str) for x in conflicting):
            raise ValidationError("conflicting_with must be a list of strings")
        rfid = v.get_raw(d, "report_finding_id")
        return cls(
            v.get_str(d, "dossier_id"),
            v.get_str(d, "statement"),
            v.get_enum(d, "status", SufficiencyStatus),
            tuple(EvidenceReference.from_dict(e) for e in evidence),
            None if rfid is None else str(rfid),
            tuple(conflicting),
            v.get_str(d, "note"),
            v.get_time(d, "created_at"),
        )


@dataclass(frozen=True)
class DossierSnapshot(Entity):
    """An immutable, independently addressable freeze of one dossier's exact evidence state.
    Identity is derived from the dossier and its evidence digest plus the exact item/finding ID
    sets captured, so a later experiment changing the investigation's evidence produces a NEW
    dossier (and could be snapshotted again) without altering this snapshot at all."""

    KIND: ClassVar[str] = "dossier_snapshot"
    PREFIX: ClassVar[str] = "dsn"
    dossier_id: str
    source_evidence_digest: str
    item_ids: tuple[str, ...]
    finding_ids: tuple[str, ...]
    created_at: datetime

    def __post_init__(self) -> None:
        v.ref("dossier_id", self.dossier_id, EvidenceDossier.PREFIX)
        v.digest("source_evidence_digest", self.source_evidence_digest)
        for name in ("item_ids", "finding_ids"):
            val = getattr(self, name)
            if not isinstance(val, tuple) or not all(isinstance(x, str) for x in val):
                raise ValidationError(f"{name} must be a tuple of strings")
        if list(self.item_ids) != sorted(self.item_ids):
            raise ValidationError("item_ids must be sorted for a deterministic snapshot identity")
        if list(self.finding_ids) != sorted(self.finding_ids):
            raise ValidationError(
                "finding_ids must be sorted for a deterministic snapshot identity"
            )
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {
            "dossier_id": self.dossier_id,
            "source_evidence_digest": self.source_evidence_digest,
            "item_ids": self.item_ids,
            "finding_ids": self.finding_ids,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d,
            cls.KIND,
            ("dossier_id", "source_evidence_digest", "item_ids", "finding_ids", "created_at"),
        )
        items = v.get_raw(d, "item_ids")
        findings = v.get_raw(d, "finding_ids")
        if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
            raise ValidationError("item_ids must be a list of strings")
        if not isinstance(findings, list) or not all(isinstance(x, str) for x in findings):
            raise ValidationError("finding_ids must be a list of strings")
        return cls(
            v.get_str(d, "dossier_id"),
            v.get_str(d, "source_evidence_digest"),
            tuple(items),
            tuple(findings),
            v.get_time(d, "created_at"),
        )


__all__ = ["DossierFinding", "DossierItem", "DossierSnapshot", "EvidenceDossier"]

"""Registry entities and embedded value objects of the research reporting layer (see
docs/reporting.md). A `Report` never mutates: regenerating against changed evidence produces a
new, distinct `Report` (its identity includes a digest of the evidence it was built from), and
old reports stay resolvable and reproducible."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Self

import experionyx.validation as v
from experionyx.domain import ClaimStatus, Entity, Investigation, check_keys, open_payload
from experionyx.errors import ValidationError
from experionyx.reporting.taxonomy import ReportStatus, ReportType, SectionKind


@dataclass(frozen=True)
class EvidenceReference:
    """A pointer to the concrete registry record backing a claim or a section, by kind and ID
    only -- never a copy of its content. `source_id` is walked by the graph builder like any
    other entity-shaped field, so a finding's evidence resolves through the SAME evidence/failure
    graph as everything else (see docs/graph.md); no second graph is created."""

    source_kind: (
        str  # e.g. "run", "fault_experiment", "reliability_profile", "statistical_analysis"
    )
    source_id: str
    description: str

    def __post_init__(self) -> None:
        v.text("source_kind", self.source_kind)
        v.text("source_id", self.source_id)
        v.text("description", self.description)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        check_keys(d, ("source_kind", "source_id", "description"))
        return cls(
            v.get_str(d, "source_kind"), v.get_str(d, "source_id"), v.get_str(d, "description")
        )


def _evidence_tuple(name: str, items: object) -> tuple[EvidenceReference, ...]:
    if not isinstance(items, tuple) or not all(isinstance(x, EvidenceReference) for x in items):
        raise ValidationError(f"{name} must be a tuple of EvidenceReference")
    return items


@dataclass(frozen=True)
class ReportSection:
    """One structural section of a report. `available=False` records, rather than hides, that
    the template called for this section but no evidence existed for it (see
    `Report.evidence_gaps` for the matching gap entries)."""

    kind: SectionKind
    title: str
    content: str
    evidence: tuple[EvidenceReference, ...]
    available: bool

    def __post_init__(self) -> None:
        v.member("kind", self.kind, SectionKind)
        v.text("title", self.title)
        if not isinstance(self.content, str):
            raise ValidationError("content must be a string")
        object.__setattr__(self, "evidence", _evidence_tuple("evidence", self.evidence))
        if not isinstance(self.available, bool):
            raise ValidationError("available must be a boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "title": self.title,
            "content": self.content,
            "evidence": [e.to_dict() for e in self.evidence],
            "available": self.available,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        check_keys(d, ("kind", "title", "content", "evidence", "available"))
        evidence = v.get_raw(d, "evidence")
        if not isinstance(evidence, list):
            raise ValidationError("evidence must be a list")
        available = v.get_raw(d, "available")
        if not isinstance(available, bool):
            raise ValidationError("available must be a boolean")
        return cls(
            v.get_enum(d, "kind", SectionKind),
            v.get_str(d, "title"),
            v.get_str(d, "content"),
            tuple(EvidenceReference.from_dict(x) for x in evidence),
            available,
        )


@dataclass(frozen=True)
class ReportTemplate(Entity):
    """A versioned, content-addressed section layout for one report type. Its ID is derived from
    (report_type, name, version, section order): editing the section list is a NEW template ID,
    so a report generated against an old template stays reproducible against it."""

    KIND: ClassVar[str] = "report_template"
    PREFIX: ClassVar[str] = "rtp"
    name: str
    version: str
    report_type: ReportType
    section_kinds: tuple[SectionKind, ...]
    created_at: datetime

    def __post_init__(self) -> None:
        v.text("name", self.name)
        v.version("version", self.version)
        v.member("report_type", self.report_type, ReportType)
        if not self.section_kinds:
            raise ValidationError("section_kinds must not be empty")
        for i, k in enumerate(self.section_kinds):
            v.member(f"section_kinds[{i}]", k, SectionKind)
        if len(set(self.section_kinds)) != len(self.section_kinds):
            raise ValidationError("section_kinds must not repeat")
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "report_type": self.report_type,
            "section_kinds": self.section_kinds,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(d, cls.KIND, ("name", "version", "report_type", "section_kinds", "created_at"))
        kinds = v.get_raw(d, "section_kinds")
        if not isinstance(kinds, list):
            raise ValidationError("section_kinds must be a list")
        return cls(
            v.get_str(d, "name"),
            v.get_str(d, "version"),
            v.get_enum(d, "report_type", ReportType),
            tuple(SectionKind(str(k)) for k in kinds),
            v.get_time(d, "created_at"),
        )


@dataclass(frozen=True)
class Report(Entity):
    """One generated research report. Identity is (spec, evidence digest): the SAME spec run
    again against UNCHANGED persisted evidence yields the identical report; a changed evidence
    state yields a new, distinct report rather than mutating this one."""

    KIND: ClassVar[str] = "report"
    PREFIX: ClassVar[str] = "rpt"
    spec_id: str
    report_type: ReportType
    investigation_id: str
    template_id: str
    title: str
    status: ReportStatus
    sections: tuple[ReportSection, ...]
    limitations: tuple[str, ...]
    evidence_gaps: tuple[str, ...]
    reproducibility: Mapping[str, object]
    source_evidence_digest: str
    engine_version: str
    generated_at: datetime

    def __post_init__(self) -> None:
        v.text("spec_id", self.spec_id)
        v.member("report_type", self.report_type, ReportType)
        v.ref("investigation_id", self.investigation_id, Investigation.PREFIX)
        v.ref("template_id", self.template_id, ReportTemplate.PREFIX)
        v.text("title", self.title)
        v.member("status", self.status, ReportStatus)
        if not isinstance(self.sections, tuple) or not all(
            isinstance(s, ReportSection) for s in self.sections
        ):
            raise ValidationError("sections must be a tuple of ReportSection")
        for name in ("limitations", "evidence_gaps"):
            val = getattr(self, name)
            if not isinstance(val, tuple) or not all(isinstance(x, str) for x in val):
                raise ValidationError(f"{name} must be a tuple of strings")
        object.__setattr__(
            self, "reproducibility", v.freeze_mapping("reproducibility", self.reproducibility)
        )
        v.digest("source_evidence_digest", self.source_evidence_digest)
        v.text("engine_version", self.engine_version)
        v.timestamp("generated_at", self.generated_at)

    def _identity(self) -> Mapping[str, object]:
        return {"spec_id": self.spec_id, "source_evidence_digest": self.source_evidence_digest}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d,
            cls.KIND,
            (
                "spec_id",
                "report_type",
                "investigation_id",
                "template_id",
                "title",
                "status",
                "sections",
                "limitations",
                "evidence_gaps",
                "reproducibility",
                "source_evidence_digest",
                "engine_version",
                "generated_at",
            ),
        )
        sections = v.get_raw(d, "sections")
        if not isinstance(sections, list):
            raise ValidationError("sections must be a list")
        limitations = v.get_raw(d, "limitations")
        gaps = v.get_raw(d, "evidence_gaps")
        if not isinstance(limitations, list) or not all(isinstance(x, str) for x in limitations):
            raise ValidationError("limitations must be a list of strings")
        if not isinstance(gaps, list) or not all(isinstance(x, str) for x in gaps):
            raise ValidationError("evidence_gaps must be a list of strings")
        return cls(
            v.get_str(d, "spec_id"),
            v.get_enum(d, "report_type", ReportType),
            v.get_str(d, "investigation_id"),
            v.get_str(d, "template_id"),
            v.get_str(d, "title"),
            v.get_enum(d, "status", ReportStatus),
            tuple(ReportSection.from_dict(s) for s in sections),
            tuple(limitations),
            tuple(gaps),
            v.get_mapping(d, "reproducibility"),
            v.get_str(d, "source_evidence_digest"),
            v.get_str(d, "engine_version"),
            v.get_time(d, "generated_at"),
        )


@dataclass(frozen=True)
class ReportFinding(Entity):
    """One claim made within a report, together with the evidence it rests on. `status` is
    `ClaimStatus.SUPPORTED` only when `evidence` is non-empty -- a report can never carry a
    silently unsupported SUPPORTED claim (see docs/reporting.md)."""

    KIND: ClassVar[str] = "report_finding"
    PREFIX: ClassVar[str] = "rfd"
    report_id: str
    statement: str
    status: ClaimStatus
    evidence: tuple[EvidenceReference, ...]
    statistical_analysis_id: str | None
    limitations: tuple[str, ...]
    created_at: datetime

    def __post_init__(self) -> None:
        v.ref("report_id", self.report_id, Report.PREFIX)
        v.text("statement", self.statement)
        v.member("status", self.status, ClaimStatus)
        object.__setattr__(self, "evidence", _evidence_tuple("evidence", self.evidence))
        if self.status is ClaimStatus.SUPPORTED and not self.evidence:
            raise ValidationError("a SUPPORTED finding must cite at least one evidence reference")
        if self.statistical_analysis_id is not None:
            v.ref("statistical_analysis_id", self.statistical_analysis_id, "sta")
        if not isinstance(self.limitations, tuple) or not all(
            isinstance(x, str) for x in self.limitations
        ):
            raise ValidationError("limitations must be a tuple of strings")
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {"report_id": self.report_id, "statement": self.statement}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d,
            cls.KIND,
            (
                "report_id",
                "statement",
                "status",
                "evidence",
                "statistical_analysis_id",
                "limitations",
                "created_at",
            ),
        )
        evidence = v.get_raw(d, "evidence")
        if not isinstance(evidence, list):
            raise ValidationError("evidence must be a list")
        limitations = v.get_raw(d, "limitations")
        if not isinstance(limitations, list) or not all(isinstance(x, str) for x in limitations):
            raise ValidationError("limitations must be a list of strings")
        sa_id = v.get_raw(d, "statistical_analysis_id")
        return cls(
            v.get_str(d, "report_id"),
            v.get_str(d, "statement"),
            v.get_enum(d, "status", ClaimStatus),
            tuple(EvidenceReference.from_dict(e) for e in evidence),
            None if sa_id is None else str(sa_id),
            tuple(limitations),
            v.get_time(d, "created_at"),
        )


__all__ = ["EvidenceReference", "Report", "ReportFinding", "ReportSection", "ReportTemplate"]

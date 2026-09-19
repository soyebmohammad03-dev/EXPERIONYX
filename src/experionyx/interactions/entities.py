"""Registry entities of interaction analysis: the analysis (with a lifecycle), its per-measure
effects, and append-only evidence. Identity is content-addressed: the same investigation and
design spec is the same logical interaction (re-registering it is a duplicate, not a new record)."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from typing import ClassVar, Self

import experionyx.validation as v
from experionyx.domain import Entity, Investigation, Run, open_payload
from experionyx.errors import ValidationError
from experionyx.interactions.taxonomy import (
    INTERACTION_TRANSITIONS,
    EffectStatus,
    EvidenceKind,
    InteractionClass,
    InteractionStatus,
    Level,
)


@dataclass(frozen=True)
class InteractionAnalysis(Entity):
    KIND: ClassVar[str] = "interaction_analysis"
    PREFIX: ClassVar[str] = "ian"
    investigation_id: str
    run_id: str  # the Run that performed (and can replay) the analysis
    spec_id: str  # isp_<hash> of the design: cells (run IDs) + configuration
    spec: Mapping[str, object]
    structural_key: (
        str  # hash of the run-independent structure (faults without seeds, model, dataset, ...)
    )
    fault_a: str  # hash of fault A's definition without seed
    fault_b: str
    model_fingerprint: str
    dataset_fingerprint: str
    primary_metric: str
    primary_class: InteractionClass
    provenance_fingerprint: str
    summary: Mapping[str, object]
    created_at: datetime
    status: InteractionStatus = InteractionStatus.DISCOVERED

    def __post_init__(self) -> None:
        v.ref("investigation_id", self.investigation_id, Investigation.PREFIX)
        v.ref("run_id", self.run_id, Run.PREFIX)
        v.text("spec_id", self.spec_id)
        object.__setattr__(self, "spec", v.freeze_mapping("spec", self.spec))
        for name in (
            "structural_key",
            "fault_a",
            "fault_b",
            "model_fingerprint",
            "dataset_fingerprint",
            "provenance_fingerprint",
        ):
            v.digest(name, getattr(self, name))
        v.text("primary_metric", self.primary_metric)
        v.member("primary_class", self.primary_class, InteractionClass)
        object.__setattr__(self, "summary", v.freeze_mapping("summary", self.summary))
        v.timestamp("created_at", self.created_at)
        v.member("status", self.status, InteractionStatus)

    def _identity(self) -> Mapping[str, object]:
        return {"investigation_id": self.investigation_id, "spec_id": self.spec_id}

    def with_status(self, status: InteractionStatus) -> Self:
        if status not in INTERACTION_TRANSITIONS[self.status]:
            raise ValidationError(f"illegal interaction transition {self.status} -> {status}")
        return replace(self, status=status)

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        names = (
            "investigation_id",
            "run_id",
            "spec_id",
            "spec",
            "structural_key",
            "fault_a",
            "fault_b",
            "model_fingerprint",
            "dataset_fingerprint",
            "primary_metric",
            "primary_class",
            "provenance_fingerprint",
            "summary",
            "created_at",
            "status",
        )
        open_payload(d, cls.KIND, names)
        return cls(
            v.get_str(d, "investigation_id"), v.get_str(d, "run_id"), v.get_str(d, "spec_id"), v.get_mapping(d, "spec"),
            v.get_str(d, "structural_key"), v.get_str(d, "fault_a"), v.get_str(d, "fault_b"),
            v.get_str(d, "model_fingerprint"), v.get_str(d, "dataset_fingerprint"), v.get_str(d, "primary_metric"),
            v.get_enum(d, "primary_class", InteractionClass), v.get_str(d, "provenance_fingerprint"),
            v.get_mapping(d, "summary"), v.get_time(d, "created_at"), v.get_enum(d, "status", InteractionStatus),
        )  # fmt: skip


@dataclass(frozen=True)
class InteractionEffect(Entity):
    """One measure's effect record (observed trial values, derived contrast, bootstrap and the
    interpreted label, kept in separate sections of `record`)."""

    KIND: ClassVar[str] = "interaction_effect"
    PREFIX: ClassVar[str] = "ief"
    analysis_id: str
    level: Level
    measure: str
    effect_status: EffectStatus
    interaction_class: InteractionClass
    record: Mapping[str, object]
    created_at: datetime

    def __post_init__(self) -> None:
        v.ref("analysis_id", self.analysis_id, InteractionAnalysis.PREFIX)
        v.member("level", self.level, Level)
        v.text("measure", self.measure)
        v.member("effect_status", self.effect_status, EffectStatus)
        v.member("interaction_class", self.interaction_class, InteractionClass)
        object.__setattr__(self, "record", v.freeze_mapping("record", self.record))
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {"analysis_id": self.analysis_id, "level": self.level, "measure": self.measure}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d,
            cls.KIND,
            (
                "analysis_id",
                "level",
                "measure",
                "effect_status",
                "interaction_class",
                "record",
                "created_at",
            ),
        )
        return cls(
            v.get_str(d, "analysis_id"),
            v.get_enum(d, "level", Level),
            v.get_str(d, "measure"),
            v.get_enum(d, "effect_status", EffectStatus),
            v.get_enum(d, "interaction_class", InteractionClass),
            v.get_mapping(d, "record"),
            v.get_time(d, "created_at"),
        )


@dataclass(frozen=True)
class InteractionEvidence(Entity):
    KIND: ClassVar[str] = "interaction_evidence"
    PREFIX: ClassVar[str] = "iev"
    analysis_id: str
    evidence_kind: EvidenceKind
    ref_id: str | None
    summary: str
    detail: Mapping[str, object]
    created_at: datetime

    def __post_init__(self) -> None:
        v.ref("analysis_id", self.analysis_id, InteractionAnalysis.PREFIX)
        v.member("evidence_kind", self.evidence_kind, EvidenceKind)
        if self.ref_id is not None:
            v.text("ref_id", self.ref_id)
        v.text("summary", self.summary)
        object.__setattr__(self, "detail", v.freeze_mapping("detail", self.detail))
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {
            "analysis_id": self.analysis_id,
            "evidence_kind": self.evidence_kind,
            "ref_id": self.ref_id,
            "summary": self.summary,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d,
            cls.KIND,
            ("analysis_id", "evidence_kind", "ref_id", "summary", "detail", "created_at"),
        )
        return cls(
            v.get_str(d, "analysis_id"),
            v.get_enum(d, "evidence_kind", EvidenceKind),
            v.get_opt_str(d, "ref_id"),
            v.get_str(d, "summary"),
            v.get_mapping(d, "detail"),
            v.get_time(d, "created_at"),
        )

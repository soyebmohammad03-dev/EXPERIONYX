"""Registry entities: the logical slice DEFINITION (`sls_`, identity = its normalized condition, so
equivalent spellings are one record) and one ANALYSIS of slices over a baseline run (`san_`). The
analysis stores headline metadata and references; membership and results live in digest-verified
run artifacts (slice/spec.json, membership.json, results.json, summary.json)."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Self

import experionyx.validation as v
from experionyx.domain import Entity, Investigation, Run, open_payload
from experionyx.errors import ValidationError
from experionyx.slices.spec import SLICE_SCHEMA, Condition, SliceSpec


@dataclass(frozen=True)
class Slice(Entity):
    KIND: ClassVar[str] = "slice"
    PREFIX: ClassVar[str] = "sls"
    name: str  # the first name it was registered under; a label, not identity
    condition: Mapping[str, object]
    fields: tuple[str, ...]
    description: str  # human-readable rendering of the condition
    slice_schema: int
    created_at: datetime

    def __post_init__(self) -> None:
        v.text("name", self.name)
        object.__setattr__(self, "condition", v.freeze_mapping("condition", self.condition))
        Condition.from_dict(self.condition)  # a stored condition must parse
        v.text("description", self.description)
        v.timestamp("created_at", self.created_at)

    @classmethod
    def of(cls, spec: SliceSpec, now: datetime) -> "Slice":
        return cls(spec.name, spec.condition.to_dict(), spec.fields, spec.human, SLICE_SCHEMA, now)

    def spec(self) -> SliceSpec:
        return SliceSpec(self.name, Condition.from_dict(self.condition))

    def _identity(self) -> Mapping[str, object]:
        return {"slice_schema": self.slice_schema, "condition": self.condition}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d,
            cls.KIND,
            ("name", "condition", "fields", "description", "slice_schema", "created_at"),
        )
        fields = d["fields"]
        if not isinstance(fields, list | tuple):
            raise ValidationError("fields must be a list")
        sc = d["slice_schema"]
        if not isinstance(sc, int):
            raise ValidationError("slice_schema must be an integer")
        return cls(
            v.get_str(d, "name"), v.get_mapping(d, "condition"), tuple(str(f) for f in fields),
            v.get_str(d, "description"), sc, v.get_time(d, "created_at"),
        )  # fmt: skip


@dataclass(frozen=True)
class SliceAnalysis(Entity):
    KIND: ClassVar[str] = "slice_analysis"
    PREFIX: ClassVar[str] = "san"
    investigation_id: str
    run_id: str  # the Run that produced (and can replay) the analysis
    spec_id: str  # ssp_<hash> of the analysis request
    spec: Mapping[str, object]
    baseline_run_id: str
    dataset_fingerprint: str
    provenance_fingerprint: str
    analysis_status: (
        str  # COMPLETE | PARTIAL (some requested evidence was insufficient/unavailable)
    )
    summary: Mapping[str, object]
    created_at: datetime

    def __post_init__(self) -> None:
        v.ref("investigation_id", self.investigation_id, Investigation.PREFIX)
        v.ref("run_id", self.run_id, Run.PREFIX)
        v.text("spec_id", self.spec_id)
        object.__setattr__(self, "spec", v.freeze_mapping("spec", self.spec))
        v.ref("baseline_run_id", self.baseline_run_id, Run.PREFIX)
        v.text("dataset_fingerprint", self.dataset_fingerprint)
        v.digest("provenance_fingerprint", self.provenance_fingerprint)
        if self.analysis_status not in ("COMPLETE", "PARTIAL"):
            raise ValidationError("analysis_status must be COMPLETE or PARTIAL")
        object.__setattr__(self, "summary", v.freeze_mapping("summary", self.summary))
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {"investigation_id": self.investigation_id, "spec_id": self.spec_id}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        names = (
            "investigation_id", "run_id", "spec_id", "spec", "baseline_run_id", "dataset_fingerprint",
            "provenance_fingerprint", "analysis_status", "summary", "created_at",
        )  # fmt: skip
        open_payload(d, cls.KIND, names)
        return cls(
            v.get_str(d, "investigation_id"), v.get_str(d, "run_id"), v.get_str(d, "spec_id"),
            v.get_mapping(d, "spec"), v.get_str(d, "baseline_run_id"), v.get_str(d, "dataset_fingerprint"),
            v.get_str(d, "provenance_fingerprint"), v.get_str(d, "analysis_status"),
            v.get_mapping(d, "summary"), v.get_time(d, "created_at"),
        )  # fmt: skip

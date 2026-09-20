"""Registry entities: a registered temporal WINDOW definition (`twn_`, identity = ordering field,
role, bounds and inclusivity, so equal definitions are one record and the ID equals
`TemporalWindow.window_id`) and one drift ANALYSIS over a baseline run (`dan_`). The analysis stores
headline metadata and references; windows, results and evidence live in digest-verified run
artifacts (drift/spec.json, windows.json, feature_results.json, distribution_results.json,
performance_results.json, summary.json)."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Self

import experionyx.validation as v
from experionyx.domain import Entity, Investigation, Run, open_payload
from experionyx.drift.spec import DRIFT_SCHEMA, Ordering, Role, TemporalWindow
from experionyx.errors import ValidationError


@dataclass(frozen=True)
class DriftWindow(Entity):
    KIND: ClassVar[str] = "temporal_window"
    PREFIX: ClassVar[str] = "twn"
    name: str  # the first label it was registered under; not identity
    ordering_field: str
    role: str
    start: int | float
    end: int | float
    start_inclusive: bool
    end_inclusive: bool
    drift_schema: int
    created_at: datetime

    def __post_init__(self) -> None:
        v.text("name", self.name)
        Ordering(self.ordering_field)
        self.window()  # a stored window must parse and be non-empty
        v.timestamp("created_at", self.created_at)

    @classmethod
    def of(cls, w: TemporalWindow, ordering_field: str, now: datetime) -> "DriftWindow":
        return cls(
            w.name or w.describe(), ordering_field, w.role.value, w.start, w.end,
            w.start_inclusive, w.end_inclusive, DRIFT_SCHEMA, now,
        )  # fmt: skip

    def window(self) -> TemporalWindow:
        if self.role not in {r.value for r in Role}:
            raise ValidationError(
                f"role must be one of {[r.value for r in Role]}, got {self.role!r}"
            )
        return TemporalWindow(
            Role(self.role), self.start, self.end,
            self.start_inclusive, self.end_inclusive, self.name,
        )  # fmt: skip

    def _identity(self) -> Mapping[str, object]:
        return self.window().identity(self.ordering_field)

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        names = ("name", "ordering_field", "role", "start", "end", "start_inclusive", "end_inclusive", "drift_schema", "created_at")  # fmt: skip
        open_payload(d, cls.KIND, names)
        s, e = d["start"], d["end"]
        if isinstance(s, bool) or isinstance(e, bool) or not isinstance(s, int | float) or not isinstance(e, int | float):  # fmt: skip
            raise ValidationError("window bounds must be numbers")
        si, ei, sc = d["start_inclusive"], d["end_inclusive"], d["drift_schema"]
        if not isinstance(si, bool) or not isinstance(ei, bool) or not isinstance(sc, int):
            raise ValidationError("start_inclusive/end_inclusive must be booleans, drift_schema an integer")  # fmt: skip
        return cls(
            v.get_str(d, "name"), v.get_str(d, "ordering_field"), v.get_str(d, "role"),
            s, e, si, ei, sc, v.get_time(d, "created_at"),
        )  # fmt: skip


@dataclass(frozen=True)
class DriftAnalysis(Entity):
    KIND: ClassVar[str] = "drift_analysis"
    PREFIX: ClassVar[str] = "dan"
    investigation_id: str
    run_id: str  # the Run that produced (and can replay) the analysis
    spec_id: str  # dsp_<hash> of the analysis request
    spec: Mapping[str, object]
    baseline_run_id: str
    dataset_fingerprint: str
    provenance_fingerprint: str
    analysis_status: (
        str  # COMPLETE | PARTIAL (some requested evidence was insufficient/unavailable/skipped)
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

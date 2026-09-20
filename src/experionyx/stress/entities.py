"""Registry entities: one stress ANALYSIS (`sxa_`) over a baseline run, and one persisted TRIAL
(`sxt_`) per expanded unit (point x repeat x cell). Measurements, lineage and statistics live in
digest-verified run artifacts (stress/spec.json, plan.json, trials.json, baseline.json,
results.json, analyses.json, summary.json); the records here are the queryable index and are
append-only."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Self

import experionyx.validation as v
from experionyx.adapters.records import RegisteredDataset, RegisteredModel
from experionyx.domain import Entity, Investigation, Run, open_payload
from experionyx.errors import ValidationError

TRIAL_STATUSES = ("COMPLETED", "FAILED", "SKIPPED", "UNSUPPORTED")


@dataclass(frozen=True)
class StressAnalysis(Entity):
    KIND: ClassVar[str] = "stress_analysis"
    PREFIX: ClassVar[str] = "sxa"
    investigation_id: str
    run_id: str  # the Run that produced (and can replay) the analysis
    spec_id: str  # sxr_<hash> of the whole design
    spec: Mapping[str, object]
    model_id: str
    dataset_id: str
    baseline_run_id: str
    provenance_fingerprint: str
    analysis_status: (
        str  # COMPLETE | PARTIAL (a trial failed/was skipped, or evidence was insufficient)
    )
    summary: Mapping[str, object]
    created_at: datetime

    def __post_init__(self) -> None:
        v.ref("investigation_id", self.investigation_id, Investigation.PREFIX)
        v.ref("run_id", self.run_id, Run.PREFIX)
        v.text("spec_id", self.spec_id)
        object.__setattr__(self, "spec", v.freeze_mapping("spec", self.spec))
        v.ref("model_id", self.model_id, RegisteredModel.PREFIX)
        v.ref("dataset_id", self.dataset_id, RegisteredDataset.PREFIX)
        v.ref("baseline_run_id", self.baseline_run_id, Run.PREFIX)
        v.digest("provenance_fingerprint", self.provenance_fingerprint)
        if self.analysis_status not in ("COMPLETE", "PARTIAL"):
            raise ValidationError("analysis_status must be COMPLETE or PARTIAL")
        object.__setattr__(self, "summary", v.freeze_mapping("summary", self.summary))
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {"investigation_id": self.investigation_id, "spec_id": self.spec_id}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        names = ("investigation_id", "run_id", "spec_id", "spec", "model_id", "dataset_id", "baseline_run_id", "provenance_fingerprint", "analysis_status", "summary", "created_at")  # fmt: skip
        open_payload(d, cls.KIND, names)
        return cls(
            v.get_str(d, "investigation_id"), v.get_str(d, "run_id"), v.get_str(d, "spec_id"),
            v.get_mapping(d, "spec"), v.get_str(d, "model_id"), v.get_str(d, "dataset_id"),
            v.get_str(d, "baseline_run_id"), v.get_str(d, "provenance_fingerprint"),
            v.get_str(d, "analysis_status"), v.get_mapping(d, "summary"), v.get_time(d, "created_at"),
        )  # fmt: skip


@dataclass(frozen=True)
class StressTrial(Entity):
    KIND: ClassVar[str] = "stress_trial"
    PREFIX: ClassVar[str] = "sxt"
    analysis_id: str
    unit_key: str  # sxu_<hash>: the deterministic identity of the expanded unit
    point_index: int
    repeat_index: int
    cell: str
    family: str  # the family of the first component (compound: see `stress_ids`)
    stress_ids: tuple[str, ...]
    origin: str  # FAULT_LABORATORY | MODEL | EVALUATION
    seed: int
    status: str  # COMPLETED | FAILED | SKIPPED | UNSUPPORTED
    run_id: str | None  # the trial's own Run (None if it never started)
    reason: str | None
    lineage: Mapping[str, object]  # fault experiment / trial refs, artifact digests
    created_at: datetime

    def __post_init__(self) -> None:
        v.ref("analysis_id", self.analysis_id, StressAnalysis.PREFIX)
        v.text("unit_key", self.unit_key)
        v.non_negative_int("point_index", self.point_index)
        v.non_negative_int("repeat_index", self.repeat_index)
        v.text("family", self.family)
        object.__setattr__(self, "stress_ids", tuple(self.stress_ids))
        if not self.stress_ids:
            raise ValidationError("a trial names at least one stress")
        v.text("origin", self.origin)
        v.non_negative_int("seed", self.seed)
        if self.status not in TRIAL_STATUSES:
            raise ValidationError(f"trial status must be one of {list(TRIAL_STATUSES)}")
        if self.run_id is not None:
            v.ref("run_id", self.run_id, Run.PREFIX)
        if self.reason is not None:
            v.text("reason", self.reason)
        object.__setattr__(self, "lineage", v.freeze_mapping("lineage", self.lineage))
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {"analysis_id": self.analysis_id, "unit_key": self.unit_key}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        names = ("analysis_id", "unit_key", "point_index", "repeat_index", "cell", "family", "stress_ids", "origin", "seed", "status", "run_id", "reason", "lineage", "created_at")  # fmt: skip
        open_payload(d, cls.KIND, names)
        ids = d["stress_ids"]
        if not isinstance(ids, list | tuple):
            raise ValidationError("stress_ids must be a list")
        return cls(
            v.get_str(d, "analysis_id"), v.get_str(d, "unit_key"), v.get_int(d, "point_index"),
            v.get_int(d, "repeat_index"), v.get_str(d, "cell"), v.get_str(d, "family"),
            tuple(str(i) for i in ids), v.get_str(d, "origin"), v.get_int(d, "seed"),
            v.get_str(d, "status"), v.get_opt_str(d, "run_id"), v.get_opt_str(d, "reason"),
            v.get_mapping(d, "lineage"), v.get_time(d, "created_at"),
        )  # fmt: skip

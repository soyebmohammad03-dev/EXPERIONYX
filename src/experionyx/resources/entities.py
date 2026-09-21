"""Registry entities: one resource ANALYSIS (`rsa_`) per measured Run, and one persisted TRIAL (`rst_`)
per warmup or measured pass over the workload. Raw per-batch timings, statistics and the environment
live in digest-verified run artifacts (resources/spec.json, environment.json, trials.json,
observations.json, statistics.json, summary.json); the records here are the queryable index and are
append-only.

An analysis' identity includes its Run: measuring the same specification again is a NEW measurement,
never an overwrite (timing is not reproducible bit for bit). `spec_id` groups repeated measurements."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Self

import experionyx.validation as v
from experionyx.domain import Entity, Investigation, Run, open_payload
from experionyx.errors import ValidationError

PHASES = ("WARMUP", "MEASURED")
TRIAL_STATUSES = ("COMPLETED", "FAILED", "TIMED_OUT")


@dataclass(frozen=True)
class ResourceAnalysis(Entity):
    KIND: ClassVar[str] = "resource_analysis"
    PREFIX: ClassVar[str] = "rsa"
    investigation_id: str
    run_id: str  # the Run that measured (and can replay the definition of) the analysis
    spec_id: str  # rsp_<hash> of the whole specification
    spec: Mapping[str, object]
    model_id: str
    dataset_id: str
    provenance_fingerprint: str
    analysis_status: (
        str  # COMPLETE | PARTIAL (a trial failed or timed out, or evidence was insufficient)
    )
    summary: Mapping[str, object]
    created_at: datetime

    def __post_init__(self) -> None:
        v.ref("investigation_id", self.investigation_id, Investigation.PREFIX)
        v.ref("run_id", self.run_id, Run.PREFIX)
        v.text("spec_id", self.spec_id)
        object.__setattr__(self, "spec", v.freeze_mapping("spec", self.spec))
        v.ref("model_id", self.model_id, "mdl")
        v.ref("dataset_id", self.dataset_id, "dst")
        v.digest("provenance_fingerprint", self.provenance_fingerprint)
        if self.analysis_status not in ("COMPLETE", "PARTIAL"):
            raise ValidationError("analysis_status must be COMPLETE or PARTIAL")
        object.__setattr__(self, "summary", v.freeze_mapping("summary", self.summary))
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {
            "investigation_id": self.investigation_id,
            "spec_id": self.spec_id,
            "run_id": self.run_id,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        names = (
            "investigation_id",
            "run_id",
            "spec_id",
            "spec",
            "model_id",
            "dataset_id",
            "provenance_fingerprint",
            "analysis_status",
            "summary",
            "created_at",
        )
        open_payload(d, cls.KIND, names)
        return cls(
            v.get_str(d, "investigation_id"),
            v.get_str(d, "run_id"),
            v.get_str(d, "spec_id"),
            v.get_mapping(d, "spec"),
            v.get_str(d, "model_id"),
            v.get_str(d, "dataset_id"),
            v.get_str(d, "provenance_fingerprint"),
            v.get_str(d, "analysis_status"),
            v.get_mapping(d, "summary"),
            v.get_time(d, "created_at"),
        )


@dataclass(frozen=True)
class ResourceTrial(Entity):
    KIND: ClassVar[str] = "resource_trial"
    PREFIX: ClassVar[str] = "rst"
    analysis_id: str
    phase: str  # WARMUP | MEASURED
    trial_index: int
    status: str  # COMPLETED | FAILED | TIMED_OUT
    planned_samples: int
    completed_samples: int
    failed_samples: int
    wall_seconds: float
    cpu_seconds: float | None  # None: process CPU accounting unavailable
    created_at: datetime

    def __post_init__(self) -> None:
        v.ref("analysis_id", self.analysis_id, ResourceAnalysis.PREFIX)
        if self.phase not in PHASES:
            raise ValidationError(f"phase must be one of {list(PHASES)}")
        v.non_negative_int("trial_index", self.trial_index)
        if self.status not in TRIAL_STATUSES:
            raise ValidationError(f"trial status must be one of {list(TRIAL_STATUSES)}")
        for k in ("planned_samples", "completed_samples", "failed_samples"):
            v.non_negative_int(k, getattr(self, k))
        if self.completed_samples + self.failed_samples > self.planned_samples:
            raise ValidationError("completed and failed samples exceed the planned samples")
        if self.status == "COMPLETED" and self.completed_samples != self.planned_samples:
            raise ValidationError("a COMPLETED trial completed every planned sample")
        if v.finite_float("wall_seconds", self.wall_seconds) < 0:
            raise ValidationError("wall_seconds must be >= 0")
        if self.cpu_seconds is not None and v.finite_float("cpu_seconds", self.cpu_seconds) < 0:
            raise ValidationError("cpu_seconds must be >= 0")
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {
            "analysis_id": self.analysis_id,
            "phase": self.phase,
            "trial_index": self.trial_index,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        names = (
            "analysis_id",
            "phase",
            "trial_index",
            "status",
            "planned_samples",
            "completed_samples",
            "failed_samples",
            "wall_seconds",
            "cpu_seconds",
            "created_at",
        )
        open_payload(d, cls.KIND, names)
        cpu = v.get_opt_float(d, "cpu_seconds")
        return cls(
            v.get_str(d, "analysis_id"),
            v.get_str(d, "phase"),
            v.get_int(d, "trial_index"),
            v.get_str(d, "status"),
            v.get_int(d, "planned_samples"),
            v.get_int(d, "completed_samples"),
            v.get_int(d, "failed_samples"),
            v.finite_float("wall_seconds", v.get_raw(d, "wall_seconds")),
            cpu,
            v.get_time(d, "created_at"),
        )

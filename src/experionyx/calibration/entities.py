"""Registry entities: one calibration ANALYSIS (`cba_`) over a baseline run, and one persisted RESULT
(`cbr_`) per analyzed context (the baseline itself, each slice, window and stress trial). Bins,
per-sample evidence, metrics, intervals and comparisons live in digest-verified run artifacts
(calibration/spec.json, predictions.json, bins.json, metrics.json, uncertainty.json, comparisons.json,
candidates.json, summary.json); the records here are the queryable index and are append-only."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Self

import experionyx.validation as v
from experionyx.domain import Entity, Investigation, Run, open_payload
from experionyx.errors import ValidationError

CONTEXT_KINDS = ("BASELINE", "CALIBRATED", "SLICE", "WINDOW", "STRESS")
RESULT_STATUSES = ("COMPUTED", "INSUFFICIENT_EVIDENCE", "UNAVAILABLE")


@dataclass(frozen=True)
class CalibrationAnalysis(Entity):
    KIND: ClassVar[str] = "calibration_analysis"
    PREFIX: ClassVar[str] = "cba"
    investigation_id: str
    run_id: str  # the Run that produced (and can replay) the analysis
    spec_id: str  # cbs_<hash> of the whole specification
    spec: Mapping[str, object]
    baseline_run_id: str
    dataset_fingerprint: str
    provenance_fingerprint: str
    analysis_status: (
        str  # COMPLETE | PARTIAL (some context was insufficient, unavailable or invalid)
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
        names = ("investigation_id", "run_id", "spec_id", "spec", "baseline_run_id", "dataset_fingerprint", "provenance_fingerprint", "analysis_status", "summary", "created_at")  # fmt: skip
        open_payload(d, cls.KIND, names)
        return cls(
            v.get_str(d, "investigation_id"), v.get_str(d, "run_id"), v.get_str(d, "spec_id"),
            v.get_mapping(d, "spec"), v.get_str(d, "baseline_run_id"), v.get_str(d, "dataset_fingerprint"),
            v.get_str(d, "provenance_fingerprint"), v.get_str(d, "analysis_status"),
            v.get_mapping(d, "summary"), v.get_time(d, "created_at"),
        )  # fmt: skip


@dataclass(frozen=True)
class CalibrationResult(Entity):
    KIND: ClassVar[str] = "calibration_result"
    PREFIX: ClassVar[str] = "cbr"
    analysis_id: str
    context_key: str  # baseline | calibrated | slice:<name> | window:<name> | stress:<unit key>
    context_kind: str
    n_samples: int  # usable observations
    status: str  # COMPUTED | INSUFFICIENT_EVIDENCE | UNAVAILABLE
    reason: str | None
    headline: Mapping[str, object]  # the scalar metrics as recorded (empty unless COMPUTED)
    sample_digest: str  # over the sorted sample IDs of the context
    created_at: datetime

    def __post_init__(self) -> None:
        v.ref("analysis_id", self.analysis_id, CalibrationAnalysis.PREFIX)
        v.text("context_key", self.context_key)
        if self.context_kind not in CONTEXT_KINDS:
            raise ValidationError(f"context_kind must be one of {list(CONTEXT_KINDS)}")
        v.non_negative_int("n_samples", self.n_samples)
        if self.status not in RESULT_STATUSES:
            raise ValidationError(f"status must be one of {list(RESULT_STATUSES)}")
        if self.reason is not None:
            v.text("reason", self.reason)
        object.__setattr__(self, "headline", v.freeze_mapping("headline", self.headline))
        v.digest("sample_digest", self.sample_digest)
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {"analysis_id": self.analysis_id, "context_key": self.context_key}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        names = ("analysis_id", "context_key", "context_kind", "n_samples", "status", "reason", "headline", "sample_digest", "created_at")  # fmt: skip
        open_payload(d, cls.KIND, names)
        return cls(
            v.get_str(d, "analysis_id"), v.get_str(d, "context_key"), v.get_str(d, "context_kind"),
            v.get_int(d, "n_samples"), v.get_str(d, "status"), v.get_opt_str(d, "reason"),
            v.get_mapping(d, "headline"), v.get_str(d, "sample_digest"), v.get_time(d, "created_at"),
        )  # fmt: skip

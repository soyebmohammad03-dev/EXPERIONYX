"""Registry entities: one quality ANALYSIS of a registered dataset (`qan_`) and one CHECK RESULT
(`qck_`, one per check and scope) so results can be queried by type and status. The analysis stores
headline metadata; measurements, violations and evidence live in digest-verified run artifacts
(data_quality/spec.json, checks.json, observations.json, violations.json, summary.json)."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Self

import experionyx.validation as v
from experionyx.adapters.records import RegisteredDataset
from experionyx.data_quality.results import Status
from experionyx.domain import Entity, Investigation, Run, open_payload
from experionyx.errors import ValidationError


@dataclass(frozen=True)
class QualityAnalysis(Entity):
    KIND: ClassVar[str] = "quality_analysis"
    PREFIX: ClassVar[str] = "qan"
    investigation_id: str
    run_id: str  # the Run that produced (and can replay) the analysis
    spec_id: str  # dqs_<hash>
    spec: Mapping[str, object]
    dataset_id: str
    dataset_fingerprint: str
    provenance_fingerprint: str
    analysis_status: str  # COMPLETE | PARTIAL (some check was inconclusive or unavailable)
    summary: Mapping[str, object]
    created_at: datetime

    def __post_init__(self) -> None:
        v.ref("investigation_id", self.investigation_id, Investigation.PREFIX)
        v.ref("run_id", self.run_id, Run.PREFIX)
        v.text("spec_id", self.spec_id)
        object.__setattr__(self, "spec", v.freeze_mapping("spec", self.spec))
        v.ref("dataset_id", self.dataset_id, RegisteredDataset.PREFIX)
        v.digest("dataset_fingerprint", self.dataset_fingerprint)
        v.digest("provenance_fingerprint", self.provenance_fingerprint)
        if self.analysis_status not in ("COMPLETE", "PARTIAL"):
            raise ValidationError("analysis_status must be COMPLETE or PARTIAL")
        object.__setattr__(self, "summary", v.freeze_mapping("summary", self.summary))
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {"investigation_id": self.investigation_id, "spec_id": self.spec_id}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        names = ("investigation_id", "run_id", "spec_id", "spec", "dataset_id", "dataset_fingerprint", "provenance_fingerprint", "analysis_status", "summary", "created_at")  # fmt: skip
        open_payload(d, cls.KIND, names)
        return cls(
            v.get_str(d, "investigation_id"), v.get_str(d, "run_id"), v.get_str(d, "spec_id"),
            v.get_mapping(d, "spec"), v.get_str(d, "dataset_id"), v.get_str(d, "dataset_fingerprint"),
            v.get_str(d, "provenance_fingerprint"), v.get_str(d, "analysis_status"),
            v.get_mapping(d, "summary"), v.get_time(d, "created_at"),
        )  # fmt: skip


@dataclass(frozen=True)
class QualityCheck(Entity):
    KIND: ClassVar[str] = "quality_check"
    PREFIX: ClassVar[str] = "qck"
    analysis_id: str
    check_id: str  # qcs_<hash> of the check definition
    check_type: str
    scope: str  # e.g. "split=train", "slice=hi | split=test", "comparison=... | reference=..."
    status: str
    reason: str | None
    evidence: Mapping[str, object]
    created_at: datetime

    def __post_init__(self) -> None:
        v.ref("analysis_id", self.analysis_id, QualityAnalysis.PREFIX)
        v.text("check_id", self.check_id)
        v.text("check_type", self.check_type)
        v.text("scope", self.scope)
        if self.status not in {s.value for s in Status}:
            raise ValidationError(f"invalid check status {self.status!r}")
        if self.reason is not None:
            v.text("reason", self.reason)
        object.__setattr__(self, "evidence", v.freeze_mapping("evidence", self.evidence))
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {"analysis_id": self.analysis_id, "check_id": self.check_id, "scope": self.scope}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        names = ("analysis_id", "check_id", "check_type", "scope", "status", "reason", "evidence", "created_at")  # fmt: skip
        open_payload(d, cls.KIND, names)
        return cls(
            v.get_str(d, "analysis_id"), v.get_str(d, "check_id"), v.get_str(d, "check_type"),
            v.get_str(d, "scope"), v.get_str(d, "status"), v.get_opt_str(d, "reason"),
            v.get_mapping(d, "evidence"), v.get_time(d, "created_at"),
        )  # fmt: skip

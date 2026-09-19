"""Registry entities of the fault laboratory. They reuse the Experiment lifecycle and link, with
real foreign keys, a fault experiment to its baseline run, its treatment trials and its analysis."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from typing import ClassVar, Self

import experionyx.validation as v
from experionyx.domain import (
    EXPERIMENT_TRANSITIONS,
    Entity,
    Experiment,
    ExperimentStatus,
    Investigation,
    Run,
    open_payload,
)
from experionyx.errors import ValidationError


class TrialStatus(StrEnum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"  # the treatment run failed or could not be created; the reason is kept
    SKIPPED = "SKIPPED"  # not executed because an early-termination limit was reached


@dataclass(frozen=True)
class FaultExperiment(Entity):
    """A designed set of treatment runs (a fault, optional sweep, seeds) against ONE baseline run.
    Identity: investigation, name, baseline run and the canonical design (fault spec included)."""

    KIND: ClassVar[str] = "fault_experiment"
    PREFIX: ClassVar[str] = "fxp"
    investigation_id: str
    name: str
    baseline_experiment_id: str
    baseline_run_id: str
    design: Mapping[str, object]  # canonical FaultDesign JSON (see faults.design)
    created_at: datetime
    status: ExperimentStatus = ExperimentStatus.DRAFT

    def __post_init__(self) -> None:
        v.ref("investigation_id", self.investigation_id, Investigation.PREFIX)
        v.text("name", self.name)
        v.ref("baseline_experiment_id", self.baseline_experiment_id, Experiment.PREFIX)
        v.ref("baseline_run_id", self.baseline_run_id, Run.PREFIX)
        object.__setattr__(self, "design", v.freeze_mapping("design", self.design))
        v.timestamp("created_at", self.created_at)
        v.member("status", self.status, ExperimentStatus)

    def _identity(self) -> Mapping[str, object]:
        return {
            "investigation_id": self.investigation_id,
            "name": self.name,
            "baseline_run_id": self.baseline_run_id,
            "design": self.design,
        }

    def with_status(self, status: ExperimentStatus) -> Self:
        if status not in EXPERIMENT_TRANSITIONS[self.status]:
            raise ValidationError(f"illegal fault-experiment transition {self.status} -> {status}")
        return replace(self, status=status)

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d, cls.KIND,
            ("investigation_id", "name", "baseline_experiment_id", "baseline_run_id", "design", "created_at", "status"),
        )  # fmt: skip
        return cls(
            investigation_id=v.get_str(d, "investigation_id"),
            name=v.get_str(d, "name"),
            baseline_experiment_id=v.get_str(d, "baseline_experiment_id"),
            baseline_run_id=v.get_str(d, "baseline_run_id"),
            design=v.get_mapping(d, "design"),
            created_at=v.get_time(d, "created_at"),
            status=v.get_enum(d, "status", ExperimentStatus),
        )


@dataclass(frozen=True)
class FaultTrial(Entity):
    """One treatment point: which fault application (seed, parameter value) was run as which
    Run, against which control run. Trials are kept even when they fail or are skipped."""

    KIND: ClassVar[str] = "fault_trial"
    PREFIX: ClassVar[str] = "ftr"
    fault_experiment_id: str
    point_index: int
    repeat_index: int
    seed: int
    fault_id: str  # FaultSpec.id (includes the seed)
    family_id: str  # FaultSpec.family_id (same across seeds)
    baseline_run_id: str
    status: TrialStatus
    created_at: datetime
    parameter_name: str | None = None
    parameter_value: float | None = None
    treatment_experiment_id: str | None = None
    treatment_run_id: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        v.ref("fault_experiment_id", self.fault_experiment_id, FaultExperiment.PREFIX)
        v.non_negative_int("point_index", self.point_index)
        v.non_negative_int("repeat_index", self.repeat_index)
        v.non_negative_int("seed", self.seed)
        v.ref("fault_id", self.fault_id, "fap")
        v.ref("family_id", self.family_id, "flt")
        v.ref("baseline_run_id", self.baseline_run_id, Run.PREFIX)
        v.member("status", self.status, TrialStatus)
        v.timestamp("created_at", self.created_at)
        if self.parameter_name is not None:
            v.text("parameter_name", self.parameter_name)
        if (self.parameter_name is None) != (self.parameter_value is None):
            raise ValidationError("parameter_name and parameter_value go together")
        if self.parameter_value is not None:
            v.finite_float("parameter_value", self.parameter_value)
        if self.treatment_experiment_id is not None:
            v.ref("treatment_experiment_id", self.treatment_experiment_id, Experiment.PREFIX)
        if self.treatment_run_id is not None:
            v.ref("treatment_run_id", self.treatment_run_id, Run.PREFIX)
        if self.status is TrialStatus.COMPLETED and self.treatment_run_id is None:
            raise ValidationError("a COMPLETED trial must reference its treatment run")
        if self.status is not TrialStatus.COMPLETED and not self.reason:
            raise ValidationError("a FAILED or SKIPPED trial must state its reason")

    def _identity(self) -> Mapping[str, object]:
        return {
            "fault_experiment_id": self.fault_experiment_id,
            "point_index": self.point_index,
            "repeat_index": self.repeat_index,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        names = (
            "fault_experiment_id", "point_index", "repeat_index", "seed", "fault_id", "family_id",
            "baseline_run_id", "status", "created_at", "parameter_name", "parameter_value",
            "treatment_experiment_id", "treatment_run_id", "reason",
        )  # fmt: skip
        open_payload(d, cls.KIND, names)
        value = v.get_raw(d, "parameter_value")
        return cls(
            fault_experiment_id=v.get_str(d, "fault_experiment_id"),
            point_index=v.get_int(d, "point_index"),
            repeat_index=v.get_int(d, "repeat_index"),
            seed=v.get_int(d, "seed"),
            fault_id=v.get_str(d, "fault_id"),
            family_id=v.get_str(d, "family_id"),
            baseline_run_id=v.get_str(d, "baseline_run_id"),
            status=v.get_enum(d, "status", TrialStatus),
            created_at=v.get_time(d, "created_at"),
            parameter_name=v.get_opt_str(d, "parameter_name"),
            parameter_value=None if value is None else v.finite_float("parameter_value", value),
            treatment_experiment_id=v.get_opt_str(d, "treatment_experiment_id"),
            treatment_run_id=v.get_opt_str(d, "treatment_run_id"),
            reason=v.get_opt_str(d, "reason"),
        )


@dataclass(frozen=True)
class FaultAnalysis(Entity):
    """Links a fault experiment to the analysis Run whose artifacts hold its degradation results."""

    KIND: ClassVar[str] = "fault_analysis"
    PREFIX: ClassVar[str] = "fan"
    fault_experiment_id: str
    run_id: str
    created_at: datetime

    def __post_init__(self) -> None:
        v.ref("fault_experiment_id", self.fault_experiment_id, FaultExperiment.PREFIX)
        v.ref("run_id", self.run_id, Run.PREFIX)
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {"fault_experiment_id": self.fault_experiment_id, "run_id": self.run_id}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(d, cls.KIND, ("fault_experiment_id", "run_id", "created_at"))
        return cls(
            v.get_str(d, "fault_experiment_id"), v.get_str(d, "run_id"), v.get_time(d, "created_at")
        )

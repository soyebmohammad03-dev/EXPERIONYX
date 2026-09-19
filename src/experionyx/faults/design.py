"""Fault experiment design: sweep, repetitions, thresholds and safety limits (all typed and
validated, all serializable into the experiment record)."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Self

from experionyx.domain import to_jsonable
from experionyx.errors import FaultLimitError, ValidationError
from experionyx.evaluation.config import EvaluationConfig, _thaw
from experionyx.evaluation.serial import from_jsonable
from experionyx.faults.degradation import FaultThresholds

DESIGN_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SweepSpec:
    """One numeric fault parameter varied over explicit values. Every value is a real run."""

    parameter: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.parameter:
            raise ValidationError("sweep parameter is required")
        if not self.values:
            raise ValidationError("a sweep needs at least one value")
        if any(not math.isfinite(x) for x in self.values):
            raise ValidationError("sweep values must be finite")
        if len(set(self.values)) != len(self.values):
            raise ValidationError("sweep values must be unique")


@dataclass(frozen=True)
class FaultLimits:
    """Safety limits. Exceeding one refuses the experiment before anything runs (no silent
    truncation); the limits in force are stored with the experiment."""

    max_points: int = 20
    max_repetitions: int = 50
    max_total_runs: int = 200  # points x repetitions
    max_total_sample_evaluations: int = 5_000_000  # runs x evaluated samples
    max_bootstrap_resamples: int = 5000  # for aggregation (per-metric intervals have their own cap)
    max_failed_trials: int | None = (
        None  # early termination: after this many failures, skip the rest
    )

    def __post_init__(self) -> None:
        for name in (
            "max_points",
            "max_repetitions",
            "max_total_runs",
            "max_total_sample_evaluations",
            "max_bootstrap_resamples",
        ):
            if getattr(self, name) < 1:
                raise ValidationError(f"{name} must be >= 1")
        if self.max_failed_trials is not None and self.max_failed_trials < 1:
            raise ValidationError("max_failed_trials must be >= 1 or null")


@dataclass(frozen=True)
class FaultDesign:
    fault: Mapping[str, object]  # the base FaultSpec (canonical dict); seeds come from `seeds`
    evaluation: EvaluationConfig
    seeds: tuple[int, ...] = (0,)
    sweep: SweepSpec | None = None
    primary_metric: str | None = None  # default: accuracy (classification) / rmse (regression)
    thresholds: FaultThresholds = field(default_factory=FaultThresholds)
    aggregation_confidence: float = 0.95
    aggregation_resamples: int = 1000
    aggregation_seed: int = 0
    limits: FaultLimits = field(default_factory=FaultLimits)
    schema_version: int = DESIGN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DESIGN_SCHEMA_VERSION:
            raise ValidationError(f"unsupported design schema {self.schema_version}")
        if (
            not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or any(s < 0 for s in self.seeds)
        ):
            raise ValidationError("seeds must be a non-empty list of unique non-negative integers")
        if not 0.0 < self.aggregation_confidence < 1.0 or self.aggregation_resamples < 1:
            raise ValidationError("aggregation confidence must be in (0, 1) and resamples >= 1")
        lim = self.limits
        if self.aggregation_resamples > lim.max_bootstrap_resamples:
            raise FaultLimitError(
                f"aggregation_resamples {self.aggregation_resamples} exceeds max_bootstrap_resamples {lim.max_bootstrap_resamples}"
            )
        if (
            self.evaluation.bootstrap.enabled
            and self.evaluation.bootstrap.resamples > lim.max_bootstrap_resamples
        ):
            raise FaultLimitError(
                f"evaluation bootstrap resamples {self.evaluation.bootstrap.resamples} exceed max_bootstrap_resamples {lim.max_bootstrap_resamples}"
            )
        if self.points > lim.max_points:
            raise FaultLimitError(f"{self.points} sweep points exceed max_points {lim.max_points}")
        if len(self.seeds) > lim.max_repetitions:
            raise FaultLimitError(
                f"{len(self.seeds)} repetitions exceed max_repetitions {lim.max_repetitions}"
            )
        if self.total_runs > lim.max_total_runs:
            raise FaultLimitError(
                f"{self.total_runs} treatment runs exceed max_total_runs {lim.max_total_runs}"
            )

    @property
    def points(self) -> int:
        return len(self.sweep.values) if self.sweep else 1

    @property
    def total_runs(self) -> int:
        return self.points * len(self.seeds)

    def check_sample_budget(self, samples_per_run: int) -> None:
        total = self.total_runs * samples_per_run
        if total > self.limits.max_total_sample_evaluations:
            raise FaultLimitError(
                f"{total} sample evaluations ({self.total_runs} runs x {samples_per_run} samples) exceed max_total_sample_evaluations {self.limits.max_total_sample_evaluations}"
            )

    def to_dict(self) -> dict[str, object]:
        data = to_jsonable(self)
        assert isinstance(data, dict)  # noqa: S101
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        result: Self = from_jsonable(cls, _thaw(dict(data)))
        return result


@dataclass(frozen=True)
class FaultEvaluationConfig:
    """The configuration of ONE faulted evaluation run: the evaluation config plus the fault."""

    evaluation: EvaluationConfig
    fault: Mapping[str, object]
    schema_version: int = 1

    def to_parameters(self) -> dict[str, object]:
        data = to_jsonable(self)
        assert isinstance(data, dict)  # noqa: S101
        return {"fault_evaluation": data}

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, object]) -> Self:
        if set(parameters) != {"fault_evaluation"}:
            raise ValidationError(
                f"a faulted evaluation's configuration must be exactly {{'fault_evaluation': ...}}; got {sorted(parameters)}"
            )
        body = _thaw(parameters["fault_evaluation"])
        if not isinstance(body, dict):
            raise ValidationError("'fault_evaluation' must be an object")
        result: Self = from_jsonable(cls, body)
        return result


@dataclass(frozen=True)
class FaultAnalysisConfig:
    fault_experiment_id: str

    def to_parameters(self) -> dict[str, object]:
        return {"fault_analysis": {"fault_experiment_id": self.fault_experiment_id}}

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, object]) -> Self:
        if set(parameters) != {"fault_analysis"}:
            raise ValidationError(
                "an analysis configuration must be exactly {'fault_analysis': ...}"
            )
        body = _thaw(parameters["fault_analysis"])
        if not isinstance(body, dict):
            raise ValidationError("'fault_analysis' must be an object")
        result: Self = from_jsonable(cls, body)
        return result

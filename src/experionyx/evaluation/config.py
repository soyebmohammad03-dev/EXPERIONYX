"""Typed, validated, serializable evaluation configuration.

An EvaluationConfig is stored as the experiment's ConfigurationRef (so it is content-addressed
and part of provenance): `{"evaluation": <config>}`. Unknown fields are rejected, never dropped.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Self

from experionyx.adapters.capabilities import TaskType
from experionyx.domain import to_jsonable
from experionyx.errors import ValidationError
from experionyx.evaluation.serial import from_jsonable

CONFIG_SCHEMA_VERSION = 1
Scalar = bool | int | float | str


class SliceKind(StrEnum):
    TARGET_EQUALS = "TARGET_EQUALS"  # true target == value (class slice)
    PREDICTED_EQUALS = "PREDICTED_EQUALS"  # model prediction == value
    FEATURE_EQUALS = "FEATURE_EQUALS"  # feature (name or column index) == value
    FEATURE_RANGE = "FEATURE_RANGE"  # low <= feature < high (either bound optional)


class ScoreSource(StrEnum):
    AUTO = "AUTO"  # predict_proba if the model supports it, otherwise no scores
    PREDICT_PROBA = "PREDICT_PROBA"  # require predict_proba; unsupported => explicit UNSUPPORTED
    SOFTMAX_LOGITS = "SOFTMAX_LOGITS"  # ASSUMES model outputs are logits; softmax applied
    NONE = "NONE"  # labels only


def _finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValidationError(f"{name} must be finite")


@dataclass(frozen=True)
class SliceCondition:
    kind: SliceKind
    field: str | None = None  # feature name or column index, for FEATURE_* kinds
    value: Scalar | None = None
    low: float | None = None
    high: float | None = None

    def __post_init__(self) -> None:
        if self.kind in (SliceKind.TARGET_EQUALS, SliceKind.PREDICTED_EQUALS):
            if self.value is None or self.field is not None or None not in (self.low, self.high):
                raise ValidationError(f"{self.kind} needs only `value`")
        elif self.kind is SliceKind.FEATURE_EQUALS:
            if not self.field or self.value is None or None not in (self.low, self.high):
                raise ValidationError("FEATURE_EQUALS needs `field` and `value`")
        else:
            if not self.field or self.value is not None:
                raise ValidationError("FEATURE_RANGE needs `field` and low and/or high")
            if self.low is None and self.high is None:
                raise ValidationError("FEATURE_RANGE needs low and/or high")
            for n, b in (("low", self.low), ("high", self.high)):
                if b is not None:
                    _finite(n, b)
            if self.low is not None and self.high is not None and self.low >= self.high:
                raise ValidationError("FEATURE_RANGE requires low < high")


@dataclass(frozen=True)
class SliceSpec:
    """A named subset of samples: the AND of its conditions."""

    name: str
    conditions: tuple[SliceCondition, ...]

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise ValidationError("slice name must be non-empty without surrounding whitespace")
        if not self.conditions:
            raise ValidationError(f"slice {self.name!r} needs at least one condition")


@dataclass(frozen=True)
class BootstrapConfig:
    enabled: bool = True
    resamples: int = 200
    confidence: float = 0.95
    seed: int = 0
    metrics: tuple[str, ...] = ()  # empty: accuracy+f1 (classification) or mae+rmse (regression)
    include_slices: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.resamples <= 100_000:
            raise ValidationError("bootstrap.resamples must be in 1..100000")
        if not 0.0 < self.confidence < 1.0:
            raise ValidationError("bootstrap.confidence must be in (0, 1)")
        if isinstance(self.seed, bool) or self.seed < 0:
            raise ValidationError("bootstrap.seed must be a non-negative integer")
        if len(set(self.metrics)) != len(self.metrics):
            raise ValidationError("bootstrap.metrics contains duplicates")


@dataclass(frozen=True)
class CalibrationConfig:
    enabled: bool = True
    bins: int = 10
    high_confidence: float = 0.9  # a wrong prediction at or above this is a high-confidence error
    low_confidence: float = 0.6  # a right prediction below this is a low-confidence correct one

    def __post_init__(self) -> None:
        if not 1 <= self.bins <= 1000:
            raise ValidationError("calibration.bins must be in 1..1000")
        if not 0.0 < self.low_confidence <= self.high_confidence <= 1.0:
            raise ValidationError("need 0 < low_confidence <= high_confidence <= 1")


@dataclass(frozen=True)
class ErrorConfig:
    record: bool = True
    max_records: int = 1000  # counts are always complete; only the retained records are bounded

    def __post_init__(self) -> None:
        if not 0 <= self.max_records <= 1_000_000:
            raise ValidationError("errors.max_records must be in 0..1000000")


@dataclass(frozen=True)
class RetentionConfig:
    predictions: bool = True  # stream sample-level predictions to an artifact
    max_samples: int = 100_000  # safety bound: metrics need all labels/scores in memory

    def __post_init__(self) -> None:
        if self.max_samples < 1:
            raise ValidationError("retention.max_samples must be >= 1")


@dataclass(frozen=True)
class Thresholds:
    """Diagnostic thresholds for autopsy rules (docs/evaluation.md). Not universally valid."""

    imbalance_ratio_min: float = 3.0  # majority/minority count at which imbalance is reported
    minority_recall_max: float = 0.5  # finding if minority-class recall is below this
    ece_max: float = 0.10  # finding if expected calibration error exceeds this
    high_confidence_error_rate_max: float = 0.05  # fraction of samples that are HC errors
    r2_min: float = 0.5  # finding if R^2 is below this
    latency_outlier_factor: float = 5.0  # finding if max batch time > factor * median (>=5 batches)
    slice_accuracy_drop_max: float = 0.10  # absolute drop of accuracy/F1 vs overall
    slice_error_increase_max: float = 0.25  # relative increase of MAE/RMSE vs overall

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _finite(name, getattr(self, name))
        if self.imbalance_ratio_min < 1.0 or self.latency_outlier_factor < 1.0:
            raise ValidationError("ratio thresholds must be >= 1")
        for name in ("minority_recall_max", "ece_max", "high_confidence_error_rate_max"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValidationError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class EvaluationConfig:
    split: str | None = None  # None: all samples
    batch_size: int = 64
    metrics: tuple[str, ...] = ()  # empty: every registered metric compatible with the task
    task: TaskType | None = None  # override when neither model nor dataset states the task
    positive_label: Scalar | None = None  # for FP/FN in binary tasks; default: second class
    score_source: ScoreSource = ScoreSource.AUTO
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    errors: ErrorConfig = field(default_factory=ErrorConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)
    slices: tuple[SliceSpec, ...] = ()
    thresholds: Thresholds = field(default_factory=Thresholds)
    schema_version: int = CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ValidationError(f"unsupported evaluation config schema {self.schema_version}")
        if isinstance(self.batch_size, bool) or self.batch_size < 1:
            raise ValidationError("batch_size must be a positive integer")
        if self.split is not None and not self.split.strip():
            raise ValidationError("split must be a non-empty name or null")
        if len(set(self.metrics)) != len(self.metrics):
            raise ValidationError("metrics contains duplicates")
        names = [s.name for s in self.slices]
        if len(set(names)) != len(names):
            raise ValidationError("slice names must be unique")

    def to_parameters(self) -> dict[str, object]:
        """The dict stored in the experiment's ConfigurationRef."""
        data = to_jsonable(self)
        assert isinstance(data, dict)  # noqa: S101  # a dataclass always serializes to a dict
        return {"evaluation": data}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        result: Self = from_jsonable(cls, dict(data))
        return result

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, object]) -> Self:
        """Read an EvaluationConfig out of ConfigurationRef parameters; rejects other keys."""
        extra = set(parameters) - {"evaluation"}
        if extra or "evaluation" not in parameters:
            raise ValidationError(
                f"an evaluation experiment's configuration must be exactly "
                f"{{'evaluation': ...}}; unexpected keys: {sorted(extra)}"
            )
        body = parameters["evaluation"]
        if not isinstance(body, Mapping):
            raise ValidationError("'evaluation' must be an object")
        return cls.from_dict({str(k): _thaw(v) for k, v in body.items()})


def _thaw(value: object) -> object:
    """Frozen ConfigurationRef values (read-only mappings, tuples) back to plain JSON types."""
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [_thaw(v) for v in value]
    return value

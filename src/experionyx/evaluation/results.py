"""Typed evaluation results. Deliberately separate concepts (see docs/observation-vs-conclusion.md):

- `MetricResult`: a measurement with the context needed to interpret it (not an Observation).
- `SamplePrediction`: what the model said about one sample (not an error).
- `ErrorRecord`: a sample where something went wrong (not a failure mode).
- `AutopsyFinding`: an OBSERVED property that crossed a stated threshold (not an interpretation).
Unavailable analyses carry a `Status` and a reason; they are never zero.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from experionyx.adapters.capabilities import TaskType
from experionyx.evaluation.config import EvaluationConfig, SliceCondition

Scalar = bool | int | float | str
Label = int | str


class Status(StrEnum):
    COMPUTED = "COMPUTED"
    UNDEFINED = "UNDEFINED"  # mathematically undefined for these data (e.g. one class only)
    UNSUPPORTED = "UNSUPPORTED"  # the model/config does not provide the required inputs
    INVALID = "INVALID"  # non-finite or inconsistent numerics


class ErrorKind(StrEnum):
    FALSE_POSITIVE = "FALSE_POSITIVE"
    FALSE_NEGATIVE = "FALSE_NEGATIVE"
    INCORRECT_CLASS = "INCORRECT_CLASS"
    LOW_CONFIDENCE_CORRECT = "LOW_CONFIDENCE_CORRECT"
    HIGH_CONFIDENCE_INCORRECT = "HIGH_CONFIDENCE_INCORRECT"
    RESIDUAL = "RESIDUAL"


class FindingType(StrEnum):
    CLASS_IMBALANCE = "CLASS_IMBALANCE"
    WEAK_MINORITY_RECALL = "WEAK_MINORITY_RECALL"
    CALIBRATION_ERROR = "CALIBRATION_ERROR"
    HIGH_CONFIDENCE_ERRORS = "HIGH_CONFIDENCE_ERRORS"
    RESIDUAL_VARIANCE = "RESIDUAL_VARIANCE"
    LATENCY_OUTLIER = "LATENCY_OUTLIER"
    SLICE_DEGRADATION = "SLICE_DEGRADATION"


class Severity(StrEnum):
    NOTICE = "NOTICE"
    WARNING = "WARNING"


class Interpretation(StrEnum):
    OBSERVED = "OBSERVED"  # a measured property crossed a stated threshold
    INTERPRETED = "INTERPRETED"  # a human/analysis assigned meaning; never emitted automatically


@dataclass(frozen=True)
class MetricContext:
    run_id: str
    split: str | None
    model_id: str | None = None
    model_fingerprint: str | None = None
    dataset_id: str | None = None
    dataset_fingerprint: str | None = None


@dataclass(frozen=True)
class Interval:
    status: Status
    method: str  # e.g. "bootstrap-percentile"
    confidence: float
    resamples: int
    valid_resamples: int
    seed: int
    n_samples: int
    lower: float | None = None
    upper: float | None = None
    reason: str | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class MetricResult:
    metric_id: str
    metric_version: str
    name: str
    status: Status
    n_samples: int
    context: MetricContext
    scale: str
    higher_is_better: bool | None
    value: float | None = None
    classes: tuple[Label, ...] | None = None
    structured: Mapping[str, object] | None = None
    reason: str | None = None
    warnings: tuple[str, ...] = ()
    interval: Interval | None = None


@dataclass(frozen=True)
class SamplePrediction:
    index: int
    true: Scalar
    predicted: Scalar
    correct: bool | None  # None for regression (no notion of exactly correct)
    confidence: float | None = None
    scores: tuple[float, ...] | None = None


@dataclass(frozen=True)
class ErrorRecord:
    sample_index: int
    kinds: tuple[ErrorKind, ...]
    true: Scalar
    predicted: Scalar
    confidence: float | None = None
    absolute_error: float | None = None
    signed_error: float | None = None  # predicted - true
    relative_error: float | None = None  # |error| / |true|, None when true == 0


@dataclass(frozen=True)
class ErrorSummary:
    status: Status
    total_samples: int
    counts: Mapping[str, int]  # by ErrorKind; always complete even when records are truncated
    recorded: int
    truncated: bool
    reason: str | None = None
    stats: Mapping[str, float] = field(default_factory=dict)  # regression residual statistics


@dataclass(frozen=True)
class ClassMetrics:
    label: Label
    support: int
    predicted: int
    precision: float | None  # None: undefined (no predictions of this class)
    recall: float | None  # None: undefined (no samples of this class)
    f1: float | None


@dataclass(frozen=True)
class ConfusionMatrix:
    labels: tuple[Label, ...]
    counts: tuple[tuple[int, ...], ...]  # rows = true, columns = predicted
    row_normalized: tuple[tuple[float | None, ...], ...]  # None for rows with no samples
    column_normalized: tuple[tuple[float | None, ...], ...]  # None for columns never predicted
    per_class: tuple[ClassMetrics, ...]


@dataclass(frozen=True)
class ImbalanceResult:
    status: Status
    class_counts: Mapping[str, int]
    class_proportions: Mapping[str, float]
    majority_class: Label | None
    minority_class: Label | None
    imbalance_ratio: float | None  # majority count / minority count (>= 1)
    note: str = ""
    reason: str | None = None


@dataclass(frozen=True)
class ConfidenceResult:
    status: Status
    source: str | None  # where the confidence signal came from (never assumed calibrated)
    n_samples: int = 0
    distribution: tuple[int, ...] = ()  # 10 equal-width bins over [0, 1]
    mean_confidence_correct: float | None = None
    mean_confidence_incorrect: float | None = None
    n_correct: int = 0
    n_incorrect: int = 0
    high_confidence_threshold: float | None = None
    low_confidence_threshold: float | None = None
    high_confidence_errors: int = 0
    low_confidence_correct: int = 0
    reason: str | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float | None
    accuracy: float | None
    gap: float | None  # |accuracy - mean_confidence|


@dataclass(frozen=True)
class CalibrationResult:
    status: Status
    confidence_source: str | None
    binning: str = "equal-width; bin k = [k/B, (k+1)/B), last bin closed"
    interpretation: str = "top-label (confidence) calibration"
    n_bins: int = 0
    n_samples: int = 0
    bins: tuple[CalibrationBin, ...] = ()
    ece: float | None = None
    mce: float | None = None
    brier_score: float | None = None
    reason: str | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SliceResult:
    name: str
    conditions: tuple[SliceCondition, ...]
    n_samples: int
    fraction_of_samples: float
    metrics: tuple[MetricResult, ...]
    deltas_vs_overall: Mapping[str, float | None]  # slice value - overall value


@dataclass(frozen=True)
class LatencySummary:
    status: Status
    load_seconds: float | None  # cold start, never mixed into inference statistics
    n_samples: int
    n_batches: int
    total_seconds: float | None = None
    mean_batch_seconds: float | None = None
    median_batch_seconds: float | None = None
    p95_batch_seconds: float | None = None  # nearest-rank percentile
    max_batch_seconds: float | None = None
    throughput_samples_per_second: float | None = None
    reason: str | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExcludedSamples:
    count: int
    reason: str
    indices: tuple[int, ...] = ()  # first indices only
    truncated: bool = False


@dataclass(frozen=True)
class AutopsyFinding:
    finding_id: str
    type: FindingType
    severity: Severity
    interpretation: Interpretation
    rule_id: str
    rule_version: str
    description: str  # deterministic template text; states what was measured, never why
    affected: str  # "overall", "class:<label>" or "slice:<name>"
    observed_value: float
    threshold: float
    comparison: str  # e.g. "observed > threshold"
    metric_ids: tuple[str, ...]
    observation_names: tuple[str, ...]  # Observations recorded for this run that support it
    artifact_paths: tuple[str, ...]
    claim_id: str | None = None
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelProfile:
    """Structured summary of measured properties. No narrative, no causal claims."""

    model_name: str | None
    model_fingerprint: str | None
    model_adapter: str | None
    model_type: str | None
    dataset_name: str | None
    dataset_fingerprint: str | None
    task: TaskType
    split: str | None
    n_samples: int
    model_capabilities: tuple[str, ...]
    parameter_count: int | None
    model_size_bytes: int | None
    metrics: Mapping[str, float | None]
    calibration_status: Status
    ece: float | None
    error_counts: Mapping[str, int]
    per_class: tuple[ClassMetrics, ...]
    latency: LatencySummary
    slices: Mapping[str, int]  # slice name -> sample count
    findings_by_type: Mapping[str, int]
    unavailable_analyses: Mapping[str, str]  # analysis -> reason


@dataclass(frozen=True)
class EvaluationResult:
    evaluator_version: str
    ruleset_version: str
    context: MetricContext
    config: EvaluationConfig
    task: TaskType
    n_samples: int
    classes: tuple[Label, ...] | None
    score_source: str | None
    metrics: tuple[MetricResult, ...]
    confusion: ConfusionMatrix | None
    imbalance: ImbalanceResult | None
    errors: ErrorSummary
    confidence: ConfidenceResult
    calibration: CalibrationResult
    slices: tuple[SliceResult, ...]
    latency: LatencySummary
    findings: tuple[AutopsyFinding, ...]
    excluded: ExcludedSamples
    artifacts: Mapping[str, str]  # logical name -> registered artifact path
    warnings: tuple[str, ...] = ()
    profile: ModelProfile | None = None

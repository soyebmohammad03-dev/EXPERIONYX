"""Degradation measurement, repeated-trial aggregation, effect thresholds and severity.

Everything here is computed from stored EvaluationResults of real runs. Nothing is interpolated or
invented. Conventions (docs/faults.md):
- delta = faulted - baseline (absolute), relative = delta / |baseline| (None when baseline is 0).
- `deterioration` is signed so that POSITIVE always means WORSE: -delta for higher-is-better
  metrics, +delta for lower-is-better ones. Metrics with no declared direction are NEUTRAL and get
  no deterioration.
- Aggregation over seeds describes variability of the *perturbation randomness* on the same
  evaluation samples. It is not a sample of the population and supports no significance claim.
"""

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from experionyx.errors import ValidationError
from experionyx.evaluation.bootstrap import quantile
from experionyx.evaluation.compare import EvaluationComparison, compare_evaluations
from experionyx.evaluation.results import EvaluationResult, MetricResult, Status


class Direction(StrEnum):
    HIGHER_IS_BETTER = "HIGHER_IS_BETTER"
    LOWER_IS_BETTER = "LOWER_IS_BETTER"
    NEUTRAL = "NEUTRAL"  # no declared direction: change is reported, never called degradation


class Change(StrEnum):
    DETERIORATED = "DETERIORATED"
    IMPROVED = "IMPROVED"
    UNCHANGED = "UNCHANGED"
    UNDETERMINED = "UNDETERMINED"  # neutral metric, or a value is missing


class EffectClass(StrEnum):
    """Diagnostic classification of a measured effect. NOT a verdict that a model 'failed'."""

    NO_MEASURED_DEGRADATION = "NO_MEASURED_DEGRADATION"
    MEASURED_DEGRADATION = "MEASURED_DEGRADATION"
    SUBSTANTIAL_DEGRADATION = "SUBSTANTIAL_DEGRADATION"
    INCONCLUSIVE = "INCONCLUSIVE"


def direction_of(higher_is_better: bool | None) -> Direction:
    if higher_is_better is None:
        return Direction.NEUTRAL
    return Direction.HIGHER_IS_BETTER if higher_is_better else Direction.LOWER_IS_BETTER


@dataclass(frozen=True)
class MetricDegradation:
    metric_id: str
    direction: Direction
    baseline: float | None
    faulted: float | None
    absolute_delta: float | None  # faulted - baseline
    relative_delta: float | None  # absolute_delta / |baseline|
    deterioration: float | None  # > 0 means worse (direction-aware)
    relative_deterioration: float | None
    change: Change
    reason: str | None = None  # why a value is missing


def _degradation(
    mid: str, base: MetricResult | None, fault: MetricResult | None
) -> MetricDegradation:
    direction = direction_of(base.higher_is_better if base else None)
    sides = {"baseline": base, "faulted": fault}
    for name, m in sides.items():
        if m is None or m.status is not Status.COMPUTED or m.value is None:
            why = "not evaluated" if m is None else f"{m.status.value}: {m.reason}"
            return MetricDegradation(
                mid,
                direction,
                None if base is None else base.value,
                None if fault is None else fault.value,
                None,
                None,
                None,
                None,
                Change.UNDETERMINED,
                f"{name} {why}",
            )
    if base is None or fault is None or base.value is None or fault.value is None:
        raise ValidationError("unreachable: sides were validated above")
    delta = fault.value - base.value
    rel = None if base.value == 0 else delta / abs(base.value)
    if direction is Direction.NEUTRAL:
        return MetricDegradation(
            mid,
            direction,
            base.value,
            fault.value,
            delta,
            rel,
            None,
            None,
            Change.UNDETERMINED,
            "no declared direction",
        )
    sign = -1.0 if direction is Direction.HIGHER_IS_BETTER else 1.0
    det = sign * delta
    rel_det = None if base.value == 0 else det / abs(base.value)
    change = Change.UNCHANGED if det == 0 else (Change.DETERIORATED if det > 0 else Change.IMPROVED)
    return MetricDegradation(
        mid, direction, base.value, fault.value, delta, rel, det, rel_det, change
    )


def measure_degradation(
    baseline: EvaluationResult, faulted: EvaluationResult
) -> tuple[MetricDegradation, ...]:
    """Per-metric baseline-vs-faulted degradation for every scalar metric of the baseline."""
    fm = {m.metric_id: m for m in faulted.metrics}
    return tuple(
        _degradation(m.metric_id, m, fm.get(m.metric_id))
        for m in baseline.metrics
        if m.structured is None  # skip structural metrics (confusion matrix)
    )


@dataclass(frozen=True)
class LatencyDegradation:
    direction: Direction
    baseline_median_seconds: float | None
    faulted_median_seconds: float | None
    absolute_delta: float | None
    relative_delta: float | None
    note: str = "timings are noisy, hardware dependent, and exclude fault-application time"


def measure_latency(baseline: EvaluationResult, faulted: EvaluationResult) -> LatencyDegradation:
    a, b = baseline.latency.median_batch_seconds, faulted.latency.median_batch_seconds
    delta = None if a is None or b is None else b - a
    return LatencyDegradation(
        Direction.LOWER_IS_BETTER, a, b, delta, None if delta is None or not a else delta / a
    )


# --- repeated trials -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Aggregate:
    n: int
    mean: float | None
    median: float | None
    std: float | None  # sample standard deviation (n - 1); None for n < 2
    minimum: float | None
    maximum: float | None
    ci_lower: float | None  # bootstrap percentile interval of the MEAN
    ci_upper: float | None
    confidence: float
    resamples: int
    seed: int
    effect_size_dz: (
        float | None
    )  # mean / std of the per-seed values (standardized mean); None if undefined
    method: str = "percentile bootstrap of the mean over seeds"
    warnings: tuple[str, ...] = ()


def aggregate(
    values: Sequence[float], *, confidence: float = 0.95, resamples: int = 1000, seed: int = 0
) -> Aggregate:
    """Summary statistics over per-seed values. The raw values must be kept by the caller."""
    if not 0.0 < confidence < 1.0 or resamples < 1:
        raise ValidationError("confidence must be in (0, 1) and resamples >= 1")
    vals = [float(v) for v in values]
    if any(not math.isfinite(v) for v in vals):
        raise ValidationError("aggregate() got a non-finite value")
    n = len(vals)
    if n == 0:
        return Aggregate(
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            confidence,
            resamples,
            seed,
            None,
            warnings=("no trials",),
        )
    mean = sum(vals) / n
    std = statistics.stdev(vals) if n > 1 else None
    warnings = []
    lo = hi = None
    if n > 1:
        rng = random.Random(seed)  # noqa: S311  # resampling, not security
        means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
        alpha = (1 - confidence) / 2
        lo, hi = quantile(means, alpha), quantile(means, 1 - alpha)
        if n < 10:
            warnings.append(f"only {n} seeds: the interval of the mean is unstable")
    else:
        warnings.append("a single trial: no spread or interval can be estimated")
    dz = mean / std if std else None
    if n > 1:
        warnings.append(
            "seeds vary only the perturbation randomness on the same evaluation samples; this is not population-level uncertainty"
        )
    return Aggregate(
        n,
        mean,
        statistics.median(vals),
        std,
        min(vals),
        max(vals),
        lo,
        hi,
        confidence,
        resamples,
        seed,
        dz,
        warnings=tuple(warnings),
    )


# --- effect assessment ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FaultThresholds:
    """Transparent thresholds for the diagnostic effect classification (not universal truths)."""

    min_absolute_degradation: float = 0.02  # in metric units
    min_relative_degradation: float = 0.05
    substantial_absolute_degradation: float = 0.10
    substantial_relative_degradation: float = 0.20
    min_affected_fraction: float = 0.0  # below this the effect is INCONCLUSIVE (too few affected)
    require_uncertainty_support: bool = True  # need the interval to exclude "no change"

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            v = getattr(self, name)
            if isinstance(v, float) and (not math.isfinite(v) or v < 0):
                raise ValidationError(f"{name} must be a finite number >= 0")
        if self.min_affected_fraction > 1.0:
            raise ValidationError("min_affected_fraction must be <= 1")
        if (self.substantial_absolute_degradation < self.min_absolute_degradation
                or self.substantial_relative_degradation < self.min_relative_degradation):  # fmt: skip
            raise ValidationError("substantial thresholds must be >= the minimum thresholds")


@dataclass(frozen=True)
class FaultEffectAssessment:
    classification: EffectClass
    primary_metric: str
    deterioration: float | None
    relative_deterioration: float | None
    affected_fraction: float | None
    uncertainty_supports_effect: bool | None  # None: not checkable
    thresholds: FaultThresholds
    basis: str  # the rule that produced the classification, in words
    reasons: tuple[str, ...] = ()


def assess_effect(
    primary_metric: str,
    deterioration: float | None,
    relative_deterioration: float | None,
    *,
    thresholds: FaultThresholds,
    affected_fraction: float | None,
    uncertainty_excludes_zero: bool | None,
) -> FaultEffectAssessment:
    """Apply the documented rule. The result describes a measured effect on ONE metric under ONE
    fault specification; it is not a statement about the model's robustness or failure."""

    def result(cls: EffectClass, basis: str, *reasons: str) -> FaultEffectAssessment:
        return FaultEffectAssessment(
            cls,
            primary_metric,
            deterioration,
            relative_deterioration,
            affected_fraction,
            uncertainty_excludes_zero,
            thresholds,
            basis,
            tuple(reasons),
        )

    if deterioration is None:
        return result(
            EffectClass.INCONCLUSIVE,
            "primary metric unavailable",
            "the primary metric is missing or undefined on one side",
        )
    if affected_fraction is not None and affected_fraction < thresholds.min_affected_fraction:
        return result(
            EffectClass.INCONCLUSIVE,
            "affected population below minimum",
            f"affected fraction {affected_fraction:.4g} < {thresholds.min_affected_fraction}",
        )
    rel = relative_deterioration
    small_abs = deterioration < thresholds.min_absolute_degradation
    small_rel = rel is not None and rel < thresholds.min_relative_degradation
    if deterioration <= 0 or small_abs or small_rel:
        why = (
            "no deterioration"
            if deterioration <= 0
            else "deterioration below the minimum thresholds"
        )
        return result(
            EffectClass.NO_MEASURED_DEGRADATION,
            why,
            f"deterioration {deterioration:.6g} (relative {rel if rel is None else round(rel, 6)})",
        )
    if thresholds.require_uncertainty_support and uncertainty_excludes_zero is False:
        return result(
            EffectClass.INCONCLUSIVE,
            "uncertainty includes no change",
            "the uncertainty interval does not exclude zero deterioration",
        )
    substantial = deterioration >= thresholds.substantial_absolute_degradation and (
        rel is None or rel >= thresholds.substantial_relative_degradation
    )
    if substantial:
        return result(EffectClass.SUBSTANTIAL_DEGRADATION, "meets both substantial thresholds")
    return result(
        EffectClass.MEASURED_DEGRADATION, "meets the minimum thresholds, below substantial"
    )


# --- severity (multi-dimensional; deliberately not a single score) -------------------------------


@dataclass(frozen=True)
class FaultSeverity:
    parameter_intensity: Mapping[str, float]  # numeric parameters of the fault
    affected_fraction: float | None
    primary_deterioration_mean: float | None
    primary_deterioration_ci: tuple[float, ...] | None  # (lower, upper)
    class_recall_deterioration: Mapping[str, float | None]  # mean over trials; > 0 means worse
    slice_deterioration: Mapping[str, float | None]  # primary metric per slice; > 0 means worse
    note: str = "multi-dimensional on purpose; no composite score is defined in this phase"


def _class_recall_deterioration(
    comparisons: Sequence[EvaluationComparison],
) -> dict[str, float | None]:
    per: dict[str, list[float]] = {}
    for c in comparisons:
        for cls in c.classes:
            if cls.recall is not None:
                per.setdefault(cls.label, []).append(-cls.recall)  # recall: higher is better
    return {k: sum(v) / len(v) for k, v in sorted(per.items())}


def build_severity(
    parameters: Mapping[str, float],
    affected_fraction: float | None,
    primary: Aggregate | None,
    baselines_and_faulted: Sequence[tuple[EvaluationResult, EvaluationResult]],
    primary_metric: str,
) -> FaultSeverity:
    comps = [compare_evaluations(b, f) for b, f in baselines_and_faulted]
    slices: dict[str, list[float]] = {}
    higher = None
    for b, _ in baselines_and_faulted[:1]:
        m = next((x for x in b.metrics if x.metric_id == primary_metric), None)
        higher = None if m is None else m.higher_is_better
    for c in comps:
        for s in c.slices:
            d = next((m.delta for m in s.metrics if m.metric_id == primary_metric), None)
            if d is not None and higher is not None:
                slices.setdefault(s.name, []).append(-d if higher else d)
    ci = (
        None
        if primary is None or primary.ci_lower is None or primary.ci_upper is None
        else (primary.ci_lower, primary.ci_upper)
    )
    return FaultSeverity(
        dict(parameters), affected_fraction, None if primary is None else primary.mean, ci,
        _class_recall_deterioration(comps), {k: sum(v) / len(v) for k, v in sorted(slices.items())},
    )  # fmt: skip


# --- interaction foundation ----------------------------------------------------------------------


@dataclass(frozen=True)
class InteractionDescription:
    """Descriptive record for a fault pair; asserts NO interaction. The additive reference is a
    convenient point of comparison, not a null hypothesis; a rigorous interaction analysis needs
    an explicit definition and factorial design (later phase)."""

    metric_id: str
    effect_a: float | None
    effect_b: float | None
    effect_ab: float | None
    additive_reference: float | None
    difference_from_additive: float | None
    note: str = "descriptive only: not evidence of interaction or its absence"


def describe_interaction(
    metric_id: str, a: Aggregate, b: Aggregate, ab: Aggregate
) -> InteractionDescription:
    ref = None if a.mean is None or b.mean is None else a.mean + b.mean
    diff = None if ref is None or ab.mean is None else ab.mean - ref
    return InteractionDescription(metric_id, a.mean, b.mean, ab.mean, ref, diff)

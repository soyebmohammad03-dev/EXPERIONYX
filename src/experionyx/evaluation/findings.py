"""Autopsy rules: explicit, configurable, versioned, deterministic (docs/evaluation.md).

A finding states that a *measured* property crossed a *stated* threshold. It is OBSERVED, never
INTERPRETED: no rule asserts a cause. Thresholds live in `Thresholds` (part of the evaluation
configuration), not in code, and are diagnostics, not scientific truths.
"""

from collections.abc import Mapping, Sequence

from experionyx.adapters.capabilities import TaskType
from experionyx.evaluation.config import Thresholds
from experionyx.evaluation.results import (
    AutopsyFinding,
    CalibrationResult,
    ConfidenceResult,
    ConfusionMatrix,
    ErrorSummary,
    FindingType,
    ImbalanceResult,
    Interpretation,
    LatencySummary,
    MetricResult,
    Severity,
    SliceResult,
    Status,
)
from experionyx.hashing import content_hash

RULESET_VERSION = "1"

# rule id -> (finding type, one-line definition). Documented in docs/evaluation.md.
RULES: dict[str, tuple[FindingType, str]] = {
    "class-imbalance": (FindingType.CLASS_IMBALANCE, "majority/minority count >= imbalance_ratio_min"),
    "minority-recall": (FindingType.WEAK_MINORITY_RECALL, "recall of the least frequent class < minority_recall_max"),
    "ece-threshold": (FindingType.CALIBRATION_ERROR, "expected calibration error > ece_max"),
    "high-confidence-errors": (FindingType.HIGH_CONFIDENCE_ERRORS, "share of samples that are high-confidence errors > high_confidence_error_rate_max"),
    "r2-floor": (FindingType.RESIDUAL_VARIANCE, "R^2 < r2_min (residuals explain little of the target variance)"),
    "latency-outlier": (FindingType.LATENCY_OUTLIER, "max batch time / median batch time > latency_outlier_factor (>= 5 batches)"),
    "slice-accuracy-drop": (FindingType.SLICE_DEGRADATION, "slice accuracy/F1 is more than slice_accuracy_drop_max below overall"),
    "slice-error-increase": (FindingType.SLICE_DEGRADATION, "slice MAE/RMSE is more than slice_error_increase_max (relative) above overall"),
}  # fmt: skip

_HIGHER_BETTER = ("accuracy", "f1")
_ERROR_LIKE = ("mae", "rmse")


def _fmt(x: float) -> str:
    return f"{x:.6g}"


def _finding(
    rule_id: str, severity: Severity, affected: str, observed: float, threshold: float,
    comparison: str, description: str, metric_ids: Sequence[str],
    observations: Sequence[str], artifacts: Sequence[str],
) -> AutopsyFinding:  # fmt: skip
    ftype, _ = RULES[rule_id]
    ident = content_hash({"rule": rule_id, "version": RULESET_VERSION, "affected": affected})
    return AutopsyFinding(
        finding_id="fnd_" + ident[len("sha256:") :][:16],
        type=ftype,
        severity=severity,
        interpretation=Interpretation.OBSERVED,
        rule_id=rule_id,
        rule_version=RULESET_VERSION,
        description=description,
        affected=affected,
        observed_value=observed,
        threshold=threshold,
        comparison=comparison,
        metric_ids=tuple(metric_ids),
        observation_names=tuple(observations),
        artifact_paths=tuple(artifacts),
    )


def derive_findings(
    *,
    task: TaskType,
    metrics: Mapping[str, MetricResult],
    confusion: ConfusionMatrix | None,
    imbalance: ImbalanceResult | None,
    confidence: ConfidenceResult,
    calibration: CalibrationResult,
    errors: ErrorSummary,
    latency: LatencySummary,
    slices: Sequence[SliceResult],
    n_samples: int,
    thresholds: Thresholds,
) -> list[AutopsyFinding]:
    t = thresholds
    out: list[AutopsyFinding] = []
    if imbalance and imbalance.status is Status.COMPUTED and imbalance.imbalance_ratio is not None:
        r = imbalance.imbalance_ratio
        if r >= t.imbalance_ratio_min:
            out.append(_finding(
                "class-imbalance", Severity.NOTICE, "overall", r, t.imbalance_ratio_min, "observed >= threshold",
                f"Majority/minority class count ratio is {_fmt(r)} (majority {imbalance.majority_class!r}, minority {imbalance.minority_class!r}); threshold {_fmt(t.imbalance_ratio_min)}.",
                (), ("imbalance.ratio",), ("evaluation/confusion.json",),
            ))  # fmt: skip
    if (
        confusion
        and imbalance
        and imbalance.status is Status.COMPUTED
        and imbalance.minority_class != imbalance.majority_class
    ):
        pc = next((c for c in confusion.per_class if c.label == imbalance.minority_class), None)
        if pc is not None and pc.recall is not None and pc.recall < t.minority_recall_max:
            out.append(_finding(
                "minority-recall", Severity.WARNING, f"class:{pc.label}", pc.recall, t.minority_recall_max, "observed < threshold",
                f"Recall of the least frequent class {pc.label!r} is {_fmt(pc.recall)} (support {pc.support}); threshold {_fmt(t.minority_recall_max)}.",
                ("recall",), (f"per_class.{pc.label}.recall",), ("evaluation/confusion.json",),
            ))  # fmt: skip
    if (
        calibration.status is Status.COMPUTED
        and calibration.ece is not None
        and calibration.ece > t.ece_max
    ):
        out.append(_finding(
            "ece-threshold", Severity.WARNING, "overall", calibration.ece, t.ece_max, "observed > threshold",
            f"Expected calibration error is {_fmt(calibration.ece)} over {calibration.n_bins} bins (confidence source {calibration.confidence_source}); threshold {_fmt(t.ece_max)}.",
            (), ("calibration.ece",), ("evaluation/calibration.json",),
        ))  # fmt: skip
    if confidence.status is Status.COMPUTED and confidence.n_samples:
        rate = confidence.high_confidence_errors / confidence.n_samples
        if rate > t.high_confidence_error_rate_max:
            out.append(_finding(
                "high-confidence-errors", Severity.WARNING, "overall", rate, t.high_confidence_error_rate_max, "observed > threshold",
                f"{confidence.high_confidence_errors} of {confidence.n_samples} samples are wrong with confidence >= {_fmt(confidence.high_confidence_threshold or 0.0)} (rate {_fmt(rate)}); threshold {_fmt(t.high_confidence_error_rate_max)}.",
                (), ("confidence.high_confidence_error_rate",), ("evaluation/calibration.json", "evaluation/errors.jsonl"),
            ))  # fmt: skip
    r2 = metrics.get("r2")
    if (
        task is TaskType.REGRESSION
        and r2
        and r2.status is Status.COMPUTED
        and r2.value is not None
        and r2.value < t.r2_min
    ):
        out.append(_finding(
            "r2-floor", Severity.WARNING, "overall", r2.value, t.r2_min, "observed < threshold",
            f"R^2 is {_fmt(r2.value)}; threshold {_fmt(t.r2_min)}.",
            ("r2",), ("metric.r2",), ("evaluation/metrics.json",),
        ))  # fmt: skip
    if (
        latency.status is Status.COMPUTED and latency.n_batches >= 5
        and latency.median_batch_seconds and latency.max_batch_seconds is not None
    ):  # fmt: skip
        ratio = latency.max_batch_seconds / latency.median_batch_seconds
        if ratio > t.latency_outlier_factor:
            out.append(_finding(
                "latency-outlier", Severity.NOTICE, "overall", ratio, t.latency_outlier_factor, "observed > threshold",
                f"Slowest batch took {_fmt(ratio)}x the median batch time over {latency.n_batches} batches; threshold {_fmt(t.latency_outlier_factor)}x.",
                (), ("latency.max_batch_seconds", "latency.median_batch_seconds"), ("evaluation/latency.json",),
            ))  # fmt: skip
    for sl in slices:
        for mid, delta in sl.deltas_vs_overall.items():
            base = metrics.get(mid)
            if delta is None or base is None or base.value is None:
                continue
            if mid in _HIGHER_BETTER and -delta > t.slice_accuracy_drop_max:
                out.append(_finding(
                    "slice-accuracy-drop", Severity.WARNING, f"slice:{sl.name}", -delta, t.slice_accuracy_drop_max, "observed drop > threshold",
                    f"On slice {sl.name!r} (n={sl.n_samples}) {mid} is {_fmt(-delta)} below overall; threshold {_fmt(t.slice_accuracy_drop_max)}.",
                    (mid,), (f"slice.{sl.name}.metric.{mid}", f"metric.{mid}"), ("evaluation/slices.json",),
                ))  # fmt: skip
            elif (
                mid in _ERROR_LIKE
                and base.value > 0
                and delta / base.value > t.slice_error_increase_max
            ):
                out.append(_finding(
                    "slice-error-increase", Severity.WARNING, f"slice:{sl.name}", delta / base.value, t.slice_error_increase_max, "observed relative increase > threshold",
                    f"On slice {sl.name!r} (n={sl.n_samples}) {mid} is {_fmt(delta / base.value)} (relative) above overall; threshold {_fmt(t.slice_error_increase_max)}.",
                    (mid,), (f"slice.{sl.name}.metric.{mid}", f"metric.{mid}"), ("evaluation/slices.json",),
                ))  # fmt: skip
    _ = (errors, n_samples)  # reserved: rules over error counts arrive with failure discovery
    return out

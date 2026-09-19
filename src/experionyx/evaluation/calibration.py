"""Confidence and calibration analysis. See docs/evaluation.md for the exact methodology.

Interpretation: *top-label (confidence) calibration*. The confidence of a sample is the largest
class probability; "correct" means the model's predicted label equals the true label. Bins are
equal-width over [0, 1]: bin k = [k/B, (k+1)/B), the last bin closed at 1. ECE is the
sample-weighted mean |accuracy - mean confidence| over non-empty bins; MCE the maximum; the Brier
score is the multiclass sum over classes of (p_k - 1[y=k])^2 averaged over samples. The confidence
source is recorded and never assumed to be calibrated.
"""

import math
from collections.abc import Sequence

from experionyx.evaluation.results import (
    CalibrationBin,
    CalibrationResult,
    ConfidenceResult,
    Label,
    Scalar,
    Status,
)

CONFIDENCE_HISTOGRAM_BINS = 10
_TOLERANCE = 1e-9


def top_confidence(scores: Sequence[Sequence[float]]) -> list[float]:
    return [max(row) for row in scores]


def _bin_index(conf: float, n_bins: int) -> int:
    return min(int(conf * n_bins), n_bins - 1)


def _valid(confidences: Sequence[float]) -> str | None:
    if any(not math.isfinite(c) or c < -_TOLERANCE or c > 1 + _TOLERANCE for c in confidences):
        return "confidences must be finite probabilities in [0, 1]"
    return None


def confidence_analysis(
    confidences: Sequence[float] | None,
    correct: Sequence[bool],
    *,
    source: str | None,
    high: float,
    low: float,
) -> ConfidenceResult:
    if confidences is None:
        return ConfidenceResult(
            Status.UNSUPPORTED, source, reason="the model exposes no probability/score output"
        )
    if not confidences:
        return ConfidenceResult(Status.UNDEFINED, source, reason="no samples")
    problem = _valid(confidences)
    if problem:
        return ConfidenceResult(Status.INVALID, source, reason=problem)
    hist = [0] * CONFIDENCE_HISTOGRAM_BINS
    right: list[float] = []
    wrong: list[float] = []
    for c, ok in zip(confidences, correct, strict=True):
        hist[_bin_index(min(max(c, 0.0), 1.0), CONFIDENCE_HISTOGRAM_BINS)] += 1
        (right if ok else wrong).append(c)
    warnings = []
    if not wrong:
        warnings.append("no incorrect predictions: confidence on errors is undefined")
    if not right:
        warnings.append("no correct predictions: confidence on correct samples is undefined")
    return ConfidenceResult(
        status=Status.COMPUTED,
        source=source,
        n_samples=len(confidences),
        distribution=tuple(hist),
        mean_confidence_correct=sum(right) / len(right) if right else None,
        mean_confidence_incorrect=sum(wrong) / len(wrong) if wrong else None,
        n_correct=len(right),
        n_incorrect=len(wrong),
        high_confidence_threshold=high,
        low_confidence_threshold=low,
        high_confidence_errors=sum(c >= high for c in wrong),
        low_confidence_correct=sum(c < low for c in right),
        warnings=tuple(warnings),
    )


def calibration_analysis(
    scores: Sequence[Sequence[float]] | None,
    y_true: Sequence[Scalar],
    y_pred: Sequence[Scalar],
    classes: tuple[Label, ...] | None,
    *,
    n_bins: int,
    source: str | None,
) -> CalibrationResult:
    if scores is None or classes is None:
        return CalibrationResult(
            Status.UNSUPPORTED,
            source,
            reason="calibration needs per-class probabilities, which are not available",
        )
    n = len(y_true)
    if n == 0:
        return CalibrationResult(Status.UNDEFINED, source, reason="no samples")
    conf = top_confidence(scores)
    problem = _valid(conf)
    if problem:
        return CalibrationResult(Status.INVALID, source, reason=problem)
    index: dict[Scalar, int] = {c: i for i, c in enumerate(classes)}
    if any(t not in index for t in y_true):
        return CalibrationResult(Status.INVALID, source, reason="targets outside the model classes")
    hits = [t == p for t, p in zip(y_true, y_pred, strict=True)]
    members: list[list[int]] = [[] for _ in range(n_bins)]
    for i, c in enumerate(conf):
        members[_bin_index(min(max(c, 0.0), 1.0), n_bins)].append(i)
    bins = []
    ece = 0.0
    mce = 0.0
    for k, idx in enumerate(members):
        lo, hi = k / n_bins, (k + 1) / n_bins
        if not idx:
            bins.append(CalibrationBin(lo, hi, 0, None, None, None))
            continue
        mean_conf = sum(conf[i] for i in idx) / len(idx)
        acc = sum(hits[i] for i in idx) / len(idx)
        gap = abs(acc - mean_conf)
        ece += len(idx) / n * gap
        mce = max(mce, gap)
        bins.append(CalibrationBin(lo, hi, len(idx), mean_conf, acc, gap))
    brier = (
        sum(
            sum((p - (1.0 if j == index[t] else 0.0)) ** 2 for j, p in enumerate(row))
            for t, row in zip(y_true, scores, strict=True)
        )
        / n
    )
    warnings = []
    if n < 100:
        warnings.append(f"only {n} samples: bin estimates are noisy; treat ECE/MCE as indicative")
    disagree = sum(
        1 for row, p in zip(scores, y_pred, strict=True) if classes[row.index(max(row))] != p
    )
    if disagree:
        warnings.append(f"{disagree} prediction(s) differ from the arg-max of the scores")
    return CalibrationResult(
        status=Status.COMPUTED,
        confidence_source=source,
        n_bins=n_bins,
        n_samples=n,
        bins=tuple(bins),
        ece=ece,
        mce=mce,
        brier_score=brier,
        warnings=tuple(warnings),
    )

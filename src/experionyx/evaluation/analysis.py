"""Error records, class imbalance, latency and slice analysis (pure Python, framework-free)."""

import heapq
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace

from experionyx.evaluation.config import (
    ErrorConfig,
    SliceCondition,
    SliceKind,
    SliceSpec,
)
from experionyx.evaluation.metrics import (
    MetricInputs,
    MetricSpec,
    compute_metric,
)
from experionyx.evaluation.results import (
    ErrorKind,
    ErrorRecord,
    ErrorSummary,
    ImbalanceResult,
    Interval,
    Label,
    LatencySummary,
    MetricContext,
    MetricResult,
    Scalar,
    SliceResult,
    Status,
)

IMBALANCE_NOTE = (
    "Accuracy can be misleading under class imbalance: a model that always predicts the "
    "majority class scores the majority proportion. Compare balanced accuracy and per-class "
    "recall. No rebalancing is applied (this is analysis, not intervention)."
)


# --- errors -----------------------------------------------------------------------------------


def classification_errors(
    indices: Sequence[int],
    y_true: Sequence[Scalar],
    y_pred: Sequence[Scalar],
    confidences: Sequence[float] | None,
    classes: tuple[Label, ...] | None,
    cfg: ErrorConfig,
    *,
    positive_label: Scalar | None,
    high: float,
    low: float,
) -> tuple[list[ErrorRecord], ErrorSummary]:
    """One record per sample with at least one error kind. Counts are complete; the retained
    records are the first `max_records` in sample order."""
    binary = classes is not None and len(classes) == 2
    positive = (
        positive_label
        if positive_label is not None
        else (classes[1] if binary and classes else None)
    )
    counts = {k.value: 0 for k in ErrorKind if k is not ErrorKind.RESIDUAL}
    records: list[ErrorRecord] = []
    flagged = 0
    for j, (t, p) in enumerate(zip(y_true, y_pred, strict=True)):
        conf = None if confidences is None else confidences[j]
        kinds: list[ErrorKind] = []
        if t != p:
            if binary and positive is not None:
                kinds.append(
                    ErrorKind.FALSE_POSITIVE if p == positive else ErrorKind.FALSE_NEGATIVE
                )
            else:
                kinds.append(ErrorKind.INCORRECT_CLASS)
            if conf is not None and conf >= high:
                kinds.append(ErrorKind.HIGH_CONFIDENCE_INCORRECT)
        elif conf is not None and conf < low:
            kinds.append(ErrorKind.LOW_CONFIDENCE_CORRECT)
        if not kinds:
            continue
        for k in kinds:
            counts[k.value] += 1
        flagged += 1
        if cfg.record and len(records) < cfg.max_records:
            records.append(ErrorRecord(indices[j], tuple(kinds), t, p, conf))
    summary = ErrorSummary(
        status=Status.COMPUTED,
        total_samples=len(y_true),
        counts=counts,
        recorded=len(records),
        truncated=cfg.record and flagged > len(records),
    )
    return records, summary


def regression_errors(
    indices: Sequence[int], y_true: Sequence[float], y_pred: Sequence[float], cfg: ErrorConfig
) -> tuple[list[ErrorRecord], ErrorSummary]:
    """Residual statistics over all samples; the `max_records` largest absolute errors retained
    (ties broken by sample order, so the selection is deterministic)."""
    res = [float(p) - float(t) for t, p in zip(y_true, y_pred, strict=True)]
    absr = [abs(r) for r in res]
    top = (
        heapq.nsmallest(cfg.max_records, range(len(res)), key=lambda j: (-absr[j], j))
        if cfg.record
        else []
    )
    records = [
        ErrorRecord(
            sample_index=indices[j],
            kinds=(ErrorKind.RESIDUAL,),
            true=y_true[j],
            predicted=y_pred[j],
            absolute_error=absr[j],
            signed_error=res[j],
            relative_error=None if y_true[j] == 0 else absr[j] / abs(y_true[j]),
        )
        for j in sorted(top)
    ]
    stats: dict[str, float] = {
        "mean_signed_error": sum(res) / len(res),
        "mean_absolute_error": sum(absr) / len(res),
        "max_absolute_error": max(absr),
        "residual_std": statistics.pstdev(res),
        "target_std": statistics.pstdev([float(t) for t in y_true]),
    }
    return records, ErrorSummary(
        status=Status.COMPUTED,
        total_samples=len(res),
        counts={ErrorKind.RESIDUAL.value: len(res)},
        recorded=len(records),
        truncated=len(records) < len(res),
        stats=stats,
    )


# --- class imbalance --------------------------------------------------------------------------


def imbalance_analysis(y_true: Sequence[Scalar]) -> ImbalanceResult:
    if not y_true:
        return ImbalanceResult(Status.UNDEFINED, {}, {}, None, None, None, reason="no samples")
    counts: dict[Scalar, int] = {}
    for t in y_true:
        counts[t] = counts.get(t, 0) + 1
    n = len(y_true)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))
    majority, minority = ordered[0], ordered[-1]
    return ImbalanceResult(
        status=Status.COMPUTED,
        class_counts={str(k): v for k, v in ordered},
        class_proportions={str(k): v / n for k, v in ordered},
        majority_class=majority[0],  # type: ignore[arg-type]
        minority_class=minority[0],  # type: ignore[arg-type]
        imbalance_ratio=majority[1] / minority[1],
        note=IMBALANCE_NOTE,
    )


# --- latency ----------------------------------------------------------------------------------


def latency_summary(
    batch_seconds: Sequence[float], n_samples: int, load_seconds: float | None
) -> LatencySummary:
    if not batch_seconds:
        return LatencySummary(
            Status.UNDEFINED, load_seconds, n_samples, 0, reason="no batches were timed"
        )
    if any(not math.isfinite(s) or s < 0 for s in batch_seconds):
        return LatencySummary(
            Status.INVALID, load_seconds, n_samples, len(batch_seconds), reason="invalid timings"
        )
    ordered = sorted(batch_seconds)
    total = sum(batch_seconds)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]  # nearest-rank
    warnings = [
        f"{len(batch_seconds)} batches / {n_samples} samples: a small sample, not a benchmark"
    ]
    if len(batch_seconds) < 20:
        warnings.append("fewer than 20 batches: p95 equals or nearly equals the maximum")
    return LatencySummary(
        status=Status.COMPUTED,
        load_seconds=load_seconds,
        n_samples=n_samples,
        n_batches=len(batch_seconds),
        total_seconds=total,
        mean_batch_seconds=total / len(batch_seconds),
        median_batch_seconds=statistics.median(ordered),
        p95_batch_seconds=p95,
        max_batch_seconds=ordered[-1],
        throughput_samples_per_second=None if total == 0 else n_samples / total,
        warnings=tuple(warnings),
    )


# --- slices -----------------------------------------------------------------------------------


def _matches(
    cond: SliceCondition,
    i: int,
    y_true: Sequence[Scalar],
    y_pred: Sequence[Scalar],
    columns: Mapping[str, Sequence[float]],
) -> bool:
    if cond.kind is SliceKind.TARGET_EQUALS:
        return y_true[i] == cond.value
    if cond.kind is SliceKind.PREDICTED_EQUALS:
        return y_pred[i] == cond.value
    assert cond.field is not None  # noqa: S101  # validated by SliceCondition
    x = columns[cond.field][i]
    if cond.kind is SliceKind.FEATURE_EQUALS:
        return x == cond.value
    return (cond.low is None or x >= cond.low) and (cond.high is None or x < cond.high)


def slice_indices(
    spec: SliceSpec,
    y_true: Sequence[Scalar],
    y_pred: Sequence[Scalar],
    columns: Mapping[str, Sequence[float]],
) -> list[int]:
    return [
        i
        for i in range(len(y_true))
        if all(_matches(c, i, y_true, y_pred, columns) for c in spec.conditions)
    ]


def evaluate_slice(
    spec: SliceSpec,
    idx: Sequence[int],
    inp: MetricInputs,
    metric_specs: Sequence[MetricSpec],
    overall: Mapping[str, MetricResult],
    context: MetricContext,
    interval_fn: Callable[[MetricSpec, MetricInputs], Interval | None] | None = None,
) -> SliceResult:
    """Scalar metrics on the slice using the same metric engine, plus deltas vs overall."""
    sub = MetricInputs(
        inp.task,
        [inp.y_true[i] for i in idx],
        [inp.y_pred[i] for i in idx],
        None if inp.scores is None else [inp.scores[i] for i in idx],
        inp.classes,
    )
    results: list[MetricResult] = []
    deltas: dict[str, float | None] = {}
    for m in metric_specs:
        if not m.scalar:
            continue
        outcome = compute_metric(m, sub)
        result = MetricResult(
            metric_id=m.id,
            metric_version=m.version,
            name=m.name,
            status=outcome.status,
            n_samples=len(idx),
            context=context,
            scale=m.scale,
            higher_is_better=m.higher_is_better,
            value=outcome.value,
            classes=inp.classes,
            reason=outcome.reason,
            warnings=outcome.warnings,
        )
        if interval_fn is not None and outcome.status is Status.COMPUTED:
            result = replace(result, interval=interval_fn(m, sub))
        results.append(result)
        base = overall.get(m.id)
        deltas[m.id] = (
            result.value - base.value
            if result.value is not None and base is not None and base.value is not None
            else None
        )
    return SliceResult(
        name=spec.name,
        conditions=spec.conditions,
        n_samples=len(idx),
        fraction_of_samples=len(idx) / len(inp.y_true) if inp.y_true else 0.0,
        metrics=tuple(results),
        deltas_vs_overall=deltas,
    )

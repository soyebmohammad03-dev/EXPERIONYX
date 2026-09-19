"""Structured comparison of two evaluation results. Reports differences and facts (such as
interval overlap); it never ranks, scores or declares a winner, and it does no significance
testing. Later phases extend this with proper statistical comparison."""

from collections.abc import Mapping
from dataclasses import dataclass

from experionyx.errors import EvaluationError
from experionyx.evaluation.results import EvaluationResult, Interval, MetricResult, Status


@dataclass(frozen=True)
class MetricDelta:
    metric_id: str
    a: float | None
    b: float | None
    delta: float | None  # b - a
    status_a: Status
    status_b: Status
    interval_a: tuple[float, float] | None
    interval_b: tuple[float, float] | None
    intervals_overlap: bool | None  # a fact about the two intervals, not a significance test


@dataclass(frozen=True)
class ClassDelta:
    label: str
    support_a: int | None
    support_b: int | None
    precision: float | None  # b - a
    recall: float | None
    f1: float | None


@dataclass(frozen=True)
class SliceComparison:
    name: str
    n_a: int | None
    n_b: int | None
    metrics: tuple[MetricDelta, ...]


@dataclass(frozen=True)
class EvaluationComparison:
    run_a: str
    run_b: str
    task: str
    same_model: bool
    same_dataset: bool
    same_split: bool
    same_config: bool
    warnings: tuple[str, ...]
    metrics: tuple[MetricDelta, ...]
    classes: tuple[ClassDelta, ...]
    slices: tuple[SliceComparison, ...]
    calibration: Mapping[str, float | None]  # b - a for ece, mce, brier_score
    latency: Mapping[str, float | None]  # b - a; timings are noisy and hardware dependent


def _diff(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else b - a


def _iv(m: MetricResult | None) -> tuple[float, float] | None:
    i: Interval | None = None if m is None else m.interval
    if i is None or i.status is not Status.COMPUTED or i.lower is None or i.upper is None:
        return None
    return (i.lower, i.upper)


def _metric_delta(mid: str, a: MetricResult | None, b: MetricResult | None) -> MetricDelta:
    ia, ib = _iv(a), _iv(b)
    overlap = None if ia is None or ib is None else (ia[0] <= ib[1] and ib[0] <= ia[1])
    return MetricDelta(
        mid,
        None if a is None else a.value,
        None if b is None else b.value,
        _diff(None if a is None else a.value, None if b is None else b.value),
        Status.UNSUPPORTED if a is None else a.status,
        Status.UNSUPPORTED if b is None else b.status,
        ia,
        ib,
        overlap,
    )


def compare_evaluations(a: EvaluationResult, b: EvaluationResult) -> EvaluationComparison:
    if a.task != b.task:
        raise EvaluationError(f"cannot compare a {a.task} evaluation with a {b.task} one")
    warnings = []
    same_model = a.context.model_fingerprint == b.context.model_fingerprint
    same_dataset = a.context.dataset_fingerprint == b.context.dataset_fingerprint
    same_split = a.context.split == b.context.split
    if not same_dataset:
        warnings.append("different datasets: differences may reflect the data, not the models")
    if not same_split:
        warnings.append("different splits")
    if a.n_samples != b.n_samples:
        warnings.append(f"different sample counts ({a.n_samples} vs {b.n_samples})")
    if a.classes != b.classes:
        warnings.append("different class sets")
    ma = {m.metric_id: m for m in a.metrics}
    mb = {m.metric_id: m for m in b.metrics}
    metrics = tuple(_metric_delta(k, ma.get(k), mb.get(k)) for k in sorted({*ma, *mb}))
    ca = {str(c.label): c for c in (a.confusion.per_class if a.confusion else ())}
    cb = {str(c.label): c for c in (b.confusion.per_class if b.confusion else ())}
    classes = tuple(
        ClassDelta(
            k,
            ca[k].support if k in ca else None,
            cb[k].support if k in cb else None,
            _diff(ca[k].precision if k in ca else None, cb[k].precision if k in cb else None),
            _diff(ca[k].recall if k in ca else None, cb[k].recall if k in cb else None),
            _diff(ca[k].f1 if k in ca else None, cb[k].f1 if k in cb else None),
        )
        for k in sorted({*ca, *cb})
    )
    sa = {s.name: s for s in a.slices}
    sb = {s.name: s for s in b.slices}
    slices = []
    for name in sorted({*sa, *sb}):
        ra = {m.metric_id: m for m in sa[name].metrics} if name in sa else {}
        rb = {m.metric_id: m for m in sb[name].metrics} if name in sb else {}
        slices.append(
            SliceComparison(
                name,
                sa[name].n_samples if name in sa else None,
                sb[name].n_samples if name in sb else None,
                tuple(_metric_delta(k, ra.get(k), rb.get(k)) for k in sorted({*ra, *rb})),
            )
        )
    la, lb = a.latency, b.latency
    return EvaluationComparison(
        run_a=a.context.run_id,
        run_b=b.context.run_id,
        task=str(a.task),
        same_model=same_model,
        same_dataset=same_dataset,
        same_split=same_split,
        same_config=a.config == b.config,
        warnings=tuple(warnings),
        metrics=metrics,
        classes=classes,
        slices=tuple(slices),
        calibration={
            "ece": _diff(a.calibration.ece, b.calibration.ece),
            "mce": _diff(a.calibration.mce, b.calibration.mce),
            "brier_score": _diff(a.calibration.brier_score, b.calibration.brier_score),
        },
        latency={
            "mean_batch_seconds": _diff(la.mean_batch_seconds, lb.mean_batch_seconds),
            "median_batch_seconds": _diff(la.median_batch_seconds, lb.median_batch_seconds),
            "p95_batch_seconds": _diff(la.p95_batch_seconds, lb.p95_batch_seconds),
            "throughput_samples_per_second": _diff(
                la.throughput_samples_per_second, lb.throughput_samples_per_second
            ),
        },
    )

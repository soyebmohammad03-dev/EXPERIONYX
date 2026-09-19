"""Pure analysis of a fault experiment: per-trial degradation, per-point aggregation over seeds,
sweep series, effect assessment and severity. Inputs are stored EvaluationResults of real runs;
this module does no I/O and invents no values."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from experionyx.adapters.capabilities import TaskType
from experionyx.errors import FaultError
from experionyx.evaluation.results import EvaluationResult, MetricResult, Status
from experionyx.faults.degradation import (
    Aggregate,
    Direction,
    FaultEffectAssessment,
    FaultSeverity,
    LatencyDegradation,
    MetricDegradation,
    aggregate,
    assess_effect,
    build_severity,
    direction_of,
    measure_degradation,
    measure_latency,
)
from experionyx.faults.design import FaultDesign


@dataclass(frozen=True)
class TrialInput:
    """What is known about one treatment trial (read from the registry and artifacts)."""

    point_index: int
    repeat_index: int
    parameter_value: float | None
    seed: int
    fault_id: str
    family_id: str
    status: str  # COMPLETED | FAILED | SKIPPED
    run_id: str | None = None
    reason: str | None = None
    affected_samples: int | None = None
    n_samples: int | None = None
    faulted: EvaluationResult | None = None
    apply_seconds: float | None = None
    evaluation_seconds: float | None = None
    total_seconds: float | None = None


@dataclass(frozen=True)
class TrialResult:
    point_index: int
    repeat_index: int
    parameter_name: str | None
    parameter_value: float | None
    seed: int
    fault_id: str
    family_id: str
    status: str
    run_id: str | None
    reason: str | None
    n_samples: int | None
    affected_samples: int | None
    affected_fraction: float | None
    degradations: tuple[MetricDegradation, ...]
    latency: LatencyDegradation | None
    apply_seconds: float | None  # fault application time
    evaluation_seconds: float | None  # evaluation time excluding fault application
    total_seconds: float | None  # whole treatment run


@dataclass(frozen=True)
class SeriesRow:
    """One point of a degradation series = one real faulted evaluation (never interpolated)."""

    parameter_name: str | None
    parameter_value: float | None
    seed: int
    baseline: float
    faulted: float
    absolute_delta: float
    relative_delta: float | None
    n_samples: int | None
    run_id: str
    fault_id: str


@dataclass(frozen=True)
class PointResult:
    point_index: int
    parameter_name: str | None
    parameter_value: float | None
    family_id: str
    n_trials: int
    n_completed: int
    primary_faulted: Aggregate  # the faulted metric across seeds
    primary_deterioration: Aggregate  # > 0 means worse, across seeds
    metric_deterioration: Mapping[str, Aggregate]
    assessment: FaultEffectAssessment
    severity: FaultSeverity


@dataclass(frozen=True)
class FaultAnalysisResult:
    baseline_run_id: str
    primary_metric: str
    primary_direction: Direction
    baseline_value: float | None
    points: tuple[PointResult, ...]
    trials: tuple[TrialResult, ...]
    series: tuple[SeriesRow, ...]
    failed_trials: int
    skipped_trials: int
    limits: Mapping[str, object]
    warnings: tuple[str, ...] = ()


def default_primary_metric(task: TaskType) -> str:
    return "accuracy" if task is TaskType.CLASSIFICATION else "rmse"


def _numeric_parameters(
    fault: Mapping[str, object], sweep_name: str | None, value: float | None
) -> dict[str, float]:
    out: dict[str, float] = {}
    comps = fault.get("components")
    leaves = list(comps) if isinstance(comps, list | tuple) and comps else [fault]
    for i, leaf in enumerate(leaves):
        params = leaf.get("parameters", {}) if isinstance(leaf, Mapping) else {}
        if not isinstance(params, Mapping):
            continue
        prefix = f"component{i}." if len(leaves) > 1 else ""
        for k, v in params.items():
            if isinstance(v, int | float) and not isinstance(v, bool):
                out[prefix + k] = float(v)
    if sweep_name is not None and value is not None:
        out[sweep_name] = value
    return out


def _uncertainty_excludes_zero(
    agg: Aggregate, base: MetricResult | None, fault: MetricResult | None
) -> bool | None:
    if agg.n > 1:
        return None if agg.ci_lower is None else agg.ci_lower > 0
    bi, fi = (base.interval if base else None), (fault.interval if fault else None)
    if (
        bi is None
        or fi is None
        or bi.status is not Status.COMPUTED
        or fi.status is not Status.COMPUTED
    ):
        return None
    if bi.lower is None or bi.upper is None or fi.lower is None or fi.upper is None or base is None:
        return None
    return fi.upper < bi.lower if base.higher_is_better else fi.lower > bi.upper


def analyze(
    design: FaultDesign,
    baseline: EvaluationResult,
    baseline_run_id: str,
    trials: Sequence[TrialInput],
) -> FaultAnalysisResult:
    primary = design.primary_metric or default_primary_metric(baseline.task)
    base_metric = next((m for m in baseline.metrics if m.metric_id == primary), None)
    if base_metric is None:
        raise FaultError(f"primary metric {primary!r} was not evaluated in the baseline")
    sweep_name = design.sweep.parameter if design.sweep else None

    def agg(values: Sequence[float]) -> Aggregate:
        return aggregate(
            values,
            confidence=design.aggregation_confidence,
            resamples=design.aggregation_resamples,
            seed=design.aggregation_seed,
        )

    results: list[TrialResult] = []
    series: list[SeriesRow] = []
    by_point: dict[int, list[TrialInput]] = {}
    for t in sorted(trials, key=lambda x: (x.point_index, x.repeat_index)):
        by_point.setdefault(t.point_index, []).append(t)
        degs: tuple[MetricDegradation, ...] = ()
        lat = None
        frac = (
            None
            if not t.n_samples or t.affected_samples is None
            else t.affected_samples / t.n_samples
        )
        if t.faulted is not None:
            degs = measure_degradation(baseline, t.faulted)
            lat = measure_latency(baseline, t.faulted)
            d = next((x for x in degs if x.metric_id == primary), None)
            if (
                d is not None
                and d.baseline is not None
                and d.faulted is not None
                and d.absolute_delta is not None
                and t.run_id
            ):
                series.append(
                    SeriesRow(
                        sweep_name,
                        t.parameter_value,
                        t.seed,
                        d.baseline,
                        d.faulted,
                        d.absolute_delta,
                        d.relative_delta,
                        t.n_samples,
                        t.run_id,
                        t.fault_id,
                    )
                )
        results.append(
            TrialResult(
                t.point_index,
                t.repeat_index,
                sweep_name,
                t.parameter_value,
                t.seed,
                t.fault_id,
                t.family_id,
                t.status,
                t.run_id,
                t.reason,
                t.n_samples,
                t.affected_samples,
                frac,
                degs,
                lat,
                t.apply_seconds,
                t.evaluation_seconds,
                t.total_seconds,
            )
        )

    points: list[PointResult] = []
    warnings: list[str] = []
    for idx in sorted(by_point):
        group = by_point[idx]
        done = [t for t in group if t.status == "COMPLETED" and t.faulted is not None]
        trial_results = [r for r in results if r.point_index == idx]
        per_metric: dict[str, list[float]] = {}
        rel: list[float] = []
        faulted_vals: list[float] = []
        for r in trial_results:
            for d in r.degradations:
                if d.deterioration is not None:
                    per_metric.setdefault(d.metric_id, []).append(d.deterioration)
                if d.metric_id == primary:
                    if d.faulted is not None:
                        faulted_vals.append(d.faulted)
                    if d.relative_deterioration is not None:
                        rel.append(d.relative_deterioration)
        det_vals = per_metric.get(primary, [])
        det_agg = agg(det_vals)
        fault_agg = agg(faulted_vals)
        fault_metric = None
        if len(done) == 1 and done[0].faulted is not None:
            fault_metric = next(
                (m for m in done[0].faulted.metrics if m.metric_id == primary), None
            )
        fractions = [r.affected_fraction for r in trial_results if r.affected_fraction is not None]
        mean_frac = sum(fractions) / len(fractions) if fractions else None
        mean_rel = sum(rel) / len(rel) if rel and len(rel) == len(det_vals) else None
        value = group[0].parameter_value
        assessment = (
            assess_effect(
                primary,
                det_agg.mean,
                mean_rel,
                thresholds=design.thresholds,
                affected_fraction=mean_frac,
                uncertainty_excludes_zero=_uncertainty_excludes_zero(
                    det_agg, base_metric, fault_metric
                ),
            )
            if det_vals
            else assess_effect(
                primary,
                None,
                None,
                thresholds=design.thresholds,
                affected_fraction=mean_frac,
                uncertainty_excludes_zero=None,
            )
        )
        severity = build_severity(
            _numeric_parameters(design.fault, sweep_name, value),
            mean_frac,
            det_agg,
            [(baseline, t.faulted) for t in done if t.faulted is not None],
            primary,
        )
        points.append(
            PointResult(
                idx,
                sweep_name,
                value,
                group[0].family_id,
                len(group),
                len(done),
                fault_agg,
                det_agg,
                {k: agg(v) for k, v in sorted(per_metric.items())},
                assessment,
                severity,
            )
        )
        if len(done) < len(group):
            warnings.append(
                f"point {idx}: {len(group) - len(done)} of {len(group)} trial(s) did not complete; aggregates use completed trials only"
            )
    failed = sum(t.status == "FAILED" for t in trials)
    skipped = sum(t.status == "SKIPPED" for t in trials)
    return FaultAnalysisResult(
        baseline_run_id, primary, direction_of(base_metric.higher_is_better), base_metric.value,
        tuple(points), tuple(results), tuple(series), failed, skipped,
        _limits_dict(design), tuple(warnings),
    )  # fmt: skip


def _limits_dict(design: FaultDesign) -> dict[str, object]:
    from experionyx.domain import to_jsonable

    data = to_jsonable(design.limits)
    assert isinstance(data, dict)  # noqa: S101
    return data

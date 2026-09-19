"""Degradation measurement, directionality, aggregation, effect assessment, severity."""

import math
import statistics
from pathlib import Path

import pytest

from eval_helpers import class_data, eval_world, load_evaluation
from experionyx.errors import ValidationError
from experionyx.evaluation.config import EvaluationConfig
from experionyx.evaluation.results import MetricContext, MetricResult, Status
from experionyx.faults.degradation import (
    Change,
    Direction,
    EffectClass,
    FaultThresholds,
    _degradation,
    aggregate,
    assess_effect,
    describe_interaction,
    direction_of,
    measure_degradation,
    measure_latency,
)

CTX = MetricContext("run_x", "test")


def metric(
    mid: str, value: float | None, better: bool | None, status: Status = Status.COMPUTED
) -> MetricResult:
    return MetricResult(
        mid,
        "1.0.0",
        mid,
        status,
        10,
        CTX,
        "",
        better,
        value=value,
        reason=None if status is Status.COMPUTED else "why",
    )


# --- directionality ----------------------------------------------------------------------------


def test_direction_is_explicit_per_metric() -> None:
    assert direction_of(True) is Direction.HIGHER_IS_BETTER
    assert direction_of(False) is Direction.LOWER_IS_BETTER
    assert direction_of(None) is Direction.NEUTRAL


def test_higher_is_better_metrics_deteriorate_when_they_fall() -> None:
    d = _degradation("accuracy", metric("accuracy", 0.9, True), metric("accuracy", 0.7, True))
    assert d.direction is Direction.HIGHER_IS_BETTER
    assert d.absolute_delta == pytest.approx(-0.2)  # faulted - baseline
    assert d.relative_delta == pytest.approx(-0.2 / 0.9)
    assert d.deterioration == pytest.approx(0.2)  # positive means worse
    assert d.relative_deterioration == pytest.approx(0.2 / 0.9)
    assert d.change is Change.DETERIORATED


def test_lower_is_better_metrics_deteriorate_when_they_rise() -> None:
    d = _degradation("mae", metric("mae", 2.0, False), metric("mae", 3.0, False))
    assert d.absolute_delta == pytest.approx(1.0)
    assert d.deterioration == pytest.approx(1.0)  # the same +1.0 is bad here, unlike for accuracy
    assert d.relative_deterioration == pytest.approx(0.5)
    assert d.change is Change.DETERIORATED


def test_improvements_and_no_change_are_reported_as_such() -> None:
    better = _degradation("accuracy", metric("accuracy", 0.7, True), metric("accuracy", 0.8, True))
    assert better.change is Change.IMPROVED
    assert better.deterioration == pytest.approx(-0.1)
    same = _degradation("mae", metric("mae", 2.0, False), metric("mae", 2.0, False))
    assert (same.change, same.deterioration, same.absolute_delta) == (Change.UNCHANGED, 0.0, 0.0)
    fell = _degradation("mae", metric("mae", 2.0, False), metric("mae", 1.0, False))
    assert fell.change is Change.IMPROVED  # a lower error is not degradation


def test_neutral_metrics_report_change_but_never_degradation() -> None:
    d = _degradation("weird", metric("weird", 1.0, None), metric("weird", 2.0, None))
    assert d.direction is Direction.NEUTRAL
    assert d.absolute_delta == 1.0
    assert (d.deterioration, d.relative_deterioration, d.change) == (
        None,
        None,
        Change.UNDETERMINED,
    )


def test_zero_baseline_has_no_relative_change_and_never_divides_by_zero() -> None:
    d = _degradation("mae", metric("mae", 0.0, False), metric("mae", 1.0, False))
    assert d.absolute_delta == 1.0
    assert (d.relative_delta, d.relative_deterioration) == (None, None)
    assert d.deterioration == 1.0


def test_missing_or_unavailable_sides_are_undetermined_with_a_reason_never_zero() -> None:
    a = _degradation(
        "roc_auc", metric("roc_auc", 0.9, True), metric("roc_auc", None, True, Status.UNSUPPORTED)
    )
    assert (a.absolute_delta, a.deterioration, a.change) == (None, None, Change.UNDETERMINED)
    assert "faulted UNSUPPORTED" in (a.reason or "")
    b = _degradation("accuracy", metric("accuracy", 0.9, True), None)
    assert "not evaluated" in (b.reason or "")
    c = _degradation(
        "accuracy", metric("accuracy", None, True, Status.UNDEFINED), metric("accuracy", 0.5, True)
    )
    assert "baseline UNDEFINED" in (c.reason or "")


# --- on real evaluation results ----------------------------------------------------------------


def test_measure_degradation_on_two_real_evaluations(tmp_path: Path) -> None:
    data = class_data(40, noise_every=5)
    cfg = EvaluationConfig(split="test")
    wa = eval_world(tmp_path / "a", model={"threshold": 5.0, "proba": True}, data=data, config=cfg)
    wb = eval_world(tmp_path / "b", model={"threshold": 8.0, "proba": True}, data=data, config=cfg)
    a, b = load_evaluation(wa.store, wa.run().run), load_evaluation(wb.store, wb.run().run)
    degs = {d.metric_id: d for d in measure_degradation(a, b)}
    assert "confusion_matrix" not in degs  # structural metrics have no scalar delta
    acc = degs["accuracy"]
    assert acc.baseline == 0.8
    assert acc.faulted is not None
    assert acc.absolute_delta == pytest.approx(acc.faulted - 0.8)
    assert acc.deterioration == pytest.approx(-acc.absolute_delta)
    assert degs["log_loss"].direction is Direction.LOWER_IS_BETTER
    assert degs["roc_auc"].direction is Direction.HIGHER_IS_BETTER
    lat = measure_latency(a, b)
    assert lat.direction is Direction.LOWER_IS_BETTER
    assert "exclude fault-application time" in lat.note
    self_deg = measure_degradation(a, a)
    assert all(d.absolute_delta in (0.0, None) for d in self_deg)
    assert all(d.change in (Change.UNCHANGED, Change.UNDETERMINED) for d in self_deg)


def test_regression_error_metrics_use_lower_is_better(tmp_path: Path) -> None:
    rows = [[float(i)] for i in range(30)]
    data = {
        "rows": rows,
        "targets": [float(i) for i in range(30)],
        "task": "REGRESSION",
        "splits": {"test": list(range(15, 30))},
        "feature_names": ["x"],
    }
    cfg = EvaluationConfig(split="test")
    wa = eval_world(tmp_path / "a", model={"value": 20.0}, data=data, config=cfg, classifier=False)
    wb = eval_world(tmp_path / "b", model={"value": 60.0}, data=data, config=cfg, classifier=False)
    degs = {
        d.metric_id: d
        for d in measure_degradation(
            load_evaluation(wa.store, wa.run().run), load_evaluation(wb.store, wb.run().run)
        )
    }
    for mid in ("mae", "rmse", "mse"):
        assert degs[mid].direction is Direction.LOWER_IS_BETTER
        assert degs[mid].deterioration is not None
        assert degs[mid].deterioration > 0
        assert degs[mid].change is Change.DETERIORATED
    assert degs["r2"].direction is Direction.HIGHER_IS_BETTER
    assert degs["r2"].deterioration is not None
    assert degs["r2"].deterioration > 0  # R^2 fell


# --- aggregation over seeds --------------------------------------------------------------------


def test_aggregate_matches_reference_statistics_and_keeps_extremes() -> None:
    vals = [0.10, 0.14, 0.09, 0.20, 0.11, 0.13, 0.16, 0.12]
    agg = aggregate(vals, confidence=0.95, resamples=2000, seed=3)
    assert agg.n == 8
    assert agg.mean == pytest.approx(statistics.mean(vals))
    assert agg.median == pytest.approx(statistics.median(vals))
    assert agg.std == pytest.approx(statistics.stdev(vals))
    assert (agg.minimum, agg.maximum) == (0.09, 0.20)
    assert agg.ci_lower is not None
    assert agg.ci_upper is not None
    assert agg.ci_lower <= agg.mean <= agg.ci_upper
    assert agg.ci_lower >= min(vals)
    assert agg.ci_upper <= max(vals)
    assert agg.effect_size_dz == pytest.approx(statistics.mean(vals) / statistics.stdev(vals))
    assert any("not population-level" in w for w in agg.warnings)
    assert (agg.confidence, agg.resamples, agg.seed) == (0.95, 2000, 3)


def test_aggregate_interval_width_matches_the_standard_error_of_the_mean() -> None:
    vals = [math.sin(i * 12.9898) for i in range(40)]  # deterministic pseudo-spread
    agg = aggregate(vals, confidence=0.95, resamples=4000, seed=1)
    assert agg.std is not None
    assert agg.ci_lower is not None
    assert agg.ci_upper is not None
    expected = 2 * 1.96 * agg.std / math.sqrt(len(vals))
    assert (agg.ci_upper - agg.ci_lower) == pytest.approx(expected, rel=0.15)


def test_aggregate_is_deterministic_and_seed_dependent() -> None:
    vals = [math.sin(i * 1.7) ** 2 for i in range(25)]
    assert aggregate(vals, seed=1) == aggregate(vals, seed=1)
    a, b = aggregate(vals, resamples=300, seed=1), aggregate(vals, resamples=300, seed=2)
    assert (a.ci_lower, a.ci_upper) != (b.ci_lower, b.ci_upper)


def test_aggregate_edge_cases_are_explicit() -> None:
    single = aggregate([0.2])
    assert (single.n, single.mean, single.std, single.ci_lower, single.effect_size_dz) == (
        1,
        0.2,
        None,
        None,
        None,
    )
    assert any("single trial" in w for w in single.warnings)
    none = aggregate([])
    assert (none.n, none.mean, none.median) == (0, None, None)
    const = aggregate([0.5, 0.5, 0.5])
    assert (const.std, const.effect_size_dz) == (0.0, None)  # effect size undefined without spread
    assert const.ci_lower == const.ci_upper == 0.5
    assert any("unstable" in w for w in aggregate([0.1, 0.2]).warnings)
    for bad in ([0.1, float("nan")], [float("inf")]):
        with pytest.raises(ValidationError):
            aggregate(bad)
    with pytest.raises(ValidationError):
        aggregate([0.1], confidence=1.0)


# --- effect assessment (thresholds) ------------------------------------------------------------

T = FaultThresholds()  # abs 0.02 / rel 0.05 / substantial abs 0.10 / rel 0.20


@pytest.mark.parametrize(
    ("det", "rel", "frac", "unc", "expected"),
    [
        (None, None, 1.0, None, EffectClass.INCONCLUSIVE),  # metric missing
        (-0.05, -0.05, 1.0, True, EffectClass.NO_MEASURED_DEGRADATION),  # an improvement
        (0.0, 0.0, 1.0, True, EffectClass.NO_MEASURED_DEGRADATION),
        (0.01, 0.02, 1.0, True, EffectClass.NO_MEASURED_DEGRADATION),  # below both minima
        (0.05, 0.01, 1.0, True, EffectClass.NO_MEASURED_DEGRADATION),  # absolute ok, relative below
        (0.05, 0.10, 1.0, True, EffectClass.MEASURED_DEGRADATION),
        (0.05, 0.10, 1.0, None, EffectClass.MEASURED_DEGRADATION),  # uncertainty not checkable
        (0.05, 0.10, 1.0, False, EffectClass.INCONCLUSIVE),  # interval includes no change
        (0.15, 0.30, 1.0, True, EffectClass.SUBSTANTIAL_DEGRADATION),
        (0.15, 0.15, 1.0, True, EffectClass.MEASURED_DEGRADATION),  # relative below substantial
        (
            0.15,
            None,
            1.0,
            True,
            EffectClass.SUBSTANTIAL_DEGRADATION,
        ),  # baseline 0: relative unavailable
    ],
)
def test_effect_classification_follows_the_documented_rule(det, rel, frac, unc, expected) -> None:  # type: ignore[no-untyped-def]
    a = assess_effect(
        "accuracy", det, rel, thresholds=T, affected_fraction=frac, uncertainty_excludes_zero=unc
    )
    assert a.classification is expected
    assert a.basis
    assert a.thresholds == T
    assert a.primary_metric == "accuracy"


def test_small_affected_populations_are_inconclusive_not_negative() -> None:
    th = FaultThresholds(min_affected_fraction=0.2)
    a = assess_effect(
        "accuracy", 0.5, 0.5, thresholds=th, affected_fraction=0.05, uncertainty_excludes_zero=True
    )
    assert a.classification is EffectClass.INCONCLUSIVE
    assert "affected fraction" in a.reasons[0]


def test_uncertainty_requirement_can_be_switched_off() -> None:
    th = FaultThresholds(require_uncertainty_support=False)
    a = assess_effect(
        "accuracy",
        0.05,
        0.10,
        thresholds=th,
        affected_fraction=1.0,
        uncertainty_excludes_zero=False,
    )
    assert a.classification is EffectClass.MEASURED_DEGRADATION


def test_thresholds_are_configuration_and_validated() -> None:
    assert (
        FaultThresholds(
            min_absolute_degradation=0.3, substantial_absolute_degradation=0.5
        ).min_absolute_degradation
        == 0.3
    )
    for bad in (
        lambda: FaultThresholds(min_absolute_degradation=-0.1),
        lambda: FaultThresholds(min_relative_degradation=float("nan")),
        lambda: FaultThresholds(min_affected_fraction=1.5),
        lambda: FaultThresholds(min_absolute_degradation=0.5, substantial_absolute_degradation=0.1),
    ):
        with pytest.raises(ValidationError):
            bad()
    loose = assess_effect(
        "accuracy",
        0.05,
        0.10,
        thresholds=FaultThresholds(
            min_absolute_degradation=0.3, substantial_absolute_degradation=0.5
        ),
        affected_fraction=1.0,
        uncertainty_excludes_zero=True,
    )
    assert (
        loose.classification is EffectClass.NO_MEASURED_DEGRADATION
    )  # same data, different thresholds


def test_classification_vocabulary_never_says_failed() -> None:
    assert {c.value for c in EffectClass} == {
        "NO_MEASURED_DEGRADATION",
        "MEASURED_DEGRADATION",
        "SUBSTANTIAL_DEGRADATION",
        "INCONCLUSIVE",
    }


# --- interaction foundation --------------------------------------------------------------------


def test_interaction_description_is_descriptive_and_asserts_nothing() -> None:
    a, b, ab = aggregate([0.10, 0.12]), aggregate([0.05, 0.07]), aggregate([0.30, 0.32])
    d = describe_interaction("accuracy", a, b, ab)
    assert d.additive_reference == pytest.approx(0.17)
    assert d.difference_from_additive == pytest.approx(0.31 - 0.17)
    assert "not evidence of interaction" in d.note
    empty = describe_interaction("accuracy", aggregate([]), b, ab)
    assert (empty.additive_reference, empty.difference_from_additive) == (None, None)

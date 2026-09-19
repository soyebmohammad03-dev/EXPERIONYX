"""Calibration, confidence, bootstrap, errors, imbalance, latency, slices, rules and config."""

import math

import pytest

from experionyx.adapters.capabilities import TaskType
from experionyx.domain import ConfigurationRef
from experionyx.errors import ValidationError
from experionyx.evaluation import analysis, calibration
from experionyx.evaluation.bootstrap import bootstrap_interval, quantile
from experionyx.evaluation.config import (
    BootstrapConfig,
    CalibrationConfig,
    ErrorConfig,
    EvaluationConfig,
    ScoreSource,
    SliceCondition,
    SliceKind,
    SliceSpec,
    Thresholds,
)
from experionyx.evaluation.findings import RULES, RULESET_VERSION, derive_findings
from experionyx.evaluation.metrics import MetricInputs, compute_metric, default_metric_registry
from experionyx.evaluation.results import (
    CalibrationResult,
    ConfidenceResult,
    ErrorKind,
    ErrorSummary,
    FindingType,
    ImbalanceResult,
    Interpretation,
    LatencySummary,
    MetricContext,
    MetricResult,
    Severity,
    Status,
)
from experionyx.evaluation.serial import from_jsonable

REGISTRY = default_metric_registry()
CLS = TaskType.CLASSIFICATION

# --- calibration ------------------------------------------------------------------------------

SCORES = [[0.9, 0.1], [0.8, 0.2], [0.6, 0.4], [0.3, 0.7], [0.45, 0.55]]
TRUE = [0, 0, 1, 1, 0]
PRED = [0, 0, 0, 1, 1]  # samples 2 and 4 are wrong


def cal(bins: int) -> CalibrationResult:
    return calibration.calibration_analysis(SCORES, TRUE, PRED, (0, 1), n_bins=bins, source="test")


def test_calibration_bins_ece_mce_are_exactly_the_documented_quantities() -> None:
    two = cal(2)
    assert [b.count for b in two.bins] == [0, 5]
    assert two.bins[1].mean_confidence == pytest.approx(0.71)
    assert two.bins[1].accuracy == pytest.approx(0.6)
    assert two.ece == pytest.approx(0.11)
    assert two.mce == pytest.approx(0.11)
    five = cal(5)
    assert [b.count for b in five.bins] == [0, 0, 1, 2, 2]
    assert five.bins[2].gap == pytest.approx(0.55)
    assert five.ece == pytest.approx(0.23)
    assert five.mce == pytest.approx(0.55)
    assert (five.bins[0].lower, five.bins[-1].upper) == (0.0, 1.0)
    assert five.bins[0].mean_confidence is None  # empty bins are None, not 0
    assert five.n_bins == 5
    assert "equal-width" in five.binning
    assert five.confidence_source == "test"


def test_brier_score_is_the_multiclass_sum_and_matches_sklearn_binary_times_two() -> None:
    result = cal(10)
    assert result.brier_score == pytest.approx(0.321)
    sk = pytest.importorskip("sklearn.metrics")
    assert result.brier_score == pytest.approx(
        2 * sk.brier_score_loss(TRUE, [row[1] for row in SCORES])
    )


def test_perfectly_calibrated_and_confident_wrong_extremes() -> None:
    perfect = calibration.calibration_analysis(
        [[1.0, 0.0]] * 4, [0] * 4, [0] * 4, (0, 1), n_bins=10, source="t"
    )
    assert (perfect.ece, perfect.mce, perfect.brier_score) == (0.0, 0.0, 0.0)
    assert perfect.bins[-1].count == 4  # confidence 1.0 lands in the last (closed) bin
    wrong = calibration.calibration_analysis(
        [[1.0, 0.0]] * 4, [1] * 4, [0] * 4, (0, 1), n_bins=10, source="t"
    )
    assert (wrong.ece, wrong.mce, wrong.brier_score) == (1.0, 1.0, 2.0)


def test_calibration_is_explicitly_unsupported_undefined_or_invalid_never_zero() -> None:
    none = calibration.calibration_analysis(None, TRUE, PRED, (0, 1), n_bins=10, source=None)
    assert (none.status, none.ece, none.brier_score) == (Status.UNSUPPORTED, None, None)
    assert none.reason
    empty = calibration.calibration_analysis([], [], [], (0, 1), n_bins=10, source="t")
    assert empty.status is Status.UNDEFINED
    bad = calibration.calibration_analysis(
        [[float("nan"), 0.5]], [0], [0], (0, 1), n_bins=10, source="t"
    )
    assert bad.status is Status.INVALID
    over = calibration.calibration_analysis([[1.5, -0.5]], [0], [0], (0, 1), n_bins=10, source="t")
    assert over.status is Status.INVALID
    outside = calibration.calibration_analysis(
        [[0.5, 0.5]], [7], [0], (0, 1), n_bins=10, source="t"
    )
    assert outside.status is Status.INVALID


def test_calibration_warns_on_small_samples_and_on_argmax_disagreement() -> None:
    assert any("only 5 samples" in w for w in cal(5).warnings)
    disagree = calibration.calibration_analysis(
        [[0.9, 0.1]], [0], [1], (0, 1), n_bins=10, source="t"
    )
    assert any("arg-max" in w for w in disagree.warnings)


def test_confidence_analysis_separates_correct_from_incorrect() -> None:
    conf = calibration.top_confidence(SCORES)
    correct = [t == p for t, p in zip(TRUE, PRED, strict=True)]
    res = calibration.confidence_analysis(conf, correct, source="s", high=0.6, low=0.75)
    assert (res.n_correct, res.n_incorrect) == (3, 2)
    assert res.mean_confidence_correct == pytest.approx((0.9 + 0.8 + 0.7) / 3)
    assert res.mean_confidence_incorrect == pytest.approx((0.6 + 0.55) / 2)
    assert res.high_confidence_errors == 1  # 0.6 >= 0.6
    assert res.low_confidence_correct == 1  # 0.7 < 0.75
    assert sum(res.distribution) == 5
    assert res.distribution == (0, 0, 0, 0, 0, 1, 1, 1, 1, 1)  # 10 equal-width bins over [0, 1]
    assert res.source == "s"


def test_confidence_analysis_edge_cases() -> None:
    assert (
        calibration.confidence_analysis(None, [], source=None, high=0.9, low=0.6).status
        is Status.UNSUPPORTED
    )
    all_right = calibration.confidence_analysis(
        [0.9, 0.8], [True, True], source="s", high=0.9, low=0.6
    )
    assert all_right.mean_confidence_incorrect is None
    assert any("no incorrect" in w for w in all_right.warnings)
    assert (
        calibration.confidence_analysis([2.0], [True], source="s", high=0.9, low=0.6).status
        is Status.INVALID
    )


# --- bootstrap --------------------------------------------------------------------------------


def acc_inputs(n: int, p_correct: float = 0.7) -> MetricInputs:
    hits = int(n * p_correct)
    truth = [0] * n
    pred = [0] * hits + [1] * (n - hits)
    return MetricInputs(CLS, truth, pred)


def test_quantile_matches_numpy_linear_interpolation() -> None:
    np = pytest.importorskip("numpy")
    values = sorted([0.3, 0.1, 0.9, 0.5, 0.7, 0.2, 0.8])
    for q in (0.0, 0.025, 0.1, 0.5, 0.975, 1.0):
        assert quantile(values, q) == pytest.approx(float(np.quantile(values, q)), abs=1e-12)
    assert quantile([4.0], 0.3) == 4.0


def test_bootstrap_is_deterministic_for_a_seed_and_records_its_parameters() -> None:
    spec = REGISTRY.get("accuracy")
    a = bootstrap_interval(spec, acc_inputs(100), resamples=300, confidence=0.9, seed=7)
    b = bootstrap_interval(spec, acc_inputs(100), resamples=300, confidence=0.9, seed=7)
    assert a == b
    truth = [float(i) for i in range(50)]
    cont = MetricInputs(
        TaskType.REGRESSION, truth, [t + (i % 7) * 0.37 for i, t in enumerate(truth)]
    )
    mae = REGISTRY.get(
        "mae"
    )  # a continuous statistic: different seeds must give different intervals
    d = bootstrap_interval(mae, cont, resamples=300, confidence=0.9, seed=7)
    e = bootstrap_interval(mae, cont, resamples=300, confidence=0.9, seed=8)
    assert d == bootstrap_interval(mae, cont, resamples=300, confidence=0.9, seed=7)
    assert (d.lower, d.upper) != (e.lower, e.upper)
    assert (a.confidence, a.resamples, a.seed, a.n_samples, a.method) == (
        0.9,
        300,
        7,
        100,
        "bootstrap-percentile",
    )
    assert a.valid_resamples == 300


def test_bootstrap_interval_bounds_and_width_behave_like_a_binomial_standard_error() -> None:
    spec = REGISTRY.get("accuracy")
    small = bootstrap_interval(spec, acc_inputs(100), resamples=2000, confidence=0.95, seed=1)
    large = bootstrap_interval(spec, acc_inputs(400), resamples=2000, confidence=0.95, seed=1)
    for iv in (small, large):
        assert iv.lower is not None
        assert iv.upper is not None
        assert 0.0 <= iv.lower <= 0.7 <= iv.upper <= 1.0  # brackets the point estimate
    assert small.lower is not None
    assert small.upper is not None
    assert large.lower is not None
    assert large.upper is not None
    width_large = large.upper - large.lower
    assert width_large == pytest.approx(2 * 1.96 * math.sqrt(0.7 * 0.3 / 400), rel=0.15)
    assert (small.upper - small.lower) / width_large == pytest.approx(
        2.0, rel=0.25
    )  # ~ sqrt(n) scaling


def test_bootstrap_confidence_level_orders_the_intervals() -> None:
    spec = REGISTRY.get("accuracy")
    narrow = bootstrap_interval(spec, acc_inputs(100), resamples=1000, confidence=0.5, seed=3)
    wide = bootstrap_interval(spec, acc_inputs(100), resamples=1000, confidence=0.99, seed=3)
    assert narrow.lower is not None
    assert narrow.upper is not None
    assert wide.lower is not None
    assert wide.upper is not None
    assert wide.lower <= narrow.lower <= narrow.upper <= wide.upper


def test_bootstrap_warns_about_small_samples_and_few_resamples_and_drops_undefined() -> None:
    warned = bootstrap_interval(
        REGISTRY.get("accuracy"), acc_inputs(10), resamples=50, confidence=0.95, seed=0
    )
    assert any("unstable" in w for w in warned.warnings)
    assert any("coarse" in w for w in warned.warnings)
    tiny = MetricInputs(CLS, [0, 1, 1], [0, 1, 1], [[1.0, 0.0], [0.1, 0.9], [0.2, 0.8]], (0, 1))
    auc = bootstrap_interval(REGISTRY.get("roc_auc"), tiny, resamples=200, confidence=0.9, seed=0)
    assert auc.valid_resamples < 200  # some resamples contain a single class and are dropped
    assert any("undefined" in w for w in auc.warnings)


def test_bootstrap_empty_and_always_undefined_are_explicit() -> None:
    empty = bootstrap_interval(
        REGISTRY.get("accuracy"), MetricInputs(CLS, [], []), resamples=10, confidence=0.9, seed=0
    )
    assert (empty.status, empty.lower, empty.upper) == (Status.UNDEFINED, None, None)
    const = MetricInputs(TaskType.REGRESSION, [1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
    iv = bootstrap_interval(REGISTRY.get("r2"), const, resamples=20, confidence=0.9, seed=0)
    assert iv.status is Status.UNDEFINED
    assert iv.reason


def test_regression_bootstrap_interval_contains_the_point_estimate() -> None:
    truth = [float(i) for i in range(60)]
    pred = [t + (1.0 if i % 3 else -0.5) for i, t in enumerate(truth)]
    inp = MetricInputs(TaskType.REGRESSION, truth, pred)
    spec = REGISTRY.get("mae")
    point = compute_metric(spec, inp).value
    iv = bootstrap_interval(spec, inp, resamples=500, confidence=0.95, seed=2)
    assert point is not None
    assert iv.lower is not None
    assert iv.upper is not None
    assert iv.lower <= point <= iv.upper


# --- errors, imbalance, latency ---------------------------------------------------------------


def test_binary_error_records_kinds_and_complete_counts() -> None:
    truth, pred = [0, 1, 0, 1, 1, 0], [1, 0, 0, 1, 1, 0]
    conf = [0.95, 0.4, 0.9, 0.99, 0.5, 0.55]
    recs, summary = analysis.classification_errors(
        list(range(100, 106)),
        truth,
        pred,
        conf,
        (0, 1),
        ErrorConfig(),
        positive_label=None,
        high=0.9,
        low=0.6,
    )
    assert summary.counts == {
        "FALSE_POSITIVE": 1, "FALSE_NEGATIVE": 1, "INCORRECT_CLASS": 0,
        "LOW_CONFIDENCE_CORRECT": 2, "HIGH_CONFIDENCE_INCORRECT": 1,
    }  # fmt: skip
    assert [r.sample_index for r in recs] == [100, 101, 104, 105]
    assert recs[0].kinds == (ErrorKind.FALSE_POSITIVE, ErrorKind.HIGH_CONFIDENCE_INCORRECT)
    assert recs[1].kinds == (ErrorKind.FALSE_NEGATIVE,)
    assert (recs[2].kinds, recs[2].confidence) == ((ErrorKind.LOW_CONFIDENCE_CORRECT,), 0.5)
    assert not summary.truncated


def test_error_positive_label_can_be_chosen_and_multiclass_uses_incorrect_class() -> None:
    recs, _ = analysis.classification_errors(
        [0], [0], [1], None, (0, 1), ErrorConfig(), positive_label=0, high=0.9, low=0.6
    )
    assert recs[0].kinds == (ErrorKind.FALSE_NEGATIVE,)  # predicted 1, but 0 is the positive class
    multi, s = analysis.classification_errors(
        [0, 1],
        ["a", "b"],
        ["b", "b"],
        None,
        ("a", "b", "c"),
        ErrorConfig(),
        positive_label=None,
        high=0.9,
        low=0.6,
    )
    assert [r.kinds for r in multi] == [(ErrorKind.INCORRECT_CLASS,)]
    assert s.counts["INCORRECT_CLASS"] == 1
    assert multi[0].confidence is None  # no confidence signal => none invented


def test_error_record_retention_is_bounded_but_counts_stay_complete() -> None:
    truth = [0] * 50
    pred = [1] * 50
    recs, summary = analysis.classification_errors(
        list(range(50)),
        truth,
        pred,
        None,
        (0, 1),
        ErrorConfig(max_records=5),
        positive_label=None,
        high=0.9,
        low=0.6,
    )
    assert len(recs) == summary.recorded == 5
    assert summary.truncated
    assert summary.counts["FALSE_POSITIVE"] == 50
    none, s2 = analysis.classification_errors(
        list(range(50)),
        truth,
        pred,
        None,
        (0, 1),
        ErrorConfig(record=False),
        positive_label=None,
        high=0.9,
        low=0.6,
    )
    assert none == []
    assert s2.counts["FALSE_POSITIVE"] == 50
    assert not s2.truncated


def test_regression_error_records_residuals_and_top_k_selection() -> None:
    recs, summary = analysis.regression_errors(
        [10, 11, 12, 13], [0.0, 2.0, 4.0, -1.0], [1.0, 2.0, 1.0, -1.0], ErrorConfig(max_records=2)
    )
    assert [r.sample_index for r in recs] == [10, 12]  # the two largest |residual|, in sample order
    first, third = recs
    assert (first.signed_error, first.absolute_error, first.relative_error) == (
        1.0,
        1.0,
        None,
    )  # true == 0
    assert (third.signed_error, third.absolute_error, third.relative_error) == (-3.0, 3.0, 0.75)
    assert summary.truncated
    assert summary.stats["mean_absolute_error"] == 1.0
    assert summary.stats["max_absolute_error"] == 3.0
    assert summary.stats["mean_signed_error"] == pytest.approx(-0.5)
    assert summary.counts == {"RESIDUAL": 4}


def test_regression_error_ties_are_broken_by_sample_order() -> None:
    recs, _ = analysis.regression_errors(
        [0, 1, 2, 3], [0.0] * 4, [1.0] * 4, ErrorConfig(max_records=2)
    )
    assert [r.sample_index for r in recs] == [0, 1]


def test_imbalance_reports_counts_proportions_ratio_and_the_accuracy_caveat() -> None:
    res = analysis.imbalance_analysis([0] * 90 + [1] * 10)
    assert res.class_counts == {"0": 90, "1": 10}
    assert res.class_proportions == {"0": 0.9, "1": 0.1}
    assert (res.majority_class, res.minority_class, res.imbalance_ratio) == (0, 1, 9.0)
    assert "misleading" in res.note
    balanced = analysis.imbalance_analysis([0, 1, 0, 1])
    assert balanced.imbalance_ratio == 1.0
    assert analysis.imbalance_analysis([]).status is Status.UNDEFINED
    assert analysis.imbalance_analysis([3, 3]).imbalance_ratio == 1.0  # single class


def test_latency_summary_statistics_and_separation_of_load_time() -> None:
    lat = analysis.latency_summary([0.1, 0.2, 0.3, 0.4], 10, load_seconds=5.0)
    assert lat.status is Status.COMPUTED
    assert lat.total_seconds == pytest.approx(1.0)
    assert lat.mean_batch_seconds == pytest.approx(0.25)
    assert lat.median_batch_seconds == pytest.approx(0.25)
    assert lat.p95_batch_seconds == 0.4  # nearest-rank
    assert lat.max_batch_seconds == 0.4
    assert lat.throughput_samples_per_second == pytest.approx(10.0)
    assert (lat.n_samples, lat.n_batches, lat.load_seconds) == (10, 4, 5.0)
    assert lat.total_seconds is not None
    assert lat.total_seconds < 5.0  # load time is not folded into inference time
    assert any("not a benchmark" in w for w in lat.warnings)
    assert any("p95" in w for w in lat.warnings)


def test_latency_edge_cases_are_explicit() -> None:
    assert analysis.latency_summary([], 0, None).status is Status.UNDEFINED
    assert analysis.latency_summary([0.1, -1.0], 2, None).status is Status.INVALID
    assert analysis.latency_summary([float("nan")], 1, None).status is Status.INVALID
    zero = analysis.latency_summary([0.0, 0.0], 4, None)
    assert zero.throughput_samples_per_second is None  # never divides by zero
    assert analysis.latency_summary([0.1] * 20, 40, None).p95_batch_seconds == pytest.approx(0.1)


# --- slices -----------------------------------------------------------------------------------

X = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
COLS = {"x": X}
ST = [0, 0, 1, 1, 1, 0]
SP = [0, 1, 1, 1, 0, 0]


def sel(*conds: SliceCondition) -> list[int]:
    return analysis.slice_indices(SliceSpec("s", tuple(conds)), ST, SP, COLS)


def test_slice_conditions_and_boundaries() -> None:
    assert sel(SliceCondition(SliceKind.TARGET_EQUALS, value=1)) == [2, 3, 4]
    assert sel(SliceCondition(SliceKind.PREDICTED_EQUALS, value=0)) == [0, 4, 5]
    assert sel(SliceCondition(SliceKind.FEATURE_EQUALS, field="x", value=3.0)) == [3]
    assert sel(SliceCondition(SliceKind.FEATURE_RANGE, field="x", low=2.0, high=4.0)) == [
        2,
        3,
    ]  # [low, high)
    assert sel(SliceCondition(SliceKind.FEATURE_RANGE, field="x", low=4.0)) == [4, 5]
    assert sel(SliceCondition(SliceKind.FEATURE_RANGE, field="x", high=1.0)) == [0]
    both = sel(
        SliceCondition(SliceKind.TARGET_EQUALS, value=1),
        SliceCondition(SliceKind.FEATURE_RANGE, field="x", high=4.0),
    )
    assert both == [2, 3]  # conditions are ANDed
    assert sel(SliceCondition(SliceKind.FEATURE_EQUALS, field="x", value=99.0)) == []


def test_slice_metrics_use_the_same_engine_and_report_deltas_vs_overall() -> None:
    inp = MetricInputs(CLS, ST, SP)
    specs = [REGISTRY.get("accuracy"), REGISTRY.get("confusion_matrix")]
    ctx = MetricContext("run_x", "test")
    overall = {
        m.id: MetricResult(
            m.id,
            m.version,
            m.name,
            Status.COMPUTED,
            6,
            ctx,
            m.scale,
            m.higher_is_better,
            value=compute_metric(m, inp).value,
        )
        for m in specs
    }
    spec = SliceSpec(
        "mid", (SliceCondition(SliceKind.FEATURE_RANGE, field="x", low=2.0, high=4.0),)
    )
    res = analysis.evaluate_slice(spec, sel(*spec.conditions), inp, specs, overall, ctx)
    assert res.n_samples == 2
    assert res.fraction_of_samples == pytest.approx(2 / 6)
    assert [m.metric_id for m in res.metrics] == ["accuracy"]  # structural metrics are skipped
    assert res.metrics[0].value == 1.0  # samples 2 and 3 are both correct
    assert res.deltas_vs_overall["accuracy"] == pytest.approx(1.0 - 4 / 6)
    empty = analysis.evaluate_slice(spec, [], inp, specs, overall, ctx)
    assert empty.n_samples == 0
    assert (empty.metrics[0].status, empty.metrics[0].value) == (Status.UNDEFINED, None)
    assert empty.deltas_vs_overall["accuracy"] is None


# --- rules / findings -------------------------------------------------------------------------

CTX = MetricContext("run_f", "test")


def metric(mid: str, value: float) -> MetricResult:
    return MetricResult(mid, "1.0.0", mid, Status.COMPUTED, 100, CTX, "", None, value=value)


def findings(**over: object):  # type: ignore[no-untyped-def]
    args: dict[str, object] = {
        "task": CLS,
        "metrics": {},
        "confusion": None,
        "imbalance": None,
        "confidence": ConfidenceResult(Status.UNSUPPORTED, None),
        "calibration": CalibrationResult(Status.UNSUPPORTED, None),
        "errors": ErrorSummary(Status.COMPUTED, 100, {}, 0, False),
        "latency": LatencySummary(Status.UNDEFINED, None, 100, 0),
        "slices": [],
        "n_samples": 100,
        "thresholds": Thresholds(),
    }
    args.update(over)
    return derive_findings(**args)  # type: ignore[arg-type]


def test_no_findings_without_evidence_and_unsupported_analyses_never_trigger_rules() -> None:
    assert findings() == []
    assert findings(calibration=CalibrationResult(Status.INVALID, "s")) == []


def test_calibration_and_confidence_rules_are_threshold_driven() -> None:
    from experionyx.evaluation.results import CalibrationBin

    bad = CalibrationResult(
        Status.COMPUTED,
        "predict_proba",
        n_bins=10,
        n_samples=100,
        bins=(CalibrationBin(0, 1, 100, 0.9, 0.6, 0.3),),
        ece=0.3,
        mce=0.3,
        brier_score=0.4,
    )
    (f,) = findings(calibration=bad)
    assert (f.type, f.severity, f.interpretation) == (
        FindingType.CALIBRATION_ERROR,
        Severity.WARNING,
        Interpretation.OBSERVED,
    )
    assert (f.rule_id, f.rule_version, f.observed_value, f.threshold) == (
        "ece-threshold",
        RULESET_VERSION,
        0.3,
        0.10,
    )
    assert f.observation_names == ("calibration.ece",)
    assert f.artifact_paths == ("evaluation/calibration.json",)
    assert (
        findings(calibration=bad, thresholds=Thresholds(ece_max=0.5)) == []
    )  # thresholds are configuration
    conf = ConfidenceResult(
        Status.COMPUTED,
        "s",
        n_samples=100,
        high_confidence_errors=10,
        high_confidence_threshold=0.9,
        n_incorrect=10,
        n_correct=90,
    )
    (g,) = findings(confidence=conf)
    assert (g.type, g.observed_value) == (FindingType.HIGH_CONFIDENCE_ERRORS, 0.1)
    assert (
        findings(confidence=conf, thresholds=Thresholds(high_confidence_error_rate_max=0.1)) == []
    )  # strict >


def test_imbalance_and_minority_recall_rules() -> None:
    from experionyx.evaluation.results import ClassMetrics, ConfusionMatrix

    imb = ImbalanceResult(Status.COMPUTED, {"0": 90, "1": 10}, {"0": 0.9, "1": 0.1}, 0, 1, 9.0)
    per_class = (
        ClassMetrics(0, 90, 98, 90 / 98, 1.0, 0.95),
        ClassMetrics(1, 10, 2, 1.0, 0.2, 1 / 3),
    )
    cm = ConfusionMatrix(
        (0, 1), ((90, 0), (8, 2)), ((1.0, 0.0), (0.8, 0.2)), ((0.9, 0.0), (0.1, 1.0)), per_class
    )
    kinds = {f.type: f for f in findings(imbalance=imb, confusion=cm)}
    assert set(kinds) == {FindingType.CLASS_IMBALANCE, FindingType.WEAK_MINORITY_RECALL}
    assert kinds[FindingType.CLASS_IMBALANCE].severity is Severity.NOTICE
    assert kinds[FindingType.WEAK_MINORITY_RECALL].affected == "class:1"
    assert kinds[FindingType.WEAK_MINORITY_RECALL].observed_value == 0.2
    balanced = ImbalanceResult(Status.COMPUTED, {"0": 50, "1": 50}, {"0": 0.5, "1": 0.5}, 0, 0, 1.0)
    assert findings(imbalance=balanced, confusion=cm) == []


def test_regression_and_latency_and_slice_rules() -> None:
    (r2,) = findings(task=TaskType.REGRESSION, metrics={"r2": metric("r2", 0.2)})
    assert (r2.type, r2.affected) == (FindingType.RESIDUAL_VARIANCE, "overall")
    assert findings(task=TaskType.REGRESSION, metrics={"r2": metric("r2", 0.9)}) == []
    assert findings(task=CLS, metrics={"r2": metric("r2", 0.0)}) == []  # rule is regression-only
    lat = analysis.latency_summary([0.001] * 9 + [0.05], 100, None)
    (outlier,) = findings(latency=lat)
    assert outlier.type is FindingType.LATENCY_OUTLIER
    few = analysis.latency_summary([0.001, 0.05], 100, None)
    assert findings(latency=few) == []  # needs >= 5 batches
    from experionyx.evaluation.results import SliceResult

    sl = SliceResult(
        "hard",
        (SliceCondition(SliceKind.TARGET_EQUALS, value=1),),
        20,
        0.2,
        (metric("accuracy", 0.5),),
        {"accuracy": -0.4},
    )
    (deg,) = findings(metrics={"accuracy": metric("accuracy", 0.9)}, slices=[sl])
    assert (deg.type, deg.affected, deg.rule_id) == (
        FindingType.SLICE_DEGRADATION,
        "slice:hard",
        "slice-accuracy-drop",
    )
    assert deg.observed_value == pytest.approx(0.4)
    err = SliceResult(
        "hard", (SliceCondition(SliceKind.TARGET_EQUALS, value=1),), 20, 0.2, (), {"mae": 5.0}
    )
    (inc,) = findings(task=TaskType.REGRESSION, metrics={"mae": metric("mae", 10.0)}, slices=[err])
    assert inc.rule_id == "slice-error-increase"
    assert inc.observed_value == pytest.approx(0.5)


def test_findings_are_deterministic_versioned_and_never_interpreted() -> None:
    bad = CalibrationResult(
        Status.COMPUTED, "s", n_bins=10, n_samples=100, ece=0.3, mce=0.3, brier_score=0.4
    )
    a, b = findings(calibration=bad), findings(calibration=bad)
    assert a == b
    assert a[0].finding_id.startswith("fnd_")
    assert a[0].claim_id is None  # linked to evidence only by the engine
    assert all(f.interpretation is Interpretation.OBSERVED for f in a)
    assert set(RULES) >= {"ece-threshold", "class-imbalance", "minority-recall", "r2-floor"}
    assert "because" not in a[0].description.lower()  # describes what was measured, not why


# --- configuration ----------------------------------------------------------------------------


def rich_config() -> EvaluationConfig:
    return EvaluationConfig(
        split="test",
        batch_size=8,
        metrics=("accuracy", "f1"),
        score_source=ScoreSource.SOFTMAX_LOGITS,
        calibration=CalibrationConfig(bins=15, high_confidence=0.95, low_confidence=0.5),
        bootstrap=BootstrapConfig(
            resamples=123, confidence=0.9, seed=4, metrics=("accuracy",), include_slices=True
        ),
        errors=ErrorConfig(max_records=7),
        slices=(
            SliceSpec(
                "s", (SliceCondition(SliceKind.FEATURE_RANGE, field="x", low=1.0, high=2.0),)
            ),
        ),
        thresholds=Thresholds(ece_max=0.2),
    )


def test_config_round_trips_through_a_frozen_configuration_ref() -> None:
    cfg = rich_config()
    ref = ConfigurationRef(cfg.to_parameters())
    assert EvaluationConfig.from_parameters(ref.parameters) == cfg


def test_config_is_content_addressed() -> None:
    a = ConfigurationRef(rich_config().to_parameters())
    b = ConfigurationRef(rich_config().to_parameters())
    changed = ConfigurationRef(EvaluationConfig(split="test", batch_size=9).to_parameters())
    assert a.id == b.id
    assert a.id != changed.id
    assert (
        ConfigurationRef(EvaluationConfig().to_parameters()).id
        == ConfigurationRef(EvaluationConfig().to_parameters()).id
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda: EvaluationConfig(batch_size=0),
        lambda: EvaluationConfig(split=" "),
        lambda: EvaluationConfig(metrics=("a", "a")),
        lambda: EvaluationConfig(schema_version=2),
        lambda: BootstrapConfig(resamples=0),
        lambda: BootstrapConfig(confidence=1.0),
        lambda: BootstrapConfig(seed=-1),
        lambda: CalibrationConfig(bins=0),
        lambda: CalibrationConfig(low_confidence=0.9, high_confidence=0.5),
        lambda: ErrorConfig(max_records=-1),
        lambda: Thresholds(ece_max=2.0),
        lambda: Thresholds(imbalance_ratio_min=0.5),
        lambda: Thresholds(r2_min=float("nan")),
        lambda: SliceSpec("", (SliceCondition(SliceKind.TARGET_EQUALS, value=1),)),
        lambda: SliceSpec("s", ()),
        lambda: SliceCondition(SliceKind.FEATURE_RANGE, field="x"),
        lambda: SliceCondition(SliceKind.FEATURE_RANGE, field="x", low=2.0, high=1.0),
        lambda: SliceCondition(SliceKind.FEATURE_EQUALS, field="x"),
        lambda: SliceCondition(SliceKind.TARGET_EQUALS),
        lambda: EvaluationConfig(
            slices=(SliceSpec("d", (SliceCondition(SliceKind.TARGET_EQUALS, value=1),)),) * 2
        ),
    ],
)
def test_invalid_configurations_are_rejected(build) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValidationError):
        build()


def test_unknown_or_malformed_config_fields_never_silently_disappear() -> None:
    good = rich_config().to_parameters()["evaluation"]
    assert isinstance(good, dict)
    with pytest.raises(ValidationError, match="unknown field"):
        EvaluationConfig.from_dict({**good, "typo_field": 1})
    nested = {**good, "bootstrap": {**good["bootstrap"], "resample": 5}}
    with pytest.raises(ValidationError, match="unknown field"):
        EvaluationConfig.from_dict(nested)
    with pytest.raises(ValidationError):
        EvaluationConfig.from_dict({**good, "batch_size": "8"})
    with pytest.raises(ValidationError):
        EvaluationConfig.from_dict({**good, "score_source": "MAGIC"})
    with pytest.raises(ValidationError, match="exactly"):
        EvaluationConfig.from_parameters({"evaluation": good, "extra": 1})
    with pytest.raises(ValidationError):
        EvaluationConfig.from_parameters({"batch_size": 4})


def test_from_jsonable_rejects_missing_required_fields_and_bad_types() -> None:
    with pytest.raises(ValidationError, match="missing field"):
        from_jsonable(CalibrationResult, {"confidence_source": None})
    with pytest.raises(ValidationError):
        from_jsonable(CalibrationResult, {"status": "COMPUTED", "confidence_source": 3})
    ok = from_jsonable(
        CalibrationResult, {"status": "UNSUPPORTED", "confidence_source": None, "reason": "x"}
    )
    assert ok.status is Status.UNSUPPORTED
    assert from_jsonable(float, 3) == 3.0
    with pytest.raises(ValidationError):
        from_jsonable(int, True)

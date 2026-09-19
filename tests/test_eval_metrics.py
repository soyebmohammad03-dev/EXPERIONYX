"""Metric registry, requirements, edge cases, and agreement with scikit-learn reference results."""

import math

import pytest

from experionyx.adapters.capabilities import TaskType
from experionyx.errors import DuplicateAdapterError, EvaluationError
from experionyx.evaluation.metrics import (
    MetricInputs,
    MetricSpec,
    Requirement,
    average_precision,
    compute_metric,
    confusion,
    default_metric_registry,
    rank_auc,
)
from experionyx.evaluation.results import Status

CLS = TaskType.CLASSIFICATION
REG = TaskType.REGRESSION
REGISTRY = default_metric_registry()


def value(metric_id: str, inp: MetricInputs) -> float:
    out = compute_metric(REGISTRY.get(metric_id), inp)
    assert out.status is Status.COMPUTED, out
    assert out.value is not None
    return out.value


# --- registry ---------------------------------------------------------------------------------


def test_registry_lists_the_documented_metrics_with_declared_requirements() -> None:
    assert set(REGISTRY.ids()) == {
        "accuracy", "balanced_accuracy", "precision", "recall", "f1", "confusion_matrix",
        "roc_auc", "pr_auc", "log_loss", "mae", "mse", "rmse", "r2", "median_absolute_error",
    }  # fmt: skip
    for mid in ("roc_auc", "pr_auc", "log_loss"):
        assert Requirement.SCORES in REGISTRY.get(mid).requires
    assert Requirement.SCORES not in REGISTRY.get("accuracy").requires
    assert {s.id for s in REGISTRY.for_task(REG)} == {
        "mae",
        "mse",
        "rmse",
        "r2",
        "median_absolute_error",
    }
    assert all(s.version for s in REGISTRY.for_task(CLS))
    assert REGISTRY.get("confusion_matrix").scalar is False


def test_registry_rejects_duplicates_unknown_and_incompatible_requests() -> None:
    reg = default_metric_registry()
    spec = reg.get("accuracy")
    with pytest.raises(DuplicateAdapterError):
        reg.register(spec)
    with pytest.raises(EvaluationError, match="unknown metric"):
        reg.get("nope")
    with pytest.raises(EvaluationError, match="do not apply"):
        reg.resolve(["mae"], CLS)
    assert [s.id for s in reg.resolve([], REG)] == [s.id for s in reg.for_task(REG)]


def test_new_metrics_plug_in_without_touching_the_engine() -> None:
    reg = default_metric_registry()

    def error_rate(inp: MetricInputs):  # type: ignore[no-untyped-def]
        from experionyx.evaluation.metrics import MetricOutcome

        wrong = sum(t != p for t, p in zip(inp.y_true, inp.y_pred, strict=True))
        return MetricOutcome(Status.COMPUTED, wrong / len(inp.y_true))

    reg.register(
        MetricSpec(
            "error_rate",
            "Error rate",
            "1.0.0",
            frozenset({CLS}),
            frozenset({Requirement.LABELS}),
            "0..1",
            False,
            True,
            "1 - accuracy",
            error_rate,
        )
    )
    inp = MetricInputs(CLS, [0, 1, 1, 0], [0, 1, 0, 0])
    out = compute_metric(reg.get("error_rate"), inp)
    assert out.value == 0.25
    assert "error_rate" not in default_metric_registry().ids()  # registries are independent


# --- agreement with scikit-learn --------------------------------------------------------------


@pytest.fixture(scope="module")
def iris_case() -> tuple[MetricInputs, list[int], object, object]:
    pytest.importorskip("sklearn")
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression

    x, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=300).fit(x[::2], y[::2])
    proba = model.predict_proba(x[1::2])
    pred = model.predict(x[1::2])
    truth = y[1::2]
    inp = MetricInputs(CLS, truth.tolist(), pred.tolist(), proba.tolist(), (0, 1, 2))
    return inp, truth.tolist(), pred, proba


def test_multiclass_metrics_match_sklearn(iris_case) -> None:  # type: ignore[no-untyped-def]
    from sklearn import metrics as sk

    inp, truth, pred, proba = iris_case
    ref = {
        "accuracy": sk.accuracy_score(truth, pred),
        "balanced_accuracy": sk.balanced_accuracy_score(truth, pred),
        "precision": sk.precision_score(truth, pred, average="macro"),
        "recall": sk.recall_score(truth, pred, average="macro"),
        "f1": sk.f1_score(truth, pred, average="macro"),
        "roc_auc": sk.roc_auc_score(truth, proba, multi_class="ovr"),
        "pr_auc": sk.average_precision_score(
            [[int(t == c) for c in (0, 1, 2)] for t in truth], proba, average="macro"
        ),
        "log_loss": sk.log_loss(truth, proba),
    }
    for mid, expected in ref.items():
        assert value(mid, inp) == pytest.approx(expected, abs=1e-12), mid


def test_binary_metrics_match_sklearn() -> None:
    sk = pytest.importorskip("sklearn.metrics")
    truth = [0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1]
    p1 = [0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.6, 0.55, 0.7, 0.3, 0.4, 0.95]
    pred = [int(p >= 0.5) for p in p1]
    inp = MetricInputs(CLS, truth, pred, [[1 - p, p] for p in p1], (0, 1))
    assert value("roc_auc", inp) == pytest.approx(sk.roc_auc_score(truth, p1), abs=1e-12)
    assert value("pr_auc", inp) == pytest.approx(sk.average_precision_score(truth, p1), abs=1e-12)
    assert value("log_loss", inp) == pytest.approx(sk.log_loss(truth, p1), abs=1e-12)
    assert value("f1", inp) == pytest.approx(sk.f1_score(truth, pred, average="macro"), abs=1e-12)


def test_ties_in_scores_are_handled_like_sklearn() -> None:
    sk = pytest.importorskip("sklearn.metrics")
    truth = [1, 0, 1, 0, 1, 0, 0, 1]
    scores = [0.5, 0.5, 0.5, 0.2, 0.9, 0.2, 0.5, 0.9]
    pos = [bool(t) for t in truth]
    assert rank_auc(scores, pos) == pytest.approx(sk.roc_auc_score(truth, scores), abs=1e-12)
    assert average_precision(scores, pos) == pytest.approx(
        sk.average_precision_score(truth, scores), abs=1e-12
    )


def test_regression_metrics_match_sklearn() -> None:
    sk = pytest.importorskip("sklearn.metrics")
    truth = [3.0, -0.5, 2.0, 7.0, 4.2, 1.1, 0.0]
    pred = [2.5, 0.0, 2.1, 7.8, 3.9, 1.0, 0.4]
    inp = MetricInputs(REG, truth, pred)
    assert value("mae", inp) == pytest.approx(sk.mean_absolute_error(truth, pred), abs=1e-12)
    assert value("mse", inp) == pytest.approx(sk.mean_squared_error(truth, pred), abs=1e-12)
    assert value("rmse", inp) == pytest.approx(
        math.sqrt(sk.mean_squared_error(truth, pred)), abs=1e-12
    )
    assert value("r2", inp) == pytest.approx(sk.r2_score(truth, pred), abs=1e-12)
    assert value("median_absolute_error", inp) == pytest.approx(
        sk.median_absolute_error(truth, pred), abs=1e-12
    )


# --- confusion matrix and per-class -----------------------------------------------------------


def test_confusion_matrix_counts_normalizations_and_per_class() -> None:
    truth = ["a", "a", "a", "b", "b", "c", "c", "c", "c"]
    pred = ["a", "a", "b", "b", "c", "c", "c", "c", "a"]
    cm = confusion(MetricInputs(CLS, truth, pred, classes=("a", "b", "c")))
    assert cm.labels == ("a", "b", "c")
    assert cm.counts == ((2, 1, 0), (0, 1, 1), (1, 0, 3))
    assert cm.row_normalized[0] == pytest.approx((2 / 3, 1 / 3, 0.0))
    assert cm.column_normalized[2] == pytest.approx(
        (1 / 3, 0.0, 3 / 4)
    )  # counts[2][j] / column j total
    a, b, c = cm.per_class
    assert (a.support, a.predicted, a.precision, a.recall) == (3, 3, 2 / 3, 2 / 3)
    assert (b.support, b.predicted) == (2, 2)
    assert (c.support, c.predicted, c.recall) == (4, 4, 3 / 4)
    assert a.f1 == pytest.approx(2 / 3)


def test_classes_with_no_predictions_or_no_samples_are_undefined_not_zero() -> None:
    cm = confusion(MetricInputs(CLS, [0, 0, 1], [0, 0, 0], classes=(0, 1, 2)))
    one, two = cm.per_class[1], cm.per_class[2]
    assert (one.precision, one.recall) == (None, 0.0)  # never predicted; recall is really 0
    assert (two.precision, two.recall, two.f1) == (None, None, None)  # absent from both
    assert cm.row_normalized[2] == (None, None, None)


def test_macro_average_warns_when_it_counts_an_undefined_class_as_zero() -> None:
    out = compute_metric(REGISTRY.get("precision"), MetricInputs(CLS, [0, 0, 1], [0, 0, 0]))
    assert out.status is Status.COMPUTED
    assert out.warnings
    assert "undefined precision" in out.warnings[0]
    sk = pytest.importorskip("sklearn.metrics")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert out.value == pytest.approx(
            sk.precision_score([0, 0, 1], [0, 0, 0], average="macro", zero_division=0)
        )


# --- explicit unsupported / undefined / invalid, never zero -------------------------------------


@pytest.mark.parametrize("metric_id", ["roc_auc", "pr_auc", "log_loss"])
def test_score_metrics_are_unsupported_without_scores_not_computed_from_labels(
    metric_id: str,
) -> None:
    out = compute_metric(REGISTRY.get(metric_id), MetricInputs(CLS, [0, 1, 1], [0, 1, 0]))
    assert out.status is Status.UNSUPPORTED
    assert out.value is None
    assert out.reason


def test_undefined_cases_are_explicit() -> None:
    one_class = MetricInputs(CLS, [1, 1, 1], [1, 1, 1], [[0.0, 1.0]] * 3, (0, 1))
    for mid in ("roc_auc", "pr_auc"):
        out = compute_metric(REGISTRY.get(mid), one_class)
        assert (out.status, out.value) == (Status.UNDEFINED, None)
    assert compute_metric(REGISTRY.get("accuracy"), MetricInputs(CLS, [1, 1], [1, 1])).value == 1.0
    assert (
        compute_metric(REGISTRY.get("r2"), MetricInputs(REG, [2.0, 2.0], [1.0, 3.0])).status
        is Status.UNDEFINED
    )
    empty = compute_metric(REGISTRY.get("accuracy"), MetricInputs(CLS, [], []))
    assert (empty.status, empty.value) == (Status.UNDEFINED, None)


def test_single_class_balanced_accuracy_and_confusion() -> None:
    inp = MetricInputs(CLS, [0, 0, 0], [0, 0, 1])
    assert value("balanced_accuracy", inp) == pytest.approx(2 / 3)
    assert confusion(inp).counts == ((2, 1), (0, 0))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_regression_values_are_invalid_not_zero(bad: float) -> None:
    for mid in ("mae", "rmse", "r2"):
        out = compute_metric(REGISTRY.get(mid), MetricInputs(REG, [1.0, bad], [1.0, 2.0]))
        assert (out.status, out.value) == (Status.INVALID, None)
        out = compute_metric(REGISTRY.get(mid), MetricInputs(REG, [1.0, 2.0], [1.0, bad]))
        assert out.status is Status.INVALID


def test_non_finite_scores_and_bad_probabilities_are_invalid() -> None:
    nan_scores = MetricInputs(CLS, [0, 1], [0, 1], [[1.0, 0.0], [float("nan"), 1.0]], (0, 1))
    assert compute_metric(REGISTRY.get("roc_auc"), nan_scores).status is Status.INVALID
    bad = MetricInputs(CLS, [0, 1], [0, 1], [[1.2, -0.2], [0.5, 0.5]], (0, 1))
    assert compute_metric(REGISTRY.get("log_loss"), bad).status is Status.INVALID
    mismatch = MetricInputs(CLS, [0, 1], [0, 1, 1])
    assert compute_metric(REGISTRY.get("accuracy"), mismatch).status is Status.INVALID
    outside = MetricInputs(CLS, [0, 5], [0, 1], [[0.5, 0.5]] * 2, (0, 1))
    assert compute_metric(REGISTRY.get("log_loss"), outside).status is Status.INVALID


def test_log_loss_clips_zero_probability_instead_of_returning_infinity() -> None:
    out = compute_metric(
        REGISTRY.get("log_loss"), MetricInputs(CLS, [1], [0], [[1.0, 0.0]], (0, 1))
    )
    assert out.value is not None
    assert math.isfinite(out.value)
    assert out.value == pytest.approx(-math.log(1e-15))

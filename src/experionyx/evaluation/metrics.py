"""Metric registry and pure-Python metric implementations (no framework dependency).

Every metric declares the task(s) it applies to and the inputs it requires. `compute_metric`
enforces that contract: missing scores => UNSUPPORTED, no samples or a mathematically undefined
value => UNDEFINED, non-finite numerics => INVALID. A value is never silently zero.

Conventions (docs/evaluation.md): precision/recall/F1 are macro-averaged over the labels present
in y_true or y_pred; a class whose precision/recall is undefined counts as 0 in the macro average
*with a recorded warning* (scikit-learn's zero_division convention). Averages are plain
double-precision means; presentation rounding is left to consumers.
"""

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from experionyx.adapters.capabilities import TaskType
from experionyx.errors import DuplicateAdapterError, EvaluationError
from experionyx.evaluation.results import ClassMetrics, ConfusionMatrix, Label, Scalar, Status

LOG_LOSS_EPS = 1e-15
IMPL_VERSION = "1.0.0"


class Requirement(StrEnum):
    LABELS = "LABELS"  # true and predicted targets
    SCORES = "SCORES"  # per-class probabilities/scores aligned with `classes`


@dataclass(frozen=True)
class MetricInputs:
    task: TaskType
    y_true: Sequence[Scalar]
    y_pred: Sequence[Scalar]
    scores: Sequence[Sequence[float]] | None = None  # rows aligned with `classes`
    classes: tuple[Label, ...] | None = None


@dataclass(frozen=True)
class MetricOutcome:
    status: Status
    value: float | None = None
    structured: dict[str, object] | None = None
    reason: str | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class MetricSpec:
    id: str
    name: str
    version: str
    tasks: frozenset[TaskType]
    requires: frozenset[Requirement]
    scale: str
    higher_is_better: bool | None
    scalar: bool  # False for structural results (confusion matrix)
    description: str
    compute: Callable[[MetricInputs], MetricOutcome] = field(repr=False, compare=False)


# --- small numeric helpers --------------------------------------------------------------------


def _ok(value: float, warnings: tuple[str, ...] = ()) -> MetricOutcome:
    return MetricOutcome(Status.COMPUTED, value, warnings=warnings)


def _undefined(reason: str) -> MetricOutcome:
    return MetricOutcome(Status.UNDEFINED, reason=reason)


def _div(a: float, b: float) -> float | None:
    return None if b == 0 else a / b


def label_order(inp: MetricInputs) -> list[Label]:
    """Declared classes first (in score-column order), then any other observed labels sorted."""
    declared = list(inp.classes) if inp.classes else []
    seen = set(declared)
    extra = {x for x in (*inp.y_true, *inp.y_pred) if x not in seen}
    try:
        tail = sorted(extra)
    except TypeError:
        tail = sorted(extra, key=str)
    return [*declared, *tail]  # type: ignore[list-item]


def confusion(inp: MetricInputs) -> ConfusionMatrix:
    labels = label_order(inp)
    index: dict[Scalar, int] = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    counts = [[0] * n for _ in range(n)]
    for t, p in zip(inp.y_true, inp.y_pred, strict=True):
        counts[index[t]][index[p]] += 1
    row_sum = [sum(r) for r in counts]
    col_sum = [sum(counts[i][j] for i in range(n)) for j in range(n)]
    per_class = []
    for i, lab in enumerate(labels):
        tp = counts[i][i]
        per_class.append(
            ClassMetrics(
                label=lab,
                support=row_sum[i],
                predicted=col_sum[i],
                precision=_div(tp, col_sum[i]),
                recall=_div(tp, row_sum[i]),
                f1=_div(2 * tp, row_sum[i] + col_sum[i]),
            )
        )
    return ConfusionMatrix(
        labels=tuple(labels),
        counts=tuple(tuple(r) for r in counts),
        row_normalized=tuple(tuple(_div(c, row_sum[i]) for c in counts[i]) for i in range(n)),
        column_normalized=tuple(
            tuple(_div(counts[i][j], col_sum[j]) for j in range(n)) for i in range(n)
        ),
        per_class=tuple(per_class),
    )


def _macro(values: list[float | None], what: str) -> MetricOutcome:
    undefined = sum(v is None for v in values)
    warnings = (
        (f"{undefined} class(es) had undefined {what} and were counted as 0 (macro average)",)
        if undefined
        else ()
    )
    return _ok(sum(v or 0.0 for v in values) / len(values), warnings)


def _present(cm: ConfusionMatrix) -> list[ClassMetrics]:
    return [c for c in cm.per_class if c.support > 0 or c.predicted > 0]


# --- classification ---------------------------------------------------------------------------


def _accuracy(inp: MetricInputs) -> MetricOutcome:
    hits = sum(t == p for t, p in zip(inp.y_true, inp.y_pred, strict=True))
    return _ok(hits / len(inp.y_true))


def _balanced_accuracy(inp: MetricInputs) -> MetricOutcome:
    recalls = [c.recall for c in confusion(inp).per_class if c.support > 0]
    return _ok(sum(r for r in recalls if r is not None) / len(recalls))


def _precision(inp: MetricInputs) -> MetricOutcome:
    return _macro([c.precision for c in _present(confusion(inp))], "precision")


def _recall(inp: MetricInputs) -> MetricOutcome:
    return _macro([c.recall for c in _present(confusion(inp))], "recall")


def _f1(inp: MetricInputs) -> MetricOutcome:
    return _macro([c.f1 for c in _present(confusion(inp))], "F1")


def _confusion_metric(inp: MetricInputs) -> MetricOutcome:
    cm = confusion(inp)
    return MetricOutcome(
        Status.COMPUTED,
        None,
        structured={
            "labels": list(cm.labels),
            "counts": [list(r) for r in cm.counts],
            "row_normalized": [list(r) for r in cm.row_normalized],
            "column_normalized": [list(r) for r in cm.column_normalized],
        },
    )


def _score_columns(inp: MetricInputs) -> tuple[list[Label], list[list[float]]] | MetricOutcome:
    """Validated (classes, per-class score columns), or an outcome explaining why not."""
    if inp.scores is None or inp.classes is None:
        return MetricOutcome(Status.UNSUPPORTED, reason="per-class scores are not available")
    k = len(inp.classes)
    if k < 2:
        return _undefined("fewer than two classes")
    for row in inp.scores:
        if len(row) != k:
            return MetricOutcome(Status.INVALID, reason="score rows do not match the classes")
    absent = [c for c in inp.classes if c not in set(inp.y_true)]
    if absent:
        return _undefined(f"class(es) {absent} have no true samples; one-vs-rest is undefined")
    return list(inp.classes), [[row[j] for row in inp.scores] for j in range(k)]


def rank_auc(scores: Sequence[float], positive: Sequence[bool]) -> float | None:
    """ROC-AUC via average ranks (Mann-Whitney U); None if one class is absent."""
    n_pos = sum(positive)
    n_neg = len(positive) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(len(scores)), key=scores.__getitem__)
    rank_sum = 0.0
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        rank_sum += avg_rank * sum(positive[order[m]] for m in range(i, j + 1))
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def average_precision(scores: Sequence[float], positive: Sequence[bool]) -> float | None:
    """Average precision (step-wise area under the PR curve, ties grouped); None if no positives."""
    n_pos = sum(positive)
    if n_pos == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp = fp = 0
    prev_recall = 0.0
    ap = 0.0
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for m in range(i, j + 1):
            if positive[order[m]]:
                tp += 1
            else:
                fp += 1
        recall = tp / n_pos
        ap += (recall - prev_recall) * (tp / (tp + fp))
        prev_recall = recall
        i = j + 1
    return ap


def _ovr(
    inp: MetricInputs, fn: Callable[[Sequence[float], Sequence[bool]], float | None]
) -> MetricOutcome:
    prepared = _score_columns(inp)
    if isinstance(prepared, MetricOutcome):
        return prepared
    classes, columns = prepared
    if len(classes) == 2:  # binary: positive class is the second one
        classes, columns = classes[1:], columns[1:]
    values = []
    for c, col in zip(classes, columns, strict=True):
        v = fn(col, [t == c for t in inp.y_true])
        if v is None:
            return _undefined(f"class {c!r} has no positives or no negatives")
        values.append(v)
    return _ok(sum(values) / len(values))


def _roc_auc(inp: MetricInputs) -> MetricOutcome:
    return _ovr(inp, rank_auc)


def _pr_auc(inp: MetricInputs) -> MetricOutcome:
    return _ovr(inp, average_precision)


def _log_loss(inp: MetricInputs) -> MetricOutcome:
    if inp.scores is None or inp.classes is None:
        return MetricOutcome(Status.UNSUPPORTED, reason="per-class probabilities are not available")
    index: dict[Scalar, int] = {c: i for i, c in enumerate(inp.classes)}
    total = 0.0
    for t, row in zip(inp.y_true, inp.scores, strict=True):
        if t not in index or len(row) != len(index):
            return MetricOutcome(Status.INVALID, reason="targets/scores do not match the classes")
        if any(not 0.0 <= p <= 1.0 for p in row):
            return MetricOutcome(Status.INVALID, reason="probabilities must lie in [0, 1]")
        total -= math.log(min(max(row[index[t]], LOG_LOSS_EPS), 1.0))
    return _ok(total / len(inp.y_true))


# --- regression -------------------------------------------------------------------------------


def _residuals(inp: MetricInputs) -> list[float]:
    return [float(p) - float(t) for t, p in zip(inp.y_true, inp.y_pred, strict=True)]


def _mae(inp: MetricInputs) -> MetricOutcome:
    r = _residuals(inp)
    return _ok(sum(abs(x) for x in r) / len(r))


def _mse(inp: MetricInputs) -> MetricOutcome:
    r = _residuals(inp)
    return _ok(sum(x * x for x in r) / len(r))


def _rmse(inp: MetricInputs) -> MetricOutcome:
    r = _residuals(inp)
    return _ok(math.sqrt(sum(x * x for x in r) / len(r)))


def _r2(inp: MetricInputs) -> MetricOutcome:
    y = [float(t) for t in inp.y_true]
    mean = sum(y) / len(y)
    ss_tot = sum((v - mean) ** 2 for v in y)
    if ss_tot == 0:
        return _undefined("the targets are constant; R^2 is undefined")
    return _ok(1.0 - sum(x * x for x in _residuals(inp)) / ss_tot)


def _median_ae(inp: MetricInputs) -> MetricOutcome:
    return _ok(float(statistics.median(abs(x) for x in _residuals(inp))))


# --- registry ---------------------------------------------------------------------------------

_CLS = frozenset({TaskType.CLASSIFICATION})
_REG = frozenset({TaskType.REGRESSION})
_LAB = frozenset({Requirement.LABELS})
_SCO = frozenset({Requirement.LABELS, Requirement.SCORES})


def _spec(
    id_: str, name: str, tasks: frozenset[TaskType], requires: frozenset[Requirement], scale: str,
    better: bool | None, fn: Callable[[MetricInputs], MetricOutcome], description: str,
    scalar: bool = True,
) -> MetricSpec:  # fmt: skip
    return MetricSpec(
        id_, name, IMPL_VERSION, tasks, requires, scale, better, scalar, description, fn
    )


class MetricRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, MetricSpec] = {}

    def register(self, spec: MetricSpec) -> None:
        if spec.id in self._specs:
            raise DuplicateAdapterError(f"metric {spec.id!r} is already registered")
        self._specs[spec.id] = spec

    def get(self, metric_id: str) -> MetricSpec:
        try:
            return self._specs[metric_id]
        except KeyError:
            raise EvaluationError(
                f"unknown metric {metric_id!r} (registered: {sorted(self._specs)})"
            ) from None

    def ids(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def for_task(self, task: TaskType) -> list[MetricSpec]:
        return [s for s in self._specs.values() if task in s.tasks]

    def resolve(self, requested: Sequence[str], task: TaskType) -> list[MetricSpec]:
        """The requested metrics (all task-compatible ones if none requested). A metric that does
        not apply to the task is an invalid request, not something to skip quietly."""
        if not requested:
            return self.for_task(task)
        specs = [self.get(m) for m in requested]
        bad = [s.id for s in specs if task not in s.tasks]
        if bad:
            raise EvaluationError(f"metric(s) {bad} do not apply to the {task} task")
        return specs


def default_metric_registry() -> MetricRegistry:
    """A fresh registry with the built-in metrics."""
    r = MetricRegistry()
    for s in (
        _spec("accuracy", "Accuracy", _CLS, _LAB, "0..1", True, _accuracy, "Fraction of exact matches"),
        _spec("balanced_accuracy", "Balanced accuracy", _CLS, _LAB, "0..1", True, _balanced_accuracy, "Mean per-class recall over classes with true samples"),
        _spec("precision", "Precision (macro)", _CLS, _LAB, "0..1", True, _precision, "Macro-averaged precision"),
        _spec("recall", "Recall (macro)", _CLS, _LAB, "0..1", True, _recall, "Macro-averaged recall"),
        _spec("f1", "F1 (macro)", _CLS, _LAB, "0..1", True, _f1, "Macro-averaged F1"),
        _spec("confusion_matrix", "Confusion matrix", _CLS, _LAB, "counts", None, _confusion_metric, "Counts (rows true, columns predicted) with row/column normalization", scalar=False),
        _spec("roc_auc", "ROC-AUC (one-vs-rest, macro)", _CLS, _SCO, "0..1", True, _roc_auc, "Needs per-class scores; binary uses the second class as positive"),
        _spec("pr_auc", "PR-AUC / average precision (one-vs-rest, macro)", _CLS, _SCO, "0..1", True, _pr_auc, "Needs per-class scores"),
        _spec("log_loss", "Log loss", _CLS, _SCO, "nats, >= 0", False, _log_loss, "Needs probabilities; probabilities clipped to [1e-15, 1]"),
        _spec("mae", "Mean absolute error", _REG, _LAB, "target units", False, _mae, "Mean of |prediction - target|"),
        _spec("mse", "Mean squared error", _REG, _LAB, "target units squared", False, _mse, "Mean of squared residuals"),
        _spec("rmse", "Root mean squared error", _REG, _LAB, "target units", False, _rmse, "sqrt(MSE)"),
        _spec("r2", "R-squared", _REG, _LAB, "<= 1", True, _r2, "1 - SS_res/SS_tot; undefined for constant targets"),
        _spec("median_absolute_error", "Median absolute error", _REG, _LAB, "target units", False, _median_ae, "Median of |prediction - target|"),
    ):  # fmt: skip
        r.register(s)
    return r


def compute_metric(spec: MetricSpec, inp: MetricInputs) -> MetricOutcome:
    """Run `spec` after enforcing its declared requirements and basic numeric validity."""
    n = len(inp.y_true)
    if n != len(inp.y_pred):
        return MetricOutcome(Status.INVALID, reason="y_true and y_pred differ in length")
    if n == 0:
        return _undefined("no samples")
    if Requirement.SCORES in spec.requires and inp.scores is None:
        return MetricOutcome(Status.UNSUPPORTED, reason="requires per-class scores, none provided")
    if (
        Requirement.SCORES in spec.requires
        and inp.scores is not None
        and (len(inp.scores) != n or any(not math.isfinite(p) for row in inp.scores for p in row))
    ):
        return MetricOutcome(Status.INVALID, reason="scores are missing or non-finite")
    if inp.task is TaskType.REGRESSION:
        for name, seq in (("y_true", inp.y_true), ("y_pred", inp.y_pred)):
            if any(
                isinstance(x, bool) or not isinstance(x, int | float) or not math.isfinite(x)
                for x in seq
            ):
                return MetricOutcome(
                    Status.INVALID, reason=f"{name} has non-numeric or non-finite values"
                )
    return spec.compute(inp)

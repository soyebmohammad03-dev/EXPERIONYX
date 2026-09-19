"""Failure signal extraction: pure functions from STORED evaluation results (and, for fault
treatments, the baseline they are compared with) to normalized FailureSignals. Nothing here
measures anything new; a signal exists only when a stored result crosses a configured threshold.
Classes are keyed by the exact JSON form of the label, so class 2 never equals class 20."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from experionyx.adapters.capabilities import TaskType
from experionyx.domain import to_jsonable
from experionyx.evaluation.config import SliceCondition
from experionyx.evaluation.results import (
    ErrorKind,
    ErrorRecord,
    EvaluationResult,
    Status,
)
from experionyx.failures.config import EXTRACTOR_VERSION, ExtractionConfig
from experionyx.failures.entities import FailureSignal
from experionyx.failures.taxonomy import Direction, FailureCategory, SignalKind
from experionyx.faults.degradation import measure_degradation


def label_key(label: object) -> str:
    """Canonical, exact class key: 2 -> '2', '2' -> '"2"', 20 -> '20' (never a prefix match)."""
    return json.dumps(label, sort_keys=True)


def band(value: float, edges: Sequence[float]) -> int:
    return sum(value >= e for e in edges)


@dataclass(frozen=True)
class FaultContext:
    """What identifies the fault behind a treatment run (from its fault.json and its trial)."""

    type: str
    family_id: str
    fault_id: str
    seed: int
    target: str  # "LABEL" | "INPUT_OR_MIXED"
    parameters: Mapping[str, object]
    parameter_value: float | None
    affected_fraction: float | None
    fault_experiment_id: str
    baseline_run_id: str
    source_dataset_fingerprint: str | None = None  # the un-faulted dataset the fault was applied to


class SignalBuilder:
    def __init__(
        self,
        run_id: str,
        experiment_id: str,
        ev: EvaluationResult,
        cfg: ExtractionConfig,
        config_hash: str,
        now: datetime,
        fault: FaultContext | None = None,
    ) -> None:
        self.run_id, self.experiment_id, self.ev, self.cfg = run_id, experiment_id, ev, cfg
        self.config_hash, self.now, self.fault = config_hash, now, fault
        self.out: list[FailureSignal] = []

    def add(
        self,
        kind: SignalKind,
        category: FailureCategory,
        direction: Direction,
        magnitude: float,
        *,
        cls: object = None,
        pred: object = None,
        slice_name: str | None = None,
        ids: Sequence[int] | None = None,
        count: int | None = None,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        cfg, f = self.cfg, self.fault
        detail: dict[str, object] = {
            "source": "fault" if f else "evaluation",
            "model_fingerprint": self.ev.context.model_fingerprint,
            "dataset_fingerprint": (
                f.source_dataset_fingerprint
                if f and f.source_dataset_fingerprint
                else self.ev.context.dataset_fingerprint
            ),
            "evaluated_dataset_fingerprint": self.ev.context.dataset_fingerprint,
            "n_samples": self.ev.n_samples,
            "task": self.ev.task.value,
            **(extra or {}),
        }
        if cls is not None:
            detail["class_label"] = label_key(cls)
        if pred is not None:
            detail["predicted_label"] = label_key(pred)
        if slice_name is not None:
            detail["slice"] = slice_name
        if f:
            detail["fault"] = {
                "type": f.type, "family_id": f.family_id, "fault_id": f.fault_id, "seed": f.seed,
                "target": f.target, "parameters": dict(f.parameters), "parameter_value": f.parameter_value,
                "affected_fraction": f.affected_fraction, "fault_experiment_id": f.fault_experiment_id,
                "baseline_run_id": f.baseline_run_id,
                "source_dataset_fingerprint": f.source_dataset_fingerprint,
            }  # fmt: skip
        sample_ids = sorted(ids or [])
        total = len(sample_ids) if count is None else count
        detail["sample_ids_recorded"] = ids is not None
        parts = [
            kind.value,
            f"class={detail.get('class_label', '-')}",
            f"pred={detail.get('predicted_label', '-')}",
            f"slice={slice_name or '-'}",
            f"fault={f.type if f else '-'}",
            f"dir={direction.value}",
            f"sev={band(magnitude, cfg.severity_edges)}",
        ]
        self.out.append(
            FailureSignal(
                run_id=self.run_id, experiment_id=self.experiment_id, signal_kind=kind, category=category,
                signature="|".join(parts), direction=direction, magnitude=magnitude, detail=detail,
                sample_ids=tuple(str(i) for i in sample_ids[: cfg.max_sample_ids_per_signal]),
                sample_count=max(total, min(len(sample_ids), cfg.max_sample_ids_per_signal)),
                extractor_version=EXTRACTOR_VERSION, config_hash=self.config_hash, created_at=self.now,
            )
        )  # fmt: skip


def _complete(ev: EvaluationResult, errors: Sequence[ErrorRecord] | None) -> bool:
    return errors is not None and not ev.errors.truncated and ev.errors.status is Status.COMPUTED


def _wrong(errors: Sequence[ErrorRecord]) -> set[int]:
    return {e.sample_index for e in errors if e.true != e.predicted}


def _category_for_slice(conditions: Sequence[SliceCondition]) -> FailureCategory:
    kinds = {c.kind.value for c in conditions}
    if kinds <= {"TARGET_EQUALS", "PREDICTED_EQUALS"}:
        return FailureCategory.CLASS_SPECIFIC_ERROR
    return FailureCategory.FEATURE_DEPENDENCY


def extract_from_evaluation(b: SignalBuilder, errors: Sequence[ErrorRecord] | None) -> None:
    """Signals visible in ONE evaluation (no baseline needed)."""
    ev, cfg = b.ev, b.cfg
    complete = _complete(ev, errors)
    recs = list(errors or [])
    if ev.task is TaskType.CLASSIFICATION and ev.confusion is not None:
        cm = ev.confusion
        for m in cm.per_class:
            if m.recall is not None and m.recall < cfg.class_recall_floor:
                ids = (
                    [e.sample_index for e in recs if e.true == m.label and e.true != e.predicted]
                    if complete
                    else None
                )
                b.add(
                    SignalKind.PER_CLASS_RECALL_LOW,
                    FailureCategory.CLASS_SPECIFIC_ERROR,
                    Direction.WORSE,
                    1.0 - m.recall,
                    cls=m.label,
                    ids=ids,
                    count=round(m.support * (1.0 - m.recall)),
                    extra={"metric": "recall", "recall": m.recall, "support": m.support},
                )
        for i, row in enumerate(cm.row_normalized):
            for j, rate in enumerate(row):
                if i != j and rate is not None and rate >= cfg.confusion_pair_rate_min:
                    a, p = cm.labels[i], cm.labels[j]
                    ids = (
                        [e.sample_index for e in recs if e.true == a and e.predicted == p]
                        if complete
                        else None
                    )
                    b.add(
                        SignalKind.CONFUSION_PAIR,
                        FailureCategory.SYSTEMATIC_MISCLASSIFICATION,
                        Direction.WORSE,
                        rate,
                        cls=a,
                        pred=p,
                        ids=ids,
                        count=cm.counts[i][j],
                        extra={"metric": "row_normalized_confusion", "support": sum(cm.counts[i])},
                    )
    c = ev.confidence
    if (
        c.status is Status.COMPUTED
        and c.n_samples
        and c.high_confidence_errors / c.n_samples >= cfg.high_confidence_error_rate_min
    ):
        ids = (
            [e.sample_index for e in recs if ErrorKind.HIGH_CONFIDENCE_INCORRECT in e.kinds]
            if complete
            else None
        )
        b.add(
            SignalKind.HIGH_CONFIDENCE_ERRORS,
            FailureCategory.HIGH_CONFIDENCE_ERROR,
            Direction.WORSE,
            c.high_confidence_errors / c.n_samples,
            ids=ids,
            count=c.high_confidence_errors,
            extra={
                "metric": "high_confidence_error_rate",
                "threshold": c.high_confidence_threshold,
            },
        )
    cal = ev.calibration
    if cal.status is Status.COMPUTED and cal.ece is not None and cal.ece >= cfg.ece_min:
        b.add(
            SignalKind.CALIBRATION_ECE,
            FailureCategory.CALIBRATION_FAILURE,
            Direction.WORSE,
            cal.ece,
            extra={"metric": "ece", "n_bins": cal.n_bins},
        )
    stats = ev.errors.stats
    mae = stats.get("mean_absolute_error")
    if ev.task is TaskType.REGRESSION and mae:
        tail = stats["max_absolute_error"] / mae
        if tail >= cfg.regression_tail_ratio_min:
            ids = [
                e.sample_index
                for e in recs
                if (e.absolute_error or 0.0) >= cfg.regression_tail_ratio_min * mae
            ]
            b.add(
                SignalKind.REGRESSION_TAIL,
                FailureCategory.REGRESSION_ERROR,
                Direction.WORSE,
                tail,
                ids=ids,
                extra={"metric": "max_abs_error/mae", "mae": mae},
            )
        bias = abs(stats["mean_signed_error"]) / mae
        if bias >= cfg.regression_bias_ratio_min:
            b.add(
                SignalKind.REGRESSION_BIAS,
                FailureCategory.REGRESSION_ERROR,
                Direction.WORSE,
                bias,
                extra={
                    "metric": "|mean_signed_error|/mae",
                    "mean_signed_error": stats["mean_signed_error"],
                },
            )
    higher = {m.metric_id: m.higher_is_better for m in ev.metrics}
    for s in ev.slices:
        for mid, delta in sorted(s.deltas_vs_overall.items()):
            hib = higher.get(mid)
            if delta is None or hib is None:
                continue
            worse = -delta if hib else delta
            if worse >= cfg.slice_deterioration_min:
                b.add(
                    SignalKind.SLICE_DEGRADATION,
                    _category_for_slice(s.conditions),
                    Direction.WORSE,
                    worse,
                    slice_name=s.name,
                    extra={
                        "metric": mid,
                        "support": s.n_samples,
                        "conditions": to_jsonable(s.conditions),
                    },
                )


def extract_from_fault(
    b: SignalBuilder,
    base: EvaluationResult,
    base_errors: Sequence[ErrorRecord] | None,
    errors: Sequence[ErrorRecord] | None,
    primary: str,
) -> None:
    """Signals that a fault made things WORSE than the baseline it is compared with."""
    ev, cfg, f = b.ev, b.cfg, b.fault
    assert f is not None  # noqa: S101
    both = _complete(base, base_errors) and _complete(ev, errors)
    newly = (
        sorted(_wrong(errors or []) - _wrong(base_errors or []))
        if both and ev.task is TaskType.CLASSIFICATION
        else None
    )
    d = next((x for x in measure_degradation(base, ev) if x.metric_id == primary), None)
    if d is not None and d.deterioration is not None and d.deterioration >= cfg.fault_effect_min:
        cat = (
            FailureCategory.REGRESSION_ERROR
            if ev.task is TaskType.REGRESSION
            else FailureCategory.LABEL_ERROR
            if f.target == "LABEL"
            else FailureCategory.INPUT_SENSITIVITY
        )
        b.add(
            SignalKind.METRIC_DEGRADATION,
            cat,
            Direction.WORSE,
            d.deterioration,
            ids=newly,
            extra={
                "metric": primary,
                "baseline": d.baseline,
                "faulted": d.faulted,
                "relative_deterioration": d.relative_deterioration,
                "ids_meaning": "newly wrong: correct in baseline, wrong under the fault",
            },
        )
    if base.confusion is not None and ev.confusion is not None:
        after = {m.label: m for m in ev.confusion.per_class}
        for m in base.confusion.per_class:
            a = after.get(m.label)
            if (
                m.recall is None
                or a is None
                or a.recall is None
                or m.recall - a.recall < cfg.class_recall_drop_min
            ):
                continue
            ids = (
                [i for i in newly if _true_of(errors, i) == m.label] if newly is not None else None
            )
            b.add(
                SignalKind.CLASS_RECALL_DEGRADATION,
                FailureCategory.CLASS_SPECIFIC_ERROR,
                Direction.WORSE,
                m.recall - a.recall,
                cls=m.label,
                ids=ids,
                extra={
                    "metric": "recall",
                    "baseline": m.recall,
                    "faulted": a.recall,
                    "support": m.support,
                },
            )
    be, fe = base.calibration.ece, ev.calibration.ece
    if (
        base.calibration.status is Status.COMPUTED
        and ev.calibration.status is Status.COMPUTED
        and be is not None
        and fe is not None
        and fe - be >= cfg.ece_increase_min
    ):
        b.add(
            SignalKind.ECE_INCREASE,
            FailureCategory.CALIBRATION_FAILURE,
            Direction.WORSE,
            fe - be,
            extra={"metric": "ece", "baseline": be, "faulted": fe},
        )


def _true_of(errors: Sequence[ErrorRecord] | None, index: int) -> object:
    return next((e.true for e in errors or [] if e.sample_index == index), None)

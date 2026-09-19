"""Slice metrics, comparisons and the fault / failure-mode / interaction integrations.

Everything numeric reuses existing engines: metric values from the evaluation metric registry,
metric intervals from the evaluation bootstrap, comparisons/effect sizes/tests/corrections from the
Phase 10 statistics core, and interaction contrasts from the Phase 7 calculators. No score ranks
slices and no result says one slice is better or worse overall; every result is scoped to a
metric, a slice pair, a sample set and a stated evidence status."""

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from experionyx.adapters.capabilities import TaskType
from experionyx.artifacts import ArtifactStore
from experionyx.domain import to_jsonable
from experionyx.errors import ExperionyxError, ValidationError
from experionyx.evaluation.bootstrap import bootstrap_interval as metric_interval
from experionyx.evaluation.calibration import calibration_analysis
from experionyx.evaluation.metrics import MetricInputs, compute_metric, default_metric_registry
from experionyx.evaluation.results import Status as MetricStatus
from experionyx.failures.entities import FailureCluster, FailureEvidence, FailureMode, FailureSignal
from experionyx.failures.extraction import label_key
from experionyx.faults.entities import FaultExperiment
from experionyx.faults.report import analysis_run_id, read_artifact
from experionyx.interactions import calc
from experionyx.interactions.config import InteractionSpec
from experionyx.interactions.entities import InteractionAnalysis
from experionyx.interactions.taxonomy import EffectStatus, Pairing
from experionyx.registry import Registry
from experionyx.slices.data import Baseline, run_rows
from experionyx.slices.evaluate import Membership, MembershipStatus, SampleId
from experionyx.stats import core as st

ANALYSIS_VERSION = "1"
ENGINE_NOTE = (
    "a per-slice, per-metric observation under the stated sample set; it ranks nothing, "
    "is not a fairness or quality score, and says nothing about why"
)
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class SliceConfig:
    min_members: int = 5  # a slice with fewer members is reported but flagged INSUFFICIENT_EVIDENCE
    min_trials: int = 3  # fault / interaction trials needed per slice
    confidence: float = 0.95
    resamples: int = 1000
    seed: int = 0
    method: str = "percentile"  # percentile | bca
    correction: str = "NONE"  # NONE | BONFERRONI | BENJAMINI_HOCHBERG across the comparisons
    alpha: float = 0.05
    metrics: tuple[str, ...] = ()  # empty: every scalar metric of the task

    def __post_init__(self) -> None:
        if self.min_members < 1 or self.min_trials < 2 or self.resamples < 1 or self.seed < 0:
            raise ValidationError("min_members >= 1, min_trials >= 2, resamples >= 1, seed >= 0")
        if not 0.0 < self.confidence < 1.0 or not 0.0 < self.alpha < 1.0:
            raise ValidationError("confidence and alpha must be in (0, 1)")
        if self.method not in st.METHODS or self.correction not in st.CORRECTIONS:
            raise ValidationError(
                f"method in {list(st.METHODS)}, correction in {list(st.CORRECTIONS)}"
            )
        if list(self.metrics) != sorted(set(self.metrics)):
            raise ValidationError("metrics must be sorted and unique")

    def to_dict(self) -> dict[str, object]:
        d = to_jsonable(self)
        assert isinstance(d, dict)  # noqa: S101
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "SliceConfig":
        extra = set(d) - set(cls.__dataclass_fields__)
        if extra:
            raise ValidationError(f"unknown slice setting(s) {sorted(extra)}")
        kw = dict(d)
        if "metrics" in kw:
            kw["metrics"] = tuple(kw["metrics"])  # type: ignore[arg-type]
        return cls(**kw)  # type: ignore[arg-type]


def measure_of(task: TaskType) -> tuple[str, bool]:
    """(per-sample measure name, higher_is_better). Classification: accuracy = mean correctness;
    regression: MAE = mean absolute error. Both are exact means of a per-sample value, which is
    what lets the comparisons use the statistics core without a resampling approximation."""
    return ("accuracy", True) if task is TaskType.CLASSIFICATION else ("mae", False)


def measure_values(
    base_rows: Mapping[int, Mapping[str, Any]], ids: Sequence[SampleId], task: TaskType
) -> tuple[dict[int, float], list[int]]:
    """({id: value}, ids without a usable value). Non-finite values are excluded and reported."""
    vals: dict[int, float] = {}
    bad: list[int] = []
    for i in ids:
        r = base_rows.get(i)  # type: ignore[arg-type]
        if r is None:
            bad.append(i)  # type: ignore[arg-type]
            continue
        t, p = r["true"], r["predicted"]
        if task is TaskType.CLASSIFICATION:
            vals[i] = float(t == p)  # type: ignore[index]
        elif all(
            isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x)
            for x in (t, p)
        ):
            vals[i] = abs(p - t)  # type: ignore[index]
        else:
            bad.append(i)  # type: ignore[arg-type]
    return vals, bad


# -- metrics ---------------------------------------------------------------------------------------------


def slice_metrics(base: Baseline, ids: Sequence[SampleId], cfg: SliceConfig) -> dict[str, Any]:
    rows = [base.rows[i] for i in ids]  # type: ignore[index]
    n = len(rows)
    out: dict[str, Any] = {
        "n_samples": n,
        "evidence": "SUFFICIENT" if n >= cfg.min_members else INSUFFICIENT,
    }
    if n < cfg.min_members:
        out["reason"] = (
            f"{n} member(s) < min_members={cfg.min_members}: values are reported but not reliable"
        )
    scored = n > 0 and all(r.get("scores") is not None for r in rows)
    inp = MetricInputs(
        base.task, [r["true"] for r in rows], [r["predicted"] for r in rows],
        [tuple(r["scores"]) for r in rows] if scored else None, base.classes,
    )  # fmt: skip
    metrics: list[dict[str, Any]] = []
    for spec in default_metric_registry().resolve(cfg.metrics, base.task):
        if not spec.scalar:
            continue
        o = compute_metric(spec, inp)
        rec: dict[str, Any] = {
            "metric_id": spec.id, "status": o.status.value, "value": o.value, "reason": o.reason,
            "higher_is_better": spec.higher_is_better, "n_samples": n, "interval": None,
        }  # fmt: skip
        if o.status is MetricStatus.COMPUTED:
            rec["interval"] = to_jsonable(
                metric_interval(
                    spec, inp, resamples=cfg.resamples, confidence=cfg.confidence, seed=cfg.seed
                )
            )
        metrics.append(rec)
    out["metrics"] = metrics
    if base.task is TaskType.CLASSIFICATION:
        wrong = [r for r in rows if r["true"] != r["predicted"]]
        by_class: dict[str, int] = {}
        for r in wrong:
            by_class[label_key(r["true"])] = by_class.get(label_key(r["true"]), 0) + 1
        out["errors"] = {"n_errors": len(wrong), "error_rate": len(wrong) / n if n else None, "errors_by_true_class": dict(sorted(by_class.items()))}  # fmt: skip
    else:
        vals, bad = measure_values(base.rows, ids, base.task)
        out["errors"] = {"absolute_error": to_jsonable(st.summarize(list(vals.values()))), "excluded_non_finite": len(bad)}  # fmt: skip
    if scored and base.classes is not None:
        cal = calibration_analysis(
            inp.scores, inp.y_true, inp.y_pred, base.classes, n_bins=10, source=None
        )
        out["calibration"] = {"status": cal.status.value, "ece": cal.ece, "mce": cal.mce, "brier_score": cal.brier_score, "reason": cal.reason, "warnings": list(cal.warnings)}  # fmt: skip
    else:
        out["calibration"] = {"status": "UNAVAILABLE", "reason": "per-class scores were not stored for every member"}  # fmt: skip
    out["latency"] = {"status": "UNAVAILABLE", "reason": "latency is measured per batch, not per sample, so it cannot be attributed to a slice"}  # fmt: skip
    return out


# -- comparisons -------------------------------------------------------------------------------------------


def _mean(x: Mapping[Any, float]) -> float | None:
    return statistics.fmean(x.values()) if x else None


def _descriptive(a: float | None, b: float | None) -> dict[str, Any]:
    if a is None or b is None:
        return {
            "difference": None,
            "relative_difference": None,
            "reason": "a group has no valid value",
        }
    return {
        "difference": a - b,
        "relative_difference": None if b == 0 else (a - b) / abs(b),
        "relative_reason": "reference value is 0" if b == 0 else None,
    }  # fmt: skip


def compare_groups(
    base: Baseline, a_ids: Sequence[SampleId], b_ids: Sequence[SampleId], cfg: SliceConfig
) -> dict[str, Any]:
    """`a` minus `b` on the task's per-sample measure. The inferential part (test, effect sizes,
    interval) treats the groups as independent samples of DIFFERENT samples, so it is refused when
    they overlap: the raw difference is still reported."""
    name, hib = measure_of(base.task)
    av, abad = measure_values(base.rows, a_ids, base.task)
    bv, bbad = measure_values(base.rows, b_ids, base.task)
    overlap = sorted(set(a_ids) & set(b_ids), key=lambda x: (isinstance(x, str), x))
    doc: dict[str, Any] = {
        "measure": name, "higher_is_better": hib, "n_a": len(av), "n_b": len(bv),
        "mean_a": _mean(av), "mean_b": _mean(bv), "excluded_a": len(abad), "excluded_b": len(bbad),
        "descriptive": _descriptive(_mean(av), _mean(bv)), "direction_note": "no group is called better or worse overall; the sign is the sign of a minus b for this metric only",
    }  # fmt: skip
    if overlap:
        doc.update(status="UNDEFINED", inference=None, reason=f"the groups share {len(overlap)} sample(s); an independent-groups comparison would be invalid")  # fmt: skip
        return doc
    if len(av) < cfg.min_members or len(bv) < cfg.min_members:
        doc.update(status=INSUFFICIENT, reason=f"a group has fewer than min_members={cfg.min_members} valid samples")  # fmt: skip
    cmp = st.compare(
        [bv[k] for k in sorted(bv)], [av[k] for k in sorted(av)], pairing=Pairing.UNPAIRED,
        method=cfg.method, confidence=cfg.confidence, resamples=cfg.resamples, seed=cfg.seed,
    )  # fmt: skip
    doc["inference"] = to_jsonable(cmp)
    doc.setdefault("status", cmp.status.value)
    return doc


def compare_to_population(
    base: Baseline, a_ids: Sequence[SampleId], cfg: SliceConfig
) -> dict[str, Any]:
    """Slice vs the FULL population. The population contains the slice, so the two are not
    independent: the delta is descriptive, and the inference is made against the complement."""
    everyone = sorted(base.rows)
    rest = [i for i in everyone if i not in set(a_ids)]
    name, _ = measure_of(base.task)
    av, _ = measure_values(base.rows, a_ids, base.task)
    pv, _ = measure_values(base.rows, everyone, base.task)
    return {
        "measure": name, "n_slice": len(av), "n_population": len(pv),
        "descriptive_vs_population": _descriptive(_mean(av), _mean(pv)),
        "vs_rest": compare_groups(base, a_ids, rest, cfg),
        "note": "delta vs the population is descriptive (the slice is part of the population); the test and effect sizes compare the slice with the rest",
    }  # fmt: skip


def correct_family(comparisons: Mapping[str, dict[str, Any]], cfg: SliceConfig) -> dict[str, Any]:
    """Multiple-comparison record over the tests actually run (Phase 10 correction)."""
    ps: dict[str, float | None] = {}
    for k, c in comparisons.items():
        inf = c.get("inference")
        test = inf.get("test") if isinstance(inf, dict) else None
        ps[k] = test.get("p_value") if isinstance(test, dict) else None
    corr = st.adjust_pvalues(ps, method=cfg.correction, alpha=cfg.alpha)
    for k, c in comparisons.items():
        c["multiplicity"] = {"raw_p": ps[k], "adjusted_p": corr.adjusted.get(k), "significant_after_correction": corr.rejected.get(k) if cfg.correction != "NONE" else None}  # fmt: skip
    return to_jsonable(corr)  # type: ignore[return-value]


# -- faults x slices --------------------------------------------------------------------------------------


def _sign(hib: bool, baseline: float, faulted: float) -> float:
    return baseline - faulted if hib else faulted - baseline  # > 0 means worse


def fault_slice(
    reg: Registry,
    store: ArtifactStore,
    base: Baseline,
    mem: Membership,
    fxp_id: str,
    cfg: SliceConfig,
) -> dict[str, Any]:
    head: dict[str, Any] = {"fault_experiment_id": fxp_id, "slice_id": mem.slice_id}
    fe = reg.get(FaultExperiment, fxp_id)
    if fe.baseline_run_id != base.run_id:
        return {**head, "status": INSUFFICIENT, "reason": f"the fault experiment was run against baseline {fe.baseline_run_id}, not {base.run_id}"}  # fmt: skip
    if mem.status is not MembershipStatus.COMPUTED:
        return {
            **head,
            "status": INSUFFICIENT,
            "reason": f"the slice has no members ({mem.status.value})",
        }
    if len(mem.sample_ids) < cfg.min_members:
        return {**head, "status": INSUFFICIENT, "reason": f"{len(mem.sample_ids)} member(s) < min_members={cfg.min_members}"}  # fmt: skip
    name, hib = measure_of(base.task)
    ids = list(mem.sample_ids)
    everyone = sorted(base.rows)
    b_slice = _mean(measure_values(base.rows, ids, base.task)[0])
    b_all = _mean(measure_values(base.rows, everyone, base.task)[0])
    trials = read_artifact(reg, store, analysis_run_id(reg, fxp_id), "fault/trials.json")
    if not isinstance(trials, list):
        return {**head, "status": "UNAVAILABLE", "reason": "fault/trials.json is not a list"}
    points: dict[int, list[dict[str, Any]]] = {}
    for t in trials:
        if t.get("status") == "COMPLETED" and t.get("run_id"):
            points.setdefault(int(t["point_index"]), []).append(t)
    out_points = []
    excluded: list[dict[str, str]] = []
    for pi in sorted(points):
        slice_det, all_det, slice_f, slice_b = {}, {}, {}, {}
        for t in points[pi]:
            key = f"seed{t['seed']}/repeat{t['repeat_index']}"
            try:
                rows = run_rows(reg, store, t["run_id"])
            except ExperionyxError as exc:
                excluded.append({"trial": key, "reason": str(exc)})
                continue
            if any(i not in rows for i in ids):
                excluded.append(
                    {
                        "trial": key,
                        "reason": "the trial run did not evaluate every member (different sample set)",
                    }
                )
                continue
            f_slice = _mean(measure_values(rows, ids, base.task)[0])
            f_all = _mean(measure_values(rows, [i for i in everyone if i in rows], base.task)[0])
            if f_slice is None or f_all is None or b_slice is None or b_all is None:
                excluded.append({"trial": key, "reason": "the measure is undefined for this trial"})
                continue
            slice_f[key], slice_b[key] = f_slice, b_slice
            slice_det[key] = _sign(hib, b_slice, f_slice)
            all_det[key] = _sign(hib, b_all, f_all)
        n = len(slice_det)
        rec: dict[str, Any] = {"point_index": pi, "parameter_value": points[pi][0].get("parameter_value"), "n_trials": n, "measure": name, "baseline_slice": b_slice, "baseline_population": b_all}  # fmt: skip
        if n < cfg.min_trials:
            rec.update(status=INSUFFICIENT, effect_classification=INSUFFICIENT, reason=f"{n} usable trial(s) < min_trials={cfg.min_trials}")  # fmt: skip
        if n >= 2:
            rec["slice_degradation"] = to_jsonable(st.compare(slice_b, slice_f, pairing=Pairing.PAIRED, method=cfg.method, confidence=cfg.confidence, resamples=cfg.resamples, seed=cfg.seed))  # fmt: skip
            diff = st.compare(all_det, slice_det, pairing=Pairing.PAIRED, method=cfg.method, confidence=cfg.confidence, resamples=cfg.resamples, seed=cfg.seed)  # fmt: skip
            rec["slice_vs_population_degradation"] = to_jsonable(diff)
            rec["trial_degradation"] = {"slice": dict(sorted(slice_det.items())), "population": dict(sorted(all_det.items()))}  # fmt: skip
            iv = diff.interval
            if n >= cfg.min_trials:
                excl = (
                    iv.lower is not None and iv.upper is not None and (iv.lower > 0 or iv.upper < 0)
                )
                rec["status"] = "COMPUTED"
                rec["effect_classification"] = (
                    ("SLICE_MORE_DEGRADED" if (iv.estimate or 0) > 0 else "SLICE_LESS_DEGRADED") if excl
                    else "NO_EVIDENCE_OF_DIFFERENCE" if iv.status is st.Status.DERIVED else INSUFFICIENT
                )  # fmt: skip
        else:
            rec.setdefault("status", INSUFFICIENT)
        rec["classification_note"] = "the interval of (slice degradation - population degradation) excludes 0, or it does not; NO_EVIDENCE_OF_DIFFERENCE is not evidence of absence, and nothing here says why"  # fmt: skip
        out_points.append(rec)
    return {**head, "status": "COMPUTED" if out_points else INSUFFICIENT, "measure": name, "points": out_points, "excluded_trials": excluded, "membership_basis": "baseline sample table (target and features of the original dataset)"}  # fmt: skip


# -- failure modes x slices ----------------------------------------------------------------------------------


def failure_slice(
    reg: Registry,
    store: ArtifactStore,
    base: Baseline,
    mem: Membership,
    mode_ids: Sequence[str],
    cfg: SliceConfig,
) -> dict[str, Any]:
    if mem.status is not MembershipStatus.COMPUTED:
        return {"slice_id": mem.slice_id, "status": INSUFFICIENT, "reason": f"the slice has no members ({mem.status.value})", "modes": []}  # fmt: skip
    members = {int(i) for i in mem.sample_ids}
    population = set(base.rows)
    modes = []
    for mid in mode_ids:
        m = reg.get(FailureMode, mid)
        cl = reg.get(FailureCluster, m.cluster_id)
        touched: set[int] = set()
        runs_touching: set[str] = set()
        runs_all: set[str] = set()
        problems: list[str] = []
        used = 0
        for sid in cl.signal_ids:
            s = reg.get(FailureSignal, sid)
            runs_all.add(s.run_id)
            fp = _signal_dataset(reg, store, s)
            if fp != base.dataset_fingerprint:
                problems.append(f"signal {sid[:12]} is from a different dataset ({fp})")
                continue
            if not s.detail.get("sample_ids_recorded") or s.sample_count > len(s.sample_ids):
                problems.append(f"signal {sid[:12]} did not record every affected sample ID")
                continue
            ids = {int(x) for x in s.sample_ids}
            used += 1
            touched |= ids
            if ids & members:
                runs_touching.add(s.run_id)
        head: dict[str, Any] = {"mode_id": mid, "lifecycle_status": m.status.value, "confirmed": m.status.value == "CONFIRMED", "confirmation_note": "only CONFIRMED modes are confirmed; every other state is unconfirmed evidence", "n_signals": len(cl.signal_ids), "n_signals_used": used, "evidence_records": len(reg.find(FailureEvidence, failure_mode_id=mid))}  # fmt: skip
        if problems or used == 0:
            modes.append({**head, "status": INSUFFICIENT, "reason": "; ".join(problems) or "no usable signals", "problems": problems})  # fmt: skip
            continue
        in_slice, outside = touched & members, touched & (population - members)
        modes.append({
            **head, "status": "COMPUTED", "present_in_slice": bool(in_slice),
            "slice": _prop(len(in_slice), len(members), cfg), "outside_slice": _prop(len(outside), len(population - members), cfg),
            "population": _prop(len(touched & population), len(population), cfg),
            "runs_with_signal": len(runs_all), "runs_touching_slice": len(runs_touching),
            "sample_basis": "distinct samples touched by at least one signal of the mode; samples are not independent trials, so the interval describes this sample set only",
        })  # fmt: skip
    return {"slice_id": mem.slice_id, "status": "COMPUTED", "modes": modes}


def _prop(k: int, n: int, cfg: SliceConfig) -> dict[str, Any]:
    return {
        "samples_affected": k,
        "samples": n,
        "interval": to_jsonable(st.proportion_interval(k, n, cfg.confidence)),
    }


def _signal_dataset(reg: Registry, store: ArtifactStore, s: FailureSignal) -> str | None:
    fault = s.detail.get("fault")
    if isinstance(fault, Mapping) and fault.get("source_dataset_fingerprint"):
        return str(fault["source_dataset_fingerprint"])
    from experionyx.evaluation.loading import load_evaluation

    try:
        return load_evaluation(reg, store, s.run_id).context.dataset_fingerprint
    except ExperionyxError:
        return None


# -- interactions x slices ---------------------------------------------------------------------------------


def interaction_slice(
    reg: Registry,
    store: ArtifactStore,
    base: Baseline,
    mem: Membership,
    ian_id: str,
    cfg: SliceConfig,
) -> dict[str, Any]:
    head: dict[str, Any] = {"interaction_id": ian_id, "slice_id": mem.slice_id}

    def no(reason: str) -> dict[str, Any]:
        return {
            **head,
            "status": INSUFFICIENT,
            "reason": reason,
            "fallback": "none: population-level results are never substituted",
        }

    if mem.status is not MembershipStatus.COMPUTED:
        return no(f"the slice has no members ({mem.status.value})")
    ids = list(mem.sample_ids)
    if len(ids) < cfg.min_members:
        return no(f"{len(ids)} member(s) < min_members={cfg.min_members}")
    a = reg.get(InteractionAnalysis, ian_id)
    spec = InteractionSpec.from_dict(a.spec)
    if base.run_id not in spec.control:
        return no(f"the interaction's control is not the baseline run {base.run_id}")
    trials = read_artifact(reg, store, a.run_id, "interaction/trials.json")
    design = read_artifact(reg, store, a.run_id, "interaction/design_validation.json")
    if not isinstance(trials, dict) or not isinstance(design, dict):
        return no("the interaction artifacts are not readable")
    name, hib = measure_of(base.task)
    cells: dict[str, dict[str, float]] = {}
    try:
        for cell, ts in trials.items():
            cells[cell] = {}
            for t in ts:
                rows = run_rows(reg, store, t["run_id"])
                missing = [i for i in ids if i not in rows]
                if missing:
                    return no(f"run {t['run_id']} did not evaluate {len(missing)} member sample(s)")
                if any(rows[i]["true"] != base.rows[i]["true"] for i in ids):  # type: ignore[index]
                    return no(
                        f"run {t['run_id']} has different ground truth for slice members (label faults are refused)"
                    )
                v = _mean(measure_values(rows, ids, base.task)[0])
                if v is None:
                    return no(f"the measure is undefined for run {t['run_id']}")
                cells[cell][t["key"]] = v
    except ExperionyxError as exc:
        return no(str(exc))
    icfg = spec.config
    pairing = Pairing(str(design["pairing"]))
    boot = calc.bootstrap(cells, icfg, pairing)
    if boot.status is not EffectStatus.COMPUTED:
        return {**head, "status": INSUFFICIENT if boot.status is EffectStatus.INSUFFICIENT_DATA else "UNDEFINED", "reason": boot.reason, "n_members": len(ids), "trials_per_cell": {c: len(v) for c, v in cells.items()}, "fallback": "none: population-level results are never substituted"}  # fmt: skip
    agg = {c: calc.aggregate(list(v.values()), icfg.aggregation) for c, v in cells.items()}
    try:
        point = calc.stats_of(agg, icfg.normalization)
    except ValueError as exc:
        return no(str(exc))
    klass, rule = calc.classify(icfg, point, boot)
    oriented, relation = calc.deterioration_view(point["interaction_contrast"] or 0.0, hib)
    return {
        **head, "status": "COMPUTED", "measure": name, "n_members": len(ids), "pairing": pairing.value,
        "trials_per_cell": {c: len(v) for c, v in cells.items()}, "cell_values": {c: dict(sorted(v.items())) for c, v in cells.items()},
        "derived": point, "bootstrap": to_jsonable(boot),
        "interpreted": {"class": klass.value, "rule": rule, "deterioration_contrast": oriented, "deterioration_relation": relation.value, "note": "an observed contrast on this slice under the interaction's design; not a causal or mechanistic claim"},
    }  # fmt: skip

"""How stressed evaluations are compared with their baseline. Everything numeric reuses existing code:
metric values and their directions come from the evaluation results, per-sample measures from the
slice layer, and comparisons, effect sizes, intervals, tests and corrections from Phase 10.

Conventions: every difference is STRESSED minus BASELINE. `deterioration` is direction-aware and
positive when the stressed value is WORSE for that metric (baseline - stressed if higher is better,
stressed - baseline otherwise). Relative change is undefined (with a reason) when the baseline is 0.
Nothing here is a score, a rank or a verdict: a p-value is a number under stated assumptions, not a
claim of practical importance, and no result says why a change happened."""

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from experionyx.domain import to_jsonable
from experionyx.evaluation.results import EvaluationResult, MetricResult
from experionyx.evaluation.results import Status as MetricStatus
from experionyx.interactions.taxonomy import Pairing
from experionyx.slices import analysis as slice_an
from experionyx.stats import core as st

NOTE = (
    "an observed change under a deliberately applied, recorded stress; it is not a robustness score, "
    "does not say why the change happened, and statistical evidence is not practical importance"
)


def _delta(base: float | None, new: float | None, hib: bool | None) -> dict[str, Any]:
    if base is None or new is None or not math.isfinite(base) or not math.isfinite(new):
        return {"absolute_change": None, "relative_change": None, "relative_reason": "a value is unavailable", "deterioration": None, "direction": "UNKNOWN"}  # fmt: skip
    d = new - base
    rel = None if base == 0 else d / abs(base)
    det = None if hib is None else (base - new if hib else new - base)
    return {
        "absolute_change": d, "relative_change": rel,
        "relative_reason": "the baseline value is 0" if base == 0 else None,
        "deterioration": det,
        "direction": "UNKNOWN" if hib is None else "HIGHER_IS_BETTER" if hib else "LOWER_IS_BETTER",
    }  # fmt: skip


def _interval(m: MetricResult | None) -> dict[str, Any] | None:
    i = None if m is None else m.interval
    if i is None or i.status is not MetricStatus.COMPUTED or i.lower is None or i.upper is None:
        return None
    return {"lower": i.lower, "upper": i.upper, "confidence": i.confidence, "resamples": i.resamples, "seed": i.seed}  # fmt: skip


def metric_deltas(base: EvaluationResult, trial: EvaluationResult) -> list[dict[str, Any]]:
    """Every scalar metric both evaluations report, raw values kept."""
    bm = {m.metric_id: m for m in base.metrics if m.structured is None}
    tm = {m.metric_id: m for m in trial.metrics if m.structured is None}
    out = []
    for mid in sorted(bm.keys() | tm.keys()):
        b, t = bm.get(mid), tm.get(mid)
        hib = (b or t).higher_is_better  # type: ignore[union-attr]
        bv = b.value if b is not None and b.status is MetricStatus.COMPUTED else None
        tv = t.value if t is not None and t.status is MetricStatus.COMPUTED else None
        out.append({"metric_id": mid, "baseline_value": bv, "stressed_value": tv, "baseline_status": None if b is None else b.status.value, "stressed_status": None if t is None else t.status.value, "n_baseline": None if b is None else b.n_samples, "n_stressed": None if t is None else t.n_samples, "baseline_interval": _interval(b), "stressed_interval": _interval(t), **_delta(bv, tv, hib)})  # fmt: skip
    return out


def sample_values(rows: Mapping[int, Mapping[str, Any]], ids: Sequence[int], task: Any) -> tuple[dict[int, float], list[int]]:  # fmt: skip
    return slice_an.measure_values(rows, ids, task)


def paired_measure(
    base_rows: Mapping[int, Mapping[str, Any]], trial_rows: Mapping[int, Mapping[str, Any]], task: Any,
    ids: Sequence[int] | None, *, min_members: int, confidence: float, resamples: int,
    permutations: int, seed: int, method: str = "percentile",
) -> dict[str, Any]:  # fmt: skip
    """Baseline vs stressed on the task's per-sample measure (accuracy or absolute error), restricted
    to `ids` (a slice's members; None = all). PAIRED when both runs evaluated the same samples with
    the same ground truth, so the difference is per sample; otherwise UNPAIRED and says why."""
    name, hib = slice_an.measure_of(task)
    keys = sorted(base_rows if ids is None else ids)
    tkeys = sorted(trial_rows if ids is None else ids)
    aligned = set(base_rows) == set(trial_rows) and all(base_rows[i]["true"] == trial_rows[i]["true"] for i in keys if i in trial_rows)  # fmt: skip
    bv, bbad = sample_values(base_rows, [i for i in keys if i in base_rows], task)
    tv, tbad = sample_values(trial_rows, [i for i in tkeys if i in trial_rows], task)
    doc: dict[str, Any] = {"measure": name, "higher_is_better": hib, "n_baseline": len(bv), "n_stressed": len(tv), "excluded_nonfinite_baseline": len(bbad), "excluded_nonfinite_stressed": len(tbad), "direction_note": "differences are stressed minus baseline; deterioration is positive when the stressed value is worse"}  # fmt: skip
    common = sorted(set(bv) & set(tv))
    pairing = Pairing.PAIRED if aligned and len(common) == len(bv) == len(tv) else Pairing.UNPAIRED
    doc["pairing"] = pairing.value
    if pairing is Pairing.UNPAIRED:
        doc["pairing_reason"] = "the two runs did not evaluate the same samples with the same ground truth (or some were excluded), so no per-sample pairing is claimed"  # fmt: skip
    doc["mean_baseline"] = statistics.fmean(bv.values()) if bv else None
    doc["mean_stressed"] = statistics.fmean(tv.values()) if tv else None
    doc |= _delta(doc["mean_baseline"], doc["mean_stressed"], hib)
    n = len(common) if pairing is Pairing.PAIRED else min(len(bv), len(tv))
    doc["n_compared"] = n
    if n < 2:
        doc.update(status="INSUFFICIENT_EVIDENCE", inference=None, reason=f"{n} comparable sample(s): a statistical comparison needs at least 2")  # fmt: skip
        return doc
    if pairing is Pairing.PAIRED:
        ref = {str(i): bv[i] for i in common}
        trt = {str(i): tv[i] for i in common}
    else:
        ref, trt = {str(i): v for i, v in bv.items()}, {str(i): v for i, v in tv.items()}
    cmp = st.compare(ref, trt, pairing=pairing, method=method, confidence=confidence, resamples=resamples, permutations=permutations, seed=seed)  # fmt: skip
    doc["inference"] = to_jsonable(cmp)
    doc["status"] = "INSUFFICIENT_EVIDENCE" if n < min_members else cmp.status.value
    if n < min_members:
        doc["reason"] = f"{n} comparable sample(s) < min_members={min_members}: values and the interval are reported, but not as reliable evidence"  # fmt: skip
    return doc


def p_value(doc: Mapping[str, Any]) -> float | None:
    inf = doc.get("inference")
    t = inf.get("test") if isinstance(inf, Mapping) else None
    p = t.get("p_value") if isinstance(t, Mapping) else None
    return p if doc.get("status") in ("DERIVED", "INCONCLUSIVE") and isinstance(p, float) else None


def correct(
    entries: Sequence[tuple[str, dict[str, Any]]], method: str, alpha: float
) -> dict[str, Any]:
    """Phase 10 correction over one family: the paired/unpaired tests of one stress design. Each
    doc gets `multiplicity`; results without a valid p-value are listed, not dropped."""
    ps = {k: p_value(d) for k, d in entries}
    corr = st.adjust_pvalues(ps, method=method, alpha=alpha)
    for k, d in entries:
        d["multiplicity"] = {"raw_p": ps[k], "adjusted_p": corr.adjusted.get(k), "adjusted_p_below_alpha": corr.rejected.get(k) if method != "NONE" else None, "method": method, "alpha": alpha}  # fmt: skip
    c = to_jsonable(corr)
    assert isinstance(c, dict)  # noqa: S101
    return {**c, "members": [k for k, _ in entries], "definition": "the hypothesis tests of this stress design: one per trial (point x repeat x cell) on the primary per-sample measure"}  # fmt: skip


def aggregate(values: Sequence[float], *, confidence: float, resamples: int, seed: int, method: str) -> dict[str, Any]:  # fmt: skip
    """Deterioration over the repeats of one point: raw values, summary, and a bootstrap interval only
    when there are at least two (otherwise INCONCLUSIVE with the reason)."""
    doc: dict[str, Any] = {"n": len(values), "values": list(values), "summary": to_jsonable(st.summarize(list(values)))}  # fmt: skip
    if len(values) < 2:
        doc["interval"] = {"status": "INCONCLUSIVE", "reason": "one trial: no spread to estimate; the value is reported as observed"}  # fmt: skip
    else:
        doc["interval"] = to_jsonable(st.bootstrap_interval(list(values), method=method, confidence=confidence, resamples=resamples, seed=seed))  # fmt: skip
    return doc


def stability(rows: Sequence[Mapping[int, Mapping[str, Any]]], digests: Sequence[str | None], metric_values: Sequence[Mapping[str, float | None]]) -> dict[str, Any]:  # fmt: skip
    """Output stability across repeats of the SAME evaluation. Observed, never manufactured: the
    observation is DETERMINISTIC only if every predictions digest and every metric value is identical."""
    if len(rows) < 2:
        return {
            "n_repeats": len(rows),
            "observed": "INSUFFICIENT_EVIDENCE",
            "reason": "fewer than two repeats: stability cannot be observed",
        }
    base = rows[0]
    agree, score_diff = [], []
    for r in rows[1:]:
        common = sorted(set(base) & set(r))
        agree.append(sum(base[i]["predicted"] == r[i]["predicted"] for i in common) / len(common) if common else None)  # fmt: skip
        diffs = [max(abs(a - b) for a, b in zip(base[i]["scores"], r[i]["scores"], strict=True)) for i in common if base[i].get("scores") and r[i].get("scores")]  # fmt: skip
        score_diff.append(max(diffs) if diffs else None)
    same_digests = len(set(digests)) == 1 and None not in digests
    metrics_same = all(m == metric_values[0] for m in metric_values[1:])
    observed = "DETERMINISTIC" if same_digests and metrics_same else "NONDETERMINISTIC" if len(digests) > 1 else "INSUFFICIENT_EVIDENCE"  # fmt: skip
    return {"n_repeats": len(rows), "observed": observed, "identical_prediction_digests": same_digests, "identical_metric_values": metrics_same, "prediction_agreement_with_first_repeat": agree, "max_score_difference_from_first_repeat": score_diff, "note": "observed over these repeats only; a DETERMINISTIC observation does not promise determinism on other hardware or library versions"}  # fmt: skip


def latency_variation(seconds: Sequence[float]) -> dict[str, Any]:
    s = st.summarize(list(seconds))
    mean = s.mean.value
    std = s.std.value
    return {"seconds": list(seconds), "summary": to_jsonable(s), "coefficient_of_variation": None if not mean or std is None else std / mean, "note": "wall-clock time is hardware and load dependent; it is reported, never used as evidence of a stress effect"}  # fmt: skip

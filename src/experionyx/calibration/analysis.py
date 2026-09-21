"""One analyzed CONTEXT (the baseline, a slice, a window, a stress trial, a calibrated evaluation
part) and the comparison of two contexts. Numbers come from `measures`; statistics come from the
Phase 10 core (Wilson intervals, `compare`, `adjust_pvalues`); distribution changes come from the
Phase 12 measures. Nothing here ranks contexts or forms a composite score, and every difference is
descriptive: it says the metric changed between the two sample sets, not why."""

import math
from collections.abc import Mapping, Sequence
from typing import Any

from experionyx.calibration import measures as m
from experionyx.calibration.spec import CalibrationSpec
from experionyx.domain import to_jsonable
from experionyx.drift import measures as dm
from experionyx.hashing import content_hash
from experionyx.interactions.taxonomy import Pairing
from experionyx.stats import core as st
from experionyx.stats.core import Status as StatStatus

COMPUTED, INSUFFICIENT, UNAVAILABLE = "COMPUTED", "INSUFFICIENT_EVIDENCE", "UNAVAILABLE"
NOTE = (
    "a per-context, per-metric observation of stored predictions; confidence is not automatically "
    "uncertainty, every value depends on the binning and the sample, and no score ranks contexts"
)
UNSUPPORTED_UNCERTAINTY = {
    "stochastic_trials": {
        "status": UNAVAILABLE,
        "reason": "no adapter in this repository exposes repeated stochastic prediction, and the baseline stores one output per sample",
    },
    "ensemble_disagreement": {
        "status": UNAVAILABLE,
        "reason": "no adapter in this repository exposes ensemble members",
    },
    "epistemic_aleatoric": {
        "status": UNAVAILABLE,
        "reason": "no decomposition is claimed: it requires an architecture that supports one, and none is declared",
    },
}
PER_SAMPLE = ("accuracy", "brier_top_label", "log_loss_top_label", "mean_confidence")
NON_MEAN = ("ece", "mce")
PRACTICAL = ("ece", "mce", "brier_top_label", "accuracy", "mean_confidence")


def digest_of(ids: Sequence[object]) -> str:
    return content_hash({"ids": sorted(ids, key=str)})


def _mean(xs: Sequence[float]) -> float | None:
    return math.fsum(xs) / len(xs) if xs else None


def prediction_uncertainty(obs: Sequence[m.Obs]) -> dict[str, Any]:
    """Descriptive dispersion of the stored predictive distribution. It is a property of THIS model's
    output vector, not an epistemic or aleatoric uncertainty."""
    ent = [m.entropy(o.probs) for o in obs]
    nent = [x for o in obs if (x := m.normalized_entropy(o.probs)) is not None]
    mar = [x for o in obs if (x := m.margin(o.probs)) is not None]
    ok = [o.correct for o in obs]

    def by(vals: Sequence[float]) -> dict[str, float | None]:
        return {"mean": _mean(vals), "mean_when_correct": _mean([x for x, c in zip(vals, ok, strict=True) if c]), "mean_when_incorrect": _mean([x for x, c in zip(vals, ok, strict=True) if not c])}  # fmt: skip

    return {
        "status": COMPUTED,
        "predictive_entropy_nats": by(ent),
        "normalized_entropy": by(nent) if len(nent) == len(ent) else None,
        "top2_margin": by(mar) if len(mar) == len(obs) else None,
        "formulas": {
            "entropy": "H = -sum_k p_k ln p_k (nats, 0 ln 0 = 0)",
            "normalized_entropy": "H / ln(K), K = number of classes (needs K >= 2)",
            "top2_margin": "p_(1) - p_(2)",
        },
        "interpretation": "dispersion of this model's predicted distribution; NOT an epistemic/aleatoric decomposition and NOT a calibrated uncertainty",
        **UNSUPPORTED_UNCERTAINTY,
    }


def context_result(
    obs: Sequence[m.Obs], spec: CalibrationSpec, classes: Sequence[object], *, key: str, kind: str,
    n_members: int, dq: Mapping[str, Any], calibrated: bool = False, extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:  # fmt: skip
    """Bins, metrics, intervals and prediction uncertainty of one context. Below `min_samples`, or
    with too few observations for the requested quantile bins, the status is INSUFFICIENT_EVIDENCE:
    the bins are shown (never hidden) but no calibration metric or bootstrap interval is produced."""
    cfg = spec.statistics
    b = spec.binning
    n = len(obs)
    out: dict[str, Any] = {
        "key": key,
        "kind": kind,
        "n_members": n_members,
        "n_samples": n,
        "sample_digest": digest_of([o.id for o in obs]),
        "status": COMPUTED,
        "reason": None,
        "data_quality": dict(dq),
        **(extra or {}),
        "binning": {**b.to_dict(), "min_bin_samples": cfg.min_bin_samples},
        "bins": {"top_label": [], "classwise": None},
        "metrics": {"top_label": None, "probability_vector": None, "classwise": None},
        "uncertainty": {"accuracy": None, "bootstrap": None},
        "prediction_uncertainty": {
            "status": UNAVAILABLE,
            "reason": "not computed for this context",
        },
    }
    if n == 0:
        return {**out, "status": UNAVAILABLE, "reason": "no usable observations in this context"}
    pairs = m.top_pairs(obs)
    hits = sum(e for _, e in pairs)
    out["uncertainty"]["accuracy"] = {"estimate": hits / n, **m.wilson(hits, n, cfg.confidence), "assumption": "independent identically distributed Bernoulli trials"}  # fmt: skip
    if b.strategy == m.QUANTILE and n < m.QUANTILE_PER_BIN * b.n_bins:
        return {**out, "status": INSUFFICIENT, "reason": f"quantile binning needs at least {m.QUANTILE_PER_BIN} observations per requested bin ({m.QUANTILE_PER_BIN * b.n_bins}); this context has {n}"}  # fmt: skip
    edges = m.make_edges(b.strategy, b.n_bins, [p for p, _ in pairs])
    out["bins"]["top_label"] = m.curve(pairs, edges, min_bin=cfg.min_bin_samples, confidence=cfg.confidence)  # fmt: skip
    if n < cfg.min_samples:
        return {**out, "status": INSUFFICIENT, "reason": f"{n} observation(s) < min_samples={cfg.min_samples}: calibration metrics and bootstrap intervals are withheld"}  # fmt: skip
    vector = not calibrated
    out["metrics"]["top_label"] = m.top_metrics(obs, b.strategy, b.n_bins, vector=False)
    if vector:
        out["metrics"]["probability_vector"] = {"brier_multiclass": m.vector_brier(obs), "nll_multiclass": m.vector_nll(obs)}  # fmt: skip
    else:
        out["metrics"]["probability_vector"] = {"status": UNAVAILABLE, "reason": "a post-hoc top-label calibrator defines a calibrated confidence, not a calibrated probability vector"}  # fmt: skip
    if vector and "CLASSWISE" in spec.objects:
        cw = m.classwise(obs, classes, b.strategy, b.n_bins, min_bin=cfg.min_bin_samples, min_positives=cfg.min_class_positives, confidence=cfg.confidence)  # fmt: skip
        out["bins"]["classwise"] = {k: r.pop("bins") for k, r in cw["per_class"].items()}
        out["metrics"]["classwise"] = cw
    boot = m.bootstrap_metrics(obs, b.strategy, b.n_bins, resamples=cfg.resamples, confidence=cfg.confidence, seed=cfg.seed, vector=vector)  # fmt: skip
    out["uncertainty"]["bootstrap"] = {
        "method": "bootstrap-percentile",
        "resamples": cfg.resamples,
        "seed": cfg.seed,
        "confidence": cfg.confidence,
        "metrics": boot,
        "assumptions": [
            "observations are independent and identically distributed",
            "the sample is representative of the population of interest",
            "ECE is positively biased in small samples; the percentile interval does not correct that",
            "quantile bin edges are recomputed on every resample",
        ],
    }
    out["prediction_uncertainty"] = prediction_uncertainty(obs) if vector else {"status": UNAVAILABLE, "reason": "a calibrated top-label confidence carries no probability vector"}  # fmt: skip
    return out


# -- comparison ---------------------------------------------------------------------------------------------------


def _per_sample(o: m.Obs) -> dict[str, float]:
    p = (o.conf, o.correct)
    return {"accuracy": float(o.correct), "mean_confidence": o.conf, "brier_top_label": m.top_brier([p]), "log_loss_top_label": m.top_log_loss([p])}  # fmt: skip


def compare_contexts(
    a: Sequence[m.Obs], b: Sequence[m.Obs], spec: CalibrationSpec, *, key_a: str, key_b: str, family: str,
) -> dict[str, Any]:  # fmt: skip
    """Metric differences (b minus a) between two sample sets. Paired when the two sets have the same
    sample IDs, unpaired when they are disjoint; partially overlapping sets are neither, so no
    inference is made. Non-mean metrics (ECE, MCE) use the bootstrap / permutation of `measures`;
    per-sample means use the Phase 10 `compare` (paired or unpaired) with its effect sizes."""
    cfg = spec.statistics
    ia, ib = {o.id for o in a}, {o.id for o in b}
    out: dict[str, Any] = {"a": key_a, "b": key_b, "family": family, "n": [len(a), len(b)], "status": COMPUTED, "reason": None, "pairing": None, "metrics": {}, "distribution": None, "note": "a difference between two sample sets; it does not say why the calibration differs"}  # fmt: skip
    if not a or not b:
        return {**out, "status": UNAVAILABLE, "reason": "a side has no usable observations"}
    if ia == ib:
        pairing = Pairing.PAIRED
    elif not ia & ib:
        pairing = Pairing.UNPAIRED
    else:
        return {**out, "status": UNAVAILABLE, "reason": f"the sample IDs partially overlap ({len(ia & ib)} shared): the sets are neither paired nor independent, so no inference is made"}  # fmt: skip
    out["pairing"] = pairing.value
    if min(len(a), len(b)) < cfg.min_samples:
        return {**out, "status": INSUFFICIENT, "reason": f"a side has fewer than min_samples={cfg.min_samples} observations"}  # fmt: skip
    sa, sb = sorted(a, key=lambda o: o.id), sorted(b, key=lambda o: o.id)
    paired = pairing is Pairing.PAIRED
    ma = m.top_metrics(sa, spec.binning.strategy, spec.binning.n_bins, vector=False)
    mb = m.top_metrics(sb, spec.binning.strategy, spec.binning.n_bins, vector=False)
    ps_a, ps_b = [_per_sample(o) for o in sa], [_per_sample(o) for o in sb]
    for name in PER_SAMPLE:
        ref: Any = {str(o.id): v[name] for o, v in zip(sa, ps_a, strict=True)} if paired else [v[name] for v in ps_a]  # fmt: skip
        trt: Any = {str(o.id): v[name] for o, v in zip(sb, ps_b, strict=True)} if paired else [v[name] for v in ps_b]  # fmt: skip
        c = st.compare(ref, trt, pairing=pairing, confidence=cfg.confidence, resamples=cfg.resamples, permutations=cfg.permutations, seed=cfg.seed)  # fmt: skip
        eff = next((e for e in c.effects if e.status is StatStatus.DERIVED), None)
        out["metrics"][name] = {
            "a": ma[name], "b": mb[name], "difference": mb[name] - ma[name],  # type: ignore[operator]
            "interval": to_jsonable(c.interval), "p_value": c.test.p_value, "test": to_jsonable(c.test),
            "effect_size": None if eff is None else {"name": eff.name, "value": eff.value},
            "source": "experionyx.stats.core.compare", "status": c.status.value,
        }  # fmt: skip
    nm = m.compare_metrics(sa, sb, paired=paired, strategy=spec.binning.strategy, n_bins=spec.binning.n_bins, names=NON_MEAN, resamples=cfg.resamples, permutations=cfg.permutations, confidence=cfg.confidence, seed=cfg.seed)  # fmt: skip
    for name, r in nm.items():
        out["metrics"][name] = {"a": ma[name], "b": mb[name], **r, "effect_size": {"name": "difference in metric units", "value": r["difference"]}, "source": "experionyx.calibration.measures.compare_metrics", "status": "DERIVED"}  # fmt: skip
    for name, r in out["metrics"].items():
        r["practically_material"] = abs(r["difference"]) >= cfg.practical_delta if name in PRACTICAL else None  # fmt: skip
        r["adjusted_p_value"] = None
        r["significant_after_correction"] = None
    conf_a, conf_b = [o.conf for o in sa], [o.conf for o in sb]
    ka, kb = _class_counts(sa), _class_counts(sb)
    out["distribution"] = {
        "confidence_ks": dm.ks_statistic(conf_a, conf_b),
        "confidence_wasserstein_1": dm.wasserstein_1(conf_a, conf_b),
        "predicted_class_total_variation": dm.total_variation(ka, kb),
        "predicted_class_counts": {"a": ka, "b": kb},
        "accuracy_change": mb["accuracy"] - ma["accuracy"],  # type: ignore[operator]
        "note": "descriptive changes in the confidence and predicted-class distributions; not causal",
    }
    return out


def _class_counts(obs: Sequence[m.Obs]) -> list[int]:
    width = len(obs[0].probs)
    counts = [0] * width
    for o in obs:
        counts[o.pred] += 1
    return counts


def correct_families(comps: Mapping[str, dict[str, Any]], spec: CalibrationSpec) -> dict[str, Any]:
    """Multiple-comparison correction, separately for each (family, metric): the family is exactly
    the comparisons of that kind in this analysis. Raw p-values are kept beside adjusted ones."""
    cfg = spec.statistics
    groups: dict[str, dict[str, float | None]] = {}
    for name, c in comps.items():
        if c["status"] != COMPUTED:
            continue
        for metric, r in c["metrics"].items():
            groups.setdefault(f"{c['family']}/{metric}", {})[name] = r["p_value"]
    report: dict[str, Any] = {}
    for gk, pv in sorted(groups.items()):
        if not any(p is not None for p in pv.values()):
            continue
        cor = st.adjust_pvalues(pv, method=cfg.correction, alpha=cfg.alpha)
        metric = gk.split("/", 1)[1]
        for name, adj in cor.adjusted.items():
            r = comps[name]["metrics"][metric]
            r["adjusted_p_value"], r["significant_after_correction"] = adj, cor.rejected[name]
        report[gk] = {"method": cor.method, "alpha": cor.alpha, "n_hypotheses": cor.n_hypotheses, "controls": cor.controls, "note": cor.note}  # fmt: skip
    return report


def candidates(obs: Sequence[m.Obs], cfg: CalibrationSpec) -> dict[str, Any]:
    """Sample-level SIGNALS for failure analysis: confident errors, unconfident successes and the
    relationship between confidence and correctness. They are evidence to look at, never failure modes."""
    s = cfg.statistics
    high = [{"sample_id": o.id, "confidence": o.conf, "true": o.true, "predicted": o.pred} for o in sorted(obs, key=lambda o: o.id) if not o.correct and o.conf >= s.high_confidence]  # fmt: skip
    low = [{"sample_id": o.id, "confidence": o.conf, "true": o.true, "predicted": o.pred} for o in sorted(obs, key=lambda o: o.id) if o.correct and o.conf < s.low_confidence]  # fmt: skip
    right = [o.conf for o in obs if o.correct]
    wrong = [o.conf for o in obs if not o.correct]
    return {
        "status": "CANDIDATE_SIGNALS_ONLY",
        "note": "candidate evidence for failure discovery: not classified as failure modes and not evidence of cause",
        "thresholds": {"high_confidence": s.high_confidence, "low_confidence": s.low_confidence},
        "high_confidence_incorrect": high,
        "low_confidence_correct": low,
        "confidence_error_relationship": {
            "n_correct": len(right),
            "n_incorrect": len(wrong),
            "mean_confidence_correct": _mean(right),
            "mean_confidence_incorrect": _mean(wrong),
            "auroc_confidence_separates_correct": m.auroc(right, wrong),
            "auroc_meaning": "probability that a random correct prediction has higher confidence than a random incorrect one (0.5 = no relationship); None if either group is empty",
        },
    }

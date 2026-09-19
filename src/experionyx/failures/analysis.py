"""Pure discovery analysis: interpretable similarity (with reasons), bounded blocked comparison,
deterministic clustering, stability diagnostics, multidimensional cluster measurements and
configurable evidence evaluation. No I/O, no randomness except a seeded bootstrap."""

import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations

from experionyx.evaluation.bootstrap import quantile
from experionyx.failures.config import (
    ClusteringAlgorithm,
    DiscoveryConfig,
    EvidenceCriteria,
    SimilarityConfig,
)
from experionyx.failures.entities import FailureSignal
from experionyx.failures.taxonomy import Direction, FailureCategory

# -- similarity ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Reason:
    dimension: str
    weight: float
    score: float
    note: str


@dataclass(frozen=True)
class Similarity:
    score: float
    reasons: tuple[Reason, ...]
    blocked: str | None = None  # a hard constraint that forced the score to 0


def _fault(s: FailureSignal) -> Mapping[str, object] | None:
    f = s.detail.get("fault")
    return f if isinstance(f, Mapping) else None


def _numeric_params(f: Mapping[str, object]) -> dict[str, float]:
    p = f.get("parameters")
    out = (
        {
            k: float(v)
            for k, v in p.items()
            if isinstance(v, int | float) and not isinstance(v, bool)
        }
        if isinstance(p, Mapping)
        else {}
    )
    return out


def similarity(a: FailureSignal, b: FailureSignal, cfg: SimilarityConfig) -> Similarity:
    """Weighted mean over the dimensions that apply to the pair. Class and slice are HARD
    constraints: signals about different classes (compared by exact label) or different slices
    are never similar, however alike everything else is."""
    w = cfg.weights
    rs: list[Reason] = []

    def add(dim: str, weight: float, score: float, note: str) -> None:
        rs.append(Reason(dim, weight, round(score, 12), note))

    da, db = a.detail, b.detail
    for dim, keys, weight in (
        ("class_label", ("class_label", "predicted_label"), w.class_label),
        ("slice", ("slice",), w.slice),
    ):
        va, vb = tuple(da.get(k) for k in keys), tuple(db.get(k) for k in keys)
        if va == (None,) * len(keys) and vb == va:
            continue
        if va != vb:
            add(dim, weight, 0.0, f"differs: {va} vs {vb}")
            return Similarity(0.0, tuple(rs), blocked=f"different {dim}")
        add(dim, weight, 1.0, f"same: {va}")
    add(
        "error_type",
        w.error_type,
        1.0 if a.signal_kind == b.signal_kind else 0.5 if a.category == b.category else 0.0,
        f"{a.signal_kind.value} vs {b.signal_kind.value}; category {a.category.value} vs {b.category.value}",
    )
    fa, fb = _fault(a), _fault(b)
    if fa or fb:
        same_type = bool(fa and fb and fa["type"] == fb["type"])
        add(
            "fault_type",
            w.fault_type,
            1.0 if same_type else 0.0,
            f"{fa['type'] if fa else 'none'} vs {fb['type'] if fb else 'none'}",
        )
        if fa and fb and same_type:
            pa, pb = _numeric_params(fa), _numeric_params(fb)
            shared = sorted(set(pa) & set(pb))
            if shared:
                rel = max(abs(pa[k] - pb[k]) / max(abs(pa[k]), abs(pb[k]), 1e-12) for k in shared)
                add(
                    "parameter_proximity",
                    w.parameter_proximity,
                    max(0.0, 1.0 - rel / cfg.parameter_relative_tolerance)
                    if cfg.parameter_relative_tolerance
                    else float(rel == 0),
                    f"max relative difference {rel:.3g} over {shared}",
                )
            xa, xb = fa.get("affected_fraction"), fb.get("affected_fraction")
            if isinstance(xa, float) and isinstance(xb, float):
                gap = abs(xa - xb)
                add(
                    "affected_fraction",
                    w.affected_fraction,
                    max(0.0, 1.0 - gap / cfg.fraction_tolerance)
                    if cfg.fraction_tolerance
                    else float(gap == 0),
                    f"affected fraction {xa:.3g} vs {xb:.3g}",
                )
    top = max(a.magnitude, b.magnitude)
    add(
        "severity",
        w.severity,
        1.0 if top == 0 else 1.0 - abs(a.magnitude - b.magnitude) / top,
        f"magnitude {a.magnitude:.3g} vs {b.magnitude:.3g}",
    )
    for dim, key, weight in (
        ("model", "model_fingerprint", w.model),
        ("dataset", "dataset_fingerprint", w.dataset),
    ):
        if da.get(key) is not None and db.get(key) is not None:
            add(
                dim,
                weight,
                float(da[key] == db[key]),
                f"{key} {'equal' if da[key] == db[key] else 'differs'}",
            )
    total = sum(r.weight for r in rs)
    if total <= 0:
        return Similarity(0.0, tuple(rs))
    return Similarity(round(sum(r.weight * r.score for r in rs) / total, 12), tuple(rs))


# -- bounded pairwise comparison -------------------------------------------------------------


@dataclass(frozen=True)
class Links:
    pairs: tuple[tuple[int, int, float], ...]  # (i, j, score) with score >= the lowest threshold
    comparisons: int
    fallback_blocks: tuple[str, ...]  # blocks compared by exact signature only (budget)
    lowest_threshold: float


def _block_key(s: FailureSignal) -> tuple[object, ...]:
    d = s.detail
    return (d.get("class_label"), d.get("predicted_label"), d.get("slice"))


def compare_all(signals: Sequence[FailureSignal], cfg: SimilarityConfig, lowest: float) -> Links:
    """Compare only signals that can be similar (same class/prediction/slice block), within a
    budget of `max_pairwise_comparisons`. A block that would exceed the budget is not compared
    pairwise: its signals are linked by IDENTICAL SIGNATURE only, and the block is recorded."""
    blocks: dict[tuple[object, ...], list[int]] = {}
    for i, s in enumerate(signals):
        blocks.setdefault(_block_key(s), []).append(i)
    pairs: list[tuple[int, int, float]] = []
    used, fallback = 0, []
    for key in sorted(blocks, key=repr):
        idx = blocks[key]
        need = len(idx) * (len(idx) - 1) // 2
        if used + need > cfg.max_pairwise_comparisons:
            fallback.append(repr(key))
            for i, j in combinations(idx, 2):
                if signals[i].signature == signals[j].signature:
                    pairs.append((i, j, 1.0))
            continue
        used += need
        for i, j in combinations(idx, 2):
            sc = similarity(signals[i], signals[j], cfg).score
            if sc >= lowest:
                pairs.append((i, j, sc))
    return Links(tuple(sorted(pairs)), used, tuple(fallback), lowest)


# -- clustering ------------------------------------------------------------------------------


def components(
    n: int, pairs: Sequence[tuple[int, int, float]], threshold: float
) -> list[list[int]]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, sc in pairs:
        if sc >= threshold:
            parent[max(find(i), find(j))] = min(find(i), find(j))
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values())


def cluster_indices(
    signals: Sequence[FailureSignal],
    links: Links,
    cfg: DiscoveryConfig,
    threshold: float | None = None,
) -> list[list[int]]:
    if cfg.clustering.algorithm is ClusteringAlgorithm.SIGNATURE_GROUPING:
        by: dict[str, list[int]] = {}
        for i, s in enumerate(signals):
            by.setdefault(s.signature, []).append(i)
        return sorted(by.values())
    t = cfg.similarity.threshold if threshold is None else threshold
    return components(len(signals), links.pairs, t)


def jaccard(a: set[int], b: set[int]) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def stability(
    base: list[list[int]], links: Links, n: int, cfg: DiscoveryConfig
) -> list[float | None]:
    """Per cluster: the worst best-Jaccard overlap with the clusters obtained when the similarity
    threshold moves by +/- sensitivity_delta. None when the algorithm has no threshold."""
    if cfg.clustering.algorithm is ClusteringAlgorithm.SIGNATURE_GROUPING:
        return [None] * len(base)
    t, d = cfg.similarity.threshold, cfg.clustering.sensitivity_delta
    alt = [[set(c) for c in components(n, links.pairs, min(1.0, max(0.0, t + s)))] for s in (-d, d)]
    return [
        round(min(max(jaccard(set(c), o) for o in variant) for variant in alt), 12) for c in base
    ]


def compactness(
    members: Sequence[FailureSignal], cfg: SimilarityConfig, cap: int = 100
) -> tuple[float | None, int]:
    """Mean pairwise similarity of (at most `cap`, lowest-ID) members; the size used is returned."""
    ms = sorted(members, key=lambda s: s.id)[:cap]
    if len(ms) < 2:
        return None, len(ms)
    vals = [similarity(a, b, cfg).score for a, b in combinations(ms, 2)]
    return round(sum(vals) / len(vals), 12), len(ms)


# -- measurements ----------------------------------------------------------------------------


def _uniq(items: Sequence[object]) -> list[object]:
    return sorted({i for i in items if i is not None}, key=repr)


def cluster_measurements(members: Sequence[FailureSignal], total_runs: int) -> dict[str, object]:
    """Prevalence, impact and severity are SEPARATE, each with its denominator. Severity is a
    vector, never a single score. Co-occurrence is not computed here (see relationships)."""
    runs = {s.run_id for s in members}
    exps = {s.experiment_id for s in members}
    faults = [f for s in members if (f := _fault(s))]
    mags = [s.magnitude for s in members]
    fracs = [
        s.sample_count / n
        for s in members
        if s.detail.get("sample_ids_recorded")
        and (n := s.detail.get("n_samples"))
        and isinstance(n, int)
    ]
    dirs = Counter(s.direction.value for s in members)
    majority = max(sorted(dirs), key=lambda k: dirs[k])
    cls = _uniq([s.detail.get("class_label") for s in members])
    return {
        "size": len(members),
        "prevalence": {
            "runs_with_signal": len(runs),
            "runs_analyzed": total_runs,
            "fraction_of_runs": len(runs) / total_runs if total_runs else None,
            "denominator": "runs analyzed by this discovery",
        },
        "impact": {
            "mean_magnitude": sum(mags) / len(mags),
            "max_magnitude": max(mags),
            "min_magnitude": min(mags),
            "unit": "signal-specific (see each signal's `detail.metric`); compared only within a kind",
        },
        "severity": {
            "magnitude_mean": sum(mags) / len(mags),
            "magnitude_max": max(mags),
            "affected_sample_fraction_mean": (sum(fracs) / len(fracs)) if fracs else None,
            "affected_sample_fraction_denominator": "per-signal n_samples; None when sample IDs were not recorded",
            "run_breadth": len(runs),
            "class_breadth": len(cls),
            "note": "multidimensional; no single severity score is computed",
        },
        "direction_counts": dict(sorted(dirs.items())),
        "direction_consistency": dirs[majority] / len(members),
        "majority_direction": majority,
        "runs": sorted(runs),
        "experiments": sorted(exps),
        "seeds": sorted({sd for f in faults if isinstance(sd := f.get("seed"), int)}),
        "fault_types": _uniq([f.get("type") for f in faults]),
        "fault_families": _uniq([f.get("family_id") for f in faults]),
        "fault_targets": _uniq([f.get("target") for f in faults]),
        "classes": cls,
        "predicted_classes": _uniq([s.detail.get("predicted_label") for s in members]),
        "slices": _uniq([s.detail.get("slice") for s in members]),
        "models": _uniq([s.detail.get("model_fingerprint") for s in members]),
        "datasets": _uniq([s.detail.get("dataset_fingerprint") for s in members]),
        "kinds": sorted({s.signal_kind.value for s in members}),
        "sources": _uniq([s.detail.get("source") for s in members]),
    }


def sample_groups(members: Sequence[FailureSignal], top: int = 50) -> dict[str, object]:
    """Per-sample grouping that preserves sample IDs: how many signals implicate each sample of
    each dataset (keyed by dataset fingerprint, so IDs of different datasets never collide)."""
    counts: Counter[tuple[str, str]] = Counter()
    for s in members:
        fp = str(s.detail.get("dataset_fingerprint"))
        counts.update((fp, i) for i in s.sample_ids)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "distinct_samples": len(counts),
        "truncated_signals": sum(s.truncated for s in members),
        "top": [
            {"dataset_fingerprint": fp, "sample_id": sid, "signals": n}
            for (fp, sid), n in ranked[:top]
        ],
        "note": "sample_id is the index into the evaluated split; only recorded (bounded) IDs are counted",
    }


def category_of(members: Sequence[FailureSignal]) -> FailureCategory:
    c = Counter(s.category for s in members)
    top = max(c.values())
    winners = sorted(k for k, v in c.items() if v == top)
    return winners[0] if len(winners) == 1 else FailureCategory.UNKNOWN


# -- evidence evaluation -----------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    name: str
    required: object
    observed: object
    passed: bool


@dataclass(frozen=True)
class Evaluation:
    criteria_name: str
    checks: tuple[Check, ...]
    passed: bool
    interval: Mapping[str, object] = field(default_factory=dict)


def bootstrap_mean_interval(
    values: Sequence[float], resamples: int, confidence: float, seed: int
) -> tuple[float, float]:
    rng = random.Random(seed)  # noqa: S311  # seeded, recorded, not security-relevant
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    alpha = (1.0 - confidence) / 2.0
    return quantile(means, alpha), quantile(means, 1.0 - alpha)


def evaluate_criteria(
    name: str,
    c: EvidenceCriteria,
    members: Sequence[FailureSignal],
    cluster_metrics: Mapping[str, object],
    resamples: int,
    confidence: float,
    seed: int,
) -> Evaluation:
    """Check a cluster against configured requirements. Every check reports required vs observed."""
    mags = [s.magnitude for s in members]
    mean = sum(mags) / len(mags)
    runs = {s.run_id for s in members}
    exps = {s.experiment_id for s in members}
    seeds = {sd for s in members if (f := _fault(s)) and isinstance(sd := f.get("seed"), int)}
    counts = Counter(s.direction for s in members)
    consistency = max(counts.values()) / len(members)
    checks = [
        Check("min_signals", c.min_signals, len(members), len(members) >= c.min_signals),
        Check("min_runs", c.min_runs, len(runs), len(runs) >= c.min_runs),
        Check("min_experiments", c.min_experiments, len(exps), len(exps) >= c.min_experiments),
        Check("min_mean_effect", c.min_mean_effect, mean, mean >= c.min_mean_effect),
        Check(
            "direction_consistency",
            c.direction_consistency_min,
            consistency,
            consistency >= c.direction_consistency_min,
        ),
    ]
    if seeds:  # only fault-derived signals have seeds
        checks.append(Check("min_seeds", c.min_seeds, len(seeds), len(seeds) >= c.min_seeds))
    comp = cluster_metrics.get("compactness")
    checks.append(
        Check(
            "min_compactness",
            c.min_compactness,
            comp,
            (comp is None and len(members) == 1)
            or (isinstance(comp, float) and comp >= c.min_compactness),
        )
    )
    stab = cluster_metrics.get("stability")
    checks.append(
        Check(
            "min_threshold_stability",
            c.min_threshold_stability,
            stab,
            stab is None or (isinstance(stab, float) and stab >= c.min_threshold_stability),
        )
    )
    interval: dict[str, object] = {}
    if c.require_interval_excludes_null:
        if len(mags) < 2:
            checks.append(
                Check(
                    "interval_excludes_null",
                    f"lower bound > {c.null_region}",
                    "n < 2: no interval",
                    False,
                )
            )
        else:
            lo, hi = bootstrap_mean_interval(mags, resamples, confidence, seed)
            interval = {
                "method": "bootstrap-percentile of the mean magnitude",
                "lower": lo,
                "upper": hi,
                "confidence": confidence,
                "resamples": resamples,
                "seed": seed,
                "n": len(mags),
            }
            checks.append(
                Check(
                    "interval_excludes_null",
                    f"lower bound > {c.null_region}",
                    lo,
                    math.isfinite(lo) and lo > c.null_region,
                )
            )
    return Evaluation(name, tuple(checks), all(k.passed for k in checks), interval)


def direction_of_majority(m: Mapping[str, object]) -> Direction:
    return Direction(str(m["majority_direction"]))

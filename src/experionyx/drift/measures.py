"""Distribution-shift measures between two independent samples (a reference and a comparison).
Pure and deterministic; the only randomness is seeded permutation resampling.

Methods, in the terms the results use:

NUMERIC  ks_statistic        largest gap between the two empirical CDFs, in [0, 1]. Measures the size of
                             the difference anywhere in the distribution, not evidence about it.
         wasserstein_1       mean distance the reference mass must move to match the comparison, in the
                             feature's own units (area between the empirical CDFs). `_scaled` divides it
                             by the reference standard deviation; undefined when that is zero.
         location            mean difference, effect sizes and a bootstrap interval from the Phase 10
                             statistics core (assumes independent samples; says nothing about shape).
         test                a permutation test of the KS statistic. Null: both samples come from one
                             distribution (labels exchangeable). Exact when few rearrangements exist,
                             otherwise a seeded Monte Carlo estimate with (hits+1)/(n+1).
CATEGORICAL / BOOLEAN
         proportions         per category, per window, with Wilson intervals (Phase 10) and their
                             difference (comparison - reference).
         jensen_shannon_divergence  symmetric, base 2, in [0, 1]; 0 means identical proportions.
         total_variation_distance   half the sum of absolute proportion differences, in [0, 1].
         test                a permutation test of the Jensen-Shannon divergence, same null.

Assumptions and limits, always: samples within a window are treated as independent draws; temporal
autocorrelation (typical of real time series) makes the p-values optimistic. Distances are
nonnegative and biased upward in small samples (two samples of the same distribution give a positive
distance), so they are not compared with zero. No bootstrap interval is reported for KS or
Wasserstein for that reason. Nothing here decides whether a shift is 'large' or 'important'."""

import bisect
import itertools
import math
import random
import statistics
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from experionyx.domain import to_jsonable
from experionyx.drift.results import Counts, DistributionShiftResult, Evidence
from experionyx.drift.spec import DriftConfig
from experionyx.failures.extraction import label_key
from experionyx.interactions.taxonomy import Pairing
from experionyx.stats import core as st

MAX_CATEGORIES = 200
_TOL = 1e-12
_PERM_ASSUMPTIONS = (
    "samples within a window are independent draws",
    "under H0 the window labels are exchangeable (one shared distribution)",
    "temporal autocorrelation, if present, makes the p-value optimistic",
)


# -- cleaning: every value is accounted for -------------------------------------------------------


def _isnan(x: object) -> bool:
    return isinstance(x, float) and math.isnan(x)


def clean(values: Iterable[object], kind: str) -> tuple[list[Any], Counts]:
    """(valid values, counts). NUMERIC keeps finite ints/floats as floats; CATEGORICAL keeps a
    canonical category key; BOOLEAN keeps 'true'/'false'. Nothing is coerced across types: a string
    in a numeric feature is INVALID, a bool is not a number, and 1 and '1' are different categories."""
    valid: list[Any] = []
    total = missing = nonfinite = invalid = 0
    for x in values:
        total += 1
        if x is None or _isnan(x):
            missing += 1
        elif isinstance(x, float) and math.isinf(x):
            nonfinite += 1
        elif kind == "NUMERIC":
            if isinstance(x, bool) or not isinstance(x, int | float):
                invalid += 1
            else:
                valid.append(float(x))
        elif kind == "BOOLEAN":
            if isinstance(x, bool):
                valid.append("true" if x else "false")
            elif isinstance(x, int | float) and x in (0, 1):
                valid.append("true" if x == 1 else "false")
            else:
                invalid += 1
        elif isinstance(x, bool | str | int | float):
            valid.append(label_key(int(x) if isinstance(x, float) and x.is_integer() else x))
        else:
            invalid += 1
    return valid, Counts(total, len(valid), missing, nonfinite, invalid)


# -- statistics -----------------------------------------------------------------------------------


def ks_statistic(x: Sequence[float], y: Sequence[float]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic; ties are handled by evaluating both ECDFs at every
    distinct value."""
    xs, ys = sorted(x), sorted(y)
    nx, ny = len(xs), len(ys)
    return max(
        abs(bisect.bisect_right(xs, v) / nx - bisect.bisect_right(ys, v) / ny)
        for v in sorted(set(xs) | set(ys))
    )


def wasserstein_1(x: Sequence[float], y: Sequence[float]) -> float:
    """First Wasserstein distance = the integral of |F_x - F_y|, exact for empirical CDFs."""
    xs, ys = sorted(x), sorted(y)
    nx, ny = len(xs), len(ys)
    pts = sorted(set(xs) | set(ys))
    return math.fsum(
        abs(bisect.bisect_right(xs, a) / nx - bisect.bisect_right(ys, a) / ny) * (b - a)
        for a, b in itertools.pairwise(pts)
    )


def _dist(counts: Sequence[int]) -> list[float]:
    n = sum(counts)
    return [c / n for c in counts]


def jensen_shannon(cx: Sequence[int], cy: Sequence[int]) -> float:
    """Jensen-Shannon divergence in bits (base 2), in [0, 1]."""
    p, q = _dist(cx), _dist(cy)
    m = [(a + b) / 2 for a, b in zip(p, q, strict=True)]

    def kl(a: Sequence[float], b: Sequence[float]) -> float:
        return math.fsum(u * math.log2(u / w) for u, w in zip(a, b, strict=True) if u > 0)

    return min(1.0, max(0.0, 0.5 * kl(p, m) + 0.5 * kl(q, m)))


def total_variation(cx: Sequence[int], cy: Sequence[int]) -> float:
    return 0.5 * math.fsum(abs(a - b) for a, b in zip(_dist(cx), _dist(cy), strict=True))


def permutation_test(
    statistic: Callable[[list[Any], list[Any]], float],
    x: Sequence[Any],
    y: Sequence[Any],
    *,
    name: str,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    """Right-tailed permutation test of `statistic(x, y)` (larger = more different). The p-value is
    the share of rearrangements of the pooled sample with a statistic at least as large as observed
    (with a small tolerance for float ties)."""
    nx, ny = len(x), len(y)
    pool = list(x) + list(y)
    obs = statistic(list(x), list(y))
    if math.comb(nx + ny, ny) <= st.EXACT_LIMIT:
        hits = count = 0
        for idx in itertools.combinations(range(nx + ny), ny):
            chosen = set(idx)
            a = [pool[i] for i in range(nx + ny) if i not in chosen]
            b = [pool[i] for i in idx]
            count += 1
            hits += statistic(a, b) >= obs - _TOL
        p, exact, used = hits / count, True, count
    else:
        rng = random.Random(seed)  # noqa: S311
        buf = list(pool)
        hits = 0
        for _ in range(permutations):
            rng.shuffle(buf)
            hits += statistic(buf[:nx], buf[nx:]) >= obs - _TOL
        p, exact, used = (hits + 1) / (permutations + 1), False, permutations
    return {
        "name": name, "status": Evidence.DERIVED.value, "statistic": obs, "p_value": p,
        "exact": exact, "permutations": used, "seed": None if exact else seed,
        "assumptions": list(_PERM_ASSUMPTIONS),
        "note": "a p-value is the probability, under the stated null, of a statistic at least this extreme; it is not a verdict, does not measure the size of the shift, and says nothing about cause",
    }  # fmt: skip


# -- numeric --------------------------------------------------------------------------------------


def _numeric(
    x: list[float], y: list[float], cfg: DriftConfig
) -> tuple[dict[str, Any], dict[str, Any] | None, list[str]]:
    warnings: list[str] = []
    for label, s in (("reference", x), ("comparison", y)):
        if len(set(s)) == 1:
            warnings.append(f"the {label} window is constant (zero variance)")
    sd = statistics.stdev(x) if len(x) > 1 else 0.0
    w1 = wasserstein_1(x, y)
    measures: dict[str, Any] = {
        "ks_statistic": ks_statistic(x, y),
        "wasserstein_1": w1,
        "wasserstein_1_scaled": w1 / sd if sd > 0 else None,
        "wasserstein_1_scaled_reason": None if sd > 0 else "the reference standard deviation is zero or undefined",
        "mean_reference": math.fsum(x) / len(x), "mean_comparison": math.fsum(y) / len(y),
        "direction": "difference is comparison minus reference",
        "distance_note": "distances are nonnegative and biased upward in small samples; compare with the permutation test, not with zero",
    }  # fmt: skip
    if min(len(x), len(y)) < cfg.min_samples:
        return measures, None, warnings
    measures["location"] = {
        "effect_sizes": [to_jsonable(e) for e in st.effect_sizes(x, y, Pairing.UNPAIRED)],
        "mean_difference_interval": to_jsonable(
            st.bootstrap_interval(
                x,
                y,
                pairing=Pairing.UNPAIRED,
                estimator="mean",
                method=cfg.method,
                confidence=cfg.confidence,
                resamples=cfg.resamples,
                seed=cfg.seed,
            )
        ),
        "note": "location only (mean); effect sizes and the interval assume independent samples and say nothing about shape",
    }
    test = permutation_test(ks_statistic, x, y, name="ks_permutation", permutations=cfg.permutations, seed=cfg.seed)  # fmt: skip
    return measures, test, warnings


# -- categorical ----------------------------------------------------------------------------------


def _categorical(
    x: list[str], y: list[str], cfg: DriftConfig
) -> tuple[dict[str, Any], dict[str, Any] | None, list[str]]:
    cats = sorted(set(x) | set(y))
    cx, cy = Counter(x), Counter(y)
    vx, vy = [cx[c] for c in cats], [cy[c] for c in cats]
    rows = []
    for c, a, b in zip(cats, vx, vy, strict=True):
        ia, ib = st.proportion_interval(a, len(x), cfg.confidence), st.proportion_interval(b, len(y), cfg.confidence)  # fmt: skip
        rows.append({
            "category": c, "n_reference": a, "n_comparison": b, "p_reference": a / len(x), "p_comparison": b / len(y),
            "difference": b / len(y) - a / len(x), "interval_reference": to_jsonable(ia), "interval_comparison": to_jsonable(ib),
        })  # fmt: skip
    measures: dict[str, Any] = {
        "categories": rows,
        "only_in_reference": [c for c in cats if cx[c] and not cy[c]],
        "only_in_comparison": [c for c in cats if cy[c] and not cx[c]],
        "jensen_shannon_divergence": jensen_shannon(vx, vy),
        "total_variation_distance": total_variation(vx, vy),
        "direction": "difference is comparison minus reference",
        "divergence_note": "base-2 Jensen-Shannon divergence in [0, 1]; biased upward in small samples",
        "proportion_note": "Wilson intervals assume independent, identically distributed draws within each window",
    }  # fmt: skip
    warnings = []
    if measures["only_in_comparison"]:
        warnings.append(
            f"categories unseen in the reference window: {measures['only_in_comparison']}"
        )
    if measures["only_in_reference"]:
        warnings.append(
            f"categories absent from the comparison window: {measures['only_in_reference']}"
        )
    if min(len(x), len(y)) < cfg.min_samples:
        return measures, None, warnings
    code = {c: i for i, c in enumerate(cats)}

    def stat(a: list[int], b: list[int]) -> float:
        ca, cb = Counter(a), Counter(b)
        return jensen_shannon([ca[i] for i in range(len(cats))], [cb[i] for i in range(len(cats))])

    test = permutation_test(stat, [code[c] for c in x], [code[c] for c in y], name="jensen_shannon_permutation", permutations=cfg.permutations, seed=cfg.seed)  # fmt: skip
    return measures, test, warnings


# -- entry ----------------------------------------------------------------------------------------


def shift(
    dimension: str,
    subject: str,
    kind: str,
    reference: Iterable[object],
    comparison: Iterable[object],
    cfg: DriftConfig,
) -> DistributionShiftResult:
    """Compare one value's distribution across two windows. Never raises for data problems: a
    problem becomes an UNDEFINED / INSUFFICIENT_EVIDENCE result with its reason."""
    x, cx = clean(reference, kind)
    y, cy = clean(comparison, kind)
    numeric = kind == "NUMERIC"
    method = "ks + wasserstein_1 + location" if numeric else "jensen_shannon + total_variation + proportions"  # fmt: skip
    head = (dimension, subject, kind)

    def out(status: Evidence, reason: str | None = None, **kw: Any) -> DistributionShiftResult:
        return DistributionShiftResult(*head, status, cx, cy, method, reason=reason, **kw)

    if not x or not y:
        side = "reference" if not x else "comparison"
        return out(Evidence.UNDEFINED, f"the {side} window has no valid values ({cx if not x else cy}); a distribution does not exist")  # fmt: skip
    if not numeric and len(set(x) | set(y)) > MAX_CATEGORIES:
        return out(Evidence.UNDEFINED, f"more than {MAX_CATEGORIES} distinct categories; a feature this granular is not a categorical distribution")  # fmt: skip
    measures, test, warnings = _numeric(x, y, cfg) if numeric else _categorical(x, y, cfg)
    for label, c in (("reference", cx), ("comparison", cy)):
        dropped = c.n_missing + c.n_nonfinite + c.n_invalid
        if dropped:
            warnings.append(f"{label} window: {dropped} of {c.n_total} value(s) excluded (missing {c.n_missing}, non-finite {c.n_nonfinite}, invalid {c.n_invalid})")  # fmt: skip
    if test is None:
        return out(
            Evidence.INSUFFICIENT_EVIDENCE,
            f"fewer than min_samples={cfg.min_samples} valid values in a window ({len(x)} reference, {len(y)} comparison): measures are reported, inference is withheld",
            measures=measures, warnings=tuple(warnings),
        )  # fmt: skip
    return out(Evidence.DERIVED, measures=measures, test=test, warnings=tuple(warnings))

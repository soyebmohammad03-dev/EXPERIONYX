"""Pure, deterministic statistics. No I/O, no hidden state, no dependencies beyond the stdlib.

Rules: an invalid statistic is never turned into a number (it is UNDEFINED with a reason); pairing is
declared and keyed, never inferred from position; every random procedure takes an explicit seed
and records its configuration; a p-value is a number under stated assumptions, never a claim about
practical importance or causation. Sorting inputs first makes results independent of input order."""

import itertools
import math
import random
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from statistics import NormalDist

from experionyx.errors import ValidationError
from experionyx.evaluation.bootstrap import quantile
from experionyx.interactions.taxonomy import Pairing

STATS_VERSION = "1"
SMALL_SAMPLE = 30
COARSE_RESAMPLES = 1000
EXACT_LIMIT = 50_000  # enumerate all rearrangements up to this many; otherwise seeded Monte Carlo
_ND = NormalDist()


class Status(StrEnum):
    OBSERVED = "OBSERVED"  # a recorded measurement
    DERIVED = "DERIVED"  # computed from observations by a stated formula
    INCONCLUSIVE = "INCONCLUSIVE"  # too little data for the requested statistic
    UNDEFINED = "UNDEFINED"  # the statistic does not exist for this input (reason given)


@dataclass(frozen=True)
class Quantity:
    value: float | None
    status: Status
    reason: str | None = None


def _d(value: float) -> Quantity:
    return Quantity(value, Status.DERIVED)


def _u(reason: str, status: Status = Status.UNDEFINED) -> Quantity:
    return Quantity(None, status, reason)


Sample = Sequence[float] | Mapping[str, float]


def _problem(values: Sequence[object]) -> str | None:
    bad = [v for v in values if isinstance(v, bool) or not isinstance(v, int | float)]
    if bad:
        return f"{len(bad)} non-numeric value(s)"
    n_bad = sum(not math.isfinite(v) for v in values)  # type: ignore[arg-type]
    return (
        f"{n_bad} non-finite value(s); observations are never silently dropped" if n_bad else None
    )


def _values(sample: Sample) -> list[float]:
    """Order-independent list of values (sorted), or ValidationError for invalid data."""
    raw = list(sample.values()) if isinstance(sample, Mapping) else list(sample)
    if (p := _problem(raw)) is not None:
        raise ValidationError(p)
    return sorted(float(v) for v in raw)


# -- descriptive summary -------------------------------------------------------------------------


@dataclass(frozen=True)
class Summary:
    n: int
    status: Status
    mean: Quantity
    median: Quantity
    variance: Quantity  # sample variance, n - 1 in the denominator
    std: Quantity
    standard_error: Quantity
    minimum: Quantity
    maximum: Quantity
    reason: str | None = None


def _variance(v: Sequence[float]) -> float:
    m = math.fsum(v) / len(v)
    return math.fsum((x - m) ** 2 for x in v) / (len(v) - 1)


def summarize(values: Sequence[float]) -> Summary:
    n = len(values)
    if n == 0 or (p := _problem(values)) is not None:
        why = "no observations" if n == 0 else str(p)
        u = _u(why)
        return Summary(n, Status.UNDEFINED, u, u, u, u, u, u, u, why)
    v = [float(x) for x in values]
    mean = _d(math.fsum(v) / n)
    if n < 2:
        u = _u("needs at least 2 observations", Status.INCONCLUSIVE)
        return Summary(
            n, Status.INCONCLUSIVE, mean, _d(v[0]), u, u, u, _d(v[0]), _d(v[0]),
            "one observation: location only, no spread",
        )  # fmt: skip
    var = _variance(v)
    sd = math.sqrt(var)
    return Summary(
        n, Status.DERIVED, mean, _d(statistics.median(v)), _d(var), _d(sd), _d(sd / math.sqrt(n)),
        _d(min(v)), _d(max(v)),
    )  # fmt: skip


# -- effect sizes --------------------------------------------------------------------------------


@dataclass(frozen=True)
class EffectSize:
    name: str
    status: Status
    value: float | None
    formula: str
    numerator: float | None
    denominator: float | None
    assumptions: tuple[str, ...]
    reason: str | None = None  # why it is undefined / inconclusive


def _effect(
    name: str, formula: str, num: float | None, den: float | None, assumptions: tuple[str, ...],
    *, needs_den: bool = True, why: str | None = None,
) -> EffectSize:  # fmt: skip
    if why is not None:
        st = Status.INCONCLUSIVE if why.startswith("needs") else Status.UNDEFINED
        return EffectSize(name, st, None, formula, num, den, assumptions, why)
    if needs_den and (den is None or den == 0.0):
        return EffectSize(
            name, Status.UNDEFINED, None, formula, num, den, assumptions, "zero denominator"
        )
    assert num is not None  # noqa: S101
    if not needs_den:
        return EffectSize(name, Status.DERIVED, num, formula, num, den, assumptions)
    assert den is not None  # noqa: S101
    return EffectSize(name, Status.DERIVED, num / den, formula, num, den, assumptions)


def effect_sizes(
    reference: Sample, treatment: Sample, pairing: Pairing = Pairing.UNPAIRED
) -> tuple[EffectSize, ...]:
    """Effects of `treatment` relative to `reference` (treatment minus reference). PAIRED needs two
    keyed mappings with identical keys. UNKNOWN is analyzed as UNPAIRED."""
    x, y = _values(reference), _values(treatment)
    if pairing is Pairing.PAIRED:
        d = _paired_diffs(reference, treatment)
        return _paired_effects(d, x, y)
    return _unpaired_effects(x, y)


def _paired_diffs(reference: Sample, treatment: Sample) -> list[float]:
    if not isinstance(reference, Mapping) or not isinstance(treatment, Mapping):
        raise ValidationError(
            "paired analysis needs keyed observations (mappings of key -> value): "
            "position is never used as identity"
        )
    if set(reference) != set(treatment):
        only = sorted(set(reference) ^ set(treatment))
        raise ValidationError(f"paired observations must share identical keys; unmatched: {only}")
    _values(reference), _values(treatment)  # validates finiteness
    return [float(treatment[k]) - float(reference[k]) for k in sorted(reference)]


def _paired_effects(d: list[float], x: list[float], y: list[float]) -> tuple[EffectSize, ...]:
    n = len(d)
    md = math.fsum(d) / n if n else None
    mx = math.fsum(x) / len(x) if x else None
    sd = math.sqrt(_variance(d)) if n > 1 else None
    a_p = ("observations are matched by key",)
    return (
        _effect("mean_difference", "mean(treatment_i - reference_i)", md, 1.0, a_p, needs_den=False,
                why=None if n else "needs at least 1 pair"),
        _effect("relative_change", "mean_difference / |mean(reference)|", md,
                None if mx is None else abs(mx), (*a_p, "reference mean is non-zero"),
                why=None if n else "needs at least 1 pair"),
        _effect("paired_standardized_difference_dz", "mean(d) / sd(d)", md, sd,
                (*a_p, "differences have non-zero spread", "roughly symmetric differences"),
                why=None if n > 1 else "needs at least 2 pairs"),
    )  # fmt: skip


def _unpaired_effects(x: list[float], y: list[float]) -> tuple[EffectSize, ...]:
    nx, ny = len(x), len(y)
    mx = math.fsum(x) / nx if nx else None
    my = math.fsum(y) / ny if ny else None
    diff = None if mx is None or my is None else my - mx
    a_u = ("the two groups are independent samples",)
    empty = None if nx and ny else "needs at least 1 observation in each group"
    pooled: float | None = None
    small = None
    if nx > 1 and ny > 1:
        df = nx + ny - 2
        pooled = math.sqrt(((nx - 1) * _variance(x) + (ny - 1) * _variance(y)) / df)
    else:
        small = "needs at least 2 observations in each group"
    df = nx + ny - 2
    g_num = None if diff is None else diff * (1.0 - 3.0 / (4.0 * df - 1.0)) if df > 0 else None
    return (
        _effect("mean_difference", "mean(treatment) - mean(reference)", diff, 1.0, a_u,
                needs_den=False, why=empty),
        _effect("relative_change", "mean_difference / |mean(reference)|", diff,
                None if mx is None else abs(mx), (*a_u, "reference mean is non-zero"), why=empty),
        _effect("cohens_d", "mean_difference / pooled_sd", diff, pooled,
                (*a_u, "similar spread in both groups", "pooled_sd > 0"), why=empty or small),
        _effect("hedges_g", "cohens_d * (1 - 3 / (4(n1+n2-2) - 1))", g_num, pooled,
                (*a_u, "similar spread in both groups", "pooled_sd > 0", "small-sample corrected d"),
                why=empty or small),
    )  # fmt: skip


# -- bootstrap confidence intervals --------------------------------------------------------------

ESTIMATORS: Mapping[str, Callable[[Sequence[float]], float]] = {
    "mean": statistics.fmean,
    "median": statistics.median,
}
METHODS = ("percentile", "bca")


@dataclass(frozen=True)
class ConfidenceInterval:
    status: Status
    method: str
    estimator: str
    confidence: float
    resamples: int
    seed: int
    pairing: str  # ONE_SAMPLE | PAIRED | UNPAIRED
    n_samples: tuple[int, ...]
    estimate: float | None = None
    lower: float | None = None
    upper: float | None = None
    valid_resamples: int = 0
    bias_correction_z0: float | None = None  # BCa only
    acceleration: float | None = None  # BCa only
    warnings: tuple[str, ...] = ()
    reason: str | None = None


def bca_levels(z0: float, a: float, alpha: float) -> tuple[float, float] | None:
    """Adjusted quantile levels of the BCa interval (Efron 1987):
    alpha_1 = Phi(z0 + (z0 + z_alpha) / (1 - a (z0 + z_alpha))), and alpha_2 likewise with
    z_(1-alpha). None when a denominator is not positive (the interval does not exist)."""
    out = []
    for z in (_ND.inv_cdf(alpha), _ND.inv_cdf(1.0 - alpha)):
        den = 1.0 - a * (z0 + z)
        if den <= 0.0:
            return None
        out.append(_ND.cdf(z0 + (z0 + z) / den))
    return out[0], out[1]


def jackknife_acceleration(jack: Sequence[float]) -> float | None:
    """a = sum(d^3) / (6 (sum d^2)^1.5), d = mean(jack) - jack_i; None if every d is 0."""
    m = math.fsum(jack) / len(jack)
    d = [m - t for t in jack]
    s2 = math.fsum(e * e for e in d)
    if s2 == 0.0:
        return None
    return float(math.fsum(e**3 for e in d) / (6.0 * s2**1.5))


def bootstrap_interval(
    reference: Sample,
    treatment: Sample | None = None,
    *,
    pairing: Pairing = Pairing.UNPAIRED,
    estimator: str = "mean",
    method: str = "percentile",
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> ConfidenceInterval:
    """CI for the estimator of one sample (`treatment` None), of the per-key differences
    treatment - reference (PAIRED), or of est(treatment) - est(reference) (independent groups)."""
    if estimator not in ESTIMATORS:
        raise ValidationError(f"estimator must be one of {sorted(ESTIMATORS)}")
    if method not in METHODS:
        raise ValidationError(f"method must be one of {list(METHODS)}")
    if not 0.0 < confidence < 1.0 or resamples < 1 or seed < 0:
        raise ValidationError("confidence in (0, 1), resamples >= 1, seed >= 0")
    est = ESTIMATORS[estimator]
    kind = "ONE_SAMPLE"
    if treatment is None:
        groups = [_values(reference)]
    elif pairing is Pairing.PAIRED:
        groups, kind = [sorted(_paired_diffs(reference, treatment))], "PAIRED"
    else:
        groups, kind = [_values(reference), _values(treatment)], "UNPAIRED"
    ns = tuple(len(g) for g in groups)
    base = {"method": method, "estimator": estimator, "confidence": confidence,
            "resamples": resamples, "seed": seed, "pairing": kind, "n_samples": ns}  # fmt: skip
    if min(ns) < 2:
        return ConfidenceInterval(
            Status.INCONCLUSIVE, reason="needs at least 2 observations per sample", **base,  # type: ignore[arg-type]
        )  # fmt: skip

    def theta(gs: Sequence[Sequence[float]]) -> float:
        return est(gs[0]) if len(gs) == 1 else est(gs[1]) - est(gs[0])

    point = theta(groups)
    rng = random.Random(seed)  # noqa: S311  # seeded and recorded; not security-relevant
    draws = sorted(theta([rng.choices(g, k=len(g)) for g in groups]) for _ in range(resamples))
    warn: list[str] = []
    if min(ns) < SMALL_SAMPLE:
        warn.append(f"n={min(ns)} < {SMALL_SAMPLE}: the interval may be unstable or too narrow")
    if resamples < COARSE_RESAMPLES:
        warn.append(f"{resamples} resamples: endpoints are coarse (< {COARSE_RESAMPLES})")
    alpha = (1.0 - confidence) / 2.0
    common = {**base, "estimate": point, "valid_resamples": len(draws)}
    if draws[0] == draws[-1]:
        warn.append("every resample gave the same value (zero variance): degenerate interval")
        return ConfidenceInterval(
            Status.DERIVED, lower=draws[0], upper=draws[0], warnings=tuple(warn), **common  # type: ignore[arg-type]
        )  # fmt: skip
    if method == "percentile":
        return ConfidenceInterval(
            Status.DERIVED, lower=quantile(draws, alpha), upper=quantile(draws, 1.0 - alpha),
            warnings=tuple(warn), **common,  # type: ignore[arg-type]
        )  # fmt: skip
    less = sum(t < point for t in draws) + 0.5 * sum(t == point for t in draws)
    p0 = less / len(draws)  # mid-rank proportion: ties count half
    if not 0.0 < p0 < 1.0:
        return ConfidenceInterval(
            Status.UNDEFINED, warnings=tuple(warn),
            reason="BCa undefined: the estimate is below or above every resample (z0 infinite)",
            **common,  # type: ignore[arg-type]
        )  # fmt: skip
    z0 = _ND.inv_cdf(p0)
    jack = [
        theta([g[:j] + g[j + 1 :] if gi == i else g for gi, g in enumerate(groups)])
        for i, gr in enumerate(groups)
        for j in range(len(gr))
    ]
    a = jackknife_acceleration(jack)
    levels = None if a is None else bca_levels(z0, a, alpha)
    if a is None or levels is None:
        return ConfidenceInterval(
            Status.UNDEFINED, bias_correction_z0=z0, acceleration=a, warnings=tuple(warn),
            reason="BCa undefined: the jackknife is constant or the adjustment denominator <= 0",
            **common,  # type: ignore[arg-type]
        )  # fmt: skip
    return ConfidenceInterval(
        Status.DERIVED, lower=quantile(draws, levels[0]), upper=quantile(draws, levels[1]),
        bias_correction_z0=z0, acceleration=a, warnings=tuple(warn), **common,  # type: ignore[arg-type]
    )  # fmt: skip


# -- tests (permutation: no distributional table needed, exact when small) -----------------------


@dataclass(frozen=True)
class TestResult:
    __test__ = False  # not a pytest class
    name: str
    status: Status
    statistic: float | None = None
    p_value: float | None = None
    exact: bool | None = None
    permutations: int = 0
    seed: int | None = None
    assumptions: tuple[str, ...] = ()
    reason: str | None = None
    note: str = (
        "a p-value is the probability, under the stated null, of a statistic at least this extreme; "
        "it says nothing about practical importance or causation"
    )


def _tol(scale: float) -> float:
    return 1e-12 * max(1.0, scale)


def paired_sign_flip_test(
    reference: Mapping[str, float], treatment: Mapping[str, float], *, permutations: int = 10_000,
    seed: int = 0,
) -> TestResult:  # fmt: skip
    """Two-sided sign-flip permutation test on the per-key differences. Null: the distribution of
    each difference is symmetric about zero (exchangeable signs). Exact for n <= 15 (2^n <= EXACT_LIMIT)."""
    name = "paired_sign_flip"
    d = _paired_diffs(reference, treatment)
    n = len(d)
    ass = ("pairs are independent of each other", "under H0 each difference is symmetric about 0")
    if n < 2:
        return TestResult(
            name, Status.INCONCLUSIVE, assumptions=ass, reason="needs at least 2 pairs"
        )
    obs = abs(math.fsum(d))
    tol = _tol(math.fsum(abs(e) for e in d))
    if 2**n <= EXACT_LIMIT:
        hits = sum(
            abs(math.fsum(e if m >> i & 1 else -e for i, e in enumerate(d))) >= obs - tol
            for m in range(2**n)
        )
        p, exact, used = hits / 2**n, True, 2**n
    else:
        rng = random.Random(seed)  # noqa: S311
        hits = sum(
            abs(math.fsum(e if rng.random() < 0.5 else -e for e in d)) >= obs - tol
            for _ in range(permutations)
        )
        p, exact, used = (hits + 1) / (permutations + 1), False, permutations
    return TestResult(
        name, Status.DERIVED, math.fsum(d) / n, p, exact, used, None if exact else seed, ass
    )


def permutation_test(
    reference: Sample, treatment: Sample, *, permutations: int = 10_000, seed: int = 0
) -> TestResult:
    """Two-sided permutation test of mean(treatment) - mean(reference) for independent groups.
    Null: the two groups come from the same distribution (exchangeable labels). Exact when the
    number of rearrangements is <= EXACT_LIMIT, otherwise a seeded Monte Carlo estimate."""
    name = "unpaired_permutation"
    x, y = _values(reference), _values(treatment)
    nx, ny = len(x), len(y)
    ass = ("the groups are independent", "under H0 the group labels are exchangeable")
    if nx < 2 or ny < 2:
        return TestResult(
            name,
            Status.INCONCLUSIVE,
            assumptions=ass,
            reason="needs at least 2 observations per group",
        )
    pool = x + y
    total = math.fsum(pool)
    obs_diff = math.fsum(y) / ny - math.fsum(x) / nx
    tol = _tol(abs(obs_diff))

    def diff(sy: float) -> float:
        return sy / ny - (total - sy) / nx

    if math.comb(nx + ny, ny) <= EXACT_LIMIT:
        hits = count = 0
        for idx in itertools.combinations(range(nx + ny), ny):
            count += 1
            hits += abs(diff(math.fsum(pool[i] for i in idx))) >= abs(obs_diff) - tol
        p, exact, used = hits / count, True, count
    else:
        rng = random.Random(seed)  # noqa: S311
        buf = list(pool)
        hits = 0
        for _ in range(permutations):
            rng.shuffle(buf)
            hits += abs(diff(math.fsum(buf[:ny]))) >= abs(obs_diff) - tol
        p, exact, used = (hits + 1) / (permutations + 1), False, permutations
    return TestResult(name, Status.DERIVED, obs_diff, p, exact, used, None if exact else seed, ass)


# -- comparison ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Comparison:
    pairing: str  # PAIRED | UNPAIRED (UNKNOWN is analyzed as UNPAIRED and says so)
    declared_pairing: str
    status: Status
    reference: Summary
    treatment: Summary
    effects: tuple[EffectSize, ...]
    test: TestResult
    interval: ConfidenceInterval
    warnings: tuple[str, ...]
    notes: tuple[str, ...]


def compare(
    reference: Sample,
    treatment: Sample,
    *,
    pairing: Pairing = Pairing.UNKNOWN,
    estimator: str = "mean",
    method: str = "percentile",
    confidence: float = 0.95,
    resamples: int = 2000,
    permutations: int = 10_000,
    seed: int = 0,
) -> Comparison:
    """Treatment vs reference: summaries, effect sizes, a permutation test and a bootstrap CI of the
    difference. Invalid pairing raises ValidationError; non-finite data raises ValidationError."""
    paired = pairing is Pairing.PAIRED
    warn = []
    if pairing is Pairing.UNKNOWN:
        warn.append("pairing was not declared; analyzed as UNPAIRED")
    x, y = _values(reference), _values(treatment)
    effects = effect_sizes(reference, treatment, pairing)
    test = (
        paired_sign_flip_test(reference, treatment, permutations=permutations, seed=seed)  # type: ignore[arg-type]
        if paired
        else permutation_test(reference, treatment, permutations=permutations, seed=seed)
    )
    iv = bootstrap_interval(
        reference, treatment, pairing=pairing, estimator=estimator, method=method,
        confidence=confidence, resamples=resamples, seed=seed,
    )  # fmt: skip
    ok = test.status is Status.DERIVED and iv.status is Status.DERIVED
    status = (
        Status.DERIVED if ok
        else Status.INCONCLUSIVE if Status.INCONCLUSIVE in (test.status, iv.status)
        else Status.UNDEFINED
    )  # fmt: skip
    return Comparison(
        "PAIRED" if paired else "UNPAIRED", pairing.value, status, summarize(x), summarize(y),
        effects, test, iv, tuple(warn),
        ("the interval and the test answer different questions and can disagree",
         "no correction for multiple comparisons is applied here (see adjust_pvalues)"),
    )  # fmt: skip


# -- multiple comparisons ------------------------------------------------------------------------

CORRECTIONS = ("NONE", "BONFERRONI", "BENJAMINI_HOCHBERG")


@dataclass(frozen=True)
class Correction:
    method: str
    alpha: float
    n_hypotheses: int  # hypotheses with a valid p-value (the correction family)
    n_total: int  # including those without a valid p-value
    adjusted: Mapping[str, float]
    rejected: Mapping[str, bool]
    excluded: tuple[str, ...]  # keys with no valid p-value; not part of the family
    controls: str
    note: str


def adjust_pvalues(
    pvalues: Mapping[str, float | None], *, method: str = "BONFERRONI", alpha: float = 0.05
) -> Correction:
    """Bonferroni (family-wise error rate) or Benjamini-Hochberg (false discovery rate, valid for
    independent or positively dependent tests). The family is exactly the keys given: the caller
    decides what counts as one hypothesis family. None p-values are excluded and listed."""
    if method not in CORRECTIONS:
        raise ValidationError(f"method must be one of {list(CORRECTIONS)}")
    if not 0.0 < alpha < 1.0:
        raise ValidationError("alpha must be in (0, 1)")
    valid = {k: p for k, p in pvalues.items() if p is not None}
    for k, p in valid.items():
        if isinstance(p, bool) or not isinstance(p, int | float) or not 0.0 <= p <= 1.0:
            raise ValidationError(f"p-value for {k!r} must be a finite number in [0, 1]: {p!r}")
    m = len(valid)
    if method == "NONE":
        adj = {k: float(p) for k, p in valid.items()}
        controls = "nothing (no correction applied)"
    elif method == "BONFERRONI":
        adj = {k: min(1.0, p * m) for k, p in valid.items()}
        controls = "family-wise error rate"
    else:
        order = sorted(valid, key=lambda k: (valid[k], k))
        adj, running = {}, 1.0
        for rank in range(m, 0, -1):
            k = order[rank - 1]
            running = min(running, valid[k] * m / rank)
            adj[k] = running
        controls = "false discovery rate (independent or positively dependent tests)"
    return Correction(
        method, alpha, m, len(pvalues), dict(sorted(adj.items())),
        {k: a <= alpha for k, a in sorted(adj.items())},
        tuple(sorted(k for k, p in pvalues.items() if p is None)), controls,
        "adjusted values are relative to exactly this family; adding or removing a hypothesis changes them",
    )  # fmt: skip


# -- proportions ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class ProportionInterval:
    status: Status
    method: str
    confidence: float
    successes: int
    trials: int
    estimate: float | None = None
    lower: float | None = None
    upper: float | None = None
    warnings: tuple[str, ...] = ()
    reason: str | None = None


def proportion_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> ProportionInterval:
    """Wilson score interval for a binomial proportion. Assumes independent, identically
    distributed Bernoulli trials; the caller must say whether that holds."""
    if not 0.0 < confidence < 1.0:
        raise ValidationError("confidence must be in (0, 1)")
    if not (isinstance(successes, int) and isinstance(trials, int)) or not 0 <= successes <= trials:
        raise ValidationError("need integers 0 <= successes <= trials")
    base = {"method": "wilson", "confidence": confidence, "successes": successes, "trials": trials}
    if trials == 0:
        return ProportionInterval(Status.UNDEFINED, reason="no trials", **base)  # type: ignore[arg-type]
    z = _ND.inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    p = successes / trials
    den = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / den
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / den
    warn = (
        (f"n={trials} < 10: the interval is very wide and the normal approximation is weak",)
        if trials < 10
        else ()
    )
    return ProportionInterval(
        Status.DERIVED, estimate=p, lower=0.0 if successes == 0 else max(0.0, centre - half),
        upper=1.0 if successes == trials else min(1.0, centre + half),  # exact: no float residue
        warnings=warn, **base,  # type: ignore[arg-type]
    )  # fmt: skip

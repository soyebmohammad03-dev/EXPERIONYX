"""Calibration mathematics, pure Python (the core stays dependency-free). Every convention is fixed
here and recorded in the artifacts; docs/calibration.md is the reference.

Objects. Three DIFFERENT things are measured and never mixed:
  TOP_LABEL           confidence = probability the model assigns to its PREDICTED class; the event is
                      "the prediction is correct". Bins, ECE, MCE, top-label Brier / log loss.
  CLASSWISE           for each class k, p_k = probability of class k; the event is "the true class is k".
                      A per-class reliability curve and ECE_k; the mean over classes is reported only
                      with its per-class values.
  PROBABILITY_VECTOR  proper scoring rules of the whole vector: multiclass Brier and negative
                      log-likelihood. These are not calibration curves.

Bins are [lower, upper) with the LAST bin closed at its upper edge, and membership is decided by
bisection on the very edges that are reported, so a confidence exactly on a boundary belongs to the
upper bin and float rounding can never disagree with the stored bounds. ECE = sum_b (n_b/n)|acc_b -
conf_b| over non-empty bins (empty bins contribute 0 and are still listed). MCE = max_b |acc_b -
conf_b| over non-empty bins. Log loss clips probabilities to [EPS, 1 - EPS]."""

import bisect
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from experionyx.evaluation.bootstrap import quantile
from experionyx.stats import core as st
from experionyx.stats.core import Status as StatStatus

EPS = 1e-15  # probability clip for logarithms
TOL = 1e-9  # a probability this far outside [0, 1] is rounding; further is invalid
SUM_TOL = 1e-6  # a probability vector must sum to 1 within this
UNIFORM, QUANTILE = "UNIFORM", "QUANTILE"
BINNINGS = (UNIFORM, QUANTILE)
QUANTILE_PER_BIN = 5  # quantile binning needs at least this many observations per requested bin


@dataclass(frozen=True)
class Obs:
    """One usable observation: sample ID, class indices of the truth and the prediction, the full
    probability vector, and (after post-hoc calibration only) the calibrated top-label confidence."""

    id: int
    true: int
    pred: int
    probs: tuple[float, ...]
    cal: float | None = None

    @property
    def conf(self) -> float:
        return self.probs[self.pred] if self.cal is None else self.cal

    @property
    def correct(self) -> bool:
        return self.true == self.pred


def to_obs(
    idx: int, true: object, pred: object, scores: object, classes: Sequence[object]
) -> Obs | str:
    """An Obs, or the reason code the row is INVALID. Nothing is repaired or dropped silently."""
    if scores is None:
        return "MISSING_SCORES"
    try:
        vec = tuple(float(x) for x in scores)  # type: ignore[attr-defined]
    except (TypeError, ValueError):
        return "NON_NUMERIC_SCORES"
    if len(vec) != len(classes):
        return "WRONG_WIDTH"
    if not all(math.isfinite(x) for x in vec):
        return "NON_FINITE"
    if any(x < -TOL or x > 1.0 + TOL for x in vec):
        return "OUT_OF_RANGE"
    vec = tuple(min(max(x, 0.0), 1.0) for x in vec)
    if abs(math.fsum(vec) - 1.0) > SUM_TOL:
        return "NOT_NORMALIZED"
    index = {c: i for i, c in enumerate(classes)}
    if true not in index:
        return "INVALID_TARGET"
    if pred not in index:
        return "INVALID_PREDICTION"
    return Obs(idx, index[true], index[pred], vec)


# -- bins ------------------------------------------------------------------------------------------------------


def make_edges(strategy: str, n_bins: int, values: Sequence[float]) -> list[float]:
    """UNIFORM: k/B. QUANTILE: the observed values at equal-count ranks (ties share a bin, so fewer
    bins than requested may result); the outer edges are the observed minimum and maximum."""
    if strategy == UNIFORM:
        return [k / n_bins for k in range(n_bins + 1)]
    s = sorted(values)
    n = len(s)
    lo, hi = s[0], s[-1]
    inner = sorted({s[(k * n) // n_bins] for k in range(1, n_bins)})
    return [lo, *(e for e in inner if lo < e <= hi), hi]


def assign(value: float, edges: Sequence[float]) -> int:
    return min(max(bisect.bisect_right(edges, value) - 1, 0), len(edges) - 2)


def _bin_evidence(count: int, min_bin: int) -> str:
    return "EMPTY" if count == 0 else "SINGLE_OBSERVATION" if count == 1 else "SPARSE" if count < min_bin else "OK"  # fmt: skip


def curve(
    pairs: Sequence[tuple[float, bool]],
    edges: Sequence[float],
    *,
    min_bin: int = QUANTILE_PER_BIN,
    confidence: float | None = None,
) -> list[dict[str, Any]]:
    """The reliability table of (probability, event) pairs. Empty bins are kept. With `confidence`,
    each non-empty bin gets a Wilson interval for its observed rate (binomial, i.i.d. assumption)."""
    members: list[list[tuple[float, bool]]] = [[] for _ in range(len(edges) - 1)]
    for p, e in pairs:
        members[assign(p, edges)].append((p, e))
    out = []
    for i, m in enumerate(members):
        n = len(m)
        row: dict[str, Any] = {
            "index": i, "lower": edges[i], "upper": edges[i + 1], "upper_closed": i == len(members) - 1,
            "count": n, "mean_confidence": None, "accuracy": None, "gap": None, "abs_gap": None,
            "evidence": _bin_evidence(n, min_bin), "interval": None,
        }  # fmt: skip
        if n:
            conf = math.fsum(p for p, _ in m) / n
            hits = sum(e for _, e in m)
            acc = hits / n
            row.update(mean_confidence=conf, accuracy=acc, gap=conf - acc, abs_gap=abs(conf - acc))
            if confidence is not None:
                row["interval"] = wilson(hits, n, confidence)
        out.append(row)
    return out


def wilson(hits: int, n: int, confidence: float) -> dict[str, Any]:
    w = st.proportion_interval(hits, n, confidence)
    return {"method": w.method, "confidence": w.confidence, "lower": w.lower, "upper": w.upper}


def ece(bins: Sequence[Mapping[str, Any]], n: int) -> float | None:
    return math.fsum(b["count"] / n * b["abs_gap"] for b in bins if b["count"]) if n else None


def mce(bins: Sequence[Mapping[str, Any]]) -> float | None:
    gaps = [b["abs_gap"] for b in bins if b["count"]]
    return max(gaps) if gaps else None


def _clip(p: float) -> float:
    return min(max(p, EPS), 1.0 - EPS)


def top_pairs(obs: Sequence[Obs]) -> list[tuple[float, bool]]:
    return [(o.conf, o.correct) for o in obs]


def top_brier(pairs: Sequence[tuple[float, bool]]) -> float:
    return math.fsum((c - e) ** 2 for c, e in pairs) / len(pairs)


def top_log_loss(pairs: Sequence[tuple[float, bool]]) -> float:
    return -math.fsum(math.log(_clip(c if e else 1.0 - c)) for c, e in pairs) / len(pairs)  # fmt: skip


def vector_brier(obs: Sequence[Obs]) -> float:
    """Multiclass Brier: mean over samples of sum_k (p_k - 1[y=k])^2 (range 0..2)."""
    return math.fsum(math.fsum((p - (k == o.true)) ** 2 for k, p in enumerate(o.probs)) for o in obs) / len(obs)  # fmt: skip


def vector_nll(obs: Sequence[Obs]) -> float:
    return -math.fsum(math.log(_clip(o.probs[o.true])) for o in obs) / len(obs)


def top_metrics(obs: Sequence[Obs], strategy: str, n_bins: int, *, vector: bool = True) -> dict[str, float | None]:  # fmt: skip
    """Every scalar of a context at once (the unit the bootstrap and permutation tests recompute)."""
    if not obs:
        return {}
    pairs = top_pairs(obs)
    bins = curve(pairs, make_edges(strategy, n_bins, [p for p, _ in pairs]))
    n = len(pairs)
    acc = sum(e for _, e in pairs) / n
    conf = math.fsum(p for p, _ in pairs) / n
    m: dict[str, float | None] = {
        "accuracy": acc, "mean_confidence": conf, "overconfidence": conf - acc, "ece": ece(bins, n),
        "mce": mce(bins), "brier_top_label": top_brier(pairs), "log_loss_top_label": top_log_loss(pairs),
    }  # fmt: skip
    if vector:
        m["brier_multiclass"] = vector_brier(obs)
        m["nll_multiclass"] = vector_nll(obs)
    return m


def classwise(
    obs: Sequence[Obs], classes: Sequence[object], strategy: str, n_bins: int, *,
    min_bin: int, min_positives: int, confidence: float,
) -> dict[str, Any]:  # fmt: skip
    """Per-class reliability of p_k against the event 'true class is k'. A class with fewer than
    `min_positives` positive samples reports INSUFFICIENT_EVIDENCE and no ECE."""
    out: dict[str, Any] = {}
    for k, label in enumerate(classes):
        pairs = [(o.probs[k], o.true == k) for o in obs]
        pos = sum(e for _, e in pairs)
        row: dict[str, Any] = {"label": label, "positives": pos, "n": len(pairs)}
        edges = make_edges(strategy, n_bins, [p for p, _ in pairs]) if pairs else None
        bins = curve(pairs, edges, min_bin=min_bin, confidence=confidence) if edges else []
        row["bins"] = bins
        if pos < min_positives:
            row.update(status="INSUFFICIENT_EVIDENCE", ece=None, mce=None, reason=f"{pos} positive sample(s) < min_class_positives={min_positives}")  # fmt: skip
        else:
            row.update(status="COMPUTED", ece=ece(bins, len(pairs)), mce=mce(bins), brier=top_brier(pairs), reason=None)  # fmt: skip
        out[str(label)] = row
    done = [r["ece"] for r in out.values() if r["status"] == "COMPUTED"]
    return {"per_class": out, "mean_ece": math.fsum(done) / len(done) if done else None, "classes_computed": len(done), "classes_total": len(out)}  # fmt: skip


# -- prediction uncertainty (descriptive, from the stored probability vector) -------------------------------


def entropy(probs: Sequence[float]) -> float:
    """Shannon entropy in nats: -sum p ln p (0 ln 0 = 0)."""
    return -math.fsum(p * math.log(p) for p in probs if p > 0.0)


def margin(probs: Sequence[float]) -> float | None:
    if len(probs) < 2:
        return None
    a, b = sorted(probs, reverse=True)[:2]
    return a - b


def normalized_entropy(probs: Sequence[float]) -> float | None:
    return entropy(probs) / math.log(len(probs)) if len(probs) >= 2 else None


def auroc(scores_pos: Sequence[float], scores_neg: Sequence[float]) -> float | None:
    """P(score of a random positive > score of a random negative), ties counting half."""
    if not scores_pos or not scores_neg:
        return None
    neg = sorted(scores_neg)
    total = 0.0
    for s in scores_pos:
        lo, hi = bisect.bisect_left(neg, s), bisect.bisect_right(neg, s)
        total += lo + 0.5 * (hi - lo)
    return total / (len(scores_pos) * len(neg))


# -- post-hoc calibrators (top-label; fit on calibration data only) --------------------------------------------


def _logit(c: float) -> float:
    c = _clip(c)
    return math.log(c / (1.0 - c))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


PLATT_RIDGE = 1e-6  # penalty on the slope only; makes separable data converge, and is recorded


def fit_platt(pairs: Sequence[tuple[float, bool]]) -> dict[str, Any]:
    """Logistic (Platt) scaling of the top-label confidence: P(correct) = sigmoid(a * logit(conf) +
    b), fitted by Newton's method with step halving from the identity map (a=1, b=0), minimizing
    sum(log loss) + ridge/2 * a^2. Deterministic; no randomness."""
    xs = [_logit(c) for c, _ in pairs]
    ys = [1.0 if e else 0.0 for _, e in pairs]

    def loss(a: float, b: float) -> float:
        return math.fsum(math.log1p(math.exp(-abs(z))) + max(z, 0.0) - y * z for z, y in ((a * x + b, y) for x, y in zip(xs, ys, strict=True))) + 0.5 * PLATT_RIDGE * a * a  # fmt: skip

    a, b, converged, it = 1.0, 0.0, False, 0
    for it in range(1, 101):  # noqa: B007
        p = [_sigmoid(a * x + b) for x in xs]
        ga = math.fsum((pi - y) * x for pi, y, x in zip(p, ys, xs, strict=True)) + PLATT_RIDGE * a
        gb = math.fsum(pi - y for pi, y in zip(p, ys, strict=True))
        w = [pi * (1 - pi) for pi in p]
        haa = math.fsum(wi * x * x for wi, x in zip(w, xs, strict=True)) + PLATT_RIDGE
        hab = math.fsum(wi * x for wi, x in zip(w, xs, strict=True))
        hbb = math.fsum(w)
        det = haa * hbb - hab * hab
        if det <= 1e-300:
            break
        da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
        step, base = 1.0, loss(a, b)
        while step > 1e-12 and loss(a - step * da, b - step * db) > base:
            step /= 2
        a, b = a - step * da, b - step * db
        if max(abs(step * da), abs(step * db)) < 1e-10:
            converged = True
            break
    return {"a": a, "b": b, "ridge": PLATT_RIDGE, "iterations": it, "converged": converged, "n_fit": len(pairs), "loss": loss(a, b)}  # fmt: skip


def fit_isotonic(pairs: Sequence[tuple[float, bool]]) -> dict[str, Any]:
    """Isotonic regression of correctness on confidence by pool-adjacent-violators; observations
    with identical confidence are pooled first. The fitted function is the monotone knot list
    (x, y), applied by linear interpolation and clipped outside the fitted range."""
    by_x: dict[float, list[float]] = {}
    for c, e in pairs:
        s = by_x.setdefault(c, [0.0, 0.0])
        s[0] += 1.0 if e else 0.0
        s[1] += 1.0
    blocks: list[list[float]] = []  # [sum, count, x_first, x_last]
    for x in sorted(by_x):
        blocks.append([by_x[x][0], by_x[x][1], x, x])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            top = blocks.pop()
            blocks[-1] = [blocks[-1][0] + top[0], blocks[-1][1] + top[1], blocks[-1][2], top[3]]
    knots: list[tuple[float, float]] = []
    for total, count, lo, hi in blocks:
        knots.extend((x, total / count) for x in ((lo, hi) if hi != lo else (lo,)))
    return {"x": [k[0] for k in knots], "y": [k[1] for k in knots], "n_fit": len(pairs), "blocks": len(blocks)}  # fmt: skip


def apply_calibrator(method: str, params: Mapping[str, Any], conf: float) -> float:
    if method == "PLATT":
        return _sigmoid(params["a"] * _logit(conf) + params["b"])
    xs, ys = params["x"], params["y"]
    if conf <= xs[0]:
        return float(ys[0])
    if conf >= xs[-1]:
        return float(ys[-1])
    j = bisect.bisect_right(xs, conf)
    x0, x1, y0, y1 = xs[j - 1], xs[j], ys[j - 1], ys[j]
    return float(y0 + (y1 - y0) * (conf - x0) / (x1 - x0))


# -- uncertainty of the metrics themselves: seeded percentile bootstrap and permutation --------------------


def _interval(draws: list[float], point: float | None, *, confidence: float, seed: int, n: int) -> dict[str, Any]:  # fmt: skip
    base: dict[str, Any] = {"method": "bootstrap-percentile", "confidence": confidence, "resamples": len(draws), "seed": seed, "n": n, "estimate": point}  # fmt: skip
    if not draws:
        return {**base, "status": StatStatus.UNDEFINED.value, "lower": None, "upper": None, "warnings": [], "reason": "no valid resample"}  # fmt: skip
    draws = sorted(draws)
    alpha = (1.0 - confidence) / 2.0
    warn = []
    if draws[0] == draws[-1]:
        warn.append("every resample gave the same value (zero variance): degenerate interval")
    return {**base, "status": StatStatus.DERIVED.value, "lower": quantile(draws, alpha), "upper": quantile(draws, 1.0 - alpha), "warnings": warn, "reason": None}  # fmt: skip


def bootstrap_metrics(
    obs: Sequence[Obs], strategy: str, n_bins: int, *, resamples: int, confidence: float, seed: int, vector: bool = True,
) -> dict[str, dict[str, Any]]:  # fmt: skip
    """Percentile bootstrap intervals of every scalar metric, resampling OBSERVATIONS with a seeded
    generator; quantile bin edges are recomputed on each resample. Assumes i.i.d. observations. ECE is
    positively biased in small samples and this percentile interval does not correct that."""
    point = top_metrics(obs, strategy, n_bins, vector=vector)
    rng = random.Random(seed)  # noqa: S311  # seeded and recorded; not security-relevant
    draws: dict[str, list[float]] = {k: [] for k in point}
    n = len(obs)
    for _ in range(resamples):
        m = top_metrics(rng.choices(obs, k=n), strategy, n_bins, vector=vector)
        for k, x in m.items():
            if x is not None:
                draws[k].append(x)
    return {
        k: _interval(d, point[k], confidence=confidence, seed=seed, n=n) for k, d in draws.items()
    }


def _metric_diff(a: Sequence[Obs], b: Sequence[Obs], strategy: str, n_bins: int, names: Sequence[str]) -> dict[str, float]:  # fmt: skip
    ma = top_metrics(a, strategy, n_bins, vector=False)
    mb = top_metrics(b, strategy, n_bins, vector=False)
    return {k: mb[k] - ma[k] for k in names if ma.get(k) is not None and mb.get(k) is not None}  # type: ignore[operator]


def compare_metrics(
    a: Sequence[Obs], b: Sequence[Obs], *, paired: bool, strategy: str, n_bins: int, names: Sequence[str],
    resamples: int, permutations: int, confidence: float, seed: int,
) -> dict[str, dict[str, Any]]:  # fmt: skip
    """Difference (b - a) of non-mean metrics (ECE, MCE) with a bootstrap interval and a permutation
    p-value. PAIRED: `a[i]` and `b[i]` are the same sample under two conditions; the bootstrap
    resamples pairs and the permutation swaps the two conditions within pairs. UNPAIRED: independent
    groups; the bootstrap resamples each group and the permutation re-splits the pooled sample."""
    point = _metric_diff(a, b, strategy, n_bins, names)
    rng = random.Random(seed)  # noqa: S311  # seeded and recorded; not security-relevant
    boot: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(resamples):
        if paired:
            idx = rng.choices(range(len(a)), k=len(a))
            ra, rb = [a[i] for i in idx], [b[i] for i in idx]
        else:
            ra, rb = rng.choices(a, k=len(a)), rng.choices(b, k=len(b))
        for k, x in _metric_diff(ra, rb, strategy, n_bins, names).items():
            boot[k].append(x)
    hits = dict.fromkeys(point, 0)
    for _ in range(permutations):
        if paired:
            swap = [rng.random() < 0.5 for _ in a]
            pa = [y if s else x for x, y, s in zip(a, b, swap, strict=True)]
            pb = [x if s else y for x, y, s in zip(a, b, swap, strict=True)]
        else:
            pool = [*a, *b]
            rng.shuffle(pool)
            pa, pb = pool[: len(a)], pool[len(a) :]
        for k, x in _metric_diff(pa, pb, strategy, n_bins, names).items():
            hits[k] += abs(x) >= abs(point[k]) - 1e-12
    return {
        k: {
            "difference": point[k],
            "interval": _interval(boot[k], point[k], confidence=confidence, seed=seed, n=len(a)),
            "p_value": (hits[k] + 1) / (permutations + 1) if permutations else None,
            "test": {
                "name": "paired condition-swap permutation" if paired else "pooled permutation",
                "permutations": permutations,
                "seed": seed,
                "exact": False,
            },
        }
        for k in point
    }

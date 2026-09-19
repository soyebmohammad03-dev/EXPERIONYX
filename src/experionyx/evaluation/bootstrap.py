"""Deterministic percentile-bootstrap confidence intervals for scalar metrics.

Method: draw `resamples` index sets of size n with replacement using `random.Random(seed)`,
recompute the metric on each resample, and take the (1-c)/2 and 1-(1-c)/2 quantiles (linear
interpolation). Limitations (docs/evaluation.md): percentile intervals can under-cover for small
n or skewed statistics, ignore any dependence between samples, and say nothing about
distribution shift; they are one reasonable choice, not a universally better one than analytical
intervals. Resamples in which the metric is undefined are dropped and counted.
"""

import random
from collections.abc import Sequence

from experionyx.evaluation.metrics import MetricInputs, MetricSpec, compute_metric
from experionyx.evaluation.results import Interval, Status

METHOD = "bootstrap-percentile"
SMALL_SAMPLE = 30
COARSE_RESAMPLES = 1000


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (the common 'type 7' definition) of sorted values."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def _subset(inp: MetricInputs, idx: Sequence[int]) -> MetricInputs:
    return MetricInputs(
        inp.task,
        [inp.y_true[i] for i in idx],
        [inp.y_pred[i] for i in idx],
        None if inp.scores is None else [inp.scores[i] for i in idx],
        inp.classes,
    )


def bootstrap_interval(
    spec: MetricSpec, inp: MetricInputs, *, resamples: int, confidence: float, seed: int
) -> Interval:
    n = len(inp.y_true)
    base = {"method": METHOD, "confidence": confidence, "resamples": resamples, "seed": seed}
    if n == 0:
        return Interval(
            Status.UNDEFINED,
            valid_resamples=0,
            n_samples=0,
            reason="no samples",
            **base,  # type: ignore[arg-type]
        )
    rng = random.Random(seed)  # noqa: S311  # statistical resampling, not security
    values: list[float] = []
    for _ in range(resamples):
        outcome = compute_metric(spec, _subset(inp, [rng.randrange(n) for _ in range(n)]))
        if outcome.status is Status.COMPUTED and outcome.value is not None:
            values.append(outcome.value)
    if not values:
        return Interval(
            Status.UNDEFINED,
            valid_resamples=0,
            n_samples=n,
            reason="the metric was undefined in every resample",
            **base,  # type: ignore[arg-type]
        )
    values.sort()
    alpha = (1.0 - confidence) / 2.0
    warnings = []
    if n < SMALL_SAMPLE:
        warnings.append(f"n={n} < {SMALL_SAMPLE}: the interval may be unstable or too narrow")
    if len(values) < resamples:
        warnings.append(f"{resamples - len(values)} resample(s) had an undefined metric (dropped)")
    if resamples < COARSE_RESAMPLES:
        warnings.append(
            f"{resamples} resamples: interval endpoints are coarse (< {COARSE_RESAMPLES})"
        )
    return Interval(
        status=Status.COMPUTED,
        valid_resamples=len(values),
        n_samples=n,
        lower=quantile(values, alpha),
        upper=quantile(values, 1.0 - alpha),
        warnings=tuple(warnings),
        **base,  # type: ignore[arg-type]
    )

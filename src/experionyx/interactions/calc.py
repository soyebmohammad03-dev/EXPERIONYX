"""The interaction mathematics. Pure functions; no I/O, no hidden state.

    effect_A = YA - Y0            effect_B = YB - Y0            combined = YAB - Y0
    expected_additive = effect_A + effect_B
    contrast = combined - expected_additive = YAB - YA - YB + Y0

The contrast is a mathematical quantity. It is not a causal effect, not a mechanism, and not a
performance deterioration: its sign means different things for HIGHER_IS_BETTER and
LOWER_IS_BETTER metrics, so the direction-aware reading is computed separately and the raw
contrast is always kept as is."""

import math
import random
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from experionyx.evaluation.bootstrap import quantile
from experionyx.interactions.config import InteractionConfig
from experionyx.interactions.taxonomy import (
    Aggregation,
    EffectStatus,
    InteractionClass,
    Normalization,
    Pairing,
    Relation,
)


def finite(x: object) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x)


def aggregate(values: Sequence[float], how: Aggregation) -> float:
    if not values:
        raise ValueError("cannot aggregate no values")
    return statistics.fmean(values) if how is Aggregation.MEAN else statistics.median(values)


def describe(values: Sequence[float]) -> dict[str, float | int | None]:
    """Repeated-trial summary; the raw values are kept by the caller (never discarded)."""
    n = len(values)
    sd = statistics.stdev(values) if n > 1 else None
    return {
        "n": n,
        "mean": statistics.fmean(values) if n else None,
        "median": statistics.median(values) if n else None,
        "std": sd,
        "standard_error": None if sd is None else sd / math.sqrt(n),
        "min": min(values) if n else None,
        "max": max(values) if n else None,
    }


@dataclass(frozen=True)
class Contrast:
    y0: float
    ya: float
    yb: float
    yab: float
    effect_a: float
    effect_b: float
    combined_effect: float
    expected_additive_effect: float
    interaction_contrast: float
    normalization: Normalization
    normalized_contrast: float | None
    normalization_note: str | None  # why the normalized value is absent, when it is


def _denominator(norm: Normalization, y0: float, ea: float, eb: float) -> float | None:
    if norm is Normalization.NONE:
        return None
    return {
        Normalization.BASELINE_MAGNITUDE: abs(y0),
        Normalization.EXPECTED_ADDITIVE_MAGNITUDE: abs(ea + eb),
        Normalization.MAX_COMPONENT_MAGNITUDE: max(abs(ea), abs(eb)),
    }[norm]


def contrast(
    y0: float, ya: float, yb: float, yab: float, norm: Normalization = Normalization.NONE
) -> Contrast:
    """Raises ValueError for non-finite inputs: undefined inputs must be handled by the caller,
    never turned into a number here."""
    if not all(finite(x) for x in (y0, ya, yb, yab)):
        raise ValueError("interaction contrast needs four finite values")
    ea, eb, comb = ya - y0, yb - y0, yab - y0
    expected = ea + eb
    c = comb - expected
    den = _denominator(norm, y0, ea, eb)
    if norm is Normalization.NONE:
        nc, note = None, "no normalization configured"
    elif den is None or den == 0.0:
        nc, note = None, f"zero denominator under {norm.value}"
    else:
        nc, note = c / den, None
    return Contrast(y0, ya, yb, yab, ea, eb, comb, expected, c, norm, nc, note)


def deterioration_view(c: float, higher_is_better: bool | None) -> tuple[float | None, Relation]:
    """(contrast oriented so that > 0 means the combination is WORSE than additive, relation)."""
    if higher_is_better is None:
        return None, Relation.UNKNOWN_DIRECTION
    oriented = -c if higher_is_better else c
    if oriented == 0.0:
        return 0.0, Relation.ADDITIVE
    return (
        oriented,
        Relation.SUPER_ADDITIVE_DETERIORATION
        if oriented > 0
        else Relation.SUB_ADDITIVE_DETERIORATION,
    )


# -- bootstrap over TRIALS ---------------------------------------------------------------------------

CellValues = Mapping[str, Mapping[str, float]]  # cell -> trial key -> metric value


@dataclass(frozen=True)
class BootstrapResult:
    status: EffectStatus
    method: str
    estimator: str
    pairing: str
    n_trials: Mapping[str, int]
    resamples: int
    valid_resamples: int
    seed: int
    confidence: float
    intervals: Mapping[str, Mapping[str, float | None]] = field(
        default_factory=dict
    )  # stat -> lower/upper/sign_fraction
    warnings: tuple[str, ...] = ()
    reason: str | None = None


def stats_of(v: Mapping[str, float], norm: Normalization) -> dict[str, float | None]:
    c = contrast(v["CONTROL"], v["A"], v["B"], v["AB"], norm)
    out: dict[str, float | None] = {
        "effect_a": c.effect_a,
        "effect_b": c.effect_b,
        "combined_effect": c.combined_effect,
        "expected_additive_effect": c.expected_additive_effect,
        "interaction_contrast": c.interaction_contrast,
        "normalized_contrast": c.normalized_contrast,
    }
    if "BA" in v:
        ba = contrast(v["CONTROL"], v["A"], v["B"], v["BA"], norm)
        out["interaction_contrast_ba"] = ba.interaction_contrast
        out["order_effect"] = c.interaction_contrast - ba.interaction_contrast  # = YAB - YBA
    return out


def bootstrap(cells: CellValues, cfg: InteractionConfig, pairing: Pairing) -> BootstrapResult:
    """Percentile bootstrap that resamples TRIALS (whole runs), never individual samples.
    UNPAIRED: each cell is resampled independently. PAIRED: trial keys are resampled jointly, so
    a key drawn for one cell is drawn for all. A single-valued control is held fixed. Below
    `min_trials` per treatment cell no interval is produced."""
    n = {k: len(v) for k, v in cells.items()}
    paired = pairing is Pairing.PAIRED

    def result(
        status: EffectStatus,
        valid: int,
        intervals: Mapping[str, Mapping[str, float | None]] | None = None,
        warnings: tuple[str, ...] = (),
        reason: str | None = None,
    ) -> BootstrapResult:
        return BootstrapResult(
            status,
            "percentile",
            cfg.aggregation.value,
            "PAIRED" if paired else "UNPAIRED",
            n,
            cfg.bootstrap_resamples,
            valid,
            cfg.bootstrap_seed,
            cfg.confidence,
            intervals or {},
            warnings,
            reason,
        )

    treat = [k for k in cells if k != "CONTROL"]
    if any(n.get(k, 0) == 0 for k in cells):
        return result(EffectStatus.UNDEFINED, 0, reason="a cell has no valid trial values")
    low = {k: n[k] for k in treat if n[k] < cfg.min_trials}
    if low:
        return result(
            EffectStatus.INSUFFICIENT_DATA,
            0,
            reason=f"fewer than min_trials={cfg.min_trials} trials in {low}",
        )
    rng = random.Random(cfg.bootstrap_seed)  # noqa: S311  # seeded and recorded; not security-relevant
    keys = {k: sorted(cells[k]) for k in cells}
    joint = sorted(cells[treat[0]]) if paired else []
    draws: dict[str, list[float]] = {}
    valid = 0
    for _ in range(cfg.bootstrap_resamples):
        pick = [joint[rng.randrange(len(joint))] for _ in joint] if paired else []
        agg: dict[str, float] = {}
        for k in cells:
            if len(keys[k]) == 1 and k == "CONTROL":
                agg[k] = cells[k][keys[k][0]]
            elif paired:
                agg[k] = aggregate([cells[k][x] for x in pick], cfg.aggregation)
            else:
                agg[k] = aggregate(
                    [cells[k][keys[k][rng.randrange(len(keys[k]))]] for _ in keys[k]],
                    cfg.aggregation,
                )
        try:
            s = stats_of(agg, cfg.normalization)
        except ValueError:
            continue
        valid += 1
        for name, val in s.items():
            if val is not None:
                draws.setdefault(name, []).append(val)
    if valid == 0:
        return result(EffectStatus.UNDEFINED, 0, reason="no valid resample")
    alpha = (1.0 - cfg.confidence) / 2.0
    point = stats_of(
        {k: aggregate(list(cells[k].values()), cfg.aggregation) for k in cells}, cfg.normalization
    )
    intervals: dict[str, dict[str, float | None]] = {}
    for name, vals in draws.items():
        vals.sort()
        p = point[name]
        same = None if p is None or p == 0 else sum((x > 0) == (p > 0) for x in vals) / len(vals)
        # smallest two-sided level at which the percentile interval would exclude zero (+1 smoothing);
        # an approximate inversion of the interval, NOT an exact test of a null hypothesis
        tail = min(sum(x <= 0 for x in vals), sum(x >= 0 for x in vals))
        intervals[name] = {
            "lower": quantile(vals, alpha),
            "upper": quantile(vals, 1.0 - alpha),
            "sign_fraction": same,
            "n": float(len(vals)),
            "bootstrap_p": min(1.0, 2.0 * (tail + 1) / (len(vals) + 1)),
        }
    warn = [
        "percentile interval: descriptive spread of trial-level resampling, not a significance test"
    ]
    if pairing is Pairing.UNKNOWN:
        warn.append("pairing was not declared; analyzed as UNPAIRED")
    if len(cells["CONTROL"]) == 1:
        warn.append(
            "control has one run: its value is held fixed, so its own variability is not represented"
        )
    return result(EffectStatus.COMPUTED, valid, intervals, tuple(warn))


# -- classification ------------------------------------------------------------------------------------


def _excludes_zero(iv: Mapping[str, float | None] | None) -> bool:
    return (
        iv is not None
        and iv["lower"] is not None
        and iv["upper"] is not None
        and (iv["lower"] > 0 or iv["upper"] < 0)
    )


def classify(
    cfg: InteractionConfig, point: Mapping[str, float | None], boot: BootstrapResult
) -> tuple[InteractionClass, dict[str, object]]:
    """Explicit, configurable rules; the returned dict records which rule fired and the inputs.

    UNDEFINED: no contrast.  INCONCLUSIVE: too few trials for an interval.
    ORDER_DEPENDENT: an order effect (AB vs BA) whose interval excludes zero, whose magnitude
    reaches min_abs_contrast, and where at least one ordered contrast's interval excludes zero.  OBSERVED: the contrast interval excludes zero, the sign is stable
    in >= min_sign_fraction of resamples and |contrast| >= min_abs_contrast.  POSSIBLE: the
    interval excludes zero, or a magnitude threshold (> 0) is configured and met, but not all
    criteria.  A threshold of 0 is 'no magnitude criterion' and never suffices by itself.
    NO_EVIDENCE: neither (this is not evidence of absence)."""
    c = point.get("interaction_contrast")
    why: dict[str, object] = {
        "min_abs_contrast": cfg.min_abs_contrast,
        "min_sign_fraction": cfg.min_sign_fraction,
        "confidence": cfg.confidence,
    }
    if c is None:
        return InteractionClass.UNDEFINED, {**why, "rule": "no computable contrast"}
    if boot.status is not EffectStatus.COMPUTED:
        return InteractionClass.INCONCLUSIVE, {
            **why,
            "rule": f"bootstrap {boot.status.value}: {boot.reason}",
        }
    iv = boot.intervals.get("interaction_contrast")
    ex = _excludes_zero(iv)
    sign = None if iv is None else iv["sign_fraction"]
    big = abs(c) >= cfg.min_abs_contrast and c != 0.0
    thr = (
        cfg.min_abs_contrast > 0
    )  # 0 means 'no magnitude criterion': size alone then proves nothing
    why.update(
        interval_excludes_zero=ex,
        sign_fraction=sign,
        magnitude_met=big,
        magnitude_threshold_applied=thr,
    )
    oe = point.get("order_effect")
    if oe is not None:
        oiv = boot.intervals.get("order_effect")
        order_ex = _excludes_zero(oiv) and oe != 0.0 and abs(oe) >= cfg.min_abs_contrast
        ex_ba = _excludes_zero(boot.intervals.get("interaction_contrast_ba"))
        why.update(
            order_effect=oe,
            order_interval_excludes_zero=order_ex,
            contrast_ab_interval_excludes_zero=ex,
            contrast_ba_interval_excludes_zero=ex_ba,
        )
        if order_ex and (ex or ex_ba):
            return InteractionClass.ORDER_DEPENDENT_INTERACTION, {
                **why,
                "rule": "order effect interval excludes zero and at least one ordered contrast's interval excludes zero",
            }
        if order_ex:
            return InteractionClass.POSSIBLE_INTERACTION, {
                **why,
                "rule": "an order effect was observed but neither ordered contrast's interval excludes zero: no interaction label is justified",
            }
    if ex and big and sign is not None and sign >= cfg.min_sign_fraction:
        return InteractionClass.OBSERVED_INTERACTION, {
            **why,
            "rule": "interval excludes zero, sign stable, magnitude threshold met",
        }
    if ex or (thr and big):
        return InteractionClass.POSSIBLE_INTERACTION, {
            **why,
            "rule": "only part of the observation criteria is met",
        }
    return InteractionClass.NO_EVIDENCE, {
        **why,
        "rule": "interval includes zero and magnitude threshold not met",
    }


CONTRAST_FORMULAS: Callable[[], dict[str, str]] = lambda: {  # noqa: E731
    "effect_a": "YA - Y0", "effect_b": "YB - Y0", "combined_effect": "YAB - Y0",
    "expected_additive_effect": "effect_a + effect_b", "interaction_contrast": "YAB - YA - YB + Y0",
    "order_effect": "I_AB - I_BA = YAB - YBA",
}  # fmt: skip

"""Mathematical ground truth for the statistics core. Every expected number here is derived by hand
or from a closed form; these are synthetic tests, not experimental findings."""

import math
from statistics import NormalDist
from typing import Any

import pytest

from experionyx.errors import ValidationError
from experionyx.interactions.taxonomy import Pairing
from experionyx.stats.core import (
    ConfidenceInterval,
    EffectSize,
    Status,
    adjust_pvalues,
    bca_levels,
    bootstrap_interval,
    compare,
    effect_sizes,
    jackknife_acceleration,
    paired_sign_flip_test,
    permutation_test,
    proportion_interval,
    summarize,
)

ND = NormalDist()


def eff(rs: tuple[EffectSize, ...], name: str) -> EffectSize:
    return next(e for e in rs if e.name == name)


# -- summary ---------------------------------------------------------------------------------------


def test_summary_ground_truth() -> None:
    s = summarize([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    assert s.status is Status.DERIVED and s.n == 8
    assert s.mean.value == 5.0 and s.median.value == 4.5
    assert s.variance.value == pytest.approx(32 / 7)  # sum of squares 32, n-1 = 7
    assert s.std.value == pytest.approx(math.sqrt(32 / 7))
    assert s.standard_error.value == pytest.approx(math.sqrt(32 / 7) / math.sqrt(8))


def test_summary_empty_one_zero_variance_nonfinite() -> None:
    e = summarize([])
    assert e.status is Status.UNDEFINED and e.mean.value is None and e.mean.reason
    one = summarize([3.0])
    assert one.status is Status.INCONCLUSIVE and one.mean.value == 3.0
    assert one.variance.value is None and one.variance.status is Status.INCONCLUSIVE
    zv = summarize([2.0, 2.0, 2.0])
    assert zv.std.value == 0.0 and zv.std.status is Status.DERIVED  # zero variance is a value
    for bad in (math.nan, math.inf, -math.inf):
        b = summarize([1.0, bad])
        assert (
            b.status is Status.UNDEFINED and b.mean.value is None and "non-finite" in str(b.reason)
        )
    assert summarize([True, 1.0]).status is Status.UNDEFINED  # bool is not a measurement


# -- effect sizes ----------------------------------------------------------------------------------


def test_unpaired_effects_ground_truth() -> None:
    x, y = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]  # means 2, 5; both variances 1 -> pooled sd 1
    r = effect_sizes(x, y)
    assert eff(r, "mean_difference").value == 3.0
    assert eff(r, "relative_change").value == pytest.approx(1.5)  # 3 / |2|
    assert eff(r, "cohens_d").value == pytest.approx(3.0)
    j = 1 - 3 / (4 * 4 - 1)  # df = 4
    assert eff(r, "hedges_g").value == pytest.approx(3.0 * j)
    d = eff(r, "cohens_d")
    assert (d.numerator, d.denominator) == (3.0, 1.0) and d.assumptions and d.formula


def test_zero_denominators_are_undefined_not_zero() -> None:
    r = effect_sizes([0.0, 0.0], [1.0, 3.0])
    assert eff(r, "relative_change").status is Status.UNDEFINED
    assert eff(r, "relative_change").value is None
    assert eff(r, "cohens_d").value is not None  # pooled sd = sqrt(2/2)... y has spread
    flat = effect_sizes([1.0, 1.0], [2.0, 2.0])
    assert eff(flat, "cohens_d").status is Status.UNDEFINED and eff(flat, "cohens_d").reason
    assert eff(flat, "mean_difference").value == 1.0
    one = effect_sizes([1.0], [2.0])
    assert eff(one, "cohens_d").status is Status.INCONCLUSIVE
    assert eff(one, "mean_difference").value == 1.0


def test_paired_effects_use_keys() -> None:
    ref = {"a": 1.0, "b": 2.0, "c": 3.0}
    trt = {"c": 5.0, "a": 3.0, "b": 3.0}  # d = a:2, b:1, c:2 regardless of insertion order
    r = effect_sizes(ref, trt, Pairing.PAIRED)
    assert eff(r, "mean_difference").value == pytest.approx(5 / 3)
    sd = math.sqrt(((2 - 5 / 3) ** 2 * 2 + (1 - 5 / 3) ** 2) / 2)
    assert eff(r, "paired_standardized_difference_dz").value == pytest.approx((5 / 3) / sd)
    assert eff(r, "relative_change").value == pytest.approx((5 / 3) / 2)


def test_invalid_pairing_rejected() -> None:
    with pytest.raises(ValidationError, match="keyed"):
        effect_sizes([1.0, 2.0], [1.0, 2.0], Pairing.PAIRED)  # position is never identity
    with pytest.raises(ValidationError, match="identical keys"):
        effect_sizes({"a": 1.0}, {"b": 1.0}, Pairing.PAIRED)
    with pytest.raises(ValidationError, match="non-finite"):
        compare({"a": 1.0, "b": 2.0}, {"a": math.nan, "b": 1.0}, pairing=Pairing.PAIRED)


# -- permutation / sign-flip tests -------------------------------------------------------------------


def test_exact_unpaired_permutation_p() -> None:
    t = permutation_test([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    assert t.exact and t.permutations == 20 and t.p_value == pytest.approx(2 / 20)
    assert t.statistic == 3.0
    same = permutation_test([1.0, 2.0], [1.0, 2.0])  # identical groups: every rearrangement counts
    assert same.p_value is not None and same.p_value > 0.5


def test_exact_paired_sign_flip_p() -> None:
    ref = dict.fromkeys("abcd", 0.0)
    t = paired_sign_flip_test(ref, {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0})
    assert t.exact and t.p_value == pytest.approx(2 / 16)  # all-plus and all-minus
    zero = paired_sign_flip_test(ref, dict(ref))
    assert zero.p_value == 1.0


def test_insufficient_samples_are_inconclusive() -> None:
    assert permutation_test([1.0], [2.0, 3.0]).status is Status.INCONCLUSIVE
    assert paired_sign_flip_test({"a": 0.0}, {"a": 1.0}).status is Status.INCONCLUSIVE
    assert permutation_test([1.0], [2.0, 3.0]).p_value is None


def test_monte_carlo_is_seeded_and_close_to_exact() -> None:
    x = [float(i) for i in range(12)]
    y = [float(i) + 0.5 for i in range(12)]  # C(24,12) = 2.7M > exact limit
    a = permutation_test(x, y, permutations=4000, seed=7)
    b = permutation_test(x, y, permutations=4000, seed=7)
    c = permutation_test(x, y, permutations=4000, seed=8)
    assert not a.exact and a == b and a.p_value != c.p_value
    assert a.p_value is not None and a.p_value > 0.5  # a half-unit shift in unit-spaced data


# -- bootstrap -------------------------------------------------------------------------------------


X = [1.0, 2.0, 6.0]


def test_bca_ingredients_ground_truth() -> None:
    # for the mean, jackknife d_i = (x_i - xbar)/(n-1) so a = sum(e^3) / (6 (sum e^2)^1.5)
    jack = [(sum(X) - x) / 2 for x in X]
    assert jackknife_acceleration(jack) == pytest.approx(18 / (6 * 14**1.5))
    assert jackknife_acceleration([1.0, 1.0, 1.0]) is None
    z0, a, alpha = 0.1, 0.05, 0.025
    lo, hi = bca_levels(z0, a, alpha) or (0, 0)
    for got, z in ((lo, ND.inv_cdf(alpha)), (hi, ND.inv_cdf(1 - alpha))):
        assert got == pytest.approx(ND.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))))
    assert bca_levels(0.0, 0.0, 0.025) == pytest.approx((0.025, 0.975))  # reduces to percentile
    assert bca_levels(0.0, 1.0, 0.025) is None  # 1 - a*z_hi <= 0: does not exist


def test_bootstrap_determinism_and_recording() -> None:
    kw: dict[str, Any] = {"resamples": 500, "seed": 3}
    a = bootstrap_interval(X, **kw)
    assert a == bootstrap_interval(list(reversed(X)), **kw)  # order-independent
    assert a != bootstrap_interval(X, resamples=500, seed=4)
    assert (a.method, a.estimator, a.confidence, a.resamples, a.seed) == (
        "percentile", "mean", 0.95, 500, 3
    )  # fmt: skip
    assert a.n_samples == (3,) and a.estimate == 3.0 and a.valid_resamples == 500
    assert a.lower is not None and a.upper is not None and 1.0 <= a.lower < a.upper <= 6.0
    assert any("n=3" in w for w in a.warnings) and any("500 resamples" in w for w in a.warnings)


def test_bootstrap_golden_value_is_stable_across_python_versions() -> None:
    # pinned so a change in resampling (or in CPython's RNG stream) is noticed by CI on 3.11 and 3.12
    iv = bootstrap_interval([float(i) for i in range(1, 11)], resamples=1000, seed=42)
    assert (iv.lower, iv.upper) == pytest.approx((GOLDEN_LOWER, GOLDEN_UPPER))


GOLDEN_LOWER, GOLDEN_UPPER = 3.8, 7.2025


def test_bca_interval_and_unsupported_cases() -> None:
    x = [float(v) for v in (1, 2, 2, 3, 3, 3, 4, 4, 5, 9)]
    iv = bootstrap_interval(x, method="bca", resamples=2000, seed=1)
    pc = bootstrap_interval(x, method="percentile", resamples=2000, seed=1)
    assert iv.status is Status.DERIVED and iv.method == "bca"
    assert iv.bias_correction_z0 is not None and iv.acceleration is not None
    assert iv.acceleration > 0  # right-skewed sample
    assert iv.lower is not None and pc.lower is not None and iv.lower <= iv.estimate <= iv.upper  # type: ignore[operator]
    assert (iv.lower, iv.upper) != (pc.lower, pc.upper)
    # median of 2 points: the estimate can sit outside every resample -> reported, not approximated
    assert bootstrap_interval([1.0, 1.0, 1.0, 9.0], estimator="median", method="bca", seed=0).status in (
        Status.DERIVED, Status.UNDEFINED
    )  # fmt: skip


def test_bootstrap_degenerate_insufficient_and_bad_input() -> None:
    z = bootstrap_interval([2.0, 2.0, 2.0], resamples=50)
    assert (z.lower, z.upper, z.status) == (2.0, 2.0, Status.DERIVED)
    assert any("zero variance" in w for w in z.warnings)
    one = bootstrap_interval([2.0])
    assert one.status is Status.INCONCLUSIVE and one.lower is None
    with pytest.raises(ValidationError):
        bootstrap_interval([1.0, math.inf])
    with pytest.raises(ValidationError):
        bootstrap_interval(X, method="bogus")
    with pytest.raises(ValidationError):
        bootstrap_interval(X, confidence=1.0)


def test_paired_and_unpaired_bootstrap() -> None:
    ref = {f"k{i}": float(i) for i in range(8)}
    trt = {k: v + 2.0 for k, v in ref.items()}  # constant paired shift of 2
    p = bootstrap_interval(ref, trt, pairing=Pairing.PAIRED, resamples=200, seed=0)
    assert p.pairing == "PAIRED" and p.estimate == 2.0 and (p.lower, p.upper) == (2.0, 2.0)
    u = bootstrap_interval(list(ref.values()), list(trt.values()), resamples=200, seed=0)
    assert u.pairing == "UNPAIRED" and u.estimate == 2.0
    assert u.lower is not None and u.upper is not None and u.lower < 2.0 < u.upper  # spread remains
    b = bootstrap_interval(
        list(ref.values()), list(trt.values()), method="bca", resamples=500, seed=0
    )
    assert b.status is Status.DERIVED and b.n_samples == (8, 8)


# -- comparison ------------------------------------------------------------------------------------


def test_compare_records_everything() -> None:
    c = compare([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], resamples=300)
    assert (
        c.status is Status.DERIVED and c.pairing == "UNPAIRED" and c.declared_pairing == "UNKNOWN"
    )
    assert any("not declared" in w for w in c.warnings)
    assert c.test.p_value == pytest.approx(0.1) and c.interval.seed == 0
    assert eff(c.effects, "mean_difference").value == 3.0
    assert c.reference.mean.value == 2.0 and c.treatment.mean.value == 5.0
    assert "practical importance" in c.test.note


def test_compare_insufficient_is_inconclusive() -> None:
    c = compare([1.0], [2.0], pairing=Pairing.UNPAIRED)
    assert c.status is Status.INCONCLUSIVE and c.test.p_value is None and c.interval.lower is None


# -- multiple comparisons ----------------------------------------------------------------------------


def test_bonferroni_and_bh_ground_truth() -> None:
    p = {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005}
    bon = adjust_pvalues(p, method="BONFERRONI")
    assert bon.adjusted == pytest.approx({"a": 0.04, "b": 0.16, "c": 0.12, "d": 0.02})
    assert bon.n_hypotheses == 4 and bon.rejected == {"a": True, "b": False, "c": False, "d": True}
    bh = adjust_pvalues(p, method="BENJAMINI_HOCHBERG")
    assert bh.adjusted == pytest.approx({"a": 0.02, "b": 0.04, "c": 0.04, "d": 0.02})
    assert bh.rejected == {"a": True, "b": True, "c": True, "d": True}
    assert adjust_pvalues(p, method="NONE").adjusted == pytest.approx(p)
    assert adjust_pvalues({"a": 0.9}, method="BONFERRONI").adjusted["a"] == 0.9


def test_correction_family_and_invalid_input() -> None:
    c = adjust_pvalues({"a": 0.01, "b": None, "c": 0.02})
    assert c.n_hypotheses == 2 and c.n_total == 3 and c.excluded == ("b",)
    assert c.adjusted == pytest.approx({"a": 0.02, "c": 0.04})
    assert adjust_pvalues({}).n_hypotheses == 0
    for bad in (-0.1, 1.5, math.nan):
        with pytest.raises(ValidationError):
            adjust_pvalues({"a": bad})
    with pytest.raises(ValidationError):
        adjust_pvalues({"a": 0.1}, method="holm")
    assert adjust_pvalues({"a": 0.5}, method="BONFERRONI").adjusted["a"] == 0.5
    assert adjust_pvalues({"a": 0.6, "b": 0.7}, method="BONFERRONI").adjusted["a"] == 1.0  # capped


# -- proportions -----------------------------------------------------------------------------------


def test_wilson_ground_truth_and_edges() -> None:
    w = proportion_interval(5, 10)
    assert (w.lower, w.upper) == pytest.approx((0.2366, 0.7634), abs=5e-4)
    z = ND.inv_cdf(0.975)
    assert w.lower == pytest.approx(
        (0.5 + z * z / 20 - z * math.sqrt(0.25 / 10 + z * z / 400)) / (1 + z * z / 10)
    )
    zero = proportion_interval(0, 20)
    assert (
        zero.lower == 0.0 and zero.upper is not None and zero.upper > 0.0
    )  # never a zero-width claim
    all_ = proportion_interval(20, 20)
    assert all_.upper == 1.0 and all_.lower is not None and all_.lower < 1.0
    assert proportion_interval(0, 0).status is Status.UNDEFINED
    assert proportion_interval(1, 5).warnings
    with pytest.raises(ValidationError):
        proportion_interval(6, 5)


def test_interval_type_is_recordable() -> None:
    from experionyx.domain import to_jsonable

    doc = to_jsonable(bootstrap_interval(X, resamples=100))
    assert isinstance(doc, dict) and doc["status"] == "DERIVED" and doc["n_samples"] == [3]
    assert isinstance(bootstrap_interval(X, resamples=10), ConfidenceInterval)

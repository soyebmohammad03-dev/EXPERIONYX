"""Interaction mathematics, configuration, identity and statistics. Synthetic values here are
mathematical verification fixtures only: they are NOT experimental evidence."""

import math
from dataclasses import replace

import pytest

from experionyx.errors import ValidationError
from experionyx.interactions import calc
from experionyx.interactions.config import InteractionConfig, InteractionSpec
from experionyx.interactions.taxonomy import (
    INTERACTION_TRANSITIONS,
    Aggregation,
    EffectStatus,
    InteractionClass,
    InteractionStatus,
    Normalization,
    Pairing,
    Relation,
)


def rid(n: int) -> str:
    return f"run_{n:032x}"


# -- ground-truth fixtures --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("y", "expected"),
    [
        ((0, 1, 2, 3), 0),
        ((0, 1, 2, 6), 3),
        ((0, 1, 2, 4), 1),
        ((5, 3, 4, 2), -0.0 + (2 - 3 - 4 + 5)),
    ],
)
def test_contrast_matches_the_hand_computed_value(y, expected) -> None:  # type: ignore[no-untyped-def]
    c = calc.contrast(*y)
    assert c.interaction_contrast == pytest.approx(expected)
    assert c.interaction_contrast == pytest.approx(c.combined_effect - c.expected_additive_effect)
    assert (
        c.effect_a == y[1] - y[0] and c.effect_b == y[2] - y[0] and c.combined_effect == y[3] - y[0]
    )
    assert c.expected_additive_effect == c.effect_a + c.effect_b  # every intermediate is kept


def test_order_dependent_fixture() -> None:
    v = {"CONTROL": 0.0, "A": 1.0, "B": 2.0, "AB": 6.0, "BA": 4.0}
    s = calc.stats_of(v, Normalization.NONE)
    assert s["interaction_contrast"] == 3 and s["interaction_contrast_ba"] == 1
    assert s["order_effect"] == 2 == v["AB"] - v["BA"]  # I_AB - I_BA = YAB - YBA


def test_non_finite_inputs_are_refused_not_propagated() -> None:
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="finite"):
            calc.contrast(0, 1, 2, bad)


@pytest.mark.parametrize(
    ("norm", "y", "expected"),
    [
        (Normalization.BASELINE_MAGNITUDE, (2, 3, 4, 8), 3 / 2),  # contrast 3 / |Y0|=2
        (Normalization.EXPECTED_ADDITIVE_MAGNITUDE, (0, 1, 2, 6), 3 / 3),
        (Normalization.MAX_COMPONENT_MAGNITUDE, (0, 1, 2, 6), 3 / 2),
    ],
)
def test_each_normalization_is_explicit(norm, y, expected) -> None:  # type: ignore[no-untyped-def]
    assert calc.contrast(y[0], y[1], y[2], y[3], norm).normalized_contrast == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    ("norm", "y"),
    [
        (Normalization.BASELINE_MAGNITUDE, (0, 1, 2, 6)),
        (Normalization.EXPECTED_ADDITIVE_MAGNITUDE, (0, 1, -1, 5)),
        (Normalization.MAX_COMPONENT_MAGNITUDE, (1, 1, 1, 4)),
    ],
)
def test_zero_denominators_give_no_number_and_a_reason(norm, y) -> None:  # type: ignore[no-untyped-def]
    c = calc.contrast(y[0], y[1], y[2], y[3], norm)
    assert c.normalized_contrast is None and "zero denominator" in (c.normalization_note or "")
    assert math.isfinite(c.interaction_contrast)  # the raw contrast is unaffected


def test_negative_metric_values_are_valid() -> None:
    assert calc.contrast(-1.0, -2.0, -3.0, -6.0).interaction_contrast == pytest.approx(
        -6 + 2 + 3 - 1
    )


def test_direction_view_never_flips_the_raw_contrast_silently() -> None:
    raw = calc.contrast(
        0.9, 0.8, 0.7, 0.4
    ).interaction_contrast  # accuracy-like: 0.4-0.8-0.7+0.9 = -0.2
    o, rel = calc.deterioration_view(raw, True)
    assert (
        raw == pytest.approx(-0.2)
        and o == pytest.approx(0.2)
        and rel is Relation.SUPER_ADDITIVE_DETERIORATION
    )
    o2, rel2 = calc.deterioration_view(raw, False)  # the same number read for an error-like metric
    assert o2 == pytest.approx(-0.2) and rel2 is Relation.SUB_ADDITIVE_DETERIORATION
    assert calc.deterioration_view(0.0, True) == (0.0, Relation.ADDITIVE)
    assert calc.deterioration_view(1.0, None) == (None, Relation.UNKNOWN_DIRECTION)


def test_aggregation_and_description_keep_n_and_spread() -> None:
    assert (
        calc.aggregate([1, 2, 9], Aggregation.MEAN) == 4
        and calc.aggregate([1, 2, 9], Aggregation.MEDIAN) == 2
    )
    d = calc.describe([1.0, 2.0, 3.0])
    assert (
        d["n"] == 3 and d["std"] == 1.0 and d["standard_error"] == pytest.approx(1 / math.sqrt(3))
    )
    assert calc.describe([4.0])["std"] is None  # no invented spread from one trial
    with pytest.raises(ValueError, match="no values"):
        calc.aggregate([], Aggregation.MEAN)


# -- bootstrap ---------------------------------------------------------------------------------------


def cells(a=(1.0, 1.1, 0.9), b=(2.0, 2.1, 1.9), ab=(6.0, 6.2, 5.8), ba=None, y0=(0.0,)):  # type: ignore[no-untyped-def]
    out = {
        "CONTROL": {f"c{i}": v for i, v in enumerate(y0)},
        "A": {str(i): v for i, v in enumerate(a)},
        "B": {str(i): v for i, v in enumerate(b)},
        "AB": {str(i): v for i, v in enumerate(ab)},
    }
    if ba is not None:
        out["BA"] = {str(i): v for i, v in enumerate(ba)}
    return out


CFG = InteractionConfig(bootstrap_resamples=500)


def test_bootstrap_is_deterministic_seeded_and_reports_its_settings() -> None:
    a = calc.bootstrap(cells(), CFG, Pairing.UNPAIRED)
    assert a == calc.bootstrap(cells(), CFG, Pairing.UNPAIRED)
    assert (
        a.status is EffectStatus.COMPUTED
        and a.seed == 0
        and a.resamples == 500
        and a.confidence == 0.95
        and a.method == "percentile"
    )
    assert a.n_trials == {"CONTROL": 1, "A": 3, "B": 3, "AB": 3}
    iv = a.intervals["interaction_contrast"]
    assert (
        iv["lower"] <= 3.0 <= iv["upper"] and iv["lower"] > 0
    )  # a real, large contrast excludes zero
    other = calc.bootstrap(cells(), replace(CFG, bootstrap_seed=1), Pairing.UNPAIRED)
    assert other.intervals != a.intervals  # a different seed is a different (recorded) resampling
    assert (
        calc.bootstrap(cells(), replace(CFG, confidence=0.5), Pairing.UNPAIRED).intervals[
            "interaction_contrast"
        ]
        != iv
    )
    assert (
        calc.bootstrap(cells(), replace(CFG, bootstrap_resamples=300), Pairing.UNPAIRED).resamples
        == 300
    )


def test_too_few_trials_gives_no_interval_and_says_why() -> None:
    r = calc.bootstrap(cells(a=(1.0,), b=(2.0,), ab=(6.0,)), CFG, Pairing.UNPAIRED)
    assert (
        r.status is EffectStatus.INSUFFICIENT_DATA
        and not r.intervals
        and "min_trials" in (r.reason or "")
    )
    two = calc.bootstrap(cells(a=(1.0, 1.1), b=(2.0, 2.1), ab=(6.0, 6.1)), CFG, Pairing.UNPAIRED)
    assert two.status is EffectStatus.INSUFFICIENT_DATA  # 2 < the default min_trials of 3
    assert calc.bootstrap(cells(ab=()), CFG, Pairing.UNPAIRED).status is EffectStatus.UNDEFINED


def test_no_interaction_case_has_an_interval_that_includes_zero() -> None:
    r = calc.bootstrap(cells(ab=(3.0, 3.1, 2.9)), CFG, Pairing.UNPAIRED)  # YAB ~ YA + YB - Y0
    iv = r.intervals["interaction_contrast"]
    assert iv["lower"] < 0 < iv["upper"]
    cls, why = calc.classify(CFG, {"interaction_contrast": 0.0}, r)
    assert cls is InteractionClass.NO_EVIDENCE and why["rule"]


def test_paired_bootstrap_resamples_keys_jointly() -> None:
    # perfectly correlated trials: pairing removes the shared noise, unpairing keeps it
    shared = (0.0, 1.0, 2.0, 3.0)
    c = {
        "CONTROL": {"k": 0.0},
        "A": dict(zip("abcd", (1 + s for s in shared), strict=True)),
        "B": dict(zip("abcd", (2 + s for s in shared), strict=True)),
        "AB": dict(zip("abcd", (3 + 2 * s for s in shared), strict=True)),
    }
    cfg = replace(CFG, bootstrap_resamples=800)
    paired = calc.bootstrap(c, cfg, Pairing.PAIRED).intervals["interaction_contrast"]
    unpaired = calc.bootstrap(c, cfg, Pairing.UNPAIRED).intervals["interaction_contrast"]
    assert paired["upper"] - paired["lower"] < unpaired["upper"] - unpaired["lower"]
    assert calc.bootstrap(c, cfg, Pairing.PAIRED).pairing == "PAIRED"


def test_unknown_pairing_is_analyzed_unpaired_and_warned() -> None:
    r = calc.bootstrap(cells(), CFG, Pairing.UNKNOWN)
    assert r.pairing == "UNPAIRED" and any("not declared" in w for w in r.warnings)
    assert any("control has one run" in w for w in r.warnings)


def test_order_effect_gets_its_own_interval() -> None:
    r = calc.bootstrap(cells(ba=(4.0, 4.1, 3.9)), CFG, Pairing.UNPAIRED)
    assert {"order_effect", "interaction_contrast_ba"} <= set(r.intervals)
    cls, _ = calc.classify(
        CFG, calc.stats_of({"CONTROL": 0, "A": 1, "B": 2, "AB": 6, "BA": 4}, Normalization.NONE), r
    )
    assert cls is InteractionClass.ORDER_DEPENDENT_INTERACTION
    same = calc.bootstrap(cells(ba=(6.0, 6.2, 5.8)), CFG, Pairing.UNPAIRED)
    cls2, _ = calc.classify(
        CFG,
        calc.stats_of({"CONTROL": 0, "A": 1, "B": 2, "AB": 6, "BA": 6}, Normalization.NONE),
        same,
    )
    assert (
        cls2 is InteractionClass.OBSERVED_INTERACTION
    )  # interaction, but no evidence of order dependence


def test_an_order_effect_alone_does_not_justify_an_interaction_label() -> None:
    c = cells(a=(0.0, 1.0, 2.0), b=(1.0, 2.0, 3.0), ab=(3.5, 3.6, 3.7), ba=(2.5, 2.6, 2.7))
    r = calc.bootstrap(c, replace(CFG, bootstrap_resamples=800), Pairing.UNPAIRED)
    point = calc.stats_of(
        {"CONTROL": 0.0, "A": 1.0, "B": 2.0, "AB": 3.6, "BA": 2.6}, Normalization.NONE
    )
    cls, why = calc.classify(replace(CFG, bootstrap_resamples=800), point, r)
    assert (
        why["order_interval_excludes_zero"] is True
        and why["contrast_ab_interval_excludes_zero"] is False
    )
    assert cls is InteractionClass.POSSIBLE_INTERACTION and "no interaction label" in str(
        why["rule"]
    )


def test_a_zero_magnitude_threshold_never_creates_evidence_by_itself() -> None:
    wide = calc.bootstrap(
        cells(a=(0.0, 1.0, 2.0), b=(1.0, 2.0, 3.0), ab=(2.0, 3.0, 4.0)), CFG, Pairing.UNPAIRED
    )
    point = calc.stats_of({"CONTROL": 0.0, "A": 1.0, "B": 2.0, "AB": 3.0}, Normalization.NONE)
    point["interaction_contrast"] = 0.3  # nonzero point contrast, interval includes zero
    assert (
        calc.classify(CFG, point, wide)[0] is InteractionClass.NO_EVIDENCE
    )  # min_abs_contrast=0 = no criterion
    assert (
        calc.classify(replace(CFG, min_abs_contrast=0.1), point, wide)[0]
        is InteractionClass.POSSIBLE_INTERACTION
    )


def test_classification_rules_and_thresholds_are_explicit() -> None:
    strong = calc.bootstrap(cells(), CFG, Pairing.UNPAIRED)
    point = {"interaction_contrast": 3.0}
    assert calc.classify(CFG, point, strong)[0] is InteractionClass.OBSERVED_INTERACTION
    strict = replace(
        CFG, min_abs_contrast=10.0
    )  # interval excludes zero, magnitude threshold not met
    assert calc.classify(strict, point, strong)[0] is InteractionClass.POSSIBLE_INTERACTION
    assert (
        calc.classify(CFG, {"interaction_contrast": None}, strong)[0] is InteractionClass.UNDEFINED
    )
    few = calc.bootstrap(cells(a=(1.0,), b=(2.0,), ab=(6.0,)), CFG, Pairing.UNPAIRED)
    assert calc.classify(CFG, point, few)[0] is InteractionClass.INCONCLUSIVE
    _, why = calc.classify(CFG, point, strong)
    assert (
        why["min_abs_contrast"] == 0.0 and why["confidence"] == 0.95
    )  # the rule inputs are recorded


# -- config, spec identity, lifecycle ----------------------------------------------------------------------


def test_config_is_strict_hashed_and_roundtrips() -> None:
    c = InteractionConfig()
    assert (
        InteractionConfig.from_dict(c.to_dict()) == c
        and c.config_hash == InteractionConfig().config_hash
    )
    for bad in (
        dict(confidence=1.0),
        dict(min_trials=1),
        dict(bootstrap_resamples=0),
        dict(min_sign_fraction=0.2),
        dict(min_abs_contrast=math.nan),
        dict(metrics=("b", "a")),
    ):
        with pytest.raises(ValidationError):
            InteractionConfig(**bad)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        InteractionConfig.from_dict({**c.to_dict(), "extra": 1})


def spec(**kw):  # type: ignore[no-untyped-def]
    base = dict(control=(rid(1),), a=(rid(2), rid(3)), b=(rid(4), rid(5)), ab=(rid(6), rid(7)))
    base.update(kw)
    return InteractionSpec(**base)


def test_spec_identity_is_deterministic_and_sensitive_to_every_meaningful_field() -> None:
    s = spec()
    assert s.spec_id == spec().spec_id == InteractionSpec.from_dict(s.to_dict()).spec_id
    assert (
        spec(a=(rid(3), rid(2))).spec_id == s.spec_id
    )  # the order of runs inside a cell is not identity
    variants = [
        spec(a=(rid(2), rid(9))),
        spec(control=(rid(8),)),
        spec(config=InteractionConfig(bootstrap_seed=1)),
        spec(config=InteractionConfig(normalization=Normalization.BASELINE_MAGNITUDE)),
        spec(config=InteractionConfig(bootstrap_resamples=99)),
        spec(config=InteractionConfig(confidence=0.9)),
        spec(discovery_run_id=rid(50)),
        spec(b=(rid(4),)),
    ]
    ids = {x.spec_id for x in variants} | {s.spec_id}
    assert len(ids) == len(variants) + 1


def test_spec_validation() -> None:
    with pytest.raises(ValidationError):
        spec(a=())
    with pytest.raises(ValidationError):
        spec(ab=(rid(2),))  # a run in two cells
    with pytest.raises(ValidationError):
        spec(ba=(rid(20),))  # BA without order_analysis
    with pytest.raises(ValidationError):
        spec(config=InteractionConfig(order_analysis=True))  # order analysis needs BA
    with pytest.raises(ValidationError):
        InteractionSpec.from_dict({**spec().to_dict(), "surprise": 1})
    assert spec(ba=(rid(20),), config=InteractionConfig(order_analysis=True)).cells()["BA"] == (
        rid(20),
    )


def test_lifecycle_table_forbids_shortcuts() -> None:
    S = InteractionStatus
    assert set(INTERACTION_TRANSITIONS) == set(S)
    assert (
        S.CONFIRMED_BY_REVIEW
        not in INTERACTION_TRANSITIONS[S.DISCOVERED] | INTERACTION_TRANSITIONS[S.SUPPORTED]
    )
    assert INTERACTION_TRANSITIONS[S.CONFIRMED_BY_REVIEW] == {S.DEPRECATED}
    assert not INTERACTION_TRANSITIONS[S.REJECTED] and not INTERACTION_TRANSITIONS[S.DEPRECATED]

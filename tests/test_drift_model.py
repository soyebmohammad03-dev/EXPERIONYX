"""Drift domain model and pure statistics: window identity and normalization, boundary semantics,
rolling plans, spec validation, and every measure against a value computed by hand. No workspace."""

import json
import math
from typing import Any

import pytest

from experionyx.domain import to_jsonable
from experionyx.drift import measures
from experionyx.drift.engine import correct
from experionyx.drift.entities import DriftWindow
from experionyx.drift.results import Evidence
from experionyx.drift.spec import (
    DriftConfig,
    FeatureDecl,
    Ordering,
    Role,
    Rolling,
    ShiftSpec,
    TemporalWindow,
    overlaps,
)
from experionyx.errors import ValidationError
from experionyx.slices.spec import SliceSpec, eq
from experionyx.stats import core as st

RUN = "run_" + "a" * 32
REF, CMP = Role.REFERENCE, Role.COMPARISON
CFG = DriftConfig(min_samples=5, resamples=100, permutations=200)


def win(start: float, end: float, role: Role = CMP, **kw: Any) -> TemporalWindow:
    return TemporalWindow(role, start, end, **kw)


def spec(**over: Any) -> ShiftSpec:
    body: dict[str, Any] = {"baseline_run": RUN, "ordering": Ordering("index"), "reference": win(0, 40, REF), "comparisons": (win(40, 80),)}  # fmt: skip
    body.update(over)
    return ShiftSpec(**body)


# -- window identity and normalization --------------------------------------------------------------


def test_window_identity_is_deterministic_and_normalized() -> None:
    a = win(0, 40)
    assert a.window_id("index") == win(0.0, 40.0).window_id("index")  # 40.0 == 40
    assert a.window_id("index") == win(0, 40, name="early").window_id("index")  # a name is a label
    assert a.window_id("index").startswith("twn_") and len(a.window_id("index")) == 36
    for other in (
        win(0, 41), win(1, 40), win(0, 40, start_inclusive=False), win(0, 40, end_inclusive=True),
        win(0, 40, REF),
    ):  # fmt: skip
        assert other.window_id("index") != a.window_id("index")
    assert a.window_id("feature:ts") != a.window_id("index")  # the ordering is part of the identity


def test_registered_window_id_equals_the_domain_window_id_and_round_trips() -> None:
    from datetime import UTC, datetime

    w = win(0, 40, REF, end_inclusive=True, name="baseline")
    rec = DriftWindow.of(w, "feature:ts", datetime(2026, 1, 1, tzinfo=UTC))
    assert rec.id == w.window_id("feature:ts")
    assert rec.window() == w
    assert DriftWindow.from_dict(rec.to_dict()) == rec
    with pytest.raises(ValidationError, match="role"):
        DriftWindow(rec.name, rec.ordering_field, "BOGUS", 0, 1, True, False, 1, rec.created_at)
    with pytest.raises(ValidationError, match="ordering field"):
        DriftWindow(rec.name, "row_number", "REFERENCE", 0, 1, True, False, 1, rec.created_at)


@pytest.mark.parametrize(
    ("start", "end", "kw"),
    [
        (float("nan"), 1, {}), (0, float("inf"), {}), (True, 5, {}), ("0", 5, {}),
        (5, 5, {}), (5, 5, {"start_inclusive": True, "end_inclusive": False}), (6, 5, {}),
    ],
)  # fmt: skip
def test_invalid_windows_are_refused(start: Any, end: Any, kw: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        TemporalWindow(CMP, start, end, **kw)


def test_a_single_point_window_needs_both_bounds_inclusive() -> None:
    w = win(5, 5, start_inclusive=True, end_inclusive=True)
    assert w.contains(5) and not w.contains(5.0001)


def test_membership_boundaries_are_exact_for_every_inclusivity() -> None:
    for si in (True, False):
        for ei in (True, False):
            w = win(10, 20, start_inclusive=si, end_inclusive=ei)
            assert w.contains(10) is si and w.contains(20) is ei
            assert w.contains(15) and not w.contains(9.999999) and not w.contains(20.000001)


def test_overlap_semantics_at_touching_boundaries() -> None:
    assert not overlaps(win(0, 5), win(5, 10))  # [0,5) and [5,10)
    assert overlaps(win(0, 5, end_inclusive=True), win(5, 10))  # [0,5] and [5,10)
    assert not overlaps(win(0, 5, end_inclusive=True), win(5, 10, start_inclusive=False))
    assert overlaps(win(0, 6), win(5, 10)) and overlaps(win(2, 3), win(0, 10))  # containment
    assert not overlaps(win(10, 20), win(0, 5)) and overlaps(win(0, 5), win(0, 5))


def test_ordering_must_be_explicit_and_well_formed() -> None:
    for good in ("index", "feature:ts", "feature:3"):
        assert Ordering(good).field == good
    for bad in ("", "row", "position", "feature:", "feature:  ", "Index"):
        with pytest.raises(ValidationError, match="ordering field"):
            Ordering(bad)
    with pytest.raises(ValidationError, match=r"missing \['ordering'\]"):
        ShiftSpec.from_dict({"baseline_run": RUN, "reference": {"start": 0, "end": 1}, "comparisons": [{"start": 1, "end": 2}]})  # fmt: skip


# -- rolling and expanding plans ------------------------------------------------------------------------


def test_rolling_windows_are_exact_and_partial_windows_are_listed_not_dropped() -> None:
    s = ShiftSpec(RUN, Ordering("index"), win(0, 20, REF), (), Rolling(20, 110, 20, 20))
    plan = s.plan()
    assert [p.comparison.describe() for p in plan.pairs] == ["[20, 40)", "[40, 60)", "[60, 80)", "[80, 100)"]  # fmt: skip
    assert all(
        p.reference == win(0, 20, REF) for p in plan.pairs
    )  # FIXED: one reference throughout
    assert [(k.window.describe(), k.reason) for k in plan.skipped] == [("[100, 120)", "PARTIAL_WINDOW")]  # fmt: skip
    assert len(plan.windows()) == 5


def test_previous_and_expanding_references() -> None:
    prev = ShiftSpec(RUN, Ordering("index"), None, (), Rolling(0, 60, 20, 20, "PREVIOUS")).plan()
    assert [p.key for p in prev.pairs] == ["[0, 20) vs [20, 40)", "[20, 40) vs [40, 60)"]
    assert [(s.window.describe(), s.reason) for s in prev.skipped] == [("[0, 20)", "NO_REFERENCE")]
    assert all(p.reference.role is REF and p.comparison.role is CMP for p in prev.pairs)
    exp = ShiftSpec(RUN, Ordering("index"), None, (), Rolling(0, 60, 20, 20, "EXPANDING")).plan()
    assert [p.key for p in exp.pairs] == ["[0, 20) vs [20, 40)", "[0, 40) vs [40, 60)"]
    assert [s.reason for s in exp.skipped] == ["NO_REFERENCE"]


def test_overlapping_windows_are_refused_before_anything_runs() -> None:
    with pytest.raises(ValidationError, match="step >= width"):
        Rolling(0, 100, 20, 10, "PREVIOUS")  # consecutive windows would share samples
    with pytest.raises(ValidationError, match="overlaps comparison"):
        spec(reference=win(0, 50, REF), comparisons=(win(40, 80),))
    with pytest.raises(ValidationError, match="overlaps rolling window"):
        ShiftSpec(RUN, Ordering("index"), win(0, 50, REF), (), Rolling(40, 100, 20, 20))
    spec(
        reference=win(0, 40, REF, end_inclusive=True),
        comparisons=(win(40, 80, start_inclusive=False),),
    )  # touching but disjoint
    with pytest.raises(ValidationError, match="overlaps"):
        spec(reference=win(0, 40, REF, end_inclusive=True), comparisons=(win(40, 80),))


def test_rolling_rejects_degenerate_or_huge_plans() -> None:
    for bad in ((0, 10, 0, 5), (0, 10, 5, 0), (10, 10, 5, 5), (0, 10, -1, 5)):
        with pytest.raises(ValidationError):
            Rolling(*bad)
    with pytest.raises(ValidationError, match="at most"):
        Rolling(0, 100_000, 1, 1)
    with pytest.raises(ValidationError, match="reference"):
        Rolling(0, 10, 1, 1, "SLIDING")


# -- spec validation and identity -------------------------------------------------------------------------


def test_spec_needs_exactly_one_window_mode_and_the_right_references() -> None:
    with pytest.raises(ValidationError, match="not both and not neither"):
        spec(comparisons=())
    with pytest.raises(ValidationError, match="not both and not neither"):
        spec(rolling=Rolling(40, 80, 20, 20))
    with pytest.raises(ValidationError, match="REFERENCE window is required"):
        spec(reference=None)
    with pytest.raises(ValidationError, match="REFERENCE window is required"):
        spec(reference=win(0, 40, CMP))
    with pytest.raises(ValidationError, match="role"):
        spec(comparisons=(win(40, 80, REF),))
    with pytest.raises(ValidationError, match="defines its own references"):
        ShiftSpec(RUN, Ordering("index"), win(0, 5, REF), (), Rolling(0, 60, 20, 20, "PREVIOUS"))
    with pytest.raises(ValidationError, match="twice"):
        spec(comparisons=(win(40, 80), win(40, 80, name="again")))
    with pytest.raises(ValidationError, match="run"):
        spec(baseline_run="nope")


def test_features_types_and_dimensions_are_validated() -> None:
    with pytest.raises(ValidationError, match="unsupported feature type 'TEXT'"):
        FeatureDecl("comment", "TEXT")
    with pytest.raises(ValidationError, match="unsupported feature type"):
        FeatureDecl("emb", "VECTOR")
    with pytest.raises(ValidationError, match="declared twice"):
        spec(features=(FeatureDecl("x", "NUMERIC"), FeatureDecl("x", "CATEGORICAL")))
    with pytest.raises(ValidationError, match="covariate dimension needs declared features"):
        spec(config=DriftConfig(dimensions=("covariate", "label")))
    with pytest.raises(ValidationError, match="declared features need the covariate"):
        spec(features=(FeatureDecl("x", "NUMERIC"),), config=DriftConfig(dimensions=("label",)))
    assert spec().config.dimensions == ("label", "performance", "prediction")
    assert spec(features=(FeatureDecl("x", "NUMERIC"),)).config.dimensions == ("covariate", "label", "performance", "prediction")  # fmt: skip


@pytest.mark.parametrize(
    "kw",
    [
        {"min_samples": 1}, {"resamples": 0}, {"permutations": 0}, {"seed": -1},
        {"confidence": 1.0}, {"confidence": 0.0}, {"alpha": 0.0}, {"alpha": 1.5},
        {"method": "magic"}, {"correction": "HOLM"}, {"correction_scope": "GLOBAL"},
        {"metrics": ("b", "a")}, {"dimensions": ("label", "bogus")}, {"dimensions": ()},
        {"dimensions": ("prediction", "label")},
    ],
)  # fmt: skip
def test_invalid_thresholds_and_configuration_are_refused(kw: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        DriftConfig(**kw)


def test_spec_identity_ignores_labels_and_order_but_not_meaning() -> None:
    a = spec(features=(FeatureDecl("a", "NUMERIC"), FeatureDecl("b", "CATEGORICAL")))
    b = spec(features=(FeatureDecl("b", "CATEGORICAL"), FeatureDecl("a", "NUMERIC")), reference=win(0, 40, REF, name="baseline"))  # fmt: skip
    assert a.spec_id == b.spec_id and a.spec_id.startswith("dsp_")
    for changed in (
        spec(features=a.features, config=DriftConfig(seed=1)),
        spec(features=a.features, config=DriftConfig(correction="BONFERRONI")),
        spec(features=a.features, comparisons=(win(40, 81),)),
        spec(features=a.features, ordering=Ordering("feature:ts")),
        spec(features=a.features, ordering=Ordering("index", unique=True)),
        spec(features=(FeatureDecl("a", "NUMERIC"), FeatureDecl("b", "BOOLEAN"))),
        spec(features=a.features, slices=(SliceSpec("s", eq("target", 1)),)),
        spec(features=a.features, failure_modes=("fmd_" + "b" * 32,)),
        spec(features=a.features, baseline_run="run_" + "c" * 32),
    ):  # fmt: skip
        assert changed.spec_id != a.spec_id
    assert spec(slices=(SliceSpec("first", eq("target", 1)),)).spec_id == spec(slices=(SliceSpec("first", eq("target", 1), "described"),)).spec_id  # fmt: skip


def test_spec_round_trips_through_json_and_rejects_unknown_or_missing_keys() -> None:
    s = spec(features=(FeatureDecl("a", "NUMERIC"),), slices=(SliceSpec("s", eq("target", 1)),), config=DriftConfig(seed=3, correction="BENJAMINI_HOCHBERG"))  # fmt: skip
    again = ShiftSpec.from_dict(json.loads(json.dumps(s.to_dict())))
    assert again == s and again.spec_id == s.spec_id
    for bad in ({**s.to_dict(), "extra": 1}, {**s.to_dict(), "drift_schema": 2}, {**s.to_dict(), "analysis_version": "9"}, {k: v for k, v in s.to_dict().items() if k != "baseline_run"}):  # fmt: skip
        with pytest.raises(ValidationError):
            ShiftSpec.from_dict(bad)
    with pytest.raises(ValidationError, match="cannot declare role"):
        ShiftSpec.from_dict({**s.to_dict(), "comparisons": [{"role": "REFERENCE", "start": 40, "end": 80}]})  # fmt: skip
    with pytest.raises(ValidationError, match="unknown drift setting"):
        DriftConfig.from_dict({"bogus": 1})


def test_duplicate_slices_are_refused() -> None:
    with pytest.raises(ValidationError, match="unique names"):
        spec(slices=(SliceSpec("s", eq("target", 1)), SliceSpec("s", eq("target", 0))))
    with pytest.raises(ValidationError, match="distinct definitions"):
        spec(slices=(SliceSpec("s", eq("target", 1)), SliceSpec("t", eq("target", 1))))


# -- numeric measures, against values computed by hand -----------------------------------------------------


def test_identical_windows_show_no_distance_and_no_evidence_of_difference() -> None:
    x = [float(i) for i in range(20)]
    r = measures.shift("COVARIATE", "f", "NUMERIC", x, x, CFG)
    assert r.status is Evidence.DERIVED
    assert r.measures["ks_statistic"] == 0.0 and r.measures["wasserstein_1"] == 0.0
    assert r.test is not None and r.test["p_value"] == 1.0
    loc = r.measures["location"]
    assert loc["mean_difference_interval"]["estimate"] == 0.0


def test_known_shift_matches_closed_forms() -> None:
    assert measures.ks_statistic([1, 2, 3], [4, 5, 6]) == 1.0
    assert measures.wasserstein_1([1, 2, 3], [4, 5, 6]) == pytest.approx(3.0, abs=1e-12)
    x = [float(i) for i in range(20)]
    y = [v + 5 for v in x]
    assert measures.wasserstein_1(x, y) == pytest.approx(
        5.0, abs=1e-12
    )  # a pure shift moves every point by 5
    assert measures.ks_statistic(x, y) == pytest.approx(
        0.25, abs=1e-12
    )  # 5 of 20 steps of the ECDF
    # ties: half the mass is equal, the rest differs by 1
    assert measures.ks_statistic([0, 0, 1, 1], [0, 0, 2, 2]) == pytest.approx(0.5)
    assert measures.wasserstein_1([0, 0, 1, 1], [0, 0, 2, 2]) == pytest.approx(0.5)
    r = measures.shift("COVARIATE", "f", "NUMERIC", x, y, CFG)
    sd = st.summarize(x).std.value
    assert sd is not None and r.measures["wasserstein_1_scaled"] == pytest.approx(
        5.0 / sd, rel=1e-12
    )
    assert r.measures["location"]["mean_difference_interval"]["estimate"] == pytest.approx(5.0)
    assert r.measures["direction"] == "difference is comparison minus reference"
    assert {e["name"] for e in r.measures["location"]["effect_sizes"]} >= {
        "cohens_d"
    } or r.measures["location"]["effect_sizes"]


def test_exact_permutation_p_value_on_a_tiny_case_is_the_enumerated_share() -> None:
    r = measures.shift(
        "COVARIATE", "f", "NUMERIC", [1.0, 2.0], [3.0, 4.0], DriftConfig(min_samples=2)
    )
    assert r.test is not None and r.test["exact"] is True and r.test["seed"] is None
    assert r.test["permutations"] == 6 and r.test["p_value"] == pytest.approx(
        2 / 6
    )  # only the 2 fully separated splits reach KS = 1


def test_monte_carlo_is_seeded_and_reproducible() -> None:
    x = [float(i) for i in range(40)]
    y = [float(i) + 3 for i in range(40)]
    cfg = DriftConfig(min_samples=5, permutations=300, resamples=100, seed=7)
    a = measures.shift("COVARIATE", "f", "NUMERIC", x, y, cfg)
    b = measures.shift(
        "COVARIATE", "f", "NUMERIC", list(reversed(x)), list(reversed(y)), cfg
    )  # input order is irrelevant
    assert a.test is not None and a.test["exact"] is False and a.test["seed"] == 7
    assert to_jsonable(a) == to_jsonable(measures.shift("COVARIATE", "f", "NUMERIC", x, y, cfg))
    assert a.test["p_value"] == b.test["p_value"]  # type: ignore[index]
    assert a.measures["location"] == b.measures["location"]
    other = measures.shift("COVARIATE", "f", "NUMERIC", x, y, DriftConfig(min_samples=5, permutations=300, resamples=100, seed=8))  # fmt: skip
    assert other.test is not None and other.test["seed"] == 8


def test_constant_data_is_handled_without_inventing_statistics() -> None:
    same = measures.shift("COVARIATE", "c", "NUMERIC", [3.0] * 20, [3.0] * 20, CFG)
    assert same.status is Evidence.DERIVED and same.measures["ks_statistic"] == 0.0
    assert same.measures["wasserstein_1_scaled"] is None and "zero" in same.measures["wasserstein_1_scaled_reason"]  # fmt: skip
    assert len(same.warnings) == 2 and all("constant" in w for w in same.warnings)
    moved = measures.shift("COVARIATE", "c", "NUMERIC", [3.0] * 20, [5.0] * 20, CFG)
    assert moved.measures["ks_statistic"] == 1.0 and moved.measures["wasserstein_1"] == pytest.approx(2.0)  # fmt: skip
    assert moved.measures["wasserstein_1_scaled"] is None  # the reference has no spread to scale by
    assert moved.test is not None and moved.test["p_value"] == pytest.approx(
        1 / 201
    )  # the smallest Monte Carlo p with 200 permutations


def test_every_value_is_accounted_for_missing_nonfinite_and_invalid() -> None:
    inf, nan = float("inf"), float("nan")
    ref = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, None, nan, -inf, "7", True]
    r = measures.shift("COVARIATE", "f", "NUMERIC", ref, [float(i) for i in range(10)], CFG)
    c = r.reference
    assert (c.n_total, c.n_valid, c.n_missing, c.n_nonfinite, c.n_invalid) == (11, 6, 2, 1, 2)
    assert c.n_total == c.n_valid + c.n_missing + c.n_nonfinite + c.n_invalid
    assert any("5 of 11" in w for w in r.warnings)  # what was excluded is reported, not hidden
    assert r.status is Evidence.DERIVED  # 6 valid >= min_samples=5
    v, _ = measures.clean([1, 2.5, True, "x", None, nan, inf], "NUMERIC")
    assert v == [1.0, 2.5]  # a bool is not a number and a string is not coerced


def test_nothing_valid_means_undefined_not_zero() -> None:
    r = measures.shift("COVARIATE", "f", "NUMERIC", [None, float("nan")] * 5, [1.0] * 10, CFG)
    assert r.status is Evidence.UNDEFINED and not r.measures and r.test is None
    assert "no valid values" in (r.reason or "") and r.reference.n_missing == 10
    only_inf = measures.shift("COVARIATE", "f", "NUMERIC", [float("inf")] * 6, [1.0] * 6, CFG)
    assert only_inf.status is Evidence.UNDEFINED and only_inf.reference.n_nonfinite == 6


def test_too_few_samples_withholds_inference_but_keeps_the_descriptive_values() -> None:
    r = measures.shift("COVARIATE", "f", "NUMERIC", [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], CFG)
    assert r.status is Evidence.INSUFFICIENT_EVIDENCE and r.test is None
    assert r.measures["ks_statistic"] == 1.0 and "location" not in r.measures
    assert "min_samples=5" in (r.reason or "")
    exact = measures.shift("COVARIATE", "f", "NUMERIC", [1.0] * 5, [2.0] * 4, CFG)
    assert exact.status is Evidence.INSUFFICIENT_EVIDENCE  # one window is short: the pair is short


# -- categorical and boolean measures -----------------------------------------------------------------------


def test_jensen_shannon_and_total_variation_match_hand_computation() -> None:
    assert measures.jensen_shannon([3, 7], [3, 7]) == 0.0
    assert measures.jensen_shannon([1, 0], [0, 1]) == pytest.approx(1.0)  # disjoint support: 1 bit
    assert measures.total_variation([1, 0], [0, 1]) == pytest.approx(1.0)
    assert measures.jensen_shannon([1, 1], [2, 0]) == pytest.approx(
        0.5 * (0.5 * math.log2(0.5 / 0.75) + 0.5 * math.log2(0.5 / 0.25))
        + 0.5 * math.log2(1 / 0.75)
    )
    assert measures.jensen_shannon([1, 1], [2, 0]) == pytest.approx(0.3112781244591328, abs=1e-12)
    assert measures.total_variation([1, 1], [2, 0]) == pytest.approx(0.5)
    assert measures.jensen_shannon([1, 3], [3, 1]) == measures.jensen_shannon(
        [3, 1], [1, 3]
    )  # symmetric


def test_categorical_proportions_use_phase_10_wilson_intervals() -> None:
    ref = ["a"] * 15 + ["b"] * 5
    cmp = ["a"] * 5 + ["b"] * 15
    r = measures.shift("LABEL", "target", "CATEGORICAL", ref, cmp, CFG)
    rows = {c["category"]: c for c in r.measures["categories"]}
    a = rows['"a"']
    assert (a["n_reference"], a["n_comparison"]) == (15, 5)
    assert (a["p_reference"], a["p_comparison"], a["difference"]) == (0.75, 0.25, -0.5)
    w = st.proportion_interval(15, 20, 0.95)
    assert (a["interval_reference"]["lower"], a["interval_reference"]["upper"]) == (
        w.lower,
        w.upper,
    )
    assert r.measures["total_variation_distance"] == pytest.approx(0.5)
    assert (
        r.test is not None
        and r.test["name"] == "jensen_shannon_permutation"
        and r.test["p_value"] < 0.05
    )
    assert r.status is Evidence.DERIVED


def test_unseen_and_vanished_categories_are_reported() -> None:
    r = measures.shift(
        "COVARIATE", "site", "CATEGORICAL", ["a"] * 8 + ["b"] * 4, ["a"] * 8 + ["c"] * 4, CFG
    )
    assert r.measures["only_in_comparison"] == ['"c"'] and r.measures["only_in_reference"] == [
        '"b"'
    ]
    assert any("unseen in the reference" in w for w in r.warnings) and any(
        "absent from the comparison" in w for w in r.warnings
    )
    # 1, 1.0 and '1' are not silently merged across types (1.0 is the same category as 1)
    v, _ = measures.clean([1, 1.0, "1", True], "CATEGORICAL")
    assert v[0] == v[1] and len(set(v)) == 3


def test_boolean_features_and_their_invalid_values() -> None:
    v, c = measures.clean([True, False, 1, 0, 1.0, 2, "yes", None], "BOOLEAN")
    assert v == ["true", "false", "true", "false", "true"] and (c.n_invalid, c.n_missing) == (2, 1)
    r = measures.shift(
        "COVARIATE", "flag", "BOOLEAN", [True] * 6 + [False] * 6, [True] * 10 + [False] * 2, CFG
    )
    assert r.kind == "BOOLEAN" and r.status is Evidence.DERIVED
    assert {c["category"] for c in r.measures["categories"]} == {"true", "false"}


def test_high_cardinality_categoricals_are_refused_as_undefined() -> None:
    ids = [f"id{i}" for i in range(measures.MAX_CATEGORIES + 5)]
    r = measures.shift("COVARIATE", "uid", "CATEGORICAL", ids, ids, CFG)
    assert r.status is Evidence.UNDEFINED and "distinct categories" in (r.reason or "")


def all_keys(x: Any) -> set[str]:
    if isinstance(x, dict):
        return set(x) | {k for v in x.values() for k in all_keys(v)}
    return {k for v in x for k in all_keys(v)} if isinstance(x, list) else set()


def test_no_result_carries_a_verdict_a_score_or_a_causal_claim() -> None:
    r = measures.shift("COVARIATE", "f", "NUMERIC", [float(i) for i in range(20)], [float(i) + 9 for i in range(20)], CFG)  # fmt: skip
    keys = all_keys(r.to_dict())
    for banned in ("drift_detected", "drift_score", "score", "verdict", "is_drift", "caused_by", "concept_drift", "severity", "rank"):  # fmt: skip
        assert banned not in keys, banned
    assert (
        "does not say why" in r.note and "concept drift" in r.note
    )  # the limits are stated, not the conclusions
    assert r.test is not None and "not a verdict" in r.test["note"] and "cause" in r.test["note"]


# -- multiple comparisons ---------------------------------------------------------------------------------------


def fake(p: float | None, subject: str) -> dict[str, Any]:
    return {"subject": subject, "measures": {}, "test": None if p is None else {"p_value": p}}


def test_correction_uses_phase_10_and_records_the_family() -> None:
    entries = [("A", "POPULATION", "covariate", fake(p, f"f{i}")) for i, p in enumerate((0.01, 0.02, 0.04, None))]  # fmt: skip
    for method, expect in (
        ("NONE", [0.01, 0.02, 0.04]),
        ("BONFERRONI", [0.03, 0.06, 0.12]),
        ("BENJAMINI_HOCHBERG", [0.03, 0.03, 0.04]),
    ):
        docs = [dict(e[3]) for e in entries]
        ents = [(e[0], e[1], e[2], d) for e, d in zip(entries, docs, strict=True)]
        fams = correct(ents, spec(config=DriftConfig(correction=method)))
        ((_, fam),) = fams.items()
        assert fam["method"] == method and fam["n_hypotheses"] == 3 and fam["n_total"] == 4
        assert [docs[i]["multiplicity"]["adjusted_p"] for i in range(3)] == pytest.approx(expect)
        assert docs[3]["multiplicity"]["raw_p"] is None and docs[3]["multiplicity"]["adjusted_p"] is None  # fmt: skip
        assert fam["members"] and "definition" in fam and len(fam["excluded"]) == 1
        m = docs[0]["multiplicity"]
        assert m["raw_p"] == 0.01 and m["method"] == method and m["alpha"] == 0.05
        assert (m["adjusted_p_below_alpha"] is None) == (
            method == "NONE"
        )  # no correction, no rejection claim


def test_family_scope_decides_what_is_corrected_together() -> None:
    def build(scope: str) -> dict[str, Any]:
        ents = [(pair, "POPULATION", "covariate", fake(0.02, f"f{j}")) for pair in ("P1", "P2") for j in range(2)]  # fmt: skip
        return correct(ents, spec(config=DriftConfig(correction="BONFERRONI", correction_scope=scope)))  # fmt: skip

    per_pair, across = build("PER_WINDOW_PAIR"), build("ACROSS_WINDOWS")
    assert len(per_pair) == 2 and all(f["n_hypotheses"] == 2 for f in per_pair.values())
    assert len(across) == 1 and next(iter(across.values()))["n_hypotheses"] == 4
    assert next(iter(across.values()))["adjusted"] and max(next(iter(across.values()))["adjusted"].values()) == pytest.approx(0.08)  # fmt: skip
    assert max(next(iter(per_pair.values()))["adjusted"].values()) == pytest.approx(0.04)


def test_raw_evidence_and_magnitude_stay_separate() -> None:
    r = measures.shift("COVARIATE", "f", "NUMERIC", [float(i) for i in range(30)], [float(i) + 0.5 for i in range(30)], CFG)  # fmt: skip
    d = r.to_dict()
    assert "ks_statistic" in d["measures"] and "p_value" in d["test"] and "p_value" not in d["measures"]  # fmt: skip
    assert d["multiplicity"] is None  # set only when a family is formed

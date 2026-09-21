"""Calibration domain and mathematics, pure (no registry). Expected numbers are hand-calculated in the
comments or computed with independently written formulas. Engineering validation of the machinery."""

import math
import random
from typing import Any

import pytest

import experionyx.calibration.measures as m
from experionyx.calibration.engine import split_ids
from experionyx.calibration.spec import CalibrationSpec, Fit, Statistics
from experionyx.errors import ValidationError

RUN = "run_" + "a" * 32
RUN2 = "run_" + "b" * 32
SXA = "sxa_" + "c" * 32


def spec(**over: Any) -> CalibrationSpec:
    d: dict[str, Any] = {"baseline_run": RUN, "prediction_source": "PREDICT_PROBA"}
    d.update(over)
    return CalibrationSpec.from_dict(d)


# -- identity, normalization, validation ------------------------------------------------------------


def test_identity_is_deterministic_and_key_order_independent() -> None:
    a = spec(binning={"strategy": "UNIFORM", "n_bins": 10}, objects=["CLASSWISE", "TOP_LABEL"])
    b = CalibrationSpec.from_dict(
        {
            "objects": ["TOP_LABEL", "CLASSWISE"],
            "prediction_source": "PREDICT_PROBA",
            "baseline_run": RUN,
            "binning": {"n_bins": 10, "strategy": "UNIFORM"},
        }
    )
    assert a.spec_id == b.spec_id and a.spec_id.startswith("cbs_") and len(a.spec_id) == 36
    assert CalibrationSpec.from_dict(a.to_dict()) == a and CalibrationSpec.from_dict(a.to_dict()).spec_id == a.spec_id  # fmt: skip


@pytest.mark.parametrize(
    "over",
    [
        {"baseline_run": RUN2},
        {"prediction_source": "SOFTMAX_LOGITS"},
        {"binning": {"n_bins": 11}},
        {"binning": {"strategy": "QUANTILE"}},
        {"objects": ["CLASSWISE", "TOP_LABEL"]},
        {"method": "PLATT"},
        {"method": "ISOTONIC"},
        {"method": "PLATT", "fit": {"seed": 1}},
        {"method": "PLATT", "fit": {"fraction": 0.4}},
        {"method": "PLATT", "fit": {"mode": "RUN", "calibration_run": RUN2}},
        {"statistics": {"seed": 1}},
        {"statistics": {"resamples": 501}},
        {"statistics": {"permutations": 7}},
        {"statistics": {"confidence": 0.9}},
        {"statistics": {"min_samples": 31}},
        {"statistics": {"correction": "BONFERRONI"}},
        {"statistics": {"practical_delta": 0.05}},
        {"statistics": {"high_confidence": 0.95}},
        {"stress_analyses": [SXA]},
        {"model_id": "mdl_" + "1" * 32},
        {"split": "test"},
        {
            "windows": {
                "ordering": {"field": "index"},
                "windows": [{"start": 0, "end": 5, "name": "w"}],
            }
        },
        {"slices": [{"name": "s", "condition": {"op": "eq", "field": "target", "values": [1]}}]},
    ],
)
def test_every_meaningful_change_alters_the_identity(over: dict[str, Any]) -> None:
    base = spec()
    changed = spec(**over)
    assert changed.spec_id != base.spec_id


def test_slice_description_is_a_label_but_its_definition_is_identity() -> None:
    cond = {"op": "eq", "field": "target", "values": [1]}
    a = spec(slices=[{"name": "s", "condition": cond, "description": "one"}])
    b = spec(slices=[{"name": "s", "condition": cond, "description": "two"}])
    c = spec(slices=[{"name": "s", "condition": {"op": "eq", "field": "target", "values": [0]}}])
    assert a.spec_id == b.spec_id != c.spec_id


def test_the_fit_protocol_is_identity_only_when_a_method_is_chosen() -> None:
    with pytest.raises(ValidationError, match="only meaningful with a calibration method"):
        spec(fit={"seed": 3})
    assert "fit" not in spec().to_dict() and "fit" in spec(method="PLATT").to_dict()


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"prediction_source": "LOGITS"}, "prediction_source"),
        ({"prediction_source": "softmax"}, "never assumed"),
        ({"target": "REGRESSION"}, "target"),
        (
            {"objects": ["CLASSWISE", "TOP_LABEL"]},
            None,
        ),  # unsorted through the dataclass is refused
        ({"objects": ["BOTH"]}, "objects"),
        ({"objects": []}, "objects"),
        ({"method": "TEMPERATURE"}, "method"),
        ({"binning": {"n_bins": 0}}, "n_bins"),
        ({"binning": {"n_bins": 1001}}, "n_bins"),
        ({"binning": {"strategy": "KMEANS"}}, "strategy"),
        ({"binning": {"bins": 5}}, "unexpected"),
        ({"statistics": {"resamples": 0}}, "resamples"),
        ({"statistics": {"confidence": 1.0}}, "confidence"),
        ({"statistics": {"confidence": float("nan")}}, "confidence"),
        ({"statistics": {"min_samples": 1}}, "min_samples"),
        ({"statistics": {"correction": "HOLM"}}, "correction"),
        ({"statistics": {"low_confidence": 0.95, "high_confidence": 0.9}}, "low_confidence"),
        ({"statistics": {"interval_method": "bca"}}, "bootstrap-percentile"),
        ({"statistics": {"bogus": 1}}, "unexpected"),
        ({"method": "PLATT", "fit": {"mode": "RUN"}}, "calibration_run"),
        ({"method": "PLATT", "fit": {"mode": "SPLIT", "calibration_run": RUN2}}, "only meaningful"),
        ({"method": "PLATT", "fit": {"mode": "RUN", "calibration_run": RUN}}, "leakage"),
        ({"method": "PLATT", "fit": {"fraction": 1.0}}, "fraction"),
        ({"method": "PLATT", "fit": {"mode": "CV"}}, "mode"),
        ({"stress_analyses": ["nope"]}, "stress_analyses"),
        ({"bogus": 1}, "unexpected"),
        ({"calibration_version": "9"}, "calibration_version"),
        ({"model_id": 3}, "model_id"),
    ],
)
def test_invalid_specs_are_refused_before_anything_runs(over: dict[str, Any], match: str | None) -> None:  # fmt: skip
    if match is None:
        with pytest.raises(ValidationError):
            CalibrationSpec(RUN, "PREDICT_PROBA", objects=("TOP_LABEL", "CLASSWISE"))
        return
    with pytest.raises(ValidationError, match=match):
        spec(**over)


def test_missing_required_fields_and_duplicate_slices_are_refused() -> None:
    with pytest.raises(ValidationError, match="missing"):
        CalibrationSpec.from_dict({"baseline_run": RUN})
    cond = {"op": "eq", "field": "target", "values": [1]}
    with pytest.raises(ValidationError, match="unique"):
        spec(slices=[{"name": "s", "condition": cond}, {"name": "s", "condition": cond}])
    with pytest.raises(ValidationError, match="POPULATION"):
        spec(slices=[{"name": "POPULATION", "condition": cond}])
    win = {"ordering": {"field": "index"}, "windows": [{"start": 0, "end": 5, "name": "a"}, {"start": 5, "end": 9, "name": "a"}]}  # fmt: skip
    with pytest.raises(ValidationError, match="unique name"):
        spec(windows=win)
    win2 = {**win, "windows": [{"start": 0, "end": 5, "name": "a"}, {"start": 5, "end": 9, "name": "b"}], "comparisons": [["a", "zzz"]]}  # fmt: skip
    with pytest.raises(ValidationError, match="two different declared windows"):
        spec(windows=win2)


def test_window_defaults_compare_every_later_window_with_the_first() -> None:
    win = {"ordering": {"field": "index"}, "windows": [{"start": 0, "end": 5, "name": "a"}, {"start": 5, "end": 9, "name": "b"}, {"start": 9, "end": 12, "name": "c"}]}  # fmt: skip
    s = spec(windows=win)
    assert s.windows is not None and s.windows.pairs() == (("a", "b"), ("a", "c"))


def test_stress_ids_are_sorted_and_deduplicated() -> None:
    other = "sxa_" + "d" * 32
    assert spec(stress_analyses=[other, SXA, SXA]).stress_analyses == (SXA, other)


# -- probability handling ---------------------------------------------------------------------------


CLASSES = ("a", "b", "c")


def obs_of(scores: Any, true: Any = "a", pred: Any = "a") -> m.Obs | str:
    return m.to_obs(7, true, pred, scores, CLASSES)


def test_a_valid_vector_becomes_an_observation_with_the_predicted_class_probability() -> None:
    o = obs_of([0.2, 0.5, 0.3], true="b", pred="b")
    assert isinstance(o, m.Obs) and o.id == 7 and (o.true, o.pred) == (1, 1) and o.conf == 0.5 and o.correct  # fmt: skip
    o2 = obs_of([0.2, 0.5, 0.3], true="a", pred="c")  # predicted != arg-max: confidence is p[pred]
    assert isinstance(o2, m.Obs) and o2.conf == 0.3 and not o2.correct


@pytest.mark.parametrize(
    ("scores", "reason"),
    [
        (None, "MISSING_SCORES"),
        ([0.5, 0.5], "WRONG_WIDTH"),
        ([0.5, 0.3, 0.3], "NOT_NORMALIZED"),
        ([0.2, 0.2, 0.2], "NOT_NORMALIZED"),
        ([1.2, -0.1, -0.1], "OUT_OF_RANGE"),
        ([-0.5, 0.5, 1.0], "OUT_OF_RANGE"),
        ([float("nan"), 0.5, 0.5], "NON_FINITE"),
        ([float("inf"), 0.0, 0.0], "NON_FINITE"),
        (["x", 0.5, 0.5], "NON_NUMERIC_SCORES"),
    ],
)
def test_invalid_probability_vectors_are_reported_with_a_reason_never_repaired(scores: Any, reason: str) -> None:  # fmt: skip
    assert obs_of(scores) == reason


def test_zero_and_one_probabilities_and_rounding_noise_are_valid() -> None:
    o = obs_of([1.0, 0.0, 0.0])
    assert isinstance(o, m.Obs) and o.conf == 1.0
    z = obs_of([0.0, 1.0, 0.0], true="a", pred="b")
    assert isinstance(z, m.Obs) and z.conf == 1.0 and not z.correct
    noisy = obs_of([1.0 + 1e-12, -1e-12, 0.0])  # within TOL: clamped, not invalid
    assert isinstance(noisy, m.Obs) and noisy.probs == (1.0, 0.0, 0.0)


def test_unknown_targets_and_predictions_are_invalid_not_dropped() -> None:
    assert obs_of([0.3, 0.3, 0.4], true="zzz") == "INVALID_TARGET"
    assert obs_of([0.3, 0.3, 0.4], pred="zzz") == "INVALID_PREDICTION"


# -- bins -------------------------------------------------------------------------------------------


def test_uniform_edges_and_boundary_membership_use_the_reported_edges() -> None:
    edges = m.make_edges("UNIFORM", 10, [])
    assert edges[0] == 0.0 and edges[-1] == 1.0 and len(edges) == 11
    assert m.assign(0.0, edges) == 0 and m.assign(1.0, edges) == 9  # last bin is closed
    assert m.assign(0.3, edges) == 3 and m.assign(0.2999999999, edges) == 2  # boundary: upper bin
    for k in range(1, 100):  # float rounding never disagrees with the stored bounds
        c = k / 100
        i = m.assign(c, edges)
        assert edges[i] <= c and (c < edges[i + 1] or i == 9)


def test_empty_and_single_observation_bins_are_kept_and_labelled() -> None:
    bins = m.curve([(0.05, True), (0.95, True), (0.96, False)], m.make_edges("UNIFORM", 10, []), min_bin=3)  # fmt: skip
    assert len(bins) == 10
    assert [b["count"] for b in bins] == [1, 0, 0, 0, 0, 0, 0, 0, 0, 2]
    assert bins[0]["evidence"] == "SINGLE_OBSERVATION" and bins[1]["evidence"] == "EMPTY"
    assert bins[9]["evidence"] == "SPARSE"
    assert bins[1]["accuracy"] is None and bins[1]["mean_confidence"] is None and bins[1]["interval"] is None  # fmt: skip
    assert bins[9]["upper_closed"] and not bins[0]["upper_closed"]


def test_quantile_edges_tie_handling_and_tiling() -> None:
    vals = [0.5, 0.5, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9]
    edges = m.make_edges("QUANTILE", 4, vals)
    assert edges[0] == 0.5 and edges[-1] == 0.9 and edges == sorted(edges)
    assert len(edges) - 1 <= 4  # ties share a bin: never more bins than requested
    bins = m.curve([(v, True) for v in vals], edges)
    assert sum(b["count"] for b in bins) == len(vals)  # every observation lands in exactly one bin
    same = m.make_edges("QUANTILE", 5, [0.7] * 10)
    assert same == [0.7, 0.7] and m.curve([(0.7, True)] * 10, same)[0]["count"] == 10


def test_bin_membership_is_independent_of_row_order() -> None:
    rng = random.Random(3)
    pairs = [(rng.random(), rng.random() < 0.5) for _ in range(60)]
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    for strategy in ("UNIFORM", "QUANTILE"):
        e1 = m.make_edges(strategy, 6, [p for p, _ in pairs])
        e2 = m.make_edges(strategy, 6, [p for p, _ in shuffled])
        assert e1 == e2 and m.curve(pairs, e1) == m.curve(shuffled, e2)


# -- metrics: hand-calculated ------------------------------------------------------------------------

# (confidence, correct): bin [0, .5): 0.3 F, 0.2 T -> mean .25, acc .5, |gap| .25, weight 2/5
#                        bin [.5, 1]: 0.9 T, 0.8 T, 0.7 F -> mean .8, acc 2/3, |gap| .13333, weight 3/5
PAIRS = [(0.9, True), (0.8, True), (0.7, False), (0.3, False), (0.2, True)]


def test_ece_mce_brier_and_log_loss_match_hand_calculation() -> None:
    bins = m.curve(PAIRS, m.make_edges("UNIFORM", 2, []))
    assert [b["count"] for b in bins] == [2, 3]
    assert bins[0]["mean_confidence"] == pytest.approx(0.25) and bins[0]["accuracy"] == 0.5
    assert bins[1]["mean_confidence"] == pytest.approx(0.8) and bins[1]["accuracy"] == pytest.approx(2 / 3)  # fmt: skip
    assert bins[0]["gap"] == pytest.approx(-0.25) and bins[1]["gap"] == pytest.approx(0.8 - 2 / 3)
    assert (
        m.ece(bins, 5) == pytest.approx(0.4 * 0.25 + 0.6 * abs(2 / 3 - 0.8)) == pytest.approx(0.18)
    )
    assert m.mce(bins) == pytest.approx(0.25)
    assert m.top_brier(PAIRS) == pytest.approx((0.01 + 0.04 + 0.49 + 0.09 + 0.64) / 5) == pytest.approx(0.254)  # fmt: skip
    ll = -(math.log(0.9) + math.log(0.8) + math.log(1 - 0.7) + math.log(1 - 0.3) + math.log(0.2)) / 5  # fmt: skip
    assert m.top_log_loss(PAIRS) == pytest.approx(ll)


def test_log_loss_clips_zero_and_one_probabilities_instead_of_exploding() -> None:
    v = m.top_log_loss([(1.0, False), (0.0, True)])
    assert math.isfinite(v) and v == pytest.approx(-math.log(1e-15), rel=1e-9)
    assert m.top_log_loss([(1.0, True), (0.0, False)]) == pytest.approx(0.0, abs=1e-12)


def _vec(probs: list[float], true: int) -> m.Obs:
    return m.Obs(0, true, max(range(len(probs)), key=probs.__getitem__), tuple(probs))


def test_multiclass_brier_and_nll_match_hand_calculation() -> None:
    obs = [_vec([0.7, 0.2, 0.1], 0), _vec([0.2, 0.5, 0.3], 2)]
    # sample 1: .3^2+.2^2+.1^2 = .14; sample 2: .2^2+.5^2+.7^2 = .78; mean .46
    assert m.vector_brier(obs) == pytest.approx(0.46)
    assert m.vector_nll(obs) == pytest.approx(-(math.log(0.7) + math.log(0.3)) / 2)


def test_top_label_and_classwise_are_different_objects() -> None:
    # three classes: the top-label event (prediction correct) differs from each classwise event
    rng = random.Random(11)
    obs = []
    for i in range(90):
        raw = [rng.random() + 0.05 for _ in range(3)]
        p = tuple(x / sum(raw) for x in raw)
        true = rng.choices(range(3), weights=p)[0]
        obs.append(m.Obs(i, true, max(range(3), key=p.__getitem__), p))
    top = m.top_metrics(obs, "UNIFORM", 5)
    cw = m.classwise(obs, ("x", "y", "z"), "UNIFORM", 5, min_bin=3, min_positives=3, confidence=0.95)  # fmt: skip
    assert top["ece"] is not None and cw["mean_ece"] is not None
    assert top["ece"] != pytest.approx(cw["mean_ece"])
    assert set(cw["per_class"]) == {"x", "y", "z"} and cw["classes_computed"] == 3


def test_classwise_hand_example_and_insufficient_classes() -> None:
    # class k=0 (label "x"): p0 = [.1, .3, .8, .6], event true==0 = [F, T, T, F]
    # bin [0,.5): .1 F, .3 T -> mean .2, rate .5, gap .3;  bin [.5,1]: .8 T, .6 F -> mean .7, rate .5, gap .2
    # ECE_0 = .5*.3 + .5*.2 = .25
    probs = [(0.1, 0.5, 0.4), (0.3, 0.3, 0.4), (0.8, 0.1, 0.1), (0.6, 0.3, 0.1)]
    truth = [1, 0, 0, 2]
    obs = [m.Obs(i, t, max(range(3), key=p.__getitem__), p) for i, (p, t) in enumerate(zip(probs, truth, strict=True))]  # fmt: skip
    cw = m.classwise(obs, ("x", "y", "z"), "UNIFORM", 2, min_bin=1, min_positives=2, confidence=0.95)  # fmt: skip
    x = cw["per_class"]["x"]
    assert x["status"] == "COMPUTED" and x["ece"] == pytest.approx(0.25) and x["positives"] == 2
    assert cw["per_class"]["y"]["status"] == "INSUFFICIENT_EVIDENCE" and cw["per_class"]["y"]["ece"] is None  # fmt: skip
    assert cw["per_class"]["z"]["status"] == "INSUFFICIENT_EVIDENCE" and "1 positive" in cw["per_class"]["z"]["reason"]  # fmt: skip


def test_per_bin_wilson_interval_uses_the_phase_10_core() -> None:
    from experionyx.stats.core import proportion_interval

    bins = m.curve(PAIRS, m.make_edges("UNIFORM", 2, []), confidence=0.95)
    w = proportion_interval(2, 3, 0.95)
    assert bins[1]["interval"]["lower"] == w.lower and bins[1]["interval"]["upper"] == w.upper
    assert bins[1]["interval"]["method"] == "wilson"


def test_entropy_margin_and_normalization() -> None:
    assert m.entropy([0.5, 0.5]) == pytest.approx(math.log(2))
    assert m.normalized_entropy([0.5, 0.5]) == pytest.approx(1.0)
    assert m.entropy([1.0, 0.0, 0.0]) == 0.0 and m.normalized_entropy([1.0, 0.0, 0.0]) == 0.0
    assert m.normalized_entropy([1.0]) is None and m.margin([1.0]) is None  # undefined, not 0
    assert m.margin([0.2, 0.5, 0.3]) == pytest.approx(0.2)
    assert m.entropy([0.25] * 4) == pytest.approx(
        math.log(4)
    )  # raw entropy is in nats, unnormalized


def test_auroc_confidence_relationship() -> None:
    assert m.auroc([0.9, 0.8], [0.3, 0.2]) == 1.0
    assert m.auroc([0.5], [0.5]) == 0.5
    assert m.auroc([0.9], []) is None


# -- uncertainty of the metrics -----------------------------------------------------------------------


def _obs(n: int, seed: int = 1) -> list[m.Obs]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        c = 0.5 + rng.random() / 2
        ok = rng.random() < c - 0.1
        out.append(m.Obs(i, 0 if ok else 1, 0, (c, 1 - c)))
    return out


def test_bootstrap_is_deterministic_for_a_seed_and_recomputes_every_metric() -> None:
    obs = _obs(80)
    kw = {"resamples": 80, "confidence": 0.9}
    a = m.bootstrap_metrics(obs, "UNIFORM", 5, seed=4, **kw)  # type: ignore[arg-type]
    b = m.bootstrap_metrics(obs, "UNIFORM", 5, seed=4, **kw)  # type: ignore[arg-type]
    c = m.bootstrap_metrics(obs, "UNIFORM", 5, seed=5, **kw)  # type: ignore[arg-type]
    assert a == b and a != c
    assert {"ece", "mce", "accuracy", "brier_top_label", "log_loss_top_label", "brier_multiclass"} <= set(a)  # fmt: skip
    for name, iv in a.items():
        assert iv["method"] == "bootstrap-percentile" and iv["resamples"] == 80 and iv["seed"] == 4, name  # fmt: skip
        assert iv["status"] == "DERIVED" and iv["lower"] <= iv["upper"]
    assert a["accuracy"]["lower"] <= a["accuracy"]["estimate"] <= a["accuracy"]["upper"]


def test_a_degenerate_sample_is_flagged_not_hidden() -> None:
    obs = [m.Obs(i, 0, 0, (1.0, 0.0)) for i in range(40)]
    r = m.bootstrap_metrics(obs, "UNIFORM", 5, resamples=30, confidence=0.95, seed=0)
    assert r["accuracy"]["lower"] == r["accuracy"]["upper"] == 1.0
    assert any("zero variance" in w for w in r["accuracy"]["warnings"])


def test_paired_comparison_is_invariant_to_row_order_and_recomputes_ece_differences() -> None:
    a = _obs(60, 2)
    b = [m.Obs(o.id, o.true, o.pred, o.probs, cal=min(0.99, o.conf * 0.9)) for o in a]
    kw = {"strategy": "UNIFORM", "n_bins": 5, "names": ("ece", "mce"), "resamples": 40, "permutations": 40, "confidence": 0.95, "seed": 3, "paired": True}  # fmt: skip
    r1 = m.compare_metrics(a, b, **kw)  # type: ignore[arg-type]
    rng = random.Random(0)
    pa, pb = list(zip(a, b, strict=True)), []
    rng.shuffle(pa)
    pb = [y for _, y in pa]
    # pairing is by position in the caller's contract: shuffling BOTH sides identically changes nothing
    r2 = m.compare_metrics([x for x, _ in pa], pb, **kw)  # type: ignore[arg-type]
    assert r1["ece"]["difference"] == pytest.approx(r2["ece"]["difference"])
    ea = m.top_metrics(a, "UNIFORM", 5, vector=False)["ece"]
    eb = m.top_metrics(b, "UNIFORM", 5, vector=False)["ece"]
    assert r1["ece"]["difference"] == pytest.approx(eb - ea)  # type: ignore[operator]
    assert 0 < r1["ece"]["p_value"] <= 1 and r1["ece"]["test"]["name"] == "paired condition-swap permutation"  # fmt: skip


# -- post-hoc calibrators ----------------------------------------------------------------------------


def test_isotonic_pool_adjacent_violators_hand_example() -> None:
    # x = .1 .2 .3 .4 .5, y = 0 1 0 1 1: (.2, .3) violate -> pooled .5; fitted 0, .5, .5, 1, 1
    pairs = [(0.1, False), (0.2, True), (0.3, False), (0.4, True), (0.5, True)]
    f = m.fit_isotonic(pairs)
    assert f["x"] == [0.1, 0.2, 0.3, 0.4, 0.5] and f["y"] == [0.0, 0.5, 0.5, 1.0, 1.0]
    assert m.apply_calibrator("ISOTONIC", f, 0.25) == pytest.approx(0.5)
    assert m.apply_calibrator("ISOTONIC", f, 0.35) == pytest.approx(0.75)  # linear between blocks
    assert m.apply_calibrator("ISOTONIC", f, 0.0) == 0.0 and m.apply_calibrator("ISOTONIC", f, 0.9) == 1.0  # fmt: skip
    ys = f["y"]
    assert ys == sorted(ys)  # monotone by construction


def test_isotonic_pools_tied_confidences_before_fitting() -> None:
    f = m.fit_isotonic([(0.5, True), (0.5, False), (0.9, True)])
    assert f["x"] == [0.5, 0.9] and f["y"] == [0.5, 1.0]


def test_platt_is_deterministic_and_recovers_a_known_map() -> None:
    rng = random.Random(5)
    pairs = []
    for _ in range(4000):  # P(correct | conf) = sigmoid(2 * logit(conf) - 0.5)
        c = 0.3 + 0.69 * rng.random()
        z = 2.0 * math.log(c / (1 - c)) - 0.5
        pairs.append((c, rng.random() < 1 / (1 + math.exp(-z))))
    f1, f2 = m.fit_platt(pairs), m.fit_platt(pairs)
    assert f1 == f2 and f1["converged"]
    assert f1["a"] == pytest.approx(2.0, abs=0.25) and f1["b"] == pytest.approx(-0.5, abs=0.25)
    assert 0 < m.apply_calibrator("PLATT", f1, 0.8) < 1


def test_platt_on_separable_data_still_converges_because_of_the_recorded_ridge() -> None:
    pairs = [(0.2, False), (0.3, False), (0.7, True), (0.9, True)] * 5
    f = m.fit_platt(pairs)
    assert f["ridge"] == m.PLATT_RIDGE and f["a"] > 0 and math.isfinite(f["a"])


# -- the deterministic calibration / evaluation split ---------------------------------------------------


def test_split_is_disjoint_covering_deterministic_and_order_independent() -> None:
    ids = list(range(50))
    cal, ev = split_ids(ids, 0.4, 3)
    assert not set(cal) & set(ev) and sorted(cal + ev) == ids and len(cal) == 20
    assert split_ids(list(reversed(ids)), 0.4, 3) == (cal, ev)  # row order does not matter
    assert split_ids(ids, 0.4, 4) != (cal, ev)  # the seed changes the partition
    tiny_cal, tiny_ev = split_ids([1, 2], 0.01, 0)
    assert len(tiny_cal) == 1 and len(tiny_ev) == 1  # never an empty side


def test_statistics_defaults_are_recorded_in_the_identity() -> None:
    d: Any = spec().to_dict()
    assert d["statistics"]["interval_method"] == "bootstrap-percentile"
    assert Statistics().min_samples == 30 and Fit().mode == "SPLIT"

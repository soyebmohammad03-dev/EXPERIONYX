"""Stress domain, transformations, capabilities and analysis helpers: identity and normalization,
validation, deterministic expansion, seeded/unseeded behaviour, model and input immutability,
restoration, thresholds, unsupported capabilities, and every statistic against a hand computation.
No workspace."""

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from eval_helpers import pure_registries
from experionyx.adapters.base import ParameterAccess
from experionyx.adapters.capabilities import DeviceKind, ModelCapability
from experionyx.errors import ValidationError
from experionyx.evaluation.config import EvaluationConfig
from experionyx.stats import core as st
from experionyx.stress import analysis as an
from experionyx.stress.capability import StressUnsupported, family_status
from experionyx.stress.spec import (
    FAMILIES,
    MAX_TRIALS,
    Origin,
    StressDesign,
    StressPlan,
    StressSpec,
    Sweep,
)
from experionyx.stress.transform import arrays_digest, build_stressed_model, perturb_arrays
from pure_adapters import LinearProbaAdapter, ThresholdClassifierAdapter
from stress_helpers import COEF, N, p1, rows

MODEL = "mdl_" + "a" * 32
DATA = "dst_" + "b" * 32
RUN = "run_" + "c" * 32


def linear(tmp_path: Path, coef: list[float] | None = None, b: float = 0.0) -> LinearProbaAdapter:
    f = tmp_path / "m.json"
    f.write_text(json.dumps({"coef": coef or COEF, "intercept": b}))
    return LinearProbaAdapter.load(f, version="1", device=DeviceKind.CPU, options={})


# -- identity, normalization, validation -------------------------------------------------------------------------------


def test_stress_identity_is_deterministic_normalized_and_complete() -> None:
    a = StressSpec("FEATURE_SCALING", {"factor": 2.0})
    b = StressSpec(
        "FEATURE_SCALING", {"factor": 2, "fraction": 1.0}
    )  # defaults spelled out, 2.0 == 2
    assert a.stress_id == b.stress_id and a.stress_id.startswith("sts_") and len(a.stress_id) == 36
    assert a.parameters == {"factor": 2, "fraction": 1} and a.scope is None
    for other in (
        StressSpec("FEATURE_SCALING", {"factor": 2.5}),
        StressSpec(
            "FEATURE_SCALING", {"factor": 2, "fraction": 0.5}, seed=1
        ),  # fraction < 1 makes the seed matter
        StressSpec(
            "FEATURE_SCALING",
            {"factor": 2},
            scope={"kind": "RANDOM_SUBSET", "fraction": 0.5},
            seed=0,
        ),
        StressSpec("FEATURE_OFFSET", {"offset": 2.0}),
    ):
        assert other.stress_id != a.stress_id
    n = StressSpec("PARAMETER_NOISE", {"relative_sigma": 0.1}, seed=3)
    assert (
        n.stress_id != StressSpec("PARAMETER_NOISE", {"relative_sigma": 0.1}, seed=4).stress_id
    )  # the seed is identity
    assert (
        n.stress_id
        != StressSpec(
            "PARAMETER_NOISE", {"relative_sigma": 0.1, "targets": ["coef_"]}, seed=3
        ).stress_id
    )
    assert StressSpec.from_dict(json.loads(json.dumps(n.to_dict()))) == n
    assert StressSpec("PARAMETER_SCALE", {"factor": 2, "targets": ["b", "a"]}).parameters[
        "targets"
    ] == ["a", "b"]  # sorted


def test_fault_families_are_the_fault_laboratory_transformations() -> None:
    from experionyx.faults.library import default_fault_registry

    for name, fam in FAMILIES.items():
        if fam.origin is Origin.FAULT_LABORATORY:
            assert fam.fault_type in default_fault_registry().names(), (
                name
            )  # nothing is re-implemented here
    s = StressSpec("FEATURE_NOISE", {"sigma": 0.5}, seed=7)
    f = s.fault_spec()
    assert f.type == "gaussian_noise" and f.seed == 7 and f.parameters.sigma == 0.5  # type: ignore[attr-defined]
    with pytest.raises(ValidationError, match="not a Fault Laboratory"):
        StressSpec("THRESHOLD", {"threshold": 0.4}).fault_spec()


@pytest.mark.parametrize(
    ("family", "params", "match"),
    [
        ("NOPE", {}, "unknown stress family"),
        ("FEATURE_SCALING", {"factor": -1}, "invalid FEATURE_SCALING"),
        ("FEATURE_SCALING", {"factor": 2, "bogus": 1}, "invalid FEATURE_SCALING"),
        ("FEATURE_NOISE", {"sigma": -1}, "invalid FEATURE_NOISE"),
        ("BLUR", {"radius": 0}, "invalid BLUR"),
        ("PARAMETER_NOISE", {}, "needs parameter"),
        ("PARAMETER_NOISE", {"relative_sigma": -0.1}, "below its valid range"),
        ("PARAMETER_NOISE", {"relative_sigma": float("nan")}, "finite number"),
        ("PARAMETER_NOISE", {"relative_sigma": 0.1, "extra": 1}, "no parameter"),
        ("PARAMETER_SCALE", {"factor": 0}, "below its valid range"),
        ("THRESHOLD", {"threshold": 0}, "below its valid range"),
        ("THRESHOLD", {"threshold": 1}, "above its valid range"),
        ("THRESHOLD", {"threshold": True}, "finite number"),
        ("BATCH_SIZE", {"batch_size": 0}, "below its valid range"),
        ("BATCH_SIZE", {"batch_size": 2.5}, "must be an integer"),
        ("REPEATED_EXECUTION", {"repeats": 1}, "below its valid range"),
        ("REPEATED_EXECUTION", {"repeats": 999}, "above its valid range"),
        ("INPUT_SHAPE", {"shape": []}, "non-empty list"),
    ],
)  # fmt: skip
def test_invalid_stress_is_refused_before_anything_runs(
    family: str, params: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        StressSpec(family, params)


def test_seed_rules_never_invent_randomness() -> None:
    with pytest.raises(ValidationError, match="deterministic: it takes no seed"):
        StressSpec("PARAMETER_SCALE", {"factor": 2}, seed=3)
    with pytest.raises(ValidationError, match="deterministic"):
        StressSpec(
            "FEATURE_SCALING", {"factor": 2}, seed=1
        )  # fraction 1: every column, nothing random
    with pytest.raises(ValidationError, match="takes no scope"):
        StressSpec("THRESHOLD", {"threshold": 0.5}, scope={"kind": "ALL"})
    with pytest.raises(ValidationError, match="seed"):
        StressSpec("PARAMETER_NOISE", {"relative_sigma": 0.1}, seed=-1)
    assert (
        StressSpec("FEATURE_NOISE", {"sigma": 1.0}, seed=5).seeded
        and not StressSpec("BRIGHTNESS", {"delta": 0.1}).seeded
    )
    assert StressSpec(
        "FEATURE_SCALING", {"factor": 2, "fraction": 0.5}, seed=2
    ).seeded  # a random column subset


# -- plans: expansion, sweeps, compounds, factorials ----------------------------------------------------------------------


def test_plan_expansion_is_explicit_deterministic_and_complete() -> None:
    s = StressSpec("PARAMETER_NOISE", {"relative_sigma": 0.1})
    plan = StressPlan((s,), Sweep("relative_sigma", (0.0, 0.1, 0.5)), (1, 2))
    units = plan.units()
    assert plan.design == "SWEEP" and len(units) == 6
    assert [(u.point_index, u.repeat_index, u.seed, u.value) for u in units] == [
        (0, 0, 1, 0.0),
        (0, 1, 2, 0.0),
        (1, 0, 1, 0.1),
        (1, 1, 2, 0.1),
        (2, 0, 1, 0.5),
        (2, 1, 2, 0.5),
    ]
    assert len({u.key for u in units}) == 6  # every unit has its own identity
    assert (
        units[3].components[0].parameters["relative_sigma"] == 0.1
        and units[3].components[0].seed == 2
    )
    assert [u.key for u in units] == [
        u.key for u in StressPlan.from_dict(json.loads(json.dumps(plan.to_dict()))).units()
    ]
    assert plan.plan_id == StressPlan.from_dict(plan.to_dict()).plan_id and plan.plan_id.startswith(
        "sxp_"
    )
    single = StressPlan((StressSpec("THRESHOLD", {"threshold": 0.4}),))
    assert (
        single.design == "SINGLE" and len(single.units()) == 1 and single.units()[0].value is None
    )
    rep = StressPlan((StressSpec("PARAMETER_NOISE", {"relative_sigma": 0.2}),), None, (1, 2, 3))
    assert rep.design == "REPEATED" and [u.seed for u in rep.units()] == [1, 2, 3]
    rex = StressPlan((StressSpec("REPEATED_EXECUTION", {"repeats": 4}),))
    assert (
        rex.design == "REPEATED_EXECUTION"
        and len(rex.units()) == 4
        and len({u.key for u in rex.units()}) == 4
    )


def test_changing_any_meaningful_part_changes_the_plan_identity() -> None:
    base = StressPlan(
        (StressSpec("PARAMETER_NOISE", {"relative_sigma": 0.1}),),
        Sweep("relative_sigma", (0.1, 0.2)),
        (1, 2),
    )
    for other in (
        StressPlan(
            (StressSpec("PARAMETER_NOISE", {"relative_sigma": 0.1}),),
            Sweep("relative_sigma", (0.1, 0.3)),
            (1, 2),
        ),
        StressPlan(
            (StressSpec("PARAMETER_NOISE", {"relative_sigma": 0.1}),),
            Sweep("relative_sigma", (0.1, 0.2)),
            (1, 3),
        ),
        StressPlan(
            (StressSpec("PARAMETER_NOISE", {"relative_sigma": 0.1, "targets": ["coef_"]}),),
            Sweep("relative_sigma", (0.1, 0.2)),
            (1, 2),
        ),
        StressPlan((StressSpec("PARAMETER_NOISE", {"relative_sigma": 0.1}),), None, (1, 2)),
    ):
        assert other.plan_id != base.plan_id


def test_compound_order_is_identity_and_only_meaningful_orders_are_allowed() -> None:
    a, b = (
        StressSpec("FEATURE_NOISE", {"sigma": 1.0}, seed=1),
        StressSpec("FEATURE_OFFSET", {"offset": 2.0}),
    )
    ab, ba = StressPlan((a, b)), StressPlan((b, a))
    assert ab.design == "COMPOUND" and ab.plan_id != ba.plan_id  # A then B is not B then A
    assert ab.fault_spec_of(ab.units()[0]).id != ba.fault_spec_of(ba.units()[0]).id
    p, t = (
        StressSpec("PARAMETER_SCALE", {"factor": 1.5}),
        StressSpec("THRESHOLD", {"threshold": 0.4}),
    )
    assert StressPlan((p, t)).design == "COMPOUND"
    with pytest.raises(ValidationError, match="parameter perturbations before"):
        StressPlan((t, p))
    with pytest.raises(ValidationError, match="mixed origins are refused"):
        StressPlan((a, t))
    with pytest.raises(ValidationError, match="at most one decision-rule"):
        StressPlan((StressSpec("THRESHOLD", {"threshold": 0.3}), t))
    with pytest.raises(ValidationError, match=r"mixed origins"):
        StressPlan((StressSpec("BATCH_SIZE", {"batch_size": 8}), p))
    with pytest.raises(ValidationError, match="cannot be swept"):
        StressPlan((a, b), Sweep("sigma", (1.0, 2.0)))


def test_factorial_designs_expand_to_three_explicit_cells_and_reject_invalid_designs() -> None:
    a, b = (
        StressSpec("FEATURE_NOISE", {"sigma": 1.0}, seed=1),
        StressSpec("FEATURE_OFFSET", {"offset": 2.0}),
    )
    f = StressPlan((a, b), None, (1, 2), True)
    us = f.units()
    assert f.design == "FACTORIAL_2X2" and [(u.cell, u.seed) for u in us] == [
        ("A", 1),
        ("B", 1),
        ("AB", 1),
        ("A", 2),
        ("B", 2),
        ("AB", 2),
    ]
    assert [len(u.components) for u in us[:3]] == [1, 1, 2] and len({u.key for u in us}) == 6
    for comps, kw, match in (
        ((a,), {}, "exactly two"),
        ((a, StressSpec("THRESHOLD", {"threshold": 0.4})), {}, "exactly two Fault Laboratory"),
        ((a, StressSpec("FEATURE_NOISE", {"sigma": 1.0}, seed=1)), {}, "must differ"),
        ((a, b), {"sweep": Sweep("sigma", (1.0, 2.0))}, "cannot be swept"),
    ):
        with pytest.raises(ValidationError, match=match):
            StressPlan(comps, seeds=(1,), factorial=True, **kw)


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: StressPlan(()), "at least one stress"),
        (lambda: StressPlan((StressSpec("THRESHOLD", {"threshold": 0.4}),), None, (1, 2)), "several seeds would repeat identical"),
        (lambda: StressPlan((StressSpec("THRESHOLD", {"threshold": 0.4}),), None, ()), "seeds"),
        (lambda: StressPlan((StressSpec("PARAMETER_NOISE", {"relative_sigma": 0.1}),), None, (1, 1)), "unique"),
        (lambda: StressPlan((StressSpec("PARAMETER_NOISE", {"relative_sigma": 0.1}),), None, (-1,)), "non-negative"),
        (lambda: StressPlan((StressSpec("PARAMETER_NOISE", {"relative_sigma": 0.1}),), Sweep("relative_sigma", (-1.0, 0.1))), "below its valid range"),
        (lambda: StressPlan((StressSpec("PARAMETER_NOISE", {"relative_sigma": 0.1}),), Sweep("nope", (1.0,))), "no parameter"),
        (lambda: StressPlan((StressSpec("PARAMETER_NOISE", {"relative_sigma": 0.1}),), Sweep("targets", (1.0,))), "no parameter"),
        (lambda: StressPlan((StressSpec("PARAMETER_NOISE", {"relative_sigma": 0.1}),), Sweep("relative_sigma", (0.1,), component=3)), "out of range"),
        (lambda: StressPlan((StressSpec("REPEATED_EXECUTION", {"repeats": 3}),), None, (1, 2)), "stands alone"),
        (lambda: StressPlan((StressSpec("PARAMETER_NOISE", {"relative_sigma": 0.1}),), Sweep("relative_sigma", tuple(0.001 * i for i in range(1, MAX_TRIALS + 5)))), "exceed the limit"),
        (lambda: Sweep("x", ()), "at least one value"),
        (lambda: Sweep("x", (1.0, 1.0)), "unique"),
        (lambda: Sweep("x", (float("inf"),)), "finite"),
    ],
)  # fmt: skip
def test_invalid_plans_are_refused_before_execution(make: Any, match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        make()


def test_design_identity_and_validation() -> None:
    plan = StressPlan((StressSpec("THRESHOLD", {"threshold": 0.4}),))
    ev = EvaluationConfig(split="test")
    d = StressDesign(MODEL, DATA, plan, ev)
    assert (
        d.spec_id.startswith("sxr_")
        and StressDesign.from_dict(json.loads(json.dumps(d.to_dict()))).spec_id == d.spec_id
    )
    assert (
        "baseline_run" not in d.to_dict()
        and StressDesign(MODEL, DATA, plan, ev, baseline_run=RUN).spec_id != d.spec_id
    )
    for changed in (
        StressDesign(MODEL, "dst_" + "d" * 32, plan, ev), StressDesign("mdl_" + "e" * 32, DATA, plan, ev),
        StressDesign(MODEL, DATA, plan, EvaluationConfig(split="test", batch_size=7)),
        StressDesign(MODEL, DATA, plan, ev, correction="BONFERRONI"), StressDesign(MODEL, DATA, plan, ev, seed=1),
        StressDesign(MODEL, DATA, plan, ev, discover_failures=True),
        StressDesign(MODEL, DATA, plan, ev, slices=({"name": "s", "condition": {"op": "eq", "field": "target", "values": [1]}},)),
    ):  # fmt: skip
        assert changed.spec_id != d.spec_id
    for kw in (
        {"min_members": 1},
        {"confidence": 1.0},
        {"alpha": 0},
        {"correction": "HOLM"},
        {"resamples": 0},
        {"seed": -1},
    ):
        with pytest.raises(ValidationError):
            StressDesign(MODEL, DATA, plan, ev, **kw)
    with pytest.raises(ValidationError, match="unique"):
        StressDesign(MODEL, DATA, plan, ev, slices=({"name": "s"}, {"name": "s"}))
    for bad in (
        {**d.to_dict(), "extra": 1},
        {**d.to_dict(), "stress_schema": 9},
        {k: v for k, v in d.to_dict().items() if k != "plan"},
    ):
        with pytest.raises(ValidationError):
            StressDesign.from_dict(bad)
    with pytest.raises(ValidationError, match="declared design"):
        StressPlan.from_dict({**plan.to_dict(), "design": "SWEEP"})
    with pytest.raises(ValidationError, match="unsupported stress_version"):
        StressPlan.from_dict({**plan.to_dict(), "stress_version": "9"})


# -- transformations: determinism, immutability, restoration ---------------------------------------------------------------------


def test_parameter_noise_is_seeded_deterministic_and_bounded(tmp_path: Path) -> None:
    m = linear(tmp_path)
    before = m.parameter_arrays()
    s = StressSpec("PARAMETER_NOISE", {"relative_sigma": 0.2}, seed=5)
    a1, rec = perturb_arrays(before, s)
    a2, _ = perturb_arrays(before, s)
    assert all(np.array_equal(a1[k], a2[k]) for k in a1)  # same seed, same perturbation
    other, _ = perturb_arrays(
        before, StressSpec("PARAMETER_NOISE", {"relative_sigma": 0.2}, seed=6)
    )
    assert not np.array_equal(a1["coef_"], other["coef_"])
    rms = math.sqrt(sum(c * c for c in COEF) / len(COEF))
    assert rec["coef_"]["noise_sigma"] == pytest.approx(0.2 * rms) and rec["coef_"]["shape"] == [2]
    assert rec["coef_"]["relative_l2_change"] == pytest.approx(
        np.linalg.norm(a1["coef_"] - before["coef_"]) / np.linalg.norm(before["coef_"])
    )
    zero, zrec = perturb_arrays(
        before, StressSpec("PARAMETER_NOISE", {"relative_sigma": 0.0}, seed=1)
    )
    assert (
        np.array_equal(zero["coef_"], before["coef_"]) and zrec["coef_"]["max_abs_change"] == 0.0
    )  # sigma 0 is the identity
    scaled, _ = perturb_arrays(
        before, StressSpec("PARAMETER_SCALE", {"factor": 3.0, "targets": ["coef_"]})
    )
    assert np.allclose(scaled["coef_"], 3 * before["coef_"]) and np.array_equal(
        scaled["intercept_"], before["intercept_"]
    )
    with pytest.raises(StressUnsupported, match="not accessible"):
        perturb_arrays(before, StressSpec("PARAMETER_SCALE", {"factor": 2.0, "targets": ["ghost"]}))


def test_the_original_model_is_never_modified_and_restoration_is_verified(tmp_path: Path) -> None:
    m = linear(tmp_path)
    digest = arrays_digest(m.parameter_arrays())
    pred_before = m.predict([[1.0, 2.0], [-1.0, 0.3]]).outputs
    for spec in (
        StressSpec("PARAMETER_NOISE", {"relative_sigma": 5.0}, seed=1),
        StressSpec("PARAMETER_SCALE", {"factor": 10.0}),
    ):
        b = build_stressed_model(m, (spec,))
        (rec,) = b.record["applied"]
        assert (
            rec["restored"] is True
            and rec["parameters_digest_original"]
            == rec["parameters_digest_original_after"]
            == digest
        )
        assert rec["parameters_digest_stressed"] != digest  # the derived model really differs
        assert arrays_digest(m.parameter_arrays()) == digest  # the registered model is untouched
        assert b.model.fingerprint() == m.fingerprint()  # identity is the registered model's
    assert m.predict([[1.0, 2.0], [-1.0, 0.3]]).outputs == pred_before
    derived = build_stressed_model(m, (StressSpec("PARAMETER_SCALE", {"factor": 0.001}),)).model
    d_out: Any = derived.predict_proba([[1.0, 2.0]]).outputs
    m_out: Any = m.predict_proba([[1.0, 2.0]]).outputs
    assert d_out[0][1] == pytest.approx(0.5, abs=1e-2)
    assert m_out[0][1] == pytest.approx(p1([1.0, 2.0]))


def test_a_parameter_stress_changes_predictions_exactly_as_the_formula_says(tmp_path: Path) -> None:
    m = linear(tmp_path)
    xs, _ = rows()
    spec = StressSpec("PARAMETER_SCALE", {"factor": 2.0})
    out = build_stressed_model(m, (spec,)).model.predict_proba(xs[:20])
    outs: Any = out.outputs
    for x, o in zip(xs[:20], outs, strict=True):
        assert o[1] == pytest.approx(p1(x, [2 * c for c in COEF]), abs=1e-12)
    assert (
        build_stressed_model(m, (spec,)).model.metadata() == m.metadata()
    )  # the stress is recorded elsewhere


def test_threshold_stress_changes_only_the_decision_rule(tmp_path: Path) -> None:
    m = linear(tmp_path)
    xs, _ = rows()
    lo = build_stressed_model(m, (StressSpec("THRESHOLD", {"threshold": 0.2}),))
    hi = build_stressed_model(m, (StressSpec("THRESHOLD", {"threshold": 0.9}),))
    probs = [p1(x) for x in xs]
    assert list(lo.model.predict(xs).outputs) == [1 if p >= 0.2 else 0 for p in probs]
    assert list(hi.model.predict(xs).outputs) == [1 if p >= 0.9 else 0 for p in probs]
    assert list(m.predict(xs).outputs) == [
        1 if p >= 0.5 else 0 for p in probs
    ]  # the default rule is untouched
    assert (
        lo.model.predict_proba(xs).outputs == m.predict_proba(xs).outputs
    )  # probabilities are the model's own
    (rec,) = lo.record["applied"]
    assert (
        rec["threshold"] == 0.2
        and rec["class_labels"] == [0, 1]
        and "argmax" in rec["default_rule"]
    )
    hi_out: Any = hi.model.predict(xs).outputs
    lo_out: Any = lo.model.predict(xs).outputs
    assert sum(hi_out) < sum(lo_out)  # monotone in the threshold
    both = build_stressed_model(
        m,
        (
            StressSpec("PARAMETER_SCALE", {"factor": 0.1}),
            StressSpec("THRESHOLD", {"threshold": 0.6}),
        ),
    )
    assert (
        len(both.record["applied"]) == 2
        and both.model.predict(xs).outputs != lo.model.predict(xs).outputs
    )


def test_unsupported_capabilities_are_reported_unavailable_not_hacked(tmp_path: Path) -> None:
    data = type(
        "D", (), {"metadata": None}
    )()  # only the Fault Laboratory families read the dataset record
    m = linear(tmp_path)
    assert isinstance(m, ParameterAccess)
    ok = family_status(m, StressSpec("PARAMETER_NOISE", {"relative_sigma": 0.1}), data)
    assert ok == ("SUPPORTED", None)
    bad = family_status(
        m, StressSpec("PARAMETER_NOISE", {"relative_sigma": 0.1, "targets": ["ghost"]}), data
    )
    assert bad[0] == "UNAVAILABLE" and "ghost" in (bad[1] or "")
    t = tmp_path / "t.json"
    t.write_text(json.dumps({"threshold": 5.0}))
    plain = ThresholdClassifierAdapter.load(t, version="1", device=DeviceKind.CPU, options={})
    assert not isinstance(plain, ParameterAccess)
    st_, why = family_status(plain, StressSpec("PARAMETER_SCALE", {"factor": 2.0}), data)
    assert st_ == "UNAVAILABLE" and "does not expose safe parameter access" in (why or "")
    with pytest.raises(StressUnsupported, match="safe parameter access"):
        build_stressed_model(plain, (StressSpec("PARAMETER_SCALE", {"factor": 2.0}),))
    shape = family_status(m, StressSpec("INPUT_SHAPE", {"shape": [3, 32, 32]}), data)
    assert shape[0] == "UNAVAILABLE" and "does not declare any supported input shapes" in (
        shape[1] or ""
    )
    assert family_status(m, StressSpec("BATCH_SIZE", {"batch_size": 4}), data)[0] == "SUPPORTED"
    repeated = family_status(m, StressSpec("REPEATED_EXECUTION", {"repeats": 3}), data)
    assert repeated[0] == "SUPPORTED"
    assert ModelCapability.PREDICT_PROBA not in plain.capabilities
    th = family_status(plain, StressSpec("THRESHOLD", {"threshold": 0.4}), data)
    assert th[0] == "UNAVAILABLE" and "does not provide predicted probabilities" in (
        th[1] or ""
    )  # ...but declares no probabilities schema here


def test_sklearn_parameter_access_is_a_safe_documented_subset() -> None:
    pytest.importorskip("sklearn")
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    from experionyx.adapters.sklearn_adapter import SklearnModelAdapter

    x, y = load_iris(return_X_y=True)

    def adapt(est: Any) -> SklearnModelAdapter:
        return SklearnModelAdapter(
            est.fit(x, y),
            version="1",
            fingerprint="sha256:" + "1" * 64,
            size_bytes=None,
            load_seconds=0.0,
            device=__import__(
                "experionyx.adapters.capabilities", fromlist=["DeviceInfo"]
            ).DeviceInfo.cpu(),
        )

    lr = adapt(LogisticRegression(max_iter=200))
    arrays = lr.parameter_arrays()
    assert set(arrays) == {"coef_", "intercept_"} and arrays["coef_"].shape == (3, 4)
    arrays["coef_"][:] = 0  # the returned arrays are COPIES
    assert np.abs(lr.parameter_arrays()["coef_"]).sum() > 0
    derived = lr.with_parameters({"coef_": np.zeros((3, 4))})
    assert (
        np.abs(derived.parameter_arrays()["coef_"]).sum() == 0
        and np.abs(lr.parameter_arrays()["coef_"]).sum() > 0
    )
    with pytest.raises(Exception, match="shape"):
        lr.with_parameters({"coef_": np.zeros((2, 2))})
    with pytest.raises(Exception, match="not a safely accessible"):
        lr.with_parameters({"n_features_in_": np.zeros(1)})
    assert (
        adapt(RandomForestClassifier(n_estimators=3, random_state=0)).parameter_arrays() == {}
    )  # nothing safe to expose


# -- analysis helpers, against hand computations ------------------------------------------------------------------------------------


def rowmap(true: list[int], pred: list[int]) -> dict[int, dict[str, Any]]:
    return {i: {"index": i, "true": t, "predicted": p, "correct": t == p, "scores": None} for i, (t, p) in enumerate(zip(true, pred, strict=True))}  # fmt: skip


def test_paired_measure_matches_the_hand_computed_difference_and_direction() -> None:
    from experionyx.adapters.capabilities import TaskType

    true = [1, 0] * 20
    base = rowmap(true, true)  # accuracy 1.0
    stressed = rowmap(
        true, [t if i % 4 else 1 - t for i, t in enumerate(true)]
    )  # every 4th sample wrong: 0.75
    d = an.paired_measure(base, stressed, TaskType.CLASSIFICATION, None, min_members=10, confidence=0.95, resamples=100, permutations=200, seed=0)  # fmt: skip
    assert d["pairing"] == "PAIRED" and d["n_compared"] == 40 and d["measure"] == "accuracy"
    assert (d["mean_baseline"], d["mean_stressed"]) == (1.0, 0.75) and d[
        "absolute_change"
    ] == pytest.approx(-0.25)
    assert d["relative_change"] == pytest.approx(-0.25) and d["deterioration"] == pytest.approx(
        0.25
    )  # baseline - stressed (higher is better)
    assert d["direction"] == "HIGHER_IS_BETTER" and d["status"] == "DERIVED"
    want = st.compare(
        {str(i): 1.0 for i in range(40)},
        {str(i): float(stressed[i]["correct"]) for i in range(40)},
        pairing=st.Pairing.PAIRED
        if hasattr(st, "Pairing")
        else __import__("experionyx.interactions.taxonomy", fromlist=["Pairing"]).Pairing.PAIRED,
        method="percentile",
        confidence=0.95,
        resamples=100,
        permutations=200,
        seed=0,
    )
    assert d["inference"]["test"]["p_value"] == pytest.approx(want.test.p_value)
    assert d["inference"]["interval"]["estimate"] == pytest.approx(want.interval.estimate)
    same = an.paired_measure(base, rowmap(true, true), TaskType.CLASSIFICATION, None, min_members=10, confidence=0.95, resamples=100, permutations=200, seed=0)  # fmt: skip
    assert (
        same["absolute_change"] == 0
        and same["deterioration"] == 0
        and same["inference"]["test"]["p_value"] == 1.0
    )
    sub = an.paired_measure(base, stressed, TaskType.CLASSIFICATION, list(range(4)), min_members=10, confidence=0.95, resamples=100, permutations=200, seed=0)  # fmt: skip
    assert (
        sub["status"] == "INSUFFICIENT_EVIDENCE"
        and "min_members=10" in sub["reason"]
        and sub["n_compared"] == 4
    )


def test_regression_direction_zero_baseline_and_nonfinite_values() -> None:
    from experionyx.adapters.capabilities import TaskType

    def reg(true: list[float], pred: list[float]) -> dict[int, dict[str, Any]]:
        return {i: {"index": i, "true": t, "predicted": p, "correct": None, "scores": None} for i, (t, p) in enumerate(zip(true, pred, strict=True))}  # fmt: skip

    true = [float(i) for i in range(20)]
    base, worse = reg(true, true), reg(true, [t + 2.0 for t in true])
    d = an.paired_measure(base, worse, TaskType.REGRESSION, None, min_members=5, confidence=0.95, resamples=100, permutations=100, seed=0)  # fmt: skip
    assert (
        d["measure"] == "mae" and d["direction"] == "LOWER_IS_BETTER" and d["mean_baseline"] == 0.0
    )
    assert d["absolute_change"] == pytest.approx(2.0) and d["deterioration"] == pytest.approx(
        2.0
    )  # stressed - baseline (lower is better)
    assert (
        d["relative_change"] is None and "baseline value is 0" in d["relative_reason"]
    )  # a zero baseline has no relative change
    nan = reg(true, [t + 1.0 if i else float("nan") for i, t in enumerate(true)])
    e = an.paired_measure(base, nan, TaskType.REGRESSION, None, min_members=5, confidence=0.95, resamples=100, permutations=100, seed=0)  # fmt: skip
    assert (
        e["excluded_nonfinite_stressed"] == 1
        and e["pairing"] == "UNPAIRED"
        and "no per-sample pairing" in e["pairing_reason"]
    )
    assert (
        e["n_compared"] == 19
    )  # the non-finite sample is reported and excluded, never silently averaged in
    tiny = an.paired_measure(reg([1.0], [1.0]), reg([1.0], [2.0]), TaskType.REGRESSION, None, min_members=5, confidence=0.95, resamples=100, permutations=100, seed=0)  # fmt: skip
    assert tiny["status"] == "INSUFFICIENT_EVIDENCE" and tiny["inference"] is None


def test_shifted_sample_ids_and_different_ground_truth_are_never_paired() -> None:
    from experionyx.adapters.capabilities import TaskType

    true = [1, 0] * 10
    base = rowmap(true, true)
    shifted = {i + 100: {**r, "index": i + 100} for i, r in rowmap(true, true).items()}
    d = an.paired_measure(base, shifted, TaskType.CLASSIFICATION, None, min_members=5, confidence=0.95, resamples=50, permutations=100, seed=0)  # fmt: skip
    assert d["pairing"] == "UNPAIRED" and d["n_compared"] == 20
    relabeled = rowmap([1 - t for t in true], true)
    r = an.paired_measure(base, relabeled, TaskType.CLASSIFICATION, None, min_members=5, confidence=0.95, resamples=50, permutations=100, seed=0)  # fmt: skip
    assert (
        r["pairing"] == "UNPAIRED"
    )  # different ground truth: an altered label set is not a per-sample stress


def test_metric_deltas_keep_raw_values_and_apply_direction_per_metric() -> None:
    from types import SimpleNamespace

    from experionyx.evaluation.results import Status

    def m(mid: str, value: float | None, hib: bool | None) -> Any:
        return SimpleNamespace(
            metric_id=mid,
            value=value,
            status=Status.COMPUTED if value is not None else Status.UNDEFINED,
            n_samples=10,
            higher_is_better=hib,
            interval=None,
            structured=None,
        )

    base = SimpleNamespace(
        metrics=(
            m("accuracy", 0.9, True),
            m("mae", 0.0, False),
            m("latency", 1.0, None),
            m("f1", None, True),
        )
    )
    trial = SimpleNamespace(
        metrics=(
            m("accuracy", 0.6, True),
            m("mae", 0.5, False),
            m("latency", 3.0, None),
            m("f1", 0.4, True),
        )
    )
    got = {d["metric_id"]: d for d in an.metric_deltas(base, trial)}  # type: ignore[arg-type]
    assert got["accuracy"]["absolute_change"] == pytest.approx(-0.3) and got["accuracy"][
        "deterioration"
    ] == pytest.approx(0.3)
    assert (
        got["mae"]["absolute_change"] == 0.5
        and got["mae"]["deterioration"] == 0.5
        and got["mae"]["relative_change"] is None
    )  # zero baseline
    assert (
        got["latency"]["deterioration"] is None and got["latency"]["direction"] == "UNKNOWN"
    )  # no direction, no verdict
    assert (
        got["f1"]["absolute_change"] is None and got["f1"]["baseline_status"] == "UNDEFINED"
    )  # an undefined value stays undefined
    assert (
        got["accuracy"]["baseline_value"] == 0.9 and got["accuracy"]["stressed_value"] == 0.6
    )  # raw values preserved


def test_multiple_comparison_correction_and_aggregation_reuse_phase_10() -> None:
    docs: list[tuple[str, dict[str, Any]]] = [
        (f"u{i}", {"status": "DERIVED", "inference": {"test": {"p_value": p}}})
        for i, p in enumerate((0.01, 0.02, 0.04))
    ]
    docs.append(
        ("u3", {"status": "INSUFFICIENT_EVIDENCE", "inference": {"test": {"p_value": 0.5}}})
    )
    fam: dict[str, Any] = an.correct(docs, "BONFERRONI", 0.05)
    assert fam["n_hypotheses"] == 3 and fam["n_total"] == 4 and fam["excluded"] == ["u3"]
    assert [d["multiplicity"]["adjusted_p"] for _, d in docs[:3]] == pytest.approx(
        [0.03, 0.06, 0.12]
    )
    assert (
        docs[3][1]["multiplicity"]["raw_p"] is None
        and docs[0][1]["multiplicity"]["adjusted_p_below_alpha"] is True
    )
    none = an.correct([(k, dict(d)) for k, d in docs], "NONE", 0.05)
    assert none["method"] == "NONE"
    agg = an.aggregate([0.1, 0.2, 0.3], confidence=0.95, resamples=200, seed=1, method="percentile")
    assert (
        agg["n"] == 3
        and agg["values"] == [0.1, 0.2, 0.3]
        and agg["summary"]["mean"]["value"] == pytest.approx(0.2)
    )
    assert agg == an.aggregate(
        [0.1, 0.2, 0.3], confidence=0.95, resamples=200, seed=1, method="percentile"
    )  # deterministic
    one = an.aggregate([0.4], confidence=0.95, resamples=200, seed=1, method="percentile")
    assert one["interval"]["status"] == "INCONCLUSIVE" and "no spread" in one["interval"]["reason"]


def test_stability_is_observed_never_manufactured() -> None:
    r = rowmap([1, 0, 1, 0], [1, 0, 1, 0])
    same = an.stability([r, r, r], ["sha256:a"] * 3, [{"accuracy": 1.0}] * 3)
    assert same["observed"] == "DETERMINISTIC" and same[
        "prediction_agreement_with_first_repeat"
    ] == [1.0, 1.0]
    other = rowmap([1, 0, 1, 0], [1, 1, 1, 0])
    diff = an.stability(
        [r, other], ["sha256:a", "sha256:b"], [{"accuracy": 1.0}, {"accuracy": 0.75}]
    )
    assert diff["observed"] == "NONDETERMINISTIC" and diff[
        "prediction_agreement_with_first_repeat"
    ] == [0.75]
    assert an.stability([r], ["x"], [{}])["observed"] == "INSUFFICIENT_EVIDENCE"
    lat = an.latency_variation([1.0, 1.0, 1.0])
    assert lat["coefficient_of_variation"] == 0.0 and "never used as evidence" in lat["note"]


def test_no_stress_document_carries_a_score_or_a_causal_claim() -> None:
    d = StressDesign(
        MODEL,
        DATA,
        StressPlan((StressSpec("THRESHOLD", {"threshold": 0.4}),)),
        EvaluationConfig(split="test"),
    )
    text = json.dumps([d.to_dict(), an.NOTE]).lower()
    for banned in ("robustness_score", "stress_score", "overall", "grade", "verdict", "caused"):
        assert banned not in text.replace("robustness score", ""), banned
    assert "not a robustness score" in an.NOTE and "does not say why" in an.NOTE
    assert pure_registries() is not None and copy.copy(N) == N

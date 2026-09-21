"""Calibration & uncertainty analyses end to end on the controlled linear world (see
calibration_helpers): real registered model and dataset, real baseline evaluation, real analysis Runs.
Expected numbers are recomputed with independently written NumPy formulas from the STORED rows, never
through `experionyx.calibration`. Engineering validation of the machinery, not findings about a model."""

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import experionyx.calibration.analysis as an
import experionyx.calibration.engine as ce
import experionyx.calibration.measures as m
from calibration_helpers import (
    SPLITS,
    analyze,
    cal_world,
    extra_run,
    make_spec,
    registry,
    spec_dict,
    stored_rows,
)
from eval_helpers import EvalWorld
from experionyx.adapters.records import RegisteredModel
from experionyx.calibration.entities import CalibrationAnalysis, CalibrationResult
from experionyx.domain import Artifact, Investigation, Run
from experionyx.errors import ArtifactIntegrityError, ExperionyxError, ValidationError
from experionyx.failures.entities import FailureMode, FailureSignal
from experionyx.interactions.samples import _rows
from stress_helpers import data as linear_data
from stress_helpers import stress_world


@pytest.fixture(scope="module")
def cw(tmp_path_factory: pytest.TempPathFactory) -> tuple[EvalWorld, str]:
    return cal_world(tmp_path_factory.mktemp("cal"))


def independent(rows: list[dict[str, Any]], bins: int) -> dict[str, float]:
    """Top-label metrics from the stored rows, written independently (NumPy, digitize)."""
    conf = np.array([r["scores"][int(r["predicted"])] for r in rows])
    ok = np.array([r["predicted"] == r["true"] for r in rows], dtype=float)
    idx = np.digitize(conf, [k / bins for k in range(1, bins)])
    ece, mce = 0.0, 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            gap = abs(ok[mask].mean() - conf[mask].mean())
            ece += mask.mean() * gap
            mce = max(mce, gap)
    y = np.array([r["true"] for r in rows])
    p = np.array([r["scores"] for r in rows])
    onehot = np.eye(p.shape[1])[y]
    return {
        "accuracy": float(ok.mean()), "mean_confidence": float(conf.mean()), "ece": float(ece), "mce": float(mce),
        "brier_top_label": float(((conf - ok) ** 2).mean()),
        "log_loss_top_label": float(-(ok * np.log(np.clip(conf, 1e-15, 1)) + (1 - ok) * np.log(np.clip(1 - conf, 1e-15, 1))).mean()),
        "brier_multiclass": float(((p - onehot) ** 2).sum(axis=1).mean()),
        "nll_multiclass": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean()),
    }  # fmt: skip


# -- the baseline analysis ---------------------------------------------------------------------------


def test_metrics_bins_and_artifacts_match_an_independent_recomputation(cw: tuple[EvalWorld, str]) -> None:  # fmt: skip
    w, base = cw
    a = analyze(w, base, objects=["TOP_LABEL"])
    cr = registry(w)
    rows = stored_rows(w, base)
    want = independent(rows, 10)
    s = a.summary["baseline"]
    assert a.analysis_status == "COMPLETE" and s["status"] == "COMPUTED"
    for k in ("accuracy", "mean_confidence", "ece", "mce", "brier_top_label", "log_loss_top_label"):
        assert s["metrics"][k] == pytest.approx(want[k], abs=1e-12), k
    assert s["probability_vector"]["brier_multiclass"] == pytest.approx(want["brier_multiclass"])
    assert s["probability_vector"]["nll_multiclass"] == pytest.approx(want["nll_multiclass"])
    assert s["metrics"]["overconfidence"] == pytest.approx(
        want["mean_confidence"] - want["accuracy"]
    )
    # the artifacts exist, are digest-verified, and keep the raw evidence (not only the final numbers)
    docs = {n: cr.document(a.id, n) for n in ce.DOCUMENTS}
    assert {x.path for x in cr.artifacts(a.id)} == {f"calibration/{n}.json" for n in ce.DOCUMENTS}
    bins = docs["bins"]["baseline"]["top_label"]
    assert len(bins) == 10 and sum(b["count"] for b in bins) == len(
        rows
    )  # empty bins are not hidden
    assert any(b["count"] == 0 for b in bins) or all(b["evidence"] != "EMPTY" for b in bins)
    for b in bins:
        assert (b["upper"] - b["lower"]) == pytest.approx(0.1) and b["upper_closed"] == (b["index"] == 9)  # fmt: skip
    pred = docs["predictions"]
    assert [r["id"] for r in pred["rows"]] == sorted(r["index"] for r in rows)
    assert pred["rows"][0]["scores"] and "entropy_nats" in pred["rows"][0] and pred["prediction_representation"] == "PREDICT_PROBA"  # fmt: skip
    assert docs["metrics"]["definitions"]["caveat"].startswith("ECE and MCE depend on the binning")
    assert "universal" in docs["metrics"]["definitions"]["caveat"]
    spec_doc = docs["spec"]
    assert spec_doc["prediction_representation"] == "PREDICT_PROBA" and "predict_proba" in spec_doc["score_assumption"]  # fmt: skip
    assert (
        spec_doc["target_representation"]["classes"] == [0, 1]
        and spec_doc["bootstrap"]["seed"] == 0
    )
    assert spec_doc["provenance_fingerprint"] == a.provenance_fingerprint
    (res,) = [r for r in cr.results(a.id) if r.context_key == "baseline"]
    assert res.status == "COMPUTED" and res.n_samples == len(rows) and res.headline["ece"] == s["metrics"]["ece"]  # fmt: skip


def test_the_bootstrap_intervals_are_recorded_with_method_seed_and_assumptions(cw: tuple[EvalWorld, str]) -> None:  # fmt: skip
    w, base = cw
    a = analyze(w, base, statistics={"seed": 7, "resamples": 50, "confidence": 0.9})
    u = registry(w).document(a.id, "uncertainty")["contexts"]["baseline"]
    boot = u["uncertainty"]["bootstrap"]
    assert boot["method"] == "bootstrap-percentile" and boot["resamples"] == 50 and boot["seed"] == 7 and boot["confidence"] == 0.9  # fmt: skip
    assert any("independent and identically distributed" in x for x in boot["assumptions"])
    assert any("positively biased" in x for x in boot["assumptions"])
    for name in (
        "ece",
        "mce",
        "accuracy",
        "brier_top_label",
        "log_loss_top_label",
        "brier_multiclass",
    ):
        iv = boot["metrics"][name]
        assert iv["status"] == "DERIVED" and iv["lower"] <= iv["upper"] and iv["seed"] == 7
    acc = u["uncertainty"]["accuracy"]
    assert acc["method"] == "wilson" and acc["lower"] <= acc["estimate"] <= acc["upper"]
    assert a.spec["statistics"]["interval_method"] == "bootstrap-percentile"


def test_a_different_bootstrap_seed_is_a_different_analysis_with_different_intervals(cw: tuple[EvalWorld, str]) -> None:  # fmt: skip
    w, base = cw
    a1 = analyze(w, base, statistics={"seed": 1})
    a2 = analyze(w, base, statistics={"seed": 2})
    assert a1.id != a2.id and a1.spec_id != a2.spec_id
    i1 = registry(w).document(a1.id, "uncertainty")["contexts"]["baseline"]["uncertainty"]["bootstrap"]["metrics"]["ece"]  # fmt: skip
    i2 = registry(w).document(a2.id, "uncertainty")["contexts"]["baseline"]["uncertainty"]["bootstrap"]["metrics"]["ece"]  # fmt: skip
    assert i1["estimate"] == i2["estimate"] and (i1["lower"], i1["upper"]) != (i2["lower"], i2["upper"])  # fmt: skip


def test_the_same_request_is_the_same_record_and_writes_nothing_twice(cw: tuple[EvalWorld, str]) -> None:  # fmt: skip
    w, base = cw
    a1 = analyze(w, base, binning={"n_bins": 7})
    n_a, n_r = len(w.registry.find(CalibrationAnalysis)), len(w.registry.find(CalibrationResult))
    a2 = analyze(w, base, binning={"n_bins": 7})
    assert a1.id == a2.id and a1.provenance_fingerprint == a2.provenance_fingerprint
    assert (len(w.registry.find(CalibrationAnalysis)), len(w.registry.find(CalibrationResult))) == (n_a, n_r)  # fmt: skip
    changed = analyze(w, base, binning={"n_bins": 8})
    assert changed.id != a1.id and changed.provenance_fingerprint != a1.provenance_fingerprint


def test_uniform_and_quantile_binning_and_their_evidence_rules(cw: tuple[EvalWorld, str]) -> None:
    w, base = cw
    q = analyze(w, base, binning={"strategy": "QUANTILE", "n_bins": 6})
    qb = registry(w).document(q.id, "bins")["baseline"]["top_label"]
    counts = [b["count"] for b in qb]
    assert (
        sum(counts) == 60 and len(qb) <= 6 and max(counts) - min(counts) <= 60 // 6 + 6
    )  # roughly equal-count
    assert all(
        qb[i]["upper"] == qb[i + 1]["lower"] for i in range(len(qb) - 1)
    )  # the bins tile the range
    assert q.spec["binning"]["strategy"] == "QUANTILE"
    # quantile bins are refused where they are statistically invalid (fewer than 5 per requested bin)
    thin = analyze(w, base, binning={"strategy": "QUANTILE", "n_bins": 20})
    r = thin.summary["baseline"]
    assert r["status"] == "INSUFFICIENT_EVIDENCE" and r["metrics"] is None and thin.analysis_status == "PARTIAL"  # fmt: skip
    assert registry(w).document(thin.id, "metrics")["contexts"]["baseline"]["reason"].startswith("quantile binning needs")  # fmt: skip


def test_ece_depends_on_the_binning_and_the_analysis_says_so(cw: tuple[EvalWorld, str]) -> None:
    w, base = cw
    rows = stored_rows(w, base)
    e5 = analyze(w, base, binning={"n_bins": 5}).summary["baseline"]["metrics"]["ece"]
    e15 = analyze(w, base, binning={"n_bins": 15}).summary["baseline"]["metrics"]["ece"]
    assert e5 == pytest.approx(independent(rows, 5)["ece"]) and e15 == pytest.approx(independent(rows, 15)["ece"])  # fmt: skip
    assert e5 != pytest.approx(e15)


def test_classwise_and_top_label_are_reported_as_different_objects(cw: tuple[EvalWorld, str]) -> None:  # fmt: skip
    w, base = cw
    a = analyze(w, base, objects=["CLASSWISE", "TOP_LABEL"], binning={"n_bins": 5})
    cr = registry(w)
    met = cr.document(a.id, "metrics")["contexts"]["baseline"]["metrics"]
    assert set(met) == {"top_label", "probability_vector", "classwise"}
    rows = stored_rows(w, base)
    cw_doc = met["classwise"]
    assert cw_doc["classes_total"] == 2 and set(cw_doc["per_class"]) == {"0", "1"}
    for k in (0, 1):  # classwise ECE of class k against the event 'true class is k' (independent)
        p = np.array([r["scores"][k] for r in rows])
        ev = np.array([r["true"] == k for r in rows], dtype=float)
        idx = np.digitize(p, [j / 5 for j in range(1, 5)])
        want = sum((idx == b).mean() * abs(ev[idx == b].mean() - p[idx == b].mean()) for b in range(5) if (idx == b).any())  # fmt: skip
        assert cw_doc["per_class"][str(k)]["ece"] == pytest.approx(want)
        assert cw_doc["per_class"][str(k)]["status"] == "COMPUTED" and cw_doc["per_class"][str(k)]["positives"] == int(ev.sum())  # fmt: skip
    assert set(cr.document(a.id, "bins")["baseline"]["classwise"]) == {
        "0",
        "1",
    }  # per-class bins are kept
    assert set(met) == {"top_label", "probability_vector", "classwise"}  # three NAMED objects, never one "calibration" number  # fmt: skip
    only = analyze(w, base, objects=["TOP_LABEL"])
    assert registry(w).document(only.id, "metrics")["contexts"]["baseline"]["metrics"]["classwise"] is None  # fmt: skip


def test_a_class_with_too_few_positives_is_insufficient_not_a_misleading_number(cw: tuple[EvalWorld, str]) -> None:  # fmt: skip
    w, base = cw
    a = analyze(w, base, objects=["CLASSWISE"], statistics={"min_class_positives": 40})
    pc = registry(w).document(a.id, "metrics")["contexts"]["baseline"]["metrics"]["classwise"]["per_class"]  # fmt: skip
    assert any(v["status"] == "INSUFFICIENT_EVIDENCE" and v["ece"] is None and "min_class_positives" in v["reason"] for v in pc.values())  # fmt: skip


def test_prediction_uncertainty_is_descriptive_and_unsupported_kinds_are_unavailable(cw: tuple[EvalWorld, str]) -> None:  # fmt: skip
    w, base = cw
    a = analyze(w, base)
    pu = registry(w).document(a.id, "uncertainty")["contexts"]["baseline"]["prediction_uncertainty"]
    rows = stored_rows(w, base)
    p = np.array([r["scores"] for r in rows])
    ent = -(p * np.log(np.clip(p, 1e-300, 1))).sum(axis=1)
    assert pu["status"] == "COMPUTED"
    assert pu["predictive_entropy_nats"]["mean"] == pytest.approx(float(ent.mean()))
    assert pu["normalized_entropy"]["mean"] == pytest.approx(
        float((ent / math.log(2)).mean())
    )  # raw != normalized
    assert pu["top2_margin"]["mean"] == pytest.approx(float(np.abs(p[:, 0] - p[:, 1]).mean()))
    assert pu["predictive_entropy_nats"]["mean"] != pytest.approx(pu["normalized_entropy"]["mean"])
    assert "H / ln(K)" in pu["formulas"]["normalized_entropy"]
    for k in ("stochastic_trials", "ensemble_disagreement", "epistemic_aleatoric"):
        assert pu[k]["status"] == "UNAVAILABLE" and pu[k]["reason"]
    assert "NOT an epistemic" in pu["interpretation"]
    assert a.summary["prediction_uncertainty"] == dict.fromkeys(("stochastic_trials", "ensemble_disagreement", "epistemic_aleatoric"), "UNAVAILABLE")  # fmt: skip


def test_no_universal_score_exists_anywhere_in_an_analysis(cw: tuple[EvalWorld, str]) -> None:
    w, base = cw
    a = analyze(w, base, objects=["CLASSWISE", "TOP_LABEL"])

    def keys(x: Any) -> set[str]:
        if isinstance(x, dict):
            return set(x) | {k for v in x.values() for k in keys(v)}
        return {k for v in x for k in keys(v)} if isinstance(x, list) else set()

    found = set().union(*(keys(registry(w).document(a.id, n)) for n in ce.DOCUMENTS))
    assert not found & {"overall_score", "calibration_score", "uncertainty_score", "reliability_score", "composite", "rank", "ranking", "verdict"}  # fmt: skip
    assert not {k for k in found if k.endswith("_score")}


# -- post-hoc calibration: protocol, leakage, before/after ----------------------------------------------


@pytest.mark.parametrize("method", ["PLATT", "ISOTONIC"])
def test_a_split_protocol_calibrates_on_one_part_and_reports_on_the_disjoint_other(cw: tuple[EvalWorld, str], method: str) -> None:  # fmt: skip
    w, base = cw
    before_reg = w.registry.get(
        RegisteredModel, next(mm.id for mm in w.registry.find(RegisteredModel))
    )
    baseline_digest = {x.path: x.digest for x in w.registry.find(Artifact, run_id=base)}
    a = analyze(w, base, method=method, fit={"mode": "SPLIT", "fraction": 0.5, "seed": 3})
    cr = registry(w)
    mdoc = cr.document(a.id, "metrics")["calibration_method"]
    proto = mdoc["protocol"]
    cal_ids, ev_ids = proto["calibration_ids"], proto["evaluation_ids"]
    assert proto["overlap"] == 0 and not set(cal_ids) & set(ev_ids) and sorted(cal_ids + ev_ids) == list(range(60))  # fmt: skip
    assert proto["calibration_data"]["ids_digest"] == an.digest_of(cal_ids) and proto["evaluation_data"]["ids_digest"] == an.digest_of(ev_ids)  # fmt: skip
    assert proto["leakage"].startswith("NONE") and (proto["seed"], proto["fraction"]) == (3, 0.5)
    assert cal_ids == ce.split_ids(list(range(60)), 0.5, 3)[0]
    rows = {r["index"]: r for r in stored_rows(w, base)}
    # 'before' is the raw model on the held-out part, recomputed independently
    held = [rows[i] for i in ev_ids]
    before = mdoc["before"]["metrics"]["top_label"]
    assert before["ece"] == pytest.approx(independent(held, 10)["ece"]) and mdoc["before"]["n_samples"] == len(ev_ids)  # fmt: skip
    # 'after' applies the recorded parameters to the same held-out samples (independent formula)
    params = mdoc["parameters"]
    conf = np.array([r["scores"][int(r["predicted"])] for r in held])
    ok = np.array([r["predicted"] == r["true"] for r in held], dtype=float)
    if method == "PLATT":
        z = (
            params["a"]
            * np.log(np.clip(conf, 1e-15, 1 - 1e-15) / (1 - np.clip(conf, 1e-15, 1 - 1e-15)))
            + params["b"]
        )
        cal = 1 / (1 + np.exp(-z))
    else:
        cal = np.interp(conf, params["x"], params["y"])
    idx = np.digitize(cal, [k / 10 for k in range(1, 10)])
    want = sum((idx == b).mean() * abs(ok[idx == b].mean() - cal[idx == b].mean()) for b in range(10) if (idx == b).any())  # fmt: skip
    after = mdoc["after"]
    assert after["metrics"]["top_label"]["ece"] == pytest.approx(want, abs=1e-9)
    assert (
        after["metrics"]["probability_vector"]["status"] == "UNAVAILABLE"
    )  # no calibrated vector is claimed
    assert after["n_samples"] == len(ev_ids) and "calibrated" in {
        r.context_key for r in cr.results(a.id)
    }
    # the comparison is paired, keeps raw metric + difference + effect size + p + interval
    cmp_ = cr.document(a.id, "comparisons")["comparisons"]["before vs after calibration"]
    assert cmp_["family"] == "POST_HOC_CALIBRATION" and cmp_["pairing"] == "PAIRED" and cmp_["status"] == "COMPUTED"  # fmt: skip
    e = cmp_["metrics"]["ece"]
    assert e["a"] == pytest.approx(before["ece"]) and e["b"] == pytest.approx(after["metrics"]["top_label"]["ece"])  # fmt: skip
    assert e["difference"] == pytest.approx(e["b"] - e["a"]) and e["interval"]["method"] == "bootstrap-percentile"  # fmt: skip
    assert e["p_value"] is not None and e["effect_size"]["value"] == pytest.approx(e["difference"])
    assert e["practically_material"] in (True, False) and cmp_["metrics"]["brier_top_label"]["effect_size"]["name"]  # fmt: skip
    # the original model and the baseline run are untouched
    assert w.registry.get(RegisteredModel, before_reg.id) == before_reg
    assert {x.path: x.digest for x in w.registry.find(Artifact, run_id=base)} == baseline_digest
    assert mdoc["model"]["immutable"].startswith("the model is never loaded")
    assert a.analysis_status == "COMPLETE"


def test_calibrators_are_deterministic_and_the_seed_changes_the_calibration_data(cw: tuple[EvalWorld, str]) -> None:  # fmt: skip
    w, base = cw
    p1 = analyze(w, base, method="PLATT", fit={"seed": 1}).summary["method"]["parameters"]
    p1b = analyze(w, base, method="PLATT", fit={"seed": 1}).summary["method"]["parameters"]
    p2 = analyze(w, base, method="PLATT", fit={"seed": 2}).summary["method"]["parameters"]
    assert p1 == p1b and p1 != p2


def test_calibrating_on_the_evaluated_run_or_its_split_is_refused_as_leakage_before_anything_runs(tmp_path: Path) -> None:  # fmt: skip
    w, base = cal_world(tmp_path)
    inv = w.registry.find(Investigation)[0]
    before = len(w.registry.find(Run))
    with pytest.raises(ValidationError, match="leakage"):
        make_spec(base, method="PLATT", fit={"mode": "RUN", "calibration_run": base})
    twin = extra_run(w, "test", seed=1)  # the same dataset fingerprint AND split: the same samples
    spec = make_spec(base, method="PLATT", fit={"mode": "RUN", "calibration_run": twin})
    with pytest.raises(ce.CalibrationError, match="LEAKAGE"):
        ce.run_calibration_request(w.registry, w.store, w.executor, inv.id, spec)
    assert len(w.registry.find(Run)) == before + 1  # only the extra evaluation run; nothing was executed for the analysis  # fmt: skip
    assert not w.registry.find(CalibrationAnalysis)


def test_a_separate_calibration_run_fits_on_it_and_evaluates_on_the_baseline(
    tmp_path: Path,
) -> None:
    w, base = cal_world(tmp_path)
    calib = extra_run(w, "calib")
    a = analyze(w, base, method="ISOTONIC", fit={"mode": "RUN", "calibration_run": calib})
    mdoc = registry(w).document(a.id, "metrics")["calibration_method"]
    proto = mdoc["protocol"]
    assert proto["mode"] == "RUN" and proto["overlap"] == 0
    assert proto["calibration_data"]["source"] == f"run {calib}" and proto["evaluation_data"]["source"] == f"run {base}"  # fmt: skip
    assert proto["calibration_data"]["split"] == "calib" and proto["evaluation_data"]["split"] == "test"  # fmt: skip
    assert proto["calibration_data"]["n"] == 60 and mdoc["after"]["n_samples"] == 60
    assert proto["calibration_data"]["predictions_digest"] != proto["evaluation_data"]["predictions_digest"]  # fmt: skip
    # the fit used ONLY the calibration run's rows: refit independently from them
    rows = stored_rows(w, calib)
    pairs = [(r["scores"][int(r["predicted"])], r["predicted"] == r["true"]) for r in rows]
    assert mdoc["parameters"] == m.fit_isotonic(pairs)
    assert a.spec["fit"] == {"mode": "RUN", "calibration_run": calib}
    lineage = registry(w).document(a.id, "spec")["lineage"]["calibration_run"]
    assert lineage["run_id"] == calib and lineage["split"] == "calib"


def test_a_calibration_run_of_another_model_or_class_set_is_refused(tmp_path: Path) -> None:
    w, base = cal_world(tmp_path)
    inv = w.registry.find(Investigation)[0]
    calib = extra_run(w, "calib")
    real = ce.load_run(w.registry, w.store, calib)
    other = ce.Loaded(**{**real.__dict__, "model_fingerprint": "sha256:" + "0" * 64})
    with pytest.raises(ce.CalibrationError, match="model fingerprint"):
        ce.check_calibration_data(ce.load_run(w.registry, w.store, base), other)
    assert inv


def test_too_little_calibration_data_yields_insufficient_evidence_not_a_calibrator(cw: tuple[EvalWorld, str]) -> None:  # fmt: skip
    w, base = cw
    a = analyze(
        w, base, method="PLATT", statistics={"min_samples": 40}
    )  # the 30-sample halves are too small
    s = a.summary
    assert a.analysis_status == "PARTIAL" and s["method"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert s["method"]["parameters"] is None and "calibration observation" in s["method"]["reason"]
    assert s["contexts"]["calibrated"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert (
        "before vs after calibration"
        not in registry(w).document(a.id, "comparisons")["comparisons"]
    )


def test_a_single_outcome_calibration_set_is_not_fitted() -> None:
    same = [m.Obs(i, 0, 0, (0.9, 0.1)) for i in range(40)]  # every prediction correct
    params, why = ce.fit_calibrator("PLATT", same, 10)
    assert params is None and "single outcome" in why  # type: ignore[operator]
    assert ce.fit_calibrator("ISOTONIC", same, 10)[0] is None


# -- slices ------------------------------------------------------------------------------------------


def test_slices_are_evaluated_only_when_requested_with_membership_counts_and_evidence(cw: tuple[EvalWorld, str]) -> None:  # fmt: skip
    w, base = cw
    plain = analyze(w, base)
    assert not [k for k in plain.summary["contexts"] if k.startswith("slice:")]  # never automatic
    pos = {"name": "pos", "condition": {"op": "range", "field": "feature:x0", "low": 0.0}}
    tiny = {"name": "tiny", "condition": {"op": "range", "field": "feature:x0", "low": 1.85}}
    a = analyze(w, base, slices=[pos, tiny], statistics={"correction": "BONFERRONI"})
    cr = registry(w)
    rows = stored_rows(w, base)
    xs = {i: linear_data()["rows"][i][0] for i in range(60)}
    members = sorted(r["index"] for r in rows if xs[r["index"]] >= 0.0)
    ctx = a.summary["contexts"]
    assert ctx["slice:pos"]["status"] == "COMPUTED" and ctx["slice:pos"]["n_samples"] == len(
        members
    )
    tiny_ids = sorted(i for i in xs if xs[i] >= 1.85)
    assert 0 < len(tiny_ids) < 20
    assert ctx["slice:tiny"]["status"] == "INSUFFICIENT_EVIDENCE" and ctx["slice:tiny"]["headline"] is None  # no misleading metric  # fmt: skip
    pred = cr.document(a.id, "predictions")["contexts"]
    assert (
        pred["slice:pos"]["sample_ids"] == members and pred["slice:tiny"]["sample_ids"] == tiny_ids
    )
    met = cr.document(a.id, "metrics")["contexts"]
    assert met["slice:tiny"]["metrics"]["top_label"] is None and "min_samples=20" in met["slice:tiny"]["reason"]  # fmt: skip
    assert cr.document(a.id, "bins")["slice:tiny"][
        "top_label"
    ]  # the bins are still shown, not hidden
    want = independent([r for r in rows if r["index"] in set(members)], 10)
    assert ctx["slice:pos"]["headline"]["ece"] == pytest.approx(want["ece"])
    assert cr.document(a.id, "uncertainty")["contexts"]["slice:pos"]["uncertainty"]["bootstrap"]
    assert a.analysis_status == "PARTIAL"
    comps = cr.document(a.id, "comparisons")["comparisons"]
    assert comps["slice:pos vs REST"]["status"] == "COMPUTED" and comps["slice:tiny vs REST"]["status"] == "INSUFFICIENT_EVIDENCE"  # fmt: skip


def test_overlapping_slices_are_each_evaluated_independently(cw: tuple[EvalWorld, str]) -> None:
    w, base = cw
    s1 = {"name": "a", "condition": {"op": "range", "field": "feature:x0", "low": -1.0}}
    s2 = {"name": "b", "condition": {"op": "range", "field": "feature:x0", "low": 0.0}}
    a = analyze(w, base, slices=[s1, s2])
    pred = registry(w).document(a.id, "predictions")["contexts"]
    assert set(pred["slice:b"]["sample_ids"]) < set(
        pred["slice:a"]["sample_ids"]
    )  # nested, both kept
    assert a.summary["contexts"]["slice:a"]["status"] == a.summary["contexts"]["slice:b"]["status"] == "COMPUTED"  # fmt: skip


def test_a_slice_over_a_missing_field_is_unavailable_not_guessed(cw: tuple[EvalWorld, str]) -> None:
    w, base = cw
    bad = {"name": "ghost", "condition": {"op": "range", "field": "feature:nope", "low": 0.0}}
    a = analyze(w, base, slices=[bad])
    assert a.summary["contexts"]["slice:ghost"]["status"] == "UNAVAILABLE" and a.analysis_status == "PARTIAL"  # fmt: skip


def test_multiple_comparison_correction_keeps_raw_adjusted_and_the_family(cw: tuple[EvalWorld, str]) -> None:  # fmt: skip
    w, base = cw
    sl = [{"name": f"s{k}", "condition": {"op": "range", "field": "feature:x0", "low": k * 0.3 - 1.0, "high": k * 0.3 - 0.3}} for k in range(4)]  # fmt: skip
    a = analyze(w, base, slices=sl, statistics={"correction": "BONFERRONI", "min_samples": 5})
    comps = registry(w).document(a.id, "comparisons")
    fams = [k for k in comps["corrections"] if k.startswith("SLICE/")]
    assert {"SLICE/ece", "SLICE/accuracy", "SLICE/brier_top_label"} <= set(fams)
    e = comps["corrections"]["SLICE/ece"]
    assert e["method"] == "BONFERRONI" and e["n_hypotheses"] == 4 and "family-wise" in e["controls"]
    for name, c in comps["comparisons"].items():
        if c["family"] != "SLICE" or c["status"] != "COMPUTED":
            continue
        r = c["metrics"]["ece"]
        assert r["adjusted_p_value"] == pytest.approx(min(1.0, r["p_value"] * 4)), name
        assert r["adjusted_p_value"] >= r["p_value"] and r["significant_after_correction"] in (
            True,
            False,
        )
        assert r["practically_material"] in (True, False)  # separate from significance
        assert r["interval"]["method"] == "bootstrap-percentile" and r["effect_size"]["value"] == pytest.approx(r["difference"])  # fmt: skip
        assert c["pairing"] == "UNPAIRED"


# -- temporal / distribution windows -------------------------------------------------------------------


def test_windows_compare_calibration_confidence_and_probability_distributions(tmp_path: Path) -> None:  # fmt: skip
    w, base = cal_world(tmp_path, splits={"test": list(range(120))})
    win = {"ordering": {"field": "index"}, "windows": [{"start": 0, "end": 60, "name": "early"}, {"start": 60, "end": 120, "name": "late"}]}  # fmt: skip
    a = analyze(w, base, windows=win)
    cr = registry(w)
    rows = stored_rows(w, base)
    c = cr.document(a.id, "comparisons")["comparisons"]["window:early vs window:late"]
    assert c["status"] == "COMPUTED" and c["pairing"] == "UNPAIRED" and c["family"] == "WINDOW"
    early = [r for r in rows if r["index"] < 60]
    late = [r for r in rows if r["index"] >= 60]
    ee, el = independent(early, 10), independent(late, 10)
    assert c["metrics"]["ece"]["a"] == pytest.approx(ee["ece"]) and c["metrics"]["ece"]["b"] == pytest.approx(el["ece"])  # fmt: skip
    assert c["metrics"]["ece"]["difference"] == pytest.approx(el["ece"] - ee["ece"])
    assert c["distribution"]["accuracy_change"] == pytest.approx(el["accuracy"] - ee["accuracy"])
    ca = np.array([r["scores"][int(r["predicted"])] for r in early])
    cb = np.array([r["scores"][int(r["predicted"])] for r in late])
    grid = np.unique(np.concatenate([ca, cb]))
    ks = max(abs((ca <= g).mean() - (cb <= g).mean()) for g in grid)
    assert c["distribution"]["confidence_ks"] == pytest.approx(ks)
    assert c["distribution"]["predicted_class_counts"]["a"] == [sum(r["predicted"] == k for r in early) for k in (0, 1)]  # fmt: skip
    assert "not causal" in c["distribution"]["note"] and "does not say why" in c["note"]
    wctx = a.summary["contexts"]
    assert wctx["window:early"]["n_samples"] == 60 and wctx["window:late"]["n_samples"] == 60
    wm = cr.document(a.id, "metrics")["contexts"]["window:late"]
    assert wm["kind"] == "WINDOW"
    win_doc = cr.document(a.id, "bins")["window:late"]["top_label"]
    assert sum(b["count"] for b in win_doc) == 60


def test_window_by_a_feature_ordering_and_by_key_not_by_row_position(tmp_path: Path) -> None:
    w, base = cal_world(tmp_path, splits={"test": list(range(120))})
    win = {"ordering": {"field": "feature:x1"}, "windows": [{"start": -3.0, "end": 0.0, "name": "neg"}, {"start": 0.0, "end": 3.0, "name": "pos"}]}  # fmt: skip
    a = analyze(w, base, windows=win)
    x1 = {i: linear_data()["rows"][i][1] for i in range(120)}
    pred = registry(w).document(a.id, "predictions")["contexts"]
    assert pred["window:neg"]["sample_ids"] == sorted(i for i in x1 if -3.0 <= x1[i] < 0.0)
    assert pred["window:pos"]["sample_ids"] == sorted(i for i in x1 if 0.0 <= x1[i] < 3.0)


def test_a_window_with_too_few_observations_is_insufficient_and_an_empty_one_unavailable(tmp_path: Path) -> None:  # fmt: skip
    w, base = cal_world(tmp_path, splits={"test": list(range(120))})
    win = {"ordering": {"field": "index"}, "windows": [{"start": 0, "end": 90, "name": "big"}, {"start": 90, "end": 100, "name": "small"}, {"start": 500, "end": 600, "name": "void"}]}  # fmt: skip
    a = analyze(w, base, windows=win)
    ctx = a.summary["contexts"]
    assert ctx["window:big"]["status"] == "COMPUTED" and ctx["window:small"]["status"] == "INSUFFICIENT_EVIDENCE"  # fmt: skip
    assert ctx["window:void"]["status"] == "UNAVAILABLE" and ctx["window:void"]["n_samples"] == 0
    comps = registry(w).document(a.id, "comparisons")["comparisons"]
    assert comps["window:big vs window:small"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert comps["window:big vs window:void"]["status"] == "UNAVAILABLE"
    assert a.analysis_status == "PARTIAL"


# -- data quality as evidence ---------------------------------------------------------------------------


def patched(monkeypatch: pytest.MonkeyPatch, fn: Any) -> None:
    orig = _rows
    monkeypatch.setattr(ce, "prediction_rows", lambda reg, store, run_id: iter(fn(list(orig(reg, store, run_id)))))  # fmt: skip


def test_invalid_probabilities_missing_scores_and_duplicate_ids_are_surfaced_not_dropped(cw: tuple[EvalWorld, str], monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    w, base = cw

    def damage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows[0] = {**rows[0], "scores": [0.9, 0.9]}  # does not sum to 1
        rows[1] = {**rows[1], "scores": [1.5, -0.5]}  # outside [0, 1]
        rows[2] = {**rows[2], "scores": None}  # missing
        rows[3] = {**rows[3], "scores": [float("nan"), 0.5]}
        rows[4] = {**rows[4], "true": 9}  # invalid target
        rows.append({**rows[10]})  # the same sample ID listed twice
        return rows

    patched(monkeypatch, damage)
    ld = ce.load_run(w.registry, w.store, base)
    reasons = {x["id"]: x["reason"] for x in ld.invalid}
    ids = sorted(r["index"] for r in stored_rows(w, base))
    assert reasons == {ids[0]: "NOT_NORMALIZED", ids[1]: "OUT_OF_RANGE", ids[2]: "MISSING_SCORES", ids[3]: "NON_FINITE", ids[4]: "INVALID_TARGET"}  # fmt: skip
    assert (
        ld.duplicates == [ids[10]] and ids[10] not in ld.obs
    )  # ambiguous rows are set aside, all of them
    assert ld.n_rows == 61 and len(ld.obs) + len(ld.invalid) + 2 * len(ld.duplicates) == 61
    a = analyze(
        w, base, binning={"n_bins": 12}
    )  # a spec not analyzed before: identity is the request
    dq = a.summary["data_quality"]
    assert a.analysis_status == "PARTIAL" and dq["affected"] and dq["n_duplicate_ids"] == 1
    assert dq["invalid_by_reason"] == {"INVALID_TARGET": 1, "MISSING_SCORES": 1, "NON_FINITE": 1, "NOT_NORMALIZED": 1, "OUT_OF_RANGE": 1}  # fmt: skip
    pred = registry(w).document(a.id, "predictions")
    assert {x["id"] for x in pred["invalid"]} == set(reasons) and pred["duplicate_ids"] == [ids[10]]
    bdq = registry(w).document(a.id, "metrics")["contexts"]["baseline"]["data_quality"]
    assert bdq["n_usable"] == len(ld.obs) == 60 - 5 - 1
    # the metrics use only the usable observations, and they are the ones an independent recompute keeps
    keep = [r for r in stored_rows(w, base) if r["index"] in ld.obs]
    assert a.summary["baseline"]["metrics"]["accuracy"] == pytest.approx(independent(keep, 10)["accuracy"])  # fmt: skip


def test_an_empty_or_score_free_evaluation_is_refused_before_execution(cw: tuple[EvalWorld, str], monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    w, base = cw
    patched(monkeypatch, lambda rows: [])
    ld = ce.load_run(w.registry, w.store, base)
    assert ld.n_rows == 0 and not ld.obs
    out = an.context_result([], make_spec(base), (0, 1), key="baseline", kind="BASELINE", n_members=0, dq=ce.dq_block(ld))  # fmt: skip
    assert out["status"] == "UNAVAILABLE" and out["metrics"]["top_label"] is None and out["bins"]["top_label"] == []  # fmt: skip


def test_a_classifier_without_probabilities_and_a_regression_model_are_unsupported(tmp_path: Path) -> None:  # fmt: skip
    from eval_helpers import class_data, eval_world
    from experionyx.evaluation.config import EvaluationConfig

    cfg = EvaluationConfig(split="test")
    labels_only = eval_world(tmp_path / "a", model={"threshold": 5.0, "proba": False}, data=class_data(40), config=cfg)  # fmt: skip
    r = labels_only.run().run.id
    spec = make_spec(r)
    with pytest.raises(ce.CalibrationError, match="stored no probability scores"):
        ce.validate(labels_only.registry, labels_only.store, spec)
    reg_data = {"rows": [[float(i)] for i in range(40)], "targets": [0.5 * i for i in range(40)], "task": "REGRESSION", "feature_names": ["x"], "splits": {"test": list(range(20, 40))}}  # fmt: skip
    regression = eval_world(tmp_path / "b", model={"value": 3.0}, data=reg_data, config=cfg, classifier=False)  # fmt: skip
    rr = regression.run().run.id
    with pytest.raises(ce.CalibrationError, match="no documented uncertainty representation"):
        ce.validate(regression.registry, regression.store, make_spec(rr))
    inv = regression.registry.find(Investigation)[0]
    with pytest.raises(ce.CalibrationError, match="UNAVAILABLE"):
        ce.run_calibration_request(regression.registry, regression.store, regression.executor, inv.id, make_spec(rr))  # fmt: skip
    assert not regression.registry.find(CalibrationAnalysis)  # refused: nothing was created


def test_a_declared_representation_that_differs_from_what_was_stored_is_refused(cw: tuple[EvalWorld, str]) -> None:  # fmt: skip
    w, base = cw
    with pytest.raises(ce.CalibrationError, match="never treated as another representation"):
        ce.validate(w.registry, w.store, make_spec(base, prediction_source="SOFTMAX_LOGITS"))
    ok = ce.validate(w.registry, w.store, make_spec(base))
    assert ok.representation == "PREDICT_PROBA" and len(ok.obs) == 60
    with pytest.raises(ce.CalibrationError, match="expects split"):
        ce.validate(w.registry, w.store, make_spec(base, split="train"))
    ce.validate(w.registry, w.store, make_spec(base, split="test"))


# -- adversarial ----------------------------------------------------------------------------------------


def docs_of(c: ce.Computed) -> str:
    return json.dumps(c.docs, sort_keys=True, default=str)


def test_reordered_rows_change_nothing(cw: tuple[EvalWorld, str], monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    w, base = cw
    spec = make_spec(base, objects=["CLASSWISE", "TOP_LABEL"])
    ref = ce.compute(w.registry, w.store, spec, None)
    patched(monkeypatch, lambda rows: list(reversed(rows)))
    again = ce.compute(w.registry, w.store, spec, None)
    assert docs_of(ref) == docs_of(again) and ref.fingerprint == again.fingerprint


def test_altered_probabilities_and_targets_alter_the_fingerprint_not_the_identity(cw: tuple[EvalWorld, str], monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    w, base = cw
    spec = make_spec(base)
    ref = ce.compute(w.registry, w.store, spec, None)

    def nudge(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        r = rows[0]
        s = list(r["scores"])
        s[0], s[1] = s[0] + 0.001, s[1] - 0.001
        rows[0] = {**r, "scores": s}
        return rows

    patched(monkeypatch, nudge)
    probs = ce.compute(w.registry, w.store, spec, None)
    monkeypatch.undo()
    patched(monkeypatch, lambda rows: [{**rows[0], "true": 1 - rows[0]["true"]}, *rows[1:]])
    targets = ce.compute(w.registry, w.store, spec, None)
    assert len({ref.fingerprint, probs.fingerprint, targets.fingerprint}) == 3
    assert probs.summary["baseline"]["metrics"]["ece"] != ref.summary["baseline"]["metrics"]["ece"]  # fmt: skip
    assert make_spec(base).spec_id == spec.spec_id  # the request identity did not change


def test_altered_stored_predictions_are_detected_by_the_artifact_digest(tmp_path: Path) -> None:
    w, base = cal_world(tmp_path)
    run = w.registry.get(Run, base)
    path = w.store.run_dir(run) / "artifacts" / "evaluation" / "predictions.jsonl"
    path.write_text(
        path.read_text().replace('"predicted": 1', '"predicted": 0', 1), encoding="utf-8"
    )
    with pytest.raises(ArtifactIntegrityError):
        analyze(w, base)


def test_shifted_or_partially_overlapping_sample_ids_are_never_treated_as_paired() -> None:
    spec = make_spec("run_" + "a" * 32)
    a = [m.Obs(i, 0, 0, (0.8, 0.2)) for i in range(40)]
    shifted = [m.Obs(i + 1, 0, 0, (0.8, 0.2)) for i in range(40)]  # IDs 1..40 vs 0..39
    c = an.compare_contexts(a, shifted, spec, key_a="a", key_b="b", family="STRESS")
    assert (
        c["status"] == "UNAVAILABLE" and "partially overlap" in c["reason"] and c["pairing"] is None
    )
    disjoint = [m.Obs(i + 100, 0, 0, (0.8, 0.2)) for i in range(40)]
    assert an.compare_contexts(a, disjoint, spec, key_a="a", key_b="b", family="WINDOW")["pairing"] == "UNPAIRED"  # fmt: skip
    same = an.compare_contexts(a, [m.Obs(i, 0, 1, (0.8, 0.2)) for i in range(40)], spec, key_a="a", key_b="b", family="STRESS")  # fmt: skip
    assert same["pairing"] == "PAIRED"


def test_a_changed_bin_configuration_is_a_different_analysis(cw: tuple[EvalWorld, str]) -> None:
    w, base = cw
    a = analyze(w, base, binning={"n_bins": 10})
    b = analyze(w, base, binning={"n_bins": 10, "strategy": "QUANTILE"})
    assert a.spec_id != b.spec_id and a.provenance_fingerprint != b.provenance_fingerprint


def test_too_few_observations_never_produce_an_interval(cw: tuple[EvalWorld, str]) -> None:
    w, base = cw
    a = analyze(w, base, statistics={"min_samples": 100})
    u = registry(w).document(a.id, "uncertainty")["contexts"]["baseline"]
    assert u["status"] == "INSUFFICIENT_EVIDENCE" and u["uncertainty"]["bootstrap"] is None
    assert u["uncertainty"]["accuracy"]["lower"] is not None  # a Wilson interval remains valid


# -- provenance, persistence, replay ----------------------------------------------------------------------


def test_provenance_records_everything_that_defines_the_analysis(cw: tuple[EvalWorld, str]) -> None:
    w, base = cw
    a = analyze(w, base, method="PLATT", objects=["CLASSWISE", "TOP_LABEL"])
    p: dict[str, Any] = registry(w).provenance(a.id)
    assert p["baseline_run_id"] == base and p["provenance_fingerprint"] == a.provenance_fingerprint
    assert p["model"]["fingerprint"] and p["dataset"]["fingerprint"] == a.dataset_fingerprint
    assert p["split"] == "test" and p["prediction_representation"] == "PREDICT_PROBA"
    assert (
        p["binning"] == {"strategy": "UNIFORM", "n_bins": 10} and p["calibration_method"] == "PLATT"
    )
    assert p["calibration_split"]["mode"] == "SPLIT" and p["bootstrap"]["resamples"] == 60
    assert (
        p["run_provenance"]["source_revision"] is not None and p["run_provenance"]["environment_id"]
    )
    assert p["source_runs"]["baseline"] == base and p["analysis_version"] == "1"
    assert p["evaluation_config_hash"].startswith("sha256:")
    run = w.registry.get(Run, a.run_id)
    assert run.status.value == "COMPLETED"


def test_replay_reproduces_identity_bins_metrics_intervals_and_artifact_references(cw: tuple[EvalWorld, str]) -> None:  # fmt: skip
    w, base = cw
    win = {"ordering": {"field": "index"}, "windows": [{"start": 0, "end": 30, "name": "a"}, {"start": 30, "end": 60, "name": "b"}]}  # fmt: skip
    sl = {"name": "pos", "condition": {"op": "range", "field": "feature:x0", "low": 0.0}}
    a = analyze(w, base, method="ISOTONIC", objects=["CLASSWISE", "TOP_LABEL"], windows=win, slices=[sl], statistics={"min_samples": 10})  # fmt: skip
    out = ce.replay_check(w.registry, w.store, w.executor, a.id)
    assert out["deterministic"] is True and out["differences"] == [] and out["compared"] == list(ce.DOCUMENTS)  # fmt: skip
    assert out["replay_run"] != out["original_run"] and out["replay_status"] == "COMPLETED"
    assert (
        len(w.registry.find(CalibrationAnalysis, spec_id=a.spec_id)) == 1
    )  # replay adds no second record


def test_recomputing_from_the_stored_predictions_reproduces_every_document(cw: tuple[EvalWorld, str]) -> None:  # fmt: skip
    w, base = cw
    a = analyze(w, base, objects=["CLASSWISE", "TOP_LABEL"])
    c = ce.compute(w.registry, w.store, make_spec(base, objects=["CLASSWISE", "TOP_LABEL"]), None)
    cr = registry(w)
    for name in ce.DOCUMENTS[1:]:
        assert json.loads(json.dumps(c.docs[name], default=str, sort_keys=True)) == json.loads(json.dumps(cr.document(a.id, name), sort_keys=True)), name  # fmt: skip
    assert c.fingerprint == a.provenance_fingerprint


def test_records_are_immutable_and_round_trip(cw: tuple[EvalWorld, str]) -> None:
    w, base = cw
    a = analyze(w, base)
    assert CalibrationAnalysis.from_dict(a.entity.to_dict()) == a.entity
    (r,) = [
        x
        for x in w.registry.find(CalibrationResult, analysis_id=a.id)
        if x.context_key == "baseline"
    ]
    assert (
        CalibrationResult.from_dict(r.to_dict()) == r
        and r.id.startswith("cbr_")
        and a.id.startswith("cba_")
    )
    with sqlite3.connect(w.workspace / "registry.sqlite") as raw:
        for sql in (
            "UPDATE calibration_analyses SET analysis_status = 'PARTIAL'",
            "DELETE FROM calibration_analyses",
            "UPDATE calibration_results SET status = 'COMPUTED'",
            "DELETE FROM calibration_results",
        ):
            with pytest.raises(sqlite3.DatabaseError, match=r"immutable|cannot be deleted"):
                raw.execute(sql)


def test_a_record_cannot_be_written_for_a_missing_run_or_analysis(cw: tuple[EvalWorld, str]) -> None:  # fmt: skip
    w, base = cw
    a = analyze(w, base)
    bad = CalibrationAnalysis(a.investigation_id, "run_" + "9" * 32, "cbs_x", {}, a.baseline_run_id, "sha256:f", a.provenance_fingerprint, "COMPLETE", {}, a.created_at)  # fmt: skip
    with pytest.raises(ExperionyxError):
        w.registry.add(bad)  # the run does not exist: a foreign-key style refusal
    with pytest.raises(ValidationError):
        CalibrationResult("cba_" + "1" * 32, "x", "NOPE", 0, "COMPUTED", None, {}, "sha256:" + "0" * 64, a.created_at)  # fmt: skip


def test_failure_candidates_are_signals_only_and_create_no_failure_records(cw: tuple[EvalWorld, str]) -> None:  # fmt: skip
    w, base = cw
    modes, signals = len(w.registry.find(FailureMode)), len(w.registry.find(FailureSignal))
    a = analyze(w, base, statistics={"high_confidence": 0.8, "low_confidence": 0.6})
    cand = registry(w).failure_candidates(a.id)
    rows = stored_rows(w, base)
    conf = {r["index"]: r["scores"][int(r["predicted"])] for r in rows}
    wrong = sorted(
        r["index"] for r in rows if r["predicted"] != r["true"] and conf[r["index"]] >= 0.8
    )
    right = sorted(
        r["index"] for r in rows if r["predicted"] == r["true"] and conf[r["index"]] < 0.6
    )
    assert [c["sample_id"] for c in cand["high_confidence_incorrect"]] == wrong and wrong
    assert [c["sample_id"] for c in cand["low_confidence_correct"]] == right
    assert (
        cand["status"] == "CANDIDATE_SIGNALS_ONLY"
        and "not classified as failure modes" in cand["note"]
    )
    rel = cand["confidence_error_relationship"]
    assert rel["n_correct"] + rel["n_incorrect"] == 60
    c_ok = np.array([conf[r["index"]] for r in rows if r["predicted"] == r["true"]])
    c_bad = np.array([conf[r["index"]] for r in rows if r["predicted"] != r["true"]])
    auc = float(((c_ok[:, None] > c_bad[None, :]) + 0.5 * (c_ok[:, None] == c_bad[None, :])).mean())
    assert rel["auroc_confidence_separates_correct"] == pytest.approx(auc)
    assert (len(w.registry.find(FailureMode)), len(w.registry.find(FailureSignal))) == (
        modes,
        signals,
    )


def test_a_referenced_analysis_that_does_not_exist_is_refused_before_execution(cw: tuple[EvalWorld, str]) -> None:  # fmt: skip
    w, base = cw
    before = len(w.registry.find(Run))
    for over in (
        {"quality_analyses": ["qan_" + "0" * 32]},
        {"stress_analyses": ["sxa_" + "0" * 32]},
    ):
        with pytest.raises(ExperionyxError):
            ce.validate(w.registry, w.store, make_spec(base, **over))
    assert len(w.registry.find(Run)) == before


def test_the_world_fixture_matches_the_documented_splits() -> None:
    assert SPLITS["test"][-1] == 59 and SPLITS["calib"][0] == 60 and spec_dict("run_" + "a" * 32)["statistics"]["min_samples"] == 20  # fmt: skip
    w = stress_world  # the helper the world is built from
    assert callable(w)

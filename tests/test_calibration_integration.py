"""Calibration linked to the rest of the lab on REAL stress trials, a real profile and a real benchmark
(the class_data world: threshold classifier with probabilities, evaluated split indices 40..79). The
linked records are referenced, never copied, and nothing here is a score or a causal claim."""

from pathlib import Path
from typing import Any

import pytest

from benchmark_helpers import ids, small_world, spec_for
from calibration_helpers import View
from experionyx.benchmark.engine import run_benchmark
from experionyx.benchmark.registry import BenchmarkRegistry
from experionyx.calibration import engine as ce
from experionyx.calibration.entities import CalibrationAnalysis
from experionyx.calibration.registry import CalibrationRegistry
from experionyx.calibration.spec import CalibrationSpec
from experionyx.errors import ProfileRefusal
from experionyx.reliability.engine import run_profile
from experionyx.reliability.entities import ReliabilityReference
from experionyx.reliability.registry import ReliabilityProfileRegistry
from experionyx.stress.engine import run_stress_experiment
from experionyx.stress.entities import StressAnalysis, StressTrial
from experionyx.stress.spec import StressDesign, StressPlan, StressSpec, Sweep
from interaction_helpers import EVAL
from reliability_helpers import EvidenceSet, full_evidence

FAST = {"resamples": 60, "permutations": 60, "min_samples": 20}


@pytest.fixture(scope="module")
def ev(tmp_path_factory: pytest.TempPathFactory) -> EvidenceSet:
    return full_evidence(tmp_path_factory.mktemp("cal-int"))


def stress(ev: EvidenceSet, plan: StressPlan, **kw: Any) -> StressAnalysis:
    w = ev.w
    model_id, data_id = ids(w)
    design = StressDesign(model_id, data_id, plan, EVAL, baseline_run=kw.pop("baseline", ev.design.baseline), min_members=5, resamples=100, permutations=100, **kw)  # fmt: skip
    out = run_stress_experiment(w.registry, w.store, w.executor, design, source_root=w.workspace.parent, investigation_id=ev.investigation)  # fmt: skip
    assert out.analysis_id, out
    return w.registry.get(StressAnalysis, out.analysis_id)


def cal(ev: EvidenceSet, **over: Any) -> View:
    spec = CalibrationSpec.from_dict({"baseline_run": ev.design.baseline, "prediction_source": "PREDICT_PROBA", "statistics": FAST, **over})  # fmt: skip
    out = ce.run_calibration_request(ev.w.registry, ev.w.store, ev.w.executor, ev.investigation, spec)  # fmt: skip
    assert out.analysis_id, out
    return View(ev.w.registry.get(CalibrationAnalysis, out.analysis_id))


NOISE = StressPlan((StressSpec("FEATURE_NOISE", {"sigma": 2.0}, seed=3),))
SWEEP = StressPlan(
    (StressSpec("FEATURE_NOISE", {"sigma": 1.0}, seed=1),), Sweep("sigma", (0.5, 2.0, 4.0)), (1, 2)
)


def test_calibration_under_an_explicitly_selected_stress_keeps_baseline_stress_and_trial_identity(ev: EvidenceSet) -> None:  # fmt: skip
    w = ev.w
    a = stress(ev, NOISE)
    c = cal(ev, stress_analyses=[a.id])
    cr = CalibrationRegistry(w.registry, w.store)
    (trial,) = w.registry.find(StressTrial, analysis_id=a.id)
    key = f"stress:{trial.unit_key}"
    ctx = c.summary["contexts"]
    assert (
        ctx["baseline"]["status"] == "COMPUTED" and ctx[key]["status"] == "COMPUTED"
    )  # baseline AND stressed
    assert ctx[key]["kind"] == "STRESS" and ctx[key]["n_samples"] == ctx["baseline"]["n_samples"]
    lin = cr.document(c.id, "spec")["lineage"]["stress"][key]
    assert lin["stress_analysis_id"] == a.id and lin["trial_id"] == trial.id and lin["trial_run_id"] == trial.run_id  # fmt: skip
    assert lin["stress_ids"] == list(trial.stress_ids) and lin["seed"] == trial.seed and lin["origin"] == "FAULT_LABORATORY"  # fmt: skip
    assert lin["stress_analysis_fingerprint"] == a.provenance_fingerprint and lin["predictions_digest"].startswith("sha256:")  # fmt: skip
    cmp_ = cr.document(c.id, "comparisons")["comparisons"][f"baseline vs {key}"]
    assert (
        cmp_["family"] == "STRESS" and cmp_["pairing"] == "PAIRED" and cmp_["status"] == "COMPUTED"
    )  # same sample IDs
    e = cmp_["metrics"]["ece"]
    assert e["difference"] == pytest.approx(e["b"] - e["a"]) and e["p_value"] is not None
    assert (
        "does not say why" in cmp_["note"] and "not causal" in cmp_["distribution"]["note"]
    )  # descriptive, not causal
    text = str(cr.document(c.id, "comparisons")).lower()
    assert "caused" not in text and "because" not in text
    assert cr.document(c.id, "spec")["lineage"]["stress"][key]["trial_dataset_fingerprint"] != c.dataset_fingerprint  # the derived dataset is recorded  # fmt: skip


def test_a_stress_sweep_is_analyzed_trial_by_trial_and_corrected_as_one_family(ev: EvidenceSet) -> None:  # fmt: skip
    a = stress(ev, SWEEP)
    trials = ev.w.registry.find(StressTrial, analysis_id=a.id)
    assert len(trials) == 6
    c = cal(ev, stress_analyses=[a.id], statistics={**FAST, "correction": "BENJAMINI_HOCHBERG"})
    cr = CalibrationRegistry(ev.w.registry, ev.w.store)
    comps = cr.document(c.id, "comparisons")
    stressed = {k: v for k, v in comps["comparisons"].items() if v["family"] == "STRESS"}
    assert len(stressed) == 6 and all(v["status"] == "COMPUTED" for v in stressed.values())
    assert comps["corrections"]["STRESS/ece"]["n_hypotheses"] == 6
    assert comps["corrections"]["STRESS/ece"]["method"] == "BENJAMINI_HOCHBERG"
    for v in stressed.values():
        r = v["metrics"]["ece"]
        assert r["adjusted_p_value"] >= r["p_value"] - 1e-12
    assert len(c.summary["contexts"]) == 1 + 6


def test_a_trial_whose_predictions_cannot_be_read_is_unavailable_evidence_not_silently_dropped(ev: EvidenceSet, monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    a = stress(ev, StressPlan((StressSpec("FEATURE_NOISE", {"sigma": 3.0}, seed=9),)))
    (trial,) = ev.w.registry.find(StressTrial, analysis_id=a.id)
    real = ce.load_run

    def flaky(reg: Any, store: Any, run_id: str) -> Any:
        if run_id == trial.run_id:
            raise ce.CalibrationError("UNAVAILABLE: the trial kept no per-sample predictions")
        return real(reg, store, run_id)

    monkeypatch.setattr(ce, "load_run", flaky)
    c = cal(ev, stress_analyses=[a.id], statistics={**FAST, "seed": 5})
    key = f"stress:{trial.unit_key}"
    assert c.summary["contexts"][key]["status"] == "UNAVAILABLE" and "cannot be read" in c.summary["contexts"][key]["reason"]  # fmt: skip
    assert (
        c.analysis_status == "PARTIAL" and c.summary["comparison_status_counts"] == {}
    )  # nothing invented


def test_a_stress_analysis_of_another_baseline_is_refused_before_execution(ev: EvidenceSet) -> None:
    from interaction_helpers import dropout, launch

    other = launch(ev.w, dropout(0.4), (1, 2), name="another baseline for calibration")
    foreign = stress(ev, NOISE, baseline=other.baseline_run_id)
    spec = CalibrationSpec.from_dict({"baseline_run": ev.design.baseline, "prediction_source": "PREDICT_PROBA", "stress_analyses": [foreign.id], "statistics": FAST})  # fmt: skip
    with pytest.raises(Exception, match="was made over baseline"):
        ce.validate(ev.w.registry, ev.w.store, spec)


def test_the_profile_gets_a_calibration_dimension_without_a_score(ev: EvidenceSet) -> None:
    w = ev.w
    c = cal(ev)
    rr = ReliabilityProfileRegistry(w.registry, w.store)
    with_c = run_profile(w.registry, w.store, w.executor, ev.investigation, ev.spec(calibration_analyses=(c.id,)))  # fmt: skip
    plain = run_profile(w.registry, w.store, w.executor, ev.investigation, ev.spec())
    assert with_c.profile_id and plain.profile_id and with_c.profile_id != plain.profile_id
    dim = rr.document(with_c.profile_id)["dimensions"]["CALIBRATION_ANALYSIS"]
    assert dim["status"] == "DERIVED" and "not an uncertainty guarantee" in dim["note"]
    obs = dim["observations"][0]
    assert obs["source"]["id"] == c.id and obs["source"]["provenance_fingerprint"] == c.provenance_fingerprint  # fmt: skip
    assert obs["baseline"]["metrics"]["ece"] == c.summary["baseline"]["metrics"]["ece"] and "no calibration or uncertainty score" in obs["note"]  # fmt: skip
    refs = {(r.ref_kind.value, r.ref_id) for r in w.registry.find(ReliabilityReference, profile_id=with_c.profile_id)}  # fmt: skip
    assert ("CALIBRATION_ANALYSIS", c.id) in refs
    off = rr.document(plain.profile_id)["dimensions"]["CALIBRATION_ANALYSIS"]
    assert off["status"] == "UNAVAILABLE" and "not evidence of good calibration" in off["reason"]  # missing != PASS  # fmt: skip
    assert (
        "CALIBRATION" in rr.document(plain.profile_id)["dimensions"]
    )  # the baseline's own dimension is untouched
    assert "calibration_analyses" not in ev.spec().to_dict() and ev.spec(calibration_analyses=(c.id,)).spec_id != ev.spec().spec_id  # fmt: skip
    assert "score" not in str(rr.document(with_c.profile_id)["dimension_status"]).lower()


def test_insufficient_calibration_evidence_stays_insufficient_in_the_profile(
    ev: EvidenceSet,
) -> None:
    w = ev.w
    thin = cal(ev, statistics={**FAST, "min_samples": 500})
    pr = run_profile(w.registry, w.store, w.executor, ev.investigation, ev.spec(calibration_analyses=(thin.id,)))  # fmt: skip
    dim = ReliabilityProfileRegistry(w.registry, w.store).document(pr.profile_id or "")["dimensions"]["CALIBRATION_ANALYSIS"]  # fmt: skip
    assert (
        dim["status"] == "INSUFFICIENT_EVIDENCE" and "enough usable observations" in dim["reason"]
    )


def test_a_profile_refuses_calibration_analyses_of_another_baseline_or_missing_ones(ev: EvidenceSet) -> None:  # fmt: skip
    from interaction_helpers import dropout, launch

    w = ev.w
    other = launch(w, dropout(0.4), (1, 2), name="foreign baseline")
    spec = CalibrationSpec.from_dict({"baseline_run": other.baseline_run_id, "prediction_source": "PREDICT_PROBA", "statistics": FAST})  # fmt: skip
    out = ce.run_calibration_request(w.registry, w.store, w.executor, ev.investigation, spec)
    assert out.analysis_id
    with pytest.raises(ProfileRefusal, match="INCOMPATIBLE_CALIBRATION_ANALYSIS"):
        run_profile(w.registry, w.store, w.executor, ev.investigation, ev.spec(calibration_analyses=(out.analysis_id,)))  # fmt: skip
    with pytest.raises(ProfileRefusal, match="CALIBRATION_ANALYSIS_MISSING"):
        run_profile(w.registry, w.store, w.executor, ev.investigation, ev.spec(calibration_analyses=("cba_" + "0" * 32,)))  # fmt: skip


BENCH = CalibrationSpec.from_dict({"baseline_run": "run_" + "0" * 32, "prediction_source": "PREDICT_PROBA", "method": "PLATT", "statistics": FAST})  # fmt: skip


def test_a_benchmark_references_a_calibration_analysis_only_when_asked_and_counts_coverage(tmp_path: Path) -> None:  # fmt: skip
    w = small_world(tmp_path)
    plain = spec_for(w)
    asked = spec_for(w, calibration=BENCH)
    assert "calibration" not in plain.to_dict() and plain.spec_id == spec_for(w, calibration=None).spec_id  # fmt: skip
    asked_doc: Any = asked.to_dict()
    assert asked.spec_id != plain.spec_id and "baseline_run" not in asked_doc["calibration"]
    from experionyx.benchmark.spec import BenchmarkSpec

    assert BenchmarkSpec.from_dict(asked.to_dict()).calibration == BENCH  # round trip
    out = run_benchmark(w.registry, w.store, w.executor, asked, source_root=w.workspace.parent)
    assert out.result_id, out.errors
    br = BenchmarkRegistry(w.registry, w.store)
    res = br.document(out.result_id, "results")["calibration_analysis"]
    assert res["status"] in ("DERIVED", "INSUFFICIENT_EVIDENCE") and res["planned"] == 2
    assert res["coverage"]["planned"] == 2 and res["coverage"]["failed"] == 0
    assert res["coverage"]["executed"] + res["coverage"]["insufficient_evidence"] + res["coverage"]["unavailable"] == 2  # fmt: skip
    assert (
        "referenced, not duplicated" in res["note"]
        and res["source"]["kind"] == "CALIBRATION_ANALYSIS"
    )
    (x,) = w.registry.find(CalibrationAnalysis)
    assert x.id == res["analysis_id"]  # not recomputed inside the benchmark
    assert (
        br.document(out.result_id, "summary")["section_status"]["calibration_analysis"]
        == res["status"]
    )
    assert any(r.ref_kind.value == "CALIBRATION_ANALYSIS" and r.ref_id == x.id for r in w.registry.find(ReliabilityReference))  # fmt: skip
    again = run_benchmark(w.registry, w.store, w.executor, spec_for(w, name="plain-again"), source_root=w.workspace.parent)  # fmt: skip
    assert "calibration_analysis" not in br.document(again.result_id or "", "results") and len(w.registry.find(CalibrationAnalysis)) == 1  # fmt: skip


def test_a_benchmark_reports_an_unsupported_calibration_as_failed_coverage_not_success(tmp_path: Path) -> None:  # fmt: skip
    w = small_world(tmp_path)
    wrong = CalibrationSpec.from_dict({"baseline_run": "run_" + "0" * 32, "prediction_source": "SOFTMAX_LOGITS", "statistics": FAST})  # fmt: skip
    out = run_benchmark(w.registry, w.store, w.executor, spec_for(w, calibration=wrong), source_root=w.workspace.parent)  # fmt: skip
    assert out.result_id, out.errors
    br = BenchmarkRegistry(w.registry, w.store)
    res = br.document(out.result_id, "results")["calibration_analysis"]
    assert (
        res["status"] == "UNAVAILABLE"
        and "never treated as another representation" in res["reason"]
    )
    assert res["coverage"] == {"planned": 1, "executed": 0, "unavailable": 0, "insufficient_evidence": 0, "failed": 1}  # fmt: skip
    assert br.document(out.result_id, "coverage")["complete"] is False
    assert not w.registry.find(CalibrationAnalysis)

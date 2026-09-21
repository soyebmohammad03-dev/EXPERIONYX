"""Resource measurements linked to the rest of the lab, on the real class_data world (a threshold
classifier with probabilities, split 'test' = samples 40..79) and the REAL clock: the reliability profile,
a benchmark's coverage, model-level stress, explicit slice / window workloads, referenced calibration
analyses and the execution engine's own records. No timing value is asserted anywhere here; structure,
identity, coverage and refusal are."""

from pathlib import Path
from typing import Any

import pytest

from benchmark_helpers import ids, small_world, spec_for
from calibration_helpers import FAST as CAL_FAST
from experionyx.adapters.records import RegisteredDataset
from experionyx.benchmark.engine import run_benchmark
from experionyx.benchmark.registry import BenchmarkRegistry
from experionyx.benchmark.spec import BenchmarkSpec
from experionyx.calibration import engine as ce
from experionyx.calibration.entities import CalibrationAnalysis
from experionyx.calibration.spec import CalibrationSpec
from experionyx.domain import Experiment, Observation, Run, to_jsonable
from experionyx.errors import NotFoundError, ProfileRefusal, ValidationError
from experionyx.execution import run_states
from experionyx.provenance import Provenance
from experionyx.reliability.engine import run_profile
from experionyx.reliability.entities import ReliabilityReference
from experionyx.reliability.registry import ReliabilityProfileRegistry
from experionyx.resources import engine as re_
from experionyx.resources.entities import ResourceAnalysis
from experionyx.resources.registry import ResourceRegistry
from experionyx.resources.spec import ResourceSpec
from experionyx.slices.spec import SliceSpec, between
from reliability_helpers import EvidenceSet, full_evidence
from resource_helpers import View

EVAL_SPLIT = "test"


@pytest.fixture(scope="module")
def ev(tmp_path_factory: pytest.TempPathFactory) -> EvidenceSet:
    return full_evidence(tmp_path_factory.mktemp("res-int"))


def rspec(ev: EvidenceSet, **over: Any) -> ResourceSpec:
    model, data = ids(ev.w)
    body: dict[str, Any] = {"model_id": model, "dataset_id": data, "split": EVAL_SPLIT, "batch_size": 8, "repeats": 5, "warmup_trials": 1}  # fmt: skip
    body.update(over)
    return ResourceSpec.from_dict(body)


def measure(ev: EvidenceSet, **over: Any) -> View:
    out = re_.run_resource_request(ev.w.registry, ev.w.store, ev.w.executor, ev.investigation, rspec(ev, **over))  # fmt: skip
    assert out.analysis_id, out
    return View(ev.w.registry.get(ResourceAnalysis, out.analysis_id))


# -- the execution engine ----------------------------------------------------------------------------------------


def test_a_measurement_is_a_normal_run_with_provenance_observations_and_artifacts(ev: EvidenceSet) -> None:  # fmt: skip
    w = ev.w
    a = measure(ev, seed=3, timeout_seconds=30.0)
    run = w.registry.get(Run, a.run_id)
    assert run.status.value == "COMPLETED" and run.seed == 3
    (prov,) = w.registry.find(Provenance, run_id=run.id)
    assert prov.execution.procedure == re_.PROCEDURE and prov.execution.resources.timeout_seconds == 30.0 and prov.execution.resources.max_workers == 1  # fmt: skip
    assert prov.inputs is not None and prov.inputs.model is not None and prov.inputs.dataset is not None  # fmt: skip
    assert prov.inputs.model.fingerprint == a.summary["model"]["fingerprint"] and prov.inputs.dataset.fingerprint == a.summary["dataset"]["fingerprint"]  # fmt: skip
    exp = w.registry.get(Experiment, run.experiment_id)
    assert exp.name == "resource measurement" and "environment-specific" in exp.hypothesis
    names = {o.name for o in w.registry.find(Observation, run_id=run.id)}
    assert {"resources.trial_wall_seconds", "resources.measured_trials", "resources.completed_trials"} <= names  # fmt: skip
    assert run.id in {r.id for rs in run_states(w.registry).values() for r in rs}
    again = measure(ev, seed=3, timeout_seconds=30.0)
    assert again.run_id != a.run_id and again.spec_id == a.spec_id and again.provenance_fingerprint == a.provenance_fingerprint  # fmt: skip
    assert w.registry.get(Run, again.run_id).attempt == run.attempt + 1  # a deliberate repeat, not an overwrite  # fmt: skip


# -- reliability profile ------------------------------------------------------------------------------------------


def test_the_profile_gets_a_resource_system_dimension_without_a_score(ev: EvidenceSet) -> None:
    w = ev.w
    ra = measure(ev)
    rr = ReliabilityProfileRegistry(w.registry, w.store)
    with_r = run_profile(w.registry, w.store, w.executor, ev.investigation, ev.spec(resource_analyses=(ra.id,)))  # fmt: skip
    plain = run_profile(w.registry, w.store, w.executor, ev.investigation, ev.spec())
    assert with_r.profile_id and plain.profile_id and with_r.profile_id != plain.profile_id
    doc = rr.document(with_r.profile_id)
    dim = doc["dimensions"]["RESOURCE_SYSTEM"]
    assert (
        dim["status"] == "DERIVED" and "performance measurement is not reliability" in dim["note"]
    )
    (obs,) = dim["observations"]
    assert obs["source"]["id"] == ra.id and obs["source"]["provenance_fingerprint"] == ra.provenance_fingerprint  # fmt: skip
    for k in ("latency", "throughput", "failures", "memory", "cpu", "environment", "concurrency", "trials", "outputs"):  # fmt: skip
        assert k in obs  # the evidence the dimension exposes
    assert obs["latency"]["trial_seconds"]["n"] == 5 and obs["failures"]["failed"] == 0 and obs["environment"]["environment_id"]  # fmt: skip
    assert obs["latency"]["trial_seconds"]["mean"] == ra.summary["steady_state"]["trial_seconds"]["mean"]  # copied as recorded  # fmt: skip
    assert (
        "no resource or reliability score" in obs["note"] and "environment-specific" in obs["note"]
    )
    refs = {(x.ref_kind.value, x.ref_id) for x in w.registry.find(ReliabilityReference, profile_id=with_r.profile_id)}  # fmt: skip
    assert ("RESOURCE_ANALYSIS", ra.id) in refs
    off = rr.document(plain.profile_id)["dimensions"]["RESOURCE_SYSTEM"]
    assert off["status"] == "UNAVAILABLE" and "not evidence of good or bad system behaviour" in off["reason"]  # missing != PASS  # fmt: skip
    assert "LATENCY" in doc["dimensions"] and "CALIBRATION_ANALYSIS" in doc["dimensions"]  # untouched neighbours  # fmt: skip
    assert "resource_analyses" not in ev.spec().to_dict() and ev.spec(resource_analyses=(ra.id,)).spec_id != ev.spec().spec_id  # fmt: skip
    assert "score" not in str(doc["dimension_status"]).lower()


def test_insufficient_resource_evidence_stays_insufficient_in_the_profile(ev: EvidenceSet) -> None:
    w = ev.w
    thin = measure(ev, repeats=2)
    timed_out = measure(ev, timeout_seconds=1e-6)  # every trial overruns a 1 microsecond timeout
    assert thin.summary["evidence_status"] == timed_out.summary["evidence_status"] == "INSUFFICIENT_EVIDENCE"  # fmt: skip
    pr = run_profile(w.registry, w.store, w.executor, ev.investigation, ev.spec(resource_analyses=(thin.id, timed_out.id)))  # fmt: skip
    dim = ReliabilityProfileRegistry(w.registry, w.store).document(pr.profile_id or "")["dimensions"]["RESOURCE_SYSTEM"]  # fmt: skip
    assert dim["status"] == "INSUFFICIENT_EVIDENCE" and "enough completed measured trials" in dim["reason"]  # fmt: skip
    assert {o["failures"]["timed_out"] for o in dim["observations"]} == {0, 5}  # the timeouts are visible evidence  # fmt: skip


def test_a_profile_refuses_resource_analyses_of_another_dataset_or_missing_ones(ev: EvidenceSet) -> None:  # fmt: skip
    import json

    from eval_helpers import NOW, class_data
    from experionyx.adapters.records import RegisteredModel
    from pure_adapters import ListDatasetAdapter

    w = ev.w
    path = w.workspace / "m" / "other.json"
    path.write_text(json.dumps(class_data(30)), encoding="utf-8")
    meta = ListDatasetAdapter.load(path, version="1", options={}).metadata()
    other = RegisteredDataset("other", meta, NOW, "m/other.json")
    w.registry.add(other)
    model = next(m for m in w.registry.find(RegisteredModel) if m.name == "model")
    foreign = View(w.registry.get(ResourceAnalysis, re_.run_resource_request(w.registry, w.store, w.executor, ev.investigation, ResourceSpec.from_dict({"model_id": model.id, "dataset_id": other.id, "split": "test", "batch_size": 4, "repeats": 5, "warmup_trials": 0})).analysis_id or ""))  # fmt: skip
    with pytest.raises(ProfileRefusal, match="INCOMPATIBLE_RESOURCE_ANALYSIS"):
        run_profile(w.registry, w.store, w.executor, ev.investigation, ev.spec(resource_analyses=(foreign.id,)))  # fmt: skip
    with pytest.raises(ProfileRefusal, match="RESOURCE_ANALYSIS_MISSING"):
        run_profile(w.registry, w.store, w.executor, ev.investigation, ev.spec(resource_analyses=("rsa_" + "0" * 32,)))  # fmt: skip


# -- model-level stress -------------------------------------------------------------------------------------------


def test_stress_identity_and_resource_identity_stay_separate_and_outputs_change_only_with_the_model(tmp_path: Path) -> None:  # fmt: skip
    from pure_adapters import LinearProbaAdapter
    from resource_helpers import resource_world

    r = resource_world(tmp_path, LinearProbaAdapter)
    base = r.run(r.spec(operation="PREDICT_PROBA", repeats=5))
    coarse = r.run(
        r.spec(operation="PREDICT_PROBA", repeats=5, batch_size=7)
    )  # execution context only
    scaled = r.run(r.spec(operation="PREDICT_PROBA", repeats=5, stress=[{"family": "PARAMETER_SCALE", "parameters": {"factor": 1.5}}]))  # fmt: skip
    thr = r.run(
        r.spec(
            operation="PREDICT_PROBA",
            repeats=5,
            stress=[{"family": "THRESHOLD", "parameters": {"threshold": 0.8}}],
        )
    )
    assert base.summary["stress"] is None and coarse.summary["stress"] is None
    assert coarse.summary["outputs"]["digest"] == base.summary["outputs"]["digest"]  # a batch size is not a model change  # fmt: skip
    assert (
        scaled.summary["outputs"]["digest"] != base.summary["outputs"]["digest"]
    )  # ...a stressed model is
    assert thr.summary["outputs"]["consistent_across_trials"] is True
    (sid,) = scaled.summary["stress"]["stress_ids"]
    assert sid.startswith("sts_") and scaled.spec["stress"][0]["family"] == "PARAMETER_SCALE"
    assert scaled.spec_id not in (base.spec_id,) and scaled.summary["stress"]["build"]["applied"][0]["restored"] is True  # fmt: skip
    assert "not a causal claim" in scaled.summary["stress"]["note"] and scaled.summary["initialization"]["stress_build_seconds"] is not None  # fmt: skip
    prov = ResourceRegistry(r.w.registry, r.w.store).provenance(scaled.id)
    assert prov["stress_ids"] == [sid] and prov["resource_configuration"]["stress"][0]["family"] == "PARAMETER_SCALE"  # fmt: skip
    assert scaled.provenance_fingerprint != base.provenance_fingerprint


def test_stress_the_adapter_cannot_apply_is_refused_as_unavailable(tmp_path: Path) -> None:
    w = small_world(tmp_path)  # a threshold classifier: no safe parameter access
    m, d = ids(w)
    from experionyx.domain import Investigation

    inv = w.registry.find(Investigation)[0]
    spec = ResourceSpec.from_dict({"model_id": m, "dataset_id": d, "split": "test", "stress": [{"family": "PARAMETER_NOISE", "parameters": {"relative_sigma": 0.1}, "seed": 1}]})  # fmt: skip
    before = len(w.registry.find(Run))
    with pytest.raises(re_.ResourceUnavailable, match=r"UNAVAILABLE.*model_stress"):
        re_.run_resource_request(w.registry, w.store, w.executor, inv.id, spec)
    assert len(w.registry.find(Run)) == before


# -- explicit slice and window workloads ---------------------------------------------------------------------------


def test_a_slice_workload_is_measured_only_when_explicitly_requested(ev: EvidenceSet) -> None:
    base = ev.design.baseline
    sl = SliceSpec("low-x0", between("feature:x0", 0, 3))
    plain = measure(ev)
    sub = measure(
        ev, subset={"kind": "SLICE", "baseline_run": base, "slice": to_jsonable(sl.to_dict())}
    )
    assert plain.summary["workload"]["subset"] is None and plain.summary["workload"]["samples"] == 40  # population by default  # fmt: skip
    sw = sub.summary["workload"]
    assert sw["samples"] == 12 and sw["subset"]["kind"] == "SLICE" and sw["subset"]["members"] == 12 and sw["subset"]["name"] == "low-x0"  # fmt: skip
    assert sw["sample_digest"] != plain.summary["workload"]["sample_digest"] and sub.spec_id != plain.spec_id  # fmt: skip
    assert sub.summary["batch_size"]["requested"] == 8 and sub.summary["batch_size"]["effective_max"] <= 8  # a subset thins each requested batch  # fmt: skip
    assert len(ev.w.registry.find(ResourceAnalysis, spec_id=plain.spec_id)) >= 1  # no other slice was run automatically  # fmt: skip


def test_a_window_workload_uses_only_the_samples_of_that_window(ev: EvidenceSet) -> None:
    win = {"kind": "WINDOW", "baseline_run": ev.design.baseline, "ordering": {"field": "feature:index"}, "window": {"name": "middle", "start": 45, "end": 55}}  # fmt: skip
    a = measure(ev, subset=win)
    assert (
        a.summary["workload"]["samples"] == 10
        and a.summary["workload"]["subset"]["kind"] == "WINDOW"
    )
    assert a.summary["workload"]["subset"]["members"] == 10 and a.summary["workload"]["subset"]["window_id"].startswith("twn_")  # fmt: skip
    idx = measure(ev, subset={**win, "ordering": {"field": "index"}, "window": {"name": "late", "start": 70, "end": 80}})  # fmt: skip
    assert idx.summary["workload"]["samples"] == 10


def test_a_subset_that_cannot_be_measured_is_refused_before_anything_runs(ev: EvidenceSet) -> None:
    w = ev.w
    before = len(w.registry.find(Run))
    empty = {"kind": "SLICE", "baseline_run": ev.design.baseline, "slice": to_jsonable(SliceSpec("none", between("feature:x0", 100, 200)).to_dict())}  # fmt: skip
    with pytest.raises(re_.ResourceError, match="no measurable members"):
        measure(ev, subset=empty)
    wrong_split = {"kind": "WINDOW", "baseline_run": ev.design.baseline, "ordering": {"field": "index"}, "window": {"name": "w", "start": 0, "end": 10}}  # fmt: skip
    with pytest.raises(re_.ResourceError, match="has no members"):
        measure(ev, subset=wrong_split)  # the evaluated split holds samples 40..79
    with pytest.raises(re_.ResourceError, match="used split"):
        measure(
            ev, subset={**wrong_split, "window": {"name": "w", "start": 40, "end": 50}}, split=None
        )
    with pytest.raises(NotFoundError):
        measure(ev, subset={**wrong_split, "baseline_run": "run_" + "0" * 32})
    assert len(w.registry.find(Run)) == before


# -- referenced analyses --------------------------------------------------------------------------------------------


def test_a_referenced_calibration_analysis_is_recorded_not_recomputed(ev: EvidenceSet) -> None:
    w = ev.w
    spec = CalibrationSpec.from_dict({"baseline_run": ev.design.baseline, "prediction_source": "PREDICT_PROBA", "statistics": CAL_FAST})  # fmt: skip
    cal = ce.run_calibration_request(w.registry, w.store, w.executor, ev.investigation, spec)
    assert cal.analysis_id
    n = len(w.registry.find(CalibrationAnalysis))
    a = measure(ev, calibration_analyses=[cal.analysis_id, cal.analysis_id])
    assert a.summary["references"]["calibration_analyses"] == [cal.analysis_id] and "nothing referenced is recomputed" in a.summary["references"]["note"]  # fmt: skip
    assert len(w.registry.find(CalibrationAnalysis)) == n  # not rerun
    assert a.spec_id != measure(ev).spec_id  # the reference is part of the definition
    with pytest.raises(NotFoundError):
        measure(ev, calibration_analyses=["cba_" + "0" * 32])
    with pytest.raises(NotFoundError):
        measure(ev, stress_analyses=["sxa_" + "0" * 32])
    with pytest.raises(NotFoundError):
        measure(ev, quality_analyses=["qan_" + "0" * 32])


# -- benchmark ---------------------------------------------------------------------------------------------------------


def _bench_specs(w: Any) -> dict[str, ResourceSpec]:
    m, d = ids(w)
    base: dict[str, Any] = {"model_id": m, "dataset_id": d, "split": "test", "batch_size": 8, "repeats": 5, "warmup_trials": 0}  # fmt: skip
    return {
        "executed": ResourceSpec.from_dict(base),
        "unsupported": ResourceSpec.from_dict({**base, "workers": 2}),
        "failed": ResourceSpec.from_dict({**base, "split": "no-such-split"}),
        "timed_out": ResourceSpec.from_dict({**base, "timeout_seconds": 1e-6}),
        "insufficient": ResourceSpec.from_dict({**base, "repeats": 2}),
    }


def test_a_benchmark_counts_every_resource_unit_by_its_real_outcome(tmp_path: Path) -> None:
    w = small_world(tmp_path)
    specs = _bench_specs(w)
    plain = spec_for(w)
    asked = spec_for(w, resources=tuple(specs.values()))
    assert "resources" not in plain.to_dict() and plain.spec_id == spec_for(w, resources=()).spec_id
    assert asked.spec_id != plain.spec_id and BenchmarkSpec.from_dict(asked.to_dict()).resources == asked.resources  # fmt: skip
    out = run_benchmark(w.registry, w.store, w.executor, asked, source_root=w.workspace.parent)
    assert out.result_id, out.errors
    br = BenchmarkRegistry(w.registry, w.store)
    res = br.document(out.result_id, "results")["resource_analysis"]
    assert res["coverage"] == {"planned": 5, "executed": 1, "unsupported": 1, "failed": 1, "timed_out": 1, "insufficient_evidence": 1}  # fmt: skip
    by_spec = {u["spec_id"]: u for u in res["units"]}
    for name, want in (("executed", "EXECUTED"), ("unsupported", "UNSUPPORTED"), ("failed", "FAILED"), ("timed_out", "TIMED_OUT"), ("insufficient", "INSUFFICIENT_EVIDENCE")):  # fmt: skip
        assert by_spec[specs[name].spec_id]["status"] == want, name
    assert by_spec[specs["unsupported"].spec_id]["analysis_id"] is None and "UNAVAILABLE" in by_spec[specs["unsupported"].spec_id]["reason"]  # fmt: skip
    assert by_spec[specs["executed"].spec_id]["source"]["kind"] == "RESOURCE_ANALYSIS"
    assert res["status"] == "DERIVED" and "referenced, not duplicated" in res["note"] and "no resource score" in res["note"]  # fmt: skip
    assert len(w.registry.find(ResourceAnalysis)) == 3  # executed, timed-out and insufficient ran; the other two never did  # fmt: skip
    assert br.document(out.result_id, "coverage")["complete"] is False
    assert br.document(out.result_id, "summary")["section_status"]["resource_analysis"] == "DERIVED"
    assert any("resource measurements" in x for x in br.document(out.result_id, "summary")["incomplete_reasons"])  # fmt: skip
    plain_out = run_benchmark(w.registry, w.store, w.executor, spec_for(w, name="plain-again"), source_root=w.workspace.parent)  # fmt: skip
    assert "resource_analysis" not in br.document(plain_out.result_id or "", "results")
    prof = ReliabilityProfileRegistry(w.registry, w.store).document(br.document(out.result_id, "summary")["reliability_profile"]["profile_id"] if False else _profile_id(w, out.result_id))  # fmt: skip
    assert prof["dimensions"]["RESOURCE_SYSTEM"]["status"] == "DERIVED"


def _profile_id(w: Any, result_id: str) -> str:
    br = BenchmarkRegistry(w.registry, w.store)
    return str(br.document(result_id, "results")["reliability_profile"]["profile_id"])


def test_a_benchmark_refuses_resource_specs_for_another_model_or_dataset_or_duplicates(tmp_path: Path) -> None:  # fmt: skip
    w = small_world(tmp_path)
    ok = _bench_specs(w)["executed"]
    other = ResourceSpec.from_dict({**ok.to_dict(), "model_id": "mdl_" + "9" * 32})
    with pytest.raises(ValidationError, match="benchmark's own model and dataset"):
        spec_for(w, resources=(other,))
    with pytest.raises(ValidationError, match="distinct"):
        spec_for(w, resources=(ok, ok))
    assert spec_for(w, resources=(ok,)).spec_id == spec_for(w, resources=(ok,)).spec_id

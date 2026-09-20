"""Drift evidence linked to the rest of the lab, on REAL fault experiments, a real discovery, a real
reliability profile and a real benchmark (the class_data world: evaluated split is indices 40..79,
feature x0 is `index % 10`). Windows are declared over the sample index; nothing here says the index
is time in any real system, and no link below is causal."""

from pathlib import Path
from typing import Any

import pytest

from benchmark_helpers import small_world, spec_for
from experionyx.benchmark.engine import run_benchmark
from experionyx.benchmark.registry import BenchmarkRegistry
from experionyx.drift.engine import run_drift_request
from experionyx.drift.entities import DriftAnalysis
from experionyx.drift.registry import DriftRegistry
from experionyx.drift.spec import Ordering, Role, ShiftSpec, TemporalWindow
from experionyx.errors import ProfileRefusal
from experionyx.failures.entities import FailureCluster, FailureMode, FailureSignal
from experionyx.reliability.engine import run_profile
from experionyx.reliability.entities import ReliabilityReference
from experionyx.reliability.registry import ReliabilityProfileRegistry
from interaction_helpers import dropout, launch
from reliability_helpers import EvidenceSet, full_evidence

W1, W2 = (40, 60), (60, 80)
CFG = {"min_samples": 5, "resamples": 50, "permutations": 100}


@pytest.fixture(scope="module")
def ev(tmp_path_factory: pytest.TempPathFactory) -> EvidenceSet:
    return full_evidence(tmp_path_factory.mktemp("drift-int"))


def spec_of(baseline: str, **over: Any) -> ShiftSpec:
    d: dict[str, Any] = {
        "baseline_run": baseline, "ordering": {"field": "index"},
        "reference": {"start": W1[0], "end": W1[1]}, "comparisons": [{"start": W2[0], "end": W2[1]}],
        "features": [{"name": "x0", "type": "NUMERIC"}], "config": CFG,
    }  # fmt: skip
    d.update(over)
    return ShiftSpec.from_dict(d)


def analyze(ev: EvidenceSet, spec: ShiftSpec) -> DriftAnalysis:
    w = ev.w
    out = run_drift_request(
        w.registry, w.store, w.executor, ev.investigation, spec
    )  # the dataset is found from the baseline
    assert out.analysis_id, out
    return w.registry.get(DriftAnalysis, out.analysis_id)


def test_failure_mode_links_match_the_signals_per_window(ev: EvidenceSet) -> None:
    w = ev.w
    modes = tuple(sorted(m.id for m in w.registry.find(FailureMode)))
    a = analyze(ev, spec_of(ev.design.baseline, failure_modes=list(modes)))
    dr = DriftRegistry(w.registry, w.store)
    doc = dr.document(a.id, "performance_results")
    assert "not an explanation" in doc["failure_mode_note"]
    wins = dr.document(a.id, "windows")["windows"]
    assert len(doc["failure_mode_evidence"]) == 2 and set(doc["failure_mode_evidence"]) == set(wins)
    computed = 0
    for wid, ev_doc in doc["failure_mode_evidence"].items():
        members = set(wins[wid]["sample_ids"])
        for row in ev_doc["modes"]:
            mode = w.registry.get(FailureMode, row["mode_id"])
            sigs = [
                w.registry.get(FailureSignal, s)
                for s in w.registry.get(FailureCluster, mode.cluster_id).signal_ids
            ]
            assert (
                row["confirmed"] is (mode.status.value == "CONFIRMED") and row["confirmed"] is False
            )
            if row["status"] == "COMPUTED":
                touched = {int(x) for s in sigs for x in s.sample_ids}
                assert (
                    row["slice"]["samples_affected"] == len(touched & members)
                    and row["slice"]["samples"] == 20
                )
                computed += 1
    assert computed >= 1
    assert "caus" not in str(doc["failure_mode_evidence"]).lower().replace("not an explanation", "")
    assert dr.document(a.id, "spec")["failure_modes"] == list(modes)


def test_reliability_profile_gets_a_drift_dimension_without_a_score(ev: EvidenceSet) -> None:
    w = ev.w
    a = analyze(ev, spec_of(ev.design.baseline))
    rr = ReliabilityProfileRegistry(w.registry, w.store)
    with_d = run_profile(
        w.registry, w.store, w.executor, ev.investigation, ev.spec(drift_analyses=(a.id,))
    )
    plain = run_profile(w.registry, w.store, w.executor, ev.investigation, ev.spec())
    assert with_d.profile_id and plain.profile_id and with_d.profile_id != plain.profile_id
    dim = rr.document(with_d.profile_id)["dimensions"]["DISTRIBUTION_SHIFT"]
    assert dim["status"] == "DERIVED" and "not shown to cause" in dim["note"]
    obs = dim["observations"][0]
    assert (
        obs["source"]["id"] == a.id
        and obs["analysis_status"] == "COMPLETE"
        and obs["n_window_pairs"] == 1
    )
    pair = next(iter(obs["pairs"].values()))["populations"]["POPULATION"]
    assert (
        pair["features"]["x0"]["status"] == "DERIVED"
        and pair["performance"]["measure"] == "accuracy"
    )
    refs = {
        (r.ref_kind.value, r.ref_id)
        for r in w.registry.find(ReliabilityReference, profile_id=with_d.profile_id)
    }
    assert ("DRIFT_ANALYSIS", a.id) in refs
    off = rr.document(plain.profile_id)["dimensions"]["DISTRIBUTION_SHIFT"]
    assert off["status"] == "UNAVAILABLE" and "no drift analysis" in off["reason"]
    assert "drift_analyses" not in ev.spec().to_dict()  # earlier profile identities are unchanged
    assert ev.spec(drift_analyses=(a.id,)).spec_id != ev.spec().spec_id
    assert "score" not in rr.document(with_d.profile_id)["no_score"].replace("no overall score", "")


def test_a_thin_drift_analysis_is_insufficient_evidence_in_the_profile(ev: EvidenceSet) -> None:
    w = ev.w
    a = analyze(ev, spec_of(ev.design.baseline, reference={"start": 40, "end": 43}, comparisons=[{"start": 60, "end": 63}], features=[], config={**CFG, "min_samples": 10}))  # fmt: skip
    assert a.analysis_status == "PARTIAL"
    p = run_profile(
        w.registry, w.store, w.executor, ev.investigation, ev.spec(drift_analyses=(a.id,))
    )
    doc = ReliabilityProfileRegistry(w.registry, w.store).document(p.profile_id or "")
    assert doc["dimensions"]["DISTRIBUTION_SHIFT"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert (
        doc["dimensions"]["DISTRIBUTION_SHIFT"]["observations"][0]["analysis_status"] == "PARTIAL"
    )


def test_profile_refuses_foreign_or_missing_drift_analyses(ev: EvidenceSet) -> None:
    w = ev.w
    other = launch(w, dropout(0.4), (1, 2), name="another baseline")
    foreign = analyze(ev, spec_of(other.baseline_run_id))
    with pytest.raises(ProfileRefusal, match="INCOMPATIBLE_DRIFT_ANALYSIS"):
        run_profile(
            w.registry, w.store, w.executor, ev.investigation, ev.spec(drift_analyses=(foreign.id,))
        )
    with pytest.raises(ProfileRefusal, match="DRIFT_ANALYSIS_MISSING"):
        run_profile(
            w.registry,
            w.store,
            w.executor,
            ev.investigation,
            ev.spec(drift_analyses=("dan_" + "0" * 32,)),
        )


def test_benchmark_runs_a_drift_analysis_only_when_explicitly_requested(tmp_path: Path) -> None:
    w = small_world(tmp_path)
    plain = spec_for(w)
    drift = spec_of("run_" + "0" * 32)
    asked = spec_for(w, drift=drift)
    assert "drift" not in plain.to_dict() and plain.spec_id == spec_for(w, drift=None).spec_id
    assert asked.spec_id != plain.spec_id and "baseline_run" not in asked.to_dict()["drift"]  # type: ignore[operator]
    out = run_benchmark(w.registry, w.store, w.executor, asked, source_root=w.workspace.parent)
    assert out.result_id, out.errors
    br = BenchmarkRegistry(w.registry, w.store)
    res, cov = br.document(out.result_id, "results"), br.document(out.result_id, "coverage")
    d = res["drift_analysis"]
    assert (
        d["status"] == "DERIVED" and d["analysis_status"] == "COMPLETE" and d["n_window_pairs"] == 1
    )
    assert br.document(out.result_id, "summary")["section_status"]["drift_analysis"] == "DERIVED"
    (a,) = w.registry.find(DriftAnalysis)
    assert (
        a.id == d["analysis_id"] and a.baseline_run_id != "run_" + "0" * 32
    )  # the placeholder became the real baseline
    assert any(
        r.ref_kind.value == "DRIFT_ANALYSIS" and r.ref_id == a.id
        for r in w.registry.find(ReliabilityReference)
    )
    assert cov["complete"] in (True, False)
    rerun = spec_for(w, name="plain-again")
    again = run_benchmark(w.registry, w.store, w.executor, rerun, source_root=w.workspace.parent)
    assert (
        "drift_analysis" not in br.document(again.result_id or "", "results")
        and len(w.registry.find(DriftAnalysis)) == 1
    )


def test_windows_are_registered_and_shared_between_analyses(ev: EvidenceSet) -> None:
    a1 = analyze(ev, spec_of(ev.design.baseline))
    a2 = analyze(ev, spec_of(ev.design.baseline, features=[], config={**CFG, "seed": 2}))
    dr = DriftRegistry(ev.w.registry, ev.w.store)
    ids = set(dr.document(a1.id, "windows")["windows"])
    assert ids == set(dr.document(a2.id, "windows")["windows"]) and a1.id != a2.id
    assert {w.id for w in dr.windows()} >= ids
    ref = TemporalWindow(Role.REFERENCE, *W1)
    assert ref.window_id(Ordering("index").field) in ids

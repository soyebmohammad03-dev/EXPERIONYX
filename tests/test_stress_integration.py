"""Stress evidence linked to the rest of the lab on REAL fault experiments, a real profile and a real
benchmark (the class_data world: threshold classifier, evaluated split indices 40..79). The linked
records are referenced, never copied, and nothing here is a score or a causal claim."""

from pathlib import Path
from typing import Any

import pytest

from benchmark_helpers import ids, small_world, spec_for
from experionyx.benchmark.engine import run_benchmark
from experionyx.benchmark.registry import BenchmarkRegistry
from experionyx.errors import ProfileRefusal
from experionyx.reliability.engine import run_profile
from experionyx.reliability.entities import ReliabilityReference
from experionyx.reliability.registry import ReliabilityProfileRegistry
from experionyx.stress.engine import run_stress_experiment
from experionyx.stress.entities import StressAnalysis
from experionyx.stress.registry import StressRegistry
from experionyx.stress.spec import StressDesign, StressPlan, StressSpec
from interaction_helpers import EVAL, dropout, launch
from reliability_helpers import EvidenceSet, full_evidence

BATCH = StressPlan((StressSpec("BATCH_SIZE", {"batch_size": 9}),))
NOISE = StressPlan((StressSpec("FEATURE_NOISE", {"sigma": 2.0}, seed=3),))


@pytest.fixture(scope="module")
def ev(tmp_path_factory: pytest.TempPathFactory) -> EvidenceSet:
    return full_evidence(tmp_path_factory.mktemp("stress-int"))


def stress(ev: EvidenceSet, plan: StressPlan, **kw: Any) -> StressAnalysis:
    w = ev.w
    model_id, data_id = ids(w)
    design = StressDesign(model_id, data_id, plan, EVAL, baseline_run=kw.pop("baseline", ev.design.baseline), min_members=5, resamples=100, permutations=100, **kw)  # fmt: skip
    out = run_stress_experiment(w.registry, w.store, w.executor, design, source_root=w.workspace.parent, investigation_id=ev.investigation)  # fmt: skip
    assert out.analysis_id, out
    return w.registry.get(StressAnalysis, out.analysis_id)


def test_the_profile_gets_a_model_stress_dimension_without_a_score(ev: EvidenceSet) -> None:
    w = ev.w
    a = stress(ev, NOISE)
    rr = ReliabilityProfileRegistry(w.registry, w.store)
    with_s = run_profile(
        w.registry, w.store, w.executor, ev.investigation, ev.spec(stress_analyses=(a.id,))
    )
    plain = run_profile(w.registry, w.store, w.executor, ev.investigation, ev.spec())
    assert with_s.profile_id and plain.profile_id and with_s.profile_id != plain.profile_id
    dim = rr.document(with_s.profile_id)["dimensions"]["MODEL_STRESS"]
    assert dim["status"] == "DERIVED" and "untested stresses remain untested" in dim["note"]
    obs = dim["observations"][0]
    assert (
        obs["source"]["id"] == a.id
        and obs["analysis_status"] == a.analysis_status
        and obs["origins"] == ["FAULT_LABORATORY"]
    )
    assert (
        obs["coverage"]["completed"] == 1
        and obs["design"] == "SINGLE"
        and "no robustness score" in obs["note"]
    )
    refs = {
        (r.ref_kind.value, r.ref_id)
        for r in w.registry.find(ReliabilityReference, profile_id=with_s.profile_id)
    }
    assert ("STRESS_ANALYSIS", a.id) in refs
    off = rr.document(plain.profile_id)["dimensions"]["MODEL_STRESS"]
    assert (
        off["status"] == "UNAVAILABLE" and "not evidence of robustness" in off["reason"]
    )  # a lack of stress evidence is not "safe"
    assert (
        "stress_analyses" not in ev.spec().to_dict()
        and ev.spec(stress_analyses=(a.id,)).spec_id != ev.spec().spec_id
    )
    assert "score" not in str(rr.document(with_s.profile_id)["dimension_status"]).lower()


def test_a_profile_refuses_stress_analyses_of_another_baseline_or_missing_ones(
    ev: EvidenceSet,
) -> None:
    w = ev.w
    other = launch(w, dropout(0.4), (1, 2), name="another baseline")
    foreign = stress(ev, BATCH, baseline=other.baseline_run_id)
    with pytest.raises(ProfileRefusal, match="INCOMPATIBLE_STRESS_ANALYSIS"):
        run_profile(
            w.registry,
            w.store,
            w.executor,
            ev.investigation,
            ev.spec(stress_analyses=(foreign.id,)),
        )
    with pytest.raises(ProfileRefusal, match="STRESS_ANALYSIS_MISSING"):
        run_profile(
            w.registry,
            w.store,
            w.executor,
            ev.investigation,
            ev.spec(stress_analyses=("sxa_" + "0" * 32,)),
        )


def test_stress_fault_experiments_feed_the_existing_fault_consumers(ev: EvidenceSet) -> None:
    from experionyx.failures.engine import run_discovery
    from experionyx.slices.engine import SliceAnalysisSpec, run_slice_analysis_request
    from experionyx.slices.spec import SliceSpec, between

    w = ev.w
    a = stress(ev, StressPlan((StressSpec("FEATURE_OFFSET", {"offset": 6.0}),)))
    fxp = StressRegistry(w.registry, w.store).document(a.id, "analyses")["links"][
        "fault_experiments"
    ]["ids"]
    assert len(fxp) == 1
    disc = run_discovery(
        w.registry, w.store, w.executor, ev.investigation, [], fxp
    )  # the unchanged Phase 6 entry point
    assert disc.run_id
    sl = run_slice_analysis_request(w.registry, w.store, w.executor, ev.investigation, SliceAnalysisSpec(ev.design.baseline, (SliceSpec("low", between("feature:x0", None, 5)),), (), tuple(fxp)))  # fmt: skip
    assert sl.analysis_id  # the unchanged Phase 11 fault-slice integration reads the same records


def test_a_benchmark_references_a_stress_analysis_only_when_asked(tmp_path: Path) -> None:
    w = small_world(tmp_path)
    plain = spec_for(w)
    asked = spec_for(w, stress=BATCH)
    assert "stress" not in plain.to_dict() and plain.spec_id == spec_for(w, stress=None).spec_id
    assert asked.spec_id != plain.spec_id and asked.to_dict()["stress"] == BATCH.to_dict()
    from experionyx.benchmark.spec import BenchmarkSpec

    assert BenchmarkSpec.from_dict(asked.to_dict()).stress == BATCH  # round trip
    out = run_benchmark(w.registry, w.store, w.executor, asked, source_root=w.workspace.parent)
    assert out.result_id, out.errors
    br = BenchmarkRegistry(w.registry, w.store)
    res = br.document(out.result_id, "results")
    s = res["stress_analysis"]
    assert s["status"] == "DERIVED" and s["planned"] == 1 and s["coverage"]["completed"] == 1
    assert "referenced, not duplicated" in s["note"] and s["source"]["kind"] == "STRESS_ANALYSIS"
    assert br.document(out.result_id, "summary")["section_status"]["stress_analysis"] == "DERIVED"
    (x,) = w.registry.find(StressAnalysis)
    assert x.id == s["analysis_id"]
    assert any(
        r.ref_kind.value == "STRESS_ANALYSIS" and r.ref_id == x.id
        for r in w.registry.find(ReliabilityReference)
    )
    again = run_benchmark(
        w.registry,
        w.store,
        w.executor,
        spec_for(w, name="plain-again"),
        source_root=w.workspace.parent,
    )
    assert (
        "stress_analysis" not in br.document(again.result_id or "", "results")
        and len(w.registry.find(StressAnalysis)) == 1
    )


def test_a_benchmark_reports_an_unsupported_stress_instead_of_faking_it(tmp_path: Path) -> None:
    w = small_world(tmp_path)
    bad = StressPlan(
        (StressSpec("PARAMETER_SCALE", {"factor": 2.0}),)
    )  # the threshold adapter exposes no parameters
    out = run_benchmark(
        w.registry, w.store, w.executor, spec_for(w, stress=bad), source_root=w.workspace.parent
    )
    assert out.result_id, out.errors
    br = BenchmarkRegistry(w.registry, w.store)
    s = br.document(out.result_id, "results")["stress_analysis"]
    assert s["status"] == "UNAVAILABLE" and "UNAVAILABLE" in s["reason"] and s["planned"] == 1
    assert br.document(out.result_id, "coverage")["complete"] is False
    assert not w.registry.find(StressAnalysis)

"""Robustness benchmarks executed for REAL through the fault laboratory, failure discovery,
interaction analysis and reliability profiles. Reported values are checked against the persisted
sources they came from, not against the benchmark's own code path."""

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from benchmark_helpers import BenchmarkLimits, ids, small_world, spec_for
from eval_helpers import EvalWorld, load_evaluation
from experionyx.benchmark.engine import replay_check, run_benchmark
from experionyx.benchmark.entities import Benchmark, BenchmarkResult, BenchmarkUnit
from experionyx.benchmark.registry import BenchmarkRegistry
from experionyx.benchmark.spec import BenchmarkSpec, FaultGrid, InteractionPair
from experionyx.benchmark.taxonomy import CoverageStatus
from experionyx.domain import Claim, Evidence, Run, RunStatus
from experionyx.errors import BenchmarkRefusal, DuplicateError
from experionyx.failures.entities import FailureMode
from experionyx.failures.lifecycle import change_status as mode_change_status
from experionyx.failures.taxonomy import FailureStatus
from experionyx.faults.entities import FaultTrial
from experionyx.faults.report import load_analysis
from experionyx.interactions.entities import InteractionAnalysis, InteractionEffect
from experionyx.reliability.entities import ReliabilityProfile
from reliability_helpers import register_variant

VERDICT_WORDS = (
    "robust",
    "unreliable",
    "safer",
    "better",
    "worse",
    "best",
    "worst",
    "rank",
    "trustworthy",
)


class Ran:
    def __init__(self, w: EvalWorld, spec: BenchmarkSpec) -> None:
        self.w, self.spec = w, spec
        self.out = run_benchmark(
            w.registry, w.store, w.executor, spec, source_root=w.workspace.parent
        )
        self.br = BenchmarkRegistry(w.registry, w.store)

    @property
    def result(self) -> BenchmarkResult:
        assert self.out.result_id, self.out.errors
        return self.br.result(self.out.result_id)

    def doc(self, name: str) -> Any:
        return self.br.document(self.result.id, name)


@pytest.fixture(scope="module")
def bm(tmp_path_factory: pytest.TempPathFactory) -> Ran:
    w = small_world(tmp_path_factory.mktemp("bm"))
    return Ran(w, spec_for(w))


def ran(w: EvalWorld, **over: Any) -> Ran:
    return Ran(w, spec_for(w, **over))


# -- 1. a real, complete benchmark ---------------------------------------------------------------------------------


def test_every_expanded_unit_traces_to_a_real_trial_run_and_evaluation(bm: Ran) -> None:
    w, r = bm.w, bm.result
    assert (
        r.coverage_status is CoverageStatus.COMPLETE
        and bm.out.status is RunStatus.COMPLETED
        and not bm.out.errors
    )
    units = {u.unit_key: u for u in bm.br.units(r.id)}
    assert len(units) == 13 and all(
        u.unit_status.value == "COMPLETED" and u.run_id for u in units.values()
    )  # baseline + 9 fault trials + 3 AB trials
    for key, u in units.items():
        if key.startswith("fault:"):
            trial = next(t for t in w.registry.find(FaultTrial) if t.treatment_run_id == u.run_id)
            assert (trial.fault_id, trial.seed, trial.point_index) == (
                u.detail["fault_id"],
                u.detail["seed"],
                u.detail["point_index"],
            )  # unit -> trial -> run
            assert (
                w.registry.get(Run, u.run_id).status is RunStatus.COMPLETED
                and load_evaluation(w.store, w.registry.get(Run, u.run_id)).n_samples > 0
            )
    assert (
        units["baseline"].run_id
        and units["fault:noise:1:2"].detail["fault"]["parameters"]["sigma"] == 6.0
    )
    ab = [u for k, u in units.items() if ":AB:" in k]
    assert len(ab) == 3 and all(u.detail["fault"]["type"] == "compound" for u in ab)


def test_results_equal_the_persisted_sources_exactly(bm: Ran) -> None:
    w = bm.w
    res = bm.doc("results")
    stored = load_evaluation(
        w.store, w.registry.get(Run, bm.doc("spec")["executed"]["baseline_run"])
    )
    assert {m["metric_id"]: m["value"] for m in res["baseline"]["metrics"]} == {
        m.metric_id: m.value for m in stored.metrics if m.structured is None
    }
    ex = bm.doc("spec")["executed"]
    for g in res["fault_responses"]["grids"]:
        persisted = load_analysis(w.registry, w.store, ex["fault_experiments"][g["grid"]])
        assert len(g["points"]) == len(persisted.points)
        for row, p in zip(g["points"], persisted.points, strict=True):
            assert (
                row["deterioration"]["mean"] == p.primary_deterioration.mean
                and row["completed"] == p.n_completed == 3
            )
            assert (
                row["effect_classification"] == p.assessment.classification.value
                and row["value"] == p.parameter_value
            )
        assert g["status"] == "OBSERVED" and g["fault"]["version"] == "1"
    (pair,) = res["interactions"]["pairs"]
    (eff,) = w.registry.find(
        InteractionEffect, analysis_id=pair["analysis_id"], measure=pair["metric"]
    )
    assert (
        pair["interaction_contrast"] == eff.record["derived"]["interaction_contrast"]
        and pair["label"]
        == w.registry.get(InteractionAnalysis, pair["analysis_id"]).primary_class.value
    )
    prof = w.registry.get(ReliabilityProfile, res["reliability_profile"]["profile_id"])
    assert res["reliability_profile"]["dimension_status"] == json.loads(
        json.dumps(dict(prof.dimension_status))
    )
    assert res["failure_modes"]["modes"] and all(
        m["lifecycle_state"] == w.registry.get(FailureMode, m["mode_id"]).status.value
        for m in res["failure_modes"]["modes"]
    )


def test_interaction_cells_a_and_b_reuse_the_grid_trials_instead_of_rerunning(bm: Ran) -> None:
    w = bm.w
    ex = bm.doc("spec")["executed"]
    (aid,) = ex["interaction_analyses"].values()
    cells: Any = w.registry.get(InteractionAnalysis, aid).spec["cells"]

    def grid_runs(grid: str, point: int) -> set[str | None]:
        found = w.registry.find(FaultTrial, fault_experiment_id=ex["fault_experiments"][grid])
        return {t.treatment_run_id for t in found if t.point_index == point}

    assert set(cells["A"]) == grid_runs("noise", 1) and set(cells["B"]) == grid_runs(
        "dropout", 0
    )  # identical runs, not copies
    assert bm.doc("spec")["interaction_pairs"]["noise+dropout"] == {
        "a_grid": "noise",
        "a_point_index": 1,
        "b_grid": "dropout",
        "b_point_index": 0,
        "order_analysis": False,
        "reuses_grid_trials": True,
    }
    assert len(cells["AB"]) == 3


def test_coverage_is_an_explicit_account_and_never_a_score(bm: Ran) -> None:
    cov = bm.doc("coverage")
    assert cov["complete"] is True and cov["incomplete_reasons"] == []
    assert cov["fault_families"] == {
        **cov["fault_families"],
        "requested": ["dropout", "noise"],
        "tested": ["dropout", "noise"],
        "not_tested": [],
    }
    assert cov["parameter_points"] == {
        "requested": 3,
        "with_any_completed": 3,
        "fully_completed": 3,
    }
    assert cov["trials"] == {
        "requested": 9,
        "completed": 9,
        "failed": 0,
        "skipped": 0,
        "not_run": 0,
    } and cov["seeds"]["distinct_seeds_with_a_completed_trial"] == [1, 2, 3]
    assert (
        cov["interactions"]["requested"]
        == cov["interactions"]["analyzed"]
        == cov["interactions"]["with_interval"]
        == 1
    )
    assert (
        cov["metrics"]["available"]
        and cov["metrics"]["undefined"] == {}
        and cov["failure_modes"]["discovery"] == "RAN"
        and cov["failure_modes"]["discovered"] > 0
    )
    assert (
        cov["failed_or_undefined"] == []
        and cov["unsupported_combinations"] == []
        and "not a measure of robustness" in cov["note"]
    )
    fam = cov["fault_families"]["by_family"]["noise"]
    assert (
        fam["points_requested"],
        fam["points_fully_completed"],
        fam["trials_requested"],
        fam["completed"],
    ) == (2, 2, 6, 6)


def test_the_result_never_contains_a_score_ranking_or_verdict(bm: Ran) -> None:
    summ = bm.doc("summary")
    blob = (
        json.dumps(bm.doc("results")).lower().replace("higher_is_better", "")
        + " "
        + " ".join(s["text"] for s in summ["interpreted"]["statements"]).lower()
    )
    for word in VERDICT_WORDS:
        assert word not in blob, word
    assert (
        summ["no_score"].startswith("no overall score")
        and "score" not in json.dumps(summ["section_status"]).lower()
    )
    assert (
        set(summ["section_status"].values())
        <= {"OBSERVED", "DERIVED", "UNAVAILABLE", "INSUFFICIENT_EVIDENCE"}
        and summ["section_status"]["baseline"] == "OBSERVED"
    )
    texts = [s["text"] for s in summ["interpreted"]["statements"]]
    assert any(
        re.search(
            r"Under fault family gaussian_noise \(sigma=6\) across 3 completed trial\(s\), accuracy changed from",
            t,
        )
        for t in texts
    )
    assert any("not a causal claim" in t for t in texts) and all(
        s["label"] == "INTERPRETED" and s["basis"] for s in summ["interpreted"]["statements"]
    )
    assert "robust" in " ".join(summ["interpreted"]["never_claimed"])


def test_artifacts_are_registered_digest_verified_and_linked_to_the_result(bm: Ran) -> None:
    w, r = bm.w, bm.result
    arts = bm.br.artifacts(r.id)
    assert {a.path for a in arts} == {
        f"benchmark/{n}.json" for n in ("spec", "units", "coverage", "results", "summary")
    }
    for a in arts:
        w.store.verify(w.registry.get(Run, a.run_id), a)
    spec_doc = bm.doc("spec")
    assert (
        spec_doc["spec_id"] == bm.spec.spec_id
        and spec_doc["protocol_hash"] == r.protocol_hash
        and spec_doc["resolved_faults"]["noise"]["version"] == "1"
    )
    prov = bm.br.provenance(r.id)
    assert (
        prov["provenance_fingerprint"] == r.provenance_fingerprint
        and prov["run_id"] == r.run_id
        and prov["model_record_id"] == bm.spec.model
    )
    claims = [c for c in w.registry.find(Claim) if f"[{r.run_id}]" in c.statement]
    assert len(claims) == 1 and "makes no assessment of robustness" in claims[0].statement
    assert (
        len(
            [
                e
                for e in w.registry.find(Evidence, claim_id=claims[0].id)
                if e.target_kind.value == "RUN"
            ]
        )
        == 13
    )


# -- 2. registry, idempotence, refusal -------------------------------------------------------------------------------------------


def test_registry_search_filters_and_duplicate_definitions_are_refused(bm: Ran) -> None:
    b = bm.br.get(bm.result.benchmark_id)
    assert b in bm.br.search(
        name="noise-dropout",
        version="1.0.0",
        model=bm.spec.model,
        dataset=bm.spec.dataset,
        protocol=b.protocol_hash,
        engine=b.engine_version,
    )
    assert (
        b in bm.br.search(fault="gaussian_noise")
        and b in bm.br.search(fault="dropout")
        and b in bm.br.search(coverage="COMPLETE")
    )
    assert (
        b not in bm.br.search(fault="salt_and_pepper")
        and b not in bm.br.search(coverage="INCOMPLETE")
        and b not in bm.br.search(version="9.9.9")
    )
    with pytest.raises(DuplicateError):
        bm.br.register(b)
    assert (
        bm.br.units(bm.result.id, unit_kind="BASELINE")[0].unit_key == "baseline"
        and len(bm.br.units(bm.result.id, unit_status="FAILED")) == 0
    )
    assert bm.br.resolve_result(b.id).id == bm.result.id and bm.br.results(b.id) == [bm.result]


def test_running_the_same_definition_again_is_idempotent(bm: Ran) -> None:
    before = len(bm.w.registry.find(Run))
    again = run_benchmark(
        bm.w.registry, bm.w.store, bm.w.executor, bm.spec, source_root=bm.w.workspace.parent
    )
    assert (
        again.already_run
        and again.result_id == bm.result.id
        and again.benchmark_id == bm.result.benchmark_id
    )
    assert len(bm.w.registry.find(Run)) == before  # nothing was re-executed


def test_invalid_definitions_are_refused_before_any_run_or_record_exists(bm: Ran) -> None:
    w = bm.w
    before = (
        len(w.registry.find(Run)),
        len(w.registry.find(Benchmark)),
        len(w.registry.find(BenchmarkResult)),
    )
    bad = spec_for(
        w, primary_metric="not_a_metric", faults=(FaultGrid("x", "no_such_fault"),), interactions=()
    )
    with pytest.raises(BenchmarkRefusal) as exc:
        run_benchmark(w.registry, w.store, w.executor, bad, source_root=w.workspace.parent)
    assert {i.code for i in exc.value.issues} == {"UNKNOWN_METRIC", "UNKNOWN_FAULT"}  # type: ignore[attr-defined]
    assert (
        len(w.registry.find(Run)),
        len(w.registry.find(Benchmark)),
        len(w.registry.find(BenchmarkResult)),
    ) == before


# -- 3. determinism across independent executions --------------------------------------------------------------------------------------------


def numbers(x: Any) -> list[float]:
    """Every numeric leaf, in key order (booleans excluded)."""
    if isinstance(x, bool):
        return []
    if isinstance(x, int | float):
        return [float(x)]
    if isinstance(x, dict):
        return [n for k in sorted(x) for n in numbers(x[k])]
    if isinstance(x, list):
        return [n for i in x for n in numbers(i)]
    return []


def test_two_independent_executions_agree_on_protocol_units_and_measurements(
    bm: Ran, tmp_path: Path
) -> None:
    w2 = small_world(tmp_path)
    second = Ran(w2, spec_for(w2))
    assert (
        second.spec.spec_id == bm.spec.spec_id
        and second.result.protocol_hash == bm.result.protocol_hash
    )  # same definition, same protocol
    assert second.doc("spec")["protocol_hash"] == bm.doc("spec")["protocol_hash"]
    ua, ub = bm.doc("units")["units"], second.doc("units")["units"]
    assert [(u["key"], u["status"], u["fault_id"]) for u in ua] == [
        (u["key"], u["status"], u["fault_id"]) for u in ub
    ]  # identical units and outcomes
    assert bm.doc("coverage") == second.doc("coverage")
    ra, rb = bm.doc("results"), second.doc("results")
    for section in ("baseline", "fault_responses"):
        a, b = numbers(ra[section]), numbers(rb[section])
        assert len(a) == len(b) and all(
            abs(x - y) <= 1e-9 * max(1.0, abs(x)) for x, y in zip(a, b, strict=True)
        ), section
    ia, ib = ra["interactions"]["pairs"][0], rb["interactions"]["pairs"][0]
    assert (
        ia["interaction_contrast"] == pytest.approx(ib["interaction_contrast"], abs=1e-12)
        and ia["label"] == ib["label"]
        and ia["interval"] == ib["interval"]
    )
    assert len(ra["failure_modes"]["modes"]) == len(rb["failure_modes"]["modes"])
    assert (
        bm.doc("summary")["interpreted"]["statements"][1]["text"]
        == second.doc("summary")["interpreted"]["statements"][1]["text"]
    )


def test_replay_of_the_collect_run_reproduces_all_documents(tmp_path: Path) -> None:
    w = small_world(tmp_path)
    r = Ran(
        w,
        spec_for(
            w,
            interactions=(),
            faults=(FaultGrid("noise", "gaussian_noise", sweep_parameter="sigma", values=(3.0,)),),
        ),
    )
    rep = replay_check(w.registry, w.store, w.executor, r.result.id)
    assert (
        rep["deterministic"] is True
        and rep["sources_changed"] is False
        and rep["differences"] == []
        and rep["replay_run"] != rep["original_run"]
    )


def test_replay_reports_changed_evidence_instead_of_claiming_nondeterminism(tmp_path: Path) -> None:
    w = small_world(tmp_path)
    r = Ran(
        w,
        spec_for(
            w,
            interactions=(),
            faults=(FaultGrid("noise", "gaussian_noise", sweep_parameter="sigma", values=(6.0,)),),
        ),
    )
    mode = next(
        m
        for m in w.registry.find(FailureMode)
        if m.status in (FailureStatus.DISCOVERED, FailureStatus.CANDIDATE)
    )
    before = r.doc("results")
    mode_change_status(
        w.registry,
        mode.id,
        FailureStatus.REJECTED,
        "reviewer",
        "test: evidence changes after collection",
    )
    rep = replay_check(w.registry, w.store, w.executor, r.result.id)
    assert (
        rep["deterministic"] is None
        and rep["sources_changed"] is True
        and "evidence changed" in rep["note"]
    )
    assert r.doc("results") == before  # the registered result is an immutable snapshot


# -- 4. incomplete coverage is reported, never hidden --------------------------------------------------------------------------------------------


def test_failed_trials_make_coverage_incomplete_with_reasons_and_no_fabricated_response(
    tmp_path: Path,
) -> None:
    w = small_world(tmp_path)
    grids = (
        FaultGrid("noise", "gaussian_noise", sweep_parameter="sigma", values=(6.0,)),
        FaultGrid("nan", "missing_values", sweep_parameter="probability", values=(0.9,)),
    )  # the model raises on NaN input
    r = Ran(w, spec_for(w, faults=grids, interactions=(), discovery=False, profile=False))
    assert (
        r.result.coverage_status is CoverageStatus.INCOMPLETE
        and r.out.status is RunStatus.COMPLETED
    )
    cov = r.doc("coverage")
    assert cov["fault_families"]["tested"] == ["noise"] and cov["fault_families"]["not_tested"] == [
        "nan"
    ]
    assert cov["trials"] == {
        "requested": 6,
        "completed": 3,
        "failed": 3,
        "skipped": 0,
        "not_run": 0,
    }
    failed = [u for u in cov["failed_or_undefined"] if u["kind"] == "FAULT_TRIAL"]
    assert len(failed) == 3 and all(
        u["status"] == "FAILED" and "NaN" in u["reason"] for u in failed
    )  # every failure has its reason
    assert any("did not complete" in x for x in cov["incomplete_reasons"]) and any(
        "completed no trial" in x for x in cov["incomplete_reasons"]
    )
    res = r.doc("results")
    nan = next(g for g in res["fault_responses"]["grids"] if g["grid"] == "nan")
    assert (
        nan["status"] == "INSUFFICIENT_EVIDENCE"
        and nan["points"][0]["completed"] == 0
        and nan["points"][0]["deterioration"]["mean"] is None
    )  # nothing invented
    first = r.doc("summary")["interpreted"]["statements"][0]["text"]
    assert first.startswith("Coverage is INCOMPLETE") and "not evidence of robustness" in first
    assert all(
        "nan" not in s["text"].lower() or "Coverage" in s["text"]
        for s in r.doc("summary")["interpreted"]["statements"][1:]
    )  # no response statement for the family without data
    assert {u.unit_status.value for u in r.br.units(r.result.id, grid="nan")} == {"FAILED"}


def test_early_termination_skips_units_visibly(tmp_path: Path) -> None:
    w = small_world(tmp_path)
    r = Ran(
        w,
        spec_for(
            w,
            faults=(
                FaultGrid("nan", "missing_values", sweep_parameter="probability", values=(0.9,)),
            ),
            interactions=(),
            discovery=False,
            profile=False,
            seeds=(1, 2, 3, 4),
            limits=BenchmarkLimits(max_failed_trials=1),
        ),
    )
    cov = r.doc("coverage")
    assert (
        cov["trials"] == {"requested": 4, "completed": 0, "failed": 1, "skipped": 3, "not_run": 0}
        and cov["complete"] is False
    )
    assert any(
        u["status"] == "SKIPPED" and "max_failed_trials=1" in u["reason"]
        for u in cov["failed_or_undefined"]
    )


def test_too_few_trials_leave_the_interaction_inconclusive_and_say_so(tmp_path: Path) -> None:
    w = small_world(tmp_path)
    r = ran(w, seeds=(1, 2), discovery=False, profile=False)
    cov = r.doc("coverage")
    assert (
        cov["interactions"]["analyzed"] == 1
        and cov["interactions"]["with_interval"] == 0
        and cov["interactions"]["inconclusive"] == ["noise+dropout"]
    )
    pair = r.doc("results")["interactions"]["pairs"][0]
    assert (
        pair["label"] == "INCONCLUSIVE"
        and pair["status"] == "INSUFFICIENT_EVIDENCE"
        and pair["interval"] is None
    )
    assert r.result.section_status["interactions"] == "INSUFFICIENT_EVIDENCE" and any(
        "inconclusive" in x for x in cov["incomplete_reasons"]
    )


def test_unsupported_grids_are_recorded_in_the_coverage_when_allowed(tmp_path: Path) -> None:
    w = small_world(tmp_path)
    grids = (
        FaultGrid("noise", "gaussian_noise", sweep_parameter="sigma", values=(6.0,)),
        FaultGrid("occ", "random_occlusion", sweep_parameter="fraction", values=(0.5,)),
    )
    r = Ran(
        w,
        spec_for(
            w,
            faults=grids,
            interactions=(),
            discovery=False,
            profile=False,
            limits=BenchmarkLimits(allow_unsupported=True),
        ),
    )
    cov = r.doc("coverage")
    assert [u["grid"] for u in cov["unsupported_combinations"]] == ["occ"] and cov[
        "fault_families"
    ]["by_family"]["occ"]["status"] == "UNSUPPORTED"
    assert (
        cov["complete"] is False
        and any("unsupported" in x for x in cov["incomplete_reasons"])
        and cov["trials"]["requested"] == 3
    )
    assert "occ" not in {
        u.grid for u in r.br.units(r.result.id)
    }  # it was never executed, and that is on record


def test_disabled_stages_are_unavailable_with_a_reason_not_silently_missing(tmp_path: Path) -> None:
    w = small_world(tmp_path)
    r = ran(w, discovery=False, profile=False, interactions=())
    res, cov = r.doc("results"), r.doc("coverage")
    assert cov["complete"] is True  # what was requested was executed
    assert (
        res["failure_modes"]["status"] == "UNAVAILABLE"
        and "disabled" in res["failure_modes"]["reason"]
    )
    assert (
        res["reliability_profile"]["status"] == "UNAVAILABLE"
        and res["interactions"]["status"] == "UNAVAILABLE"
    )
    assert (
        r.result.section_status["failure_modes"] == "UNAVAILABLE"
        and cov["failure_modes"]["discovery"] == "DISABLED"
    )


# -- 5. versioning and comparison ------------------------------------------------------------------------------------------------------------------


def test_a_new_definition_version_is_detectable_and_old_results_stay_interpretable(bm: Ran) -> None:
    w = bm.w
    v2 = Ran(w, replace(bm.spec, version="1.0.1", discovery=False, profile=False, interactions=()))
    assert (
        v2.result.protocol_hash != bm.result.protocol_hash
        and v2.result.provenance_fingerprint != bm.result.provenance_fingerprint
    )
    assert bm.br.get(v2.result.benchmark_id).spec_id != bm.br.get(bm.result.benchmark_id).spec_id
    assert (
        bm.br.get(bm.result.benchmark_id).spec["version"] == "1.0.0"
        and bm.doc("spec")["spec"]["version"] == "1.0.0"
    )  # the old result still carries its exact definition
    with pytest.raises(BenchmarkRefusal) as exc:
        bm.br.compare(bm.result.id, v2.result.id)
    assert {i.code for i in exc.value.issues} == {"PROTOCOL_DIFFERS"}  # type: ignore[attr-defined]
    assert any("version" in str(i) for i in exc.value.issues) and any(
        "interactions" in str(i) for i in exc.value.issues
    )  # type: ignore[attr-defined]
    assert len(bm.br.search(name="noise-dropout")) >= 2


def test_two_models_under_the_same_protocol_compare_by_raw_differences_with_no_winner(
    bm: Ran,
) -> None:
    w = bm.w
    m2, _ = register_variant(w, threshold=3.0, tag="cmp")
    other = Ran(w, replace(bm.spec, model=m2))
    assert (
        other.result.protocol_hash == bm.result.protocol_hash
        and other.result.benchmark_id != bm.result.benchmark_id
    )
    out = bm.br.compare(bm.result.id, other.result.id)
    assert out["same_model"] is False and out["protocol_hash"] == bm.result.protocol_hash
    acc = next(m for m in out["baseline"]["metrics"] if m["metric_id"] == "accuracy")
    assert acc["difference"] == pytest.approx(acc["b"] - acc["a"]) and acc["a"] != acc["b"]
    fam = {f["grid"]: f for f in out["fault_responses"]["families"]}
    assert (
        set(fam) == {"noise", "dropout"}
        and len(fam["noise"]["points"]) == 2
        and not fam["noise"]["points_only_in_a"]
    )
    pt = fam["noise"]["points"][1]["deterioration_mean"]
    assert pt["difference"] == pytest.approx(pt["b"] - pt["a"])
    assert out["coverage"]["complete"] == {"a": True, "b": True} and set(
        out["section_status"]
    ) == set(bm.result.section_status)
    blob = (
        json.dumps(out)
        .lower()
        .replace("higher_is_better", "")
        .replace("no overall ranking, winner or verdict is computed", "")
    )
    for word in ("winner", "better", "worse", "best", "worst", "rank", "verdict"):
        assert word not in blob
    same = bm.br.compare(bm.result.id, bm.result.id)
    assert same["same_model"] is True and all(
        m["difference"] == 0 for m in same["baseline"]["metrics"]
    )


def test_provenance_identity_changes_with_every_meaningful_input(bm: Ran) -> None:
    w = bm.w
    base = bm.result.provenance_fingerprint
    variants = {
        "seeds": replace(bm.spec, seeds=(1, 2, 4), discovery=False, profile=False),
        "grid point": replace(
            bm.spec,
            faults=(bm.spec.faults[0], replace(bm.spec.faults[1], values=(2.0, 5.0))),
            interactions=(InteractionPair("noise", "dropout", 5.0, 0.5),),
            discovery=False,
            profile=False,
        ),
    }
    seen = {base}
    for name, spec in variants.items():
        r = Ran(w, spec)
        assert (
            r.result.provenance_fingerprint not in seen
            and r.result.protocol_hash != bm.result.protocol_hash
        ), name
        seen.add(r.result.provenance_fingerprint)
    assert (
        bm.result.id
        == BenchmarkResult(
            bm.result.benchmark_id,
            bm.result.investigation_id,
            bm.result.run_id,
            bm.result.protocol_hash,
            base,
            bm.result.coverage_status,
            bm.result.section_status,
            bm.result.summary,
            bm.result.created_at,
        ).id
    )


def test_unit_and_result_records_are_immutable_and_reference_real_runs(bm: Ran) -> None:
    import sqlite3

    for u in bm.br.units(bm.result.id):
        if u.run_id:
            assert bm.w.registry.exists(Run, u.run_id)
    path = bm.w.workspace / "registry.sqlite"
    with sqlite3.connect(path) as raw:
        for table in ("benchmarks", "benchmark_results", "benchmark_units"):
            assert raw.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] > 0  # noqa: S608
            with pytest.raises(sqlite3.DatabaseError):
                raw.execute(f"DELETE FROM {table}")  # noqa: S608
            with pytest.raises(sqlite3.DatabaseError):
                raw.execute(f"UPDATE {table} SET payload = '{{}}'")  # noqa: S608
    assert ids(bm.w)[0] == bm.spec.model and isinstance(bm.br.units(bm.result.id)[0], BenchmarkUnit)

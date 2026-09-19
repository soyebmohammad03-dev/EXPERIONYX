"""Slice analysis end to end on REAL fault experiments, a real discovery, a real interaction
analysis, a real reliability profile and a real benchmark. Every expected number is recomputed
independently from the raw per-sample prediction files (not through the slice engine) and from the
dataset's own formula: the evaluated split is indices 40..79 and feature x0 is `index % 10`."""

import json
import shutil
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from benchmark_helpers import small_world, spec_for
from eval_helpers import EvalWorld, load_evaluation, pure_registries
from experionyx.benchmark.engine import run_benchmark
from experionyx.benchmark.registry import BenchmarkRegistry
from experionyx.domain import Run, RunStatus
from experionyx.errors import ArtifactIntegrityError, ProfileRefusal, ValidationError
from experionyx.failures.config import DiscoveryConfig, ExtractionConfig
from experionyx.failures.engine import run_discovery
from experionyx.failures.entities import FailureCluster, FailureMode, FailureSignal
from experionyx.faults.entities import FaultTrial
from experionyx.reliability.engine import run_profile
from experionyx.reliability.entities import ReliabilityReference
from experionyx.reliability.registry import ReliabilityProfileRegistry
from experionyx.slices.data import load_baseline, sample_table
from experionyx.slices.engine import SliceAnalysisSpec, compute, run_slice_analysis_request
from experionyx.slices.entities import Slice, SliceAnalysis
from experionyx.slices.evaluate import evaluate
from experionyx.slices.registry import SliceRegistry
from experionyx.slices.spec import SliceSpec, all_of, between, eq, label
from interaction_helpers import dropout, launch
from reliability_helpers import EvidenceSet, full_evidence

TEST_IDS = range(40, 80)  # class_data(80): the test split
CFG = {"resamples": 200, "min_members": 5, "min_trials": 3}


@pytest.fixture(scope="module")
def ev(tmp_path_factory: pytest.TempPathFactory) -> EvidenceSet:
    return full_evidence(tmp_path_factory.mktemp("slice"))


def raw_rows(w: EvalWorld, run_id: str) -> dict[int, dict[str, Any]]:
    """A run's predictions read straight from its artifact file."""
    run = w.registry.get(Run, run_id)
    path = w.store.run_dir(run) / "artifacts" / "evaluation" / "predictions.jsonl"
    return {r["index"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def acc(rows: dict[int, dict[str, Any]], ids: list[int]) -> float:
    return float(sum(rows[i]["true"] == rows[i]["predicted"] for i in ids) / len(ids))


def x0(i: int) -> int:
    return i % 10


LOW = SliceSpec("low-x0", between("feature:x0", None, 5))
ONES = SliceSpec("ones", label(1))
HARD = SliceSpec("hard", all_of(label(1), between("feature:x0", 5, None)))
TINY = SliceSpec("tiny", eq("feature:x0", 3))  # 4 members: below min_members
WRONG = SliceSpec("wrong", eq("correct", False))  # reads a model OUTPUT: run-dependent membership
LOW_IDS = [i for i in TEST_IDS if x0(i) < 5]


def request(
    ev: EvidenceSet, slices: tuple[SliceSpec, ...] = (LOW, ONES, TINY), **over: Any
) -> SliceAnalysisSpec:
    cfg = {**CFG, **over.pop("config", {})}
    from experionyx.slices.analysis import SliceConfig

    return SliceAnalysisSpec(ev.design.baseline, slices, config=SliceConfig(**cfg), **over)


def analyze(ev: EvidenceSet, spec: SliceAnalysisSpec) -> SliceAnalysis:
    w = ev.w
    out = run_slice_analysis_request(w.registry, w.store, w.executor, ev.investigation, spec)
    assert out.status is RunStatus.COMPLETED and out.analysis_id
    return w.registry.get(SliceAnalysis, out.analysis_id)


def results(ev: EvidenceSet, a: SliceAnalysis, name: str = "results") -> Any:
    return SliceRegistry(ev.w.registry, ev.w.store).document(a.id, name)


# -- membership, metrics, comparisons vs independent recomputation --------------------------------------------------


def test_membership_metrics_and_comparisons_match_independent_recomputation(
    ev: EvidenceSet,
) -> None:
    a = analyze(ev, request(ev))
    mem, res = results(ev, a, "membership"), results(ev, a)
    rows = raw_rows(ev.w, ev.design.baseline)
    assert sorted(mem["low-x0"]["sample_ids"]) == LOW_IDS and mem["low-x0"]["n_members"] == 20
    assert mem["low-x0"]["prevalence"] == 20 / 40 and mem["low-x0"]["status"] == "COMPUTED"
    ones = [i for i in TEST_IDS if rows[i]["true"] == 1]
    assert sorted(mem["ones"]["sample_ids"]) == ones
    assert mem["tiny"]["sample_ids"] == [i for i in TEST_IDS if x0(i) == 3]

    m = next(x for x in res["slices"]["low-x0"]["metrics"] if x["metric_id"] == "accuracy")
    assert m["value"] == pytest.approx(acc(rows, LOW_IDS), abs=1e-12) and m["n_samples"] == 20
    assert m["interval"]["seed"] == 0 and m["interval"]["resamples"] == 200
    errs = res["slices"]["low-x0"]["errors"]
    assert errs["n_errors"] == sum(rows[i]["true"] != rows[i]["predicted"] for i in LOW_IDS)
    assert (
        res["slices"]["tiny"]["evidence"] == "INSUFFICIENT_EVIDENCE"
        and a.analysis_status == "PARTIAL"
    )
    assert res["slices"]["low-x0"]["latency"]["status"] == "UNAVAILABLE"

    cmp = res["comparisons"]["low-x0 vs POPULATION"]
    pop = acc(rows, list(TEST_IDS))
    assert cmp["descriptive_vs_population"]["difference"] == pytest.approx(
        acc(rows, LOW_IDS) - pop, abs=1e-12
    )
    rest = [i for i in TEST_IDS if i not in LOW_IDS]
    inf = cmp["vs_rest"]
    assert inf["descriptive"]["difference"] == pytest.approx(
        acc(rows, LOW_IDS) - acc(rows, rest), abs=1e-12
    )
    assert inf["inference"]["pairing"] == "UNPAIRED" and inf["n_a"] == 20 and inf["n_b"] == 20
    assert res["multiple_comparisons"]["method"] == "NONE" and "multiplicity" in cmp
    assert "score" in res["note"] and "ranks nothing" in res["note"]
    text = json.dumps(res).lower()
    assert "fairness_score" not in text and "rank_" not in text and "winner" not in text


def test_explicit_comparisons_and_correction_are_configured_and_recorded(ev: EvidenceSet) -> None:
    spec = request(
        ev,
        (LOW, ONES, HARD),
        comparisons=(("hard", "ones"), ("low-x0", "REST"), ("low-x0", "hard")),
        config={"correction": "BENJAMINI_HOCHBERG", "alpha": 0.1},
    )
    res = results(ev, analyze(ev, spec))
    assert sorted(res["comparisons"]) == ["hard vs ones", "low-x0 vs REST", "low-x0 vs hard"]
    rows = raw_rows(ev.w, ev.design.baseline)
    ones = [i for i in TEST_IDS if rows[i]["true"] == 1]
    hard = [i for i in ones if x0(i) >= 5]
    assert set(hard) <= set(ones)
    # hard is a subset of ones: the groups overlap, so no independent-groups test is run
    sub = res["comparisons"]["hard vs ones"]
    assert sub["status"] == "UNDEFINED" and sub["inference"] is None and "share" in sub["reason"]
    assert sub["descriptive"]["difference"] == pytest.approx(
        acc(rows, hard) - acc(rows, ones), abs=1e-12
    )
    corr = res["multiple_comparisons"]
    assert corr["method"] == "BENJAMINI_HOCHBERG" and corr["alpha"] == 0.1
    tested = {k for k, c in res["comparisons"].items() if c.get("inference")}
    assert corr["n_hypotheses"] == len(tested) and set(corr["adjusted"]) == tested
    for k in tested:
        assert (
            res["comparisons"][k]["multiplicity"]["adjusted_p"]
            >= res["comparisons"][k]["multiplicity"]["raw_p"]
        )


# -- provenance, identity, artifacts ------------------------------------------------------------------------------


def test_provenance_identity_and_artifacts(ev: EvidenceSet) -> None:
    w = ev.w
    a = analyze(ev, request(ev))
    sr = SliceRegistry(w.registry, w.store)
    base_ev = load_evaluation(w.store, w.registry.get(Run, ev.design.baseline))
    assert (
        a.dataset_fingerprint == base_ev.context.dataset_fingerprint
        and a.baseline_run_id == ev.design.baseline
    )
    prov: Any = sr.provenance(a.id)
    assert prov["split"] == "test" and set(prov["slice_ids"]) == {"low-x0", "ones", "tiny"}
    assert prov["slice_ids"]["low-x0"] == LOW.slice_id and prov["metric_config"]["min_members"] == 5
    assert prov["evaluation_config_hash"] and prov["source_runs"]["baseline"] == ev.design.baseline
    assert {x.path for x in sr.artifacts(a.id)} == {
        "slice/spec.json",
        "slice/membership.json",
        "slice/results.json",
        "slice/summary.json",
    }
    assert sr.document(a.id, "summary")["slices"]["low-x0"]["headline"]["value"] is not None
    assert "sample_ids" in json.dumps(
        sr.document(a.id, "membership")
    )  # IDs only: no raw dataset copy
    assert "rows" not in json.dumps(sr.document(a.id, "spec"))

    again = analyze(ev, request(ev))
    assert again.id == a.id  # the same request is the same analysis
    changed = analyze(
        ev, request(ev, (SliceSpec("low-x0", between("feature:x0", None, 6)), ONES, TINY))
    )
    assert changed.id != a.id and changed.spec_id != a.spec_id
    assert changed.provenance_fingerprint != a.provenance_fingerprint
    reconfigured = analyze(ev, request(ev, config={"resamples": 300}))
    assert (
        reconfigured.id != a.id and reconfigured.provenance_fingerprint != a.provenance_fingerprint
    )
    renamed = analyze(
        ev, request(ev, (SliceSpec("lower", between("feature:x0", None, 5)), ONES, TINY))
    )
    assert renamed.spec_id != a.spec_id  # names appear in outputs, so they are part of the request
    assert (
        len(sr.search(field="feature:x0", text="< 5")) == 1
    )  # one logical record for the equivalent slice
    assert sum(s.spec().slice_id == LOW.slice_id for s in sr.list()) == 1
    assert sr.get(LOW.slice_id).fields == ("feature:x0",)


def test_registry_register_get_search_evaluate(ev: EvidenceSet) -> None:
    from experionyx.slices.data import dataset_for_run

    w = ev.w
    sr = SliceRegistry(w.registry, w.store)
    s, _ = sr.register(HARD)
    again, created2 = sr.register(
        SliceSpec("other name", all_of(between("feature:x0", 5, None), label(1)))
    )
    assert not created2 and again.id == s.id == HARD.slice_id  # equivalent: one record
    assert sr.get(s.id) == s and s.name in {
        "hard",
        "other name",
    }  # the first registered name is the label
    assert HARD.slice_id in {x.id for x in sr.search(field="target")}
    assert HARD.slice_id in {x.id for x in sr.search(static=True)}
    sr.register(WRONG)
    assert WRONG.slice_id not in {x.id for x in sr.search(static=True)}
    assert WRONG.slice_id in {x.id for x in sr.search(static=False, field="correct")}
    assert [x.id for x in sr.search(name="hard")] == [HARD.slice_id]
    assert {x.id for x in sr.search(text="target == 1")} >= {HARD.slice_id}
    assert [x.id for x in sr.list()] == sorted(x.id for x in sr.list())
    with pytest.raises(Exception, match="dataset"):
        sr.evaluate(
            HARD, ev.design.baseline
        )  # feature slices need the dataset adapter: refused, not guessed
    ds = dataset_for_run(w.registry, pure_registries(), w.workspace, ev.design.baseline)
    assert ds is not None
    m = sr.evaluate(HARD, ev.design.baseline, ds)
    rows = raw_rows(w, ev.design.baseline)
    assert list(m.sample_ids) == [i for i in TEST_IDS if rows[i]["true"] == 1 and x0(i) >= 5]
    assert list(sr.evaluate(LOW, ev.design.baseline, ds).sample_ids) == LOW_IDS
    m2 = sr.evaluate(ONES, ev.design.baseline)  # target-only slices need no dataset
    assert list(m2.sample_ids) == [i for i in TEST_IDS if rows[i]["true"] == 1]
    w_out = sr.evaluate(WRONG, ev.design.baseline)  # an outcome slice is evaluable on ONE run
    assert list(w_out.sample_ids) == [
        i for i in TEST_IDS if rows[i]["true"] != rows[i]["predicted"]
    ]
    with pytest.raises(ValidationError):
        SliceRegistry(w.registry).evaluate(ONES, ev.design.baseline)  # no store


def test_invalid_requests_are_refused_before_any_run(ev: EvidenceSet) -> None:
    w = ev.w
    n = len(w.registry.find(Run))
    cases: tuple[Callable[[], object], ...] = (
        lambda: SliceAnalysisSpec(ev.design.baseline, ()),
        lambda: SliceAnalysisSpec(ev.design.baseline, (LOW, SliceSpec("low-x0", label(1)))),
        lambda: SliceAnalysisSpec(
            ev.design.baseline, (LOW, SliceSpec("again", between("feature:x0", None, 5)))
        ),
        lambda: SliceAnalysisSpec(ev.design.baseline, (LOW,), comparisons=(("low-x0", "nope"),)),
        lambda: SliceAnalysisSpec(
            ev.design.baseline, (LOW,), comparisons=(("POPULATION", "low-x0"),)
        ),
        lambda: SliceAnalysisSpec.from_dict(
            {"baseline_run": ev.design.baseline, "slices": [LOW.to_dict()], "bogus": 1}
        ),
    )
    for bad in cases:
        with pytest.raises(ValidationError):
            bad()
    with pytest.raises(Exception, match=r"not found|NotFound|unknown|no such|does not exist"):
        run_slice_analysis_request(
            w.registry,
            w.store,
            w.executor,
            ev.investigation,
            SliceAnalysisSpec(ev.design.baseline, (LOW,), failure_modes=("fmd_" + "0" * 32,)),
        )
    assert len(w.registry.find(Run)) == n  # nothing was created


def test_missing_feature_data_makes_the_slice_unavailable_not_empty(ev: EvidenceSet) -> None:
    res = results(ev, analyze(ev, request(ev, (SliceSpec("ghost", eq("feature:nope", 1)), ONES))))
    assert (
        res["slices"]["ghost"]["status"] == "UNAVAILABLE"
        and "not a feature" in res["slices"]["ghost"]["reason"]
    )
    assert res["slices"]["ones"]["status"] == "COMPUTED"


# -- adversarial: order cannot change membership or results ------------------------------------------------------------


def test_reordering_samples_cannot_change_membership_or_results(ev: EvidenceSet) -> None:
    w = ev.w
    base = load_baseline(w.registry, w.store, ev.design.baseline)
    from dataclasses import replace

    from experionyx.slices import analysis as an

    rev = replace(base, rows=dict(reversed(list(base.rows.items()))))
    cfg = an.SliceConfig(resamples=200)
    t1, t2 = sample_table(base, None, ("target",)), sample_table(rev, None, ("target",))
    m1, m2 = evaluate(ONES, t1), evaluate(ONES, t2)
    assert m1 == m2 and m1.membership_digest == m2.membership_digest
    assert an.slice_metrics(base, m1.sample_ids, cfg) == an.slice_metrics(rev, m2.sample_ids, cfg)
    rest = [i for i in sorted(base.rows) if i not in set(m1.sample_ids)]
    assert an.compare_groups(base, m1.sample_ids, rest, cfg) == an.compare_groups(
        rev, m2.sample_ids, list(reversed(rest)), cfg
    )


def test_a_reordered_or_edited_predictions_file_is_detected_not_silently_used(
    ev: EvidenceSet, tmp_path: Path
) -> None:
    w = ev.w
    run = w.registry.get(Run, ev.design.baseline)
    path = w.store.run_dir(run) / "artifacts" / "evaluation" / "predictions.jsonl"
    keep = tmp_path / "keep.jsonl"
    shutil.copy(path, keep)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text(
            "\n".join(reversed(lines)) + "\n", encoding="utf-8"
        )  # same rows, different order
        with pytest.raises(ArtifactIntegrityError):
            load_baseline(w.registry, w.store, ev.design.baseline)
    finally:
        shutil.copy(keep, path)
    assert load_baseline(w.registry, w.store, ev.design.baseline).rows


# -- faults x slices ---------------------------------------------------------------------------------------------------


def test_fault_slice_matches_independent_trial_recomputation(ev: EvidenceSet) -> None:
    w = ev.w
    fx = ev.design.a.fault_experiment.id  # gaussian noise, 4 seeds
    a = analyze(ev, request(ev, (LOW, ONES), fault_experiments=(fx,)))
    res = results(ev, a)
    rows = raw_rows(w, ev.design.baseline)
    pt = res["faults"]["low-x0"][0]["points"][0]
    assert pt["n_trials"] == 4 and pt["baseline_slice"] == pytest.approx(
        acc(rows, LOW_IDS), abs=1e-12
    )
    trials = sorted(w.registry.find(FaultTrial, fault_experiment_id=fx), key=lambda t: t.seed)
    want_slice, want_all = {}, {}
    for t in trials:
        r = raw_rows(w, t.treatment_run_id or "")
        key = f"seed{t.seed}/repeat{t.repeat_index}"
        want_slice[key] = acc(rows, LOW_IDS) - acc(r, LOW_IDS)  # accuracy: positive means worse
        want_all[key] = acc(rows, list(TEST_IDS)) - acc(r, list(TEST_IDS))
    assert pt["trial_degradation"]["slice"] == pytest.approx(want_slice, abs=1e-12)
    assert pt["trial_degradation"]["population"] == pytest.approx(want_all, abs=1e-12)
    diff = pt["slice_vs_population_degradation"]
    md = next(e for e in diff["effects"] if e["name"] == "mean_difference")["value"]
    assert md == pytest.approx(
        statistics.fmean(want_slice[k] - want_all[k] for k in want_slice), abs=1e-12
    )
    assert diff["pairing"] == "PAIRED" and pt["effect_classification"] in {
        "SLICE_MORE_DEGRADED",
        "SLICE_LESS_DEGRADED",
        "NO_EVIDENCE_OF_DIFFERENCE",
    }
    assert (
        "not evidence of absence" in pt["classification_note"]
        and "nothing here says why" in pt["classification_note"]
    )
    # the classification follows the interval of the paired difference, nothing else
    iv = diff["interval"]
    excludes = iv["lower"] > 0 or iv["upper"] < 0
    assert (pt["effect_classification"] != "NO_EVIDENCE_OF_DIFFERENCE") == excludes


def test_fault_slice_refusals_and_insufficiency_are_explicit(ev: EvidenceSet) -> None:
    w = ev.w
    fx = ev.design.a.fault_experiment.id
    few = results(
        ev, analyze(ev, request(ev, (LOW,), fault_experiments=(fx,), config={"min_trials": 5}))
    )
    pt = few["faults"]["low-x0"][0]["points"][0]
    assert (
        pt["effect_classification"] == "INSUFFICIENT_EVIDENCE"
        and pt["status"] == "INSUFFICIENT_EVIDENCE"
    )
    assert (
        "min_trials=5" in pt["reason"] and pt["slice_degradation"]["status"] == "DERIVED"
    )  # raw evidence kept
    nonstatic = results(ev, analyze(ev, request(ev, (LOW, WRONG), fault_experiments=(fx,))))
    assert (
        nonstatic["faults"]["wrong"][0]["status"] == "REFUSED"
        and "differs between runs" in nonstatic["faults"]["wrong"][0]["reason"]
    )
    small = results(ev, analyze(ev, request(ev, (TINY,), fault_experiments=(fx,))))
    assert (
        small["faults"]["tiny"][0]["status"] == "INSUFFICIENT_EVIDENCE"
        and "min_members" in small["faults"]["tiny"][0]["reason"]
    )
    other = launch(
        w, dropout(0.5), (1, 2, 3, 4), name="other baseline"
    )  # its own, different baseline run
    assert other.baseline_run_id != ev.design.baseline
    res = results(
        ev, analyze(ev, request(ev, (LOW,), fault_experiments=(other.fault_experiment.id,)))
    )
    r0 = res["faults"]["low-x0"][0]
    assert (
        r0["status"] == "INSUFFICIENT_EVIDENCE"
        and "run against baseline" in r0["reason"]
        and ev.design.baseline in r0["reason"]
    )


# -- failure modes x slices -----------------------------------------------------------------------------------------------


def test_failure_mode_slice_counts_match_the_signals_and_respect_lifecycle(ev: EvidenceSet) -> None:
    w = ev.w
    all_modes = tuple(sorted(m.id for m in w.registry.find(FailureMode)))
    res = results(ev, analyze(ev, request(ev, (LOW, ONES), failure_modes=all_modes)))
    members = set(LOW_IDS)
    seen = {"COMPUTED": 0, "INSUFFICIENT_EVIDENCE": 0}
    for row in res["failure_modes"]["low-x0"]["modes"]:
        mode = w.registry.get(FailureMode, row["mode_id"])
        sigs = [
            w.registry.get(FailureSignal, sid)
            for sid in w.registry.get(FailureCluster, mode.cluster_id).signal_ids
        ]
        usable = [
            s
            for s in sigs
            if s.detail.get("sample_ids_recorded") and s.sample_count == len(s.sample_ids)
        ]
        assert row["lifecycle_status"] == mode.status.value and row["confirmed"] is (
            mode.status.value == "CONFIRMED"
        )
        assert row["confirmed"] is False  # nothing in this registry was confirmed by a person
        if len(usable) == len(sigs):  # every signal recorded all of its sample IDs
            touched: set[int] = {int(x) for s in sigs for x in s.sample_ids}
            assert row["status"] == "COMPUTED"
            assert (
                row["slice"]["samples_affected"] == len(touched & members)
                and row["slice"]["samples"] == 20
            )
            assert row["present_in_slice"] == bool(touched & members)
            assert row["outside_slice"]["samples_affected"] == len(
                touched & (set(TEST_IDS) - members)
            )
            assert (
                row["population"]["samples_affected"] == len(touched & set(TEST_IDS))
                and row["population"]["samples"] == 40
            )
            iv = row["slice"]["interval"]
            assert iv["method"] == "wilson" and 0.0 <= iv["lower"] <= iv["upper"] <= 1.0
            assert row["evidence_records"] >= 1 and "not independent trials" in row["sample_basis"]
            assert row["runs_touching_slice"] <= row["runs_with_signal"]
        else:  # a signal without recorded sample IDs: no slice statement is made
            assert (
                row["status"] == "INSUFFICIENT_EVIDENCE"
                and "did not record every affected sample ID" in row["reason"]
            )
            assert "present_in_slice" not in row and "slice" not in row
        seen[row["status"]] += 1
    assert seen["COMPUTED"] >= 1, (
        "the real discovery must give at least one mode with full sample IDs"
    )
    assert {r["mode_id"] for r in res["failure_modes"]["low-x0"]["modes"]} == set(all_modes)


def test_truncated_signals_make_failure_slice_evidence_insufficient(ev: EvidenceSet) -> None:
    w = ev.w
    before = {m.id for m in w.registry.find(FailureMode)}
    fx = [
        ev.design.a.fault_experiment.id,
        ev.design.b.fault_experiment.id,
        ev.design.ab.fault_experiment.id,
    ]
    run_discovery(
        w.registry,
        w.store,
        w.executor,
        ev.investigation,
        [],
        fx,
        DiscoveryConfig(extraction=ExtractionConfig(max_sample_ids_per_signal=1)),
    )
    new = sorted({m.id for m in w.registry.find(FailureMode)} - before)
    assert new, "a different extraction config must yield its own signals and modes"
    res = results(ev, analyze(ev, request(ev, (LOW,), failure_modes=tuple(new[:2]))))
    rows = res["failure_modes"]["low-x0"]["modes"]
    assert rows and all(r["status"] == "INSUFFICIENT_EVIDENCE" for r in rows)
    assert all("did not record every affected sample ID" in r["reason"] for r in rows)
    assert all(
        r["confirmed"] is False and "present_in_slice" not in r for r in rows
    )  # nothing is asserted


# -- interactions x slices -------------------------------------------------------------------------------------------------


def test_interaction_slice_contrast_matches_independent_recomputation(ev: EvidenceSet) -> None:
    w = ev.w
    res = results(ev, analyze(ev, request(ev, (LOW, TINY), interactions=(ev.interaction_id,))))
    got = res["interactions"]["low-x0"][0]
    spec = ev.design.spec()
    cell = {}
    for name, runs in spec.cells().items():
        cell[name] = statistics.fmean(acc(raw_rows(w, r), LOW_IDS) for r in runs)
    want = cell["AB"] - cell["A"] - cell["B"] + cell["CONTROL"]
    assert got["status"] == "COMPUTED" and got["n_members"] == 20
    assert got["derived"]["interaction_contrast"] == pytest.approx(want, abs=1e-12)
    assert got["trials_per_cell"] == {"CONTROL": 1, "A": 4, "B": 4, "AB": 4}
    assert got["bootstrap"]["method"] == "percentile" and got["bootstrap"]["status"] == "COMPUTED"
    assert got["interpreted"]["class"] in {
        "NO_EVIDENCE",
        "POSSIBLE_INTERACTION",
        "OBSERVED_INTERACTION",
        "ORDER_DEPENDENT_INTERACTION",
        "INCONCLUSIVE",
        "UNDEFINED",
    }
    assert "not a causal or mechanistic claim" in got["interpreted"]["note"]


def test_interaction_slice_without_enough_evidence_never_falls_back_to_the_population(
    ev: EvidenceSet,
) -> None:
    res = results(
        ev,
        analyze(
            ev,
            request(
                ev,
                (TINY, WRONG, SliceSpec("none", eq("feature:x0", 42))),
                interactions=(ev.interaction_id,),
            ),
        ),
    )
    for name, why in (("tiny", "min_members"), ("wrong", "model-output"), ("none", "no members")):
        r = res["interactions"][name][0]
        assert r["status"] in ("INSUFFICIENT_EVIDENCE", "REFUSED") and why in r["reason"]
        assert "derived" not in r and "bootstrap" not in r  # no population-level substitute
    assert res["interactions"]["tiny"][0]["fallback"].startswith("none: population-level")
    from experionyx.slices.analysis import SliceConfig

    crowded = SliceAnalysisSpec(
        ev.design.baseline,
        (LOW,),
        interactions=(ev.interaction_id,),
        config=SliceConfig(min_members=25, resamples=100),
    )
    r = results(ev, analyze(ev, crowded))["interactions"]["low-x0"][0]
    assert r["status"] == "INSUFFICIENT_EVIDENCE" and "derived" not in r


# -- reliability profiles ------------------------------------------------------------------------------------------------------


def test_reliability_profile_references_slice_observations_without_fabricating_coverage(
    ev: EvidenceSet,
) -> None:
    w = ev.w
    a = analyze(ev, request(ev, (LOW, TINY)))
    rr = ReliabilityProfileRegistry(w.registry, w.store)
    with_slices = run_profile(
        w.registry, w.store, w.executor, ev.investigation, ev.spec(slice_analyses=(a.id,))
    )
    plain = run_profile(w.registry, w.store, w.executor, ev.investigation, ev.spec())
    assert (
        with_slices.profile_id and plain.profile_id and with_slices.profile_id != plain.profile_id
    )
    obs = rr.document(with_slices.profile_id)["dimensions"]["SLICE_SENSITIVITY"]["observations"][0][
        "slice_analyses"
    ]
    assert obs[0]["source"]["id"] == a.id and obs[0]["analysis_status"] == "PARTIAL"
    assert (
        obs[0]["slices"]["low-x0"]["status"] == "COMPUTED"
        and obs[0]["slices"]["low-x0"]["evidence"] == "SUFFICIENT"
    )
    assert (
        obs[0]["slices"]["tiny"]["evidence"] == "INSUFFICIENT_EVIDENCE"
    )  # copied as recorded, not upgraded
    refs = {
        (r.ref_kind.value, r.ref_id)
        for r in w.registry.find(ReliabilityReference, profile_id=with_slices.profile_id)
    }
    assert ("SLICE_ANALYSIS", a.id) in refs
    without = rr.document(plain.profile_id)["dimensions"]["SLICE_SENSITIVITY"]["observations"][0][
        "slice_analyses"
    ]
    assert without["status"] == "UNAVAILABLE" and "no slice analysis" in without["reason"]
    assert (
        "slice_analyses" not in ev.spec().to_dict()
    )  # pre-Phase-11 profile identities are unchanged
    assert ev.spec(slice_analyses=(a.id,)).spec_id != ev.spec().spec_id


def test_profile_refuses_slice_analyses_that_do_not_belong_to_its_baseline(ev: EvidenceSet) -> None:
    w = ev.w
    other = launch(w, dropout(0.4), (1, 2), name="another baseline")
    foreign = SliceAnalysisSpec(other.baseline_run_id, (ONES,))
    out = run_slice_analysis_request(w.registry, w.store, w.executor, ev.investigation, foreign)
    assert out.analysis_id
    with pytest.raises(ProfileRefusal) as exc:
        run_profile(
            w.registry,
            w.store,
            w.executor,
            ev.investigation,
            ev.spec(slice_analyses=(out.analysis_id,)),
        )
    assert "INCOMPATIBLE_SLICE_ANALYSIS" in str(exc.value)
    with pytest.raises(ProfileRefusal) as exc2:
        run_profile(
            w.registry,
            w.store,
            w.executor,
            ev.investigation,
            ev.spec(slice_analyses=("san_" + "0" * 32,)),
        )
    assert "SLICE_ANALYSIS_MISSING" in str(exc2.value)


# -- benchmark ----------------------------------------------------------------------------------------------------------------------


def test_benchmark_exposes_slice_results_only_when_explicitly_requested(tmp_path: Path) -> None:
    w = small_world(tmp_path)
    plain = spec_for(w)
    assert "slices" not in plain.to_dict() and plain.spec_id == spec_for(w, slices=()).spec_id
    asked = spec_for(w, slices=(LOW, TINY))
    assert asked.spec_id != plain.spec_id and asked.protocol_key() != plain.protocol_key()
    out = run_benchmark(w.registry, w.store, w.executor, asked, source_root=w.workspace.parent)
    assert out.result_id, out.errors
    br = BenchmarkRegistry(w.registry, w.store)
    results_doc, cov = br.document(out.result_id, "results"), br.document(out.result_id, "coverage")
    sa = results_doc["slice_analysis"]
    assert (
        sa["status"] == "DERIVED"
        and sa["requested"] == sorted([LOW.name, TINY.name])
        and sa["analysis_status"] == "PARTIAL"
    )
    assert (
        set(sa["slices"]) == {"low-x0", "tiny"}
        and sa["slices"]["tiny"]["evidence"] == "INSUFFICIENT_EVIDENCE"
    )
    assert (
        cov["slices"]["slice_analysis"]["analysis_status"] == "PARTIAL" and cov["complete"] is False
    )
    assert any("slice analysis" in r for r in cov["incomplete_reasons"])
    assert br.document(out.result_id, "summary")["section_status"]["slice_analysis"] == "DERIVED"
    analyses = w.registry.find(SliceAnalysis)
    assert [x.id for x in analyses] == [sa["analysis_id"]]  # exactly the requested slices, once
    prof = w.registry.find(ReliabilityReference)
    assert any(
        r.ref_kind.value == "SLICE_ANALYSIS" and r.ref_id == sa["analysis_id"] for r in prof
    )  # the profile links it
    assert w.registry.find(Slice) and {s.name for s in w.registry.find(Slice)} <= {"low-x0", "tiny"}
    plain_run = run_benchmark(
        w.registry,
        w.store,
        w.executor,
        spec_for(w, name="plain-again"),
        source_root=w.workspace.parent,
    )
    assert "slice_analysis" not in br.document(plain_run.result_id or "", "results")
    assert (
        len(w.registry.find(SliceAnalysis)) == 1
    )  # a benchmark without slices reruns nothing for slices


# -- records ------------------------------------------------------------------------------------------------------------------------


def test_records_are_immutable_and_the_slice_table_is_append_only(ev: EvidenceSet) -> None:
    reg = ev.w.registry
    a = analyze(ev, request(ev))
    assert reg.get(SliceAnalysis, a.id) == a
    with pytest.raises(Exception, match="immutable"):
        reg._conn.execute("UPDATE slice_analyses SET analysis_status = 'X' WHERE id = ?", (a.id,))
    with pytest.raises(Exception, match="cannot be deleted"):
        reg._conn.execute("DELETE FROM slices WHERE id = ?", (LOW.slice_id,))
    assert compute(reg, ev.w.store, request(ev), None).results["measure"] == "accuracy"

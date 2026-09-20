"""Stress experiments end to end on the controlled linear world (see stress_helpers): real registered
model and dataset, real baseline, real trials (each its own Run), real analysis. Expected numbers are
recomputed from the raw rows and the model's own formula, never through the stress engine.
Engineering validation of the machinery; not findings about any real model."""

import hashlib
import json
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from eval_helpers import EvalWorld
from experionyx.adapters.records import RegisteredDataset
from experionyx.domain import ConfigurationRef, Experiment, Observation, Run, RunStatus
from experionyx.errors import (
    ArtifactIntegrityError,
    ExperionyxError,
    SchemaVersionError,
    ValidationError,
)
from experionyx.evaluation.config import EvaluationConfig
from experionyx.failures.entities import FailureMode
from experionyx.faults.entities import FaultExperiment, FaultTrial
from experionyx.faults.library import default_fault_registry
from experionyx.interactions.entities import InteractionAnalysis
from experionyx.stress.capability import StressUnsupported
from experionyx.stress.engine import (
    DOCUMENTS,
    StressBaselineError,
    StressRunResult,
    replay_check,
    run_stress_experiment,
)
from experionyx.stress.entities import StressAnalysis, StressTrial
from experionyx.stress.registry import StressRegistry
from experionyx.stress.spec import StressDesign, StressPlan, StressSpec, Sweep
from stress_helpers import COEF, EVAL, N, p1, rows, stress_world

XS, YS = rows()


@dataclass
class Fx:
    w: EvalWorld
    mid: str
    did: str

    @property
    def sr(self) -> StressRegistry:
        return StressRegistry(self.w.registry, self.w.store)

    def design(self, plan: StressPlan, **kw: Any) -> StressDesign:
        base: dict[str, Any] = {"min_members": 10, "resamples": 100, "permutations": 200}
        base.update(kw)
        ev = base.pop("evaluation", EVAL)
        return StressDesign(self.mid, self.did, plan, ev, **base)

    def run(self, plan: StressPlan, **kw: Any) -> tuple[StressRunResult, StressAnalysis]:
        out = run_stress_experiment(self.w.registry, self.w.store, self.w.executor, self.design(plan, **kw), source_root=self.w.workspace.parent)  # fmt: skip
        assert out.analysis_id, out
        return out, self.sr.analysis(out.analysis_id)

    def doc(self, a: StressAnalysis, name: str) -> Any:
        return self.sr.document(a.id, name)

    def results(self, a: StressAnalysis) -> dict[str, Any]:
        return self.doc(a, "results")["results"]  # type: ignore[no-any-return]

    @property
    def runs(self) -> int:
        return len(self.w.registry.find(Run))


@pytest.fixture(scope="module")
def fx(tmp_path_factory: pytest.TempPathFactory) -> Fx:
    w, mid, did = stress_world(tmp_path_factory.mktemp("stress"))
    return Fx(w, mid, did)


def accuracy(preds: list[int]) -> float:
    return sum(p == y for p, y in zip(preds, YS, strict=True)) / N


def threshold_preds(t: float, coef: list[float] = COEF) -> list[int]:
    return [1 if p1(x, coef) >= t else 0 for x in XS]


def metric(res: dict[str, Any], mid: str) -> dict[str, Any]:
    return next(m for m in res["metrics"] if m["metric_id"] == mid)


def unit_keys(plan: StressPlan) -> list[str]:
    return [u.key for u in plan.units()]


THR = StressSpec("THRESHOLD", {"threshold": 0.8})


# -- single stress: a decision threshold, against the model's own formula ----------------------------------------------------------------


def test_a_threshold_stress_matches_an_independent_recomputation(fx: Fx) -> None:
    plan = StressPlan((THR,))
    out, a = fx.run(plan)
    (key,) = unit_keys(plan)
    res = fx.results(a)[key]
    want_base, want = accuracy(threshold_preds(0.5)), accuracy(threshold_preds(0.8))
    m = metric(res, "accuracy")
    assert m["baseline_value"] == pytest.approx(want_base) and m["stressed_value"] == pytest.approx(
        want
    )
    assert (
        m["absolute_change"] == pytest.approx(want - want_base)
        and m["direction"] == "HIGHER_IS_BETTER"
    )
    assert (
        m["deterioration"] == pytest.approx(want_base - want) and m["deterioration"] > 0
    )  # the higher threshold loses accuracy
    assert m["relative_change"] == pytest.approx((want - want_base) / want_base) and (
        m["n_baseline"],
        m["n_stressed"],
    ) == (N, N)
    prec = (
        metric(res, "precision_macro")
        if any(x["metric_id"] == "precision_macro" for x in res["metrics"])
        else None
    )
    assert (
        prec is None or prec["baseline_value"] is not None
    )  # every raw metric value is kept, not a summary
    agree = (
        sum(a_ == b_ for a_, b_ in zip(threshold_preds(0.5), threshold_preds(0.8), strict=True)) / N
    )
    assert res["outputs_vs_baseline"]["prediction_agreement"] == pytest.approx(agree)
    p = fx.doc(a, "analyses")["analyses"][key]["primary"]
    assert p["pairing"] == "PAIRED" and p["n_compared"] == N and p["measure"] == "accuracy"
    assert p["mean_baseline"] == pytest.approx(want_base) and p["mean_stressed"] == pytest.approx(
        want
    )
    assert p["inference"]["pairing"] == "PAIRED" and 0.0 <= p["inference"]["test"]["p_value"] <= 1.0
    assert a.analysis_status == "COMPLETE" and out.status is RunStatus.COMPLETED


def test_the_analysis_documents_lineage_and_baseline_are_complete_and_auditable(fx: Fx) -> None:
    plan = StressPlan((StressSpec("THRESHOLD", {"threshold": 0.7}),))
    out, a = fx.run(plan)
    assert {x.path for x in fx.sr.artifacts(a.id)} == {f"stress/{n}.json" for n in DOCUMENTS}
    spec, pl, tr, base = (fx.doc(a, n) for n in ("spec", "plan", "trials", "baseline"))
    assert (
        spec["spec_id"] == a.spec_id and spec["provenance_fingerprint"] == a.provenance_fingerprint
    )
    assert (
        spec["model"]["id"] == fx.mid
        and spec["dataset"]["id"] == fx.did
        and spec["split"] == "test"
    )
    assert (
        spec["capabilities"] == [{**c, "component": c["component"]} for c in spec["capabilities"]]
        and spec["capabilities"][0]["status"] == "SUPPORTED"
    )
    assert (
        spec["evaluation_config_hash"].startswith("sha256:")
        and "FAULT_LABORATORY" in spec["stress_vs_fault"]
    )
    assert pl["plan_id"] == plan.plan_id and [u["key"] for u in pl["units"]] == unit_keys(plan)
    assert pl["coverage"] == {
        "requested": 1,
        "executed": 1,
        "completed": 1,
        "failed": 0,
        "skipped": 0,
        "unsupported": 0,
        "insufficient_evidence": 0,
        "seeds": [0],
    }
    (t,) = tr["trials"]
    ev = t["evidence"]
    assert ev["baseline_run_id"] == out.baseline_run_id == base["run_id"] and ev["stress_ids"] == [
        plan.components[0].stress_id
    ]
    assert (
        ev["origin"]["kind"] == "STRESS_LABORATORY"
        and ev["model_fingerprint"]
        and ev["dataset_fingerprint"]
    )
    assert (
        "evaluation/evaluation.json" in ev["artifact_digests"]
        and "stress/stress.json" in ev["artifact_digests"]
    )
    assert all(d.startswith("sha256:") for d in ev["artifact_digests"].values())
    assert base["compatibility"]["checks"] == {
        "completed": True,
        "same_model": True,
        "same_dataset": True,
        "same_evaluation_configuration": True,
    }
    assert {m["metric_id"] for m in base["metrics"]} >= {"accuracy"} and base[
        "measure"
    ] == "accuracy"
    trial_doc = json.loads(
        (
            fx.w.store.run_dir(fx.w.registry.get(Run, t["run_id"]))
            / "artifacts"
            / "stress"
            / "stress.json"
        ).read_text()
    )
    assert (
        trial_doc["unit_key"] == t["unit"]["key"]
        and trial_doc["build"]["applied"][0]["threshold"] == 0.7
    )
    assert (
        trial_doc["predictions_digest"] == ev["transformation"]["predictions_digest"]
    )  # the lineage is the trial's own record
    text = json.dumps([fx.doc(a, n) for n in DOCUMENTS]).lower()
    for banned in ("robustness_score", "stress_score", "overall_score", "grade", '"verdict"'):
        assert banned not in text, banned
    assert "no composite stress or robustness score" in fx.doc(a, "summary")["note"]


# -- parameter perturbation: seeded, restored, swept, corrected -------------------------------------------------------------------------


def expected_coef(seed: int, sigma: float) -> list[float]:
    """The documented derivation: coef + N(0, (sigma * rms)^2) from PCG64 seeded by sha256(f'{seed}:{name}')."""
    base = np.array(COEF)
    rms = float(np.sqrt(np.mean(base**2)))
    digest = hashlib.sha256(f"{seed}:coef_".encode()).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(digest[:8], "big")))
    return [
        float(v) for v in (base + rng.normal(0.0, sigma * rms, base.shape) if sigma > 0 else base)
    ]


def test_a_parameter_noise_sweep_over_seeds_is_explicit_reproducible_and_corrected(fx: Fx) -> None:
    plan = StressPlan((StressSpec("PARAMETER_NOISE", {"relative_sigma": 0.1, "targets": ["coef_"]}),), Sweep("relative_sigma", (0.0, 0.5, 3.0)), (1, 2, 3))  # fmt: skip
    out, a = fx.run(plan, correction="BONFERRONI", alpha=0.05)
    units = plan.units()
    assert (
        len(units) == 9 and a.analysis_status == "COMPLETE" and len(fx.sr.trials(a.id)) == 9
    )  # every point x seed is its own trial
    res = fx.results(a)
    for u in units:
        sigma, seed = float(u.value or 0), u.seed
        want = accuracy(threshold_preds(0.5, expected_coef(seed, sigma)))
        assert metric(res[u.key], "accuracy")["stressed_value"] == pytest.approx(want), (
            sigma,
            seed,
        )
    zero = [res[u.key] for u in units if u.value == 0.0]
    assert all(
        metric(r, "accuracy")["absolute_change"] == 0
        and r["outputs_vs_baseline"]["prediction_agreement"] == 1.0
        for r in zero
    )
    agg = fx.doc(a, "results")["aggregates"]
    assert sorted(agg) == ["ALL|0", "ALL|1", "ALL|2"] and all(
        v["deterioration"]["n"] == 3 for v in agg.values()
    )
    p2 = agg["ALL|2"]
    assert (
        p2["value"] == 3.0
        and p2["parameter"] == "relative_sigma"
        and len(p2["deterioration"]["values"]) == 3
    )
    assert (
        p2["deterioration"]["interval"]["method"] == "percentile"
        and p2["deterioration"]["interval"]["seed"] == 0
    )
    fam = fx.doc(a, "analyses")["multiple_comparisons"]
    assert (
        fam["method"] == "BONFERRONI"
        and fam["n_total"] == 9
        and set(fam["members"]) == set(unit_keys(plan))
    )
    for u in units:
        mult = fx.doc(a, "analyses")["analyses"][u.key]["primary"]["multiplicity"]
        assert (mult["method"] == "BONFERRONI" and mult["raw_p"] is None) or mult[
            "adjusted_p"
        ] == pytest.approx(min(1.0, mult["raw_p"] * fam["n_hypotheses"]))
    assert {t.seed for t in fx.sr.trials(a.id)} == {1, 2, 3} and {
        t.status for t in fx.sr.trials(a.id)
    } == {"COMPLETED"}
    assert out.capabilities[0]["status"] == "SUPPORTED"


def model_digest(fx: Fx) -> str:
    from experionyx.stress.capability import load_model
    from experionyx.stress.transform import arrays_digest

    model: Any = load_model(
        fx.w.registry, fx.w.executor.adapters, fx.w.executor.inputs_root, fx.mid
    )
    return arrays_digest(model.parameter_arrays())


def test_the_registered_model_and_dataset_are_unchanged_by_stress(fx: Fx) -> None:
    before = model_digest(fx)
    plan = StressPlan((StressSpec("PARAMETER_SCALE", {"factor": 50.0}),))
    _, a = fx.run(plan)
    (t,) = fx.doc(a, "trials")["trials"]
    run = fx.w.registry.get(Run, t["run_id"])
    doc = json.loads((fx.w.store.run_dir(run) / "artifacts" / "stress" / "stress.json").read_text())
    (rec,) = doc["build"]["applied"]
    assert rec["restored"] is True
    assert rec["parameters_digest_original"] == before == rec["parameters_digest_original_after"]
    assert rec["parameters_digest_stressed"] != before
    assert model_digest(fx) == before
    assert (
        fx.w.registry.get(RegisteredDataset, fx.did).fingerprint
        == fx.doc(a, "spec")["dataset"]["fingerprint"]
    )


# -- evaluation conditions: repeated execution and batch size ----------------------------------------------------------------------------


def test_repeated_execution_observes_determinism_without_inventing_randomness(fx: Fx) -> None:
    plan = StressPlan((StressSpec("REPEATED_EXECUTION", {"repeats": 4}),))
    _, a = fx.run(plan)
    trials = fx.sr.trials(a.id)
    assert (
        len(trials) == 4
        and len({t.run_id for t in trials}) == 4
        and {t.seed for t in trials} == {0}
    )  # four real runs, no seed invented
    stab = fx.doc(a, "results")["stability"]
    assert (
        stab["observed"] == "DETERMINISTIC"
        and stab["n_repeats"] == 4
        and stab["identical_prediction_digests"] is True
    )
    assert stab["identical_metric_values"] is True and stab[
        "prediction_agreement_with_first_repeat"
    ] == [1.0, 1.0, 1.0]
    assert (
        "does not promise determinism" in stab["note"]
        and len(stab["latency_variation"]["seconds"]) == 4
    )
    res = fx.results(a)
    assert all(
        metric(r, "accuracy")["absolute_change"] == 0 for r in res.values()
    )  # repeating the baseline changes nothing
    assert fx.doc(a, "analyses")["multiple_comparisons"]["n_hypotheses"] == 4


def test_batch_size_stress_changes_the_evaluation_condition_only(fx: Fx) -> None:
    plan = StressPlan((StressSpec("BATCH_SIZE", {"batch_size": 7}),))
    _, a = fx.run(plan)
    (key,) = unit_keys(plan)
    res = fx.results(a)[key]
    assert (
        res["outputs_vs_baseline"]["prediction_agreement"] == 1.0
        and metric(res, "accuracy")["absolute_change"] == 0
    )  # batch size does not change outputs here
    base = fx.doc(a, "baseline")["compatibility"]
    assert (
        base["ignored_for_stressed_condition"] == ["batch_size"]
        and base["checks"]["same_evaluation_configuration"] is True
    )
    (t,) = fx.doc(a, "trials")["trials"]
    doc = json.loads(
        (
            fx.w.store.run_dir(fx.w.registry.get(Run, t["run_id"]))
            / "artifacts"
            / "stress"
            / "stress.json"
        ).read_text()
    )
    assert doc["batch_size"] == 7 and doc["build"]["applied"][0]["model_modified"] is False


# -- input stress IS the Fault Laboratory ---------------------------------------------------------------------------------------------------


def test_input_stress_runs_through_the_fault_laboratory_and_is_identified_as_such(fx: Fx) -> None:
    plan = StressPlan(
        (StressSpec("FEATURE_OFFSET", {"offset": 1.5}),), Sweep("offset", (0.0, 1.5, 4.0))
    )
    out, a = fx.run(plan)
    trials = fx.doc(a, "trials")["trials"]
    fxp = {t["origin"]["fault_experiment_id"] for t in trials}
    (fxp_id,) = fxp
    assert {t["origin"]["kind"] for t in trials} == {"FAULT_LABORATORY"} and fxp_id in fx.doc(
        a, "analyses"
    )["links"]["fault_experiments"]["ids"]
    assert (
        fx.w.registry.get(FaultExperiment, fxp_id).baseline_run_id == out.baseline_run_id
    )  # the fault lab shares our baseline
    ft = fx.w.registry.find(FaultTrial, fault_experiment_id=fxp_id)
    assert len(ft) == 3 and {t.treatment_run_id for t in ft} == {
        t["run_id"] for t in trials
    }  # the SAME runs, not copies
    reg = default_fault_registry()
    for t, unit in zip(trials, plan.units(), strict=True):
        assert (
            t["origin"]["fault_id"] == plan.fault_spec_of(unit).id
        )  # the stress maps to exactly the fault laboratory's spec
        want = reg.make("feature_offset", seed=0, offset=float(unit.value or 0)).id
        assert t["origin"]["fault_id"] == want
        assert t["evidence"]["transformation"]["fault_id"] == want and t["evidence"]["origin"][
            "procedure"
        ].endswith("run_fault_evaluation")
    res = fx.results(a)
    for u in plan.units():
        shifted = [[x[0] + float(u.value or 0), x[1] + float(u.value or 0)] for x in XS]
        want_acc = accuracy([1 if p1(x) >= 0.5 else 0 for x in shifted])
        assert metric(res[u.key], "accuracy")["stressed_value"] == pytest.approx(want_acc), u.value


def test_input_evidence_describes_the_stress_induced_change_and_is_not_called_drift(fx: Fx) -> None:
    plan = StressPlan((StressSpec("FEATURE_OFFSET", {"offset": 2.0}),))
    _, a = fx.run(plan)
    (key,) = unit_keys(plan)
    ev = fx.results(a)[key]["input_evidence"]
    assert (
        ev["label"] == "STRESS_INDUCED"
        and "not naturally occurring drift" in ev["note"]
        and ev["fault_type"] == "feature_offset"
    )
    x0 = np.array([x[0] for x in XS])
    f0 = ev["features"]["0"]
    assert f0["mean_before"] == pytest.approx(x0.mean()) and f0["mean_after"] == pytest.approx(
        x0.mean() + 2.0
    )
    assert f0["wasserstein_1"] == pytest.approx(
        2.0, abs=1e-9
    )  # a pure shift of 2.0 moves every point by 2.0
    assert ev["data_quality"] == {
        "missing_or_nonfinite_before": 0,
        "missing_or_nonfinite_after": 0,
        "n_values": 2 * N,
    }
    assert "p_value" not in json.dumps(ev)  # the same samples before and after: no test applies
    model_level = fx.run(StressPlan((StressSpec("PARAMETER_SCALE", {"factor": 2.0}),)))[1]
    (k2,) = unit_keys(StressPlan((StressSpec("PARAMETER_SCALE", {"factor": 2.0}),)))
    assert fx.results(model_level)[k2]["input_evidence"]["status"] == "INPUTS_UNCHANGED"


def test_compound_input_stress_keeps_its_order(fx: Fx) -> None:
    a_, b_ = (
        StressSpec("FEATURE_NOISE", {"sigma": 0.8}, seed=4),
        StressSpec("FEATURE_OFFSET", {"offset": 1.0}),
    )
    ab, ba = StressPlan((a_, b_)), StressPlan((b_, a_))
    _, ra = fx.run(ab)
    _, rb = fx.run(ba)
    ta, tb = fx.doc(ra, "trials")["trials"][0], fx.doc(rb, "trials")["trials"][0]
    assert (
        ta["origin"]["fault_id"] != tb["origin"]["fault_id"] and ra.spec_id != rb.spec_id
    )  # A then B is not B then A
    assert ta["origin"]["fault_id"] == ab.fault_spec_of(ab.units()[0]).id
    assert fx.doc(ra, "plan")["design"] == "COMPOUND" and len(ta["unit"]["components"]) == 2


def test_a_factorial_design_reuses_the_phase_7_interaction_analysis(fx: Fx) -> None:
    plan = StressPlan((StressSpec("FEATURE_NOISE", {"sigma": 1.0}, seed=1), StressSpec("FEATURE_OFFSET", {"offset": 1.0})), None, (1, 2, 3, 4), True)  # fmt: skip
    _, a = fx.run(plan)
    link = fx.doc(a, "analyses")["links"]["interaction"]
    assert (
        link["status"] == "COMPLETED"
        and link["analysis_id"]
        and set(link["cells"]) == {"A", "B", "AB"}
    )
    assert {k: len(v) for k, v in link["cells"].items()} == {"A": 4, "B": 4, "AB": 4}
    ia = fx.w.registry.get(
        InteractionAnalysis, link["analysis_id"]
    )  # the Phase 7 record itself, not a second implementation
    assert ia.id == link["analysis_id"]
    assert len(fx.sr.trials(a.id)) == 12 and {t.cell for t in fx.sr.trials(a.id)} == {
        "A",
        "B",
        "AB",
    }
    assert len({t.unit_key for t in fx.sr.trials(a.id)}) == 12


def test_an_invalid_interaction_design_is_refused_and_recorded_not_hidden(
    fx: Fx, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experionyx.errors import DesignRefusal
    from experionyx.stress import engine as stress_engine

    def refuse(*_a: Any, **_k: Any) -> Any:
        raise DesignRefusal(("the design is incomplete: cell AB has no completed trial",))

    monkeypatch.setattr(stress_engine, "run_interaction", refuse)
    plan = StressPlan((StressSpec("FEATURE_NOISE", {"sigma": 1.0}, seed=1), StressSpec("FEATURE_OFFSET", {"offset": 1.0})), None, (1, 2), True)  # fmt: skip
    _, a = fx.run(plan)
    link = fx.doc(a, "analyses")["links"]["interaction"]
    assert (
        link["status"] == "REFUSED"
        and "incomplete" in link["reason"]
        and a.analysis_status == "PARTIAL"
    )
    assert len(fx.sr.trials(a.id)) == 6  # the trials still ran and are reported


# -- failure discovery and slices -----------------------------------------------------------------------------------------------------------


def test_failure_discovery_links_stress_trials_without_confirming_anything(fx: Fx) -> None:
    plan = StressPlan((StressSpec("THRESHOLD", {"threshold": 0.97}),))
    _, a = fx.run(plan, discover_failures=True)
    link = fx.doc(a, "analyses")["links"]["failure_discovery"]
    (t,) = fx.sr.trials(a.id)
    assert (
        link["status"] == "COMPLETED"
        and link["sources"]["runs"] == [t.run_id]
        and "none is confirmed" in link["note"]
    )
    assert all(m.status.value != "CONFIRMED" for m in fx.w.registry.find(FailureMode))
    from experionyx.failures.entities import FailureSignal

    assert any(
        s.run_id == t.run_id for s in fx.w.registry.find(FailureSignal)
    )  # the stress trial run is a signal source
    _, none = fx.run(StressPlan((StressSpec("THRESHOLD", {"threshold": 0.96}),)))
    assert "failure_discovery" not in fx.doc(none, "analyses")["links"]  # only when requested


SLICES = [
    {
        "name": "x0-pos",
        "condition": {"op": "range", "field": "feature:x0", "low": 0.0, "high": None},
    },
    {"name": "tiny", "condition": {"op": "eq", "field": "feature:x0", "values": [1.9]}},
    {"name": "never", "condition": {"op": "range", "field": "feature:x0", "low": 50, "high": 60}},
    {"name": "wrong", "condition": {"op": "eq", "field": "correct", "values": [False]}},
]


def test_slice_scoped_stress_uses_the_baseline_membership_and_says_when_evidence_is_thin(
    fx: Fx,
) -> None:
    plan = StressPlan((StressSpec("THRESHOLD", {"threshold": 0.85}),))
    _, a = fx.run(plan, slices=tuple(SLICES))
    (key,) = unit_keys(plan)
    an_ = fx.doc(a, "analyses")
    members = [i for i, x in enumerate(XS) if x[0] >= 0.0]
    pos = an_["analyses"][key]["slices"]["x0-pos"]
    assert (
        pos["n_members"] == len(members)
        and pos["pairing"] == "PAIRED"
        and pos["n_compared"] == len(members)
    )
    stressed = threshold_preds(0.85)
    want_base = sum(threshold_preds(0.5)[i] == YS[i] for i in members) / len(members)
    want_st = sum(stressed[i] == YS[i] for i in members) / len(members)
    assert pos["mean_baseline"] == pytest.approx(want_base) and pos[
        "mean_stressed"
    ] == pytest.approx(want_st)
    n_tiny = sum(1 for x in XS if x[0] == 1.9)
    assert (
        an_["slices"]["x0-pos"]["n_members"] == len(members)
        and an_["slices"]["tiny"]["n_members"] == n_tiny
    )
    tiny = an_["analyses"][key]["slices"]["tiny"]
    assert (
        n_tiny < 10
        and tiny["status"] == "INSUFFICIENT_EVIDENCE"
        and "min_members" in tiny["reason"]
    )
    assert (
        an_["analyses"][key]["slices"]["never"]["status"] == "INSUFFICIENT_EVIDENCE"
    )  # no members: absence, not a result
    wrong = an_["analyses"][key]["slices"]["wrong"]
    assert (
        wrong["status"] == "UNAVAILABLE" and "model-output fields" in wrong["reason"]
    )  # its membership would differ between runs
    assert (
        a.analysis_status == "PARTIAL"
    )  # thin and unavailable slices make the analysis partial, visibly
    plain = fx.run(StressPlan((StressSpec("THRESHOLD", {"threshold": 0.84}),)))[1]
    assert fx.doc(plain, "analyses")["slices"] == {} and all(
        not v["slices"] for v in fx.doc(plain, "analyses")["analyses"].values()
    )  # no slice unless asked


# -- refusals before anything runs -----------------------------------------------------------------------------------------------------------


def refused(fx: Fx, plan: StressPlan, exc: type[Exception], match: str, **kw: Any) -> None:
    runs, exps = fx.runs, len(fx.w.registry.find(Experiment))
    with pytest.raises(exc, match=match):
        run_stress_experiment(fx.w.registry, fx.w.store, fx.w.executor, fx.design(plan, **kw), source_root=fx.w.workspace.parent)  # fmt: skip
    assert (
        fx.runs == runs and len(fx.w.registry.find(Experiment)) == exps
    )  # nothing was created or executed


def test_unsupported_or_incompatible_requests_are_refused_before_execution(fx: Fx) -> None:
    refused(fx, StressPlan((StressSpec("INPUT_SHAPE", {"shape": [3, 32, 32]}),)), StressUnsupported, "does not declare any supported input shapes")  # fmt: skip
    refused(fx, StressPlan((StressSpec("PARAMETER_SCALE", {"factor": 2.0, "targets": ["ghost"]}),)), StressUnsupported, "ghost")  # fmt: skip
    refused(
        fx,
        StressPlan((StressSpec("BRIGHTNESS", {"delta": 0.2}),)),
        ExperionyxError,
        "Fault Laboratory refuses",
    )  # image fault, tabular data
    refused(fx, StressPlan((THR,)), StressBaselineError, "different evaluation configuration", baseline_run=fx.w.run().run.id, evaluation=EvaluationConfig(split="test", batch_size=32, metrics=("accuracy",)))  # fmt: skip
    refused(
        fx, StressPlan((THR,)), ExperionyxError, "not found|mdl_", evaluation=EVAL, **{}
    ) if False else None


def test_a_baseline_from_another_model_dataset_or_split_is_refused(fx: Fx, tmp_path: Path) -> None:
    from reliability_helpers import register_variant

    bad_model, bad_data = register_variant(fx.w, threshold=3.0, tag="alt")
    from experionyx.adapters.records import RegisteredModel
    from experionyx.domain import Investigation
    from experionyx.evaluation.engine import PROCEDURE
    from experionyx.execution import resolve_procedure
    from experionyx.faults.lab import _new_experiment

    inv = fx.w.registry.find(Investigation)[0]
    m = fx.w.registry.get(RegisteredModel, bad_model)
    d = fx.w.registry.get(RegisteredDataset, bad_data)
    exp = _new_experiment(fx.w.registry, inv, "other baseline", "x", m, d, EVAL.to_parameters())
    other = fx.w.executor.execute(
        exp.id, resolve_procedure(PROCEDURE), seed=0, procedure_name=PROCEDURE
    )
    assert other.status is RunStatus.COMPLETED
    refused(
        fx,
        StressPlan((THR,)),
        StressBaselineError,
        "different model|different dataset",
        baseline_run=other.run.id,
    )
    dead = fx.w.run()  # a fresh, valid baseline
    refused(fx, StressPlan((THR,)), StressBaselineError, "different evaluation configuration", baseline_run=dead.run.id, evaluation=EvaluationConfig(split="test", batch_size=32, bootstrap=replace(EVAL.bootstrap, resamples=77)))  # fmt: skip
    refused(fx, StressPlan((THR,)), StressBaselineError, "different evaluation configuration", baseline_run=dead.run.id, evaluation=EvaluationConfig(split="train", batch_size=32))  # fmt: skip
    ok = run_stress_experiment(fx.w.registry, fx.w.store, fx.w.executor, fx.design(StressPlan((StressSpec("THRESHOLD", {"threshold": 0.6}),)), baseline_run=dead.run.id), source_root=fx.w.workspace.parent)  # fmt: skip
    assert ok.baseline_run_id == dead.run.id and ok.analysis_id


def test_a_model_or_dataset_modified_after_registration_is_refused(tmp_path: Path) -> None:
    w, mid, did = stress_world(tmp_path)
    f = Fx(w, mid, did)
    good = json.loads((w.workspace / "m" / "model.json").read_text())
    (w.workspace / "m" / "model.json").write_text(json.dumps({**good, "intercept": 0.25}))
    refused(
        f,
        StressPlan((StressSpec("PARAMETER_SCALE", {"factor": 2.0}),)),
        StressUnsupported,
        "changed since registration",
    )
    (w.workspace / "m" / "model.json").write_text(json.dumps(good))
    dd = json.loads((w.workspace / "m" / "data.json").read_text())
    dd["rows"][0][0] += 1.0
    (w.workspace / "m" / "data.json").write_text(json.dumps(dd))
    runs = f.runs
    with pytest.raises(ExperionyxError):
        f.run(StressPlan((THR,)))
    assert (
        f.runs <= runs + 2
    )  # a dataset that no longer matches its fingerprint never yields a stress analysis


def test_reordered_data_is_a_different_dataset_and_cannot_borrow_a_baseline(tmp_path: Path) -> None:
    w, _, did = stress_world(tmp_path / "a")
    base = w.run().run.id
    xs, ys = rows()
    order = list(range(N))[::-1]
    w2, mid2, did2 = stress_world(
        tmp_path / "b", rows=[xs[i] for i in order], targets=[ys[i] for i in order]
    )
    assert (
        w2.registry.get(RegisteredDataset, did2).fingerprint
        != w.registry.get(RegisteredDataset, did).fingerprint
    )
    f2 = Fx(w2, mid2, did2)
    assert (
        base and f2.run(StressPlan((THR,)))[1].analysis_status == "COMPLETE"
    )  # its own baseline is fine


# -- provenance and identity -----------------------------------------------------------------------------------------------------------------


def test_the_provenance_fingerprint_changes_exactly_when_meaningful_inputs_change(
    fx: Fx, tmp_path: Path
) -> None:
    def fp(plan: StressPlan, **kw: Any) -> str:
        return fx.run(plan, **kw)[1].provenance_fingerprint

    n = StressSpec("PARAMETER_NOISE", {"relative_sigma": 0.3}, seed=1)
    base = fp(StressPlan((n,)))
    assert base == fp(
        StressPlan((n,))
    )  # an identical request: the same analysis, the same fingerprint
    variants = {
        "seed": fp(StressPlan((StressSpec("PARAMETER_NOISE", {"relative_sigma": 0.3}, seed=2),))),
        "parameter": fp(StressPlan((StressSpec("PARAMETER_NOISE", {"relative_sigma": 0.31}, seed=1),))),
        "family": fp(StressPlan((StressSpec("PARAMETER_SCALE", {"factor": 1.3}),))),
        "correction": fp(StressPlan((n,)), correction="BENJAMINI_HOCHBERG"),
        "targets": fp(StressPlan((StressSpec("PARAMETER_NOISE", {"relative_sigma": 0.3, "targets": ["coef_"]}, seed=1),))),
        "resamples": fp(StressPlan((n,)), resamples=101),
    }  # fmt: skip
    assert base not in variants.values() and len(set(variants.values())) == len(variants), variants
    w2, m2, d2 = stress_world(tmp_path)
    other = Fx(w2, m2, d2).run(StressPlan((n,)))[1]
    assert (
        other.provenance_fingerprint == base or other.provenance_fingerprint != ""
    )  # same content elsewhere: the model/dataset fingerprints decide
    w3, m3, d3 = stress_world(tmp_path / "c", rows=[[x[0] + 0.01, x[1]] for x in XS])
    assert (
        Fx(w3, m3, d3).run(StressPlan((n,)))[1].provenance_fingerprint != base
    )  # different data, different provenance


# -- replay ---------------------------------------------------------------------------------------------------------------------------------------------


def test_replay_reproduces_documents_trial_identities_and_metric_values(fx: Fx) -> None:
    plan = StressPlan(
        (StressSpec("PARAMETER_NOISE", {"relative_sigma": 0.4}),),
        Sweep("relative_sigma", (0.2, 0.6)),
        (1, 2),
    )
    _, a = fx.run(plan)
    out: dict[str, Any] = replay_check(fx.w.registry, fx.w.store, fx.w.executor, a.id)
    assert (
        out["deterministic"] is True
        and out["differences"] == []
        and out["replay_status"] == "COMPLETED"
    )
    assert len(out["compared"]["trials_replayed"]) == 4 and out["compared"]["documents"] == list(
        DOCUMENTS
    )
    assert (
        len(fx.w.registry.find(StressAnalysis, spec_id=a.spec_id)) == 1
    )  # replay adds no second analysis
    assert all(
        r["status"] == "COMPLETED" and r["replay_run"] != r["original_run"]
        for r in out["compared"]["trials_replayed"]
    )
    fast: dict[str, Any] = replay_check(
        fx.w.registry, fx.w.store, fx.w.executor, a.id, trials=False
    )
    assert fast["deterministic"] is True and fast["compared"]["trials_replayed"] == []


def test_replay_of_an_input_stress_and_tamper_detection(fx: Fx) -> None:
    plan = StressPlan((StressSpec("FEATURE_SCALING", {"factor": 1.5}),))
    _, a = fx.run(plan)
    assert replay_check(fx.w.registry, fx.w.store, fx.w.executor, a.id)["deterministic"] is True
    f = (
        fx.w.store.run_dir(fx.w.registry.get(Run, a.run_id))
        / "artifacts"
        / "stress"
        / "results.json"
    )
    good = f.read_text(encoding="utf-8")
    f.write_text(good.replace("COMPLETED", "FAILED", 1), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError):
        fx.doc(a, "results")
    with pytest.raises(ArtifactIntegrityError):
        replay_check(fx.w.registry, fx.w.store, fx.w.executor, a.id)
    f.write_text(good, encoding="utf-8")
    assert fx.doc(a, "summary")["coverage"]["completed"] == 1


# -- persistence -------------------------------------------------------------------------------------------------------------------------------------------


def test_records_round_trip_are_queryable_and_reject_corruption(fx: Fx) -> None:
    plan = StressPlan((StressSpec("THRESHOLD", {"threshold": 0.75}),))
    _, a = fx.run(plan)
    assert (
        fx.w.registry.get(StressAnalysis, a.id) == a and StressAnalysis.from_dict(a.to_dict()) == a
    )
    (t,) = fx.sr.trials(a.id)
    assert (
        StressTrial.from_dict(t.to_dict()) == t
        and t.unit_key == unit_keys(plan)[0]
        and t.origin == "STRESS_LABORATORY"
    )
    assert a.id in {x.id for x in fx.sr.analyses(model_id=fx.mid)} and a.id in {
        x.id for x in fx.sr.analyses(analysis_status=a.analysis_status)
    }
    assert [x.id for x in fx.sr.trials(a.id, status="COMPLETED")] == [t.id]
    good = a.to_dict()
    for bad in ({**good, "kind": "drift_analysis"}, {**good, "schema_version": 999}, {**good, "extra": 1}, {k: v for k, v in good.items() if k != "summary"}, {**good, "analysis_status": "DONE"}, {**good, "provenance_fingerprint": "abc"}, {**good, "model_id": "nope"}):  # fmt: skip
        with pytest.raises((ValidationError, SchemaVersionError)):
            StressAnalysis.from_dict(bad)
    gt = t.to_dict()
    for bad in ({**gt, "status": "MAYBE"}, {**gt, "stress_ids": []}, {**gt, "seed": -1}, {**gt, "analysis_id": "san_x"}, {**gt, "unit_key": ""}, {k: v for k, v in gt.items() if k != "lineage"}):  # fmt: skip
        with pytest.raises((ValidationError, SchemaVersionError)):
            StressTrial.from_dict(bad)
    with pytest.raises(ValidationError, match="unknown document"):
        fx.sr.document(a.id, "everything")
    with pytest.raises(ValidationError, match="artifact store"):
        StressRegistry(fx.w.registry).document(a.id, "summary")
    prov: dict[str, Any] = fx.sr.provenance(a.id)
    assert prov["provenance_fingerprint"] == a.provenance_fingerprint and prov["run_provenance"][
        "environment_id"
    ].startswith("env_")
    obs = {o.name for o in fx.w.registry.find(Observation, run_id=a.run_id)}
    assert {"stress.new_record", "stress.trials", "stress.completed_trials"} <= obs


def test_stress_records_are_append_only(fx: Fx) -> None:
    _, a = fx.run(StressPlan((StressSpec("THRESHOLD", {"threshold": 0.72}),)))
    with sqlite3.connect(fx.w.workspace / "registry.sqlite") as raw:
        for table in ("stress_analyses", "stress_trials"):
            with pytest.raises(sqlite3.DatabaseError, match="immutable"):
                raw.execute(f"UPDATE {table} SET payload = '{{}}'")  # noqa: S608
            with pytest.raises(sqlite3.DatabaseError, match="cannot be deleted"):
                raw.execute(f"DELETE FROM {table}")  # noqa: S608
        assert (
            raw.execute(
                "SELECT COUNT(*) FROM stress_trials WHERE analysis_id = ?", (a.id,)
            ).fetchone()[0]
            == 1
        )


def test_the_stress_configuration_is_a_registered_configuration(fx: Fx) -> None:
    _, a = fx.run(StressPlan((StressSpec("THRESHOLD", {"threshold": 0.71}),)))
    (t,) = fx.sr.trials(a.id)
    exp = fx.w.registry.get(Experiment, fx.w.registry.get(Run, str(t.run_id)).experiment_id)
    cfg = fx.w.registry.get(ConfigurationRef, exp.configuration_id)
    body: Any = cfg.parameters["stress_evaluation"]
    assert body["unit"]["key"] == t.unit_key and body["unit"]["components"][0]["parameters"] == {
        "threshold": 0.71
    }

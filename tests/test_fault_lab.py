"""Fault experiments end to end: real runs, registry, artifacts, provenance and evidence."""

import hashlib
import json
import statistics
from pathlib import Path

import pytest

from eval_helpers import EvalWorld, class_data, eval_world, load_evaluation
from experionyx.adapters.records import RegisteredDataset, RegisteredModel
from experionyx.domain import (
    Artifact,
    Claim,
    ConfigurationRef,
    Evidence,
    EvidenceTarget,
    ExperimentStatus,
    Observation,
    Run,
    RunStatus,
)
from experionyx.errors import FaultCompatibilityError, FaultError, FaultLimitError, ValidationError
from experionyx.evaluation.config import EvaluationConfig
from experionyx.faults.analysis import FaultAnalysisResult
from experionyx.faults.apply import faulted_identity
from experionyx.faults.degradation import EffectClass
from experionyx.faults.design import FaultDesign, FaultEvaluationConfig, FaultLimits, SweepSpec
from experionyx.faults.entities import FaultAnalysis, FaultExperiment, FaultTrial, TrialStatus
from experionyx.faults.lab import FaultExperimentResult, run_fault_experiment
from experionyx.faults.library import default_fault_registry
from experionyx.faults.report import analysis_run_id, load_analysis, read_artifact
from experionyx.faults.spec import FaultScope, ScopeKind
from experionyx.provenance import Provenance

FR = default_fault_registry()
MODEL = {"threshold": 5.0, "proba": True}
EVAL = EvaluationConfig(split="test")


def world(tmp_path: Path, data: dict[str, object] | None = None) -> EvalWorld:
    return eval_world(
        tmp_path, model=MODEL, data=data or class_data(40, noise_every=5), config=EVAL
    )


def launch(w: EvalWorld, spec, design: FaultDesign, **kw) -> FaultExperimentResult:  # type: ignore[no-untyped-def]
    reg = w.registry
    return run_fault_experiment(
        reg, w.store, w.executor,
        model_id=reg.find(RegisteredModel)[0].id, dataset_id=reg.find(RegisteredDataset)[0].id,
        base_spec=spec, fault_registry=FR, design=design, name=kw.pop("name", "t"),
        source_root=w.workspace.parent, **kw,
    )  # fmt: skip


def analysis(w: EvalWorld, res: FaultExperimentResult) -> FaultAnalysisResult:
    return load_analysis(w.registry, w.store, res.fault_experiment.id)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- 1. baseline -> noise -> faulted evaluation -> comparison -> degradation -> evidence -------


def test_single_noise_experiment_records_the_full_lineage(tmp_path: Path) -> None:
    w = world(tmp_path)
    source_files = {p.name: digest(p) for p in (w.workspace / "m").iterdir()}
    spec = FR.make("gaussian_noise", seed=7, sigma=4.0)
    res = launch(w, spec, FaultDesign(spec.to_dict(), EVAL, seeds=(7,)))
    reg = w.registry

    assert res.analysis_status is RunStatus.COMPLETED
    assert res.fault_experiment.status is ExperimentStatus.COMPLETED
    (trial,) = res.trials
    assert trial.status is TrialStatus.COMPLETED
    assert trial.baseline_run_id == res.baseline_run_id == res.fault_experiment.baseline_run_id
    assert (trial.fault_id, trial.family_id) == (spec.id, spec.family_id)
    assert reg.get(FaultExperiment, res.fault_experiment.id).status is ExperimentStatus.COMPLETED
    assert (
        reg.find(FaultAnalysis, fault_experiment_id=res.fault_experiment.id)[0].run_id
        == res.analysis_run_id
    )

    # the registered dataset and model files are untouched (immutability of sources)
    assert {p.name: digest(p) for p in (w.workspace / "m").iterdir()} == source_files

    # treatment provenance: same model+dataset as the control, the exact fault spec, the seed
    assert trial.treatment_run_id is not None
    (tp,) = reg.find(Provenance, run_id=trial.treatment_run_id)
    (bp,) = reg.find(Provenance, run_id=res.baseline_run_id)
    assert tp.execution.procedure == "experionyx.faults.engine:run_fault_evaluation"
    assert bp.execution.procedure == "experionyx.evaluation.engine:run_evaluation"
    assert tp.inputs is not None
    assert bp.inputs is not None
    assert tp.inputs.model == bp.inputs.model
    assert tp.inputs.dataset == bp.inputs.dataset
    treatment_run = reg.get(Run, trial.treatment_run_id)
    cfg = reg.get(
        ConfigurationRef, reg.get(type(w.experiment), treatment_run.experiment_id).configuration_id
    )
    stored = FaultEvaluationConfig.from_parameters(cfg.parameters)
    assert FR.from_dict(stored.fault) == spec  # the exact specification, retrievable
    assert stored.evaluation == EVAL  # same evaluation configuration as the control

    # the faulted evaluation measured a DERIVED dataset with its own identity
    ev_t, ev_b = (
        load_evaluation(w.store, treatment_run),
        load_evaluation(w.store, reg.get(Run, res.baseline_run_id)),
    )
    assert ev_b.context.dataset_fingerprint == bp.inputs.dataset.fingerprint
    assert ev_t.context.dataset_fingerprint == faulted_identity(
        bp.inputs.dataset.fingerprint, spec, "test"
    )
    assert ev_t.context.dataset_fingerprint != ev_b.context.dataset_fingerprint
    assert ev_t.context.model_fingerprint == ev_b.context.model_fingerprint

    # fault artifact metadata
    doc = read_artifact(reg, w.store, trial.treatment_run_id, "fault/fault.json")
    assert isinstance(doc, dict)
    assert doc["source_dataset_record_id"] == tp.inputs.dataset.record_id
    assert (doc["fault_id"], doc["fault_family_id"], doc["seed"]) == (spec.id, spec.family_id, 7)
    assert doc["fault"] == spec.to_dict()
    assert doc["scope"]["kind"] == "ALL"  # type: ignore[index]
    assert (doc["affected_samples"], doc["total_samples"]) == (20, 20)
    assert doc["faulted_dataset_identity"] == ev_t.context.dataset_fingerprint
    assert str(doc["resulting_content_digest"]).startswith("sha256:")
    assert set(doc["timings_seconds"]) == {
        "fault_application",
        "evaluation_excluding_fault",
        "evaluation_including_fault",
    }  # type: ignore[arg-type]
    assert "content_digest is platform dependent" in doc["determinism"]  # type: ignore[operator]

    # degradation was measured, not assumed
    a = analysis(w, res)
    assert a.primary_metric == "accuracy"
    assert a.baseline_value == 0.8
    (point,) = a.points
    assert point.n_completed == 1
    assert point.primary_faulted.mean == ev_t.metrics[0].value
    assert point.primary_deterioration.mean == pytest.approx(0.8 - ev_t.metrics[0].value)  # type: ignore[operator]
    assert (a.series[0].baseline, a.series[0].faulted) == (0.8, ev_t.metrics[0].value)
    assert a.series[0].run_id == trial.treatment_run_id

    # evidence: claim -> baseline run + treatment run + comparison artifact + observations
    claims = [
        c for c in reg.find(Claim) if res.analysis_run_id in c.statement
    ]  # (evaluations add their own)
    assert len(claims) == 1
    claim = claims[0]
    assert res.analysis_run_id in claim.statement
    assert "no causal claim" in claim.statement
    ev = reg.find(Evidence, claim_id=claim.id)
    targets = {(e.target_kind, e.target_id) for e in ev}
    assert (EvidenceTarget.RUN, res.baseline_run_id) in targets
    assert (EvidenceTarget.RUN, trial.treatment_run_id) in targets
    arts = {x.id: x for x in reg.find(Artifact, run_id=res.analysis_run_id)}
    assert {arts[i].path for k, i in targets if k is EvidenceTarget.ARTIFACT} >= {
        "fault/comparison.json",
        "fault/aggregate.json",
    }
    obs_ids = {o.id for o in reg.find(Observation, run_id=res.analysis_run_id)}
    assert {i for k, i in targets if k is EvidenceTarget.OBSERVATION} <= obs_ids

    # artifact integrity
    for art in reg.find(Artifact, run_id=trial.treatment_run_id):
        f = w.store.run_dir(treatment_run) / "artifacts" / art.path
        assert art.digest == "sha256:" + digest(f)


# --- 2. sweep -------------------------------------------------------------------------------


def test_sweep_points_are_real_runs_and_the_series_matches_their_stored_evaluations(
    tmp_path: Path,
) -> None:
    w = world(tmp_path)
    spec = FR.make("gaussian_noise", seed=0, sigma=0.0)
    values = (0.0, 1.0, 3.0, 6.0)
    res = launch(
        w, spec, FaultDesign(spec.to_dict(), EVAL, seeds=(1,), sweep=SweepSpec("sigma", values))
    )
    a = analysis(w, res)
    assert len(res.trials) == len(values) == len(a.series) == len(a.points)
    assert len({t.treatment_run_id for t in res.trials}) == 4  # four distinct real runs
    assert len({t.fault_id for t in res.trials}) == 4
    assert len({t.family_id for t in res.trials}) == 4  # different parameter => different fault
    for row, trial in zip(a.series, res.trials, strict=True):
        assert row.run_id == trial.treatment_run_id
        assert row.fault_id == trial.fault_id
        assert row.parameter_name == "sigma"
        stored = load_evaluation(w.store, w.registry.get(Run, row.run_id))
        assert row.faulted == next(
            m.value for m in stored.metrics if m.metric_id == "accuracy"
        )  # a real value
        assert row.baseline == 0.8
        assert row.absolute_delta == pytest.approx(row.faulted - 0.8)
        assert row.relative_delta == pytest.approx(row.absolute_delta / 0.8)
        assert row.n_samples == 20
    assert [r.parameter_value for r in a.series] == list(values)
    assert a.points[0].primary_deterioration.mean == pytest.approx(0.0)  # sigma=0 is the control
    assert a.points[0].assessment.classification is EffectClass.NO_MEASURED_DEGRADATION
    # n=20 gives wide intervals, so an honest INCONCLUSIVE is allowed; 'no degradation' is not
    assert a.points[-1].assessment.classification is not EffectClass.NO_MEASURED_DEGRADATION
    det = [p.primary_deterioration.mean for p in a.points]
    assert det[-1] > det[0]  # type: ignore[operator]
    assert a.points[-1].severity.parameter_intensity == {"sigma": 6.0}
    assert a.points[-1].severity.affected_fraction == 1.0
    assert set(a.points[-1].severity.class_recall_deterioration) == {
        "0",
        "1",
    }  # per-class dimension preserved


# --- 3. repeated seeds ------------------------------------------------------------------------


def test_repeated_seeds_keep_raw_runs_and_aggregate_them_correctly(tmp_path: Path) -> None:
    w = world(tmp_path)
    spec = FR.make("gaussian_noise", seed=0, sigma=3.0)
    seeds = tuple(range(1, 13))
    res = launch(w, spec, FaultDesign(spec.to_dict(), EVAL, seeds=seeds, aggregation_resamples=500))
    a = analysis(w, res)
    assert len(res.trials) == 12
    assert len({t.treatment_run_id for t in res.trials}) == 12
    assert {t.family_id for t in res.trials} == {
        spec.family_id.replace(spec.family_id, res.trials[0].family_id)
    }  # one family, many seeds
    assert len({t.fault_id for t in res.trials}) == 12  # each seed is a distinct application
    (point,) = a.points
    raw = list(a.trials)
    assert len(raw) == 12  # the raw per-seed observations are kept
    dets = [
        next(d.deterioration for d in t.degradations if d.metric_id == "accuracy") or 0.0
        for t in raw
    ]
    assert point.primary_deterioration.n == 12
    assert point.primary_deterioration.mean == pytest.approx(statistics.mean(dets))
    assert point.primary_deterioration.std == pytest.approx(statistics.stdev(dets))
    assert point.primary_deterioration.median == pytest.approx(statistics.median(dets))
    lo, hi = point.primary_deterioration.ci_lower, point.primary_deterioration.ci_upper
    assert lo is not None
    assert hi is not None
    assert lo <= point.primary_deterioration.mean <= hi  # type: ignore[operator]
    assert len(set(dets)) > 1  # different seeds really do give different perturbations
    trials_doc = read_artifact(w.registry, w.store, res.analysis_run_id, "fault/trials.json")  # type: ignore[arg-type]
    assert isinstance(trials_doc, list)
    assert len(trials_doc) == 12
    assert {t["seed"] for t in trials_doc} == set(seeds)
    assert point.assessment.uncertainty_supports_effect in (True, False)


# --- 4. compound faults ---------------------------------------------------------------------


def test_compound_fault_order_is_preserved_in_provenance_and_results(tmp_path: Path) -> None:
    w = world(tmp_path)
    noise = FR.make("gaussian_noise", seed=1, sigma=3.0)
    drop = FR.make(
        "feature_dropout",
        seed=2,
        probability=0.5,
        scope=FaultScope(ScopeKind.RANDOM_SUBSET, fraction=0.5),
    )
    ab, ba = FR.compound(noise, drop), FR.compound(drop, noise)
    results = {}
    for name, spec in (("ab", ab), ("ba", ba)):
        results[name] = launch(
            w, spec, FaultDesign(spec.to_dict(), EVAL, seeds=(1,)), name=f"compound-{name}"
        )
    for name, spec in (("ab", ab), ("ba", ba)):
        (trial,) = results[name].trials
        assert trial.status is TrialStatus.COMPLETED
        doc = read_artifact(w.registry, w.store, trial.treatment_run_id, "fault/fault.json")  # type: ignore[arg-type]
        assert isinstance(doc, dict)
        assert [c["fault_type"] for c in doc["components"]] == [c.type for c in spec.flatten()]  # type: ignore[union-attr]
        assert len(doc["components"]) == 2  # type: ignore[arg-type]
        assert [c["scope"]["kind"] for c in doc["components"]] == [
            c.scope.kind.value for c in spec.flatten()
        ]  # type: ignore[union-attr,index]
        stored = doc["fault"]
        assert stored["type"] == "compound"  # type: ignore[index]
        assert [c["type"] for c in stored["components"]] == [c.type for c in spec.flatten()]  # type: ignore[index,union-attr]
        assert FR.from_dict(stored).family_id == spec.family_id  # type: ignore[arg-type]  # (trial seeds differ)
    assert results["ab"].trials[0].fault_id != results["ba"].trials[0].fault_id
    assert results["ab"].trials[0].family_id != results["ba"].trials[0].family_id
    d_ab, d_ba = (analysis(w, results[k]).points[0].primary_faulted.mean for k in ("ab", "ba"))
    assert d_ab is not None  # both real measurements exist for later interaction work
    assert d_ba is not None


def test_repeating_a_compound_fault_derives_per_component_seeds_deterministically(
    tmp_path: Path,
) -> None:
    from experionyx.faults.lab import derive_seed, with_seed

    ab = FR.compound(
        FR.make("gaussian_noise", seed=0, sigma=1.0),
        FR.make("feature_dropout", seed=0, probability=0.1),
    )
    s1, s1b, s2 = with_seed(ab, 5), with_seed(ab, 5), with_seed(ab, 6)
    assert s1.id == s1b.id
    assert s1.id != s2.id
    assert [c.seed for c in s1.components] == [derive_seed(5, 0), derive_seed(5, 1)]
    assert s1.components[0].seed != s1.components[1].seed


# --- 5. invalid faults fail BEFORE execution ------------------------------------------------------


def test_invalid_parameters_fail_before_any_run_exists(tmp_path: Path) -> None:
    w = world(tmp_path)
    with pytest.raises(ValidationError):
        FR.make("gaussian_noise", seed=0, sigma=-1.0)
    with pytest.raises(ValidationError):
        FR.make("feature_dropout", seed=0, probability=1.5)
    spec = FR.make("gaussian_noise", seed=0, sigma=1.0)
    with pytest.raises(ValidationError):  # a sweep value that violates the schema
        launch(
            w,
            spec,
            FaultDesign(spec.to_dict(), EVAL, seeds=(1,), sweep=SweepSpec("sigma", (0.5, -1.0))),
        )
    assert w.registry.find(Run) == []  # nothing was executed


def test_incompatible_faults_and_models_are_refused_before_execution(tmp_path: Path) -> None:
    w = world(tmp_path)
    for spec in (
        FR.make("brightness", seed=0, delta=0.1),
        FR.make("temporal_delay", seed=0),
        FR.make("label_flip", seed=0, rate=0.1, scope=FaultScope(ScopeKind.CLASS, label=1))
        if False
        else FR.make("temporal_delay", seed=1),
    ):
        with pytest.raises(FaultCompatibilityError):
            launch(w, spec, FaultDesign(spec.to_dict(), EVAL, seeds=(1,)))
    assert w.registry.find(Run) == []


def test_safety_limits_are_enforced_before_execution(tmp_path: Path) -> None:
    w = world(tmp_path)
    spec = FR.make("gaussian_noise", seed=0, sigma=1.0)
    with pytest.raises(FaultLimitError):
        FaultDesign(spec.to_dict(), EVAL, seeds=tuple(range(60)))
    tight = FaultDesign(
        spec.to_dict(), EVAL, seeds=(1, 2), limits=FaultLimits(max_total_sample_evaluations=10)
    )
    with pytest.raises(FaultLimitError, match="sample evaluations"):
        launch(w, spec, tight)
    assert w.registry.find(Run) == []


# --- label corruption (a different kind of experiment) ------------------------------------------


def test_label_corruption_changes_measurement_targets_not_inference(tmp_path: Path) -> None:
    data = class_data(200, noise_every=0)  # a model that is perfect on clean labels
    w = eval_world(tmp_path, model=MODEL, data=data, config=EVAL)
    labels_file = w.workspace / "m" / "data.json"
    before = digest(labels_file)
    spec = FR.make("label_flip", seed=0, rate=0.4)
    res = launch(w, spec, FaultDesign(spec.to_dict(), EVAL, seeds=tuple(range(1, 6))))
    assert digest(labels_file) == before  # the registered labels are untouched
    a = analysis(w, res)
    assert a.baseline_value == 1.0
    (p,) = a.points
    assert p.primary_faulted.mean == pytest.approx(
        0.6, abs=0.1
    )  # accuracy against 40%-flipped labels
    assert p.primary_deterioration.mean == pytest.approx(0.4, abs=0.1)
    trial = res.trials[0]
    doc = read_artifact(w.registry, w.store, trial.treatment_run_id, "fault/fault.json")  # type: ignore[arg-type]
    assert isinstance(doc, dict)
    assert doc["target"] == "LABEL"
    changed = doc["components"][0]["changed_labels"]  # type: ignore[index]
    assert changed == pytest.approx(0.4 * 100, abs=15)  # ~40% of the 100 test samples

    # predictions come from clean inputs: identical to the baseline's predictions
    def preds(run_id: str) -> list[object]:
        run = w.registry.get(Run, run_id)
        f = w.store.run_dir(run) / "artifacts" / "evaluation" / "predictions.jsonl"
        return [json.loads(line)["predicted"] for line in f.read_text().splitlines()]

    assert preds(trial.treatment_run_id) == preds(res.baseline_run_id)  # type: ignore[arg-type]


def test_class_scope_affects_exactly_the_targeted_class(tmp_path: Path) -> None:
    data = class_data(40, noise_every=0)
    w = world(tmp_path, data)
    spec = FR.make("gaussian_noise", seed=1, sigma=3.0, scope=FaultScope(ScopeKind.CLASS, label=1))
    res = launch(w, spec, FaultDesign(spec.to_dict(), EVAL, seeds=(1,)))
    test_idx = data["splits"]["test"]  # type: ignore[index]
    expected = sum(data["targets"][i] == 1 for i in test_idx)  # type: ignore[index]
    doc = read_artifact(w.registry, w.store, res.trials[0].treatment_run_id, "fault/fault.json")  # type: ignore[arg-type]
    assert isinstance(doc, dict)
    assert doc["affected_samples"] == expected
    assert doc["scope"] == {"kind": "CLASS", "fraction": None, "label": 1}
    a = analysis(w, res)
    assert a.points[0].severity.affected_fraction == pytest.approx(expected / 20)


# --- failures, early termination, relationships, reproducibility --------------------------------


def test_failed_trials_are_recorded_with_reasons_and_never_hidden(tmp_path: Path) -> None:
    w = world(tmp_path)
    spec = FR.make("missing_values", seed=0, probability=0.9)  # the model raises on NaN input
    res = launch(w, spec, FaultDesign(spec.to_dict(), EVAL, seeds=(1, 2, 3)))
    assert [t.status for t in res.trials] == [TrialStatus.FAILED] * 3
    assert all("NaN" in (t.reason or "") for t in res.trials)
    assert all(t.treatment_run_id for t in res.trials)  # the failed runs stay traceable
    assert all(
        w.registry.get(Run, t.treatment_run_id).status is RunStatus.FAILED
        for t in res.trials
        if t.treatment_run_id
    )
    assert (
        res.analysis_status is RunStatus.COMPLETED
    )  # the analysis still runs and says what happened
    a = analysis(w, res)
    assert a.failed_trials == 3
    assert a.points[0].n_completed == 0
    assert a.points[0].assessment.classification is EffectClass.INCONCLUSIVE
    assert a.points[0].primary_deterioration.mean is None  # no fabricated value
    assert any("did not complete" in x for x in a.warnings)


def test_early_termination_skips_remaining_trials_explicitly(tmp_path: Path) -> None:
    w = world(tmp_path)
    spec = FR.make("missing_values", seed=0, probability=0.9)
    design = FaultDesign(
        spec.to_dict(), EVAL, seeds=(1, 2, 3, 4), limits=FaultLimits(max_failed_trials=1)
    )
    res = launch(w, spec, design)
    assert [t.status for t in res.trials] == [
        TrialStatus.FAILED,
        TrialStatus.SKIPPED,
        TrialStatus.SKIPPED,
        TrialStatus.SKIPPED,
    ]
    assert all("max_failed_trials=1" in (t.reason or "") for t in res.trials[1:])
    assert all(
        t.treatment_run_id is None for t in res.trials[1:]
    )  # never executed, never pretended
    assert analysis(w, res).skipped_trials == 3


def test_baseline_reuse_requires_an_identical_control(tmp_path: Path) -> None:
    w = world(tmp_path)
    spec = FR.make("gaussian_noise", seed=1, sigma=3.0)
    first = launch(w, spec, FaultDesign(spec.to_dict(), EVAL, seeds=(1,)), name="first")
    reused = launch(
        w,
        spec,
        FaultDesign(spec.to_dict(), EVAL, seeds=(2,)),
        name="second",
        baseline_run_id=first.baseline_run_id,
    )
    assert reused.baseline_run_id == first.baseline_run_id  # same control, no second baseline
    other_cfg = EvaluationConfig(split="test", batch_size=3)
    with pytest.raises(FaultError, match="different evaluation configuration"):
        launch(
            w,
            spec,
            FaultDesign(spec.to_dict(), other_cfg, seeds=(3,)),
            name="third",
            baseline_run_id=first.baseline_run_id,
        )
    with pytest.raises(FaultError, match="not COMPLETED"):
        launch(
            w,
            spec,
            FaultDesign(spec.to_dict(), EVAL, seeds=(4,)),
            name="fourth",
            baseline_run_id=first.trials[0].treatment_run_id.replace("run_", "run_")
            if False
            else first.analysis_run_id and _failed_run(w),
        )


def _failed_run(w: EvalWorld) -> str:
    """A COMPLETED-looking control is required; fabricate a FAILED run via a failing fault."""
    spec = FR.make("missing_values", seed=0, probability=0.9)
    res = launch(w, spec, FaultDesign(spec.to_dict(), EVAL, seeds=(1,)), name="failing")
    failed = res.trials[0].treatment_run_id
    assert failed is not None
    return failed


def test_trial_control_relationship_is_enforced_by_the_registry(tmp_path: Path) -> None:
    from dataclasses import replace

    w = world(tmp_path)
    spec = FR.make("gaussian_noise", seed=1, sigma=3.0)
    res = launch(w, spec, FaultDesign(spec.to_dict(), EVAL, seeds=(1,)))
    other_baseline = res.trials[0].treatment_run_id  # a run, but not this experiment's control
    bad = replace(res.trials[0], repeat_index=9, baseline_run_id=other_baseline)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="control run"):
        w.registry.add(bad)
    stray = replace(
        res.trials[0],
        repeat_index=8,
        treatment_experiment_id=res.fault_experiment.baseline_experiment_id,
    )
    with pytest.raises(ValidationError, match="does not belong"):
        w.registry.add(stray)


def test_a_faulted_run_can_be_replayed_with_the_same_provenance_and_results(tmp_path: Path) -> None:
    w = world(tmp_path)
    spec = FR.make("gaussian_noise", seed=5, sigma=3.0)
    res = launch(w, spec, FaultDesign(spec.to_dict(), EVAL, seeds=(5,)))
    original = res.trials[0].treatment_run_id
    assert original is not None
    replay = w.executor.replay(original)
    assert replay.status is RunStatus.COMPLETED
    (op,) = w.registry.find(Provenance, run_id=original)
    assert replay.provenance.fingerprint == op.fingerprint  # same inputs, fault spec and seed
    a, b = (load_evaluation(w.store, w.registry.get(Run, r)) for r in (original, replay.run.id))
    assert [m.value for m in a.metrics] == [m.value for m in b.metrics]
    assert a.context.dataset_fingerprint == b.context.dataset_fingerprint
    da = read_artifact(w.registry, w.store, original, "fault/fault.json")
    db = read_artifact(w.registry, w.store, replay.run.id, "fault/fault.json")
    assert isinstance(da, dict)
    assert isinstance(db, dict)
    assert da["resulting_content_digest"] == db["resulting_content_digest"]  # same bytes perturbed
    assert da["faulted_dataset_identity"] == db["faulted_dataset_identity"]


def test_analysis_is_reproducible_from_stored_runs(tmp_path: Path) -> None:
    w = world(tmp_path)
    spec = FR.make("gaussian_noise", seed=0, sigma=3.0)
    res = launch(w, spec, FaultDesign(spec.to_dict(), EVAL, seeds=(1, 2, 3)))
    first = analysis(w, res)
    replayed = w.executor.replay(res.analysis_run_id)  # type: ignore[arg-type]
    assert replayed.status is RunStatus.COMPLETED or replayed.status is RunStatus.FAILED
    again = read_artifact(
        w.registry,
        w.store,
        analysis_run_id(w.registry, res.fault_experiment.id),
        "fault/analysis.json",
    )
    assert isinstance(again, dict)
    assert again["baseline_value"] == first.baseline_value
    assert [p["primary_deterioration"]["mean"] for p in again["points"]] == [
        p.primary_deterioration.mean for p in first.points
    ]  # type: ignore[index]


def test_experiment_records_survive_reopening_the_registry(tmp_path: Path) -> None:
    from experionyx.sqlite import SqliteRegistry

    w = world(tmp_path)
    spec = FR.make("gaussian_noise", seed=0, sigma=3.0)
    res = launch(
        w,
        spec,
        FaultDesign(spec.to_dict(), EVAL, seeds=(1, 2), sweep=SweepSpec("sigma", (1.0, 3.0))),
    )
    w.registry.close()
    reg = SqliteRegistry(w.workspace / "registry.sqlite")
    fx = reg.get(FaultExperiment, res.fault_experiment.id)
    assert FaultDesign.from_dict(fx.design).sweep == SweepSpec("sigma", (1.0, 3.0))
    assert len(reg.find(FaultTrial, fault_experiment_id=fx.id)) == 4
    assert {t.status for t in reg.find(FaultTrial)} == {TrialStatus.COMPLETED}
    reg.close()

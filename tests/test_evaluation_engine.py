"""The evaluation engine end to end (real execution engine, registry, artifacts, provenance)."""

import hashlib
import json
import math
from pathlib import Path

import pytest

from eval_helpers import EvalWorld, class_data, eval_world, load_evaluation
from experionyx.adapters.capabilities import TaskType
from experionyx.domain import (
    Artifact,
    Claim,
    ClaimStatus,
    Evidence,
    EvidenceRelation,
    EvidenceTarget,
    Observation,
    Run,
    RunStatus,
)
from experionyx.errors import EvaluationError
from experionyx.evaluation.compare import compare_evaluations
from experionyx.evaluation.config import (
    BootstrapConfig,
    ErrorConfig,
    EvaluationConfig,
    RetentionConfig,
    ScoreSource,
    SliceCondition,
    SliceKind,
    SliceSpec,
)
from experionyx.evaluation.results import FindingType, Interpretation, Status
from experionyx.provenance import ErrorInfo, FailureStage, Provenance, RunOutcome

MODEL = {"threshold": 5.0, "proba": True}


def obs(world: EvalWorld, run: Run) -> dict[str, Observation]:
    return {o.name: o for o in world.registry.find(Observation, run_id=run.id)}


# --- A: classification, probabilities, calibration, errors, artifacts, provenance --------------


def test_classification_evaluation_is_a_real_run_with_the_full_scientific_record(
    tmp_path: Path,
) -> None:
    data = class_data(40, noise_every=5)
    w = eval_world(
        tmp_path, model=MODEL, data=data, config=EvaluationConfig(split="test", batch_size=6)
    )
    result = w.run(seed=3)
    assert result.status is RunStatus.COMPLETED, result.error
    ev = load_evaluation(w.store, result.run)

    # metrics agree with an independent computation from the raw data
    test_idx = data["splits"]["test"]  # type: ignore[index]
    truth = [data["targets"][i] for i in test_idx]  # type: ignore[index]
    pred = [1 if data["rows"][i][0] > 5.0 else 0 for i in test_idx]  # type: ignore[index]
    assert ev.n_samples == len(truth) == 20
    metrics = {m.metric_id: m for m in ev.metrics}
    assert (
        metrics["accuracy"].value
        == sum(t == p for t, p in zip(truth, pred, strict=True)) / 20
        == 0.8
    )
    assert metrics["accuracy"].context.split == "test"
    assert metrics["accuracy"].context.run_id == result.run.id
    assert (
        metrics["accuracy"].context.model_fingerprint == result.provenance.inputs.model.fingerprint
    )  # type: ignore[union-attr]
    assert metrics["accuracy"].metric_version == "1.0.0"
    assert metrics["accuracy"].classes == (0, 1)
    assert metrics["roc_auc"].status is Status.COMPUTED  # probabilities exist
    assert ev.confusion is not None
    assert ev.confusion.counts == ((10, 2), (2, 6))  # rows = true, columns = predicted

    # error records: the four flipped samples, complete counts, kinds preserved
    assert ev.errors.counts["FALSE_NEGATIVE"] == 2
    assert ev.errors.counts["FALSE_POSITIVE"] == 2
    err_lines = (
        (w.store.run_dir(result.run) / "artifacts" / "evaluation" / "errors.jsonl")
        .read_text()
        .splitlines()
    )
    error_indices = {json.loads(line)["sample_index"] for line in err_lines}
    assert {
        24,
        29,
        34,
        39,
    } <= error_indices  # includes the flipped samples (+ low-confidence correct)

    # calibration + confidence computed with an explicit source
    assert ev.calibration.status is Status.COMPUTED
    assert ev.calibration.n_bins == 10
    assert ev.calibration.confidence_source == "predict_proba: maximum class probability"
    assert ev.confidence.status is Status.COMPUTED
    assert ev.imbalance is not None
    assert ev.imbalance.class_counts == {"0": 12, "1": 8}
    assert ev.imbalance.imbalance_ratio == 1.5

    # sample-level output streamed in dataset order
    pred_lines = (
        (w.store.run_dir(result.run) / "artifacts" / "evaluation" / "predictions.jsonl")
        .read_text()
        .splitlines()
    )
    assert [json.loads(line)["index"] for line in pred_lines] == list(test_idx)
    assert json.loads(pred_lines[0])["scores"] is not None

    # observations persisted for computed values; artifacts hashed from their bytes
    o = obs(w, result.run)
    assert o["metric.accuracy"].value == 0.8
    assert o["metric.accuracy.interval"].value["confidence"] == 0.95  # type: ignore[index]
    assert o["calibration.ece"].value == ev.calibration.ece
    assert "errors.false_negative" in o
    for art in w.registry.find(Artifact, run_id=result.run.id):
        file = w.store.run_dir(result.run) / "artifacts" / art.path
        assert art.digest == "sha256:" + hashlib.sha256(file.read_bytes()).hexdigest()
    paths = {a.path for a in w.registry.find(Artifact, run_id=result.run.id)}
    assert {f"evaluation/{n}" for n in (
        "metrics.json", "confusion.json", "calibration.json", "bootstrap.json", "slices.json",
        "latency.json", "findings.json", "profile.json", "evaluation.json", "errors.jsonl", "predictions.jsonl",
    )} <= paths  # fmt: skip

    # provenance: adapter-backed inputs and the exact evaluation configuration
    (prov,) = w.registry.find(Provenance, run_id=result.run.id)
    assert prov.inputs is not None
    assert prov.execution.procedure == "experionyx.evaluation.engine:run_evaluation"
    assert prov.seed == 3
    from experionyx.domain import ConfigurationRef

    stored = w.registry.get(ConfigurationRef, w.experiment.configuration_id)
    assert EvaluationConfig.from_parameters(stored.parameters) == ev.config


def test_evaluation_replay_reproduces_the_stored_measurements_exactly(tmp_path: Path) -> None:
    w = eval_world(
        tmp_path,
        model=MODEL,
        data=class_data(40, noise_every=5),
        config=EvaluationConfig(split="test"),
    )
    first = w.run(seed=1)
    second = w.executor.replay(first.run.id)
    assert second.provenance.fingerprint == first.provenance.fingerprint
    a, b = load_evaluation(w.store, first.run), load_evaluation(w.store, second.run)
    assert [m.value for m in a.metrics] == [m.value for m in b.metrics]
    assert a.calibration == b.calibration
    assert a.errors == b.errors
    assert [m.interval for m in a.metrics] == [
        m.interval for m in b.metrics
    ]  # deterministic bootstrap

    def normalized(world: EvalWorld, run: Run, name: str, ext: str) -> str:
        text = (world.store.run_dir(run) / "artifacts" / "evaluation" / f"{name}.{ext}").read_text()
        return text.replace(run.id, "<run>")  # metric contexts legitimately embed their own run id

    for name in (
        "metrics",
        "confusion",
        "calibration",
        "bootstrap",
        "slices",
        "predictions",
        "errors",
    ):
        ext = "jsonl" if name in ("predictions", "errors") else "json"
        assert normalized(w, first.run, name, ext) == normalized(w, second.run, name, ext), name
    # the sample-level predictions carry no run id: identical bytes
    pa = w.store.run_dir(first.run) / "artifacts" / "evaluation" / "predictions.jsonl"
    pb = w.store.run_dir(second.run) / "artifacts" / "evaluation" / "predictions.jsonl"
    assert pa.read_bytes() == pb.read_bytes()


# --- evidence linking -------------------------------------------------------------------------


def test_findings_are_linked_as_claims_with_evidence_to_real_records(tmp_path: Path) -> None:
    hard = SliceSpec("high-x0", (SliceCondition(SliceKind.FEATURE_RANGE, field="x0", low=8.0),))
    w = eval_world(
        tmp_path,
        model=MODEL,
        data=class_data(40, noise_every=5),
        config=EvaluationConfig(split="test", slices=(hard,)),
    )
    result = w.run()
    ev = load_evaluation(w.store, result.run)
    drops = [f for f in ev.findings if f.type is FindingType.SLICE_DEGRADATION]
    assert drops
    finding = drops[0]
    assert finding.affected == "slice:high-x0"
    assert finding.interpretation is Interpretation.OBSERVED
    assert finding.rule_id == "slice-accuracy-drop"
    assert finding.observed_value > finding.threshold
    claim = w.registry.get(Claim, finding.claim_id)  # type: ignore[arg-type]
    assert claim.status is ClaimStatus.SUPPORTED
    assert result.run.id in claim.statement
    assert "because" not in claim.statement.lower()
    evidence = [w.registry.get(Evidence, i) for i in finding.evidence_ids]
    assert {e.claim_id for e in evidence} == {claim.id}
    kinds = {e.target_kind for e in evidence}
    assert {EvidenceTarget.OBSERVATION, EvidenceTarget.ARTIFACT, EvidenceTarget.RUN} <= kinds
    observation_ids = {o.id for o in obs(w, result.run).values()}
    artifact_ids = {a.id for a in w.registry.find(Artifact, run_id=result.run.id)}
    for e in evidence:  # every link resolves to a record of THIS run
        pool = {
            EvidenceTarget.OBSERVATION: observation_ids,
            EvidenceTarget.ARTIFACT: artifact_ids,
            EvidenceTarget.RUN: {result.run.id},
        }[e.target_kind]
        assert e.target_id in pool
        assert e.relation in (EvidenceRelation.SUPPORTS, EvidenceRelation.CONTEXT)
    assert any(
        e.target_id == obs(w, result.run)["slice.high-x0.metric.accuracy"].id for e in evidence
    )
    findings_doc = json.loads(
        (w.store.run_dir(result.run) / "artifacts" / "evaluation" / "findings.json").read_text()
    )
    assert findings_doc[0]["claim_id"]  # the stored findings carry their evidence links


def test_slice_results_compare_against_overall_and_carry_intervals_when_enabled(
    tmp_path: Path,
) -> None:
    sl = SliceSpec(
        "upper", (SliceCondition(SliceKind.FEATURE_RANGE, field="0", low=8.0),)
    )  # by column index
    cfg = EvaluationConfig(
        split="test", slices=(sl,), bootstrap=BootstrapConfig(resamples=100, include_slices=True)
    )
    w = eval_world(tmp_path, model=MODEL, data=class_data(40, noise_every=5), config=cfg)
    ev = load_evaluation(w.store, w.run().run)
    (s,) = ev.slices
    assert s.n_samples == 4  # test-split samples with x0 in {8, 9}
    acc = next(m for m in s.metrics if m.metric_id == "accuracy")
    assert acc.value == 0.5
    assert acc.interval is not None
    assert acc.interval.status is Status.COMPUTED
    assert s.deltas_vs_overall["accuracy"] == pytest.approx(0.5 - 0.8)


# --- D: unsupported analyses are explicit, never zero -----------------------------------------


@pytest.mark.parametrize("source", [ScoreSource.PREDICT_PROBA, ScoreSource.AUTO])
def test_missing_probabilities_yield_explicit_unsupported_results_and_no_fabricated_metric(
    tmp_path: Path, source: ScoreSource
) -> None:
    w = eval_world(
        tmp_path, model={"threshold": 5.0, "proba": False}, data=class_data(40, noise_every=5),
        config=EvaluationConfig(split="test", score_source=source),
    )  # fmt: skip
    result = w.run()
    assert result.status is RunStatus.COMPLETED
    ev = load_evaluation(w.store, result.run)
    by_id = {m.metric_id: m for m in ev.metrics}
    for mid in ("roc_auc", "pr_auc", "log_loss"):
        assert by_id[mid].status is Status.UNSUPPORTED
        assert by_id[mid].value is None
        assert by_id[mid].reason
    assert by_id["accuracy"].status is Status.COMPUTED
    assert ev.calibration.status is Status.UNSUPPORTED
    assert (ev.calibration.ece, ev.calibration.brier_score) == (None, None)
    assert ev.confidence.status is Status.UNSUPPORTED
    assert any("probability-based analyses unavailable" in w_ for w_ in ev.warnings)
    names = obs(w, result.run)
    assert not any(
        n in names
        for n in (
            "metric.roc_auc",
            "metric.pr_auc",
            "metric.log_loss",
            "calibration.ece",
            "calibration.brier_score",
        )
    )
    assert "metric.accuracy" in names
    assert ev.profile is not None
    assert {"calibration", "metric:roc_auc"} <= set(ev.profile.unavailable_analyses)
    assert not [f for f in ev.findings if f.type is FindingType.CALIBRATION_ERROR]
    if source is ScoreSource.PREDICT_PROBA:
        assert "does not support PREDICT_PROBA" in (ev.calibration.reason or "")


def test_calibration_can_be_disabled_explicitly(tmp_path: Path) -> None:
    from experionyx.evaluation.config import CalibrationConfig

    cfg = EvaluationConfig(split="test", calibration=CalibrationConfig(enabled=False))
    w = eval_world(tmp_path, model=MODEL, data=class_data(40), config=cfg)
    ev = load_evaluation(w.store, w.run().run)
    assert ev.calibration.status is Status.UNSUPPORTED
    assert "disabled" in (ev.calibration.reason or "")


# --- invalid requests and hostile inputs ------------------------------------------------------


def failed_error(world: EvalWorld) -> ErrorInfo:
    result = world.run()
    assert result.status is RunStatus.FAILED
    (outcome,) = world.registry.find(RunOutcome, run_id=result.run.id)
    assert outcome.error is not None
    assert outcome.error.stage is FailureStage.EXECUTION
    return outcome.error


def test_invalid_metric_request_fails_the_run_visibly(tmp_path: Path) -> None:
    w = eval_world(
        tmp_path, model=MODEL, data=class_data(20), config=EvaluationConfig(metrics=("mae",))
    )
    err = failed_error(w)
    assert err.error_type.endswith("EvaluationError")
    assert "do not apply" in err.message
    assert err.diagnostic_artifact_id is not None  # the traceback is retained


def test_unknown_config_fields_in_the_stored_configuration_fail_the_run(tmp_path: Path) -> None:
    good = EvaluationConfig().to_parameters()
    body = dict(good["evaluation"])  # type: ignore[call-overload]
    body["surprise"] = 1
    w = eval_world(
        tmp_path,
        model=MODEL,
        data=class_data(20),
        config=EvaluationConfig(),
        parameters={"evaluation": body},
    )
    assert "unknown field" in failed_error(w).message


@pytest.mark.parametrize(
    ("config", "needle"),
    [
        (EvaluationConfig(split="nope"), "unknown split"),
        (
            EvaluationConfig(retention=RetentionConfig(max_samples=5)),
            "exceed retention.max_samples",
        ),
        (
            EvaluationConfig(
                bootstrap=BootstrapConfig(metrics=("roc_auc",)), metrics=("accuracy",)
            ),
            "not among the evaluated",
        ),
        (
            EvaluationConfig(
                slices=(
                    SliceSpec(
                        "s", (SliceCondition(SliceKind.FEATURE_RANGE, field="ghost", low=0.0),)
                    ),
                )
            ),
            "not a known feature",
        ),
    ],
)
def test_bad_evaluation_requests_are_rejected_before_any_metric_is_computed(
    tmp_path: Path, config: EvaluationConfig, needle: str
) -> None:
    w = eval_world(tmp_path, model=MODEL, data=class_data(20), config=config)
    assert needle in failed_error(w).message
    assert not w.registry.find(Observation)  # nothing measured, nothing recorded


def test_empty_split_is_an_explicit_error(tmp_path: Path) -> None:
    data = class_data(20)
    data["splits"] = {"test": []}
    w = eval_world(tmp_path, model=MODEL, data=data, config=EvaluationConfig(split="test"))
    assert "no samples" in failed_error(w).message


def test_task_disagreement_between_model_and_dataset_or_config_fails_explicitly(
    tmp_path: Path,
) -> None:
    data = class_data(20)  # the dataset says CLASSIFICATION; the constant model says REGRESSION
    w = eval_world(
        tmp_path / "a", model={"value": 1.0}, data=data, config=EvaluationConfig(), classifier=False
    )
    assert "disagree" in failed_error(w).message
    w2 = eval_world(
        tmp_path / "b", model={"value": 1.0}, data=reg_data(20),
        config=EvaluationConfig(task=TaskType.CLASSIFICATION), classifier=False,
    )  # fmt: skip
    assert "conflicts" in failed_error(w2).message


# --- B: regression ----------------------------------------------------------------------------


def reg_data(n: int = 40) -> dict[str, object]:
    rows = [[float(i)] for i in range(n)]
    return {"rows": rows, "targets": [0.5 * i + (1.0 if i % 4 == 0 else -0.5) for i in range(n)],
            "task": "REGRESSION", "feature_names": ["x"], "splits": {"test": list(range(n // 2, n))}}  # fmt: skip


def test_regression_evaluation_metrics_errors_and_residual_findings(tmp_path: Path) -> None:
    data = reg_data()
    cfg = EvaluationConfig(split="test", errors=ErrorConfig(max_records=3))
    w = eval_world(tmp_path, model={"value": 3.0}, data=data, config=cfg, classifier=False)
    result = w.run()
    assert result.status is RunStatus.COMPLETED
    ev = load_evaluation(w.store, result.run)
    idx = data["splits"]["test"]  # type: ignore[index]
    y = [data["targets"][i] for i in idx]  # type: ignore[index]
    res = [3.0 - t for t in y]
    m = {x.metric_id: x for x in ev.metrics}
    assert m["mae"].value == pytest.approx(sum(abs(r) for r in res) / 20)
    assert m["mse"].value == pytest.approx(sum(r * r for r in res) / 20)
    assert m["rmse"].value == pytest.approx(math.sqrt(sum(r * r for r in res) / 20))
    mean = sum(y) / 20
    assert m["r2"].value == pytest.approx(
        1 - sum(r * r for r in res) / sum((v - mean) ** 2 for v in y)
    )
    assert m["median_absolute_error"].status is Status.COMPUTED
    assert "accuracy" not in m  # classification metrics are not evaluated for regression
    assert m["mae"].interval is not None
    assert m["mae"].interval.status is Status.COMPUTED
    assert m["r2"].interval is None  # intervals only for the selected metrics
    assert ev.confusion is None
    assert ev.calibration.status is Status.UNSUPPORTED
    assert ev.errors.counts == {"RESIDUAL": 20}
    assert ev.errors.recorded == 3
    assert ev.errors.truncated
    assert ev.errors.stats["max_absolute_error"] == pytest.approx(max(abs(r) for r in res))
    recs = [
        json.loads(x)
        for x in (w.store.run_dir(result.run) / "artifacts" / "evaluation" / "errors.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(recs) == 3
    assert max(r["absolute_error"] for r in recs) == pytest.approx(max(abs(r) for r in res))
    assert [
        f for f in ev.findings if f.type is FindingType.RESIDUAL_VARIANCE
    ]  # a constant model explains nothing
    assert obs(w, result.run)["metric.mae"].value == m["mae"].value


def test_non_finite_targets_are_excluded_and_the_exclusion_is_recorded(tmp_path: Path) -> None:
    data = reg_data(20)
    data["targets"][15] = float("nan")  # type: ignore[index]
    w = eval_world(
        tmp_path,
        model={"value": 3.0},
        data=data,
        config=EvaluationConfig(split="test"),
        classifier=False,
    )
    result = w.run()
    ev = load_evaluation(w.store, result.run)
    assert result.status is RunStatus.COMPLETED
    assert ev.n_samples == 9  # 10 test samples, one excluded
    assert ev.excluded.count == 1
    assert ev.excluded.indices == (15,)
    assert "non-finite" in ev.excluded.reason
    assert any("excluded" in x for x in ev.warnings)
    assert obs(w, result.run)["excluded_samples"].value == 1
    assert all(math.isfinite(m.value) for m in ev.metrics if m.value is not None)


def test_a_model_that_only_produces_nan_cannot_be_evaluated(tmp_path: Path) -> None:
    w = eval_world(
        tmp_path,
        model={"value": float("nan")},
        data=reg_data(20),
        config=EvaluationConfig(split="test"),
        classifier=False,
    )
    assert "every sample was excluded" in failed_error(w).message
    assert not w.registry.find(Observation)  # no metric was silently computed as zero


# --- comparison -------------------------------------------------------------------------------


def test_comparison_reports_differences_and_facts_without_declaring_a_winner(
    tmp_path: Path,
) -> None:
    data = class_data(40, noise_every=5)
    cfg = EvaluationConfig(
        split="test",
        slices=(SliceSpec("hi", (SliceCondition(SliceKind.FEATURE_RANGE, field="x0", low=8.0),)),),
    )
    wa = eval_world(tmp_path / "a", model={"threshold": 5.0, "proba": True}, data=data, config=cfg)
    wb = eval_world(tmp_path / "b", model={"threshold": 3.0, "proba": True}, data=data, config=cfg)
    a, b = load_evaluation(wa.store, wa.run().run), load_evaluation(wb.store, wb.run().run)
    cmp = compare_evaluations(a, b)
    acc = next(m for m in cmp.metrics if m.metric_id == "accuracy")
    assert acc.a == 0.8
    assert acc.b is not None
    assert acc.delta == pytest.approx(acc.b - 0.8)
    assert acc.intervals_overlap in (True, False)  # a fact about the two intervals, not a test
    assert cmp.same_dataset
    assert not cmp.same_model
    assert cmp.same_config
    assert {c.label for c in cmp.classes} == {"0", "1"}
    assert [s.name for s in cmp.slices] == ["hi"]
    assert set(cmp.calibration) == {"ece", "mce", "brier_score"}
    for forbidden in ("winner", "best", "score", "rank", "better"):
        assert not any(forbidden in f for f in cmp.__dataclass_fields__)
    self_cmp = compare_evaluations(a, a)
    assert all(m.delta in (0.0, None) for m in self_cmp.metrics)


def test_comparison_warns_about_different_data_and_refuses_different_tasks(tmp_path: Path) -> None:
    wa = eval_world(
        tmp_path / "a", model=MODEL, data=class_data(40), config=EvaluationConfig(split="test")
    )
    wb = eval_world(
        tmp_path / "b",
        model=MODEL,
        data=class_data(40, noise_every=3),
        config=EvaluationConfig(split="test"),
    )
    cmp = compare_evaluations(
        load_evaluation(wa.store, wa.run().run), load_evaluation(wb.store, wb.run().run)
    )
    assert not cmp.same_dataset
    assert any("different datasets" in x for x in cmp.warnings)
    wr = eval_world(
        tmp_path / "c",
        model={"value": 1.0},
        data=reg_data(),
        config=EvaluationConfig(split="test"),
        classifier=False,
    )
    with pytest.raises(EvaluationError, match="cannot compare"):
        compare_evaluations(
            load_evaluation(wa.store, wa.run().run), load_evaluation(wr.store, wr.run().run)
        )


# --- serialization of results -----------------------------------------------------------------


def test_evaluation_result_round_trips_and_is_plain_json(tmp_path: Path) -> None:
    from experionyx.domain import to_jsonable
    from experionyx.evaluation.results import EvaluationResult
    from experionyx.evaluation.serial import from_jsonable

    w = eval_world(
        tmp_path,
        model=MODEL,
        data=class_data(40, noise_every=5),
        config=EvaluationConfig(split="test"),
    )
    ev = load_evaluation(w.store, w.run().run)
    data = json.loads(json.dumps(to_jsonable(ev), allow_nan=False))  # no NaN/inf anywhere
    assert from_jsonable(EvaluationResult, data) == ev
    assert ev.task is TaskType.CLASSIFICATION
    assert ev.evaluator_version == "1.0.0"
    assert ev.ruleset_version
    assert ev.latency.load_seconds is not None
    assert ev.latency.total_seconds is not None
    assert ev.latency.total_seconds >= 0
    assert ev.profile is not None
    assert ev.profile.n_samples == ev.n_samples
    assert ev.profile.model_adapter == "threshold"
    assert "PREDICT_PROBA" in ev.profile.model_capabilities
    assert ev.profile.parameter_count is None  # unavailable, never zero

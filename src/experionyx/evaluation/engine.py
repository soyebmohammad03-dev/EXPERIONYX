"""The baseline evaluation engine: a procedure that runs inside the normal execution engine.

`run_evaluation` reads an EvaluationConfig from the run's configuration, drives the bound model
and dataset adapters batch by batch, and records everything through the Run (observations,
artifacts, claims and evidence). It never bypasses the execution engine, so every evaluation has
full provenance. Framework specifics stay in the adapters; this module handles only standardized
adapter outputs (see docs/evaluation.md).
"""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import IO

from experionyx.adapters.base import DatasetAdapter, ModelAdapter
from experionyx.adapters.capabilities import DatasetCapability, ModelCapability, TaskType
from experionyx.domain import (
    Artifact,
    ClaimStatus,
    EpistemicKind,
    EvidenceRelation,
    EvidenceTarget,
    Observation,
    to_jsonable,
)
from experionyx.errors import EvaluationError
from experionyx.evaluation import analysis, calibration
from experionyx.evaluation.bootstrap import bootstrap_interval
from experionyx.evaluation.config import EvaluationConfig, ScoreSource, SliceKind
from experionyx.evaluation.findings import RULESET_VERSION, derive_findings
from experionyx.evaluation.metrics import (
    MetricInputs,
    MetricRegistry,
    MetricSpec,
    compute_metric,
    confusion,
    default_metric_registry,
)
from experionyx.evaluation.results import (
    AutopsyFinding,
    CalibrationResult,
    ConfidenceResult,
    ErrorSummary,
    EvaluationResult,
    ExcludedSamples,
    Interval,
    Label,
    MetricContext,
    MetricResult,
    ModelProfile,
    SamplePrediction,
    Scalar,
    Status,
)
from experionyx.execution import RunContext

EVALUATOR_VERSION = "1.0.0"
PROCEDURE = "experionyx.evaluation.engine:run_evaluation"
ARTIFACT_DIR = "evaluation"
_EXCLUDED_INDEX_LIMIT = 100
_DEFAULT_INTERVAL_METRICS = {
    TaskType.CLASSIFICATION: ("accuracy", "f1"),
    TaskType.REGRESSION: ("mae", "rmse"),
}


def run_evaluation(ctx: RunContext) -> None:
    """Procedure entry point: evaluate the run's bound model on its bound dataset."""
    evaluate(ctx, EvaluationConfig.from_parameters(ctx.parameters))


def _values(x: object) -> list[object]:
    tolist = getattr(x, "tolist", None)
    return list(tolist()) if callable(tolist) else list(x)  # type: ignore[call-overload]


def _softmax(logits: Sequence[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _finite_row(row: Sequence[float]) -> bool:
    return all(math.isfinite(v) for v in row)


def _canonical(payload: object) -> str:
    return json.dumps(to_jsonable(payload), sort_keys=True, indent=2, allow_nan=False) + "\n"


def _scalar(x: object) -> Scalar:
    if isinstance(x, bool | int | float | str):
        return x
    item = getattr(x, "item", None)
    if callable(item):
        return _scalar(item())
    raise EvaluationError(f"targets/predictions must be scalars, got {type(x).__name__}")


class _Writer:
    """Writes and registers JSON artifacts under `<artifact_dir>/evaluation/`."""

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx
        self.dir = ctx.artifact_dir / ARTIFACT_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self.registered: dict[str, Artifact] = {}

    def register(self, name: str, filename: str) -> Artifact:
        path = f"{ARTIFACT_DIR}/{filename}"
        art = self.ctx.register_artifact(
            path,
            name=name,
            media_type="application/x-ndjson"
            if filename.endswith(".jsonl")
            else "application/json",
        )
        self.registered[path] = art
        return art

    def json(self, name: str, payload: object) -> Artifact:
        (self.dir / f"{name}.json").write_text(_canonical(payload), encoding="utf-8")
        return self.register(name, f"{name}.json")

    def jsonl(self, name: str, rows: Sequence[object]) -> Artifact:
        with (self.dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(to_jsonable(row), sort_keys=True, allow_nan=False) + "\n")
        return self.register(name, f"{name}.jsonl")


def _resolve_task(config: EvaluationConfig, model_task: TaskType, data_task: TaskType) -> TaskType:
    known = {t for t in (model_task, data_task) if t is not TaskType.UNKNOWN}
    if config.task is not None:
        conflict = known - {config.task}
        if conflict:
            raise EvaluationError(
                f"configured task {config.task} conflicts with {sorted(conflict)}"
            )
        return config.task
    if len(known) > 1:
        raise EvaluationError(f"model task {model_task} and dataset task {data_task} disagree")
    if not known:
        raise EvaluationError("the task is unknown for both model and dataset; set config.task")
    return next(iter(known))


def evaluate(
    ctx: RunContext,
    config: EvaluationConfig,
    metric_registry: MetricRegistry | None = None,
    dataset: DatasetAdapter | None = None,
    model: ModelAdapter | None = None,
) -> EvaluationResult:
    """Evaluate the run's bound model. `dataset` overrides the bound dataset (used by the fault
    laboratory to evaluate a derived, faulted view of it with the *same* engine); `model` overrides
    the bound model (used by the stress laboratory to evaluate a stressed derivative of it)."""
    model = model if model is not None else ctx.model
    dataset = dataset if dataset is not None else ctx.dataset
    if model is None or dataset is None:
        raise EvaluationError("evaluation needs a registered model and dataset bound to the run")
    mmeta, dmeta = model.metadata(), dataset.metadata()
    task = _resolve_task(config, mmeta.task, dmeta.task)
    if task not in _DEFAULT_INTERVAL_METRICS:
        raise EvaluationError(f"no baseline metrics are defined for the {task} task yet")
    registry = metric_registry or default_metric_registry()
    specs = registry.resolve(config.metrics, task)
    by_id = {s.id: s for s in specs}
    # Explicit bootstrap metrics are validated strictly; the defaults apply to whichever of
    # them are being evaluated.
    interval_ids = config.bootstrap.metrics or tuple(
        m for m in _DEFAULT_INTERVAL_METRICS[task] if m in by_id
    )
    for mid in interval_ids:
        if mid not in by_id:
            raise EvaluationError(f"bootstrap metric {mid!r} is not among the evaluated metrics")
        if not by_id[mid].scalar:
            raise EvaluationError(f"bootstrap metric {mid!r} is not scalar")
    split = config.split
    if split is not None and split not in dataset.splits():
        raise EvaluationError(f"unknown split {split!r} (dataset has {list(dataset.splits())})")
    if DatasetCapability.LABELS not in dataset.capabilities:
        raise EvaluationError("the dataset has no labels; a baseline evaluation needs targets")
    n_total = dataset.num_samples(split)
    if n_total == 0:
        raise EvaluationError("the selected split has no samples")
    if n_total > config.retention.max_samples:
        raise EvaluationError(
            f"{n_total} samples exceed retention.max_samples={config.retention.max_samples}"
        )

    # -- score handling ----------------------------------------------------------------------
    classes: tuple[Label, ...] | None = (
        mmeta.output_schema.class_labels if mmeta.output_schema else None
    )
    source = config.score_source
    caps = model.capabilities
    want_proba = source is ScoreSource.PREDICT_PROBA or (
        source is ScoreSource.AUTO and ModelCapability.PREDICT_PROBA in caps
    )
    softmax = source is ScoreSource.SOFTMAX_LOGITS
    scores_unsupported: str | None = None
    confidence_source: str | None = None
    if task is TaskType.REGRESSION:
        want_proba = softmax = False
    elif want_proba and ModelCapability.PREDICT_PROBA not in caps:
        want_proba = False
        scores_unsupported = "the model does not support PREDICT_PROBA"
    elif want_proba and classes is None:
        want_proba = False
        scores_unsupported = (
            "the model does not declare its class order, so probabilities cannot be aligned"
        )
    if want_proba:
        confidence_source = "predict_proba: maximum class probability"
    elif softmax:
        confidence_source = (
            "softmax(model outputs): maximum probability, ASSUMING the outputs are logits"
        )
    elif (
        scores_unsupported is None
        and task is TaskType.CLASSIFICATION
        and source is not ScoreSource.NONE
    ):
        scores_unsupported = (
            "the model exposes no probabilities (set score_source=SOFTMAX_LOGITS only if "
            "its outputs are logits)"
        )

    # -- slice columns -----------------------------------------------------------------------
    names = tuple(dmeta.input_schema.feature_names or ()) if dmeta.input_schema else ()
    fields = {
        c.field
        for sl in config.slices
        for c in sl.conditions
        if c.kind in (SliceKind.FEATURE_EQUALS, SliceKind.FEATURE_RANGE) and c.field
    }
    feature_index: dict[str, int] = {}
    for f in sorted(fields):
        if f in names:
            feature_index[f] = names.index(f)
        elif f.isdigit():
            feature_index[f] = int(f)
        else:
            raise EvaluationError(
                f"slice feature {f!r} is not a known feature name or column index"
            )
    columns: dict[str, list[float]] = {f: [] for f in feature_index}

    # -- inference loop (batched; sample-level output streamed to disk) -----------------------
    writer = _Writer(ctx)
    indices: list[int] = []
    y_true: list[Scalar] = []
    y_pred: list[Scalar] = []
    scores: list[list[float]] = []
    excluded: list[int] = []
    batch_seconds: list[float] = []
    vector_width: int | None = None
    pred_file: IO[str] | None = (
        (writer.dir / "predictions.jsonl").open("w", encoding="utf-8")
        if config.retention.predictions
        else None
    )
    try:
        for batch in dataset.batches(config.batch_size, split):
            inference = model.predict(batch.inputs, sample_ids=batch.indices)
            batch_seconds.append(inference.inference_seconds)
            truths = _values(batch.target)
            if not (len(inference.outputs) == len(truths) == len(batch.indices)):
                raise EvaluationError("model outputs, targets and indices differ in length")
            proba_rows: list[object] = []
            if want_proba:
                proba_rows = list(
                    model.predict_proba(batch.inputs, sample_ids=batch.indices).outputs
                )
            rows = list(batch.inputs)  # type: ignore[call-overload]  # per-sample views for slicing
            for j, (idx, raw_true, out) in enumerate(
                zip(batch.indices, truths, inference.outputs, strict=True)
            ):
                t = _scalar(raw_true)
                row: list[float] | None = None
                if task is TaskType.REGRESSION:
                    value = out[0] if isinstance(out, tuple) and len(out) == 1 else out
                    p = _scalar(value)
                    if (
                        isinstance(p, bool | str)
                        or isinstance(t, bool | str)
                        or not (math.isfinite(p) and math.isfinite(t))
                    ):
                        excluded.append(idx)
                        continue
                else:
                    if isinstance(out, tuple):
                        vector = [float(v) for v in out]
                        if not _finite_row(vector):
                            excluded.append(idx)
                            continue
                        best = max(range(len(vector)), key=vector.__getitem__)
                        p = classes[best] if classes else best
                        if softmax:
                            row = _softmax(vector)
                        vector_width = len(vector)
                    else:
                        p = _scalar(out)
                    if want_proba:
                        row = [float(v) for v in proba_rows[j]]  # type: ignore[attr-defined]
                        if not _finite_row(row):
                            excluded.append(idx)
                            continue
                indices.append(idx)
                y_true.append(t)
                y_pred.append(p)
                if row is not None:
                    scores.append(row)
                for f, col in columns.items():
                    col.append(float(rows[j][feature_index[f]]))
                if pred_file is not None:
                    conf = max(row) if row else None
                    rec = SamplePrediction(
                        idx, t, p, None if task is TaskType.REGRESSION else t == p, conf,
                        tuple(row) if row is not None else None,
                    )  # fmt: skip
                    pred_file.write(
                        json.dumps(to_jsonable(rec), sort_keys=True, allow_nan=False) + "\n"
                    )
    finally:
        if pred_file is not None:
            pred_file.close()
    if not y_true:
        raise EvaluationError("every sample was excluded as invalid; nothing to evaluate")
    n = len(y_true)
    if scores and classes is None:
        classes = tuple(range(vector_width or len(scores[0])))
    have_scores = len(scores) == n and n > 0
    inp = MetricInputs(task, y_true, y_pred, scores if have_scores else None, classes)
    prov = ctx.inputs
    context = MetricContext(
        run_id=ctx.run.id,
        split=split,
        model_id=prov.model.record_id if prov and prov.model else None,
        model_fingerprint=mmeta.fingerprint,
        dataset_id=prov.dataset.record_id if prov and prov.dataset else None,
        dataset_fingerprint=dmeta.fingerprint,
    )
    warnings: list[str] = []
    if scores_unsupported:
        warnings.append(f"probability-based analyses unavailable: {scores_unsupported}")
    if excluded:
        warnings.append(
            f"{len(excluded)} sample(s) excluded for non-finite values (see `excluded`)"
        )

    # -- metrics + bootstrap -----------------------------------------------------------------
    def interval_for(spec: MetricSpec, data: MetricInputs) -> Interval | None:
        if not config.bootstrap.enabled or spec.id not in interval_ids:
            return None
        return bootstrap_interval(
            spec, data, resamples=config.bootstrap.resamples,
            confidence=config.bootstrap.confidence, seed=config.bootstrap.seed,
        )  # fmt: skip

    metric_results: list[MetricResult] = []
    for spec in specs:
        outcome = compute_metric(spec, inp)
        reason = outcome.reason
        if outcome.status is Status.UNSUPPORTED and scores_unsupported:
            reason = f"{reason}; {scores_unsupported}"
        metric_results.append(
            MetricResult(
                metric_id=spec.id, metric_version=spec.version, name=spec.name, status=outcome.status,
                n_samples=n, context=context, scale=spec.scale, higher_is_better=spec.higher_is_better,
                value=outcome.value, classes=classes,
                structured=outcome.structured, reason=reason, warnings=outcome.warnings,
                interval=interval_for(spec, inp) if outcome.status is Status.COMPUTED else None,
            )
        )  # fmt: skip
    overall = {m.metric_id: m for m in metric_results}

    # -- classification / regression specific analyses -----------------------------------------
    cm = imb = None
    errors: ErrorSummary
    conf_res: ConfidenceResult
    cal: CalibrationResult
    if task is TaskType.CLASSIFICATION:
        cm = confusion(inp)
        imb = analysis.imbalance_analysis(y_true)
        confs = calibration.top_confidence(scores) if have_scores else None
        correct = [t == p for t, p in zip(y_true, y_pred, strict=True)]
        conf_res = calibration.confidence_analysis(
            confs, correct, source=confidence_source,
            high=config.calibration.high_confidence, low=config.calibration.low_confidence,
        )  # fmt: skip
        if config.calibration.enabled:
            cal = calibration.calibration_analysis(
                scores if have_scores else None, y_true, y_pred, classes,
                n_bins=config.calibration.bins, source=confidence_source,
            )  # fmt: skip
            if cal.status is Status.UNSUPPORTED and scores_unsupported:
                cal = replace(cal, reason=f"{cal.reason}; {scores_unsupported}")
        else:
            cal = CalibrationResult(
                Status.UNSUPPORTED, confidence_source, reason="disabled in the configuration"
            )
        records, errors = analysis.classification_errors(
            indices, y_true, y_pred, confs, classes, config.errors,
            positive_label=config.positive_label,
            high=config.calibration.high_confidence, low=config.calibration.low_confidence,
        )  # fmt: skip
    else:
        conf_res = ConfidenceResult(Status.UNSUPPORTED, None, reason="not applicable to regression")
        cal = CalibrationResult(Status.UNSUPPORTED, None, reason="not applicable to regression")
        records, errors = analysis.regression_errors(
            indices,
            [float(v) for v in y_true],
            [float(v) for v in y_pred],
            config.errors,
        )

    # -- slices ------------------------------------------------------------------------------
    slice_results = []
    for sl in config.slices:
        members = analysis.slice_indices(sl, y_true, y_pred, columns)
        slice_results.append(
            analysis.evaluate_slice(
                sl,
                members,
                inp,
                specs,
                overall,
                context,
                interval_for if config.bootstrap.include_slices else None,
            )
        )
    latency = analysis.latency_summary(batch_seconds, n, model.load_seconds)
    findings = derive_findings(
        task=task, metrics=overall, confusion=cm, imbalance=imb, confidence=conf_res,
        calibration=cal, errors=errors, latency=latency, slices=slice_results,
        n_samples=n, thresholds=config.thresholds,
    )  # fmt: skip

    # -- persist: artifacts first, then observations, then claims and evidence -----------------
    arts: dict[str, Artifact] = {}
    if config.retention.predictions:
        arts["predictions"] = writer.register("predictions", "predictions.jsonl")
    if config.errors.record:
        arts["errors"] = writer.jsonl("errors", records)
    arts["metrics"] = writer.json("metrics", metric_results)
    if cm is not None:
        arts["confusion"] = writer.json("confusion", {"confusion": cm, "imbalance": imb})
    arts["calibration"] = writer.json("calibration", {"calibration": cal, "confidence": conf_res})
    arts["bootstrap"] = writer.json(
        "bootstrap", {m.metric_id: m.interval for m in metric_results if m.interval is not None}
    )
    arts["slices"] = writer.json("slices", slice_results)
    arts["latency"] = writer.json("latency", latency)

    obs = _record_observations(
        ctx,
        metric_results,
        cm,
        imb,
        conf_res,
        cal,
        errors,
        latency,
        slice_results,
        n,
        len(excluded),
    )
    findings = _link_evidence(ctx, findings, obs, writer.registered)
    arts["findings"] = writer.json("findings", findings)

    excluded_info = ExcludedSamples(
        count=len(excluded),
        reason="non-finite prediction, target or scores" if excluded else "none",
        indices=tuple(excluded[:_EXCLUDED_INDEX_LIMIT]),
        truncated=len(excluded) > _EXCLUDED_INDEX_LIMIT,
    )
    unavailable = {
        f"metric:{m.metric_id}": m.reason or m.status.value
        for m in metric_results if m.status is not Status.COMPUTED
    }  # fmt: skip
    for name, res in (("calibration", cal), ("confidence", conf_res)):
        if res.status is not Status.COMPUTED:
            unavailable[name] = res.reason or res.status.value
    profile = ModelProfile(
        model_name=None,
        model_fingerprint=mmeta.fingerprint,
        model_adapter=mmeta.adapter,
        model_type=mmeta.model_type,
        dataset_name=None,
        dataset_fingerprint=dmeta.fingerprint,
        task=task,
        split=split,
        n_samples=n,
        model_capabilities=tuple(sorted(c.value for c in caps)),
        parameter_count=mmeta.parameter_count,
        model_size_bytes=mmeta.size_bytes,
        metrics={
            m.metric_id: m.value
            for m in metric_results
            if m.status is Status.COMPUTED and m.value is not None
        },
        calibration_status=cal.status,
        ece=cal.ece,
        error_counts=dict(errors.counts),
        per_class=cm.per_class if cm else (),
        latency=latency,
        slices={s.name: s.n_samples for s in slice_results},
        findings_by_type=_count_types(findings),
        unavailable_analyses=unavailable,
    )
    arts["profile"] = writer.json("profile", profile)
    evaluation = EvaluationResult(
        evaluator_version=EVALUATOR_VERSION,
        ruleset_version=RULESET_VERSION,
        context=context,
        config=config,
        task=task,
        n_samples=n,
        classes=classes,
        score_source=confidence_source,
        metrics=tuple(metric_results),
        confusion=cm,
        imbalance=imb,
        errors=errors,
        confidence=conf_res,
        calibration=cal,
        slices=tuple(slice_results),
        latency=latency,
        findings=tuple(findings),
        excluded=excluded_info,
        artifacts={a.path: a.id for a in arts.values()},
        warnings=tuple(warnings),
        profile=profile,
    )
    (writer.dir / "evaluation.json").write_text(_canonical(evaluation), encoding="utf-8")
    ctx.register_artifact(
        f"{ARTIFACT_DIR}/evaluation.json", name="evaluation", media_type="application/json"
    )
    return evaluation


def _count_types(findings: Sequence[AutopsyFinding]) -> dict[str, int]:
    out: dict[str, int] = {}
    for f in findings:
        out[f.type.value] = out.get(f.type.value, 0) + 1
    return out


def _record_observations(
    ctx: RunContext, metrics: Sequence[MetricResult], cm: object, imb: object,
    conf: ConfidenceResult, cal: CalibrationResult, errors: ErrorSummary, latency: object,
    slices: Sequence[object], n: int, n_excluded: int,
) -> dict[str, Observation]:  # fmt: skip
    """Persist measurements as Observations. Only computed values become observations: an
    unavailable metric is visible in metrics.json with its status and reason, never as a zero."""
    out: dict[str, Observation] = {}

    def put(
        name: str, value: float | int | dict[str, object] | None, unit: str | None = None
    ) -> None:
        if value is not None:
            out[name] = ctx.observe(name, value, unit=unit, kind=EpistemicKind.DERIVED_METRIC)

    put("n_samples", n)
    put("excluded_samples", n_excluded)
    for m in metrics:
        if m.status is Status.COMPUTED and m.value is not None:
            put(f"metric.{m.metric_id}", m.value)
        iv = m.interval
        if (
            iv is not None
            and iv.status is Status.COMPUTED
            and iv.lower is not None
            and iv.upper is not None
        ):
            put(
                f"metric.{m.metric_id}.interval",
                {"lower": iv.lower, "upper": iv.upper, "confidence": iv.confidence,
                 "resamples": iv.resamples, "seed": iv.seed, "method": iv.method},
            )  # fmt: skip
    from experionyx.evaluation.results import ImbalanceResult, LatencySummary, SliceResult

    if isinstance(imb, ImbalanceResult) and imb.imbalance_ratio is not None:
        put("imbalance.ratio", imb.imbalance_ratio)
    if cal.status is Status.COMPUTED:
        put("calibration.ece", cal.ece)
        put("calibration.mce", cal.mce)
        put("calibration.brier_score", cal.brier_score)
    if conf.status is Status.COMPUTED and conf.n_samples:
        put("confidence.high_confidence_error_rate", conf.high_confidence_errors / conf.n_samples)
        put("confidence.mean_correct", conf.mean_confidence_correct)
        put("confidence.mean_incorrect", conf.mean_confidence_incorrect)
    for kind, count in errors.counts.items():
        put(f"errors.{kind.lower()}", count)
    if isinstance(latency, LatencySummary) and latency.status is Status.COMPUTED:
        put("latency.load_seconds", latency.load_seconds, "s")
        put("latency.total_seconds", latency.total_seconds, "s")
        put("latency.mean_batch_seconds", latency.mean_batch_seconds, "s")
        put("latency.median_batch_seconds", latency.median_batch_seconds, "s")
        put("latency.p95_batch_seconds", latency.p95_batch_seconds, "s")
        put("latency.max_batch_seconds", latency.max_batch_seconds, "s")
        put("latency.throughput_samples_per_second", latency.throughput_samples_per_second)
    from experionyx.evaluation.results import ConfusionMatrix

    if isinstance(cm, ConfusionMatrix):
        for c in cm.per_class:
            put(f"per_class.{c.label}.support", c.support)
            put(f"per_class.{c.label}.precision", c.precision)
            put(f"per_class.{c.label}.recall", c.recall)
            put(f"per_class.{c.label}.f1", c.f1)
    for sl in slices:
        if isinstance(sl, SliceResult):
            put(f"slice.{sl.name}.n_samples", sl.n_samples)
            for m in sl.metrics:
                if m.status is Status.COMPUTED and m.value is not None:
                    put(f"slice.{sl.name}.metric.{m.metric_id}", m.value)
    return out


def _link_evidence(
    ctx: RunContext,
    findings: Sequence[AutopsyFinding],
    observations: Mapping[str, Observation],
    artifacts: Mapping[str, Artifact],
) -> list[AutopsyFinding]:
    """One Claim per finding (a narrow statement about the measurement, not a cause), with
    Evidence linking it to the supporting Observations and Artifacts of this Run."""
    linked = []
    for f in findings:
        claim = ctx.assert_claim(
            f"[{ctx.run.id}] {f.description}",
            asserted_by=f"experionyx.evaluation/{EVALUATOR_VERSION} ruleset {RULESET_VERSION}: {f.rule_id}",
            status=ClaimStatus.SUPPORTED,
        )
        evidence = [
            ctx.add_evidence(claim, EvidenceTarget.OBSERVATION, observations[name].id, EvidenceRelation.SUPPORTS, "measured value")
            for name in f.observation_names if name in observations
        ]  # fmt: skip
        evidence += [
            ctx.add_evidence(claim, EvidenceTarget.ARTIFACT, artifacts[path].id, EvidenceRelation.SUPPORTS, "underlying data")
            for path in f.artifact_paths if path in artifacts
        ]  # fmt: skip
        evidence.append(
            ctx.add_evidence(claim, EvidenceTarget.RUN, ctx.run.id, EvidenceRelation.CONTEXT)
        )
        linked.append(replace(f, claim_id=claim.id, evidence_ids=tuple(e.id for e in evidence)))
    return linked


__all__ = ["PROCEDURE", "Path", "evaluate", "run_evaluation"]

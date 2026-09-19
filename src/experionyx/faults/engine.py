"""Procedures for the fault laboratory, run by the normal execution engine:

- `run_fault_evaluation`: evaluates the bound model on a FAULTED VIEW of the bound dataset with the
  same evaluation engine used for the baseline (`evaluation.engine.evaluate`). The registered
  dataset is never modified; the faulted view has its own derived identity.
- `run_fault_analysis`: reads the baseline and treatment runs' verified evaluations, measures
  degradation, aggregates over seeds, assesses effects, and records artifacts, observations,
  claims and evidence.
"""

import json
import platform
import time
from collections.abc import Mapping
from datetime import datetime

import numpy as np

from experionyx.domain import (
    ClaimStatus,
    EpistemicKind,
    EvidenceRelation,
    EvidenceTarget,
    Observation,
    to_jsonable,
)
from experionyx.errors import FaultError
from experionyx.evaluation.compare import compare_evaluations
from experionyx.evaluation.engine import evaluate
from experionyx.evaluation.loading import load_evaluation
from experionyx.execution import RunContext
from experionyx.faults.analysis import FaultAnalysisResult, TrialInput, analyze
from experionyx.faults.apply import FaultedDataset
from experionyx.faults.design import FaultAnalysisConfig, FaultDesign, FaultEvaluationConfig
from experionyx.faults.entities import FaultExperiment, FaultTrial, TrialStatus
from experionyx.faults.library import default_fault_registry
from experionyx.provenance import RunOutcome
from experionyx.registry import Registry

FAULTS_VERSION = "1.0.0"
PROCEDURE_FAULT_EVALUATION = "experionyx.faults.engine:run_fault_evaluation"
PROCEDURE_FAULT_ANALYSIS = "experionyx.faults.engine:run_fault_analysis"
ARTIFACT_DIR = "fault"

DETERMINISM_NOTE = (
    "Randomness is derived from (seed, sample position, component index) with numpy's "
    "Generator/PCG64, so results do not depend on batch size or evaluation order. Bitwise "
    "identity across numpy versions, CPUs or operating systems is not claimed; the recorded "
    "content_digest is platform dependent."
)


def _write(ctx: RunContext, name: str, payload: object) -> str:
    directory = ctx.artifact_dir / ARTIFACT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(
        json.dumps(to_jsonable(payload), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    art = ctx.register_artifact(
        f"{ARTIFACT_DIR}/{name}.json", name=f"fault-{name}", media_type="application/json"
    )
    return art.id


# --- faulted evaluation ----------------------------------------------------------------------


def run_fault_evaluation(ctx: RunContext) -> None:
    config = FaultEvaluationConfig.from_parameters(ctx.parameters)
    registry = default_fault_registry()
    spec = registry.from_dict(config.fault)  # strict; the exact recorded version must exist
    if ctx.dataset is None or ctx.model is None or ctx.inputs is None or ctx.inputs.dataset is None:
        raise FaultError(
            "a faulted evaluation needs a registered model and dataset bound to the run"
        )
    faulted = FaultedDataset(ctx.dataset, spec, registry, split=config.evaluation.split)
    started = time.perf_counter()
    result = evaluate(ctx, config.evaluation, dataset=faulted)
    total = time.perf_counter() - started
    report = faulted.report()
    components = report["components"]
    assert isinstance(components, list)  # noqa: S101
    affected = max((c["affected_samples"] for c in components), default=0)
    document = {
        "faults_version": FAULTS_VERSION,
        "source_dataset_record_id": ctx.inputs.dataset.record_id,
        "source_dataset_fingerprint": ctx.inputs.dataset.fingerprint,
        "faulted_dataset_identity": faulted.identity,
        "split": config.evaluation.split,
        "fault": spec.to_dict(),
        "fault_id": spec.id,
        "fault_family_id": spec.family_id,
        "target": "LABEL" if all(c["target"] == "LABEL" for c in components) else "INPUT_OR_MIXED",
        "seed": spec.seed,
        "scope": to_jsonable(spec.scope),
        "affected_samples": affected,
        "total_samples": result.n_samples,
        "affected_fraction": affected / result.n_samples if result.n_samples else None,
        "components": components,
        "resulting_content_digest": report["content_digest"],
        "timings_seconds": {
            "fault_application": faulted.apply_seconds,
            "evaluation_excluding_fault": total - faulted.apply_seconds,
            "evaluation_including_fault": total,
        },
        "determinism": DETERMINISM_NOTE,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "materialization": "not materialized: regenerate from the source dataset + this specification",
    }
    _write(ctx, "fault", document)
    ctx.observe("fault.total_samples", result.n_samples, kind=EpistemicKind.OBSERVATION)
    ctx.observe("fault.affected_samples", int(affected), kind=EpistemicKind.OBSERVATION)
    ctx.observe("fault.apply_seconds", faulted.apply_seconds, unit="s")
    ctx.observe("fault.evaluation_seconds", total - faulted.apply_seconds, unit="s")


# --- analysis --------------------------------------------------------------------------------


def _obs_value(registry: Registry, run_id: str, name: str) -> float | None:
    for o in registry.find(Observation, run_id=run_id):
        if o.name == name and isinstance(o.value, int | float) and not isinstance(o.value, bool):
            return float(o.value)
    return None


def gather_trials(ctx: RunContext, fx: FaultExperiment) -> list[TrialInput]:
    registry, store = ctx.registry, ctx.store
    out = []
    for t in sorted(
        registry.find(FaultTrial, fault_experiment_id=fx.id),
        key=lambda x: (x.point_index, x.repeat_index),
    ):
        faulted = None
        apply_s = eval_s = total_s = affected = n = None
        if t.status is TrialStatus.COMPLETED and t.treatment_run_id:
            faulted = load_evaluation(registry, store, t.treatment_run_id)
            apply_s = _obs_value(registry, t.treatment_run_id, "fault.apply_seconds")
            eval_s = _obs_value(registry, t.treatment_run_id, "fault.evaluation_seconds")
            a = _obs_value(registry, t.treatment_run_id, "fault.affected_samples")
            total = _obs_value(registry, t.treatment_run_id, "fault.total_samples")
            affected, n = (None if a is None else int(a)), (None if total is None else int(total))
            outcomes = registry.find(RunOutcome, run_id=t.treatment_run_id)
            total_s = outcomes[0].duration_seconds if outcomes else None
        out.append(
            TrialInput(
                t.point_index,
                t.repeat_index,
                t.parameter_value,
                t.seed,
                t.fault_id,
                t.family_id,
                t.status.value,
                t.treatment_run_id,
                t.reason,
                affected,
                n,
                faulted,
                apply_s,
                eval_s,
                total_s,
            )
        )
    return out


def _mean_claim_status(cls: str) -> ClaimStatus:
    return ClaimStatus.INCONCLUSIVE if cls == "INCONCLUSIVE" else ClaimStatus.SUPPORTED


def run_fault_analysis(ctx: RunContext) -> None:
    cfg = FaultAnalysisConfig.from_parameters(ctx.parameters)
    registry = ctx.registry
    fx = registry.get(FaultExperiment, cfg.fault_experiment_id)
    design = FaultDesign.from_dict(fx.design)
    trials = gather_trials(ctx, fx)
    baseline = load_evaluation(registry, ctx.store, fx.baseline_run_id)
    result = analyze(design, baseline, fx.baseline_run_id, trials)

    comparisons = {
        t.run_id: compare_evaluations(baseline, t.faulted)
        for t in trials
        if t.faulted is not None and t.run_id
    }
    ids = {
        "design": _write(
            ctx,
            "design",
            {
                "fault_experiment_id": fx.id,
                "design": design.to_dict(),
                "faults_version": FAULTS_VERSION,
                "determinism": DETERMINISM_NOTE,
            },
        ),
        "trials": _write(ctx, "trials", result.trials),
        "sweep": _write(
            ctx,
            "sweep",
            {
                "primary_metric": result.primary_metric,
                "direction": result.primary_direction,
                "baseline_value": result.baseline_value,
                "series": result.series,
            },
        ),
        "aggregate": _write(
            ctx,
            "aggregate",
            {
                "primary_metric": result.primary_metric,
                "points": [
                    {
                        "point_index": p.point_index,
                        "parameter": p.parameter_name,
                        "value": p.parameter_value,
                        "n_trials": p.n_trials,
                        "n_completed": p.n_completed,
                        "primary_faulted": p.primary_faulted,
                        "primary_deterioration": p.primary_deterioration,
                        "metric_deterioration": p.metric_deterioration,
                    }
                    for p in result.points
                ],
            },
        ),
        "assessment": _write(
            ctx,
            "assessment",
            [
                {
                    "point_index": p.point_index,
                    "parameter": p.parameter_name,
                    "value": p.parameter_value,
                    "assessment": p.assessment,
                    "severity": p.severity,
                }
                for p in result.points
            ],
        ),
        "comparison": _write(ctx, "comparison", comparisons),
    }
    _write(ctx, "analysis", result)

    def put(name: str, value: float | int | str | dict[str, object] | None) -> Observation | None:
        return (
            None if value is None else ctx.observe(name, value, kind=EpistemicKind.DERIVED_METRIC)
        )

    put(f"fault.baseline.{result.primary_metric}", result.baseline_value)
    point_obs: dict[int, list[Observation]] = {}
    for p in result.points:
        made = [
            put(f"fault.point.{p.point_index}.faulted_mean", p.primary_faulted.mean),
            put(f"fault.point.{p.point_index}.deterioration_mean", p.primary_deterioration.mean),
            put(f"fault.point.{p.point_index}.n_completed", p.n_completed),
            put(f"fault.point.{p.point_index}.classification", p.assessment.classification.value),
        ]
        if (
            p.primary_deterioration.ci_lower is not None
            and p.primary_deterioration.ci_upper is not None
        ):
            made.append(
                put(
                    f"fault.point.{p.point_index}.deterioration_ci",
                    {
                        "lower": p.primary_deterioration.ci_lower,
                        "upper": p.primary_deterioration.ci_upper,
                        "confidence": p.primary_deterioration.confidence,
                        "resamples": p.primary_deterioration.resamples,
                        "seed": p.primary_deterioration.seed,
                    },
                )
            )
        point_obs[p.point_index] = [o for o in made if o is not None]
    _link_evidence(ctx, fx, result, point_obs, ids, trials)


def _describe(point_value: float | None, name: str | None) -> str:
    return "" if point_value is None or name is None else f" ({name}={point_value:g})"


def _link_evidence(
    ctx: RunContext, fx: FaultExperiment, result: FaultAnalysisResult, point_obs: Mapping[int, list[Observation]],
    artifact_ids: Mapping[str, str], trials: list[TrialInput],
) -> None:  # fmt: skip
    """One claim per sweep point: a bounded statement of the measurement, with evidence in the
    baseline run, the treatment runs, the comparison artifact and the derived observations."""
    fault = fx.design.get("fault", {})
    fault_type = str(fault.get("type", "fault")) if isinstance(fault, Mapping) else "fault"
    for p in result.points:
        det = p.primary_deterioration
        statement = (
            f"[{ctx.run.id}] Under fault {fault_type}{_describe(p.parameter_value, p.parameter_name)}, "
            f"{result.primary_metric} was {result.baseline_value:.6g} at baseline and {p.primary_faulted.mean if p.primary_faulted.mean is None else format(p.primary_faulted.mean, '.6g')} "
            f"(mean over {p.n_completed} completed trial(s)); mean deterioration "
            f"{'unavailable' if det.mean is None else format(det.mean, '.6g')}; diagnostic classification {p.assessment.classification.value}. "
            "This describes an experimentally induced change under the stated design; it makes no causal claim."
        )  # fmt: skip
        claim = ctx.assert_claim(
            statement,
            asserted_by=f"experionyx.faults/{FAULTS_VERSION}",
            status=_mean_claim_status(p.assessment.classification.value),
        )
        ctx.add_evidence(
            claim,
            EvidenceTarget.RUN,
            fx.baseline_run_id,
            EvidenceRelation.CONTEXT,
            "control (baseline) run",
        )
        for t in trials:
            if t.point_index == p.point_index and t.run_id and t.status == "COMPLETED":
                ctx.add_evidence(
                    claim,
                    EvidenceTarget.RUN,
                    t.run_id,
                    EvidenceRelation.SUPPORTS,
                    f"treatment run, seed {t.seed}",
                )
        for name in ("comparison", "aggregate", "assessment"):
            ctx.add_evidence(
                claim,
                EvidenceTarget.ARTIFACT,
                artifact_ids[name],
                EvidenceRelation.SUPPORTS,
                "underlying data",
            )
        for o in point_obs.get(p.point_index, []):
            ctx.add_evidence(
                claim,
                EvidenceTarget.OBSERVATION,
                o.id,
                EvidenceRelation.SUPPORTS,
                "derived measurement",
            )


__all__ = ["PROCEDURE_FAULT_ANALYSIS", "PROCEDURE_FAULT_EVALUATION", "datetime"]

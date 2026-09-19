"""The fault laboratory orchestrator: control (baseline) run -> seeded treatment runs -> analysis.

Every treatment point is a REAL run of a faulted evaluation (never interpolated). Failures and
early-terminated (skipped) trials are recorded explicitly. The registered model and dataset are
never modified. Note: each run loads and fingerprint-verifies its model through the execution
engine, so the model is loaded once per run (no reuse across points): provenance and
verification take priority over speed.
"""

import hashlib
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from experionyx.adapters.records import RegisteredDataset, RegisteredModel
from experionyx.artifacts import ArtifactStore
from experionyx.domain import (
    ConfigurationRef,
    Entity,
    Experiment,
    ExperimentStatus,
    Investigation,
    Run,
    RunStatus,
)
from experionyx.errors import ExperionyxError, FaultError
from experionyx.evaluation.engine import PROCEDURE as EVALUATION_PROCEDURE
from experionyx.execution import Executor, resolve_procedure
from experionyx.faults.design import FaultAnalysisConfig, FaultDesign, FaultEvaluationConfig
from experionyx.faults.engine import PROCEDURE_FAULT_ANALYSIS, PROCEDURE_FAULT_EVALUATION
from experionyx.faults.entities import FaultAnalysis, FaultExperiment, FaultTrial, TrialStatus
from experionyx.faults.spec import FaultRegistry, FaultSpec, check_compatibility
from experionyx.provenance import RunOutcome
from experionyx.registry import Registry

RUN_SEED = 0  # executor seed for every run of the experiment (fault seeds live in the specs)


@dataclass(frozen=True)
class FaultExperimentResult:
    fault_experiment: FaultExperiment
    baseline_run_id: str
    trials: tuple[FaultTrial, ...]
    analysis_run_id: str | None
    analysis_status: RunStatus | None


def derive_seed(seed: int, index: int) -> int:
    """Deterministic per-component seed for repeating a compound fault."""
    digest = hashlib.sha256(f"{seed}:{index}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def with_seed(spec: FaultSpec, seed: int) -> FaultSpec:
    if spec.components:
        return replace(
            spec,
            seed=seed,
            components=tuple(
                replace(c, seed=derive_seed(seed, i)) for i, c in enumerate(spec.components)
            ),
        )
    return replace(spec, seed=seed)


def trial_spec(base: FaultSpec, parameter: str | None, value: float | None, seed: int) -> FaultSpec:
    spec = base if parameter is None or value is None else base.with_parameter(parameter, value)
    return with_seed(spec, seed)


def _ensure(registry: Registry, entity: Entity) -> None:
    if not registry.exists(type(entity), entity.id):
        registry.add(entity)


def _new_experiment(
    registry: Registry,
    inv: Investigation,
    name: str,
    hypothesis: str,
    model: RegisteredModel,
    data: RegisteredDataset,
    parameters: dict[str, object],
) -> Experiment:
    cfg = ConfigurationRef(parameters)
    exp = Experiment(inv.id, name, hypothesis, model.ref(), data.ref(), cfg.id, datetime.now(UTC))
    _ensure(registry, cfg)
    if not registry.exists(Experiment, exp.id):
        registry.add(exp)
        registry.update_status(exp.with_status(ExperimentStatus.READY))
    return registry.get(Experiment, exp.id)


def _trial(
    fx_id: str, p_index: int, r_index: int, spec: FaultSpec, baseline_run_id: str,
    parameter: str | None, value: float | None, status: TrialStatus, **extra: str | None,
) -> FaultTrial:  # fmt: skip
    return FaultTrial(
        fx_id, p_index, r_index, spec.seed, spec.id, spec.family_id, baseline_run_id, status,
        datetime.now(UTC), parameter if value is not None else None, value,
        extra.get("treatment_experiment_id"), extra.get("treatment_run_id"), extra.get("reason"),
    )  # fmt: skip


def _dtype_kind(dtype: str | None) -> str | None:
    if dtype is None:
        return None
    return (
        "f" if dtype.startswith("float") else "i" if dtype.startswith(("int", "uint")) else "other"
    )


def preflight(spec: FaultSpec, fault_registry: FaultRegistry, data: RegisteredDataset) -> None:
    """Compatibility check against dataset metadata, before any run exists."""
    meta = data.metadata
    schema = meta.input_schema
    ndim = len(schema.shape) if schema is not None and schema.shape is not None else None
    dtype = schema.dtype if schema is not None else None
    labels = meta.target_schema.class_labels if meta.target_schema is not None else None
    explicit = any(getattr(leaf.parameters, "classes", None) for leaf in spec.flatten())
    check_compatibility(
        spec,
        fault_registry,
        ndim=ndim,
        dtype_kind=_dtype_kind(dtype),
        has_classes=bool(labels) or explicit,
    )


def _reason(registry: Registry, run_id: str) -> str:
    outcomes = registry.find(RunOutcome, run_id=run_id)
    err = outcomes[0].error if outcomes else None
    return f"{err.error_type}: {err.message}" if err else "the run did not complete"


def _validate_baseline(
    registry: Registry,
    run_id: str,
    design: FaultDesign,
    model: RegisteredModel,
    data: RegisteredDataset,
) -> Run:
    run = registry.get(Run, run_id)
    exp = registry.get(Experiment, run.experiment_id)
    cfg = registry.get(ConfigurationRef, exp.configuration_id)
    if run.status is not RunStatus.COMPLETED:
        raise FaultError(f"baseline run {run_id} is {run.status}, not COMPLETED")
    if _plain(cfg.parameters) != _plain(design.evaluation.to_parameters()):
        raise FaultError(
            "the baseline run used a different evaluation configuration; "
            "a control must differ only by the fault"
        )
    if exp.model != model.ref() or exp.dataset != data.ref():
        raise FaultError("the baseline run used a different model or dataset")
    return run


def _plain(value: object) -> object:
    from experionyx.evaluation.config import _thaw

    return _thaw(value)


def run_fault_experiment(
    registry: Registry,
    store: ArtifactStore,
    executor: Executor,
    *,
    model_id: str,
    dataset_id: str,
    base_spec: FaultSpec,
    fault_registry: FaultRegistry,
    design: FaultDesign,
    name: str,
    source_root: Path,
    baseline_run_id: str | None = None,
) -> FaultExperimentResult:
    model = registry.get(RegisteredModel, model_id)
    data = registry.get(RegisteredDataset, dataset_id)
    sweep = design.sweep
    points = list(sweep.values) if sweep else [None]

    # -- validate everything before any run is created --------------------------------------
    if dict(design.fault) != base_spec.to_dict():
        raise FaultError("design.fault must be the canonical form of base_spec")
    for value in points:
        preflight(
            trial_spec(base_spec, sweep.parameter if sweep else None, value, design.seeds[0]),
            fault_registry,
            data,
        )
    split = design.evaluation.split
    samples = data.metadata.splits.get(split) if split else data.metadata.num_samples
    if isinstance(samples, int):
        design.check_sample_budget(samples)
        if samples > design.evaluation.retention.max_samples:
            raise FaultError(f"{samples} samples exceed the evaluation's retention.max_samples")

    inv = Investigation(
        "fault-injection",
        "How does measured behaviour change under controlled faults?",
        datetime.now(UTC),
    )
    _ensure(registry, inv)

    # -- control run -------------------------------------------------------------------------
    if baseline_run_id is None:
        exp = _new_experiment(
            registry,
            inv,
            f"baseline {model.name} {model.version} on {data.name} {data.version}",
            "Baseline evaluation (control condition)",
            model,
            data,
            design.evaluation.to_parameters(),
        )
        result = executor.execute(
            exp.id,
            resolve_procedure(EVALUATION_PROCEDURE),
            seed=RUN_SEED,
            procedure_name=EVALUATION_PROCEDURE,
        )
        if result.status is not RunStatus.COMPLETED:
            raise FaultError(f"the baseline evaluation failed: {result.error}")
        baseline_run = result.run
    else:
        baseline_run = _validate_baseline(registry, baseline_run_id, design, model, data)

    fx = FaultExperiment(
        inv.id,
        name,
        baseline_run.experiment_id,
        baseline_run.id,
        design.to_dict(),
        datetime.now(UTC),
    )
    _ensure(registry, fx)
    for status in (ExperimentStatus.READY, ExperimentStatus.RUNNING):
        fx = fx.with_status(status)
        registry.update_status(fx)

    # -- treatment trials (each a real, separately recorded run) -----------------------------
    trials: list[FaultTrial] = []
    failures = 0
    limit = design.limits.max_failed_trials
    parameter = sweep.parameter if sweep else None
    for p_index, value in enumerate(points):
        for r_index, seed in enumerate(design.seeds):
            spec = trial_spec(base_spec, parameter, value, seed)

            record = partial(
                _trial, fx.id, p_index, r_index, spec, baseline_run.id, parameter, value
            )
            if limit is not None and failures >= limit:
                trial = record(
                    TrialStatus.SKIPPED,
                    reason=f"early termination: max_failed_trials={limit} reached",
                )
            else:
                label = (
                    f"{spec.type} "
                    + (f"{parameter}={value:g} " if parameter and value is not None else "")
                    + f"seed={seed}"
                )
                try:
                    cfg = FaultEvaluationConfig(design.evaluation, spec.to_dict()).to_parameters()
                    texp = _new_experiment(
                        registry,
                        inv,
                        f"fault {label} of {fx.id}",
                        f"Faulted evaluation: {label}",
                        model,
                        data,
                        cfg,
                    )
                    res = executor.execute(
                        texp.id,
                        resolve_procedure(PROCEDURE_FAULT_EVALUATION),
                        seed=RUN_SEED,
                        procedure_name=PROCEDURE_FAULT_EVALUATION,
                    )
                    ok = res.status is RunStatus.COMPLETED
                    trial = record(
                        TrialStatus.COMPLETED if ok else TrialStatus.FAILED,
                        treatment_experiment_id=texp.id,
                        treatment_run_id=res.run.id,
                        reason=None if ok else _reason(registry, res.run.id),
                    )
                except ExperionyxError as exc:
                    trial = record(TrialStatus.FAILED, reason=f"{type(exc).__name__}: {exc}")
                failures += trial.status is TrialStatus.FAILED
            registry.add(trial)
            trials.append(trial)

    # -- analysis (a run of its own, so its artifacts, observations and evidence have provenance)
    analysis_exec = Executor(registry, store, source_root=source_root)
    aexp = _new_experiment(
        registry,
        inv,
        f"analysis of {fx.id}",
        "Degradation analysis of a fault experiment",
        model,
        data,
        FaultAnalysisConfig(fx.id).to_parameters(),
    )
    ares = analysis_exec.execute(
        aexp.id,
        resolve_procedure(PROCEDURE_FAULT_ANALYSIS),
        seed=RUN_SEED,
        procedure_name=PROCEDURE_FAULT_ANALYSIS,
    )
    registry.add(FaultAnalysis(fx.id, ares.run.id, datetime.now(UTC)))
    fx = fx.with_status(
        ExperimentStatus.COMPLETED
        if ares.status is RunStatus.COMPLETED
        else ExperimentStatus.FAILED
    )
    registry.update_status(fx)
    return FaultExperimentResult(fx, baseline_run.id, tuple(trials), ares.run.id, ares.status)

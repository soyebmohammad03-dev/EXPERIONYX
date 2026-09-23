"""The experiment dispatcher: the ONLY place the scheduler calls into an existing engine.

    Scheduler -> Experiment Dispatcher -> Existing Engine

No fault, benchmark, stress, calibration, resource, drift, quality, discovery, interaction,
profile or statistical logic is reimplemented here. Each `_dispatch_*` function deserializes a
unit's stored JSON parameters with the target engine's OWN `Spec.from_dict`, calls its OWN entry
point, and normalizes the result into a `DispatchOutcome`. Every dispatched unit keeps its
original engine identity: the record IDs a dispatch produces (a Run, a BenchmarkResult, a
StressAnalysis, ...) are exactly what calling that engine directly would have produced.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from experionyx.artifacts import ArtifactStore
from experionyx.benchmark.spec import BenchmarkSpec
from experionyx.calibration.engine import run_calibration_request
from experionyx.calibration.spec import CalibrationSpec
from experionyx.data_quality.engine import run_quality_request
from experionyx.data_quality.spec import QualitySpec
from experionyx.domain import RunStatus
from experionyx.drift.engine import run_drift_request
from experionyx.drift.spec import ShiftSpec
from experionyx.errors import (
    AdapterUnavailableError,
    DeviceUnavailableError,
    ExperionyxError,
    NotFoundError,
    UnsupportedCapabilityError,
    ValidationError,
)
from experionyx.execution import Executor, resolve_procedure
from experionyx.failures.config import DiscoveryConfig
from experionyx.failures.engine import run_discovery
from experionyx.faults.design import FaultDesign
from experionyx.faults.lab import run_fault_experiment
from experionyx.faults.library import default_fault_registry
from experionyx.interactions.config import InteractionSpec
from experionyx.interactions.engine import run_interaction
from experionyx.provenance import FailureStage, RunOutcome
from experionyx.registry import Registry
from experionyx.reliability.engine import run_profile
from experionyx.reliability.spec import ProfileSpec
from experionyx.resources.engine import run_resource_request
from experionyx.resources.spec import ResourceSpec
from experionyx.scheduler.taxonomy import FailureCategory, UnitKind, UnitState
from experionyx.stats import store as stats_store
from experionyx.stress.spec import StressDesign

_DEP_PREFIX = "$dep:"


class DispatchError(ExperionyxError):
    """A unit's parameters reference a dependency that was not resolved, or name a kind this
    dispatcher does not (yet) support. Refused before any engine call is made."""


def referenced_dependencies(value: object) -> set[str]:
    """Every dependency key named by a `"$dep:<key>"` sentinel anywhere in `value` (recursively).
    Used by dry-run to catch a parameters block that references a key the unit never declared in
    `depends_on`, before anything is dispatched."""
    found: set[str] = set()
    if isinstance(value, str) and value.startswith(_DEP_PREFIX):
        found.add(value[len(_DEP_PREFIX) :])
    elif isinstance(value, Mapping):
        for v in value.values():
            found |= referenced_dependencies(v)
    elif isinstance(value, list | tuple):
        for v in value:
            found |= referenced_dependencies(v)
    return found


def substitute(value: object, refs: Mapping[str, str]) -> object:
    """Recursively replace `"$dep:<key>"` strings with `refs[key]` (a dependency's primary
    reference). Raises DispatchError for an unresolved key; everything else passes through."""
    if isinstance(value, str) and value.startswith(_DEP_PREFIX):
        key = value[len(_DEP_PREFIX) :]
        if key not in refs:
            raise DispatchError(f"unresolved dependency reference {value!r}")
        return refs[key]
    if isinstance(value, Mapping):
        return {k: substitute(v, refs) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [substitute(v, refs) for v in value]
    return value


@dataclass(frozen=True)
class DispatchOutcome:
    status: UnitState  # SUCCEEDED or FAILED
    primary_ref: str | None = None
    run_id: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    failure_category: FailureCategory | None = None


def _ok(primary_ref: str | None, run_id: str | None) -> DispatchOutcome:
    return DispatchOutcome(UnitState.SUCCEEDED, primary_ref, run_id)


_STAGE_CATEGORY = {
    FailureStage.PREPARATION: FailureCategory.CONFIGURATION,
    FailureStage.EXECUTION: FailureCategory.DATA_MODEL,
    FailureStage.ARTIFACT_REGISTRATION: FailureCategory.TRANSIENT,
    FailureStage.INTERRUPTED: FailureCategory.TRANSIENT,
}
_CONFIGURATION_ERRORS = (ValidationError, NotFoundError)
_UNSUPPORTED_ERRORS = (UnsupportedCapabilityError, AdapterUnavailableError, DeviceUnavailableError)


def _from_run(
    registry: Registry, run_id: str, primary_ref: str | None, status: RunStatus
) -> DispatchOutcome:
    """Classify a dispatch that went through the execution engine using its own recorded outcome
    (never guessed): `RunOutcome.error.stage` says exactly why it failed."""
    if status is RunStatus.COMPLETED:
        return _ok(primary_ref, run_id)
    found = registry.find(RunOutcome, run_id=run_id)
    error = found[0].error if found else None
    return DispatchOutcome(
        UnitState.FAILED,
        primary_ref,
        run_id,
        error.error_type if error else "UnknownFailure",
        error.message if error else f"run {run_id} ended {status.value} with no recorded outcome",
        _STAGE_CATEGORY.get(error.stage, FailureCategory.TRANSIENT)
        if error
        else FailureCategory.TRANSIENT,
    )


def _from_exception(exc: Exception) -> DispatchOutcome:
    if isinstance(exc, _UNSUPPORTED_ERRORS):
        category = FailureCategory.UNSUPPORTED
    elif isinstance(exc, _CONFIGURATION_ERRORS):
        category = FailureCategory.CONFIGURATION
    elif isinstance(exc, ExperionyxError):
        category = FailureCategory.CONFIGURATION  # a *Refusal: refused before anything ran
    else:
        category = FailureCategory.TRANSIENT
    return DispatchOutcome(
        UnitState.FAILED,
        None,
        None,
        f"{type(exc).__module__}.{type(exc).__qualname__}",
        str(exc),
        category,
    )


def _dispatch_baseline(
    registry: Registry, executor: Executor, params: Mapping[str, object]
) -> DispatchOutcome:
    from experionyx.evaluation.engine import PROCEDURE

    experiment_id = str(params["experiment_id"])
    seed = int(params.get("seed", 0))  # type: ignore[call-overload]
    result = executor.execute(
        experiment_id, resolve_procedure(PROCEDURE), seed=seed, procedure_name=PROCEDURE
    )
    return _from_run(registry, result.run.id, result.run.id, result.status)


def _dispatch_fault(
    registry: Registry,
    store: ArtifactStore,
    executor: Executor,
    params: Mapping[str, object],
    source_root: Path,
) -> DispatchOutcome:
    base_spec_raw, design_raw = params["base_spec"], params["design"]
    if not isinstance(base_spec_raw, Mapping) or not isinstance(design_raw, Mapping):
        raise DispatchError("base_spec and design must be objects")
    fr = default_fault_registry()
    base = fr.from_dict(base_spec_raw)
    design = FaultDesign.from_dict(design_raw)
    res = run_fault_experiment(
        registry, store, executor,
        model_id=str(params["model_id"]), dataset_id=str(params["dataset_id"]),
        base_spec=base, fault_registry=fr, design=design, name=str(params["name"]),
        source_root=source_root, baseline_run_id=params.get("baseline_run_id"),  # type: ignore[arg-type]
    )  # fmt: skip
    return _ok(res.fault_experiment.id, res.baseline_run_id)


def _dispatch_interaction(registry: Registry, store: ArtifactStore, executor: Executor, investigation_id: str, params: Mapping[str, object]) -> DispatchOutcome:  # fmt: skip
    spec = InteractionSpec.from_dict(params)
    out = run_interaction(registry, store, executor, investigation_id, spec)
    if out.analysis_id is None:
        return DispatchOutcome(UnitState.FAILED, None, None, "InteractionRefused", f"the interaction run ended {out.status.value}", FailureCategory.CONFIGURATION)  # fmt: skip
    return _ok(out.analysis_id, None)


def _dispatch_benchmark(registry: Registry, store: ArtifactStore, executor: Executor, params: Mapping[str, object], source_root: Path) -> DispatchOutcome:  # fmt: skip
    from experionyx.benchmark.engine import run_benchmark

    body = dict(params)
    disc = body.pop("discovery_config", None)
    if disc is not None and not isinstance(disc, Mapping):
        raise DispatchError("discovery_config must be an object")
    spec = BenchmarkSpec.from_dict(body)
    res = run_benchmark(registry, store, executor, spec, source_root=source_root, discovery_config=DiscoveryConfig.from_dict(disc) if disc else None)  # fmt: skip
    if res.result_id is None:
        return DispatchOutcome(UnitState.FAILED, None, res.run_id, "BenchmarkIncomplete", f"errors: {dict(res.errors)}", FailureCategory.DATA_MODEL)  # fmt: skip
    return _ok(res.result_id, res.run_id)


def _dispatch_stress(registry: Registry, store: ArtifactStore, executor: Executor, investigation_id: str, params: Mapping[str, object], source_root: Path) -> DispatchOutcome:  # fmt: skip
    from experionyx.stress.engine import run_stress_experiment

    design = StressDesign.from_dict(params)
    out = run_stress_experiment(registry, store, executor, design, source_root=source_root, investigation_id=investigation_id)  # fmt: skip
    if out.analysis_id is None:
        return DispatchOutcome(UnitState.FAILED, None, None, "StressIncomplete", f"the stress run ended {out.status.value}", FailureCategory.DATA_MODEL)  # fmt: skip
    return _ok(out.analysis_id, None)


def _dispatch_calibration(registry: Registry, store: ArtifactStore, executor: Executor, investigation_id: str, params: Mapping[str, object]) -> DispatchOutcome:  # fmt: skip
    spec = CalibrationSpec.from_dict(params)
    out = run_calibration_request(registry, store, executor, investigation_id, spec)
    if out.analysis_id is None:
        return DispatchOutcome(UnitState.FAILED, None, None, "CalibrationIncomplete", f"the calibration run ended {out.status.value}", FailureCategory.DATA_MODEL)  # fmt: skip
    return _ok(out.analysis_id, None)


def _dispatch_resource(registry: Registry, store: ArtifactStore, executor: Executor, investigation_id: str, params: Mapping[str, object]) -> DispatchOutcome:  # fmt: skip
    spec = ResourceSpec.from_dict(params)
    out = run_resource_request(registry, store, executor, investigation_id, spec)
    if out.analysis_id is None:
        return DispatchOutcome(UnitState.FAILED, None, None, "ResourceIncomplete", f"the resource run ended {out.status.value}", FailureCategory.TRANSIENT)  # fmt: skip
    return _ok(out.analysis_id, None)


def _dispatch_drift(registry: Registry, store: ArtifactStore, executor: Executor, investigation_id: str, params: Mapping[str, object]) -> DispatchOutcome:  # fmt: skip
    spec = ShiftSpec.from_dict(params)
    out = run_drift_request(registry, store, executor, investigation_id, spec)
    if out.analysis_id is None:
        return DispatchOutcome(UnitState.FAILED, None, None, "DriftIncomplete", f"the drift run ended {out.status.value}", FailureCategory.DATA_MODEL)  # fmt: skip
    return _ok(out.analysis_id, None)


def _dispatch_quality(registry: Registry, store: ArtifactStore, executor: Executor, investigation_id: str, params: Mapping[str, object]) -> DispatchOutcome:  # fmt: skip
    spec = QualitySpec.from_dict(params)
    res = run_quality_request(registry, store, executor, investigation_id, spec)
    return _from_run(registry, res.run_id, res.analysis_id, res.status)


def _as_id_list(params: Mapping[str, object], *keys: str) -> list[str]:
    for key in keys:
        raw = params.get(key)
        if raw is not None:
            if not isinstance(raw, list | tuple):
                raise DispatchError(f"{key} must be a list of IDs")
            return [str(x) for x in raw]
    return []


def _dispatch_discovery(registry: Registry, store: ArtifactStore, executor: Executor, investigation_id: str, params: Mapping[str, object]) -> DispatchOutcome:  # fmt: skip
    cfg = params.get("config")
    if cfg is not None and not isinstance(cfg, Mapping):
        raise DispatchError("config must be an object")
    run_ids = _as_id_list(params, "run_ids", "baseline_run_ids")
    fx_ids = _as_id_list(params, "fault_experiment_ids")
    res = run_discovery(registry, store, executor, investigation_id, run_ids, fx_ids, DiscoveryConfig.from_dict(cfg) if cfg else None)  # fmt: skip
    return _from_run(registry, res.run_id, res.run_id, res.status)


def _dispatch_profile(registry: Registry, store: ArtifactStore, executor: Executor, investigation_id: str, params: Mapping[str, object]) -> DispatchOutcome:  # fmt: skip
    spec = ProfileSpec.from_dict(params)
    out = run_profile(registry, store, executor, investigation_id, spec)
    if out.profile_id is None:
        return DispatchOutcome(UnitState.FAILED, None, None, "ProfileRefused", f"the profile run ended {out.status.value}", FailureCategory.CONFIGURATION)  # fmt: skip
    return _ok(out.profile_id, None)


def _dispatch_statistical(registry: Registry, store: ArtifactStore, params: Mapping[str, object]) -> DispatchOutcome:  # fmt: skip
    kind = str(params["kind"])
    sources_raw = params["sources"]
    if not isinstance(sources_raw, Mapping):
        raise DispatchError("sources must be an object")
    config_raw = params.get("config")
    if config_raw is not None and not isinstance(config_raw, Mapping):
        raise DispatchError("config must be an object")
    analysis, _ = stats_store.create(registry, store, kind, dict(sources_raw), dict(config_raw) if config_raw else None)  # fmt: skip
    return _ok(analysis.id, None)


def dispatch(
    kind: UnitKind,
    registry: Registry,
    store: ArtifactStore,
    executor: Executor,
    investigation_id: str,
    parameters: Mapping[str, object],
    refs: Mapping[str, str],
    source_root: Path,
) -> DispatchOutcome:
    """Resolve `$dep:` references against `refs` (unit key -> primary_ref of a SUCCEEDED
    dependency), then call the one engine `kind` names. Never raises: every failure -- a refusal
    before any run exists, or a real execution failure -- becomes a FAILED DispatchOutcome."""
    try:
        payload = substitute(parameters, refs)
        if not isinstance(payload, dict):
            raise DispatchError("unit parameters must be a JSON object")
        if kind is UnitKind.BASELINE_EVALUATION:
            return _dispatch_baseline(registry, executor, payload)
        if kind is UnitKind.FAULT_EXPERIMENT:
            return _dispatch_fault(registry, store, executor, payload, source_root)
        if kind is UnitKind.INTERACTION:
            return _dispatch_interaction(registry, store, executor, investigation_id, payload)
        if kind is UnitKind.BENCHMARK:
            return _dispatch_benchmark(registry, store, executor, payload, source_root)
        if kind is UnitKind.STRESS:
            return _dispatch_stress(
                registry, store, executor, investigation_id, payload, source_root
            )
        if kind is UnitKind.CALIBRATION:
            return _dispatch_calibration(registry, store, executor, investigation_id, payload)
        if kind is UnitKind.RESOURCE:
            return _dispatch_resource(registry, store, executor, investigation_id, payload)
        if kind is UnitKind.DRIFT:
            return _dispatch_drift(registry, store, executor, investigation_id, payload)
        if kind is UnitKind.DATA_QUALITY:
            return _dispatch_quality(registry, store, executor, investigation_id, payload)
        if kind is UnitKind.FAILURE_DISCOVERY:
            return _dispatch_discovery(registry, store, executor, investigation_id, payload)
        if kind is UnitKind.RELIABILITY_PROFILE:
            return _dispatch_profile(registry, store, executor, investigation_id, payload)
        if kind is UnitKind.STATISTICAL_ANALYSIS:
            return _dispatch_statistical(registry, store, payload)
        raise DispatchError(f"no dispatcher for unit kind {kind.value}")
    except Exception as exc:  # a dispatch NEVER raises: every failure becomes a recorded outcome
        return _from_exception(exc)


__all__ = ["DispatchError", "DispatchOutcome", "dispatch", "referenced_dependencies", "substitute"]

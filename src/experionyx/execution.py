"""Execution engine: turns a registered Experiment into a traced Run (see docs/execution.md).

Lifecycle of `Executor.execute`:
  validate -> capture environment/source -> create Run (PENDING) -> record Provenance and move to
  RUNNING (one transaction) -> call the procedure -> verify artifacts -> record RunOutcome and the
  final status (one transaction).
Failures inside the procedure are recorded, not raised. Failures before a Run exists are raised
as PreparationError because there is no Run to attach them to.
"""

import importlib
import logging
import os
import sqlite3
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import experionyx
import experionyx.validation as v
from experionyx.adapters.base import DatasetAdapter, ModelAdapter
from experionyx.adapters.capabilities import DeviceInfo, DeviceKind
from experionyx.adapters.records import RegisteredDataset, RegisteredModel
from experionyx.adapters.registry import AdapterRegistries
from experionyx.artifacts import ArtifactStore
from experionyx.capture import (
    EnvironmentCapture,
    capture_environment,
    capture_source,
    seed_everything,
)
from experionyx.domain import (
    Artifact,
    ArtifactCategory,
    ConfigurationRef,
    EnvironmentSnapshot,
    EpistemicKind,
    Experiment,
    ExperimentStatus,
    Observation,
    ObservationValue,
    Run,
    RunStatus,
)
from experionyx.errors import (
    ArtifactError,
    DatasetFingerprintError,
    ExperionyxError,
    ModelFingerprintError,
    PreparationError,
    ReplayError,
)
from experionyx.provenance import (
    AdapterInputs,
    ErrorInfo,
    ExecutionParameters,
    FailureStage,
    InputBinding,
    Provenance,
    ResourceLimits,
    RunOutcome,
    SourceRevision,
)
from experionyx.registry import Registry

logger = logging.getLogger(__name__)

Clock = Callable[[], datetime]
_NOT_EXECUTABLE = frozenset(
    {ExperimentStatus.DRAFT, ExperimentStatus.CANCELLED, ExperimentStatus.FAILED}
)
_DIAGNOSTIC_PATH = "diagnostics/traceback.txt"


def utc_now() -> datetime:
    return datetime.now(UTC)


class _Recorder:
    """Persists what a running procedure produces, as it produces it."""

    def __init__(
        self, registry: Registry, store: ArtifactStore, run: Run, artifact_dir: Path, clock: Clock
    ) -> None:
        self.registry, self.store, self.run = registry, store, run
        self.artifact_dir, self.clock = artifact_dir, clock
        self.observations: list[Observation] = []
        self.artifacts: list[Artifact] = []
        self._sequence: dict[str, int] = {}

    def observe(
        self, name: str, value: ObservationValue, unit: str | None, kind: EpistemicKind
    ) -> Observation:
        sequence = self._sequence.get(name, 0)
        obs = Observation(
            self.run.id, name, value, self.clock(), sequence, unit, epistemic_kind=kind
        )
        self.registry.add(obs)
        self._sequence[name] = sequence + 1
        self.observations.append(obs)
        return obs

    def register(
        self,
        path: str,
        name: str,
        category: ArtifactCategory,
        media_type: str | None,
    ) -> Artifact:
        artifact = self.store.register(
            self.run,
            path,
            name=name,
            category=category,
            media_type=media_type,
            created_at=self.clock(),
        )
        self.registry.add(artifact)  # only a successfully hashed+stored file is ever referenced
        self.artifacts.append(artifact)
        return artifact


@dataclass(frozen=True)
class RunContext:
    """Everything a procedure receives. Immutable; results are reported via `observe` and
    `register_artifact`, which persist immediately."""

    experiment: Experiment
    run: Run
    configuration: ConfigurationRef
    seed: int
    started_at: datetime
    source: SourceRevision
    execution: ExecutionParameters
    artifact_dir: Path  # write output files here, then register them
    project_root: Path
    recorder: _Recorder = field(repr=False, compare=False)
    model: ModelAdapter | None = None  # loaded and fingerprint-verified, if the experiment has one
    dataset: DatasetAdapter | None = None
    device: DeviceInfo | None = None

    @property
    def parameters(self) -> Mapping[str, object]:
        return self.configuration.parameters

    def observe(
        self,
        name: str,
        value: ObservationValue,
        *,
        unit: str | None = None,
        kind: EpistemicKind = EpistemicKind.OBSERVATION,
    ) -> Observation:
        return self.recorder.observe(name, value, unit, kind)

    def register_artifact(
        self,
        path: str,
        *,
        name: str | None = None,
        category: ArtifactCategory = ArtifactCategory.OUTPUT,
        media_type: str | None = None,
    ) -> Artifact:
        """Hash and register a file already written under `artifact_dir` (`path` is relative)."""
        return self.recorder.register(path, name or path, category, media_type)


Procedure = Callable[[RunContext], None]


@dataclass(frozen=True)
class ExecutionResult:
    run: Run  # final state
    outcome: RunOutcome
    provenance: Provenance
    observations: tuple[Observation, ...]
    artifacts: tuple[Artifact, ...]

    @property
    def status(self) -> RunStatus:
        return self.run.status

    @property
    def error(self) -> ErrorInfo | None:
        return self.outcome.error

    @property
    def duration_seconds(self) -> float | None:
        return self.outcome.duration_seconds


def _sanitize(text: str) -> str:
    """Strip the user's home directory from diagnostics destined for public repositories."""
    home = str(Path.home())
    return text.replace(home, "~") if len(home) > 1 else text


def _type_name(exc: BaseException) -> str:
    return f"{type(exc).__module__}.{type(exc).__qualname__}"


def _qualified_name(procedure: Procedure) -> str:
    module = getattr(procedure, "__module__", None)
    qualname = getattr(procedure, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(qualname, str) or "<" in qualname:
        raise PreparationError(
            "cannot derive an importable name for this procedure (lambda, local or callable "
            "object); pass procedure_name='module:function'"
        )
    return f"{module}:{qualname}"


def resolve_procedure(ref: str) -> Procedure:
    """Import `module:qualified.name` and return the callable."""
    module_name, sep, attr = ref.partition(":")
    if not (module_name and sep and attr):
        raise PreparationError(f"procedure must look like 'module:function', got {ref!r}")
    try:
        obj: object = importlib.import_module(module_name)
        for part in attr.split("."):
            obj = getattr(obj, part)
    except (ImportError, AttributeError) as exc:
        raise PreparationError(f"cannot import procedure {ref!r}: {exc}") from exc
    if not callable(obj):
        raise PreparationError(f"{ref!r} is not callable")
    return cast(Procedure, obj)


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def run_states(registry: Registry, experiment_id: str | None = None) -> dict[RunStatus, list[Run]]:
    """Runs grouped by lifecycle state (the resumability view: pending/running/completed/failed)."""
    runs = (
        registry.find(Run)
        if experiment_id is None
        else registry.find(Run, experiment_id=experiment_id)
    )
    return {s: [r for r in runs if r.status is s] for s in RunStatus}


@dataclass(frozen=True)
class _Bound:
    """Adapters resolved for one run (all None for experiments without registered inputs)."""

    inputs: AdapterInputs | None = None
    model: ModelAdapter | None = None
    dataset: DatasetAdapter | None = None
    runtime: Mapping[str, object] = field(default_factory=dict)


class Executor:
    def __init__(
        self,
        registry: Registry,
        store: ArtifactStore,
        *,
        source_root: Path,
        clock: Clock = utc_now,
        adapters: AdapterRegistries | None = None,
        inputs_root: Path | None = None,
        device: DeviceKind = DeviceKind.CPU,
    ) -> None:
        """`adapters`: if given, registered models/datasets referenced by an experiment (a
        ModelRef/DatasetRef with a digest) are resolved, loaded and verified before each run.
        Relative `source` paths of registered inputs resolve against `inputs_root`.
        `device`: default device policy (CPU unless changed)."""
        self._registry, self._store = registry, store
        self._source_root, self._clock = source_root, clock
        self._adapters, self._inputs_root, self._device = adapters, inputs_root, device

    # -- public API ---------------------------------------------------------------------------

    def execute(
        self,
        experiment_id: str,
        procedure: Procedure,
        *,
        seed: int,
        procedure_name: str | None = None,
        resources: ResourceLimits | None = None,
        metadata: Mapping[str, object] | None = None,
        replay_of: str | None = None,
        device: DeviceKind | None = None,
    ) -> ExecutionResult:
        """Run `procedure` as a new Run of the experiment. Never mutates earlier runs."""
        v.non_negative_int("seed", seed)
        experiment = self._registry.get(Experiment, experiment_id)
        if experiment.status in _NOT_EXECUTABLE:
            raise PreparationError(f"experiment is {experiment.status}; it cannot be executed")
        configuration = self._registry.get(ConfigurationRef, experiment.configuration_id)
        execution = ExecutionParameters(
            procedure_name or _qualified_name(procedure),
            resources or ResourceLimits(),
            metadata or {},
        )
        try:
            seeded = seed_everything(seed)
            capture = capture_environment(seeded)
            source = capture_source(self._source_root)
        except Exception as exc:
            raise PreparationError(f"environment capture failed: {exc}") from exc

        run = self._create_run(experiment, capture, seed)
        logger.info("run %s created for experiment %s (seed %d)", run.id, experiment.id, seed)
        try:
            bound = self._bind_inputs(experiment, device or self._device)
        except Exception as exc:
            self._record_unstarted_failure(run, exc)
            raise PreparationError(f"run {run.id}: cannot bind inputs: {exc}") from exc
        artifact_dir, provenance = self._start(
            run, experiment, capture, source, seed, execution, replay_of, bound
        )
        running = run.with_status(RunStatus.RUNNING)
        recorder = _Recorder(self._registry, self._store, running, artifact_dir, self._clock)
        context = RunContext(
            experiment, running, configuration, seed, provenance.started_at, source,
            execution, artifact_dir, self._source_root, recorder,
            bound.model, bound.dataset, bound.inputs.device if bound.inputs else None,
        )  # fmt: skip

        error: ErrorInfo | None = None
        diagnostic_text: str | None = None
        interrupt: BaseException | None = None
        began = time.perf_counter()
        try:
            procedure(context)
        except Exception as exc:
            error = ErrorInfo(FailureStage.EXECUTION, _type_name(exc), _sanitize(str(exc)))
            diagnostic_text = _sanitize(traceback.format_exc())
        except BaseException as exc:  # KeyboardInterrupt / SystemExit: record, then re-raise
            interrupt = exc
            error = ErrorInfo(FailureStage.INTERRUPTED, _type_name(exc), _sanitize(str(exc)))
            diagnostic_text = _sanitize(traceback.format_exc())
        duration = time.perf_counter() - began

        result = self._finalize(running, recorder, provenance, error, diagnostic_text, duration)
        if interrupt is not None:
            raise interrupt
        return result

    def replay(self, run_id: str, procedure: Procedure | None = None) -> ExecutionResult:
        """Request a replay: a NEW Run built from a finished run's recorded inputs.

        This is REPLAY_REQUESTED only. Nothing here verifies that results reproduce.
        """
        original = self._registry.get(Run, run_id)
        if original.status not in (RunStatus.COMPLETED, RunStatus.FAILED):
            raise ReplayError(
                f"run {run_id} is {original.status}; only finished runs can be replayed"
            )
        recorded = self._registry.find(Provenance, run_id=run_id)
        if not recorded:
            raise ReplayError(f"run {run_id} has no provenance (it failed before starting)")
        prov = recorded[0]
        try:
            target = procedure or resolve_procedure(prov.execution.procedure)
        except PreparationError as exc:
            raise ReplayError(f"cannot replay {run_id}: {exc}") from exc
        return self.execute(
            original.experiment_id,
            target,
            seed=prov.seed,
            procedure_name=None if procedure else prov.execution.procedure,
            resources=prov.execution.resources,
            metadata=prov.execution.metadata,
            replay_of=run_id,
            device=prov.inputs.device.requested if prov.inputs and prov.inputs.device else None,
        )

    def recover_interrupted(self) -> list[Run]:
        """Close RUNNING runs whose process is gone (e.g. killed) as FAILED/INTERRUPTED.

        Detection is by the pid recorded in provenance, so it is only meaningful on the host that
        ran them; a run whose pid is still alive is left alone. Pid reuse can hide a dead run.
        """
        recovered: list[Run] = []
        for run in self._registry.find(Run, status=RunStatus.RUNNING.value):
            recorded = self._registry.find(Provenance, run_id=run.id)
            pid = recorded[0].runtime.get("pid") if recorded else None
            if isinstance(pid, int) and _process_alive(pid):
                continue
            artifacts = self._registry.find(Artifact, run_id=run.id)
            outcome = RunOutcome(
                run.id,
                self._clock(),
                RunStatus.FAILED,
                None,  # duration unknown
                len(self._registry.find(Observation, run_id=run.id)),
                tuple(sorted(a.id for a in artifacts)),
                ErrorInfo(
                    FailureStage.INTERRUPTED,
                    "experionyx.ProcessEnded",
                    "the process ended without recording an outcome (recovered later)",
                ),
            )
            final = run.with_status(RunStatus.FAILED)
            with self._registry.transaction():
                self._registry.add(outcome)
                self._registry.update_status(final)
            logger.warning("run %s recovered as FAILED (interrupted)", run.id)
            recovered.append(final)
        return recovered

    # -- internals ----------------------------------------------------------------------------

    def _source_path(self, source: str | None, what: str) -> str:
        if source is None:
            raise PreparationError(f"the registered {what} has no source to load from")
        if source.startswith("builtin:") or Path(source).is_absolute() or self._inputs_root is None:
            return source
        return str(self._inputs_root / source)

    def _bind_inputs(self, experiment: Experiment, device: DeviceKind) -> _Bound:
        """Resolve, load and fingerprint-verify the experiment's registered model/dataset."""
        if self._adapters is None:
            return _Bound()
        model = dataset = None
        model_binding = dataset_binding = None
        runtime: dict[str, object] = {}
        if experiment.model.digest is not None and (loaded := self._load_model(experiment, device)):
            model, model_binding = loaded
            runtime["model_load_seconds"] = model.load_seconds
        if experiment.dataset.digest is not None and (data := self._load_dataset(experiment)):
            dataset, dataset_binding = data
        if model_binding is None and dataset_binding is None:
            return _Bound()
        inputs = AdapterInputs(model_binding, dataset_binding, model.device if model else None)
        return _Bound(inputs, model, dataset, runtime)

    def _load_model(
        self, experiment: Experiment, device: DeviceKind
    ) -> tuple[ModelAdapter, InputBinding] | None:
        """None if the ref has no registered record (an unbound external reference); a registered
        model that fails to load or no longer matches its fingerprint is an error."""
        assert self._adapters is not None  # noqa: S101  # guarded by _bind_inputs
        ref = experiment.model
        records = self._registry.find(
            RegisteredModel, name=ref.name, version=ref.version, fingerprint=str(ref.digest)
        )
        if not records:
            logger.warning(
                "model %s %s is not registered; running without an adapter", ref.name, ref.version
            )
            return None
        usable = []
        for rec in records:
            adapter_cls = self._adapters.models.resolve(rec.adapter)
            if rec.adapter_version == adapter_cls.VERSION:
                usable.append((rec, adapter_cls))
        if not usable:
            raise PreparationError(
                f"model {ref.name} {ref.version} was registered with adapter version(s) "
                f"{sorted(r.adapter_version for r in records)}, none of which is installed; "
                "re-register it with the current adapter"
            )
        if len(usable) > 1:
            raise PreparationError(f"model {ref.name} {ref.version} is registered ambiguously")
        rec, adapter_cls = usable[0]
        adapter = adapter_cls.load(
            self._source_path(rec.source, "model"),
            version=rec.version,
            device=device,
            options=rec.options,
        )
        if adapter.fingerprint() != rec.fingerprint:
            raise ModelFingerprintError(
                f"model {rec.name} {rec.version} has changed since registration: "
                f"registered {rec.fingerprint}, found {adapter.fingerprint()}"
            )
        return adapter, InputBinding(rec.id, adapter_cls.NAME, adapter_cls.VERSION, rec.fingerprint)

    def _load_dataset(self, experiment: Experiment) -> tuple[DatasetAdapter, InputBinding] | None:
        assert self._adapters is not None  # noqa: S101
        ref = experiment.dataset
        records = self._registry.find(
            RegisteredDataset, name=ref.name, version=ref.version, fingerprint=str(ref.digest)
        )
        if not records:
            logger.warning(
                "dataset %s %s is not registered; running without an adapter", ref.name, ref.version
            )
            return None
        usable = []
        for rec in records:
            adapter_cls = self._adapters.datasets.resolve(rec.adapter)
            if rec.adapter_version == adapter_cls.VERSION:
                usable.append((rec, adapter_cls))
        if len(usable) != 1:
            raise PreparationError(
                f"dataset {ref.name} {ref.version}: no unique registered record matches an "
                "installed adapter version; re-register it with the current adapter"
            )
        rec, adapter_cls = usable[0]
        adapter = adapter_cls.load(
            self._source_path(rec.source, "dataset"), version=rec.version, options=rec.options
        )
        if adapter.fingerprint() != rec.fingerprint:
            raise DatasetFingerprintError(
                f"dataset {rec.name} {rec.version} has changed since registration: "
                f"registered {rec.fingerprint}, found {adapter.fingerprint()}"
            )
        return adapter, InputBinding(rec.id, adapter_cls.NAME, adapter_cls.VERSION, rec.fingerprint)

    def _create_run(self, experiment: Experiment, capture: EnvironmentCapture, seed: int) -> Run:
        env = capture.snapshot
        with self._registry.transaction():
            if not self._registry.exists(EnvironmentSnapshot, env.id):
                self._registry.add(env)
            earlier = [
                r.attempt
                for r in self._registry.find(
                    Run, experiment_id=experiment.id, environment_id=env.id
                )
                if r.seed == seed
            ]
            run = Run(
                experiment.id, env.id, seed, self._clock(), attempt=max(earlier, default=-1) + 1
            )
            self._registry.add(run)
        return run

    def _start(
        self,
        run: Run,
        experiment: Experiment,
        capture: EnvironmentCapture,
        source: SourceRevision,
        seed: int,
        execution: ExecutionParameters,
        replay_of: str | None,
        bound: _Bound,
    ) -> tuple[Path, Provenance]:
        try:
            artifact_dir = self._store.prepare(run)
            provenance = Provenance(
                run_id=run.id,
                experiment_id=experiment.id,
                environment_id=capture.snapshot.id,
                dependency_digest=capture.dependency_digest,
                configuration_id=experiment.configuration_id,
                seed=seed,
                source=source,
                executor_version=experionyx.__version__,
                execution=execution,
                started_at=self._clock(),
                runtime={
                    **capture.runtime,
                    **bound.runtime,
                    "dependency_issues": list(capture.dependency_issues),
                },
                replay_of=replay_of,
                inputs=bound.inputs,
            )
            self._store.write_metadata(run, "provenance.json", _document(provenance))
            with self._registry.transaction():
                self._registry.add(provenance)
                self._registry.update_status(run.with_status(RunStatus.RUNNING))
        except Exception as exc:
            self._record_unstarted_failure(run, exc)
            raise PreparationError(f"run {run.id} could not be started: {exc}") from exc
        return artifact_dir, provenance

    def _record_unstarted_failure(self, run: Run, exc: Exception) -> None:
        """PENDING -> FAILED so the aborted preparation stays visible."""
        outcome = RunOutcome(
            run.id,
            self._clock(),
            RunStatus.FAILED,
            None,
            0,
            (),
            ErrorInfo(FailureStage.PREPARATION, _type_name(exc), _sanitize(str(exc))),
        )
        try:
            with self._registry.transaction():
                self._registry.add(outcome)
                self._registry.update_status(run.with_status(RunStatus.FAILED))
        except (ExperionyxError, sqlite3.Error):
            logger.exception("could not record the preparation failure of run %s", run.id)

    def _finalize(
        self,
        running: Run,
        recorder: _Recorder,
        provenance: Provenance,
        error: ErrorInfo | None,
        diagnostic_text: str | None,
        duration: float,
    ) -> ExecutionResult:
        if error is None:
            try:
                for artifact in tuple(recorder.artifacts):
                    self._store.verify(running, artifact)
            except ArtifactError as exc:
                error = ErrorInfo(FailureStage.ARTIFACT_REGISTRATION, _type_name(exc), str(exc))
        if error is not None:
            text = diagnostic_text or f"{error.error_type}: {error.message}\n"
            diagnostic = self._register_diagnostic(recorder, text)
            if diagnostic is not None:
                error = replace(error, diagnostic_artifact_id=diagnostic.id)

        status = RunStatus.FAILED if error else RunStatus.COMPLETED
        outcome = RunOutcome(
            running.id,
            self._clock(),
            status,
            duration,
            len(recorder.observations),
            tuple(sorted({a.id for a in recorder.artifacts})),
            error,
        )
        final = running.with_status(status)
        with self._registry.transaction():  # outcome and final status commit together or not at all
            self._registry.add(outcome)
            self._registry.update_status(final)
        logger.info("run %s %s in %.3fs", final.id, status, duration)
        try:
            self._store.write_metadata(
                final, "outcome.json", {"id": outcome.id, **outcome.to_dict()}
            )
        except OSError:
            logger.warning("could not write outcome.json for run %s", final.id, exc_info=True)
        return ExecutionResult(
            final, outcome, provenance, tuple(recorder.observations), tuple(recorder.artifacts)
        )

    def _register_diagnostic(self, recorder: _Recorder, text: str) -> Artifact | None:
        try:
            target = recorder.artifact_dir / _DIAGNOSTIC_PATH
            target.parent.mkdir(exist_ok=True)
            target.write_text(text, encoding="utf-8")
            return recorder.register(
                _DIAGNOSTIC_PATH, "traceback", ArtifactCategory.DIAGNOSTIC, "text/plain"
            )
        except (OSError, ExperionyxError, sqlite3.Error):
            logger.warning(
                "could not retain diagnostics for run %s", recorder.run.id, exc_info=True
            )
            return None


def _document(provenance: Provenance) -> dict[str, object]:
    return {"id": provenance.id, **provenance.to_dict(), "fingerprint": provenance.fingerprint}

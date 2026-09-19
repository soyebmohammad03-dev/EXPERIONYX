"""Adapter-backed execution: registered models/datasets resolved, verified and recorded."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

import procedures
from experionyx.adapters.capabilities import DeviceKind
from experionyx.adapters.records import RegisteredDataset, RegisteredModel
from experionyx.adapters.registry import AdapterRegistries, AdapterRegistry
from experionyx.artifacts import LocalArtifactStore
from experionyx.domain import (
    ConfigurationRef,
    Experiment,
    ExperimentStatus,
    Investigation,
    Observation,
    Run,
    RunStatus,
)
from experionyx.errors import (
    AdapterNotFoundError,
    DuplicateError,
    PreparationError,
    ValidationError,
)
from experionyx.execution import Executor
from experionyx.provenance import FailureStage, Provenance, RunOutcome, compare_provenance
from experionyx.sqlite import SqliteRegistry
from pure_adapters import ConstantModelAdapter, ListDatasetAdapter

T0 = datetime(2026, 9, 19, 12, tzinfo=UTC)
ROWS = {
    "rows": [[float(i)] for i in range(10)],
    "targets": list(range(10)),
    "splits": {"test": [7, 8, 9, 1, 2]},
}


def registries() -> AdapterRegistries:
    models: AdapterRegistry = AdapterRegistry("model")  # type: ignore[type-arg]
    datasets: AdapterRegistry = AdapterRegistry("dataset")  # type: ignore[type-arg]
    models.register("constant", ConstantModelAdapter)
    datasets.register("list", ListDatasetAdapter)
    return AdapterRegistries(models, datasets)


@dataclass
class World:
    registry: SqliteRegistry
    workspace: Path
    model: RegisteredModel
    dataset: RegisteredDataset
    experiment: Experiment
    executor: Executor


def build(
    tmp_path: Path, *, adapters: AdapterRegistries | None = None, model_value: float = 1.5
) -> World:
    ws = tmp_path / "ws"
    (ws / "models").mkdir(parents=True)
    (ws / "data").mkdir()
    (ws / "models" / "m.json").write_text(json.dumps({"value": model_value}))
    (ws / "data" / "d.json").write_text(json.dumps(ROWS))
    m = ConstantModelAdapter.load(
        ws / "models" / "m.json", version="1", device=DeviceKind.CPU, options={}
    )
    d = ListDatasetAdapter.load(ws / "data" / "d.json", version="1", options={})
    rec_m = RegisteredModel("const", m.metadata(), T0, "models/m.json")  # workspace-relative source
    rec_d = RegisteredDataset("rows", d.metadata(), T0, "data/d.json")
    reg = SqliteRegistry(ws / "registry.sqlite")
    inv, cfg = (
        Investigation("adapters", "Does the run record its inputs?", T0),
        ConfigurationRef({"k": 1}),
    )
    exp = Experiment(inv.id, "e", "h", rec_m.ref(), rec_d.ref(), cfg.id, T0)
    for e in (rec_m, rec_d, inv, cfg, exp):
        reg.add(e)
    reg.update_status(exp.with_status(ExperimentStatus.READY))
    store = LocalArtifactStore(ws / "experiments")
    executor = Executor(
        reg, store, source_root=tmp_path, adapters=adapters or registries(), inputs_root=ws
    )
    return World(reg, ws, rec_m, rec_d, reg.get(Experiment, exp.id), executor)


@pytest.fixture
def world(tmp_path: Path):  # type: ignore[no-untyped-def]
    w = build(tmp_path)
    yield w
    w.registry.close()


def test_end_to_end_run_records_what_was_actually_investigated(world: World) -> None:
    result = world.executor.execute(world.experiment.id, procedures.adapter_eval, seed=3)
    assert result.status is RunStatus.COMPLETED

    (prov,) = world.registry.find(Provenance, run_id=result.run.id)
    assert prov.inputs is not None
    assert prov.inputs.model is not None
    assert prov.inputs.dataset is not None
    assert prov.inputs.model.record_id == world.model.id
    assert prov.inputs.model.fingerprint == world.model.fingerprint
    assert (prov.inputs.model.adapter, prov.inputs.model.adapter_version) == ("constant", "1.0.0")
    assert prov.inputs.dataset.record_id == world.dataset.id
    assert prov.inputs.dataset.fingerprint == world.dataset.fingerprint
    assert prov.inputs.device is not None
    assert (prov.inputs.device.requested, prov.inputs.device.resolved) == (
        DeviceKind.CPU,
        DeviceKind.CPU,
    )
    load_seconds = prov.runtime["model_load_seconds"]
    assert isinstance(load_seconds, float)
    assert load_seconds >= 0

    obs = {o.name for o in world.registry.find(Observation, run_id=result.run.id)}
    assert {"batch_seconds", "n_predictions"} <= obs
    assert len(result.artifacts) == 1
    assert world.registry.get(RegisteredModel, world.model.id) == world.model


def test_inputs_participate_in_the_fingerprint_and_replay_preserves_them(world: World) -> None:
    first = world.executor.execute(world.experiment.id, procedures.adapter_eval, seed=1)
    replay = world.executor.replay(first.run.id)
    assert replay.provenance.inputs == first.provenance.inputs
    assert replay.provenance.fingerprint == first.provenance.fingerprint
    assert "inputs" in first.provenance.components()
    without = Provenance.from_dict({**first.provenance.to_dict(), "inputs": None})
    assert without.fingerprint != first.provenance.fingerprint
    assert compare_provenance(first.provenance, without) == ("inputs",)


def test_experiments_without_adapters_are_unaffected(tmp_path: Path) -> None:
    w = build(tmp_path)
    plain = Executor(w.registry, LocalArtifactStore(w.workspace / "x"), source_root=tmp_path)
    result = plain.execute(w.experiment.id, procedures.nothing, seed=0)
    assert result.provenance.inputs is None
    assert "inputs" not in result.provenance.components()
    w.registry.close()


def test_model_changed_after_registration_is_detected_before_running(world: World) -> None:
    (world.workspace / "models" / "m.json").write_text(json.dumps({"value": 99}))
    with pytest.raises(PreparationError, match="changed since registration"):
        world.executor.execute(world.experiment.id, procedures.adapter_eval, seed=0)
    (run,) = world.registry.find(Run)
    assert run.status is RunStatus.FAILED  # visible, never RUNNING
    (outcome,) = world.registry.find(RunOutcome, run_id=run.id)
    assert outcome.error is not None
    assert outcome.error.stage is FailureStage.PREPARATION
    assert outcome.error.error_type.endswith("ModelFingerprintError")
    assert world.registry.find(Provenance, run_id=run.id) == []


def test_dataset_changed_after_registration_is_detected(world: World) -> None:
    changed = {**ROWS, "targets": [*range(9), 42]}
    (world.workspace / "data" / "d.json").write_text(json.dumps(changed))
    with pytest.raises(PreparationError, match="changed since registration"):
        world.executor.execute(world.experiment.id, procedures.adapter_eval, seed=0)
    (run,) = world.registry.find(Run)
    (outcome,) = world.registry.find(RunOutcome, run_id=run.id)
    assert outcome.error is not None
    assert outcome.error.error_type.endswith("DatasetFingerprintError")


def test_refs_without_a_registered_record_run_unbound_and_visibly_so(world: World) -> None:
    from dataclasses import replace

    from experionyx.domain import ModelRef

    inv = world.registry.find(Investigation)[0]
    cfg = world.registry.find(ConfigurationRef)[0]
    ghost = replace(
        world.experiment, name="ghost", model=ModelRef("ghost", "1", "sha256:" + "5" * 64)
    )
    ghost = Experiment(inv.id, "ghost", "h", ghost.model, world.dataset.ref(), cfg.id, T0)
    world.registry.add(ghost)
    world.registry.update_status(ghost.with_status(ExperimentStatus.READY))
    result = world.executor.execute(ghost.id, procedures.nothing, seed=0)
    assert result.status is RunStatus.COMPLETED
    assert result.provenance.inputs is not None  # the dataset *is* registered and bound ...
    assert result.provenance.inputs.model is None  # ... the model is not, and that is recorded


def test_missing_adapter_and_stale_adapter_version(tmp_path: Path) -> None:
    w = build(tmp_path)
    # adapter not installed/registered
    empty = AdapterRegistries(AdapterRegistry("model"), registries().datasets)
    ex = Executor(
        w.registry,
        LocalArtifactStore(w.workspace / "x"),
        source_root=tmp_path,
        adapters=empty,
        inputs_root=w.workspace,
    )
    with pytest.raises(PreparationError, match="cannot bind inputs"):
        ex.execute(w.experiment.id, procedures.nothing, seed=0)
    assert w.registry.find(Run)[0].status is RunStatus.FAILED
    w.registry.close()

    # record registered with an adapter version that is no longer installed
    w2 = build(tmp_path / "second")
    stale_meta = ConstantModelAdapter.load(
        w2.workspace / "models" / "m.json", version="2", device=DeviceKind.CPU, options={}
    ).metadata()
    import dataclasses

    stale = RegisteredModel(
        "old", dataclasses.replace(stale_meta, adapter_version="0.1.0"), T0, "models/m.json"
    )
    inv = w2.registry.find(Investigation)[0]
    cfg = w2.registry.find(ConfigurationRef)[0]
    exp = Experiment(inv.id, "old", "h", stale.ref(), w2.dataset.ref(), cfg.id, T0)
    w2.registry.add(stale)
    w2.registry.add(exp)
    w2.registry.update_status(exp.with_status(ExperimentStatus.READY))
    with pytest.raises(PreparationError, match="re-register"):
        w2.executor.execute(exp.id, procedures.nothing, seed=0)
    w2.registry.close()


def test_unknown_adapter_name_error_is_typed() -> None:
    with pytest.raises(AdapterNotFoundError):
        registries().models.resolve("nope")


# --- persistence of model/dataset identity ----------------------------------------------------


def test_model_and_dataset_identity_persist_and_are_retrievable(tmp_path: Path) -> None:
    w = build(tmp_path)
    w.registry.close()
    reg = SqliteRegistry(w.workspace / "registry.sqlite")
    (m,) = reg.find(RegisteredModel, name="const", version="1")
    assert m == w.model
    assert m.fingerprint == w.model.metadata.fingerprint
    assert reg.find(RegisteredModel, fingerprint=w.model.fingerprint) == [m]
    assert reg.find(RegisteredDataset, adapter="list") == [w.dataset]
    assert m.ref().digest == m.fingerprint
    reg.close()


def test_registration_is_immutable_and_versions_cannot_silently_change_content(
    world: World,
) -> None:
    with pytest.raises(DuplicateError):
        world.registry.add(world.model)
    other = ConstantModelAdapter.load(
        _write(world.workspace / "models" / "n.json", 7),
        version="1",
        device=DeviceKind.CPU,
        options={},
    )
    clash = RegisteredModel("const", other.metadata(), T0, "models/n.json")
    with pytest.raises(ValidationError, match="different fingerprint"):
        world.registry.add(clash)
    newer = ConstantModelAdapter.load(
        world.workspace / "models" / "n.json", version="2", device=DeviceKind.CPU, options={}
    )
    world.registry.add(
        RegisteredModel("const", newer.metadata(), T0, "models/n.json")
    )  # new version: fine


def _write(path: Path, value: float) -> Path:
    path.write_text(json.dumps({"value": value}))
    return path


def test_provenance_inputs_must_match_the_experiments_registered_refs(world: World) -> None:
    from dataclasses import replace

    from experionyx.provenance import (
        AdapterInputs,
        ExecutionParameters,
        InputBinding,
        SourceRevision,
        SourceState,
    )

    env_run = world.executor.execute(world.experiment.id, procedures.nothing, seed=0)
    base = env_run.provenance
    pending = Run(base_run_experiment(base), base.environment_id, 77, T0)
    world.registry.add(pending)
    bad = InputBinding(world.model.id, "constant", "1.0.0", "sha256:" + "0" * 64)
    with pytest.raises(ValidationError, match="does not match"):
        world.registry.add(
            replace(
                base,
                run_id=pending.id,
                seed=77,
                inputs=AdapterInputs(bad, None, None),
                source=SourceRevision(SourceState.UNKNOWN),
                execution=ExecutionParameters("m:f"),
            )
        )


def base_run_experiment(prov: Provenance) -> str:
    return prov.experiment_id

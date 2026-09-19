"""Helpers for evaluation-engine tests: a real workspace, registry and executor wired to the
pure-Python reference adapters (no ML framework needed)."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from experionyx.adapters.capabilities import DeviceKind
from experionyx.adapters.records import RegisteredDataset, RegisteredModel
from experionyx.adapters.registry import AdapterRegistries, AdapterRegistry
from experionyx.artifacts import LocalArtifactStore
from experionyx.domain import ConfigurationRef, Experiment, ExperimentStatus, Investigation, Run
from experionyx.evaluation.config import EvaluationConfig
from experionyx.evaluation.engine import PROCEDURE
from experionyx.evaluation.results import EvaluationResult
from experionyx.evaluation.serial import from_jsonable
from experionyx.execution import ExecutionResult, Executor, resolve_procedure
from experionyx.sqlite import SqliteRegistry
from pure_adapters import ConstantModelAdapter, ListDatasetAdapter, ThresholdClassifierAdapter

NOW = datetime(2026, 9, 19, 12, tzinfo=UTC)


def pure_registries() -> AdapterRegistries:
    models: AdapterRegistry = AdapterRegistry("model")  # type: ignore[type-arg]
    datasets: AdapterRegistry = AdapterRegistry("dataset")  # type: ignore[type-arg]
    models.register("constant", ConstantModelAdapter)
    models.register("threshold", ThresholdClassifierAdapter)
    datasets.register("list", ListDatasetAdapter)
    return AdapterRegistries(models, datasets)


@dataclass
class EvalWorld:
    registry: SqliteRegistry
    workspace: Path
    executor: Executor
    experiment: Experiment
    store: LocalArtifactStore

    def run(self, seed: int = 0) -> ExecutionResult:
        return self.executor.execute(
            self.experiment.id, resolve_procedure(PROCEDURE), seed=seed, procedure_name=PROCEDURE
        )


def eval_world(
    tmp_path: Path,
    *,
    model: Mapping[str, object],
    data: Mapping[str, object],
    config: EvaluationConfig,
    classifier: bool = True,
    parameters: Mapping[str, object] | None = None,
) -> EvalWorld:
    ws = tmp_path / "ws"
    (ws / "m").mkdir(parents=True)
    (ws / "m" / "model.json").write_text(json.dumps(model), encoding="utf-8")
    (ws / "m" / "data.json").write_text(json.dumps(data), encoding="utf-8")
    model_cls = ThresholdClassifierAdapter if classifier else ConstantModelAdapter
    m = model_cls.load(ws / "m" / "model.json", version="1", device=DeviceKind.CPU, options={})
    d = ListDatasetAdapter.load(ws / "m" / "data.json", version="1", options={})
    rec_m = RegisteredModel("model", m.metadata(), NOW, "m/model.json")
    rec_d = RegisteredDataset("data", d.metadata(), NOW, "m/data.json")
    reg = SqliteRegistry(ws / "registry.sqlite")
    inv = Investigation("eval-tests", "Does the evaluation engine record what it measures?", NOW)
    cfg = ConfigurationRef(dict(parameters) if parameters is not None else config.to_parameters())
    exp = Experiment(inv.id, "eval", "baseline", rec_m.ref(), rec_d.ref(), cfg.id, NOW)
    for e in (rec_m, rec_d, inv, cfg, exp):
        reg.add(e)
    reg.update_status(exp.with_status(ExperimentStatus.READY))
    store = LocalArtifactStore(ws / "experiments")
    executor = Executor(
        reg, store, source_root=tmp_path, adapters=pure_registries(), inputs_root=ws
    )
    return EvalWorld(reg, ws, executor, reg.get(Experiment, exp.id), store)


def class_data(n: int = 40, *, boundary: float = 5.0, noise_every: int = 0) -> dict[str, object]:
    """Binary data: feature 0 counts 0..9 repeated; target 1 iff feature0 > boundary (optionally
    with every `noise_every`-th target flipped, to create errors)."""
    rows = [[float(i % 10), float(i)] for i in range(n)]
    targets = [1 if r[0] > boundary else 0 for r in rows]
    if noise_every:
        targets = [1 - t if (i + 1) % noise_every == 0 else t for i, t in enumerate(targets)]
    return {
        "rows": rows,
        "targets": targets,
        "task": "CLASSIFICATION",
        "feature_names": ["x0", "index"],
        "splits": {"test": list(range(n // 2, n))},
    }


def load_evaluation(store: LocalArtifactStore, run: Run) -> EvaluationResult:
    """Read and parse the stored evaluation.json of a run (strict deserialization)."""
    path = store.run_dir(run) / "artifacts" / "evaluation" / "evaluation.json"
    result: EvaluationResult = from_jsonable(EvaluationResult, json.loads(path.read_text()))
    return result

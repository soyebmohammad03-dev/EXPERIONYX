"""Small, real evaluation experiments (`experionyx demo <name>`, and `examples/`).

Each demo trains a tiny model on tiny data inside the workspace, registers the model and dataset,
creates an evaluation experiment, and executes it through the normal execution engine using
the baseline evaluation engine (experionyx.evaluation). Every reported number
is measured during the run; nothing is precomputed. Requires the matching optional extra.
"""

from datetime import UTC, datetime
from pathlib import Path

from experionyx.adapters.base import DatasetAdapter, ModelAdapter
from experionyx.adapters.capabilities import DeviceKind
from experionyx.adapters.records import RegisteredDataset, RegisteredModel
from experionyx.adapters.registry import default_registries
from experionyx.artifacts import LocalArtifactStore
from experionyx.domain import ConfigurationRef, Entity, Experiment, ExperimentStatus, Investigation
from experionyx.errors import AdapterUnavailableError, ExperionyxError
from experionyx.evaluation.config import (
    EvaluationConfig,
    ScoreSource,
    SliceCondition,
    SliceKind,
    SliceSpec,
)
from experionyx.evaluation.engine import PROCEDURE
from experionyx.execution import ExecutionResult, Executor, resolve_procedure
from experionyx.sqlite import SqliteRegistry

DEMOS = ("sklearn-classification", "sklearn-regression", "torch-classification")


def _ensure(registry: SqliteRegistry, entity: Entity) -> None:
    if not registry.exists(type(entity), entity.id):
        registry.add(entity)


def _sklearn_config(regression: bool) -> EvaluationConfig:
    if regression:
        bmi = SliceCondition(SliceKind.FEATURE_RANGE, field="bmi", low=0.0)
        return EvaluationConfig(
            split="test", batch_size=16, slices=(SliceSpec("high-bmi", (bmi,)),)
        )
    wide = SliceCondition(SliceKind.FEATURE_RANGE, field="petal width (cm)", low=1.0)
    class0 = SliceCondition(SliceKind.TARGET_EQUALS, value=0)
    return EvaluationConfig(
        split="test",
        batch_size=16,
        slices=(SliceSpec("class-0", (class0,)), SliceSpec("wide-petals", (wide,))),
    )


# The torch demo model outputs logits, so probabilities are derived with softmax (an explicit,
# recorded assumption); slices use a column index because the tensors have no feature names.
_TORCH_CONFIG = EvaluationConfig(
    split="test",
    batch_size=16,
    score_source=ScoreSource.SOFTMAX_LOGITS,
    slices=(
        SliceSpec("x0-positive", (SliceCondition(SliceKind.FEATURE_RANGE, field="0", low=0.0),)),
    ),
)


def _register_and_run(
    workspace: Path,
    *,
    name: str,
    model: ModelAdapter,
    model_source: str,
    dataset: DatasetAdapter,
    dataset_source: str,
    dataset_options: dict[str, object],
    config: EvaluationConfig,
    seed: int,
) -> ExecutionResult:
    now = datetime.now(UTC)
    registry = SqliteRegistry(workspace / "registry.sqlite")
    try:
        rec_model = RegisteredModel(name, model.metadata(), now, model_source, {})
        rec_data = RegisteredDataset(name, dataset.metadata(), now, dataset_source, dataset_options)
        inv = Investigation(f"demo-{name}", "Does an adapter-backed run record what it used?", now)
        cfg = ConfigurationRef(config.to_parameters())
        exp = Experiment(
            inv.id, name, f"{name} runs end to end through the adapters",
            rec_model.ref(), rec_data.ref(), cfg.id, now,
        )  # fmt: skip
        for entity in (rec_model, rec_data, inv, cfg, exp):
            _ensure(registry, entity)
        if registry.get(Experiment, exp.id).status is ExperimentStatus.DRAFT:
            registry.update_status(exp.with_status(ExperimentStatus.READY))
        executor = Executor(
            registry,
            LocalArtifactStore(workspace / "experiments"),
            source_root=Path.cwd(),
            adapters=default_registries(),
            inputs_root=workspace,
            device=DeviceKind.CPU,
        )
        return executor.execute(
            exp.id, resolve_procedure(PROCEDURE), seed=seed, procedure_name=PROCEDURE
        )
    finally:
        registry.close()


def _sklearn_demo(workspace: Path, seed: int, *, regression: bool) -> ExecutionResult:
    try:
        from sklearn.linear_model import LogisticRegression, Ridge

        from experionyx.adapters.sklearn_adapter import SklearnDatasetAdapter, SklearnModelAdapter
    except ImportError as exc:
        raise AdapterUnavailableError(
            f"{exc}. Install with: pip install 'experionyx[sklearn]'"
        ) from exc
    name, builtin = ("diabetes-ridge", "diabetes") if regression else ("iris-logreg", "iris")
    options: dict[str, object] = {"seed": 0, "test_size": 0.25}
    dataset = SklearnDatasetAdapter.load(f"builtin:{builtin}", version="1", options=options)
    model_path = workspace / "models" / f"{name}.joblib"
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        (train,) = dataset.batches(10**6, "train")  # the whole (tiny) training split
        estimator = Ridge(alpha=1.0) if regression else LogisticRegression(max_iter=200)
        estimator.fit(train.inputs, train.target)
        SklearnModelAdapter.save(estimator, model_path)
    model = SklearnModelAdapter.load(model_path, version="1", device=DeviceKind.CPU, options={})
    return _register_and_run(
        workspace, name=name, model=model, model_source=f"models/{name}.joblib",
        dataset=dataset, dataset_source=f"builtin:{builtin}", dataset_options=options,
        config=_sklearn_config(regression), seed=seed,
    )  # fmt: skip


def _torch_demo(workspace: Path, seed: int) -> ExecutionResult:
    try:
        import torch

        from experionyx.adapters.capabilities import TaskType
        from experionyx.adapters.torch_adapter import TorchDatasetAdapter, TorchModelAdapter
    except ImportError as exc:
        raise AdapterUnavailableError(
            f"{exc}. Install with: pip install 'experionyx[torch]'"
        ) from exc
    name = "blobs-linear"
    data_path, model_path = (
        workspace / "datasets" / f"{name}.pt",
        workspace / "models" / f"{name}.pt",
    )
    if not data_path.exists():
        data_path.parent.mkdir(parents=True, exist_ok=True)
        gen = torch.Generator().manual_seed(0)
        x = torch.randn(200, 4, generator=gen)
        y = (x[:, 0] + x[:, 1] > 0).long()
        torch.save(
            {
                "X": x,
                "y": y,
                "split_train": torch.arange(0, 150),
                "split_test": torch.arange(150, 200),
            },
            data_path,
        )
    options: dict[str, object] = {"task": TaskType.CLASSIFICATION.value}
    dataset = TorchDatasetAdapter.load(data_path, version="1", options=options)
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(0)
        net = torch.nn.Linear(4, 2)
        (train,) = dataset.batches(10**6, "train")
        optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
        for _ in range(100):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(net(train.inputs), train.target)  # type: ignore[arg-type]
            loss.backward()
            optimizer.step()
        TorchModelAdapter.save(net, model_path, torch.zeros(1, 4))
    model = TorchModelAdapter.load(
        model_path, version="1", device=DeviceKind.CPU,
        options={"task": TaskType.CLASSIFICATION.value, "input_shape": [4]},
    )  # fmt: skip
    return _register_and_run(
        workspace, name=name, model=model, model_source=f"models/{name}.pt",
        dataset=dataset, dataset_source=f"datasets/{name}.pt", dataset_options=options,
        config=_TORCH_CONFIG, seed=seed,
    )  # fmt: skip


def run_demo(name: str, workspace: Path, seed: int = 0) -> ExecutionResult:
    workspace.mkdir(parents=True, exist_ok=True)
    if name == "sklearn-classification":
        return _sklearn_demo(workspace, seed, regression=False)
    if name == "sklearn-regression":
        return _sklearn_demo(workspace, seed, regression=True)
    if name == "torch-classification":
        return _torch_demo(workspace, seed)
    raise ExperionyxError(f"unknown demo {name!r} (choose from {list(DEMOS)})")

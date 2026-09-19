"""Real end-to-end runs: dataset + model through adapters, the execution engine, observations,
artifacts and provenance; plus the demos and the adapter CLI."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from conftest import make_repo
from experionyx.adapters.capabilities import DeviceKind, ModelCapability
from experionyx.adapters.records import RegisteredDataset, RegisteredModel
from experionyx.adapters.registry import default_registries
from experionyx.artifacts import LocalArtifactStore
from experionyx.cli import main
from experionyx.domain import (
    Artifact,
    ConfigurationRef,
    Experiment,
    ExperimentStatus,
    Investigation,
    Observation,
    RunStatus,
)
from experionyx.execution import Executor, RunContext
from experionyx.provenance import Provenance, SourceState
from experionyx.sqlite import SqliteRegistry

NOW = datetime(2026, 9, 19, 12, tzinfo=UTC)


def evaluate(ctx: RunContext) -> None:
    """Predict the whole test split; report accuracy (measured here) and store predictions."""
    assert ctx.model is not None
    assert ctx.dataset is not None
    preds: list[object] = []
    truth: list[object] = []
    for batch in ctx.dataset.batches(int(str(ctx.parameters["batch_size"])), "test"):
        result = ctx.model.predict(batch.inputs, sample_ids=batch.indices)
        preds.extend(result.outputs)
        truth.extend(batch.target.tolist())  # type: ignore[attr-defined]
    if preds and isinstance(preds[0], tuple):  # logits -> class
        preds = [max(range(len(p)), key=p.__getitem__) for p in preds]  # type: ignore[arg-type]
    ctx.observe(
        "accuracy",
        sum(p == t for p, t in zip(preds, truth, strict=True)) / len(preds),
        unit="ratio",
    )
    ctx.observe("n_test", len(preds))
    (ctx.artifact_dir / "predictions.json").write_text(json.dumps(preds))
    ctx.register_artifact("predictions.json")


def run_experiment(
    tmp_path: Path,
    model: object,
    dataset: object,
    *,
    model_source: str,
    dataset_source: str,
    dataset_options: dict[str, object],
) -> tuple[SqliteRegistry, Executor, Experiment, RegisteredModel, RegisteredDataset]:
    ws = tmp_path / "ws"
    reg = SqliteRegistry(ws / "registry.sqlite")
    rm = RegisteredModel("m", model.metadata(), NOW, model_source)  # type: ignore[attr-defined]
    rd = RegisteredDataset("d", dataset.metadata(), NOW, dataset_source, dataset_options)  # type: ignore[attr-defined]
    inv, cfg = (
        Investigation("e2e", "Does the whole chain hold together?", NOW),
        ConfigurationRef({"batch_size": 8}),
    )
    exp = Experiment(inv.id, "e2e", "h", rm.ref(), rd.ref(), cfg.id, NOW)
    for e in (rm, rd, inv, cfg, exp):
        reg.add(e)
    reg.update_status(exp.with_status(ExperimentStatus.READY))
    executor = Executor(
        reg,
        LocalArtifactStore(ws / "experiments"),
        source_root=make_repo(tmp_path / "repo"),
        adapters=default_registries(),
        inputs_root=ws,
    )
    return reg, executor, exp, rm, rd


def check_chain(
    tmp_path: Path,
    reg: SqliteRegistry,
    executor: Executor,
    exp: Experiment,
    rm: RegisteredModel,
    rd: RegisteredDataset,
    adapter: str,
) -> None:
    result = executor.execute(exp.id, evaluate, seed=5)
    assert result.status is RunStatus.COMPLETED, result.error
    (prov,) = reg.find(Provenance, run_id=result.run.id)
    assert prov.source.state is SourceState.REPRODUCIBLE_SOURCE
    assert prov.inputs is not None
    assert prov.inputs.model is not None
    assert prov.inputs.dataset is not None
    assert (prov.inputs.model.record_id, prov.inputs.model.adapter) == (rm.id, adapter)
    assert (prov.inputs.dataset.record_id, prov.inputs.dataset.adapter) == (rd.id, adapter)
    assert prov.inputs.model.fingerprint == rm.fingerprint
    assert prov.inputs.dataset.fingerprint == rd.fingerprint
    assert prov.inputs.device.resolved is DeviceKind.CPU  # type: ignore[union-attr]
    obs = {o.name: o.value for o in reg.find(Observation, run_id=result.run.id)}
    assert 0.0 <= obs["accuracy"] <= 1.0  # type: ignore[operator]
    assert obs["n_test"] > 0  # type: ignore[operator]
    (art,) = reg.find(Artifact, run_id=result.run.id)
    file = (
        LocalArtifactStore(tmp_path / "ws" / "experiments").run_dir(result.run)
        / "artifacts"
        / art.path
    )
    assert art.digest == "sha256:" + hashlib.sha256(file.read_bytes()).hexdigest()
    replay = executor.replay(result.run.id)
    assert replay.provenance.fingerprint == result.provenance.fingerprint
    assert {o.name: o.value for o in replay.observations}["accuracy"] == obs[
        "accuracy"
    ]  # deterministic here


def test_sklearn_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    from experionyx.adapters.sklearn_adapter import SklearnDatasetAdapter, SklearnModelAdapter

    opts: dict[str, object] = {"seed": 0, "test_size": 0.25}
    dataset = SklearnDatasetAdapter.load("builtin:iris", version="1", options=opts)
    (train,) = dataset.batches(1000, "train")
    est = LogisticRegression(max_iter=300).fit(train.inputs, train.target)
    (tmp_path / "ws" / "models").mkdir(parents=True)
    SklearnModelAdapter.save(est, tmp_path / "ws" / "models" / "m.joblib")
    model = SklearnModelAdapter.load(
        tmp_path / "ws" / "models" / "m.joblib", version="1", device=DeviceKind.CPU, options={}
    )
    assert ModelCapability.PREDICT_PROBA in model.capabilities
    reg, executor, exp, rm, rd = run_experiment(
        tmp_path,
        model,
        dataset,
        model_source="models/m.joblib",
        dataset_source="builtin:iris",
        dataset_options=opts,
    )
    check_chain(tmp_path, reg, executor, exp, rm, rd, "sklearn")
    reg.close()


def test_sklearn_model_changed_on_disk_is_refused(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    from experionyx.adapters.sklearn_adapter import SklearnDatasetAdapter, SklearnModelAdapter
    from experionyx.errors import PreparationError

    opts: dict[str, object] = {"seed": 0}
    dataset = SklearnDatasetAdapter.load("builtin:iris", version="1", options=opts)
    (train,) = dataset.batches(1000, "train")
    (tmp_path / "ws" / "models").mkdir(parents=True)
    path = tmp_path / "ws" / "models" / "m.joblib"
    SklearnModelAdapter.save(LogisticRegression(max_iter=300).fit(train.inputs, train.target), path)
    model = SklearnModelAdapter.load(path, version="1", device=DeviceKind.CPU, options={})
    reg, executor, exp, *_ = run_experiment(
        tmp_path,
        model,
        dataset,
        model_source="models/m.joblib",
        dataset_source="builtin:iris",
        dataset_options=opts,
    )
    SklearnModelAdapter.save(
        LogisticRegression(max_iter=300, C=0.001).fit(train.inputs, train.target), path
    )
    with pytest.raises(PreparationError, match="changed since registration"):
        executor.execute(exp.id, evaluate, seed=0)
    reg.close()


def test_torch_end_to_end(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from experionyx.adapters.capabilities import TaskType
    from experionyx.adapters.torch_adapter import TorchDatasetAdapter, TorchModelAdapter

    ws = tmp_path / "ws"
    (ws / "models").mkdir(parents=True)
    (ws / "datasets").mkdir()
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(120, 4, generator=gen)
    y = (x[:, 0] + x[:, 1] > 0).long()
    torch.save(
        {"X": x, "y": y, "split_train": torch.arange(0, 90), "split_test": torch.arange(90, 120)},
        ws / "datasets" / "d.pt",
    )
    opts: dict[str, object] = {"task": TaskType.CLASSIFICATION.value}
    dataset = TorchDatasetAdapter.load(ws / "datasets" / "d.pt", version="1", options=opts)
    torch.manual_seed(0)
    module = torch.nn.Linear(4, 2)
    optim = torch.optim.SGD(module.parameters(), lr=0.5)
    for _ in range(60):
        optim.zero_grad()
        torch.nn.functional.cross_entropy(module(x[:90]), y[:90]).backward()
        optim.step()
    TorchModelAdapter.save(module, ws / "models" / "m.pt", torch.zeros(1, 4))
    model = TorchModelAdapter.load(
        ws / "models" / "m.pt",
        version="1",
        device=DeviceKind.CPU,
        options={"task": "CLASSIFICATION"},
    )
    reg, executor, exp, rm, rd = run_experiment(
        tmp_path,
        model,
        dataset,
        model_source="models/m.pt",
        dataset_source="datasets/d.pt",
        dataset_options=opts,
    )
    check_chain(tmp_path, reg, executor, exp, rm, rd, "torch")
    reg.close()


# --- demos and CLI ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["sklearn-classification", "sklearn-regression", "torch-classification"]
)
def test_demos_run_for_real_and_are_repeatable(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("torch" if name.startswith("torch") else "sklearn")
    monkeypatch.chdir(tmp_path)
    assert main(["--workspace", str(tmp_path / "w"), "demo", name]) == 0
    first = capsys.readouterr().out
    assert "status: COMPLETED" in first
    assert (
        main(["--workspace", str(tmp_path / "w"), "demo", name]) == 0
    )  # second run: new run, same records
    reg = SqliteRegistry(tmp_path / "w" / "registry.sqlite")
    runs = reg.find(__import__("experionyx.domain", fromlist=["Run"]).Run)
    assert len(runs) == 2
    assert len({r.experiment_id for r in runs}) == 1
    assert len(reg.find(RegisteredModel)) == 1
    values = [{o.name: o.value for o in reg.find(Observation, run_id=r.id)} for r in runs]
    key = "metric.mae" if "regression" in name else "metric.accuracy"
    assert values[0][key] == values[1][key]
    reg.close()


def test_adapter_cli_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("sklearn")
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression

    from experionyx.adapters.sklearn_adapter import SklearnModelAdapter

    monkeypatch.chdir(tmp_path)
    ws = str(tmp_path / "w")
    assert main(["--workspace", ws, "init"]) == 0
    capsys.readouterr()

    assert main(["adapters", "list"]) == 0
    listing = capsys.readouterr().out
    assert "sklearn" in listing
    assert "PREDICT_PROBA" in listing
    assert "torch" in listing

    assert main(["adapters", "inspect", "sklearn"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["model"]["available"] is True
    assert info["model"]["capabilities"] == ["BATCH_PREDICT", "PREDICT", "PREDICT_PROBA"]
    assert set(info) == {"model", "dataset"}
    assert main(["adapters", "inspect", "nonexistent"]) == 2
    capsys.readouterr()

    data = load_iris()
    model_path = SklearnModelAdapter.save(
        LogisticRegression(max_iter=300).fit(data.data, data.target), tmp_path / "m.joblib"
    )
    assert main(["model", "inspect", str(model_path), "--adapter", "sklearn"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["metadata"]["task"] == "CLASSIFICATION"
    assert (
        inspected["metadata"]["fingerprint"]
        == "sha256:" + hashlib.sha256(model_path.read_bytes()).hexdigest()
    )
    assert inspected["device"] == {"requested": "CPU", "resolved": "CPU"}

    assert (
        main(
            [
                "--workspace",
                ws,
                "model",
                "register",
                "--name",
                "iris",
                "--source",
                str(model_path),
                "--adapter",
                "sklearn",
            ]
        )
        == 0
    )
    model_id = next(
        line.split()[-1]
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("registered model")
    )
    assert main(["--workspace", ws, "model", "inspect", model_id]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == model_id
    assert (
        main(
            [
                "--workspace",
                ws,
                "model",
                "register",
                "--name",
                "iris",
                "--source",
                str(model_path),
                "--adapter",
                "sklearn",
            ]
        )
        == 2
    )  # duplicate
    capsys.readouterr()

    assert (
        main(
            [
                "dataset",
                "inspect",
                "builtin:iris",
                "--adapter",
                "sklearn",
                "--deep",
                "--option",
                "seed=2",
            ]
        )
        == 0
    )
    meta = json.loads(capsys.readouterr().out)
    assert meta["num_samples"] == 150
    assert meta["missing_values"] == 0
    assert meta["source"]["seed"] == 2
    assert (
        main(
            [
                "--workspace",
                ws,
                "dataset",
                "register",
                "--name",
                "iris",
                "--source",
                "builtin:iris",
                "--adapter",
                "sklearn",
            ]
        )
        == 0
    )
    assert "registered dataset dst_" in capsys.readouterr().out
    assert main(["model", "inspect", str(model_path)]) == 2  # --adapter required for files
    assert (
        main(["model", "inspect", str(model_path), "--adapter", "sklearn", "--device", "MPS"]) == 2
    )  # unavailable device, no fallback
    capsys.readouterr()

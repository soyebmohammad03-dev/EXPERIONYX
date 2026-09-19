"""Baseline evaluation with real scikit-learn and PyTorch models, verified against the
frameworks' own reference implementations; plus the evaluation CLI."""

import json
import math
from pathlib import Path

import pytest

from experionyx.cli import main
from experionyx.demos import run_demo
from experionyx.domain import Artifact, Claim, Evidence, Observation, Run, RunStatus
from experionyx.evaluation.results import EvaluationResult, Status
from experionyx.evaluation.serial import from_jsonable
from experionyx.sqlite import SqliteRegistry


def load_ev(ws: Path, run) -> EvaluationResult:  # type: ignore[no-untyped-def]
    path = (
        ws
        / "experiments"
        / run.experiment_id
        / "runs"
        / run.id
        / "artifacts"
        / "evaluation"
        / "evaluation.json"
    )
    return from_jsonable(EvaluationResult, json.loads(path.read_text()))  # type: ignore[no-any-return]


def ece(conf: list[float], hit: list[bool], bins: int = 10) -> float:
    total = 0.0
    for k in range(bins):
        idx = [i for i, c in enumerate(conf) if min(int(c * bins), bins - 1) == k]
        if idx:
            total += (
                len(idx)
                / len(conf)
                * abs(sum(hit[i] for i in idx) / len(idx) - sum(conf[i] for i in idx) / len(idx))
            )
    return total


def check_integrity(ws: Path, run_id: str) -> None:
    """Every stored artifact matches its digest; every claim's evidence resolves in this run."""
    reg = SqliteRegistry(ws / "registry.sqlite")
    try:
        run = reg.get(Run, run_id)
        arts = reg.find(Artifact, run_id=run_id)
        assert len(arts) >= 10
        import hashlib

        for a in arts:
            f = ws / "experiments" / run.experiment_id / "runs" / run_id / "artifacts" / a.path
            assert a.digest == "sha256:" + hashlib.sha256(f.read_bytes()).hexdigest()
        ids = {a.id for a in arts} | {o.id for o in reg.find(Observation, run_id=run_id)} | {run_id}
        for e in reg.find(Evidence):
            assert e.target_id in ids or reg.find(
                Claim, investigation_id=reg.get(Claim, e.claim_id).investigation_id
            )
    finally:
        reg.close()


# --- A: scikit-learn classification -----------------------------------------------------------


def test_sklearn_classification_baseline_matches_sklearn_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("sklearn")
    import joblib
    import numpy as np
    from sklearn import metrics as sk
    from sklearn.datasets import load_iris

    monkeypatch.chdir(tmp_path)
    ws = tmp_path / "w"
    result = run_demo("sklearn-classification", ws)
    assert result.status is RunStatus.COMPLETED, result.error
    ev = load_ev(ws, result.run)

    data = load_iris()
    order = np.random.RandomState(0).permutation(150)
    test = order[150 - round(150 * 0.25) :]
    model = joblib.load(ws / "models" / "iris-logreg.joblib")
    x, y = data.data[test], data.target[test]
    pred, proba = model.predict(x), model.predict_proba(x)
    m = {r.metric_id: r for r in ev.metrics}
    assert ev.n_samples == len(test) == 38
    assert m["accuracy"].value == pytest.approx(sk.accuracy_score(y, pred), abs=1e-12)
    assert m["balanced_accuracy"].value == pytest.approx(
        sk.balanced_accuracy_score(y, pred), abs=1e-12
    )
    assert m["f1"].value == pytest.approx(sk.f1_score(y, pred, average="macro"), abs=1e-12)
    assert m["precision"].value == pytest.approx(
        sk.precision_score(y, pred, average="macro"), abs=1e-12
    )
    assert m["recall"].value == pytest.approx(sk.recall_score(y, pred, average="macro"), abs=1e-12)
    assert m["roc_auc"].value == pytest.approx(
        sk.roc_auc_score(y, proba, multi_class="ovr"), abs=1e-12
    )
    assert m["log_loss"].value == pytest.approx(sk.log_loss(y, proba), abs=1e-12)
    assert ev.confusion is not None
    assert np.array(ev.confusion.counts).tolist() == sk.confusion_matrix(y, pred).tolist()
    assert ev.calibration.ece == pytest.approx(
        ece(proba.max(axis=1).tolist(), (pred == y).tolist()), abs=1e-12
    )
    brier = float(np.mean(np.sum((proba - np.eye(3)[y]) ** 2, axis=1)))
    assert ev.calibration.brier_score == pytest.approx(brier, abs=1e-12)
    assert {s.name for s in ev.slices} == {"class-0", "wide-petals"}
    class0 = next(s for s in ev.slices if s.name == "class-0")
    assert class0.n_samples == int((y == 0).sum())
    assert m["accuracy"].interval is not None
    assert m["accuracy"].interval.status is Status.COMPUTED
    assert ev.errors.counts["INCORRECT_CLASS"] == int((pred != y).sum())
    assert ev.latency.n_batches == 3
    assert ev.latency.load_seconds is not None
    check_integrity(ws, result.run.id)


# --- B: scikit-learn regression ---------------------------------------------------------------


def test_sklearn_regression_baseline_matches_sklearn_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("sklearn")
    import joblib
    import numpy as np
    from sklearn import metrics as sk
    from sklearn.datasets import load_diabetes

    monkeypatch.chdir(tmp_path)
    ws = tmp_path / "w"
    result = run_demo("sklearn-regression", ws)
    assert result.status is RunStatus.COMPLETED, result.error
    ev = load_ev(ws, result.run)
    data = load_diabetes()
    order = np.random.RandomState(0).permutation(len(data.data))
    test = order[len(data.data) - round(len(data.data) * 0.25) :]
    pred = joblib.load(ws / "models" / "diabetes-ridge.joblib").predict(data.data[test])
    y = data.target[test]
    m = {r.metric_id: r for r in ev.metrics}
    assert m["mae"].value == pytest.approx(sk.mean_absolute_error(y, pred), rel=1e-12)
    assert m["mse"].value == pytest.approx(sk.mean_squared_error(y, pred), rel=1e-12)
    assert m["rmse"].value == pytest.approx(math.sqrt(sk.mean_squared_error(y, pred)), rel=1e-12)
    assert m["r2"].value == pytest.approx(sk.r2_score(y, pred), rel=1e-9)
    assert m["median_absolute_error"].value == pytest.approx(
        sk.median_absolute_error(y, pred), rel=1e-12
    )
    assert ev.confusion is None
    assert ev.calibration.status is Status.UNSUPPORTED
    assert ev.errors.counts == {"RESIDUAL": len(test)}
    assert ev.errors.stats["mean_absolute_error"] == pytest.approx(m["mae"].value)
    assert m["rmse"].interval is not None
    assert m["rmse"].interval.lower <= m["rmse"].value <= m["rmse"].interval.upper  # type: ignore[operator]
    high_bmi = ev.slices[0]
    assert high_bmi.name == "high-bmi"
    assert high_bmi.n_samples == int((data.data[test][:, 2] >= 0.0).sum())
    check_integrity(ws, result.run.id)


# --- C: PyTorch classification ----------------------------------------------------------------


def test_torch_classification_baseline_with_explicit_softmax_assumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    import warnings

    monkeypatch.chdir(tmp_path)
    ws = tmp_path / "w"
    result = run_demo("torch-classification", ws)
    assert result.status is RunStatus.COMPLETED, result.error
    ev = load_ev(ws, result.run)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = torch.jit.load(str(ws / "models" / "blobs-linear.pt"))
    data = torch.load(ws / "datasets" / "blobs-linear.pt", weights_only=True)
    test = data["split_test"]
    with torch.no_grad():
        logits = net(data["X"][test])
    y = data["y"][test]
    m = {r.metric_id: r for r in ev.metrics}
    assert m["accuracy"].value == pytest.approx(
        float((logits.argmax(1) == y).float().mean()), abs=1e-12
    )
    probs = torch.softmax(logits, dim=1)
    assert m["log_loss"].value == pytest.approx(
        float(-torch.log(probs[torch.arange(len(y)), y]).mean()), abs=1e-5
    )
    assert ev.calibration.status is Status.COMPUTED
    assert "ASSUMING the outputs are logits" in (ev.calibration.confidence_source or "")
    assert (
        ev.confidence.source == ev.calibration.confidence_source
    )  # the assumption is always recorded
    assert ev.profile is not None
    assert ev.profile.model_adapter == "torch"
    assert ev.profile.parameter_count == 4 * 2 + 2
    assert ev.slices[0].name == "x0-positive"
    assert result.provenance.inputs is not None
    assert result.provenance.inputs.device is not None
    assert result.provenance.inputs.device.resolved.value == "CPU"
    check_integrity(ws, result.run.id)


# --- CLI --------------------------------------------------------------------------------------


@pytest.fixture
def demo_ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    pytest.importorskip("sklearn")
    monkeypatch.chdir(tmp_path)
    ws = tmp_path / "w"
    assert main(["--workspace", str(ws), "demo", "sklearn-classification"]) == 0
    return ws


def ids(ws: Path, table: str) -> str:
    import sqlite3

    with sqlite3.connect(ws / "registry.sqlite") as c:
        return str(c.execute(f"SELECT id FROM {table} LIMIT 1").fetchone()[0])  # noqa: S608


def test_cli_metrics_list_and_filter(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["metrics", "list"]) == 0
    out = capsys.readouterr().out
    assert "roc_auc" in out
    assert "requires=LABELS,SCORES" in out
    assert "mae" in out
    assert main(["metrics", "list", "--task", "REGRESSION"]) == 0
    reg = capsys.readouterr().out
    assert "mae" in reg
    assert "accuracy" not in reg


def test_cli_evaluate_inspect_compare_and_autopsy(
    demo_ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = str(demo_ws)
    mid, did = ids(demo_ws, "models"), ids(demo_ws, "datasets")
    capsys.readouterr()
    assert (
        main(
            [
                "--workspace",
                ws,
                "evaluate",
                "--model",
                mid,
                "--dataset",
                did,
                "--split",
                "test",
                "--metric",
                "accuracy",
                "--metric",
                "log_loss",
                "--bins",
                "5",
                "--bootstrap-resamples",
                "50",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    run_a = next(x.split(": ")[1] for x in out.splitlines() if x.startswith("run: "))
    assert "status: COMPLETED" in out

    assert main(["--workspace", ws, "evaluation", "inspect", run_a]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert set(summary["metrics"]) == {"accuracy", "log_loss"}
    assert summary["n_samples"] == 38
    assert summary["profile"]["ece"] is not None
    assert main(["--workspace", ws, "evaluation", "inspect", run_a, "--full"]) == 0
    full = json.loads(capsys.readouterr().out)
    assert full["calibration"]["n_bins"] == 5
    assert full["metrics"][0]["interval"]["resamples"] == 50

    assert (
        main(
            [
                "--workspace",
                ws,
                "evaluate",
                "--model",
                mid,
                "--dataset",
                did,
                "--split",
                "test",
                "--metric",
                "accuracy",
                "--score-source",
                "NONE",
                "--bootstrap-resamples",
                "0",
            ]
        )
        == 0
    )
    run_b = next(
        x.split(": ")[1] for x in capsys.readouterr().out.splitlines() if x.startswith("run: ")
    )
    assert main(["--workspace", ws, "evaluation", "compare", run_a, run_b]) == 0
    cmp = json.loads(capsys.readouterr().out)
    assert cmp["same_model"]
    assert cmp["same_dataset"]
    assert not cmp["same_config"]
    assert [m["metric_id"] for m in cmp["metrics"]] == ["accuracy", "log_loss"]
    assert cmp["metrics"][1]["b"] is None  # log_loss was not evaluated in B: absent, not zero
    assert not any(k in cmp for k in ("winner", "score", "best"))

    assert (
        main(["--workspace", ws, "model", "autopsy", mid, "--dataset", did, "--split", "test"]) == 0
    )
    autopsy = json.loads(capsys.readouterr().out)
    assert autopsy["profile"]["task"] == "CLASSIFICATION"
    assert "metrics" in autopsy["profile"]
    assert isinstance(autopsy["findings"], list)


def test_cli_evaluate_with_config_file_is_strict_and_reports_failures(
    demo_ws: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = str(demo_ws)
    mid, did = ids(demo_ws, "models"), ids(demo_ws, "datasets")
    capsys.readouterr()
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"split": "test", "batch_szie": 4}))  # a typo must not be ignored
    assert (
        main(
            ["--workspace", ws, "evaluate", "--model", mid, "--dataset", did, "--config", str(cfg)]
        )
        == 2
    )
    assert "unknown field" in capsys.readouterr().err
    assert (
        main(["--workspace", ws, "evaluate", "--model", mid, "--dataset", did, "--metric", "mae"])
        == 1
    )  # invalid for the task
    assert "status: FAILED" in capsys.readouterr().out


def test_cli_inspect_detects_a_tampered_evaluation_artifact(
    demo_ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = str(demo_ws)
    reg = SqliteRegistry(demo_ws / "registry.sqlite")
    run = reg.find(Run)[0]
    reg.close()
    file = (
        demo_ws
        / "experiments"
        / run.experiment_id
        / "runs"
        / run.id
        / "artifacts"
        / "evaluation"
        / "evaluation.json"
    )
    capsys.readouterr()
    file.write_text(file.read_text().replace('"n_samples": 38', '"n_samples": 39', 1))
    assert main(["--workspace", ws, "evaluation", "inspect", run.id]) == 2
    assert "changed since registration" in capsys.readouterr().err

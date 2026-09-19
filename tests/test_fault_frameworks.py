"""Fault experiments with real scikit-learn and PyTorch models, and the fault CLI."""

import json
import sqlite3
from pathlib import Path

import pytest

from experionyx.adapters.records import RegisteredDataset, RegisteredModel
from experionyx.artifacts import LocalArtifactStore
from experionyx.cli import main
from experionyx.domain import Run
from experionyx.faults.demos import run_fault_demo
from experionyx.faults.entities import FaultExperiment, TrialStatus
from experionyx.faults.report import load_analysis
from experionyx.sqlite import SqliteRegistry


def ids(ws: Path, table: str) -> str:
    with sqlite3.connect(ws / "registry.sqlite") as c:
        return str(c.execute(f"SELECT id FROM {table} LIMIT 1").fetchone()[0])  # noqa: S608


@pytest.fixture
def demo_ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    pytest.importorskip("sklearn")
    monkeypatch.chdir(tmp_path)
    ws = tmp_path / "w"
    assert main(["--workspace", str(ws), "demo", "sklearn-classification"]) == 0
    return ws


def out_run(text: str, key: str) -> str:
    return next(
        line.split(": ")[1].split()[0] for line in text.splitlines() if line.startswith(key)
    )


# --- CLI --------------------------------------------------------------------------------------


def test_cli_lists_and_inspects_faults(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["faults"]) == 0
    listing = capsys.readouterr().out
    assert "gaussian_noise" in listing
    assert "label_flip" in listing
    assert "NOT IMPLEMENTED" in listing  # reserved future capabilities are labelled honestly
    assert main(["fault", "inspect", "feature_dropout"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["category"] == "MISSINGNESS"
    assert [p["name"] for p in info["parameters"]] == ["probability", "value"]
    assert info["parameters"][0]["required"] is True
    assert info["parameters"][1]["default"] == 0.0
    assert "probability" in info["sweepable"]
    assert main(["fault", "inspect", "nope"]) == 2
    assert "no fault named" in capsys.readouterr().err


def test_cli_run_sweep_compare_and_inspect_on_real_data(
    demo_ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = str(demo_ws)
    mid, did = ids(demo_ws, "models"), ids(demo_ws, "datasets")
    capsys.readouterr()
    base = ["--workspace", ws, "fault"]
    assert (
        main(
            [
                *base,
                "run",
                "--model",
                mid,
                "--dataset",
                did,
                "--split",
                "test",
                "--type",
                "gaussian_noise",
                "--param",
                "sigma=1.0",
                "--seed",
                "3",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    fx = out_run(out, "fault experiment:")
    baseline = out_run(out, "baseline run:")
    assert "trials:           {'COMPLETED': 1}" in out
    assert "accuracy (HIGHER_IS_BETTER)" in out

    assert (
        main(
            [
                *base,
                "sweep",
                "--model",
                mid,
                "--dataset",
                did,
                "--split",
                "test",
                "--type",
                "feature_dropout",
                "--param",
                "probability=0.0",
                "--sweep",
                "probability=0,0.5,1.0",
                "--seeds",
                "1,2",
                "--baseline-run",
                baseline,
            ]
        )
        == 0
    )
    sweep_out = capsys.readouterr().out
    assert "trials:           {'COMPLETED': 6}" in sweep_out
    assert out_run(sweep_out, "baseline run:") == baseline  # the control was reused
    fx2 = out_run(sweep_out, "fault experiment:")
    assert "probability=1" in sweep_out

    assert main([*base, "experiment", "inspect", fx2]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert len(doc["trials"]) == 6
    assert {t["status"] for t in doc["trials"]} == {"COMPLETED"}
    assert [r["value"] for r in doc["summary"]] == [0.0, 0.5, 1.0]
    assert doc["fault_experiment"]["status"] == "COMPLETED"
    assert doc["primary_metric"] == "accuracy"
    treat = doc["trials"][-1]["treatment_run"]

    assert main([*base, "compare", baseline, treat]) == 0
    cmp = json.loads(capsys.readouterr().out)
    acc = next(d for d in cmp["degradation"] if d["metric_id"] == "accuracy")
    assert acc["direction"] == "HIGHER_IS_BETTER"
    assert acc["baseline"] == pytest.approx(0.9473684210526315)
    assert acc["absolute_delta"] == pytest.approx(acc["faulted"] - acc["baseline"])
    assert acc["deterioration"] == pytest.approx(-acc["absolute_delta"])
    assert cmp["fault"]["total_samples"] == 38
    assert "not a verdict" in cmp["note"]
    reg = SqliteRegistry(demo_ws / "registry.sqlite")
    assert reg.get(FaultExperiment, fx).id == fx
    reg.close()


def test_cli_rejects_invalid_and_incompatible_faults_before_running(
    demo_ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = str(demo_ws)
    mid, did = ids(demo_ws, "models"), ids(demo_ws, "datasets")
    reg = SqliteRegistry(demo_ws / "registry.sqlite")
    runs_before = len(reg.find(Run))
    reg.close()
    capsys.readouterr()
    common = [
        "--workspace",
        ws,
        "fault",
        "run",
        "--model",
        mid,
        "--dataset",
        did,
        "--split",
        "test",
    ]
    assert main([*common, "--type", "gaussian_noise", "--param", "sigma=-1"]) == 2
    assert "sigma" in capsys.readouterr().err
    assert (
        main([*common, "--type", "brightness", "--param", "delta=0.1"]) == 2
    )  # tabular data, image fault
    assert "IMAGE" in capsys.readouterr().err
    assert main([*common, "--type", "temporal_delay"]) == 2
    assert main([*common]) == 2  # no fault given
    assert main([*common, "--type", "nope"]) == 2
    assert (
        main(
            [
                *common,
                "--type",
                "gaussian_noise",
                "--param",
                "sigma=1",
                "--scope-fraction",
                "0.5",
                "--scope-class",
                "1",
            ]
        )
        == 2
    )
    assert (
        main(
            [
                "--workspace",
                ws,
                "fault",
                "sweep",
                "--model",
                mid,
                "--dataset",
                did,
                "--type",
                "gaussian_noise",
                "--param",
                "sigma=0",
                "--sweep",
                "sigma=0,x",
            ]
        )
        == 2
    )
    reg = SqliteRegistry(demo_ws / "registry.sqlite")
    assert len(reg.find(Run)) == runs_before  # none of the refused requests ran anything
    reg.close()


def test_cli_scope_class_and_spec_file_for_compound_faults(
    demo_ws: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from experionyx.faults.library import default_fault_registry

    ws = str(demo_ws)
    mid, did = ids(demo_ws, "models"), ids(demo_ws, "datasets")
    capsys.readouterr()
    assert (
        main(
            [
                "--workspace",
                ws,
                "fault",
                "run",
                "--model",
                mid,
                "--dataset",
                did,
                "--split",
                "test",
                "--type",
                "gaussian_noise",
                "--param",
                "sigma=1.0",
                "--scope-class",
                "1",
                "--seed",
                "1",
            ]
        )
        == 0
    )
    assert "COMPLETED" in capsys.readouterr().out
    fr = default_fault_registry()
    compound = fr.compound(
        fr.make("gaussian_noise", seed=1, sigma=0.5),
        fr.make("feature_dropout", seed=2, probability=0.2),
    )
    path = tmp_path / "compound.json"
    path.write_text(json.dumps(compound.to_dict()))
    assert (
        main(
            [
                "--workspace",
                ws,
                "fault",
                "run",
                "--model",
                mid,
                "--dataset",
                did,
                "--split",
                "test",
                "--spec-file",
                str(path),
                "--seed",
                "4",
            ]
        )
        == 0
    )
    assert "COMPLETED" in capsys.readouterr().out


# --- real scikit-learn experiments ---------------------------------------------------------------


def test_sklearn_faults_show_measured_degradation_and_record_failures(demo_ws: Path) -> None:
    from experionyx.adapters.registry import default_registries
    from experionyx.demos import _sklearn_config
    from experionyx.execution import Executor
    from experionyx.faults.design import FaultDesign, FaultLimits, SweepSpec
    from experionyx.faults.lab import run_fault_experiment
    from experionyx.faults.library import default_fault_registry

    fr = default_fault_registry()
    reg = SqliteRegistry(demo_ws / "registry.sqlite")
    store = LocalArtifactStore(demo_ws / "experiments")
    ex = Executor(
        reg, store, source_root=demo_ws.parent, adapters=default_registries(), inputs_root=demo_ws
    )
    mid, did = reg.find(RegisteredModel)[0].id, reg.find(RegisteredDataset)[0].id
    cfg = _sklearn_config(False)

    noise = fr.make("gaussian_noise", seed=0, sigma=0.0)
    res = run_fault_experiment(reg, store, ex, model_id=mid, dataset_id=did, base_spec=noise, fault_registry=fr,
                               design=FaultDesign(noise.to_dict(), cfg, seeds=(1, 2, 3), sweep=SweepSpec("sigma", (0.0, 1.0, 3.0))),
                               name="noise", source_root=demo_ws.parent)  # fmt: skip
    a = load_analysis(reg, store, res.fault_experiment.id)
    means = [p.primary_faulted.mean for p in a.points]
    assert means[0] == pytest.approx(a.baseline_value)  # sigma=0 changes nothing
    assert means[0] > means[-1]  # accuracy falls under heavy noise
    assert a.points[-1].assessment.classification.value in (
        "MEASURED_DEGRADATION",
        "SUBSTANTIAL_DEGRADATION",
    )
    assert len(a.series) == 9
    assert all(t.status is TrialStatus.COMPLETED for t in res.trials)

    # sklearn estimators reject NaN: the failure is recorded, and early termination is explicit
    missing = fr.make("missing_values", seed=0, probability=0.5)
    res2 = run_fault_experiment(reg, store, ex, model_id=mid, dataset_id=did, base_spec=missing, fault_registry=fr,
                                design=FaultDesign(missing.to_dict(), cfg, seeds=(1, 2, 3), limits=FaultLimits(max_failed_trials=1)),
                                name="missing", source_root=demo_ws.parent, baseline_run_id=res.baseline_run_id)  # fmt: skip
    assert [t.status for t in res2.trials] == [
        TrialStatus.FAILED,
        TrialStatus.SKIPPED,
        TrialStatus.SKIPPED,
    ]
    assert "ValueError" in (res2.trials[0].reason or "") or "NaN" in (res2.trials[0].reason or "")
    reg.close()


def test_label_demo_measures_label_noise_separately_from_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("sklearn")
    monkeypatch.chdir(tmp_path)
    (result,) = run_fault_demo("labels", tmp_path / "w")
    reg = SqliteRegistry(tmp_path / "w" / "registry.sqlite")
    a = load_analysis(
        reg, LocalArtifactStore(tmp_path / "w" / "experiments"), result.fault_experiment.id
    )
    assert [p.parameter_value for p in a.points] == [0.0, 0.1, 0.2, 0.3, 0.5]
    assert all(p.n_completed == 10 for p in a.points)  # ten seeds, raw runs kept
    assert a.points[0].primary_deterioration.mean == pytest.approx(0.0)
    assert a.points[-1].primary_deterioration.mean > a.points[1].primary_deterioration.mean  # type: ignore[operator]
    assert a.points[-1].primary_deterioration.n == 10
    reg.close()


def test_tabular_demo_runs_three_sweeps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("sklearn")
    monkeypatch.chdir(tmp_path)
    results = run_fault_demo("tabular", tmp_path / "w")
    assert len(results) == 3
    reg = SqliteRegistry(tmp_path / "w" / "registry.sqlite")
    store = LocalArtifactStore(tmp_path / "w" / "experiments")
    for r in results:
        a = load_analysis(reg, store, r.fault_experiment.id)
        assert a.failed_trials == 0
        assert all(len(p.trials if hasattr(p, "trials") else [1]) for p in a.points)
    reg.close()


# --- real PyTorch experiments ----------------------------------------------------------------------


def test_tensor_demo_runs_image_faults_on_a_torch_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    monkeypatch.chdir(tmp_path)
    results = run_fault_demo("tensor", tmp_path / "w")
    assert [r.fault_experiment.name for r in results] == [
        "stripes: gaussian noise",
        "stripes: random occlusion",
        "stripes: brightness",
        "stripes: contrast",
    ]
    reg = SqliteRegistry(tmp_path / "w" / "registry.sqlite")
    store = LocalArtifactStore(tmp_path / "w" / "experiments")
    analyses = {
        r.fault_experiment.name: load_analysis(reg, store, r.fault_experiment.id) for r in results
    }
    for name, a in analyses.items():
        assert a.failed_trials == 0, name
        assert a.baseline_value == pytest.approx(1.0)  # the tiny CNN separates the stripes
    noise = analyses["stripes: gaussian noise"]
    assert noise.points[0].primary_deterioration.mean == pytest.approx(0.0)  # sigma=0
    assert noise.points[2].primary_deterioration.mean == pytest.approx(
        0.0
    )  # sigma=1: measured robust
    assert noise.points[-1].primary_faulted.mean < 1.0  # sigma=6 finally hurts (sigma<=1 does not)
    bright = analyses["stripes: brightness"]
    zero = next(p for p in bright.points if p.parameter_value == 0.0)
    assert zero.primary_deterioration.mean == pytest.approx(0.0)  # delta=0 is the identity
    assert len(bright.series) == 4  # deterministic fault: one seed per point
    reg.close()

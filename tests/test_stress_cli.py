"""The stress CLI and REAL-DATA engineering validation experiments: (1) the bundled iris dataset with
a scikit-learn logistic regression (tabular: an input-stress sweep, a seeded parameter-noise sweep
over repeats, a repeated-execution run); (2) the existing stripes image dataset with a small PyTorch
CNN (tensor/image: brightness, blur, occlusion and parameter noise). Expected values come from numpy,
scikit-learn and PyTorch directly. These check that the machinery reproduces independent numbers on
real models; they are NOT scientific findings about iris, about stripes or about robustness."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("sklearn")

import joblib
from sklearn.datasets import load_iris

from experionyx.adapters.records import RegisteredDataset, RegisteredModel
from experionyx.cli import main
from experionyx.demos import _sklearn_config, run_demo
from experionyx.domain import Run, to_jsonable
from experionyx.sqlite import SqliteRegistry
from experionyx.stress.entities import StressAnalysis

CLASSES = 3


@dataclass
class Iris:
    ws: Path
    baseline: str
    mid: str
    did: str


@pytest.fixture(scope="module")
def iris(tmp_path_factory: pytest.TempPathFactory) -> Iris:
    ws = tmp_path_factory.mktemp("iris-stress") / "w"
    base = run_demo("sklearn-classification", ws)
    with SqliteRegistry(ws / "registry.sqlite") as reg:
        return Iris(
            ws, base.run.id, reg.find(RegisteredModel)[0].id, reg.find(RegisteredDataset)[0].id
        )


def cli(ws: Path, capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str, str]:
    code = main(["--workspace", str(ws), "stress", *args])
    out = capsys.readouterr()
    return code, out.out, out.err


def write(tmp_path: Path, name: str, body: Any) -> str:
    p = tmp_path / name
    p.write_text(body if isinstance(body, str) else json.dumps(body), encoding="utf-8")
    return str(p)


def design(iris: Iris, plan: dict[str, Any], **over: Any) -> dict[str, Any]:
    d: dict[str, Any] = {"model_id": iris.mid, "dataset_id": iris.did, "plan": plan, "evaluation": to_jsonable(_sklearn_config(False)), "baseline_run": iris.baseline, "min_members": 5, "resamples": 100, "permutations": 200}  # fmt: skip
    d.update(over)
    return d


def n_runs(ws: Path) -> int:
    with SqliteRegistry(ws / "registry.sqlite") as reg:
        return len(reg.find(Run))


def truth(iris: Iris) -> tuple[Any, Any, Any, Any]:
    data = load_iris()
    order = np.random.RandomState(0).permutation(150)
    test = sorted(int(i) for i in order[150 - round(150 * 0.25) :])
    return (
        data.data[test],
        data.target[test],
        joblib.load(iris.ws / "models" / "iris-logreg.joblib"),
        test,
    )


def results_of(ws: Path, aid: str) -> dict[str, Any]:
    from experionyx.artifacts import LocalArtifactStore
    from experionyx.stress.registry import StressRegistry

    with SqliteRegistry(ws / "registry.sqlite") as reg:
        doc: dict[str, Any] = StressRegistry(reg, LocalArtifactStore(ws / "experiments")).document(
            aid, "results"
        )
        return dict(doc["results"])


def acc_of(res: dict[str, Any]) -> float:
    return next(m for m in res["metrics"] if m["metric_id"] == "accuracy")["stressed_value"]  # type: ignore[no-any-return]


# -- real tabular experiments ---------------------------------------------------------------------------------------------------------------


def test_real_iris_input_stress_sweep_matches_scikit_learn(
    iris: Iris, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    plan = {"components": [{"family": "FEATURE_SCALING", "parameters": {"factor": 1.0}}], "sweep": {"parameter": "factor", "values": [0.5, 1.0, 2.0, 4.0]}}  # fmt: skip
    f = write(tmp_path, "s.json", design(iris, plan))
    code, out, _ = cli(iris.ws, capsys, "sweep", f)
    doc = json.loads(out)
    assert code in (0, 3) and doc["status"] == "COMPLETED" and doc["analysis_id"].startswith("sxa_")
    assert code == (
        0 if doc["analysis_status"] == "COMPLETE" else 3
    )  # the exit code mirrors the evidence
    x, y, model, _ = truth(iris)
    res = results_of(iris.ws, doc["analysis_id"])
    units = doc["summary"]["coverage"]["requested"]
    assert units == 4 and len(res) == 4
    got = sorted(acc_of(r) for r in res.values())
    want = sorted(float((model.predict(x * f_) == y).mean()) for f_ in (0.5, 1.0, 2.0, 4.0))
    assert got == pytest.approx(want, abs=1e-12)
    trials = doc["summary"]["design"]
    assert trials == "SWEEP" and doc["summary"]["origins"] == ["FAULT_LABORATORY"]


def test_real_iris_parameter_noise_over_seeds_matches_an_independent_derivation(
    iris: Iris, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    import hashlib

    plan = {
        "components": [{"family": "PARAMETER_NOISE", "parameters": {"relative_sigma": 0.8}}],
        "seeds": [1, 2, 3],
    }
    _, out, _ = cli(iris.ws, capsys, "run", write(tmp_path, "n.json", design(iris, plan)))
    doc = json.loads(out)
    assert (
        doc["status"] == "COMPLETED"
        and doc["summary"]["design"] == "REPEATED"
        and doc["summary"]["coverage"]["completed"] == 3
    )
    x, y, model, _ = truth(iris)
    coef, b = model.coef_.copy(), model.intercept_.copy()

    def noisy(seed: int, name: str, arr: Any) -> Any:
        rms = float(np.sqrt(np.mean(arr**2)))
        rng = np.random.Generator(
            np.random.PCG64(
                int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "big")
            )
        )
        return arr + rng.normal(0.0, 0.8 * rms, arr.shape)

    want = []
    for seed in (1, 2, 3):
        c, i = noisy(seed, "coef_", coef), noisy(seed, "intercept_", b)
        want.append(float(((x @ c.T + i).argmax(axis=1) == y).mean()))
    res = results_of(iris.ws, doc["analysis_id"])
    assert sorted(acc_of(r) for r in res.values()) == pytest.approx(sorted(want), abs=1e-12)
    (agg,) = [
        v for k, v in json.loads(json.dumps(_aggregates(iris.ws, doc["analysis_id"]))).items()
    ]
    assert agg["deterioration"]["n"] == 3 and sorted(
        agg["deterioration"]["values"]
    ) == pytest.approx(sorted(float((model.predict(x) == y).mean()) - w for w in want), abs=1e-12)
    # the registered model file on disk is untouched
    assert np.array_equal(joblib.load(iris.ws / "models" / "iris-logreg.joblib").coef_, coef)


def _aggregates(ws: Path, aid: str) -> dict[str, Any]:
    from experionyx.artifacts import LocalArtifactStore
    from experionyx.stress.registry import StressRegistry

    with SqliteRegistry(ws / "registry.sqlite") as reg:
        doc: dict[str, Any] = StressRegistry(reg, LocalArtifactStore(ws / "experiments")).document(
            aid, "results"
        )
        return dict(doc["aggregates"])


def test_real_iris_repeated_execution_replay_compare_list_and_inspect(
    iris: Iris, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    plan = {"components": [{"family": "REPEATED_EXECUTION", "parameters": {"repeats": 3}}]}
    f = write(tmp_path, "r.json", design(iris, plan))
    code, out, _ = cli(iris.ws, capsys, "run", f)
    doc = json.loads(out)
    aid = doc["analysis_id"]
    assert (
        code == 0 and doc["summary"]["stability_observed"] == "DETERMINISTIC"
    )  # sklearn logistic regression is deterministic
    code, out, _ = cli(iris.ws, capsys, "replay", aid)
    rep = json.loads(out)
    assert (
        code == 0
        and rep["deterministic"] is True
        and rep["differences"] == []
        and len(rep["compared"]["trials_replayed"]) == 3
    )
    code, out, _ = cli(iris.ws, capsys, "compare", aid)
    cmp = json.loads(out)
    before = n_runs(iris.ws)
    assert (
        code == 0
        and cmp["reproduced"] is True
        and cmp["provenance_fingerprint"]["stored"] == cmp["provenance_fingerprint"]["recomputed"]
    )
    assert n_runs(iris.ws) == before and "nothing was stored" in cmp["not_stored"]
    code, out, _ = cli(iris.ws, capsys, "list", "--model-id", iris.mid)
    assert code == 0 and aid in {a["id"] for a in json.loads(out)["analyses"]}
    code, out, _ = cli(iris.ws, capsys, "inspect", aid)
    d = json.loads(out)
    assert (
        d["provenance"]["provenance_fingerprint"]
        and d["provenance"]["run_provenance"]["environment_id"]
        and len(d["trials"]) == 3
    )
    assert {a["path"] for a in d["artifacts"]} == {
        f"stress/{n}.json"
        for n in ("spec", "plan", "trials", "baseline", "results", "analyses", "summary")
    }
    code, out, _ = cli(iris.ws, capsys, "inspect", d["trials"][0]["id"])
    assert json.loads(out)["origin"] == "STRESS_LABORATORY"
    code, out, _ = cli(iris.ws, capsys, "inspect", aid, "--full")
    assert set(json.loads(out)["documents"]) == {
        "spec",
        "plan",
        "trials",
        "baseline",
        "results",
        "analyses",
        "summary",
    }
    code, out, _ = cli(iris.ws, capsys, "list", "--format", "text")
    assert aid in out and code == 0


def test_families_validate_and_preflight_report_capabilities_deterministically(
    iris: Iris, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code, out, _ = cli(iris.ws, capsys, "families")
    fams = json.loads(out)["families"]
    assert (
        code == 0
        and fams["FEATURE_SCALING"]["origin"] == "FAULT_LABORATORY"
        and fams["THRESHOLD"]["origin"] == "MODEL"
    )
    plan = {
        "components": [
            {"family": "PARAMETER_SCALE", "parameters": {"factor": 2.0}},
            {"family": "THRESHOLD", "parameters": {"threshold": 0.4}},
        ]
    }
    f = write(tmp_path, "c.json", design(iris, plan))
    code, out, _ = cli(iris.ws, capsys, "validate", f)
    d = json.loads(out)
    assert (
        code == 0
        and d["valid"] is True
        and d["design"] == "COMPOUND"
        and d["n_trials"] == 1
        and d["spec_id"].startswith("sxr_")
    )
    assert out == cli(iris.ws, capsys, "validate", f)[1]  # identical input, identical bytes
    code, out, err = cli(iris.ws, capsys, "validate", f, "--preflight")
    assert (
        code == 2
        and "THRESHOLD is UNAVAILABLE" in err
        and "binary classifiers" in err
        and out == ""
    )  # iris has three classes: no meaningful threshold
    ok = {"components": [{"family": "PARAMETER_SCALE", "parameters": {"factor": 2.0}}]}
    code, out, _ = cli(
        iris.ws, capsys, "validate", write(tmp_path, "ok.json", design(iris, ok)), "--preflight"
    )
    caps = json.loads(out)["capabilities"]
    assert code == 0 and caps[0]["status"] == "SUPPORTED" and caps[0]["family"] == "PARAMETER_SCALE"
    code, out, _ = cli(
        iris.ws, capsys, "validate", write(tmp_path, "t.json", design(iris, ok)), "--format", "text"
    )
    assert out.startswith("VALID: sxr_")


# -- invalid requests: clean errors, nothing executed ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("what", "plan", "over", "message"),
    [
        ("unknown-family", {"components": [{"family": "VIBES"}]}, {}, "unknown stress family"),
        ("bad-range", {"components": [{"family": "PARAMETER_SCALE", "parameters": {"factor": -2}}]}, {}, "below its valid range"),
        ("fake-seeds", {"components": [{"family": "PARAMETER_SCALE", "parameters": {"factor": 2}}], "seeds": [1, 2]}, {}, "deterministic"),
        ("mixed-compound", {"components": [{"family": "FEATURE_NOISE", "parameters": {"sigma": 1}, "seed": 1}, {"family": "THRESHOLD", "parameters": {"threshold": 0.4}}]}, {}, "mixed origins"),
        ("shape", {"components": [{"family": "INPUT_SHAPE", "parameters": {"shape": [3, 32, 32]}}]}, {"_preflight": True}, "does not declare any supported input shapes"),
        ("image-on-tabular", {"components": [{"family": "BRIGHTNESS", "parameters": {"delta": 0.1}}]}, {"_preflight": True}, "Fault Laboratory refuses"),
        ("bad-target", {"components": [{"family": "PARAMETER_SCALE", "parameters": {"factor": 2, "targets": ["ghost"]}}]}, {"_preflight": True}, "ghost"),
        ("bad-correction", {"components": [{"family": "PARAMETER_SCALE", "parameters": {"factor": 2}}]}, {"correction": "HOLM"}, "correction"),
        ("unknown-key", {"components": [{"family": "PARAMETER_SCALE", "parameters": {"factor": 2}}]}, {"surprise": 1}, "unexpected"),
        ("other-baseline", {"components": [{"family": "PARAMETER_SCALE", "parameters": {"factor": 2}}]}, {"evaluation": {"split": "test", "batch_size": 3}}, "different evaluation configuration"),
    ],
)  # fmt: skip
def test_invalid_requests_fail_cleanly_and_execute_nothing(
    iris: Iris,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    what: str,
    plan: dict[str, Any],
    over: dict[str, Any],
    message: str,
) -> None:
    over = dict(over)
    over.pop("_preflight", None)
    f = write(tmp_path, f"{what}.json", design(iris, plan, **over))
    before = n_runs(iris.ws)
    code, out, err = cli(iris.ws, capsys, "run", f)
    assert code == 2 and out == "" and err.startswith("error:") and "Traceback" not in err, (
        what,
        err,
    )
    assert message in err, (what, err)
    assert n_runs(iris.ws) == before  # refusals create nothing


def test_unreadable_specs_unknown_ids_and_sweep_misuse_are_user_errors(
    iris: Iris, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    for bad in (
        write(tmp_path, "junk.json", "{not json"),
        write(tmp_path, "list.json", "[1]"),
        str(tmp_path / "missing.json"),
    ):
        code, out, err = cli(iris.ws, capsys, "validate", bad)
        assert code == 2 and out == "" and err.startswith("error:") and "Traceback" not in err
    for args in (
        ("inspect", "sxa_" + "0" * 32),
        ("inspect", "sxt_" + "0" * 32),
        ("replay", "sxa_" + "0" * 32),
        ("compare", "sxa_" + "0" * 32),
    ):
        code, out, err = cli(iris.ws, capsys, *args)
        assert code == 2 and err.startswith("error:") and "Traceback" not in err, args
    single = {"components": [{"family": "PARAMETER_SCALE", "parameters": {"factor": 2.0}}]}
    code, out, err = cli(iris.ws, capsys, "sweep", write(tmp_path, "s.json", design(iris, single)))
    assert code == 2 and "needs a plan with a sweep or several seeds" in err and out == ""
    code = main(["--workspace", str(tmp_path / "nowhere"), "stress", "list"])
    assert code == 2 and "no registry" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "no robustness score" in " ".join(capsys.readouterr().out.split())


# -- real tensor / image experiments --------------------------------------------------------------------------------------------------------------


def test_real_image_stress_on_a_torch_cnn_matches_pytorch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    torch = pytest.importorskip("torch")
    from experionyx.adapters.capabilities import DeviceKind
    from experionyx.adapters.torch_adapter import TorchModelAdapter
    from experionyx.faults.demos import run_fault_demo
    from experionyx.faults.entities import FaultExperiment

    monkeypatch.chdir(tmp_path)
    ws = tmp_path / "w"
    results = run_fault_demo("tensor", ws)  # the existing stripes dataset + tiny CNN + baseline
    with SqliteRegistry(ws / "registry.sqlite") as reg:
        fxp = reg.get(FaultExperiment, results[0].fault_experiment.id)
        mid, did = reg.find(RegisteredModel)[0].id, reg.find(RegisteredDataset)[0].id
        baseline, evaluation = fxp.baseline_run_id, to_jsonable(fxp.design["evaluation"])

    def spec(plan: dict[str, Any], **over: Any) -> dict[str, Any]:
        return {"model_id": mid, "dataset_id": did, "plan": plan, "evaluation": evaluation, "baseline_run": baseline, "min_members": 5, "resamples": 100, "permutations": 100, **over}  # fmt: skip

    payload = torch.load(next((ws / "datasets").glob("*.pt")))
    x, y = payload["X"][160:240], payload["y"][160:240]
    model = TorchModelAdapter.load(
        next((ws / "models").glob("*.pt")),
        version="1",
        device=DeviceKind.CPU,
        options={"task": "CLASSIFICATION", "input_shape": [1, 8, 8]},
    )

    def acc(inputs: Any, m: Any = model) -> float:
        out = m.predict(inputs).outputs
        return float(
            (
                torch.as_tensor(
                    [int(np.argmax(o)) if hasattr(o, "__len__") else int(o) for o in out]
                )
                == y
            )
            .float()
            .mean()
        )

    plan_b = {
        "components": [{"family": "BRIGHTNESS", "parameters": {"delta": 0.0}}],
        "sweep": {"parameter": "delta", "values": [-0.5, 0.0, 0.25]},
    }
    code, out, _ = cli(ws, capsys, "sweep", write(tmp_path, "b.json", spec(plan_b)))
    doc = json.loads(out)
    assert (
        code in (0, 3)
        and doc["summary"]["coverage"]["completed"] == 3
        and doc["summary"]["origins"] == ["FAULT_LABORATORY"]
    )
    got = sorted(acc_of(r) for r in results_of(ws, doc["analysis_id"]).values())
    want = sorted(acc((x + d).clamp(0, 1)) for d in (-0.5, 0.0, 0.25))
    assert got == pytest.approx(want, abs=1e-12)
    plan_p = {
        "components": [{"family": "PARAMETER_NOISE", "parameters": {"relative_sigma": 1.0}}],
        "seeds": [1, 2],
    }
    code, out, _ = cli(ws, capsys, "run", write(tmp_path, "p.json", spec(plan_p)))
    pdoc = json.loads(out)
    assert pdoc["summary"]["coverage"]["completed"] == 2 and pdoc["summary"]["origins"] == ["MODEL"]
    base_acc = acc(x)
    assert base_acc == pytest.approx(1.0)  # the tiny CNN separates the stripes
    for r in results_of(ws, pdoc["analysis_id"]).values():
        assert r["input_evidence"]["status"] == "INPUTS_UNCHANGED" and 0.0 <= acc_of(r) <= 1.0
    for name, params, extra in (
        ("BLUR", {"radius": 1}, {}),
        ("OCCLUSION", {"fraction": 0.5}, {"seeds": [1, 2]}),
    ):
        plan = {
            "components": [
                {"family": name, "parameters": params, **({"seed": 1} if extra else {})}
            ],
            **extra,
        }
        c, o, e = cli(ws, capsys, "run", write(tmp_path, f"{name}.json", spec(plan)))
        assert c in (0, 3) and json.loads(o)["summary"]["coverage"]["failed"] == 0, (name, e)
    thr = {"components": [{"family": "THRESHOLD", "parameters": {"threshold": 0.4}}]}
    c, o, e = cli(ws, capsys, "run", write(tmp_path, "thr.json", spec(thr)))
    assert (
        c == 2 and "THRESHOLD is UNAVAILABLE" in e and o == ""
    )  # a logits model exposes no probabilities: refused, not approximated
    code, out, _ = cli(ws, capsys, "replay", pdoc["analysis_id"])
    assert code == 0 and json.loads(out)["deterministic"] is True
    assert StressAnalysis.PREFIX == "sxa"

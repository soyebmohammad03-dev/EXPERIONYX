"""The drift CLI and a REAL-DATA scenario: a scikit-learn model trained on the bundled iris dataset
and evaluated by the normal engine. Iris rows are stored class-sorted, so declaring the SAMPLE INDEX
as the ordering makes the two windows differ in class balance BY CONSTRUCTION: this is a validation
fixture that shows the machinery reproduces scikit-learn/numpy values, not a finding about iris or
about time. Expected values come from scikit-learn and numpy directly."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sklearn")

import joblib
import numpy as np
from sklearn.datasets import load_iris

from experionyx.cli import main
from experionyx.demos import run_demo
from experionyx.domain import Run
from experionyx.sqlite import SqliteRegistry

PW = "petal width (cm)"
REF, CMP = (0, 75), (75, 150)
PAIR = "[0, 75) vs [75, 150)"


@dataclass
class Iris:
    ws: Path
    baseline: str


@pytest.fixture(scope="module")
def iris(tmp_path_factory: pytest.TempPathFactory) -> Iris:
    ws = tmp_path_factory.mktemp("iris-drift") / "w"
    base = run_demo("sklearn-classification", ws)
    assert base.run is not None
    return Iris(ws, base.run.id)


def cli(iris: Iris, capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str, str]:
    code = main(["--workspace", str(iris.ws), "drift", *args])
    out = capsys.readouterr()
    return code, out.out, out.err


def spec(iris: Iris, **over: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "baseline_run": iris.baseline, "ordering": {"field": "index"},
        "reference": {"start": REF[0], "end": REF[1]}, "comparisons": [{"start": CMP[0], "end": CMP[1]}],
        "features": [{"name": PW, "type": "NUMERIC"}, {"name": "sepal length (cm)", "type": "NUMERIC"}],
        "slices": [{"name": "wide", "condition": {"op": "range", "field": f"feature:{PW}", "low": 1.0, "high": None}}],
        "config": {"min_samples": 5, "resamples": 200, "permutations": 500, "correction": "BENJAMINI_HOCHBERG"},
    }  # fmt: skip
    d.update(over)
    return d


def write(tmp_path: Path, name: str, body: Any) -> str:
    p = tmp_path / name
    p.write_text(body if isinstance(body, str) else json.dumps(body), encoding="utf-8")
    return str(p)


def truth() -> tuple[list[int], Any, Any]:
    data = load_iris()
    order = np.random.RandomState(0).permutation(150)
    test = sorted(int(i) for i in order[150 - round(150 * 0.25) :])
    return test, data, None


def n_runs(iris: Iris) -> int:
    with SqliteRegistry(iris.ws / "registry.sqlite") as reg:
        return len(reg.find(Run))


# -- real data: values equal scikit-learn / numpy ------------------------------------------------------------


def test_real_iris_analysis_matches_scikit_learn_and_numpy(
    iris: Iris, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    f = write(tmp_path, "s.json", spec(iris))
    code, out, _ = cli(iris, capsys, "evaluate", f)
    doc = json.loads(out)
    assert code in (0, 3) and doc["status"] == "COMPLETED" and doc["analysis_id"].startswith("dan_")
    assert code == (
        0 if doc["analysis_status"] == "COMPLETE" else 3
    )  # the exit code mirrors the evidence
    test, data, _ = truth()
    early, late = [i for i in test if i < 75], [i for i in test if i >= 75]
    code, out, _ = cli(iris, capsys, "inspect", doc["analysis_id"], "--full")
    full = json.loads(out)["documents"]
    pop = full["feature_results"]["results"][PAIR]["populations"]["POPULATION"]
    x, y = np.sort(data.data[early, 3]), np.sort(data.data[late, 3])
    grid = np.unique(np.concatenate([x, y]))
    ks = np.max(
        np.abs(
            np.searchsorted(x, grid, "right") / len(x) - np.searchsorted(y, grid, "right") / len(y)
        )
    )
    assert pop[PW]["measures"]["ks_statistic"] == pytest.approx(ks, abs=1e-12)
    assert pop[PW]["measures"]["location"]["mean_difference_interval"]["estimate"] == pytest.approx(
        y.mean() - x.mean()
    )
    assert (pop[PW]["reference"]["n_valid"], pop[PW]["comparison"]["n_valid"]) == (
        len(early),
        len(late),
    )
    lab = full["distribution_results"]["results"][PAIR]["populations"]["POPULATION"]["label"]
    cats = {c["category"]: c for c in lab["measures"]["categories"]}
    for cls in (0, 1, 2):
        assert (cats[str(cls)]["n_reference"], cats[str(cls)]["n_comparison"]) == (
            int((data.target[early] == cls).sum()),
            int((data.target[late] == cls).sum()),
        )
    model = joblib.load(iris.ws / "models" / "iris-logreg.joblib")

    def acc(ids: list[int]) -> float:
        return float((model.predict(data.data[ids]) == data.target[ids]).mean())

    perf = full["performance_results"]["results"][PAIR]["populations"]["POPULATION"]
    m = {r["metric_id"]: r for r in perf["metrics"]}["accuracy"]
    assert m["reference_value"] == pytest.approx(acc(early)) and m[
        "comparison_value"
    ] == pytest.approx(acc(late))
    wide_e = [i for i in early if data.data[i, 3] >= 1.0]
    assert full["windows"]["slices"]["wide"]["windows"][next(iter(full["windows"]["windows"]))][
        "n_members"
    ] in (len(wide_e), len([i for i in late if data.data[i, 3] >= 1.0]))
    assert "n_members" in json.dumps(full["windows"]["slices"])


def test_replay_and_list_and_inspect_on_the_real_analysis(
    iris: Iris, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    f = write(tmp_path, "s.json", spec(iris, slices=[]))
    _, out, _ = cli(iris, capsys, "evaluate", f)
    aid = json.loads(out)["analysis_id"]
    code, out, _ = cli(iris, capsys, "replay", aid)
    rep = json.loads(out)
    assert (
        code == 0
        and rep["deterministic"] is True
        and rep["differences"] == []
        and rep["replay_status"] == "COMPLETED"
    )
    code, out, _ = cli(iris, capsys, "list", "--windows", "--baseline-run", iris.baseline)
    listing = json.loads(out)
    assert code == 0 and aid in {a["id"] for a in listing["analyses"]} and listing["windows"]
    wid = listing["windows"][0]["id"]
    code, out, _ = cli(iris, capsys, "inspect", wid)
    assert json.loads(out)["used_by"] and wid.startswith("twn_")
    code, out, _ = cli(iris, capsys, "inspect", aid)
    prov = json.loads(out)["provenance"]
    assert (
        prov["provenance_fingerprint"]
        and prov["model_fingerprint"]
        and prov["run_provenance"]["environment_id"]
    )
    assert {a["path"] for a in json.loads(out)["artifacts"]} == {
        f"drift/{n}.json"
        for n in (
            "spec",
            "windows",
            "feature_results",
            "distribution_results",
            "performance_results",
            "summary",
        )
    }
    code, out, _ = cli(iris, capsys, "list", "--format", "text")
    assert aid in out and code == 0


def test_compare_stores_nothing_and_windows_previews_membership(
    iris: Iris, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    f = write(
        tmp_path,
        "s.json",
        spec(iris, config={"min_samples": 5, "resamples": 50, "permutations": 100, "seed": 9}),
    )
    before = n_runs(iris)
    code, out, _ = cli(iris, capsys, "compare", f, "--full")
    doc = json.loads(out)
    assert code == 0 and n_runs(iris) == before and "nothing was stored" in doc["not_stored"]
    assert doc["summary"]["n_pairs"] == 1 and "feature_results" in doc
    code, out, _ = cli(iris, capsys, "windows", f)
    w = json.loads(out)
    assert code == 0 and n_runs(iris) == before and len(w["windows"]) == 2
    assert all("omitted" in x["sample_ids"] for x in w["windows"])
    test, _, _ = truth()
    assert sorted(x["n_samples"] for x in w["windows"]) == sorted(
        [len([i for i in test if i < 75]), len([i for i in test if i >= 75])]
    )
    code, out, _ = cli(iris, capsys, "windows", f, "--ids")
    assert sorted(i for x in json.loads(out)["windows"] for i in x["sample_ids"]) == test


def test_validate_prints_identity_plan_and_can_preflight_the_data(
    iris: Iris, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    roll = spec(iris, reference=None, comparisons=[], rolling={"start": 0, "stop": 160, "width": 50, "step": 50, "reference": "EXPANDING"}, slices=[])  # fmt: skip
    f = write(tmp_path, "r.json", roll)
    code, out, _ = cli(iris, capsys, "validate", f)
    d = json.loads(out)
    assert code == 0 and d["valid"] is True and d["spec_id"].startswith("dsp_")
    assert [p["key"] for p in d["pairs"]] == ["[0, 50) vs [50, 100)", "[0, 100) vs [100, 150)"]
    assert [s["reason"] for s in d["skipped"]] == ["NO_REFERENCE", "PARTIAL_WINDOW"] and d[
        "features"
    ][PW] == "NUMERIC"
    code, out, _ = cli(iris, capsys, "validate", f, "--preflight")
    assert json.loads(out)["preflight"]["ok"] is True and code == 0
    code, out, _ = cli(iris, capsys, "validate", f, "--format", "text")
    assert out.startswith("VALID: dsp_")
    again = json.loads(cli(iris, capsys, "validate", f)[1])["spec_id"]
    assert again == d["spec_id"]  # deterministic output


# -- invalid requests: clean errors, nothing executed -------------------------------------------------------------


@pytest.mark.parametrize(
    ("what", "body", "message"),
    [
        ("overlap", {"reference": {"start": 0, "end": 100}}, "overlaps"),
        ("bad-type", {"features": [{"name": PW, "type": "TEXT"}]}, "unsupported feature type"),
        ("no-feature", {"features": [{"name": "nope", "type": "NUMERIC"}]}, "not a feature of this dataset"),
        ("bad-ordering", {"ordering": {"field": "row"}}, "ordering field"),
        ("empty-window", {"reference": {"start": 500, "end": 600}}, "no samples"),
        ("bad-threshold", {"config": {"min_samples": 1}}, "min_samples"),
        ("bad-correction", {"config": {"correction": "HOLM"}}, "correction"),
        ("both-modes", {"rolling": {"start": 0, "stop": 10, "width": 5, "step": 5}}, "not both"),
        ("unknown-key", {"surprise": 1}, "unexpected"),
        ("unknown-run", {"baseline_run": "run_" + "7" * 32}, "not found|No such|run_"),
    ],
)  # fmt: skip
def test_invalid_requests_fail_cleanly_and_execute_nothing(
    iris: Iris,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    what: str,
    body: dict[str, Any],
    message: str,
) -> None:
    f = write(tmp_path, f"{what}.json", spec(iris, **body))
    before = n_runs(iris)
    for cmd in ("evaluate", "compare", "windows"):
        code, out, err = cli(iris, capsys, cmd, f)
        assert code == 2 and out == "" and err.startswith("error:") and "Traceback" not in err, (
            what,
            cmd,
            err,
        )
        assert n_runs(iris) == before
    if what in (
        "overlap",
        "bad-type",
        "bad-ordering",
        "bad-threshold",
        "bad-correction",
        "both-modes",
        "unknown-key",
    ):
        code, out, err = cli(
            iris, capsys, "validate", f
        )  # structural problems are caught without any data
        assert code == 2 and out == "" and "Traceback" not in err


def test_unreadable_specs_and_unknown_ids_are_user_errors(
    iris: Iris, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    for bad in (
        write(tmp_path, "junk.json", "{not json"),
        write(tmp_path, "list.json", "[1, 2]"),
        str(tmp_path / "missing.json"),
    ):
        code, out, err = cli(iris, capsys, "validate", bad)
        assert code == 2 and out == "" and err.startswith("error:") and "Traceback" not in err
    for args in (
        ("inspect", "dan_" + "0" * 32),
        ("inspect", "twn_" + "0" * 32),
        ("replay", "dan_" + "0" * 32),
    ):
        code, out, err = cli(iris, capsys, *args)
        assert code == 2 and err.startswith("error:") and "Traceback" not in err
    code = main(["--workspace", str(tmp_path / "nowhere"), "drift", "list"])
    assert code == 2 and "no registry" in capsys.readouterr().err


def test_help_text_and_json_are_deterministic(
    iris: Iris, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    for argv in (["--help"], ["drift", "--help"]):
        with pytest.raises(SystemExit) as e:
            main(argv)
        assert e.value.code == 0
        text = capsys.readouterr().out
    for cmd in ("validate", "list", "inspect", "windows", "evaluate", "compare", "replay"):
        assert cmd in text
    with pytest.raises(SystemExit):
        main(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "no drift score" in text and "no causal claim" in text
    f = write(
        tmp_path,
        "s.json",
        spec(iris, slices=[], config={"min_samples": 5, "resamples": 50, "permutations": 100}),
    )
    a = cli(iris, capsys, "compare", f)[1]
    assert a == cli(iris, capsys, "compare", f)[1]  # identical input, identical bytes

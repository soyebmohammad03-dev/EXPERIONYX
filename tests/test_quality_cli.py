"""The data-quality CLI and a REAL-DATA engineering validation: the bundled iris dataset registered
by the sklearn demo. Expected values come from scikit-learn/numpy directly (the same random split
the adapter uses). This checks that the machinery reproduces independent numbers on a real dataset; it
is NOT a finding about iris or about data quality in general. Iris does contain one genuine exact
duplicate row, which makes the duplicate and split-overlap indicators meaningful to compare."""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sklearn")

import numpy as np
from sklearn.datasets import load_iris

from experionyx.adapters.records import RegisteredDataset
from experionyx.cli import main
from experionyx.demos import run_demo
from experionyx.domain import Investigation, Run
from experionyx.sqlite import SqliteRegistry

FEATS = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]


@dataclass
class Iris:
    ws: Path
    did: str
    inv: str


@pytest.fixture(scope="module")
def iris(tmp_path_factory: pytest.TempPathFactory) -> Iris:
    ws = tmp_path_factory.mktemp("iris-quality") / "w"
    run_demo("sklearn-classification", ws)
    with SqliteRegistry(ws / "registry.sqlite") as reg:
        return Iris(ws, reg.find(RegisteredDataset)[0].id, reg.find(Investigation)[0].id)


def spec(iris: Iris, **over: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "dataset_id": iris.did, "splits": ["test", "train"], "target": {"task": "CLASSIFICATION"},
        "features": [{"name": n, "type": "NUMERIC"} for n in FEATS],
        "checks": [
            {"type": "sample_count", "config": {"min": 30}}, {"type": "missingness", "config": {}},
            {"type": "duplicates", "config": {}}, {"type": "numeric", "config": {"bounds": {n: [0, 10] for n in FEATS}}},
            {"type": "target", "config": {"min_class_count": 5}},
            {"type": "split_overlap", "config": {"reference": "train", "comparison": "test"}},
            {"type": "group_comparison", "config": {"reference": {"split": "train"}, "comparison": {"split": "test"}, "aspects": ["missingness", "target"]}},
        ],
        "config": {"min_members": 10, "resamples": 100, "permutations": 200},
    }  # fmt: skip
    d.update(over)
    return d


def cli(iris: Iris, capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str, str]:
    code = main(["--workspace", str(iris.ws), "data-quality", *args])
    out = capsys.readouterr()
    return code, out.out, out.err


def write(tmp_path: Path, name: str, body: Any) -> str:
    p = tmp_path / name
    p.write_text(body if isinstance(body, str) else json.dumps(body), encoding="utf-8")
    return str(p)


def n_runs(iris: Iris) -> int:
    with SqliteRegistry(iris.ws / "registry.sqlite") as reg:
        return len(reg.find(Run))


def truth() -> tuple[Any, dict[str, list[int]]]:
    data = load_iris()
    order = np.random.RandomState(0).permutation(150)
    return data, {"test": sorted(int(i) for i in order[150 - round(150 * 0.25) :]), "train": sorted(int(i) for i in order[: 150 - round(150 * 0.25)])}  # fmt: skip


# -- real data: values equal numpy / scikit-learn -----------------------------------------------------------------------


def test_real_iris_analysis_matches_numpy_and_scikit_learn(
    iris: Iris, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code, out, _ = cli(iris, capsys, "run", write(tmp_path, "s.json", spec(iris)))
    doc = json.loads(out)
    assert doc["status"] == "COMPLETED" and doc["analysis_id"].startswith("qan_")
    assert code == (
        0
        if doc["analysis_status"] == "COMPLETE" and not doc["summary"]["status_counts"].get("FAIL")
        else 3
    )
    data, splits = truth()
    code, out, _ = cli(iris, capsys, "inspect", doc["analysis_id"], "--full")
    docs = json.loads(out)["documents"]
    by_type = {c["type"]: c for c in docs["checks"]["checks"]}

    def obs(kind: str, scope: str) -> dict[str, Any]:
        return docs["observations"][by_type[kind]["check_id"]][scope]["observations"]  # type: ignore[no-any-return]

    for split, ids in splits.items():
        x, y = data.data[ids], data.target[ids]
        assert obs("sample_count", f"split={split}")["n_rows"] == len(ids)
        counts = Counter(int(v) for v in y)
        assert {
            int(k): v["count"] for k, v in obs("target", f"split={split}")["classes"].items()
        } == dict(counts)
        assert obs("missingness", f"split={split}")["rows_with_any_missing"] == 0
        uniq = len({tuple(r) for r in x.tolist()})
        dup = obs("duplicates", f"split={split}")
        assert dup["n_unique_rows"] == uniq and dup["n_duplicate_rows"] == len(ids) - uniq
    train_vecs = {tuple(r) for r in data.data[splits["train"]].tolist()}
    shared = [i for i in splits["test"] if tuple(data.data[i].tolist()) in train_vecs]
    (key,) = list(docs["observations"][by_type["split_overlap"]["check_id"]])
    ov = docs["observations"][by_type["split_overlap"]["check_id"]][key]["observations"]
    assert ov["sample_id_overlap"] == 0 and ov[
        "comparison_rows_with_identical_vector_in_reference"
    ] == len(shared)
    for name in FEATS:
        j = FEATS.index(name)
        col = data.data[splits["train"], j]
        f = obs("numeric", "split=train")["features"][name]
        assert f["summary"]["mean"]["value"] == pytest.approx(float(col.mean()), abs=1e-12)
        q1, q3 = np.percentile(
            col, [25, 75]
        )  # linear interpolation = the documented type-7 quantile
        o = f["outliers"]
        assert (o["q1"], o["q3"]) == (pytest.approx(q1), pytest.approx(q3))
        want = int(((col < q1 - 1.5 * (q3 - q1)) | (col > q3 + 1.5 * (q3 - q1))).sum())
        assert o["n_outliers"] == want and f["bounds"]["n_outside"] == 0
    gc = docs["observations"][by_type["group_comparison"]["check_id"]]
    assert len(gc) == 1


def test_replay_list_inspect_and_checks_on_the_real_analysis(
    iris: Iris, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    f = write(
        tmp_path,
        "s.json",
        spec(iris, config={"min_members": 10, "resamples": 50, "permutations": 100, "seed": 3}),
    )
    _, out, _ = cli(iris, capsys, "run", f)
    aid = json.loads(out)["analysis_id"]
    code, out, _ = cli(iris, capsys, "replay", aid)
    rep = json.loads(out)
    assert (
        code == 0
        and rep["deterministic"] is True
        and rep["differences"] == []
        and rep["replay_status"] == "COMPLETED"
    )
    code, out, _ = cli(iris, capsys, "list", "--dataset-id", iris.did)
    listing = json.loads(out)
    assert code == 0 and aid in {a["id"] for a in listing["analyses"]}
    code, out, _ = cli(iris, capsys, "inspect", aid)
    d = json.loads(out)
    assert (
        d["provenance"]["dataset"]["adapter"] == "sklearn"
        and d["provenance"]["run_provenance"]["environment_id"]
    )
    assert {a["path"] for a in d["artifacts"]} == {
        f"data_quality/{n}.json"
        for n in ("spec", "checks", "observations", "violations", "summary")
    }
    code, out, _ = cli(iris, capsys, "checks", aid, "--type", "target")
    rows = json.loads(out)["checks"]
    assert (
        code == 0
        and {r["scope"] for r in rows} == {"split=test", "split=train"}
        and all(r["check_type"] == "target" for r in rows)
    )
    code, out, _ = cli(iris, capsys, "inspect", rows[0]["id"])
    assert json.loads(out)["check_type"] == "target"
    code, out, _ = cli(iris, capsys, "checks", aid, "--status", "PASS", "--format", "text")
    assert code == 0 and "PASS" in out
    code, out, _ = cli(iris, capsys, "list", "--format", "text")
    assert aid in out


def test_the_catalog_and_validate_are_deterministic(
    iris: Iris, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code, out, _ = cli(iris, capsys, "checks")
    cat = json.loads(out)["check_types"]
    assert code == 0 and {
        "schema",
        "missingness",
        "duplicates",
        "target_leakage",
        "group_comparison",
        "temporal_order",
    } <= set(cat)
    f = write(tmp_path, "s.json", spec(iris))
    code, out, _ = cli(iris, capsys, "validate", f)
    d = json.loads(out)
    assert (
        code == 0
        and d["valid"] is True
        and d["spec_id"].startswith("dqs_")
        and len(d["checks"]) == 7
    )
    assert out == cli(iris, capsys, "validate", f)[1]  # identical input, identical bytes
    code, out, _ = cli(iris, capsys, "validate", f, "--preflight")
    pre = json.loads(out)["preflight"]
    _, splits = truth()
    assert pre["ok"] is True and {k: v["n_rows"] for k, v in pre["splits"].items()} == {
        k: len(v) for k, v in splits.items()
    }
    code, out, _ = cli(iris, capsys, "validate", f, "--format", "text")
    assert out.startswith("VALID: dqs_")


def test_help_and_exit_codes_reflect_findings(
    iris: Iris, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "data-quality" in text and "no quality score" in text
    failing = spec(iris, checks=[{"type": "sample_count", "config": {"min": 10_000}}])
    code, out, _ = cli(iris, capsys, "run", write(tmp_path, "f.json", failing))
    doc = json.loads(out)
    assert (
        code == 3 and doc["summary"]["status_counts"]["FAIL"] == 2
    )  # a configured rule failed on both splits
    ok = spec(iris, checks=[{"type": "sample_count", "config": {"min": 1}}])
    code, out, _ = cli(iris, capsys, "run", write(tmp_path, "ok.json", ok))
    assert code == 0 and json.loads(out)["analysis_status"] == "COMPLETE"


# -- invalid requests: clean errors, nothing executed ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("what", "body", "message"),
    [
        ("unknown-check", {"checks": [{"type": "vibes"}]}, "unknown check type"),
        ("bad-threshold", {"checks": [{"type": "missingness", "config": {"max_feature_rate": 9}}]}, "must be in"),
        ("bad-bounds", {"checks": [{"type": "numeric", "config": {"bounds": {FEATS[0]: [9, 1]}}}]}, "low <= high"),
        ("bad-type", {"features": [{"name": FEATS[0], "type": "TEXT"}]}, "unsupported feature type"),
        ("no-feature", {"features": [{"name": "ghost", "type": "NUMERIC"}]}, "not a column"),
        ("no-split", {"splits": ["ghost"]}, "not declared by the dataset"),
        ("no-target-decl", {"target": None}, "declared `target`"),
        ("leak-needs-names", {"checks": [{"type": "target_leakage", "config": {}}]}, "candidates"),
        ("unknown-key", {"surprise": 1}, "unexpected"),
        ("no-dataset", {"dataset_id": "dst_" + "0" * 32}, "not found|dst_"),
        ("bad-schema", {"quality_schema": 7}, "quality_schema"),
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
    code, out, err = cli(iris, capsys, "run", f)
    assert code == 2 and out == "" and err.startswith("error:") and "Traceback" not in err, (
        what,
        err,
    )
    assert n_runs(iris) == before
    code, out, err = cli(iris, capsys, "validate", f, "--preflight")
    assert code == 2 and out == "" and "Traceback" not in err


def test_unreadable_specs_unknown_ids_and_ambiguity_are_user_errors(
    iris: Iris, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    for bad in (
        write(tmp_path, "junk.json", "{not json"),
        write(tmp_path, "list.json", "[1]"),
        str(tmp_path / "missing.json"),
    ):
        code, out, err = cli(iris, capsys, "validate", bad)
        assert code == 2 and out == "" and err.startswith("error:") and "Traceback" not in err
    for args in (
        ("inspect", "qan_" + "0" * 32),
        ("inspect", "qck_" + "0" * 32),
        ("replay", "qan_" + "0" * 32),
        ("checks", "qan_" + "0" * 32),
    ):
        code, out, err = cli(iris, capsys, *args)
        assert (code == 2 and err.startswith("error:") and "Traceback" not in err) or (
            code == 0 and json.loads(out)["checks"] == []
        ), args
    f = write(tmp_path, "s.json", spec(iris))
    code, out, err = cli(iris, capsys, "run", f, "--investigation", "inv_" + "0" * 32)
    assert code == 2 and err.startswith("error:") and "Traceback" not in err
    code = main(["--workspace", str(tmp_path / "nowhere"), "data-quality", "list"])
    assert code == 2 and "no registry" in capsys.readouterr().err

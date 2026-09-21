"""The resources CLI and REAL-DATA engineering measurements: the bundled iris dataset with a real
scikit-learn logistic regression and the existing stripes dataset with a small PyTorch CNN, timed with
the real clock. What is checked is structure, identity, counts, sample digests recomputed independently
with numpy, output digests across batch sizes and worker counts, statuses, JSON determinism, replay and
clean errors. NO timing value is asserted or presented as a benchmark: every number produced here is an
environment-specific engineering measurement of this machine at this moment."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("sklearn")

from experionyx.adapters.records import RegisteredDataset, RegisteredModel
from experionyx.cli import main
from experionyx.demos import run_demo
from experionyx.domain import Run
from experionyx.hashing import content_hash
from experionyx.sqlite import SqliteRegistry


@dataclass
class Iris:
    ws: Path
    mid: str
    did: str


@pytest.fixture(scope="module")
def iris(tmp_path_factory: pytest.TempPathFactory) -> Iris:
    ws = tmp_path_factory.mktemp("iris-res") / "w"
    run_demo("sklearn-classification", ws)
    with SqliteRegistry(ws / "registry.sqlite") as reg:
        return Iris(ws, reg.find(RegisteredModel)[0].id, reg.find(RegisteredDataset)[0].id)


def cli(ws: Path, capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str, str]:
    code = main(["--workspace", str(ws), "resources", *args])
    out = capsys.readouterr()
    return code, out.out, out.err


def write(tmp_path: Path, name: str, body: Any) -> str:
    p = tmp_path / name
    p.write_text(body if isinstance(body, str) else json.dumps(body), encoding="utf-8")
    return str(p)


def spec(iris: Iris, **over: Any) -> dict[str, Any]:
    d: dict[str, Any] = {"model_id": iris.mid, "dataset_id": iris.did, "split": "test", "batch_size": 8, "repeats": 5, "warmup_trials": 1}  # fmt: skip
    d.update(over)
    return d


def n_runs(ws: Path) -> int:
    with SqliteRegistry(ws / "registry.sqlite") as reg:
        return len(reg.find(Run))


def run_ok(
    iris: Iris, tmp_path: Path, capsys: pytest.CaptureFixture[str], **over: Any
) -> dict[str, Any]:
    code, out, err = cli(iris.ws, capsys, "run", write(tmp_path, f"s{len(over)}{abs(hash(str(over)))}.json", spec(iris, **over)))  # fmt: skip
    assert code == 0, err
    doc: dict[str, Any] = json.loads(out)
    return doc


def iris_test_ids() -> list[int]:
    order = np.random.RandomState(0).permutation(150)
    return [int(i) for i in order[150 - round(150 * 0.25) :]]  # the dataset's own iteration order


# -- validate ------------------------------------------------------------------------------------------------------


def test_validate_prints_a_deterministic_identity_and_preflights_without_running(iris: Iris, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    f = write(tmp_path, "s.json", spec(iris))
    code, out, _ = cli(iris.ws, capsys, "validate", f)
    doc = json.loads(out)
    assert code == 0 and doc["valid"] is True and doc["spec_id"].startswith("rsp_") and doc["batch_size"] == 8  # fmt: skip
    assert "timing is environment-specific" in doc["note"] and cli(iris.ws, capsys, "validate", f)[1] == out  # fmt: skip
    before = n_runs(iris.ws)
    code, out, _ = cli(iris.ws, capsys, "validate", f, "--preflight")
    pre = json.loads(out)["preflight"]
    assert code == 0 and pre["samples_in_split"] == 38 and pre["model"] == "sklearn"
    caps = {c["capability"]: c["status"] for c in pre["capabilities"]}
    assert caps["operation:PREDICT"] == "SUPPORTED" and caps["memory_accounting"] in (
        "SUPPORTED",
        "UNAVAILABLE",
    )
    assert n_runs(iris.ws) == before  # nothing was executed
    code, out, _ = cli(iris.ws, capsys, "validate", f, "--format", "text")
    assert code == 0 and out.startswith("VALID: rsp_")
    other = json.loads(cli(iris.ws, capsys, "validate", write(tmp_path, "o.json", spec(iris, batch_size=9)))[1])  # fmt: skip
    assert other["spec_id"] != doc["spec_id"]


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"batch_size": 0}, "batch_size"), ({"batch_size": -3}, "batch_size"), ({"repeats": 0}, "repeats"),
        ({"workers": 0}, "workers"), ({"timeout_seconds": -1}, "timeout_seconds"), ({"operation": "TRAIN"}, "operation"),
        ({"bogus": 1}, "unexpected"), ({"model_id": "x"}, "model_id"), ({"measurement": {"percentiles": [120]}}, "percentiles"),
    ],
)  # fmt: skip
def test_invalid_specs_are_clean_errors_that_execute_nothing(iris: Iris, tmp_path: Path, capsys: pytest.CaptureFixture[str], over: dict[str, Any], match: str) -> None:  # fmt: skip
    before = n_runs(iris.ws)
    for cmd in ("validate", "run"):
        code, out, err = cli(iris.ws, capsys, cmd, write(tmp_path, "bad.json", spec(iris, **over)))
        assert code == 2 and out == "" and err.startswith("error:") and match in err and "Traceback" not in err  # fmt: skip
    assert n_runs(iris.ws) == before


def test_unreadable_spec_files_and_unsupported_requests_are_clean_errors(iris: Iris, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    before = n_runs(iris.ws)
    code, _, err = cli(iris.ws, capsys, "run", str(tmp_path / "missing.json"))
    assert code == 2 and "cannot read resource spec" in err
    code, _, err = cli(iris.ws, capsys, "run", write(tmp_path, "junk.json", "{not json"))
    assert code == 2 and "cannot read resource spec" in err
    code, _, err = cli(iris.ws, capsys, "run", write(tmp_path, "list.json", "[1]"))
    assert code == 2 and "must be a JSON object" in err
    code, _, err = cli(
        iris.ws, capsys, "run", write(tmp_path, "m.json", spec(iris, memory_limit_mb=256))
    )
    assert code == 2 and "UNAVAILABLE" in err and "memory_limit" in err
    code, _, err = cli(iris.ws, capsys, "run", write(tmp_path, "sp.json", spec(iris, split="nope")))
    assert code == 2 and "unknown split" in err
    code, _, err = cli(iris.ws, capsys, "validate", write(tmp_path, "sp2.json", spec(iris, workers=2)), "--preflight")  # fmt: skip
    assert code == 0  # sklearn declares thread-safe inference, so this is supported
    code = main(["--workspace", str(tmp_path / "nowhere"), "resources", "list"])
    assert code == 2 and "no registry" in capsys.readouterr().err
    assert n_runs(iris.ws) == before


# -- run, list, inspect, replay ---------------------------------------------------------------------------------


def test_a_real_sklearn_measurement_is_recorded_listed_inspected_and_replayed(iris: Iris, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    doc = run_ok(iris, tmp_path, capsys)
    assert doc["status"] == "COMPLETED" and doc["analysis_status"] == "COMPLETE" and "environment-specific" in doc["note"]  # fmt: skip
    s = doc["summary"]
    assert s["evidence_status"] == "MEASURED" and s["workload"]["samples"] == 38 and s["workload"]["batches"] == 5  # fmt: skip
    assert s["batch_size"] == {"requested": 8, "effective_min": 6, "effective_max": 8, "effective_mean": 38 / 5, "final_batch": 6}  # fmt: skip
    assert s["workload"]["sample_digest"] == content_hash({"ids": iris_test_ids()})  # the samples numpy says are 'test', in the dataset's order  # fmt: skip
    assert s["trials"]["used"] == 5 and s["trials"]["warmup"] == 1 and s["outputs"]["consistent_across_trials"] is True  # fmt: skip
    assert s["model"]["adapter"] == "sklearn" and s["environment"]["measurement_backends"]["gpu_memory"].startswith("UNAVAILABLE")  # fmt: skip
    for k in ("median", "mean", "min", "max"):
        assert s["steady_state"]["trial_seconds"][k] > 0
    assert math.isfinite(s["steady_state"]["throughput_samples_per_second"]["mean"])
    aid = doc["analysis_id"]
    code, out, _ = cli(iris.ws, capsys, "list")
    assert code == 0 and aid in [a["id"] for a in json.loads(out)["analyses"]]
    code, out, _ = cli(iris.ws, capsys, "list", "--spec-id", s["spec_id"], "--format", "text")
    assert code == 0 and aid in out
    code, out, _ = cli(iris.ws, capsys, "inspect", aid)
    ins = json.loads(out)
    assert code == 0 and len(ins["trials"]) == 6 and {t["phase"] for t in ins["trials"]} == {"WARMUP", "MEASURED"}  # fmt: skip
    prov = ins["provenance"]
    assert prov["run_provenance"]["seed"] == 0 and prov["model"]["model_id"] == iris.mid and prov["provenance_fingerprint"] == s["provenance_fingerprint"]  # fmt: skip
    assert len(prov["artifact_digests"]) == 6 and all(v.startswith("sha256:") for v in prov["artifact_digests"].values())  # fmt: skip
    code, out, _ = cli(iris.ws, capsys, "inspect", aid, "--document", "trials")
    raw = json.loads(out)["document"]["trials"]
    assert len(raw) == 6 and all(len(t["batch_seconds"]) == 5 for t in raw)  # raw per-batch timings are preserved  # fmt: skip
    code, out, _ = cli(iris.ws, capsys, "inspect", aid, "--full")
    assert code == 0 and set(json.loads(out)["documents"]) == {"spec", "environment", "trials", "observations", "statistics", "summary"}  # fmt: skip
    code, out, _ = cli(iris.ws, capsys, "inspect", ins["trials"][0]["id"])
    assert code == 0 and json.loads(out)["analysis_id"] == aid
    code, out, err = cli(iris.ws, capsys, "replay", aid)
    rep = json.loads(out)
    assert code == 0, err
    assert rep["definition_reproduced"] is True and rep["outputs_reproduced"] is True and rep["timing"] == "NOT_EXPECTED_TO_REPRODUCE"  # fmt: skip
    assert rep["replay_analysis"] != aid and rep["compared"] == ["spec", "outputs"]


def test_a_missing_analysis_is_a_clean_error(
    iris: Iris, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = cli(iris.ws, capsys, "inspect", "rsa_" + "0" * 32)
    assert code == 2 and out == "" and err.startswith("error:") and "Traceback" not in err
    code, _, err = cli(iris.ws, capsys, "replay", "rsa_" + "0" * 32)
    assert code == 2 and err.startswith("error:")


# -- sweep and compare (real batch sizes, real workers) ------------------------------------------------------------


def test_a_real_batch_size_sweep_persists_each_size_and_compares_with_a_correction(iris: Iris, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    before = n_runs(iris.ws)
    f = write(tmp_path, "sw.json", spec(iris, warmup_trials=1))
    code, out, err = cli(iris.ws, capsys, "sweep", f, "--batch-sizes", "1,8,38", "--correction", "BENJAMINI_HOCHBERG")  # fmt: skip
    doc = json.loads(out)
    assert code == 0, err
    assert n_runs(iris.ws) == before + 3 and [u["batch_size"] for u in doc["units"]] == [1, 8, 38]
    assert len({u["analysis_id"] for u in doc["units"]}) == 3 and {u["status"] for u in doc["units"]} == {"COMPLETED"}  # fmt: skip
    c = doc["comparison"]
    assert c["reference"] == doc["units"][0]["analysis_id"] and c["pairing"] == "UNPAIRED" and c["correction"]["method"] == "BENJAMINI_HOCHBERG"  # fmt: skip
    assert len(c["comparisons"]) == 2 and all(r["differs_from_reference"] == ["batch_size"] for r in c["comparisons"])  # fmt: skip
    digests = set()
    for u in doc["units"]:
        ins = json.loads(cli(iris.ws, capsys, "inspect", u["analysis_id"])[1])
        digests.add(ins["summary"]["outputs"]["digest"])
        assert ins["summary"]["batch_size"]["requested"] == u["batch_size"]
    assert len(digests) == 1  # a real sklearn model gives the same outputs at every batch size
    ids = [u["analysis_id"] for u in doc["units"]]
    code, out, _ = cli(iris.ws, capsys, "compare", *ids, "--metric", "trial_seconds")
    assert code == 0 and json.loads(out)["metric"] == "trial_seconds"
    first = cli(iris.ws, capsys, "compare", *ids)[1]
    assert cli(iris.ws, capsys, "compare", *ids)[1] == first  # deterministic given the seed


def test_an_invalid_sweep_or_comparison_executes_and_stores_nothing(iris: Iris, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    before = n_runs(iris.ws)
    f = write(tmp_path, "sw.json", spec(iris))
    for sizes, match in (("8,0", "batch_size"), ("8,-2", "batch_size"), ("8,8", "distinct"), ("8,x", "comma-separated"), ("", "comma-separated")):  # fmt: skip
        code, out, err = cli(iris.ws, capsys, "sweep", f, "--batch-sizes", sizes)
        assert code == 2 and out == "" and match in err and "Traceback" not in err, sizes
    code, _, err = cli(iris.ws, capsys, "compare", "rsa_" + "0" * 32)
    assert code == 2 and err.startswith("error:")
    code, _, err = cli(iris.ws, capsys, "compare", "rsa_" + "0" * 32, "rsa_" + "1" * 32)
    assert code == 2 and err.startswith("error:")
    assert n_runs(iris.ws) == before


def test_real_workers_are_supported_for_sklearn_and_preserve_the_outputs(iris: Iris, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    one = run_ok(iris, tmp_path, capsys, workers=1, batch_size=4)
    three = run_ok(iris, tmp_path, capsys, workers=3, batch_size=4)
    assert three["summary"]["concurrency"]["workers"] == 3 and three["summary"]["concurrency"]["adapter_declared_thread_safe"] is True  # fmt: skip
    assert three["summary"]["outputs"]["digest"] == one["summary"]["outputs"]["digest"] and three["analysis_status"] == "COMPLETE"  # fmt: skip
    assert three["summary"]["concurrency"]["trial_failures"] == 0


def test_a_real_timeout_is_a_timed_out_run_not_a_success(iris: Iris, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    code, out, err = cli(iris.ws, capsys, "run", write(tmp_path, "t.json", spec(iris, timeout_seconds=1e-6, batch_size=2)))  # fmt: skip
    doc = json.loads(out)
    assert code == 3, err  # the run finished but its evidence is not complete
    assert doc["status"] == "COMPLETED" and doc["analysis_status"] == "PARTIAL" and doc["summary"]["evidence_status"] == "INSUFFICIENT_EVIDENCE"  # fmt: skip
    assert doc["summary"]["failures"]["timed_out"] == 5 and doc["summary"]["trials"]["used"] == 0
    assert doc["summary"]["steady_state"]["status"] == "UNAVAILABLE"


def test_the_measurement_is_labelled_environment_specific_everywhere(iris: Iris, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    doc = run_ok(iris, tmp_path, capsys, repeats=5, seed=2)
    assert "environment-specific" in doc["note"]
    lim = " ".join(doc["summary"]["limitations"])
    for phrase in (
        "not a universal hardware benchmark",
        "not independent samples",
        "high-water mark",
        "cooperative",
    ):
        assert phrase in lim


# -- a real PyTorch CNN -------------------------------------------------------------------------------------------


def test_a_real_torch_cnn_workload_is_measured_with_batch_and_worker_variation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    pytest.importorskip("torch")
    ws = tmp_path / "w"
    run_demo("torch-classification", ws)
    with SqliteRegistry(ws / "registry.sqlite") as reg:
        mid, did = reg.find(RegisteredModel)[0].id, reg.find(RegisteredDataset)[0].id
    base = {"model_id": mid, "dataset_id": did, "batch_size": 4, "repeats": 5, "warmup_trials": 1}
    code, out, err = cli(ws, capsys, "validate", write(tmp_path, "v.json", base), "--preflight")
    assert code == 0, err
    n = json.loads(out)["preflight"]["samples_in_split"]
    docs = []
    for name, over in (("seq", {}), ("big", {"batch_size": 16}), ("par", {"workers": 2})):
        code, out, err = cli(ws, capsys, "run", write(tmp_path, f"{name}.json", {**base, **over}))
        assert code == 0, err
        docs.append(json.loads(out)["summary"])
    assert {d["workload"]["samples"] for d in docs} == {n} and {
        d["model"]["adapter"] for d in docs
    } == {"torch"}
    assert len({d["outputs"]["digest"] for d in docs}) == 1 and all(
        d["outputs"]["consistent_across_trials"] for d in docs
    )
    assert (
        docs[2]["concurrency"]["adapter_declared_thread_safe"] is True
        and docs[1]["batch_size"]["requested"] == 16
    )

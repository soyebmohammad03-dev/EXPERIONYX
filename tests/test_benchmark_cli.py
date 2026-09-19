"""The benchmark CLI on a real sklearn workspace (real model, real dataset, real fault runs)."""

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sklearn")

from experionyx.adapters.records import RegisteredDataset, RegisteredModel
from experionyx.benchmark.entities import Benchmark, BenchmarkResult
from experionyx.cli import main
from experionyx.domain import Run
from experionyx.sqlite import SqliteRegistry


class Ws:
    def __init__(self, path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        self.path, self.capsys = path, capsys
        assert main(["--workspace", str(path), "demo", "sklearn-classification"]) == 0
        capsys.readouterr()
        with SqliteRegistry(path / "registry.sqlite") as reg:
            self.model = reg.find(RegisteredModel)[0].id
            self.dataset = reg.find(RegisteredDataset)[0].id

    def cli(self, *args: str) -> tuple[int, str, str]:
        code = main(["--workspace", str(self.path), *args])
        out = self.capsys.readouterr()
        return code, out.out, out.err

    def spec(self, fname: str, **over: Any) -> str:
        body: dict[str, Any] = {
            "name": "iris-noise",
            "version": "1.0.0",
            "model": self.model,
            "dataset": self.dataset,
            "faults": [
                {
                    "name": "noise",
                    "fault": "gaussian_noise",
                    "sweep_parameter": "sigma",
                    "values": [0.5, 2.0],
                }
            ],
            "seeds": [1, 2, 3],
            "aggregation": {"resamples": 200},
        }
        body.update(over)
        f = self.path / f"{fname}.json"
        f.write_text(json.dumps(body), encoding="utf-8")
        return str(f)

    def counts(self) -> tuple[int, int, int]:
        with SqliteRegistry(self.path / "registry.sqlite") as reg:
            return len(reg.find(Run)), len(reg.find(Benchmark)), len(reg.find(BenchmarkResult))


@pytest.fixture
def ws(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Ws:
    return Ws(tmp_path / "ws", capsys)


def test_validate_run_list_inspect_coverage_replay(ws: Ws) -> None:
    spec = ws.spec("v1")
    code, out, _ = ws.cli("benchmark", "validate", spec)
    rep = json.loads(out)
    assert (
        code == 0
        and rep["valid"]
        and rep["units"] == 1 + 6
        and rep["units_by_kind"] == {"BASELINE": 1, "FAULT_TRIAL": 6}
    )
    text = ws.cli("benchmark", "validate", spec, "--format", "text")[1]
    assert "VALID" in text and "7 experiment unit(s)" in text
    runs_before = ws.counts()[0]
    assert (
        ws.cli("benchmark", "validate", spec)[0] == 0 and ws.counts()[0] == runs_before
    )  # validate never writes

    code, out, _ = ws.cli("benchmark", "run", spec)
    doc = json.loads(out)
    assert (
        code == 0
        and doc["status"] == "COMPLETED"
        and doc["already_run"] is False
        and doc["result"]["coverage_status"] == "COMPLETE"
    )
    bid, rid = doc["benchmark_id"], doc["result_id"]
    code, out, _ = ws.cli("benchmark", "run", spec)
    assert (
        code == 0 and json.loads(out)["already_run"] is True and json.loads(out)["result_id"] == rid
    )  # idempotent

    listed = json.loads(ws.cli("benchmark", "list")[1])["benchmarks"]
    assert [b["id"] for b in listed] == [bid]
    assert [
        b["id"]
        for b in json.loads(
            ws.cli(
                "benchmark",
                "list",
                "--fault",
                "gaussian_noise",
                "--coverage",
                "COMPLETE",
                "--name",
                "iris-noise",
            )[1]
        )["benchmarks"]
    ] == [bid]
    assert (
        json.loads(ws.cli("benchmark", "list", "--fault", "salt_and_pepper")[1])["benchmarks"] == []
    )
    assert "iris-noise" in ws.cli("benchmark", "list", "--format", "text")[1]

    b = json.loads(ws.cli("benchmark", "inspect", bid)[1])
    assert b["spec"]["faults"][0]["name"] == "noise" and [r["id"] for r in b["results"]] == [rid]
    r = json.loads(ws.cli("benchmark", "inspect", rid)[1])
    assert (
        r["coverage_status"] == "COMPLETE"
        and r["summary"]["no_score"].startswith("no overall score")
        and {a["path"] for a in r["artifacts"]}
        == {f"benchmark/{n}.json" for n in ("spec", "units", "coverage", "results", "summary")}
    )
    full = json.loads(ws.cli("benchmark", "inspect", rid, "--full")[1])["documents"]
    assert (
        set(full) == {"spec", "units", "coverage", "results", "summary"}
        and len(full["units"]["units"]) == 7
    )
    assert (
        "sections (no overall score)" in ws.cli("benchmark", "inspect", rid, "--format", "text")[1]
    )

    code, out, _ = ws.cli(
        "benchmark", "coverage", bid
    )  # a benchmark with one result resolves to it
    cov = json.loads(out)["coverage"]
    assert (
        code == 0
        and cov["complete"]
        and cov["trials"]["completed"] == 6
        and cov["fault_families"]["tested"] == ["noise"]
    )
    assert "COMPLETE" in ws.cli("benchmark", "coverage", rid, "--format", "text")[1]

    code, out, _ = ws.cli("benchmark", "replay", rid)
    rp = json.loads(out)
    assert code == 0 and rp["deterministic"] is True and rp["differences"] == []


def test_invalid_definitions_are_refused_before_execution_with_every_issue(ws: Ws) -> None:
    bad = ws.spec("bad", primary_metric="nope", faults=[{"name": "x", "fault": "no_such_fault"}])
    code, out, _ = ws.cli("benchmark", "validate", bad)
    rep = json.loads(out)
    assert (
        code == 1
        and rep["valid"] is False
        and {i["code"] for i in rep["issues"]} == {"UNKNOWN_METRIC", "UNKNOWN_FAULT"}
    )
    assert all(i["required"] and i["found"] and i["why"] for i in rep["issues"])
    assert "REFUSED" in ws.cli("benchmark", "validate", bad, "--format", "text")[1]
    before = ws.counts()
    code, _, err = ws.cli("benchmark", "run", bad)
    assert (
        code == 2
        and "nothing was executed and nothing was recorded" in err
        and "UNKNOWN_FAULT" in err
    )
    assert ws.counts() == before
    for name, body, needle in (
        ("nofaults.json", {"faults": []}, "at least one"),
        ("badver.json", {"version": "one"}, "major.minor.patch"),
        ("extra.json", {"surprise": 1}, "unexpected"),
    ):
        code, _, err = ws.cli("benchmark", "validate", ws.spec(name.removesuffix(".json"), **body))
        assert code == 2 and needle in err, (name, err)
    assert ws.cli("benchmark", "validate", str(ws.path / "missing.json"))[0] == 2
    (ws.path / "arr.json").write_text("[1]", encoding="utf-8")
    assert ws.cli("benchmark", "validate", str(ws.path / "arr.json"))[0] == 2
    assert (
        ws.cli("benchmark", "inspect", "bmk_" + "0" * 32)[0] == 2
        and ws.cli("benchmark", "inspect", "brs_" + "0" * 32)[0] == 2
    )


def test_incomplete_coverage_is_reported_with_exit_code_3(ws: Ws) -> None:
    spec = ws.spec(
        "incomplete",
        name="iris-with-unsupported",
        faults=[
            {
                "name": "noise",
                "fault": "gaussian_noise",
                "sweep_parameter": "sigma",
                "values": [1.0],
            },
            {
                "name": "occ",
                "fault": "random_occlusion",
                "sweep_parameter": "fraction",
                "values": [0.5],
            },
        ],
        limits={"allow_unsupported": True},
        discovery=False,
        profile=False,
    )
    code, out, _ = ws.cli("benchmark", "run", spec)
    doc = json.loads(out)
    assert (
        code == 3
        and doc["status"] == "COMPLETED"
        and doc["result"]["coverage_status"] == "INCOMPLETE"
    )  # ran, but the coverage is incomplete: never exit 0
    rid = doc["result_id"]
    code, out, _ = ws.cli("benchmark", "coverage", rid)
    cov = json.loads(out)["coverage"]
    assert (
        code == 3
        and cov["unsupported_combinations"][0]["grid"] == "occ"
        and any("unsupported" in r for r in cov["incomplete_reasons"])
    )
    text = ws.cli("benchmark", "coverage", rid, "--format", "text")[1]
    assert "INCOMPLETE" in text and "not a robustness measure" in text and "! " in text


def test_compare_needs_an_identical_protocol_and_reports_no_winner(ws: Ws) -> None:
    a = json.loads(ws.cli("benchmark", "run", ws.spec("a"))[1])["result_id"]
    same_protocol_other_run = ws.spec("a2")  # identical definition: idempotent, same result
    assert json.loads(ws.cli("benchmark", "run", same_protocol_other_run)[1])["result_id"] == a
    b = json.loads(ws.cli("benchmark", "run", ws.spec("b", version="1.0.1"))[1])["result_id"]
    code, _, err = ws.cli("benchmark", "compare", a, b)
    assert (
        code == 2 and "not comparable" in err and "PROTOCOL_DIFFERS" in err and "version" in err
    )  # a changed definition is detected and explained
    code, out, _ = ws.cli("benchmark", "compare", a, a)
    cmp = json.loads(out)
    assert (
        code == 0
        and cmp["same_model"] is True
        and cmp["note"].startswith("raw differences")
        and all(m["difference"] == 0 for m in cmp["baseline"]["metrics"])
    )
    assert "no winner" in ws.cli("benchmark", "compare", a, a, "--format", "text")[1]
    blob = (
        json.dumps(cmp)
        .lower()
        .replace("higher_is_better", "")
        .replace("no overall ranking, winner or verdict is computed", "")
    )
    for word in ("winner", "better", "worse", "best", "worst", "rank", "verdict"):
        assert word not in blob

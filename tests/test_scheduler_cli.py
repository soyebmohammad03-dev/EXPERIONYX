"""The scheduler CLI on a real workspace: validate, expand, run, status, inspect, graph, resume,
cancel, retry, replay -- JSON and text output, deterministic output, and clean errors for invalid
requests."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from experionyx.cli import main
from experionyx.domain import Investigation
from experionyx.sqlite import SqliteRegistry


class Ws:
    def __init__(self, path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        self.path, self.capsys = path, capsys
        assert main(["--workspace", str(path), "init"]) == 0
        capsys.readouterr()
        with SqliteRegistry(path / "registry.sqlite") as reg:
            inv = Investigation("cli-e2e", "does the scheduler CLI work?", datetime.now(UTC))
            reg.add(inv)
            self.investigation = inv.id

    def cli(self, *args: str) -> tuple[int, str, str]:
        code = main(["--workspace", str(self.path), *args])
        out = self.capsys.readouterr()
        return code, out.out, out.err

    def spec(self, fname: str, units: list[dict[str, Any]], **over: Any) -> str:
        body: dict[str, Any] = {"name": fname, "version": "1.0.0", "units": units}
        body.update(over)
        f = self.path / f"{fname}.json"
        f.write_text(json.dumps(body), encoding="utf-8")
        return str(f)


def bootstrap_unit(key: str, values: list[float], depends_on: list[str] | None = None) -> dict[str, Any]:  # fmt: skip
    return {
        "key": key,
        "kind": "STATISTICAL_ANALYSIS",
        "parameters": {"kind": "BOOTSTRAP", "sources": {"kind": "inline", "reference": values}},
        "depends_on": depends_on or [],
    }


@pytest.fixture
def ws(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Ws:
    return Ws(tmp_path / "ws", capsys)


def test_validate_accepts_a_good_spec_and_refuses_a_cyclic_one(ws: Ws) -> None:
    good = ws.spec("good", [bootstrap_unit("a", [1, 2, 3])])
    code, out, _ = ws.cli("scheduler", "validate", good)
    report = json.loads(out)
    assert code == 0 and report["valid"] is True and report["order"] == ["a"]

    cyclic = ws.spec(
        "cyclic",
        [bootstrap_unit("a", [1], depends_on=["b"]), bootstrap_unit("b", [1], depends_on=["a"])],
    )
    code, out, _ = ws.cli("scheduler", "validate", cyclic)
    report = json.loads(out)
    assert code == 1 and report["valid"] is False and report["issues"][0]["code"] == "CYCLE"


def test_expand_materializes_units_in_dependency_order(ws: Ws) -> None:
    spec = ws.spec("expand", [bootstrap_unit("a", [1, 2, 3]), bootstrap_unit("b", [4], ["a"])])
    code, out, _ = ws.cli("scheduler", "expand", spec)
    doc = json.loads(out)
    assert code == 0
    assert doc["order"] == ["a", "b"]
    assert len(doc["units"]) == 2


def test_run_executes_the_dag_and_status_reflects_it(ws: Ws) -> None:
    spec = ws.spec("run", [bootstrap_unit("a", [1, 2, 3]), bootstrap_unit("b", [4], ["a"])])
    code, out, _ = ws.cli("scheduler", "run", spec, "--investigation", ws.investigation)
    doc = json.loads(out)
    assert code == 0 and doc["status"] == "COMPLETED" and doc["counts"] == {"SUCCEEDED": 2}
    sched_id = doc["schedule_id"]

    code, out, _ = ws.cli("scheduler", "status", sched_id)
    status = json.loads(out)
    assert code == 0
    assert status["counts"] == {"SUCCEEDED": 2}
    assert {u["key"] for u in status["units"]} == {"a", "b"}
    assert all(u["status"] == "SUCCEEDED" for u in status["units"])


def test_run_dry_run_executes_zero_experiments(ws: Ws) -> None:
    spec = ws.spec("dry", [bootstrap_unit("a", [1, 2, 3])])
    code, out, _ = ws.cli(
        "scheduler", "run", spec, "--investigation", ws.investigation, "--dry-run"
    )
    doc = json.loads(out)
    assert code == 0 and doc["dry_run"] is True
    sched_id = doc["schedule_id"]
    code, out, _ = ws.cli("scheduler", "status", sched_id)
    status = json.loads(out)
    assert status["counts"] == {"PLANNED": 1}  # nothing ran


def test_resume_is_idempotent_for_already_succeeded_units(ws: Ws) -> None:
    spec = ws.spec("resume", [bootstrap_unit("a", [1, 2, 3])])
    code, out, _ = ws.cli("scheduler", "run", spec, "--investigation", ws.investigation)
    sched_id = json.loads(out)["schedule_id"]
    code, out, _ = ws.cli("scheduler", "resume", sched_id, "--investigation", ws.investigation)
    doc = json.loads(out)
    assert code == 0 and doc["status"] == "COMPLETED" and doc["counts"] == {"SUCCEEDED": 1}
    assert doc["schedule_id"] == sched_id


def test_inspect_and_graph_show_the_definition_and_dependencies(ws: Ws) -> None:
    spec = ws.spec("graph", [bootstrap_unit("a", [1, 2, 3]), bootstrap_unit("b", [4], ["a"])])
    code, out, _ = ws.cli("scheduler", "run", spec, "--investigation", ws.investigation)
    sched_id = json.loads(out)["schedule_id"]

    code, out, _ = ws.cli("scheduler", "inspect", sched_id)
    doc = json.loads(out)
    assert code == 0 and doc["name"] == "graph"

    code, out, _ = ws.cli("scheduler", "graph", sched_id)
    doc = json.loads(out)
    assert code == 0 and doc["edges"] == [["a", "b"]]


def test_cancel_marks_open_units_and_preserves_completed_evidence(ws: Ws) -> None:
    spec = ws.spec("cancel", [bootstrap_unit("a", [1, 2, 3])])
    code, out, _ = ws.cli("scheduler", "expand", spec)  # materialize only, do not execute
    sched_id = json.loads(out)["schedule_id"]
    code, out, _ = ws.cli("scheduler", "cancel", sched_id)
    doc = json.loads(out)
    assert code == 0 and doc["cancelled_from"] == {"PLANNED": 1}
    code, out, _ = ws.cli("scheduler", "status", sched_id)
    assert json.loads(out)["counts"] == {"CANCELLED": 1}


def test_retry_resets_a_failed_unit(ws: Ws) -> None:
    bad = {"key": "bad", "kind": "FAULT_EXPERIMENT", "parameters": {}}
    spec = ws.spec("retry", [bad])
    code, out, _ = ws.cli("scheduler", "run", spec, "--investigation", ws.investigation)
    doc = json.loads(out)
    assert code == 1 and doc["status"] == "FAILED"
    sched_id = doc["schedule_id"]
    code, out, _ = ws.cli("scheduler", "retry", sched_id, "--unit", "bad")
    doc = json.loads(out)
    assert code == 0 and doc["status"] == "RETRY_PENDING"
    code, _out, _err = ws.cli("scheduler", "retry", sched_id, "--unit", "does-not-exist")
    assert code == 2  # a clean, non-crashing error for an invalid request


def test_replay_reports_zero_differences_for_a_deterministic_schedule(ws: Ws) -> None:
    spec = ws.spec("replay", [bootstrap_unit("a", [1, 2, 3])])
    code, out, _ = ws.cli("scheduler", "run", spec, "--investigation", ws.investigation)
    sched_id = json.loads(out)["schedule_id"]
    code, out, _ = ws.cli("scheduler", "replay", sched_id)
    doc = json.loads(out)
    assert code == 0 and doc["differences"] == {}


def test_text_format_is_human_readable(ws: Ws) -> None:
    spec = ws.spec("text", [bootstrap_unit("a", [1, 2, 3])])
    code, out, _ = ws.cli(
        "scheduler", "run", spec, "--investigation", ws.investigation, "--format", "text"
    )
    assert code == 0
    assert "COMPLETED" in out
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)  # text output is not JSON


def test_validate_output_is_deterministic_across_repeated_runs(ws: Ws) -> None:
    spec = ws.spec("det", [bootstrap_unit("b", [1]), bootstrap_unit("a", [2])])
    _, out1, _ = ws.cli("scheduler", "validate", spec)
    _, out2, _ = ws.cli("scheduler", "validate", spec)
    assert out1 == out2


def test_run_on_an_unreadable_spec_file_is_a_clean_error(ws: Ws) -> None:
    code, _, err = ws.cli("scheduler", "run", str(ws.path / "missing.json"))
    assert code == 2 and "cannot read schedule spec" in err

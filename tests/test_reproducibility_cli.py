"""The reproduce CLI on a real workspace: validate, run, inspect, compare, verify, replay, diff
-- JSON and text output, and clean errors for invalid requests."""

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
            inv = Investigation("cli-reproduce", "does the reproduce CLI work?", datetime.now(UTC))  # fmt: skip
            reg.add(inv)
            self.investigation = inv.id

    def cli(self, *args: str) -> tuple[int, str, str]:
        code = main(["--workspace", str(self.path), *args])
        out = self.capsys.readouterr()
        return code, out.out, out.err

    def cli_json(self, *args: str) -> Any:
        _code, out, err = self.cli(*args)
        assert out, f"expected JSON output, got empty stdout (stderr: {err})"
        return json.loads(out)


@pytest.fixture
def ws(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Ws:
    return Ws(tmp_path / "ws", capsys)


def _build_snapshot(ws: Ws) -> str:
    spec = ws.path / "g.json"
    spec.write_text(json.dumps({"name": "g", "version": "1.0.0"}), encoding="utf-8")
    doc = ws.cli_json("graph", "build", str(spec), "--investigation", ws.investigation)
    snap_id = doc["snapshot_id"]
    assert isinstance(snap_id, str)
    return snap_id


def test_validate_reports_a_real_target_as_valid(ws: Ws) -> None:
    snap_id = _build_snapshot(ws)
    doc = ws.cli_json("reproduce", "validate", "GRAPH_SNAPSHOT", snap_id)
    assert doc["valid"] is True
    assert doc["investigation_id"] == ws.investigation


def test_validate_reports_an_unknown_target_as_invalid(ws: Ws) -> None:
    code, out, _err = ws.cli("reproduce", "validate", "GRAPH_SNAPSHOT", "gsn_" + "9" * 32)
    assert code == 1
    assert json.loads(out)["valid"] is False


def test_run_deterministic_on_unchanged_evidence_is_equal_and_exits_zero(ws: Ws) -> None:
    snap_id = _build_snapshot(ws)
    code, out, _err = ws.cli(
        "reproduce", "run", "GRAPH_SNAPSHOT", snap_id, "--mode", "DETERMINISTIC", "--format", "text"
    )
    assert code == 0
    assert "EQUAL" in out


def test_run_persists_an_attempt_inspectable_afterward(ws: Ws) -> None:
    snap_id = _build_snapshot(ws)
    doc = ws.cli_json("reproduce", "run", "GRAPH_SNAPSHOT", snap_id, "--mode", "EXACT")
    attempt_id = doc["id"]
    inspected = ws.cli_json("reproduce", "inspect", attempt_id)
    assert inspected["id"] == attempt_id
    assert inspected["outcome"] == "EQUAL"


def test_repeated_run_increments_attempt_number(ws: Ws) -> None:
    snap_id = _build_snapshot(ws)
    first = ws.cli_json("reproduce", "run", "GRAPH_SNAPSHOT", snap_id, "--mode", "DETERMINISTIC")
    second = ws.cli_json("reproduce", "run", "GRAPH_SNAPSHOT", snap_id, "--mode", "DETERMINISTIC")
    assert first["attempt"] == 0
    assert second["attempt"] == 1
    assert first["id"] != second["id"]


def test_provenance_only_mode_never_re_executes(ws: Ws) -> None:
    snap_id = _build_snapshot(ws)
    doc = ws.cli_json("reproduce", "run", "GRAPH_SNAPSHOT", snap_id, "--mode", "PROVENANCE_ONLY")
    assert doc["replay_run_id"] is None
    assert doc["outcome"] == "EQUAL"


def test_verify_a_clean_run_reports_no_corruption(ws: Ws) -> None:
    snap_id = _build_snapshot(ws)
    inspected = ws.cli_json("graph", "inspect", snap_id)
    run_id = inspected["run_id"]
    code, _out, _err = ws.cli("reproduce", "verify", run_id)
    assert code == 0
    doc = ws.cli_json("reproduce", "verify", run_id)
    assert doc["corrupted"] == []
    assert doc["checked"] >= 1


def test_replay_creates_a_new_attempt_referencing_the_prior_one(ws: Ws) -> None:
    snap_id = _build_snapshot(ws)
    first = ws.cli_json("reproduce", "run", "GRAPH_SNAPSHOT", snap_id, "--mode", "DETERMINISTIC")
    replayed = ws.cli_json("reproduce", "replay", first["id"])
    assert replayed["id"] != first["id"]
    assert replayed["target_id"] == snap_id
    assert replayed["attempt"] == 1


def test_diff_between_two_attempts_of_the_same_target(ws: Ws) -> None:
    snap_id = _build_snapshot(ws)
    a = ws.cli_json("reproduce", "run", "GRAPH_SNAPSHOT", snap_id, "--mode", "DETERMINISTIC")
    b = ws.cli_json("reproduce", "run", "GRAPH_SNAPSHOT", snap_id, "--mode", "DETERMINISTIC")
    doc = ws.cli_json("reproduce", "diff", a["id"], b["id"])
    assert doc["outcome_changed"] is False
    assert doc["a"]["id"] == a["id"] and doc["b"]["id"] == b["id"]


def test_compare_identical_artifacts_is_equal(ws: Ws) -> None:
    snap_id = _build_snapshot(ws)
    inspected = ws.cli_json("graph", "inspect", snap_id)
    run_id = inspected["run_id"]
    doc = ws.cli_json(
        "reproduce", "compare", run_id, "graph/summary.json", run_id, "graph/summary.json"
    )
    assert doc["outcome"] == "EQUAL"


def test_schedule_target_without_investigation_is_a_clean_error(ws: Ws) -> None:
    code, _out, err = ws.cli("reproduce", "run", "SCHEDULE", "sch_" + "1" * 32, "--mode", "DETERMINISTIC")  # fmt: skip
    assert code == 2
    assert "investigation" in err

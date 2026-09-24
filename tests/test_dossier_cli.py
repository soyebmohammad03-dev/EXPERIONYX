"""The dossier CLI on a real workspace: build, list, show, items, findings, snapshot, validate,
export -- JSON output and non-zero exit on unresolved evidence (see docs/dossier.md)."""

import json
from pathlib import Path
from typing import Any

import pytest

from experionyx.cli import main
from experionyx.domain import (
    ConfigurationRef,
    DatasetRef,
    Experiment,
    ExperimentStatus,
    Investigation,
    ModelRef,
)
from experionyx.sqlite import SqliteRegistry


class Ws:
    def __init__(self, path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        self.path, self.capsys = path, capsys
        assert main(["--workspace", str(path), "init"]) == 0
        capsys.readouterr()
        with SqliteRegistry(path / "registry.sqlite") as reg:
            from datetime import UTC, datetime

            now = datetime.now(UTC)
            inv = Investigation("cli-dossier", "does the dossier CLI work?", now)
            cfg = ConfigurationRef({"lr": 0.1})
            exp = Experiment(
                inv.id, "baseline", "h", ModelRef("m", "1"), DatasetRef("d", "1"), cfg.id, now
            )
            reg.add(inv)
            reg.add(cfg)
            reg.add(exp)
            reg.update_status(exp.with_status(ExperimentStatus.READY))
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


def _build(ws: Ws) -> Any:
    return ws.cli_json(
        "dossier",
        "build",
        "--investigation",
        ws.investigation,
        "--question",
        "does this reproduce?",
    )


def test_build_creates_a_dossier_and_list_finds_it(ws: Ws) -> None:
    doc = _build(ws)
    assert doc["id"].startswith("dsr_")
    assert doc["investigation_id"] == ws.investigation
    listed = ws.cli_json("dossier", "list", "--investigation", ws.investigation)
    assert [d["id"] for d in listed] == [doc["id"]]


def test_build_is_idempotent_against_unchanged_evidence(ws: Ws) -> None:
    first = _build(ws)
    second = _build(ws)
    assert first["id"] == second["id"]
    assert second["warnings"]


def test_show_returns_the_dossier_and_items_lists_evidence_lineage(ws: Ws) -> None:
    doc = _build(ws)
    shown = ws.cli_json("dossier", "show", doc["id"])
    assert shown["id"] == doc["id"]
    items = ws.cli_json("dossier", "items", doc["id"])
    assert any(i["source_kind"] == "experiment" for i in items)


def test_findings_surfaces_missing_evidence_gaps(ws: Ws) -> None:
    doc = _build(ws)
    findings = ws.cli_json("dossier", "findings", doc["id"])
    assert any(f["status"] == "MISSING_EXPECTED_EVIDENCE" for f in findings)


def test_validate_lists_unresolved_findings_but_exits_zero_without_unavailable_evidence(
    ws: Ws,
) -> None:
    doc = _build(ws)
    code, out, _err = ws.cli("dossier", "validate", doc["id"])
    assert code == 0
    assert json.loads(out)  # gaps exist for an untouched investigation, still exit 0


def test_snapshot_is_immutable_and_idempotent_through_the_cli(ws: Ws) -> None:
    doc = _build(ws)
    first = ws.cli_json("dossier", "snapshot", doc["id"])
    second = ws.cli_json("dossier", "snapshot", doc["id"])
    assert first["id"] == second["id"]
    assert first["id"].startswith("dsn_")


def test_export_markdown_contains_the_dossier_id_and_research_question(ws: Ws) -> None:
    doc = _build(ws)
    code, out, _err = ws.cli("dossier", "export", doc["id"])
    assert code == 0
    assert doc["id"] in out
    assert "does this reproduce?" in out


def test_build_with_an_unknown_investigation_errors_cleanly(ws: Ws) -> None:
    code, _out, err = ws.cli(
        "dossier",
        "build",
        "--investigation",
        "inv_" + "9" * 32,
        "--question",
        "q",
    )
    assert code == 2
    assert "not found" in err

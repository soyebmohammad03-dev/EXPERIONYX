"""The report CLI on a real workspace: generate, list, show, sections, claims, validate, export
-- JSON output and non-zero exit on validation failure (see docs/reporting.md)."""

import json
from pathlib import Path
from typing import Any

import pytest

from experionyx.cli import main
from experionyx.domain import ConfigurationRef, Experiment, ExperimentStatus, Investigation
from experionyx.sqlite import SqliteRegistry


class Ws:
    def __init__(self, path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        self.path, self.capsys = path, capsys
        assert main(["--workspace", str(path), "init"]) == 0
        capsys.readouterr()
        with SqliteRegistry(path / "registry.sqlite") as reg:
            from datetime import UTC, datetime

            now = datetime.now(UTC)
            inv = Investigation("cli-report", "does the report CLI work?", now)
            cfg = ConfigurationRef({"lr": 0.1})
            from experionyx.domain import DatasetRef, ModelRef

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


def test_generate_creates_a_report_and_list_finds_it(ws: Ws) -> None:
    doc = ws.cli_json(
        "report", "generate", "INVESTIGATION_DOSSIER", "--investigation", ws.investigation
    )
    assert doc["id"].startswith("rpt_")
    assert doc["investigation_id"] == ws.investigation
    listed = ws.cli_json("report", "list", "--investigation", ws.investigation)
    assert [r["id"] for r in listed] == [doc["id"]]


def test_generate_is_idempotent_against_unchanged_evidence(ws: Ws) -> None:
    first = ws.cli_json(
        "report", "generate", "INVESTIGATION_DOSSIER", "--investigation", ws.investigation
    )
    second = ws.cli_json(
        "report", "generate", "INVESTIGATION_DOSSIER", "--investigation", ws.investigation
    )
    assert first["id"] == second["id"]
    assert second["warnings"]


def test_show_returns_the_full_report_and_sections_lists_every_section(ws: Ws) -> None:
    doc = ws.cli_json(
        "report", "generate", "INVESTIGATION_DOSSIER", "--investigation", ws.investigation
    )
    shown = ws.cli_json("report", "show", doc["id"])
    assert shown["id"] == doc["id"]
    sections = ws.cli_json("report", "sections", doc["id"])
    assert {s["kind"] for s in sections} >= {"EXECUTIVE_SUMMARY", "LIMITATIONS", "EVIDENCE_GAPS"}


def test_claims_lists_findings_with_resolvable_evidence(ws: Ws) -> None:
    doc = ws.cli_json(
        "report", "generate", "INVESTIGATION_DOSSIER", "--investigation", ws.investigation
    )
    claims = ws.cli_json("report", "claims", doc["id"])
    assert claims
    for c in claims:
        assert c["status"] == "SUPPORTED"
        assert c["evidence"]


def test_validate_exits_zero_on_a_well_formed_report(ws: Ws) -> None:
    doc = ws.cli_json(
        "report", "generate", "INVESTIGATION_DOSSIER", "--investigation", ws.investigation
    )
    code, out, _err = ws.cli("report", "validate", doc["id"])
    assert code == 0
    assert json.loads(out) == []


def test_export_markdown_contains_the_report_id_and_matrix(ws: Ws) -> None:
    doc = ws.cli_json(
        "report", "generate", "INVESTIGATION_DOSSIER", "--investigation", ws.investigation
    )
    code, out, _err = ws.cli("report", "export", doc["id"], "--export-format", "markdown")
    assert code == 0
    assert doc["id"] in out
    assert "Claim / Evidence Matrix" in out


def test_export_html_wraps_the_markdown(ws: Ws) -> None:
    doc = ws.cli_json(
        "report", "generate", "INVESTIGATION_DOSSIER", "--investigation", ws.investigation
    )
    code, out, _err = ws.cli("report", "export", doc["id"], "--export-format", "html")
    assert code == 0
    assert "<pre>" in out and doc["id"] in out


def test_generate_with_an_unknown_investigation_errors_cleanly(ws: Ws) -> None:
    code, _out, err = ws.cli(
        "report", "generate", "INVESTIGATION_DOSSIER", "--investigation", "inv_" + "9" * 32
    )
    assert code == 2
    assert "not found" in err

"""The graph CLI on a real workspace: build, list, inspect, neighbors, path, query, diff, replay
-- JSON and text output, deterministic output, and clean errors for invalid requests."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from experionyx.cli import main
from experionyx.domain import ConfigurationRef, Investigation
from experionyx.sqlite import SqliteRegistry


class Ws:
    def __init__(self, path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        self.path, self.capsys = path, capsys
        assert main(["--workspace", str(path), "init"]) == 0
        capsys.readouterr()
        with SqliteRegistry(path / "registry.sqlite") as reg:
            inv = Investigation("cli-graph", "does the graph CLI work?", datetime.now(UTC))
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

    def spec(self, fname: str, **over: Any) -> str:
        body: dict[str, Any] = {"name": fname, "version": "1.0.0"}
        body.update(over)
        f = self.path / f"{fname}.json"
        f.write_text(json.dumps(body), encoding="utf-8")
        return str(f)


@pytest.fixture
def ws(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Ws:
    return Ws(tmp_path / "ws", capsys)


def test_build_constructs_a_snapshot_and_is_idempotent(ws: Ws) -> None:
    spec = ws.spec("g1")
    doc = ws.cli_json("graph", "build", spec, "--investigation", ws.investigation)
    assert doc["status"] == "COMPLETED"
    assert doc["already_built"] is False
    snapshot_id = doc["snapshot_id"]
    assert doc["snapshot"]["node_count"] >= 1

    again = ws.cli_json("graph", "build", spec, "--investigation", ws.investigation)
    assert again["already_built"] is True
    assert again["snapshot_id"] == snapshot_id


def test_build_grows_after_new_evidence_and_is_reflected_in_list_and_inspect(ws: Ws) -> None:
    spec = ws.spec("g2")
    first = ws.cli_json("graph", "build", spec, "--investigation", ws.investigation)
    with SqliteRegistry(ws.path / "registry.sqlite") as reg:
        reg.add(ConfigurationRef({"extra": True}))
    second = ws.cli_json("graph", "build", spec, "--investigation", ws.investigation)
    assert second["already_built"] is False
    assert second["snapshot_id"] != first["snapshot_id"]

    listed = ws.cli_json("graph", "list")
    assert len(listed["graphs"]) == 1  # one definition; two snapshots of it

    inspected = ws.cli_json("graph", "inspect", second["graph_id"])
    assert len(inspected["snapshots"]) == 2


def test_inspect_a_snapshot_shows_kind_and_relation_breakdowns(ws: Ws) -> None:
    spec = ws.spec("g3")
    built = ws.cli_json("graph", "build", spec, "--investigation", ws.investigation)
    doc = ws.cli_json("graph", "inspect", built["snapshot_id"])
    assert "INVESTIGATION" in doc["nodes_by_kind"]
    assert doc["node_count"] == built["snapshot"]["node_count"]


def test_neighbors_and_path_over_a_real_snapshot(ws: Ws) -> None:
    spec = ws.spec("g4")
    built = ws.cli_json("graph", "build", spec, "--investigation", ws.investigation)
    snap = built["snapshot_id"]

    nb = ws.cli_json("graph", "neighbors", snap, ws.investigation)
    assert any(n["ref_id"] == ws.investigation for n in nb["nodes"])

    with SqliteRegistry(ws.path / "registry.sqlite") as reg:
        cfg = reg.find(ConfigurationRef)
    assert cfg, "expected a bookkeeping ConfigurationRef from graph build itself"

    code, out, _err = ws.cli("graph", "path", snap, ws.investigation, ws.investigation)
    assert code == 0
    doc = json.loads(out)
    assert doc["found"] is True
    assert len(doc["nodes"]) == 1  # a path to itself


def test_path_reports_not_found_between_unconnected_nodes(ws: Ws) -> None:
    spec = ws.spec("g5")
    ws.cli_json("graph", "build", spec, "--investigation", ws.investigation)
    with SqliteRegistry(ws.path / "registry.sqlite") as reg:
        reg.add(ConfigurationRef({"isolated": True}))
    built2 = ws.cli_json("graph", "build", spec, "--investigation", ws.investigation)
    with SqliteRegistry(ws.path / "registry.sqlite") as reg:
        isolated_cfg = next(c for c in reg.find(ConfigurationRef) if c.parameters.get("isolated"))
    code, out, _err = ws.cli(
        "graph",
        "path",
        built2["snapshot_id"],
        ws.investigation,
        isolated_cfg.id,
        "--direction",
        "OUT",
    )
    doc = json.loads(out)
    assert doc["found"] is False
    assert code == 1


def test_query_evidence_for_claim_on_an_unsupported_claim(ws: Ws) -> None:
    from experionyx.domain import Claim, ClaimStatus

    with SqliteRegistry(ws.path / "registry.sqlite") as reg:
        claim = Claim(
            ws.investigation,
            "nothing backs this",
            "tester",
            datetime.now(UTC),
            ClaimStatus.INSUFFICIENT_EVIDENCE,
        )
        reg.add(claim)
    spec = ws.spec("g6")
    built = ws.cli_json("graph", "build", spec, "--investigation", ws.investigation)
    doc = ws.cli_json("graph", "query", built["snapshot_id"], "evidence-for-claim", claim.id)
    assert len(doc["nodes"]) == 1  # only the claim itself: unsupported


def test_query_rejects_an_unknown_query_name(ws: Ws) -> None:
    spec = ws.spec("g7")
    built = ws.cli_json("graph", "build", spec, "--investigation", ws.investigation)
    code, _out, err = ws.cli(
        "graph", "query", built["snapshot_id"], "not-a-real-query", ws.investigation
    )
    assert code == 2 and "unknown query" in err


def test_diff_reports_added_nodes_between_two_builds(ws: Ws) -> None:
    spec = ws.spec("g8")
    first = ws.cli_json("graph", "build", spec, "--investigation", ws.investigation)
    with SqliteRegistry(ws.path / "registry.sqlite") as reg:
        reg.add(ConfigurationRef({"diff-me": True}))
    second = ws.cli_json("graph", "build", spec, "--investigation", ws.investigation)
    doc = ws.cli_json("graph", "diff", first["snapshot_id"], second["snapshot_id"])
    assert len(doc["added_nodes"]) >= 1
    assert doc["removed_nodes"] == []


def test_replay_is_deterministic_for_an_unchanged_registry(ws: Ws) -> None:
    spec = ws.spec("g9")
    built = ws.cli_json("graph", "build", spec, "--investigation", ws.investigation)
    code, out, _err = ws.cli("graph", "replay", built["snapshot_id"])
    doc = json.loads(out)
    assert code == 0
    assert doc["deterministic"] is True


def test_build_on_an_unreadable_spec_file_is_a_clean_error(ws: Ws) -> None:
    code, _out, err = ws.cli("graph", "build", str(ws.path / "missing.json"))
    assert code == 2 and "cannot read graph spec" in err


def test_inspect_an_unknown_id_is_a_clean_error(ws: Ws) -> None:
    code, _out, err = ws.cli("graph", "inspect", "grh_" + "0" * 32)
    assert code == 2
    assert err  # a clean error, not a traceback


def test_text_format_is_human_readable(ws: Ws) -> None:
    spec = ws.spec("g10")
    code, out, _err = ws.cli(
        "graph", "build", spec, "--investigation", ws.investigation, "--format", "text"
    )
    assert code == 0
    assert "snapshot" in out
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)


def test_build_output_is_deterministic_when_registry_content_is_unchanged(ws: Ws) -> None:
    spec = ws.spec("g11")
    ws.cli_json("graph", "build", spec, "--investigation", ws.investigation)
    _code, out1, _err = ws.cli("graph", "inspect", ws.cli_json("graph", "list")["graphs"][0]["id"])
    _code, out2, _err = ws.cli("graph", "inspect", ws.cli_json("graph", "list")["graphs"][0]["id"])
    assert out1 == out2

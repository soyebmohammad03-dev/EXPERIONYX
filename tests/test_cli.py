import json
import sqlite3
from pathlib import Path

import pytest

from conftest import Lab
from experionyx.cli import main
from experionyx.domain import Artifact, Run


@pytest.fixture
def ws(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> Lab:
    lab.registry.close()  # the CLI opens its own connection to the same workspace
    monkeypatch.chdir(lab.workspace.parent)  # not a git repo -> source state UNKNOWN
    from experionyx.sqlite import SqliteRegistry

    lab.registry = SqliteRegistry(lab.workspace / "registry.sqlite")
    return lab


def cli(lab: Lab, *args: str) -> int:
    return main(["--workspace", str(lab.workspace), *args])


def run_cli_json(lab: Lab, capsys: pytest.CaptureFixture[str], *args: str) -> dict[str, object]:
    assert cli(lab, *args) == 0
    return json.loads(capsys.readouterr().out)  # type: ignore[no-any-return]


def test_status_reports_real_registry_counts(ws: Lab, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli(ws, "status") == 0
    out = capsys.readouterr().out
    assert "experiments: 1" in out
    assert "runs COMPLETED: 0" in out
    assert cli(ws, "execute", ws.experiment.id, "--procedure", "procedures:ok", "--seed", "3") == 0
    capsys.readouterr()
    cli(ws, "status")
    out = capsys.readouterr().out
    assert "runs COMPLETED: 1" in out
    assert "observations: 3" in out
    assert "artifacts: 1" in out


def test_execute_show_run_provenance_and_experiment(
    ws: Lab, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli(ws, "execute", ws.experiment.id, "--procedure", "procedures:ok", "--seed", "3") == 0
    out = capsys.readouterr().out
    assert "status: COMPLETED" in out
    run_id = next(line.split(": ")[1] for line in out.splitlines() if line.startswith("run: "))

    run = run_cli_json(ws, capsys, "run", run_id)
    assert run["run"]["status"] == "COMPLETED"  # type: ignore[index]
    assert run["outcome"]["status"] == "COMPLETED"  # type: ignore[index]
    assert len(run["observations"]) == 3  # type: ignore[arg-type]
    assert run["artifacts"][0]["path"] == "result.txt"  # type: ignore[index]

    prov = run_cli_json(ws, capsys, "provenance", run_id)
    assert prov["seed"] == 3
    assert prov["execution"]["procedure"] == "procedures:ok"  # type: ignore[index]
    assert str(prov["fingerprint"]).startswith("sha256:")

    exp = run_cli_json(ws, capsys, "experiment", ws.experiment.id)
    assert [r["id"] for r in exp["runs"]] == [run_id]  # type: ignore[attr-defined]
    assert exp["configuration"]["parameters"]["lr"] == 0.1  # type: ignore[index]


def test_failed_execution_exits_nonzero_and_is_inspectable(
    ws: Lab, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli(ws, "execute", ws.experiment.id, "--procedure", "procedures:fails", "--seed", "0")
    out = capsys.readouterr().out
    assert code == 1
    assert "status: FAILED" in out
    assert "builtins.ValueError: boom" in out
    (run,) = ws.registry.find(Run)
    shown = run_cli_json(ws, capsys, "run", run.id)
    assert shown["outcome"]["error"]["message"] == "boom"  # type: ignore[index]


def test_replay_command_creates_a_new_run(ws: Lab, capsys: pytest.CaptureFixture[str]) -> None:
    cli(ws, "execute", ws.experiment.id, "--procedure", "procedures:ok", "--seed", "1")
    (original,) = ws.registry.find(Run)
    capsys.readouterr()
    assert cli(ws, "replay", original.id) == 0
    out = capsys.readouterr().out
    assert "does not verify reproduction" in out
    runs = ws.registry.find(Run)
    assert len(runs) == 2
    new = next(r for r in runs if r.id != original.id)
    prov = run_cli_json(ws, capsys, "provenance", new.id)
    assert prov["replay_of"] == original.id


def test_verify_detects_a_modified_artifact(ws: Lab, capsys: pytest.CaptureFixture[str]) -> None:
    cli(ws, "execute", ws.experiment.id, "--procedure", "procedures:ok", "--seed", "1")
    (run,) = ws.registry.find(Run)
    capsys.readouterr()
    assert cli(ws, "verify", run.id) == 0
    assert "OK" in capsys.readouterr().out
    (art,) = ws.registry.find(Artifact, run_id=run.id)
    (ws.store.run_dir(run) / "artifacts" / art.path).write_text("tampered")
    assert cli(ws, "verify", run.id) == 1
    assert "MISMATCH" in capsys.readouterr().out


def test_recover_command(ws: Lab, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli(ws, "recover") == 0
    assert "0 run(s) recovered" in capsys.readouterr().out


def test_errors_are_reported_with_exit_code_2(
    ws: Lab, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert main(["--workspace", str(tmp_path / "missing"), "status"]) == 2
    assert "no registry" in capsys.readouterr().err
    assert cli(ws, "run", "run_" + "0" * 32) == 2
    assert "not found" in capsys.readouterr().err
    assert cli(ws, "execute", ws.experiment.id, "--procedure", "nope:fn", "--seed", "0") == 2
    assert cli(ws, "provenance", "run_" + "0" * 32) == 2


def test_missing_registry_is_never_created_by_read_commands(tmp_path: Path) -> None:
    main(["--workspace", str(tmp_path / "w"), "status"])
    assert not (tmp_path / "w").exists()


def test_execute_requires_an_explicit_seed(ws: Lab) -> None:
    with pytest.raises(SystemExit):
        cli(ws, "execute", ws.experiment.id, "--procedure", "procedures:ok")


def test_registry_file_is_a_real_sqlite_database(ws: Lab) -> None:
    cli(ws, "execute", ws.experiment.id, "--procedure", "procedures:ok", "--seed", "1")
    with sqlite3.connect(ws.workspace / "registry.sqlite") as raw:
        assert raw.execute("SELECT status FROM runs").fetchall() == [("COMPLETED",)]
        assert raw.execute("SELECT count(*) FROM provenance").fetchone()[0] == 1

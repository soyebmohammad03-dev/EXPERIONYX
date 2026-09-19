"""The interaction CLI against a real workspace."""

import json
from pathlib import Path

import pytest

from experionyx.cli import main
from experionyx.domain import Run
from experionyx.interactions.entities import InteractionAnalysis
from experionyx.sqlite import SqliteRegistry
from interaction_helpers import Design, four_cell, world


@pytest.fixture
def d(tmp_path: Path) -> Design:
    design = four_cell(world(tmp_path), seeds=(1, 2, 3), order=False, sigma=6.0, p=0.5)
    design.w.registry.close()
    return design


def cli(d: Design, capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str, str]:
    code = main(["--workspace", str(d.w.workspace), *args])
    out = capsys.readouterr()
    return code, out.out, out.err


def spec_file(d: Design, tmp_path: Path, name: str = "spec.json", **over: object) -> Path:
    with SqliteRegistry(d.w.workspace / "registry.sqlite") as reg:
        body = {
            "control": [d.baseline],
            "a": list(runs_of_reg(reg, d.a)),
            "b": list(runs_of_reg(reg, d.b)),
            "ab": list(runs_of_reg(reg, d.ab)),
            "config": {"bootstrap_resamples": 200},
        }
    body.update(over)
    path = tmp_path / name
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def runs_of_reg(reg: SqliteRegistry, res):  # type: ignore[no-untyped-def]
    from experionyx.faults.entities import FaultTrial

    return tuple(
        t.treatment_run_id
        for t in reg.find(FaultTrial, fault_experiment_id=res.fault_experiment.id)
        if t.treatment_run_id
    )


def count(d: Design) -> tuple[int, int]:
    with SqliteRegistry(d.w.workspace / "registry.sqlite") as reg:
        return len(reg.find(Run)), len(reg.find(InteractionAnalysis))


def test_validate_analyze_list_inspect_export_related_failures_replay(
    d: Design, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:  # type: ignore[no-untyped-def]
    spec = spec_file(d, tmp_path)
    code, out, _ = cli(d, capsys, "interaction", "validate", str(spec))
    rep = json.loads(out)
    assert code == 0 and rep["valid"] is True and rep["trials"]["A"] == 3 and rep["checks"]
    code, out, _ = cli(d, capsys, "interaction", "validate", str(spec), "--format", "text")
    assert code == 0 and "VALID" in out and out.lstrip().startswith("spec isp_")
    runs_before = count(d)[0]
    assert (
        cli(d, capsys, "interaction", "validate", str(spec))[0] == 0 and count(d)[0] == runs_before
    )  # validate never writes

    code, out, _ = cli(d, capsys, "interaction", "analyze", str(spec))
    doc = json.loads(out)
    assert (
        code == 0
        and doc["status"] == "COMPLETED"
        and doc["interaction"]["statement"].endswith("not a causal or mechanistic claim.")
    )
    aid = doc["analysis_id"]
    assert cli(d, capsys, "interaction", "analyze", str(spec), "--format", "text")[1].startswith(
        "COMPLETED:"
    )

    _, out, _ = cli(d, capsys, "interaction", "list")
    assert aid in {x["id"] for x in json.loads(out)["interactions"]}
    _, out, _ = cli(
        d,
        capsys,
        "interaction",
        "list",
        "--metric",
        "accuracy",
        "--fault",
        "gaussian_noise",
        "--status",
        doc["interaction"]["status"],
    )
    assert [x["id"] for x in json.loads(out)["interactions"]] == [aid]
    _, out, _ = cli(d, capsys, "interaction", "list", "--metric", "nope")
    assert json.loads(out)["interactions"] == []
    assert "accuracy" in cli(d, capsys, "interaction", "list", "--format", "text")[1]

    _, out, _ = cli(d, capsys, "interaction", "inspect", aid)
    ins = json.loads(out)
    assert (
        ins["id"] == aid
        and any(e["measure"] == "accuracy" for e in ins["effects"])
        and {"DESIGN", "OBSERVATION"} <= {e["kind"] for e in ins["evidence"]}
    )
    _, out, _ = cli(d, capsys, "interaction", "inspect", aid, "--full")
    assert len(json.loads(out)["raw_trials"]["AB"]) == 3  # raw trials on request

    out_file = tmp_path / "bundle.json"
    assert cli(d, capsys, "interaction", "export", aid, "--out", str(out_file))[0] == 0
    bundle = json.loads(out_file.read_text())
    assert (
        set(bundle) == {"analysis", "provenance", "effects", "evidence", "artifacts"}
        and "trials" in bundle["artifacts"]
    )

    _, out, _ = cli(d, capsys, "interaction", "related", aid)
    assert json.loads(out)["related"] == []
    _, out, _ = cli(d, capsys, "interaction", "failures", aid)
    assert json.loads(out)["status"] == "NOT_AVAILABLE"
    code, out, _ = cli(d, capsys, "interaction", "replay", aid)
    rp = json.loads(out)
    assert code == 0 and rp["deterministic"] is True and rp["differences"] == []


def test_invalid_designs_are_refused_before_execution_with_every_issue(
    d: Design, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:  # type: ignore[no-untyped-def]
    with SqliteRegistry(d.w.workspace / "registry.sqlite") as reg:
        a = runs_of_reg(reg, d.a)
    bad = spec_file(d, tmp_path, name="bad_design.json", control=[a[0]], a=list(a[1:]))
    before = count(d)
    code, out, _ = cli(d, capsys, "interaction", "validate", str(bad))
    rep = json.loads(out)
    assert (
        code == 1
        and rep["valid"] is False
        and rep["issues"][0]["code"] == "CONTROL_HAS_FAULT"
        and rep["issues"][0]["why"]
    )
    assert "REFUSED" in cli(d, capsys, "interaction", "validate", str(bad), "--format", "text")[1]
    code, _, err = cli(d, capsys, "interaction", "analyze", str(bad))
    assert (
        code == 2
        and "nothing was analyzed and nothing was recorded" in err
        and "CONTROL_HAS_FAULT" in err
    )
    assert count(d) == before  # no run, no analysis: refused before execution


def test_input_errors_and_lifecycle_gates(
    d: Design, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = spec_file(d, tmp_path)
    code, _, err = cli(d, capsys, "interaction", "validate", str(tmp_path / "missing.json"))
    assert code == 2 and "cannot read spec" in err
    (tmp_path / "bad.json").write_text("[1]", encoding="utf-8")
    assert cli(d, capsys, "interaction", "validate", str(tmp_path / "bad.json"))[0] == 2
    empty = spec_file(d, tmp_path, name="empty.json", a=[])
    assert cli(d, capsys, "interaction", "validate", str(empty))[0] == 2
    aid = json.loads(cli(d, capsys, "interaction", "analyze", str(spec))[1])["analysis_id"]
    code, _, err = cli(
        d, capsys, "interaction", "confirm", aid, "--by", "me", "--reason", "hopeful"
    )
    assert code == 2 and "REPRODUCIBLE" in err
    code, out, _ = cli(
        d,
        capsys,
        "interaction",
        "set-status",
        aid,
        "--to",
        "REJECTED",
        "--by",
        "me",
        "--reason",
        "demo",
    )
    assert code == 0 and json.loads(out)["status"] == "REJECTED"
    code, _, err = cli(
        d,
        capsys,
        "interaction",
        "set-status",
        aid,
        "--to",
        "DEPRECATED",
        "--by",
        "me",
        "--reason",
        "x",
    )
    assert code == 2 and "illegal" in err
    with pytest.raises(SystemExit):
        main(
            [
                "--workspace",
                str(d.w.workspace),
                "interaction",
                "set-status",
                aid,
                "--to",
                "CONFIRMED_BY_REVIEW",
                "--by",
                "me",
                "--reason",
                "x",
            ]
        )
    assert cli(d, capsys, "interaction", "inspect", "ian_" + "0" * 32)[0] == 2


def test_interaction_tables_are_append_only_except_the_analysis_status(
    d: Design, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import sqlite3

    assert cli(d, capsys, "interaction", "analyze", str(spec_file(d, tmp_path)))[0] == 0
    with sqlite3.connect(d.w.workspace / "registry.sqlite") as raw:
        for table in ("interaction_analyses", "interaction_effects", "interaction_evidence"):
            assert raw.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] > 0  # noqa: S608
            with pytest.raises(sqlite3.DatabaseError):
                raw.execute(f"DELETE FROM {table}")  # noqa: S608
        for table in ("interaction_effects", "interaction_evidence"):
            with pytest.raises(sqlite3.DatabaseError):
                raw.execute(f"UPDATE {table} SET payload = '{{}}'")  # noqa: S608
        with pytest.raises(sqlite3.DatabaseError):
            raw.execute(
                "UPDATE interaction_analyses SET run_id = 'run_x'"
            )  # identity columns are guarded

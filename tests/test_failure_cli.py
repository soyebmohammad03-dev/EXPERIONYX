"""The failure CLI against a real workspace (registry + artifacts written by real runs)."""

import json
from pathlib import Path

import pytest

from eval_helpers import EvalWorld
from experionyx.cli import main
from experionyx.failures.entities import FailureMode
from experionyx.failures.taxonomy import FailureStatus
from failure_helpers import noise_sweep, noisy_world


@pytest.fixture
def ws(tmp_path: Path) -> tuple[EvalWorld, str, str]:
    w = noisy_world(tmp_path)
    res = noise_sweep(w)
    w.registry.close()
    return w, home_of_path(w, res), res.fault_experiment.id


def home_of_path(w: EvalWorld, res) -> str:  # type: ignore[no-untyped-def]
    from experionyx.sqlite import SqliteRegistry

    with SqliteRegistry(w.workspace / "registry.sqlite") as reg:
        from experionyx.domain import Experiment, Run

        return reg.get(Experiment, reg.get(Run, res.baseline_run_id).experiment_id).investigation_id


def cli(w: EvalWorld, capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str, str]:
    code = main(["--workspace", str(w.workspace), *args])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_discover_list_inspect_candidates_evidence_graph_cluster(ws, capsys) -> None:  # type: ignore[no-untyped-def]
    w, _, fxp = ws
    code, out, _ = cli(w, capsys, "failure", "discover", "--fault-experiment", fxp)
    assert code == 0
    doc = json.loads(out)
    assert doc["status"] == "COMPLETED" and doc["failure_modes"]
    mode_id = next(m["id"] for m in doc["failure_modes"] if m["status"] == "CANDIDATE")

    _, out, _ = cli(w, capsys, "failures")
    listed = json.loads(out)["failure_modes"]
    assert {m["id"] for m in listed} >= {m["id"] for m in doc["failure_modes"]}
    _, out, _ = cli(
        w, capsys, "failures", "--status", "CANDIDATE", "--category", listed[0]["category"]
    )
    assert all(m["status"] == "CANDIDATE" for m in json.loads(out)["failure_modes"])

    _, out, _ = cli(w, capsys, "failure", "inspect", mode_id)
    rec = json.loads(out)
    assert rec["id"] == mode_id and rec["structured"]["discovered_by"].startswith(
        "deterministic rules"
    )
    _, out, _ = cli(w, capsys, "failure", "inspect", rec["cluster_id"])
    cluster = json.loads(out)
    _, out, _ = cli(w, capsys, "failure", "inspect", cluster["signal_ids"][0])
    assert json.loads(out)["kind"] == "failure_signal"

    _, out, _ = cli(w, capsys, "failure", "candidates")
    cands = json.loads(out)["candidates"]
    assert cands and all("criteria" in c for c in cands)
    assert not any(c["status"] == "CONFIRMED" for c in cands)

    _, out, _ = cli(w, capsys, "failure", "evidence", mode_id)
    kinds = {e["evidence_kind"] for e in json.loads(out)["evidence"]}
    assert {"SIGNAL", "OBSERVATION", "TRANSITION"} <= kinds

    _, out, _ = cli(w, capsys, "failure", "graph", mode_id)
    assert {e["predicate"] for e in json.loads(out)["edges"]} >= {"EXHIBITS", "SUPPORTED_BY"}

    _, out, _ = cli(w, capsys, "failure", "cluster")
    assert json.loads(out)["clusters"]
    _, out, _ = cli(w, capsys, "failure", "cluster", rec["cluster_id"])
    shown = json.loads(out)
    assert {m["id"] for m in shown["members"]} == set(cluster["signal_ids"])


def _mode_with(w: EvalWorld, status: str) -> str:
    from experionyx.sqlite import SqliteRegistry

    with SqliteRegistry(w.workspace / "registry.sqlite") as reg:
        return next(m.id for m in reg.find(FailureMode) if m.status.value == status)


def test_confirm_is_refused_and_status_changes_need_reasons(ws, capsys) -> None:  # type: ignore[no-untyped-def]
    w, _, fxp = ws
    assert cli(w, capsys, "failure", "discover", "--fault-experiment", fxp)[0] == 0
    candidate = _mode_with(w, "CANDIDATE")
    code, _, err = cli(
        w, capsys, "failure", "confirm", candidate, "--by", "me", "--reason", "hopeful"
    )
    assert code == 2 and "SUPPORTED" in err  # a CANDIDATE cannot be confirmed
    code, _, err = cli(
        w,
        capsys,
        "failure",
        "set-status",
        candidate,
        "--to",
        "REJECTED",
        "--by",
        " ",
        "--reason",
        "x",
    )
    assert code == 2 and "actor" in err  # an actor is required
    code, out, _ = cli(
        w,
        capsys,
        "failure",
        "set-status",
        candidate,
        "--to",
        "DEPRECATED",
        "--by",
        "me",
        "--reason",
        "demo",
    )
    assert (
        code == 0 and json.loads(out)["status"] == "DEPRECATED"
    )  # CANDIDATE -> DEPRECATED is legal
    code, _, err = cli(
        w,
        capsys,
        "failure",
        "set-status",
        candidate,
        "--to",
        "CANDIDATE",
        "--by",
        "me",
        "--reason",
        "undo",
    )
    assert code == 2 and "illegal" in err  # DEPRECATED is terminal
    with pytest.raises(SystemExit):  # CONFIRMED is not offered by set-status at all
        main(
            [
                "--workspace",
                str(w.workspace),
                "failure",
                "set-status",
                candidate,
                "--to",
                "CONFIRMED",
                "--by",
                "me",
                "--reason",
                "x",
            ]
        )


def test_input_errors_are_clear(ws, capsys) -> None:  # type: ignore[no-untyped-def]
    w, _, fxp = ws
    code, _, err = cli(w, capsys, "failure", "discover")
    assert (
        code == 2 and "investigations" in err
    )  # nothing selected: no sources to infer a home from
    code, _, err = cli(w, capsys, "failure", "inspect", "run_" + "0" * 32)
    assert code == 2 and "expected a failure mode" in err
    code, _, err = cli(w, capsys, "failure", "inspect", "fmd_" + "0" * 32)
    assert code == 2 and "not found" in err
    code, _, err = cli(
        w,
        capsys,
        "failure",
        "discover",
        "--fault-experiment",
        fxp,
        "--config",
        str(w.workspace / "nope.json"),
    )
    assert code == 2 and "cannot read --config" in err
    bad = w.workspace / "bad.json"
    bad.write_text('{"similarity": {"threshold": 5}}', encoding="utf-8")
    code, _, err = cli(
        w, capsys, "failure", "discover", "--fault-experiment", fxp, "--config", str(bad)
    )
    assert code == 2 and "threshold" in err


def test_multi_investigation_sources_need_an_explicit_home(ws, capsys) -> None:  # type: ignore[no-untyped-def]
    w, inv, fxp = ws
    code, out, _ = cli(
        w, capsys, "failure", "discover", "--fault-experiment", fxp, "--investigation", inv
    )
    assert code == 0 and json.loads(out)["status"] == "COMPLETED"
    code, _, _ = cli(
        w,
        capsys,
        "failure",
        "discover",
        "--fault-experiment",
        fxp,
        "--investigation",
        "inv_" + "0" * 32,
    )
    assert code == 2


def test_a_custom_config_file_is_used_and_recorded(ws, capsys, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from dataclasses import replace

    from experionyx.failures.config import DiscoveryConfig, SimilarityConfig

    w, _, fxp = ws
    cfg = replace(DiscoveryConfig(), similarity=SimilarityConfig(threshold=0.9))
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    code, out, _ = cli(
        w, capsys, "failure", "discover", "--fault-experiment", fxp, "--config", str(path)
    )
    assert code == 0 and json.loads(out)["config_hash"] == cfg.config_hash
    from experionyx.sqlite import SqliteRegistry

    with SqliteRegistry(w.workspace / "registry.sqlite") as reg:
        assert all(m.status is not FailureStatus.CONFIRMED for m in reg.find(FailureMode))

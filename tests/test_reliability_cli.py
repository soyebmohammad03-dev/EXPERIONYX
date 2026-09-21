"""The reliability CLI against a real workspace."""

import json
import sqlite3
from pathlib import Path

import pytest

from eval_helpers import EvalWorld
from experionyx.cli import main
from experionyx.domain import Run
from experionyx.reliability.entities import ReliabilityProfile
from experionyx.sqlite import SqliteRegistry
from interaction_helpers import launch, noise, world


@pytest.fixture
def ws(tmp_path: Path) -> tuple[EvalWorld, str, str]:
    w = world(tmp_path)
    res = launch(w, noise(6.0), (1, 2, 3), name="noise")
    fx = res.fault_experiment.id
    base = res.baseline_run_id
    w.registry.close()
    return w, base, fx


def cli(w: EvalWorld, capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str, str]:
    code = main(["--workspace", str(w.workspace), *args])
    out = capsys.readouterr()
    return code, out.out, out.err


def write(tmp_path: Path, name: str, body: dict[str, object]) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(body), encoding="utf-8")
    return str(path)


def runs(w: EvalWorld) -> int:
    with SqliteRegistry(w.workspace / "registry.sqlite") as reg:
        return len(reg.find(Run))


def test_profile_list_inspect_evidence_compare_replay(
    ws: tuple[EvalWorld, str, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    w, base, fx = ws
    full = write(
        tmp_path,
        "full.json",
        {"scope": "MODEL_DATASET_EVALUATION", "baseline_run": base, "fault_experiments": [fx]},
    )
    bare = write(tmp_path, "bare.json", {"scope": "MODEL_DATASET_EVALUATION", "baseline_run": base})
    code, out, _ = cli(w, capsys, "reliability", "profile", full)
    doc = json.loads(out)
    assert (
        code == 0
        and doc["status"] == "COMPLETED"
        and doc["profile"]["dimension_status"]["FAULT_SENSITIVITY"] == "OBSERVED"
    )
    pid = doc["profile_id"]
    text = cli(w, capsys, "reliability", "profile", full, "--format", "text")[1]
    assert pid in text and "no overall score" in text and "Under fault gaussian_noise" in text
    bare_id = json.loads(cli(w, capsys, "reliability", "profile", bare)[1])["profile_id"]

    listed = json.loads(cli(w, capsys, "reliability", "list")[1])["profiles"]
    assert {p["id"] for p in listed} == {pid, bare_id}
    only = json.loads(cli(w, capsys, "reliability", "list", "--ref", fx)[1])["profiles"]
    assert [p["id"] for p in only] == [pid]
    filt = json.loads(
        cli(
            w,
            capsys,
            "reliability",
            "list",
            "--dimension",
            "FAULT_SENSITIVITY",
            "--dimension-status",
            "UNAVAILABLE",
        )[1]
    )["profiles"]
    assert [p["id"] for p in filt] == [bare_id]
    assert (
        "MODEL_DATASET_EVALUATION" in cli(w, capsys, "reliability", "list", "--format", "text")[1]
    )

    ins = json.loads(cli(w, capsys, "reliability", "inspect", pid)[1])
    assert ins["id"] == pid and ins["interpreted"]["statements"] and "document" not in ins
    full_doc = json.loads(cli(w, capsys, "reliability", "inspect", pid, "--full")[1])["document"]
    assert (
        set(full_doc["dimensions"]) == set(full_doc["dimension_status"])
        and len(full_doc["dimensions"]) == 14
    )
    dim = json.loads(
        cli(w, capsys, "reliability", "inspect", pid, "--dimension", "FAULT_SENSITIVITY")[1]
    )
    assert (
        dim["status"] == "OBSERVED" and dim["observations"][0]["fault"]["type"] == "gaussian_noise"
    )

    ev = json.loads(cli(w, capsys, "reliability", "evidence", pid)[1])
    assert {r["kind"] for r in ev["references"]} >= {"RUN", "FAULT_EXPERIMENT", "ARTIFACT"} and {
        a["path"] for a in ev["artifacts"]
    } == {"reliability/profile.json", "reliability/sources.json"}
    assert ev["provenance"]["provenance_fingerprint"].startswith("sha256:")
    assert all(
        r["dimension"] == "FAULT_SENSITIVITY"
        for r in json.loads(
            cli(w, capsys, "reliability", "evidence", pid, "--dimension", "FAULT_SENSITIVITY")[1]
        )["references"]
    )

    code, out, _ = cli(w, capsys, "reliability", "compare", bare_id, pid)
    cmp = json.loads(out)
    assert (
        code == 0
        and cmp["same_model"] is True
        and cmp["fault_responses"]["only_in_b"]
        and cmp["note"].startswith("raw differences")
    )
    assert (
        "no winner" in cli(w, capsys, "reliability", "compare", bare_id, pid, "--format", "text")[1]
    )

    code, out, _ = cli(w, capsys, "reliability", "replay", pid)
    rp = json.loads(out)
    assert code == 0 and rp["deterministic"] is True and rp["differences"] == []


def test_incompatible_or_invalid_requests_fail_clearly_and_record_nothing(
    ws: tuple[EvalWorld, str, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    w, base, fx = ws
    before = runs(w)
    faulted_anchor = write(
        tmp_path, "a.json", {"scope": "MODEL_DATASET", "baseline_run": _treatment(w, fx)}
    )
    code, _, err = cli(w, capsys, "reliability", "profile", faulted_anchor)
    assert (
        code == 2
        and "nothing was built and nothing was recorded" in err
        and "BASELINE_IS_FAULTED" in err
    )
    for name, body, needle in (
        ("bad_scope.json", {"scope": "CROSS_DATASET", "baseline_run": base}, "unknown scope"),
        ("no_baseline.json", {"scope": "MODEL_DATASET"}, "baseline_run"),
        ("extra.json", {"scope": "MODEL_DATASET", "baseline_run": base, "x": 1}, "unexpected"),
    ):
        code, _, err = cli(w, capsys, "reliability", "profile", write(tmp_path, name, body))
        assert code == 2 and needle in err, (name, err)
    assert cli(w, capsys, "reliability", "profile", str(tmp_path / "missing.json"))[0] == 2
    assert cli(w, capsys, "reliability", "inspect", "rpf_" + "0" * 32)[0] == 2
    assert runs(w) == before  # every refusal happened before execution


def _treatment(w: EvalWorld, fx: str) -> str:
    from experionyx.faults.entities import FaultTrial

    with SqliteRegistry(w.workspace / "registry.sqlite") as reg:
        return next(
            t.treatment_run_id
            for t in reg.find(FaultTrial, fault_experiment_id=fx)
            if t.treatment_run_id
        )


def test_incomparable_profiles_are_refused_by_the_cli(
    ws: tuple[EvalWorld, str, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    w, base, _ = ws
    a = json.loads(
        cli(
            w,
            capsys,
            "reliability",
            "profile",
            write(tmp_path, "a.json", {"scope": "MODEL_DATASET", "baseline_run": base}),
        )[1]
    )["profile_id"]
    b = json.loads(
        cli(
            w,
            capsys,
            "reliability",
            "profile",
            write(tmp_path, "b.json", {"scope": "MODEL_DATASET_EVALUATION", "baseline_run": base}),
        )[1]
    )["profile_id"]
    code, _, err = cli(w, capsys, "reliability", "compare", a, b)
    assert code == 2 and "not comparable" in err and "SCOPE_MISMATCH" in err


def test_profile_tables_are_append_only(
    ws: tuple[EvalWorld, str, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    w, base, fx = ws
    assert (
        cli(
            w,
            capsys,
            "reliability",
            "profile",
            write(
                tmp_path,
                "s.json",
                {"scope": "MODEL_DATASET", "baseline_run": base, "fault_experiments": [fx]},
            ),
        )[0]
        == 0
    )
    with sqlite3.connect(w.workspace / "registry.sqlite") as raw:
        for table in ("reliability_profiles", "reliability_references"):
            assert raw.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] > 0  # noqa: S608
            with pytest.raises(sqlite3.DatabaseError):
                raw.execute(f"DELETE FROM {table}")  # noqa: S608
            with pytest.raises(sqlite3.DatabaseError):
                raw.execute(f"UPDATE {table} SET payload = '{{}}'")  # noqa: S608
    with SqliteRegistry(w.workspace / "registry.sqlite") as reg:
        (p,) = reg.find(ReliabilityProfile)
        assert p.scope.value == "MODEL_DATASET"

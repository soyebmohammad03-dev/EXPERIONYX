"""The benchmark protocol/submit and leaderboard CLI on a real sklearn workspace -- JSON output
and clean errors for invalid requests."""

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sklearn")

from experionyx.adapters.records import RegisteredDataset, RegisteredModel
from experionyx.cli import main
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

    def cli_json(self, *args: str) -> Any:
        _code, out, err = self.cli(*args)
        assert out, f"expected JSON output, got empty stdout (stderr: {err})"
        return json.loads(out)

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


@pytest.fixture
def ws(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Ws:
    return Ws(tmp_path / "ws", capsys)


def test_protocol_register_and_submit_produce_real_evidence(ws: Ws) -> None:
    spec = ws.spec("s1")
    protocol = ws.cli_json("benchmark", "protocol", spec)
    assert protocol["id"].startswith("bpr_")

    submission = ws.cli_json("benchmark", "submit", spec, "--protocol", protocol["id"])
    assert submission["submission_id"].startswith("bsb_")
    assert submission["status"] == "COMPLETED"
    assert submission["already_submitted"] is False

    again = ws.cli_json("benchmark", "submit", spec, "--protocol", protocol["id"])
    assert again["already_submitted"] is True
    assert again["submission_id"] == submission["submission_id"]


def test_leaderboard_list_and_inspect(ws: Ws) -> None:
    spec = ws.spec("s2")
    protocol = ws.cli_json("benchmark", "protocol", spec)
    ws.cli_json("benchmark", "submit", spec, "--protocol", protocol["id"])

    listed = ws.cli_json("leaderboard", "list")
    assert protocol["id"] in [p["id"] for p in listed["protocols"]]

    inspected = ws.cli_json("leaderboard", "inspect", protocol["id"])
    assert len(inspected["submissions"]) == 1


def test_leaderboard_snapshot_and_compare_flow(ws: Ws) -> None:
    spec = ws.spec("s3")
    protocol = ws.cli_json("benchmark", "protocol", spec)
    sub_a = ws.cli_json("benchmark", "submit", spec, "--protocol", protocol["id"])

    snap = ws.cli_json("leaderboard", "snapshot", protocol["id"])
    assert snap["id"].startswith("lbs_")
    assert sub_a["submission_id"] in snap["submission_ids"]
    assert len(snap["entries"]) == 1

    inspected = ws.cli_json("leaderboard", "inspect", snap["id"])
    assert len(inspected["entries"]) == 1

    code, _out, _err = ws.cli("leaderboard", "verify", sub_a["submission_id"])
    assert code == 0

    code, _out, _err = ws.cli("leaderboard", "replay", snap["id"])
    assert code == 0  # unchanged evidence -> the same snapshot


def test_leaderboard_compare_refuses_a_protocol_mismatch(ws: Ws) -> None:
    spec_a = ws.spec("s4a")
    protocol_a = ws.cli_json("benchmark", "protocol", spec_a)
    sub_a = ws.cli_json("benchmark", "submit", spec_a, "--protocol", protocol_a["id"])

    spec_b = ws.spec("s4b", seeds=[9, 10])
    protocol_b = ws.cli_json("benchmark", "protocol", spec_b)
    sub_b = ws.cli_json("benchmark", "submit", spec_b, "--protocol", protocol_b["id"])

    code, _out, err = ws.cli(
        "leaderboard", "compare", sub_a["submission_id"], sub_b["submission_id"]
    )
    assert code == 2
    assert "not comparable" in err


def test_submit_refuses_a_spec_that_does_not_match_the_protocol(ws: Ws) -> None:
    spec = ws.spec("s5")
    protocol = ws.cli_json("benchmark", "protocol", spec)
    mismatched = ws.spec("s5-mismatched", seeds=[42, 43])
    code, _out, err = ws.cli("benchmark", "submit", mismatched, "--protocol", protocol["id"])
    assert code == 2
    assert "does not match protocol" in err

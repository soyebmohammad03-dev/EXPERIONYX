"""The calibration CLI on the real iris workspace (a scikit-learn logistic regression with real
predict_proba): validation, refusal before execution, evaluation, inspection of provenance and
evidence, recompute-and-compare, replay, JSON determinism and clean errors."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sklearn")

from experionyx.cli import main
from experionyx.demos import run_demo
from experionyx.domain import Run
from experionyx.sqlite import SqliteRegistry


@dataclass
class Iris:
    ws: Path
    baseline: str


@pytest.fixture(scope="module")
def iris(tmp_path_factory: pytest.TempPathFactory) -> Iris:
    ws = tmp_path_factory.mktemp("iris-cal-cli") / "w"
    return Iris(ws, run_demo("sklearn-classification", ws).run.id)


def cli(ws: Path, capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str, str]:
    code = main(["--workspace", str(ws), "calibration", *args])
    out = capsys.readouterr()
    return code, out.out, out.err


def write(tmp_path: Path, name: str, body: Any) -> str:
    p = tmp_path / name
    p.write_text(body if isinstance(body, str) else json.dumps(body), encoding="utf-8")
    return str(p)


def spec(iris: Iris, **over: Any) -> dict[str, Any]:
    d: dict[str, Any] = {"baseline_run": iris.baseline, "prediction_source": "PREDICT_PROBA", "statistics": {"resamples": 40, "permutations": 40, "min_samples": 10}, "binning": {"n_bins": 5}}  # fmt: skip
    d.update(over)
    return d


def n_runs(ws: Path) -> int:
    with SqliteRegistry(ws / "registry.sqlite") as reg:
        return len(reg.find(Run))


def test_validate_prints_a_deterministic_identity_and_preflights_the_baseline(iris: Iris, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    f = write(tmp_path, "s.json", spec(iris))
    code, out, _ = cli(iris.ws, capsys, "validate", f)
    doc = json.loads(out)
    assert code == 0 and doc["valid"] is True and doc["spec_id"].startswith("cbs_") and doc["method"] == "NONE"  # fmt: skip
    assert cli(iris.ws, capsys, "validate", f)[1] == out  # deterministic output
    before = n_runs(iris.ws)
    code, out, _ = cli(iris.ws, capsys, "validate", f, "--preflight")
    pre = json.loads(out)["preflight"]
    assert code == 0 and pre["baseline_usable"] == pre["baseline_rows"] == 38 and pre["invalid_rows"] == 0  # fmt: skip
    assert pre["prediction_representation"] == "PREDICT_PROBA" and pre["classes"] == [0, 1, 2]
    assert n_runs(iris.ws) == before  # nothing was executed
    code, out, _ = cli(iris.ws, capsys, "validate", f, "--format", "text")
    assert code == 0 and out.startswith("VALID: cbs_")
    other = json.loads(cli(iris.ws, capsys, "validate", write(tmp_path, "o.json", spec(iris, binning={"n_bins": 6})))[1])  # fmt: skip
    assert other["spec_id"] != doc["spec_id"]


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"prediction_source": "LOGITS"}, "prediction_source"),
        ({"binning": {"n_bins": 0}}, "n_bins"),
        ({"method": "TEMPERATURE"}, "method"),
        ({"statistics": {"confidence": 2}}, "confidence"),
        ({"unknown_field": 1}, "unexpected"),
        (
            {"method": "PLATT", "fit": {"mode": "RUN", "calibration_run": "run_" + "a" * 32}},
            "does not exist|unknown|not found|no such",
        ),
    ],
)
def test_invalid_specs_and_requests_are_refused_cleanly_before_anything_runs(iris: Iris, tmp_path: Path, capsys: pytest.CaptureFixture[str], over: dict[str, Any], match: str) -> None:  # fmt: skip
    import re

    f = write(tmp_path, "bad.json", spec(iris, **over))
    before = n_runs(iris.ws)
    for cmd in (["validate", f, "--preflight"], ["evaluate", f]):
        code, out, err = cli(iris.ws, capsys, *cmd)
        assert code == 2 and out == "" and "Traceback" not in err and re.search(match, err, re.I), (
            err
        )
    assert n_runs(iris.ws) == before


def test_a_representation_mismatch_and_a_leaky_calibration_run_are_refused(iris: Iris, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    code, _, err = cli(iris.ws, capsys, "evaluate", write(tmp_path, "m.json", spec(iris, prediction_source="SOFTMAX_LOGITS")))  # fmt: skip
    assert code == 2 and "never treated as another representation" in err
    code, _, err = cli(iris.ws, capsys, "validate", write(tmp_path, "l.json", spec(iris, method="ISOTONIC", fit={"mode": "RUN", "calibration_run": iris.baseline})), "--preflight")  # fmt: skip
    assert code == 2 and "leakage" in err
    code, _, err = cli(iris.ws, capsys, "validate", str(tmp_path / "missing.json"))
    assert code == 2 and "cannot read calibration spec" in err
    code, _, err = cli(iris.ws, capsys, "validate", write(tmp_path, "arr.json", "[1]"))
    assert code == 2 and "must be a JSON object" in err
    code, _, err = cli(iris.ws, capsys, "validate", write(tmp_path, "junk.json", "{nope"))
    assert code == 2 and "cannot read calibration spec" in err


def test_evaluate_list_inspect_compare_and_replay_round_trip(iris: Iris, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    assert json.loads(cli(iris.ws, capsys, "list")[1]) == {"analyses": []}
    f = write(tmp_path, "e.json", spec(iris, objects=["CLASSWISE", "TOP_LABEL"], method="PLATT", fit={"seed": 1}))  # fmt: skip
    code, out, _ = cli(iris.ws, capsys, "evaluate", f)
    doc = json.loads(out)
    aid = doc["analysis_id"]
    assert code in (0, 3) and doc["status"] == "COMPLETED" and aid.startswith("cba_")
    assert (
        doc["summary"]["baseline"]["status"] == "COMPUTED"
        and doc["summary"]["method"]["method"] == "PLATT"
    )
    assert "confidence is not automatically uncertainty" in doc["summary"]["note"]
    again = json.loads(cli(iris.ws, capsys, "evaluate", f)[1])
    assert again["analysis_id"] == aid  # the same request is the same record
    rows = json.loads(cli(iris.ws, capsys, "list")[1])["analyses"]
    assert [r["id"] for r in rows] == [aid] and rows[0]["method"] == "PLATT" and rows[0]["baseline"]["status"] == "COMPUTED"  # fmt: skip
    assert json.loads(cli(iris.ws, capsys, "list", "--status", "COMPLETE" if rows[0]["status"] == "COMPLETE" else "PARTIAL")[1])["analyses"]  # fmt: skip
    assert json.loads(cli(iris.ws, capsys, "list", "--baseline-run", "run_" + "0" * 32)[1])["analyses"] == []  # fmt: skip
    # inspect: provenance, per-context results, artifacts, and evidence documents
    code, out, _ = cli(iris.ws, capsys, "inspect", aid)
    ins = json.loads(out)
    assert (
        code == 0 and ins["provenance"]["provenance_fingerprint"] == ins["provenance_fingerprint"]
    )
    assert ins["provenance"]["binning"] == {"strategy": "UNIFORM", "n_bins": 5} and ins["provenance"]["calibration_method"] == "PLATT"  # fmt: skip
    assert {a["path"] for a in ins["artifacts"]} == {f"calibration/{n}.json" for n in ("spec", "predictions", "bins", "metrics", "uncertainty", "comparisons", "candidates", "summary")}  # fmt: skip
    assert {r["context_key"] for r in ins["results"]} == {"baseline", "calibrated"}
    assert cli(iris.ws, capsys, "inspect", aid)[1] == out  # deterministic JSON
    ev = json.loads(cli(iris.ws, capsys, "inspect", aid, "--document", "bins")[1])["document"]
    assert len(ev["baseline"]["top_label"]) == 5 and ev["baseline"]["classwise"]
    full = json.loads(cli(iris.ws, capsys, "inspect", aid, "--full")[1])["documents"]
    assert set(full) == {
        "spec",
        "predictions",
        "bins",
        "metrics",
        "uncertainty",
        "comparisons",
        "candidates",
        "summary",
    }
    cbr = ins["results"][0]["id"]
    one = json.loads(cli(iris.ws, capsys, "inspect", cbr)[1])
    assert one["analysis_id"] == aid and one["id"] == cbr
    # compare recomputes from the stored predictions and stores nothing; replay re-executes as a new run
    runs = n_runs(iris.ws)
    code, out, _ = cli(iris.ws, capsys, "compare", aid)
    cmp_ = json.loads(out)
    assert code == 0 and cmp_["reproduced"] is True and cmp_["differences"] == [] and "nothing was stored" in cmp_["not_stored"]  # fmt: skip
    assert cmp_["provenance_fingerprint"]["stored"] == cmp_["provenance_fingerprint"]["recomputed"] and n_runs(iris.ws) == runs  # fmt: skip
    code, out, _ = cli(iris.ws, capsys, "replay", aid)
    rep = json.loads(out)
    assert code == 0 and rep["deterministic"] is True and rep["differences"] == [] and n_runs(iris.ws) == runs + 1  # fmt: skip
    code, out, _ = cli(iris.ws, capsys, "replay", aid, "--format", "text")
    assert code == 0 and out.startswith("reproduced:")


def test_incomplete_evidence_exits_3_and_is_reported_as_partial(iris: Iris, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    f = write(tmp_path, "q.json", spec(iris, binning={"strategy": "QUANTILE", "n_bins": 20}))
    code, out, _ = cli(iris.ws, capsys, "evaluate", f)
    doc = json.loads(out)
    assert code == 3 and doc["analysis_status"] == "PARTIAL" and doc["summary"]["baseline"]["status"] == "INSUFFICIENT_EVIDENCE"  # fmt: skip
    assert doc["summary"]["baseline"]["metrics"] is None
    assert json.loads(cli(iris.ws, capsys, "list", "--status", "PARTIAL")[1])["analyses"]


def test_unknown_ids_missing_workspaces_and_help(iris: Iris, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    code, _, err = cli(iris.ws, capsys, "inspect", "cba_" + "0" * 32)
    assert code == 2 and "Traceback" not in err and err
    code, _, err = cli(iris.ws, capsys, "replay", "cba_" + "0" * 32)
    assert code == 2
    code = main(["--workspace", str(tmp_path / "nowhere"), "calibration", "list"])
    assert code == 2 and "no registry" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "no universal score" in " ".join(capsys.readouterr().out.split())

"""Builds one real, populated workspace exercised by tests/test_api_*.py, reusing the exact
engines/CLI surface test_cross_phase.py already exercises. No mocks, no synthetic API-only data."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experionyx.adapters.records import RegisteredDataset, RegisteredModel
from experionyx.adapters.registry import default_registries
from experionyx.artifacts import LocalArtifactStore
from experionyx.cli import main
from experionyx.execution import Executor
from experionyx.failures.engine import run_discovery
from experionyx.failures.entities import FailureMode
from experionyx.faults.demos import run_fault_demo
from experionyx.faults.entities import FaultAnalysis, FaultExperiment
from experionyx.reliability.engine import run_profile
from experionyx.reliability.spec import ProfileSpec
from experionyx.reliability.taxonomy import Scope
from experionyx.sqlite import SqliteRegistry

FEATS = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]


@dataclass
class ApiWorld:
    ws: Path
    investigation_id: str
    baseline_run_id: str
    fault_experiment_ids: tuple[str, ...]
    fault_analysis_ids: tuple[str, ...]
    model_id: str
    dataset_id: str
    failure_mode_ids: tuple[str, ...]
    drift_analysis_id: str
    stats_analysis_id: str
    reliability_profile_id: str
    graph_snapshot_id: str
    report_id: str
    dossier_id: str
    reproduction_attempt_id: str


def _cli(ws: Path, *args: str) -> dict[str, Any]:
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = main(["--workspace", str(ws), *args])
    out = buf.getvalue()
    if code == 2:
        raise AssertionError(f"cli {args} errored ({code}): {out}")
    return json.loads(out) if out.strip() else {}


def _write(tmp: Path, name: str, body: Any) -> str:
    p = tmp / name
    p.write_text(json.dumps(body), encoding="utf-8")
    return str(p)


def build_api_world(tmp_path: Path) -> ApiWorld:
    ws = tmp_path / "w"
    results = run_fault_demo("tabular", ws)
    fxps = tuple(r.fault_experiment.id for r in results)
    with SqliteRegistry(ws / "registry.sqlite") as reg:
        fx0 = reg.get(FaultExperiment, results[0].fault_experiment.id)
        baseline = fx0.baseline_run_id
        investigation_id = fx0.investigation_id
        model_id = reg.find(RegisteredModel)[0].id
        dataset_id = reg.find(RegisteredDataset)[0].id
        fault_analysis_ids = tuple(a.id for a in reg.find(FaultAnalysis))

    with SqliteRegistry(ws / "registry.sqlite") as reg:
        store = LocalArtifactStore(ws / "experiments")
        executor = Executor(
            reg, store, source_root=Path.cwd(), adapters=default_registries(), inputs_root=ws
        )
        disc = run_discovery(reg, store, executor, investigation_id, [], list(fxps))
        assert disc.status.value == "COMPLETED"
        failure_mode_ids = tuple(m.id for m in reg.find(FailureMode))

    drift_spec = {
        "baseline_run": baseline,
        "ordering": {"field": "index"},
        "reference": {"start": 0, "end": 75},
        "comparisons": [{"start": 75, "end": 150}],
        "features": [{"name": n, "type": "NUMERIC"} for n in FEATS],
        "config": {"min_samples": 5, "resamples": 100, "permutations": 200},
    }
    drift_doc = _cli(ws, "drift", "evaluate", _write(ws, "dr.json", drift_spec))
    drift_id = drift_doc["analysis_id"]

    stats_doc = _cli(
        ws,
        "stats",
        "compare",
        "--reference-values",
        "1,2,3,4,5",
        "--treatment-values",
        "2,3,4,5,7",
        "--resamples",
        "200",
        "--permutations",
        "200",
    )
    stats_id = stats_doc["id"]

    with SqliteRegistry(ws / "registry.sqlite") as reg:
        store = LocalArtifactStore(ws / "experiments")
        executor = Executor(
            reg, store, source_root=Path.cwd(), adapters=default_registries(), inputs_root=ws
        )
        spec = ProfileSpec(
            Scope.MODEL_DATASET_EVALUATION,
            baseline,
            fxps,
            (),
            failure_mode_ids[:2],
            (),
            (drift_id,),
        )
        prof = run_profile(reg, store, executor, investigation_id, spec)
    assert prof.profile_id is not None
    profile_id = prof.profile_id

    graph_spec = {"name": "api-tests", "version": "1.0.0"}
    graph_doc = _cli(
        ws, "graph", "build", _write(ws, "gr.json", graph_spec), "--investigation", investigation_id
    )
    snapshot_id = graph_doc["snapshot_id"]

    report_doc = _cli(
        ws,
        "report",
        "generate",
        "--investigation",
        investigation_id,
        "MODEL_RELIABILITY_REPORT",
    )
    report_id = report_doc["id"]

    dossier_doc = _cli(
        ws,
        "dossier",
        "build",
        "--investigation",
        investigation_id,
        "--question",
        "Is this model reliable under the applied faults?",
    )
    dossier_id = dossier_doc["id"]

    reproduce_doc = _cli(ws, "reproduce", "run", "--mode", "PROVENANCE_ONLY", "RUN", baseline)
    attempt_id = reproduce_doc["id"]

    return ApiWorld(
        ws,
        investigation_id,
        baseline,
        fxps,
        fault_analysis_ids,
        model_id,
        dataset_id,
        failure_mode_ids,
        drift_id,
        stats_id,
        profile_id,
        snapshot_id,
        report_id,
        dossier_id,
        attempt_id,
    )


__all__ = ["ApiWorld", "build_api_world"]

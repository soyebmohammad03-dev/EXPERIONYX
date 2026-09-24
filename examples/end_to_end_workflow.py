"""Canonical end-to-end research workflow: model/dataset -> baseline -> controlled fault
-> failure discovery -> drift -> statistics -> reliability -> graph -> reproducibility
-> report -> dossier -> immutable snapshot -> Markdown export, with the API/UI paths a
researcher would open to inspect each result. Real engines and a real CLI end to end, no mocks.

    pip install 'experionyx[sklearn,faults,viz]'
    python examples/end_to_end_workflow.py [workspace]
    experionyx --workspace <workspace> viz serve   # then open the printed URLs
"""

import json
import sys
from pathlib import Path
from typing import Any

from experionyx.adapters.records import RegisteredDataset, RegisteredModel
from experionyx.adapters.registry import default_registries
from experionyx.artifacts import LocalArtifactStore
from experionyx.cli import main as cli_main
from experionyx.execution import Executor
from experionyx.failures.engine import run_discovery
from experionyx.failures.entities import FailureMode
from experionyx.faults.demos import run_fault_demo
from experionyx.faults.entities import FaultAnalysis, FaultExperiment
from experionyx.reliability.engine import run_profile
from experionyx.reliability.spec import ProfileSpec
from experionyx.reliability.taxonomy import Scope
from experionyx.sqlite import SqliteRegistry

FEATURES = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]


def _cli(ws: Path, *args: str) -> dict[str, Any]:
    """Invoke the real CLI in-process and parse its JSON stdout (same entry point as the shell)."""
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli_main(["--workspace", str(ws), *args])
    out = buf.getvalue()
    if code == 2:
        raise SystemExit(f"cli {args} failed ({code}): {out}")
    return json.loads(out) if out.strip() else {}


def _write(ws: Path, name: str, body: Any) -> str:
    p = ws / name
    p.write_text(json.dumps(body), encoding="utf-8")
    return str(p)


def main(ws: Path) -> None:
    print(f"workspace: {ws}\n")

    # 1-3. register model/dataset, run baseline, apply a controlled fault (all one real call).
    print("1. register model/dataset + baseline + controlled fault experiments ...")
    results = run_fault_demo("tabular", ws)
    fault_experiment_ids = tuple(r.fault_experiment.id for r in results)
    with SqliteRegistry(ws / "registry.sqlite") as reg:
        fx0 = reg.get(FaultExperiment, results[0].fault_experiment.id)
        baseline_run_id = fx0.baseline_run_id
        investigation_id = fx0.investigation_id
        model_id = reg.find(RegisteredModel)[0].id
        dataset_id = reg.find(RegisteredDataset)[0].id
        fault_analysis_ids = tuple(a.id for a in reg.find(FaultAnalysis))
    print(f"   model={model_id}  dataset={dataset_id}")
    print(f"   investigation={investigation_id}  baseline_run={baseline_run_id}")
    print(f"   fault_experiments={fault_experiment_ids}")

    # 4. discover/analyze failures from the baseline + fault evidence.
    print("2. discover failures ...")
    with SqliteRegistry(ws / "registry.sqlite") as reg:
        store = LocalArtifactStore(ws / "experiments")
        executor = Executor(
            reg, store, source_root=Path.cwd(), adapters=default_registries(), inputs_root=ws
        )
        fx_ids = list(fault_experiment_ids)
        discovery = run_discovery(reg, store, executor, investigation_id, [], fx_ids)
        if discovery.status.value != "COMPLETED":
            raise SystemExit(f"failure discovery did not complete: {discovery}")
        failure_mode_ids = tuple(m.id for m in reg.find(FailureMode))
    print(f"   failure_modes={failure_mode_ids}")

    # 5. distribution shift over the baseline's own predictions (two windows of one real split).
    print("3. drift analysis ...")
    drift_spec = {
        "baseline_run": baseline_run_id,
        "ordering": {"field": "index"},
        "reference": {"start": 0, "end": 75},
        "comparisons": [{"start": 75, "end": 150}],
        "features": [{"name": name, "type": "NUMERIC"} for name in FEATURES],
        "config": {"min_samples": 5, "resamples": 100, "permutations": 200},
    }
    drift_spec_path = _write(ws, "drift_spec.json", drift_spec)
    drift_analysis_id = _cli(ws, "drift", "evaluate", drift_spec_path)["analysis_id"]
    print(f"   drift_analysis={drift_analysis_id}")

    # 6. a standalone statistical comparison, registered as its own persisted analysis.
    print("4. statistical comparison ...")
    stats_analysis_id = _cli(
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
    )["id"]
    print(f"   stats_analysis={stats_analysis_id}")

    # 7. reliability evidence: an explicit per-dimension summary, never a single score.
    print("5. reliability profile ...")
    with SqliteRegistry(ws / "registry.sqlite") as reg:
        store = LocalArtifactStore(ws / "experiments")
        executor = Executor(
            reg, store, source_root=Path.cwd(), adapters=default_registries(), inputs_root=ws
        )
        spec = ProfileSpec(
            Scope.MODEL_DATASET_EVALUATION,
            baseline_run_id,
            fault_experiment_ids,
            (),
            failure_mode_ids[:2],
            (),
            (drift_analysis_id,),
        )
        profile_id = run_profile(reg, store, executor, investigation_id, spec).profile_id
    if profile_id is None:
        raise SystemExit("reliability profile was not produced")
    print(f"   reliability_profile={profile_id}")

    # 8. inspect the investigation's evidence as one queryable graph snapshot.
    print("6. graph snapshot ...")
    graph_spec = {"name": "end-to-end-example", "version": "1.0.0"}
    graph_spec_path = _write(ws, "graph_spec.json", graph_spec)
    snapshot_id = _cli(ws, "graph", "build", graph_spec_path, "--investigation", investigation_id)[
        "snapshot_id"
    ]
    print(f"   graph_snapshot={snapshot_id}")

    # 9. reproducibility check against the baseline run (provenance-only: no re-execution here).
    print("7. reproducibility check ...")
    attempt_id = _cli(ws, "reproduce", "run", "--mode", "PROVENANCE_ONLY", "RUN", baseline_run_id)[
        "id"
    ]
    print(f"   reproduction_attempt={attempt_id}")
    # Benchmark/leaderboard is skipped here: a protocol+submission needs its own registered
    # protocol grid, which would roughly double this example's size for one more subsystem
    # already covered end-to-end in tests/test_leaderboard_e2e.py.

    # 10-12. report -> dossier -> immutable snapshot, all assembled from evidence above.
    print("8. research report ...")
    report_id = _cli(
        ws, "report", "generate", "--investigation", investigation_id, "MODEL_RELIABILITY_REPORT"
    )["id"]
    print(f"   report={report_id}")

    print("9. evidence dossier ...")
    dossier_id = _cli(
        ws,
        "dossier",
        "build",
        "--investigation",
        investigation_id,
        "--question",
        "Is this model reliable under the applied faults?",
    )["id"]
    print(f"   dossier={dossier_id}")

    print("10. immutable dossier snapshot ...")
    snapshot_doc = _cli(ws, "dossier", "snapshot", dossier_id)
    print(f"    dossier_snapshot={snapshot_doc.get('id', snapshot_doc)}")

    # 13. Markdown export -- the canonical, human-readable artifact of this whole run.
    print("11. Markdown export ...")
    report_md = ws / "report.md"
    dossier_md = ws / "dossier.md"
    _cli(ws, "report", "export", report_id, "--output", str(report_md))
    _cli(ws, "dossier", "export", dossier_id, "--output", str(dossier_md))
    print(f"    {report_md}\n    {dossier_md}")

    # 14. what a researcher would open next: `experionyx viz serve` over this same workspace.
    print(f"\n12. inspect via API/UI -- run `experionyx --workspace {ws} viz serve`, then open:")
    for path in (
        f"/#/investigations/{investigation_id}",
        f"/#/reliability/profiles/{profile_id}",
        f"/#/failures/modes/{failure_mode_ids[0]}" if failure_mode_ids else "/#/failures",
        f"/#/faults/analyses/{fault_analysis_ids[0]}" if fault_analysis_ids else "/#/faults",
        f"/#/drift/analyses/{drift_analysis_id}",
        f"/#/stats/analyses/{stats_analysis_id}",
        f"/#/reproducibility/attempts/{attempt_id}",
        f"/#/graph/snapshots/{snapshot_id}",
        f"/#/reports/{report_id}",
        f"/#/dossiers/{dossier_id}",
    ):
        print(f"    http://127.0.0.1:8420{path}")
    print(f"\n    http://127.0.0.1:8420/api/reports/{report_id}/export.md  (raw Markdown)")
    print(f"    http://127.0.0.1:8420/api/dossiers/{dossier_id}/export.md  (raw Markdown)")


if __name__ == "__main__":
    workspace = Path(sys.argv[1] if len(sys.argv) > 1 else ".experionyx-example")
    main(workspace)

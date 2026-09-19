"""Read stored fault-experiment results (verified against their artifact digests)."""

import json
from pathlib import Path

from experionyx.artifacts import ArtifactStore, LocalArtifactStore
from experionyx.domain import Artifact, Run
from experionyx.errors import ExperionyxError
from experionyx.evaluation.serial import from_jsonable
from experionyx.faults.analysis import FaultAnalysisResult
from experionyx.faults.entities import FaultAnalysis, FaultExperiment
from experionyx.registry import Registry


def read_artifact(registry: Registry, store: ArtifactStore, run_id: str, path: str) -> object:
    """The JSON content of `path` in a run's artifacts, after verifying its digest."""
    run = registry.get(Run, run_id)
    found = [a for a in registry.find(Artifact, run_id=run_id) if a.path == path]
    if not found:
        raise ExperionyxError(f"run {run_id} has no artifact {path!r}")
    store.verify(run, found[0])
    if not isinstance(store, LocalArtifactStore):
        raise ExperionyxError("reading artifact contents requires a LocalArtifactStore")
    return json.loads(
        Path(store.run_dir(run) / "artifacts" / found[0].path).read_text(encoding="utf-8")
    )


def analysis_run_id(registry: Registry, fault_experiment_id: str) -> str:
    found = registry.find(FaultAnalysis, fault_experiment_id=fault_experiment_id)
    if not found:
        raise ExperionyxError(f"fault experiment {fault_experiment_id} has no analysis run")
    return found[-1].run_id


def load_analysis(
    registry: Registry, store: ArtifactStore, fault_experiment_id: str
) -> FaultAnalysisResult:
    registry.get(FaultExperiment, fault_experiment_id)
    data = read_artifact(
        registry, store, analysis_run_id(registry, fault_experiment_id), "fault/analysis.json"
    )
    result: FaultAnalysisResult = from_jsonable(FaultAnalysisResult, data)
    return result


def summary_rows(result: FaultAnalysisResult) -> list[dict[str, object]]:
    """One row per sweep point: what was measured, with its classification."""
    return [
        {
            "point": p.point_index,
            "parameter": p.parameter_name,
            "value": p.parameter_value,
            "trials": f"{p.n_completed}/{p.n_trials}",
            "baseline": result.baseline_value,
            "faulted_mean": p.primary_faulted.mean,
            "deterioration_mean": p.primary_deterioration.mean,
            "deterioration_ci": None
            if p.primary_deterioration.ci_lower is None
            else [p.primary_deterioration.ci_lower, p.primary_deterioration.ci_upper],
            "classification": p.assessment.classification.value,
        }
        for p in result.points
    ]

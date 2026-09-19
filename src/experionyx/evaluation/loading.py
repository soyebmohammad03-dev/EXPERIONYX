"""Load a run's stored evaluation, verifying the artifact digest first."""

import json
from pathlib import Path

from experionyx.artifacts import ArtifactStore, LocalArtifactStore
from experionyx.domain import Artifact, Run
from experionyx.errors import ExperionyxError
from experionyx.evaluation.results import EvaluationResult
from experionyx.evaluation.serial import from_jsonable
from experionyx.registry import Registry

EVALUATION_PATH = "evaluation/evaluation.json"


def load_evaluation(registry: Registry, store: ArtifactStore, run_id: str) -> EvaluationResult:
    """The strictly parsed EvaluationResult of `run_id`; raises if the artifact is missing or has
    changed since it was registered."""
    run = registry.get(Run, run_id)
    found = [a for a in registry.find(Artifact, run_id=run.id) if a.path == EVALUATION_PATH]
    if not found:
        raise ExperionyxError(f"run {run.id} has no evaluation artifact")
    store.verify(run, found[0])
    if not isinstance(store, LocalArtifactStore):
        raise ExperionyxError("reading artifact contents requires a LocalArtifactStore")
    text = Path(store.run_dir(run) / "artifacts" / found[0].path).read_text(encoding="utf-8")
    result: EvaluationResult = from_jsonable(EvaluationResult, json.loads(text))
    return result

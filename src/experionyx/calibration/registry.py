"""The calibration registry: list analyses and their per-context results, and read their
digest-verified documents, provenance and the candidate failure signals."""

import builtins  # used in string annotations
from typing import Any

from experionyx.artifacts import ArtifactStore
from experionyx.calibration.engine import ARTIFACT_DIR, DOCUMENTS
from experionyx.calibration.entities import CalibrationAnalysis, CalibrationResult
from experionyx.domain import Artifact, to_jsonable
from experionyx.errors import ValidationError
from experionyx.faults.report import read_artifact
from experionyx.provenance import Provenance
from experionyx.registry import Registry


class CalibrationRegistry:
    def __init__(self, registry: Registry, store: ArtifactStore | None = None) -> None:
        self.reg, self.store = registry, store

    def analysis(self, analysis_id: str) -> CalibrationAnalysis:
        return self.reg.get(CalibrationAnalysis, analysis_id)

    def analyses(self, **columns: str) -> "builtins.list[CalibrationAnalysis]":
        return sorted(self.reg.find(CalibrationAnalysis, **columns), key=lambda a: a.id)

    def results(self, analysis_id: str, **columns: str) -> "builtins.list[CalibrationResult]":
        return sorted(self.reg.find(CalibrationResult, analysis_id=analysis_id, **columns), key=lambda r: r.context_key)  # fmt: skip

    def artifacts(self, analysis_id: str) -> "builtins.list[Artifact]":
        return [a for a in self.reg.find(Artifact, run_id=self.analysis(analysis_id).run_id) if a.path.startswith(f"{ARTIFACT_DIR}/")]  # fmt: skip

    def document(self, analysis_id: str, name: str) -> Any:
        if self.store is None:
            raise ValidationError("an artifact store is needed to read calibration artifacts")
        if name not in DOCUMENTS:
            raise ValidationError(f"unknown document {name!r}; choose from {list(DOCUMENTS)}")
        return read_artifact(self.reg, self.store, self.analysis(analysis_id).run_id, f"{ARTIFACT_DIR}/{name}.json")  # fmt: skip

    def failure_candidates(self, analysis_id: str) -> Any:
        """Sample-level candidate signals (confident errors, unconfident successes) for failure
        analysis. They are evidence, not failure modes."""
        return self.document(analysis_id, "candidates")

    def provenance(self, analysis_id: str) -> dict[str, object]:
        a = self.analysis(analysis_id)
        spec = self.document(analysis_id, "spec") if self.store is not None else {}
        recorded = self.reg.find(Provenance, run_id=a.run_id)
        run_prov = None if not recorded else {
            "source_revision": to_jsonable(recorded[0].source), "environment_id": recorded[0].environment_id,
            "dependency_digest": recorded[0].dependency_digest, "executor_version": recorded[0].executor_version,
            "seed": recorded[0].seed, "run_fingerprint": recorded[0].fingerprint,
        }  # fmt: skip
        keys = ("model", "dataset", "split", "evaluation_config_hash", "prediction_representation", "score_assumption", "target_representation", "binning", "calibration_method", "calibration_split", "bootstrap", "lineage", "source_runs", "analysis_version")  # fmt: skip
        return {
            "analysis_id": a.id, "run_id": a.run_id, "spec_id": a.spec_id, "baseline_run_id": a.baseline_run_id,
            "dataset_fingerprint": a.dataset_fingerprint, "provenance_fingerprint": a.provenance_fingerprint,
            "run_provenance": run_prov, **{k: spec.get(k) for k in keys},
        }  # fmt: skip

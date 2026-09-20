"""The quality registry: list analyses and their per-check results, and read their digest-verified
documents and provenance."""

import builtins  # used in string annotations
from typing import Any

from experionyx.artifacts import ArtifactStore
from experionyx.data_quality.entities import QualityAnalysis, QualityCheck
from experionyx.domain import Artifact, to_jsonable
from experionyx.errors import ValidationError
from experionyx.faults.report import read_artifact
from experionyx.provenance import Provenance
from experionyx.registry import Registry

DOCUMENTS = ("spec", "checks", "observations", "violations", "summary")


class QualityRegistry:
    def __init__(self, registry: Registry, store: ArtifactStore | None = None) -> None:
        self.reg, self.store = registry, store

    def analysis(self, analysis_id: str) -> QualityAnalysis:
        return self.reg.get(QualityAnalysis, analysis_id)

    def analyses(self, **columns: str) -> "builtins.list[QualityAnalysis]":
        return sorted(self.reg.find(QualityAnalysis, **columns), key=lambda a: a.id)

    def checks(self, analysis_id: str, **columns: str) -> "builtins.list[QualityCheck]":
        return sorted(self.reg.find(QualityCheck, analysis_id=analysis_id, **columns), key=lambda c: (c.check_type, c.scope, c.id))  # fmt: skip

    def artifacts(self, analysis_id: str) -> "builtins.list[Artifact]":
        return [a for a in self.reg.find(Artifact, run_id=self.analysis(analysis_id).run_id) if a.path.startswith("data_quality/")]  # fmt: skip

    def document(self, analysis_id: str, name: str) -> Any:
        if self.store is None:
            raise ValidationError("an artifact store is needed to read data-quality artifacts")
        if name not in DOCUMENTS:
            raise ValidationError(f"unknown document {name!r}; choose from {list(DOCUMENTS)}")
        return read_artifact(self.reg, self.store, self.analysis(analysis_id).run_id, f"data_quality/{name}.json")  # fmt: skip

    def provenance(self, analysis_id: str) -> dict[str, object]:
        a = self.analysis(analysis_id)
        spec = self.document(analysis_id, "spec") if self.store is not None else {}
        recorded = self.reg.find(Provenance, run_id=a.run_id)
        run_prov = None if not recorded else {
            "source_revision": to_jsonable(recorded[0].source), "environment_id": recorded[0].environment_id,
            "dependency_digest": recorded[0].dependency_digest, "executor_version": recorded[0].executor_version,
            "seed": recorded[0].seed, "run_fingerprint": recorded[0].fingerprint,
        }  # fmt: skip
        keys = ("dataset", "columns", "splits", "check_ids", "thresholds", "random_seed", "versions", "slice_ids", "slice_membership", "missing_values", "artifact_digests")  # fmt: skip
        return {
            "analysis_id": a.id, "run_id": a.run_id, "spec_id": a.spec_id, "dataset_id": a.dataset_id,
            "dataset_fingerprint": a.dataset_fingerprint, "provenance_fingerprint": a.provenance_fingerprint,
            "run_provenance": run_prov, **{k: spec.get(k) for k in keys},
        }  # fmt: skip

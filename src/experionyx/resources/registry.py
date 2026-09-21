"""The resource registry: list analyses and their trials, and read their digest-verified documents
and provenance."""

import builtins  # used in string annotations
from typing import Any

from experionyx.artifacts import ArtifactStore
from experionyx.domain import Artifact, to_jsonable
from experionyx.errors import ValidationError
from experionyx.faults.report import read_artifact
from experionyx.provenance import Provenance
from experionyx.registry import Registry
from experionyx.resources.engine import ARTIFACT_DIR, DOCUMENTS
from experionyx.resources.entities import ResourceAnalysis, ResourceTrial


class ResourceRegistry:
    def __init__(self, registry: Registry, store: ArtifactStore | None = None) -> None:
        self.reg, self.store = registry, store

    def analysis(self, analysis_id: str) -> ResourceAnalysis:
        return self.reg.get(ResourceAnalysis, analysis_id)

    def analyses(self, **columns: str) -> "builtins.list[ResourceAnalysis]":
        return sorted(
            self.reg.find(ResourceAnalysis, **columns), key=lambda a: (a.created_at, a.id)
        )

    def trials(self, analysis_id: str, **columns: str) -> "builtins.list[ResourceTrial]":
        return sorted(
            self.reg.find(ResourceTrial, analysis_id=analysis_id, **columns),
            key=lambda t: (t.phase, t.trial_index),
        )

    def artifacts(self, analysis_id: str) -> "builtins.list[Artifact]":
        return [
            a
            for a in self.reg.find(Artifact, run_id=self.analysis(analysis_id).run_id)
            if a.path.startswith(f"{ARTIFACT_DIR}/")
        ]

    def document(self, analysis_id: str, name: str) -> Any:
        if self.store is None:
            raise ValidationError("an artifact store is needed to read resource artifacts")
        if name not in DOCUMENTS:
            raise ValidationError(f"unknown document {name!r}; choose from {list(DOCUMENTS)}")
        return read_artifact(
            self.reg, self.store, self.analysis(analysis_id).run_id, f"{ARTIFACT_DIR}/{name}.json"
        )

    def provenance(self, analysis_id: str) -> dict[str, Any]:
        a = self.analysis(analysis_id)
        spec = self.document(analysis_id, "spec") if self.store is not None else {}
        env = self.document(analysis_id, "environment") if self.store is not None else {}
        recorded = self.reg.find(Provenance, run_id=a.run_id)
        run_prov = (
            None
            if not recorded
            else {
                "source_revision": to_jsonable(recorded[0].source),
                "environment_id": recorded[0].environment_id,
                "dependency_digest": recorded[0].dependency_digest,
                "executor_version": recorded[0].executor_version,
                "seed": recorded[0].seed,
                "run_fingerprint": recorded[0].fingerprint,
                "execution": to_jsonable(recorded[0].execution),
            }
        )
        keys = (
            "model",
            "dataset",
            "split",
            "source_revision",
            "environment_id",
            "seed",
            "stress_ids",
            "workload_digest",
            "analysis_version",
        )
        s = a.spec
        return {
            "analysis_id": a.id,
            "run_id": a.run_id,
            "spec_id": a.spec_id,
            "provenance_fingerprint": a.provenance_fingerprint,
            "run_provenance": run_prov,
            **{k: spec.get(k) for k in keys},
            "resource_configuration": {
                k: to_jsonable(s.get(k))
                for k in (
                    "operation",
                    "batch_size",
                    "n_samples",
                    "repeats",
                    "warmup_trials",
                    "workers",
                    "timeout_seconds",
                    "measurement",
                    "subset",
                    "stress",
                    "device",
                )
            },
            "artifact_digests": {x.path: x.digest for x in self.artifacts(a.id)},
            "environment": {
                k: env.get(k)
                for k in (
                    "environment_id",
                    "python_version",
                    "os",
                    "machine",
                    "cpu_count",
                    "measurement_backends",
                )
            },
        }

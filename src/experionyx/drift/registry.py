"""The drift registry: register window definitions (idempotent per definition), list and search
them, and retrieve analyses with their digest-verified documents and provenance."""

import builtins  # used in a string annotation
from datetime import UTC, datetime
from typing import Any

from experionyx.artifacts import ArtifactStore
from experionyx.domain import Artifact, to_jsonable
from experionyx.drift.entities import DriftAnalysis, DriftWindow
from experionyx.drift.spec import Ordering, TemporalWindow
from experionyx.errors import ValidationError
from experionyx.faults.report import read_artifact
from experionyx.provenance import Provenance
from experionyx.registry import Registry

DOCUMENTS = ("spec", "windows", "feature_results", "distribution_results", "performance_results", "summary")  # fmt: skip


class DriftRegistry:
    def __init__(self, registry: Registry, store: ArtifactStore | None = None) -> None:
        self.reg, self.store = registry, store

    def register_window(self, w: TemporalWindow, ordering: Ordering, now: datetime | None = None) -> tuple[DriftWindow, bool]:  # fmt: skip
        """(record, created). An equal definition is the SAME record whatever it is named."""
        entity = DriftWindow.of(w, ordering.field, now or datetime.now(UTC))
        if self.reg.exists(DriftWindow, entity.id):
            return self.reg.get(DriftWindow, entity.id), False
        self.reg.add(entity)
        return entity, True

    def window(self, window_id: str) -> DriftWindow:
        return self.reg.get(DriftWindow, window_id)

    def windows(self, *, ordering: str | None = None, role: str | None = None) -> "builtins.list[DriftWindow]":  # fmt: skip
        cols = {k: v for k, v in (("ordering_field", ordering), ("role", role)) if v}
        return sorted(self.reg.find(DriftWindow, **cols), key=lambda w: w.id)

    def analysis(self, analysis_id: str) -> DriftAnalysis:
        return self.reg.get(DriftAnalysis, analysis_id)

    def analyses(self, **columns: str) -> "builtins.list[DriftAnalysis]":
        return sorted(self.reg.find(DriftAnalysis, **columns), key=lambda a: a.id)

    def artifacts(self, analysis_id: str) -> "builtins.list[Artifact]":
        return [
            a
            for a in self.reg.find(Artifact, run_id=self.analysis(analysis_id).run_id)
            if a.path.startswith("drift/")
        ]

    def document(self, analysis_id: str, name: str) -> Any:
        if self.store is None:
            raise ValidationError("an artifact store is needed to read drift artifacts")
        if name not in DOCUMENTS:
            raise ValidationError(f"unknown document {name!r}; choose from {list(DOCUMENTS)}")
        return read_artifact(self.reg, self.store, self.analysis(analysis_id).run_id, f"drift/{name}.json")  # fmt: skip

    def provenance(self, analysis_id: str) -> dict[str, object]:
        a = self.analysis(analysis_id)
        spec = self.document(analysis_id, "spec") if self.store is not None else {}
        keys = ("model_fingerprint", "model_id", "dataset_id", "split", "evaluation_config_hash", "ordering", "window_ids", "features", "metrics", "methods", "random_seed", "source_revision", "versions", "slice_ids", "failure_modes")  # fmt: skip
        recorded = self.reg.find(Provenance, run_id=a.run_id)
        run_prov = None if not recorded else {
            "source_revision": to_jsonable(recorded[0].source), "environment_id": recorded[0].environment_id,
            "dependency_digest": recorded[0].dependency_digest, "executor_version": recorded[0].executor_version,
            "seed": recorded[0].seed, "run_fingerprint": recorded[0].fingerprint,
        }  # fmt: skip
        return {
            "analysis_id": a.id, "run_id": a.run_id, "spec_id": a.spec_id, "run_provenance": run_prov,
            "baseline_run_id": a.baseline_run_id, "dataset_fingerprint": a.dataset_fingerprint,
            "provenance_fingerprint": a.provenance_fingerprint,
            **{k: spec.get(k) for k in keys},
        }  # fmt: skip

"""The slice registry: register (idempotent per logical definition), get, list, search, evaluate a
slice against a baseline run, and retrieve analyses, their digest-verified documents and provenance."""

import builtins  # used in a string annotation
from datetime import UTC, datetime
from typing import Any

from experionyx.adapters.base import DatasetAdapter
from experionyx.artifacts import ArtifactStore
from experionyx.domain import Artifact, to_jsonable
from experionyx.errors import ValidationError
from experionyx.faults.report import read_artifact
from experionyx.registry import Registry
from experionyx.slices.data import load_baseline, sample_table
from experionyx.slices.entities import Slice, SliceAnalysis
from experionyx.slices.evaluate import Membership, evaluate
from experionyx.slices.spec import SliceSpec

DOCUMENTS = ("spec", "membership", "results", "summary")


class SliceRegistry:
    def __init__(self, registry: Registry, store: ArtifactStore | None = None) -> None:
        self.reg, self.store = registry, store

    def register(self, spec: SliceSpec, now: datetime | None = None) -> tuple[Slice, bool]:
        """(record, created). An equivalent definition (same normalized condition) is the SAME
        record whatever it is named, so it is returned, not duplicated."""
        entity = Slice.of(spec, now or datetime.now(UTC))
        if self.reg.exists(Slice, entity.id):
            return self.reg.get(Slice, entity.id), False
        self.reg.add(entity)
        return entity, True

    def get(self, slice_id: str) -> Slice:
        return self.reg.get(Slice, slice_id)

    def search(
        self, *, name: str | None = None, field: str | None = None, text: str | None = None,
        static: bool | None = None,
    ) -> list[Slice]:  # fmt: skip
        found = self.reg.find(Slice, **({"name": name} if name else {}))
        if field:
            found = [s for s in found if field in s.fields]
        if text:
            found = [s for s in found if text.lower() in s.description.lower()]
        if static is not None:
            found = [s for s in found if s.spec().static is static]
        return sorted(found, key=lambda s: s.id)

    def evaluate(
        self, spec: SliceSpec, baseline_run: str, dataset: DatasetAdapter | None = None
    ) -> Membership:
        """Membership of `spec` over a baseline run's samples. Nothing is stored."""
        if self.store is None:
            raise ValidationError("an artifact store is needed to read a run's samples")
        base = load_baseline(self.reg, self.store, baseline_run)
        return evaluate(spec, sample_table(base, dataset, spec.fields))

    def analysis(self, analysis_id: str) -> SliceAnalysis:
        return self.reg.get(SliceAnalysis, analysis_id)

    def analyses(self, **columns: str) -> list[SliceAnalysis]:
        return sorted(self.reg.find(SliceAnalysis, **columns), key=lambda a: a.id)

    def artifacts(self, analysis_id: str) -> list[Artifact]:
        return [
            a
            for a in self.reg.find(Artifact, run_id=self.analysis(analysis_id).run_id)
            if a.path.startswith("slice/")
        ]

    def document(self, analysis_id: str, name: str) -> Any:
        if self.store is None:
            raise ValidationError("an artifact store is needed to read slice artifacts")
        if name not in DOCUMENTS:
            raise ValidationError(f"unknown document {name!r}; choose from {list(DOCUMENTS)}")
        return read_artifact(
            self.reg, self.store, self.analysis(analysis_id).run_id, f"slice/{name}.json"
        )

    def provenance(self, analysis_id: str) -> dict[str, object]:
        a = self.analysis(analysis_id)
        spec = self.document(analysis_id, "spec") if self.store is not None else {}
        return {
            "analysis_id": a.id, "run_id": a.run_id, "spec_id": a.spec_id,
            "baseline_run_id": a.baseline_run_id, "dataset_fingerprint": a.dataset_fingerprint,
            "provenance_fingerprint": a.provenance_fingerprint,
            "split": spec.get("split"), "evaluation_config_hash": spec.get("evaluation_config_hash"),
            "slice_ids": spec.get("slice_ids"), "metric_config": to_jsonable(a.spec.get("config")),
            "source_runs": spec.get("source_runs"),
        }  # fmt: skip

    def list(
        self,
    ) -> "builtins.list[Slice]":  # defined last: the name shadows the builtin in this class body
        return sorted(self.reg.find(Slice), key=lambda s: s.id)

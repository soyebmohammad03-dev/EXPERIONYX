"""The interaction registry: register, get, list, search, and retrieve evidence, raw trials and
artifacts. IDs are content-addressed, so a logical interaction (investigation + design spec) can
only be registered once."""

from dataclasses import dataclass
from typing import Any

from experionyx.artifacts import ArtifactStore
from experionyx.domain import Artifact, Run
from experionyx.errors import DuplicateError
from experionyx.failures.entities import FailureRelationship
from experionyx.failures.taxonomy import NodeKind
from experionyx.faults.report import read_artifact
from experionyx.interactions.entities import (
    InteractionAnalysis,
    InteractionEffect,
    InteractionEvidence,
)
from experionyx.registry import Registry


@dataclass(frozen=True)
class Related:
    analysis_id: str
    relation: str  # SAME_STRUCTURE | POTENTIALLY_RELATED (never 'same interaction')
    differences: tuple[str, ...]


class InteractionRegistry:
    def __init__(self, registry: Registry, store: ArtifactStore | None = None) -> None:
        self.reg, self.store = registry, store

    def register(
        self, analysis: InteractionAnalysis, effects: list[InteractionEffect] | None = None
    ) -> InteractionAnalysis:
        if self.reg.exists(InteractionAnalysis, analysis.id):
            raise DuplicateError(
                f"interaction {analysis.id} is already registered (same investigation and design spec)"
            )
        with self.reg.transaction():
            self.reg.add(analysis)
            for e in effects or []:
                self.reg.add(e)
        return analysis

    def get(self, analysis_id: str) -> InteractionAnalysis:
        return self.reg.get(InteractionAnalysis, analysis_id)

    def find(self, **columns: str) -> list[InteractionAnalysis]:
        return self.reg.find(InteractionAnalysis, **columns)

    def search(
        self,
        *,
        fault: str | None = None,
        metric: str | None = None,
        status: str | None = None,
        failure_mode: str | None = None,
        experiment: str | None = None,
        dataset: str | None = None,
        model: str | None = None,
        primary_class: str | None = None,
        investigation: str | None = None,
    ) -> list[InteractionAnalysis]:
        cols = {
            k: v
            for k, v in (
                ("status", status),
                ("dataset_fingerprint", dataset),
                ("model_fingerprint", model),
                ("primary_class", primary_class),
                ("investigation_id", investigation),
            )
            if v
        }
        found = self.reg.find(InteractionAnalysis, **cols)
        if fault:
            found = [a for a in found if fault in (a.fault_a, a.fault_b) or fault in _types(a)]
        if metric:
            with_metric = {e.analysis_id for e in self.reg.find(InteractionEffect, measure=metric)}
            found = [a for a in found if a.id in with_metric]
        if failure_mode:
            linked = {
                r.object_id
                for r in self.reg.find(FailureRelationship, subject_id=failure_mode)
                if r.object_kind is NodeKind.INTERACTION
            }
            found = [a for a in found if a.id in linked]
        if experiment:
            found = [a for a in found if experiment in self._experiments(a)]
        return sorted(found, key=lambda a: a.id)

    def _experiments(self, a: InteractionAnalysis) -> set[str]:
        cells: Any = a.spec["cells"]
        return {self.reg.get(Run, str(r)).experiment_id for runs in cells.values() for r in runs}

    def effects(self, analysis_id: str, *, level: str | None = None) -> list[InteractionEffect]:
        extra = {"level": level} if level else {}
        return self.reg.find(InteractionEffect, analysis_id=analysis_id, **extra)

    def evidence(self, analysis_id: str) -> list[InteractionEvidence]:
        return self.reg.find(InteractionEvidence, analysis_id=analysis_id)

    def artifacts(self, analysis_id: str) -> list[Artifact]:
        return [
            a
            for a in self.reg.find(Artifact, run_id=self.get(analysis_id).run_id)
            if a.path.startswith("interaction/")
        ]

    def raw_trials(self, analysis_id: str) -> object:
        """Every trial of every cell (the per-trial values behind each aggregate), digest-verified."""
        if self.store is None:
            raise ValueError("an artifact store is needed to read raw trials")
        return read_artifact(
            self.reg, self.store, self.get(analysis_id).run_id, "interaction/trials.json"
        )

    def artifact(self, analysis_id: str, name: str) -> object:
        if self.store is None:
            raise ValueError("an artifact store is needed to read artifacts")
        return read_artifact(
            self.reg, self.store, self.get(analysis_id).run_id, f"interaction/{name}.json"
        )

    def provenance(self, analysis_id: str) -> dict[str, object]:
        a = self.get(analysis_id)
        return {
            "analysis_id": a.id,
            "run_id": a.run_id,
            "spec_id": a.spec_id,
            "provenance_fingerprint": a.provenance_fingerprint,
            "model_fingerprint": a.model_fingerprint,
            "dataset_fingerprint": a.dataset_fingerprint,
            "cells": a.spec["cells"],
        }

    def related(self, analysis_id: str) -> list[Related]:
        """Analyses with the same or a similar STRUCTURE. Nothing is merged and parameter
        differences stay visible; a match is a pointer for a human, not an identification."""
        a = self.get(analysis_id)
        out: list[Related] = []
        for o in self.reg.find(InteractionAnalysis):
            if o.id == a.id:
                continue
            if o.structural_key == a.structural_key and o.primary_metric == a.primary_metric:
                out.append(
                    Related(
                        o.id,
                        "SAME_STRUCTURE",
                        ("different trial runs/seeds or configuration only",),
                    )
                )
                continue
            diffs = _differences(a, o)
            if diffs is not None:
                out.append(Related(o.id, "POTENTIALLY_RELATED", tuple(diffs)))
        return sorted(out, key=lambda r: r.analysis_id)


def _desc(a: InteractionAnalysis) -> dict[str, Any]:
    d: Any = a.summary["descriptors"]
    return dict(d)


def _types(a: InteractionAnalysis) -> set[str]:
    d = _desc(a)
    return {str(d["fault_a"]["type"]), str(d["fault_b"]["type"])}


def _differences(a: InteractionAnalysis, o: InteractionAnalysis) -> list[str] | None:
    """None when the two are not even plausibly related (different fault types/versions in the
    same roles, or a different metric); otherwise every visible difference."""
    da, do = _desc(a), _desc(o)
    if a.primary_metric != o.primary_metric:
        return None
    diffs: list[str] = []
    for role in ("fault_a", "fault_b"):
        fa, fo = da[role], do[role]
        if (fa["type"], fa["version"]) != (fo["type"], fo["version"]):
            return None
        for k in ("parameters", "scope"):
            if fa[k] != fo[k]:
                diffs.append(f"{role} {k}: {fa[k]} vs {fo[k]}")
    for label, x, y in (
        ("model_fingerprint", a.model_fingerprint, o.model_fingerprint),
        ("dataset_fingerprint", a.dataset_fingerprint, o.dataset_fingerprint),
        ("split", da.get("split"), do.get("split")),
        (
            "evaluation_config_hash",
            da.get("evaluation_config_hash"),
            do.get("evaluation_config_hash"),
        ),
        ("order_analysis", da.get("order_analysis"), do.get("order_analysis")),
    ):
        if x != y:
            diffs.append(f"{label} differs")
    return diffs or None

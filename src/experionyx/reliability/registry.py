"""The reliability profile registry: register, get, list, search, and retrieve evidence,
provenance and the profile document. A logical profile (investigation + spec) is registered once."""

from typing import Any

from experionyx.artifacts import ArtifactStore
from experionyx.domain import Artifact
from experionyx.errors import DuplicateError
from experionyx.faults.report import read_artifact
from experionyx.registry import Registry
from experionyx.reliability.compare import compare_profiles
from experionyx.reliability.entities import ReliabilityProfile, ReliabilityReference


class ReliabilityProfileRegistry:
    def __init__(self, registry: Registry, store: ArtifactStore | None = None) -> None:
        self.reg, self.store = registry, store

    def register(
        self, profile: ReliabilityProfile, references: list[ReliabilityReference] | None = None
    ) -> ReliabilityProfile:
        if self.reg.exists(ReliabilityProfile, profile.id):
            raise DuplicateError(
                f"profile {profile.id} is already registered (same investigation and spec)"
            )
        with self.reg.transaction():
            self.reg.add(profile)
            for r in references or []:
                self.reg.add(r)
        return profile

    def get(self, profile_id: str) -> ReliabilityProfile:
        return self.reg.get(ReliabilityProfile, profile_id)

    def find(self, **columns: str) -> list[ReliabilityProfile]:
        return self.reg.find(ReliabilityProfile, **columns)

    def search(
        self,
        *,
        model: str | None = None,
        dataset: str | None = None,
        scope: str | None = None,
        split: str | None = None,
        evaluation: str | None = None,
        investigation: str | None = None,
        ref: str | None = None,
        dimension_status: tuple[str, str] | None = None,
    ) -> list[ReliabilityProfile]:
        cols = {
            k: v
            for k, v in (
                ("model_fingerprint", model),
                ("dataset_fingerprint", dataset),
                ("scope", scope),
                ("split", split),
                ("evaluation_config_hash", evaluation),
                ("investigation_id", investigation),
            )
            if v
        }
        found = self.reg.find(ReliabilityProfile, **cols)
        if ref:  # profiles that reference a given run / fault experiment / mode / interaction
            linked = {r.profile_id for r in self.reg.find(ReliabilityReference, ref_id=ref)}
            found = [p for p in found if p.id in linked]
        if dimension_status:
            dim, st = dimension_status
            found = [p for p in found if p.dimension_status.get(dim) == st]
        return sorted(found, key=lambda p: p.id)

    def references(
        self, profile_id: str, *, dimension: str | None = None
    ) -> list[ReliabilityReference]:
        extra = {"dimension": dimension} if dimension else {}
        return self.reg.find(ReliabilityReference, profile_id=profile_id, **extra)

    def artifacts(self, profile_id: str) -> list[Artifact]:
        return [
            a
            for a in self.reg.find(Artifact, run_id=self.get(profile_id).run_id)
            if a.path.startswith("reliability/")
        ]

    def document(self, profile_id: str) -> dict[str, Any]:
        """The digest-verified profile document (every observation with its source reference)."""
        if self.store is None:
            raise ValueError("an artifact store is needed to read the profile document")
        doc = read_artifact(
            self.reg, self.store, self.get(profile_id).run_id, "reliability/profile.json"
        )
        assert isinstance(doc, dict)  # noqa: S101
        return doc

    def provenance(self, profile_id: str) -> dict[str, object]:
        p = self.get(profile_id)
        return {
            "profile_id": p.id,
            "run_id": p.run_id,
            "spec_id": p.spec_id,
            "provenance_fingerprint": p.provenance_fingerprint,
            "model_fingerprint": p.model_fingerprint,
            "dataset_fingerprint": p.dataset_fingerprint,
            "spec": p.spec,
        }

    def compare(self, a_id: str, b_id: str) -> dict[str, Any]:
        return compare_profiles(self.document(a_id), self.document(b_id))

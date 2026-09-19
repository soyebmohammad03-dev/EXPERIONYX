"""Registry entities of reliability profiles: the profile (structured metadata only) and its
evidence references. Observations themselves live in the digest-verified `reliability/profile.json`
artifact, never in a database blob. Both are immutable and content-addressed."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Self

import experionyx.validation as v
from experionyx.domain import Entity, Investigation, Run, open_payload
from experionyx.errors import ValidationError
from experionyx.reliability.taxonomy import Dimension, DimensionStatus, RefKind, Scope


@dataclass(frozen=True)
class ReliabilityProfile(Entity):
    KIND: ClassVar[str] = "reliability_profile"
    PREFIX: ClassVar[str] = "rpf"
    investigation_id: str
    run_id: str  # the Run that built (and can replay) the profile
    spec_id: str
    spec: Mapping[str, object]
    scope: Scope
    model_fingerprint: str
    dataset_fingerprint: str
    split: str  # "*" when the profile is not split-specific
    evaluation_config_hash: str
    provenance_fingerprint: str
    dimension_status: Mapping[str, object]  # dimension -> DimensionStatus (no score)
    summary: Mapping[str, object]
    created_at: datetime

    def __post_init__(self) -> None:
        v.ref("investigation_id", self.investigation_id, Investigation.PREFIX)
        v.ref("run_id", self.run_id, Run.PREFIX)
        v.text("spec_id", self.spec_id)
        object.__setattr__(self, "spec", v.freeze_mapping("spec", self.spec))
        v.member("scope", self.scope, Scope)
        for n in (
            "model_fingerprint",
            "dataset_fingerprint",
            "evaluation_config_hash",
            "provenance_fingerprint",
        ):
            v.digest(n, getattr(self, n))
        v.text("split", self.split)
        object.__setattr__(
            self, "dimension_status", v.freeze_mapping("dimension_status", self.dimension_status)
        )
        for dim, st in self.dimension_status.items():
            Dimension(dim)
            if st not in {s.value for s in DimensionStatus}:
                raise ValidationError(f"invalid dimension status {st!r}")
        object.__setattr__(self, "summary", v.freeze_mapping("summary", self.summary))
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {"investigation_id": self.investigation_id, "spec_id": self.spec_id}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        names = (
            "investigation_id",
            "run_id",
            "spec_id",
            "spec",
            "scope",
            "model_fingerprint",
            "dataset_fingerprint",
            "split",
            "evaluation_config_hash",
            "provenance_fingerprint",
            "dimension_status",
            "summary",
            "created_at",
        )
        open_payload(d, cls.KIND, names)
        return cls(
            v.get_str(d, "investigation_id"),
            v.get_str(d, "run_id"),
            v.get_str(d, "spec_id"),
            v.get_mapping(d, "spec"),
            v.get_enum(d, "scope", Scope),
            v.get_str(d, "model_fingerprint"),
            v.get_str(d, "dataset_fingerprint"),
            v.get_str(d, "split"),
            v.get_str(d, "evaluation_config_hash"),
            v.get_str(d, "provenance_fingerprint"),
            v.get_mapping(d, "dimension_status"),
            v.get_mapping(d, "summary"),
            v.get_time(d, "created_at"),
        )


@dataclass(frozen=True)
class ReliabilityReference(Entity):
    """Where an observation of a dimension came from (a run, artifact, fault experiment,
    failure mode or interaction)."""

    KIND: ClassVar[str] = "reliability_reference"
    PREFIX: ClassVar[str] = "rrf"
    profile_id: str
    dimension: Dimension
    ref_kind: RefKind
    ref_id: str
    note: str
    created_at: datetime

    def __post_init__(self) -> None:
        v.ref("profile_id", self.profile_id, ReliabilityProfile.PREFIX)
        v.member("dimension", self.dimension, Dimension)
        v.member("ref_kind", self.ref_kind, RefKind)
        v.text("ref_id", self.ref_id)
        v.text("note", self.note)
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {
            "profile_id": self.profile_id,
            "dimension": self.dimension,
            "ref_kind": self.ref_kind,
            "ref_id": self.ref_id,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d, cls.KIND, ("profile_id", "dimension", "ref_kind", "ref_id", "note", "created_at")
        )
        return cls(
            v.get_str(d, "profile_id"),
            v.get_enum(d, "dimension", Dimension),
            v.get_enum(d, "ref_kind", RefKind),
            v.get_str(d, "ref_id"),
            v.get_str(d, "note"),
            v.get_time(d, "created_at"),
        )

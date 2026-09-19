"""Immutable, validated domain objects (see docs/domain-model.md).

Entities validate in `__post_init__`, expose deterministic `to_dict`/`from_dict`, and derive a
content-addressed `id` from their *identity fields* (see docs/identity.md). Lifecycle `status` and
`created_at` are deliberately not part of identity.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import datetime
from enum import Enum, StrEnum
from typing import ClassVar, Self

import experionyx.validation as v
from experionyx.errors import SchemaVersionError, ValidationError
from experionyx.hashing import HASH_PREFIX, content_hash

PAYLOAD_SCHEMA_VERSION = 1
_ID_HEX_LENGTH = 32


class ClaimStatus(StrEnum):
    """Outcome of evaluating a claim against evidence. Negative results are first-class."""

    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    CONFOUNDED = "CONFOUNDED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class EpistemicKind(StrEnum):
    """What kind of statement a piece of information is; never conflate these."""

    OBSERVATION = "OBSERVATION"
    DERIVED_METRIC = "DERIVED_METRIC"
    INTERPRETATION = "INTERPRETATION"
    HYPOTHESIS = "HYPOTHESIS"


class ExperimentStatus(StrEnum):
    DRAFT = "DRAFT"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class RunStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ArtifactCategory(StrEnum):
    OUTPUT = "OUTPUT"
    DATA = "DATA"
    LOG = "LOG"
    DIAGNOSTIC = "DIAGNOSTIC"


class EvidenceTarget(StrEnum):
    """What kind of record a piece of evidence points at."""

    EXPERIMENT = "EXPERIMENT"
    RUN = "RUN"
    OBSERVATION = "OBSERVATION"
    ARTIFACT = "ARTIFACT"
    STATISTICAL_ANALYSIS = "STATISTICAL_ANALYSIS"


class EvidenceRelation(StrEnum):
    SUPPORTS = "SUPPORTS"
    REFUTES = "REFUTES"
    CONTEXT = "CONTEXT"


_S = ExperimentStatus
_EXPERIMENT_TRANSITIONS: Mapping[ExperimentStatus, frozenset[ExperimentStatus]] = {
    _S.DRAFT: frozenset({_S.READY, _S.CANCELLED}),
    _S.READY: frozenset({_S.RUNNING, _S.CANCELLED}),
    _S.RUNNING: frozenset({_S.COMPLETED, _S.FAILED, _S.CANCELLED}),
    _S.COMPLETED: frozenset(),
    _S.FAILED: frozenset(),
    _S.CANCELLED: frozenset(),
}
EXPERIMENT_TRANSITIONS = _EXPERIMENT_TRANSITIONS  # reused by other lifecycle-bearing entities
_R = RunStatus
_RUN_TRANSITIONS: Mapping[RunStatus, frozenset[RunStatus]] = {
    _R.PENDING: frozenset({_R.RUNNING, _R.FAILED}),
    _R.RUNNING: frozenset({_R.COMPLETED, _R.FAILED}),
    _R.COMPLETED: frozenset(),
    _R.FAILED: frozenset(),
}


# --- serialization plumbing -------------------------------------------------------------------


def to_jsonable(value: object) -> object:
    """Convert a domain value into plain JSON types."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {k: to_jsonable(x) for k, x in value.items()}
    if isinstance(value, tuple | list):
        return [to_jsonable(x) for x in value]
    return value


def check_keys(
    d: Mapping[str, object], names: tuple[str, ...], extra: tuple[str, ...] = ()
) -> None:
    expected = set(names) | set(extra)
    if set(d) != expected:
        raise ValidationError(
            f"unexpected fields {sorted(set(d) - expected)}, missing {sorted(expected - set(d))}"
        )


def open_payload(d: Mapping[str, object], kind: str, names: tuple[str, ...]) -> None:
    """Check the envelope (`kind`, `schema_version`) and the exact field set."""
    if d.get("kind") != kind:
        raise ValidationError(f"expected kind {kind!r}, got {d.get('kind')!r}")
    if d.get("schema_version") != PAYLOAD_SCHEMA_VERSION:
        raise SchemaVersionError(
            f"unsupported payload schema_version {d.get('schema_version')!r} "
            f"(supported: {PAYLOAD_SCHEMA_VERSION})"
        )
    check_keys(d, names, ("kind", "schema_version"))


class Entity:
    """Base for registry entities. Subclasses are frozen dataclasses."""

    KIND: ClassVar[str]
    PREFIX: ClassVar[str]

    def _identity(self) -> Mapping[str, object]:
        raise NotImplementedError

    @property
    def id(self) -> str:
        """Deterministic ID: `<prefix>_<first 32 hex of sha256(kind + identity fields)>`."""
        h = content_hash({"kind": self.KIND, "identity": to_jsonable(self._identity())})
        return f"{self.PREFIX}_{h[len(HASH_PREFIX) :][:_ID_HEX_LENGTH]}"

    def to_dict(self) -> dict[str, object]:
        body = to_jsonable(self)
        if not isinstance(body, dict):
            raise TypeError(f"{type(self).__name__} is not a dataclass")
        return {"kind": self.KIND, "schema_version": PAYLOAD_SCHEMA_VERSION, **body}

    def content_hash(self) -> str:
        """Hash of the *entire* serialized record (all fields, including status/timestamps)."""
        return content_hash(self.to_dict())

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        raise NotImplementedError


# --- value objects ----------------------------------------------------------------------------


@dataclass(frozen=True)
class _VersionedRef:
    name: str
    version: str
    digest: str | None = None

    def __post_init__(self) -> None:
        v.text("name", self.name)
        v.version("version", self.version)
        if self.digest is not None:
            v.digest("digest", self.digest)

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        check_keys(d, ("name", "version", "digest"))
        return cls(v.get_str(d, "name"), v.get_str(d, "version"), v.get_opt_str(d, "digest"))


class ModelRef(_VersionedRef):
    """Reference to a versioned model under investigation. Embedded in Experiment."""


class DatasetRef(_VersionedRef):
    """Reference to a versioned dataset. Embedded in Experiment."""


# --- entities ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Investigation(Entity):
    KIND: ClassVar[str] = "investigation"
    PREFIX: ClassVar[str] = "inv"
    name: str
    question: str
    created_at: datetime

    def __post_init__(self) -> None:
        v.text("name", self.name)
        v.text("question", self.question)
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {"name": self.name, "question": self.question}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(d, cls.KIND, ("name", "question", "created_at"))
        return cls(v.get_str(d, "name"), v.get_str(d, "question"), v.get_time(d, "created_at"))


@dataclass(frozen=True)
class ConfigurationRef(Entity):
    """Content-addressed configuration: identical parameters always yield the same ID."""

    KIND: ClassVar[str] = "configuration"
    PREFIX: ClassVar[str] = "cfg"
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", v.freeze_mapping("parameters", self.parameters))

    def _identity(self) -> Mapping[str, object]:
        return {"parameters": self.parameters}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(d, cls.KIND, ("parameters",))
        return cls(v.get_mapping(d, "parameters"))


@dataclass(frozen=True)
class EnvironmentSnapshot(Entity):
    """Content-addressed description of the software/hardware environment of a run."""

    KIND: ClassVar[str] = "environment"
    PREFIX: ClassVar[str] = "env"
    python_version: str
    os: str
    machine: str
    packages: Mapping[str, object]
    source_revision: str | None = None

    def __post_init__(self) -> None:
        v.version("python_version", self.python_version)
        v.text("os", self.os)
        v.text("machine", self.machine)
        packages = v.freeze_mapping("packages", self.packages)
        for name, ver in packages.items():
            v.version(f"packages.{name}", ver)
        object.__setattr__(self, "packages", packages)
        if self.source_revision is not None:
            v.text("source_revision", self.source_revision)

    def _identity(self) -> Mapping[str, object]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d, cls.KIND, ("python_version", "os", "machine", "packages", "source_revision")
        )
        return cls(
            v.get_str(d, "python_version"),
            v.get_str(d, "os"),
            v.get_str(d, "machine"),
            v.get_mapping(d, "packages"),
            v.get_opt_str(d, "source_revision"),
        )


@dataclass(frozen=True)
class Experiment(Entity):
    KIND: ClassVar[str] = "experiment"
    PREFIX: ClassVar[str] = "exp"
    investigation_id: str
    name: str
    hypothesis: str
    model: ModelRef
    dataset: DatasetRef
    configuration_id: str
    created_at: datetime
    status: ExperimentStatus = ExperimentStatus.DRAFT

    def __post_init__(self) -> None:
        v.ref("investigation_id", self.investigation_id, Investigation.PREFIX)
        v.text("name", self.name)
        v.text("hypothesis", self.hypothesis)
        v.member("model", self.model, ModelRef)
        v.member("dataset", self.dataset, DatasetRef)
        v.ref("configuration_id", self.configuration_id, ConfigurationRef.PREFIX)
        v.timestamp("created_at", self.created_at)
        v.member("status", self.status, ExperimentStatus)

    def _identity(self) -> Mapping[str, object]:
        return {
            "investigation_id": self.investigation_id,
            "name": self.name,
            "hypothesis": self.hypothesis,
            "model": self.model,
            "dataset": self.dataset,
            "configuration_id": self.configuration_id,
        }

    def with_status(self, status: ExperimentStatus) -> Self:
        if status not in _EXPERIMENT_TRANSITIONS[self.status]:
            raise ValidationError(f"illegal experiment transition {self.status} -> {status}")
        return replace(self, status=status)

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d,
            cls.KIND,
            (
                "investigation_id",
                "name",
                "hypothesis",
                "model",
                "dataset",
                "configuration_id",
                "created_at",
                "status",
            ),
        )
        return cls(
            investigation_id=v.get_str(d, "investigation_id"),
            name=v.get_str(d, "name"),
            hypothesis=v.get_str(d, "hypothesis"),
            model=ModelRef.from_dict(v.get_mapping(d, "model")),
            dataset=DatasetRef.from_dict(v.get_mapping(d, "dataset")),
            configuration_id=v.get_str(d, "configuration_id"),
            created_at=v.get_time(d, "created_at"),
            status=v.get_enum(d, "status", ExperimentStatus),
        )


@dataclass(frozen=True)
class Run(Entity):
    """One execution of an experiment under an environment and seed. `attempt` distinguishes
    deliberate repeats with identical experiment/environment/seed."""

    KIND: ClassVar[str] = "run"
    PREFIX: ClassVar[str] = "run"
    experiment_id: str
    environment_id: str
    seed: int
    created_at: datetime
    attempt: int = 0
    status: RunStatus = RunStatus.PENDING

    def __post_init__(self) -> None:
        v.ref("experiment_id", self.experiment_id, Experiment.PREFIX)
        v.ref("environment_id", self.environment_id, EnvironmentSnapshot.PREFIX)
        v.non_negative_int("seed", self.seed)
        v.timestamp("created_at", self.created_at)
        v.non_negative_int("attempt", self.attempt)
        v.member("status", self.status, RunStatus)

    def _identity(self) -> Mapping[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "environment_id": self.environment_id,
            "seed": self.seed,
            "attempt": self.attempt,
        }

    def with_status(self, status: RunStatus) -> Self:
        if status not in _RUN_TRANSITIONS[self.status]:
            raise ValidationError(f"illegal run transition {self.status} -> {status}")
        return replace(self, status=status)

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d,
            cls.KIND,
            ("experiment_id", "environment_id", "seed", "created_at", "attempt", "status"),
        )
        return cls(
            experiment_id=v.get_str(d, "experiment_id"),
            environment_id=v.get_str(d, "environment_id"),
            seed=v.get_int(d, "seed"),
            created_at=v.get_time(d, "created_at"),
            attempt=v.get_int(d, "attempt"),
            status=v.get_enum(d, "status", RunStatus),
        )


# Scalars or JSON-structured values (sequences become tuples, objects read-only mappings).
ObservationValue = bool | int | float | str | tuple[object, ...] | Mapping[str, object]


@dataclass(frozen=True)
class Observation(Entity):
    """A measured value from a run. Identity is (run, name, sequence): the value is content, not
    identity, so a different value cannot silently replace an existing observation."""

    KIND: ClassVar[str] = "observation"
    PREFIX: ClassVar[str] = "obs"
    run_id: str
    name: str
    value: ObservationValue
    created_at: datetime
    sequence: int = 0
    unit: str | None = None
    epistemic_kind: EpistemicKind = EpistemicKind.OBSERVATION

    def __post_init__(self) -> None:
        v.ref("run_id", self.run_id, Run.PREFIX)
        v.text("name", self.name)
        if self.value is None:
            raise ValidationError("value must not be None")
        object.__setattr__(self, "value", v.freeze("value", self.value))  # JSON-only, finite
        v.timestamp("created_at", self.created_at)
        v.non_negative_int("sequence", self.sequence)
        if self.unit is not None:
            v.text("unit", self.unit)
        if v.member("epistemic_kind", self.epistemic_kind, EpistemicKind) not in _OBSERVATION_KINDS:
            raise ValidationError(f"observation kind must be one of {sorted(_OBSERVATION_KINDS)}")

    def _identity(self) -> Mapping[str, object]:
        return {"run_id": self.run_id, "name": self.name, "sequence": self.sequence}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d,
            cls.KIND,
            ("run_id", "name", "value", "created_at", "sequence", "unit", "epistemic_kind"),
        )
        return cls(
            run_id=v.get_str(d, "run_id"),
            name=v.get_str(d, "name"),
            value=v.get_raw(d, "value"),  # type: ignore[arg-type]  # validated in __post_init__
            created_at=v.get_time(d, "created_at"),
            sequence=v.get_int(d, "sequence"),
            unit=v.get_opt_str(d, "unit"),
            epistemic_kind=v.get_enum(d, "epistemic_kind", EpistemicKind),
        )


_OBSERVATION_KINDS = frozenset({EpistemicKind.OBSERVATION, EpistemicKind.DERIVED_METRIC})


@dataclass(frozen=True)
class Artifact(Entity):
    """Metadata for a file produced by a run. The file itself is not stored in the registry."""

    KIND: ClassVar[str] = "artifact"
    PREFIX: ClassVar[str] = "art"
    run_id: str
    name: str
    path: str  # relative to the run's artifact directory
    digest: str
    size_bytes: int
    media_type: str
    created_at: datetime
    category: ArtifactCategory = ArtifactCategory.OUTPUT

    def __post_init__(self) -> None:
        v.ref("run_id", self.run_id, Run.PREFIX)
        v.text("name", self.name)
        v.member("category", self.category, ArtifactCategory)
        v.relative_path("path", self.path)
        v.digest("digest", self.digest)
        v.non_negative_int("size_bytes", self.size_bytes)
        if "/" not in v.text("media_type", self.media_type):
            raise ValidationError(f"media_type must look like 'type/subtype': {self.media_type!r}")
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {"run_id": self.run_id, "path": self.path, "digest": self.digest}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d,
            cls.KIND,
            (
                "run_id",
                "name",
                "path",
                "digest",
                "size_bytes",
                "media_type",
                "created_at",
                "category",
            ),
        )
        return cls(
            run_id=v.get_str(d, "run_id"),
            name=v.get_str(d, "name"),
            category=v.get_enum(d, "category", ArtifactCategory),
            path=v.get_str(d, "path"),
            digest=v.get_str(d, "digest"),
            size_bytes=v.get_int(d, "size_bytes"),
            media_type=v.get_str(d, "media_type"),
            created_at=v.get_time(d, "created_at"),
        )


@dataclass(frozen=True)
class Claim(Entity):
    """A statement made within an investigation. `status` is a domain state only; nothing in
    Phase 1 evaluates it against evidence."""

    KIND: ClassVar[str] = "claim"
    PREFIX: ClassVar[str] = "clm"
    investigation_id: str
    statement: str
    asserted_by: str
    created_at: datetime
    status: ClaimStatus = ClaimStatus.INSUFFICIENT_EVIDENCE

    def __post_init__(self) -> None:
        v.ref("investigation_id", self.investigation_id, Investigation.PREFIX)
        v.text("statement", self.statement)
        v.text("asserted_by", self.asserted_by)
        v.timestamp("created_at", self.created_at)
        v.member("status", self.status, ClaimStatus)

    def _identity(self) -> Mapping[str, object]:
        return {"investigation_id": self.investigation_id, "statement": self.statement}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d,
            cls.KIND,
            ("investigation_id", "statement", "asserted_by", "created_at", "status"),
        )
        return cls(
            investigation_id=v.get_str(d, "investigation_id"),
            statement=v.get_str(d, "statement"),
            asserted_by=v.get_str(d, "asserted_by"),
            created_at=v.get_time(d, "created_at"),
            status=v.get_enum(d, "status", ClaimStatus),
        )


@dataclass(frozen=True)
class Evidence(Entity):
    """Links a claim to the specific record that supports, refutes or contextualizes it."""

    KIND: ClassVar[str] = "evidence"
    PREFIX: ClassVar[str] = "evd"
    claim_id: str
    target_kind: EvidenceTarget
    target_id: str
    relation: EvidenceRelation
    created_at: datetime
    note: str | None = None

    def __post_init__(self) -> None:
        v.ref("claim_id", self.claim_id, Claim.PREFIX)
        kind = v.member("target_kind", self.target_kind, EvidenceTarget)
        v.ref("target_id", self.target_id, evidence_target_type(kind).PREFIX)
        v.member("relation", self.relation, EvidenceRelation)
        v.timestamp("created_at", self.created_at)
        if self.note is not None:
            v.text("note", self.note)

    def _identity(self) -> Mapping[str, object]:
        return {
            "claim_id": self.claim_id,
            "target_kind": self.target_kind,
            "target_id": self.target_id,
            "relation": self.relation,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d,
            cls.KIND,
            ("claim_id", "target_kind", "target_id", "relation", "created_at", "note"),
        )
        return cls(
            claim_id=v.get_str(d, "claim_id"),
            target_kind=v.get_enum(d, "target_kind", EvidenceTarget),
            target_id=v.get_str(d, "target_id"),
            relation=v.get_enum(d, "relation", EvidenceRelation),
            created_at=v.get_time(d, "created_at"),
            note=v.get_opt_str(d, "note"),
        )


EVIDENCE_TARGET_TYPES: Mapping[EvidenceTarget, type[Entity]] = {
    EvidenceTarget.EXPERIMENT: Experiment,
    EvidenceTarget.RUN: Run,
    EvidenceTarget.OBSERVATION: Observation,
    EvidenceTarget.ARTIFACT: Artifact,
}


def evidence_target_type(kind: EvidenceTarget) -> type[Entity]:
    if kind is EvidenceTarget.STATISTICAL_ANALYSIS:  # lazy: the stats package imports this module
        from experionyx.stats.entities import StatisticalAnalysis

        return StatisticalAnalysis
    return EVIDENCE_TARGET_TYPES[kind]

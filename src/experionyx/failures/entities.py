"""Registry entities of failure discovery. Four distinct concepts are kept apart on purpose:
FAILURE_SIGNAL (one normalized observation from one run), FAILURE_CLUSTER (signals grouped by a
deterministic rule), and FAILURE_MODE (a cluster that has been given a lifecycle status; CANDIDATE
until evidence criteria are met, REGISTERED only through explicit transitions), plus the evidence
and relationships that back a mode. Structured fields are authoritative; descriptions explain."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from typing import ClassVar, Self

import experionyx.validation as v
from experionyx.domain import Entity, Experiment, Investigation, Run, open_payload
from experionyx.errors import ValidationError
from experionyx.failures.taxonomy import (
    ALLOWED_RELATIONSHIPS,
    FAILURE_TRANSITIONS,
    Direction,
    EvidenceKind,
    FailureCategory,
    FailureStatus,
    NodeKind,
    Predicate,
    SignalKind,
)


def _strs(field: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple) or not all(isinstance(x, str) for x in value):
        raise ValidationError(f"{field} must be a list of strings")
    return tuple(value)


@dataclass(frozen=True)
class FailureSignal(Entity):
    """One normalized failure observation extracted from one Run's stored results. It keeps its
    lineage (run, kind, structured detail, the affected sample IDs) and never a bare score."""

    KIND: ClassVar[str] = "failure_signal"
    PREFIX: ClassVar[str] = "fsg"
    run_id: str
    experiment_id: str
    signal_kind: SignalKind
    category: FailureCategory
    signature: str  # deterministic key of the normalized failure (see failures.signature)
    direction: Direction
    magnitude: float  # >= 0; the size of the observed effect, in the unit stated in `detail`
    detail: Mapping[str, object]  # class_label, slice, metric, fault family/params, seed, ...
    sample_ids: tuple[str, ...]  # affected samples, bounded (see sample_count / truncated)
    sample_count: int  # true number of affected samples (may exceed len(sample_ids))
    extractor_version: str
    config_hash: str
    created_at: datetime

    def __post_init__(self) -> None:
        v.ref("run_id", self.run_id, Run.PREFIX)
        v.ref("experiment_id", self.experiment_id, Experiment.PREFIX)
        v.member("signal_kind", self.signal_kind, SignalKind)
        v.member("category", self.category, FailureCategory)
        v.text("signature", self.signature)
        v.member("direction", self.direction, Direction)
        if v.finite_float("magnitude", self.magnitude) < 0:
            raise ValidationError("magnitude must be >= 0")
        object.__setattr__(self, "detail", v.freeze_mapping("detail", self.detail))
        object.__setattr__(self, "sample_ids", _strs("sample_ids", self.sample_ids))
        v.non_negative_int("sample_count", self.sample_count)
        if self.sample_count < len(self.sample_ids):
            raise ValidationError("sample_count is smaller than the stored sample IDs")
        v.text("extractor_version", self.extractor_version)
        v.digest("config_hash", self.config_hash)
        v.timestamp("created_at", self.created_at)

    @property
    def truncated(self) -> bool:
        return self.sample_count > len(self.sample_ids)

    def _identity(self) -> Mapping[str, object]:
        return {
            "run_id": self.run_id,
            "signal_kind": self.signal_kind,
            "signature": self.signature,
            "extractor_version": self.extractor_version,
            "config_hash": self.config_hash,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d, cls.KIND,
            ("run_id", "experiment_id", "signal_kind", "category", "signature", "direction", "magnitude", "detail", "sample_ids", "sample_count", "extractor_version", "config_hash", "created_at"),
        )  # fmt: skip
        return cls(
            run_id=v.get_str(d, "run_id"),
            experiment_id=v.get_str(d, "experiment_id"),
            signal_kind=v.get_enum(d, "signal_kind", SignalKind),
            category=v.get_enum(d, "category", FailureCategory),
            signature=v.get_str(d, "signature"),
            direction=v.get_enum(d, "direction", Direction),
            magnitude=v.finite_float("magnitude", v.get_raw(d, "magnitude")),
            detail=v.get_mapping(d, "detail"),
            sample_ids=_strs("sample_ids", v.get_raw(d, "sample_ids")),
            sample_count=v.get_int(d, "sample_count"),
            extractor_version=v.get_str(d, "extractor_version"),
            config_hash=v.get_str(d, "config_hash"),
            created_at=v.get_time(d, "created_at"),
        )


@dataclass(frozen=True)
class FailureCluster(Entity):
    """Signals grouped by a named, versioned, deterministic algorithm. A cluster is a statistical
    grouping only: it is not a failure mode until it is registered as one."""

    KIND: ClassVar[str] = "failure_cluster"
    PREFIX: ClassVar[str] = "fcl"
    investigation_id: str
    signal_ids: tuple[str, ...]  # sorted
    algorithm: str
    algorithm_version: str
    config_hash: str
    threshold: float
    metrics: Mapping[
        str, object
    ]  # size, compactness, stability, prevalence, ... (with denominators)
    created_at: datetime

    def __post_init__(self) -> None:
        v.ref("investigation_id", self.investigation_id, Investigation.PREFIX)
        ids = _strs("signal_ids", self.signal_ids)
        if not ids or list(ids) != sorted(set(ids)):
            raise ValidationError("signal_ids must be non-empty, unique and sorted")
        for s in ids:
            v.ref("signal_ids[]", s, FailureSignal.PREFIX)
        object.__setattr__(self, "signal_ids", ids)
        v.text("algorithm", self.algorithm)
        v.text("algorithm_version", self.algorithm_version)
        v.digest("config_hash", self.config_hash)
        v.finite_float("threshold", self.threshold)
        object.__setattr__(self, "metrics", v.freeze_mapping("metrics", self.metrics))
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {
            "investigation_id": self.investigation_id,
            "signal_ids": self.signal_ids,
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
            "config_hash": self.config_hash,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d, cls.KIND,
            ("investigation_id", "signal_ids", "algorithm", "algorithm_version", "config_hash", "threshold", "metrics", "created_at"),
        )  # fmt: skip
        return cls(
            investigation_id=v.get_str(d, "investigation_id"),
            signal_ids=_strs("signal_ids", v.get_raw(d, "signal_ids")),
            algorithm=v.get_str(d, "algorithm"),
            algorithm_version=v.get_str(d, "algorithm_version"),
            config_hash=v.get_str(d, "config_hash"),
            threshold=v.finite_float("threshold", v.get_raw(d, "threshold")),
            metrics=v.get_mapping(d, "metrics"),
            created_at=v.get_time(d, "created_at"),
        )


@dataclass(frozen=True)
class FailureMode(Entity):
    """A failure mode record. `structured` (classes, slices, fault families, models, datasets,
    direction, effect statistics, prevalence/impact with denominators, multidimensional severity)
    is authoritative; `title` and `description` only explain it. Status moves only through
    validated transitions and never automatically to CONFIRMED."""

    KIND: ClassVar[str] = "failure_mode"
    PREFIX: ClassVar[str] = "fmd"
    investigation_id: str
    cluster_id: str
    category: FailureCategory
    title: str
    description: str
    structured: Mapping[str, object]
    created_at: datetime
    status: FailureStatus = FailureStatus.DISCOVERED

    def __post_init__(self) -> None:
        v.ref("investigation_id", self.investigation_id, Investigation.PREFIX)
        v.ref("cluster_id", self.cluster_id, FailureCluster.PREFIX)
        v.member("category", self.category, FailureCategory)
        v.text("title", self.title)
        v.text("description", self.description)
        object.__setattr__(self, "structured", v.freeze_mapping("structured", self.structured))
        v.timestamp("created_at", self.created_at)
        v.member("status", self.status, FailureStatus)

    def _identity(self) -> Mapping[str, object]:
        return {"investigation_id": self.investigation_id, "cluster_id": self.cluster_id}

    def with_status(self, status: FailureStatus) -> Self:
        if status not in FAILURE_TRANSITIONS[self.status]:
            raise ValidationError(f"illegal failure-mode transition {self.status} -> {status}")
        return replace(self, status=status)

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d, cls.KIND,
            ("investigation_id", "cluster_id", "category", "title", "description", "structured", "created_at", "status"),
        )  # fmt: skip
        return cls(
            investigation_id=v.get_str(d, "investigation_id"),
            cluster_id=v.get_str(d, "cluster_id"),
            category=v.get_enum(d, "category", FailureCategory),
            title=v.get_str(d, "title"),
            description=v.get_str(d, "description"),
            structured=v.get_mapping(d, "structured"),
            created_at=v.get_time(d, "created_at"),
            status=v.get_enum(d, "status", FailureStatus),
        )


@dataclass(frozen=True)
class FailureEvidence(Entity):
    """One retained piece of evidence for (or a recorded event about) a failure mode. Append-only:
    it survives every later status change, including rejection and deprecation."""

    KIND: ClassVar[str] = "failure_evidence"
    PREFIX: ClassVar[str] = "fev"
    failure_mode_id: str
    evidence_kind: EvidenceKind
    ref_id: str | None  # the signal / run / claim this evidence points at, if any
    summary: str
    detail: Mapping[str, object]
    created_at: datetime

    def __post_init__(self) -> None:
        v.ref("failure_mode_id", self.failure_mode_id, FailureMode.PREFIX)
        v.member("evidence_kind", self.evidence_kind, EvidenceKind)
        if self.ref_id is not None:
            v.text("ref_id", self.ref_id)
        v.text("summary", self.summary)
        object.__setattr__(self, "detail", v.freeze_mapping("detail", self.detail))
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {
            "failure_mode_id": self.failure_mode_id,
            "evidence_kind": self.evidence_kind,
            "ref_id": self.ref_id,
            "summary": self.summary,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d,
            cls.KIND,
            ("failure_mode_id", "evidence_kind", "ref_id", "summary", "detail", "created_at"),
        )
        return cls(
            failure_mode_id=v.get_str(d, "failure_mode_id"),
            evidence_kind=v.get_enum(d, "evidence_kind", EvidenceKind),
            ref_id=v.get_opt_str(d, "ref_id"),
            summary=v.get_str(d, "summary"),
            detail=v.get_mapping(d, "detail"),
            created_at=v.get_time(d, "created_at"),
        )


@dataclass(frozen=True)
class FailureRelationship(Entity):
    """A typed edge of the (backend-only) failure knowledge graph. Only combinations listed in
    ALLOWED_RELATIONSHIPS are valid. CO_OCCURS_WITH records observed co-occurrence, which is not
    an interaction claim; INTERACTS_WITH is reserved for an explicitly designed analysis."""

    KIND: ClassVar[str] = "failure_relationship"
    PREFIX: ClassVar[str] = "frl"
    investigation_id: str
    subject_kind: NodeKind
    subject_id: str
    predicate: Predicate
    object_kind: NodeKind
    object_id: str
    detail: Mapping[str, object]
    created_at: datetime

    def __post_init__(self) -> None:
        v.ref("investigation_id", self.investigation_id, Investigation.PREFIX)
        v.member("subject_kind", self.subject_kind, NodeKind)
        v.member("predicate", self.predicate, Predicate)
        v.member("object_kind", self.object_kind, NodeKind)
        v.text("subject_id", self.subject_id)
        v.text("object_id", self.object_id)
        if (self.subject_kind, self.predicate, self.object_kind) not in ALLOWED_RELATIONSHIPS:
            raise ValidationError(
                f"relationship {self.subject_kind} {self.predicate} {self.object_kind} is not allowed"
            )
        object.__setattr__(self, "detail", v.freeze_mapping("detail", self.detail))
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {
            "investigation_id": self.investigation_id,
            "subject_kind": self.subject_kind,
            "subject_id": self.subject_id,
            "predicate": self.predicate,
            "object_kind": self.object_kind,
            "object_id": self.object_id,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d, cls.KIND,
            ("investigation_id", "subject_kind", "subject_id", "predicate", "object_kind", "object_id", "detail", "created_at"),
        )  # fmt: skip
        return cls(
            investigation_id=v.get_str(d, "investigation_id"),
            subject_kind=v.get_enum(d, "subject_kind", NodeKind),
            subject_id=v.get_str(d, "subject_id"),
            predicate=v.get_enum(d, "predicate", Predicate),
            object_kind=v.get_enum(d, "object_kind", NodeKind),
            object_id=v.get_str(d, "object_id"),
            detail=v.get_mapping(d, "detail"),
            created_at=v.get_time(d, "created_at"),
        )

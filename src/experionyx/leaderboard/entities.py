"""Registry entities of the leaderboard layer. Immutable and content-addressed, exactly like the
benchmark engine's own `Benchmark`/`BenchmarkResult` (see docs/leaderboard.md). None of these
duplicate Phase 9 data: a `BenchmarkSubmission` REFERENCES an existing `Benchmark`/`BenchmarkResult`
rather than copying it, and a `LeaderboardEntry`'s metrics are read from that result's own stored
document, never recomputed."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Self

import experionyx.validation as v
from experionyx.adapters.records import RegisteredDataset, RegisteredModel
from experionyx.benchmark.entities import Benchmark, BenchmarkResult
from experionyx.domain import Entity, Investigation, open_payload
from experionyx.errors import ValidationError
from experionyx.leaderboard.taxonomy import ReproducibilityState


@dataclass(frozen=True)
class BenchmarkProtocol(Entity):
    """The model-independent identity of a benchmark: dataset, split, metrics, fault/stress
    conditions, slices, statistical configuration, resource policy, calibration requirements,
    evaluation configuration, seeds and software constraints -- everything
    `BenchmarkSpec.protocol_key()` already hashes into `protocol_hash`, promoted to its own
    registry row so many submissions (different models) can be grouped under it."""

    KIND: ClassVar[str] = "benchmark_protocol"
    PREFIX: ClassVar[str] = "bpr"
    name: str
    version: str
    protocol_hash: str
    dataset_record_id: str
    engine_version: str
    created_at: datetime

    def __post_init__(self) -> None:
        v.text("name", self.name)
        v.text("version", self.version)
        v.digest("protocol_hash", self.protocol_hash)
        v.ref("dataset_record_id", self.dataset_record_id, RegisteredDataset.PREFIX)
        v.text("engine_version", self.engine_version)
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {"protocol_hash": self.protocol_hash}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(d, cls.KIND, ("name", "version", "protocol_hash", "dataset_record_id", "engine_version", "created_at"))  # fmt: skip
        return cls(
            v.get_str(d, "name"), v.get_str(d, "version"), v.get_str(d, "protocol_hash"),
            v.get_str(d, "dataset_record_id"), v.get_str(d, "engine_version"), v.get_time(d, "created_at"),
        )  # fmt: skip


@dataclass(frozen=True)
class BenchmarkSubmission(Entity):
    """One model's real, provenance-linked binding to a `BenchmarkProtocol`. Identity is
    `(protocol_id, result_id)`: the same evidence submitted twice is the same submission, never
    duplicated -- the same idempotence convention as `GraphSnapshot`/`BenchmarkResult` themselves."""

    KIND: ClassVar[str] = "benchmark_submission"
    PREFIX: ClassVar[str] = "bsb"
    protocol_id: str
    investigation_id: str
    model_record_id: str
    benchmark_id: str  # the Phase 9 Benchmark (this model's own full definition)
    result_id: str  # the Phase 9 BenchmarkResult (the real evidence)
    submitted_at: datetime

    def __post_init__(self) -> None:
        v.ref("protocol_id", self.protocol_id, BenchmarkProtocol.PREFIX)
        v.ref("investigation_id", self.investigation_id, Investigation.PREFIX)
        v.ref("model_record_id", self.model_record_id, RegisteredModel.PREFIX)
        v.ref("benchmark_id", self.benchmark_id, Benchmark.PREFIX)
        v.ref("result_id", self.result_id, BenchmarkResult.PREFIX)
        v.timestamp("submitted_at", self.submitted_at)

    def _identity(self) -> Mapping[str, object]:
        return {"protocol_id": self.protocol_id, "result_id": self.result_id}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(d, cls.KIND, ("protocol_id", "investigation_id", "model_record_id", "benchmark_id", "result_id", "submitted_at"))  # fmt: skip
        return cls(
            v.get_str(d, "protocol_id"), v.get_str(d, "investigation_id"), v.get_str(d, "model_record_id"),
            v.get_str(d, "benchmark_id"), v.get_str(d, "result_id"), v.get_time(d, "submitted_at"),
        )  # fmt: skip


@dataclass(frozen=True)
class LeaderboardSnapshot(Entity):
    """One immutable snapshot of every submission to a protocol that met its filtering rules and
    evidence requirements at a point in time. Identity is `(protocol_id, source_fingerprint)`:
    rebuilding from unchanged submissions returns the same snapshot, never a duplicate -- the same
    convention as `GraphSnapshot`."""

    KIND: ClassVar[str] = "leaderboard_snapshot"
    PREFIX: ClassVar[str] = "lbs"
    protocol_id: str
    investigation_id: str
    engine_version: str
    metric_ids: tuple[str, ...]
    filtering_rules: Mapping[str, object]
    evidence_requirements: Mapping[str, object]
    submission_ids: tuple[str, ...]
    excluded: Mapping[str, object]  # submission_id -> reason it was filtered out
    source_fingerprint: str
    created_at: datetime

    def __post_init__(self) -> None:
        v.ref("protocol_id", self.protocol_id, BenchmarkProtocol.PREFIX)
        v.ref("investigation_id", self.investigation_id, Investigation.PREFIX)
        v.text("engine_version", self.engine_version)
        if not isinstance(self.metric_ids, tuple) or not all(isinstance(x, str) for x in self.metric_ids):  # fmt: skip
            raise ValidationError("metric_ids must be a tuple of strings")
        object.__setattr__(self, "filtering_rules", v.freeze_mapping("filtering_rules", self.filtering_rules))  # fmt: skip
        object.__setattr__(self, "evidence_requirements", v.freeze_mapping("evidence_requirements", self.evidence_requirements))  # fmt: skip
        if not isinstance(self.submission_ids, tuple) or not all(isinstance(x, str) for x in self.submission_ids):  # fmt: skip
            raise ValidationError("submission_ids must be a tuple of strings")
        object.__setattr__(self, "excluded", v.freeze_mapping("excluded", self.excluded))
        v.digest("source_fingerprint", self.source_fingerprint)
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {"protocol_id": self.protocol_id, "source_fingerprint": self.source_fingerprint}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d, cls.KIND,
            ("protocol_id", "investigation_id", "engine_version", "metric_ids", "filtering_rules",
             "evidence_requirements", "submission_ids", "excluded", "source_fingerprint", "created_at"),
        )  # fmt: skip
        metric_ids = v.get_raw(d, "metric_ids")
        if not isinstance(metric_ids, list):
            raise ValidationError("metric_ids must be a list")
        submission_ids = v.get_raw(d, "submission_ids")
        if not isinstance(submission_ids, list):
            raise ValidationError("submission_ids must be a list")
        return cls(
            v.get_str(d, "protocol_id"), v.get_str(d, "investigation_id"), v.get_str(d, "engine_version"),
            tuple(str(x) for x in metric_ids), v.get_mapping(d, "filtering_rules"),
            v.get_mapping(d, "evidence_requirements"), tuple(str(x) for x in submission_ids),
            v.get_mapping(d, "excluded"), v.get_str(d, "source_fingerprint"), v.get_time(d, "created_at"),
        )  # fmt: skip


@dataclass(frozen=True)
class LeaderboardEntry(Entity):
    """One submission's reported evidence within one snapshot: raw per-metric values (read from
    the underlying `BenchmarkResult`, never recomputed), resource context kept separate, and a
    reproducibility evidence state. No composite score."""

    KIND: ClassVar[str] = "leaderboard_entry"
    PREFIX: ClassVar[str] = "lbe"
    snapshot_id: str
    submission_id: str
    model_record_id: str
    metrics: Mapping[str, object]  # metric_id -> {value, status, higher_is_better, interval, n_samples}  # fmt: skip
    resource_context: Mapping[str, object]
    reproducibility_state: ReproducibilityState
    created_at: datetime

    def __post_init__(self) -> None:
        v.ref("snapshot_id", self.snapshot_id, LeaderboardSnapshot.PREFIX)
        v.ref("submission_id", self.submission_id, BenchmarkSubmission.PREFIX)
        v.ref("model_record_id", self.model_record_id, RegisteredModel.PREFIX)
        object.__setattr__(self, "metrics", v.freeze_mapping("metrics", self.metrics))
        object.__setattr__(self, "resource_context", v.freeze_mapping("resource_context", self.resource_context))  # fmt: skip
        v.member("reproducibility_state", self.reproducibility_state, ReproducibilityState)
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {"snapshot_id": self.snapshot_id, "submission_id": self.submission_id}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(d, cls.KIND, ("snapshot_id", "submission_id", "model_record_id", "metrics", "resource_context", "reproducibility_state", "created_at"))  # fmt: skip
        return cls(
            v.get_str(d, "snapshot_id"), v.get_str(d, "submission_id"), v.get_str(d, "model_record_id"),
            v.get_mapping(d, "metrics"), v.get_mapping(d, "resource_context"),
            v.get_enum(d, "reproducibility_state", ReproducibilityState), v.get_time(d, "created_at"),
        )  # fmt: skip


__all__ = ["BenchmarkProtocol", "BenchmarkSubmission", "LeaderboardEntry", "LeaderboardSnapshot"]

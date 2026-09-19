"""Registry entities of the benchmark engine: the versioned DEFINITION, each RESULT of running it,
and the explicit UNITS (experiment trials) a result expanded into. Immutable and content-addressed;
observations stay in digest-verified artifacts, only structured metadata is stored here."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Self

import experionyx.validation as v
from experionyx.adapters.records import RegisteredDataset, RegisteredModel
from experionyx.benchmark.taxonomy import CoverageStatus, UnitKind, UnitStatus
from experionyx.domain import Entity, Investigation, Run, open_payload


@dataclass(frozen=True)
class Benchmark(Entity):
    KIND: ClassVar[str] = "benchmark"
    PREFIX: ClassVar[str] = "bmk"
    name: str
    version: str
    spec_id: str  # bsp_<hash> of the full definition (includes model, dataset, faults, seeds, analysis settings)
    spec: Mapping[str, object]
    protocol_hash: (
        str  # model-independent hash of the expanded protocol (resolved fault versions included)
    )
    engine_version: str
    model_record_id: str
    dataset_record_id: str
    created_at: datetime

    def __post_init__(self) -> None:
        v.text("name", self.name)
        v.text("version", self.version)
        v.text("spec_id", self.spec_id)
        object.__setattr__(self, "spec", v.freeze_mapping("spec", self.spec))
        v.digest("protocol_hash", self.protocol_hash)
        v.text("engine_version", self.engine_version)
        v.ref("model_record_id", self.model_record_id, RegisteredModel.PREFIX)
        v.ref("dataset_record_id", self.dataset_record_id, RegisteredDataset.PREFIX)
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {"spec_id": self.spec_id}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d,
            cls.KIND,
            (
                "name",
                "version",
                "spec_id",
                "spec",
                "protocol_hash",
                "engine_version",
                "model_record_id",
                "dataset_record_id",
                "created_at",
            ),
        )
        return cls(
            v.get_str(d, "name"),
            v.get_str(d, "version"),
            v.get_str(d, "spec_id"),
            v.get_mapping(d, "spec"),
            v.get_str(d, "protocol_hash"),
            v.get_str(d, "engine_version"),
            v.get_str(d, "model_record_id"),
            v.get_str(d, "dataset_record_id"),
            v.get_time(d, "created_at"),
        )


@dataclass(frozen=True)
class BenchmarkResult(Entity):
    KIND: ClassVar[str] = "benchmark_result"
    PREFIX: ClassVar[str] = "brs"
    benchmark_id: str
    investigation_id: str
    run_id: str  # the collect Run that read the persisted evidence and wrote the artifacts
    protocol_hash: str
    provenance_fingerprint: str
    coverage_status: CoverageStatus
    section_status: Mapping[
        str, object
    ]  # section -> OBSERVED | DERIVED | UNAVAILABLE | INSUFFICIENT_EVIDENCE
    summary: Mapping[str, object]
    created_at: datetime

    def __post_init__(self) -> None:
        v.ref("benchmark_id", self.benchmark_id, Benchmark.PREFIX)
        v.ref("investigation_id", self.investigation_id, Investigation.PREFIX)
        v.ref("run_id", self.run_id, Run.PREFIX)
        v.digest("protocol_hash", self.protocol_hash)
        v.digest("provenance_fingerprint", self.provenance_fingerprint)
        v.member("coverage_status", self.coverage_status, CoverageStatus)
        object.__setattr__(
            self, "section_status", v.freeze_mapping("section_status", self.section_status)
        )
        object.__setattr__(self, "summary", v.freeze_mapping("summary", self.summary))
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {
            "benchmark_id": self.benchmark_id,
            "provenance_fingerprint": self.provenance_fingerprint,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d,
            cls.KIND,
            (
                "benchmark_id",
                "investigation_id",
                "run_id",
                "protocol_hash",
                "provenance_fingerprint",
                "coverage_status",
                "section_status",
                "summary",
                "created_at",
            ),
        )
        return cls(
            v.get_str(d, "benchmark_id"),
            v.get_str(d, "investigation_id"),
            v.get_str(d, "run_id"),
            v.get_str(d, "protocol_hash"),
            v.get_str(d, "provenance_fingerprint"),
            v.get_enum(d, "coverage_status", CoverageStatus),
            v.get_mapping(d, "section_status"),
            v.get_mapping(d, "summary"),
            v.get_time(d, "created_at"),
        )


@dataclass(frozen=True)
class BenchmarkUnit(Entity):
    """One expanded experiment unit and what happened to it. benchmark -> unit -> run is a real
    foreign-key chain; failed, skipped and unsupported units are kept with their reasons."""

    KIND: ClassVar[str] = "benchmark_unit"
    PREFIX: ClassVar[str] = "bun"
    result_id: str
    unit_key: str
    unit_kind: UnitKind
    unit_status: UnitStatus
    grid: str  # "-" for units that belong to no grid (baseline, interaction cells)
    detail: Mapping[str, object]  # fault spec, point, seed, reason, reused-from, ...
    created_at: datetime
    run_id: str | None = None  # the run that executed it (None if it never ran)

    def __post_init__(self) -> None:
        v.ref("result_id", self.result_id, BenchmarkResult.PREFIX)
        v.text("unit_key", self.unit_key)
        v.member("unit_kind", self.unit_kind, UnitKind)
        v.member("unit_status", self.unit_status, UnitStatus)
        v.text("grid", self.grid)
        object.__setattr__(self, "detail", v.freeze_mapping("detail", self.detail))
        v.timestamp("created_at", self.created_at)
        if self.run_id is not None:
            v.ref("run_id", self.run_id, Run.PREFIX)

    def _identity(self) -> Mapping[str, object]:
        return {"result_id": self.result_id, "unit_key": self.unit_key}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d,
            cls.KIND,
            (
                "result_id",
                "unit_key",
                "unit_kind",
                "unit_status",
                "grid",
                "detail",
                "created_at",
                "run_id",
            ),
        )
        return cls(
            v.get_str(d, "result_id"),
            v.get_str(d, "unit_key"),
            v.get_enum(d, "unit_kind", UnitKind),
            v.get_enum(d, "unit_status", UnitStatus),
            v.get_str(d, "grid"),
            v.get_mapping(d, "detail"),
            v.get_time(d, "created_at"),
            v.get_opt_str(d, "run_id"),
        )

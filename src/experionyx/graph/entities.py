"""Registry entities of the evidence/failure knowledge graph: the versioned DEFINITION
(`EvidenceGraph`), each construction's SNAPSHOT (`GraphSnapshot`), and the explicit `GraphNode`s
and `GraphEdge`s a snapshot contains. Immutable and content-addressed, exactly like the benchmark
engine's `Benchmark` / `BenchmarkResult` / `BenchmarkUnit` (see docs/graph.md). A `GraphNode` never
duplicates the entity it references -- it stores only the kind, the referenced ID, and whether
that ID actually resolves in this registry."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Self

import experionyx.validation as v
from experionyx.domain import Entity, Investigation, Run, open_payload
from experionyx.errors import ValidationError
from experionyx.graph.taxonomy import NodeKind, RelationType


@dataclass(frozen=True)
class EvidenceGraph(Entity):
    KIND: ClassVar[str] = "evidence_graph"
    PREFIX: ClassVar[str] = "grh"
    name: str
    version: str
    spec_id: str  # gsp_<hash> of the full construction request
    spec: Mapping[str, object]
    engine_version: str
    created_at: datetime

    def __post_init__(self) -> None:
        v.text("name", self.name)
        v.text("version", self.version)
        v.text("spec_id", self.spec_id)
        object.__setattr__(self, "spec", v.freeze_mapping("spec", self.spec))
        v.text("engine_version", self.engine_version)
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {"spec_id": self.spec_id}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(d, cls.KIND, ("name", "version", "spec_id", "spec", "engine_version", "created_at"))  # fmt: skip
        return cls(
            v.get_str(d, "name"), v.get_str(d, "version"), v.get_str(d, "spec_id"),
            v.get_mapping(d, "spec"), v.get_str(d, "engine_version"), v.get_time(d, "created_at"),
        )  # fmt: skip


@dataclass(frozen=True)
class GraphSnapshot(Entity):
    """One construction of a graph over the registry at a point in time. Identity is
    `(graph_id, source_fingerprint)`: constructing the SAME definition against UNCHANGED registry
    content always yields the SAME snapshot -- re-collection is idempotent, not duplicated, the
    same way a `BenchmarkResult` is keyed on `(benchmark_id, provenance_fingerprint)`."""

    KIND: ClassVar[str] = "graph_snapshot"
    PREFIX: ClassVar[str] = "gsn"
    graph_id: str
    investigation_id: str  # the workspace's home investigation this snapshot was built under
    run_id: str  # the collect Run that read the registry and wrote the artifacts
    source_fingerprint: str  # content hash of every (kind, id, content_hash) row actually included
    provenance_fingerprint: str
    node_count: int
    edge_count: int
    unresolved_count: int  # references that do not resolve in this registry (kept, never dropped)
    truncated: bool  # True if max_nodes/max_edges cut the construction short
    summary: Mapping[str, object]
    created_at: datetime

    def __post_init__(self) -> None:
        v.ref("graph_id", self.graph_id, EvidenceGraph.PREFIX)
        v.ref("investigation_id", self.investigation_id, Investigation.PREFIX)
        v.ref("run_id", self.run_id, Run.PREFIX)
        v.digest("source_fingerprint", self.source_fingerprint)
        v.digest("provenance_fingerprint", self.provenance_fingerprint)
        v.non_negative_int("node_count", self.node_count)
        v.non_negative_int("edge_count", self.edge_count)
        v.non_negative_int("unresolved_count", self.unresolved_count)
        if not isinstance(self.truncated, bool):
            raise ValidationError("truncated must be a boolean")
        object.__setattr__(self, "summary", v.freeze_mapping("summary", self.summary))
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {"graph_id": self.graph_id, "source_fingerprint": self.source_fingerprint}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d, cls.KIND,
            ("graph_id", "investigation_id", "run_id", "source_fingerprint", "provenance_fingerprint",
             "node_count", "edge_count", "unresolved_count", "truncated", "summary", "created_at"),
        )  # fmt: skip
        truncated = v.get_raw(d, "truncated")
        if not isinstance(truncated, bool):
            raise ValidationError("truncated must be a boolean")
        return cls(
            v.get_str(d, "graph_id"), v.get_str(d, "investigation_id"), v.get_str(d, "run_id"),
            v.get_str(d, "source_fingerprint"), v.get_str(d, "provenance_fingerprint"),
            v.get_int(d, "node_count"), v.get_int(d, "edge_count"), v.get_int(d, "unresolved_count"),
            truncated, v.get_mapping(d, "summary"), v.get_time(d, "created_at"),
        )  # fmt: skip


@dataclass(frozen=True)
class GraphNode(Entity):
    """One node: a reference to an existing entity (or, for a synthetic kind, a graph-only
    concept), never a copy of its data. `resolved=False` represents a dangling reference
    explicitly -- it is never silently dropped."""

    KIND: ClassVar[str] = "graph_node"
    PREFIX: ClassVar[str] = "gnd"
    snapshot_id: str
    node_kind: NodeKind
    ref_id: str  # an existing entity ID, or a synthetic key for SPLIT/DESCRIPTOR/UNRESOLVED
    label: str
    resolved: bool
    detail: Mapping[str, object]
    created_at: datetime

    def __post_init__(self) -> None:
        v.ref("snapshot_id", self.snapshot_id, GraphSnapshot.PREFIX)
        v.member("node_kind", self.node_kind, NodeKind)
        v.text("ref_id", self.ref_id)
        v.text("label", self.label)
        if not isinstance(self.resolved, bool):
            raise ValidationError("resolved must be a boolean")
        object.__setattr__(self, "detail", v.freeze_mapping("detail", self.detail))
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {"snapshot_id": self.snapshot_id, "node_kind": self.node_kind, "ref_id": self.ref_id}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d,
            cls.KIND,
            ("snapshot_id", "node_kind", "ref_id", "label", "resolved", "detail", "created_at"),
        )
        resolved = v.get_raw(d, "resolved")
        if not isinstance(resolved, bool):
            raise ValidationError("resolved must be a boolean")
        return cls(
            v.get_str(d, "snapshot_id"), v.get_enum(d, "node_kind", NodeKind), v.get_str(d, "ref_id"),
            v.get_str(d, "label"), resolved, v.get_mapping(d, "detail"), v.get_time(d, "created_at"),
        )  # fmt: skip


@dataclass(frozen=True)
class GraphEdge(Entity):
    """One directed, typed edge between two `GraphNode`s of the same snapshot. Parallel edges of
    different relation/detail between the same two nodes are legal (the graph is not a DAG);
    an identical edge added twice collapses to one immutable record (content-addressed)."""

    KIND: ClassVar[str] = "graph_edge"
    PREFIX: ClassVar[str] = "ged"
    snapshot_id: str
    from_node_id: str
    to_node_id: str
    relation: RelationType
    detail: Mapping[str, object]
    created_at: datetime

    def __post_init__(self) -> None:
        v.ref("snapshot_id", self.snapshot_id, GraphSnapshot.PREFIX)
        v.ref("from_node_id", self.from_node_id, GraphNode.PREFIX)
        v.ref("to_node_id", self.to_node_id, GraphNode.PREFIX)
        v.member("relation", self.relation, RelationType)
        object.__setattr__(self, "detail", v.freeze_mapping("detail", self.detail))
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {
            "snapshot_id": self.snapshot_id,
            "from_node_id": self.from_node_id,
            "to_node_id": self.to_node_id,
            "relation": self.relation,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d,
            cls.KIND,
            ("snapshot_id", "from_node_id", "to_node_id", "relation", "detail", "created_at"),
        )
        return cls(
            v.get_str(d, "snapshot_id"), v.get_str(d, "from_node_id"), v.get_str(d, "to_node_id"),
            v.get_enum(d, "relation", RelationType), v.get_mapping(d, "detail"), v.get_time(d, "created_at"),
        )  # fmt: skip

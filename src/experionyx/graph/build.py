"""Deterministic, generic construction of graph nodes and edges from the registry's OWN content.

Every registry entity is already, by this project's own convention, content-addressed and refers
to the entities it depends on by ID (`<prefix>_<32 hex>`, see docs/identity.md). This module never
hand-writes "read a fault trial and link it to its baseline run" for every phase; instead it walks
each entity's dataclass fields (recursively through nested dataclasses, never into free-form JSON
blobs such as `spec`/`detail`/`summary`/`parameters`) and turns every string that LOOKS like an
entity reference into an edge, resolving its target kind from the reference's own prefix. Two
entity types need a few lines of special handling because their fields are not literal registry
IDs: `Evidence.relation` picks the edge label (SUPPORTS/REFUTES/CONTEXTUALIZES) instead of a fixed
default, and `FailureRelationship`'s `FAULT`/`CLASS`/`SLICE` endpoints are descriptor strings, not
IDs (see `failures.taxonomy.NodeKind`), so they become synthetic `DESCRIPTOR` nodes. A `split`
value paired with a `dataset_fingerprint` value on ANY entity becomes a synthetic `SPLIT` node,
handled once, generically, rather than per analysis kind.

This design means a future entity kind needs NO new code here to appear in the graph: it just
needs to be added to `ENTITY_TYPES`.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass, fields, is_dataclass

from experionyx.adapters.records import RegisteredDataset, RegisteredModel
from experionyx.benchmark.entities import Benchmark, BenchmarkResult, BenchmarkUnit
from experionyx.calibration.entities import CalibrationAnalysis, CalibrationResult
from experionyx.data_quality.entities import QualityAnalysis, QualityCheck
from experionyx.domain import (
    Artifact,
    Claim,
    ConfigurationRef,
    Entity,
    EnvironmentSnapshot,
    Evidence,
    EvidenceRelation,
    Experiment,
    Investigation,
    Observation,
    Run,
)
from experionyx.drift.entities import DriftAnalysis, DriftWindow
from experionyx.failures.entities import (
    FailureCluster,
    FailureEvidence,
    FailureMode,
    FailureRelationship,
    FailureSignal,
)
from experionyx.failures.taxonomy import NodeKind as FailureEndpointKind
from experionyx.faults.entities import FaultAnalysis, FaultExperiment, FaultTrial
from experionyx.graph.spec import GraphSpec
from experionyx.graph.taxonomy import PREFIX_TO_KIND, NodeKind, RelationType
from experionyx.hashing import HASH_PREFIX, content_hash
from experionyx.interactions.entities import (
    InteractionAnalysis,
    InteractionEffect,
    InteractionEvidence,
)
from experionyx.provenance import Provenance, RunOutcome
from experionyx.registry import Registry
from experionyx.reliability.entities import ReliabilityProfile, ReliabilityReference
from experionyx.reproducibility.entities import ReproductionAttempt
from experionyx.resources.entities import ResourceAnalysis, ResourceTrial
from experionyx.scheduler.entities import (
    ExecutionAttempt,
    Schedule,
    ScheduleRun,
    ScheduleUnit,
    UnitStateTransition,
)
from experionyx.slices.entities import Slice, SliceAnalysis
from experionyx.stats.entities import StatisticalAnalysis
from experionyx.stress.entities import StressAnalysis, StressTrial

ENTITY_TYPES: tuple[type[Entity], ...] = (
    Investigation, ConfigurationRef, EnvironmentSnapshot, Experiment, Run, Observation, Artifact,
    Claim, Evidence, Provenance, RunOutcome, RegisteredModel, RegisteredDataset,
    FaultExperiment, FaultTrial, FaultAnalysis,
    FailureSignal, FailureCluster, FailureMode, FailureEvidence, FailureRelationship,
    InteractionAnalysis, InteractionEffect, InteractionEvidence,
    ReliabilityProfile, ReliabilityReference,
    Benchmark, BenchmarkResult, BenchmarkUnit,
    StatisticalAnalysis,
    Slice, SliceAnalysis,
    DriftWindow, DriftAnalysis,
    QualityAnalysis, QualityCheck,
    StressAnalysis, StressTrial,
    CalibrationAnalysis, CalibrationResult,
    ResourceAnalysis, ResourceTrial,
    Schedule, ScheduleRun, ScheduleUnit, ExecutionAttempt, UnitStateTransition,
    ReproductionAttempt,
)  # fmt: skip

_ID_RE = re.compile(r"[a-z]{3}_[0-9a-f]{32}")

# field name -> edge label, for the generic (non-special-cased) reference scan. Anything not
# listed defaults to REFERENCES -- still a real, typed edge, just the least specific one.
FIELD_RELATION: dict[str, RelationType] = {
    "investigation_id": RelationType.MEMBER_OF,
    "schedule_id": RelationType.MEMBER_OF,
    "schedule_run_id": RelationType.MEMBER_OF,
    "unit_id": RelationType.MEMBER_OF,
    "cluster_id": RelationType.MEMBER_OF,
    "benchmark_id": RelationType.MEMBER_OF,
    "analysis_id": RelationType.ANALYZED_BY,
    "result_id": RelationType.ANALYZED_BY,
    "profile_id": RelationType.ANALYZED_BY,
    "configuration_id": RelationType.DEPENDS_ON,
    "depends_on": RelationType.DEPENDS_ON,
    "experiment_id": RelationType.DERIVED_FROM,
    "baseline_experiment_id": RelationType.DERIVED_FROM,
    "treatment_experiment_id": RelationType.DERIVED_FROM,
    "fault_experiment_id": RelationType.DISCOVERED_FROM,
    "run_id": RelationType.DERIVED_FROM,
    "baseline_run_id": RelationType.DERIVED_FROM,
    "treatment_run_id": RelationType.DERIVED_FROM,
    "environment_id": RelationType.OBSERVED_IN,
    "replay_of": RelationType.REPRODUCED_BY,
    "diagnostic_artifact_id": RelationType.PRODUCED_ARTIFACT,
    "artifact_ids": RelationType.PRODUCED_ARTIFACT,
    "claim_id": RelationType.EVIDENCE_FOR,
    "signal_ids": RelationType.MEMBER_OF,
    "failure_mode_id": RelationType.EVIDENCE_FOR,
    "model_id": RelationType.USES_MODEL,
    "model_record_id": RelationType.USES_MODEL,
    "dataset_id": RelationType.USES_DATASET,
    "dataset_record_id": RelationType.USES_DATASET,
    "target_id": RelationType.REPRODUCED_BY,
    "replay_run_id": RelationType.COMPARED_WITH,
}

_DESCRIPTOR_ENDPOINTS = frozenset(
    {FailureEndpointKind.FAULT, FailureEndpointKind.CLASS, FailureEndpointKind.SLICE}
)


@dataclass(frozen=True)
class NodeDraft:
    kind: NodeKind
    ref_id: str
    resolved: bool
    detail: dict[str, object]


@dataclass(frozen=True)
class EdgeDraft:
    from_key: tuple[NodeKind, str]
    to_key: tuple[NodeKind, str]
    relation: RelationType
    detail: dict[str, object]


@dataclass(frozen=True)
class Constructed:
    nodes: dict[tuple[NodeKind, str], NodeDraft]  # keyed by (kind, ref_id): naturally deduplicated
    edges: list[EdgeDraft]  # de-duplicated by (from_key, to_key, relation, detail) before return
    unresolved_count: int
    truncated: bool
    source_fingerprint: str


def _split_node(dataset_fingerprint: str, split: str) -> str:
    return (
        "spl_"
        + content_hash({"dataset_fingerprint": dataset_fingerprint, "split": split})[
            len(HASH_PREFIX) :
        ][:32]
    )


def _descriptor_node(kind: FailureEndpointKind, raw: str) -> str:
    return "dsc_" + content_hash({"kind": kind.value, "value": raw})[len(HASH_PREFIX) :][:32]


def _walk(value: object, field_name: str, out: list[tuple[str, str]]) -> None:
    """Collect (field_name, string_value) pairs from `value`, recursing through nested
    dataclasses (config/provenance value objects) but never into `Mapping`-typed JSON blobs."""
    if isinstance(value, str):
        out.append((field_name, value))
    elif isinstance(value, tuple):
        for item in value:
            _walk(item, field_name, out)
    elif is_dataclass(value) and not isinstance(value, type):
        for f in fields(value):
            _walk(getattr(value, f.name), f.name, out)


def _resolve_ref(registry: Registry, ref: str) -> tuple[NodeKind, bool] | None:
    m = _ID_RE.fullmatch(ref)
    if not m:
        return None
    prefix = ref[:3]
    kind = PREFIX_TO_KIND.get(prefix)
    if kind is None:
        return None  # a content-identity hash embedded in JSON, not a registry table reference
    cls = _KIND_TO_CLASS.get(kind)
    resolved = cls is not None and registry.exists(cls, ref)
    return kind, resolved


class _Builder:
    def __init__(self, spec: GraphSpec) -> None:
        self.spec = spec
        self.nodes: dict[tuple[NodeKind, str], NodeDraft] = {}
        self.edges: dict[
            tuple[tuple[NodeKind, str], tuple[NodeKind, str], RelationType, str], dict[str, object]
        ] = {}
        self.unresolved = 0
        self.truncated = False

    def _add_node(self, kind: NodeKind, ref_id: str, resolved: bool, detail: dict[str, object]) -> tuple[NodeKind, str]:  # fmt: skip
        key = (kind, ref_id)
        if key not in self.nodes:
            if len(self.nodes) >= self.spec.max_nodes:
                self.truncated = True
                return key
            self.nodes[key] = NodeDraft(kind, ref_id, resolved, detail)
            if not resolved:
                self.unresolved += 1
        return key

    def _add_edge(self, frm: tuple[NodeKind, str], to: tuple[NodeKind, str], relation: RelationType, detail: dict[str, object]) -> None:  # fmt: skip
        key = (frm, to, relation, content_hash(detail))
        if key not in self.edges and len(self.edges) >= self.spec.max_edges:
            self.truncated = True
            return
        self.edges[key] = detail

    def _self_node(self, entity: Entity) -> tuple[NodeKind, str] | None:
        kind = PREFIX_TO_KIND.get(entity.PREFIX)
        if kind is None or not self.spec.includes(kind):
            return None
        return self._add_node(kind, entity.id, True, {})

    def _generic_edges(self, registry: Registry, entity: Entity, self_key: tuple[NodeKind, str]) -> None:  # fmt: skip
        overrides = _overrides(entity)
        found: list[tuple[str, str]] = []
        if is_dataclass(entity) and not isinstance(entity, type):
            for f in fields(entity):
                if f.name == "id":
                    continue
                _walk(getattr(entity, f.name), f.name, found)
        for field_name, value in found:
            resolved_info = _resolve_ref(registry, value)
            if resolved_info is None:
                continue
            kind, resolved = resolved_info
            if not self.spec.includes(kind):
                continue
            to_key = self._add_node(kind, value, resolved, {})
            relation = overrides.get(field_name) or FIELD_RELATION.get(field_name, RelationType.REFERENCES)  # fmt: skip
            self._add_edge(self_key, to_key, relation, {"field": field_name})

    def _split_edge(self, entity: Entity, self_key: tuple[NodeKind, str]) -> None:
        split = getattr(entity, "split", None)
        fp = getattr(entity, "dataset_fingerprint", None)
        if not isinstance(split, str) or split in ("", "*") or not isinstance(fp, str) or not fp:
            return
        if not self.spec.includes(NodeKind.SPLIT):
            return
        node = self._add_node(NodeKind.SPLIT, _split_node(fp, split), True, {"dataset_fingerprint": fp, "split": split})  # fmt: skip
        self._add_edge(self_key, node, RelationType.USES_SPLIT, {})

    def _failure_relationship_edges(self, entity: FailureRelationship, self_key: tuple[NodeKind, str]) -> None:  # fmt: skip
        if not self.spec.includes(NodeKind.DESCRIPTOR):
            return
        for endpoint_kind, raw_id, field_name in (
            (entity.subject_kind, entity.subject_id, "subject_id"),
            (entity.object_kind, entity.object_id, "object_id"),
        ):
            if endpoint_kind not in _DESCRIPTOR_ENDPOINTS:
                continue
            node = self._add_node(NodeKind.DESCRIPTOR, _descriptor_node(endpoint_kind, raw_id), True, {"kind": endpoint_kind.value, "value": raw_id})  # fmt: skip
            self._add_edge(self_key, node, RelationType.REFERENCES, {"predicate": entity.predicate.value, "field": field_name})  # fmt: skip

    def add(self, registry: Registry, entity: Entity) -> None:
        self_key = self._self_node(entity)
        if self_key is None:
            return
        self._generic_edges(registry, entity, self_key)
        self._split_edge(entity, self_key)
        if isinstance(entity, FailureRelationship):
            self._failure_relationship_edges(entity, self_key)


_KIND_TO_CLASS: dict[NodeKind, type[Entity]] = {}
for _cls in ENTITY_TYPES:
    _kind = PREFIX_TO_KIND.get(_cls.PREFIX)
    if _kind is not None:
        _KIND_TO_CLASS[_kind] = _cls


def _overrides(entity: Entity) -> dict[str, RelationType]:
    if isinstance(entity, Evidence):
        return {
            "target_id": {
                EvidenceRelation.SUPPORTS: RelationType.SUPPORTS,
                EvidenceRelation.REFUTES: RelationType.REFUTES,
                EvidenceRelation.CONTEXT: RelationType.CONTEXTUALIZES,
            }[entity.relation]
        }
    return {}


def _in_scope(registry: Registry, spec: GraphSpec, entity: Entity) -> bool:
    if spec.investigation_id is None:
        return True
    inv = getattr(entity, "investigation_id", None)
    return inv is None or inv == spec.investigation_id


@dataclass(frozen=True)
class _Excluded:
    """The graph engine's own collect-run bookkeeping (see `engine.run_graph_collect`): a graph
    construction never treats ITSELF, or an earlier graph construction's own
    ConfigurationRef/Experiment/Run/Claim (and anything keyed off that run or claim), as evidence.
    Without this, every graph would permanently absorb the previous one's bookkeeping and never
    reach a stable, idempotent identity. `EvidenceGraph`/`GraphSnapshot`/`GraphNode`/`GraphEdge`
    are never in `ENTITY_TYPES` at all, so a graph of graphs is excluded structurally, not here."""

    config_ids: frozenset[str]
    experiment_ids: frozenset[str]
    run_ids: frozenset[str]
    claim_ids: frozenset[str]
    environment_ids: frozenset[str]  # environments used EXCLUSIVELY by excluded runs


_CLAIM_ASSERTER_PREFIX = "experionyx.graph/"


def _excluded(registry: Registry) -> _Excluded:
    from experionyx.domain import Claim, ConfigurationRef, EnvironmentSnapshot, Experiment, Run

    config_ids = frozenset(
        c.id for c in registry.find(ConfigurationRef) if "graph_collect" in c.parameters
    )
    experiment_ids = frozenset(
        e.id for e in registry.find(Experiment) if e.configuration_id in config_ids
    )
    excluded_runs = [r for e in experiment_ids for r in registry.find(Run, experiment_id=e)]
    run_ids = frozenset(r.id for r in excluded_runs)
    claim_ids = frozenset(
        c.id for c in registry.find(Claim) if c.asserted_by.startswith(_CLAIM_ASSERTER_PREFIX)
    )
    environment_ids = frozenset(
        env.id
        for env in (registry.get(EnvironmentSnapshot, r.environment_id) for r in excluded_runs)
        if {r.id for r in registry.find(Run, environment_id=env.id)} <= run_ids
    )
    return _Excluded(config_ids, experiment_ids, run_ids, claim_ids, environment_ids)


def _is_bookkeeping(entity: Entity, excluded: _Excluded) -> bool:
    from experionyx.domain import (  # fmt: skip
        Claim,
        ConfigurationRef,
        EnvironmentSnapshot,
        Evidence,
        Experiment,
        Run,
    )

    if isinstance(entity, ConfigurationRef):
        return entity.id in excluded.config_ids
    if isinstance(entity, Experiment):
        return entity.id in excluded.experiment_ids
    if isinstance(entity, Run):
        return entity.id in excluded.run_ids
    if isinstance(entity, Claim):
        return entity.id in excluded.claim_ids
    if isinstance(entity, Evidence):
        return entity.claim_id in excluded.claim_ids
    if isinstance(entity, EnvironmentSnapshot):
        return entity.id in excluded.environment_ids
    run_id = getattr(entity, "run_id", None)
    return isinstance(run_id, str) and run_id in excluded.run_ids


def _iter_entities(registry: Registry, spec: GraphSpec) -> Iterable[Entity]:
    excluded = _excluded(registry)
    for cls in ENTITY_TYPES:
        kind = PREFIX_TO_KIND[cls.PREFIX]
        if not spec.includes(kind):
            continue
        for entity in registry.find(cls):
            if _is_bookkeeping(entity, excluded):
                continue
            if _in_scope(registry, spec, entity):
                yield entity


def construct(registry: Registry, spec: GraphSpec) -> Constructed:
    """Deterministically build the node/edge set for `spec` against `registry`'s CURRENT content.
    Calling this twice against unchanged registry content produces an identical result (same
    nodes, same edges, same `source_fingerprint`); it never executes anything."""
    b = _Builder(spec)
    fp_parts: list[tuple[str, str, str]] = []
    for entity in _iter_entities(registry, spec):
        fp_parts.append((entity.PREFIX, entity.id, entity.content_hash()))
        b.add(registry, entity)
    fp_parts.sort()
    source_fingerprint = content_hash({"engine": "1.0.0", "rows": fp_parts})
    edges = [
        EdgeDraft(frm, to, relation, detail)
        for (frm, to, relation, _digest), detail in sorted(
            b.edges.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2].value, kv[0][3])
        )
    ]
    return Constructed(b.nodes, edges, b.unresolved, b.truncated, source_fingerprint)


__all__ = ["ENTITY_TYPES", "Constructed", "EdgeDraft", "NodeDraft", "construct"]

"""Graph domain: deterministic identity, spec validation, the generic reference extractor and the
adjacency/traversal primitives. No registry beyond what a test needs directly."""

from datetime import UTC, datetime

import pytest

from experionyx.errors import ValidationError
from experionyx.graph.build import _descriptor_node, _resolve_ref, _split_node, _walk
from experionyx.graph.entities import EvidenceGraph, GraphEdge, GraphNode, GraphSnapshot
from experionyx.graph.spec import DEFAULT_QUERY, GraphQuery, GraphSpec
from experionyx.graph.taxonomy import PREFIX_TO_KIND, NodeKind, RelationType
from experionyx.sqlite import SqliteRegistry

T0 = datetime(2026, 9, 24, tzinfo=UTC)
DIGEST = "sha256:" + "ab" * 32


# --- GraphSpec --------------------------------------------------------------------------------


def test_spec_id_is_deterministic_and_content_addressed() -> None:
    a = GraphSpec("g", "1.0.0")
    b = GraphSpec("g", "1.0.0")
    c = GraphSpec("g", "1.0.1")
    assert a.spec_id == b.spec_id
    assert a.spec_id != c.spec_id


def test_spec_id_changes_with_included_kinds_and_bounds() -> None:
    base = GraphSpec("g", "1.0.0")
    kinds = GraphSpec("g", "1.0.0", included_kinds=(NodeKind.RUN,))
    bounds = GraphSpec("g", "1.0.0", max_nodes=10)
    assert base.spec_id != kinds.spec_id
    assert base.spec_id != bounds.spec_id


def test_spec_rejects_bad_version() -> None:
    with pytest.raises(ValidationError):
        GraphSpec("g", "1.0")


def test_spec_round_trips_through_dict() -> None:
    spec = GraphSpec("g", "1.0.0", investigation_id="inv_" + "1" * 32, included_kinds=(NodeKind.RUN, NodeKind.MODEL))  # fmt: skip
    again = GraphSpec.from_dict(spec.to_dict())
    assert again.spec_id == spec.spec_id
    assert again.included_kinds == (NodeKind.MODEL, NodeKind.RUN)  # sorted, deterministic


def test_spec_includes_respects_included_kinds() -> None:
    everything = GraphSpec("g", "1.0.0")
    assert everything.includes(NodeKind.RUN)
    restricted = GraphSpec("g", "1.0.0", included_kinds=(NodeKind.RUN,))
    assert restricted.includes(NodeKind.RUN)
    assert not restricted.includes(NodeKind.MODEL)


def test_graph_query_rejects_invalid_bounds() -> None:
    with pytest.raises(ValidationError):
        GraphQuery(max_depth=-1)
    with pytest.raises(ValidationError):
        GraphQuery(max_visited=0)


# --- generic reference extraction ---------------------------------------------------------------


def test_walk_collects_strings_recursively_through_tuples_and_dataclasses() -> None:
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Inner:
        ref: str

    @dataclass(frozen=True)
    class Outer:
        direct: str
        nested: Inner
        many: tuple[str, ...]

    out: list[tuple[str, str]] = []
    _walk(Outer("a", Inner("b"), ("c", "d")), "top", out)
    assert ("direct", "a") in out
    assert ("ref", "b") in out
    assert ("many", "c") in out and ("many", "d") in out


def test_walk_never_descends_into_mappings() -> None:
    out: list[tuple[str, str]] = []
    _walk({"hidden": "sta_" + "0" * 32}, "spec", out)
    assert out == []  # a Mapping value is opaque; nothing inside it is scanned


def test_resolve_ref_recognizes_known_prefixes_only() -> None:
    with SqliteRegistry(":memory:") as reg:
        assert _resolve_ref(reg, "run_" + "0" * 32) == (NodeKind.RUN, False)  # doesn't exist
        assert _resolve_ref(reg, "xyz_" + "0" * 32) is None  # unrecognized prefix: not scanned
        assert _resolve_ref(reg, "not-an-id") is None


def test_split_node_and_descriptor_node_are_deterministic() -> None:
    from experionyx.failures.taxonomy import NodeKind as FailureEndpointKind

    a = _split_node("sha256:" + "1" * 64, "train")
    b = _split_node("sha256:" + "1" * 64, "train")
    c = _split_node("sha256:" + "1" * 64, "test")
    assert a == b
    assert a != c
    d1 = _descriptor_node(FailureEndpointKind.FAULT, "gaussian_noise")
    d2 = _descriptor_node(FailureEndpointKind.FAULT, "gaussian_noise")
    assert d1 == d2


# --- entity identity and validation --------------------------------------------------------------


def test_evidence_graph_identity_is_spec_id_only() -> None:
    a = EvidenceGraph("g", "1.0.0", "gsp_" + "1" * 32, {"x": 1}, "1.0.0", T0)
    b = EvidenceGraph("g", "1.0.0", "gsp_" + "1" * 32, {"x": 1}, "1.0.0", datetime(2020, 1, 1, tzinfo=UTC))  # fmt: skip
    assert a.id == b.id  # created_at is not identity


def test_graph_snapshot_identity_is_graph_and_source_fingerprint() -> None:
    def make(fp: str) -> GraphSnapshot:
        return GraphSnapshot(
            "grh_" + "1" * 32, "inv_" + "1" * 32, "run_" + "1" * 32, fp, DIGEST, 1, 1, 0, False, {}, T0
        )  # fmt: skip

    a = make(DIGEST)
    b = make(DIGEST)
    c = make("sha256:" + "cd" * 32)
    assert a.id == b.id
    assert a.id != c.id


def test_graph_node_identity_changes_with_kind_or_ref_id() -> None:
    def make(kind: NodeKind, ref: str) -> GraphNode:
        return GraphNode("gsn_" + "1" * 32, kind, ref, "label", True, {}, T0)

    base = make(NodeKind.RUN, "run_" + "1" * 32)
    assert make(NodeKind.RUN, "run_" + "1" * 32).id == base.id
    assert make(NodeKind.MODEL, "run_" + "1" * 32).id != base.id
    assert make(NodeKind.RUN, "run_" + "2" * 32).id != base.id


def test_graph_edge_identity_changes_with_relation_or_detail() -> None:
    def make(relation: RelationType, detail: dict[str, object]) -> GraphEdge:
        return GraphEdge("gsn_" + "1" * 32, "gnd_" + "1" * 32, "gnd_" + "2" * 32, relation, detail, T0)  # fmt: skip

    base = make(RelationType.DERIVED_FROM, {"field": "run_id"})
    assert make(RelationType.DERIVED_FROM, {"field": "run_id"}).id == base.id
    assert make(RelationType.MEMBER_OF, {"field": "run_id"}).id != base.id
    assert make(RelationType.DERIVED_FROM, {"field": "other"}).id != base.id


def test_graph_edge_rejects_unknown_relation() -> None:
    with pytest.raises(ValidationError):
        GraphEdge.from_dict(
            {
                "kind": "graph_edge", "schema_version": 1,
                "snapshot_id": "gsn_" + "1" * 32, "from_node_id": "gnd_" + "1" * 32,
                "to_node_id": "gnd_" + "2" * 32, "relation": "NOT_A_REAL_RELATION",
                "detail": {}, "created_at": T0.isoformat(),
            }
        )  # fmt: skip


def test_prefix_to_kind_covers_every_node_kind_except_synthetic() -> None:
    synthetic = {NodeKind.SPLIT, NodeKind.DESCRIPTOR, NodeKind.UNRESOLVED}
    covered = set(PREFIX_TO_KIND.values())
    assert covered == set(NodeKind) - synthetic


def test_default_query_has_sane_bounds() -> None:
    assert DEFAULT_QUERY.max_depth > 0
    assert DEFAULT_QUERY.max_visited > 0


# --- split synthesis and FailureRelationship descriptor handling (white-box on _Builder) --------


def test_split_synthesis_creates_a_uses_split_edge() -> None:
    from dataclasses import dataclass

    from experionyx.graph.build import _Builder
    from experionyx.graph.taxonomy import NodeKind as GK
    from experionyx.graph.taxonomy import RelationType as GR

    @dataclass(frozen=True)
    class FakeAnalysis:
        PREFIX = "sta"
        id = "sta_" + "1" * 32
        split: str = "train"
        dataset_fingerprint: str = "sha256:" + "2" * 64

    b = _Builder(GraphSpec("g", "1.0.0"))
    self_key = b._self_node(FakeAnalysis())  # type: ignore[arg-type]
    assert self_key is not None
    b._split_edge(FakeAnalysis(), self_key)  # type: ignore[arg-type]
    split_nodes = [k for k in b.nodes if k[0] is GK.SPLIT]
    assert len(split_nodes) == 1
    edges = [e for e in b.edges if e[2] is GR.USES_SPLIT]
    assert len(edges) == 1


def test_split_synthesis_skips_wildcard_and_empty_split() -> None:
    from dataclasses import dataclass

    from experionyx.graph.build import _Builder
    from experionyx.graph.taxonomy import NodeKind as GK

    @dataclass(frozen=True)
    class FakeAnalysis:
        PREFIX = "sta"
        id = "sta_" + "1" * 32
        split: str = "*"
        dataset_fingerprint: str = "sha256:" + "2" * 64

    b = _Builder(GraphSpec("g", "1.0.0"))
    self_key = b._self_node(FakeAnalysis())  # type: ignore[arg-type]
    assert self_key is not None
    b._split_edge(FakeAnalysis(), self_key)  # type: ignore[arg-type]
    assert not [k for k in b.nodes if k[0] is GK.SPLIT]


def test_failure_relationship_descriptor_endpoints_become_synthetic_nodes() -> None:
    from experionyx.failures.entities import FailureRelationship
    from experionyx.failures.taxonomy import NodeKind as FK
    from experionyx.failures.taxonomy import Predicate
    from experionyx.graph.build import _Builder
    from experionyx.graph.taxonomy import NodeKind as GK

    rel = FailureRelationship(
        "inv_" + "1" * 32, FK.FAULT, "gaussian_noise", Predicate.REVEALS, FK.FAILURE_MODE,
        "fmd_" + "1" * 32, {}, T0,
    )  # fmt: skip
    b = _Builder(GraphSpec("g", "1.0.0"))
    self_key = b._self_node(rel)
    assert self_key is not None
    b._failure_relationship_edges(rel, self_key)
    descriptor_nodes = [k for k in b.nodes if k[0] is GK.DESCRIPTOR]
    assert len(descriptor_nodes) == 1  # FAULT is a descriptor; FAILURE_MODE (a real ID) is not

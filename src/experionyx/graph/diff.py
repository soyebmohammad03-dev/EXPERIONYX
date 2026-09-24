"""Comparison between two graph snapshots. Reports raw set differences (added/removed nodes,
added/removed edges); it never labels a difference an improvement or a regression -- the same
"raw differences, no verdict" stance as `benchmark.compare` and `resources.compare` (see
docs/graph.md)."""

from dataclasses import dataclass

from experionyx.graph.query import EdgeView, NodeView, SnapshotIndex
from experionyx.registry import Registry


@dataclass(frozen=True)
class GraphDiff:
    snapshot_a: str
    snapshot_b: str
    added_nodes: tuple[NodeView, ...]  # in b, not in a
    removed_nodes: tuple[NodeView, ...]  # in a, not in b
    added_edges: tuple[EdgeView, ...]
    removed_edges: tuple[EdgeView, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "snapshot_a": self.snapshot_a,
            "snapshot_b": self.snapshot_b,
            "added_nodes": [n.to_dict() for n in self.added_nodes],
            "removed_nodes": [n.to_dict() for n in self.removed_nodes],
            "added_edges": [e.to_dict() for e in self.added_edges],
            "removed_edges": [e.to_dict() for e in self.removed_edges],
        }


def _node_key(n: NodeView) -> tuple[str, str]:
    return (n.kind.value, n.ref_id)


def _edge_key(e: EdgeView, nodes: dict[str, NodeView]) -> tuple[tuple[str, str], tuple[str, str], str]:  # fmt: skip
    return (_node_key(nodes[e.from_node_id]), _node_key(nodes[e.to_node_id]), e.relation)


def diff(registry: Registry, snapshot_id_a: str, snapshot_id_b: str) -> GraphDiff:
    """Compare two snapshots by the content each node/edge actually represents (kind + referenced
    ID for nodes; endpoint identity + relation for edges) rather than by internal row ID, so the
    same logical node/edge across two different snapshots is recognized as unchanged."""
    a = SnapshotIndex(registry, snapshot_id_a)
    b = SnapshotIndex(registry, snapshot_id_b)
    a_nodes = {_node_key(NodeView.of(n)): NodeView.of(n) for n in a.nodes.values()}
    b_nodes = {_node_key(NodeView.of(n)): NodeView.of(n) for n in b.nodes.values()}
    added_node_keys = set(b_nodes) - set(a_nodes)
    removed_node_keys = set(a_nodes) - set(b_nodes)

    a_node_views = {n.id: NodeView.of(n) for n in a.nodes.values()}
    b_node_views = {n.id: NodeView.of(n) for n in b.nodes.values()}
    a_edges = {_edge_key(EdgeView.of(e), a_node_views): EdgeView.of(e) for e in a.edges.values()}
    b_edges = {_edge_key(EdgeView.of(e), b_node_views): EdgeView.of(e) for e in b.edges.values()}
    added_edge_keys = set(b_edges) - set(a_edges)
    removed_edge_keys = set(a_edges) - set(b_edges)

    return GraphDiff(
        snapshot_id_a,
        snapshot_id_b,
        tuple(sorted((b_nodes[k] for k in added_node_keys), key=_node_key)),
        tuple(sorted((a_nodes[k] for k in removed_node_keys), key=_node_key)),
        tuple(sorted((b_edges[k] for k in added_edge_keys), key=lambda e: e.id)),
        tuple(sorted((a_edges[k] for k in removed_edge_keys), key=lambda e: e.id)),
    )


__all__ = ["GraphDiff", "diff"]

"""Deterministic, bounded traversal over a persisted `GraphSnapshot`. Every traversal stops at
whichever of `GraphQuery.max_depth` / `max_visited` it hits first and reports `truncated=True`;
nothing is silently cut off (see docs/graph.md). An edge is evidence/provenance metadata; these
functions never infer, rank or explain WHY two nodes are connected -- they report which paths
exist, not what they mean.
"""

from collections import deque
from dataclasses import dataclass

from experionyx.graph.entities import GraphEdge, GraphNode, GraphSnapshot
from experionyx.graph.spec import DEFAULT_QUERY, GraphQuery
from experionyx.graph.taxonomy import NodeKind, RelationType, TraversalDirection
from experionyx.registry import Registry


@dataclass(frozen=True)
class NodeView:
    id: str
    kind: NodeKind
    ref_id: str
    label: str
    resolved: bool

    @classmethod
    def of(cls, n: GraphNode) -> "NodeView":
        return cls(n.id, n.node_kind, n.ref_id, n.label, n.resolved)

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, "kind": self.kind.value, "ref_id": self.ref_id, "label": self.label, "resolved": self.resolved}  # fmt: skip


@dataclass(frozen=True)
class EdgeView:
    id: str
    from_node_id: str
    to_node_id: str
    relation: str
    detail: dict[str, object]

    @classmethod
    def of(cls, e: GraphEdge) -> "EdgeView":
        return cls(e.id, e.from_node_id, e.to_node_id, e.relation.value, dict(e.detail))

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, "from": self.from_node_id, "to": self.to_node_id, "relation": self.relation, "detail": self.detail}  # fmt: skip


@dataclass(frozen=True)
class TraversalResult:
    nodes: tuple[NodeView, ...]  # in visitation order; the start node(s) come first
    edges: tuple[EdgeView, ...]  # edges actually traversed, in the order they were followed
    truncated: bool
    visited_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "truncated": self.truncated,
            "visited_count": self.visited_count,
        }


@dataclass(frozen=True)
class PathResult:
    found: bool
    nodes: tuple[NodeView, ...]  # the path, start to end (empty if not found)
    edges: tuple[EdgeView, ...]  # one fewer than nodes
    truncated: bool  # the search hit a bound before it could rule the path out

    def to_dict(self) -> dict[str, object]:
        return {
            "found": self.found,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "truncated": self.truncated,
        }


class SnapshotIndex:
    """An in-memory adjacency index of one snapshot's nodes/edges, built once and reused across
    several queries against it (avoids re-reading the whole snapshot per call)."""

    def __init__(self, registry: Registry, snapshot_id: str) -> None:
        registry.get(GraphSnapshot, snapshot_id)  # raises NotFoundError for an unknown snapshot
        self.snapshot_id = snapshot_id
        self.nodes: dict[str, GraphNode] = {n.id: n for n in registry.find(GraphNode, snapshot_id=snapshot_id)}  # fmt: skip
        self.edges: dict[str, GraphEdge] = {e.id: e for e in registry.find(GraphEdge, snapshot_id=snapshot_id)}  # fmt: skip
        out: dict[str, list[str]] = {n: [] for n in self.nodes}
        inc: dict[str, list[str]] = {n: [] for n in self.nodes}
        for e in sorted(self.edges.values(), key=lambda e: e.id):
            out.setdefault(e.from_node_id, []).append(e.id)
            inc.setdefault(e.to_node_id, []).append(e.id)
        self._out, self._in = out, inc

    def node(self, node_id: str) -> GraphNode:
        return self.nodes[node_id]

    def find_by_ref(self, kind: NodeKind, ref_id: str) -> GraphNode | None:
        for n in self.nodes.values():
            if n.node_kind is kind and n.ref_id == ref_id:
                return n
        return None

    def _adjacent(self, node_id: str, direction: TraversalDirection) -> list[tuple[str, str]]:
        """[(edge_id, neighbor_node_id), ...] for `node_id` in `direction`."""
        out: list[tuple[str, str]] = []
        if direction in (TraversalDirection.OUT, TraversalDirection.BOTH):
            for eid in self._out.get(node_id, ()):
                out.append((eid, self.edges[eid].to_node_id))
        if direction in (TraversalDirection.IN, TraversalDirection.BOTH):
            for eid in self._in.get(node_id, ()):
                out.append((eid, self.edges[eid].from_node_id))
        return out

    def _allowed(self, query: GraphQuery, node_id: str, edge: GraphEdge | None) -> bool:
        if query.kinds and self.nodes[node_id].node_kind not in query.kinds:
            return False
        return not (edge is not None and query.relations and edge.relation.value not in query.relations)  # fmt: skip

    def neighbors(self, node_id: str, direction: TraversalDirection = TraversalDirection.BOTH) -> TraversalResult:  # fmt: skip
        start = self.node(node_id)
        edges = [self.edges[eid] for eid, _n in self._adjacent(node_id, direction)]
        neighbor_ids = {n for _eid, n in self._adjacent(node_id, direction)}
        nodes = [start] + [self.nodes[n] for n in sorted(neighbor_ids)]
        return TraversalResult(
            tuple(NodeView.of(n) for n in nodes),
            tuple(EdgeView.of(e) for e in sorted(edges, key=lambda e: e.id)),
            False,
            len(nodes),
        )

    def _bfs(self, start: str, direction: TraversalDirection, query: GraphQuery) -> TraversalResult:  # fmt: skip
        visited: dict[str, int] = {start: 0}
        order = [start]
        used_edges: list[GraphEdge] = []
        q: deque[str] = deque([start])
        truncated = False
        while q:
            cur = q.popleft()
            depth = visited[cur]
            if depth >= query.max_depth:
                continue
            for eid, nxt in self._adjacent(cur, direction):
                edge = self.edges[eid]
                if not self._allowed(query, nxt, edge):
                    continue
                if nxt in visited:
                    continue
                if len(visited) >= query.max_visited:
                    truncated = True
                    break
                visited[nxt] = depth + 1
                order.append(nxt)
                used_edges.append(edge)
                q.append(nxt)
            else:
                continue
            break
        return TraversalResult(
            tuple(NodeView.of(self.nodes[n]) for n in order),
            tuple(EdgeView.of(e) for e in used_edges),
            truncated,
            len(order),
        )

    def ancestors(self, node_id: str, query: GraphQuery = DEFAULT_QUERY) -> TraversalResult:
        """Nodes reached by following edges BACKWARD (things whose OWN field pointed at this
        node -- e.g. an Evidence row that names this Claim). Most edges in this graph point from
        the referencing record to the record it references, so "what supports X" is usually an
        ancestor query, not a descendant one."""
        return self._bfs(node_id, TraversalDirection.IN, query)

    def descendants(self, node_id: str, query: GraphQuery = DEFAULT_QUERY) -> TraversalResult:
        """Nodes reached by following edges FORWARD (what this node's OWN fields point at)."""
        return self._bfs(node_id, TraversalDirection.OUT, query)

    def related(self, node_id: str, query: GraphQuery = DEFAULT_QUERY) -> TraversalResult:
        """Everything reachable in EITHER direction. Use this for "what connects to X" questions
        that should not depend on which side of each edge happened to name the other -- an
        Evidence row points AT its Claim (an ancestor of the Claim) but also points AT the
        run/artifact/analysis it draws on (a descendant of the Evidence); `related` follows both."""
        return self._bfs(node_id, TraversalDirection.BOTH, query)

    def path(
        self,
        from_id: str,
        to_id: str,
        direction: TraversalDirection = TraversalDirection.OUT,
        query: GraphQuery = DEFAULT_QUERY,
    ) -> PathResult:
        if from_id == to_id:
            return PathResult(True, (NodeView.of(self.nodes[from_id]),), (), False)
        parent: dict[str, tuple[str, str]] = {}  # node -> (edge_id, parent_node)
        visited = {from_id}
        q: deque[tuple[str, int]] = deque([(from_id, 0)])
        truncated = False
        while q:
            cur, depth = q.popleft()
            if depth >= query.max_depth:
                truncated = True
                continue
            for eid, nxt in self._adjacent(cur, direction):
                edge = self.edges[eid]
                if not self._allowed(query, nxt, edge) or nxt in visited:
                    continue
                if len(visited) >= query.max_visited:
                    truncated = True
                    break
                visited.add(nxt)
                parent[nxt] = (eid, cur)
                if nxt == to_id:
                    return PathResult(True, *_reconstruct(self, from_id, to_id, parent), truncated)
                q.append((nxt, depth + 1))
            else:
                continue
            break
        return PathResult(False, (), (), truncated)


def _reconstruct(
    idx: SnapshotIndex, from_id: str, to_id: str, parent: dict[str, tuple[str, str]]
) -> tuple[tuple[NodeView, ...], tuple[EdgeView, ...]]:
    node_ids = [to_id]
    edge_ids: list[str] = []
    cur = to_id
    while cur != from_id:
        eid, prev = parent[cur]
        edge_ids.append(eid)
        node_ids.append(prev)
        cur = prev
    node_ids.reverse()
    edge_ids.reverse()
    return (
        tuple(NodeView.of(idx.nodes[n]) for n in node_ids),
        tuple(EdgeView.of(idx.edges[e]) for e in edge_ids),
    )


# --- composed queries, all expressed via the primitives above -----------------------------------


def evidence_for_claim(idx: SnapshotIndex, claim_node_id: str, query: GraphQuery = DEFAULT_QUERY) -> TraversalResult:  # fmt: skip
    """The Evidence row(s) that name this claim (an EVIDENCE_FOR edge INTO the claim), and
    transitively what THEY draw on (an artifact, a run, an analysis, ...). Deliberately does NOT
    follow the claim's own unrelated fields (e.g. `investigation_id`) -- that would pull in every
    other record merely sharing the same investigation, which is not evidence for this claim. A
    claim with no EVIDENCE_FOR edge here is unsupported in this graph -- reported, not hidden."""
    claim = idx.node(claim_node_id)
    incoming = (e for e in idx.edges.values() if e.to_node_id == claim_node_id)
    direct = sorted(
        (e for e in incoming if e.relation is RelationType.EVIDENCE_FOR), key=lambda e: e.id
    )
    nodes: dict[str, NodeView] = {claim_node_id: NodeView.of(claim)}
    edges: dict[str, EdgeView] = {}
    truncated = False
    for e in direct:
        edges[e.id] = EdgeView.of(e)
        nodes.setdefault(e.from_node_id, NodeView.of(idx.node(e.from_node_id)))
        sub = idx.descendants(e.from_node_id, query)
        truncated = truncated or sub.truncated
        for n in sub.nodes:
            nodes.setdefault(n.id, n)
        for se in sub.edges:
            edges.setdefault(se.id, se)
    return TraversalResult(tuple(nodes.values()), tuple(edges.values()), truncated, len(nodes))


def runs_contributing_to_failure_mode(idx: SnapshotIndex, mode_node_id: str, query: GraphQuery = DEFAULT_QUERY) -> TraversalResult:  # fmt: skip
    """Every RUN node reachable by following the failure mode's OWN fields forward (mode ->
    cluster -> signal -> run). Pure `descendants` is correct and safe here: none of these records
    has an outgoing edge back out to "everything else in the investigation" the way a bidirectional
    walk would. An empty result means no run is reachable in this graph -- not that none was
    checked; see `truncated`/`visited_count`."""
    r = idx.descendants(mode_node_id, query)
    return TraversalResult(
        tuple(n for n in r.nodes if n.kind is NodeKind.RUN), r.edges, r.truncated, r.visited_count
    )


def analyses_depending_on_run(idx: SnapshotIndex, run_node_id: str, query: GraphQuery = DEFAULT_QUERY) -> TraversalResult:  # fmt: skip
    """Every record that names this run via `run_id` (an analysis, an observation, an artifact,
    ...) -- they are the run's ANCESTORS in this graph's edge direction (each points AT the run,
    not the other way around)."""
    return idx.ancestors(run_node_id, query)


def artifacts_for_analysis(idx: SnapshotIndex, analysis_node_id: str, query: GraphQuery = DEFAULT_QUERY) -> TraversalResult:  # fmt: skip
    """Every ARTIFACT connected to an analysis. The real path is two hops in DIFFERENT directions
    (analysis -[DERIVED_FROM]-> run, then artifact -[DERIVED_FROM]-> run, i.e. the artifact is an
    ANCESTOR of the run): forward to the run the analysis names, then backward to what names that
    run. An empty result means no artifact is reachable in this graph -- not that none was
    checked."""
    to_run = idx.descendants(analysis_node_id, query)
    nodes: dict[str, NodeView] = {n.id: n for n in to_run.nodes}
    edges: dict[str, EdgeView] = {e.id: e for e in to_run.edges}
    truncated = to_run.truncated
    for n in to_run.nodes:
        if n.kind is not NodeKind.RUN:
            continue
        sub = idx.ancestors(n.id, query)
        truncated = truncated or sub.truncated
        for sn in sub.nodes:
            nodes.setdefault(sn.id, sn)
        for se in sub.edges:
            edges.setdefault(se.id, se)
    artifacts = tuple(n for n in nodes.values() if n.kind is NodeKind.ARTIFACT)
    return TraversalResult(artifacts, tuple(edges.values()), truncated, len(nodes))


def model_to_failure_paths(idx: SnapshotIndex, model_ref_id: str, mode_ref_id: str, query: GraphQuery = DEFAULT_QUERY) -> PathResult:  # fmt: skip
    model = idx.find_by_ref(NodeKind.MODEL, model_ref_id)
    mode = idx.find_by_ref(NodeKind.FAILURE_MODE, mode_ref_id)
    if model is None or mode is None:
        return PathResult(False, (), (), False)
    return idx.path(model.id, mode.id, TraversalDirection.BOTH, query)


def dataset_to_failure_paths(idx: SnapshotIndex, dataset_ref_id: str, mode_ref_id: str, query: GraphQuery = DEFAULT_QUERY) -> PathResult:  # fmt: skip
    dataset = idx.find_by_ref(NodeKind.DATASET, dataset_ref_id)
    mode = idx.find_by_ref(NodeKind.FAILURE_MODE, mode_ref_id)
    if dataset is None or mode is None:
        return PathResult(False, (), (), False)
    return idx.path(dataset.id, mode.id, TraversalDirection.BOTH, query)


__all__ = [
    "EdgeView",
    "NodeView",
    "PathResult",
    "SnapshotIndex",
    "TraversalResult",
    "analyses_depending_on_run",
    "artifacts_for_analysis",
    "dataset_to_failure_paths",
    "evidence_for_claim",
    "model_to_failure_paths",
    "runs_contributing_to_failure_mode",
]

"""Bounded, inspectable traversal over a persisted `GraphSnapshot`. Every traversal call forwards
`max_depth`/`max_visited` straight into `graph.query`'s existing bounded BFS -- this router never
loads a whole snapshot's nodes/edges into one unbounded response."""

from typing import Any

from fastapi import APIRouter

import experionyx.validation as v
from experionyx.api.common import page
from experionyx.api.deps import RegistryDep
from experionyx.errors import NotFoundError, ValidationError
from experionyx.graph.engine import resolve_graph, resolve_snapshot
from experionyx.graph.entities import GraphNode, GraphSnapshot
from experionyx.graph.query import SnapshotIndex
from experionyx.graph.spec import DEFAULT_MAX_TRAVERSAL_DEPTH, DEFAULT_MAX_VISITED, GraphQuery
from experionyx.graph.taxonomy import TraversalDirection

router = APIRouter(prefix="/api/graph", tags=["graph"])

_MAX_QUERY_VISITED = 2_000  # API-side hard cap, independent of a caller-supplied max_visited


def _bounds(max_depth: int | None, max_visited: int | None) -> GraphQuery:
    depth = DEFAULT_MAX_TRAVERSAL_DEPTH if max_depth is None else max(0, min(max_depth, 50))
    visited = (
        DEFAULT_MAX_VISITED if max_visited is None else max(1, min(max_visited, _MAX_QUERY_VISITED))
    )
    return GraphQuery(max_depth=depth, max_visited=visited)


def _direction(value: str) -> TraversalDirection:
    try:
        return TraversalDirection(value)
    except ValueError as exc:
        raise ValidationError(f"direction: {value!r} is not a valid TraversalDirection") from exc


def _require_node(idx: SnapshotIndex, node_id: str) -> None:
    v.ref("node_id", node_id, GraphNode.PREFIX)
    if node_id not in idx.nodes:
        raise NotFoundError(f"no node {node_id!r} in snapshot {idx.snapshot_id}")


@router.get("/snapshots")
def list_snapshots(
    registry: RegistryDep,
    investigation: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> dict[str, object]:
    filters = {"investigation_id": investigation} if investigation else {}
    return page(registry.find(GraphSnapshot, **filters), limit, offset)


@router.get("/snapshots/{ref}")
def get_snapshot(ref: str, registry: RegistryDep) -> dict[str, object]:
    snapshot = resolve_snapshot(registry, ref)
    return {"id": snapshot.id, **snapshot.to_dict()}


@router.get("/snapshots/{ref}/nodes/{node_id}")
def get_node(ref: str, node_id: str, registry: RegistryDep) -> dict[str, object]:
    snapshot = resolve_snapshot(registry, ref)
    idx = SnapshotIndex(registry, snapshot.id)
    _require_node(idx, node_id)
    return idx.node(node_id).to_dict()


@router.get("/snapshots/{ref}/nodes/{node_id}/neighbors")
def neighbors(
    ref: str,
    node_id: str,
    registry: RegistryDep,
    direction: str = "BOTH",
) -> dict[str, Any]:
    snapshot = resolve_snapshot(registry, ref)
    idx = SnapshotIndex(registry, snapshot.id)
    _require_node(idx, node_id)
    return idx.neighbors(node_id, _direction(direction)).to_dict()


@router.get("/snapshots/{ref}/nodes/{node_id}/ancestors")
def ancestors(
    ref: str,
    node_id: str,
    registry: RegistryDep,
    max_depth: int | None = None,
    max_visited: int | None = None,
) -> dict[str, Any]:
    snapshot = resolve_snapshot(registry, ref)
    idx = SnapshotIndex(registry, snapshot.id)
    _require_node(idx, node_id)
    return idx.ancestors(node_id, _bounds(max_depth, max_visited)).to_dict()


@router.get("/snapshots/{ref}/nodes/{node_id}/descendants")
def descendants(
    ref: str,
    node_id: str,
    registry: RegistryDep,
    max_depth: int | None = None,
    max_visited: int | None = None,
) -> dict[str, Any]:
    snapshot = resolve_snapshot(registry, ref)
    idx = SnapshotIndex(registry, snapshot.id)
    _require_node(idx, node_id)
    return idx.descendants(node_id, _bounds(max_depth, max_visited)).to_dict()


@router.get("/snapshots/{ref}/nodes/{node_id}/related")
def related(
    ref: str,
    node_id: str,
    registry: RegistryDep,
    max_depth: int | None = None,
    max_visited: int | None = None,
) -> dict[str, Any]:
    snapshot = resolve_snapshot(registry, ref)
    idx = SnapshotIndex(registry, snapshot.id)
    _require_node(idx, node_id)
    return idx.related(node_id, _bounds(max_depth, max_visited)).to_dict()


@router.get("/snapshots/{ref}/path")
def path(
    ref: str,
    from_id: str,
    to_id: str,
    registry: RegistryDep,
    direction: str = "OUT",
    max_depth: int | None = None,
    max_visited: int | None = None,
) -> dict[str, Any]:
    snapshot = resolve_snapshot(registry, ref)
    idx = SnapshotIndex(registry, snapshot.id)
    _require_node(idx, from_id)
    _require_node(idx, to_id)
    bounds = _bounds(max_depth, max_visited)
    return idx.path(from_id, to_id, _direction(direction), bounds).to_dict()


@router.get("/graphs/{ref}")
def get_graph(ref: str, registry: RegistryDep) -> dict[str, object]:
    graph = resolve_graph(registry, ref)
    return {"id": graph.id, **graph.to_dict()}


__all__ = ["router"]

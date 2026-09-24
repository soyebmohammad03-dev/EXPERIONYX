"""API tests for bounded graph traversal over the real snapshot built in `api_world`."""

from api_client import client
from api_world import ApiWorld
from experionyx.graph.entities import GraphNode
from experionyx.sqlite import SqliteRegistry


def _investigation_node_id(api_world: ApiWorld) -> str:
    c = client(api_world)
    snapshot = c.get(f"/api/graph/snapshots/{api_world.graph_snapshot_id}").json()
    assert snapshot["id"] == api_world.graph_snapshot_id
    with SqliteRegistry(api_world.ws / "registry.sqlite") as reg:
        nodes = reg.find(GraphNode, snapshot_id=api_world.graph_snapshot_id)
        inv_nodes = [n for n in nodes if n.ref_id == api_world.investigation_id]
        assert inv_nodes, "the investigation must be a node in its own graph snapshot"
        return inv_nodes[0].id


def test_get_snapshot(api_world: ApiWorld) -> None:
    res = client(api_world).get(f"/api/graph/snapshots/{api_world.graph_snapshot_id}")
    assert res.status_code == 200
    body = res.json()
    assert body["investigation_id"] == api_world.investigation_id
    assert body["node_count"] > 0


def test_snapshot_not_found(api_world: ApiWorld) -> None:
    res = client(api_world).get(f"/api/graph/snapshots/gsn_{'0' * 32}")
    assert res.status_code == 404


def test_node_get_and_bad_node_is_404(api_world: ApiWorld) -> None:
    node_id = _investigation_node_id(api_world)
    c = client(api_world)
    res = c.get(f"/api/graph/snapshots/{api_world.graph_snapshot_id}/nodes/{node_id}")
    assert res.status_code == 200
    assert res.json()["ref_id"] == api_world.investigation_id

    missing = c.get(f"/api/graph/snapshots/{api_world.graph_snapshot_id}/nodes/gnd_{'0' * 32}")
    assert missing.status_code == 404


def test_related_traversal_is_bounded_and_reports_truncation(api_world: ApiWorld) -> None:
    node_id = _investigation_node_id(api_world)
    c = client(api_world)
    res = c.get(
        f"/api/graph/snapshots/{api_world.graph_snapshot_id}/nodes/{node_id}/related",
        params={"max_depth": 1, "max_visited": 2},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["visited_count"] <= 2
    assert "truncated" in body


def test_ancestors_max_visited_hard_cap_is_enforced(api_world: ApiWorld) -> None:
    node_id = _investigation_node_id(api_world)
    c = client(api_world)
    res = c.get(
        f"/api/graph/snapshots/{api_world.graph_snapshot_id}/nodes/{node_id}/ancestors",
        params={"max_visited": 10_000_000},
    )
    assert res.status_code == 200
    # the API-side hard cap (2000) must win over an absurd caller-supplied value
    assert res.json()["visited_count"] <= 2001

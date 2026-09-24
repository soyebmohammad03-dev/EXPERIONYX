"""API tests for drift and statistics -- no significance is implied beyond the stored result."""

from api_client import client
from api_world import ApiWorld


def test_drift_analysis_get(api_world: ApiWorld) -> None:
    res = client(api_world).get(f"/api/drift/analyses/{api_world.drift_analysis_id}")
    assert res.status_code == 200
    assert res.json()["id"] == api_world.drift_analysis_id


def test_drift_analysis_not_found(api_world: ApiWorld) -> None:
    res = client(api_world).get(f"/api/drift/analyses/dan_{'0' * 32}")
    assert res.status_code == 404


def test_stats_analysis_get_returns_stored_result_verbatim(api_world: ApiWorld) -> None:
    res = client(api_world).get(f"/api/stats/analyses/{api_world.stats_analysis_id}")
    assert res.status_code == 200
    body = res.json()
    assert body["id"] == api_world.stats_analysis_id
    assert "result" in body
    assert "config" in body
    assert "sources" in body


def test_stats_analysis_list_filters_by_kind(api_world: ApiWorld) -> None:
    res = client(api_world).get("/api/stats/analyses", params={"analysis_kind": "COMPARE"})
    assert res.status_code == 200
    assert all(a["analysis_kind"] == "COMPARE" for a in res.json()["items"])


def test_viz_drift_trajectory_is_deterministic(api_world: ApiWorld) -> None:
    c = client(api_world)
    url = f"/api/viz/drift/{api_world.drift_analysis_id}/trajectory"
    first, second = c.get(url).content, c.get(url).content
    assert first == second


def test_viz_stats_effect_never_fabricates_significance(api_world: ApiWorld) -> None:
    res = client(api_world).get(f"/api/viz/stats/{api_world.stats_analysis_id}/effect")
    assert res.status_code == 200
    body = res.json()
    assert body["points"] != "unavailable"
    assert "uncertainty" in body

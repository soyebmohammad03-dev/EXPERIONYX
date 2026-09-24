"""API tests for the investigation-overview endpoints, using the shared real `api_world`."""

from api_client import client
from api_world import ApiWorld


def test_list_investigations(api_world: ApiWorld) -> None:
    res = client(api_world).get("/api/investigations")
    assert res.status_code == 200
    page = res.json()
    assert page["total"] >= 1
    assert any(i["id"] == api_world.investigation_id for i in page["items"])


def test_get_investigation_includes_experiments_reports_dossiers(api_world: ApiWorld) -> None:
    res = client(api_world).get(f"/api/investigations/{api_world.investigation_id}")
    assert res.status_code == 200
    body = res.json()
    assert body["id"] == api_world.investigation_id
    assert api_world.report_id in body["report_ids"]
    assert api_world.dossier_id in body["dossier_ids"]
    assert len(body["experiment_ids"]) >= 1


def test_get_investigation_unknown_id_is_404(api_world: ApiWorld) -> None:
    res = client(api_world).get(f"/api/investigations/inv_{'0' * 32}")
    assert res.status_code == 404
    assert res.json()["error"] == "NotFoundError"


def test_get_investigation_malformed_id_is_422(api_world: ApiWorld) -> None:
    res = client(api_world).get("/api/investigations/not-an-id")
    assert res.status_code == 422
    assert res.json()["error"] == "ValidationError"


def test_get_run_includes_outcome_and_provenance(api_world: ApiWorld) -> None:
    res = client(api_world).get(f"/api/investigations/runs/{api_world.baseline_run_id}")
    assert res.status_code == 200
    body = res.json()
    assert body["id"] == api_world.baseline_run_id
    assert isinstance(body["outcome"], dict)
    assert isinstance(body["provenance"], dict)


def test_pagination_bounds_are_enforced(api_world: ApiWorld) -> None:
    res = client(api_world).get("/api/investigations", params={"limit": 10_000, "offset": -5})
    assert res.status_code == 200
    page = res.json()
    assert page["limit"] <= 500
    assert page["offset"] == 0

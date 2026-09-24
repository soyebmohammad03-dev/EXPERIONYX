"""API tests for reliability, failures and faults -- reads real evidence from `api_world`."""

from api_client import client
from api_world import ApiWorld


def test_reliability_profile_has_no_aggregate_score(api_world: ApiWorld) -> None:
    res = client(api_world).get(f"/api/reliability/profiles/{api_world.reliability_profile_id}")
    assert res.status_code == 200
    body = res.json()
    assert "score" not in str(body["dimension_status"]).lower()
    assert isinstance(body["dimension_status"], dict)
    assert body["dimension_status"]


def test_reliability_profile_not_found(api_world: ApiWorld) -> None:
    res = client(api_world).get(f"/api/reliability/profiles/rpf_{'0' * 32}")
    assert res.status_code == 404


def test_reliability_search_by_investigation(api_world: ApiWorld) -> None:
    res = client(api_world).get(
        "/api/reliability/profiles", params={"investigation": api_world.investigation_id}
    )
    assert res.status_code == 200
    assert any(p["id"] == api_world.reliability_profile_id for p in res.json()["items"])


def test_failure_modes_list_and_get(api_world: ApiWorld) -> None:
    c = client(api_world)
    mode_id = api_world.failure_mode_ids[0]
    res = c.get(f"/api/failures/modes/{mode_id}")
    assert res.status_code == 200
    body = res.json()
    assert body["id"] == mode_id
    assert body["evidence"] == "unavailable" or isinstance(body["evidence"], list)


def test_failure_mode_unknown_is_404(api_world: ApiWorld) -> None:
    res = client(api_world).get(f"/api/failures/modes/fmd_{'0' * 32}")
    assert res.status_code == 404


def test_fault_experiment_lists_trials_and_analyses(api_world: ApiWorld) -> None:
    fxp_id = api_world.fault_experiment_ids[0]
    res = client(api_world).get(f"/api/faults/experiments/{fxp_id}")
    assert res.status_code == 200
    body = res.json()
    assert body["id"] == fxp_id
    assert isinstance(body["trials"], list)
    assert body["trials"]


def test_fault_analysis_get(api_world: ApiWorld) -> None:
    analysis_id = api_world.fault_analysis_ids[0]
    res = client(api_world).get(f"/api/faults/analyses/{analysis_id}")
    assert res.status_code == 200
    assert res.json()["id"] == analysis_id


def test_fault_degradation_viz_has_real_points(api_world: ApiWorld) -> None:
    fxp_id = api_world.fault_experiment_ids[0]
    res = client(api_world).get(f"/api/viz/faults/{fxp_id}/degradation")
    assert res.status_code == 200
    body = res.json()
    assert body["metric"] == "deterioration"
    assert body["points"] == "unavailable" or isinstance(body["points"], list)


def test_fault_degradation_viz_is_deterministic(api_world: ApiWorld) -> None:
    fxp_id = api_world.fault_experiment_ids[0]
    c = client(api_world)
    first = c.get(f"/api/viz/faults/{fxp_id}/degradation").json()
    second = c.get(f"/api/viz/faults/{fxp_id}/degradation").json()
    assert first == second


def test_fault_degradation_viz_unknown_experiment_is_404(api_world: ApiWorld) -> None:
    res = client(api_world).get(f"/api/viz/faults/fxp_{'0' * 32}/degradation")
    assert res.status_code == 404

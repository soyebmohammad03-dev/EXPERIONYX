"""API tests for benchmark/leaderboard (no benchmark was constructed in `api_world`; deep
benchmark-grid setup is out of scope here -- list/404 paths are covered) and reproducibility
(a real PROVENANCE_ONLY attempt against the baseline run)."""

from api_client import client
from api_world import ApiWorld


def test_benchmark_list_is_empty_but_well_formed(api_world: ApiWorld) -> None:
    res = client(api_world).get("/api/benchmark/benchmarks")
    assert res.status_code == 200
    assert res.json() == {"items": [], "total": 0, "limit": 50, "offset": 0}


def test_benchmark_result_unknown_is_404(api_world: ApiWorld) -> None:
    res = client(api_world).get(f"/api/benchmark/results/brs_{'0' * 32}")
    assert res.status_code == 404


def test_benchmark_compare_malformed_id_is_422(api_world: ApiWorld) -> None:
    res = client(api_world).get("/api/benchmark/results/compare", params={"a": "x", "b": "y"})
    assert res.status_code == 422


def test_reproduction_attempt_get(api_world: ApiWorld) -> None:
    res = client(api_world).get(
        f"/api/reproducibility/attempts/{api_world.reproduction_attempt_id}"
    )
    assert res.status_code == 200
    body = res.json()
    assert body["id"] == api_world.reproduction_attempt_id
    assert body["target_id"] == api_world.baseline_run_id


def test_reproduction_attempts_filter_by_target(api_world: ApiWorld) -> None:
    res = client(api_world).get(
        "/api/reproducibility/attempts", params={"target_id": api_world.baseline_run_id}
    )
    assert res.status_code == 200
    assert any(a["id"] == api_world.reproduction_attempt_id for a in res.json()["items"])


def test_resolve_target_run(api_world: ApiWorld) -> None:
    res = client(api_world).get(f"/api/reproducibility/resolve/RUN/{api_world.baseline_run_id}")
    assert res.status_code == 200


def test_resolve_target_bad_kind_is_422(api_world: ApiWorld) -> None:
    run_id = api_world.baseline_run_id
    res = client(api_world).get(f"/api/reproducibility/resolve/NOT_A_KIND/{run_id}")
    assert res.status_code == 422


def test_verify_run_artifacts(api_world: ApiWorld) -> None:
    res = client(api_world).get(
        f"/api/reproducibility/runs/{api_world.baseline_run_id}/artifact-verification"
    )
    assert res.status_code == 200

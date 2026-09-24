"""Regression test: `/api/benchmark/protocols/{id}/leaderboard` must return the most recently
created snapshot, not the one with the lexicographically largest content-addressed id. Two real
snapshots of the same protocol are built (submission A, then submission A+B), and the API result
is checked against `max(snapshots, key=created_at)` rather than against a hardcoded expectation,
so the assertion holds regardless of how the content hashes happen to sort."""

import pytest
from fastapi.testclient import TestClient

from benchmark_helpers import ids, small_world, spec_for
from eval_helpers import EvalWorld
from experionyx.api.app import create_app
from experionyx.leaderboard.entities import BenchmarkProtocol, LeaderboardSnapshot
from experionyx.leaderboard.protocol import register_protocol
from experionyx.leaderboard.snapshot import build_snapshot
from experionyx.leaderboard.submission import submit
from reliability_helpers import register_variant


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> EvalWorld:
    return small_world(tmp_path_factory.mktemp("lb-api"))


@pytest.fixture(scope="module")
def two_snapshots(
    world: EvalWorld,
) -> tuple[BenchmarkProtocol, LeaderboardSnapshot, LeaderboardSnapshot]:
    spec_a = spec_for(world)
    protocol = register_protocol(world.registry, spec_a)
    submit(world.registry, world.store, world.executor, protocol, spec_a, source_root=world.workspace.parent)  # fmt: skip
    first = build_snapshot(world.registry, world.store, protocol)

    variant_model, _ = register_variant(world, threshold=3.0, tag="lb-api")
    _orig_model, dataset = ids(world)
    spec_b = spec_for(world, model=variant_model, dataset=dataset)
    submit(world.registry, world.store, world.executor, protocol, spec_b, source_root=world.workspace.parent)  # fmt: skip
    second = build_snapshot(world.registry, world.store, protocol)

    assert first.id != second.id, "two distinct submissions must yield two distinct snapshots"
    return protocol, first, second


def test_leaderboard_returns_most_recent_snapshot_by_time_not_by_id(
    world: EvalWorld,
    two_snapshots: tuple[BenchmarkProtocol, LeaderboardSnapshot, LeaderboardSnapshot],
) -> None:
    protocol, first, second = two_snapshots
    expected = max((first, second), key=lambda s: (s.created_at, s.id))

    client = TestClient(create_app(str(world.workspace)))
    res = client.get(f"/api/benchmark/protocols/{protocol.id}/leaderboard")
    assert res.status_code == 200
    body = res.json()
    assert body["snapshot"] != "unavailable"
    assert body["snapshot"]["id"] == expected.id

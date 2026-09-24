"""Real leaderboard workflow: register a protocol, submit real benchmark evidence for two models,
build an immutable snapshot, compare submissions, verify artifact integrity and graph
traceability. Engineering validation of the machinery, not a scientific finding about which model
is "better" -- comparisons stay protocol-specific and metric-specific throughout."""

from collections.abc import Mapping

import pytest

from benchmark_helpers import ids, small_world, spec_for
from eval_helpers import EvalWorld
from experionyx.benchmark.entities import BenchmarkResult
from experionyx.benchmark.registry import BenchmarkRegistry
from experionyx.errors import ValidationError
from experionyx.graph.build import construct
from experionyx.graph.spec import GraphSpec
from experionyx.graph.taxonomy import NodeKind, RelationType
from experionyx.leaderboard.compare import compare_protocol_constrained, compare_submissions
from experionyx.leaderboard.entities import (
    BenchmarkProtocol,
    BenchmarkSubmission,
    LeaderboardEntry,
)
from experionyx.leaderboard.protocol import protocol_hash_of, register_protocol
from experionyx.leaderboard.snapshot import build_snapshot
from experionyx.leaderboard.submission import submit
from experionyx.reproducibility.engine import verify_run_artifacts
from reliability_helpers import register_variant


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> EvalWorld:
    return small_world(tmp_path_factory.mktemp("lb"))


@pytest.fixture(scope="module")
def two_submissions(world: EvalWorld) -> tuple[BenchmarkProtocol, BenchmarkSubmission, BenchmarkSubmission]:  # fmt: skip
    """Real engineering validation: A. a real protocol over a real model+dataset. B. real
    benchmark units/results (Phase 9). C. two protocol-compatible submissions (two models)."""
    spec_a = spec_for(world)
    protocol = register_protocol(world.registry, spec_a)
    result_a = submit(world.registry, world.store, world.executor, protocol, spec_a, source_root=world.workspace.parent)  # fmt: skip
    assert result_a.submission_id is not None, result_a

    variant_model, _variant_dataset = register_variant(world, threshold=3.0, tag="lb")
    _orig_model, dataset = ids(world)
    spec_b = spec_for(world, model=variant_model, dataset=dataset)
    assert protocol_hash_of(world.registry, spec_b) == protocol.protocol_hash  # still the same protocol  # fmt: skip
    result_b = submit(world.registry, world.store, world.executor, protocol, spec_b, source_root=world.workspace.parent)  # fmt: skip
    assert result_b.submission_id is not None, result_b

    sub_a = world.registry.get(BenchmarkSubmission, result_a.submission_id)
    sub_b = world.registry.get(BenchmarkSubmission, result_b.submission_id)
    return protocol, sub_a, sub_b


# --- protocol and submission ---------------------------------------------------------------


def test_protocol_registration_is_idempotent(world: EvalWorld) -> None:
    spec = spec_for(world)
    a = register_protocol(world.registry, spec)
    b = register_protocol(world.registry, spec)
    assert a.id == b.id


def test_protocol_hash_is_model_independent(world: EvalWorld) -> None:
    model_a, dataset = ids(world)
    model_b, _ = register_variant(world, threshold=9.0, tag="hashcheck")
    a = protocol_hash_of(world.registry, spec_for(world, model=model_a, dataset=dataset))
    b = protocol_hash_of(world.registry, spec_for(world, model=model_b, dataset=dataset))
    assert a == b


def test_protocol_hash_changes_with_seeds(world: EvalWorld) -> None:
    a = protocol_hash_of(world.registry, spec_for(world, seeds=(1, 2, 3)))
    b = protocol_hash_of(world.registry, spec_for(world, seeds=(1, 2, 3, 4)))
    assert a != b


def test_submitting_the_same_evidence_twice_is_idempotent(world: EvalWorld, two_submissions: tuple[BenchmarkProtocol, BenchmarkSubmission, BenchmarkSubmission]) -> None:  # fmt: skip
    protocol, sub_a, _sub_b = two_submissions
    spec = spec_for(world)
    again = submit(world.registry, world.store, world.executor, protocol, spec, source_root=world.workspace.parent)  # fmt: skip
    assert again.already_submitted is True
    assert again.submission_id == sub_a.id


def test_a_spec_for_a_different_protocol_is_refused(world: EvalWorld, two_submissions: tuple[BenchmarkProtocol, BenchmarkSubmission, BenchmarkSubmission]) -> None:  # fmt: skip
    protocol, _sub_a, _sub_b = two_submissions
    other = spec_for(world, name="different-protocol", seeds=(0, 1, 2, 3))
    with pytest.raises(ValidationError):
        submit(world.registry, world.store, world.executor, protocol, other, source_root=world.workspace.parent)  # fmt: skip


# --- snapshot ------------------------------------------------------------------------------


def test_snapshot_includes_both_submissions_and_is_idempotent(world: EvalWorld, two_submissions: tuple[BenchmarkProtocol, BenchmarkSubmission, BenchmarkSubmission]) -> None:  # fmt: skip
    protocol, sub_a, sub_b = two_submissions
    snap = build_snapshot(world.registry, world.store, protocol)
    assert set(snap.submission_ids) == {sub_a.id, sub_b.id}
    entries = world.registry.find(LeaderboardEntry, snapshot_id=snap.id)
    assert {e.submission_id for e in entries} == {sub_a.id, sub_b.id}

    again = build_snapshot(world.registry, world.store, protocol)
    assert again.id == snap.id  # unchanged evidence -> the same snapshot, not a duplicate


def test_snapshot_entries_carry_raw_metric_values_no_composite_score(world: EvalWorld, two_submissions: tuple[BenchmarkProtocol, BenchmarkSubmission, BenchmarkSubmission]) -> None:  # fmt: skip
    protocol, sub_a, _sub_b = two_submissions
    snap = build_snapshot(world.registry, world.store, protocol)
    entry = next(e for e in world.registry.find(LeaderboardEntry, snapshot_id=snap.id) if e.submission_id == sub_a.id)  # fmt: skip
    assert entry.metrics  # at least one real metric was carried through
    for m in entry.metrics.values():
        assert isinstance(m, Mapping) and "value" in m and "higher_is_better" in m
    assert "score" not in entry.metrics and "overall" not in entry.metrics


def test_snapshot_replay_confirms_deterministic_identity(world: EvalWorld, two_submissions: tuple[BenchmarkProtocol, BenchmarkSubmission, BenchmarkSubmission]) -> None:  # fmt: skip
    protocol, _sub_a, _sub_b = two_submissions
    snap = build_snapshot(world.registry, world.store, protocol)
    rebuilt = build_snapshot(world.registry, world.store, protocol)
    assert rebuilt.id == snap.id


# --- comparison (protocol-constrained, no winner) -------------------------------------------


def test_compare_two_submissions_reports_raw_differences(world: EvalWorld, two_submissions: tuple[BenchmarkProtocol, BenchmarkSubmission, BenchmarkSubmission]) -> None:  # fmt: skip
    _protocol, sub_a, sub_b = two_submissions
    compare_protocol_constrained(world.registry, sub_a.id, sub_b.id)  # does not raise: same protocol  # fmt: skip
    out = compare_submissions(world.registry, world.store, sub_a.id, sub_b.id)
    assert "winner" not in out and "best" not in out and "rank" not in out
    assert out["baseline"]["metrics"]  # at least one real metric compared


def test_compare_refuses_submissions_of_different_protocols(world: EvalWorld, two_submissions: tuple[BenchmarkProtocol, BenchmarkSubmission, BenchmarkSubmission]) -> None:  # fmt: skip
    _protocol, sub_a, _sub_b = two_submissions
    other_spec = spec_for(world, name="a-different-protocol", seeds=(5, 6))
    other_protocol = register_protocol(world.registry, other_spec)
    other_result = submit(world.registry, world.store, world.executor, other_protocol, other_spec, source_root=world.workspace.parent)  # fmt: skip
    assert other_result.submission_id is not None
    with pytest.raises(ValidationError):
        compare_protocol_constrained(world.registry, sub_a.id, other_result.submission_id)


# --- artifact verification and graph traceability -------------------------------------------


def test_verify_a_submissions_evidence_reports_no_corruption(world: EvalWorld, two_submissions: tuple[BenchmarkProtocol, BenchmarkSubmission, BenchmarkSubmission]) -> None:  # fmt: skip
    _protocol, sub_a, _sub_b = two_submissions
    result = world.registry.get(BenchmarkResult, sub_a.result_id)
    out = verify_run_artifacts(world.registry, world.store, result.run_id)
    assert out["corrupted"] == []


def test_leaderboard_result_traces_through_the_graph_to_concrete_runs(world: EvalWorld, two_submissions: tuple[BenchmarkProtocol, BenchmarkSubmission, BenchmarkSubmission]) -> None:  # fmt: skip
    """E. at least one leaderboard result traced through the evidence graph back to concrete
    runs/artifacts -- engineering validation, not a scientific claim."""
    protocol, sub_a, _sub_b = two_submissions
    build_snapshot(world.registry, world.store, protocol)
    got = construct(world.registry, GraphSpec("g", "1.0.0"))
    submission_node = (NodeKind.BENCHMARK_RESULT, sub_a.result_id)
    assert submission_node in got.nodes  # the submission's real BenchmarkResult is a real node
    result = BenchmarkRegistry(world.registry, world.store).result(sub_a.result_id)
    run_edges = [
        e
        for e in got.edges
        if e.from_key == submission_node and e.to_key == (NodeKind.RUN, result.run_id)
    ]
    assert run_edges  # the result traces to its own real collect Run
    assert any(e.relation is RelationType.DERIVED_FROM for e in run_edges)

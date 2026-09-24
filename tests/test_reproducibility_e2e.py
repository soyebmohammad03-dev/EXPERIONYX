"""Real reproduction attempts against a real registry: dispatch to an existing engine's own
`replay_check`, classification per mode, environment/artifact checks, attempt numbering, and
provenance-only short-circuiting -- driven through `run_reproduction` directly (see
docs/reproducibility.md)."""

from conftest import Lab
from experionyx.errors import NotFoundError, ValidationError
from experionyx.graph.engine import run_graph
from experionyx.graph.spec import GraphSpec
from experionyx.reproducibility.engine import run_reproduction
from experionyx.reproducibility.entities import ReproductionAttempt
from experionyx.reproducibility.spec import ReproductionSpec
from experionyx.reproducibility.taxonomy import ComparisonOutcome, ReproductionMode, TargetKind


def _build_graph(lab: Lab) -> tuple[str, str]:
    """Returns (snapshot_id, collect_run_id)."""
    result = run_graph(lab.registry, lab.store, lab.executor, GraphSpec("g", "1.0.0"))
    assert result.snapshot_id is not None and result.run_id is not None
    return result.snapshot_id, result.run_id


def test_deterministic_mode_on_an_unchanged_graph_snapshot_is_equal(lab: Lab) -> None:
    snapshot_id, _run_id = _build_graph(lab)
    spec = ReproductionSpec(TargetKind.GRAPH_SNAPSHOT, snapshot_id, ReproductionMode.DETERMINISTIC)
    result = run_reproduction(lab.registry, lab.store, lab.executor, spec)
    attempt = lab.registry.get(ReproductionAttempt, result.attempt_id)
    assert result.outcome is ComparisonOutcome.EQUAL
    assert attempt.target_kind is TargetKind.GRAPH_SNAPSHOT
    assert attempt.attempt == 0
    assert attempt.replay_run_id is not None
    assert attempt.sources_changed is False


def test_exact_mode_on_an_unchanged_graph_snapshot_is_equal(lab: Lab) -> None:
    snapshot_id, _ = _build_graph(lab)
    spec = ReproductionSpec(TargetKind.GRAPH_SNAPSHOT, snapshot_id, ReproductionMode.EXACT)
    result = run_reproduction(lab.registry, lab.store, lab.executor, spec)
    assert result.outcome is ComparisonOutcome.EQUAL


def test_repeated_attempts_at_the_same_target_number_sequentially_and_never_overwrite(lab: Lab) -> None:  # fmt: skip
    snapshot_id, _ = _build_graph(lab)
    spec = ReproductionSpec(TargetKind.GRAPH_SNAPSHOT, snapshot_id, ReproductionMode.DETERMINISTIC)
    first = run_reproduction(lab.registry, lab.store, lab.executor, spec)
    second = run_reproduction(lab.registry, lab.store, lab.executor, spec)
    a = lab.registry.get(ReproductionAttempt, first.attempt_id)
    b = lab.registry.get(ReproductionAttempt, second.attempt_id)
    assert a.attempt == 0 and b.attempt == 1
    assert a.id != b.id
    both = lab.registry.find(ReproductionAttempt, target_id=snapshot_id)
    assert {x.id for x in both} == {a.id, b.id}  # the first attempt is still there, unmutated


def test_provenance_only_never_re_executes_and_checks_artifact_integrity(lab: Lab) -> None:
    snapshot_id, _run_id = _build_graph(lab)
    from experionyx.domain import Run

    before = len(lab.registry.find(Run))
    spec = ReproductionSpec(
        TargetKind.GRAPH_SNAPSHOT, snapshot_id, ReproductionMode.PROVENANCE_ONLY
    )
    result = run_reproduction(lab.registry, lab.store, lab.executor, spec)
    after = len(lab.registry.find(Run))
    assert after == before  # no replay run was created
    attempt = lab.registry.get(ReproductionAttempt, result.attempt_id)
    assert attempt.replay_run_id is None
    assert result.outcome is ComparisonOutcome.EQUAL
    assert attempt.artifact_verification["corrupted"] == ()
    assert attempt.note is not None and "not re-executed" in attempt.note


def test_run_target_kind_reproduces_the_underlying_collect_run(lab: Lab) -> None:
    _snapshot_id, run_id = _build_graph(lab)
    spec = ReproductionSpec(TargetKind.RUN, run_id, ReproductionMode.EXACT)
    result = run_reproduction(lab.registry, lab.store, lab.executor, spec)
    assert result.outcome is ComparisonOutcome.EQUAL
    attempt = lab.registry.get(ReproductionAttempt, result.attempt_id)
    assert attempt.run_id == run_id


def test_numeric_tolerance_mode_on_an_unchanged_target_is_equal(lab: Lab) -> None:
    snapshot_id, _ = _build_graph(lab)
    spec = ReproductionSpec(TargetKind.GRAPH_SNAPSHOT, snapshot_id, ReproductionMode.NUMERIC_TOLERANCE)  # fmt: skip
    result = run_reproduction(lab.registry, lab.store, lab.executor, spec)
    assert result.outcome is ComparisonOutcome.EQUAL


def test_environment_diff_reports_identical_environment_for_replay_on_the_same_machine(lab: Lab) -> None:  # fmt: skip
    snapshot_id, _ = _build_graph(lab)
    spec = ReproductionSpec(TargetKind.GRAPH_SNAPSHOT, snapshot_id, ReproductionMode.DETERMINISTIC)
    result = run_reproduction(lab.registry, lab.store, lab.executor, spec)
    attempt = lab.registry.get(ReproductionAttempt, result.attempt_id)
    assert attempt.environment_diff["comparable"] is True
    assert attempt.environment_diff["changed"] == {}


def test_unknown_target_raises_not_found(lab: Lab) -> None:
    spec = ReproductionSpec(TargetKind.GRAPH_SNAPSHOT, "gsn_" + "9" * 32, ReproductionMode.EXACT)
    import pytest

    with pytest.raises(NotFoundError):
        run_reproduction(lab.registry, lab.store, lab.executor, spec)


def test_schedule_target_without_investigation_is_refused(lab: Lab) -> None:
    import pytest

    spec = ReproductionSpec(TargetKind.SCHEDULE, "sch_" + "9" * 32, ReproductionMode.DETERMINISTIC)
    with pytest.raises(ValidationError):
        run_reproduction(lab.registry, lab.store, lab.executor, spec)

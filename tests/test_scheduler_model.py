"""Scheduler domain model: deterministic identity, validation, the DAG expansion, the state
machine and the `$dep:` substitution helper. No registry, no execution."""

from datetime import UTC, datetime

import pytest

from experionyx.errors import SchedulerRefusal, ValidationError
from experionyx.scheduler.dispatch import DispatchError, referenced_dependencies, substitute
from experionyx.scheduler.entities import (
    ExecutionAttempt,
    Schedule,
    ScheduleRun,
    ScheduleUnit,
    UnitStateTransition,
)
from experionyx.scheduler.graph import expand, validation_report
from experionyx.scheduler.spec import (
    ExecutionPolicy,
    ResourceRequirement,
    RetryPolicy,
    ScheduleSpec,
    UnitDef,
)
from experionyx.scheduler.taxonomy import (
    FailureCategory,
    RetryPolicyKind,
    ScheduleRunState,
    UnitKind,
    UnitState,
)

T0 = datetime(2026, 9, 23, tzinfo=UTC)


def stats_unit(key: str, value: float, *deps: str) -> UnitDef:
    return UnitDef(
        key,
        UnitKind.STATISTICAL_ANALYSIS,
        {"kind": "BOOTSTRAP", "sources": {"kind": "inline", "reference": [value] * 5}},
        depends_on=tuple(deps),
    )


# --- ScheduleSpec / UnitDef validation ----------------------------------------------------------


def test_spec_id_is_deterministic_and_content_addressed() -> None:
    a = ScheduleSpec("s", "1.0.0", (stats_unit("x", 1.0),))
    b = ScheduleSpec("s", "1.0.0", (stats_unit("x", 1.0),))
    c = ScheduleSpec("s", "1.0.0", (stats_unit("x", 2.0),))  # different parameters
    assert a.spec_id == b.spec_id
    assert a.spec_id != c.spec_id


def test_spec_rejects_duplicate_keys() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        ScheduleSpec("s", "1.0.0", (stats_unit("x", 1.0), stats_unit("x", 2.0)))


def test_spec_rejects_unknown_dependency_reference() -> None:
    with pytest.raises(ValidationError, match="unknown"):
        ScheduleSpec("s", "1.0.0", (stats_unit("a", 1.0, "missing"),))


def test_unit_cannot_depend_on_itself() -> None:
    with pytest.raises(ValidationError, match="itself"):
        UnitDef("x", UnitKind.STATISTICAL_ANALYSIS, depends_on=("x",))


def test_spec_rejects_bad_version_and_empty_units() -> None:
    with pytest.raises(ValidationError):
        ScheduleSpec("s", "1.0", (stats_unit("x", 1.0),))
    with pytest.raises(ValidationError):
        ScheduleSpec("s", "1.0.0", ())


def test_retry_policy_none_forces_a_single_attempt() -> None:
    with pytest.raises(ValidationError):
        RetryPolicy(RetryPolicyKind.NONE, max_attempts=3)


def test_fixed_retry_policy_needs_retryable_categories() -> None:
    with pytest.raises(ValidationError):
        RetryPolicy(RetryPolicyKind.FIXED, max_attempts=3, retryable=())


def test_execution_policy_bounds() -> None:
    with pytest.raises(ValidationError):
        ExecutionPolicy(max_workers=0)
    with pytest.raises(ValidationError):
        ExecutionPolicy(max_workers=65)


def test_resource_requirement_round_trips() -> None:
    r = ResourceRequirement("HIGH", memory_mb=512, exclusive=True, workload_size=10_000)
    assert ResourceRequirement.from_dict(r.to_dict()) == r


# --- graph expansion -----------------------------------------------------------------------------


def test_expand_orders_independent_units_deterministically_by_priority_then_key() -> None:
    params = {"kind": "BOOTSTRAP", "sources": {"kind": "inline", "reference": [1]}}
    spec = ScheduleSpec(
        "s",
        "1.0.0",
        (
            UnitDef("b", UnitKind.STATISTICAL_ANALYSIS, params, priority=0),
            UnitDef("a", UnitKind.STATISTICAL_ANALYSIS, params, priority=5),
            UnitDef("c", UnitKind.STATISTICAL_ANALYSIS, params, priority=0),
        ),
    )
    plan = expand(spec)
    assert [u.key for u in plan.units] == ["a", "b", "c"]  # highest priority first, then key


def test_expand_orders_dependents_after_their_dependencies() -> None:
    spec = ScheduleSpec("s", "1.0.0", (stats_unit("a", 1.0), stats_unit("b", 2.0, "a")))
    plan = expand(spec)
    assert [u.key for u in plan.units].index("a") < [u.key for u in plan.units].index("b")


def test_expand_refuses_a_cycle_and_names_every_stuck_unit() -> None:
    a = UnitDef("a", UnitKind.STATISTICAL_ANALYSIS, {"kind": "BOOTSTRAP", "sources": {"kind": "inline", "reference": [1]}}, depends_on=("b",))  # fmt: skip
    b = UnitDef("b", UnitKind.STATISTICAL_ANALYSIS, {"kind": "BOOTSTRAP", "sources": {"kind": "inline", "reference": [1]}}, depends_on=("a",))  # fmt: skip
    # bypass ScheduleSpec's own eager checks by constructing directly and calling expand
    spec = object.__new__(ScheduleSpec)
    object.__setattr__(spec, "units", (a, b))
    object.__setattr__(spec, "policy", ExecutionPolicy())
    with pytest.raises(SchedulerRefusal) as exc:
        expand(spec)
    issue = exc.value.issues[0]
    assert issue.code == "CYCLE" and "'a'" in issue.found and "'b'" in issue.found  # type: ignore[attr-defined]  # fmt: skip


def test_validation_report_non_raising_form() -> None:
    spec = ScheduleSpec("s", "1.0.0", (stats_unit("a", 1.0),))
    report = validation_report(spec)
    assert report["valid"] is True
    assert report["order"] == ["a"]


def test_graph_hash_changes_with_dependencies() -> None:
    a = ScheduleSpec("s", "1.0.0", (stats_unit("a", 1.0), stats_unit("b", 2.0)))
    b = ScheduleSpec("s", "1.0.0", (stats_unit("a", 1.0), stats_unit("b", 2.0, "a")))
    assert expand(a).graph_hash != expand(b).graph_hash


# --- entity identity and state machine ------------------------------------------------------------


def _unit(**over: object) -> ScheduleUnit:
    base: dict[str, object] = dict(
        schedule_id="sch_" + "1" * 32,
        unit_key="a",
        unit_kind=UnitKind.STATISTICAL_ANALYSIS,
        parameters={"kind": "BOOTSTRAP"},
        depends_on=(),
        retry_policy={"kind": "NONE", "max_attempts": 1, "retryable": []},
        resources=None,
        timeout_seconds=None,
        priority=0,
        created_at=T0,
    )
    base.update(over)
    return ScheduleUnit(**base)  # type: ignore[arg-type]


def test_schedule_unit_identity_changes_with_every_meaningful_input() -> None:
    base = _unit()
    assert _unit().id == base.id  # identical inputs: identical identity
    assert _unit(parameters={"kind": "COMPARE"}).id != base.id
    assert _unit(unit_kind=UnitKind.RESOURCE).id != base.id
    assert _unit(priority=1).id != base.id
    assert _unit(depends_on=("sun_" + "2" * 32,)).id != base.id
    assert _unit(retry_policy={"kind": "FIXED", "max_attempts": 2, "retryable": ["TRANSIENT"]}).id != base.id  # fmt: skip
    assert _unit(timeout_seconds=30.0).id != base.id
    assert _unit(schedule_id="sch_" + "9" * 32).id != base.id


def test_schedule_identity_is_the_spec_id_only() -> None:
    spec = ScheduleSpec("s", "1.0.0", (stats_unit("a", 1.0),))
    s1 = Schedule(spec.name, spec.version, spec.spec_id, spec.to_dict(), "sha256:" + "0" * 64, "1.0.0", T0)  # fmt: skip
    s2 = Schedule(spec.name, spec.version, spec.spec_id, spec.to_dict(), "sha256:" + "0" * 64, "1.0.0", datetime(2026, 1, 1, tzinfo=UTC))  # fmt: skip
    assert s1.id == s2.id  # created_at is not identity


def test_unit_state_machine_allows_only_declared_transitions() -> None:
    u = _unit()
    assert u.with_status(UnitState.READY).status is UnitState.READY
    with pytest.raises(ValidationError, match="illegal"):
        u.with_status(UnitState.SUCCEEDED)  # PLANNED cannot jump straight to SUCCEEDED
    ready = u.with_status(UnitState.READY)
    running = ready.with_status(UnitState.RUNNING)
    done = running.with_status(UnitState.SUCCEEDED)
    with pytest.raises(ValidationError, match="illegal"):
        done.with_status(UnitState.FAILED)  # SUCCEEDED is terminal


def test_schedule_run_state_machine() -> None:
    r = ScheduleRun("sch_" + "1" * 32, "inv_" + "1" * 32, {}, False, 0, T0)
    running = r.with_status(ScheduleRunState.RUNNING)
    with pytest.raises(ValidationError):
        r.with_status(ScheduleRunState.COMPLETED)  # PLANNED cannot jump straight to COMPLETED
    running.with_status(ScheduleRunState.COMPLETED)  # legal from RUNNING


def test_execution_attempt_requires_error_info_unless_succeeded() -> None:
    with pytest.raises(ValidationError):
        ExecutionAttempt("sun_" + "1" * 32, "scr_" + "1" * 32, 0, T0, T0, UnitState.FAILED, None, None, None, None, None)  # fmt: skip
    ExecutionAttempt("sun_" + "1" * 32, "scr_" + "1" * 32, 0, T0, T0, UnitState.FAILED, FailureCategory.CONFIGURATION, "E", "bad", None, None)  # fmt: skip
    ExecutionAttempt("sun_" + "1" * 32, "scr_" + "1" * 32, 0, T0, T0, UnitState.SUCCEEDED, None, None, None, None, None)  # fmt: skip


def test_unit_state_transition_rejects_illegal_pairs() -> None:
    with pytest.raises(ValidationError, match="illegal"):
        UnitStateTransition("sun_" + "1" * 32, "scr_" + "1" * 32, 0, UnitState.SUCCEEDED, UnitState.FAILED, "x", T0)  # fmt: skip


# --- $dep substitution -----------------------------------------------------------------------------


def test_substitute_replaces_dep_markers_recursively() -> None:
    payload = {"a": "$dep:baseline", "b": [1, "$dep:other", {"c": "$dep:baseline"}]}
    out = substitute(payload, {"baseline": "run_1", "other": "run_2"})
    assert out == {"a": "run_1", "b": [1, "run_2", {"c": "run_1"}]}


def test_substitute_raises_on_unresolved_reference() -> None:
    with pytest.raises(DispatchError, match="unresolved"):
        substitute({"a": "$dep:missing"}, {})


def test_referenced_dependencies_finds_every_dep_marker() -> None:
    payload = {"a": "$dep:x", "b": {"c": "$dep:y"}, "d": ["$dep:x", "plain"]}
    assert referenced_dependencies(payload) == {"x", "y"}

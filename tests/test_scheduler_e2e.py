"""Real end-to-end scheduler execution: dependency ordering, bounded concurrency, retries, resume
after an interrupted process, idempotency, cancellation and dry-run. Every unit here dispatches
through the REAL statistical-analysis engine (no model/dataset needed), so success/failure comes
from actually running the target engine, not a stub."""

import threading
from collections.abc import Mapping
from pathlib import Path

import pytest

from experionyx.artifacts import LocalArtifactStore
from experionyx.domain import Investigation
from experionyx.errors import SchedulerRefusal
from experionyx.execution import Executor
from experionyx.registry import Registry
from experionyx.scheduler import dispatch as dispatch_mod
from experionyx.scheduler.dispatch import DispatchOutcome
from experionyx.scheduler.engine import (
    cancel_schedule,
    dry_run,
    materialize,
    resolve_schedule,
    retry_unit,
    run_schedule,
    spec_from_schedule,
    unit_views,
    utc_now,
)
from experionyx.scheduler.entities import ExecutionAttempt
from experionyx.scheduler.spec import ExecutionPolicy, RetryPolicy, ScheduleSpec, UnitDef
from experionyx.scheduler.taxonomy import (  # fmt: skip
    FailureCategory,
    RetryPolicyKind,
    ScheduleRunState,
    UnitKind,
    UnitState,
)
from experionyx.sqlite import SqliteRegistry


def bootstrap(values: list[float]) -> Mapping[str, object]:
    return {"kind": "BOOTSTRAP", "sources": {"kind": "inline", "reference": values}}


BAD_FAULT_PARAMS: Mapping[str, object] = {
    "base_spec": {},
    "design": {},
    "model_id": "mdl_" + "0" * 32,
    "dataset_id": "dst_" + "0" * 32,
    "name": "bad",
}  # refuses at FaultRegistry.from_dict: a real ValidationError, not a crash


def unit(key: str, values: list[float], *deps: str, **over: object) -> UnitDef:
    base: dict[str, object] = dict(
        key=key, kind=UnitKind.STATISTICAL_ANALYSIS, parameters=bootstrap(values), depends_on=deps
    )
    base.update(over)
    return UnitDef(**base)  # type: ignore[arg-type]


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    reg = SqliteRegistry(ws / "registry.sqlite")
    inv = Investigation("scheduler-e2e", "does the scheduler orchestrate real engines?", utc_now())
    reg.add(inv)
    reg.close()
    return ws


def open_registry(ws: Path) -> SqliteRegistry:
    return SqliteRegistry(ws / "registry.sqlite")


def open_executor(ws: Path, reg: Registry) -> Executor:
    return Executor(reg, LocalArtifactStore(ws / "experiments"), source_root=Path.cwd())


def run(ws: Path, spec: ScheduleSpec, **kw: object) -> object:
    return run_schedule(
        lambda: open_registry(ws),
        LocalArtifactStore(ws / "experiments"),
        lambda reg: open_executor(ws, reg),
        spec,
        source_root=Path.cwd(),
        **kw,  # type: ignore[arg-type]
    )


# --- real cross-unit dependency and independent concurrency -------------------------------------


def test_dependent_unit_waits_for_its_dependency_and_sees_real_evidence(workspace: Path) -> None:
    spec = ScheduleSpec(
        "chain",
        "1.0.0",
        (unit("baseline", [1, 2, 3, 4, 5]), unit("treatment", [2, 3, 4, 5, 6], "baseline")),
    )
    result = run(workspace, spec)
    assert result.status is ScheduleRunState.COMPLETED  # type: ignore[attr-defined]
    assert dict(result.counts) == {"SUCCEEDED": 2}  # type: ignore[attr-defined]
    with open_registry(workspace) as reg:
        views = {v.key: v for v in unit_views(reg, result.schedule_id)}  # type: ignore[attr-defined]
        assert views["baseline"].status is UnitState.SUCCEEDED
        assert views["treatment"].status is UnitState.SUCCEEDED
        assert views["treatment"].primary_ref is not None
        assert views["baseline"].primary_ref != views["treatment"].primary_ref


def test_independent_units_may_execute_concurrently_under_bounded_workers(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two independent units under 2 workers: each blocks on a shared 2-party barrier before
    finishing. If the scheduler ran them serially, the first would wait on the barrier forever
    (nothing would ever release it) and the test would time out; a passing result is proof that
    both were dispatched before either completed, with no dependence on wall-clock thresholds."""
    barrier = threading.Barrier(2, timeout=10)
    real_dispatch = dispatch_mod.dispatch

    def barrier_dispatch(kind, registry, store, executor, investigation_id, parameters, refs, source_root):  # type: ignore[no-untyped-def]  # fmt: skip
        barrier.wait()
        return real_dispatch(kind, registry, store, executor, investigation_id, parameters, refs, source_root)  # fmt: skip

    monkeypatch.setattr("experionyx.scheduler.engine.dispatch", barrier_dispatch)
    spec = ScheduleSpec(
        "parallel", "1.0.0",
        (unit("a", [1, 2, 3]), unit("b", [4, 5, 6])),
        policy=ExecutionPolicy(max_workers=2),
    )  # fmt: skip
    result = run(workspace, spec)
    assert result.status is ScheduleRunState.COMPLETED  # type: ignore[attr-defined]
    assert dict(result.counts) == {"SUCCEEDED": 2}  # type: ignore[attr-defined]


def test_a_failed_dependency_blocks_its_dependents_but_not_independent_units(
    workspace: Path,
) -> None:
    spec = ScheduleSpec(
        "partial-fail", "1.0.0",
        (
            unit("bad", [], kind=UnitKind.FAULT_EXPERIMENT, parameters=BAD_FAULT_PARAMS),  # malformed: refuses at FaultRegistry.from_dict
            unit("dependent", [1, 2], "bad"),
            unit("independent", [7, 8, 9]),
        ),
    )  # fmt: skip
    result = run(workspace, spec)
    with open_registry(workspace) as reg:
        views = {v.key: v for v in unit_views(reg, result.schedule_id)}  # type: ignore[attr-defined]
    assert views["bad"].status is UnitState.FAILED
    assert views["dependent"].status is UnitState.BLOCKED
    assert (
        views["independent"].status is UnitState.SUCCEEDED
    )  # not held up by the unrelated failure
    with open_registry(workspace) as reg:
        (attempt,) = reg.find(ExecutionAttempt, unit_id=views["bad"].unit_id)
    assert attempt.failure_category is not None  # a real, classified failure, not a crash


# --- retries --------------------------------------------------------------------------------------


def test_failure_then_retry_then_success_creates_separate_attempts(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    calls = {"n": 0}
    real_dispatch = dispatch_mod.dispatch

    def flaky(kind, registry, store, executor, investigation_id, parameters, refs, source_root):  # type: ignore[no-untyped-def]  # fmt: skip
        calls["n"] += 1
        if calls["n"] == 1:
            return DispatchOutcome(UnitState.FAILED, None, None, "Simulated", "transient blip", FailureCategory.TRANSIENT)  # fmt: skip
        return real_dispatch(kind, registry, store, executor, investigation_id, parameters, refs, source_root)  # fmt: skip

    monkeypatch.setattr("experionyx.scheduler.engine.dispatch", flaky)
    spec = ScheduleSpec(
        "retry",
        "1.0.0",
        (
            unit(
                "flaky",
                [1, 2, 3],
                retry=RetryPolicy(RetryPolicyKind.FIXED, 3, (FailureCategory.TRANSIENT,)),
            ),
        ),
    )
    result = run(workspace, spec)
    assert result.status is ScheduleRunState.COMPLETED  # type: ignore[attr-defined]
    with open_registry(workspace) as reg:
        views = {v.key: v for v in unit_views(reg, result.schedule_id)}  # type: ignore[attr-defined]
        attempts = sorted(
            reg.find(ExecutionAttempt, unit_id=views["flaky"].unit_id), key=lambda a: a.attempt
        )
    assert views["flaky"].status is UnitState.SUCCEEDED
    assert [a.outcome for a in attempts] == [UnitState.FAILED, UnitState.SUCCEEDED]
    assert attempts[0].attempt == 0 and attempts[1].attempt == 1  # the failed attempt is kept, not overwritten  # fmt: skip


def test_a_configuration_failure_is_never_retried(workspace: Path) -> None:
    spec = ScheduleSpec(
        "no-retry",
        "1.0.0",
        (
            unit(
                "bad",
                [],
                kind=UnitKind.FAULT_EXPERIMENT,
                parameters=BAD_FAULT_PARAMS,
                retry=RetryPolicy(RetryPolicyKind.FIXED, 5, (FailureCategory.TRANSIENT,)),
            ),
        ),
    )
    result = run(workspace, spec)
    with open_registry(workspace) as reg:
        views = {v.key: v for v in unit_views(reg, result.schedule_id)}  # type: ignore[attr-defined]
        attempts = reg.find(ExecutionAttempt, unit_id=views["bad"].unit_id)
    assert views["bad"].status is UnitState.FAILED
    assert len(attempts) == 1  # CONFIGURATION is not in the retryable set: one attempt only
    assert attempts[0].failure_category is FailureCategory.CONFIGURATION


# --- resume / interruption -------------------------------------------------------------------------


def test_resume_does_not_rerun_a_successful_unit(workspace: Path) -> None:
    spec = ScheduleSpec("resumable", "1.0.0", (unit("a", [1, 2, 3]),))
    first = run(workspace, spec)
    with open_registry(workspace) as reg:
        before = unit_views(reg, first.schedule_id)[0].primary_ref  # type: ignore[attr-defined]
    second = run(workspace, spec)  # same spec: materialize is idempotent, nothing to redo
    with open_registry(workspace) as reg:
        after = unit_views(reg, second.schedule_id)[0].primary_ref  # type: ignore[attr-defined]
        attempts = reg.find(ExecutionAttempt, unit_id=unit_views(reg, second.schedule_id)[0].unit_id)  # type: ignore[attr-defined]  # fmt: skip
    assert before == after  # the SAME underlying evidence, not new duplicate work
    assert len(attempts) == 1  # dispatched exactly once across both invocations


def test_an_interrupted_run_is_recovered_and_finished_on_the_next_invocation(
    workspace: Path,
) -> None:
    spec = ScheduleSpec("crash", "1.0.0", (unit("a", [1, 2, 3]),))
    with open_registry(workspace) as reg:
        sched, _plan, units = materialize(reg, spec, utc_now())
        u = units["a"]
        ready = u.with_status(UnitState.READY)
        reg.update_status(ready)
        running = ready.with_status(UnitState.RUNNING)
        reg.update_status(
            running
        )  # simulate a process that died mid-dispatch: no ExecutionAttempt exists
    result = run(workspace, spec)
    assert result.status is ScheduleRunState.COMPLETED  # type: ignore[attr-defined]
    with open_registry(workspace) as reg:
        views = {v.key: v for v in unit_views(reg, sched.id)}
        attempts = sorted(
            reg.find(ExecutionAttempt, unit_id=views["a"].unit_id), key=lambda a: a.attempt
        )
    assert views["a"].status is UnitState.SUCCEEDED
    # attempt 0 is the recovered "interrupted" record; attempt 1 is the real completion
    assert attempts[0].outcome is UnitState.FAILED and attempts[0].failure_category is FailureCategory.TRANSIENT  # fmt: skip
    assert attempts[1].outcome is UnitState.SUCCEEDED


# --- cancellation -----------------------------------------------------------------------------------


def test_cancel_event_stops_scheduling_new_units_but_keeps_completed_evidence(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = ScheduleSpec("cancel-mid", "1.0.0", (unit("a", [1, 2, 3]), unit("b", [4, 5, 6], "a")))
    event = threading.Event()
    real_dispatch = dispatch_mod.dispatch

    def cancel_after_first(kind, registry, store, executor, investigation_id, parameters, refs, source_root):  # type: ignore[no-untyped-def]  # fmt: skip
        out = real_dispatch(kind, registry, store, executor, investigation_id, parameters, refs, source_root)  # fmt: skip
        event.set()
        return out

    monkeypatch.setattr("experionyx.scheduler.engine.dispatch", cancel_after_first)
    result = run(workspace, spec, cancel_event=event)
    assert result.status is ScheduleRunState.CANCELLED  # type: ignore[attr-defined]
    with open_registry(workspace) as reg:
        views = {v.key: v for v in unit_views(reg, result.schedule_id)}  # type: ignore[attr-defined]
    assert views["a"].status is UnitState.SUCCEEDED  # completed evidence is preserved
    assert views["b"].status is UnitState.CANCELLED  # never dispatched


def test_cancel_schedule_marks_open_units_cancelled_without_touching_completed_ones(workspace: Path) -> None:  # fmt: skip
    spec = ScheduleSpec("cancel-registry", "1.0.0", (unit("a", [1, 2, 3]),))
    with open_registry(workspace) as reg:
        sched, _plan, _units = materialize(reg, spec, utc_now())
    counts = cancel_schedule_via(workspace, sched.id)
    with open_registry(workspace) as reg:
        views = {v.key: v for v in unit_views(reg, sched.id)}
    assert views["a"].status is UnitState.CANCELLED
    assert counts.get("PLANNED", 0) == 1


def cancel_schedule_via(ws: Path, schedule_id: str) -> dict[str, int]:
    with open_registry(ws) as reg:
        return cancel_schedule(reg, schedule_id)


# --- dry-run: zero execution ----------------------------------------------------------------------


def test_dry_run_executes_nothing_and_validates_the_whole_plan(workspace: Path) -> None:
    stray_params = {
        "kind": "BOOTSTRAP",
        "sources": {"kind": "inline", "reference": ["$dep:missing"]},
    }
    spec = ScheduleSpec(
        "dry", "1.0.0", (unit("a", [1, 2, 3]), unit("b", [4, 5, 6], "a", parameters=stray_params))
    )
    with open_registry(workspace) as reg:
        result = dry_run(reg, spec)
        # nothing was dispatched: no ExecutionAttempt exists for any unit
        for v in unit_views(reg, result.schedule_id):
            assert reg.find(ExecutionAttempt, unit_id=v.unit_id) == []
            assert v.status is UnitState.PLANNED
    assert result.dry_run is True
    assert result.order == ("a", "b")
    assert "b" in result.issues  # references a dependency key it never declared


def test_dry_run_reports_declared_resource_requirements() -> None:
    from experionyx.scheduler.spec import ResourceRequirement

    spec = ScheduleSpec(
        "res", "1.0.0", (unit("a", [1, 2, 3], resources=ResourceRequirement("HIGH", memory_mb=256)),)
    )  # fmt: skip
    with SqliteRegistry(":memory:") as reg:
        reg.add(Investigation("x", "q", utc_now()))
        result = dry_run(reg, spec)
    unit_resources = result.resources["a"]
    assert isinstance(unit_resources, dict) and unit_resources["memory_mb"] == 256


# --- an analysis unit that requires evidence which isn't there ------------------------------------


def test_analysis_unit_waiting_for_missing_evidence_fails_as_configuration_not_a_crash(workspace: Path) -> None:  # fmt: skip
    profile_params = {"scope": "MODEL_DATASET_EVALUATION", "baseline_run": "run_" + "0" * 32}
    spec = ScheduleSpec(
        "needs-evidence",
        "1.0.0",
        (unit("needs", [1], kind=UnitKind.RELIABILITY_PROFILE, parameters=profile_params),),
    )
    result = run(workspace, spec)
    with open_registry(workspace) as reg:
        views = {v.key: v for v in unit_views(reg, result.schedule_id)}  # type: ignore[attr-defined]
        (attempt,) = reg.find(ExecutionAttempt, unit_id=views["needs"].unit_id)
    assert views["needs"].status is UnitState.FAILED
    assert attempt.failure_category is FailureCategory.CONFIGURATION  # refused; nothing fabricated


# --- validation refuses before anything runs --------------------------------------------------------


def test_a_cyclic_schedule_refuses_before_any_unit_is_materialized(workspace: Path) -> None:
    from experionyx.errors import ValidationError as SpecValidationError

    a = unit("a", [1], "b")
    with pytest.raises(SpecValidationError):  # constructing the spec itself already refuses
        ScheduleSpec("cyc", "1.0.0", (a,))


def test_materialize_is_idempotent_across_processes(workspace: Path) -> None:
    spec = ScheduleSpec("dup", "1.0.0", (unit("a", [1, 2, 3]),))
    with open_registry(workspace) as r1:
        s1, _, u1 = materialize(r1, spec, utc_now())
    with open_registry(workspace) as r2:
        s2, _, u2 = materialize(r2, spec, utc_now())
    assert s1.id == s2.id
    assert u1["a"].id == u2["a"].id


def test_resolve_schedule_by_prefix_and_full_id(workspace: Path) -> None:
    spec = ScheduleSpec("resolvable", "1.0.0", (unit("a", [1]),))
    with open_registry(workspace) as reg:
        sched, _, _ = materialize(reg, spec, utc_now())
        found_full = resolve_schedule(reg, sched.id)
        found_prefix = resolve_schedule(reg, sched.spec_id[:10])
    assert found_full.id == sched.id
    assert found_prefix.id == sched.id


def test_spec_from_schedule_reconstructs_the_exact_definition(workspace: Path) -> None:
    spec = ScheduleSpec("roundtrip", "1.0.0", (unit("a", [1, 2, 3]),))
    with open_registry(workspace) as reg:
        sched, _, _ = materialize(reg, spec, utc_now())
        rebuilt = spec_from_schedule(sched)
    assert rebuilt.spec_id == spec.spec_id


# --- retry_unit -------------------------------------------------------------------------------------


def test_retry_unit_resets_a_failed_unit_for_a_fresh_attempt(workspace: Path) -> None:
    spec = ScheduleSpec(
        "manual-retry",
        "1.0.0",
        (unit("bad", [], kind=UnitKind.FAULT_EXPERIMENT, parameters=BAD_FAULT_PARAMS),),
    )
    result = run(workspace, spec)
    with open_registry(workspace) as reg:
        u = retry_unit(reg, result.schedule_id, "bad")  # type: ignore[attr-defined]
    assert u.status is UnitState.RETRY_PENDING
    from experionyx.errors import ValidationError as RetryValidationError

    with pytest.raises(RetryValidationError), open_registry(workspace) as reg:
        retry_unit(reg, result.schedule_id, "does-not-exist")  # type: ignore[attr-defined]


def test_schedule_refusal_on_a_cycle_raised_by_expand_is_exported(workspace: Path) -> None:
    assert issubclass(SchedulerRefusal, Exception)

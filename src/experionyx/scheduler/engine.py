"""Schedule materialization, dry-run and the concurrent DAG-execution orchestrator.

Concurrency: each worker thread opens its OWN `Registry` connection to the same SQLite file (the
same pattern the registry already supports for any two writers: `BEGIN IMMEDIATE` plus a
connection timeout serializes them) and its own `Executor`. Only the coordinator thread ever calls
`Registry.update_status` on a `ScheduleUnit`/`ScheduleRun` or writes a `UnitStateTransition`, so
there is exactly one writer of scheduler bookkeeping at a time; workers only ever ADD immutable
records (through the engines they dispatch to, and their own `ExecutionAttempt`).

Timeout is honest about what a thread-based scheduler can and cannot do: Python cannot revoke a
running thread. A unit whose `timeout_seconds` elapses is recorded TIMED_OUT and the coordinator
stops waiting on it, but the underlying dispatch keeps running in the background until it finishes
naturally (see docs/scheduler.md). For a hard kill, run the engine out-of-process.
"""

import threading
from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from experionyx.artifacts import ArtifactStore
from experionyx.data_quality.engine import baseline_investigation
from experionyx.errors import ValidationError
from experionyx.execution import Executor
from experionyx.registry import Registry
from experionyx.scheduler.dispatch import DispatchOutcome, dispatch, referenced_dependencies
from experionyx.scheduler.entities import (
    ExecutionAttempt,
    Schedule,
    ScheduleRun,
    ScheduleUnit,
    UnitStateTransition,
)
from experionyx.scheduler.graph import Plan, expand
from experionyx.scheduler.spec import ENGINE_VERSION, ScheduleSpec
from experionyx.scheduler.taxonomy import (
    FailureCategory,
    RetryPolicyKind,
    ScheduleRunState,
    UnitState,
)

BLOCKING_STATES = frozenset(
    {UnitState.FAILED, UnitState.CANCELLED, UnitState.SKIPPED, UnitState.TIMED_OUT}
)
OpenRegistry = Callable[[], Registry]
OpenExecutor = Callable[[Registry], Executor]


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class UnitView:
    """Read-only snapshot of one unit, for CLI `status`/`inspect`/`graph` output."""

    unit_id: str
    key: str
    kind: str
    status: UnitState
    depends_on: tuple[str, ...]  # keys
    priority: int
    attempts: int
    primary_ref: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "unit_id": self.unit_id,
            "key": self.key,
            "kind": self.kind,
            "status": self.status.value,
            "depends_on": list(self.depends_on),
            "priority": self.priority,
            "attempts": self.attempts,
            "primary_ref": self.primary_ref,
        }


@dataclass(frozen=True)
class ScheduleRunResult:
    schedule_id: str
    schedule_run_id: str | None  # None for a dry run: nothing was recorded but the plan itself
    status: ScheduleRunState | None
    dry_run: bool
    counts: Mapping[str, int]  # UnitState value -> count
    order: tuple[str, ...]  # planned or actual dispatch order, by key
    issues: Mapping[str, str]  # key -> problem, for dry-run structural findings
    resources: Mapping[str, object]  # key -> declared ResourceRequirement, where known

    def to_dict(self) -> dict[str, object]:
        return {
            "schedule_id": self.schedule_id,
            "schedule_run_id": self.schedule_run_id,
            "status": self.status.value if self.status else None,
            "dry_run": self.dry_run,
            "counts": dict(self.counts),
            "order": list(self.order),
            "issues": dict(self.issues),
            "resources": dict(self.resources),
        }


def materialize(
    registry: Registry, spec: ScheduleSpec, now: datetime
) -> tuple[Schedule, Plan, dict[str, ScheduleUnit]]:
    """Idempotently create the `Schedule` and every `ScheduleUnit` for `spec`. Raises
    SchedulerRefusal (from `graph.expand`) before anything is persisted if the DAG is invalid."""
    plan = expand(spec)
    sched = Schedule(
        spec.name, spec.version, spec.spec_id, spec.to_dict(), plan.graph_hash, ENGINE_VERSION, now
    )
    units: dict[str, ScheduleUnit] = {}
    with registry.transaction():
        if not registry.exists(Schedule, sched.id):
            registry.add(sched)
        for pu in plan.units:  # topological order: every dependency is already materialized
            dep_ids = tuple(units[k].id for k in pu.depends_on)
            su = ScheduleUnit(
                sched.id, pu.key, pu.kind, pu.parameters, dep_ids, pu.retry.to_dict(),
                pu.resources.to_dict() if pu.resources else None, pu.timeout_seconds, pu.priority, now,
            )  # fmt: skip
            if not registry.exists(ScheduleUnit, su.id):
                registry.add(su)
            else:
                su = registry.get(ScheduleUnit, su.id)
            units[pu.key] = su
    return sched, plan, units


def dry_run(
    registry: Registry, spec: ScheduleSpec, now: datetime | None = None
) -> ScheduleRunResult:
    """Validate and expand the WHOLE plan; execute ZERO experiments. Reports the deterministic
    dispatch order, every structural issue (a parameter referencing an undeclared dependency), and
    every statically-known resource requirement."""
    sched, plan, units = materialize(registry, spec, now or utc_now())
    counts: dict[str, int] = {}
    for u in units.values():
        counts[u.status.value] = counts.get(u.status.value, 0) + 1
    issues: dict[str, str] = {}
    resources: dict[str, object] = {}
    for pu in plan.units:
        stray = referenced_dependencies(pu.parameters) - set(pu.depends_on)
        if stray:
            issues[pu.key] = f"parameters reference undeclared dependency key(s) {sorted(stray)}"
        if pu.resources is not None:
            resources[pu.key] = pu.resources.to_dict()
    return ScheduleRunResult(
        sched.id, None, None, True, counts, tuple(u.key for u in plan.units), issues, resources
    )


def _seed(
    coord: Registry, units: dict[str, ScheduleUnit]
) -> tuple[dict[str, UnitState], dict[str, str], dict[str, int]]:
    status = {k: u.status for k, u in units.items()}
    primary_ref: dict[str, str] = {}
    attempt_no: dict[str, int] = {}
    for k, u in units.items():
        attempts = coord.find(ExecutionAttempt, unit_id=u.id)
        attempt_no[k] = len(attempts)
        if u.status is UnitState.SUCCEEDED:
            ok = [a for a in attempts if a.outcome is UnitState.SUCCEEDED]
            if ok:
                primary_ref[k] = ok[-1].primary_ref or ok[-1].run_id or ""
    return status, primary_ref, attempt_no


def _should_retry(unit: ScheduleUnit, outcome: DispatchOutcome, attempts_used: int) -> bool:
    policy = unit.retry_policy
    if policy.get("kind") != RetryPolicyKind.FIXED.value:
        return False
    max_attempts = policy.get("max_attempts")
    if attempts_used >= (max_attempts if isinstance(max_attempts, int) else 1):
        return False
    if outcome.failure_category is None:
        return False
    retryable = policy.get("retryable")
    return outcome.failure_category.value in (
        retryable if isinstance(retryable, list | tuple) else ()
    )


def _mem_mb(resources: Mapping[str, object]) -> int:
    raw = resources.get("memory_mb")
    return raw if isinstance(raw, int) else 0


class _Orchestrator:
    """Owns the mutable scheduling state of one `_execute` call. Every ScheduleUnit/ScheduleRun
    status transition and every UnitStateTransition is written by the coordinator thread only,
    under `self._lock`; worker threads only run `dispatch`, on their own registry connection."""

    def __init__(
        self,
        coord: Registry,
        open_registry: OpenRegistry,
        store: ArtifactStore,
        open_executor: OpenExecutor,
        run: ScheduleRun,
        plan: Plan,
        units: dict[str, ScheduleUnit],
        source_root: Path,
        max_workers: int,
        memory_budget_mb: int | None,
        cancel_event: threading.Event | None,
        clock: Callable[[], datetime],
    ) -> None:
        self.coord, self.open_registry, self.store, self.open_executor = (
            coord,
            open_registry,
            store,
            open_executor,
        )
        self.run, self.plan, self.units, self.source_root = run, plan, units, source_root
        self.max_workers, self.memory_budget_mb = max_workers, memory_budget_mb
        self.cancel_event, self.clock = cancel_event, clock
        self.status, self.primary_ref, self.attempt_no = _seed(coord, units)
        self.seq: dict[str, int] = {}
        self.started_at: dict[str, datetime] = {}
        self.lock = threading.Lock()
        self.running_memory = 0
        self.running_exclusive = False
        self.dispatch_order: list[str] = []
        self._key_of_id = {u.id: k for k, u in units.items()}

    def _dep_keys(self, key: str) -> tuple[str, ...]:
        return tuple(self._key_of_id[d] for d in self.units[key].depends_on)

    def _transition(self, key: str, to: UnitState, reason: str) -> None:
        unit = self.units[key]
        updated = unit.with_status(
            to
        )  # raises ValidationError on an illegal transition: a real bug
        n = self.seq.get(unit.id, 0)
        self.seq[unit.id] = n + 1
        with self.coord.transaction():
            self.coord.update_status(updated)
            self.coord.add(
                UnitStateTransition(unit.id, self.run.id, n, unit.status, to, reason, self.clock())
            )
        self.units[key] = updated
        self.status[key] = to

    def _record_attempt(
        self, key: str, started: datetime, finished: datetime, outcome: DispatchOutcome
    ) -> None:
        n = self.attempt_no[key]
        self.attempt_no[key] = n + 1
        self.coord.add(
            ExecutionAttempt(
                self.units[key].id, self.run.id, n, started, finished, outcome.status,
                outcome.failure_category, outcome.error_type, outcome.error_message,
                outcome.primary_ref, outcome.run_id,
            )
        )  # fmt: skip

    def reconcile_interrupted(self) -> None:
        """A ScheduleUnit found RUNNING at the start of a fresh invocation belongs to a process
        that ended without finishing it (this is a resume/recovery entry point, never a live
        concurrent writer -- the scheduler runs one orchestrator per schedule at a time)."""
        for key in list(self.units):
            if self.status[key] is UnitState.RUNNING:
                now = self.clock()
                self._record_attempt(
                    key, now, now,
                    DispatchOutcome(UnitState.FAILED, None, None, "Interrupted",
                                     "the previous process ended while this unit was RUNNING",
                                     FailureCategory.TRANSIENT),
                )  # fmt: skip
                self._transition(
                    key, UnitState.FAILED, "recovered: found RUNNING at schedule-run start"
                )
                # recovery always grants one more attempt, independent of the unit's own retry
                # policy: an interrupted process is not a retry decision, it is unfinished work.
                self._transition(key, UnitState.RETRY_PENDING, "recovered: scheduled for a fresh attempt")  # fmt: skip

    def propagate_blocked(self) -> None:
        changed = True
        while changed:
            changed = False
            for key in list(self.units):
                st = self.status[key]
                deps = self._dep_keys(key)
                if st in (UnitState.PLANNED, UnitState.READY, UnitState.BLOCKED):
                    stuck = [d for d in deps if self.status[d] in BLOCKING_STATES]
                    if stuck and st is not UnitState.BLOCKED:
                        self._transition(
                            key, UnitState.BLOCKED, f"dependency did not succeed: {stuck}"
                        )
                        changed = True
                    elif not stuck and st is UnitState.BLOCKED and all(self.status[d] is UnitState.SUCCEEDED for d in deps):  # fmt: skip
                        self._transition(key, UnitState.READY, "every dependency now SUCCEEDED")
                        changed = True

    def _admissible(self, key: str) -> bool:
        if self.running_exclusive:
            return False
        res = self.units[key].resources or {}
        exclusive, mem = bool(res.get("exclusive")), _mem_mb(res)
        running = any(s is UnitState.RUNNING for s in self.status.values())
        if exclusive and running:
            return False
        return not (
            self.memory_budget_mb is not None
            and mem
            and running
            and self.running_memory + mem > self.memory_budget_mb
        )

    def ready_candidates(self) -> list[str]:
        out = [
            k for k, u in self.units.items()
            if self.status[k] in (UnitState.PLANNED, UnitState.READY, UnitState.RETRY_PENDING)
            and all(self.status[d] is UnitState.SUCCEEDED for d in self._dep_keys(k))
        ]  # fmt: skip
        out.sort(key=lambda k: (-self.units[k].priority, k))
        return out

    def submit_more(
        self, pool: ThreadPoolExecutor, futures: dict[Future[DispatchOutcome], str]
    ) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            return
        for key in self.ready_candidates():
            if len(futures) >= self.max_workers:
                break
            if not self._admissible(key):
                continue
            unit = self.units[key]
            if self.status[key] in (UnitState.PLANNED, UnitState.RETRY_PENDING):
                self._transition(key, UnitState.READY, "dependencies satisfied")
            self._transition(key, UnitState.RUNNING, "dispatched")
            res = unit.resources or {}
            if res.get("exclusive"):
                self.running_exclusive = True
            self.running_memory += _mem_mb(res)
            self.started_at[key] = self.clock()
            refs = dict(self.primary_ref)
            fut = pool.submit(_run_one, self.open_registry, self.store, self.open_executor, self.source_root, self.run.investigation_id, unit, refs)  # fmt: skip
            futures[fut] = key
            self.dispatch_order.append(key)

    def _release(self, key: str) -> None:
        res = self.units[key].resources or {}
        if res.get("exclusive"):
            self.running_exclusive = False
        self.running_memory = max(0, self.running_memory - _mem_mb(res))

    def finish(self, key: str, outcome: DispatchOutcome) -> None:
        started = self.started_at.pop(key, self.clock())
        finished = self.clock()
        self._release(key)
        self._record_attempt(key, started, finished, outcome)
        if outcome.status is UnitState.SUCCEEDED:
            self._transition(key, UnitState.SUCCEEDED, "dispatch succeeded")
            self.primary_ref[key] = outcome.primary_ref or outcome.run_id or ""
            return
        used = self.attempt_no[key]
        self._transition(key, UnitState.FAILED, f"attempt {used} failed ({outcome.failure_category}): {outcome.error_message}")  # fmt: skip
        if _should_retry(self.units[key], outcome, used):
            self._transition(key, UnitState.RETRY_PENDING, f"attempt {used} is retryable; retrying")

    def timeout(self, key: str) -> None:
        started = self.started_at.pop(key, self.clock())
        self._release(key)
        self._record_attempt(
            key, started, self.clock(),
            DispatchOutcome(UnitState.TIMED_OUT, None, None, "Timeout",
                             f"exceeded timeout_seconds={self.units[key].timeout_seconds}; the scheduler stopped "
                             "waiting but did not (cannot) kill the underlying thread", FailureCategory.TIMEOUT),
        )  # fmt: skip
        used = self.attempt_no[key]
        self._transition(key, UnitState.TIMED_OUT, f"attempt {used} timed out")
        if _should_retry(self.units[key], DispatchOutcome(UnitState.TIMED_OUT, failure_category=FailureCategory.TIMEOUT), used):  # fmt: skip
            self._transition(key, UnitState.RETRY_PENDING, f"attempt {used} is retryable; retrying")

    def cancel_open(self) -> None:
        for key in list(self.units):
            if self.status[key] in (
                UnitState.PLANNED,
                UnitState.READY,
                UnitState.BLOCKED,
                UnitState.RETRY_PENDING,
            ):
                self._transition(key, UnitState.CANCELLED, "schedule run cancelled")

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.status.values():
            out[s.value] = out.get(s.value, 0) + 1
        return out


def _run_one(
    open_registry: OpenRegistry,
    store: ArtifactStore,
    open_executor: OpenExecutor,
    source_root: Path,
    investigation_id: str,
    unit: ScheduleUnit,
    refs: dict[str, str],
) -> DispatchOutcome:
    reg = open_registry()
    try:
        return dispatch(unit.unit_kind, reg, store, open_executor(reg), investigation_id, unit.parameters, refs, source_root)  # fmt: skip
    finally:
        reg.close()


def _next_seconds(o: _Orchestrator) -> float | None:
    now = o.clock()
    remaining: list[float] = []
    for k in o.started_at:
        limit = o.units[k].timeout_seconds
        if limit is not None:
            remaining.append(limit - (now - o.started_at[k]).total_seconds())
    return max(0.0, min(remaining)) if remaining else None


def _execute(o: _Orchestrator) -> dict[str, int]:
    o.reconcile_interrupted()
    o.propagate_blocked()
    futures: dict[Future[DispatchOutcome], str] = {}
    pool = ThreadPoolExecutor(max_workers=o.max_workers)
    try:
        with o.lock:
            o.submit_more(pool, futures)
        while futures:
            wait_for = _next_seconds(o)
            done, _pending = wait(list(futures), timeout=wait_for, return_when=FIRST_COMPLETED)
            with o.lock:
                for fut in done:
                    key = futures.pop(fut)
                    o.finish(key, fut.result())
                now = o.clock()
                for fut, key in list(futures.items()):
                    tl = o.units[key].timeout_seconds
                    if tl is not None and (now - o.started_at[key]).total_seconds() >= tl:
                        futures.pop(fut)
                        o.timeout(key)
                o.propagate_blocked()
                if o.cancel_event is not None and o.cancel_event.is_set():
                    o.cancel_open()
                    break
                o.submit_more(pool, futures)
    finally:
        # Orphaned (timed-out) threads may still be running: don't block process exit on them.
        pool.shutdown(wait=False, cancel_futures=True)
    return o.counts()


def run_schedule(
    open_registry: OpenRegistry,
    store: ArtifactStore,
    open_executor: OpenExecutor,
    spec: ScheduleSpec,
    *,
    investigation_id: str | None = None,
    max_workers: int | None = None,
    dry_run_only: bool = False,
    source_root: Path,
    cancel_event: threading.Event | None = None,
    clock: Callable[[], datetime] = utc_now,
) -> ScheduleRunResult:
    """Materialize `spec` (idempotent), then either report the plan (`dry_run_only`) or execute the
    DAG to completion, honoring dependencies, retries, resources and cancellation. Safe to call
    again for the same `spec`: already-SUCCEEDED units are never re-dispatched (resume), and a
    fresh `ScheduleRun` records the new invocation without losing any earlier one."""
    coord = open_registry()
    try:
        now = clock()
        if dry_run_only:
            return dry_run(coord, spec, now)
        sched, plan, units = materialize(coord, spec, now)
        inv_id = baseline_investigation(coord, investigation_id)
        policy = (
            spec.policy if max_workers is None else replace(spec.policy, max_workers=max_workers)
        )
        earlier = [r.sequence for r in coord.find(ScheduleRun, schedule_id=sched.id, investigation_id=inv_id)]  # fmt: skip
        run = ScheduleRun(
            sched.id, inv_id, policy.to_dict(), False, max(earlier, default=-1) + 1, now
        )
        with coord.transaction():
            coord.add(run)
            run = run.with_status(ScheduleRunState.RUNNING)
            coord.update_status(run)
        orch = _Orchestrator(
            coord, open_registry, store, open_executor, run, plan, units, source_root,
            policy.max_workers, policy.memory_budget_mb, cancel_event, clock,
        )  # fmt: skip
        counts = _execute(orch)
        cancelled = cancel_event is not None and cancel_event.is_set()
        final = (
            ScheduleRunState.CANCELLED if cancelled
            else ScheduleRunState.FAILED if counts.get(UnitState.FAILED.value) or counts.get(UnitState.TIMED_OUT.value)
            else ScheduleRunState.COMPLETED
        )  # fmt: skip
        coord.update_status(run.with_status(final))
        return ScheduleRunResult(
            sched.id, run.id, final, False, counts, tuple(orch.dispatch_order), {}, {}
        )
    finally:
        coord.close()


def unit_views(registry: Registry, schedule_id: str) -> list[UnitView]:
    units = registry.find(ScheduleUnit, schedule_id=schedule_id)
    out = []
    for u in units:
        attempts = registry.find(ExecutionAttempt, unit_id=u.id)
        ref = None
        for a in sorted(attempts, key=lambda a: a.attempt):
            if a.outcome is UnitState.SUCCEEDED:
                ref = a.primary_ref or a.run_id
        deps = {d: registry.get(ScheduleUnit, d).unit_key for d in u.depends_on}
        out.append(UnitView(u.id, u.unit_key, u.unit_kind.value, u.status, tuple(deps[d] for d in u.depends_on), u.priority, len(attempts), ref))  # fmt: skip
    return sorted(out, key=lambda v: v.key)


def spec_from_schedule(schedule: Schedule) -> ScheduleSpec:
    """Reconstruct the exact `ScheduleSpec` a `Schedule` was materialized from. Resume and replay
    need no original spec file: the definition is fully preserved in the registry."""
    return ScheduleSpec.from_dict(dict(schedule.spec))


def cancel_schedule(
    registry: Registry, schedule_id: str, now: datetime | None = None
) -> dict[str, int]:
    """Mark every still-open unit of `schedule_id` CANCELLED, and its latest ScheduleRun CANCELLED
    if it is still RUNNING. A unit already RUNNING is marked CANCELLED too (best effort: this
    records intent, it cannot forcibly stop a dispatch already in flight -- see docs/scheduler.md).
    Completed evidence (SUCCEEDED units and their records) is never touched."""  # fmt: skip
    now = now or utc_now()
    units = registry.find(ScheduleUnit, schedule_id=schedule_id)
    runs = sorted(registry.find(ScheduleRun, schedule_id=schedule_id), key=lambda r: r.sequence)
    counts: dict[str, int] = {}
    open_states = frozenset(
        {UnitState.PLANNED, UnitState.READY, UnitState.RUNNING, UnitState.BLOCKED, UnitState.RETRY_PENDING}
    )  # fmt: skip
    for u in units:
        if u.status not in open_states:
            continue
        n = len(registry.find(UnitStateTransition, unit_id=u.id))
        with registry.transaction():
            registry.update_status(u.with_status(UnitState.CANCELLED))
            run_id = runs[-1].id if runs else None
            if run_id is not None:
                registry.add(
                    UnitStateTransition(
                        u.id, run_id, n, u.status, UnitState.CANCELLED, "cancelled by request", now
                    )
                )
        counts[u.status.value] = counts.get(u.status.value, 0) + 1
    if runs and runs[-1].status is ScheduleRunState.RUNNING:
        registry.update_status(runs[-1].with_status(ScheduleRunState.CANCELLED))
    return counts


def retry_unit(registry: Registry, schedule_id: str, unit_key: str, now: datetime | None = None) -> ScheduleUnit:  # fmt: skip
    """Explicitly request a new attempt at a unit that stopped in FAILED or TIMED_OUT, regardless
    of whether its retry policy is exhausted. Recorded as its own auditable transition; the
    dependents that were BLOCKED on it are re-evaluated the next time the schedule is run."""
    now = now or utc_now()
    matches = [
        u for u in registry.find(ScheduleUnit, schedule_id=schedule_id) if u.unit_key == unit_key
    ]
    if not matches:
        raise ValidationError(f"no unit {unit_key!r} in schedule {schedule_id}")
    unit = matches[0]
    if unit.status not in (UnitState.FAILED, UnitState.TIMED_OUT):
        raise ValidationError(f"unit {unit_key!r} is {unit.status.value}; only FAILED or TIMED_OUT units can be retried")  # fmt: skip
    runs = sorted(registry.find(ScheduleRun, schedule_id=schedule_id), key=lambda r: r.sequence)
    updated = unit.with_status(UnitState.RETRY_PENDING)
    with registry.transaction():
        registry.update_status(updated)
        if runs:
            n = len(registry.find(UnitStateTransition, unit_id=unit.id))
            registry.add(
                UnitStateTransition(
                    unit.id,
                    runs[-1].id,
                    n,
                    unit.status,
                    UnitState.RETRY_PENDING,
                    "manual retry requested",
                    now,
                )
            )
    return updated


def resolve_schedule(registry: Registry, ref: str) -> Schedule:
    if ref.startswith(Schedule.PREFIX + "_"):
        return registry.get(Schedule, ref)
    matches = [s for s in registry.find(Schedule) if s.spec_id == ref or s.spec_id.startswith(ref)]
    if len(matches) != 1:
        raise ValidationError(f"{'no' if not matches else 'ambiguous'} schedule matches {ref!r}")
    return matches[0]


__all__ = [
    "BLOCKING_STATES",
    "OpenExecutor",
    "OpenRegistry",
    "ScheduleRunResult",
    "UnitView",
    "cancel_schedule",
    "dry_run",
    "materialize",
    "resolve_schedule",
    "retry_unit",
    "run_schedule",
    "spec_from_schedule",
    "unit_views",
    "utc_now",
]

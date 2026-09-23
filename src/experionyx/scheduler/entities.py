"""Registry entities of the experiment scheduler (see docs/scheduler.md).

`Schedule` is the immutable, content-addressed DEFINITION (identical to how `Benchmark` holds a
definition separate from its results). `ScheduleRun` is one orchestration session over it --
mutable status, exactly like `Experiment`/`Run`/`FaultExperiment`. `ScheduleUnit` is one expanded,
dependency-ordered unit; its identity is derived from every meaningful input INCLUDING the resolved
IDs of the units it depends on, so a change anywhere upstream changes its own ID too. Both
`ScheduleRun` and `ScheduleUnit` participate in `Registry.update_status`; every transition is also
appended, immutably, as a `UnitStateTransition`. `ExecutionAttempt` is append-only: a retry creates
a new record and never overwrites the one before it.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from typing import ClassVar, Self

import experionyx.validation as v
from experionyx.domain import Entity, Investigation, Run, open_payload
from experionyx.errors import ValidationError
from experionyx.scheduler.taxonomy import (
    SCHEDULE_RUN_TRANSITIONS,
    UNIT_TRANSITIONS,
    FailureCategory,
    ScheduleRunState,
    UnitKind,
    UnitState,
)


@dataclass(frozen=True)
class Schedule(Entity):
    KIND: ClassVar[str] = "schedule"
    PREFIX: ClassVar[str] = "sch"
    name: str
    version: str
    spec_id: str  # ssp_<hash> of the full compact definition (units, dependencies, policy)
    spec: Mapping[str, object]
    graph_hash: str  # hash of the expanded, ordered DAG (deterministic given spec_id)
    engine_version: str
    created_at: datetime

    def __post_init__(self) -> None:
        v.text("name", self.name)
        v.text("version", self.version)
        v.text("spec_id", self.spec_id)
        object.__setattr__(self, "spec", v.freeze_mapping("spec", self.spec))
        v.digest("graph_hash", self.graph_hash)
        v.text("engine_version", self.engine_version)
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {"spec_id": self.spec_id}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d,
            cls.KIND,
            ("name", "version", "spec_id", "spec", "graph_hash", "engine_version", "created_at"),
        )
        return cls(
            v.get_str(d, "name"),
            v.get_str(d, "version"),
            v.get_str(d, "spec_id"),
            v.get_mapping(d, "spec"),
            v.get_str(d, "graph_hash"),
            v.get_str(d, "engine_version"),
            v.get_time(d, "created_at"),
        )


@dataclass(frozen=True)
class ScheduleRun(Entity):
    """One orchestration session over a `Schedule`. `sequence` distinguishes repeated invocations
    (an initial run, a resume, a replay) the same way `Run.attempt` does; the schedule's own
    identity (`schedule_id`) never changes across them."""

    KIND: ClassVar[str] = "schedule_run"
    PREFIX: ClassVar[str] = "scr"
    schedule_id: str
    investigation_id: str
    policy: Mapping[
        str, object
    ]  # the ExecutionPolicy actually used (may override the spec default)
    dry_run: bool
    sequence: int
    created_at: datetime
    status: ScheduleRunState = ScheduleRunState.PLANNED

    def __post_init__(self) -> None:
        v.ref("schedule_id", self.schedule_id, Schedule.PREFIX)
        v.ref("investigation_id", self.investigation_id, Investigation.PREFIX)
        object.__setattr__(self, "policy", v.freeze_mapping("policy", self.policy))
        if not isinstance(self.dry_run, bool):
            raise ValidationError("dry_run must be a boolean")
        v.non_negative_int("sequence", self.sequence)
        v.timestamp("created_at", self.created_at)
        v.member("status", self.status, ScheduleRunState)

    def _identity(self) -> Mapping[str, object]:
        return {"schedule_id": self.schedule_id, "investigation_id": self.investigation_id, "sequence": self.sequence}  # fmt: skip

    def with_status(self, status: ScheduleRunState) -> Self:
        if status not in SCHEDULE_RUN_TRANSITIONS[self.status]:
            raise ValidationError(f"illegal schedule run transition {self.status} -> {status}")
        return replace(self, status=status)

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d,
            cls.KIND,
            (
                "schedule_id",
                "investigation_id",
                "policy",
                "dry_run",
                "sequence",
                "created_at",
                "status",
            ),
        )
        raw = v.get_raw(d, "dry_run")
        if not isinstance(raw, bool):
            raise ValidationError("dry_run must be a boolean")
        return cls(
            v.get_str(d, "schedule_id"),
            v.get_str(d, "investigation_id"),
            v.get_mapping(d, "policy"),
            raw,
            v.get_int(d, "sequence"),
            v.get_time(d, "created_at"),
            v.get_enum(d, "status", ScheduleRunState),
        )


@dataclass(frozen=True)
class ScheduleUnit(Entity):
    """One expanded unit of a schedule. Identity covers everything that changes what would be
    dispatched (kind, parameters, retry policy, resources, timeout) AND the resolved IDs of every
    unit it depends on -- so a definition change anywhere upstream changes every downstream unit's
    identity too, exactly as an upstream content-addressed record change would."""

    KIND: ClassVar[str] = "schedule_unit"
    PREFIX: ClassVar[str] = "sun"
    schedule_id: str
    unit_key: str
    unit_kind: UnitKind
    parameters: Mapping[str, object]
    depends_on: tuple[str, ...]  # ScheduleUnit IDs (sun_...), not raw keys
    retry_policy: Mapping[str, object]
    resources: Mapping[str, object] | None
    timeout_seconds: float | None
    priority: int
    created_at: datetime
    status: UnitState = UnitState.PLANNED

    def __post_init__(self) -> None:
        v.ref("schedule_id", self.schedule_id, Schedule.PREFIX)
        v.text("unit_key", self.unit_key)
        v.member("unit_kind", self.unit_kind, UnitKind)
        object.__setattr__(self, "parameters", v.freeze_mapping("parameters", self.parameters))
        deps = tuple(self.depends_on)
        for i, d in enumerate(deps):
            v.ref(f"depends_on[{i}]", d, self.PREFIX)
        object.__setattr__(self, "depends_on", deps)
        object.__setattr__(
            self, "retry_policy", v.freeze_mapping("retry_policy", self.retry_policy)
        )
        if self.resources is not None:
            object.__setattr__(self, "resources", v.freeze_mapping("resources", self.resources))
        if self.timeout_seconds is not None:
            v.finite_float("timeout_seconds", self.timeout_seconds)
        if not isinstance(self.priority, int) or isinstance(self.priority, bool):
            raise ValidationError("priority must be an integer")
        v.timestamp("created_at", self.created_at)
        v.member("status", self.status, UnitState)

    def _identity(self) -> Mapping[str, object]:
        return {
            "schedule_id": self.schedule_id,
            "unit_key": self.unit_key,
            "unit_kind": self.unit_kind,
            "parameters": self.parameters,
            "depends_on": self.depends_on,
            "retry_policy": self.retry_policy,
            "resources": self.resources,
            "timeout_seconds": self.timeout_seconds,
            "priority": self.priority,
        }

    def with_status(self, status: UnitState) -> Self:
        if status not in UNIT_TRANSITIONS[self.status]:
            raise ValidationError(f"illegal unit transition {self.status} -> {status}")
        return replace(self, status=status)

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d,
            cls.KIND,
            (
                "schedule_id", "unit_key", "unit_kind", "parameters", "depends_on", "retry_policy",
                "resources", "timeout_seconds", "priority", "created_at", "status",
            ),
        )  # fmt: skip
        deps = v.get_raw(d, "depends_on")
        if not isinstance(deps, list) or not all(isinstance(x, str) for x in deps):
            raise ValidationError("depends_on must be a list of strings")
        resources = v.get_raw(d, "resources")
        return cls(
            v.get_str(d, "schedule_id"),
            v.get_str(d, "unit_key"),
            v.get_enum(d, "unit_kind", UnitKind),
            v.get_mapping(d, "parameters"),
            tuple(deps),
            v.get_mapping(d, "retry_policy"),
            None if resources is None else v.get_mapping(d, "resources"),
            v.get_opt_float(d, "timeout_seconds"),
            v.get_int(d, "priority"),
            v.get_time(d, "created_at"),
            v.get_enum(d, "status", UnitState),
        )


@dataclass(frozen=True)
class ExecutionAttempt(Entity):
    """One immutable, terminal outcome of one attempt at a `ScheduleUnit`. `attempt` numbers from
    0; a retry NEVER overwrites an earlier attempt, it adds a new record."""

    KIND: ClassVar[str] = "execution_attempt"
    PREFIX: ClassVar[str] = "att"
    unit_id: str
    schedule_run_id: str
    attempt: int
    started_at: datetime
    finished_at: datetime
    outcome: UnitState  # SUCCEEDED | FAILED | TIMED_OUT | CANCELLED
    failure_category: FailureCategory | None
    error_type: str | None
    error_message: str | None
    primary_ref: str | None  # the engine-produced record ID this unit resolved to
    run_id: str | None  # the underlying executed Run, if the dispatched engine created one

    _TERMINAL: ClassVar[frozenset[UnitState]] = frozenset(
        {UnitState.SUCCEEDED, UnitState.FAILED, UnitState.TIMED_OUT, UnitState.CANCELLED}
    )

    def __post_init__(self) -> None:
        v.ref("unit_id", self.unit_id, ScheduleUnit.PREFIX)
        v.ref("schedule_run_id", self.schedule_run_id, ScheduleRun.PREFIX)
        v.non_negative_int("attempt", self.attempt)
        v.timestamp("started_at", self.started_at)
        v.timestamp("finished_at", self.finished_at)
        if self.finished_at < self.started_at:
            raise ValidationError("finished_at must not precede started_at")
        if v.member("outcome", self.outcome, UnitState) not in self._TERMINAL:
            raise ValidationError(
                f"outcome must be one of {sorted(s.value for s in self._TERMINAL)}"
            )
        ok = self.outcome is UnitState.SUCCEEDED
        if ok and (self.failure_category is not None or self.error_type is not None):
            raise ValidationError("a SUCCEEDED attempt carries no failure information")
        if not ok and self.error_type is None:
            raise ValidationError("a non-SUCCEEDED attempt needs error_type")
        if self.failure_category is not None:
            v.member("failure_category", self.failure_category, FailureCategory)
        if self.error_message is not None and not isinstance(self.error_message, str):
            raise ValidationError("error_message must be a string")
        if self.primary_ref is not None:
            v.text("primary_ref", self.primary_ref)
        if self.run_id is not None:
            v.ref("run_id", self.run_id, Run.PREFIX)

    def _identity(self) -> Mapping[str, object]:
        return {"unit_id": self.unit_id, "attempt": self.attempt}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d,
            cls.KIND,
            (
                "unit_id", "schedule_run_id", "attempt", "started_at", "finished_at", "outcome",
                "failure_category", "error_type", "error_message", "primary_ref", "run_id",
            ),
        )  # fmt: skip
        fc = v.get_raw(d, "failure_category")
        return cls(
            v.get_str(d, "unit_id"),
            v.get_str(d, "schedule_run_id"),
            v.get_int(d, "attempt"),
            v.get_time(d, "started_at"),
            v.get_time(d, "finished_at"),
            v.get_enum(d, "outcome", UnitState),
            None if fc is None else v.get_enum(d, "failure_category", FailureCategory),
            v.get_opt_str(d, "error_type"),
            v.get_opt_str(d, "error_message"),
            v.get_opt_str(d, "primary_ref"),
            v.get_opt_str(d, "run_id"),
        )


@dataclass(frozen=True)
class UnitStateTransition(Entity):
    """An append-only audit record of one `ScheduleUnit` status change. `sequence` is assigned by
    the orchestrator, monotonically increasing per unit within one `ScheduleRun`."""

    KIND: ClassVar[str] = "unit_state_transition"
    PREFIX: ClassVar[str] = "utr"
    unit_id: str
    schedule_run_id: str
    sequence: int
    from_status: UnitState
    to_status: UnitState
    reason: str
    created_at: datetime

    def __post_init__(self) -> None:
        v.ref("unit_id", self.unit_id, ScheduleUnit.PREFIX)
        v.ref("schedule_run_id", self.schedule_run_id, ScheduleRun.PREFIX)
        v.non_negative_int("sequence", self.sequence)
        v.member("from_status", self.from_status, UnitState)
        v.member("to_status", self.to_status, UnitState)
        if self.to_status not in UNIT_TRANSITIONS[self.from_status]:
            raise ValidationError(f"illegal unit transition {self.from_status} -> {self.to_status}")
        v.text("reason", self.reason)
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {"unit_id": self.unit_id, "schedule_run_id": self.schedule_run_id, "sequence": self.sequence}  # fmt: skip

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d,
            cls.KIND,
            (
                "unit_id",
                "schedule_run_id",
                "sequence",
                "from_status",
                "to_status",
                "reason",
                "created_at",
            ),
        )
        return cls(
            v.get_str(d, "unit_id"),
            v.get_str(d, "schedule_run_id"),
            v.get_int(d, "sequence"),
            v.get_enum(d, "from_status", UnitState),
            v.get_enum(d, "to_status", UnitState),
            v.get_str(d, "reason"),
            v.get_time(d, "created_at"),
        )

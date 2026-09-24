"""Enumerations and state machines for the experiment scheduler (see docs/scheduler.md).

Every transition table here follows the same shape as `domain._EXPERIMENT_TRANSITIONS`: a mapping
from state to the frozenset of states it may legally move to. There is no other way to change a
scheduler entity's status.
"""

from collections.abc import Mapping
from enum import StrEnum


class UnitKind(StrEnum):
    """Which existing engine a scheduled unit dispatches to. The scheduler never re-implements any
    of these; it only calls the existing entry point (see `scheduler.dispatch`)."""

    BASELINE_EVALUATION = "BASELINE_EVALUATION"
    FAULT_EXPERIMENT = "FAULT_EXPERIMENT"
    INTERACTION = "INTERACTION"
    BENCHMARK = "BENCHMARK"
    STRESS = "STRESS"
    CALIBRATION = "CALIBRATION"
    RESOURCE = "RESOURCE"
    DRIFT = "DRIFT"
    DATA_QUALITY = "DATA_QUALITY"
    FAILURE_DISCOVERY = "FAILURE_DISCOVERY"
    RELIABILITY_PROFILE = "RELIABILITY_PROFILE"
    STATISTICAL_ANALYSIS = "STATISTICAL_ANALYSIS"
    REPRODUCTION = "REPRODUCTION"  # a reproduction attempt at another unit's evidence; never targets a SCHEDULE  # fmt: skip


# RESOURCE measures wall-clock/CPU/memory of the live machine: replaying it can reproduce the
# SCIENTIFIC outputs (percentiles, comparisons) only up to environment variance, never the exact
# timings. Every other kind is a deterministic computation given the same seeds and evidence.
ENVIRONMENT_DEPENDENT_KINDS = frozenset({UnitKind.RESOURCE})


class UnitState(StrEnum):
    PLANNED = "PLANNED"
    READY = "READY"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    SKIPPED = "SKIPPED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"
    RETRY_PENDING = "RETRY_PENDING"


_U = UnitState
UNIT_TRANSITIONS: Mapping[UnitState, frozenset[UnitState]] = {
    _U.PLANNED: frozenset({_U.READY, _U.BLOCKED, _U.CANCELLED}),
    _U.READY: frozenset({_U.RUNNING, _U.BLOCKED, _U.CANCELLED}),
    _U.RUNNING: frozenset({_U.SUCCEEDED, _U.FAILED, _U.TIMED_OUT, _U.CANCELLED}),
    _U.FAILED: frozenset({_U.RETRY_PENDING, _U.CANCELLED}),
    _U.TIMED_OUT: frozenset({_U.RETRY_PENDING, _U.CANCELLED}),
    _U.RETRY_PENDING: frozenset({_U.READY, _U.CANCELLED}),
    # BLOCKED is re-evaluated every scheduling pass: a retried dependency that later succeeds moves
    # its blocked dependents back to READY.
    _U.BLOCKED: frozenset({_U.READY, _U.SKIPPED, _U.CANCELLED}),
    _U.SUCCEEDED: frozenset(),
    _U.SKIPPED: frozenset(),
    _U.CANCELLED: frozenset(),
}
TERMINAL_UNIT_STATES = frozenset({_U.SUCCEEDED, _U.FAILED, _U.SKIPPED, _U.CANCELLED, _U.TIMED_OUT})
# States from which the scheduling loop still expects further progress (directly, or after a
# retry/dependency resolution moves them out of BLOCKED/RETRY_PENDING).
OPEN_UNIT_STATES = frozenset(UnitState) - frozenset({_U.SUCCEEDED, _U.SKIPPED, _U.CANCELLED})


class ScheduleRunState(StrEnum):
    PLANNED = "PLANNED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


_S = ScheduleRunState
SCHEDULE_RUN_TRANSITIONS: Mapping[ScheduleRunState, frozenset[ScheduleRunState]] = {
    _S.PLANNED: frozenset({_S.RUNNING, _S.CANCELLED}),
    _S.RUNNING: frozenset({_S.COMPLETED, _S.FAILED, _S.CANCELLED}),
    _S.COMPLETED: frozenset(),
    _S.FAILED: frozenset(),
    _S.CANCELLED: frozenset(),
}


class FailureCategory(StrEnum):
    """Why an attempt failed, coarse enough to decide retryability without guessing.

    Derived from real information the execution engine already records (`provenance.FailureStage`)
    or from the type of exception a dispatch raised before any run existed -- never invented."""

    TRANSIENT = "TRANSIENT"  # worth retrying: interrupted process, artifact I/O, lock contention
    CONFIGURATION = "CONFIGURATION"  # the definition is invalid; retrying cannot help
    UNSUPPORTED = "UNSUPPORTED"  # the requested capability is not available in this environment
    TIMEOUT = "TIMEOUT"  # the unit- or attempt-level timeout elapsed
    DATA_MODEL = "DATA_MODEL"  # the model/data raised during real execution


RETRYABLE_BY_DEFAULT = frozenset({FailureCategory.TRANSIENT, FailureCategory.TIMEOUT})


class RetryPolicyKind(StrEnum):
    NONE = "NONE"
    FIXED = "FIXED"

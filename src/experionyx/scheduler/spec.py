"""The typed, immutable, content-addressed definition of a schedule: a compact set of unit
definitions and the policy that governs their execution. Identity is the content hash of the
canonical form, so changing anything meaningful (a unit's kind, parameters, dependencies, retry
policy, resource requirement or timeout, or the execution policy) changes `spec_id`. Nothing here
executes anything; see `scheduler.graph.expand` and `scheduler.engine.run_schedule`."""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Self

import experionyx.validation as v
from experionyx.errors import ValidationError
from experionyx.hashing import HASH_PREFIX, content_hash
from experionyx.scheduler.taxonomy import FailureCategory, RetryPolicyKind, UnitKind

SPEC_SCHEMA_VERSION = 1
ENGINE_VERSION = "1.0.0"  # the scheduler's planning/execution methodology; bump when either changes
MAX_UNITS = 2000
MAX_ATTEMPTS = 20
_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_:.-]*")


def _closed(d: Mapping[str, Any], what: str, allowed: set[str]) -> None:
    extra = set(d) - allowed
    if extra:
        raise ValidationError(f"unexpected {what} field(s) {sorted(extra)}")


def _unit_key(field_name: str, value: object) -> str:
    text = v.text(field_name, value)
    if not _KEY.fullmatch(text):
        raise ValidationError(f"{field_name} must match [A-Za-z0-9][A-Za-z0-9_:.-]*, got {text!r}")
    return text


@dataclass(frozen=True)
class RetryPolicy:
    """How many times a unit may be re-attempted, and which failure categories qualify."""

    kind: RetryPolicyKind = RetryPolicyKind.NONE
    max_attempts: int = 1  # total attempts including the first
    retryable: tuple[FailureCategory, ...] = (FailureCategory.TRANSIENT, FailureCategory.TIMEOUT)

    def __post_init__(self) -> None:
        v.member("kind", self.kind, RetryPolicyKind)
        if not isinstance(self.max_attempts, int) or isinstance(self.max_attempts, bool):
            raise ValidationError("max_attempts must be an integer")
        if self.kind is RetryPolicyKind.NONE:
            if self.max_attempts != 1:
                raise ValidationError("a NONE retry policy must have max_attempts == 1")
        elif not 1 < self.max_attempts <= MAX_ATTEMPTS:
            raise ValidationError(f"FIXED retry policy needs 1 < max_attempts <= {MAX_ATTEMPTS}")
        retryable = tuple(sorted(set(self.retryable), key=lambda c: c.value))
        if self.kind is RetryPolicyKind.FIXED and not retryable:
            raise ValidationError("a FIXED retry policy needs at least one retryable category")
        object.__setattr__(self, "retryable", retryable)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "max_attempts": self.max_attempts,
            "retryable": [c.value for c in self.retryable],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        _closed(d, "retry policy", {"kind", "max_attempts", "retryable"})
        retryable = d.get("retryable", ["TRANSIENT", "TIMEOUT"])
        if not isinstance(retryable, list | tuple):
            raise ValidationError("retryable must be a list")
        return cls(
            RetryPolicyKind(str(d.get("kind", "NONE"))),
            int(d.get("max_attempts", 1)),  # type: ignore[call-overload]
            tuple(FailureCategory(str(c)) for c in retryable),
        )


NO_RETRY = RetryPolicy()


@dataclass(frozen=True)
class ResourceRequirement:
    """An explicitly DECLARED resource need. Advisory bookkeeping only: nothing here is enforced
    at the OS level, and a requirement the scheduler cannot validate is reported UNAVAILABLE, never
    assumed satisfied (see docs/scheduler.md)."""

    cpu_intensity: str = "LOW"  # LOW | MEDIUM | HIGH; informational
    memory_mb: int | None = None
    exclusive: bool = False  # no other unit runs while this one is RUNNING
    workload_size: int | None = None  # expected batch/sample count; informational

    def __post_init__(self) -> None:
        if self.cpu_intensity not in ("LOW", "MEDIUM", "HIGH"):
            raise ValidationError("cpu_intensity must be one of LOW, MEDIUM, HIGH")
        if self.memory_mb is not None and (
            isinstance(self.memory_mb, bool)
            or not isinstance(self.memory_mb, int)
            or self.memory_mb < 1
        ):
            raise ValidationError("memory_mb must be a positive integer")
        if not isinstance(self.exclusive, bool):
            raise ValidationError("exclusive must be a boolean")
        if self.workload_size is not None and (
            isinstance(self.workload_size, bool)
            or not isinstance(self.workload_size, int)
            or self.workload_size < 1
        ):
            raise ValidationError("workload_size must be a positive integer")

    def to_dict(self) -> dict[str, object]:
        return {
            "cpu_intensity": self.cpu_intensity,
            "memory_mb": self.memory_mb,
            "exclusive": self.exclusive,
            "workload_size": self.workload_size,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        _closed(
            d, "resource requirement", {"cpu_intensity", "memory_mb", "exclusive", "workload_size"}
        )
        return cls(
            str(d.get("cpu_intensity", "LOW")),
            None if d.get("memory_mb") is None else int(d["memory_mb"]),  # type: ignore[call-overload]
            bool(d.get("exclusive", False)),
            None if d.get("workload_size") is None else int(d["workload_size"]),  # type: ignore[call-overload]
        )


@dataclass(frozen=True)
class ExecutionPolicy:
    """How the DAG is executed. Never affects what a unit computes, only when/how it runs."""

    max_workers: int = 1
    default_timeout_seconds: float | None = None
    default_retry: RetryPolicy = NO_RETRY
    fail_fast: bool = False  # a FAILED (exhausted) unit blocks scheduling of new units
    memory_budget_mb: int | None = None  # advisory cap on the sum of concurrently RUNNING units'
    # declared memory_mb; None = not tracked (not "unlimited") -- see ResourceRequirement.

    def __post_init__(self) -> None:
        if not isinstance(self.max_workers, int) or isinstance(self.max_workers, bool) or not 1 <= self.max_workers <= 64:  # fmt: skip
            raise ValidationError("max_workers must be an integer in [1, 64]")
        if self.default_timeout_seconds is not None:
            v.finite_float("default_timeout_seconds", self.default_timeout_seconds)
            if self.default_timeout_seconds <= 0:
                raise ValidationError("default_timeout_seconds must be > 0")
        v.member("default_retry", self.default_retry, RetryPolicy)
        if not isinstance(self.fail_fast, bool):
            raise ValidationError("fail_fast must be a boolean")
        if self.memory_budget_mb is not None and (
            isinstance(self.memory_budget_mb, bool)
            or not isinstance(self.memory_budget_mb, int)
            or self.memory_budget_mb < 1
        ):
            raise ValidationError("memory_budget_mb must be a positive integer")

    def to_dict(self) -> dict[str, object]:
        return {
            "max_workers": self.max_workers,
            "default_timeout_seconds": self.default_timeout_seconds,
            "default_retry": self.default_retry.to_dict(),
            "fail_fast": self.fail_fast,
            "memory_budget_mb": self.memory_budget_mb,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        _closed(
            d,
            "execution policy",
            {
                "max_workers",
                "default_timeout_seconds",
                "default_retry",
                "fail_fast",
                "memory_budget_mb",
            },
        )
        retry = d.get("default_retry")
        return cls(
            int(d.get("max_workers", 1)),  # type: ignore[call-overload]
            None
            if d.get("default_timeout_seconds") is None
            else float(d["default_timeout_seconds"]),  # type: ignore[arg-type]
            NO_RETRY if not retry else RetryPolicy.from_dict(retry),  # type: ignore[arg-type]
            bool(d.get("fail_fast", False)),
            None if d.get("memory_budget_mb") is None else int(d["memory_budget_mb"]),  # type: ignore[call-overload]
        )


DEFAULT_POLICY = ExecutionPolicy()


@dataclass(frozen=True)
class UnitDef:
    """One compact unit definition inside a `ScheduleSpec`. `parameters` is the raw JSON body the
    dispatcher will hand to the target engine's own `Spec.from_dict` (see `scheduler.dispatch`);
    it may embed `"$dep:<key>"` sentinel strings in place of a run/analysis ID, resolved once the
    named dependency succeeds (see `scheduler.dispatch.substitute`)."""

    key: str
    kind: UnitKind
    parameters: Mapping[str, object] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    retry: RetryPolicy | None = None  # None: use the schedule's default_retry
    resources: ResourceRequirement | None = None
    timeout_seconds: float | None = None  # None: use the schedule's default_timeout_seconds
    priority: int = 0  # higher runs first among units that are simultaneously READY

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _unit_key("key", self.key))
        v.member("kind", self.kind, UnitKind)
        object.__setattr__(self, "parameters", v.freeze_mapping("parameters", self.parameters))
        deps = tuple(sorted({_unit_key("depends_on[]", d) for d in self.depends_on}))
        if self.key in deps:
            raise ValidationError(f"unit {self.key!r} cannot depend on itself")
        object.__setattr__(self, "depends_on", deps)
        if self.retry is not None:
            v.member("retry", self.retry, RetryPolicy)
        if self.resources is not None:
            v.member("resources", self.resources, ResourceRequirement)
        if self.timeout_seconds is not None:
            v.finite_float("timeout_seconds", self.timeout_seconds)
            if self.timeout_seconds <= 0:
                raise ValidationError("timeout_seconds must be > 0")
        if not isinstance(self.priority, int) or isinstance(self.priority, bool):
            raise ValidationError("priority must be an integer")

    def to_dict(self) -> dict[str, object]:
        from experionyx.domain import to_jsonable

        return {
            "key": self.key,
            "kind": self.kind.value,
            "parameters": to_jsonable(self.parameters),
            "depends_on": list(self.depends_on),
            "retry": None if self.retry is None else self.retry.to_dict(),
            "resources": None if self.resources is None else self.resources.to_dict(),
            "timeout_seconds": self.timeout_seconds,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        _closed(
            d,
            "unit",
            {
                "key",
                "kind",
                "parameters",
                "depends_on",
                "retry",
                "resources",
                "timeout_seconds",
                "priority",
            },
        )
        deps = d.get("depends_on", [])
        if not isinstance(deps, list | tuple):
            raise ValidationError("depends_on must be a list")
        params = d.get("parameters", {})
        if not isinstance(params, Mapping):
            raise ValidationError("parameters must be an object")
        retry, res = d.get("retry"), d.get("resources")
        return cls(
            str(d["key"]),
            UnitKind(str(d["kind"])),
            dict(params),
            tuple(str(k) for k in deps),
            None if retry is None else RetryPolicy.from_dict(retry),  # type: ignore[arg-type]
            None if res is None else ResourceRequirement.from_dict(res),  # type: ignore[arg-type]
            None if d.get("timeout_seconds") is None else float(d["timeout_seconds"]),  # type: ignore[arg-type]
            int(d.get("priority", 0)),  # type: ignore[call-overload]
        )


@dataclass(frozen=True)
class ScheduleSpec:
    """A compact schedule definition. `units` is expanded into explicit, persisted, dependency-
    ordered `ScheduleUnit` records by `scheduler.graph.expand` before anything runs."""

    name: str
    version: str  # the DEFINITION version (major.minor.patch); schedule runs keep the exact spec
    units: tuple[UnitDef, ...]
    policy: ExecutionPolicy = DEFAULT_POLICY
    schema_version: int = SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SPEC_SCHEMA_VERSION:
            raise ValidationError(f"unsupported schedule spec schema {self.schema_version}")
        v.text("name", self.name)
        if not re.fullmatch(r"\d+\.\d+\.\d+", self.version):
            raise ValidationError(
                f"schedule version must be major.minor.patch, got {self.version!r}"
            )
        if not self.units:
            raise ValidationError("a schedule needs at least one unit")
        if len(self.units) > MAX_UNITS:
            raise ValidationError(f"a schedule may define at most {MAX_UNITS} units")
        keys = [u.key for u in self.units]
        if len(set(keys)) != len(keys):
            dupes = sorted({k for k in keys if keys.count(k) > 1})
            raise ValidationError(f"duplicate unit key(s): {dupes}")
        unknown = sorted({d for u in self.units for d in u.depends_on if d not in keys})
        if unknown:
            raise ValidationError(f"unit(s) depend on unknown key(s): {unknown}")
        object.__setattr__(self, "units", tuple(sorted(self.units, key=lambda u: u.key)))
        v.member("policy", self.policy, ExecutionPolicy)

    def unit(self, key: str) -> UnitDef:
        for u in self.units:
            if u.key == key:
                return u
        raise ValidationError(f"no unit named {key!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "units": [u.to_dict() for u in self.units],
            "policy": self.policy.to_dict(),
            "schema_version": self.schema_version,
        }

    @property
    def spec_id(self) -> str:
        return "ssp_" + content_hash(self.to_dict())[len(HASH_PREFIX) :][:32]

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        _closed(d, "schedule spec", {"name", "version", "units", "policy", "schema_version"})
        units = d.get("units")
        if not isinstance(units, list | tuple) or not units:
            raise ValidationError("units must be a non-empty list")
        policy = d.get("policy")
        return cls(
            str(d["name"]),
            str(d["version"]),
            tuple(UnitDef.from_dict(u) for u in units),
            DEFAULT_POLICY if not policy else ExecutionPolicy.from_dict(policy),  # type: ignore[arg-type]
            int(d.get("schema_version", SPEC_SCHEMA_VERSION)),  # type: ignore[call-overload]
        )

"""Execution provenance: what was run, on what, from which source (see docs/provenance.md).

`Provenance` is written once, when a run starts, and records only inputs and context. Results of
an execution live in `RunOutcome`, written once when the run ends. Neither is ever updated.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import ClassVar, Self

import experionyx.validation as v
from experionyx.domain import (
    Artifact,
    ConfigurationRef,
    Entity,
    EnvironmentSnapshot,
    Experiment,
    Run,
    RunStatus,
    check_keys,
    open_payload,
    to_jsonable,
)
from experionyx.errors import ValidationError
from experionyx.hashing import content_hash

_COMMIT = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


class SourceState(StrEnum):
    """Whether the source that ran can be recovered from version control."""

    REPRODUCIBLE_SOURCE = "REPRODUCIBLE_SOURCE"  # clean worktree at `commit`
    MODIFIED_WORKTREE = "MODIFIED_WORKTREE"  # uncommitted/untracked changes on top of `commit`
    UNKNOWN = "UNKNOWN"  # not a git repo, git unavailable, or git failed


@dataclass(frozen=True)
class SourceRevision:
    state: SourceState
    commit: str | None = None
    branch: str | None = None  # informational; not part of the fingerprint

    def __post_init__(self) -> None:
        v.member("state", self.state, SourceState)
        if self.state is SourceState.UNKNOWN:
            if self.commit is not None or self.branch is not None:
                raise ValidationError("an UNKNOWN source must not carry a commit or branch")
        elif self.commit is None or not _COMMIT.fullmatch(self.commit):
            raise ValidationError(f"{self.state} requires a full lowercase hex commit SHA")
        if self.branch is not None:
            v.text("branch", self.branch)

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        check_keys(d, ("state", "commit", "branch"))
        return cls(
            v.get_enum(d, "state", SourceState),
            v.get_opt_str(d, "commit"),
            v.get_opt_str(d, "branch"),
        )


@dataclass(frozen=True)
class ResourceLimits:
    """Advisory settings recorded with the run. The executor does NOT enforce them."""

    max_workers: int = 1
    memory_limit_mb: int | None = None
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if v.non_negative_int("max_workers", self.max_workers) < 1:
            raise ValidationError("max_workers must be >= 1")
        if (
            self.memory_limit_mb is not None
            and v.non_negative_int("memory_limit_mb", self.memory_limit_mb) < 1
        ):
            raise ValidationError("memory_limit_mb must be >= 1")
        if (
            self.timeout_seconds is not None
            and v.finite_float("timeout_seconds", self.timeout_seconds) <= 0
        ):
            raise ValidationError("timeout_seconds must be > 0")

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        check_keys(d, ("max_workers", "memory_limit_mb", "timeout_seconds"))
        memory = v.get_raw(d, "memory_limit_mb")
        return cls(
            v.get_int(d, "max_workers"),
            None if memory is None else v.get_int(d, "memory_limit_mb"),
            v.get_opt_float(d, "timeout_seconds"),
        )


@dataclass(frozen=True)
class ExecutionParameters:
    """How a run was executed. `procedure` is the import path (`module:function`) of the code."""

    procedure: str
    resources: ResourceLimits = ResourceLimits()
    metadata: Mapping[str, object] = field(default_factory=dict)  # controlled extension point

    def __post_init__(self) -> None:
        v.text("procedure", self.procedure)
        v.member("resources", self.resources, ResourceLimits)
        object.__setattr__(self, "metadata", v.freeze_mapping("metadata", self.metadata))

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        check_keys(d, ("procedure", "resources", "metadata"))
        return cls(
            v.get_str(d, "procedure"),
            ResourceLimits.from_dict(v.get_mapping(d, "resources")),
            v.get_mapping(d, "metadata"),
        )


class FailureStage(StrEnum):
    PREPARATION = "PREPARATION"  # failed before the procedure started
    EXECUTION = "EXECUTION"  # the procedure raised
    ARTIFACT_REGISTRATION = "ARTIFACT_REGISTRATION"  # artifact missing/changed at completion
    INTERRUPTED = "INTERRUPTED"  # KeyboardInterrupt/SystemExit, or process died (recovered)


@dataclass(frozen=True)
class ErrorInfo:
    stage: FailureStage
    error_type: str  # qualified exception class name
    message: str  # may legitimately be empty
    diagnostic_artifact_id: str | None = None  # e.g. sanitized traceback artifact

    def __post_init__(self) -> None:
        v.member("stage", self.stage, FailureStage)
        v.text("error_type", self.error_type)
        if not isinstance(self.message, str):
            raise ValidationError("message must be a string")
        if self.diagnostic_artifact_id is not None:
            v.ref("diagnostic_artifact_id", self.diagnostic_artifact_id, Artifact.PREFIX)

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        check_keys(d, ("stage", "error_type", "message", "diagnostic_artifact_id"))
        return cls(
            v.get_enum(d, "stage", FailureStage),
            v.get_str(d, "error_type"),
            v.get_str(d, "message"),
            v.get_opt_str(d, "diagnostic_artifact_id"),
        )


@dataclass(frozen=True)
class Provenance(Entity):
    """Inputs and context of one run. One per run; identity is the run.

    Reproducibility-relevant fields participate in `fingerprint`; `run_id`, `replay_of`,
    `started_at` and `runtime` (informational) do not.
    """

    KIND: ClassVar[str] = "provenance"
    PREFIX: ClassVar[str] = "prv"
    run_id: str
    experiment_id: str
    environment_id: str
    dependency_digest: str
    configuration_id: str
    seed: int
    source: SourceRevision
    executor_version: str
    execution: ExecutionParameters
    started_at: datetime
    runtime: Mapping[str, object] = field(default_factory=dict)  # informational only
    replay_of: str | None = None  # set => this run is a REPLAY_REQUESTED of that run

    def __post_init__(self) -> None:
        v.ref("run_id", self.run_id, Run.PREFIX)
        v.ref("experiment_id", self.experiment_id, Experiment.PREFIX)
        v.ref("environment_id", self.environment_id, EnvironmentSnapshot.PREFIX)
        v.digest("dependency_digest", self.dependency_digest)
        v.ref("configuration_id", self.configuration_id, ConfigurationRef.PREFIX)
        v.non_negative_int("seed", self.seed)
        v.member("source", self.source, SourceRevision)
        v.version("executor_version", self.executor_version)
        v.member("execution", self.execution, ExecutionParameters)
        v.timestamp("started_at", self.started_at)
        object.__setattr__(self, "runtime", v.freeze_mapping("runtime", self.runtime))
        if self.replay_of is not None:
            v.ref("replay_of", self.replay_of, Run.PREFIX)
            if self.replay_of == self.run_id:
                raise ValidationError("a run cannot be a replay of itself")

    @property
    def is_replay(self) -> bool:
        return self.replay_of is not None

    def components(self) -> dict[str, object]:
        """The reproducibility-relevant inputs, by name. Basis of `fingerprint`."""
        return {
            "experiment": self.experiment_id,
            "source": {"state": self.source.state.value, "commit": self.source.commit},
            "environment": self.environment_id,
            "dependencies": self.dependency_digest,
            "configuration": self.configuration_id,
            "seed": self.seed,
            "executor": self.executor_version,
            "execution": to_jsonable(self.execution),
        }

    @property
    def fingerprint(self) -> str:
        return content_hash(self.components())

    def _identity(self) -> Mapping[str, object]:
        return {"run_id": self.run_id}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d,
            cls.KIND,
            (
                "run_id",
                "experiment_id",
                "environment_id",
                "dependency_digest",
                "configuration_id",
                "seed",
                "source",
                "executor_version",
                "execution",
                "started_at",
                "runtime",
                "replay_of",
            ),
        )
        return cls(
            run_id=v.get_str(d, "run_id"),
            experiment_id=v.get_str(d, "experiment_id"),
            environment_id=v.get_str(d, "environment_id"),
            dependency_digest=v.get_str(d, "dependency_digest"),
            configuration_id=v.get_str(d, "configuration_id"),
            seed=v.get_int(d, "seed"),
            source=SourceRevision.from_dict(v.get_mapping(d, "source")),
            executor_version=v.get_str(d, "executor_version"),
            execution=ExecutionParameters.from_dict(v.get_mapping(d, "execution")),
            started_at=v.get_time(d, "started_at"),
            runtime=v.get_mapping(d, "runtime"),
            replay_of=v.get_opt_str(d, "replay_of"),
        )


def compare_provenance(a: Provenance, b: Provenance) -> tuple[str, ...]:
    """Names of the reproducibility-relevant components that differ (empty => same fingerprint)."""
    ca, cb = a.components(), b.components()
    return tuple(k for k in ca if ca[k] != cb[k])


@dataclass(frozen=True)
class RunOutcome(Entity):
    """How a run ended. One per run; written atomically with the final status change."""

    KIND: ClassVar[str] = "outcome"
    PREFIX: ClassVar[str] = "out"
    run_id: str
    finished_at: datetime
    status: RunStatus
    duration_seconds: float | None  # None: unknown (e.g. recovered after the process died)
    observation_count: int
    artifact_ids: tuple[str, ...] = ()  # sorted, unique
    error: ErrorInfo | None = None

    def __post_init__(self) -> None:
        v.ref("run_id", self.run_id, Run.PREFIX)
        v.timestamp("finished_at", self.finished_at)
        if v.member("status", self.status, RunStatus) not in (
            RunStatus.COMPLETED,
            RunStatus.FAILED,
        ):
            raise ValidationError("an outcome status must be COMPLETED or FAILED")
        if self.duration_seconds is not None and (
            v.finite_float("duration_seconds", self.duration_seconds) < 0
        ):
            raise ValidationError("duration_seconds must be >= 0")
        v.non_negative_int("observation_count", self.observation_count)
        ids = tuple(self.artifact_ids)
        for i, a in enumerate(ids):
            v.ref(f"artifact_ids[{i}]", a, Artifact.PREFIX)
        if list(ids) != sorted(set(ids)):
            raise ValidationError("artifact_ids must be sorted and unique")
        object.__setattr__(self, "artifact_ids", ids)
        if (self.error is None) != (self.status is RunStatus.COMPLETED):
            raise ValidationError("error is required exactly when status is FAILED")
        if self.error is not None:
            v.member("error", self.error, ErrorInfo)
            diag = self.error.diagnostic_artifact_id
            if diag is not None and diag not in ids:
                raise ValidationError("diagnostic artifact must be listed in artifact_ids")

    def _identity(self) -> Mapping[str, object]:
        return {"run_id": self.run_id}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d,
            cls.KIND,
            (
                "run_id",
                "finished_at",
                "status",
                "duration_seconds",
                "observation_count",
                "artifact_ids",
                "error",
            ),
        )
        ids = v.get_raw(d, "artifact_ids")
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            raise ValidationError("artifact_ids must be a list of strings")
        error = v.get_raw(d, "error")
        return cls(
            run_id=v.get_str(d, "run_id"),
            finished_at=v.get_time(d, "finished_at"),
            status=v.get_enum(d, "status", RunStatus),
            duration_seconds=v.get_opt_float(d, "duration_seconds"),
            observation_count=v.get_int(d, "observation_count"),
            artifact_ids=tuple(ids),
            error=None if error is None else ErrorInfo.from_dict(v.get_mapping(d, "error")),
        )


@dataclass(frozen=True)
class ConfigDiff:
    added: tuple[str, ...]
    removed: tuple[str, ...]
    changed: tuple[str, ...]

    @property
    def identical(self) -> bool:
        return not (self.added or self.removed or self.changed)


def _flatten(prefix: str, value: object, out: dict[str, object]) -> None:
    if isinstance(value, Mapping) and value:
        for k, x in value.items():
            _flatten(f"{prefix}.{k}" if prefix else k, x, out)
    else:
        out[prefix] = value


def config_diff(a: ConfigurationRef, b: ConfigurationRef) -> ConfigDiff:
    """Compare two configurations by dotted key path (sequences are compared whole)."""
    fa: dict[str, object] = {}
    fb: dict[str, object] = {}
    _flatten("", a.parameters, fa)
    _flatten("", b.parameters, fb)
    return ConfigDiff(
        added=tuple(sorted(set(fb) - set(fa))),
        removed=tuple(sorted(set(fa) - set(fb))),
        changed=tuple(sorted(k for k in set(fa) & set(fb) if fa[k] != fb[k])),
    )


__all__ = [
    "ConfigDiff",
    "ErrorInfo",
    "ExecutionParameters",
    "FailureStage",
    "Provenance",
    "ResourceLimits",
    "RunOutcome",
    "SourceRevision",
    "SourceState",
    "compare_provenance",
    "config_diff",
]

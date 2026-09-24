"""The registry entity of one reproduction attempt. Immutable and content-addressed like every
other collect-style record in this project (`BenchmarkResult`, `GraphSnapshot`, ...); a
reproduction attempt never mutates the target it reproduces (see docs/reproducibility.md)."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Self

import experionyx.validation as v
from experionyx.domain import Entity, Investigation, Run, open_payload
from experionyx.errors import ValidationError
from experionyx.reproducibility.taxonomy import ComparisonOutcome, TargetKind


@dataclass(frozen=True)
class ReproductionAttempt(Entity):
    """One immutable, terminal outcome of one attempt at reproducing `target_id`. `attempt` numbers
    from 0 per target; a re-attempt NEVER overwrites an earlier one, it adds a new record -- the
    same convention as `scheduler.entities.ExecutionAttempt`."""

    KIND: ClassVar[str] = "reproduction_attempt"
    PREFIX: ClassVar[str] = "rpa"
    target_kind: TargetKind
    target_id: str
    spec_id: str
    spec: Mapping[str, object]
    attempt: int
    investigation_id: str
    run_id: str | None  # the ORIGINAL run this attempt reproduces (None when the target has no single representative run, e.g. a SCHEDULE)  # fmt: skip
    replay_run_id: str | None  # the new Run produced while reproducing, if one was produced
    replay_status: str | None
    secondary_ref: str | None  # a non-Run identifier the reproduction also produced (e.g. a ScheduleRun id)  # fmt: skip
    outcome: ComparisonOutcome
    sources_changed: bool  # persisted evidence changed since the target was collected
    differences: tuple[str, ...]  # artifact-name-level diffs from the underlying engine's own replay_check  # fmt: skip
    field_differences: tuple[Mapping[str, object], ...]  # FieldDifference.to_dict(), when NUMERIC_TOLERANCE re-compared something  # fmt: skip
    environment_diff: Mapping[str, object]
    artifact_verification: Mapping[str, object]
    note: str | None
    engine_version: str
    created_at: datetime

    def __post_init__(self) -> None:
        v.member("target_kind", self.target_kind, TargetKind)
        v.text("target_id", self.target_id)
        v.text("spec_id", self.spec_id)
        object.__setattr__(self, "spec", v.freeze_mapping("spec", self.spec))
        v.non_negative_int("attempt", self.attempt)
        v.ref("investigation_id", self.investigation_id, Investigation.PREFIX)
        if self.run_id is not None:
            v.ref("run_id", self.run_id, Run.PREFIX)
        if self.replay_run_id is not None:
            v.ref("replay_run_id", self.replay_run_id, Run.PREFIX)
        if self.replay_status is not None:
            v.text("replay_status", self.replay_status)
        if self.secondary_ref is not None:
            v.text("secondary_ref", self.secondary_ref)
        v.member("outcome", self.outcome, ComparisonOutcome)
        if not isinstance(self.sources_changed, bool):
            raise ValidationError("sources_changed must be a boolean")
        if not isinstance(self.differences, tuple) or not all(isinstance(x, str) for x in self.differences):  # fmt: skip
            raise ValidationError("differences must be a tuple of strings")
        object.__setattr__(
            self,
            "field_differences",
            tuple(
                v.freeze_mapping(f"field_differences[{i}]", d)
                for i, d in enumerate(self.field_differences)
            ),
        )
        object.__setattr__(self, "environment_diff", v.freeze_mapping("environment_diff", self.environment_diff))  # fmt: skip
        object.__setattr__(self, "artifact_verification", v.freeze_mapping("artifact_verification", self.artifact_verification))  # fmt: skip
        if self.note is not None:
            v.text("note", self.note)
        v.text("engine_version", self.engine_version)
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {"target_id": self.target_id, "attempt": self.attempt}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(
            d, cls.KIND,
            ("target_kind", "target_id", "spec_id", "spec", "attempt", "investigation_id",
             "run_id", "replay_run_id", "replay_status", "secondary_ref", "outcome", "sources_changed",
             "differences", "field_differences", "environment_diff", "artifact_verification",
             "note", "engine_version", "created_at"),
        )  # fmt: skip
        sources_changed = v.get_raw(d, "sources_changed")
        if not isinstance(sources_changed, bool):
            raise ValidationError("sources_changed must be a boolean")
        differences = v.get_raw(d, "differences")
        if not isinstance(differences, list) or not all(isinstance(x, str) for x in differences):
            raise ValidationError("differences must be a list of strings")
        field_differences = v.get_raw(d, "field_differences")
        if not isinstance(field_differences, list):
            raise ValidationError("field_differences must be a list")
        run_id = v.get_raw(d, "run_id")
        replay_run_id = v.get_raw(d, "replay_run_id")
        replay_status = v.get_raw(d, "replay_status")
        secondary_ref = v.get_raw(d, "secondary_ref")
        note = v.get_raw(d, "note")
        return cls(
            v.get_enum(d, "target_kind", TargetKind), v.get_str(d, "target_id"), v.get_str(d, "spec_id"),
            v.get_mapping(d, "spec"), v.get_int(d, "attempt"), v.get_str(d, "investigation_id"),
            None if run_id is None else str(run_id), None if replay_run_id is None else str(replay_run_id),
            None if replay_status is None else str(replay_status),
            None if secondary_ref is None else str(secondary_ref),
            v.get_enum(d, "outcome", ComparisonOutcome), sources_changed, tuple(differences),
            tuple(field_differences), v.get_mapping(d, "environment_diff"),
            v.get_mapping(d, "artifact_verification"), None if note is None else str(note),
            v.get_str(d, "engine_version"), v.get_time(d, "created_at"),
        )  # fmt: skip


__all__ = ["ReproductionAttempt"]

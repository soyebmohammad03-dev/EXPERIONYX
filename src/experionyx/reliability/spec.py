"""The request for a profile: a scope and the persisted sources it summarizes. Identity is the
content hash of the canonical form, so a different source set or scope is a different profile."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

import experionyx.validation as v
from experionyx.errors import ValidationError
from experionyx.hashing import HASH_PREFIX, content_hash
from experionyx.reliability.taxonomy import Scope

PROFILE_VERSION = "1.0.0"


def _ids(name: str, ids: tuple[str, ...], prefix: str) -> tuple[str, ...]:
    if isinstance(ids, str) or not all(isinstance(i, str) for i in ids):
        raise ValidationError(f"{name} must be a list of IDs")
    for i in ids:
        v.ref(name, i, prefix)
    return tuple(sorted(set(ids)))


@dataclass(frozen=True)
class ProfileSpec:
    scope: Scope
    baseline_run: str  # a COMPLETED baseline evaluation run (no fault): the profile's anchor
    fault_experiments: tuple[str, ...] = ()
    interactions: tuple[str, ...] = ()
    failure_modes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        v.member("scope", self.scope, Scope)
        v.ref("baseline_run", self.baseline_run, "run")
        object.__setattr__(
            self, "fault_experiments", _ids("fault_experiments", self.fault_experiments, "fxp")
        )
        object.__setattr__(self, "interactions", _ids("interactions", self.interactions, "ian"))
        object.__setattr__(self, "failure_modes", _ids("failure_modes", self.failure_modes, "fmd"))

    def to_dict(self) -> dict[str, object]:
        return {
            "scope": self.scope.value,
            "baseline_run": self.baseline_run,
            "fault_experiments": list(self.fault_experiments),
            "interactions": list(self.interactions),
            "failure_modes": list(self.failure_modes),
            "profile_version": PROFILE_VERSION,
        }

    @property
    def spec_id(self) -> str:
        return "rsp_" + content_hash(self.to_dict())[len(HASH_PREFIX) :][:32]

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        extra = set(d) - {
            "scope",
            "baseline_run",
            "fault_experiments",
            "interactions",
            "failure_modes",
            "profile_version",
        }
        if extra or "scope" not in d or "baseline_run" not in d:
            raise ValidationError(
                f"malformed profile spec (unexpected {sorted(extra)}; scope and baseline_run are required)"
            )
        try:
            scope = Scope(str(d["scope"]))
        except ValueError:
            raise ValidationError(
                f"unknown scope {d['scope']!r}; choose from {[s.value for s in Scope]}"
            ) from None

        def ids(k: str) -> tuple[str, ...]:
            x = d.get(k, [])
            if not isinstance(x, list | tuple):
                raise ValidationError(f"{k} must be a list")
            return tuple(str(i) for i in x)

        return cls(
            scope,
            str(d["baseline_run"]),
            ids("fault_experiments"),
            ids("interactions"),
            ids("failure_modes"),
        )

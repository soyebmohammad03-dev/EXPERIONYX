"""Typed check results. One result is one check on one scope (a split, a slice within a split, a
group pair). There is no quality score and no overall verdict: each result keeps the observed
measurements, the configured thresholds, any statistical evidence and its interpretation apart.

PASS           no configured expectation was violated (observations are still recorded)
FAIL           a configured rule was violated (a declared type, range, allowed set, threshold, ...)
WARNING        a phenomenon that needs interpretation was observed (duplicates, identifier-like
               columns, non-finite values, leakage indicators, unseen categories, ...)
INCONCLUSIVE   too little valid data to say
UNAVAILABLE    the data the check needs does not exist
NOT_APPLICABLE the check does not apply to this scope"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from experionyx.domain import to_jsonable

NOTE = "a measurement under a configured expectation; it does not say the data is good or bad, why it looks this way, or what it does to a model"  # fmt: skip


class Status(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    INCONCLUSIVE = "INCONCLUSIVE"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


_ORDER = (Status.FAIL, Status.WARNING, Status.INCONCLUSIVE, Status.UNAVAILABLE, Status.NOT_APPLICABLE, Status.PASS)  # fmt: skip


def worst(statuses: Sequence[Status]) -> Status:
    """The most attention-needing status of several sub-results of ONE check (per feature, ...)."""
    return next((s for s in _ORDER if s in statuses), Status.NOT_APPLICABLE)


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    check_type: str
    scope: Mapping[str, Any]
    status: Status
    reason: str | None = None
    observations: Mapping[str, Any] = field(default_factory=dict)
    violations: tuple[Mapping[str, Any], ...] = ()
    thresholds: Mapping[str, Any] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)  # counts the result rests on
    statistics: Mapping[str, Any] | None = None  # Phase 10 blocks, kept apart from observations
    note: str = NOTE

    @property
    def scope_key(self) -> str:
        return (
            " | ".join(
                f"{k}={self.scope[k]}" for k in sorted(self.scope) if self.scope[k] is not None
            )
            or "dataset"
        )

    def to_dict(self) -> dict[str, Any]:
        d = to_jsonable(self)
        assert isinstance(d, dict)  # noqa: S101
        return d


def viol(
    rule: str, subject: str, ids: Sequence[object], examples: int, **detail: Any
) -> dict[str, Any]:
    """A violation with its exact count and (up to `examples`) affected sample IDs."""
    shown = sorted(ids, key=lambda x: (isinstance(x, str), x))[:examples]
    return {"rule": rule, "subject": subject, "count": len(ids), "affected_rows": shown, "truncated": len(ids) > len(shown), **detail}  # fmt: skip

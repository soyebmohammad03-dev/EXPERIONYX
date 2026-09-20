"""Typed result documents. Every result carries an explicit evidence status and its sample counts;
none carries a score, a rank, or a causal claim.

OBSERVED  a value read from the data (counts, per-window metric values)
DERIVED   computed from observations by a stated method (distances, tests, differences)
INCONCLUSIVE           the requested statistic exists but the data cannot support it
UNDEFINED              the statistic does not exist for this input (reason given)
UNAVAILABLE            the data needed does not exist (reason given)
INSUFFICIENT_EVIDENCE  too few valid samples for inference; descriptive values are still shown"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from experionyx.domain import to_jsonable

NOTE = (
    "an observed difference between the distributions of two sets of samples; it does not say why "
    "they differ, whether the model's behaviour changed because of it, or that the change is "
    "concept drift"
)


class Evidence(StrEnum):
    OBSERVED = "OBSERVED"
    DERIVED = "DERIVED"
    INCONCLUSIVE = "INCONCLUSIVE"
    UNDEFINED = "UNDEFINED"
    UNAVAILABLE = "UNAVAILABLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class Counts:
    """What became of the values of one window: every value is accounted for."""

    n_total: int
    n_valid: int
    n_missing: int  # None or NaN
    n_nonfinite: int  # +/- infinity
    n_invalid: int  # present but not representable as the declared type


@dataclass(frozen=True)
class DistributionShiftResult:
    """One comparison of a value's distribution between a reference and a comparison window. The
    same type serves covariate (per feature), label, prediction and confidence shift."""

    dimension: str  # COVARIATE | LABEL | PREDICTION | CONFIDENCE
    subject: str  # feature name, or target / predicted / confidence
    kind: str  # NUMERIC | CATEGORICAL | BOOLEAN
    status: Evidence
    reference: Counts
    comparison: Counts
    method: str
    measures: Mapping[str, Any] = field(default_factory=dict)
    test: Mapping[str, Any] | None = (
        None  # the one hypothesis test of this result (the family member)
    )
    multiplicity: Mapping[str, Any] | None = None
    warnings: tuple[str, ...] = ()
    reason: str | None = None
    note: str = NOTE

    def to_dict(self) -> dict[str, Any]:
        d = to_jsonable(self)
        assert isinstance(d, dict)  # noqa: S101
        return d


@dataclass(frozen=True)
class PerformanceDriftResult:
    """Window-specific metric values (OBSERVED) and their differences (DERIVED). The comparison
    window's value minus the reference's, with each metric's own direction."""

    task: str
    measure: str  # the per-sample measure used for inference (accuracy | mae)
    status: Evidence
    reference: Mapping[str, Any]
    comparison: Mapping[str, Any]
    metrics: tuple[Mapping[str, Any], ...] = ()
    inference: Mapping[str, Any] | None = None  # unpaired comparison of the per-sample measure
    multiplicity: Mapping[str, Any] | None = None
    reason: str | None = None
    note: str = "windows contain different samples, so this is an unpaired, descriptive comparison of two sample sets; a worse metric is not shown to be caused by any change in the inputs"  # fmt: skip

    def to_dict(self) -> dict[str, Any]:
        d = to_jsonable(self)
        assert isinstance(d, dict)  # noqa: S101
        return d


@dataclass(frozen=True)
class DriftEvidence:
    """The exact inputs behind one window pair's results, so each result can be traced and replayed."""

    pair: str
    reference_window_id: str
    comparison_window_id: str
    reference_digest: str  # over the sorted member sample IDs
    comparison_digest: str
    n_reference: int
    n_comparison: int
    dataset_fingerprint: str
    baseline_run_id: str
    ordering: str

    def to_dict(self) -> dict[str, Any]:
        d = to_jsonable(self)
        assert isinstance(d, dict)  # noqa: S101
        return d

"""Failure vocabulary: EXPERIONYX *diagnostic* categories (not universal scientific truths),
signal kinds, lifecycle states and the validated transitions between them."""

from collections.abc import Mapping
from enum import StrEnum


class FailureCategory(StrEnum):
    """Diagnostic categories used to organize evidence. They are a working vocabulary of this
    project, may be extended, and make no claim to be a universally valid classification."""

    INPUT_SENSITIVITY = "INPUT_SENSITIVITY"
    DISTRIBUTION_SHIFT = "DISTRIBUTION_SHIFT"  # reserved: no shift faults exist yet
    LABEL_ERROR = "LABEL_ERROR"
    CLASS_SPECIFIC_ERROR = "CLASS_SPECIFIC_ERROR"
    FEATURE_DEPENDENCY = "FEATURE_DEPENDENCY"
    CALIBRATION_FAILURE = "CALIBRATION_FAILURE"
    HIGH_CONFIDENCE_ERROR = "HIGH_CONFIDENCE_ERROR"
    LOW_CONFIDENCE_ERROR = "LOW_CONFIDENCE_ERROR"
    SYSTEMATIC_MISCLASSIFICATION = "SYSTEMATIC_MISCLASSIFICATION"
    REGRESSION_ERROR = "REGRESSION_ERROR"
    PERFORMANCE_DEGRADATION = "PERFORMANCE_DEGRADATION"
    RESOURCE_SENSITIVITY = "RESOURCE_SENSITIVITY"  # reserved: no resource faults yet
    UNKNOWN = "UNKNOWN"


class SignalKind(StrEnum):
    """What a normalized failure signal measures (its provenance in the source analysis)."""

    PER_CLASS_RECALL_LOW = "PER_CLASS_RECALL_LOW"  # evaluation: a class's recall is low
    CONFUSION_PAIR = "CONFUSION_PAIR"  # evaluation: true class a is often predicted as b
    HIGH_CONFIDENCE_ERRORS = "HIGH_CONFIDENCE_ERRORS"
    CALIBRATION_ECE = "CALIBRATION_ECE"
    REGRESSION_TAIL = "REGRESSION_TAIL"  # max |residual| far above the typical residual
    REGRESSION_BIAS = "REGRESSION_BIAS"  # signed residuals systematically one-sided
    SLICE_DEGRADATION = "SLICE_DEGRADATION"  # a slice's metric worse than overall
    METRIC_DEGRADATION = "METRIC_DEGRADATION"  # fault experiment: a metric got worse
    CLASS_RECALL_DEGRADATION = "CLASS_RECALL_DEGRADATION"  # fault: a class's recall fell
    ECE_INCREASE = "ECE_INCREASE"  # fault: calibration error rose


class Direction(StrEnum):
    WORSE = "WORSE"
    BETTER = "BETTER"
    NEUTRAL = "NEUTRAL"


class FailureStatus(StrEnum):
    """Lifecycle. FAILURE_SIGNAL and FAILURE_CLUSTER are separate records; a mode starts as
    DISCOVERED/CANDIDATE and is never CONFIRMED automatically."""

    DISCOVERED = "DISCOVERED"  # a cluster exists but does not meet candidate criteria
    CANDIDATE = "CANDIDATE"  # meets the configured candidate evidence requirements
    SUPPORTED = "SUPPORTED"  # meets the stricter supported requirements
    CONFIRMED = "CONFIRMED"  # supported AND reproduced AND explicitly confirmed by a person
    DEPRECATED = "DEPRECATED"
    REJECTED = "REJECTED"


FAILURE_TRANSITIONS: Mapping[FailureStatus, frozenset[FailureStatus]] = {
    FailureStatus.DISCOVERED: frozenset({FailureStatus.CANDIDATE, FailureStatus.REJECTED}),
    FailureStatus.CANDIDATE: frozenset({FailureStatus.SUPPORTED, FailureStatus.REJECTED, FailureStatus.DEPRECATED}),
    FailureStatus.SUPPORTED: frozenset({FailureStatus.CONFIRMED, FailureStatus.CANDIDATE, FailureStatus.REJECTED, FailureStatus.DEPRECATED}),
    FailureStatus.CONFIRMED: frozenset({FailureStatus.DEPRECATED}),
    FailureStatus.DEPRECATED: frozenset(),
    FailureStatus.REJECTED: frozenset(),
}  # fmt: skip


class EvidenceKind(StrEnum):
    SIGNAL = "SIGNAL"
    RUN = "RUN"
    ARTIFACT = "ARTIFACT"
    OBSERVATION = "OBSERVATION"
    REPRODUCTION = "REPRODUCTION"
    CLAIM = "CLAIM"
    TRANSITION = "TRANSITION"  # a recorded lifecycle change (who, why, criteria met)


class NodeKind(StrEnum):
    MODEL = "MODEL"
    DATASET = "DATASET"
    FAULT = "FAULT"
    FAILURE_MODE = "FAILURE_MODE"
    CLASS = "CLASS"
    SLICE = "SLICE"
    EVIDENCE = "EVIDENCE"
    RUN = "RUN"


class Predicate(StrEnum):
    EXHIBITS = "EXHIBITS"
    EXPOSES = "EXPOSES"
    REVEALS = "REVEALS"
    AFFECTS = "AFFECTS"
    SUPPORTED_BY = "SUPPORTED_BY"
    REPRODUCED_BY = "REPRODUCED_BY"
    CO_OCCURS_WITH = "CO_OCCURS_WITH"  # observed co-occurrence: NOT an interaction
    INTERACTS_WITH = "INTERACTS_WITH"  # reserved for a future, explicitly designed analysis


ALLOWED_RELATIONSHIPS: frozenset[tuple[NodeKind, Predicate, NodeKind]] = frozenset(
    {
        (NodeKind.MODEL, Predicate.EXHIBITS, NodeKind.FAILURE_MODE),
        (NodeKind.DATASET, Predicate.EXPOSES, NodeKind.FAILURE_MODE),
        (NodeKind.FAULT, Predicate.REVEALS, NodeKind.FAILURE_MODE),
        (NodeKind.FAILURE_MODE, Predicate.AFFECTS, NodeKind.CLASS),
        (NodeKind.FAILURE_MODE, Predicate.AFFECTS, NodeKind.SLICE),
        (NodeKind.FAILURE_MODE, Predicate.SUPPORTED_BY, NodeKind.EVIDENCE),
        (NodeKind.FAILURE_MODE, Predicate.REPRODUCED_BY, NodeKind.RUN),
        (NodeKind.FAILURE_MODE, Predicate.CO_OCCURS_WITH, NodeKind.FAILURE_MODE),
        (NodeKind.FAILURE_MODE, Predicate.INTERACTS_WITH, NodeKind.FAILURE_MODE),
    }
)

"""Capabilities, task types and devices: the vocabulary adapters use to describe themselves.

Adding a member to any of these enums is backward compatible: existing adapters simply do not
declare it. Callers must check `supports()` (or handle UnsupportedCapabilityError) before use.
"""

from dataclasses import dataclass
from enum import StrEnum

import experionyx.validation as v
from experionyx.errors import ValidationError


class ModelCapability(StrEnum):
    PREDICT = "PREDICT"
    BATCH_PREDICT = "BATCH_PREDICT"
    PREDICT_PROBA = "PREDICT_PROBA"


class DatasetCapability(StrEnum):
    RANDOM_ACCESS = "RANDOM_ACCESS"
    ITERATION = "ITERATION"
    BATCHING = "BATCHING"
    LABELS = "LABELS"
    TRAIN_SPLIT = "TRAIN_SPLIT"
    VALIDATION_SPLIT = "VALIDATION_SPLIT"
    TEST_SPLIT = "TEST_SPLIT"


class TaskType(StrEnum):
    """Small extensible taxonomy. UNKNOWN means 'not safely inferable'; CUSTOM means 'known, but
    outside the named tasks'."""

    CLASSIFICATION = "CLASSIFICATION"
    REGRESSION = "REGRESSION"
    MULTILABEL_CLASSIFICATION = "MULTILABEL_CLASSIFICATION"
    IMAGE_CLASSIFICATION = "IMAGE_CLASSIFICATION"
    TIME_SERIES = "TIME_SERIES"
    ANOMALY_DETECTION = "ANOMALY_DETECTION"
    CUSTOM = "CUSTOM"
    UNKNOWN = "UNKNOWN"


class DeviceKind(StrEnum):
    CPU = "CPU"
    MPS = "MPS"
    CUDA = "CUDA"


@dataclass(frozen=True)
class DeviceInfo:
    """What was asked for and what was used. Policy: CPU unless explicitly requested; an
    unavailable request raises DeviceUnavailableError rather than falling back."""

    requested: DeviceKind
    resolved: DeviceKind

    def __post_init__(self) -> None:
        v.member("requested", self.requested, DeviceKind)
        v.member("resolved", self.resolved, DeviceKind)
        if self.requested is not self.resolved:
            raise ValidationError("resolved device must equal the requested device (no fallback)")

    @classmethod
    def cpu(cls) -> "DeviceInfo":
        return cls(DeviceKind.CPU, DeviceKind.CPU)

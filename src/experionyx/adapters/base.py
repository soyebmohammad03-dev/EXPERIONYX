"""Adapter protocols and shared result types. No ML framework is imported here.

Protocols describe what the execution engine and analyses rely on. `BaseModelAdapter` and
`BaseDatasetAdapter` are optional conveniences: they implement capability checking and make
unsupported operations raise `UnsupportedCapabilityError`, so adapters only implement what they
actually support.
"""

from abc import ABC
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Protocol, Self

from experionyx.adapters.capabilities import (
    DatasetCapability,
    DeviceInfo,
    DeviceKind,
    ModelCapability,
)
from experionyx.adapters.metadata import DatasetMetadata, ModelMetadata
from experionyx.errors import UnsupportedCapabilityError

Inputs = object  # framework-native batch of inputs (array, tensor, ...); adapters validate it
SampleId = int | str


@dataclass(frozen=True)
class AdapterInfo:
    kind: str  # "model" | "dataset"
    name: str
    version: str
    framework: str
    capabilities: tuple[str, ...]  # what the adapter *can* support; instances may support fewer
    description: str = ""


@dataclass(frozen=True)
class InferenceResult:
    """Standardized inference output. Outputs are plain nested tuples (no framework objects).
    No metrics are computed here; evaluation consumes these results."""

    outputs: tuple[object, ...]  # one entry per sample: a scalar or a (nested) tuple
    capability: ModelCapability  # which operation produced it
    sample_count: int
    batch_count: int
    batch_size: int | None  # requested batch size, None for a single call
    inference_seconds: float  # monotonic wall time of inference only (excludes model load)
    batch_seconds: tuple[float, ...]  # per-batch timings; a single measurement is not a benchmark
    model_fingerprint: str
    adapter: str
    adapter_version: str
    device: DeviceInfo
    sample_ids: tuple[SampleId, ...] | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Sample:
    index: int
    inputs: object
    target: object | None = None


@dataclass(frozen=True)
class Batch:
    indices: tuple[int, ...]  # dataset indices, in order; usable as sample_ids
    inputs: object
    target: object | None = None


class ModelAdapter(Protocol):
    NAME: ClassVar[str]
    VERSION: ClassVar[str]
    FRAMEWORK: ClassVar[str]

    device: DeviceInfo
    load_seconds: float  # cold-start cost of loading; separate from inference time

    @classmethod
    def info(cls) -> AdapterInfo: ...

    @classmethod
    def load(
        cls,
        source: str | Path,
        *,
        version: str,
        device: DeviceKind,
        options: Mapping[str, object],
    ) -> Self: ...

    @property
    def capabilities(self) -> frozenset[ModelCapability]: ...

    def metadata(self) -> ModelMetadata: ...

    def fingerprint(self) -> str: ...

    def predict(
        self, inputs: Inputs, *, sample_ids: Sequence[SampleId] | None = None
    ) -> InferenceResult: ...

    def batch_predict(
        self, inputs: Inputs, batch_size: int, *, sample_ids: Sequence[SampleId] | None = None
    ) -> InferenceResult: ...

    def predict_proba(
        self, inputs: Inputs, *, sample_ids: Sequence[SampleId] | None = None
    ) -> InferenceResult: ...


class DatasetAdapter(Protocol):
    NAME: ClassVar[str]
    VERSION: ClassVar[str]
    FRAMEWORK: ClassVar[str]

    @classmethod
    def info(cls) -> AdapterInfo: ...

    @classmethod
    def load(cls, source: str | Path, *, version: str, options: Mapping[str, object]) -> Self: ...

    @property
    def capabilities(self) -> frozenset[DatasetCapability]: ...

    def metadata(self, *, deep: bool = False) -> DatasetMetadata: ...

    def fingerprint(self) -> str: ...

    def splits(self) -> tuple[str, ...]: ...

    def num_samples(self, split: str | None = None) -> int: ...

    def sample(self, index: int, split: str | None = None) -> Sample: ...

    def batches(self, batch_size: int, split: str | None = None) -> Iterator[Batch]: ...


class BaseModelAdapter(ABC):  # abstract only by convention: shared behaviour
    NAME: ClassVar[str]
    VERSION: ClassVar[str]
    FRAMEWORK: ClassVar[str]
    POSSIBLE_CAPABILITIES: ClassVar[frozenset[ModelCapability]]
    DESCRIPTION: ClassVar[str] = ""

    @classmethod
    def info(cls) -> AdapterInfo:
        return AdapterInfo(
            "model",
            cls.NAME,
            cls.VERSION,
            cls.FRAMEWORK,
            tuple(sorted(c.value for c in cls.POSSIBLE_CAPABILITIES)),
            cls.DESCRIPTION,
        )

    @property
    def capabilities(self) -> frozenset[ModelCapability]:
        raise NotImplementedError

    def supports(self, capability: ModelCapability) -> bool:
        return capability in self.capabilities

    def require(self, capability: ModelCapability) -> None:
        if not self.supports(capability):
            raise UnsupportedCapabilityError(
                f"{self.NAME} model does not support {capability} "
                f"(supports: {sorted(c.value for c in self.capabilities)})"
            )

    def predict(
        self, inputs: Inputs, *, sample_ids: Sequence[SampleId] | None = None
    ) -> InferenceResult:
        raise UnsupportedCapabilityError(f"{self.NAME} does not implement predict")

    def batch_predict(
        self, inputs: Inputs, batch_size: int, *, sample_ids: Sequence[SampleId] | None = None
    ) -> InferenceResult:
        raise UnsupportedCapabilityError(f"{self.NAME} does not implement batch_predict")

    def predict_proba(
        self, inputs: Inputs, *, sample_ids: Sequence[SampleId] | None = None
    ) -> InferenceResult:
        raise UnsupportedCapabilityError(f"{self.NAME} does not implement predict_proba")


class BaseDatasetAdapter(ABC):
    NAME: ClassVar[str]
    VERSION: ClassVar[str]
    FRAMEWORK: ClassVar[str]
    POSSIBLE_CAPABILITIES: ClassVar[frozenset[DatasetCapability]]
    DESCRIPTION: ClassVar[str] = ""

    @classmethod
    def info(cls) -> AdapterInfo:
        return AdapterInfo(
            "dataset",
            cls.NAME,
            cls.VERSION,
            cls.FRAMEWORK,
            tuple(sorted(c.value for c in cls.POSSIBLE_CAPABILITIES)),
            cls.DESCRIPTION,
        )

    @property
    def capabilities(self) -> frozenset[DatasetCapability]:
        raise NotImplementedError

    def supports(self, capability: DatasetCapability) -> bool:
        return capability in self.capabilities

    def require(self, capability: DatasetCapability) -> None:
        if not self.supports(capability):
            raise UnsupportedCapabilityError(
                f"{self.NAME} dataset does not support {capability} "
                f"(supports: {sorted(c.value for c in self.capabilities)})"
            )

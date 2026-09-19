"""Tiny real adapters with no ML framework: a stand-in for a third-party adapter package.
They exercise the protocols, registry, executor and contract suites without sklearn/torch."""

import json
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import ClassVar, Self

from experionyx.adapters.base import (
    BaseDatasetAdapter,
    BaseModelAdapter,
    Batch,
    InferenceResult,
    Inputs,
    Sample,
    SampleId,
)
from experionyx.adapters.batching import batch_slices
from experionyx.adapters.capabilities import (
    DatasetCapability,
    DeviceInfo,
    DeviceKind,
    ModelCapability,
    TaskType,
)
from experionyx.adapters.metadata import DatasetMetadata, ModelMetadata
from experionyx.adapters.schema import Modality, TensorSchema
from experionyx.artifacts import sha256_file
from experionyx.errors import (
    DatasetLoadError,
    DeviceUnavailableError,
    InferenceError,
    InvalidDatasetError,
    ModelLoadError,
)
from experionyx.hashing import content_hash


class ConstantModelAdapter(BaseModelAdapter):
    """A 'model' stored as JSON `{"value": <number>}` that predicts that value for every row."""

    NAME: ClassVar[str] = "constant"
    VERSION: ClassVar[str] = "1.0.0"
    FRAMEWORK: ClassVar[str] = "pure-python"
    POSSIBLE_CAPABILITIES: ClassVar[frozenset[ModelCapability]] = frozenset(
        {ModelCapability.PREDICT, ModelCapability.BATCH_PREDICT}
    )

    def __init__(
        self, value: float, fingerprint: str, size: int, version: str, seconds: float
    ) -> None:
        self._value, self._fp, self._size, self._version = value, fingerprint, size, version
        self.load_seconds = seconds
        self.device = DeviceInfo.cpu()

    @classmethod
    def load(
        cls, source: str | Path, *, version: str, device: DeviceKind, options: Mapping[str, object]
    ) -> Self:
        if device is not DeviceKind.CPU:
            raise DeviceUnavailableError("constant adapter is CPU only")
        started = time.perf_counter()
        try:
            digest, size = sha256_file(Path(source))
            value = json.loads(Path(source).read_text())["value"]
        except (OSError, ValueError, KeyError) as exc:
            raise ModelLoadError(f"cannot load {source}: {exc}") from exc
        return cls(float(value), digest, size, version, time.perf_counter() - started)

    @property
    def capabilities(self) -> frozenset[ModelCapability]:
        return self.POSSIBLE_CAPABILITIES

    def fingerprint(self) -> str:
        return self._fp

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            adapter=self.NAME,
            adapter_version=self.VERSION,
            framework=self.FRAMEWORK,
            model_type="ConstantModel",
            version=self._version,
            fingerprint=self._fp,
            task=TaskType.REGRESSION,
            input_schema=TensorSchema(Modality.TABULAR, shape=(None, None)),
            serialization_format="json",
            size_bytes=self._size,
            capabilities=tuple(self.capabilities),
        )

    def _rows(self, inputs: Inputs) -> list[object]:
        if (
            not isinstance(inputs, list)
            or not inputs
            or not all(isinstance(r, list) for r in inputs)
        ):
            raise InferenceError("inputs must be a non-empty list of rows")
        return list(inputs)

    def _result(
        self,
        cap: ModelCapability,
        rows: list[object],
        timings: Sequence[float],
        batch_size: int | None,
        ids: Sequence[SampleId] | None,
    ) -> InferenceResult:
        if ids is not None and len(ids) != len(rows):
            raise InferenceError("sample id count mismatch")
        return InferenceResult(
            outputs=tuple(self._value for _ in rows), capability=cap, sample_count=len(rows),
            batch_count=len(timings), batch_size=batch_size,
            inference_seconds=sum(timings), batch_seconds=tuple(timings),
            model_fingerprint=self._fp, adapter=self.NAME, adapter_version=self.VERSION,
            device=self.device, sample_ids=None if ids is None else tuple(ids),
        )  # fmt: skip

    def predict(
        self, inputs: Inputs, *, sample_ids: Sequence[SampleId] | None = None
    ) -> InferenceResult:
        rows = self._rows(inputs)
        started = time.perf_counter()
        return self._result(
            ModelCapability.PREDICT, rows, (time.perf_counter() - started,), None, sample_ids
        )

    def batch_predict(
        self, inputs: Inputs, batch_size: int, *, sample_ids: Sequence[SampleId] | None = None
    ) -> InferenceResult:
        rows = self._rows(inputs)
        timings = []
        for chunk in batch_slices(len(rows), batch_size):
            started = time.perf_counter()
            _ = rows[chunk]
            timings.append(time.perf_counter() - started)
        return self._result(ModelCapability.BATCH_PREDICT, rows, timings, batch_size, sample_ids)


class ListDatasetAdapter(BaseDatasetAdapter):
    """A dataset stored as JSON `{"rows": [[...]], "targets": [...], "splits": {"test": [idx]}}`."""

    NAME: ClassVar[str] = "list"
    VERSION: ClassVar[str] = "1.0.0"
    FRAMEWORK: ClassVar[str] = "pure-python"
    POSSIBLE_CAPABILITIES: ClassVar[frozenset[DatasetCapability]] = frozenset(DatasetCapability)

    def __init__(self, payload: Mapping[str, object], version: str) -> None:
        rows, targets = payload["rows"], payload.get("targets")
        if not isinstance(rows, list) or not rows:
            raise InvalidDatasetError("rows must be a non-empty list")
        if targets is not None and (not isinstance(targets, list) or len(targets) != len(rows)):
            raise InvalidDatasetError("targets must match rows")
        self._rows, self._targets, self._version = rows, targets, version
        raw = payload.get("splits", {})
        self._splits = {k: list(v) for k, v in raw.items()} if isinstance(raw, dict) else {}
        self._payload = dict(payload)

    @classmethod
    def load(cls, source: str | Path, *, version: str, options: Mapping[str, object]) -> Self:
        try:
            return cls(json.loads(Path(source).read_text()), version)
        except (OSError, ValueError, KeyError) as exc:
            raise DatasetLoadError(f"cannot load {source}: {exc}") from exc

    @property
    def capabilities(self) -> frozenset[DatasetCapability]:
        caps = {
            DatasetCapability.RANDOM_ACCESS,
            DatasetCapability.ITERATION,
            DatasetCapability.BATCHING,
        }
        if self._targets is not None:
            caps.add(DatasetCapability.LABELS)
        if "test" in self._splits:
            caps.add(DatasetCapability.TEST_SPLIT)
        return frozenset(caps)

    def fingerprint(self) -> str:
        return content_hash(self._payload)

    def splits(self) -> tuple[str, ...]:
        return tuple(sorted(self._splits))

    def num_samples(self, split: str | None = None) -> int:
        return len(self._rows) if split is None else len(self._indices(split))

    def _indices(self, split: str | None) -> list[int]:
        if split is None:
            return list(range(len(self._rows)))
        if split not in self._splits:
            raise InvalidDatasetError(f"unknown split {split!r}")
        return self._splits[split]

    def metadata(self, *, deep: bool = False) -> DatasetMetadata:
        return DatasetMetadata(
            adapter=self.NAME, adapter_version=self.VERSION, family="list", version=self._version,
            fingerprint=self.fingerprint(), task=TaskType.REGRESSION,
            num_samples=len(self._rows), num_features=len(self._rows[0]),
            splits={k: len(v) for k, v in self._splits.items()},
            missing_values=0 if deep else None,
            capabilities=tuple(self.capabilities),
        )  # fmt: skip

    def sample(self, index: int, split: str | None = None) -> Sample:
        idx = self._indices(split)
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(idx):
            raise InvalidDatasetError(f"index {index!r} out of range")
        i = idx[index]
        return Sample(i, self._rows[i], None if self._targets is None else self._targets[i])

    def batches(self, batch_size: int, split: str | None = None) -> Iterator[Batch]:
        idx = self._indices(split)
        for chunk in batch_slices(len(idx), batch_size):
            ids = idx[chunk]
            yield Batch(
                tuple(ids),
                [self._rows[i] for i in ids],
                None if self._targets is None else [self._targets[i] for i in ids],
            )

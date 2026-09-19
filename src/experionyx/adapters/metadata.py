"""Standardized model and dataset metadata.

Every optional field is `None` when it cannot be measured reliably. `None` means "unavailable",
never zero. Dataset metadata is cheap by default; expensive statistics are opt-in (`deep=True`).
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Self

import experionyx.validation as v
from experionyx.adapters.capabilities import DatasetCapability, ModelCapability, TaskType
from experionyx.adapters.schema import TensorSchema
from experionyx.domain import check_keys
from experionyx.errors import ValidationError


def _opt_count(field_name: str, value: int | None) -> None:
    if value is not None:
        v.non_negative_int(field_name, value)


def _schema(d: Mapping[str, object], key: str) -> TensorSchema | None:
    raw = v.get_raw(d, key)
    return None if raw is None else TensorSchema.from_dict(v.get_mapping(d, key))


def _opt_int(d: Mapping[str, object], key: str) -> int | None:
    return None if v.get_raw(d, key) is None else v.get_int(d, key)


@dataclass(frozen=True)
class ModelMetadata:
    adapter: str
    adapter_version: str
    framework: str
    model_type: str  # qualified class name of the underlying model, as observed
    version: str  # user-supplied model version
    fingerprint: str
    task: TaskType = TaskType.UNKNOWN
    family: str | None = None
    input_schema: TensorSchema | None = None
    output_schema: TensorSchema | None = None
    parameter_count: int | None = None
    trainable_parameter_count: int | None = None
    serialization_format: str | None = None
    size_bytes: int | None = None  # size of the artifact on disk, if file-backed
    capabilities: tuple[ModelCapability, ...] = ()

    def __post_init__(self) -> None:
        v.text("adapter", self.adapter)
        v.version("adapter_version", self.adapter_version)
        v.text("framework", self.framework)
        v.text("model_type", self.model_type)
        v.version("version", self.version)
        v.digest("fingerprint", self.fingerprint)
        v.member("task", self.task, TaskType)
        if self.family is not None:
            v.text("family", self.family)
        for name in ("input_schema", "output_schema"):
            if getattr(self, name) is not None:
                v.member(name, getattr(self, name), TensorSchema)
        _opt_count("parameter_count", self.parameter_count)
        _opt_count("trainable_parameter_count", self.trainable_parameter_count)
        if (
            self.parameter_count is not None
            and self.trainable_parameter_count is not None
            and self.trainable_parameter_count > self.parameter_count
        ):
            raise ValidationError("trainable_parameter_count exceeds parameter_count")
        if self.serialization_format is not None:
            v.text("serialization_format", self.serialization_format)
        _opt_count("size_bytes", self.size_bytes)
        caps = tuple(
            sorted({v.member("capability", c, ModelCapability) for c in self.capabilities})
        )
        object.__setattr__(self, "capabilities", caps)

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        check_keys(d, tuple(cls.__dataclass_fields__))
        caps = v.get_raw(d, "capabilities")
        if not isinstance(caps, list | tuple):
            raise ValidationError("capabilities must be a list")
        return cls(
            adapter=v.get_str(d, "adapter"),
            adapter_version=v.get_str(d, "adapter_version"),
            framework=v.get_str(d, "framework"),
            model_type=v.get_str(d, "model_type"),
            version=v.get_str(d, "version"),
            fingerprint=v.get_str(d, "fingerprint"),
            task=v.get_enum(d, "task", TaskType),
            family=v.get_opt_str(d, "family"),
            input_schema=_schema(d, "input_schema"),
            output_schema=_schema(d, "output_schema"),
            parameter_count=_opt_int(d, "parameter_count"),
            trainable_parameter_count=_opt_int(d, "trainable_parameter_count"),
            serialization_format=v.get_opt_str(d, "serialization_format"),
            size_bytes=_opt_int(d, "size_bytes"),
            capabilities=tuple(ModelCapability(str(c)) for c in caps),
        )


@dataclass(frozen=True)
class DatasetMetadata:
    adapter: str
    adapter_version: str
    family: str  # e.g. "tabular", "torch-map-style"
    version: str
    fingerprint: str
    task: TaskType = TaskType.UNKNOWN
    num_samples: int | None = None
    num_features: int | None = None
    input_schema: TensorSchema | None = None  # includes shape and dtype where known
    target_schema: TensorSchema | None = None  # includes class labels where known
    class_count: int | None = None
    splits: Mapping[str, object] = field(default_factory=dict)  # split name -> sample count
    missing_values: int | None = None  # only populated by deep inspection; None = not inspected
    source: Mapping[str, object] = field(default_factory=dict)  # how it was produced/loaded
    capabilities: tuple[DatasetCapability, ...] = ()

    def __post_init__(self) -> None:
        v.text("adapter", self.adapter)
        v.version("adapter_version", self.adapter_version)
        v.text("family", self.family)
        v.version("version", self.version)
        v.digest("fingerprint", self.fingerprint)
        v.member("task", self.task, TaskType)
        _opt_count("num_samples", self.num_samples)
        _opt_count("num_features", self.num_features)
        _opt_count("class_count", self.class_count)
        _opt_count("missing_values", self.missing_values)
        for name in ("input_schema", "target_schema"):
            if getattr(self, name) is not None:
                v.member(name, getattr(self, name), TensorSchema)
        splits = v.freeze_mapping("splits", self.splits)
        for name, count in splits.items():
            v.non_negative_int(f"splits.{name}", count)
        object.__setattr__(self, "splits", splits)
        object.__setattr__(self, "source", v.freeze_mapping("source", self.source))
        caps = tuple(
            sorted({v.member("capability", c, DatasetCapability) for c in self.capabilities})
        )
        object.__setattr__(self, "capabilities", caps)

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        check_keys(d, tuple(cls.__dataclass_fields__))
        caps = v.get_raw(d, "capabilities")
        if not isinstance(caps, list | tuple):
            raise ValidationError("capabilities must be a list")
        return cls(
            adapter=v.get_str(d, "adapter"),
            adapter_version=v.get_str(d, "adapter_version"),
            family=v.get_str(d, "family"),
            version=v.get_str(d, "version"),
            fingerprint=v.get_str(d, "fingerprint"),
            task=v.get_enum(d, "task", TaskType),
            num_samples=_opt_int(d, "num_samples"),
            num_features=_opt_int(d, "num_features"),
            input_schema=_schema(d, "input_schema"),
            target_schema=_schema(d, "target_schema"),
            class_count=_opt_int(d, "class_count"),
            splits=v.get_mapping(d, "splits"),
            missing_values=_opt_int(d, "missing_values"),
            source=v.get_mapping(d, "source"),
            capabilities=tuple(DatasetCapability(str(c)) for c in caps),
        )

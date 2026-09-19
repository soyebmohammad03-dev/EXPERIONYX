"""Registry entities that make models and datasets identifiable, persistent and retrievable.

`RegisteredModel`/`RegisteredDataset` bind a user-facing name+version to a fingerprint, the
adapter (and adapter version) that interprets it, and where to load it from. An `Experiment`
references one through its `ModelRef`/`DatasetRef` (name, version, digest == fingerprint).
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar, Self

import experionyx.validation as v
from experionyx.adapters.metadata import DatasetMetadata, ModelMetadata
from experionyx.domain import DatasetRef, Entity, ModelRef, open_payload


@dataclass(frozen=True)
class RegisteredModel(Entity):
    """Identity: name, version, adapter, adapter version, fingerprint, load options.
    `source` (a locator such as a file path) is deliberately NOT identity: the same bytes at a
    different path are the same model. Do not register absolute private paths in public repos."""

    KIND: ClassVar[str] = "model"
    PREFIX: ClassVar[str] = "mdl"
    name: str
    metadata: ModelMetadata
    created_at: datetime
    source: str | None = None
    options: Mapping[str, object] = field(default_factory=dict)  # adapter load options

    def __post_init__(self) -> None:
        v.text("name", self.name)
        v.member("metadata", self.metadata, ModelMetadata)
        v.timestamp("created_at", self.created_at)
        if self.source is not None:
            v.text("source", self.source)
        object.__setattr__(self, "options", v.freeze_mapping("options", self.options))

    @property
    def version(self) -> str:
        return self.metadata.version

    @property
    def adapter(self) -> str:
        return self.metadata.adapter

    @property
    def adapter_version(self) -> str:
        return self.metadata.adapter_version

    @property
    def fingerprint(self) -> str:
        return self.metadata.fingerprint

    def ref(self) -> ModelRef:
        return ModelRef(self.name, self.version, self.fingerprint)

    def _identity(self) -> Mapping[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "adapter": self.adapter,
            "adapter_version": self.adapter_version,
            "fingerprint": self.fingerprint,
            "options": self.options,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(d, cls.KIND, ("name", "metadata", "created_at", "source", "options"))
        return cls(
            name=v.get_str(d, "name"),
            metadata=ModelMetadata.from_dict(v.get_mapping(d, "metadata")),
            created_at=v.get_time(d, "created_at"),
            source=v.get_opt_str(d, "source"),
            options=v.get_mapping(d, "options"),
        )


@dataclass(frozen=True)
class RegisteredDataset(Entity):
    KIND: ClassVar[str] = "dataset"
    PREFIX: ClassVar[str] = "dst"
    name: str
    metadata: DatasetMetadata
    created_at: datetime
    source: str | None = None
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        v.text("name", self.name)
        v.member("metadata", self.metadata, DatasetMetadata)
        v.timestamp("created_at", self.created_at)
        if self.source is not None:
            v.text("source", self.source)
        object.__setattr__(self, "options", v.freeze_mapping("options", self.options))

    @property
    def version(self) -> str:
        return self.metadata.version

    @property
    def adapter(self) -> str:
        return self.metadata.adapter

    @property
    def adapter_version(self) -> str:
        return self.metadata.adapter_version

    @property
    def fingerprint(self) -> str:
        return self.metadata.fingerprint

    def ref(self) -> DatasetRef:
        return DatasetRef(self.name, self.version, self.fingerprint)

    def _identity(self) -> Mapping[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "adapter": self.adapter,
            "adapter_version": self.adapter_version,
            "fingerprint": self.fingerprint,
            "options": self.options,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        open_payload(d, cls.KIND, ("name", "metadata", "created_at", "source", "options"))
        return cls(
            name=v.get_str(d, "name"),
            metadata=DatasetMetadata.from_dict(v.get_mapping(d, "metadata")),
            created_at=v.get_time(d, "created_at"),
            source=v.get_opt_str(d, "source"),
            options=v.get_mapping(d, "options"),
        )

"""Descriptions of model inputs/outputs. Unknown information is None, never guessed."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Self

import experionyx.validation as v
from experionyx.domain import check_keys
from experionyx.errors import ValidationError


class Modality(StrEnum):
    TABULAR = "TABULAR"
    IMAGE = "IMAGE"
    TEXT = "TEXT"
    SERIES = "SERIES"
    UNKNOWN = "UNKNOWN"


class FeatureKind(StrEnum):
    NUMERIC = "NUMERIC"
    CATEGORICAL = "CATEGORICAL"
    UNKNOWN = "UNKNOWN"


Label = int | str


@dataclass(frozen=True)
class TensorSchema:
    """Shape of data flowing in or out. A `None` dimension is variable (e.g. the batch axis);
    a `None` field is unknown."""

    modality: Modality = Modality.UNKNOWN
    dtype: str | None = None
    shape: tuple[int | None, ...] | None = None
    feature_names: tuple[str, ...] | None = None
    feature_kinds: tuple[FeatureKind, ...] | None = None
    class_labels: tuple[Label, ...] | None = None

    def __post_init__(self) -> None:
        v.member("modality", self.modality, Modality)
        if self.dtype is not None:
            v.text("dtype", self.dtype)
        if self.shape is not None:
            object.__setattr__(self, "shape", tuple(self.shape))
            for d in self.shape:
                if d is not None:
                    v.non_negative_int("shape dimension", d)
        for name in ("feature_names", "feature_kinds", "class_labels"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(value))
        if self.feature_names is not None:
            for n in self.feature_names:
                v.text("feature name", n)
        if self.feature_kinds is not None:
            for k in self.feature_kinds:
                v.member("feature kind", k, FeatureKind)
            if self.feature_names is not None and len(self.feature_kinds) != len(
                self.feature_names
            ):
                raise ValidationError("feature_kinds and feature_names must have equal length")
        if self.class_labels is not None:
            for label in self.class_labels:
                if isinstance(label, bool) or not isinstance(label, int | str):
                    raise ValidationError(f"class labels must be int or str, got {label!r}")

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        check_keys(
            d, ("modality", "dtype", "shape", "feature_names", "feature_kinds", "class_labels")
        )

        def seq(key: str) -> tuple[object, ...] | None:
            raw = v.get_raw(d, key)
            if raw is None:
                return None
            if not isinstance(raw, list | tuple):
                raise ValidationError(f"{key} must be a list or null")
            return tuple(raw)

        kinds = seq("feature_kinds")
        return cls(
            modality=v.get_enum(d, "modality", Modality),
            dtype=v.get_opt_str(d, "dtype"),
            shape=seq("shape"),  # type: ignore[arg-type]
            feature_names=seq("feature_names"),  # type: ignore[arg-type]
            feature_kinds=None if kinds is None else tuple(FeatureKind(str(k)) for k in kinds),
            class_labels=seq("class_labels"),  # type: ignore[arg-type]
        )

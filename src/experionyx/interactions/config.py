"""Typed, strict, hashed configuration and specification of an interaction analysis. Every
threshold that influences a label is here and is recorded with the result."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Self

import experionyx.validation as v
from experionyx.domain import Run, to_jsonable
from experionyx.errors import ValidationError
from experionyx.evaluation.config import _thaw
from experionyx.evaluation.serial import from_jsonable
from experionyx.hashing import HASH_PREFIX, content_hash
from experionyx.interactions.taxonomy import Aggregation, Normalization, Pairing

INTERACTION_VERSION = "1.0.0"
CONFIG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class InteractionConfig:
    aggregation: Aggregation = Aggregation.MEAN
    normalization: Normalization = Normalization.NONE
    pairing: Pairing = (
        Pairing.UNPAIRED
    )  # conservative default; PAIRED must be declared and is checked
    primary_metric: str | None = None  # default: accuracy (classification) / rmse (regression)
    metrics: tuple[str, ...] = ()  # empty: every scalar metric of the control evaluation
    include_classes: bool = True  # per-class recall contrasts (classification)
    include_slices: bool = True  # per-slice metric contrasts
    order_analysis: bool = False  # requires the BA cell
    per_sample: bool = True
    max_samples: int = 100_000  # per-sample analysis is refused (not truncated) beyond this
    bootstrap_resamples: int = 2000
    bootstrap_seed: int = 0
    confidence: float = 0.95
    min_trials: int = 3  # per treatment cell (A, B, AB, BA) for an interval
    min_abs_contrast: float = 0.0  # magnitude threshold in metric units (0 = none)
    min_sign_fraction: float = 0.95  # share of resamples with the sign of the point contrast
    require_same_environment: bool = True
    prevalence_delta_min: float = 0.2  # failure-mode prevalence change treated as increased/reduced
    schema_version: int = CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ValidationError(f"unsupported interaction config schema {self.schema_version}")
        if self.bootstrap_resamples < 1 or self.bootstrap_seed < 0 or self.min_trials < 2:
            raise ValidationError("bootstrap_resamples >= 1, bootstrap_seed >= 0, min_trials >= 2")
        if not 0.0 < self.confidence < 1.0:
            raise ValidationError("confidence must be in (0, 1)")
        for name in ("min_abs_contrast", "prevalence_delta_min"):
            if not (math.isfinite(getattr(self, name)) and getattr(self, name) >= 0):
                raise ValidationError(f"{name} must be a finite number >= 0")
        if not 0.5 <= self.min_sign_fraction <= 1.0:
            raise ValidationError("min_sign_fraction must be in [0.5, 1]")
        if self.max_samples < 1:
            raise ValidationError("max_samples must be >= 1")
        if list(self.metrics) != sorted(set(self.metrics)):
            raise ValidationError("metrics must be sorted and unique")

    def to_dict(self) -> dict[str, object]:
        data = to_jsonable(self)
        assert isinstance(data, dict)  # noqa: S101
        return data

    @property
    def config_hash(self) -> str:
        return content_hash(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        result: Self = from_jsonable(cls, _thaw(dict(data)))
        return result


def _runs(field_name: str, ids: tuple[str, ...], *, required: bool) -> tuple[str, ...]:
    if isinstance(ids, str) or not all(isinstance(i, str) for i in ids):
        raise ValidationError(f"{field_name} must be a list of run IDs")
    if required and not ids:
        raise ValidationError(f"{field_name} needs at least one run")
    for i in ids:
        v.ref(field_name, i, Run.PREFIX)
    return tuple(sorted(set(ids)))  # the order of runs inside a cell is not part of the identity


@dataclass(frozen=True)
class InteractionSpec:
    """A four-cell (optionally five-cell) design given as sets of finished runs. Its identity is
    the content hash of the canonical form, so any change to a run, a cell or a rule changes it."""

    control: tuple[str, ...]
    a: tuple[str, ...]
    b: tuple[str, ...]
    ab: tuple[str, ...]
    ba: tuple[str, ...] = ()
    discovery_run_id: str | None = None  # a Phase 6 discovery that covered the fault cells
    config: InteractionConfig = field(default_factory=InteractionConfig)

    def __post_init__(self) -> None:
        for name, req in (
            ("control", True),
            ("a", True),
            ("b", True),
            ("ab", True),
            ("ba", self.config.order_analysis),
        ):
            object.__setattr__(self, name, _runs(name, getattr(self, name), required=req))
        if self.ba and not self.config.order_analysis:
            raise ValidationError("a BA cell was given but order_analysis is off")
        if self.discovery_run_id is not None:
            v.ref("discovery_run_id", self.discovery_run_id, Run.PREFIX)
        cells = [self.control, self.a, self.b, self.ab, self.ba]
        flat = [r for c in cells for r in c]
        if len(flat) != len(set(flat)):
            raise ValidationError("a run may belong to only one cell")

    def cells(self) -> dict[str, tuple[str, ...]]:
        out = {"CONTROL": self.control, "A": self.a, "B": self.b, "AB": self.ab}
        if self.ba:
            out["BA"] = self.ba
        return out

    def to_dict(self) -> dict[str, object]:
        return {
            "cells": {k: list(x) for k, x in self.cells().items()},
            "discovery_run_id": self.discovery_run_id,
            "config": self.config.to_dict(),
            "interaction_version": INTERACTION_VERSION,
        }

    @property
    def spec_id(self) -> str:
        return "isp_" + content_hash(self.to_dict())[len(HASH_PREFIX) :][:32]

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        cells, cfg = d.get("cells"), d.get("config")
        if (
            not isinstance(cells, Mapping)
            or not isinstance(cfg, Mapping)
            or set(cells) - {"CONTROL", "A", "B", "AB", "BA"}
        ):
            raise ValidationError("malformed interaction spec")
        extra = set(d) - {"cells", "discovery_run_id", "config", "interaction_version"}
        if extra:
            raise ValidationError(f"unexpected interaction spec fields {sorted(extra)}")

        def cell(k: str) -> tuple[str, ...]:
            x = cells.get(k, [])
            if not isinstance(x, list | tuple):
                raise ValidationError(f"cell {k} must be a list")
            return tuple(str(i) for i in x)

        disc = d.get("discovery_run_id")
        return cls(
            cell("CONTROL"),
            cell("A"),
            cell("B"),
            cell("AB"),
            cell("BA"),
            None if disc is None else str(disc),
            InteractionConfig.from_dict(cfg),
        )

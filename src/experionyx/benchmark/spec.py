"""The benchmark definition. Immutable and strictly validated; its identity is the content hash of
its canonical form, so changing anything scientifically meaningful (faults, grids, seeds,
evaluation, analysis settings, limits, version, model, dataset) changes it. Faults and
interaction pairs are kept in a canonical order, so listing them differently does not."""

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Self

import experionyx.validation as v
from experionyx.calibration.spec import CalibrationSpec
from experionyx.domain import to_jsonable
from experionyx.drift.spec import ShiftSpec
from experionyx.errors import ValidationError
from experionyx.evaluation.config import EvaluationConfig
from experionyx.hashing import HASH_PREFIX, content_hash
from experionyx.interactions.config import InteractionConfig
from experionyx.resources.spec import ResourceSpec
from experionyx.slices.analysis import SliceConfig
from experionyx.slices.spec import SliceSpec
from experionyx.stress.spec import StressPlan

DRIFT_PLACEHOLDER_RUN = (
    "run_" + "0" * 32
)  # stands in for the baseline, which does not exist until the protocol runs
ENGINE_VERSION = "1.0.0"  # the benchmark protocol/analysis methodology; bump when either changes
SPEC_SCHEMA_VERSION = 1


def _plain(x: object) -> dict[str, Any]:
    out: Any = to_jsonable(x)
    return dict(out)


def _num(name: str, x: object) -> float:
    if isinstance(x, bool) or not isinstance(x, int | float) or not math.isfinite(x):
        raise ValidationError(f"{name} must be a finite number, got {x!r}")
    return float(x)


@dataclass(frozen=True)
class FaultGrid:
    """One fault family with a fixed parameter set and, optionally, ONE swept parameter. Every
    swept value becomes an explicit experiment point; nothing is interpolated or skipped."""

    name: str  # unique label of this grid within the benchmark
    fault: str  # registered fault type
    parameters: Mapping[str, object] = field(default_factory=dict)
    sweep_parameter: str | None = None
    values: tuple[float, ...] = ()
    fault_version: str | None = None  # pin a fault version; None = the registered latest, recorded
    scope_fraction: float | None = None  # apply to this share of samples (None: all)

    def __post_init__(self) -> None:
        v.text("name", self.name)
        v.text("fault", self.fault)
        object.__setattr__(
            self, "parameters", v.freeze_mapping("parameters", dict(self.parameters))
        )
        if (self.sweep_parameter is None) != (not self.values):
            raise ValidationError(f"grid {self.name!r}: sweep_parameter and values go together")
        object.__setattr__(self, "values", tuple(_num("sweep value", x) for x in self.values))
        if len(set(self.values)) != len(self.values):
            raise ValidationError(f"grid {self.name!r}: sweep values must be unique")
        if self.sweep_parameter is not None and self.sweep_parameter in self.parameters:
            raise ValidationError(
                f"grid {self.name!r}: {self.sweep_parameter!r} is both fixed and swept"
            )
        if self.scope_fraction is not None:
            f = _num("scope_fraction", self.scope_fraction)
            if not 0.0 < f <= 1.0:
                raise ValidationError("scope_fraction must be in (0, 1]")

    def points(self) -> tuple[float | None, ...]:
        return self.values if self.values else (None,)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "fault": self.fault,
            "fault_version": self.fault_version,
            "parameters": _plain(self.parameters),
            "sweep_parameter": self.sweep_parameter,
            "values": list(self.values),
            "scope_fraction": self.scope_fraction,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Self:
        extra = set(d) - {
            "name",
            "fault",
            "fault_version",
            "parameters",
            "sweep_parameter",
            "values",
            "scope_fraction",
        }
        if extra or "name" not in d or "fault" not in d:
            raise ValidationError(
                f"malformed fault grid (unexpected {sorted(extra)}; name and fault are required)"
            )
        params, values = d.get("parameters", {}), d.get("values", [])
        if not isinstance(params, Mapping) or not isinstance(values, list | tuple):
            raise ValidationError("grid parameters must be an object and values a list")
        sp, sv, sf = d.get("sweep_parameter"), d.get("fault_version"), d.get("scope_fraction")
        return cls(
            str(d["name"]),
            str(d["fault"]),
            dict(params),
            None if sp is None else str(sp),
            tuple(values),
            None if sv is None else str(sv),
            None if sf is None else float(sf),
        )


@dataclass(frozen=True)
class InteractionPair:
    """An interaction experiment between two grids at one parameter point each (a four-cell design
    analyzed by the Phase 7 engine)."""

    a: str
    b: str
    a_value: float | None = None
    b_value: float | None = None
    order_analysis: bool = False

    def __post_init__(self) -> None:
        v.text("a", self.a)
        v.text("b", self.b)
        if self.a == self.b:
            raise ValidationError("an interaction needs two different grids")
        for n, x in (("a_value", self.a_value), ("b_value", self.b_value)):
            if x is not None:
                object.__setattr__(self, n, _num(n, x))

    @property
    def name(self) -> str:
        return f"{self.a}+{self.b}"

    def to_dict(self) -> dict[str, object]:
        return {
            "a": self.a,
            "b": self.b,
            "a_value": self.a_value,
            "b_value": self.b_value,
            "order_analysis": self.order_analysis,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Self:
        extra = set(d) - {"a", "b", "a_value", "b_value", "order_analysis"}
        if extra or "a" not in d or "b" not in d:
            raise ValidationError(
                f"malformed interaction pair (unexpected {sorted(extra)}; a and b are required)"
            )
        av, bv = d.get("a_value"), d.get("b_value")
        return cls(
            str(d["a"]),
            str(d["b"]),
            None if av is None else float(av),
            None if bv is None else float(bv),
            bool(d.get("order_analysis", False)),
        )


@dataclass(frozen=True)
class BenchmarkLimits:
    max_units: int = (
        200  # expanded experiment units (baseline + trials); exceeding it refuses the benchmark
    )
    max_failed_trials: int | None = (
        None  # per fault experiment: skip the rest after this many failures
    )
    allow_unsupported: bool = (
        False  # True: record unsupported grids in the coverage instead of refusing
    )

    def __post_init__(self) -> None:
        if self.max_units < 1 or (
            self.max_failed_trials is not None and self.max_failed_trials < 1
        ):
            raise ValidationError("max_units and max_failed_trials must be >= 1")

    def to_dict(self) -> dict[str, object]:
        return {
            "max_units": self.max_units,
            "max_failed_trials": self.max_failed_trials,
            "allow_unsupported": self.allow_unsupported,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Self:
        extra = set(d) - {"max_units", "max_failed_trials", "allow_unsupported"}
        if extra:
            raise ValidationError(f"unexpected limits {sorted(extra)}")
        mf = d.get("max_failed_trials")
        return cls(
            int(d.get("max_units", 200)),
            None if mf is None else int(mf),
            bool(d.get("allow_unsupported", False)),
        )


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    version: (
        str  # the benchmark DEFINITION version (semver-like); results keep the exact spec they ran
    )
    model: str  # registered model record ID (mdl_...)
    dataset: str  # registered dataset record ID (dst_...)
    faults: tuple[FaultGrid, ...]
    seeds: tuple[int, ...]
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    interactions: tuple[InteractionPair, ...] = ()
    primary_metric: str | None = None
    aggregation_confidence: float = 0.95
    aggregation_resamples: int = 1000
    aggregation_seed: int = 0
    interaction_config: InteractionConfig = field(default_factory=InteractionConfig)
    discovery: bool = True  # run Phase 6 failure discovery over the benchmark's experiments
    profile: bool = True  # assemble a Phase 8 reliability profile from the evidence
    limits: BenchmarkLimits = field(default_factory=BenchmarkLimits)
    schema_version: int = SPEC_SCHEMA_VERSION
    slices: tuple[
        SliceSpec, ...
    ] = ()  # explicit request for a Phase 11 slice analysis (empty: none)
    slice_config: SliceConfig = field(default_factory=SliceConfig)
    stress: StressPlan | None = (
        None  # explicit request for a Phase 14 stress analysis on the benchmark's model, dataset, evaluation and baseline
    )
    calibration: CalibrationSpec | None = (
        None  # explicit request for a Phase 15 calibration analysis of the benchmark's baseline; its baseline_run is a placeholder here
    )
    drift: ShiftSpec | None = (
        None  # explicit request for a Phase 12 drift analysis; its baseline_run is a placeholder here and is replaced by the benchmark's baseline
    )
    resources: tuple[
        ResourceSpec, ...
    ] = ()  # explicit request for Phase 16 resource measurements (one unit each; a batch-size sweep is several specs) over the benchmark's model and dataset

    def __post_init__(self) -> None:
        if self.schema_version != SPEC_SCHEMA_VERSION:
            raise ValidationError(f"unsupported benchmark spec schema {self.schema_version}")
        v.text("name", self.name)
        if not re.fullmatch(r"\d+\.\d+\.\d+", self.version):
            raise ValidationError(
                f"benchmark version must be major.minor.patch, got {self.version!r}"
            )
        v.ref("model", self.model, "mdl")
        v.ref("dataset", self.dataset, "dst")
        ids = [r.spec_id for r in self.resources]
        if len(set(ids)) != len(ids):
            raise ValidationError("resource specs must be distinct")
        if any(r.model_id != self.model or r.dataset_id != self.dataset for r in self.resources):
            raise ValidationError(
                "every resource spec must measure the benchmark's own model and dataset"
            )
        object.__setattr__(
            self, "resources", tuple(sorted(self.resources, key=lambda r: r.spec_id))
        )
        if not self.faults:
            raise ValidationError("a benchmark needs at least one fault grid")
        names = [g.name for g in self.faults]
        if len(set(names)) != len(names):
            raise ValidationError(
                f"fault grid names must be unique: {sorted({n for n in names if names.count(n) > 1})}"
            )
        object.__setattr__(self, "faults", tuple(sorted(self.faults, key=lambda g: g.name)))
        object.__setattr__(
            self, "interactions", tuple(sorted(self.interactions, key=lambda p: (p.a, p.b)))
        )
        if len({p.name for p in self.interactions}) != len(self.interactions):
            raise ValidationError("duplicate interaction pair")
        if (
            not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in self.seeds)
        ):
            raise ValidationError("seeds must be a non-empty list of unique non-negative integers")
        object.__setattr__(self, "seeds", tuple(sorted(self.seeds)))
        object.__setattr__(self, "slices", tuple(sorted(self.slices, key=lambda x: x.slice_id)))
        if len({x.name for x in self.slices}) != len(self.slices) or len(
            {x.slice_id for x in self.slices}
        ) != len(self.slices):
            raise ValidationError(
                "benchmark slices must have unique names and distinct definitions"
            )
        if (
            not 0.0 < self.aggregation_confidence < 1.0
            or self.aggregation_resamples < 1
            or self.aggregation_seed < 0
        ):
            raise ValidationError("invalid aggregation settings")

    def to_dict(self) -> dict[str, object]:
        ev = to_jsonable(self.evaluation)
        d: dict[str, object] = {
            "name": self.name, "version": self.version, "model": self.model, "dataset": self.dataset,
            "evaluation": ev, "faults": [g.to_dict() for g in self.faults], "seeds": list(self.seeds),
            "interactions": [p.to_dict() for p in self.interactions], "primary_metric": self.primary_metric,
            "aggregation": {"confidence": self.aggregation_confidence, "resamples": self.aggregation_resamples, "seed": self.aggregation_seed},
            "interaction_config": self.interaction_config.to_dict(), "discovery": self.discovery, "profile": self.profile,
            "limits": self.limits.to_dict(), "schema_version": self.schema_version,
        }  # fmt: skip
        if (
            self.slices
        ):  # explicit request only; keeps the identity of every earlier benchmark unchanged
            d["slices"] = [x.to_dict() for x in self.slices]
            d["slice_config"] = self.slice_config.to_dict()
        if (
            self.stress is not None
        ):  # explicit request only; keeps every earlier benchmark's identity
            d["stress"] = self.stress.to_dict()
        if (
            self.drift is not None
        ):  # explicit request only; keeps every earlier benchmark's identity
            d["drift"] = {k: x for k, x in self.drift.to_dict().items() if k != "baseline_run"}
        if (
            self.calibration is not None
        ):  # explicit request only; keeps every earlier benchmark's identity
            d["calibration"] = {k: x for k, x in self.calibration.to_dict().items() if k != "baseline_run"}  # fmt: skip
        if self.resources:  # explicit request only; keeps every earlier benchmark's identity
            d["resources"] = [r.to_dict() for r in self.resources]
        return d

    @property
    def spec_id(self) -> str:
        return "bsp_" + content_hash(self.to_dict())[len(HASH_PREFIX) :][:32]

    def protocol_key(self) -> dict[str, object]:
        """Everything except the MODEL: two results are comparable only under an identical key."""
        d = self.to_dict()
        d.pop("model")
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Self:
        known = {
            "name",
            "version",
            "model",
            "dataset",
            "evaluation",
            "faults",
            "seeds",
            "interactions",
            "primary_metric",
            "aggregation",
            "interaction_config",
            "discovery",
            "profile",
            "limits",
            "schema_version",
            "slices",
            "slice_config",
            "drift",
            "calibration",
            "resources",
            "stress",
        }
        extra = set(d) - known
        missing = {"name", "version", "model", "dataset", "faults", "seeds"} - set(d)
        if extra or missing:
            raise ValidationError(
                f"malformed benchmark spec (unexpected {sorted(extra)}, missing {sorted(missing)})"
            )
        agg = d.get("aggregation", {})
        if not isinstance(agg, Mapping) or set(agg) - {"confidence", "resamples", "seed"}:
            raise ValidationError("aggregation must be an object with confidence, resamples, seed")
        faults, inter, seeds = d["faults"], d.get("interactions", []), d["seeds"]
        if not all(isinstance(x, list | tuple) for x in (faults, inter, seeds)):
            raise ValidationError("faults, interactions and seeds must be lists")
        return cls(
            str(d["name"]), str(d["version"]), str(d["model"]), str(d["dataset"]),
            tuple(FaultGrid.from_dict(g) for g in faults), tuple(seeds),
            EvaluationConfig.from_dict(d.get("evaluation", {})) if d.get("evaluation") else EvaluationConfig(),
            tuple(InteractionPair.from_dict(p) for p in inter),
            None if d.get("primary_metric") is None else str(d["primary_metric"]),
            float(agg.get("confidence", 0.95)), int(agg.get("resamples", 1000)), int(agg.get("seed", 0)),
            InteractionConfig.from_dict(d["interaction_config"]) if d.get("interaction_config") else InteractionConfig(),
            bool(d.get("discovery", True)), bool(d.get("profile", True)),
            BenchmarkLimits.from_dict(d.get("limits", {})), int(d.get("schema_version", SPEC_SCHEMA_VERSION)),
            tuple(SliceSpec.from_dict(x) for x in d.get("slices", [])),
            SliceConfig.from_dict(d["slice_config"]) if d.get("slice_config") else SliceConfig(),
            StressPlan.from_dict(d["stress"]) if d.get("stress") else None,
            CalibrationSpec.from_dict({**d["calibration"], "baseline_run": DRIFT_PLACEHOLDER_RUN}) if d.get("calibration") else None,
            ShiftSpec.from_dict({**d["drift"], "baseline_run": DRIFT_PLACEHOLDER_RUN}) if d.get("drift") else None,
            tuple(ResourceSpec.from_dict(x) for x in d.get("resources", [])),
        )  # fmt: skip

"""Fault specifications: typed, versioned, canonically identified (see docs/faults.md).

Terminology (docs/faults.md): a *fault* is a specification; a *perturbation* is its application to
data; an *observation* is a measured value; *degradation* is a measured, directional change of a
metric between control and treatment; a *failure* / *failure mode* are later-phase concepts that
this package never asserts.

This module has no numpy dependency: implementations live in `library.py`.
"""

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Self

from experionyx.domain import to_jsonable
from experionyx.errors import FaultCompatibilityError, FaultNotFoundError, ValidationError
from experionyx.evaluation.serial import from_jsonable

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    Array = NDArray[np.generic]
else:
    Array = object

FAULT_SCHEMA_VERSION = 1


class FaultCategory(StrEnum):
    INPUT_CORRUPTION = "INPUT_CORRUPTION"
    LABEL_CORRUPTION = "LABEL_CORRUPTION"
    MISSINGNESS = "MISSINGNESS"
    NOISE = "NOISE"
    TRANSFORMATION = "TRANSFORMATION"
    DISTRIBUTION_SHIFT = "DISTRIBUTION_SHIFT"  # future
    TEMPORAL_CORRUPTION = "TEMPORAL_CORRUPTION"  # future
    RESOURCE_FAULT = "RESOURCE_FAULT"  # future
    COMPOUND = "COMPOUND"


class FaultTarget(StrEnum):
    """What a fault changes. LABEL faults alter the *evaluation targets* (data quality), not what
    the model sees; they are a different kind of experiment from input robustness."""

    INPUT = "INPUT"
    LABEL = "LABEL"
    COMPOUND = "COMPOUND"


class Requirement(StrEnum):
    FLOAT_ARRAY = "FLOAT_ARRAY"  # floating-point inputs, shape (n, ...)
    TABULAR = "TABULAR"  # exactly 2-D: (samples, features)
    IMAGE = "IMAGE"  # at least 3-D: (samples, [channels,] height, width)
    CLASS_LABELS = "CLASS_LABELS"  # a classification target with known classes


class ScopeKind(StrEnum):
    ALL = "ALL"
    RANDOM_SUBSET = "RANDOM_SUBSET"
    CLASS = "CLASS"
    # Reserved vocabulary; validation rejects these until they are implemented.
    SLICE = "SLICE"
    FEATURE = "FEATURE"
    REGION = "REGION"
    TIME_WINDOW = "TIME_WINDOW"


IMPLEMENTED_SCOPES = frozenset({ScopeKind.ALL, ScopeKind.RANDOM_SUBSET, ScopeKind.CLASS})


@dataclass(frozen=True)
class FaultScope:
    """Which samples a fault applies to. 100% and 10% are scientifically different faults, so the
    scope is part of the fault's identity."""

    kind: ScopeKind = ScopeKind.ALL
    fraction: float | None = None  # RANDOM_SUBSET: exact share (rounded) of the evaluated samples
    label: int | str | None = None  # CLASS: samples whose true target equals this

    def __post_init__(self) -> None:
        if self.kind not in IMPLEMENTED_SCOPES:
            raise ValidationError(f"scope {self.kind} is reserved and not implemented yet")
        if self.kind is ScopeKind.ALL and (self.fraction is not None or self.label is not None):
            raise ValidationError("scope ALL takes no fraction or label")
        if self.kind is ScopeKind.RANDOM_SUBSET:
            if self.fraction is None or self.label is not None:
                raise ValidationError("scope RANDOM_SUBSET needs `fraction` only")
            if not (math.isfinite(self.fraction) and 0.0 < self.fraction <= 1.0):
                raise ValidationError("scope fraction must be in (0, 1]")
        if self.kind is ScopeKind.CLASS and (self.label is None or self.fraction is not None):
            raise ValidationError("scope CLASS needs `label` only")


@dataclass(frozen=True)
class FaultParameters:
    """Base class of every fault's typed parameter set. Subclasses are frozen dataclasses that
    validate in `__post_init__`; the authoritative parameter representation is never a dict."""


def finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValidationError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def in_range(name: str, value: float, lo: float, hi: float, *, lo_open: bool = False) -> float:
    v = finite(name, value)
    if (v <= lo if lo_open else v < lo) or v > hi:
        bracket = "(" if lo_open else "["
        raise ValidationError(f"{name} must be in {bracket}{lo}, {hi}], got {v}")
    return v


@dataclass(frozen=True)
class FaultContext:
    """Deterministic randomness and geometry handed to a fault implementation.

    Every random draw comes from `row_rng(position)` (one stream per sample position and
    component) or `columns(...)` (one stream per run), never from global state, so results do not
    depend on batch size or evaluation order."""

    seed: int
    component: int
    positions: "NDArray[np.int64]"  # global position (within the evaluated split) of each row
    classes: tuple[int | str, ...] | None = None
    _cache: dict[str, object] = field(default_factory=dict, repr=False, compare=False)

    def row_rng(self, position: int) -> "np.random.Generator":
        import numpy as np

        return np.random.default_rng(
            np.random.SeedSequence([self.seed, int(position), self.component, 1])
        )

    def columns(self, width: int, fraction: float, salt: int = 0) -> "NDArray[np.int64]":
        """A fixed random subset of round(fraction*width) columns, the same for every batch."""
        import numpy as np

        key = f"cols:{width}:{fraction}:{salt}"
        if key not in self._cache:
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, self.component, salt, 2])
            )
            k = min(width, max(1 if fraction > 0 else 0, round(fraction * width)))
            self._cache[key] = np.sort(rng.permutation(width)[:k])
        cols: NDArray[np.int64] = self._cache[key]  # type: ignore[assignment]
        return cols


ApplyFn = Callable[["Array", "NDArray[np.bool_]", FaultParameters, FaultContext], "Array"]


@dataclass(frozen=True)
class FaultType:
    """Registered fault implementation and its metadata."""

    name: str
    version: str
    category: FaultCategory
    target: FaultTarget
    params_type: type[FaultParameters]
    description: str
    requires: frozenset[Requirement] = frozenset()
    stochastic: bool = True
    param_docs: Mapping[str, str] = field(default_factory=dict)
    apply: ApplyFn | None = field(default=None, repr=False, compare=False)  # None: not implemented

    @property
    def implemented(self) -> bool:
        return self.apply is not None

    def sweepable(self) -> tuple[str, ...]:
        return tuple(
            f.name for f in fields(self.params_type) if f.type in ("float", "int", float, int)
        )


def _canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(payload: object, prefix: str) -> str:
    return prefix + hashlib.sha256(_canonical(payload).encode()).hexdigest()[:32]


@dataclass(frozen=True)
class FaultSpec:
    """One fully specified, reproducible fault application.

    `family_id` identifies the fault without its seed (type, version, parameters, scope, and the
    ordered components); `id` additionally includes every seed. Timestamps never enter identity.
    """

    type: str
    version: str
    parameters: FaultParameters
    seed: int
    scope: FaultScope = field(default_factory=FaultScope)
    components: tuple["FaultSpec", ...] = ()  # compound faults: ordered, order is identity

    def __post_init__(self) -> None:
        # Identity must not depend on how a number was written (1 vs 1.0): normalize float fields.
        if is_dataclass(self.parameters):
            fixes = {
                f.name: float(getattr(self.parameters, f.name))
                for f in fields(self.parameters)
                if f.type in (float, "float", float | None)
                and isinstance(getattr(self.parameters, f.name), int)
                and not isinstance(getattr(self.parameters, f.name), bool)
            }
            if fixes:
                object.__setattr__(self, "parameters", replace(self.parameters, **fixes))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValidationError("seed must be a non-negative integer")
        if not self.type or not self.version:
            raise ValidationError("fault type and version are required")
        if self.components and self.type != COMPOUND:
            raise ValidationError("only a compound fault has components")
        if self.type == COMPOUND and len(self.components) < 2:
            raise ValidationError("a compound fault needs at least two components")

    def _canon(self, *, with_seed: bool) -> dict[str, object]:
        body: dict[str, object] = {
            "type": self.type,
            "version": self.version,
            "parameters": to_jsonable(self.parameters),
            "scope": to_jsonable(self.scope),
            "components": [c._canon(with_seed=with_seed) for c in self.components],
        }
        if with_seed:
            body["seed"] = self.seed
        return body

    @property
    def family_id(self) -> str:
        return _digest(self._canon(with_seed=False), "flt_")

    @property
    def id(self) -> str:
        return _digest(self._canon(with_seed=True), "fap_")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": FAULT_SCHEMA_VERSION,
            **self._canon(with_seed=True),
            "components": [c.to_dict() for c in self.components],
        }

    def flatten(self) -> tuple["FaultSpec", ...]:
        """The leaf faults in execution order (a non-compound fault is its own single leaf)."""
        return self.components if self.components else (self,)

    def with_parameter(self, name: str, value: float) -> Self:
        if self.components:
            raise ValidationError("a compound fault has no parameters of its own to sweep")
        if not is_dataclass(self.parameters) or name not in {
            f.name for f in fields(self.parameters)
        }:
            raise ValidationError(f"{self.type} has no parameter {name!r}")
        return replace(self, parameters=replace(self.parameters, **{name: value}))


COMPOUND = "compound"


@dataclass(frozen=True)
class NoParameters(FaultParameters):
    """Parameters of a compound fault: none of its own (its components carry them)."""


class FaultRegistry:
    """Registry of fault types, keyed by (name, version); resolution defaults to the newest."""

    def __init__(self) -> None:
        self._types: dict[str, dict[str, FaultType]] = {}

    def register(self, fault_type: FaultType) -> None:
        versions = self._types.setdefault(fault_type.name, {})
        if fault_type.version in versions:
            raise ValidationError(
                f"fault {fault_type.name} v{fault_type.version} is already registered"
            )
        versions[fault_type.version] = fault_type

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._types))

    def versions(self, name: str) -> tuple[str, ...]:
        return tuple(self._types.get(name, {}))

    def resolve(self, name: str, version: str | None = None) -> FaultType:
        if name not in self._types:
            raise FaultNotFoundError(f"no fault named {name!r} (registered: {list(self.names())})")
        versions = self._types[name]
        if version is None:
            return versions[
                max(versions, key=lambda v: [int(p) if p.isdigit() else 0 for p in v.split(".")])
            ]
        if version not in versions:
            raise FaultNotFoundError(
                f"fault {name!r} has no version {version!r} (registered: {list(versions)}); a "
                "stored specification is reproducible only with the implementation it names"
            )
        return versions[version]

    def all(self) -> list[FaultType]:
        return [self.resolve(n) for n in self.names()]

    def make(
        self,
        name: str,
        *,
        seed: int,
        scope: FaultScope | None = None,
        version: str | None = None,
        **params: object,
    ) -> FaultSpec:
        """Build a validated spec. Parameter errors surface here, before any experiment starts."""
        ft = self.resolve(name, version)
        try:
            parameters = ft.params_type(**params)
        except TypeError as exc:
            raise ValidationError(f"invalid parameters for {name}: {exc}") from exc
        return FaultSpec(ft.name, ft.version, parameters, seed, scope or FaultScope())

    def compound(self, *components: FaultSpec) -> FaultSpec:
        """An ordered composition. Order is part of the identity: A→B and B→A differ."""
        return FaultSpec(COMPOUND, "1", NoParameters(), 0, FaultScope(), tuple(components))

    def from_dict(self, data: Mapping[str, object]) -> FaultSpec:
        """Strictly parse a stored spec; the exact recorded version must still be registered."""
        known = {"schema_version", "type", "version", "parameters", "scope", "components", "seed"}
        extra = set(data) - known
        if extra or not {"type", "version", "parameters", "seed"} <= set(data):
            raise ValidationError(f"malformed fault specification (unexpected {sorted(extra)})")
        if data.get("schema_version", FAULT_SCHEMA_VERSION) != FAULT_SCHEMA_VERSION:
            raise ValidationError("unsupported fault schema version")
        name, version = str(data["type"]), str(data["version"])
        raw_components = data.get("components", [])
        if not isinstance(raw_components, list | tuple):
            raise ValidationError("components must be a list")
        if name == COMPOUND:
            return FaultSpec(
                COMPOUND, version, NoParameters(), int(str(data["seed"])), FaultScope(),
                tuple(self.from_dict(c) for c in raw_components),
            )  # fmt: skip
        ft = self.resolve(name, version)
        params: Any = from_jsonable(ft.params_type, data["parameters"])
        scope: FaultScope = from_jsonable(FaultScope, data.get("scope", {"kind": "ALL"}))
        seed = data["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValidationError("seed must be an integer")
        return FaultSpec(ft.name, ft.version, params, seed, scope)


def check_compatibility(
    spec: FaultSpec,
    registry: FaultRegistry,
    *,
    ndim: int | None,
    dtype_kind: str | None,
    has_classes: bool,
) -> None:
    """Pre-flight: does every leaf fault suit the data described by dataset metadata? Raises
    FaultCompatibilityError before any run is created. `None` means the property is unknown."""
    for leaf in spec.flatten():
        ft = registry.resolve(leaf.type, leaf.version)
        if not ft.implemented:
            raise FaultCompatibilityError(f"fault {ft.name} is not implemented yet ({ft.category})")
        for req in ft.requires:
            bad = (
                (req is Requirement.FLOAT_ARRAY and dtype_kind not in (None, "f"))
                or (req is Requirement.TABULAR and ndim not in (None, 2))
                or (req is Requirement.IMAGE and ndim is not None and ndim < 3)
                or (req is Requirement.CLASS_LABELS and not has_classes)
            )
            if bad:
                raise FaultCompatibilityError(
                    f"fault {ft.name} requires {req} but the data has ndim={ndim}, "
                    f"dtype kind={dtype_kind}, class labels={'yes' if has_classes else 'no'}"
                )
        if leaf.scope.kind is ScopeKind.CLASS and not has_classes:
            raise FaultCompatibilityError("class-targeted scope needs a classification target")

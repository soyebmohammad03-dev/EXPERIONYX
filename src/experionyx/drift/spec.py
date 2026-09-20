"""Typed, immutable definitions for temporal / distribution-shift analysis with deterministic IDs.

A window is an interval over an EXPLICITLY DECLARED ordering (`Ordering`): either the dataset
sample index (`index`, declared by the author to be the temporal order) or a numeric dataset
feature (`feature:<name>`, e.g. a timestamp column). Row position is never silently used as time.
Bounds are finite numbers with explicit inclusivity. Reference and comparison windows of a pair must
be disjoint (the tests used assume independent groups), which is decided from the intervals alone
and therefore refused before anything runs.

Rolling plans generate comparison windows `[start + k*step, start + k*step + width)`; every
candidate is either used or listed as skipped with its reason. Nothing is dropped silently."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Self

import experionyx.validation as v
from experionyx.errors import ValidationError
from experionyx.hashing import HASH_PREFIX, content_hash
from experionyx.slices.analysis import SliceConfig
from experionyx.slices.spec import SliceSpec
from experionyx.stats import core as st

DRIFT_SCHEMA = 1
ANALYSIS_VERSION = "1"
MAX_WINDOWS = 500
DIMENSIONS = ("covariate", "label", "performance", "prediction")
FEATURE_KINDS = ("NUMERIC", "CATEGORICAL", "BOOLEAN")
REFERENCE_MODES = ("FIXED", "PREVIOUS", "EXPANDING")
CORRECTION_SCOPES = ("PER_WINDOW_PAIR", "ACROSS_WINDOWS")
INDEX = "index"
FEATURE_PREFIX = "feature:"

Number = int | float


class Role(StrEnum):
    REFERENCE = "REFERENCE"
    COMPARISON = "COMPARISON"


def _num(name: str, x: object) -> Number:
    if isinstance(x, bool) or not isinstance(x, int | float) or not math.isfinite(x):
        raise ValidationError(f"{name} must be a finite number, got {x!r}")
    return int(x) if isinstance(x, float) and x.is_integer() else x


def _flag(d: Mapping[str, object], key: str, default: bool) -> bool:
    x = d.get(key, default)
    if not isinstance(x, bool):
        raise ValidationError(f"{key} must be a boolean")
    return x


def _strict(d: Mapping[str, object], what: str, allowed: set[str], required: set[str]) -> None:
    if set(d) - allowed or required - set(d):
        raise ValidationError(
            f"malformed {what} (unexpected {sorted(set(d) - allowed)}, missing {sorted(required - set(d))})"
        )


@dataclass(frozen=True)
class Ordering:
    """The declared ordering of samples. `unique` refuses ties (duplicate keys) when the caller
    needs a strict order; ties are otherwise legitimate because membership depends on the key's
    VALUE, never on position."""

    field: str
    unique: bool = False

    def __post_init__(self) -> None:
        f = self.field
        if not isinstance(f, str) or not (
            f == INDEX or (f.startswith(FEATURE_PREFIX) and f[len(FEATURE_PREFIX) :].strip())
        ):
            raise ValidationError(
                f"ordering field must be {INDEX!r} or {FEATURE_PREFIX}<name or column index>, got {f!r}"
            )

    def to_dict(self) -> dict[str, object]:
        return {"field": self.field, "unique": self.unique}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        _strict(d, "ordering", {"field", "unique"}, {"field"})
        return cls(str(d["field"]), _flag(d, "unique", False))


@dataclass(frozen=True)
class TemporalWindow:
    """An interval `start .. end` over the declared ordering. The role is part of the identity;
    the name is a label."""

    role: Role
    start: Number
    end: Number
    start_inclusive: bool = True
    end_inclusive: bool = False
    name: str | None = None

    def __post_init__(self) -> None:
        v.member("role", self.role, Role)
        object.__setattr__(self, "start", _num("start", self.start))
        object.__setattr__(self, "end", _num("end", self.end))
        if self.start > self.end or (
            self.start == self.end and not (self.start_inclusive and self.end_inclusive)
        ):
            raise ValidationError(
                f"window {self.describe()} is empty by construction (need start < end, or start == end with both bounds inclusive)"
            )

    def contains(self, key: float) -> bool:
        lo = key >= self.start if self.start_inclusive else key > self.start
        hi = key <= self.end if self.end_inclusive else key < self.end
        return lo and hi

    def describe(self) -> str:
        return f"{'[' if self.start_inclusive else '('}{self.start:g}, {self.end:g}{']' if self.end_inclusive else ')'}"

    def identity(self, ordering_field: str) -> dict[str, object]:
        return {
            "drift_schema": DRIFT_SCHEMA, "ordering": ordering_field, "role": self.role.value,
            "start": self.start, "end": self.end,
            "start_inclusive": self.start_inclusive, "end_inclusive": self.end_inclusive,
        }  # fmt: skip

    def window_id(self, ordering_field: str) -> str:
        # the same derivation as Entity.id for the registered DriftWindow, so the two always agree
        h = content_hash({"kind": "temporal_window", "identity": self.identity(ordering_field)})
        return "twn_" + h[len(HASH_PREFIX) :][:32]

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role.value, "start": self.start, "end": self.end,
            "start_inclusive": self.start_inclusive, "end_inclusive": self.end_inclusive,
            "name": self.name,
        }  # fmt: skip

    @classmethod
    def from_dict(cls, d: Mapping[str, object], role: Role) -> Self:
        _strict(d, "window", {"role", "start", "end", "start_inclusive", "end_inclusive", "name"}, {"start", "end"})  # fmt: skip
        if "role" in d and d["role"] != role.value:
            raise ValidationError(
                f"a {role.value.lower()} window cannot declare role {d['role']!r}"
            )
        name = d.get("name")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise ValidationError("window name must be a non-empty string")
        return cls(
            role, _num("start", d["start"]), _num("end", d["end"]),
            _flag(d, "start_inclusive", True), _flag(d, "end_inclusive", False), name,
        )  # fmt: skip


def _before(a: TemporalWindow, b: TemporalWindow) -> bool:
    return a.end < b.start or (a.end == b.start and not (a.end_inclusive and b.start_inclusive))


def overlaps(a: TemporalWindow, b: TemporalWindow) -> bool:
    return not (_before(a, b) or _before(b, a))


@dataclass(frozen=True)
class Rolling:
    """Sequential windows of `width`, `step` apart, over `[start, stop)`. `reference` says what each
    is compared against: FIXED (the spec's `reference` window), PREVIOUS (the window before it) or
    EXPANDING (everything from `start` up to the window's start)."""

    start: Number
    stop: Number
    width: Number
    step: Number
    reference: str = "FIXED"

    def __post_init__(self) -> None:
        for n in ("start", "stop", "width", "step"):
            object.__setattr__(self, n, _num(n, getattr(self, n)))
        if self.reference not in REFERENCE_MODES:
            raise ValidationError(f"rolling reference must be one of {list(REFERENCE_MODES)}")
        if self.width <= 0 or self.step <= 0 or self.stop <= self.start:
            raise ValidationError("rolling needs width > 0, step > 0 and stop > start")
        if self.reference == "PREVIOUS" and self.step < self.width:
            raise ValidationError(
                "rolling with a PREVIOUS reference needs step >= width, otherwise consecutive windows overlap and the comparison would share samples"
            )
        if math.ceil((self.stop - self.start) / self.step) > MAX_WINDOWS:
            raise ValidationError(f"a rolling plan may have at most {MAX_WINDOWS} windows")

    def candidates(self) -> list[TemporalWindow]:
        out = []
        k = 0
        while self.start + k * self.step < self.stop:
            s = self.start + k * self.step
            out.append(TemporalWindow(Role.COMPARISON, s, s + self.width))
            k += 1
        return out

    def to_dict(self) -> dict[str, object]:
        return {"start": self.start, "stop": self.stop, "width": self.width, "step": self.step, "reference": self.reference}  # fmt: skip

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        _strict(d, "rolling", {"start", "stop", "width", "step", "reference"}, {"start", "stop", "width", "step"})  # fmt: skip
        return cls(d["start"], d["stop"], d["width"], d["step"], str(d.get("reference", "FIXED")))  # type: ignore[arg-type]


@dataclass(frozen=True)
class FeatureDecl:
    """A dataset feature to analyze and how to read it. The kind is declared, never guessed."""

    name: str
    kind: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or self.name != self.name.strip():
            raise ValidationError("a feature name must be non-empty without surrounding whitespace")
        if self.kind not in FEATURE_KINDS:
            raise ValidationError(
                f"unsupported feature type {self.kind!r} for {self.name!r}; supported: {list(FEATURE_KINDS)}"
            )

    @property
    def column(self) -> str:
        return FEATURE_PREFIX + self.name

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "type": self.kind}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        _strict(d, "feature", {"name", "type"}, {"name", "type"})
        return cls(str(d["name"]), str(d["type"]))


@dataclass(frozen=True)
class DriftConfig:
    min_samples: int = 10  # valid values per window below which inference is withheld
    confidence: float = 0.95
    resamples: int = 1000
    permutations: int = 2000
    seed: int = 0
    method: str = "percentile"
    correction: str = "NONE"  # NONE | BONFERRONI | BENJAMINI_HOCHBERG within each family
    alpha: float = 0.05
    correction_scope: str = (
        "PER_WINDOW_PAIR"  # or ACROSS_WINDOWS (one family over every window pair)
    )
    metrics: tuple[str, ...] = ()  # performance metrics; empty: every scalar metric of the task
    dimensions: tuple[str, ...] | None = (
        None  # None: label, prediction, performance (+ covariate when features are declared)
    )

    def __post_init__(self) -> None:
        if self.min_samples < 2 or self.resamples < 1 or self.permutations < 1 or self.seed < 0:
            raise ValidationError("min_samples >= 2, resamples >= 1, permutations >= 1, seed >= 0")
        if not 0.0 < self.confidence < 1.0 or not 0.0 < self.alpha < 1.0:
            raise ValidationError("confidence and alpha must be in (0, 1)")
        if self.method not in st.METHODS or self.correction not in st.CORRECTIONS:
            raise ValidationError(
                f"method in {list(st.METHODS)}, correction in {list(st.CORRECTIONS)}"
            )
        if self.correction_scope not in CORRECTION_SCOPES:
            raise ValidationError(f"correction_scope must be one of {list(CORRECTION_SCOPES)}")
        if list(self.metrics) != sorted(set(self.metrics)):
            raise ValidationError("metrics must be sorted and unique")
        if self.dimensions is not None:
            bad = set(self.dimensions) - set(DIMENSIONS)
            if bad or not self.dimensions or list(self.dimensions) != sorted(set(self.dimensions)):
                raise ValidationError(f"dimensions must be a sorted, unique, non-empty subset of {list(DIMENSIONS)}")  # fmt: skip

    def slice_config(self) -> SliceConfig:
        """The Phase 11 metric/comparison settings this configuration implies."""
        return SliceConfig(
            min_members=self.min_samples, confidence=self.confidence, resamples=self.resamples,
            seed=self.seed, method=self.method, correction=self.correction, alpha=self.alpha,
            metrics=self.metrics,
        )  # fmt: skip

    def to_dict(self) -> dict[str, object]:
        return {
            "min_samples": self.min_samples, "confidence": self.confidence, "resamples": self.resamples,
            "permutations": self.permutations, "seed": self.seed, "method": self.method,
            "correction": self.correction, "alpha": self.alpha, "correction_scope": self.correction_scope,
            "metrics": list(self.metrics),
            "dimensions": None if self.dimensions is None else list(self.dimensions),
        }  # fmt: skip

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        extra = set(d) - set(cls.__dataclass_fields__)
        if extra:
            raise ValidationError(f"unknown drift setting(s) {sorted(extra)}")
        kw = dict(d)
        for k in ("metrics", "dimensions"):
            if kw.get(k) is not None:
                if not isinstance(kw[k], list | tuple):
                    raise ValidationError(f"{k} must be a list")
                kw[k] = tuple(kw[k])  # type: ignore[arg-type]
        return cls(**kw)  # type: ignore[arg-type]


@dataclass(frozen=True)
class Pair:
    reference: TemporalWindow
    comparison: TemporalWindow

    @property
    def key(self) -> str:
        return f"{self.reference.describe()} vs {self.comparison.describe()}"


@dataclass(frozen=True)
class Skipped:
    window: TemporalWindow
    reason: str  # PARTIAL_WINDOW | NO_REFERENCE | EMPTY_WINDOW | EMPTY_REFERENCE
    detail: str

    def to_dict(self, ordering_field: str) -> dict[str, object]:
        return {"window": self.window.to_dict(), "window_id": self.window.window_id(ordering_field), "reason": self.reason, "detail": self.detail}  # fmt: skip


@dataclass(frozen=True)
class Plan:
    pairs: tuple[Pair, ...]
    skipped: tuple[Skipped, ...]

    def windows(self) -> list[TemporalWindow]:
        seen: dict[TemporalWindow, None] = {}
        for p in self.pairs:
            seen[p.reference] = None
            seen[p.comparison] = None
        return list(seen)


@dataclass(frozen=True)
class ShiftSpec:
    baseline_run: str
    ordering: Ordering
    reference: TemporalWindow | None = None
    comparisons: tuple[TemporalWindow, ...] = ()
    rolling: Rolling | None = None
    features: tuple[FeatureDecl, ...] = ()
    slices: tuple[SliceSpec, ...] = ()
    failure_modes: tuple[str, ...] = ()
    config: DriftConfig = field(default_factory=DriftConfig)

    def __post_init__(self) -> None:
        v.ref("baseline_run", self.baseline_run, "run")
        if bool(self.comparisons) == (self.rolling is not None):
            raise ValidationError("declare either explicit `comparisons` or a `rolling` plan, not both and not neither")  # fmt: skip
        if self.rolling is None or self.rolling.reference == "FIXED":
            if self.reference is None or self.reference.role is not Role.REFERENCE:
                raise ValidationError("a REFERENCE window is required")
        elif self.reference is not None:
            raise ValidationError(f"a {self.rolling.reference} rolling plan defines its own references; do not declare `reference`")  # fmt: skip
        for c in self.comparisons:
            if c.role is not Role.COMPARISON:
                raise ValidationError(f"comparison window {c.describe()} has role {c.role.value}")
            if self.reference is not None and overlaps(self.reference, c):
                raise ValidationError(
                    f"reference {self.reference.describe()} overlaps comparison {c.describe()}; the tests need disjoint windows"
                )
        if self.rolling is not None and self.reference is not None:
            hit = next((c for c in self.rolling.candidates() if overlaps(self.reference, c)), None)
            if hit is not None:
                raise ValidationError(f"reference {self.reference.describe()} overlaps rolling window {hit.describe()}; the tests need disjoint windows")  # fmt: skip
        oid = self.ordering.field
        if len({c.window_id(oid) for c in self.comparisons}) != len(self.comparisons):
            raise ValidationError("the same comparison window is declared twice")
        names = [f.name for f in self.features]
        if len(set(names)) != len(names):
            raise ValidationError("a feature is declared twice")
        object.__setattr__(self, "features", tuple(sorted(self.features, key=lambda f: f.name)))
        object.__setattr__(self, "comparisons", tuple(sorted(self.comparisons, key=lambda w: (w.start, w.end))))  # fmt: skip
        if len({s.name for s in self.slices}) != len(self.slices) or len({s.slice_id for s in self.slices}) != len(self.slices):  # fmt: skip
            raise ValidationError("drift slices must have unique names and distinct definitions")
        object.__setattr__(self, "slices", tuple(sorted(self.slices, key=lambda s: s.slice_id)))
        for i in self.failure_modes:
            v.ref("failure_modes", i, "fmd")
        object.__setattr__(self, "failure_modes", tuple(sorted(set(self.failure_modes))))
        dims = self.config.dimensions
        if dims is None:
            dims = tuple(sorted({"label", "performance", "prediction"} | ({"covariate"} if self.features else set())))  # fmt: skip
        if ("covariate" in dims) != bool(self.features):
            raise ValidationError("the covariate dimension needs declared features, and declared features need the covariate dimension")  # fmt: skip
        if list(dims) != sorted(dims):
            raise ValidationError("dimensions must be sorted")
        d = self.config.to_dict()
        d["dimensions"] = list(dims)
        object.__setattr__(self, "config", DriftConfig.from_dict(d))

    # -- windows ---------------------------------------------------------------------------------

    def plan(self) -> Plan:
        pairs: list[Pair] = []
        skipped: list[Skipped] = []
        if self.rolling is None:
            assert self.reference is not None  # noqa: S101
            return Plan(tuple(Pair(self.reference, c) for c in self.comparisons), ())
        r = self.rolling
        prev: TemporalWindow | None = None
        for w in r.candidates():
            if w.end > r.stop:
                skipped.append(Skipped(w, "PARTIAL_WINDOW", f"{w.describe()} extends past stop={r.stop:g}"))  # fmt: skip
            elif r.reference == "FIXED":
                assert self.reference is not None  # noqa: S101
                pairs.append(Pair(self.reference, w))
            elif r.reference == "PREVIOUS":
                if prev is None:
                    skipped.append(Skipped(w, "NO_REFERENCE", "the first window has no preceding window to serve as reference"))  # fmt: skip
                else:
                    pairs.append(Pair(TemporalWindow(Role.REFERENCE, prev.start, prev.end), w))
            elif w.start == r.start:  # EXPANDING
                skipped.append(Skipped(w, "NO_REFERENCE", "the expanding reference [start, window start) is empty for the first window"))  # fmt: skip
            else:
                pairs.append(Pair(TemporalWindow(Role.REFERENCE, r.start, w.start), w))
            if w.end <= r.stop:
                prev = w
        return Plan(tuple(pairs), tuple(skipped))

    # -- identity --------------------------------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline_run": self.baseline_run, "ordering": self.ordering.to_dict(),
            "reference": None if self.reference is None else self.reference.to_dict(),
            "comparisons": [c.to_dict() for c in self.comparisons],
            "rolling": None if self.rolling is None else self.rolling.to_dict(),
            "features": [f.to_dict() for f in self.features],
            "slices": [s.to_dict() for s in self.slices],
            "failure_modes": list(self.failure_modes), "config": self.config.to_dict(),
            "analysis_version": ANALYSIS_VERSION, "drift_schema": DRIFT_SCHEMA,
        }  # fmt: skip

    @property
    def spec_id(self) -> str:
        # names are labels; slice definitions enter through their identities
        d = self.to_dict()
        d["slices"] = [{"name": s.name, "slice_id": s.slice_id} for s in self.slices]
        for w in (d["reference"], *d["comparisons"]):  # type: ignore[misc]
            if isinstance(w, dict):
                w.pop("name", None)
        return "dsp_" + content_hash(d)[len(HASH_PREFIX) :][:32]

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        allowed = {"baseline_run", "ordering", "reference", "comparisons", "rolling", "features", "slices", "failure_modes", "config", "analysis_version", "drift_schema"}  # fmt: skip
        _strict(d, "drift spec", allowed, {"baseline_run", "ordering"})
        if d.get("drift_schema", DRIFT_SCHEMA) != DRIFT_SCHEMA:
            raise ValidationError(f"unsupported drift_schema {d.get('drift_schema')!r}")
        if d.get("analysis_version", ANALYSIS_VERSION) != ANALYSIS_VERSION:
            raise ValidationError(f"unsupported analysis_version {d.get('analysis_version')!r}")

        def obj(k: str) -> Mapping[str, Any] | None:
            x = d.get(k)
            if x is not None and not isinstance(x, Mapping):
                raise ValidationError(f"{k} must be an object")
            return x

        def objs(k: str) -> Sequence[Mapping[str, Any]]:
            x = d.get(k) or []
            if not isinstance(x, list | tuple) or not all(isinstance(i, Mapping) for i in x):
                raise ValidationError(f"{k} must be a list of objects")
            return x

        ordering, ref, roll, cfg = obj("ordering"), obj("reference"), obj("rolling"), obj("config")
        if ordering is None:
            raise ValidationError("ordering is required: time is never inferred from row position")
        modes = d.get("failure_modes") or []
        if not isinstance(modes, list | tuple):
            raise ValidationError("failure_modes must be a list")
        return cls(
            str(d["baseline_run"]), Ordering.from_dict(ordering),
            None if ref is None else TemporalWindow.from_dict(ref, Role.REFERENCE),
            tuple(TemporalWindow.from_dict(c, Role.COMPARISON) for c in objs("comparisons")),
            None if roll is None else Rolling.from_dict(roll),
            tuple(FeatureDecl.from_dict(f) for f in objs("features")),
            tuple(SliceSpec.from_dict(s) for s in objs("slices")),
            tuple(str(m) for m in modes),
            DriftConfig.from_dict(cfg or {}),
        )  # fmt: skip

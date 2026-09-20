"""A versioned, deterministic specification of data-quality checks. Nothing is inferred: every
expectation (types, ranges, allowed categories, thresholds, which splits are reference and
comparison, which column is an identifier) is declared here, and an invalid declaration is refused
before anything runs.

A check is a type plus a NORMALIZED configuration; its identity (`qcs_`) is a hash of both, so
spelling differences (order, 5.0 vs 5) do not change it and any real change does. The spec identity
(`dqs_`) covers the dataset, the declared schema and every check. There is no overall score: each
check reports its own status on its own scope."""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Self

import experionyx.validation as v
from experionyx.drift.spec import FeatureDecl, Ordering, Role, TemporalWindow
from experionyx.errors import ValidationError
from experionyx.failures.extraction import label_key
from experionyx.hashing import HASH_PREFIX, content_hash
from experionyx.slices.spec import SliceSpec
from experionyx.stats import core as st

QUALITY_SCHEMA = 1
ANALYSIS_VERSION = "1"
CHECK_TYPES = (
    "schema", "sample_count", "missingness", "duplicates", "identifiers", "numeric", "categorical",
    "target", "distribution", "split_overlap", "target_leakage", "temporal_order", "group_comparison",
)  # fmt: skip
CHECK_DOCS = {
    "schema": "required/extra columns, ragged rows, values that do not fit a declared type",
    "sample_count": "row-count rules; metadata/feature/target count consistency",
    "missingness": "per-feature and per-row missing counts, rates, patterns; optional maximum rates",
    "duplicates": "exact duplicate rows; identical features with different targets",
    "identifiers": "a declared id column's duplicates and conflicts; identifier-like columns",
    "numeric": "non-finite values, constant/near-constant, bounds, outliers under a documented method",
    "categorical": "counts, cardinality, allowed set, rare categories",
    "target": "missing/invalid targets, class counts and proportions, regression summary",
    "distribution": "skewness, dominant values, distinct-value counts (descriptive)",
    "split_overlap": "sample-ID and feature-vector overlap between two declared splits (leakage indicator)",
    "target_leakage": "explicitly named candidates: equality, correlation, purity against the target",
    "temporal_order": "whether a comparison split lies after a reference split in the declared ordering",
    "group_comparison": "missingness, validity, categories and target between two groups (splits or windows) with Phase 10 statistics",
}
PER_TABLE = ("schema", "sample_count", "missingness", "duplicates", "identifiers", "numeric", "categorical", "target", "distribution", "target_leakage")  # fmt: skip
SLICEABLE = ("categorical", "distribution", "duplicates", "missingness", "numeric", "target")
ASPECTS = ("categories", "missingness", "target", "validity")
TASKS = ("CLASSIFICATION", "REGRESSION")
OUTLIER_METHODS = ("iqr", "mad")
MAX_CHECKS = 200


# -- small validators (each returns the normalized value) ---------------------------------------------------


def _strict(
    d: Mapping[str, object], what: str, allowed: set[str], required: set[str] | None = None
) -> None:
    need = required or set()
    if set(d) - allowed or need - set(d):
        raise ValidationError(f"malformed {what} (unexpected {sorted(set(d) - allowed)}, missing {sorted(need - set(d))})")  # fmt: skip


def _num(name: str, x: object) -> int | float:
    if isinstance(x, bool) or not isinstance(x, int | float) or not math.isfinite(x):
        raise ValidationError(f"{name} must be a finite number, got {x!r}")
    return int(x) if isinstance(x, float) and x.is_integer() else x


def _rate(name: str, x: object) -> float | None:
    if x is None:
        return None
    n = _num(name, x)
    if not 0 <= n <= 1:
        raise ValidationError(f"{name} must be in [0, 1], got {x!r}")
    return float(n)


def _count(name: str, x: object, low: int = 0) -> int | None:
    if x is None:
        return None
    if isinstance(x, bool) or not isinstance(x, int) or x < low:
        raise ValidationError(f"{name} must be an integer >= {low}, got {x!r}")
    return x


def _pos(name: str, x: object) -> float:
    n = _num(name, x)
    if n <= 0:
        raise ValidationError(f"{name} must be > 0, got {x!r}")
    return float(n)


def _names(name: str, x: object) -> list[str]:
    if x is None:
        return []
    if not isinstance(x, list | tuple) or isinstance(x, str) or not all(isinstance(i, str) and i and i == i.strip() for i in x):  # fmt: skip
        raise ValidationError(
            f"{name} must be a list of non-empty names without surrounding whitespace"
        )
    if len(set(x)) != len(x):
        raise ValidationError(f"{name} lists a name twice")
    return sorted(x)


def _flag(name: str, x: object) -> bool:
    if not isinstance(x, bool):
        raise ValidationError(f"{name} must be a boolean, got {x!r}")
    return x


def _scalar(x: object) -> bool | int | float | str:
    if isinstance(x, bool | str):
        return x
    if isinstance(x, int | float):
        return _num("value", x)
    raise ValidationError(f"category values must be bool, number or string, got {type(x).__name__}")


def _opt_str(name: str, x: object) -> str | None:
    if x is None:
        return None
    if not isinstance(x, str) or not x or x != x.strip():
        raise ValidationError(f"{name} must be a non-empty name or null")
    return x


def _group(name: str, x: object, role: Role) -> dict[str, object]:
    if not isinstance(x, Mapping):
        raise ValidationError(f"{name} must be an object with a `split` and optionally a `window`")
    _strict(x, name, {"split", "window"})
    split, win = _opt_str(f"{name}.split", x.get("split")), x.get("window")
    if split is None:
        raise ValidationError(
            f"{name} needs a declared `split` (a window narrows a split; it never replaces it)"
        )
    out: dict[str, object] = {"split": split, "window": None}
    if win is not None:
        if not isinstance(win, Mapping):
            raise ValidationError(f"{name}.window must be an object")
        w = TemporalWindow.from_dict({k: val for k, val in win.items() if k != "name"}, role)
        out["window"] = {k: val for k, val in w.to_dict().items() if k not in ("name", "role")}
    return out


# -- per-type normalizers -----------------------------------------------------------------------------------


def _n_common(c: Mapping[str, object], allowed: set[str], what: str) -> dict[str, object]:
    _strict(c, f"{what} check", {"features", "examples", *allowed})
    return {"features": _names("features", c.get("features")), "examples": _count("examples", c.get("examples", 20), 0)}  # fmt: skip


def _schema(c: Mapping[str, object]) -> dict[str, object]:
    out = _n_common(c, {"required", "allow_extra"}, "schema")
    return {**out, "required": _names("required", c.get("required")), "allow_extra": _flag("allow_extra", c.get("allow_extra", True))}  # fmt: skip


def _sample_count(c: Mapping[str, object]) -> dict[str, object]:
    _strict(c, "sample_count check", {"min", "max", "expected"})
    lo, hi, ex = (
        _count("min", c.get("min")),
        _count("max", c.get("max")),
        _count("expected", c.get("expected")),
    )
    if lo is not None and hi is not None and lo > hi:
        raise ValidationError("sample_count needs min <= max")
    return {"min": lo, "max": hi, "expected": ex}


def _missingness(c: Mapping[str, object]) -> dict[str, object]:
    out = _n_common(c, {"max_feature_rate", "max_row_rate", "patterns"}, "missingness")
    return {**out, "max_feature_rate": _rate("max_feature_rate", c.get("max_feature_rate")), "max_row_rate": _rate("max_row_rate", c.get("max_row_rate")), "patterns": _count("patterns", c.get("patterns", 5), 0)}  # fmt: skip


def _duplicates(c: Mapping[str, object]) -> dict[str, object]:
    out = _n_common(
        c, {"include_target", "max_duplicate_rate", "max_conflicting_rate"}, "duplicates"
    )
    return {**out, "include_target": _flag("include_target", c.get("include_target", False)), "max_duplicate_rate": _rate("max_duplicate_rate", c.get("max_duplicate_rate")), "max_conflicting_rate": _rate("max_conflicting_rate", c.get("max_conflicting_rate"))}  # fmt: skip


def _identifiers(c: Mapping[str, object]) -> dict[str, object]:
    out = _n_common(c, {"unique_ratio", "min_samples", "require_unique"}, "identifiers")
    ratio = _rate("unique_ratio", c.get("unique_ratio", 0.98))
    if ratio is None or ratio == 0:
        raise ValidationError("unique_ratio must be in (0, 1]")
    return {**out, "unique_ratio": ratio, "min_samples": _count("min_samples", c.get("min_samples", 20), 2), "require_unique": _flag("require_unique", c.get("require_unique", False))}  # fmt: skip


def _numeric(c: Mapping[str, object]) -> dict[str, object]:
    out = _n_common(
        c,
        {
            "bounds",
            "near_constant_ratio",
            "min_variance",
            "outlier_method",
            "outlier_k",
            "max_outlier_rate",
        },
        "numeric",
    )
    raw = c.get("bounds") or {}
    if not isinstance(raw, Mapping):
        raise ValidationError("bounds must map a feature to [low, high]")
    bounds: dict[str, list[int | float | None]] = {}
    for name in sorted(raw):
        b = raw[name]
        if not isinstance(name, str) or not isinstance(b, list | tuple) or len(b) != 2:
            raise ValidationError(f"bounds[{name!r}] must be [low, high] (either may be null)")
        lo, hi = (None if x is None else _num(f"bounds[{name!r}]", x) for x in b)
        if lo is None and hi is None:
            raise ValidationError(f"bounds[{name!r}] needs a low and/or a high")
        if lo is not None and hi is not None and lo > hi:
            raise ValidationError(f"bounds[{name!r}] needs low <= high")
        bounds[name] = [lo, hi]
    method = c.get("outlier_method", "iqr")
    if method not in OUTLIER_METHODS:
        raise ValidationError(f"outlier_method must be one of {list(OUTLIER_METHODS)}")
    ratio = _rate("near_constant_ratio", c.get("near_constant_ratio", 0.99))
    if ratio is None or ratio == 0:
        raise ValidationError("near_constant_ratio must be in (0, 1]")
    mv = c.get("min_variance")
    return {**out, "bounds": bounds, "near_constant_ratio": ratio, "min_variance": None if mv is None else float(_num("min_variance", mv)), "outlier_method": method, "outlier_k": _pos("outlier_k", c.get("outlier_k", 1.5 if method == "iqr" else 3.5)), "max_outlier_rate": _rate("max_outlier_rate", c.get("max_outlier_rate"))}  # fmt: skip


def _categorical(c: Mapping[str, object]) -> dict[str, object]:
    out = _n_common(c, {"allowed", "rare_fraction", "max_cardinality"}, "categorical")
    raw = c.get("allowed") or {}
    if not isinstance(raw, Mapping):
        raise ValidationError("allowed must map a feature to its list of allowed values")
    allowed: dict[str, list[object]] = {}
    for name in sorted(raw):
        vals = raw[name]
        if not isinstance(name, str) or not isinstance(vals, list | tuple) or not vals:
            raise ValidationError(f"allowed[{name!r}] must be a non-empty list of values")
        allowed[name] = [
            x for _, x in sorted({json_key(_scalar(a)): _scalar(a) for a in vals}.items())
        ]
    rare = _rate("rare_fraction", c.get("rare_fraction", 0.01))
    return {**out, "allowed": allowed, "rare_fraction": rare, "max_cardinality": _count("max_cardinality", c.get("max_cardinality"), 1)}  # fmt: skip


def json_key(x: object) -> str:
    return label_key(int(x) if isinstance(x, float) and x.is_integer() else x)


def _target(c: Mapping[str, object]) -> dict[str, object]:
    _strict(c, "target check", {"allowed", "min_class_count", "min_class_proportion", "max_missing_rate", "outlier_method", "outlier_k", "examples"})  # fmt: skip
    allowed = c.get("allowed")
    if allowed is not None:
        if not isinstance(allowed, list | tuple) or not allowed:
            raise ValidationError("allowed must be a non-empty list of target values")
        allowed = [
            x for _, x in sorted({json_key(_scalar(a)): _scalar(a) for a in allowed}.items())
        ]
    method = c.get("outlier_method", "iqr")
    if method not in OUTLIER_METHODS:
        raise ValidationError(f"outlier_method must be one of {list(OUTLIER_METHODS)}")
    return {"allowed": allowed, "min_class_count": _count("min_class_count", c.get("min_class_count"), 1), "min_class_proportion": _rate("min_class_proportion", c.get("min_class_proportion")), "max_missing_rate": _rate("max_missing_rate", c.get("max_missing_rate")), "outlier_method": method, "outlier_k": _pos("outlier_k", c.get("outlier_k", 1.5 if method == "iqr" else 3.5)), "examples": _count("examples", c.get("examples", 20), 0)}  # fmt: skip


def _distribution(c: Mapping[str, object]) -> dict[str, object]:
    out = _n_common(c, {"max_abs_skew", "max_modal_fraction", "min_unique"}, "distribution")
    skew = c.get("max_abs_skew")
    return {**out, "max_abs_skew": None if skew is None else _pos("max_abs_skew", skew), "max_modal_fraction": _rate("max_modal_fraction", c.get("max_modal_fraction")), "min_unique": _count("min_unique", c.get("min_unique"), 1)}  # fmt: skip


def _split_pair(c: Mapping[str, object], what: str, extra: set[str]) -> dict[str, object]:
    _strict(c, f"{what} check", {"reference", "comparison", *extra}, {"reference", "comparison"})
    ref, cmp = _opt_str("reference", c["reference"]), _opt_str("comparison", c["comparison"])
    if ref is None or cmp is None or ref == cmp:
        raise ValidationError(
            f"{what} needs two different declared splits as `reference` and `comparison`"
        )
    return {"reference": ref, "comparison": cmp}


def _split_overlap(c: Mapping[str, object]) -> dict[str, object]:
    out = _split_pair(c, "split_overlap", {"features", "max_overlap", "examples"})
    return {**out, "features": _names("features", c.get("features")), "max_overlap": _count("max_overlap", c.get("max_overlap")), "examples": _count("examples", c.get("examples", 20), 0)}  # fmt: skip


def _temporal_order(c: Mapping[str, object]) -> dict[str, object]:
    out = _split_pair(c, "temporal_order", {"allow_ties", "examples"})
    return {**out, "allow_ties": _flag("allow_ties", c.get("allow_ties", True)), "examples": _count("examples", c.get("examples", 20), 0)}  # fmt: skip


def _target_leakage(c: Mapping[str, object]) -> dict[str, object]:
    _strict(c, "target_leakage check", {"candidates", "max_abs_correlation", "min_purity", "min_rows", "examples"}, {"candidates"})  # fmt: skip
    cands = _names("candidates", c["candidates"])
    if not cands:
        raise ValidationError(
            "target_leakage needs explicitly named candidate features; none are guessed"
        )
    return {"candidates": cands, "max_abs_correlation": _rate("max_abs_correlation", c.get("max_abs_correlation", 0.99)), "min_purity": _rate("min_purity", c.get("min_purity", 0.99)), "min_rows": _count("min_rows", c.get("min_rows", 20), 2), "examples": _count("examples", c.get("examples", 20), 0)}  # fmt: skip


def _group_comparison(c: Mapping[str, object]) -> dict[str, object]:
    _strict(c, "group_comparison check", {"reference", "comparison", "aspects", "features", "examples"}, {"reference", "comparison"})  # fmt: skip
    aspects = _names("aspects", c.get("aspects") or list(ASPECTS))
    if set(aspects) - set(ASPECTS):
        raise ValidationError(f"aspects must be a subset of {list(ASPECTS)}")
    ref, cmp = _group("reference", c["reference"], Role.REFERENCE), _group("comparison", c["comparison"], Role.COMPARISON)  # fmt: skip
    if ref == cmp:
        raise ValidationError("group_comparison needs two different groups")
    return {"reference": ref, "comparison": cmp, "aspects": aspects, "features": _names("features", c.get("features")), "examples": _count("examples", c.get("examples", 20), 0)}  # fmt: skip


NORMALIZERS: dict[str, Callable[[Mapping[str, object]], dict[str, object]]] = {
    "schema": _schema, "sample_count": _sample_count, "missingness": _missingness, "duplicates": _duplicates,
    "identifiers": _identifiers, "numeric": _numeric, "categorical": _categorical, "target": _target,
    "distribution": _distribution, "split_overlap": _split_overlap, "target_leakage": _target_leakage,
    "temporal_order": _temporal_order, "group_comparison": _group_comparison,
}  # fmt: skip


@dataclass(frozen=True)
class CheckSpec:
    type: str
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in NORMALIZERS:
            raise ValidationError(
                f"unknown check type {self.type!r}; choose from {list(CHECK_TYPES)}"
            )
        if not isinstance(self.config, Mapping):
            raise ValidationError("a check configuration must be an object")
        object.__setattr__(
            self, "config", v.freeze_mapping("config", NORMALIZERS[self.type](self.config))
        )

    @property
    def check_id(self) -> str:
        h = content_hash(
            {
                "kind": "quality_check",
                "schema": QUALITY_SCHEMA,
                "type": self.type,
                "config": _thaw(self.config),
            }
        )
        return "qcs_" + h[len(HASH_PREFIX) :][:32]

    def to_dict(self) -> dict[str, object]:
        return {"type": self.type, "config": _thaw(self.config)}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        _strict(d, "check", {"type", "config"}, {"type"})
        cfg = d.get("config", {})
        if not isinstance(cfg, Mapping):
            raise ValidationError("a check configuration must be an object")
        return cls(str(d["type"]), cfg)


def _thaw(x: object) -> object:
    if isinstance(x, Mapping):
        return {k: _thaw(val) for k, val in x.items()}
    if isinstance(x, list | tuple):
        return [_thaw(i) for i in x]
    return x


@dataclass(frozen=True)
class TargetDecl:
    task: str

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValidationError(f"target task must be one of {list(TASKS)}")

    def to_dict(self) -> dict[str, object]:
        return {"task": self.task}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        _strict(d, "target", {"task"}, {"task"})
        return cls(str(d["task"]))


@dataclass(frozen=True)
class QualityConfig:
    min_members: int = 10  # rows a group/slice needs before inference is attempted
    confidence: float = 0.95
    resamples: int = 1000
    permutations: int = 2000
    seed: int = 0
    method: str = "percentile"
    correction: str = "NONE"
    alpha: float = 0.05
    max_examples: int = 20  # default cap on listed affected rows
    slice_checks: tuple[str, ...] = SLICEABLE

    def __post_init__(self) -> None:
        if self.min_members < 2 or self.resamples < 1 or self.permutations < 1 or self.seed < 0 or self.max_examples < 0:  # fmt: skip
            raise ValidationError("min_members >= 2, resamples >= 1, permutations >= 1, seed >= 0, max_examples >= 0")  # fmt: skip
        if not 0.0 < self.confidence < 1.0 or not 0.0 < self.alpha < 1.0:
            raise ValidationError("confidence and alpha must be in (0, 1)")
        if self.method not in st.METHODS or self.correction not in st.CORRECTIONS:
            raise ValidationError(
                f"method in {list(st.METHODS)}, correction in {list(st.CORRECTIONS)}"
            )
        if list(self.slice_checks) != sorted(set(self.slice_checks)) or set(
            self.slice_checks
        ) - set(SLICEABLE):
            raise ValidationError(
                f"slice_checks must be a sorted, unique subset of {list(SLICEABLE)}"
            )

    def to_dict(self) -> dict[str, object]:
        return {"min_members": self.min_members, "confidence": self.confidence, "resamples": self.resamples, "permutations": self.permutations, "seed": self.seed, "method": self.method, "correction": self.correction, "alpha": self.alpha, "max_examples": self.max_examples, "slice_checks": list(self.slice_checks)}  # fmt: skip

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        extra = set(d) - set(cls.__dataclass_fields__)
        if extra:
            raise ValidationError(f"unknown quality setting(s) {sorted(extra)}")
        kw: dict[str, Any] = dict(d)
        if "slice_checks" in kw:
            kw["slice_checks"] = tuple(kw["slice_checks"])
        return cls(**kw)


@dataclass(frozen=True)
class QualitySpec:
    dataset_id: str
    checks: tuple[CheckSpec, ...]
    splits: tuple[str, ...] = ()  # empty: every split the dataset declares (or the whole dataset)
    features: tuple[FeatureDecl, ...] = ()
    target: TargetDecl | None = None
    id_column: str | None = None
    ordering: Ordering | None = None
    missing_values: tuple[
        str, ...
    ] = ()  # extra string tokens that mean "missing" (None and NaN always do)
    slices: tuple[SliceSpec, ...] = ()
    config: QualityConfig = field(default_factory=QualityConfig)

    def __post_init__(self) -> None:
        v.ref("dataset_id", self.dataset_id, "dst")
        if not self.checks or len(self.checks) > MAX_CHECKS:
            raise ValidationError(f"a quality spec needs between 1 and {MAX_CHECKS} checks")
        ids = [c.check_id for c in self.checks]
        if len(set(ids)) != len(ids):
            raise ValidationError("the same check is declared twice")
        object.__setattr__(self, "checks", tuple(sorted(self.checks, key=lambda c: c.check_id)))
        names = [f.name for f in self.features]
        if len(set(names)) != len(names):
            raise ValidationError("a feature is declared twice")
        object.__setattr__(self, "features", tuple(sorted(self.features, key=lambda f: f.name)))
        if len(set(self.splits)) != len(self.splits) or not all(
            isinstance(s, str) and s for s in self.splits
        ):
            raise ValidationError("splits must be unique, non-empty names")
        object.__setattr__(self, "splits", tuple(sorted(self.splits)))
        if any(not isinstance(t, str) for t in self.missing_values):
            raise ValidationError("missing_values must be strings (the empty string is allowed)")
        object.__setattr__(self, "missing_values", tuple(sorted(set(self.missing_values))))
        if len({s.name for s in self.slices}) != len(self.slices) or len({s.slice_id for s in self.slices}) != len(self.slices):  # fmt: skip
            raise ValidationError("quality slices must have unique names and distinct definitions")
        object.__setattr__(self, "slices", tuple(sorted(self.slices, key=lambda s: s.slice_id)))
        self._cross_check()

    def _cross_check(self) -> None:
        declared = {f.name: f.kind for f in self.features}
        for c in self.checks:
            cfg: Mapping[str, Any] = c.config
            t = c.type
            for n in list(cfg.get("features") or []) + list(cfg.get("candidates") or []):
                if n not in declared and t != "target_leakage" and t != "schema":
                    raise ValidationError(
                        f"{t} check names feature {n!r}, which is not declared in `features`"
                    )
                if t == "target_leakage" and n not in declared:
                    raise ValidationError(
                        f"target_leakage candidate {n!r} is not a declared feature"
                    )
            if t in ("numeric", "categorical", "distribution") and not declared:
                raise ValidationError(
                    f"a {t} check needs declared `features` (types are never guessed)"
                )
            if t == "numeric":
                for n in list(cfg["bounds"]) + list(cfg["features"]):
                    if declared.get(n) != "NUMERIC":
                        raise ValidationError(
                            f"numeric check names {n!r}, which is not a declared NUMERIC feature"
                        )
            if t == "categorical":
                for n in list(cfg["allowed"]) + list(cfg["features"]):
                    if declared.get(n) not in ("CATEGORICAL", "BOOLEAN"):
                        raise ValidationError(f"categorical check names {n!r}, which is not a declared CATEGORICAL/BOOLEAN feature")  # fmt: skip
            if t in ("target", "target_leakage") and self.target is None:
                raise ValidationError(f"a {t} check needs a declared `target` (task)")
            if (
                t == "target"
                and cfg["allowed"] is not None
                and self.target
                and self.target.task == "REGRESSION"
            ):
                raise ValidationError("allowed target values apply to classification only")
            if t in ("split_overlap", "temporal_order") and (cfg["reference"] not in self.splits or cfg["comparison"] not in self.splits):  # fmt: skip
                raise ValidationError(f"{t} names splits that are not listed in `splits` ({list(self.splits)}); declare them explicitly")  # fmt: skip
            if t == "temporal_order" and self.ordering is None:
                raise ValidationError(
                    "temporal_order needs a declared `ordering`; time is never inferred"
                )
            if t == "group_comparison":
                for g in (cfg["reference"], cfg["comparison"]):
                    if g["split"] is not None and g["split"] not in self.splits:
                        raise ValidationError(
                            f"group_comparison names split {g['split']!r}, which is not listed in `splits`"
                        )
                    if g["window"] is not None and self.ordering is None:
                        raise ValidationError("a window group needs a declared `ordering`")
                if "target" in cfg["aspects"] and self.target is None:
                    raise ValidationError("the target aspect needs a declared `target`")
        if self.slices and not any(c.type in self.config.slice_checks and c.type in SLICEABLE for c in self.checks):  # fmt: skip
            raise ValidationError("slices were requested but no declared check is slice-capable")

    # -- identity --------------------------------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id, "splits": list(self.splits), "features": [f.to_dict() for f in self.features],
            "target": None if self.target is None else self.target.to_dict(), "id_column": self.id_column,
            "ordering": None if self.ordering is None else self.ordering.to_dict(),
            "missing_values": list(self.missing_values), "checks": [c.to_dict() for c in self.checks],
            "slices": [s.to_dict() for s in self.slices], "config": self.config.to_dict(),
            "analysis_version": ANALYSIS_VERSION, "quality_schema": QUALITY_SCHEMA,
        }  # fmt: skip

    @property
    def spec_id(self) -> str:
        d = self.to_dict()
        d["slices"] = [{"name": s.name, "slice_id": s.slice_id} for s in self.slices]
        return "dqs_" + content_hash(d)[len(HASH_PREFIX) :][:32]

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        allowed = {"dataset_id", "splits", "features", "target", "id_column", "ordering", "missing_values", "checks", "slices", "config", "analysis_version", "quality_schema"}  # fmt: skip
        _strict(d, "quality spec", allowed, {"dataset_id", "checks"})
        if d.get("quality_schema", QUALITY_SCHEMA) != QUALITY_SCHEMA:
            raise ValidationError(f"unsupported quality_schema {d.get('quality_schema')!r}")
        if d.get("analysis_version", ANALYSIS_VERSION) != ANALYSIS_VERSION:
            raise ValidationError(f"unsupported analysis_version {d.get('analysis_version')!r}")

        def objs(k: str) -> Sequence[Mapping[str, Any]]:
            x = d.get(k) or []
            if not isinstance(x, list | tuple) or not all(isinstance(i, Mapping) for i in x):
                raise ValidationError(f"{k} must be a list of objects")
            return x

        def strs(k: str) -> tuple[str, ...]:
            x = d.get(k) or []
            if not isinstance(x, list | tuple) or not all(isinstance(i, str) for i in x):
                raise ValidationError(f"{k} must be a list of strings")
            return tuple(x)

        tgt, order, cfg = d.get("target"), d.get("ordering"), d.get("config") or {}
        if not (isinstance(tgt, Mapping) or tgt is None) or not (isinstance(order, Mapping) or order is None) or not isinstance(cfg, Mapping):  # fmt: skip
            raise ValidationError("target, ordering and config must be objects")
        return cls(
            str(d["dataset_id"]), tuple(CheckSpec.from_dict(c) for c in objs("checks")), strs("splits"),
            tuple(FeatureDecl.from_dict(f) for f in objs("features")),
            None if tgt is None else TargetDecl.from_dict(tgt), _opt_str("id_column", d.get("id_column")),
            None if order is None else Ordering.from_dict(order), strs("missing_values"),
            tuple(SliceSpec.from_dict(s) for s in objs("slices")), QualityConfig.from_dict(cfg),
        )  # fmt: skip

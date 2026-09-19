"""Built-in fault implementations (numpy). Each fault declares typed parameters, the data it
requires, and whether it is stochastic. Mutation guarantee: implementations NEVER modify their
inputs; they return a new array (copy-on-write per batch) in which only affected rows differ.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from experionyx.errors import FaultCompatibilityError, ValidationError
from experionyx.faults.spec import (
    ApplyFn,
    FaultCategory,
    FaultContext,
    FaultParameters,
    FaultRegistry,
    FaultTarget,
    FaultType,
    NoParameters,
    Requirement,
    finite,
    in_range,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    Arr = NDArray[np.generic]
    Mask = NDArray[np.bool_]

P = TypeVar("P", bound=FaultParameters)
_TAB = frozenset({Requirement.FLOAT_ARRAY, Requirement.TABULAR})
_ANY = frozenset({Requirement.FLOAT_ARRAY})
_IMG = frozenset({Requirement.FLOAT_ARRAY, Requirement.IMAGE})
_LAB = frozenset({Requirement.CLASS_LABELS})


def _p(params: FaultParameters, kind: type[P]) -> P:
    if not isinstance(params, kind):
        raise FaultCompatibilityError(f"expected {kind.__name__}, got {type(params).__name__}")
    return params


def _rows(mask: "Mask") -> "NDArray[np.int64]":
    return np.flatnonzero(mask)


def _pos(ctx: FaultContext, i: int) -> int:
    return int(ctx.positions[i])


# --- parameter sets ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GaussianNoiseParams(FaultParameters):
    sigma: float  # standard deviation, in the units of the data

    def __post_init__(self) -> None:
        if finite("sigma", self.sigma) < 0:
            raise ValidationError("sigma must be >= 0")


@dataclass(frozen=True)
class UniformNoiseParams(FaultParameters):
    half_width: float  # noise ~ Uniform(-half_width, +half_width)

    def __post_init__(self) -> None:
        if finite("half_width", self.half_width) < 0:
            raise ValidationError("half_width must be >= 0")


@dataclass(frozen=True)
class FeatureDropoutParams(FaultParameters):
    probability: float  # each element of an affected sample is replaced with `value` w.p. this
    value: float = 0.0

    def __post_init__(self) -> None:
        in_range("probability", self.probability, 0.0, 1.0)
        finite("value", self.value)


@dataclass(frozen=True)
class MissingValuesParams(FaultParameters):
    probability: float
    fill_value: float | None = None  # None means NaN

    def __post_init__(self) -> None:
        in_range("probability", self.probability, 0.0, 1.0)
        if self.fill_value is not None:
            finite("fill_value", self.fill_value)


@dataclass(frozen=True)
class FeatureMaskParams(FaultParameters):
    fraction: float  # share of feature columns masked (the same columns for every sample)
    value: float = 0.0

    def __post_init__(self) -> None:
        in_range("fraction", self.fraction, 0.0, 1.0)
        finite("value", self.value)


@dataclass(frozen=True)
class FeatureScalingParams(FaultParameters):
    factor: float
    fraction: float = 1.0  # share of feature columns scaled

    def __post_init__(self) -> None:
        if finite("factor", self.factor) <= 0:
            raise ValidationError("factor must be > 0")
        in_range("fraction", self.fraction, 0.0, 1.0)


@dataclass(frozen=True)
class FeatureOffsetParams(FaultParameters):
    offset: float
    fraction: float = 1.0

    def __post_init__(self) -> None:
        finite("offset", self.offset)
        in_range("fraction", self.fraction, 0.0, 1.0)


@dataclass(frozen=True)
class FeaturePermutationParams(FaultParameters):
    fraction: float  # share of columns whose values are cyclically permuted among themselves

    def __post_init__(self) -> None:
        in_range("fraction", self.fraction, 0.0, 1.0)


@dataclass(frozen=True)
class OutlierParams(FaultParameters):
    probability: float  # per element
    magnitude: float  # absolute size of the +/- spike, in the units of the data

    def __post_init__(self) -> None:
        in_range("probability", self.probability, 0.0, 1.0)
        if finite("magnitude", self.magnitude) <= 0:
            raise ValidationError("magnitude must be > 0")


@dataclass(frozen=True)
class SaltPepperParams(FaultParameters):
    probability: float  # per element; half become `low`, half `high`
    low: float = 0.0
    high: float = 1.0

    def __post_init__(self) -> None:
        in_range("probability", self.probability, 0.0, 1.0)
        if finite("low", self.low) >= finite("high", self.high):
            raise ValidationError("low must be < high")


@dataclass(frozen=True)
class BrightnessParams(FaultParameters):
    delta: float
    value_min: float = 0.0
    value_max: float = 1.0

    def __post_init__(self) -> None:
        finite("delta", self.delta)
        if finite("value_min", self.value_min) >= finite("value_max", self.value_max):
            raise ValidationError("value_min must be < value_max")


@dataclass(frozen=True)
class ContrastParams(FaultParameters):
    factor: float
    value_min: float = 0.0
    value_max: float = 1.0

    def __post_init__(self) -> None:
        if finite("factor", self.factor) <= 0:
            raise ValidationError("factor must be > 0")
        if finite("value_min", self.value_min) >= finite("value_max", self.value_max):
            raise ValidationError("value_min must be < value_max")


@dataclass(frozen=True)
class OcclusionParams(FaultParameters):
    fraction: float  # side length of the square patch as a share of the image side
    value: float = 0.0

    def __post_init__(self) -> None:
        in_range("fraction", self.fraction, 0.0, 1.0, lo_open=True)
        finite("value", self.value)


@dataclass(frozen=True)
class BoxBlurParams(FaultParameters):
    radius: int  # neighbourhood is (2r+1) x (2r+1); edge pixels are replicated

    def __post_init__(self) -> None:
        if isinstance(self.radius, bool) or not isinstance(self.radius, int) or self.radius < 1:
            raise ValidationError("radius must be an integer >= 1")


@dataclass(frozen=True)
class LabelRateParams(FaultParameters):
    rate: float  # probability that an affected sample's label is corrupted
    classes: tuple[int | str, ...] | None = None  # default: the dataset's class labels

    def __post_init__(self) -> None:
        in_range("rate", self.rate, 0.0, 1.0)
        if self.classes is not None and len(set(self.classes)) < 2:
            raise ValidationError("classes must contain at least two distinct labels")


# --- input faults -----------------------------------------------------------------------------


def gaussian_noise(x: "Arr", mask: "Mask", params: FaultParameters, ctx: FaultContext) -> "Arr":
    p = _p(params, GaussianNoiseParams)
    out = x.copy()
    for i in _rows(mask):
        rng = ctx.row_rng(_pos(ctx, int(i)))
        out[i] = (x[i] + rng.normal(0.0, p.sigma, x[i].shape)).astype(x.dtype)
    return out


def uniform_noise(x: "Arr", mask: "Mask", params: FaultParameters, ctx: FaultContext) -> "Arr":
    p = _p(params, UniformNoiseParams)
    out = x.copy()
    for i in _rows(mask):
        rng = ctx.row_rng(_pos(ctx, int(i)))
        out[i] = (x[i] + rng.uniform(-p.half_width, p.half_width, x[i].shape)).astype(x.dtype)
    return out


def _elementwise(x: "Arr", mask: "Mask", ctx: FaultContext, prob: float, fill: float) -> "Arr":
    out = x.copy()
    for i in _rows(mask):
        drop = ctx.row_rng(_pos(ctx, int(i))).random(x[i].shape) < prob
        out[i] = np.where(drop, np.asarray(fill, dtype=x.dtype), x[i])
    return out


def feature_dropout(x: "Arr", mask: "Mask", params: FaultParameters, ctx: FaultContext) -> "Arr":
    p = _p(params, FeatureDropoutParams)
    return _elementwise(x, mask, ctx, p.probability, p.value)


def missing_values(x: "Arr", mask: "Mask", params: FaultParameters, ctx: FaultContext) -> "Arr":
    p = _p(params, MissingValuesParams)
    return _elementwise(
        x, mask, ctx, p.probability, float("nan") if p.fill_value is None else p.fill_value
    )


def _columns(x: "Arr", ctx: FaultContext, fraction: float) -> "NDArray[np.int64]":
    return ctx.columns(x.shape[1], fraction)


def feature_mask(x: "Arr", mask: "Mask", params: FaultParameters, ctx: FaultContext) -> "Arr":
    p = _p(params, FeatureMaskParams)
    out = x.copy()
    rows, cols = _rows(mask), _columns(x, ctx, p.fraction)
    if len(rows) and len(cols):
        out[np.ix_(rows, cols)] = np.asarray(p.value, dtype=x.dtype)
    return out


def feature_scaling(x: "Arr", mask: "Mask", params: FaultParameters, ctx: FaultContext) -> "Arr":
    p = _p(params, FeatureScalingParams)
    out = x.copy()
    rows, cols = _rows(mask), _columns(x, ctx, p.fraction)
    if len(rows) and len(cols):
        out[np.ix_(rows, cols)] = (x[np.ix_(rows, cols)] * p.factor).astype(x.dtype)
    return out


def feature_offset(x: "Arr", mask: "Mask", params: FaultParameters, ctx: FaultContext) -> "Arr":
    p = _p(params, FeatureOffsetParams)
    out = x.copy()
    rows, cols = _rows(mask), _columns(x, ctx, p.fraction)
    if len(rows) and len(cols):
        out[np.ix_(rows, cols)] = (x[np.ix_(rows, cols)] + p.offset).astype(x.dtype)
    return out


def feature_permutation(
    x: "Arr", mask: "Mask", params: FaultParameters, ctx: FaultContext
) -> "Arr":
    p = _p(params, FeaturePermutationParams)
    cols = _columns(x, ctx, p.fraction)
    if len(cols) < 2:
        raise FaultCompatibilityError(
            f"feature_permutation needs at least 2 selected columns (fraction {p.fraction} of "
            f"{x.shape[1]} features selects {len(cols)})"
        )
    out = x.copy()
    rows = _rows(mask)
    if len(rows):
        out[np.ix_(rows, cols)] = x[np.ix_(rows, np.roll(cols, 1))]
    return out


def outlier_injection(x: "Arr", mask: "Mask", params: FaultParameters, ctx: FaultContext) -> "Arr":
    p = _p(params, OutlierParams)
    out = x.copy()
    for i in _rows(mask):
        rng = ctx.row_rng(_pos(ctx, int(i)))
        hit = rng.random(x[i].shape) < p.probability
        sign = rng.choice(np.array([-1.0, 1.0]), size=x[i].shape)
        out[i] = (x[i] + hit * sign * p.magnitude).astype(x.dtype)
    return out


def salt_and_pepper(x: "Arr", mask: "Mask", params: FaultParameters, ctx: FaultContext) -> "Arr":
    p = _p(params, SaltPepperParams)
    out = x.copy()
    for i in _rows(mask):
        r = ctx.row_rng(_pos(ctx, int(i))).random(x[i].shape)
        out[i] = np.where(
            r < p.probability / 2, p.low, np.where(r < p.probability, p.high, x[i])
        ).astype(x.dtype)
    return out


def brightness(x: "Arr", mask: "Mask", params: FaultParameters, ctx: FaultContext) -> "Arr":
    p = _p(params, BrightnessParams)
    out = x.copy()
    rows = _rows(mask)
    if len(rows):
        out[rows] = np.clip(x[rows] + p.delta, p.value_min, p.value_max).astype(x.dtype)
    return out


def contrast(x: "Arr", mask: "Mask", params: FaultParameters, ctx: FaultContext) -> "Arr":
    p = _p(params, ContrastParams)
    out = x.copy()
    rows = _rows(mask)
    if len(rows):
        sub = x[rows]
        mean = sub.mean(axis=tuple(range(1, sub.ndim)), keepdims=True)
        out[rows] = np.clip((sub - mean) * p.factor + mean, p.value_min, p.value_max).astype(
            x.dtype
        )
    return out


def random_occlusion(x: "Arr", mask: "Mask", params: FaultParameters, ctx: FaultContext) -> "Arr":
    p = _p(params, OcclusionParams)
    out = x.copy()
    h, w = x.shape[-2:]
    ph, pw = max(1, round(p.fraction * h)), max(1, round(p.fraction * w))
    for i in _rows(mask):
        rng = ctx.row_rng(_pos(ctx, int(i)))
        top, left = int(rng.integers(0, h - ph + 1)), int(rng.integers(0, w - pw + 1))
        out[i][..., top : top + ph, left : left + pw] = np.asarray(p.value, dtype=x.dtype)
    return out


def box_blur(x: "Arr", mask: "Mask", params: FaultParameters, ctx: FaultContext) -> "Arr":
    p = _p(params, BoxBlurParams)
    out = x.copy()
    r = p.radius
    h, w = x.shape[-2:]
    for i in _rows(mask):
        a = x[i].astype(np.float64)
        padded = np.pad(a, [(0, 0)] * (a.ndim - 2) + [(r, r), (r, r)], mode="edge")
        total = np.zeros_like(a)
        for dy in range(2 * r + 1):
            for dx in range(2 * r + 1):
                total += padded[..., dy : dy + h, dx : dx + w]
        out[i] = (total / (2 * r + 1) ** 2).astype(x.dtype)
    return out


# --- label faults (operate on the 1-D evaluation targets, not on model inputs) ----------------


def _label_classes(p: LabelRateParams, ctx: FaultContext) -> list[int | str]:
    classes = p.classes or ctx.classes
    if not classes or len(classes) < 2:
        raise FaultCompatibilityError(
            "label faults need the class labels (dataset metadata or `classes`)"
        )
    return list(classes)


def _label_out(y: "Arr", classes: list[int | str]) -> "Arr":
    if y.dtype.kind in "US":  # avoid silent truncation of longer replacement strings
        return y.astype(np.result_type(y.dtype, np.array(classes).dtype))
    return y.copy()


def label_flip(y: "Arr", mask: "Mask", params: FaultParameters, ctx: FaultContext) -> "Arr":
    p = _p(params, LabelRateParams)
    classes = _label_classes(p, ctx)
    out = _label_out(y, classes)
    for i in _rows(mask):
        rng = ctx.row_rng(_pos(ctx, int(i)))
        if rng.random() < p.rate:
            current = y[i].item() if hasattr(y[i], "item") else y[i]
            if current not in classes:
                raise FaultCompatibilityError(
                    f"label {current!r} is not among the classes {classes}"
                )
            others = [c for c in classes if c != current]
            out[i] = others[int(rng.integers(len(others)))]
    return out


def label_randomize(y: "Arr", mask: "Mask", params: FaultParameters, ctx: FaultContext) -> "Arr":
    p = _p(params, LabelRateParams)
    classes = _label_classes(p, ctx)
    out = _label_out(y, classes)
    for i in _rows(mask):
        rng = ctx.row_rng(_pos(ctx, int(i)))
        if rng.random() < p.rate:
            out[i] = classes[int(rng.integers(len(classes)))]  # may re-draw the same label
    return out


# --- registration -----------------------------------------------------------------------------

_V = "1"


def _t(
    name: str,
    cat: FaultCategory,
    target: FaultTarget,
    params: type[FaultParameters],
    requires: frozenset[Requirement],
    desc: str,
    fn: ApplyFn,
    docs: Mapping[str, str],
    stochastic: bool = True,
) -> FaultType:
    return FaultType(name, _V, cat, target, params, desc, requires, stochastic, docs, fn)


def register_builtin_faults(reg: FaultRegistry) -> None:
    C, T = FaultCategory, FaultTarget
    for ft in (
        _t("gaussian_noise", C.NOISE, T.INPUT, GaussianNoiseParams, _ANY, "Add N(0, sigma^2) noise to every element of affected samples", gaussian_noise, {"sigma": "standard deviation, data units, >= 0"}),
        _t("uniform_noise", C.NOISE, T.INPUT, UniformNoiseParams, _ANY, "Add Uniform(-w, w) noise", uniform_noise, {"half_width": "noise half-width, data units, >= 0"}),
        _t("feature_dropout", C.MISSINGNESS, T.INPUT, FeatureDropoutParams, _ANY, "Replace random elements with a constant (default 0)", feature_dropout, {"probability": "per-element, [0, 1]", "value": "replacement value"}),
        _t("missing_values", C.MISSINGNESS, T.INPUT, MissingValuesParams, _ANY, "Replace random elements with NaN (or fill_value); a model that cannot take NaN fails visibly", missing_values, {"probability": "per-element, [0, 1]", "fill_value": "null = NaN"}),
        _t("feature_mask", C.MISSINGNESS, T.INPUT, FeatureMaskParams, _TAB, "Set a fixed random subset of feature columns to a constant", feature_mask, {"fraction": "share of columns, [0, 1]", "value": "constant"}),
        _t("feature_scaling", C.TRANSFORMATION, T.INPUT, FeatureScalingParams, _TAB, "Multiply selected feature columns by a factor (deterministic given the columns)", feature_scaling, {"factor": "> 0", "fraction": "share of columns"}),
        _t("feature_offset", C.TRANSFORMATION, T.INPUT, FeatureOffsetParams, _TAB, "Add a constant to selected feature columns", feature_offset, {"offset": "data units", "fraction": "share of columns"}),
        _t("feature_permutation", C.TRANSFORMATION, T.INPUT, FeaturePermutationParams, _TAB, "Cyclically permute the values of selected columns among themselves (a schema/column-order fault, not permutation importance)", feature_permutation, {"fraction": "share of columns (>= 2 selected)"}),
        _t("outlier_injection", C.INPUT_CORRUPTION, T.INPUT, OutlierParams, _ANY, "Add +/- magnitude spikes to random elements", outlier_injection, {"probability": "per-element", "magnitude": "> 0, data units"}),
        _t("salt_and_pepper", C.NOISE, T.INPUT, SaltPepperParams, _IMG, "Set random elements to low or high", salt_and_pepper, {"probability": "per-element", "low": "pepper value", "high": "salt value"}),
        _t("brightness", C.TRANSFORMATION, T.INPUT, BrightnessParams, _IMG, "Add delta and clip to [value_min, value_max]", brightness, {"delta": "additive shift", "value_min": "clip floor", "value_max": "clip ceiling"}, False),
        _t("contrast", C.TRANSFORMATION, T.INPUT, ContrastParams, _IMG, "Scale deviations from each sample's mean by factor, then clip", contrast, {"factor": "> 0", "value_min": "clip floor", "value_max": "clip ceiling"}, False),
        _t("random_occlusion", C.INPUT_CORRUPTION, T.INPUT, OcclusionParams, _IMG, "Overwrite one random square patch per sample", random_occlusion, {"fraction": "patch side / image side, (0, 1]", "value": "fill value"}),
        _t("box_blur", C.TRANSFORMATION, T.INPUT, BoxBlurParams, _IMG, "Average over a (2r+1)^2 neighbourhood (numpy; no extra dependency)", box_blur, {"radius": "integer >= 1"}, False),
        _t("label_flip", C.LABEL_CORRUPTION, T.LABEL, LabelRateParams, _LAB, "With probability `rate` replace an affected label with a DIFFERENT class (a data-quality fault, not inference robustness)", label_flip, {"rate": "[0, 1]", "classes": "default: dataset classes"}),
        _t("label_randomize", C.LABEL_CORRUPTION, T.LABEL, LabelRateParams, _LAB, "With probability `rate` replace an affected label with a uniformly drawn class (may equal the original)", label_randomize, {"rate": "[0, 1]", "classes": "default: dataset classes"}),
    ):  # fmt: skip
        reg.register(ft)
    # Reserved vocabulary for later phases: registered so they are discoverable, but not runnable.
    for name, cat, desc in (
        (
            "covariate_shift",
            C.DISTRIBUTION_SHIFT,
            "Shift the input distribution (needs a shift model; future)",
        ),
        (
            "temporal_delay",
            C.TEMPORAL_CORRUPTION,
            "Delay/reorder time-indexed inputs (needs temporal datasets; future)",
        ),
        (
            "resource_throttle",
            C.RESOURCE_FAULT,
            "Constrain compute/memory (the adapter layer cannot do this safely yet; future)",
        ),
    ):
        reg.register(
            FaultType(name, _V, cat, T.INPUT, NoParameters, desc, frozenset(), False, {}, None)
        )


def default_fault_registry() -> FaultRegistry:
    """A fresh registry with the built-in faults (no global state)."""
    reg = FaultRegistry()
    register_builtin_faults(reg)
    return reg

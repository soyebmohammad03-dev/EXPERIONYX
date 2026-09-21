"""The typed, immutable, content-addressed definition of one calibration & uncertainty analysis.

A `CalibrationSpec` names EVERYTHING that can change a result: the baseline run whose stored
predictions are analyzed (and, optionally, the model / dataset / split it is expected to be over), the
probability representation the scores are declared to be, the target, the calibration objects, the
binning strategy and count, the post-hoc method and its calibration/evaluation protocol, the
statistical configuration (with seeds), and the explicitly requested slices, temporal windows, stress
analyses and data-quality analyses. Any meaningful change alters `spec_id`. Optional blocks that are
empty are omitted from the identity so adding a feature never renames an older spec.

There is no composite calibration or uncertainty score anywhere in this package."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Self

import experionyx.validation as v
from experionyx.calibration.measures import BINNINGS, QUANTILE_PER_BIN
from experionyx.drift.spec import Ordering, Role, TemporalWindow
from experionyx.errors import ValidationError
from experionyx.hashing import HASH_PREFIX, content_hash
from experionyx.slices.spec import SliceSpec
from experionyx.stats import core as st

CALIBRATION_VERSION = "1"
PREDICTION_SOURCES = ("PREDICT_PROBA", "SOFTMAX_LOGITS")
TARGETS = ("CLASS_LABEL",)
OBJECTS = ("TOP_LABEL", "CLASSWISE")
METHODS = ("NONE", "PLATT", "ISOTONIC")
FIT_MODES = ("SPLIT", "RUN")
MAX_CONTEXTS = (
    100  # slices + windows + stress trials per analysis (each is a separate tested context)
)
POPULATION = "POPULATION"


def _closed(d: Mapping[str, object], what: str, allowed: set[str], required: frozenset[str] = frozenset()) -> None:  # fmt: skip
    extra, missing = set(d) - allowed, set(required) - set(d)
    if extra or missing:
        raise ValidationError(f"malformed {what} (unexpected {sorted(extra)}, missing {sorted(missing)})")  # fmt: skip


def _int(name: str, x: object, lo: int, hi: int) -> int:
    if isinstance(x, bool) or not isinstance(x, int) or not lo <= x <= hi:
        raise ValidationError(f"{name} must be an integer in {lo}..{hi}, got {x!r}")
    return x


def _frac(name: str, x: object, *, open_: bool = True) -> float:
    if isinstance(x, bool) or not isinstance(x, int | float) or not math.isfinite(x) or not (0.0 < x < 1.0 if open_ else 0.0 <= x <= 1.0):  # fmt: skip
        raise ValidationError(f"{name} must be a number in (0, 1), got {x!r}")
    return float(x)


@dataclass(frozen=True)
class Binning:
    strategy: str = "UNIFORM"
    n_bins: int = 10

    def __post_init__(self) -> None:
        if self.strategy not in BINNINGS:
            raise ValidationError(f"binning strategy must be one of {list(BINNINGS)}")
        _int("n_bins", self.n_bins, 1, 1000)

    def to_dict(self) -> dict[str, object]:
        return {"strategy": self.strategy, "n_bins": self.n_bins}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        _closed(d, "binning", {"strategy", "n_bins"})
        return cls(str(d.get("strategy", "UNIFORM")), d.get("n_bins", 10))  # type: ignore[arg-type]


@dataclass(frozen=True)
class Fit:
    """How a post-hoc calibrator's TRAINING data is separated from the EVALUATION data.
    SPLIT: the baseline's samples are split deterministically (seeded hash of the sample ID) into a
    calibration part (`fraction`) and a held-out evaluation part; final numbers use only the held-out
    part. RUN: the calibrator is fitted on a DIFFERENT run's predictions and evaluated on the baseline;
    a run over the same dataset fingerprint and split is refused as leakage."""

    mode: str = "SPLIT"
    fraction: float = 0.5
    seed: int = 0
    calibration_run: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in FIT_MODES:
            raise ValidationError(f"calibration fit mode must be one of {list(FIT_MODES)}")
        _frac("fraction", self.fraction)
        _int("seed", self.seed, 0, 2**63 - 1)
        if self.mode == "RUN":
            v.ref("calibration_run", self.calibration_run, "run")
        elif self.calibration_run is not None:
            raise ValidationError("calibration_run is only meaningful with mode RUN")

    def to_dict(self) -> dict[str, object]:
        if self.mode == "RUN":
            return {"mode": "RUN", "calibration_run": self.calibration_run}
        return {"mode": "SPLIT", "fraction": self.fraction, "seed": self.seed}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        _closed(d, "fit", {"mode", "fraction", "seed", "calibration_run"})
        kw: dict[str, Any] = dict(d)
        kw.setdefault("mode", "SPLIT")
        return cls(**kw)


@dataclass(frozen=True)
class Statistics:
    confidence: float = 0.95
    resamples: int = 500
    permutations: int = 500
    seed: int = 0
    min_samples: int = 30  # below this a context is INSUFFICIENT_EVIDENCE: no metric, no interval
    min_bin_samples: int = QUANTILE_PER_BIN  # bins below this are flagged SPARSE
    min_class_positives: int = 5
    correction: str = "NONE"  # across each family of comparisons, per metric
    alpha: float = 0.05
    practical_delta: float = 0.02  # a difference of at least this many metric units is "material"
    high_confidence: float = 0.9
    low_confidence: float = 0.6

    def __post_init__(self) -> None:
        _frac("confidence", self.confidence)
        _frac("alpha", self.alpha)
        _int("resamples", self.resamples, 1, 100_000)
        _int("permutations", self.permutations, 0, 100_000)
        _int("seed", self.seed, 0, 2**63 - 1)
        _int("min_samples", self.min_samples, 2, 10**9)
        _int("min_bin_samples", self.min_bin_samples, 1, 10**9)
        _int("min_class_positives", self.min_class_positives, 1, 10**9)
        if self.correction not in st.CORRECTIONS:
            raise ValidationError(f"correction must be one of {list(st.CORRECTIONS)}")
        _frac("practical_delta", self.practical_delta, open_=True)
        _frac("high_confidence", self.high_confidence, open_=False)
        _frac("low_confidence", self.low_confidence, open_=False)
        if self.low_confidence > self.high_confidence:
            raise ValidationError("need low_confidence <= high_confidence")

    def to_dict(self) -> dict[str, object]:
        return {**{k: getattr(self, k) for k in self.__dataclass_fields__}, "interval_method": "bootstrap-percentile"}  # fmt: skip

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        kw = {k: x for k, x in d.items() if k != "interval_method"}
        if d.get("interval_method", "bootstrap-percentile") != "bootstrap-percentile":
            raise ValidationError("interval_method must be 'bootstrap-percentile' (the only method: BCa is undefined for non-smooth metrics such as ECE)")  # fmt: skip
        _closed(kw, "statistics", set(cls.__dataclass_fields__))
        return cls(**kw)  # type: ignore[arg-type]


@dataclass(frozen=True)
class WindowSet:
    """Explicitly declared temporal / distribution windows over one ordering, and the pairs compared
    (a, b are window names; default: every later window against the first)."""

    ordering: Ordering
    windows: tuple[TemporalWindow, ...]
    comparisons: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        names = [w.name for w in self.windows]
        if len(self.windows) < 1 or any(not n for n in names) or len(set(names)) != len(names):
            raise ValidationError("windows need at least one window, each with a unique name")
        for a, b in self.comparisons:
            if a == b or a not in names or b not in names:
                raise ValidationError(f"comparison ({a}, {b}) must name two different declared windows")  # fmt: skip
        object.__setattr__(self, "comparisons", tuple(sorted(set(self.comparisons))))

    def pairs(self) -> tuple[tuple[str, str], ...]:
        first = str(self.windows[0].name)
        return self.comparisons or tuple((first, str(w.name)) for w in self.windows[1:])

    def to_dict(self) -> dict[str, object]:
        return {"ordering": self.ordering.to_dict(), "windows": [w.to_dict() for w in self.windows], "comparisons": [list(c) for c in self.comparisons]}  # fmt: skip

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        _closed(d, "windows", {"ordering", "windows", "comparisons"}, frozenset({"ordering", "windows"}))  # fmt: skip
        ws, cs = d["windows"], d.get("comparisons", [])
        if not isinstance(ws, list | tuple) or not isinstance(cs, list | tuple) or any(not isinstance(c, list | tuple) or len(c) != 2 for c in cs):  # fmt: skip
            raise ValidationError("windows must be a list and comparisons a list of [a, b] pairs")
        if not isinstance(d["ordering"], Mapping) or any(not isinstance(w, Mapping) for w in ws):
            raise ValidationError("ordering and every window must be objects")
        return cls(
            Ordering.from_dict(d["ordering"]),
            tuple(TemporalWindow.from_dict(w, Role.COMPARISON) for w in ws),
            tuple((str(c[0]), str(c[1])) for c in cs),
        )


def _ids(name: str, ids: object, prefix: str) -> tuple[str, ...]:
    if isinstance(ids, str) or not isinstance(ids, list | tuple):
        raise ValidationError(f"{name} must be a list of IDs")
    return tuple(sorted({v.ref(name, i, prefix) for i in ids}))


@dataclass(frozen=True)
class CalibrationSpec:
    baseline_run: str
    prediction_source: str  # what the stored scores are DECLARED to be; checked against the run
    binning: Binning = field(default_factory=Binning)
    objects: tuple[str, ...] = ("TOP_LABEL",)
    target: str = "CLASS_LABEL"
    method: str = "NONE"
    fit: Fit = field(default_factory=Fit)
    statistics: Statistics = field(default_factory=Statistics)
    slices: tuple[SliceSpec, ...] = ()
    windows: WindowSet | None = None
    stress_analyses: tuple[str, ...] = ()  # sxa_ IDs whose completed trials are analyzed
    quality_analyses: tuple[str, ...] = ()  # qan_ IDs referenced as data-quality context
    model_id: str | None = None  # optional expectations, verified against the baseline run
    dataset_id: str | None = None
    split: str | None = None

    def __post_init__(self) -> None:
        v.ref("baseline_run", self.baseline_run, "run")
        if self.prediction_source not in PREDICTION_SOURCES:
            raise ValidationError(f"prediction_source must be one of {list(PREDICTION_SOURCES)}: scores are never assumed to be probabilities")  # fmt: skip
        if self.target not in TARGETS:
            raise ValidationError(f"target must be one of {list(TARGETS)} (regression has no calibration object here)")  # fmt: skip
        if not self.objects or list(self.objects) != sorted(set(self.objects)) or set(self.objects) - set(OBJECTS):  # fmt: skip
            raise ValidationError(f"objects must be a sorted, unique subset of {list(OBJECTS)}")
        if self.method not in METHODS:
            raise ValidationError(f"method must be one of {list(METHODS)}")
        if self.method == "NONE" and self.fit != Fit():
            raise ValidationError("a fit protocol is only meaningful with a calibration method")
        if self.method != "NONE" and self.fit.mode == "RUN" and self.fit.calibration_run == self.baseline_run:  # fmt: skip
            raise ValidationError("the calibration run must differ from the evaluated baseline run (calibrating and evaluating on the same data is leakage)")  # fmt: skip
        names = [s.name for s in self.slices]
        if len(set(names)) != len(names) or POPULATION in names:
            raise ValidationError(f"slice names must be unique and not {POPULATION}")
        object.__setattr__(self, "slices", tuple(sorted(self.slices, key=lambda s: s.slice_id)))
        object.__setattr__(self, "stress_analyses", _ids("stress_analyses", self.stress_analyses, "sxa"))  # fmt: skip
        object.__setattr__(self, "quality_analyses", _ids("quality_analyses", self.quality_analyses, "qan"))  # fmt: skip
        if self.model_id is not None:
            v.ref("model_id", self.model_id, "mdl")
        if self.dataset_id is not None:
            v.ref("dataset_id", self.dataset_id, "dst")
        if self.split is not None:
            v.text("split", self.split)
        if len(self.slices) + (len(self.windows.windows) if self.windows else 0) > MAX_CONTEXTS:
            raise ValidationError(f"at most {MAX_CONTEXTS} slices and windows per analysis")

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "calibration_version": CALIBRATION_VERSION, "baseline_run": self.baseline_run,
            "prediction_source": self.prediction_source, "target": self.target,
            "objects": list(self.objects), "binning": self.binning.to_dict(), "method": self.method,
            "statistics": self.statistics.to_dict(),
        }  # fmt: skip
        if self.method != "NONE":
            d["fit"] = self.fit.to_dict()
        if self.slices:
            d["slices"] = [s.to_dict() for s in self.slices]
        if self.windows is not None:
            d["windows"] = self.windows.to_dict()
        for k in ("stress_analyses", "quality_analyses"):
            if getattr(self, k):
                d[k] = list(getattr(self, k))
        for k in ("model_id", "dataset_id", "split"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        return d

    @property
    def spec_id(self) -> str:
        d = self.to_dict()
        if self.slices:  # a slice's description is a label; its name is used in outputs
            d["slices"] = [{"name": s.name, "slice_id": s.slice_id} for s in self.slices]
        return "cbs_" + content_hash(d)[len(HASH_PREFIX) :][:32]

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        allowed = {"calibration_version", "baseline_run", "prediction_source", "target", "objects", "binning", "method", "fit", "statistics", "slices", "windows", "stress_analyses", "quality_analyses", "model_id", "dataset_id", "split"}  # fmt: skip
        _closed(d, "calibration spec", allowed, frozenset({"baseline_run", "prediction_source"}))
        if d.get("calibration_version", CALIBRATION_VERSION) != CALIBRATION_VERSION:
            raise ValidationError(f"unsupported calibration_version {d.get('calibration_version')!r}")  # fmt: skip
        sl = d.get("slices", [])
        objects = d.get("objects", ["TOP_LABEL"])
        if not isinstance(sl, list | tuple) or not isinstance(objects, list | tuple):
            raise ValidationError("slices and objects must be lists")

        def block(key: str, cls_: Any, default: Any) -> Any:
            x = d.get(key)
            if x is None:
                return default
            if not isinstance(x, Mapping):
                raise ValidationError(f"{key} must be an object")
            return cls_.from_dict(x)

        def opt(key: str) -> str | None:
            x = d.get(key)
            if x is not None and not isinstance(x, str):
                raise ValidationError(f"{key} must be a string")
            return x

        return cls(
            str(d["baseline_run"]), str(d["prediction_source"]),
            block("binning", Binning, Binning()), tuple(sorted(str(o) for o in objects)),
            str(d.get("target", "CLASS_LABEL")), str(d.get("method", "NONE")),
            block("fit", Fit, Fit()), block("statistics", Statistics, Statistics()),
            tuple(SliceSpec.from_dict(s) for s in sl), block("windows", WindowSet, None),
            _ids("stress_analyses", d.get("stress_analyses", []), "sxa"),
            _ids("quality_analyses", d.get("quality_analyses", []), "qan"),
            opt("model_id"), opt("dataset_id"), opt("split"),
        )  # fmt: skip

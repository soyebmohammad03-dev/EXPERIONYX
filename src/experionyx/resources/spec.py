"""The typed, immutable, content-addressed definition of one resource & systems measurement.

A `ResourceSpec` names EVERYTHING that can change what is measured: the registered model and dataset
(and split), the timed operation, the requested batch size, the number of samples, the warmup and
measured trial counts, the worker count, the per-trial timeout, the measurement configuration (with
seeds), an optional explicit workload subset (a slice or a window of a baseline run), optional
model-level stress components, references to other analyses, the requested device and a free-text
note about the environment. Any meaningful change alters `spec_id`. Optional blocks that are empty
are omitted from the identity so adding a feature never renames an older spec.

The identity is the DEFINITION of the experiment. It says nothing about the machine: the environment
is captured per run and enters the provenance fingerprint, never the spec. There is no composite
resource or systems score anywhere in this package."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Self

import experionyx.validation as v
from experionyx.adapters.capabilities import DeviceKind
from experionyx.calibration.spec import _closed, _ids, _int
from experionyx.drift.spec import Ordering, Role, TemporalWindow
from experionyx.errors import ValidationError
from experionyx.hashing import HASH_PREFIX, content_hash
from experionyx.slices.spec import SliceSpec
from experionyx.stress.spec import Origin, StressSpec

RESOURCE_VERSION = "1"
OPERATIONS = ("PREDICT", "PREDICT_PROBA")
MODEL_STRESS_FAMILIES = ("PARAMETER_NOISE", "PARAMETER_SCALE", "THRESHOLD")
MAX_BATCH_SIZE = 1_000_000
MAX_SAMPLES = 10_000_000
MAX_REPEATS = 500
MAX_WARMUP = 50
MAX_WORKERS = 64
MAX_TIMEOUT_SECONDS = 86_400.0
SUBSET_KINDS = ("SLICE", "WINDOW")


def _finite(name: str, x: object, lo: float, hi: float) -> float:
    if (
        isinstance(x, bool)
        or not isinstance(x, int | float)
        or not math.isfinite(x)
        or not lo < x <= hi
    ):
        raise ValidationError(f"{name} must be a finite number in ({lo}, {hi}], got {x!r}")
    return float(x)


@dataclass(frozen=True)
class Measurement:
    """What is computed from the raw trial measurements. The raw measurements are always kept."""

    percentiles: tuple[float, ...] = (50.0, 90.0, 95.0, 99.0)
    confidence: float = 0.95
    resamples: int = 500
    seed: int = 0
    min_trials: int = (
        5  # below this many completed measured trials: descriptive statistics only, no interval
    )
    cpu: bool = (
        True  # request process CPU accounting (reported UNAVAILABLE where the platform has none)
    )
    memory: bool = True  # request process memory high-water accounting (likewise)

    def __post_init__(self) -> None:
        ps = tuple(float(p) for p in self.percentiles)
        if (
            not ps
            or any(isinstance(p, bool) or not math.isfinite(p) or not 0.0 < p < 100.0 for p in ps)
            or list(ps) != sorted(set(ps))
        ):
            raise ValidationError(
                "percentiles must be a sorted, unique, non-empty list of numbers in (0, 100)"
            )
        object.__setattr__(self, "percentiles", ps)
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, int | float)
            or not 0.0 < self.confidence < 1.0
        ):
            raise ValidationError(f"confidence must be a number in (0, 1), got {self.confidence!r}")
        _int("resamples", self.resamples, 1, 100_000)
        _int("seed", self.seed, 0, 2**63 - 1)
        _int("min_trials", self.min_trials, 2, MAX_REPEATS)
        if not isinstance(self.cpu, bool) or not isinstance(self.memory, bool):
            raise ValidationError("cpu and memory must be booleans")

    def to_dict(self) -> dict[str, object]:
        return {
            "percentiles": list(self.percentiles),
            "confidence": float(self.confidence),
            "resamples": self.resamples,
            "seed": self.seed,
            "min_trials": self.min_trials,
            "cpu": self.cpu,
            "memory": self.memory,
            "interval_method": "bootstrap-percentile",
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        kw = {k: x for k, x in d.items() if k != "interval_method"}
        if d.get("interval_method", "bootstrap-percentile") != "bootstrap-percentile":
            raise ValidationError(
                "interval_method must be 'bootstrap-percentile' (the only method)"
            )
        _closed(kw, "measurement", set(cls.__dataclass_fields__))
        if "percentiles" in kw:
            ps = kw["percentiles"]
            if isinstance(ps, str) or not isinstance(ps, list | tuple):
                raise ValidationError("percentiles must be a list")
            kw["percentiles"] = tuple(ps)
        return cls(**kw)  # type: ignore[arg-type]


@dataclass(frozen=True)
class Subset:
    """An EXPLICITLY requested workload subset: the samples of a baseline run that belong to a slice
    or a temporal window. Resource behaviour is population-level by default; a subset is meaningful
    only where the workload itself differs (size, content) between the subset and the population."""

    kind: str
    baseline_run: str
    slice: SliceSpec | None = None
    ordering: Ordering | None = None
    window: TemporalWindow | None = None

    def __post_init__(self) -> None:
        if self.kind not in SUBSET_KINDS:
            raise ValidationError(f"subset kind must be one of {list(SUBSET_KINDS)}")
        v.ref("baseline_run", self.baseline_run, "run")
        if self.kind == "SLICE" and (
            self.slice is None or self.ordering is not None or self.window is not None
        ):
            raise ValidationError("a SLICE subset takes exactly a slice")
        if self.kind == "WINDOW" and (
            self.slice is not None or self.ordering is None or self.window is None
        ):
            raise ValidationError("a WINDOW subset takes exactly an ordering and a window")

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {"kind": self.kind, "baseline_run": self.baseline_run}
        if self.slice is not None:
            d["slice"] = self.slice.to_dict()
        if self.ordering is not None and self.window is not None:
            d["ordering"], d["window"] = self.ordering.to_dict(), self.window.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        _closed(
            d,
            "subset",
            {"kind", "baseline_run", "slice", "ordering", "window"},
            frozenset({"kind", "baseline_run"}),
        )
        for k in ("slice", "ordering", "window"):
            if k in d and not isinstance(d[k], Mapping):
                raise ValidationError(f"{k} must be an object")
        return cls(
            str(d["kind"]),
            str(d["baseline_run"]),
            SliceSpec.from_dict(d["slice"]) if "slice" in d else None,  # type: ignore[arg-type]
            Ordering.from_dict(d["ordering"]) if "ordering" in d else None,  # type: ignore[arg-type]
            TemporalWindow.from_dict(d["window"], Role.COMPARISON) if "window" in d else None,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ResourceSpec:
    model_id: str
    dataset_id: str
    split: str | None = None
    operation: str = "PREDICT"
    batch_size: int = 32
    n_samples: int | None = (
        None  # None: every sample of the split (or subset); otherwise the first n
    )
    repeats: int = 10  # measured trials (each is one full pass over the workload)
    warmup_trials: int = (
        1  # full passes run first, recorded, and excluded from steady-state statistics
    )
    workers: int = (
        1  # >1: batches run on a thread pool; needs an adapter that declares thread-safe inference
    )
    timeout_seconds: float | None = None  # per trial; cooperative: checked as each batch completes
    memory_limit_mb: int | None = (
        None  # cannot be enforced in-process: requesting it is refused as UNAVAILABLE
    )
    measurement: Measurement = field(default_factory=Measurement)
    subset: Subset | None = None
    stress: tuple[
        StressSpec, ...
    ] = ()  # model-level stress applied before measuring (identity kept separate)
    calibration_analyses: tuple[
        str, ...
    ] = ()  # cba_ references (execution context, not recomputed)
    stress_analyses: tuple[str, ...] = ()  # sxa_ references
    quality_analyses: tuple[str, ...] = ()  # qan_ references
    device: str = "CPU"
    seed: int = 0
    environment_note: str | None = None  # declared conditions (power mode, other load), free text

    def __post_init__(self) -> None:
        v.ref("model_id", self.model_id, "mdl")
        v.ref("dataset_id", self.dataset_id, "dst")
        if self.split is not None:
            v.text("split", self.split)
        if self.operation not in OPERATIONS:
            raise ValidationError(f"operation must be one of {list(OPERATIONS)}")
        _int("batch_size", self.batch_size, 1, MAX_BATCH_SIZE)
        if self.n_samples is not None:
            _int("n_samples", self.n_samples, 1, MAX_SAMPLES)
        _int("repeats", self.repeats, 1, MAX_REPEATS)
        _int("warmup_trials", self.warmup_trials, 0, MAX_WARMUP)
        _int("workers", self.workers, 1, MAX_WORKERS)
        if self.timeout_seconds is not None:
            object.__setattr__(
                self,
                "timeout_seconds",
                _finite("timeout_seconds", self.timeout_seconds, 0.0, MAX_TIMEOUT_SECONDS),
            )  # 5 and 5.0 are one definition
        if self.memory_limit_mb is not None:
            _int("memory_limit_mb", self.memory_limit_mb, 1, 10**9)
        if self.device not in {d.value for d in DeviceKind}:
            raise ValidationError(f"device must be one of {[d.value for d in DeviceKind]}")
        _int("seed", self.seed, 0, 2**63 - 1)
        if self.environment_note is not None:
            v.text("environment_note", self.environment_note)
        for c in self.stress:
            if c.info.origin is not Origin.MODEL or c.family not in MODEL_STRESS_FAMILIES:
                raise ValidationError(
                    f"stress component {c.family} is not a model-level stress usable here ({list(MODEL_STRESS_FAMILIES)}); batch size and repeats are resource parameters and input stress runs through the Fault Laboratory"
                )
        object.__setattr__(self, "stress", tuple(self.stress))
        for name, prefix in (
            ("calibration_analyses", "cba"),
            ("stress_analyses", "sxa"),
            ("quality_analyses", "qan"),
        ):
            object.__setattr__(self, name, _ids(name, getattr(self, name), prefix))

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "resource_version": RESOURCE_VERSION,
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "split": self.split,
            "operation": self.operation,
            "batch_size": self.batch_size,
            "repeats": self.repeats,
            "warmup_trials": self.warmup_trials,
            "workers": self.workers,
            "measurement": self.measurement.to_dict(),
            "device": self.device,
            "seed": self.seed,
        }
        for k in ("n_samples", "timeout_seconds", "memory_limit_mb", "environment_note"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        if self.subset is not None:
            d["subset"] = self.subset.to_dict()
        if self.stress:
            d["stress"] = [c.to_dict() for c in self.stress]
        for k in ("calibration_analyses", "stress_analyses", "quality_analyses"):
            if getattr(self, k):
                d[k] = list(getattr(self, k))
        return d

    @property
    def spec_id(self) -> str:
        return "rsp_" + content_hash(self.to_dict())[len(HASH_PREFIX) :][:32]

    @property
    def stress_ids(self) -> tuple[str, ...]:
        return tuple(c.stress_id for c in self.stress)

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        allowed = {
            "resource_version",
            "model_id",
            "dataset_id",
            "split",
            "operation",
            "batch_size",
            "n_samples",
            "repeats",
            "warmup_trials",
            "workers",
            "timeout_seconds",
            "memory_limit_mb",
            "measurement",
            "subset",
            "stress",
            "calibration_analyses",
            "stress_analyses",
            "quality_analyses",
            "device",
            "seed",
            "environment_note",
        }
        _closed(d, "resource spec", allowed, frozenset({"model_id", "dataset_id"}))
        if d.get("resource_version", RESOURCE_VERSION) != RESOURCE_VERSION:
            raise ValidationError(f"unsupported resource_version {d.get('resource_version')!r}")
        kw: dict[str, Any] = {k: x for k, x in d.items() if k != "resource_version"}
        for k in ("measurement", "subset"):
            if kw.get(k) is not None and not isinstance(kw[k], Mapping):
                raise ValidationError(f"{k} must be an object")
        if kw.get("measurement") is not None:
            kw["measurement"] = Measurement.from_dict(kw["measurement"])
        else:
            kw.pop("measurement", None)
        kw["subset"] = Subset.from_dict(kw["subset"]) if kw.get("subset") is not None else None
        st = kw.get("stress", [])
        if (
            isinstance(st, str)
            or not isinstance(st, list | tuple)
            or any(not isinstance(c, Mapping) for c in st)
        ):
            raise ValidationError("stress must be a list of stress components")
        kw["stress"] = tuple(StressSpec.from_dict(c) for c in st)
        for k in ("model_id", "dataset_id"):
            kw[k] = str(kw[k])
        return cls(**kw)

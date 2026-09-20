"""Typed, immutable, content-addressed definitions of model stress.

Stress vs fault. The Fault Laboratory applies controlled DATA/SYSTEM faults (noise, missing values,
corruption, label noise). The Stress Laboratory asks how a MODEL behaves when a controlled aspect of
its inputs, parameters, decision rule or evaluation conditions is deliberately stressed. Where a
stress is semantically an input fault (feature scaling, offsets, noise, image corruption) it IS the
Fault Laboratory's implementation: the stress spec maps to a fault spec, trials run through the fault
procedure, and the evidence names the fault it came from. Nothing about those transformations is
re-implemented here. Model-level families (parameter perturbation, decision threshold, batch size,
input shape, repeated execution) are new and exist only where an adapter declares the capability.

A `StressSpec` is one stress (family, parameters, seed, scope). A `StressPlan` orders one or more of
them (compound stress: ORDER IS IDENTITY), optionally sweeps one numeric parameter, repeats over seeds,
or forms a 2x2 factorial (A, B, A then B) for the interaction analysis. `expand()` turns a plan into
explicit `StressUnit`s: every point, repeat and cell is a separate, deterministic, persisted trial.
There is no composite stress or robustness score anywhere in this package."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Self

from experionyx.errors import ValidationError
from experionyx.evaluation.config import EvaluationConfig
from experionyx.faults.lab import with_seed
from experionyx.faults.library import default_fault_registry
from experionyx.faults.spec import FaultScope, FaultSpec, ScopeKind
from experionyx.hashing import HASH_PREFIX, content_hash

STRESS_VERSION = "1"
STRESS_SCHEMA = 1
MAX_TRIALS = 200
MAX_REPEATS = 50


class Origin(StrEnum):
    FAULT_LABORATORY = "FAULT_LABORATORY"  # the transformation IS a Phase 5 fault
    MODEL = "MODEL"  # a derived model, or its decision rule
    EVALUATION = "EVALUATION"  # an evaluation condition (batch size, shape, repetition)


@dataclass(frozen=True)
class Param:
    kind: str  # float | int | names
    low: float | None = None
    high: float | None = None
    low_open: bool = False
    high_open: bool = False
    required: bool = False
    default: object = None


@dataclass(frozen=True)
class Family:
    name: str
    origin: Origin
    target: str  # what is stressed
    description: str
    fault_type: str | None = None  # FAULT_LABORATORY families
    default_sweep: str | None = None
    seeded: bool = False  # does the seed change the result? If not, seeds are refused, never faked
    params: Mapping[str, Param] = field(default_factory=dict)


FAMILIES: dict[str, Family] = {
    f.name: f
    for f in (
        Family(
            "FEATURE_SCALING",
            Origin.FAULT_LABORATORY,
            "INPUT",
            "multiply selected tabular feature columns by a factor",
            "feature_scaling",
            "factor",
            False,
        ),
        Family(
            "FEATURE_OFFSET",
            Origin.FAULT_LABORATORY,
            "INPUT",
            "add a constant to selected tabular feature columns",
            "feature_offset",
            "offset",
            False,
        ),
        Family(
            "FEATURE_NOISE",
            Origin.FAULT_LABORATORY,
            "INPUT",
            "add Gaussian noise of standard deviation sigma to inputs",
            "gaussian_noise",
            "sigma",
            True,
        ),
        Family(
            "IMAGE_NOISE",
            Origin.FAULT_LABORATORY,
            "INPUT",
            "salt-and-pepper noise on image tensors",
            "salt_and_pepper",
            "probability",
            True,
        ),
        Family(
            "BRIGHTNESS",
            Origin.FAULT_LABORATORY,
            "INPUT",
            "shift image brightness and clip",
            "brightness",
            "delta",
            False,
        ),
        Family(
            "CONTRAST",
            Origin.FAULT_LABORATORY,
            "INPUT",
            "scale image contrast and clip",
            "contrast",
            "factor",
            False,
        ),
        Family(
            "OCCLUSION",
            Origin.FAULT_LABORATORY,
            "INPUT",
            "overwrite one random square patch per image",
            "random_occlusion",
            "fraction",
            True,
        ),
        Family(
            "BLUR",
            Origin.FAULT_LABORATORY,
            "INPUT",
            "box blur of an image with an integer radius",
            "box_blur",
            "radius",
            False,
        ),
        Family(
            "PARAMETER_NOISE",
            Origin.MODEL,
            "MODEL_PARAMETERS",
            "add seeded Gaussian noise, relative to each parameter array's RMS, to safely accessible parameters",
            None,
            "relative_sigma",
            True,
            {"relative_sigma": Param("float", 0.0, None, required=True), "targets": Param("names")},
        ),
        Family(
            "PARAMETER_SCALE",
            Origin.MODEL,
            "MODEL_PARAMETERS",
            "multiply safely accessible parameters by a factor",
            None,
            "factor",
            False,
            {
                "factor": Param("float", 0.0, None, low_open=True, required=True),
                "targets": Param("names"),
            },
        ),
        Family(
            "THRESHOLD",
            Origin.MODEL,
            "DECISION_RULE",
            "classify by comparing the positive-class probability with a threshold (binary models with predicted probabilities)",
            None,
            "threshold",
            False,
            {"threshold": Param("float", 0.0, 1.0, True, True, required=True)},
        ),
        Family(
            "BATCH_SIZE",
            Origin.EVALUATION,
            "EVALUATION_CONDITION",
            "evaluate with a different batch size (outputs are expected not to depend on it)",
            None,
            "batch_size",
            False,
            {"batch_size": Param("int", 1, None, required=True)},
        ),
        Family(
            "INPUT_SHAPE",
            Origin.EVALUATION,
            "INPUT_SHAPE",
            "feed inputs of another declared shape (only for adapters that declare supported shapes)",
            None,
            None,
            False,
            {"shape": Param("names", required=True)},
        ),
        Family(
            "REPEATED_EXECUTION",
            Origin.EVALUATION,
            "EXECUTION",
            "evaluate the unmodified model repeatedly; report output stability and latency variation",
            None,
            None,
            False,
            {"repeats": Param("int", 2, MAX_REPEATS, required=True)},
        ),
    )
}
COMPOUND_RANK = {
    "PARAMETER_NOISE": 0,
    "PARAMETER_SCALE": 0,
    "THRESHOLD": 1,
}  # model-level order: parameters, then the rule


def _canon(x: object) -> object:
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return x
    if isinstance(x, int | float):
        if not math.isfinite(x):
            raise ValidationError("stress parameters must be finite")
        return int(x) if isinstance(x, float) and x.is_integer() else x
    if isinstance(x, list | tuple):
        return [_canon(i) for i in x]
    raise ValidationError(f"unsupported stress parameter value {x!r}")


def _check_param(fam: str, name: str, p: Param, value: object) -> object:
    if p.kind == "names":
        if (
            not isinstance(value, list | tuple)
            or not value
            or not all(isinstance(i, str | int) and not isinstance(i, bool) for i in value)
        ):
            raise ValidationError(f"{fam}.{name} must be a non-empty list")
        return list(value) if name == "shape" else sorted({str(i) for i in value})
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValidationError(f"{fam}.{name} must be a finite number, got {value!r}")
    if p.kind == "int" and (isinstance(value, float) and not value.is_integer()):
        raise ValidationError(f"{fam}.{name} must be an integer, got {value!r}")
    if p.low is not None and (value < p.low or (p.low_open and value == p.low)):
        raise ValidationError(f"{fam}.{name}={value!r} is below its valid range")
    if p.high is not None and (value > p.high or (p.high_open and value == p.high)):
        raise ValidationError(f"{fam}.{name}={value!r} is above its valid range")
    return int(value) if p.kind == "int" else _canon(value)


@dataclass(frozen=True)
class StressSpec:
    """One stress. Identity (`sts_…`) covers the version, family, every normalized parameter, the
    seed and the scope."""

    family: str
    parameters: Mapping[str, object] = field(default_factory=dict)
    seed: int = 0
    scope: Mapping[str, object] | None = (
        None  # FAULT_LABORATORY families: which samples (default ALL)
    )
    version: str = STRESS_VERSION

    def __post_init__(self) -> None:
        fam = FAMILIES.get(self.family)
        if fam is None:
            raise ValidationError(
                f"unknown stress family {self.family!r}; choose from {sorted(FAMILIES)}"
            )
        if self.version != STRESS_VERSION:
            raise ValidationError(f"unsupported stress version {self.version!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValidationError("seed must be a non-negative integer")
        if not isinstance(self.parameters, Mapping):
            raise ValidationError("parameters must be an object")
        if fam.origin is Origin.FAULT_LABORATORY:
            spec = self._build_fault(self.parameters, self.scope)  # strict, before any run
            params: dict[str, object] = {k: _canon(v) for k, v in _fault_params(spec).items()}
            scope = _scope_dict(spec.scope)
        else:
            if self.scope is not None:
                raise ValidationError(f"{self.family} takes no scope (it does not select samples)")
            extra = set(self.parameters) - set(fam.params)
            if extra:
                raise ValidationError(f"{self.family} has no parameter(s) {sorted(extra)}")
            params = {}
            for name, p in fam.params.items():
                if name in self.parameters:
                    params[name] = _check_param(self.family, name, p, self.parameters[name])
                elif p.required:
                    raise ValidationError(f"{self.family} needs parameter {name!r}")
                elif p.default is not None:
                    params[name] = p.default
            scope = None
        object.__setattr__(self, "parameters", dict(sorted(params.items())))
        object.__setattr__(self, "scope", scope)
        if not self.seeded and self.seed != 0:
            raise ValidationError(
                f"{self.family} with these parameters is deterministic: it takes no seed (seed must be 0), so no seed is invented"
            )

    # -- structure ---------------------------------------------------------------------------------

    @property
    def info(self) -> Family:
        return FAMILIES[self.family]

    @property
    def seeded(self) -> bool:
        """Does the seed change the result? Stated per family AND parameters, never assumed."""
        fam = self.info
        if fam.origin is Origin.FAULT_LABORATORY and fam.fault_type in (
            "feature_scaling",
            "feature_offset",
        ):
            return float(self.parameters.get("fraction", 1.0)) < 1.0  # type: ignore[arg-type]
        return fam.seeded

    def _build_fault(
        self, parameters: Mapping[str, object], scope: Mapping[str, object] | None
    ) -> FaultSpec:
        fam = FAMILIES[self.family]
        assert fam.fault_type is not None  # noqa: S101
        sc = FaultScope()
        if scope:
            raw: dict[str, Any] = dict(scope)
            sc = FaultScope(
                ScopeKind(str(raw.get("kind", "ALL"))), raw.get("fraction"), raw.get("label")
            )
        kwargs: dict[str, Any] = dict(parameters)
        try:
            return default_fault_registry().make(fam.fault_type, seed=self.seed, scope=sc, **kwargs)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"invalid {self.family} parameters: {exc}") from exc

    def fault_spec(self) -> FaultSpec:
        """The Fault Laboratory spec this stress IS (FAULT_LABORATORY families only)."""
        if self.info.origin is not Origin.FAULT_LABORATORY:
            raise ValidationError(f"{self.family} is not a Fault Laboratory transformation")
        return self._build_fault(self.parameters, self.scope)

    def with_value(self, parameter: str, value: float) -> "StressSpec":
        numeric = {n for n, p in self.info.params.items() if p.kind != "names"}
        if (parameter not in self.parameters and parameter not in numeric) or parameter in {
            "targets",
            "shape",
        }:
            raise ValidationError(f"{self.family} has no parameter {parameter!r} to sweep")
        return StressSpec(
            self.family, {**self.parameters, parameter: value}, self.seed, self.scope, self.version
        )

    def with_seed(self, seed: int) -> "StressSpec":
        return replace(self, seed=seed) if self.seeded else self

    @property
    def stress_id(self) -> str:
        h = content_hash({"kind": "stress", "schema": STRESS_SCHEMA, "identity": self.to_dict()})
        return "sts_" + h[len(HASH_PREFIX) :][:32]

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "version": self.version,
            "parameters": dict(self.parameters),
            "seed": self.seed,
            "scope": None if self.scope is None else dict(self.scope),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        extra = set(d) - {"family", "version", "parameters", "seed", "scope"}
        if extra or "family" not in d:
            raise ValidationError(
                f"malformed stress (unexpected {sorted(extra)}; family is required)"
            )
        params = d.get("parameters", {})
        scope = d.get("scope")
        if not isinstance(params, Mapping) or not (scope is None or isinstance(scope, Mapping)):
            raise ValidationError("parameters and scope must be objects")
        seed = d.get("seed", 0)
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValidationError("seed must be an integer")
        return cls(str(d["family"]), params, seed, scope, str(d.get("version", STRESS_VERSION)))


def _fault_params(spec: FaultSpec) -> dict[str, object]:
    from dataclasses import asdict

    return asdict(spec.parameters)


def _scope_dict(scope: FaultScope) -> dict[str, object] | None:
    if scope.kind is ScopeKind.ALL:
        return None
    return {
        k: v
        for k, v in (
            ("kind", scope.kind.value),
            ("fraction", scope.fraction),
            ("label", scope.label),
        )
        if v is not None
    }


@dataclass(frozen=True)
class Sweep:
    """One numeric parameter of one component varied over explicit values; every value is a trial."""

    parameter: str
    values: tuple[float, ...]
    component: int = 0

    def __post_init__(self) -> None:
        if not self.parameter or not self.values:
            raise ValidationError("a sweep needs a parameter and at least one value")
        if any(
            isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v)
            for v in self.values
        ):
            raise ValidationError("sweep values must be finite numbers")
        if len(set(self.values)) != len(self.values):
            raise ValidationError("sweep values must be unique")
        if self.component < 0:
            raise ValidationError("sweep component must be >= 0")

    def to_dict(self) -> dict[str, object]:
        return {
            "parameter": self.parameter,
            "values": [_canon(v) for v in self.values],
            "component": self.component,
        }


@dataclass(frozen=True)
class StressUnit:
    """One explicit trial: a point of the sweep, a repeat, and (factorial) a cell."""

    point_index: int
    repeat_index: int
    seed: int
    cell: str  # "" (not factorial) | "A" | "B" | "AB"
    parameter: str | None
    value: float | None
    components: tuple[StressSpec, ...]

    @property
    def key(self) -> str:
        h = content_hash(
            {
                "point": self.point_index,
                "repeat": self.repeat_index,
                "cell": self.cell,
                "seed": self.seed,
                "components": [c.stress_id for c in self.components],
            }
        )
        return "sxu_" + h[len(HASH_PREFIX) :][:32]

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "point_index": self.point_index,
            "repeat_index": self.repeat_index,
            "seed": self.seed,
            "cell": self.cell,
            "parameter": self.parameter,
            "value": None if self.value is None else _canon(self.value),
            "components": [c.to_dict() for c in self.components],
            "stress_ids": [c.stress_id for c in self.components],
        }


@dataclass(frozen=True)
class StressPlan:
    components: tuple[StressSpec, ...]
    sweep: Sweep | None = None
    seeds: tuple[int, ...] = (0,)
    factorial: bool = False

    def __post_init__(self) -> None:
        if not self.components:
            raise ValidationError("a stress plan needs at least one stress")
        origins = {c.info.origin for c in self.components}
        if self.factorial:
            if len(self.components) != 2 or origins != {Origin.FAULT_LABORATORY}:
                raise ValidationError(
                    "a 2x2 factorial needs exactly two Fault Laboratory stress factors (A, B) so the interaction analysis can be used"
                )
            if self.sweep is not None:
                raise ValidationError("a factorial design cannot be swept")
            if (
                self.components[0].fault_spec().family_id
                == self.components[1].fault_spec().family_id
            ):
                raise ValidationError("the two factors of a factorial design must differ")
        if len(self.components) > 1 and origins != {Origin.FAULT_LABORATORY}:
            if origins != {Origin.MODEL} or any(
                c.family not in COMPOUND_RANK for c in self.components
            ):
                raise ValidationError(
                    "a compound stress must be all input stress (Fault Laboratory) or model-level stress applied in a meaningful order (parameters, then the decision rule); mixed origins are refused"
                )
            ranks = [COMPOUND_RANK[c.family] for c in self.components]
            if ranks != sorted(ranks):
                raise ValidationError(
                    "model-level compound stress must apply parameter perturbations before the decision threshold"
                )
            if sum(r == 1 for r in ranks) > 1:
                raise ValidationError(
                    "a compound stress may contain at most one decision-rule stress"
                )
        if (
            not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in self.seeds)
        ):
            raise ValidationError("seeds must be a non-empty list of unique non-negative integers")
        rep = [c for c in self.components if c.family == "REPEATED_EXECUTION"]
        if rep and (len(self.components) > 1 or self.sweep is not None or self.seeds != (0,)):
            raise ValidationError("REPEATED_EXECUTION stands alone: no sweep, compound or seeds")
        seeded = any(c.seeded for c in self.components)
        if len(self.seeds) > 1 and not seeded:
            raise ValidationError(
                "this stress is deterministic: several seeds would repeat identical trials. Use REPEATED_EXECUTION to observe repeat-to-repeat behaviour"
            )
        if len(self.seeds) > MAX_REPEATS:
            raise ValidationError(f"at most {MAX_REPEATS} seeds")
        if (
            self.sweep is not None
            and len(self.components) > 1
            and Origin.FAULT_LABORATORY in origins
        ):
            raise ValidationError(
                "a compound input stress cannot be swept (the Fault Laboratory has no parameter of a compound fault to vary); sweep a single stress or run separate plans"
            )
        if self.sweep is not None:
            if self.sweep.component >= len(self.components):
                raise ValidationError("sweep.component is out of range")
            comp = self.components[self.sweep.component]
            for v in self.sweep.values:  # every value must be valid before anything runs
                comp.with_value(self.sweep.parameter, v)
        if len(self.units()) > MAX_TRIALS:
            raise ValidationError(f"{len(self.units())} trials exceed the limit of {MAX_TRIALS}")

    # -- expansion ---------------------------------------------------------------------------------

    @property
    def design(self) -> str:
        if self.factorial:
            return "FACTORIAL_2X2"
        if self.components[0].family == "REPEATED_EXECUTION":
            return "REPEATED_EXECUTION"
        if len(self.components) > 1:
            return "COMPOUND"
        if self.sweep is not None:
            return "SWEEP"
        return "REPEATED" if len(self.seeds) > 1 else "SINGLE"

    def _resolve(self, value: float | None, seed: int) -> tuple[StressSpec, ...]:
        out = []
        for i, c in enumerate(self.components):
            if self.sweep is not None and value is not None and i == self.sweep.component:
                c = c.with_value(self.sweep.parameter, value)
            out.append(c.with_seed(seed))
        return tuple(out)

    def units(self) -> tuple[StressUnit, ...]:
        if self.components[0].family == "REPEATED_EXECUTION":
            n = int(self.components[0].parameters["repeats"])  # type: ignore[call-overload]
            return tuple(StressUnit(0, r, 0, "", None, None, self.components) for r in range(n))
        cells = (("A", 0), ("B", 1), ("AB", None)) if self.factorial else (("", None),)
        points: Sequence[float | None] = self.sweep.values if self.sweep else (None,)
        out: list[StressUnit] = []
        for p, value in enumerate(points):
            for r, seed in enumerate(self.seeds):
                for cell, only in cells:
                    comps = self._resolve(value, seed)
                    if only is not None:
                        comps = (comps[only],)
                    out.append(
                        StressUnit(
                            p,
                            r,
                            seed,
                            cell,
                            self.sweep.parameter if self.sweep else None,
                            value,
                            comps,
                        )
                    )
        return tuple(out)

    def fault_spec_of(self, unit: StressUnit) -> FaultSpec:
        """The Fault Laboratory spec of a unit (FAULT_LABORATORY plans), with the same seed derivation
        the Fault Laboratory uses for compound faults."""
        specs = [c.fault_spec() for c in unit.components]
        if len(specs) == 1:
            return specs[0]
        return with_seed(default_fault_registry().compound(*specs), unit.seed)

    # -- identity ----------------------------------------------------------------------------------

    @property
    def origin(self) -> Origin:
        return self.components[0].info.origin

    def to_dict(self) -> dict[str, object]:
        return {
            "components": [c.to_dict() for c in self.components],
            "sweep": None if self.sweep is None else self.sweep.to_dict(),
            "seeds": list(self.seeds),
            "factorial": self.factorial,
            "design": self.design,
            "stress_version": STRESS_VERSION,
        }

    @property
    def plan_id(self) -> str:
        return "sxp_" + content_hash(self.to_dict())[len(HASH_PREFIX) :][:32]

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Self:
        allowed = {"components", "sweep", "seeds", "factorial", "design", "stress_version"}
        if set(d) - allowed or "components" not in d:
            raise ValidationError(
                f"malformed stress plan (unexpected {sorted(set(d) - allowed)}; components required)"
            )
        if d.get("stress_version", STRESS_VERSION) != STRESS_VERSION:
            raise ValidationError("unsupported stress_version")
        comps = d["components"]
        if not isinstance(comps, list | tuple) or not all(isinstance(c, Mapping) for c in comps):
            raise ValidationError("components must be a list of stress objects")
        sw = d.get("sweep")
        if sw is not None and (
            not isinstance(sw, Mapping)
            or set(sw) - {"parameter", "values", "component"}
            or "parameter" not in sw
            or "values" not in sw
        ):
            raise ValidationError("malformed sweep")
        seeds = d.get("seeds", [0])
        if not isinstance(seeds, list | tuple):
            raise ValidationError("seeds must be a list")
        fact = d.get("factorial", False)
        if not isinstance(fact, bool):
            raise ValidationError("factorial must be a boolean")
        plan = cls(
            tuple(StressSpec.from_dict(c) for c in comps),
            None
            if sw is None
            else Sweep(str(sw["parameter"]), tuple(sw["values"]), int(sw.get("component", 0))),
            tuple(seeds),
            fact,
        )
        if "design" in d and d["design"] != plan.design:
            raise ValidationError(
                f"declared design {d['design']!r} does not match the plan ({plan.design})"
            )
        return plan


@dataclass(frozen=True)
class StressDesign:
    """A whole stress experiment: what is stressed, on which model and data, under which evaluation.
    `baseline_run` (optional) is part of the identity only when supplied explicitly."""

    model_id: str
    dataset_id: str
    plan: StressPlan
    evaluation: EvaluationConfig
    baseline_run: str | None = None
    slices: tuple[Mapping[str, object], ...] = ()  # Phase 11 slice definitions (explicit only)
    min_members: int = 10
    confidence: float = 0.95
    resamples: int = 1000
    permutations: int = 2000
    seed: int = 0
    correction: str = "NONE"
    alpha: float = 0.05
    discover_failures: bool = False
    input_evidence: bool = True

    def __post_init__(self) -> None:
        import experionyx.validation as v
        from experionyx.stats import core as st

        v.ref("model_id", self.model_id, "mdl")
        v.ref("dataset_id", self.dataset_id, "dst")
        if self.baseline_run is not None:
            v.ref("baseline_run", self.baseline_run, "run")
        if self.min_members < 2 or self.resamples < 1 or self.permutations < 1 or self.seed < 0:
            raise ValidationError("min_members >= 2, resamples >= 1, permutations >= 1, seed >= 0")
        if not 0.0 < self.confidence < 1.0 or not 0.0 < self.alpha < 1.0:
            raise ValidationError("confidence and alpha must be in (0, 1)")
        if self.correction not in st.CORRECTIONS:
            raise ValidationError(f"correction must be one of {list(st.CORRECTIONS)}")
        names = [s.get("name") for s in self.slices]
        if len(set(names)) != len(names):
            raise ValidationError("slice names must be unique")

    def to_dict(self) -> dict[str, object]:
        from experionyx.domain import to_jsonable

        d: dict[str, object] = {
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "plan": self.plan.to_dict(),
            "evaluation": to_jsonable(self.evaluation),
            "slices": [dict(s) for s in self.slices],
            "min_members": self.min_members,
            "confidence": self.confidence,
            "resamples": self.resamples,
            "permutations": self.permutations,
            "seed": self.seed,
            "correction": self.correction,
            "alpha": self.alpha,
            "discover_failures": self.discover_failures,
            "input_evidence": self.input_evidence,
            "stress_schema": STRESS_SCHEMA,
        }
        if self.baseline_run is not None:
            d["baseline_run"] = self.baseline_run
        return d

    @property
    def spec_id(self) -> str:
        return "sxr_" + content_hash(self.to_dict())[len(HASH_PREFIX) :][:32]

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Self:
        allowed = {
            "model_id",
            "dataset_id",
            "plan",
            "evaluation",
            "slices",
            "min_members",
            "confidence",
            "resamples",
            "permutations",
            "seed",
            "correction",
            "alpha",
            "discover_failures",
            "input_evidence",
            "stress_schema",
            "baseline_run",
        }
        if set(d) - allowed or not {"model_id", "dataset_id", "plan"} <= set(d):
            raise ValidationError(
                f"malformed stress design (unexpected {sorted(set(d) - allowed)})"
            )
        if d.get("stress_schema", STRESS_SCHEMA) != STRESS_SCHEMA:
            raise ValidationError("unsupported stress_schema")
        plan = d["plan"]
        ev = d.get("evaluation", {})
        if not isinstance(plan, Mapping) or not isinstance(ev, Mapping):
            raise ValidationError("plan and evaluation must be objects")
        sl = d.get("slices") or []
        if not isinstance(sl, list | tuple) or not all(isinstance(s, Mapping) for s in sl):
            raise ValidationError("slices must be a list of slice definitions")
        kw: dict[str, Any] = {
            k: d[k]
            for k in (
                "min_members",
                "confidence",
                "resamples",
                "permutations",
                "seed",
                "correction",
                "alpha",
                "discover_failures",
                "input_evidence",
            )
            if k in d
        }
        return cls(
            str(d["model_id"]),
            str(d["dataset_id"]),
            StressPlan.from_dict(plan),
            EvaluationConfig.from_dict(ev),
            d.get("baseline_run"),
            tuple(dict(s) for s in sl),
            **kw,
        )

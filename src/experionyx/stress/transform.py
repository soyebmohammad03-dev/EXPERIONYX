"""Model-level stress: derived models and decision rules, built WITHOUT touching the registered model.

Parameter stress uses the adapter's optional `ParameterAccess`: arrays are copied, perturbed with a
seeded generator derived from (seed, array name), and handed to `with_parameters`, which builds a NEW
adapter from a private copy of the model. The original is untouched by construction; the build
record still verifies it by digest (`restored`). A threshold stress changes only the decision rule
applied to the model's own predicted probabilities. Randomness comes only from the recorded seed."""

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar

import numpy as np

from experionyx.adapters.base import (
    InferenceResult,
    Inputs,
    ModelAdapter,
    ParameterAccess,
    SampleId,
)
from experionyx.adapters.capabilities import DeviceInfo, ModelCapability
from experionyx.adapters.metadata import ModelMetadata
from experionyx.errors import ExperionyxError
from experionyx.stress.capability import StressUnsupported, _binary_labels
from experionyx.stress.spec import StressSpec


def arrays_digest(arrays: Mapping[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(f"{name}|{a.dtype}|{list(a.shape)}|".encode())
        h.update(a.tobytes())
    return "sha256:" + h.hexdigest()


def _rng(seed: int, name: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(digest[:8], "big")))


def perturb_arrays(arrays: Mapping[str, np.ndarray], spec: StressSpec) -> tuple[dict[str, np.ndarray], dict[str, Any]]:  # fmt: skip
    """(new arrays, per-array record). Deterministic given the spec (seed included)."""
    targets = sorted(spec.parameters.get("targets") or arrays)  # type: ignore[call-overload]
    out = {k: v.copy() for k, v in arrays.items()}
    record: dict[str, Any] = {}
    for name in targets:
        if name not in arrays:
            raise StressUnsupported(
                f"parameter {name!r} is not accessible (available: {sorted(arrays)})"
            )
        base = arrays[name].astype(float)
        if spec.family == "PARAMETER_SCALE":
            new = base * float(spec.parameters["factor"])  # type: ignore[arg-type]
            sigma = None
        else:
            rms = float(np.sqrt(np.mean(base**2)))
            sigma = float(spec.parameters["relative_sigma"]) * rms  # type: ignore[arg-type]
            new = (
                base + _rng(spec.seed, name).normal(0.0, sigma, base.shape)
                if sigma > 0
                else base.copy()
            )
        out[name] = new.astype(arrays[name].dtype)
        delta = out[name].astype(float) - base
        norm = float(np.linalg.norm(base))
        record[name] = {"shape": list(base.shape), "noise_sigma": sigma, "max_abs_change": float(np.max(np.abs(delta))) if delta.size else 0.0, "relative_l2_change": float(np.linalg.norm(delta) / norm) if norm > 0 else None}  # fmt: skip
    return out, record


class StressedModel:
    """A ModelAdapter view of a (possibly derived) model with an optional decision-threshold rule.
    Identity (fingerprint, metadata) is the REGISTERED model's: the stress is recorded separately."""

    NAME: ClassVar[str] = "stressed"
    VERSION: ClassVar[str] = "1.0.0"
    FRAMEWORK: ClassVar[str] = "experionyx-stress"

    def __init__(self, base: ModelAdapter, predictor: ModelAdapter, threshold: float | None = None) -> None:  # fmt: skip
        self._base, self._predictor, self._threshold = base, predictor, threshold
        self.device: DeviceInfo = base.device
        self.load_seconds: float = base.load_seconds
        self._labels = _binary_labels(base) if threshold is not None else None

    @property
    def capabilities(self) -> frozenset[ModelCapability]:
        return self._base.capabilities

    def fingerprint(self) -> str:
        return self._base.fingerprint()

    def metadata(self) -> ModelMetadata:
        return self._base.metadata()

    def predict(
        self, inputs: Inputs, *, sample_ids: Sequence[SampleId] | None = None
    ) -> InferenceResult:
        if self._threshold is None:
            return self._predictor.predict(inputs, sample_ids=sample_ids)
        assert self._labels is not None  # noqa: S101
        probs = self._predictor.predict_proba(inputs, sample_ids=sample_ids)
        neg, pos = self._labels
        rows: Sequence[Any] = probs.outputs
        labels = tuple(pos if float(row[1]) >= self._threshold else neg for row in rows)
        return replace(probs, outputs=labels, capability=ModelCapability.PREDICT)

    def predict_proba(
        self, inputs: Inputs, *, sample_ids: Sequence[SampleId] | None = None
    ) -> InferenceResult:
        return self._predictor.predict_proba(inputs, sample_ids=sample_ids)

    def batch_predict(self, inputs: Inputs, batch_size: int, *, sample_ids: Sequence[SampleId] | None = None) -> InferenceResult:  # fmt: skip
        return self._predictor.batch_predict(inputs, batch_size, sample_ids=sample_ids)

    @classmethod
    def info(cls) -> Any:  # pragma: no cover - not registered as an adapter
        raise ExperionyxError("StressedModel is a wrapper, not a registrable adapter")

    @classmethod
    def load(cls, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise ExperionyxError("StressedModel cannot be loaded from a file")


@dataclass(frozen=True)
class Build:
    model: ModelAdapter
    record: dict[str, Any]


def build_stressed_model(base: ModelAdapter, components: Sequence[StressSpec]) -> Build:
    """Apply model-level components in order (parameters first, then the decision rule)."""
    predictor: ModelAdapter = base
    threshold: float | None = None
    record: dict[str, Any] = {"applied": [], "registered_fingerprint": base.fingerprint()}
    for c in components:
        if c.family in ("PARAMETER_NOISE", "PARAMETER_SCALE"):
            if not isinstance(predictor, ParameterAccess):
                raise StressUnsupported(
                    f"the {base.NAME} adapter does not expose safe parameter access"
                )
            before = predictor.parameter_arrays()
            if not before:
                raise StressUnsupported("this model has no safely accessible numeric parameters")
            d0 = arrays_digest(before)
            new, per = perturb_arrays(before, c)
            derived = predictor.with_parameters(new)
            d_after_original = arrays_digest(predictor.parameter_arrays())
            d_derived = arrays_digest(derived.parameter_arrays()) if isinstance(derived, ParameterAccess) else None  # fmt: skip
            record["applied"].append({"stress_id": c.stress_id, "family": c.family, "seed": c.seed, "parameters_digest_original": d0, "parameters_digest_original_after": d_after_original, "restored": d0 == d_after_original, "parameters_digest_stressed": d_derived, "arrays": per})  # fmt: skip
            if d0 != d_after_original:
                raise StressUnsupported(
                    "the original model's parameters changed while building the stressed model"
                )
            predictor = derived
        elif c.family == "THRESHOLD":
            threshold = float(c.parameters["threshold"])  # type: ignore[arg-type]
            record["applied"].append({"stress_id": c.stress_id, "family": c.family, "threshold": threshold, "default_rule": "argmax of predicted probabilities", "class_labels": list(_binary_labels(base) or ())})  # fmt: skip
        elif c.family in ("BATCH_SIZE", "REPEATED_EXECUTION"):
            record["applied"].append({"stress_id": c.stress_id, "family": c.family, "model_modified": False})  # fmt: skip
        else:
            raise StressUnsupported(f"{c.family} is not a model-level stress")
    return Build(StressedModel(base, predictor, threshold), record)


__all__ = ["Build", "StressedModel", "arrays_digest", "build_stressed_model", "perturb_arrays"]

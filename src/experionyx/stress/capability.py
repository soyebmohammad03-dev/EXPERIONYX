"""What each stress family needs from an adapter, checked BEFORE anything runs.

A family that an adapter cannot support safely is reported UNAVAILABLE with the reason and refused;
it is never approximated with an unsafe hack (no reaching into a model's private state, no
unsupported reshaping)."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experionyx.adapters.base import ModelAdapter, ParameterAccess
from experionyx.adapters.capabilities import DeviceKind, ModelCapability, TaskType
from experionyx.adapters.records import RegisteredDataset, RegisteredModel
from experionyx.errors import ExperionyxError
from experionyx.faults.lab import preflight as fault_preflight
from experionyx.faults.library import default_fault_registry
from experionyx.registry import Registry
from experionyx.stress.spec import Origin, StressPlan, StressSpec

SUPPORTED = "SUPPORTED"
UNAVAILABLE = "UNAVAILABLE"


class StressUnsupported(ExperionyxError):
    """A requested stress cannot be applied safely to this model or dataset (reason attached)."""


def load_model(
    registry: Registry, adapters: Any, inputs_root: Path | None, model_id: str
) -> ModelAdapter:
    """The registered model's adapter, loaded (CPU) and fingerprint-verified."""
    if adapters is None:
        raise StressUnsupported("no adapter registry is available to load the model")
    rec = registry.get(RegisteredModel, model_id)
    if rec.source is None:
        raise StressUnsupported(f"model {rec.name} has no source to load from")
    src = rec.source
    path = src if src.startswith("builtin:") or src.startswith("/") or inputs_root is None else str(inputs_root / src)  # fmt: skip
    cls = adapters.models.resolve(rec.adapter)
    model: ModelAdapter = cls.load(
        path, version=rec.version, device=DeviceKind.CPU, options=rec.options
    )
    if model.fingerprint() != rec.fingerprint:
        raise StressUnsupported(
            f"model {rec.name} changed since registration (fingerprint mismatch)"
        )
    return model


def _binary_labels(model: ModelAdapter) -> tuple[Any, ...] | None:
    schema = model.metadata().output_schema
    labels = None if schema is None else schema.class_labels
    return tuple(labels) if labels is not None and len(labels) == 2 else None


def family_status(
    model: ModelAdapter, spec: StressSpec, data: RegisteredDataset
) -> tuple[str, str | None]:
    """(SUPPORTED | UNAVAILABLE, reason) of one stress on this model and dataset."""
    fam = spec.family
    origin = spec.info.origin
    if origin is Origin.FAULT_LABORATORY:
        try:
            fault_preflight(spec.fault_spec(), default_fault_registry(), data)
        except ExperionyxError as exc:
            return (
                UNAVAILABLE,
                f"the Fault Laboratory refuses this transformation on this dataset: {exc}",
            )
        return SUPPORTED, None
    if fam in ("PARAMETER_NOISE", "PARAMETER_SCALE"):
        if not isinstance(model, ParameterAccess):
            return UNAVAILABLE, f"the {model.NAME} adapter does not expose safe parameter access"
        names = set(model.parameter_arrays())
        if not names:
            return UNAVAILABLE, "this model has no safely accessible numeric parameters (the adapter exposes none for this estimator)"  # fmt: skip
        wanted: Any = spec.parameters.get("targets") or []
        missing = sorted(set(wanted) - names)
        if missing:
            return UNAVAILABLE, f"parameter target(s) {missing} are not accessible (available: {sorted(names)})"  # fmt: skip
        return SUPPORTED, None
    if fam == "THRESHOLD":
        if ModelCapability.PREDICT_PROBA not in model.capabilities:
            return UNAVAILABLE, "the model does not provide predicted probabilities, so a decision threshold is not defined for it"  # fmt: skip
        if model.metadata().task is not TaskType.CLASSIFICATION or _binary_labels(model) is None:
            return UNAVAILABLE, "a decision threshold is only defined here for binary classifiers with two declared class labels"  # fmt: skip
        return SUPPORTED, None
    if fam == "INPUT_SHAPE":
        declared = getattr(type(model), "SUPPORTED_INPUT_SHAPES", None)
        raw_shape: Any = spec.parameters["shape"]
        want = tuple(int(x) for x in raw_shape)
        if not declared:
            return UNAVAILABLE, f"the {model.NAME} adapter does not declare any supported input shapes; shape stress is refused, not emulated"  # fmt: skip
        if want not in {tuple(s) for s in declared}:
            return UNAVAILABLE, f"shape {list(want)} is not among the adapter's declared shapes {[list(s) for s in declared]}"  # fmt: skip
        return SUPPORTED, None
    if ModelCapability.PREDICT not in model.capabilities:
        return UNAVAILABLE, "the model cannot predict"
    return SUPPORTED, None


def capability_report(model: ModelAdapter, plan: StressPlan, data: RegisteredDataset) -> list[dict[str, Any]]:  # fmt: skip
    rows = []
    for i, c in enumerate(plan.components):
        status, reason = family_status(model, c, data)
        rows.append({"component": i, "family": c.family, "stress_id": c.stress_id, "origin": c.info.origin.value, "status": status, "reason": reason})  # fmt: skip
    return rows


def refuse_unavailable(report: Sequence[Mapping[str, Any]]) -> None:
    bad = [r for r in report if r["status"] != SUPPORTED]
    if bad:
        raise StressUnsupported(
            "; ".join(f"{r['family']} is UNAVAILABLE: {r['reason']}" for r in bad)
        )

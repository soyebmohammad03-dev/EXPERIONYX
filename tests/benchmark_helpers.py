"""Helpers for benchmark tests: a real registry world and small, real benchmark specs."""

from pathlib import Path
from typing import Any

from eval_helpers import EvalWorld
from experionyx.adapters.records import RegisteredDataset, RegisteredModel
from experionyx.benchmark.spec import BenchmarkLimits, BenchmarkSpec, FaultGrid, InteractionPair
from experionyx.evaluation.config import EvaluationConfig
from experionyx.faults.library import default_fault_registry
from experionyx.interactions.config import InteractionConfig
from interaction_helpers import world

FR = default_fault_registry()
EVAL = EvaluationConfig(split="test")


def ids(w: EvalWorld) -> tuple[str, str]:
    return (
        next(m.id for m in w.registry.find(RegisteredModel) if m.name == "model"),
        next(d.id for d in w.registry.find(RegisteredDataset) if d.name == "data"),
    )


def spec_for(w: EvalWorld, **over: Any) -> BenchmarkSpec:
    model, data = ids(w)
    body: dict[str, Any] = {
        "name": "noise-dropout", "version": "1.0.0", "model": model, "dataset": data,
        "faults": (FaultGrid("noise", "gaussian_noise", sweep_parameter="sigma", values=(2.0, 6.0)), FaultGrid("dropout", "feature_dropout", sweep_parameter="probability", values=(0.5,))),
        "seeds": (1, 2, 3), "evaluation": EVAL, "interactions": (InteractionPair("noise", "dropout", 6.0, 0.5),),
        "interaction_config": InteractionConfig(bootstrap_resamples=200), "aggregation_resamples": 200,
    }  # fmt: skip
    body.update(over)
    return BenchmarkSpec(**body)


def small_world(tmp_path: Path) -> EvalWorld:
    return world(tmp_path)


__all__ = ["FR", "BenchmarkLimits", "small_world", "spec_for"]

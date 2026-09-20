"""A controlled stress VALIDATION fixture: a framework-free binary linear classifier
(`LinearProbaAdapter`, with the optional ParameterAccess capability) on 120 two-feature samples, split
"test" = all of them. The label is 1 when x0 + 0.5*x1 > 0 with every 9th label flipped, so the model
has known accuracy and non-trivial probabilities. Every stress result recomputed in the tests comes
from the raw rows and the model's own formula, never through the stress engine."""

import math
from pathlib import Path
from typing import Any

from eval_helpers import EvalWorld, eval_world
from experionyx.adapters.records import RegisteredDataset, RegisteredModel
from experionyx.evaluation.config import EvaluationConfig
from pure_adapters import LinearProbaAdapter

N = 120
COEF = [1.0, 0.5]
EVAL = EvaluationConfig(split="test", batch_size=32)


def rows() -> tuple[list[list[float]], list[int]]:
    xs = [[((i * 37) % 41 - 20) / 10, ((i * 53) % 47 - 23) / 10] for i in range(N)]
    ys = [1 if x[0] + 0.5 * x[1] > 0 else 0 for x in xs]
    for i in range(8, N, 9):
        ys[i] = 1 - ys[i]
    return xs, ys


def p1(x: list[float], coef: list[float] = COEF, b: float = 0.0) -> float:
    return 1.0 / (1.0 + math.exp(-(b + sum(c * v for c, v in zip(coef, x, strict=True)))))


def data(**over: Any) -> dict[str, Any]:
    xs, ys = rows()
    d: dict[str, Any] = {"rows": xs, "targets": ys, "task": "CLASSIFICATION", "feature_names": ["x0", "x1"], "splits": {"test": list(range(N))}}  # fmt: skip
    d.update(over)
    return d


def stress_world(tmp_path: Path, **over: Any) -> tuple[EvalWorld, str, str]:
    """(world, registered model id, registered dataset id)."""
    w = eval_world(tmp_path, model={"coef": COEF, "intercept": 0.0}, data=data(**over), config=EVAL, model_cls=LinearProbaAdapter)  # fmt: skip
    (m,) = [m for m in w.registry.find(RegisteredModel) if m.name == "model"]
    (d,) = [d for d in w.registry.find(RegisteredDataset) if d.name == "data"]
    return w, m.id, d.id

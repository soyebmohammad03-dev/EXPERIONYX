"""A controlled temporal validation fixture for drift tests. VALIDATION DATA, not a research finding:
every distributional change below is written into the data on purpose so the expected behaviour is
known in advance. Two halves of 80 samples each, ordered by sample index (index 0..79 = early,
80..159 = late) and by a `ts` timestamp feature with the same ordering.

column            early (0..79)                     late (80..159)                 designed change
x0                i % 10 (0..9 uniform)             5 + (i % 10) / 2  (5..9.5)     shifted numeric, drives predictions
stable            i % 7                             i % 7                          identical distribution
shifted           i % 10                            i % 10 + 6                     location shift
const             3.0                               3.0                            constant feature
nanheavy          None for half, inf for 1 in 16, else i % 10  same                       missing/non-finite-heavy, otherwise stable
site (category)   a,b,c evenly                      c for 3 in 4, else a/b         changed proportions (+ new 'd')
flag (boolean)    alternates                        alternates                     identical
ts                float(i)                          float(i)                       the timestamp ordering
target            x0 > 5                            1 for 3 in 4 (class balance)   label shift
The threshold model predicts x0 > 5, so predictions shift and accuracy falls in the late half."""

from pathlib import Path
from typing import Any

from eval_helpers import EvalWorld, eval_world
from experionyx.adapters.registry import AdapterRegistries
from experionyx.evaluation.config import EvaluationConfig
from experionyx.registry import Registry
from pure_adapters import ListDatasetAdapter

N = 160
HALF = N // 2
FEATURES = ["x0", "stable", "shifted", "const", "nanheavy", "site", "flag", "ts"]
EVAL = EvaluationConfig(split="test")


def rows() -> tuple[list[list[Any]], list[int]]:
    out: list[list[Any]] = []
    targets: list[int] = []
    for i in range(N):
        late = i >= HALF
        x0 = 5 + (i % 10) / 2 if late else float(i % 10)
        site = (
            ("c" if i % 4 else ("a" if i % 8 == 0 else "d" if i % 16 == 4 else "b"))
            if late
            else "abc"[i % 3]
        )
        out.append([x0, float(i % 7), float(i % 10) + (6 if late else 0), 3.0, (None if i % 2 else (float('inf') if i % 16 == 0 else float(i % 10))), site, i % 2 == 0, float(i)])  # fmt: skip
        targets.append((1 if i % 4 else 0) if late else int(x0 > 5))
    return out, targets


def data(**over: Any) -> dict[str, Any]:
    r, t = rows()
    d: dict[str, Any] = {"rows": r, "targets": t, "task": "CLASSIFICATION", "feature_names": FEATURES, "splits": {"test": list(range(N))}}  # fmt: skip
    d.update(over)
    return d


def drift_world(tmp_path: Path, **over: Any) -> tuple[EvalWorld, str]:
    """(world, baseline run id): the threshold model evaluated on the whole fixture."""
    import json

    w = eval_world(
        tmp_path, model={"threshold": 5.0, "proba": True}, data=data(**over), config=EVAL
    )
    res = w.run()
    assert res.status.value == "COMPLETED"
    del json
    return w, res.run.id


def dataset_of(w: EvalWorld) -> Any:
    return ListDatasetAdapter.load(w.workspace / "m" / "data.json", version="1", options={})


def spec_dict(baseline: str, **over: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "baseline_run": baseline,
        "ordering": {"field": "index"},
        "reference": {"start": 0, "end": HALF},
        "comparisons": [{"start": HALF, "end": N}],
        "features": [
            {"name": "x0", "type": "NUMERIC"},
            {"name": "stable", "type": "NUMERIC"},
            {"name": "shifted", "type": "NUMERIC"},
            {"name": "const", "type": "NUMERIC"},
            {"name": "nanheavy", "type": "NUMERIC"},
            {"name": "site", "type": "CATEGORICAL"},
            {"name": "flag", "type": "BOOLEAN"},
        ],
        "config": {"min_samples": 10, "resamples": 100, "permutations": 200},
    }
    d.update(over)
    return d


__all__ = [
    "EVAL",
    "HALF",
    "AdapterRegistries",
    "N",
    "Registry",
    "dataset_of",
    "drift_world",
    "spec_dict",
]

"""A controlled data-quality VALIDATION fixture. VALIDATION DATA, not a finding: every defect below is
written in on purpose so the expected behaviour is known. 160 rows, splits train = rows 0..119 and
test = rows 110..159 (so ids 110..119 are in both).

column   contents                                                        designed property
row_id   i, except row 41 which repeats 40                               duplicate + conflicting id
age      20 + (7i mod 40); row 3 = 900, row 4 = -5, row 9 = +inf         out of bounds, negative, non-finite
income   None when i%5 == 0, else 1000 + 10*(i%13); row 6 = "n/a"         20% missing, one invalid string
site     a/b/c cycling; row 7 = "z", row 8 = "q"                          rare 'z' (allowed), invalid 'q'
flag     i even                                                          boolean
const    7.0                                                             constant
score    3i                                                              integer, all distinct: identifier-like
leak     the target's own value (row 101 excepted)                        target leakage candidate
ts       float(i); row 5 = None                                          ordering key, one missing
target   1 when i%10 == 0 else 0; row 12 = 2 (singleton), row 13 = None  imbalance, singleton class, missing
Row 100 repeats row 99 (except row_id); row 101 repeats its features with a flipped target; row 150
repeats row 5's features (a vector present in train AND test) with the opposite target."""

from pathlib import Path
from typing import Any

from eval_helpers import EvalWorld, eval_world
from experionyx.adapters.records import RegisteredDataset
from experionyx.evaluation.config import EvaluationConfig
from pure_adapters import ListDatasetAdapter

N = 160
COLUMNS = ["row_id", "age", "income", "site", "flag", "const", "score", "leak", "ts"]
INF = float("inf")


def rows() -> tuple[list[list[Any]], list[Any]]:
    out: list[list[Any]] = []
    tg: list[Any] = []
    for i in range(N):
        t: Any = 1 if i % 10 == 0 else 0
        if i == 12:
            t = 2
        row: list[Any] = [i, float(20 + (7 * i) % 40), None if i % 5 == 0 else 1000.0 + 10 * (i % 13), "abc"[i % 3], i % 2 == 0, 7.0, 3 * i, float(t), float(i)]  # fmt: skip
        out.append(row)
        tg.append(t)
    out[41][0] = 40
    out[3][1], out[4][1], out[9][1] = 900.0, -5.0, INF
    out[6][2] = "n/a"
    out[7][3], out[8][3] = "z", "q"
    out[5][8] = None
    out[100] = [100, *out[99][1:]]
    tg[100] = tg[99]
    out[101] = [101, *out[99][1:]]
    tg[101] = 1 - tg[99] if isinstance(tg[99], int) and tg[99] in (0, 1) else 1
    out[150] = [150, *out[5][1:8], 150.0]
    tg[150] = 1 - tg[5]
    tg[13] = None
    return out, tg


def data(**over: Any) -> dict[str, Any]:
    r, t = rows()
    d: dict[str, Any] = {"rows": r, "targets": t, "task": "CLASSIFICATION", "feature_names": COLUMNS, "splits": {"train": list(range(0, 120)), "test": list(range(110, N))}}  # fmt: skip
    d.update(over)
    return d


def quality_world(tmp_path: Path, **over: Any) -> tuple[EvalWorld, str]:
    """(world, registered dataset id). The world's model/experiment are unused by quality analyses."""
    w = eval_world(tmp_path, model={"threshold": 5.0, "proba": True}, data=data(**over), config=EvaluationConfig(split="test"))  # fmt: skip
    (rec,) = [d for d in w.registry.find(RegisteredDataset) if d.name == "data"]
    return w, rec.id


def dataset_of(w: EvalWorld) -> Any:
    return ListDatasetAdapter.load(w.workspace / "m" / "data.json", version="1", options={})


FEATURES = [
    {"name": "age", "type": "NUMERIC"}, {"name": "income", "type": "NUMERIC"}, {"name": "site", "type": "CATEGORICAL"},
    {"name": "flag", "type": "BOOLEAN"}, {"name": "const", "type": "NUMERIC"}, {"name": "score", "type": "NUMERIC"},
    {"name": "leak", "type": "NUMERIC"},
]  # fmt: skip


def spec_dict(dataset_id: str, checks: list[dict[str, Any]], **over: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "dataset_id": dataset_id, "splits": ["test", "train"], "features": FEATURES, "target": {"task": "CLASSIFICATION"},
        "id_column": "row_id", "ordering": {"field": "feature:ts"}, "checks": checks,
        "config": {"min_members": 10, "resamples": 100, "permutations": 200},
    }  # fmt: skip
    d.update(over)
    return d

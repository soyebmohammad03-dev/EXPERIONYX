"""The rows a quality analysis reads. A `Table` is one split (or the whole dataset) as the adapter
serves it: sample IDs (dataset indices), the raw cells of each row, the raw target. Nothing is
coerced here; cells stay as stored so a check can tell missing from invalid from valid. Structural
problems (ragged rows, a target of the wrong length, repeated sample IDs, a wrong sample count in the
metadata) are recorded on the table and reported by the checks, never repaired."""

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from experionyx.adapters.base import DatasetAdapter
from experionyx.drift.measures import clean
from experionyx.errors import ExperionyxError
from experionyx.hashing import content_hash

MAX_ROWS = 2_000_000


class QualityDataError(ExperionyxError):
    """The data a quality analysis needs cannot be read (reason attached); never approximated."""


def _py(x: Any) -> Any:
    return x.item() if hasattr(x, "item") and not isinstance(x, str | bytes) else x


def _as_list(x: object) -> list[Any]:
    return x.tolist() if hasattr(x, "tolist") else list(x)  # type: ignore[call-overload, no-any-return]


@dataclass(frozen=True)
class Table:
    split: str | None
    ids: tuple[int, ...]
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    target: tuple[Any, ...] | None
    ragged: tuple[int, ...]  # sample IDs whose row width differs from the number of columns
    duplicate_ids: tuple[int, ...]  # sample IDs the adapter served more than once
    declared_n: int | None  # the adapter's own count for this split
    target_len: int | None  # how many targets the adapter served (None: no target)

    @property
    def n(self) -> int:
        return len(self.ids)

    def index(self, name: str) -> int:
        try:
            return self.columns.index(name)
        except ValueError:
            raise QualityDataError(f"{name!r} is not a column of this dataset (columns: {list(self.columns)})") from None  # fmt: skip

    def column(self, name: str) -> list[Any]:
        j = self.index(name)
        return [r[j] if j < len(r) else None for r in self.rows]

    def select(self, positions: Sequence[int]) -> "Table":
        ids = tuple(self.ids[i] for i in positions)
        keep = set(ids)
        return Table(
            self.split, ids, self.columns, tuple(self.rows[i] for i in positions),
            None if self.target is None else tuple(self.target[i] for i in positions),
            tuple(i for i in self.ragged if i in keep), tuple(i for i in self.duplicate_ids if i in keep),
            None, None if self.target is None else len(ids),
        )  # fmt: skip

    def digest(self) -> str:
        return content_hash({"ids": list(self.ids)})


def dataset_columns(dataset: DatasetAdapter) -> tuple[str, ...]:
    meta = dataset.metadata()
    names = tuple(meta.input_schema.feature_names or ()) if meta.input_schema else ()
    if names:
        return tuple(str(n) for n in names)
    width = meta.num_features or (meta.input_schema.shape[-1] if meta.input_schema and meta.input_schema.shape else None)  # fmt: skip
    if width is None:
        raise QualityDataError("the dataset declares neither feature names nor a feature count")
    return tuple(str(i) for i in range(int(width)))


def load_table(dataset: DatasetAdapter, split: str | None, columns: tuple[str, ...]) -> Table:
    ids: list[int] = []
    rows: list[tuple[Any, ...]] = []
    target: list[Any] = []
    saw_target = False
    for batch in dataset.batches(1024, split):
        chunk = [tuple(_py(c) for c in _as_list(r)) for r in _as_list(batch.inputs)]
        ids.extend(int(i) for i in batch.indices)
        rows.extend(chunk)
        if batch.target is not None:
            saw_target = True
            target.extend(_py(t) for t in _as_list(batch.target))
        if len(ids) > MAX_ROWS:
            raise QualityDataError(f"more than {MAX_ROWS} rows; analyze a split or a subset")
    try:
        declared = dataset.num_samples(split)
    except ExperionyxError:
        declared = None
    counts = Counter(ids)
    aligned = saw_target and len(target) == len(ids)
    return Table(
        split, tuple(ids), columns, tuple(rows), tuple(target) if aligned else None,
        tuple(i for i, r in zip(ids, rows, strict=True) if len(r) != len(columns)),
        tuple(sorted(i for i, c in counts.items() if c > 1)), declared, len(target) if saw_target else None,
    )  # fmt: skip


# -- cell classification ------------------------------------------------------------------------------------


def is_missing_token(x: object, tokens: frozenset[str]) -> bool:
    return isinstance(x, str) and x in tokens


def classify(values: Sequence[Any], kind: str, tokens: frozenset[str]) -> list[tuple[str, Any]]:
    """Per cell: ('valid', normalized) | ('missing', None) | ('nonfinite', None) | ('invalid', None).
    Missing means None, NaN or a declared missing token; the rest follows the Phase 12 cleaning rules."""
    out: list[tuple[str, Any]] = []
    for x in values:
        if is_missing_token(x, tokens):
            out.append(("missing", None))
            continue
        v, c = clean([x], kind)
        state = (
            "valid"
            if v
            else "missing"
            if c.n_missing
            else "nonfinite"
            if c.n_nonfinite
            else "invalid"
        )
        out.append((state, v[0] if v else None))
    return out


def cell_key(x: Any) -> str:
    """A canonical, hashable spelling of any cell (NaN == NaN, 1 == 1.0, None distinct from 'None')."""
    if isinstance(x, float):
        return (
            "nan"
            if math.isnan(x)
            else repr(int(x))
            if x.is_integer() and abs(x) < 1e15
            else repr(x)
        )
    return repr(x)

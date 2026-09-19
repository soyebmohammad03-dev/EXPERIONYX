"""Sample tables for slice analysis. A table maps each sample ID (the dataset index within the
evaluated split) to its fields: `target`, `predicted`, `correct`, `confidence` from a run's stored
per-sample predictions, plus `feature:<name or index>` read from the dataset adapter. Predictions
come from digest-verified artifacts; nothing is duplicated into the registry."""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from experionyx.adapters.base import DatasetAdapter
from experionyx.adapters.capabilities import TaskType
from experionyx.adapters.records import RegisteredDataset
from experionyx.artifacts import ArtifactStore
from experionyx.errors import ExperionyxError, ValidationError
from experionyx.evaluation.loading import load_evaluation
from experionyx.hashing import content_hash
from experionyx.interactions.samples import PREDICTIONS_PATH, AlignmentError
from experionyx.interactions.samples import _rows as prediction_rows
from experionyx.provenance import Provenance
from experionyx.registry import Registry
from experionyx.slices.evaluate import SampleTable, build_table

FEATURE_PREFIX = "feature:"


class SliceDataError(ExperionyxError):
    """The data a slice needs is not available (reason attached); never approximated."""


@dataclass(frozen=True)
class Baseline:
    run_id: str
    task: TaskType
    classes: tuple[Any, ...] | None
    split: str | None
    dataset_id: str | None
    dataset_fingerprint: str | None
    evaluation_config_hash: str
    rows: dict[int, dict[str, Any]]  # sample index -> stored prediction row


def run_rows(registry: Registry, store: ArtifactStore, run_id: str) -> dict[int, dict[str, Any]]:
    """A run's per-sample rows keyed by sample index; duplicates and non-integer IDs are refused."""
    out: dict[int, dict[str, Any]] = {}
    for row in prediction_rows(registry, store, run_id):
        idx = row["index"]
        if not isinstance(idx, int) or isinstance(idx, bool):
            raise AlignmentError(f"run {run_id} has a non-integer sample index {idx!r}")
        if idx in out:
            raise AlignmentError(
                f"run {run_id} lists sample index {idx} more than once (duplicate sample IDs)"
            )
        out[idx] = row
    return out


def load_baseline(registry: Registry, store: ArtifactStore, run_id: str) -> Baseline:
    ev = load_evaluation(registry, store, run_id)
    try:
        rows = run_rows(registry, store, run_id)
    except AlignmentError as exc:
        raise SliceDataError(str(exc)) from exc
    return Baseline(
        run_id, ev.task, ev.classes, ev.context.split, ev.context.dataset_id,
        ev.context.dataset_fingerprint, content_hash(_jsonable(ev.config)), rows,
    )  # fmt: skip


def _jsonable(x: object) -> object:
    from experionyx.domain import to_jsonable

    return to_jsonable(x)


def feature_columns(
    dataset: DatasetAdapter, split: str | None, fields: Iterable[str], indices: set[int]
) -> dict[str, dict[int, float]]:
    """`feature:<name|index>` -> {sample index: value} for the requested samples only."""
    wanted = sorted({f for f in fields if f.startswith(FEATURE_PREFIX)})
    if not wanted:
        return {}
    meta = dataset.metadata()
    names = tuple(meta.input_schema.feature_names or ()) if meta.input_schema else ()
    col: dict[str, int] = {}
    for f in wanted:
        key = f[len(FEATURE_PREFIX) :]
        if key in names:
            col[f] = names.index(key)
        elif key.isdigit():
            col[f] = int(key)
        else:
            raise SliceDataError(f"{f!r} is not a feature of this dataset (names: {list(names)})")
    out: dict[str, dict[int, float]] = {f: {} for f in wanted}
    for batch in dataset.batches(1024, split):
        rows = list(batch.inputs)  # type: ignore[call-overload]
        for j, idx in enumerate(batch.indices):
            if idx in indices:
                for f, c in col.items():
                    try:
                        out[f][idx] = float(rows[j][c])
                    except (IndexError, TypeError, ValueError):
                        out[f][idx] = math.nan  # unusable: the evaluator will call it UNKNOWN
    return out


def sample_table(
    base: Baseline, dataset: DatasetAdapter | None, fields: Sequence[str]
) -> SampleTable:
    """The baseline's samples as a table. Feature fields need the dataset adapter; if a slice
    needs them and none is given the analysis raises SliceDataError instead of guessing."""
    feats = [f for f in fields if f.startswith(FEATURE_PREFIX)]
    cols: dict[str, dict[int, float]] = {}
    if feats:
        if dataset is None:
            raise SliceDataError(f"fields {feats} need the dataset, which is not available")
        cols = feature_columns(dataset, base.split, feats, set(base.rows))
    regression = base.task is TaskType.REGRESSION
    table_rows = []
    for idx, r in base.rows.items():
        fields_: dict[str, object] = {"target": r["true"], "predicted": r["predicted"]}
        if not regression and r.get("correct") is not None:
            fields_["correct"] = bool(r["correct"])
        if r.get("confidence") is not None:
            fields_["confidence"] = r["confidence"]
        for f, values in cols.items():
            if idx in values:
                fields_[f] = values[idx]
        table_rows.append((idx, fields_))
    try:
        return build_table(table_rows)
    except ValidationError as exc:
        raise SliceDataError(str(exc)) from exc


def dataset_for_run(
    registry: Registry, adapters: Any, inputs_root: Any, run_id: str
) -> DatasetAdapter | None:
    """The dataset adapter a run was executed on (fingerprint-verified), or None if the run has no
    registered dataset. `adapters` is an adapter registry bundle with a `datasets` registry."""
    provs = registry.find(Provenance, run_id=run_id)
    if not provs or provs[0].inputs is None or provs[0].inputs.dataset is None:
        return None
    rec = registry.get(RegisteredDataset, provs[0].inputs.dataset.record_id)
    cls = adapters.datasets.resolve(rec.adapter)
    src = rec.source
    if src is None:
        raise SliceDataError(f"dataset {rec.name} has no source to load from")
    path = src if src.startswith("builtin:") or str(src).startswith("/") else str(inputs_root / src)
    ds: DatasetAdapter = cls.load(path, version=rec.version, options=rec.options)
    if ds.fingerprint() != rec.fingerprint:
        raise SliceDataError(
            f"dataset {rec.name} changed since registration (fingerprint mismatch)"
        )
    return ds


__all__ = [
    "PREDICTIONS_PATH",
    "Baseline",
    "SliceDataError",
    "dataset_for_run",
    "feature_columns",
    "load_baseline",
    "run_rows",
    "sample_table",
]

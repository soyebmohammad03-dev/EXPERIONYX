"""What a drift analysis reads and how windows become sample sets.

A `Frame` is the baseline run's per-sample predictions plus the raw dataset columns the spec names
(the ordering feature and the declared features), keyed by dataset sample index. Window membership
is decided from each sample's ordering KEY, never from row position. A sample whose key is missing,
non-numeric or non-finite belongs to no window and is counted, not guessed."""

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

from experionyx.adapters.base import DatasetAdapter
from experionyx.drift.spec import INDEX, ShiftSpec, Skipped, TemporalWindow
from experionyx.errors import ExperionyxError
from experionyx.hashing import content_hash
from experionyx.slices.data import Baseline, feature_columns

MAX_LISTED = 20


class DriftDataError(ExperionyxError):
    """The data a drift analysis needs is missing or invalid (reason attached); never approximated."""


@dataclass(frozen=True)
class Frame:
    base: Baseline
    order: dict[int, float]  # sample index -> ordering key, for samples with a usable key
    order_excluded: dict[str, list[int]]  # reason -> sample indices (all of them, sorted)
    columns: dict[str, dict[int, Any]]  # declared feature name -> {sample index: raw cell}


def load_frame(base: Baseline, dataset: DatasetAdapter | None, spec: ShiftSpec) -> Frame:
    ordering = spec.ordering.field
    wanted = [f.column for f in spec.features] + ([] if ordering == INDEX else [ordering])
    raw: dict[str, dict[int, Any]] = {}
    if wanted:
        if dataset is None:
            raise DriftDataError(
                f"fields {sorted(wanted)} need the dataset, which is not available"
            )
        raw = feature_columns(dataset, base.split, wanted, set(base.rows), raw=True)
    order: dict[int, float] = {}
    excluded: dict[str, list[int]] = {"MISSING": [], "NON_FINITE": [], "INVALID": []}
    for i in sorted(base.rows):
        if ordering == INDEX:
            order[i] = float(i)
            continue
        x = raw[ordering].get(i)
        if x is None or (isinstance(x, float) and math.isnan(x)):
            excluded["MISSING"].append(i)
        elif isinstance(x, bool) or not isinstance(x, int | float):
            excluded["INVALID"].append(i)
        elif not math.isfinite(x):
            excluded["NON_FINITE"].append(i)
        else:
            order[i] = float(x)
    if spec.ordering.unique:
        dupes = sorted(k for k, n in Counter(order.values()).items() if n > 1)
        if dupes:
            raise DriftDataError(f"the ordering {ordering!r} is required to be unique but {len(dupes)} key(s) occur more than once, e.g. {dupes[:5]}")  # fmt: skip
    cols = {f.name: raw.get(f.column, {}) for f in spec.features}
    return Frame(base, order, {k: v for k, v in excluded.items() if v}, cols)


@dataclass(frozen=True)
class ResolvedWindow:
    window: TemporalWindow
    window_id: str
    sample_ids: tuple[int, ...]
    digest: str
    order_min: float | None
    order_max: float | None

    @property
    def n(self) -> int:
        return len(self.sample_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "window_id": self.window_id, "window": self.window.to_dict(), "n_samples": self.n,
            "sample_digest": self.digest, "order_min": self.order_min, "order_max": self.order_max,
            "sample_ids": list(self.sample_ids),
        }  # fmt: skip


def resolve_window(
    w: TemporalWindow, ordering_field: str, order: dict[int, float]
) -> ResolvedWindow:
    ids = tuple(i for i in sorted(order) if w.contains(order[i]))
    keys = [order[i] for i in ids]
    return ResolvedWindow(
        w, w.window_id(ordering_field), ids, content_hash({"ids": list(ids)}),
        min(keys) if keys else None, max(keys) if keys else None,
    )  # fmt: skip


@dataclass(frozen=True)
class Resolution:
    windows: dict[str, ResolvedWindow]  # window_id -> members
    pairs: tuple[tuple[str, str, str], ...]  # (pair key, reference window_id, comparison window_id)
    skipped: tuple[Skipped, ...]


def resolve(spec: ShiftSpec, frame: Frame) -> Resolution:
    """Members of every window of the plan. An explicitly declared window with no members is
    refused; a generated (rolling) one is skipped and listed with the reason."""
    oid = spec.ordering.field
    plan = spec.plan()
    cache: dict[str, ResolvedWindow] = {}

    def get(w: TemporalWindow) -> ResolvedWindow:
        wid = w.window_id(oid)
        if wid not in cache:
            cache[wid] = resolve_window(w, oid, frame.order)
        return cache[wid]

    skipped = list(plan.skipped)
    pairs: list[tuple[str, str, str]] = []
    for p in plan.pairs:
        r, c = get(p.reference), get(p.comparison)
        generated_ref = spec.reference is None or p.reference != spec.reference
        generated_cmp = spec.rolling is not None
        for rw, generated, label, reason in (
            (r, generated_ref, "reference", "EMPTY_REFERENCE"),
            (c, generated_cmp, "comparison", "EMPTY_WINDOW"),
        ):
            if rw.n == 0 and not generated:
                raise DriftDataError(f"the {label} window {rw.window.describe()} has no samples: no sample's {oid} key falls inside it")  # fmt: skip
            if rw.n == 0:
                skipped.append(Skipped(c.window, reason, f"the generated {label} window {rw.window.describe()} has no samples"))  # fmt: skip
                break
        else:
            if set(r.sample_ids) & set(c.sample_ids):  # cannot happen for disjoint intervals
                raise DriftDataError(f"windows {r.window.describe()} and {c.window.describe()} share samples")  # fmt: skip
            pairs.append((p.key, r.window_id, c.window_id))
    used = {w for _, a, b in pairs for w in (a, b)}
    windows = {k: v for k, v in sorted(cache.items()) if k in used}
    return Resolution(windows, tuple(pairs), tuple(skipped))

"""Per-sample interaction contrasts, computed only when samples are PROVEN to be aligned.

Alignment key: the sample's dataset index within the evaluated split. Alignment is verified, not
assumed: identical index sets in every run, no duplicate indices, identical ground-truth targets
per index (label faults change targets, so they are refused), and the design-level checks on
source dataset fingerprint and split. Row order in a file never matters (samples are keyed by
index), and nothing is ever matched by position. If alignment cannot be established the
per-sample analysis is REFUSED with the reason; it is never approximated."""

import json
import math
from collections import defaultdict
from collections.abc import Iterator, Mapping
from pathlib import Path

from experionyx.artifacts import ArtifactStore, LocalArtifactStore
from experionyx.domain import Artifact, Run
from experionyx.errors import ExperionyxError
from experionyx.failures.extraction import label_key
from experionyx.interactions.design import LoadedDesign
from experionyx.registry import Registry

PREDICTIONS_PATH = "evaluation/predictions.jsonl"


class AlignmentError(ExperionyxError):
    """Samples of the compared runs cannot be shown to correspond."""


def _rows(registry: Registry, store: ArtifactStore, run_id: str) -> Iterator[dict[str, object]]:
    run = registry.get(Run, run_id)
    found = [a for a in registry.find(Artifact, run_id=run_id) if a.path == PREDICTIONS_PATH]
    if not found:
        raise AlignmentError(
            f"run {run_id} kept no per-sample predictions (retention.predictions is off)"
        )
    store.verify(run, found[0])
    if not isinstance(store, LocalArtifactStore):
        raise ExperionyxError("reading artifacts requires a LocalArtifactStore")
    with Path(store.run_dir(run) / "artifacts" / found[0].path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _fields(
    row: Mapping[str, object], classes: tuple[object, ...] | None, regression: bool
) -> dict[str, float]:
    out: dict[str, float] = {}
    t, p = row["true"], row["predicted"]
    if regression:
        if (
            isinstance(t, int | float)
            and isinstance(p, int | float)
            and math.isfinite(t)
            and math.isfinite(p)
        ):
            out.update(prediction=float(p), abs_error=abs(p - t), signed_error=float(p - t))
    else:
        out["error"] = float(t != p)
    conf = row.get("confidence")
    if isinstance(conf, int | float) and math.isfinite(conf):
        out["confidence"] = float(conf)
    scores = row.get("scores")
    if (
        not regression
        and classes is not None
        and isinstance(scores, list)
        and t in classes
        and len(scores) == len(classes)
    ):
        out["true_class_probability"] = float(scores[classes.index(t)])
    return out


def per_sample(
    registry: Registry,
    store: ArtifactStore,
    d: LoadedDesign,
    ev_classes: tuple[object, ...] | None,
    top: int = 20,
) -> dict[str, object]:
    cfg = d.spec.config
    n_expected = next(iter(d.trials["CONTROL"])).n_samples
    if n_expected > cfg.max_samples:
        return {
            "status": "REFUSED",
            "reason": f"{n_expected} samples exceed max_samples={cfg.max_samples}; per-sample analysis is refused, not truncated",
        }
    regression = d.task == "REGRESSION"
    sums: dict[str, dict[int, dict[str, list[float]]]] = {
        c: defaultdict(lambda: defaultdict(list)) for c in d.trials
    }
    truth: dict[int, object] = {}
    index_sets: dict[str, set[int]] = {}
    try:
        for cell, ts in d.trials.items():
            for t in ts:
                seen: set[int] = set()
                for row in _rows(registry, store, t.run_id):
                    idx = row["index"]
                    if not isinstance(idx, int) or isinstance(idx, bool):
                        raise AlignmentError(
                            f"run {t.run_id} has a non-integer sample index {idx!r}"
                        )
                    if idx in seen:
                        raise AlignmentError(
                            f"run {t.run_id} lists sample index {idx} more than once (duplicate sample IDs)"
                        )
                    seen.add(idx)
                    if idx in truth and truth[idx] != row["true"]:
                        raise AlignmentError(
                            f"sample {idx} has a different ground-truth target in {t.run_id} ({row['true']!r}) than in an earlier run ({truth[idx]!r}); per-sample contrasts would compare different ground truth (label faults are refused)"
                        )
                    truth[idx] = row["true"]
                    for name, val in _fields(row, ev_classes, regression).items():
                        sums[cell][idx][name].append(val)
                first = index_sets.setdefault("ref", seen)
                if seen != first:
                    raise AlignmentError(
                        f"run {t.run_id} evaluated a different sample set than the other runs ({len(seen ^ first)} indices differ)"
                    )
    except (
        ExperionyxError
    ) as exc:  # AlignmentError, or an artifact that no longer matches its digest
        return {
            "status": "REFUSED",
            "reason": str(exc),
            "alignment": {"key": "dataset index within split", "verified": False},
        }
    fields = sorted({f for cell in sums.values() for per in cell.values() for f in per})
    records: list[dict[str, object]] = []
    valid: list[tuple[int, str, float]] = []
    summary: dict[str, dict[str, float | int]] = {}
    by_class: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for idx in sorted(truth):
        rec_fields: dict[str, object] = {}
        for f in fields:
            cell_means: dict[str, float] = {}
            for cell in d.trials:
                vals = sums[cell].get(idx, {}).get(f)
                if vals:
                    cell_means[cell] = sum(vals) / len(vals)
            if {"CONTROL", "A", "B", "AB"} <= set(cell_means):
                y0, ya, yb, yab = (cell_means[c] for c in ("CONTROL", "A", "B", "AB"))
                c = yab - ya - yb + y0
                rec_fields[f] = {
                    "y0": y0,
                    "ya": ya,
                    "yb": yb,
                    "yab": yab,
                    "effect_a": ya - y0,
                    "effect_b": yb - y0,
                    "combined_effect": yab - y0,
                    "interaction_contrast": c,
                    "status": "VALID",
                }
                s = summary.setdefault(
                    f, {"n": 0, "sum": 0.0, "positive": 0, "negative": 0, "zero": 0}
                )
                s["n"] += 1
                s["sum"] += c
                s["positive" if c > 0 else "negative" if c < 0 else "zero"] += 1
                valid.append((idx, f, c))
                if not regression:
                    by_class[label_key(truth[idx])][f].append(c)
            else:
                rec_fields[f] = {
                    "status": "UNDEFINED",
                    "reason": f"missing in cell(s) {sorted({'CONTROL', 'A', 'B', 'AB'} - set(cell_means))}",
                }
        records.append(
            {
                "sample_id": idx,
                "true": truth[idx],
                "class": None if regression else label_key(truth[idx]),
                "slice_memberships": "not recorded per sample (slice contrasts are reported at metric level)",
                "fields": rec_fields,
            }
        )
    for s in summary.values():
        s["mean_contrast"] = float(s.pop("sum")) / int(s["n"])
    lead = [
        {"sample_id": i, "field": f, "interaction_contrast": c}
        for i, f, c in sorted(valid, key=lambda x: (-abs(x[2]), x[0], x[1]))
    ]
    return {
        "status": "COMPUTED", "alignment": {"key": "dataset index within split", "verified": True, "checks": ["identical index sets in every run", "no duplicate indices", "identical ground-truth targets per index", "source dataset fingerprint and split (design validation)"]},
        "n_samples": len(records), "fields": fields, "summary": summary,
        "by_class": {c: {f: {"n": len(v), "mean_contrast": sum(v) / len(v)} for f, v in per.items()} for c, per in sorted(by_class.items())},
        "largest_absolute_contrasts": lead[:top], "records": records,
        "note": "per-sample contrasts use cell means over trials; samples are not independent draws, so no interval is computed at sample level",
    }  # fmt: skip

"""Per-measure interaction effects. Each record keeps OBSERVED trial values, DERIVED quantities
and INTERPRETED labels in separate sections; no single score is produced, and hypotheses counted
for multiplicity are reported, not corrected."""

from collections.abc import Mapping
from dataclasses import asdict

from experionyx.domain import to_jsonable
from experionyx.interactions import calc
from experionyx.interactions.design import LoadedDesign, Trial
from experionyx.interactions.taxonomy import (
    EffectStatus,
    InteractionClass,
    Level,
    Pairing,
)

NOTE = "an observed interaction contrast under the stated design; not a causal or mechanistic claim"


def _level(name: str) -> Level:
    return (
        Level.CLASS
        if name.startswith("class:")
        else Level.SLICE
        if name.startswith("slice:")
        else Level.METRIC
    )


def select_measures(d: LoadedDesign) -> list[str]:
    cfg = d.spec.config
    control = set().union(*(t.values.keys() for t in d.trials["CONTROL"]))
    primary = primary_metric(d)
    scalar = [m for m in sorted(control) if _level(m) is Level.METRIC]
    chosen = (
        [primary, *(m for m in cfg.metrics if m != primary)]
        if cfg.metrics
        else [primary, *(m for m in scalar if m != primary)]
    )
    if cfg.include_classes:
        chosen += [m for m in sorted(control) if _level(m) is Level.CLASS]
    if cfg.include_slices:
        chosen += [m for m in sorted(control) if _level(m) is Level.SLICE]
    return [m for m in chosen if m in control]


def primary_metric(d: LoadedDesign) -> str:
    return d.spec.config.primary_metric or ("accuracy" if d.task == "CLASSIFICATION" else "rmse")


def _cell_values(
    trials: tuple[Trial, ...], name: str
) -> tuple[dict[str, float], list[dict[str, str]]]:
    vals, dropped = {}, []
    for t in trials:
        if name in t.values:
            vals[t.key] = t.values[name]
        else:
            dropped.append(
                {
                    "run_id": t.run_id,
                    "key": t.key,
                    "reason": t.undefined.get(name, "measure absent"),
                }
            )
    return vals, dropped


def effect_for(d: LoadedDesign, name: str) -> dict[str, object]:
    cfg = d.spec.config
    cells: dict[str, dict[str, float]] = {}
    dropped: dict[str, list[dict[str, str]]] = {}
    for cell, ts in d.trials.items():
        cells[cell], dropped[cell] = _cell_values(ts, name)
    paired = d.pairing is Pairing.PAIRED
    if (
        paired
    ):  # only keys present in every treatment cell are paired; the rest are recorded, not used
        common = set.intersection(*(set(v) for c, v in cells.items() if c != "CONTROL"))
        for c in cells:
            if c != "CONTROL":
                for k in sorted(set(cells[c]) - common):
                    dropped[c].append(
                        {
                            "run_id": "",
                            "key": k,
                            "reason": "no matching key in every treatment cell (paired analysis)",
                        }
                    )
                cells[c] = {k: v for k, v in cells[c].items() if k in common}
    hib = d.direction.get(name)
    rec: dict[str, object] = {
        "level": _level(name).value, "name": name,
        "direction": "UNKNOWN" if hib is None else "HIGHER_IS_BETTER" if hib else "LOWER_IS_BETTER",
        "observed": {c: {"trials": dict(sorted(v.items())), "describe": calc.describe(list(v.values())), "dropped_trials": dropped[c]} for c, v in cells.items()},
        "pairing": d.pairing.value if d.pairing is not Pairing.UNKNOWN else "UNPAIRED (declared UNKNOWN)",
    }  # fmt: skip
    if any(not v for v in cells.values()):
        empty = sorted(c for c, v in cells.items() if not v)
        rec.update(
            status=EffectStatus.UNDEFINED.value,
            undefined_reason=f"no defined value in cell(s) {empty}",
            derived=None,
            bootstrap=None,
            interpreted={
                "class": InteractionClass.UNDEFINED.value,
                "rule": {"rule": "no defined value in a cell"},
                "note": NOTE,
            },
        )
        return rec
    agg = {c: calc.aggregate(list(v.values()), cfg.aggregation) for c, v in cells.items()}
    try:
        cons = calc.contrast(agg["CONTROL"], agg["A"], agg["B"], agg["AB"], cfg.normalization)
    except ValueError as exc:
        rec.update(
            status=EffectStatus.UNDEFINED.value,
            undefined_reason=str(exc),
            derived=None,
            bootstrap=None,
            interpreted={
                "class": InteractionClass.UNDEFINED.value,
                "rule": {"rule": str(exc)},
                "note": NOTE,
            },
        )
        return rec
    derived: dict[str, object] = {
        "aggregation": cfg.aggregation.value,
        **asdict(cons),
        "formulas": calc.CONTRAST_FORMULAS(),
    }
    derived["normalization"] = cons.normalization.value
    if "BA" in cells:
        s = calc.stats_of(agg, cfg.normalization)
        derived.update(
            yba=agg["BA"],
            interaction_contrast_ba=s["interaction_contrast_ba"],
            order_effect=s["order_effect"],
            order_note="I_AB - I_BA = YAB - YBA",
        )
    else:
        derived["order_note"] = (
            "order analysis not requested" if not cfg.order_analysis else "BA cell missing"
        )
    boot = calc.bootstrap(cells, cfg, d.pairing)
    point = calc.stats_of(agg, cfg.normalization)
    klass, rule = calc.classify(cfg, point, boot)
    oriented, relation = calc.deterioration_view(cons.interaction_contrast, hib)
    status = (
        EffectStatus.COMPUTED
        if boot.status is EffectStatus.COMPUTED
        else EffectStatus.INSUFFICIENT_DATA
        if boot.status is EffectStatus.INSUFFICIENT_DATA
        else EffectStatus.UNDEFINED
    )
    rec.update(
        status=status.value, undefined_reason=boot.reason if status is not EffectStatus.COMPUTED else None,
        derived=derived, bootstrap=to_jsonable(boot),
        interpreted={"class": klass.value, "rule": rule, "raw_contrast": cons.interaction_contrast, "deterioration_contrast": oriented, "deterioration_relation": relation.value, "direction_note": "raw_contrast is the mathematical contrast; deterioration_contrast > 0 means the combination is worse than additive for this metric's direction", "note": NOTE},
    )  # fmt: skip
    return rec


def analyze(d: LoadedDesign) -> dict[str, object]:
    measures = select_measures(d)
    effects = [effect_for(d, m) for m in measures]
    primary = next(e for e in effects if e["name"] == primary_metric(d))
    interp = primary["interpreted"]
    assert isinstance(interp, Mapping)  # noqa: S101
    by_level: dict[str, int] = {}
    for e in effects:
        by_level[str(e["level"])] = by_level.get(str(e["level"]), 0) + 1
    return {
        "primary_metric": primary_metric(d), "primary_class": interp["class"], "effects": effects,
        "multiplicity": {"hypotheses_tested": len(effects), "by_level": by_level, "adjustment": "none", "note": "exploratory: every measure is a separate contrast and no multiplicity adjustment is applied; treat non-primary results as hypotheses to confirm with new trials"},
    }  # fmt: skip


def classes_of(result: Mapping[str, object]) -> dict[str, str]:
    out = {}
    effects = result["effects"]
    assert isinstance(effects, list)  # noqa: S101
    for e in effects:
        i = e["interpreted"]
        assert isinstance(i, Mapping)  # noqa: S101
        out[str(e["name"])] = str(i["class"])
    return out

"""Structured comparison of two compatible profiles. Output is per-dimension raw changes
(b minus a) with metric directions stated but NO 'better', 'worse', winner or ranking."""

import json
from collections.abc import Mapping
from typing import Any

from experionyx.errors import ProfileRefusal
from experionyx.interactions.design import DesignIssue
from experionyx.reliability.taxonomy import Scope

NOTE = "raw differences (b minus a) between independently inspectable profiles; no overall ranking, winner or verdict is computed"


def _diff(a: Any, b: Any) -> float | None:
    return b - a if isinstance(a, int | float) and isinstance(b, int | float) else None


def check_comparable(a: Mapping[str, Any], b: Mapping[str, Any]) -> None:
    ca, cb = a["context"], b["context"]
    issues: list[DesignIssue] = []
    if a["scope"] != b["scope"]:
        issues.append(
            DesignIssue(
                "SCOPE_MISMATCH",
                str(a["scope"]),
                str(b["scope"]),
                "profiles of different scopes summarize different things",
            )
        )
    if ca["dataset_fingerprint"] != cb["dataset_fingerprint"]:
        issues.append(
            DesignIssue(
                "DATASET_MISMATCH",
                ca["dataset_fingerprint"],
                cb["dataset_fingerprint"],
                "profiles of different datasets are never compared",
            )
        )
    scope = Scope(str(a["scope"]))
    if (
        scope in (Scope.MODEL_DATASET_SPLIT, Scope.MODEL_DATASET_EVALUATION)
        and ca["split"] != cb["split"]
    ):
        issues.append(
            DesignIssue(
                "SPLIT_MISMATCH", str(ca["split"]), str(cb["split"]), "different evaluated splits"
            )
        )
    if (
        scope is Scope.MODEL_DATASET_EVALUATION
        and ca["evaluation_config_hash"] != cb["evaluation_config_hash"]
    ):
        issues.append(
            DesignIssue(
                "EVALUATION_CONFIG_MISMATCH",
                ca["evaluation_config_hash"],
                cb["evaluation_config_hash"],
                "different evaluation configurations",
            )
        )
    if issues:
        raise ProfileRefusal(tuple(issues))


def _obs(doc: Mapping[str, Any], dim: str) -> list[Any]:
    return list(doc["dimensions"][dim]["observations"] or [])


def _side(a: Mapping[str, Any], b: Mapping[str, Any], dim: str) -> dict[str, Any]:
    return {"status_a": a["dimension_status"][dim], "status_b": b["dimension_status"][dim]}


def _by(items: list[Any], key: Any) -> dict[str, Any]:
    return {key(i): i for i in items}


def compare_profiles(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    check_comparable(a, b)
    out: dict[str, Any] = {
        "note": NOTE,
        "a": {
            "spec_id": a["spec_id"],
            "model_fingerprint": a["context"]["model_fingerprint"],
            "provenance_fingerprint": a["provenance_fingerprint"],
        },
        "b": {
            "spec_id": b["spec_id"],
            "model_fingerprint": b["context"]["model_fingerprint"],
            "provenance_fingerprint": b["provenance_fingerprint"],
        },
        "same_model": a["context"]["model_fingerprint"] == b["context"]["model_fingerprint"],
        "dimension_status": {
            d: {"a": a["dimension_status"][d], "b": b["dimension_status"][d]}
            for d in a["dimension_status"]
        },
    }
    # baseline: matched metrics with raw differences
    ma = {
        m["metric_id"]: m
        for m in (_obs(a, "BASELINE_PERFORMANCE") or [{"metrics": []}])[0]["metrics"]
    }
    mb = {
        m["metric_id"]: m
        for m in (_obs(b, "BASELINE_PERFORMANCE") or [{"metrics": []}])[0]["metrics"]
    }
    out["baseline"] = {
        "metrics": [
            {
                "metric_id": k,
                "a": ma[k]["value"],
                "b": mb[k]["value"],
                "difference": _diff(ma[k]["value"], mb[k]["value"]),
                "higher_is_better": ma[k]["higher_is_better"],
                "interval_a": ma[k]["interval"],
                "interval_b": mb[k]["interval"],
            }
            for k in sorted(set(ma) & set(mb))
        ],
        "only_in_a": sorted(set(ma) - set(mb)),
        "only_in_b": sorted(set(mb) - set(ma)),
    }
    for dim, key in (("CALIBRATION", "calibration"), ("LATENCY", "latency")):
        oa, ob = _obs(a, dim), _obs(b, dim)
        if oa and ob:
            out[key] = {
                k: {"a": oa[0][k], "b": ob[0][k], "difference": _diff(oa[0][k], ob[0][k])}
                for k in oa[0]
                if k not in ("source", "note", "confidence_source", "binning", "interpretation")
                and k in ob[0]
            }
        else:
            out[key] = _side(a, b, dim)
    # fault responses matched by fault definition (type, version, parameters, scope, components)
    fa, fb = (
        _by(_obs(a, "FAULT_SENSITIVITY"), _fault_key),
        _by(_obs(b, "FAULT_SENSITIVITY"), _fault_key),
    )
    faults = []
    for k in sorted(set(fa) & set(fb)):
        pa = {(p["parameter"], p["value"]): p for p in fa[k]["points"]}
        pb = {(p["parameter"], p["value"]): p for p in fb[k]["points"]}
        faults.append(
            {
                "fault": json.loads(k),
                "primary_metric": fa[k]["primary_metric"],
                "same_metric": fa[k]["primary_metric"] == fb[k]["primary_metric"],
                "baseline_value": {"a": fa[k]["baseline_value"], "b": fb[k]["baseline_value"]},
                "points": [
                    {
                        "parameter": pr,
                        "value": v,
                        "completed": {"a": pa[(pr, v)]["completed"], "b": pb[(pr, v)]["completed"]},
                        "deterioration_mean": {
                            "a": pa[(pr, v)]["deterioration"]["mean"],
                            "b": pb[(pr, v)]["deterioration"]["mean"],
                            "difference": _diff(
                                pa[(pr, v)]["deterioration"]["mean"],
                                pb[(pr, v)]["deterioration"]["mean"],
                            ),
                        },
                        "deterioration_interval": {
                            "a": [
                                pa[(pr, v)]["deterioration"]["ci_lower"],
                                pa[(pr, v)]["deterioration"]["ci_upper"],
                            ],
                            "b": [
                                pb[(pr, v)]["deterioration"]["ci_lower"],
                                pb[(pr, v)]["deterioration"]["ci_upper"],
                            ],
                        },
                        "effect_classification": {
                            "a": pa[(pr, v)]["effect_classification"],
                            "b": pb[(pr, v)]["effect_classification"],
                        },
                    }
                    for (pr, v) in sorted(set(pa) & set(pb), key=str)
                ],
                "points_only_in_a": sorted(str(x) for x in set(pa) - set(pb)),
                "points_only_in_b": sorted(str(x) for x in set(pb) - set(pa)),
            }
        )
    out["fault_responses"] = {
        **_side(a, b, "FAULT_SENSITIVITY"),
        "matched": faults,
        "only_in_a": sorted(set(fa) - set(fb)),
        "only_in_b": sorted(set(fb) - set(fa)),
    }
    # failure modes matched by structure (category, classes, slices, fault types)
    ra = _by(
        [
            {**p, **s}
            for p, s in zip(
                _obs(a, "FAILURE_PREVALENCE"), _obs(a, "FAILURE_SEVERITY"), strict=False
            )
        ],
        _mode_id,
    )
    rb = _by(
        [
            {**p, **s}
            for p, s in zip(
                _obs(b, "FAILURE_PREVALENCE"), _obs(b, "FAILURE_SEVERITY"), strict=False
            )
        ],
        _mode_id,
    )
    out["failure_modes"] = {
        **_side(a, b, "FAILURE_PREVALENCE"),
        "matched": [
            {
                "structure": json.loads(k)[0],
                "ids": {"a": ra[k]["source"]["id"], "b": rb[k]["source"]["id"]},
                "lifecycle_state": {"a": ra[k]["lifecycle_state"], "b": rb[k]["lifecycle_state"]},
                "prevalence_fraction": {
                    "a": ra[k]["prevalence"]["fraction_of_runs"],
                    "b": rb[k]["prevalence"]["fraction_of_runs"],
                    "difference": _diff(
                        ra[k]["prevalence"]["fraction_of_runs"],
                        rb[k]["prevalence"]["fraction_of_runs"],
                    ),
                },
                "severity_magnitude_mean": {
                    "a": ra[k]["severity"]["magnitude_mean"],
                    "b": rb[k]["severity"]["magnitude_mean"],
                    "difference": _diff(
                        ra[k]["severity"]["magnitude_mean"], rb[k]["severity"]["magnitude_mean"]
                    ),
                },
            }
            for k in sorted(set(ra) & set(rb))
        ],
        "only_in_a": [ra[k]["source"]["id"] for k in sorted(set(ra) - set(rb))],
        "only_in_b": [rb[k]["source"]["id"] for k in sorted(set(rb) - set(ra))],
        "note": "modes of different profiles are matched only by structure (category, classes, slices, fault types), never by name",
    }
    ia, ib = (
        _by(_obs(a, "INTERACTION_SENSITIVITY"), _inter_key),
        _by(_obs(b, "INTERACTION_SENSITIVITY"), _inter_key),
    )
    out["interactions"] = {
        **_side(a, b, "INTERACTION_SENSITIVITY"),
        "matched": [
            {
                "faults": [ia[k]["components"]],
                "metric": ia[k]["metric"],
                "contrast": {
                    "a": ia[k]["interaction_contrast"],
                    "b": ib[k]["interaction_contrast"],
                    "difference": _diff(
                        ia[k]["interaction_contrast"], ib[k]["interaction_contrast"]
                    ),
                },
                "interval": {"a": ia[k]["interval"], "b": ib[k]["interval"]},
                "label": {"a": ia[k]["label"], "b": ib[k]["label"]},
                "lifecycle_state": {"a": ia[k]["lifecycle_state"], "b": ib[k]["lifecycle_state"]},
                "order_dependent": {"a": ia[k]["order_dependent"], "b": ib[k]["order_dependent"]},
            }
            for k in sorted(set(ia) & set(ib))
        ],
        "only_in_a": [ia[k]["source"]["id"] for k in sorted(set(ia) - set(ib))],
        "only_in_b": [ib[k]["source"]["id"] for k in sorted(set(ib) - set(ia))],
    }
    sa, sb = _obs(a, "SLICE_SENSITIVITY"), _obs(b, "SLICE_SENSITIVITY")
    slices: dict[str, Any] = _side(a, b, "SLICE_SENSITIVITY")
    if sa and sb:
        ca = {c["class"]: c for c in sa[0]["classes"]} if isinstance(sa[0]["classes"], list) else {}
        cb = {c["class"]: c for c in sb[0]["classes"]} if isinstance(sb[0]["classes"], list) else {}
        sla = {s["name"]: s for s in sa[0]["slices"]} if isinstance(sa[0]["slices"], list) else {}
        slb = {s["name"]: s for s in sb[0]["slices"]} if isinstance(sb[0]["slices"], list) else {}
        slices.update(
            classes=[
                {
                    "class": k,
                    "recall": {
                        "a": ca[k]["recall"],
                        "b": cb[k]["recall"],
                        "difference": _diff(ca[k]["recall"], cb[k]["recall"]),
                    },
                    "support": {"a": ca[k]["support"], "b": cb[k]["support"]},
                }
                for k in sorted(set(ca) & set(cb))
            ],
            slices=[
                {
                    "name": k,
                    "n_samples": {"a": sla[k]["n_samples"], "b": slb[k]["n_samples"]},
                    "deltas_vs_overall": {
                        "a": sla[k]["deltas_vs_overall"],
                        "b": slb[k]["deltas_vs_overall"],
                    },
                }
                for k in sorted(set(sla) & set(slb))
            ],
        )
    out["slices"] = slices
    ra_, rb_ = _obs(a, "REPRODUCIBILITY"), _obs(b, "REPRODUCIBILITY")
    out["reproducibility"] = {
        **_side(a, b, "REPRODUCIBILITY"),
        "a": ra_[0] if ra_ else None,
        "b": rb_[0] if rb_ else None,
    }
    ua, ub = _obs(a, "UNCERTAINTY"), _obs(b, "UNCERTAINTY")
    out["uncertainty"] = {
        **_side(a, b, "UNCERTAINTY"),
        "intervals_a": len(ua[0]["intervals"]) if ua else 0,
        "intervals_b": len(ub[0]["intervals"]) if ub else 0,
        "caveats_a": ua[0]["caveats"] if ua else [],
        "caveats_b": ub[0]["caveats"] if ub else [],
    }
    return out


def _mode_key(m: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "category": m["category"],
        "classes": sorted(m["classes"]),
        "slices": sorted(m["slices"]),
        "fault_types": sorted(m["fault_types"]),
    }


def _fault_key(o: Mapping[str, Any]) -> str:
    return json.dumps(
        {k: o["fault"][k] for k in ("type", "version", "parameters", "scope", "components")},
        sort_keys=True,
    )


def _mode_id(m: Mapping[str, Any]) -> str:
    return json.dumps([_mode_key(m)], sort_keys=True)


def _inter_key(i: Mapping[str, Any]) -> str:
    return json.dumps([i["fault_a"], i["fault_b"], i["metric"]], sort_keys=True)

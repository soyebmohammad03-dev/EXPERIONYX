"""Comparison of two benchmark RESULTS. They must have been produced under the identical protocol
(same definition, dataset, faults, seeds, evaluation, analysis settings and resolved fault
versions); only the model may differ. Output is raw per-section differences (b minus a) and the
compatibility constraints; there is no winner, ranking or verdict."""

import json
from collections.abc import Mapping
from typing import Any

from experionyx.errors import BenchmarkRefusal
from experionyx.interactions.design import DesignIssue
from experionyx.reliability.compare import _diff, _mode_key

NOTE = "raw differences (b minus a) between two results of the same protocol; no overall ranking, winner or verdict is computed"


def _protocol_diff(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    sa, sb = a["spec"]["spec"], b["spec"]["spec"]
    out = [
        f"{k}: {json.dumps(sa[k], sort_keys=True)[:80]} vs {json.dumps(sb[k], sort_keys=True)[:80]}"
        for k in sorted(set(sa) | set(sb))
        if k != "model" and sa.get(k) != sb.get(k)
    ]
    if a["spec"]["engine_version"] != b["spec"]["engine_version"]:
        out.append(
            f"engine_version: {a['spec']['engine_version']} vs {b['spec']['engine_version']}"
        )
    if a["spec"]["resolved_faults"] != b["spec"]["resolved_faults"]:
        out.append(
            f"resolved fault versions: {a['spec']['resolved_faults']} vs {b['spec']['resolved_faults']}"
        )
    return out


def check_comparable(a: Mapping[str, Any], b: Mapping[str, Any]) -> None:
    if a["spec"]["protocol_hash"] != b["spec"]["protocol_hash"]:
        diffs = _protocol_diff(a, b) or [
            "the expanded protocols differ (unit list or unsupported grids)"
        ]
        raise BenchmarkRefusal(
            tuple(
                DesignIssue(
                    "PROTOCOL_DIFFERS",
                    a["spec"]["protocol_hash"],
                    b["spec"]["protocol_hash"],
                    "results are comparable only under an identical protocol: " + d,
                )
                for d in diffs
            )
        )


def compare_results(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    check_comparable(a, b)
    ca, cb = a["coverage"], b["coverage"]
    ra, rb = a["results"], b["results"]
    out: dict[str, Any] = {
        "note": NOTE,
        "protocol_hash": a["spec"]["protocol_hash"],
        "same_model": a["spec"]["spec"]["model"] == b["spec"]["spec"]["model"],
        "benchmark": {"name": a["spec"]["spec"]["name"], "version": a["spec"]["spec"]["version"]},
    }
    out["coverage"] = {
        "complete": {"a": ca["complete"], "b": cb["complete"]},
        "trials": {
            k: {
                "a": ca["trials"][k],
                "b": cb["trials"][k],
                "difference": _diff(ca["trials"][k], cb["trials"][k]),
            }
            for k in ca["trials"]
        },
        "parameter_points": {
            k: {
                "a": ca["parameter_points"][k],
                "b": cb["parameter_points"][k],
                "difference": _diff(ca["parameter_points"][k], cb["parameter_points"][k]),
            }
            for k in ca["parameter_points"]
        },
        "interactions": {
            k: {"a": ca["interactions"][k], "b": cb["interactions"][k]}
            for k in ("requested", "analyzed", "with_interval")
        },
        "failure_modes_discovered": {
            "a": ca["failure_modes"]["discovered"],
            "b": cb["failure_modes"]["discovered"],
        },
        "incomplete_reasons": {"a": ca["incomplete_reasons"], "b": cb["incomplete_reasons"]},
    }
    ma = {m["metric_id"]: m for m in ra["baseline"].get("metrics", [])}
    mb = {m["metric_id"]: m for m in rb["baseline"].get("metrics", [])}
    out["baseline"] = {
        "metrics": [
            {
                "metric_id": k,
                "a": ma[k]["value"],
                "b": mb[k]["value"],
                "difference": _diff(ma[k]["value"], mb[k]["value"]),
                "higher_is_better": ma[k]["higher_is_better"],
            }
            for k in sorted(set(ma) & set(mb))
        ],
        "only_in_a": sorted(set(ma) - set(mb)),
        "only_in_b": sorted(set(mb) - set(ma)),
    }
    ga = {g["grid"]: g for g in ra["fault_responses"]["grids"]}
    gb = {g["grid"]: g for g in rb["fault_responses"]["grids"]}
    fam = []
    for k in sorted(set(ga) & set(gb)):
        pa = {(p["parameter"], p["value"]): p for p in ga[k]["points"]}
        pb = {(p["parameter"], p["value"]): p for p in gb[k]["points"]}
        fam.append(
            {
                "grid": k,
                "fault": ga[k]["fault"],
                "status": {"a": ga[k]["status"], "b": gb[k]["status"]},
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
        "families": fam,
        "only_in_a": sorted(set(ga) - set(gb)),
        "only_in_b": sorted(set(gb) - set(ga)),
    }
    ia = {i["pair"]: i for i in ra["interactions"]["pairs"]}
    ib = {i["pair"]: i for i in rb["interactions"]["pairs"]}
    out["interactions"] = {
        "pairs": [
            {
                "pair": k,
                "label": {"a": ia[k].get("label"), "b": ib[k].get("label")},
                "lifecycle_state": {
                    "a": ia[k].get("lifecycle_state"),
                    "b": ib[k].get("lifecycle_state"),
                },
                "contrast": {
                    "a": ia[k].get("interaction_contrast"),
                    "b": ib[k].get("interaction_contrast"),
                    "difference": _diff(
                        ia[k].get("interaction_contrast"), ib[k].get("interaction_contrast")
                    ),
                },
                "interval": {"a": ia[k].get("interval"), "b": ib[k].get("interval")},
                "order_effect": {"a": ia[k].get("order_effect"), "b": ib[k].get("order_effect")},
            }
            for k in sorted(set(ia) & set(ib))
        ],
        "only_in_a": sorted(set(ia) - set(ib)),
        "only_in_b": sorted(set(ib) - set(ia)),
    }
    fa = {json.dumps(_mode_key(m), sort_keys=True): m for m in ra["failure_modes"]["modes"]}
    fb = {json.dumps(_mode_key(m), sort_keys=True): m for m in rb["failure_modes"]["modes"]}
    out["failure_modes"] = {
        "status": {"a": ra["failure_modes"]["status"], "b": rb["failure_modes"]["status"]},
        "matched": [
            {
                "structure": json.loads(k),
                "lifecycle_state": {"a": fa[k]["lifecycle_state"], "b": fb[k]["lifecycle_state"]},
                "prevalence_fraction": {
                    "a": fa[k]["prevalence"]["fraction_of_runs"],
                    "b": fb[k]["prevalence"]["fraction_of_runs"],
                    "difference": _diff(
                        fa[k]["prevalence"]["fraction_of_runs"],
                        fb[k]["prevalence"]["fraction_of_runs"],
                    ),
                },
            }
            for k in sorted(set(fa) & set(fb))
        ],
        "only_in_a": len(set(fa) - set(fb)),
        "only_in_b": len(set(fb) - set(fa)),
        "note": "modes of different results are matched only by structure (category, classes, slices, fault types), never by ID",
    }
    out["uncertainty"] = {
        "intervals_available": {
            "a": ra["uncertainty"]["intervals_available"],
            "b": rb["uncertainty"]["intervals_available"],
        },
        "status": {"a": ra["uncertainty"]["status"], "b": rb["uncertainty"]["status"]},
    }
    out["reproducibility"] = {"a": ra["reproducibility"], "b": rb["reproducibility"]}
    out["section_status"] = {
        k: {"a": a["summary"]["section_status"][k], "b": b["summary"]["section_status"][k]}
        for k in a["summary"]["section_status"]
    }
    return out

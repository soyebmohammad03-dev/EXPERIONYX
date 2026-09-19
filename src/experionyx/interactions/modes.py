"""How Phase 6 failure modes are observed across the design's cells.

Modes are only comparable when ONE discovery covered every treatment cell (otherwise modes found
separately would look 'newly observed' purely because they are different cluster IDs). The
analysis therefore needs `discovery_run_id`, checks that the discovery's sources cover every
treatment run's fault experiment, and otherwise reports NOT_AVAILABLE with the reason. States
describe what was observed, never what caused it, and a mode below CANDIDATE is reported as
INSUFFICIENT_EVIDENCE."""

from collections.abc import Mapping

from experionyx.artifacts import ArtifactStore
from experionyx.domain import Run, RunStatus
from experionyx.errors import ExperionyxError
from experionyx.failures.entities import FailureCluster, FailureMode, FailureSignal
from experionyx.failures.taxonomy import FailureStatus
from experionyx.faults.entities import FaultTrial
from experionyx.faults.report import read_artifact
from experionyx.interactions.design import LoadedDesign
from experionyx.interactions.taxonomy import ModeState
from experionyx.registry import Registry

WEAK = {FailureStatus.DISCOVERED}


def failure_modes(registry: Registry, store: ArtifactStore, d: LoadedDesign) -> dict[str, object]:
    cfg = d.spec.config
    rid = d.spec.discovery_run_id
    if rid is None:
        return {
            "status": "NOT_AVAILABLE",
            "reason": "no discovery_run_id in the spec: failure-mode interaction needs one Phase 6 discovery covering every treatment cell",
        }
    try:
        run = registry.get(Run, rid)
        if run.status is not RunStatus.COMPLETED:
            return {
                "status": "NOT_AVAILABLE",
                "reason": f"discovery run {rid} is {run.status.value}",
            }
        analysis = read_artifact(registry, store, rid, "failure-analysis.json")
        candidates = read_artifact(registry, store, rid, "failure-candidates.json")
    except ExperionyxError as exc:
        return {"status": "NOT_AVAILABLE", "reason": f"cannot read discovery run {rid}: {exc}"}
    if not isinstance(analysis, dict) or not isinstance(candidates, dict):
        return {"status": "NOT_AVAILABLE", "reason": "discovery artifacts are malformed"}
    covered = {
        s.get("fault_experiment_id") for s in analysis.get("sources", []) if isinstance(s, dict)
    }
    uncovered = []
    for cell, ts in d.trials.items():
        if cell == "CONTROL":
            continue
        for t in ts:
            fx = {
                x.fault_experiment_id for x in registry.find(FaultTrial, treatment_run_id=t.run_id)
            }
            if not fx or not fx <= covered:
                uncovered.append({"cell": cell, "run_id": t.run_id})
    if uncovered:
        return {
            "status": "NOT_AVAILABLE",
            "reason": f"the discovery did not cover {len(uncovered)} treatment run(s); modes from separate discoveries are not comparable",
            "uncovered_runs": uncovered[:20],
        }
    cell_runs = {c: {t.run_id for t in ts} for c, ts in d.trials.items() if c != "CONTROL"}
    rows: list[dict[str, object]] = []
    for entry in candidates.get("modes", []):
        mode = registry.get(FailureMode, str(entry["id"]))
        sig_runs: dict[str, set[str]] = {c: set() for c in cell_runs}
        sig_ids: dict[str, list[str]] = {c: [] for c in cell_runs}
        for sid in registry.get(FailureCluster, mode.cluster_id).signal_ids:
            sig = registry.get(FailureSignal, sid)
            for c, runs in cell_runs.items():
                if sig.run_id in runs:
                    sig_runs[c].add(sig.run_id)
                    sig_ids[c].append(sid)
        prev = {c: len(sig_runs[c]) / len(cell_runs[c]) for c in cell_runs}
        if not any(prev.values()):
            continue
        states = _states(prev, mode.status, cfg.prevalence_delta_min)
        rows.append({
            "mode_id": mode.id, "mode_status": mode.status.value, "category": mode.category.value,
            "prevalence": {c: {"runs_with_signal": len(sig_runs[c]), "runs_in_cell": len(cell_runs[c]), "fraction": prev[c]} for c in cell_runs},
            "states": [s.value for s in states], "evidence": {c: sorted(v)[:50] for c, v in sig_ids.items()},
            "low_trial_cells": sorted(c for c, r in cell_runs.items() if len(r) < cfg.min_trials),
            "note": "observed prevalence under each treatment; not a statement about cause",
        })  # fmt: skip
    counts: dict[str, int] = {}
    for r in rows:
        for s in r["states"]:  # type: ignore[attr-defined]
            counts[s] = counts.get(s, 0) + 1
    return {
        "status": "COMPUTED",
        "discovery_run_id": rid,
        "discovery_config_hash": analysis.get("config_hash"),
        "prevalence_delta_min": cfg.prevalence_delta_min,
        "modes": sorted(rows, key=lambda r: str(r["mode_id"])),
        "state_counts": dict(sorted(counts.items())),
        "definition": "prevalence = treatment runs of the cell that contain a signal of the mode / runs in the cell; compound is compared with the larger single-fault prevalence",
    }


def _states(prev: Mapping[str, float], status: FailureStatus, delta: float) -> list[ModeState]:
    pa, pb, pab = prev.get("A", 0.0), prev.get("B", 0.0), prev.get("AB", 0.0)
    out: list[ModeState] = []
    if status in WEAK:
        out.append(ModeState.INSUFFICIENT_EVIDENCE)
    base = max(pa, pb)
    if pab > 0:
        out.append(ModeState.OBSERVED_UNDER_COMPOUND_TREATMENT)
        if base == 0:
            out.append(ModeState.NEWLY_OBSERVED)
        elif pab - base >= delta:
            out.append(ModeState.INCREASED_PREVALENCE)
        elif base - pab >= delta:
            out.append(ModeState.REDUCED_PREVALENCE)
        else:
            out.append(ModeState.PERSISTING)
    elif base > 0:
        out.append(ModeState.REDUCED_PREVALENCE)
    return out

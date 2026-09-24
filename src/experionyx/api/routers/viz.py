"""Chart-ready reshaping of already-persisted analysis fields. No new statistics are computed
here -- every number returned already exists on a persisted entity's `summary`/`result`/`headline`/
`metrics` mapping. Missing/unavailable evidence is always returned as an explicit "unavailable"
marker, never as 0 or by omission. No aggregate score or "best model" field is ever introduced."""

from typing import Any

from fastapi import APIRouter

import experionyx.validation as v
from experionyx.api.deps import RegistryDep, StoreDep
from experionyx.calibration.entities import CalibrationAnalysis, CalibrationResult
from experionyx.domain import to_jsonable
from experionyx.drift.entities import DriftAnalysis
from experionyx.errors import ExperionyxError
from experionyx.failures.entities import FailureMode
from experionyx.faults.entities import FaultExperiment
from experionyx.faults.report import load_analysis, summary_rows
from experionyx.reliability.entities import ReliabilityProfile
from experionyx.resources.entities import ResourceAnalysis, ResourceTrial
from experionyx.stats.entities import StatisticalAnalysis

router = APIRouter(prefix="/api/viz", tags=["viz"])


def _chart(
    *,
    metric: str,
    unit: str | None,
    population: str,
    comparison: Any,
    points: Any,
    uncertainty: Any = "unavailable",
    provenance_ref: str | None = None,
) -> dict[str, object]:
    payload = {
        "metric": metric,
        "unit": unit if unit is not None else "unavailable",
        "population": population,
        "comparison": comparison,
        "points": points,
        "uncertainty": uncertainty,
        "provenance_ref": provenance_ref if provenance_ref is not None else "unavailable",
    }
    jsonable = to_jsonable(payload)
    assert isinstance(jsonable, dict)  # noqa: S101  # to_jsonable(dict) is always a dict
    return jsonable


@router.get("/reliability/{profile_id}/dimensions")
def reliability_dimensions(profile_id: str, registry: RegistryDep) -> dict[str, object]:
    """Per-dimension status counts -- deliberately NOT a single reliability score."""
    v.ref("profile_id", profile_id, ReliabilityProfile.PREFIX)
    profile = registry.get(ReliabilityProfile, profile_id)
    points = [
        {"dimension": dim, "status": status}
        for dim, status in sorted(profile.dimension_status.items())
    ]
    return _chart(
        metric="dimension_status",
        unit=None,
        population=f"profile:{profile_id}",
        comparison="per-dimension (no aggregate score)",
        points=points,
        provenance_ref=profile.run_id,
    )


@router.get("/failures/frequency")
def failure_mode_frequency(registry: RegistryDep, investigation: str) -> dict[str, object]:
    v.ref("investigation", investigation, "inv")
    modes = registry.find(FailureMode, investigation_id=investigation)
    counts: dict[str, int] = {}
    for m in modes:
        counts[m.category.value] = counts.get(m.category.value, 0) + 1
    points = [{"category": k, "count": v_} for k, v_ in sorted(counts.items())]
    return _chart(
        metric="failure_mode_count",
        unit="count",
        population=f"investigation:{investigation}",
        comparison="by category",
        points=points,
    )


@router.get("/drift/{analysis_id}/trajectory")
def drift_trajectory(analysis_id: str, registry: RegistryDep) -> dict[str, object]:
    v.ref("analysis_id", analysis_id, DriftAnalysis.PREFIX)
    analysis = registry.get(DriftAnalysis, analysis_id)
    return _chart(
        metric="drift_summary",
        unit=None,
        population=f"run:{analysis.run_id}",
        comparison=f"baseline:{analysis.baseline_run_id}",
        points=dict(analysis.summary) or "unavailable",
        provenance_ref=analysis.provenance_fingerprint,
    )


@router.get("/calibration/{analysis_id}/curve")
def calibration_curve(analysis_id: str, registry: RegistryDep) -> dict[str, object]:
    v.ref("analysis_id", analysis_id, CalibrationAnalysis.PREFIX)
    analysis = registry.get(CalibrationAnalysis, analysis_id)
    results = sorted(
        registry.find(CalibrationResult, analysis_id=analysis_id), key=lambda r: r.context_key
    )
    points = [
        {
            "context": r.context_key,
            "n_samples": r.n_samples,
            "status": r.status,
            "headline": dict(r.headline) if r.status == "COMPUTED" else "unavailable",
        }
        for r in results
    ]
    return _chart(
        metric="calibration_headline",
        unit=None,
        population=f"run:{analysis.run_id}",
        comparison="by context (baseline / calibrated / slice / window / stress)",
        points=points,
        provenance_ref=analysis.provenance_fingerprint,
    )


@router.get("/resources/{analysis_id}/measurements")
def resource_measurements(analysis_id: str, registry: RegistryDep) -> dict[str, object]:
    v.ref("analysis_id", analysis_id, ResourceAnalysis.PREFIX)
    analysis = registry.get(ResourceAnalysis, analysis_id)
    trials = sorted(
        registry.find(ResourceTrial, analysis_id=analysis_id),
        key=lambda t: (t.phase, t.trial_index),
    )
    points = [
        {
            "phase": t.phase,
            "trial_index": t.trial_index,
            "wall_seconds": t.wall_seconds,
            "cpu_seconds": t.cpu_seconds if t.cpu_seconds is not None else "unavailable",
            "status": t.status,
        }
        for t in trials
    ]
    return _chart(
        metric="wall_seconds",
        unit="seconds",
        population=f"run:{analysis.run_id}",
        comparison="by trial (WARMUP vs MEASURED)",
        points=points,
        provenance_ref=analysis.provenance_fingerprint,
    )


@router.get("/faults/{fault_experiment_id}/degradation")
def fault_degradation(
    fault_experiment_id: str, registry: RegistryDep, store: StoreDep
) -> dict[str, object]:
    """One point per sweep point: parameter value vs. deterioration, with its CI band and
    classification -- read straight from the persisted `fault/analysis.json` artifact via the
    existing `faults.report` helpers. No new statistics are computed here."""
    v.ref("fault_experiment_id", fault_experiment_id, FaultExperiment.PREFIX)
    exp = registry.get(FaultExperiment, fault_experiment_id)
    try:
        result = load_analysis(registry, store, fault_experiment_id)
    except ExperionyxError:
        return _chart(
            metric="deterioration",
            unit=None,
            population=f"fault_experiment:{fault_experiment_id}",
            comparison="faulted vs. baseline, by sweep point",
            points="unavailable",
        )
    rows = summary_rows(result)
    points = []
    for row in rows:
        if row["deterioration_mean"] is None:
            continue
        ci = row["deterioration_ci"]
        lo, hi = (None, None) if not isinstance(ci, list) else (ci[0], ci[1])
        points.append(
            {
                "x": row["value"] if row["value"] is not None else row["point"],
                "y": row["deterioration_mean"],
                "lo": lo,
                "hi": hi,
                "classification": row["classification"],
                "parameter": row["parameter"],
            }
        )
    return _chart(
        metric="deterioration",
        unit=None,
        population=f"fault_experiment:{fault_experiment_id}",
        comparison=f"faulted vs. baseline={result.baseline_value}, by sweep point",
        points=points if points else "unavailable",
        provenance_ref=exp.investigation_id,
    )


@router.get("/stats/{analysis_id}/effect")
def stats_effect(analysis_id: str, registry: RegistryDep) -> dict[str, object]:
    v.ref("analysis_id", analysis_id, StatisticalAnalysis.PREFIX)
    analysis = registry.get(StatisticalAnalysis, analysis_id)
    result = dict(analysis.result)
    return _chart(
        metric=analysis.analysis_kind,
        unit=None,
        population=f"sources.kind={analysis.sources.get('kind', 'unavailable')}",
        comparison=dict(analysis.sources),
        points=result or "unavailable",
        uncertainty=result.get("confidence_interval", "unavailable"),
        provenance_ref=analysis.input_hash,
    )


__all__ = ["router"]

"""Fault/robustness experiments: treatment/control trials and the resulting analyses (degradation,
effect estimates, interactions) exactly as persisted -- no new statistics are computed here."""

from fastapi import APIRouter

import experionyx.validation as v
from experionyx.api.common import page, record
from experionyx.api.deps import RegistryDep
from experionyx.faults.entities import FaultAnalysis, FaultExperiment, FaultTrial

router = APIRouter(prefix="/api/faults", tags=["faults"])


@router.get("/experiments")
def list_experiments(
    registry: RegistryDep,
    investigation: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> dict[str, object]:
    filters = {"investigation_id": investigation} if investigation else {}
    return page(registry.find(FaultExperiment, **filters), limit, offset)


@router.get("/experiments/{fault_experiment_id}")
def get_experiment(fault_experiment_id: str, registry: RegistryDep) -> dict[str, object]:
    v.ref("fault_experiment_id", fault_experiment_id, FaultExperiment.PREFIX)
    exp = registry.get(FaultExperiment, fault_experiment_id)
    trials = sorted(
        registry.find(FaultTrial, fault_experiment_id=fault_experiment_id), key=lambda t: t.id
    )
    analyses = registry.find(FaultAnalysis, fault_experiment_id=fault_experiment_id)
    return {
        **record(exp),
        "trials": [record(t) for t in trials],
        "analysis_ids": sorted(a.id for a in analyses),
    }


@router.get("/analyses/{analysis_id}")
def get_analysis(analysis_id: str, registry: RegistryDep) -> dict[str, object]:
    v.ref("analysis_id", analysis_id, FaultAnalysis.PREFIX)
    return record(registry.get(FaultAnalysis, analysis_id))


__all__ = ["router"]

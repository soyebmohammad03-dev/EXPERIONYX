"""Investigation identity, scope, experiments, runs, provenance and evidence-reference coverage."""

from fastapi import APIRouter

import experionyx.validation as v
from experionyx.api.common import page, record
from experionyx.api.deps import RegistryDep
from experionyx.domain import Experiment, Investigation, Run
from experionyx.dossier.entities import EvidenceDossier
from experionyx.provenance import Provenance, RunOutcome
from experionyx.reporting.entities import Report

router = APIRouter(prefix="/api/investigations", tags=["investigations"])


@router.get("")
def list_investigations(
    registry: RegistryDep, limit: int | None = None, offset: int | None = None
) -> dict[str, object]:
    return page(registry.find(Investigation), limit, offset)


@router.get("/{investigation_id}")
def get_investigation(investigation_id: str, registry: RegistryDep) -> dict[str, object]:
    v.ref("investigation_id", investigation_id, Investigation.PREFIX)
    inv = registry.get(Investigation, investigation_id)
    experiments = sorted(
        registry.find(Experiment, investigation_id=investigation_id), key=lambda e: e.id
    )
    reports = registry.find(Report, investigation_id=investigation_id)
    dossiers = registry.find(EvidenceDossier, investigation_id=investigation_id)
    return {
        **record(inv),
        "experiment_ids": [e.id for e in experiments],
        "report_ids": sorted(r.id for r in reports),
        "dossier_ids": sorted(d.id for d in dossiers),
    }


@router.get("/{investigation_id}/experiments")
def list_experiments(
    investigation_id: str,
    registry: RegistryDep,
    limit: int | None = None,
    offset: int | None = None,
) -> dict[str, object]:
    v.ref("investigation_id", investigation_id, Investigation.PREFIX)
    registry.get(Investigation, investigation_id)
    return page(registry.find(Experiment, investigation_id=investigation_id), limit, offset)


@router.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: str, registry: RegistryDep) -> dict[str, object]:
    v.ref("experiment_id", experiment_id, Experiment.PREFIX)
    exp = registry.get(Experiment, experiment_id)
    runs = sorted(registry.find(Run, experiment_id=experiment_id), key=lambda r: r.id)
    return {**record(exp), "run_ids": [r.id for r in runs]}


@router.get("/experiments/{experiment_id}/runs")
def list_runs(
    experiment_id: str, registry: RegistryDep, limit: int | None = None, offset: int | None = None
) -> dict[str, object]:
    v.ref("experiment_id", experiment_id, Experiment.PREFIX)
    registry.get(Experiment, experiment_id)
    return page(registry.find(Run, experiment_id=experiment_id), limit, offset)


@router.get("/runs/{run_id}")
def get_run(run_id: str, registry: RegistryDep) -> dict[str, object]:
    v.ref("run_id", run_id, Run.PREFIX)
    run = registry.get(Run, run_id)
    outcome = registry.find(RunOutcome, run_id=run_id)
    provenance = registry.find(Provenance, run_id=run_id)
    return {
        **record(run),
        "outcome": record(outcome[0]) if outcome else "unavailable",
        "provenance": record(provenance[0]) if provenance else "unavailable",
    }


__all__ = ["router"]

"""Distribution shift: temporal windows and drift analyses, with feature/label/prediction/
performance results read from their persisted artifact documents (via `DriftRegistry.document`)."""

from typing import Any

from fastapi import APIRouter

import experionyx.validation as v
from experionyx.api.common import clamp_pagination, page, record
from experionyx.api.deps import RegistryDep, StoreDep
from experionyx.drift.entities import DriftAnalysis
from experionyx.drift.registry import DriftRegistry

router = APIRouter(prefix="/api/drift", tags=["drift"])


@router.get("/windows")
def list_windows(
    registry: RegistryDep,
    role: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> dict[str, object]:
    reg = DriftRegistry(registry)
    windows = reg.windows(role=role)
    lim, off = clamp_pagination(limit, offset)
    ordered = sorted(windows, key=lambda w: w.id)
    return {
        "items": [record(w) for w in ordered[off : off + lim]],
        "total": len(ordered),
        "limit": lim,
        "offset": off,
    }


@router.get("/analyses")
def list_analyses(
    registry: RegistryDep,
    investigation: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> dict[str, object]:
    filters = {"investigation_id": investigation} if investigation else {}
    return page(registry.find(DriftAnalysis, **filters), limit, offset)


@router.get("/analyses/{analysis_id}")
def get_analysis(analysis_id: str, registry: RegistryDep) -> dict[str, object]:
    v.ref("analysis_id", analysis_id, DriftAnalysis.PREFIX)
    return record(DriftRegistry(registry).analysis(analysis_id))


@router.get("/analyses/{analysis_id}/document")
def get_analysis_document(
    analysis_id: str, name: str, registry: RegistryDep, store: StoreDep
) -> Any:
    v.ref("analysis_id", analysis_id, DriftAnalysis.PREFIX)
    reg = DriftRegistry(registry, store)
    reg.analysis(analysis_id)  # NotFoundError if unknown
    return reg.document(analysis_id, name)


__all__ = ["router"]

"""Statistical analyses exactly as persisted: effect size, CI, p-value, correction method, sample
size and comparison identity live in `result`/`config`/`sources` and are returned verbatim -- this
router never (re)computes a statistic or implies significance beyond what `result` states."""

from fastapi import APIRouter

import experionyx.validation as v
from experionyx.api.common import page
from experionyx.api.deps import RegistryDep
from experionyx.stats.entities import StatisticalAnalysis

router = APIRouter(prefix="/api/stats", tags=["stats"])


@router.get("/analyses")
def list_analyses(
    registry: RegistryDep,
    analysis_kind: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> dict[str, object]:
    filters = {"analysis_kind": analysis_kind} if analysis_kind else {}
    return page(registry.find(StatisticalAnalysis, **filters), limit, offset)


@router.get("/analyses/{analysis_id}")
def get_analysis(analysis_id: str, registry: RegistryDep) -> dict[str, object]:
    v.ref("analysis_id", analysis_id, StatisticalAnalysis.PREFIX)
    entity = registry.get(StatisticalAnalysis, analysis_id)
    return {"id": entity.id, **entity.to_dict()}


__all__ = ["router"]

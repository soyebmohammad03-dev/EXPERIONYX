"""Reproduction attempts, their classifications, environment differences and artifact-digest
verification -- all read from persisted `ReproductionAttempt` rows or the existing read-only
helpers `resolve_target`/`verify_run_artifacts`. No replay is triggered from this router."""

from fastapi import APIRouter

import experionyx.validation as v
from experionyx.api.common import page, record
from experionyx.api.deps import RegistryDep, StoreDep
from experionyx.errors import ValidationError
from experionyx.reproducibility.engine import resolve_target, verify_run_artifacts
from experionyx.reproducibility.entities import ReproductionAttempt
from experionyx.reproducibility.taxonomy import TargetKind

router = APIRouter(prefix="/api/reproducibility", tags=["reproducibility"])


@router.get("/attempts")
def list_attempts(
    registry: RegistryDep,
    investigation: str | None = None,
    target_id: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> dict[str, object]:
    filters: dict[str, str] = {}
    if investigation:
        filters["investigation_id"] = investigation
    if target_id:
        filters["target_id"] = target_id
    return page(registry.find(ReproductionAttempt, **filters), limit, offset)


@router.get("/attempts/{attempt_id}")
def get_attempt(attempt_id: str, registry: RegistryDep) -> dict[str, object]:
    v.ref("attempt_id", attempt_id, ReproductionAttempt.PREFIX)
    return record(registry.get(ReproductionAttempt, attempt_id))


@router.get("/resolve/{target_kind}/{target_id}")
def resolve(target_kind: str, target_id: str, registry: RegistryDep) -> object:
    try:
        kind = TargetKind(target_kind)
    except ValueError as exc:
        raise ValidationError(f"target_kind: {target_kind!r} is not a valid TargetKind") from exc
    return resolve_target(registry, kind, target_id)


@router.get("/runs/{run_id}/artifact-verification")
def verify_artifacts(run_id: str, registry: RegistryDep, store: StoreDep) -> object:
    v.ref("run_id", run_id, "run")
    return verify_run_artifacts(registry, store, run_id)


__all__ = ["router"]

"""Reliability dimensions with explicit per-dimension status and supporting/missing evidence.
Never aggregates `dimension_status` into a single score -- see docs/reliability.md."""

from fastapi import APIRouter

import experionyx.validation as v
from experionyx.api.common import clamp_pagination, record
from experionyx.api.deps import RegistryDep
from experionyx.reliability.entities import ReliabilityProfile
from experionyx.reliability.registry import ReliabilityProfileRegistry

router = APIRouter(prefix="/api/reliability", tags=["reliability"])


@router.get("/profiles")
def search_profiles(
    registry: RegistryDep,
    investigation: str | None = None,
    model: str | None = None,
    dataset: str | None = None,
    scope: str | None = None,
    split: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> dict[str, object]:
    reg = ReliabilityProfileRegistry(registry)
    found = reg.search(
        investigation=investigation, model=model, dataset=dataset, scope=scope, split=split
    )
    lim, off = clamp_pagination(limit, offset)
    window = found[off : off + lim]
    return {
        "items": [record(p) for p in window],
        "total": len(found),
        "limit": lim,
        "offset": off,
    }


@router.get("/profiles/{profile_id}")
def get_profile(profile_id: str, registry: RegistryDep) -> dict[str, object]:
    v.ref("profile_id", profile_id, ReliabilityProfile.PREFIX)
    reg = ReliabilityProfileRegistry(registry)
    profile = reg.get(profile_id)
    refs = reg.references(profile_id)
    return {
        **record(profile),
        "references": [record(r) for r in refs],
    }


@router.get("/profiles/{profile_id}/provenance")
def get_profile_provenance(profile_id: str, registry: RegistryDep) -> object:
    v.ref("profile_id", profile_id, ReliabilityProfile.PREFIX)
    return ReliabilityProfileRegistry(registry).provenance(profile_id)


@router.get("/compare")
def compare_profiles(a: str, b: str, registry: RegistryDep) -> object:
    v.ref("a", a, ReliabilityProfile.PREFIX)
    v.ref("b", b, ReliabilityProfile.PREFIX)
    return ReliabilityProfileRegistry(registry).compare(a, b)


__all__ = ["router"]

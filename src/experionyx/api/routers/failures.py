"""Discovered failures: clusters, modes, evidence, and graph-taxonomy relationships. Cluster
`metrics` (size, compactness, stability, prevalence with denominators) are returned verbatim."""

from fastapi import APIRouter

import experionyx.validation as v
from experionyx.api.common import page, record
from experionyx.api.deps import RegistryDep
from experionyx.failures.entities import (
    FailureCluster,
    FailureEvidence,
    FailureMode,
    FailureRelationship,
)

router = APIRouter(prefix="/api/failures", tags=["failures"])


@router.get("/clusters")
def list_clusters(
    registry: RegistryDep,
    investigation: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> dict[str, object]:
    filters = {"investigation_id": investigation} if investigation else {}
    return page(registry.find(FailureCluster, **filters), limit, offset)


@router.get("/clusters/{cluster_id}")
def get_cluster(cluster_id: str, registry: RegistryDep) -> dict[str, object]:
    v.ref("cluster_id", cluster_id, FailureCluster.PREFIX)
    cluster = registry.get(FailureCluster, cluster_id)
    modes = registry.find(FailureMode, cluster_id=cluster_id)
    return {**record(cluster), "mode_ids": sorted(m.id for m in modes)}


@router.get("/modes")
def list_modes(
    registry: RegistryDep,
    investigation: str | None = None,
    cluster: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> dict[str, object]:
    filters: dict[str, str] = {}
    if investigation:
        filters["investigation_id"] = investigation
    if cluster:
        filters["cluster_id"] = cluster
    return page(registry.find(FailureMode, **filters), limit, offset)


@router.get("/modes/{mode_id}")
def get_mode(mode_id: str, registry: RegistryDep) -> dict[str, object]:
    v.ref("mode_id", mode_id, FailureMode.PREFIX)
    mode = registry.get(FailureMode, mode_id)
    evidence = registry.find(FailureEvidence, failure_mode_id=mode_id)
    by_id = {
        r.id: r
        for r in (
            *registry.find(FailureRelationship, subject_id=mode_id),
            *registry.find(FailureRelationship, object_id=mode_id),
        )
    }
    related = list(by_id.values())
    return {
        **record(mode),
        "evidence": [record(e) for e in evidence] or "unavailable",
        "relationships": [record(r) for r in related],
    }


__all__ = ["router"]

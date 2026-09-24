"""Evidence dossiers: list, retrieve, inspect items/findings/limitations, and export Markdown via
`dossier.render.render_markdown` -- the domain-layer renderer added alongside this API."""

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

import experionyx.validation as v
from experionyx.api.common import page, record
from experionyx.api.deps import RegistryDep
from experionyx.dossier.entities import DossierFinding, DossierItem, EvidenceDossier
from experionyx.dossier.render import render_markdown

router = APIRouter(prefix="/api/dossiers", tags=["dossier"])


@router.get("")
def list_dossiers(
    registry: RegistryDep,
    investigation: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> dict[str, object]:
    filters = {"investigation_id": investigation} if investigation else {}
    return page(registry.find(EvidenceDossier, **filters), limit, offset)


@router.get("/{dossier_id}")
def get_dossier(dossier_id: str, registry: RegistryDep) -> dict[str, object]:
    v.ref("dossier_id", dossier_id, EvidenceDossier.PREFIX)
    dossier = registry.get(EvidenceDossier, dossier_id)
    items = registry.find(DossierItem, dossier_id=dossier_id)
    findings = registry.find(DossierFinding, dossier_id=dossier_id)
    return {
        **record(dossier),
        "item_ids": sorted(i.id for i in items),
        "finding_ids": sorted(f.id for f in findings),
    }


@router.get("/{dossier_id}/items")
def get_items(dossier_id: str, registry: RegistryDep) -> dict[str, object]:
    v.ref("dossier_id", dossier_id, EvidenceDossier.PREFIX)
    registry.get(EvidenceDossier, dossier_id)
    items = sorted(registry.find(DossierItem, dossier_id=dossier_id), key=lambda i: i.id)
    return {"items": [record(i) for i in items], "total": len(items)}


@router.get("/{dossier_id}/findings")
def get_findings(dossier_id: str, registry: RegistryDep) -> dict[str, object]:
    v.ref("dossier_id", dossier_id, EvidenceDossier.PREFIX)
    registry.get(EvidenceDossier, dossier_id)
    findings = sorted(registry.find(DossierFinding, dossier_id=dossier_id), key=lambda f: f.id)
    return {"items": [record(f) for f in findings], "total": len(findings)}


@router.get("/{dossier_id}/export.md", response_class=PlainTextResponse)
def export_markdown(dossier_id: str, registry: RegistryDep) -> str:
    v.ref("dossier_id", dossier_id, EvidenceDossier.PREFIX)
    dossier = registry.get(EvidenceDossier, dossier_id)
    return render_markdown(registry, dossier)


__all__ = ["router"]

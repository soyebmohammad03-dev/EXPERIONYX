"""Research reports: list, retrieve, inspect claims (`ReportFinding`), and export Markdown via
the existing `reporting.render.render_markdown` -- no re-rendering logic lives in the API."""

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

import experionyx.validation as v
from experionyx.api.common import page, record
from experionyx.api.deps import RegistryDep
from experionyx.reporting.entities import Report, ReportFinding
from experionyx.reporting.render import render_markdown

router = APIRouter(prefix="/api/reports", tags=["reporting"])


@router.get("")
def list_reports(
    registry: RegistryDep,
    investigation: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> dict[str, object]:
    filters = {"investigation_id": investigation} if investigation else {}
    return page(registry.find(Report, **filters), limit, offset)


@router.get("/{report_id}")
def get_report(report_id: str, registry: RegistryDep) -> dict[str, object]:
    v.ref("report_id", report_id, Report.PREFIX)
    report = registry.get(Report, report_id)
    findings = registry.find(ReportFinding, report_id=report_id)
    return {**record(report), "finding_ids": sorted(f.id for f in findings)}


@router.get("/{report_id}/findings")
def get_findings(report_id: str, registry: RegistryDep) -> dict[str, object]:
    v.ref("report_id", report_id, Report.PREFIX)
    registry.get(Report, report_id)
    findings = sorted(registry.find(ReportFinding, report_id=report_id), key=lambda f: f.id)
    return {"items": [record(f) for f in findings], "total": len(findings)}


@router.get("/{report_id}/export.md", response_class=PlainTextResponse)
def export_markdown(report_id: str, registry: RegistryDep) -> str:
    v.ref("report_id", report_id, Report.PREFIX)
    report = registry.get(Report, report_id)
    return render_markdown(registry, report)


__all__ = ["router"]

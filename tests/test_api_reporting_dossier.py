"""API tests for reports and evidence dossiers: retrieval, findings and Markdown export."""

from api_client import client
from api_world import ApiWorld


def test_report_get_and_findings(api_world: ApiWorld) -> None:
    c = client(api_world)
    report = c.get(f"/api/reports/{api_world.report_id}").json()
    assert report["id"] == api_world.report_id
    findings = c.get(f"/api/reports/{api_world.report_id}/findings").json()
    assert findings["total"] == len(report["finding_ids"])


def test_report_export_markdown_matches_engine_render(api_world: ApiWorld) -> None:
    from experionyx.reporting.entities import Report
    from experionyx.reporting.render import render_markdown
    from experionyx.sqlite import SqliteRegistry

    res = client(api_world).get(f"/api/reports/{api_world.report_id}/export.md")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/plain")
    with SqliteRegistry(api_world.ws / "registry.sqlite") as reg:
        expected = render_markdown(reg, reg.get(Report, api_world.report_id))
    assert res.text == expected


def test_report_not_found(api_world: ApiWorld) -> None:
    res = client(api_world).get(f"/api/reports/rpt_{'0' * 32}")
    assert res.status_code == 404


def test_dossier_get_items_and_findings(api_world: ApiWorld) -> None:
    c = client(api_world)
    dossier = c.get(f"/api/dossiers/{api_world.dossier_id}").json()
    assert dossier["id"] == api_world.dossier_id
    items = c.get(f"/api/dossiers/{api_world.dossier_id}/items").json()
    assert items["total"] == len(dossier["item_ids"])


def test_dossier_export_markdown_matches_engine_render(api_world: ApiWorld) -> None:
    from experionyx.dossier.entities import EvidenceDossier
    from experionyx.dossier.render import render_markdown
    from experionyx.sqlite import SqliteRegistry

    res = client(api_world).get(f"/api/dossiers/{api_world.dossier_id}/export.md")
    assert res.status_code == 200
    with SqliteRegistry(api_world.ws / "registry.sqlite") as reg:
        expected = render_markdown(reg, reg.get(EvidenceDossier, api_world.dossier_id))
    assert res.text == expected


def test_dossier_not_found(api_world: ApiWorld) -> None:
    res = client(api_world).get(f"/api/dossiers/dsr_{'0' * 32}")
    assert res.status_code == 404

"""Deterministic exports of a persisted `Report`. Markdown is canonical; HTML is a thin,
dependency-free derived view (no template engine, no frontend framework) of the same content.
Both preserve the report's identity and every evidence reference (see docs/reporting.md)."""

import html as _html

from experionyx.domain import to_jsonable
from experionyx.registry import Registry
from experionyx.reporting.entities import Report, ReportFinding


def _findings_table(findings: tuple[ReportFinding, ...]) -> str:
    if not findings:
        return "_No findings were generated._"
    lines = ["| ID | Status | Statement | Evidence |", "| --- | --- | --- | --- |"]
    for f in findings:
        evidence = "; ".join(f"`{e.source_id}` ({e.source_kind})" for e in f.evidence) or "—"
        lines.append(f"| `{f.id}` | {f.status.value} | {f.statement} | {evidence} |")
    return "\n".join(lines)


def render_markdown(registry: Registry, report: Report) -> str:
    """Canonical human-readable export. Byte-identical for byte-identical report content."""
    findings = tuple(sorted(registry.find(ReportFinding, report_id=report.id), key=lambda f: f.id))
    parts = [
        f"# {report.title}",
        "",
        f"- **Report ID:** `{report.id}`",
        f"- **Type:** {report.report_type.value}",
        f"- **Investigation:** `{report.investigation_id}`",
        f"- **Template:** `{report.template_id}`",
        f"- **Status:** {report.status.value}",
        f"- **Generated:** {report.generated_at.isoformat()}",
        f"- **Engine version:** {report.engine_version}",
        f"- **Evidence digest:** `{report.source_evidence_digest}`",
        "",
    ]
    for section in report.sections:
        parts.append(f"## {section.title}" + ("" if section.available else " (unavailable)"))
        parts.append("")
        parts.append(section.content)
        parts.append("")
    parts.append("## Claim / Evidence Matrix")
    parts.append("")
    parts.append(_findings_table(findings))
    parts.append("")
    parts.append("## Reproducibility")
    parts.append("")
    parts.append("```json")
    import json

    parts.append(json.dumps(to_jsonable(report.reproducibility), indent=2, sort_keys=True))
    parts.append("```")
    parts.append("")
    return "\n".join(parts)


def render_html(registry: Registry, report: Report) -> str:
    """A minimal derived HTML view: the Markdown export wrapped as escaped `<pre>` content, so
    there is no markdown-to-HTML dependency to maintain."""
    body = _html.escape(render_markdown(registry, report))
    return (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{_html.escape(report.title)}</title></head>"
        f"<body><pre>{body}</pre></body></html>"
    )


__all__ = ["render_html", "render_markdown"]

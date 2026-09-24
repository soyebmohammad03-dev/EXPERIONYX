"""Deterministic Markdown export of a persisted `EvidenceDossier` (see docs/dossier.md). Mirrors
`reporting.render`: byte-identical for byte-identical dossier content, no template engine."""

from experionyx.dossier.entities import DossierFinding, DossierItem, EvidenceDossier
from experionyx.registry import Registry


def render_markdown(registry: Registry, dossier: EvidenceDossier) -> str:
    """Canonical human-readable export. Byte-identical for byte-identical dossier content."""
    items = tuple(sorted(registry.find(DossierItem, dossier_id=dossier.id), key=lambda i: i.id))
    findings = tuple(
        sorted(registry.find(DossierFinding, dossier_id=dossier.id), key=lambda f: f.id)
    )
    lines = [
        f"# Evidence Dossier: {dossier.research_question}",
        "",
        f"- **Dossier ID:** `{dossier.id}`",
        f"- **Investigation:** `{dossier.investigation_id}`",
        f"- **Source:** {dossier.source_kind.value} `{dossier.source_id}`",
        f"- **Reports:** {', '.join(dossier.report_ids) or 'none'}",
        f"- **Evidence digest:** `{dossier.source_evidence_digest}`",
        "",
        "## Findings (sufficiency analysis)",
        "",
        "| ID | Status | Statement |",
        "| --- | --- | --- |",
        *(f"| `{f.id}` | {f.status.value} | {f.statement} |" for f in findings),
        "",
        "## Evidence Items",
        "",
        *(f"- `{i.id}` ({i.item_kind.value}/{i.source_kind}): `{i.source_id}`" for i in items),
        "",
        "## Limitations",
        "",
        *(f"- {x}" for x in dossier.limitations),
    ]
    return "\n".join(lines)


__all__ = ["render_markdown"]

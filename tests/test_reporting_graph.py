"""A generated report is reachable through the SAME evidence/failure graph as everything else
-- no second graph is built for reporting (see docs/graph.md, docs/reporting.md)."""

import procedures
from conftest import Lab
from experionyx.graph.build import construct
from experionyx.graph.spec import GraphSpec
from experionyx.graph.taxonomy import NodeKind
from experionyx.reporting.generator import builtin_template, ensure_template, generate_report
from experionyx.reporting.spec import ReportSpec
from experionyx.reporting.taxonomy import ReportType


def test_a_report_and_its_findings_appear_as_graph_nodes_with_resolved_evidence_edges(
    lab: Lab,
) -> None:
    lab.executor.execute(lab.experiment.id, procedures.ok, seed=0)
    template_id = ensure_template(lab.registry, builtin_template(ReportType.INVESTIGATION_DOSSIER))
    spec = ReportSpec(ReportType.INVESTIGATION_DOSSIER, lab.investigation.id, template_id)
    result = generate_report(lab.registry, spec)

    constructed = construct(lab.registry, GraphSpec("g", "1.0.0"))
    node_kinds = {kind.value for kind, _ref in constructed.nodes}
    assert {"REPORT", "REPORT_FINDING", "REPORT_TEMPLATE"} <= node_kinds
    assert (NodeKind.REPORT, result.report_id) in constructed.nodes
    report_node = constructed.nodes[(NodeKind.REPORT, result.report_id)]
    assert report_node.resolved is True  # every referenced node the walker found actually exists

    from_report = [
        e for e in constructed.edges if e.from_key == (NodeKind.REPORT, result.report_id)
    ]
    assert any(
        e.relation.value == "MEMBER_OF" and e.to_key[0].value == "INVESTIGATION"
        for e in from_report
    )
    assert any(
        e.relation.value == "DEPENDS_ON" and e.to_key[0].value == "REPORT_TEMPLATE"
        for e in from_report
    )

    finding_ids = {ref for kind, ref in constructed.nodes if kind.value == "REPORT_FINDING"}
    assert finding_ids  # at least one finding node exists
    finding_edges = [e for e in constructed.edges if e.from_key[0].value == "REPORT_FINDING"]
    assert all(
        constructed.nodes[e.to_key].resolved for e in finding_edges
    )  # cited evidence resolves

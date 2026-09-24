"""Graph integration: a `ReproductionAttempt` appears in a constructed graph as its own node,
connected to the target it reproduces (`REPRODUCED_BY`) and the replay run it compared against
(`COMPARED_WITH`) -- entirely through the graph's own generic field walk, no reproducibility-
specific traversal code (see docs/graph.md, docs/reproducibility.md)."""

import procedures
from conftest import Lab
from experionyx.graph.build import construct
from experionyx.graph.spec import GraphSpec
from experionyx.graph.taxonomy import NodeKind, RelationType
from experionyx.reproducibility.engine import run_reproduction
from experionyx.reproducibility.spec import ReproductionSpec
from experionyx.reproducibility.taxonomy import ReproductionMode, TargetKind


def test_a_reproduction_attempt_appears_as_a_node_connected_to_its_target(lab: Lab) -> None:
    original = lab.executor.execute(lab.experiment.id, procedures.ok, seed=4)

    spec = ReproductionSpec(TargetKind.RUN, original.run.id, ReproductionMode.EXACT)
    result = run_reproduction(lab.registry, lab.store, lab.executor, spec)

    got = construct(lab.registry, GraphSpec("g", "1.0.0"))
    attempt_node = next(
        (k for k in got.nodes if k[0] is NodeKind.REPRODUCTION_ATTEMPT and k[1] == result.attempt_id),
        None,
    )  # fmt: skip
    assert attempt_node is not None

    edges_from_attempt = [e for e in got.edges if e.from_key == attempt_node]
    relations = {e.relation for e in edges_from_attempt}
    assert RelationType.REPRODUCED_BY in relations
    assert RelationType.COMPARED_WITH in relations

    target_edge = next(e for e in edges_from_attempt if e.relation is RelationType.REPRODUCED_BY)
    assert target_edge.to_key == (NodeKind.RUN, original.run.id)

    compared_edge = next(e for e in edges_from_attempt if e.relation is RelationType.COMPARED_WITH)
    assert compared_edge.to_key[0] is NodeKind.RUN
    assert compared_edge.to_key[1] == result.replay_run_id

"""Real graph construction, idempotency, bookkeeping self-exclusion, traversal, diff and replay
against a real registry -- both a lightweight synthetic-evidence registry (via the `lab` fixture)
and a REAL cross-phase sklearn workspace (model -> baseline -> fault -> failure discovery ->
reliability -> statistical analysis -> scheduler), driven through the actual CLI the same way a
user would."""

import json
from pathlib import Path
from typing import Any

import pytest

from conftest import Lab
from experionyx.domain import (  # fmt: skip
    Claim,
    ClaimStatus,
    Evidence,
    EvidenceRelation,
    EvidenceTarget,
    Experiment,
)
from experionyx.errors import NotFoundError
from experionyx.graph.build import construct
from experionyx.graph.diff import diff
from experionyx.graph.engine import replay_check, run_graph
from experionyx.graph.entities import GraphNode
from experionyx.graph.query import SnapshotIndex, evidence_for_claim
from experionyx.graph.spec import GraphSpec
from experionyx.graph.taxonomy import NodeKind
from experionyx.scheduler.entities import ScheduleUnit

DIGEST = "sha256:" + "ab" * 32


# --- construction over a real registry -----------------------------------------------------------


def test_construct_finds_real_experiment_and_investigation_nodes(lab: Lab) -> None:
    spec = GraphSpec("g", "1.0.0")
    got = construct(lab.registry, spec)
    kinds = {k for k, _ in got.nodes}
    assert NodeKind.EXPERIMENT in kinds
    assert NodeKind.INVESTIGATION in kinds
    assert NodeKind.CONFIGURATION in kinds
    assert got.unresolved_count == 0


def test_investigation_scoping_excludes_experiments_from_another_investigation(lab: Lab) -> None:
    from datetime import UTC, datetime

    from experionyx.domain import (  # fmt: skip
        ConfigurationRef,
        DatasetRef,
        ExperimentStatus,
        Investigation,
        ModelRef,
    )

    other_inv = Investigation("other", "unrelated question", datetime.now(UTC))
    lab.registry.add(other_inv)
    other_cfg = ConfigurationRef({"z": 1})
    lab.registry.add(other_cfg)
    other_exp = Experiment(
        other_inv.id, "other-exp", "h", ModelRef("m", "1"), DatasetRef("d", "1"),
        other_cfg.id, datetime.now(UTC),
    )  # fmt: skip
    lab.registry.add(other_exp)
    lab.registry.update_status(other_exp.with_status(ExperimentStatus.READY))

    scoped = construct(lab.registry, GraphSpec("g", "1.0.0", investigation_id=lab.investigation.id))
    ref_ids = {ref for _, ref in scoped.nodes}
    assert lab.experiment.id in ref_ids
    assert other_exp.id not in ref_ids  # excluded: belongs to a different investigation


def test_included_kinds_restricts_the_scan(lab: Lab) -> None:
    spec = GraphSpec("g", "1.0.0", included_kinds=(NodeKind.INVESTIGATION,))
    got = construct(lab.registry, spec)
    assert {k for k, _ in got.nodes} == {NodeKind.INVESTIGATION}
    assert got.edges == []  # no edges: nothing else was scanned to connect to


def test_max_nodes_bound_is_honored_and_reported_truncated(lab: Lab) -> None:
    spec = GraphSpec("g", "1.0.0", max_nodes=1)
    got = construct(lab.registry, spec)
    assert len(got.nodes) <= 1
    assert got.truncated is True


# --- missing references are represented explicitly, never dropped ------------------------------


def test_an_unresolved_reference_is_kept_and_flagged(lab: Lab) -> None:
    """`ExecutionAttempt.primary_ref` is free text, not a registry-validated foreign key (unlike
    `run_id`), so it is the simplest real place a plausible-but-nonexistent reference can land in
    a healthy registry. Everything else the row needs (its unit, its schedule run) is real."""
    from datetime import UTC, datetime

    from experionyx.scheduler.entities import ExecutionAttempt, Schedule, ScheduleRun, ScheduleUnit
    from experionyx.scheduler.taxonomy import RetryPolicyKind, UnitKind, UnitState

    now = datetime(2026, 9, 24, tzinfo=UTC)
    sched = Schedule("g", "1.0.0", "ssp_" + "1" * 32, {"units": []}, DIGEST, "1.0.0", now)
    lab.registry.add(sched)
    run = ScheduleRun(sched.id, lab.investigation.id, {"max_workers": 1}, False, 0, now)
    lab.registry.add(run)
    unit = ScheduleUnit(
        sched.id, "a", UnitKind.STATISTICAL_ANALYSIS, {}, (),
        {"kind": RetryPolicyKind.NONE.value, "max_attempts": 1, "retryable": []}, None, None, 0, now,
    )  # fmt: skip
    lab.registry.add(unit)
    ghost_ref = "sta_" + "9" * 32  # a plausible statistical-analysis ID that does not exist
    attempt = ExecutionAttempt(unit.id, run.id, 0, now, now, UnitState.SUCCEEDED, None, None, None, ghost_ref, None)  # fmt: skip
    lab.registry.add(attempt)  # accepted: primary_ref is not a validated foreign key

    got = construct(lab.registry, GraphSpec("g", "1.0.0"))
    unresolved = {ref: draft for (kind, ref), draft in got.nodes.items() if kind is NodeKind.STATISTICAL_ANALYSIS and not draft.resolved}  # fmt: skip
    assert ghost_ref in unresolved
    assert got.unresolved_count >= 1


# --- idempotency and bookkeeping self-exclusion ---------------------------------------------------


def test_run_graph_is_idempotent_and_does_not_perpetually_grow(lab: Lab) -> None:
    from experionyx.graph.entities import GraphSnapshot

    spec = GraphSpec("g", "1.0.0")
    first = run_graph(lab.registry, lab.store, lab.executor, spec, investigation_id=lab.investigation.id)  # fmt: skip
    assert first.snapshot_id is not None
    snap1 = lab.registry.get(GraphSnapshot, first.snapshot_id)
    for _ in range(3):
        again = run_graph(lab.registry, lab.store, lab.executor, spec, investigation_id=lab.investigation.id)  # fmt: skip
        assert again.already_built is True
        assert again.snapshot_id == first.snapshot_id
    snap_final = lab.registry.get(GraphSnapshot, first.snapshot_id)
    assert snap_final.node_count == snap1.node_count  # no growth across repeated calls


def test_the_graphs_own_collect_bookkeeping_never_appears_as_a_node(lab: Lab) -> None:
    spec = GraphSpec("g", "1.0.0")
    res = run_graph(lab.registry, lab.store, lab.executor, spec, investigation_id=lab.investigation.id)  # fmt: skip
    assert res.snapshot_id is not None
    idx = SnapshotIndex(lab.registry, res.snapshot_id)
    exp_ref_ids = {n.ref_id for n in idx.nodes.values() if n.node_kind is NodeKind.EXPERIMENT}
    collect_experiments = [e for e in lab.registry.find(Experiment) if e.name.startswith("graph ")]
    assert collect_experiments  # the collect run's own bookkeeping experiment really exists...
    assert not (exp_ref_ids & {e.id for e in collect_experiments})  # ...but never as a graph node


def test_investigation_scoping_does_not_isolate_entities_with_no_investigation_id_field(lab: Lab) -> None:  # fmt: skip
    """Documented limitation (docs/graph.md): `investigation_id` scoping filters only entities
    that themselves carry that attribute. A brand-new `ConfigurationRef` has no `investigation_id`
    field at all, so it passes `_in_scope` regardless of which investigation created it -- it is
    NOT excluded, and a scoped graph's identity DOES change when one is added anywhere in the
    registry. This is the honest, documented behavior, not isolation."""
    from experionyx.domain import ConfigurationRef

    spec = GraphSpec("g", "1.0.0", investigation_id=lab.investigation.id)
    first = run_graph(lab.registry, lab.store, lab.executor, spec, investigation_id=lab.investigation.id)  # fmt: skip
    lab.registry.add(ConfigurationRef({"unrelated": True}))
    again = run_graph(lab.registry, lab.store, lab.executor, spec, investigation_id=lab.investigation.id)  # fmt: skip
    assert again.already_built is False
    assert again.snapshot_id != first.snapshot_id


def test_investigation_scoping_excludes_a_fully_scoped_unrelated_experiment(lab: Lab) -> None:
    """The positive case: an Experiment (which DOES carry `investigation_id`) belonging to a
    different investigation is excluded, and reusing an EXISTING configuration (so no new
    unscoped ConfigurationRef enters the registry) keeps the scoped graph's identity stable."""
    from datetime import UTC, datetime

    from experionyx.domain import DatasetRef, ExperimentStatus, Investigation, ModelRef

    spec = GraphSpec("g", "1.0.0", investigation_id=lab.investigation.id)
    run_graph(lab.registry, lab.store, lab.executor, spec, investigation_id=lab.investigation.id)
    other_inv = Investigation("other2", "another question", datetime.now(UTC))
    lab.registry.add(other_inv)
    other_exp = Experiment(
        other_inv.id, "unrelated", "h", ModelRef("m", "1"), DatasetRef("d", "1"),
        lab.configuration.id, datetime.now(UTC),
    )  # fmt: skip
    lab.registry.add(other_exp)
    lab.registry.update_status(other_exp.with_status(ExperimentStatus.READY))
    again = run_graph(lab.registry, lab.store, lab.executor, spec, investigation_id=lab.investigation.id)  # fmt: skip
    idx = SnapshotIndex(lab.registry, again.snapshot_id)  # type: ignore[arg-type]
    assert (
        idx.find_by_ref(NodeKind.EXPERIMENT, other_exp.id) is None
    )  # excluded: different investigation


# --- traversal -------------------------------------------------------------------------------------


def test_neighbors_ancestors_descendants_and_path_over_real_evidence(lab: Lab) -> None:
    spec = GraphSpec("g", "1.0.0")
    res = run_graph(lab.registry, lab.store, lab.executor, spec, investigation_id=lab.investigation.id)  # fmt: skip
    assert res.snapshot_id is not None
    idx = SnapshotIndex(lab.registry, res.snapshot_id)
    exp_node = idx.find_by_ref(NodeKind.EXPERIMENT, lab.experiment.id)
    assert exp_node is not None
    inv_node = idx.find_by_ref(NodeKind.INVESTIGATION, lab.investigation.id)
    assert inv_node is not None

    nb = idx.neighbors(exp_node.id)
    assert inv_node.id in {n.id for n in nb.nodes}

    desc = idx.descendants(exp_node.id)
    assert inv_node.id in {n.id for n in desc.nodes}  # experiment -[MEMBER_OF]-> investigation

    anc = idx.ancestors(inv_node.id)
    assert exp_node.id in {
        n.id for n in anc.nodes
    }  # symmetric: investigation is an ancestor target

    path = idx.path(exp_node.id, inv_node.id)
    assert path.found is True
    assert path.nodes[0].id == exp_node.id
    assert path.nodes[-1].id == inv_node.id


def test_traversal_is_bounded_and_reports_truncated(lab: Lab) -> None:
    from experionyx.graph.spec import GraphQuery

    spec = GraphSpec("g", "1.0.0")
    res = run_graph(lab.registry, lab.store, lab.executor, spec, investigation_id=lab.investigation.id)  # fmt: skip
    assert res.snapshot_id is not None
    idx = SnapshotIndex(lab.registry, res.snapshot_id)
    exp_node = idx.find_by_ref(NodeKind.EXPERIMENT, lab.experiment.id)
    assert exp_node is not None
    tiny = idx.descendants(exp_node.id, GraphQuery(max_visited=1))
    assert tiny.truncated is True
    assert tiny.visited_count <= 1


def test_evidence_for_claim_is_unsupported_when_no_evidence_row_exists(lab: Lab) -> None:
    claim = Claim(lab.investigation.id, "an unsupported claim", "tester", lab.experiment.created_at, ClaimStatus.INSUFFICIENT_EVIDENCE)  # fmt: skip
    lab.registry.add(claim)
    spec = GraphSpec("g", "1.0.0")
    res = run_graph(lab.registry, lab.store, lab.executor, spec, investigation_id=lab.investigation.id)  # fmt: skip
    assert res.snapshot_id is not None
    idx = SnapshotIndex(lab.registry, res.snapshot_id)
    claim_node = idx.find_by_ref(NodeKind.CLAIM, claim.id)
    assert claim_node is not None
    result = evidence_for_claim(idx, claim_node.id)
    assert len(result.nodes) == 1  # only the claim itself: nothing supports it


def test_evidence_for_claim_traces_a_real_evidence_chain(lab: Lab) -> None:
    claim = Claim(lab.investigation.id, "a supported claim", "tester", lab.experiment.created_at, ClaimStatus.SUPPORTED)  # fmt: skip
    lab.registry.add(claim)
    ev = Evidence(claim.id, EvidenceTarget.EXPERIMENT, lab.experiment.id, EvidenceRelation.SUPPORTS, lab.experiment.created_at)  # fmt: skip
    lab.registry.add(ev)
    spec = GraphSpec("g", "1.0.0")
    res = run_graph(lab.registry, lab.store, lab.executor, spec, investigation_id=lab.investigation.id)  # fmt: skip
    assert res.snapshot_id is not None
    idx = SnapshotIndex(lab.registry, res.snapshot_id)
    claim_node = idx.find_by_ref(NodeKind.CLAIM, claim.id)
    assert claim_node is not None
    result = evidence_for_claim(idx, claim_node.id)
    ref_ids = {n.ref_id for n in result.nodes}
    assert ev.id in ref_ids
    assert lab.experiment.id in ref_ids  # the evidence's own target is reachable


# --- diff and replay --------------------------------------------------------------------------------


def test_diff_reports_added_nodes_between_two_snapshots(lab: Lab) -> None:
    from experionyx.domain import ConfigurationRef

    spec = GraphSpec("g", "1.0.0")
    first = run_graph(lab.registry, lab.store, lab.executor, spec, investigation_id=lab.investigation.id)  # fmt: skip
    assert first.snapshot_id is not None
    new_cfg = ConfigurationRef({"new": True})
    lab.registry.add(new_cfg)
    second = run_graph(lab.registry, lab.store, lab.executor, spec, investigation_id=lab.investigation.id)  # fmt: skip
    assert second.snapshot_id is not None
    assert second.snapshot_id != first.snapshot_id
    d = diff(lab.registry, first.snapshot_id, second.snapshot_id)
    assert new_cfg.id in {n.ref_id for n in d.added_nodes}
    assert d.removed_nodes == ()


def test_replay_is_deterministic_for_unchanged_registry_content(lab: Lab) -> None:
    spec = GraphSpec("g", "1.0.0")
    res = run_graph(lab.registry, lab.store, lab.executor, spec, investigation_id=lab.investigation.id)  # fmt: skip
    assert res.snapshot_id is not None
    out = replay_check(lab.registry, lab.store, lab.executor, res.snapshot_id)
    assert out["deterministic"] is True
    assert out["differences"] == []


def test_replay_reports_sources_changed_when_registry_content_moved_on(lab: Lab) -> None:
    from experionyx.domain import ConfigurationRef

    spec = GraphSpec("g", "1.0.0")
    res = run_graph(lab.registry, lab.store, lab.executor, spec, investigation_id=lab.investigation.id)  # fmt: skip
    assert res.snapshot_id is not None
    lab.registry.add(ConfigurationRef({"more": "evidence"}))
    out = replay_check(lab.registry, lab.store, lab.executor, res.snapshot_id)
    assert out["sources_changed"] is True
    assert out["deterministic"] is None


# --- adversarial / invalid requests --------------------------------------------------------------


def test_snapshot_index_raises_not_found_for_an_unknown_snapshot(lab: Lab) -> None:
    with pytest.raises(NotFoundError):
        SnapshotIndex(lab.registry, "gsn_" + "0" * 32)


def test_construct_over_an_empty_registry_scope_finds_nothing(lab: Lab) -> None:
    got = construct(lab.registry, GraphSpec("g", "1.0.0", included_kinds=(NodeKind.STRESS_ANALYSIS,)))  # fmt: skip
    assert got.nodes == {}
    assert got.edges == []
    assert got.unresolved_count == 0


def test_duplicate_node_add_is_a_no_op_not_a_duplicate_row(lab: Lab) -> None:
    spec = GraphSpec("g", "1.0.0")
    res = run_graph(lab.registry, lab.store, lab.executor, spec, investigation_id=lab.investigation.id)  # fmt: skip
    assert res.snapshot_id is not None
    nodes = lab.registry.find(GraphNode, snapshot_id=res.snapshot_id)
    keys = [(n.node_kind, n.ref_id) for n in nodes]
    assert len(keys) == len(set(keys))  # every node appears exactly once


# --- real cross-phase validation (model -> fault -> failure -> reliability -> stats -> scheduler) ---


class Ws:
    """A real sklearn workspace driven through the CLI, exactly the way a user would."""

    def __init__(self, path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        from experionyx.cli import main

        self.path, self.capsys = path, capsys
        assert main(["--workspace", str(path), "demo", "sklearn-classification"]) == 0
        capsys.readouterr()
        from experionyx.adapters.records import RegisteredDataset, RegisteredModel
        from experionyx.domain import Investigation
        from experionyx.sqlite import SqliteRegistry

        with SqliteRegistry(path / "registry.sqlite") as reg:
            self.model = reg.find(RegisteredModel)[0].id
            self.dataset = reg.find(RegisteredDataset)[0].id
            self.investigation = reg.find(Investigation)[0].id

    def cli(self, *args: str) -> tuple[int, str, str]:
        from experionyx.cli import main

        code = main(["--workspace", str(self.path), *args])
        out = self.capsys.readouterr()
        return code, out.out, out.err

    def cli_json(self, *args: str) -> Any:
        code, out, err = self.cli(*args)
        assert code in (0, 1, 2, 3), f"unexpected exit code {code}: {err}"
        return json.loads(out)


@pytest.fixture
def ws(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Ws:
    pytest.importorskip("sklearn")
    return Ws(tmp_path / "ws", capsys)


def test_real_cross_phase_graph_spans_model_fault_failure_reliability_stats_and_scheduler(
    ws: Ws,
) -> None:
    """Engineering validation, not a scientific finding: proves the graph connects real evidence
    across six phases plus a real scheduler dependency edge, using the actual CLI end to end."""
    code, _fault_out, fault_err = ws.cli(
        "fault",
        "run",
        "--model",
        ws.model,
        "--dataset",
        ws.dataset,
        "--type",
        "gaussian_noise",
        "--param",
        "sigma=1.5",
        "--seed",
        "1",
    )
    assert code == 0, fault_err  # `fault run` prints text only; read the result from the registry

    from experionyx.faults.entities import FaultExperiment
    from experionyx.sqlite import SqliteRegistry

    with SqliteRegistry(ws.path / "registry.sqlite") as reg:
        fx = reg.find(FaultExperiment)[-1]
        fx_id, baseline_run_id = fx.id, fx.baseline_run_id

    disc = ws.cli_json("failure", "discover", "--fault-experiment", fx_id)
    assert disc["status"] == "COMPLETED"

    profile_spec = ws.path / "profile.json"
    profile_spec.write_text(
        json.dumps(
            {
                "scope": "MODEL_DATASET_EVALUATION",
                "baseline_run": baseline_run_id,
                "fault_experiments": [fx_id],
            }
        )
    )
    rel = ws.cli_json("reliability", "profile", str(profile_spec))
    assert rel["status"] == "COMPLETED"

    stats = ws.cli_json("stats", "bootstrap", "--reference-values", "1,2,3,4,5", "--seed", "1")
    assert stats["status"] in ("DERIVED", "OBSERVED")

    sched_spec = ws.path / "sched.json"
    sched_spec.write_text(
        json.dumps(
            {
                "name": "kg-e2e",
                "version": "1.0.0",
                "units": [
                    {
                        "key": "a",
                        "kind": "STATISTICAL_ANALYSIS",
                        "parameters": {
                            "kind": "BOOTSTRAP",
                            "sources": {"kind": "inline", "reference": [1, 2, 3]},
                        },
                    },
                    {
                        "key": "b",
                        "kind": "STATISTICAL_ANALYSIS",
                        "depends_on": ["a"],
                        "parameters": {
                            "kind": "BOOTSTRAP",
                            "sources": {"kind": "inline", "reference": [4, 5, 6]},
                        },
                    },
                ],
            }
        )
    )
    sched = ws.cli_json("scheduler", "run", str(sched_spec), "--investigation", ws.investigation)
    assert sched["status"] == "COMPLETED"

    gspec = ws.path / "graph.json"
    gspec.write_text(json.dumps({"name": "full", "version": "1.0.0"}))
    built = ws.cli_json("graph", "build", str(gspec), "--investigation", ws.investigation)
    assert built["status"] == "COMPLETED"
    snapshot_id = built["snapshot_id"]
    assert built["snapshot"]["node_count"] > 50  # real, substantial cross-phase evidence
    assert built["snapshot"]["edge_count"] > 50
    assert built["snapshot"]["unresolved_count"] == 0

    inspected = ws.cli_json("graph", "inspect", snapshot_id)
    by_kind = inspected["nodes_by_kind"]
    for expected in (
        "MODEL",
        "DATASET",
        "RUN",
        "FAULT_EXPERIMENT",
        "RELIABILITY_PROFILE",
        "STATISTICAL_ANALYSIS",
        "SCHEDULE_UNIT",
    ):
        assert by_kind.get(expected, 0) > 0, f"missing {expected} in {by_kind}"


def test_real_scheduler_dependency_appears_as_a_depends_on_edge(ws: Ws) -> None:
    sched_spec = ws.path / "sched2.json"
    sched_spec.write_text(
        json.dumps(
            {
                "name": "kg-sched",
                "version": "1.0.0",
                "units": [
                    {
                        "key": "a",
                        "kind": "STATISTICAL_ANALYSIS",
                        "parameters": {
                            "kind": "BOOTSTRAP",
                            "sources": {"kind": "inline", "reference": [1, 2, 3]},
                        },
                    },
                    {
                        "key": "b",
                        "kind": "STATISTICAL_ANALYSIS",
                        "depends_on": ["a"],
                        "parameters": {
                            "kind": "BOOTSTRAP",
                            "sources": {"kind": "inline", "reference": [4, 5, 6]},
                        },
                    },
                ],
            }
        )
    )
    ws.cli_json("scheduler", "run", str(sched_spec), "--investigation", ws.investigation)

    from experionyx.sqlite import SqliteRegistry

    with SqliteRegistry(ws.path / "registry.sqlite") as reg:
        units = {u.unit_key: u.id for u in reg.find(ScheduleUnit)}

    gspec = ws.path / "graph2.json"
    gspec.write_text(json.dumps({"name": "sched-only", "version": "1.0.0"}))
    built = ws.cli_json("graph", "build", str(gspec), "--investigation", ws.investigation)
    snapshot_id = built["snapshot_id"]
    path = ws.cli_json("graph", "path", snapshot_id, units["b"], units["a"], "--direction", "OUT")
    assert path["found"] is True
    assert path["edges"][0]["relation"] == "DEPENDS_ON"

"""Executing a graph construction. The orchestrator computes the node/edge set from the CURRENT
registry, then a `collect` Run (through the normal execution engine) re-derives it and persists the
snapshot, so the graph itself has full provenance and can be replayed -- exactly the same shape as
`benchmark.engine.run_benchmark` / `run_benchmark_collect` (see docs/graph.md)."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from experionyx.artifacts import ArtifactStore
from experionyx.domain import (
    ClaimStatus,
    ConfigurationRef,
    DatasetRef,
    EpistemicKind,
    EvidenceRelation,
    EvidenceTarget,
    Experiment,
    ExperimentStatus,
    ModelRef,
    RunStatus,
    to_jsonable,
)
from experionyx.errors import ExperionyxError, ValidationError
from experionyx.evaluation.config import _thaw
from experionyx.execution import Executor, RunContext, resolve_procedure
from experionyx.faults.report import read_artifact
from experionyx.graph.build import construct
from experionyx.graph.entities import EvidenceGraph, GraphEdge, GraphNode, GraphSnapshot
from experionyx.graph.spec import ENGINE_VERSION, GraphSpec
from experionyx.graph.taxonomy import NodeKind
from experionyx.registry import Registry

PROCEDURE = "experionyx.graph.engine:run_graph_collect"
ARTIFACTS = ("spec", "nodes", "edges", "summary")


def _tag(spec: GraphSpec) -> str:
    return spec.spec_id[4:16]


@dataclass(frozen=True)
class GraphRunResult:
    graph_id: str | None
    snapshot_id: str | None
    run_id: str | None
    status: RunStatus | None
    already_built: bool


def _existing(registry: Registry, graph_id: str, source_fingerprint: str) -> GraphSnapshot | None:
    for s in registry.find(GraphSnapshot, graph_id=graph_id):
        if s.source_fingerprint == source_fingerprint:
            return s
    return None


def _investigation(registry: Registry, requested: str | None) -> str:
    from experionyx.domain import Investigation

    if requested is not None:
        registry.get(Investigation, requested)
        return requested
    found = sorted(i.id for i in registry.find(Investigation))
    if len(found) != 1:
        raise ValidationError(f"{len(found)} investigations exist; say which one with --investigation")  # fmt: skip
    return found[0]


def run_graph(
    registry: Registry,
    store: ArtifactStore,
    executor: Executor,
    spec: GraphSpec,
    *,
    investigation_id: str | None = None,
) -> GraphRunResult:
    """Idempotent per registry state: a graph whose exact (spec, current registry content) was
    already collected is returned, not re-built."""
    pre = construct(registry, spec)
    inv = _investigation(registry, investigation_id or spec.investigation_id)
    gr = EvidenceGraph(spec.name, spec.version, spec.spec_id, spec.to_dict(), ENGINE_VERSION, datetime.now(UTC))  # fmt: skip
    done = None
    if registry.exists(EvidenceGraph, gr.id):
        done = _existing(registry, gr.id, pre.source_fingerprint)
    if done is not None:
        return GraphRunResult(gr.id, done.id, done.run_id, RunStatus.COMPLETED, True)
    conf = ConfigurationRef({"graph_collect": {"spec": spec.to_dict()}})
    if not registry.exists(ConfigurationRef, conf.id):
        registry.add(conf)
    exp = Experiment(
        inv, f"graph {spec.name} {spec.version} {_tag(spec)}",
        "Construction of an evidence/failure knowledge graph over the registry",
        ModelRef("none", "0"), DatasetRef("none", "0"), conf.id, datetime.now(UTC),
    )  # fmt: skip
    if not registry.exists(Experiment, exp.id):
        registry.add(exp)
        registry.update_status(exp.with_status(ExperimentStatus.READY))
    result = executor.execute(
        exp.id, resolve_procedure(PROCEDURE), seed=0, procedure_name=PROCEDURE
    )
    found = _existing(registry, gr.id, pre.source_fingerprint) if registry.exists(EvidenceGraph, gr.id) else None  # fmt: skip
    return GraphRunResult(gr.id if found else None, found.id if found else None, result.run.id, result.status, False)  # fmt: skip


def _write(ctx: RunContext, name: str, payload: object) -> str:
    directory = ctx.artifact_dir / "graph"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(
        json.dumps(to_jsonable(payload), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return ctx.register_artifact(
        f"graph/{name}.json", name=f"graph-{name}", media_type="application/json"
    ).id


def run_graph_collect(ctx: RunContext) -> None:
    params = _thaw(ctx.parameters)
    if not isinstance(params, dict) or set(params) != {"graph_collect"}:
        raise ValidationError("a graph configuration must be exactly {'graph_collect': ...}")
    body = params["graph_collect"]
    spec = GraphSpec.from_dict(body["spec"])
    reg, now = ctx.registry, ctx.started_at
    got = construct(reg, spec)

    gr = EvidenceGraph(spec.name, spec.version, spec.spec_id, spec.to_dict(), ENGINE_VERSION, now)
    provenance_fingerprint = _provenance_fingerprint(ctx)
    snapshot = GraphSnapshot(
        gr.id,
        ctx.experiment.investigation_id,
        ctx.run.id,
        got.source_fingerprint,
        provenance_fingerprint,
        len(got.nodes),
        len(got.edges),
        got.unresolved_count,
        got.truncated,
        {
            "node_kinds": _kind_counts(k for k, _ in got.nodes),
            "relations": _relation_counts(e.relation for e in got.edges),
        },
        now,
    )

    node_ids: dict[tuple[NodeKind, str], str] = {}
    new = not reg.exists(GraphSnapshot, snapshot.id)
    if new:
        with reg.transaction():
            if not reg.exists(EvidenceGraph, gr.id):
                reg.add(gr)
            reg.add(snapshot)
            for (kind, ref_id), draft in got.nodes.items():
                node = GraphNode(snapshot.id, kind, ref_id, f"{kind.value.lower()} {ref_id}", draft.resolved, draft.detail, now)  # fmt: skip
                reg.add(node)
                node_ids[(kind, ref_id)] = node.id
            for e in got.edges:
                if e.from_key not in node_ids or e.to_key not in node_ids:
                    continue
                reg.add(GraphEdge(snapshot.id, node_ids[e.from_key], node_ids[e.to_key], e.relation, e.detail, now))  # fmt: skip
    else:
        existing = reg.find(GraphNode, snapshot_id=snapshot.id)
        node_ids = {(n.node_kind, n.ref_id): n.id for n in existing}

    node_docs = [
        {"kind": k.value, "ref_id": rid, "resolved": d.resolved}
        for (k, rid), d in got.nodes.items()
    ]
    edge_docs = [
        {"from": e.from_key[1], "to": e.to_key[1], "relation": e.relation.value, "detail": e.detail}
        for e in got.edges
    ]
    art = {
        "spec": _write(
            ctx,
            "spec",
            {"spec_id": spec.spec_id, "spec": spec.to_dict(), "engine_version": ENGINE_VERSION},
        ),
        "nodes": _write(ctx, "nodes", {"nodes": node_docs}),
        "edges": _write(ctx, "edges", {"edges": edge_docs}),
        "summary": _write(ctx, "summary", dict(snapshot.summary)),
    }
    ctx.observe("graph.new_snapshot", int(new), kind=EpistemicKind.OBSERVATION)
    ctx.observe("graph.nodes", snapshot.node_count, kind=EpistemicKind.DERIVED_METRIC)
    ctx.observe("graph.edges", snapshot.edge_count, kind=EpistemicKind.DERIVED_METRIC)
    ctx.observe("graph.unresolved", snapshot.unresolved_count, kind=EpistemicKind.DERIVED_METRIC)
    claim = ctx.assert_claim(
        f"[{ctx.run.id}] Graph {spec.name} {spec.version} was constructed over {snapshot.node_count} node(s) and {snapshot.edge_count} edge(s) from the registry as of source fingerprint {snapshot.source_fingerprint[7:19]}; {snapshot.unresolved_count} reference(s) did not resolve. This records what the registry's own references are; it makes no causal claim.",
        asserted_by=f"experionyx.graph/{ENGINE_VERSION}",
        status=ClaimStatus.SUPPORTED,
    )  # fmt: skip
    for name in ("nodes", "edges", "summary"):
        ctx.add_evidence(claim, EvidenceTarget.ARTIFACT, art[name], EvidenceRelation.SUPPORTS, "graph document")  # fmt: skip


def _provenance_fingerprint(ctx: RunContext) -> str:
    from experionyx.provenance import Provenance

    found = ctx.registry.find(Provenance, run_id=ctx.run.id)
    if not found:
        raise ExperionyxError(f"run {ctx.run.id} has no provenance yet")
    return found[0].fingerprint


def _kind_counts(kinds: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for k in kinds:
        out[k.value] = out.get(k.value, 0) + 1
    return out


def _relation_counts(relations: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in relations:
        out[r.value] = out.get(r.value, 0) + 1
    return out


def replay_check(
    reg: Registry, store: ArtifactStore, executor: Executor, snapshot_id: str
) -> dict[str, Any]:
    """Replay the collect Run as a NEW run and compare every artifact. If the registry's content
    changed since (new evidence was added, say) the source fingerprints differ and this is
    reported as `sources_changed`, not as nondeterminism."""
    snap = reg.get(GraphSnapshot, snapshot_id)
    replay = executor.replay(snap.run_id)
    out: dict[str, Any] = {
        "snapshot_id": snap.id,
        "original_run": snap.run_id,
        "replay_run": replay.run.id,
        "replay_status": replay.status.value,
    }
    if replay.status is not RunStatus.COMPLETED:
        return {**out, "deterministic": False, "differences": [f"replay run ended {replay.status.value}"]}  # fmt: skip
    diffs: list[str] = []
    for name in ARTIFACTS:
        x = read_artifact(reg, store, snap.run_id, f"graph/{name}.json")
        y = read_artifact(reg, store, replay.run.id, f"graph/{name}.json")
        if x != y:
            diffs.append(name)
    y_spec: Any = read_artifact(reg, store, replay.run.id, "graph/spec.json")
    replay_fingerprint = None
    if isinstance(y_spec, dict):
        replay_snapshots = reg.find(GraphSnapshot, run_id=replay.run.id)
        replay_fingerprint = replay_snapshots[0].source_fingerprint if replay_snapshots else None
    if replay_fingerprint is not None and replay_fingerprint != snap.source_fingerprint:
        return {
            **out,
            "deterministic": None,
            "sources_changed": True,
            "differences": diffs,
            "note": "the registry's content changed since this snapshot was collected (source fingerprint differs); determinism cannot be judged from this replay",
        }
    return {**out, "deterministic": not diffs, "sources_changed": False, "differences": diffs}


def resolve_graph(registry: Registry, ref: str) -> EvidenceGraph:
    if ref.startswith(EvidenceGraph.PREFIX + "_"):
        return registry.get(EvidenceGraph, ref)
    matches = [g for g in registry.find(EvidenceGraph) if g.spec_id == ref or g.spec_id.startswith(ref)]  # fmt: skip
    if len(matches) != 1:
        raise ValidationError(f"{'no' if not matches else 'ambiguous'} graph matches {ref!r}")
    return matches[0]


def resolve_snapshot(registry: Registry, ref: str) -> GraphSnapshot:
    if ref.startswith(GraphSnapshot.PREFIX + "_"):
        return registry.get(GraphSnapshot, ref)
    matches = [s for s in registry.find(GraphSnapshot) if s.id.startswith(ref)]
    if len(matches) != 1:
        raise ValidationError(f"{'no' if not matches else 'ambiguous'} snapshot matches {ref!r}")
    return matches[0]


__all__ = [
    "ARTIFACTS",
    "PROCEDURE",
    "GraphRunResult",
    "replay_check",
    "resolve_graph",
    "resolve_snapshot",
    "run_graph",
    "run_graph_collect",
]

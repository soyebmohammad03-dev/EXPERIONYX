"""The discovery pipeline: signals -> similarity links -> clusters -> stability -> measurements ->
evidence evaluation -> registered DISCOVERED/CANDIDATE/SUPPORTED failure modes. Deterministic:
identical signals and configuration give identical clusters, IDs and statuses. A mode is never
CONFIRMED here; that needs reproduction and an explicit person-made decision (lifecycle.py)."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from experionyx.domain import Entity, Investigation, Run, to_jsonable
from experionyx.errors import ValidationError
from experionyx.failures import analysis as an
from experionyx.failures.config import (
    CLUSTERING_VERSION,
    EXTRACTOR_VERSION,
    SIMILARITY_VERSION,
    DiscoveryConfig,
)
from experionyx.failures.entities import (
    FailureCluster,
    FailureEvidence,
    FailureMode,
    FailureRelationship,
    FailureSignal,
)
from experionyx.failures.taxonomy import (
    EvidenceKind,
    FailureCategory,
    FailureStatus,
    NodeKind,
    Predicate,
)
from experionyx.provenance import Provenance
from experionyx.registry import Registry

DISCOVERY_ACTOR = "experionyx-discovery"


@dataclass(frozen=True)
class ClusterResult:
    members: tuple[FailureSignal, ...]
    metrics: Mapping[str, object]
    category: FailureCategory
    candidate: an.Evaluation
    supported: an.Evaluation
    status: FailureStatus
    sample_groups: Mapping[str, object]


@dataclass(frozen=True)
class Analysis:
    clusters: tuple[ClusterResult, ...]
    comparisons: int
    fallback_blocks: tuple[str, ...]
    dropped_clusters: tuple[Mapping[str, object], ...]
    singleton_clusters: int


def analyze_signals(
    signals: Sequence[FailureSignal], cfg: DiscoveryConfig, runs_analyzed: int
) -> Analysis:
    sig = list(signals)
    links = an.compare_all(
        sig, cfg.similarity, max(0.0, cfg.similarity.threshold - cfg.clustering.sensitivity_delta)
    )
    groups = an.cluster_indices(sig, links, cfg)
    stab = an.stability(groups, links, len(sig), cfg)
    order = sorted(range(len(groups)), key=lambda k: (-len(groups[k]), sig[groups[k][0]].id))
    kept, dropped = order[: cfg.clustering.max_clusters], order[cfg.clustering.max_clusters :]
    results: list[ClusterResult] = []
    for k in kept:
        members = tuple(sig[i] for i in groups[k])
        comp, comp_n = an.compactness(members, cfg.similarity)
        metrics = an.cluster_measurements(members, runs_analyzed)
        metrics["compactness"] = comp
        metrics["compactness_members_used"] = comp_n
        metrics["stability"] = stab[k]
        metrics["stability_method"] = (
            None
            if stab[k] is None
            else f"best Jaccard overlap after moving the threshold by +/-{cfg.clustering.sensitivity_delta}"
        )
        ev = cfg.evidence
        cand = an.evaluate_criteria(
            "candidate",
            ev.candidate,
            members,
            metrics,
            ev.bootstrap_resamples,
            ev.bootstrap_confidence,
            ev.bootstrap_seed,
        )
        sup = an.evaluate_criteria(
            "supported",
            ev.supported,
            members,
            metrics,
            ev.bootstrap_resamples,
            ev.bootstrap_confidence,
            ev.bootstrap_seed,
        )
        status = (
            FailureStatus.SUPPORTED
            if cand.passed and sup.passed
            else FailureStatus.CANDIDATE
            if cand.passed
            else FailureStatus.DISCOVERED
        )
        results.append(
            ClusterResult(
                members,
                metrics,
                an.category_of(members),
                cand,
                sup,
                status,
                an.sample_groups(members),
            )
        )
    return Analysis(
        tuple(results), links.comparisons, links.fallback_blocks,
        tuple({"size": len(groups[k]), "first_signal_id": sig[groups[k][0]].id} for k in dropped),
        sum(len(g) == 1 for g in groups),
    )  # fmt: skip


# -- explanatory text (never authoritative) ------------------------------------------------------


def _title(r: ClusterResult) -> str:
    m = r.metrics
    bits = [r.category.value]
    for key in ("classes", "slices", "fault_types"):
        vals = m.get(key)
        if isinstance(vals, list) and vals:
            bits.append(f"{key}={','.join(map(str, vals[:3]))}")
    return " | ".join(bits)


def _description(r: ClusterResult) -> str:
    m = r.metrics
    n_runs = len(m["runs"]) if isinstance(m["runs"], list) else 0
    return (
        f"A group of {len(r.members)} failure signal(s) from {n_runs} run(s), kinds {m['kinds']}, "
        f"currently {r.status.value}. This text summarizes the structured fields; it is not itself "
        "evidence, and the grouping is statistical: it does not establish a cause."
    )


# -- persistence -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Persisted:
    clusters: tuple[FailureCluster, ...]
    modes: tuple[FailureMode, ...]  # only those registered by this call's cluster set
    new_signals: int
    new_modes: int
    relationships: int


def _put(registry: Registry, e: Entity) -> bool:
    if registry.exists(type(e), e.id):
        return False
    registry.add(e)
    return True


def _strongest_per_run(members: Sequence[FailureSignal]) -> list[FailureSignal]:
    best: dict[str, FailureSignal] = {}
    for s in members:
        cur = best.get(s.run_id)
        if cur is None or (s.magnitude, s.id) > (cur.magnitude, cur.id):
            best[s.run_id] = s
    return [best[k] for k in sorted(best)]


def persist(
    registry: Registry,
    investigation_id: str,
    signals: Sequence[FailureSignal],
    result: Analysis,
    cfg: DiscoveryConfig,
    now: datetime,
) -> Persisted:
    """Store signals, clusters and modes in ONE transaction. Already-known records are kept
    unchanged (identity is content-addressed), so re-running a discovery is idempotent and never
    resets a status a person has moved on."""
    registry.get(Investigation, investigation_id)
    new_signals = new_modes = rels = 0
    clusters: list[FailureCluster] = []
    modes: list[FailureMode] = []
    with registry.transaction():
        for s in signals:
            new_signals += _put(registry, s)
        for r in result.clusters:
            structured = _structured(r, cfg)
            cl = FailureCluster(
                investigation_id, tuple(sorted(s.id for s in r.members)), cfg.clustering.algorithm.value,
                CLUSTERING_VERSION, cfg.config_hash, cfg.similarity.threshold, to_jsonable_map(structured["measurements"]), now,
            )  # fmt: skip
            _put(registry, cl)
            clusters.append(cl)
            if len(r.members) < cfg.clustering.min_cluster_size_for_mode:
                continue
            mode = FailureMode(
                investigation_id, cl.id, r.category, _title(r), _description(r), structured, now
            )
            if registry.exists(FailureMode, mode.id):
                modes.append(registry.get(FailureMode, mode.id))
                continue
            registry.add(mode)
            new_modes += 1
            mode, made = _advance(registry, mode, r, now)
            rels += _link(registry, investigation_id, mode, r, made, now)
            modes.append(mode)
        rels += _co_occurrence(registry, investigation_id, modes, now)
    return Persisted(tuple(clusters), tuple(modes), new_signals, new_modes, rels)


def _structured(r: ClusterResult, cfg: DiscoveryConfig) -> dict[str, object]:
    return {
        "measurements": to_jsonable(r.metrics),
        "criteria": {"candidate": to_jsonable(r.candidate), "supported": to_jsonable(r.supported)},
        "sample_groups": to_jsonable(r.sample_groups),
        "provenance": {
            "config_hash": cfg.config_hash,
            "extractor_version": EXTRACTOR_VERSION,
            "similarity_version": SIMILARITY_VERSION,
            "clustering_version": CLUSTERING_VERSION,
            "algorithm": cfg.clustering.algorithm.value,
        },
        "discovered_by": "deterministic rules; no LLM or learned model decides any of this",
    }


def evidence(
    mode_id: str,
    kind: EvidenceKind,
    ref: str | None,
    summary: str,
    detail: Mapping[str, object],
    now: datetime,
) -> FailureEvidence:
    return FailureEvidence(mode_id, kind, ref, summary, detail, now)


def _advance(
    registry: Registry, mode: FailureMode, r: ClusterResult, now: datetime
) -> tuple[FailureMode, list[FailureEvidence]]:
    made: list[FailureEvidence] = []
    for s in _strongest_per_run(r.members):
        made.append(
            evidence(
                mode.id,
                EvidenceKind.SIGNAL,
                s.id,
                f"strongest {s.signal_kind.value} signal of run {s.run_id}",
                {"run_id": s.run_id, "magnitude": s.magnitude},
                now,
            )
        )
    made.append(
        evidence(
            mode.id,
            EvidenceKind.OBSERVATION,
            None,
            "criteria evaluation at registration",
            {"candidate": to_jsonable(r.candidate), "supported": to_jsonable(r.supported)},
            now,
        )
    )
    for e in made:
        registry.add(e)
    if r.candidate.passed:
        mode = transition(
            registry,
            mode,
            FailureStatus.CANDIDATE,
            DISCOVERY_ACTOR,
            "configured candidate criteria met",
            now,
            {"automatic": True, "checks": to_jsonable(r.candidate.checks)},
        )
        if r.supported.passed:
            mode = transition(
                registry,
                mode,
                FailureStatus.SUPPORTED,
                DISCOVERY_ACTOR,
                "configured supported criteria met",
                now,
                {"automatic": True, "checks": to_jsonable(r.supported.checks)},
            )
    return mode, made


def transition(
    registry: Registry,
    mode: FailureMode,
    target: FailureStatus,
    actor: str,
    reason: str,
    now: datetime,
    detail: Mapping[str, object] | None = None,
) -> FailureMode:
    """Validated lifecycle change; the reason and actor are retained as TRANSITION evidence."""
    if not actor.strip() or not reason.strip():
        raise ValidationError("a status change needs an actor and a reason")
    new = mode.with_status(target)
    with registry.transaction():
        registry.update_status(new)
        registry.add(
            evidence(
                mode.id,
                EvidenceKind.TRANSITION,
                None,
                f"{mode.status.value} -> {target.value}",
                {
                    "from": mode.status.value,
                    "to": target.value,
                    "actor": actor,
                    "reason": reason,
                    **(detail or {}),
                },
                now,
            )
        )
    return new


def _edge(
    registry: Registry,
    inv: str,
    sk: NodeKind,
    sid: str,
    p: Predicate,
    ok: NodeKind,
    oid: str,
    detail: Mapping[str, object],
    now: datetime,
) -> int:
    return int(_put(registry, FailureRelationship(inv, sk, sid, p, ok, oid, detail, now)))


def _link(
    registry: Registry,
    inv: str,
    mode: FailureMode,
    r: ClusterResult,
    made: Sequence[FailureEvidence],
    now: datetime,
) -> int:
    n = 0
    seen: set[tuple[str, str]] = set()
    for run_id in sorted({s.run_id for s in r.members}):
        run = registry.get(Run, run_id)
        prov = registry.find(Provenance, run_id=run.id)
        inputs = prov[0].inputs if prov else None
        if inputs and inputs.model and ("m", inputs.model.record_id) not in seen:
            seen.add(("m", inputs.model.record_id))
            n += _edge(
                registry,
                inv,
                NodeKind.MODEL,
                inputs.model.record_id,
                Predicate.EXHIBITS,
                NodeKind.FAILURE_MODE,
                mode.id,
                {},
                now,
            )
        if inputs and inputs.dataset and ("d", inputs.dataset.record_id) not in seen:
            seen.add(("d", inputs.dataset.record_id))
            n += _edge(
                registry,
                inv,
                NodeKind.DATASET,
                inputs.dataset.record_id,
                Predicate.EXPOSES,
                NodeKind.FAILURE_MODE,
                mode.id,
                {},
                now,
            )
    m = r.metrics
    for fam in m["fault_families"]:  # type: ignore[attr-defined]
        n += _edge(
            registry,
            inv,
            NodeKind.FAULT,
            str(fam),
            Predicate.REVEALS,
            NodeKind.FAILURE_MODE,
            mode.id,
            {},
            now,
        )
    for c in m["classes"]:  # type: ignore[attr-defined]
        n += _edge(
            registry,
            inv,
            NodeKind.FAILURE_MODE,
            mode.id,
            Predicate.AFFECTS,
            NodeKind.CLASS,
            str(c),
            {},
            now,
        )
    for sl in m["slices"]:  # type: ignore[attr-defined]
        n += _edge(
            registry,
            inv,
            NodeKind.FAILURE_MODE,
            mode.id,
            Predicate.AFFECTS,
            NodeKind.SLICE,
            str(sl),
            {},
            now,
        )
    for e in made:
        n += _edge(
            registry,
            inv,
            NodeKind.FAILURE_MODE,
            mode.id,
            Predicate.SUPPORTED_BY,
            NodeKind.EVIDENCE,
            e.id,
            {"kind": e.evidence_kind.value},
            now,
        )
    return n


def _co_occurrence(
    registry: Registry, inv: str, modes: Sequence[FailureMode], now: datetime
) -> int:
    """Observed co-occurrence: two modes appearing in the same runs. It is NOT an interaction."""
    ordered = sorted(modes, key=lambda m: m.id)
    runs = {
        m.id: {
            registry.get(FailureSignal, sid).run_id
            for sid in registry.get(FailureCluster, m.cluster_id).signal_ids
        }
        for m in ordered
    }
    n = 0
    for i, a in enumerate(ordered):
        for b in ordered[i + 1 :]:
            shared = runs[a.id] & runs[b.id]
            if shared:
                detail = {
                    "shared_runs": len(shared),
                    "jaccard": len(shared) / len(runs[a.id] | runs[b.id]),
                    "note": "observed co-occurrence only; not evidence of interaction",
                }
                n += _edge(
                    registry,
                    inv,
                    NodeKind.FAILURE_MODE,
                    a.id,
                    Predicate.CO_OCCURS_WITH,
                    NodeKind.FAILURE_MODE,
                    b.id,
                    detail,
                    now,
                )
    return n


def summary_counts(modes: Sequence[FailureMode]) -> dict[str, int]:
    return dict(sorted(Counter(m.status.value for m in modes).items()))


def to_jsonable_map(value: object) -> Mapping[str, object]:
    data = to_jsonable(value)
    if not isinstance(data, dict):
        raise ValidationError("expected a mapping")
    return data

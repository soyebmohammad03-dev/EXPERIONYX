"""The discovery procedure, run by the normal execution engine so that every discovery is a Run
with provenance, artifacts, observations, claims and evidence, and can itself be replayed."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

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
from experionyx.errors import ValidationError
from experionyx.evaluation.config import _thaw
from experionyx.execution import Executor, RunContext, resolve_procedure
from experionyx.failures.config import (
    CLUSTERING_VERSION,
    EXTRACTOR_VERSION,
    SIMILARITY_VERSION,
    DiscoveryConfig,
)
from experionyx.failures.discovery import analyze_signals, persist, summary_counts
from experionyx.failures.entities import FailureMode
from experionyx.failures.sources import collect_signals
from experionyx.failures.taxonomy import FailureStatus
from experionyx.registry import Registry

PROCEDURE = "experionyx.failures.engine:run_failure_discovery"
FAILURES_VERSION = "1.0.0"
CONCLUSIONS = {
    "can_conclude": "Which stored failure signals are similar under the configured rules, how the runs group, and whether each group meets the configured evidence requirements.",
    "cannot_conclude": "Why a failure happens (no causal claim), that a group is a real failure mode of the model in general, or that it will recur on other data. A CONFIRMED status needs a passing reproduction check and an explicit decision by a person.",
    "interpretation_note": "Titles and descriptions are explanatory templates; the structured fields are authoritative.",
}


@dataclass(frozen=True)
class DiscoveryRequest:
    investigation_id: str
    run_ids: tuple[str, ...]
    fault_experiment_ids: tuple[str, ...]
    config: Mapping[str, object]

    def to_parameters(self) -> dict[str, object]:
        return {
            "failure_discovery": {
                "investigation_id": self.investigation_id,
                "run_ids": list(self.run_ids),
                "fault_experiment_ids": list(self.fault_experiment_ids),
                "config": dict(self.config),
            }
        }

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, object]) -> "DiscoveryRequest":
        if set(parameters) != {"failure_discovery"}:
            raise ValidationError(
                "a discovery configuration must be exactly {'failure_discovery': ...}"
            )
        body = _thaw(parameters["failure_discovery"])
        if not isinstance(body, dict) or set(body) != {
            "investigation_id",
            "run_ids",
            "fault_experiment_ids",
            "config",
        }:
            raise ValidationError("'failure_discovery' has the wrong fields")
        ids = [body["run_ids"], body["fault_experiment_ids"]]
        if (
            not isinstance(body["investigation_id"], str)
            or not isinstance(body["config"], dict)
            or not all(isinstance(x, list) and all(isinstance(i, str) for i in x) for x in ids)
        ):
            raise ValidationError("'failure_discovery' has invalid field types")
        return cls(
            body["investigation_id"],
            tuple(body["run_ids"]),
            tuple(body["fault_experiment_ids"]),
            body["config"],
        )


def _write(ctx: RunContext, name: str, payload: object) -> str:
    (ctx.artifact_dir / f"{name}.json").write_text(
        json.dumps(to_jsonable(payload), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return ctx.register_artifact(f"{name}.json", name=name, media_type="application/json").id


def run_failure_discovery(ctx: RunContext) -> None:
    req = DiscoveryRequest.from_parameters(ctx.parameters)
    cfg = DiscoveryConfig.from_dict(req.config)
    now = ctx.started_at
    collected = collect_signals(
        ctx.registry,
        ctx.store,
        req.investigation_id,
        req.run_ids,
        req.fault_experiment_ids,
        cfg,
        now,
    )
    analysis = analyze_signals(collected.signals, cfg, len(collected.runs_analyzed))
    persisted = persist(ctx.registry, req.investigation_id, collected.signals, analysis, cfg, now)
    modes = list(persisted.modes)
    ids = {
        "signals": _write(
            ctx,
            "failure-signals",
            {
                "config_hash": cfg.config_hash,
                "extractor_version": EXTRACTOR_VERSION,
                "count": len(collected.signals),
                "signals": [
                    s.to_dict() | {"id": s.id, "truncated": s.truncated} for s in collected.signals
                ],
            },
        ),
        "clusters": _write(
            ctx,
            "failure-clusters",
            {
                "algorithm": cfg.clustering.algorithm,
                "algorithm_version": CLUSTERING_VERSION,
                "similarity_version": SIMILARITY_VERSION,
                "threshold": cfg.similarity.threshold,
                "pairwise_comparisons": analysis.comparisons,
                "fallback_blocks": analysis.fallback_blocks,
                "dropped_clusters": analysis.dropped_clusters,
                "singleton_clusters": analysis.singleton_clusters,
                "clusters": [
                    {"id": c.id, "signal_ids": c.signal_ids, "metrics": c.metrics}
                    for c in persisted.clusters
                ],
            },
        ),
        "candidates": _write(
            ctx,
            "failure-candidates",
            {
                "modes": [
                    {
                        "id": m.id,
                        "cluster_id": m.cluster_id,
                        "status": m.status,
                        "category": m.category,
                        "title": m.title,
                        "description": m.description,
                        "structured": m.structured,
                    }
                    for m in modes
                ]
            },
        ),
    }
    status_counts = summary_counts(modes)
    _write(ctx, "failure-analysis", {
        "config": cfg.to_dict(), "config_hash": cfg.config_hash, "failures_version": FAILURES_VERSION,
        "investigation_id": req.investigation_id, "sources": collected.sources, "skipped": collected.skipped,
        "runs_analyzed": len(collected.runs_analyzed),
        "counts": {"signals": len(collected.signals), "new_signals": persisted.new_signals, "clusters": len(persisted.clusters), "modes": len(modes), "new_modes": persisted.new_modes, "relationships_created": persisted.relationships, "status": status_counts},
        "limits": {"max_source_runs": cfg.max_source_runs, "max_signals": cfg.extraction.max_signals, "max_pairwise_comparisons": cfg.similarity.max_pairwise_comparisons, "pairwise_comparisons_made": analysis.comparisons, "blocks_compared_by_exact_signature_only": analysis.fallback_blocks, "max_clusters": cfg.clustering.max_clusters, "clusters_dropped_over_limit": analysis.dropped_clusters, "signals_with_truncated_sample_ids": sum(s.truncated for s in collected.signals)},
        "determinism": "no randomness except a seeded bootstrap (seed recorded in the config); identical stored results and configuration give identical signals, clusters, IDs and statuses",
        **CONCLUSIONS,
    })  # fmt: skip
    ctx.observe("failures.signals", len(collected.signals), kind=EpistemicKind.DERIVED_METRIC)
    ctx.observe("failures.clusters", len(persisted.clusters), kind=EpistemicKind.DERIVED_METRIC)
    ctx.observe("failures.modes", len(modes), kind=EpistemicKind.DERIVED_METRIC)
    for status, n in status_counts.items():
        ctx.observe(f"failures.modes.{status.lower()}", n, kind=EpistemicKind.DERIVED_METRIC)
    _claims(ctx, modes, ids["candidates"])


def _claims(ctx: RunContext, modes: Sequence[FailureMode], candidates_artifact: str) -> None:
    for m in modes:
        if m.status not in (FailureStatus.CANDIDATE, FailureStatus.SUPPORTED):
            continue
        meas = m.structured["measurements"]
        assert isinstance(meas, Mapping)  # noqa: S101
        runs = meas["runs"]
        assert isinstance(runs, list | tuple)  # noqa: S101
        statement = (
            f"[{ctx.run.id}] Failure signals of kinds {list(meas['kinds'])} group into {m.id} ({m.category.value}), observed in {len(runs)} run(s) "
            f"and currently {m.status.value} under the configured evidence requirements. This is a statistical grouping of measured signals; it makes no causal claim."
        )
        claim = ctx.assert_claim(
            statement,
            asserted_by=f"experionyx.failures/{FAILURES_VERSION}",
            status=ClaimStatus.SUPPORTED
            if m.status is FailureStatus.SUPPORTED
            else ClaimStatus.INCONCLUSIVE,
        )
        for r in runs:
            ctx.add_evidence(
                claim,
                EvidenceTarget.RUN,
                str(r),
                EvidenceRelation.SUPPORTS,
                "run containing a signal of this group",
            )
        ctx.add_evidence(
            claim,
            EvidenceTarget.ARTIFACT,
            candidates_artifact,
            EvidenceRelation.SUPPORTS,
            "structured mode record",
        )


# -- launcher ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscoveryRunResult:
    experiment_id: str
    run_id: str
    status: RunStatus


def run_discovery(
    registry: Registry,
    store: ArtifactStore,
    executor: Executor,
    investigation_id: str,
    run_ids: Sequence[str],
    fault_experiment_ids: Sequence[str],
    cfg: DiscoveryConfig | None = None,
    *,
    seed: int = 0,
) -> DiscoveryRunResult:
    """Run one discovery as a Run. The experiment is a placeholder-bound, model-free one: it
    references no model or dataset (their identities live in the signals' lineage)."""
    cfg = cfg or DiscoveryConfig()
    req = DiscoveryRequest(
        investigation_id,
        tuple(sorted(set(run_ids))),
        tuple(sorted(set(fault_experiment_ids))),
        cfg.to_dict(),
    )
    conf = ConfigurationRef(req.to_parameters())
    if not registry.exists(ConfigurationRef, conf.id):
        registry.add(conf)
    exp = Experiment(
        investigation_id,
        "failure discovery",
        "Deterministic failure discovery over stored results",
        ModelRef("none", "0"),
        DatasetRef("none", "0"),
        conf.id,
        datetime.now(UTC),
    )
    if not registry.exists(Experiment, exp.id):
        registry.add(exp)
        registry.update_status(exp.with_status(ExperimentStatus.READY))
    result = executor.execute(
        exp.id, resolve_procedure(PROCEDURE), seed=seed, procedure_name=PROCEDURE
    )
    return DiscoveryRunResult(exp.id, result.run.id, result.status)

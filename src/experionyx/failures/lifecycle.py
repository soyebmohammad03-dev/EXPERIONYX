"""Lifecycle operations on registered failure modes: reproduction, explicit confirmation and
status changes. Every change keeps its actor and reason as evidence; nothing here confirms a
mode on its own."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from experionyx.artifacts import ArtifactStore
from experionyx.domain import RunStatus, to_jsonable
from experionyx.errors import FailureError, ValidationError
from experionyx.evaluation.loading import load_evaluation
from experionyx.execution import Executor
from experionyx.failures.config import DiscoveryConfig
from experionyx.failures.discovery import evidence, transition
from experionyx.failures.entities import (
    FailureCluster,
    FailureEvidence,
    FailureMode,
    FailureRelationship,
    FailureSignal,
)
from experionyx.failures.extraction import (
    FaultContext,
    SignalBuilder,
    extract_from_evaluation,
    extract_from_fault,
)
from experionyx.failures.sources import load_errors
from experionyx.failures.taxonomy import (
    EvidenceKind,
    FailureStatus,
    NodeKind,
    Predicate,
)
from experionyx.faults.analysis import default_primary_metric
from experionyx.faults.entities import FaultExperiment
from experionyx.provenance import Provenance
from experionyx.registry import Registry


def match_key(s: FailureSignal) -> tuple[object, ...]:
    """What makes a reproduced signal 'the same finding': same kind, class, prediction, slice and
    fault type (the magnitude is compared separately, within a tolerance)."""
    d = s.detail
    fault = d.get("fault")
    ftype = fault.get("type") if isinstance(fault, Mapping) else None
    return (s.signal_kind, d.get("class_label"), d.get("predicted_label"), d.get("slice"), ftype)


def _fault_context(d: Mapping[str, object]) -> FaultContext:
    p = d.get("parameters")
    frac = d.get("affected_fraction")
    pv = d.get("parameter_value")
    src = d.get("source_dataset_fingerprint")
    return FaultContext(
        type=str(d["type"]), family_id=str(d["family_id"]), fault_id=str(d["fault_id"]), seed=int(str(d["seed"])),
        target=str(d["target"]), parameters=dict(p) if isinstance(p, Mapping) else {},
        parameter_value=float(pv) if isinstance(pv, int | float) else None,
        affected_fraction=float(frac) if isinstance(frac, int | float) else None,
        fault_experiment_id=str(d["fault_experiment_id"]), baseline_run_id=str(d["baseline_run_id"]),
        source_dataset_fingerprint=str(src) if src else None,
    )  # fmt: skip


@dataclass(frozen=True)
class Reproduction:
    mode_id: str
    attempted: int
    passed_count: int
    passed: bool
    records: tuple[Mapping[str, object], ...]
    evidence_ids: tuple[str, ...]
    runs_not_replayed: int  # supporting runs beyond max_runs (bounded, recorded)


def reproduce(
    registry: Registry,
    store: ArtifactStore,
    executor: Executor,
    mode_id: str,
    cfg: DiscoveryConfig,
    now: datetime | None = None,
) -> Reproduction:
    """Replay up to `max_runs` supporting runs as NEW runs, re-extract signals with the ORIGINAL
    extraction configuration, and compare each reproduced magnitude with the original within the
    configured tolerance. Records original result, reproduction result, tolerance, pass/fail and
    provenance as evidence. It never changes the mode's status."""
    now = now or datetime.now(UTC)
    mode = registry.get(FailureMode, mode_id)
    recorded = mode.structured["provenance"]
    assert isinstance(recorded, Mapping)  # noqa: S101
    if recorded.get("config_hash") != cfg.config_hash:
        raise ValidationError(
            "reproduction must use the configuration the mode was discovered with (config hash differs)"
        )
    rc = cfg.reproduction
    signals = [
        registry.get(FailureSignal, i)
        for i in registry.get(FailureCluster, mode.cluster_id).signal_ids
    ]
    best: dict[str, FailureSignal] = {}
    for s in signals:
        if s.run_id not in best or (s.magnitude, s.id) > (
            best[s.run_id].magnitude,
            best[s.run_id].id,
        ):
            best[s.run_id] = s
    chosen = sorted(best.values(), key=lambda s: (-s.magnitude, s.run_id))[: rc.max_runs]
    records: list[dict[str, object]] = []
    made: list[FailureEvidence] = []
    for orig in sorted(chosen, key=lambda s: s.run_id):
        tol = {
            "absolute": rc.magnitude_tolerance_abs,
            "relative": rc.magnitude_tolerance_relative,
            "rule": "pass if |reproduced - original| <= max(absolute, relative * original)",
        }
        rec: dict[str, object] = {
            "original_run_id": orig.run_id,
            "original_signal_id": orig.id,
            "signal_kind": orig.signal_kind.value,
            "original_magnitude": orig.magnitude,
            "tolerance": tol,
        }
        replay = executor.replay(orig.run_id)
        rec["replay_run_id"] = replay.run.id
        rec["replay_status"] = replay.status.value
        a, b = (
            registry.find(Provenance, run_id=orig.run_id),
            registry.find(Provenance, run_id=replay.run.id),
        )
        rec["provenance"] = {
            "original_environment_id": a[0].environment_id if a else None,
            "replay_environment_id": b[0].environment_id if b else None,
            "seed": b[0].seed if b else None,
            "source_state": b[0].source.state.value if b else None,
        }
        reproduced: float | None = None
        if replay.status is RunStatus.COMPLETED:
            ev = load_evaluation(registry, store, replay.run.id)
            builder = SignalBuilder(
                replay.run.id,
                replay.run.experiment_id,
                ev,
                cfg.extraction,
                cfg.config_hash,
                now,
                _fault_context(f) if isinstance(f := orig.detail.get("fault"), Mapping) else None,
            )
            if builder.fault is not None:
                fx = registry.get(FaultExperiment, builder.fault.fault_experiment_id)
                base = load_evaluation(registry, store, fx.baseline_run_id)
                extract_from_fault(
                    builder,
                    base,
                    load_errors(registry, store, fx.baseline_run_id),
                    load_errors(registry, store, replay.run.id),
                    str(fx.design.get("primary_metric") or default_primary_metric(base.task)),
                )
            else:
                extract_from_evaluation(builder, load_errors(registry, store, replay.run.id))
            same = [s.magnitude for s in builder.out if match_key(s) == match_key(orig)]
            reproduced = max(same) if same else None
            rec["reason"] = (
                None
                if same
                else "the finding did not reappear (absent or below the configured threshold)"
            )
        else:
            rec["reason"] = f"replay run ended {replay.status.value}"
        ok = reproduced is not None and abs(reproduced - orig.magnitude) <= max(
            rc.magnitude_tolerance_abs, rc.magnitude_tolerance_relative * orig.magnitude
        )
        rec["reproduced_magnitude"] = reproduced
        rec["passed"] = ok
        records.append(rec)
        made.append(
            evidence(
                mode.id,
                EvidenceKind.REPRODUCTION,
                replay.run.id,
                f"replay of {orig.run_id}: {'reproduced' if ok else 'not reproduced'}",
                rec,
                now,
            )
        )
    n_ok = sum(bool(r["passed"]) for r in records)
    frac = n_ok / len(records) if records else 0.0
    passed = bool(records) and frac >= rc.min_pass_fraction
    summary = {
        "summary": True,
        "attempted": len(records),
        "passed_count": n_ok,
        "pass_fraction": frac,
        "min_pass_fraction": rc.min_pass_fraction,
        "passed": passed,
        "runs_not_replayed": len(best) - len(chosen),
        "config_hash": cfg.config_hash,
    }
    made.append(
        evidence(
            mode.id,
            EvidenceKind.REPRODUCTION,
            None,
            f"reproduction {'passed' if passed else 'failed'}: {n_ok}/{len(records)} runs",
            summary,
            now,
        )
    )
    with registry.transaction():
        for e in made:
            if not registry.exists(FailureEvidence, e.id):
                registry.add(e)
        for r in records:
            edge = FailureRelationship(
                mode.investigation_id,
                NodeKind.FAILURE_MODE,
                mode.id,
                Predicate.REPRODUCED_BY,
                NodeKind.RUN,
                str(r["replay_run_id"]),
                {"passed": r["passed"]},
                now,
            )
            if not registry.exists(FailureRelationship, edge.id):
                registry.add(edge)
    return Reproduction(
        mode.id,
        len(records),
        n_ok,
        passed,
        tuple(records),
        tuple(e.id for e in made),
        len(best) - len(chosen),
    )


def _summaries(registry: Registry, mode_id: str) -> list[FailureEvidence]:
    return [
        e
        for e in registry.find(
            FailureEvidence, failure_mode_id=mode_id, evidence_kind=EvidenceKind.REPRODUCTION.value
        )
        if e.detail.get("summary") is True
    ]


def confirm(
    registry: Registry, mode_id: str, actor: str, reason: str, now: datetime | None = None
) -> FailureMode:
    """CONFIRMED needs: status SUPPORTED, a passing reproduction, and an explicit actor + reason."""
    mode = registry.get(FailureMode, mode_id)
    if mode.status is not FailureStatus.SUPPORTED:
        raise ValidationError(
            f"only a SUPPORTED mode can be confirmed (this one is {mode.status.value})"
        )
    passing = [e for e in _summaries(registry, mode_id) if e.detail.get("passed") is True]
    if not passing:
        raise ValidationError(
            "confirmation requires a passing reproduction check; run `failure reproduce` first"
        )
    return transition(
        registry,
        mode,
        FailureStatus.CONFIRMED,
        actor,
        reason,
        now or datetime.now(UTC),
        {"automatic": False, "reproduction_evidence_id": sorted(e.id for e in passing)[0]},
    )


def change_status(
    registry: Registry,
    mode_id: str,
    target: FailureStatus,
    actor: str,
    reason: str,
    now: datetime | None = None,
) -> FailureMode:
    """Reject, deprecate or otherwise move a mode along a legal transition (never to CONFIRMED)."""
    if target is FailureStatus.CONFIRMED:
        raise FailureError("use confirm(): it enforces support and reproduction")
    return transition(
        registry,
        registry.get(FailureMode, mode_id),
        target,
        actor,
        reason,
        now or datetime.now(UTC),
    )


def graph(registry: Registry, mode_id: str) -> dict[str, object]:
    """The backend-only knowledge graph around one mode: edges in both directions."""
    registry.get(FailureMode, mode_id)
    edges = [
        e for e in registry.find(FailureRelationship) if mode_id in (e.subject_id, e.object_id)
    ]
    return {"mode_id": mode_id, "edges": [to_jsonable(e.to_dict() | {"id": e.id}) for e in edges]}

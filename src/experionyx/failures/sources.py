"""Reading the stored results that discovery starts from. Only verified artifacts of finished
runs are used; anything that cannot be used is recorded as skipped, with its reason."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from experionyx.artifacts import ArtifactStore, LocalArtifactStore
from experionyx.domain import Artifact, Experiment, Run, RunStatus
from experionyx.errors import ExperionyxError, FailureLimitError, ValidationError
from experionyx.evaluation.loading import load_evaluation
from experionyx.evaluation.results import ErrorRecord
from experionyx.evaluation.serial import from_jsonable
from experionyx.failures.config import DiscoveryConfig
from experionyx.failures.entities import FailureSignal
from experionyx.failures.extraction import (
    FaultContext,
    SignalBuilder,
    extract_from_evaluation,
    extract_from_fault,
)
from experionyx.faults.analysis import default_primary_metric
from experionyx.faults.entities import FaultExperiment, FaultTrial, TrialStatus
from experionyx.registry import Registry

ERRORS_PATH = "evaluation/errors.jsonl"
FAULT_PATH = "fault/fault.json"


@dataclass(frozen=True)
class Collected:
    signals: tuple[FailureSignal, ...]
    runs_analyzed: tuple[str, ...]
    sources: tuple[Mapping[str, object], ...]
    skipped: tuple[Mapping[str, object], ...]


def _artifact_file(registry: Registry, store: ArtifactStore, run: Run, path: str) -> Path | None:
    found = [a for a in registry.find(Artifact, run_id=run.id) if a.path == path]
    if not found:
        return None
    store.verify(run, found[0])
    if not isinstance(store, LocalArtifactStore):
        raise ExperionyxError("reading artifact contents requires a LocalArtifactStore")
    return store.run_dir(run) / "artifacts" / path


def load_errors(registry: Registry, store: ArtifactStore, run_id: str) -> list[ErrorRecord] | None:
    """The run's stored error records (verified), or None if it recorded none."""
    file = _artifact_file(registry, store, registry.get(Run, run_id), ERRORS_PATH)
    if file is None:
        return None
    lines = file.read_text(encoding="utf-8").splitlines()
    return [from_jsonable(ErrorRecord, json.loads(line)) for line in lines if line.strip()]


def load_fault(registry: Registry, store: ArtifactStore, run_id: str) -> Mapping[str, object]:
    file = _artifact_file(registry, store, registry.get(Run, run_id), FAULT_PATH)
    if file is None:
        raise ExperionyxError(f"run {run_id} has no fault artifact")
    data = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ExperionyxError("fault artifact is not an object")
    return data


# Sources may come from ANY investigation: cross-experiment reuse is the point. The discovery
# result is homed in `investigation_id`; each source's own investigation is recorded as lineage.
def _investigation_of(registry: Registry, run: Run) -> str:
    return registry.get(Experiment, run.experiment_id).investigation_id


def fault_context(
    fault_doc: Mapping[str, object], trial: FaultTrial, fx: FaultExperiment
) -> FaultContext:
    spec = fault_doc["fault"]
    assert isinstance(spec, Mapping)  # noqa: S101
    params = spec.get("parameters", {})
    frac = fault_doc.get("affected_fraction")
    return FaultContext(
        type=str(spec["type"]), family_id=trial.family_id, fault_id=trial.fault_id, seed=trial.seed,
        target=str(fault_doc.get("target", "INPUT_OR_MIXED")),
        parameters=dict(params) if isinstance(params, Mapping) else {},
        parameter_value=trial.parameter_value,
        affected_fraction=float(frac) if isinstance(frac, int | float) else None,
        fault_experiment_id=fx.id, baseline_run_id=fx.baseline_run_id,
        source_dataset_fingerprint=str(fault_doc.get("source_dataset_fingerprint")) if fault_doc.get("source_dataset_fingerprint") else None,
    )  # fmt: skip


def collect_signals(
    registry: Registry, store: ArtifactStore, investigation_id: str, run_ids: Sequence[str],
    fault_experiment_ids: Sequence[str], cfg: DiscoveryConfig, now: datetime,
) -> Collected:  # fmt: skip
    """Extract signals from the selected runs. Explicit `run_ids` contribute the failures visible
    in their own evaluation; fault experiments contribute how each treatment run is WORSE than
    its baseline (baseline weaknesses are not attributed to the fault)."""
    ex, chash = cfg.extraction, cfg.config_hash
    runs, fxps = sorted(set(run_ids)), sorted(set(fault_experiment_ids))
    if not runs and not fxps:
        raise ValidationError("discovery needs at least one run or fault experiment")
    fx_records = [registry.get(FaultExperiment, i) for i in fxps]
    trials = {fx.id: registry.find(FaultTrial, fault_experiment_id=fx.id) for fx in fx_records}
    planned = len(runs) + sum(1 + len(t) for t in trials.values())
    if planned > cfg.max_source_runs:
        raise FailureLimitError(
            f"{planned} source runs exceed max_source_runs={cfg.max_source_runs}; nothing was analyzed"
        )
    signals: dict[str, FailureSignal] = {}
    analyzed: dict[str, None] = {}
    sources: list[Mapping[str, object]] = []
    skipped: list[Mapping[str, object]] = []

    def skip(run_id: str | None, why: str, **extra: object) -> None:
        skipped.append({"run_id": run_id, "reason": why, **extra})

    def keep(built: SignalBuilder, run_id: str) -> None:
        analyzed[run_id] = None
        for s in built.out:
            signals[s.id] = s

    for rid in runs:
        run = registry.get(Run, rid)
        if run.status is not RunStatus.COMPLETED:
            skip(rid, f"run is {run.status.value}")
            continue
        try:
            ev = load_evaluation(registry, store, rid)
            errors = load_errors(registry, store, rid)
        except ExperionyxError as exc:
            skip(rid, f"unusable evaluation: {exc}")
            continue
        b = SignalBuilder(rid, run.experiment_id, ev, ex, chash, now)
        extract_from_evaluation(b, errors)
        keep(b, rid)
        sources.append(
            {
                "kind": "evaluation-run",
                "run_id": rid,
                "investigation_id": _investigation_of(registry, run),
                "signals": len(b.out),
            }
        )
    for fx in fx_records:
        try:
            base = load_evaluation(registry, store, fx.baseline_run_id)
            base_errors = load_errors(registry, store, fx.baseline_run_id)
        except ExperionyxError as exc:
            skip(fx.baseline_run_id, f"unusable baseline: {exc}", fault_experiment_id=fx.id)
            continue
        primary = str(fx.design.get("primary_metric") or default_primary_metric(base.task))
        used = 0
        for t in sorted(trials[fx.id], key=lambda x: (x.point_index, x.repeat_index)):
            if t.status is not TrialStatus.COMPLETED or t.treatment_run_id is None:
                skip(
                    t.treatment_run_id,
                    f"trial {t.status.value}: {t.reason}",
                    fault_experiment_id=fx.id,
                )
                continue
            try:
                run = registry.get(Run, t.treatment_run_id)
                ev = load_evaluation(registry, store, t.treatment_run_id)
                errors = load_errors(registry, store, t.treatment_run_id)
                ctx = fault_context(load_fault(registry, store, t.treatment_run_id), t, fx)
            except ExperionyxError as exc:
                skip(
                    t.treatment_run_id, f"unusable treatment run: {exc}", fault_experiment_id=fx.id
                )
                continue
            b = SignalBuilder(t.treatment_run_id, run.experiment_id, ev, ex, chash, now, ctx)
            extract_from_fault(b, base, base_errors, errors, primary)
            keep(b, t.treatment_run_id)
            used += 1
        sources.append(
            {
                "kind": "fault-experiment",
                "investigation_id": fx.investigation_id,
                "fault_experiment_id": fx.id,
                "baseline_run_id": fx.baseline_run_id,
                "treatment_runs_used": used,
                "trials": len(trials[fx.id]),
                "primary_metric": primary,
            }
        )
    if len(signals) > ex.max_signals:
        raise FailureLimitError(
            f"{len(signals)} signals exceed max_signals={ex.max_signals}; nothing was clustered"
        )
    return Collected(
        tuple(sorted(signals.values(), key=lambda s: s.id)),
        tuple(analyzed),
        tuple(sources),
        tuple(skipped),
    )

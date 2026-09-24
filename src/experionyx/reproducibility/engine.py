"""Dispatch one reproduction attempt to the target kind's OWN, already-audited replay path (every
analysis engine in this project already exposes `replay_check(reg, store, executor, id) ->
dict[str, object]` with the same shape: `original_run`, `replay_run`, `replay_status`,
`deterministic`, `sources_changed`, `differences`). This module never re-implements what any of
those already do; it classifies their result against the requested `ReproductionMode`, optionally
re-compares differing artifacts within a numeric tolerance, checks environment and artifact
integrity, and persists one `ReproductionAttempt` (see docs/reproducibility.md)."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from experionyx.artifacts import ArtifactStore
from experionyx.domain import Artifact, Investigation, Run, RunStatus
from experionyx.errors import ArtifactIntegrityError, ExperionyxError, ValidationError
from experionyx.execution import Executor
from experionyx.faults.report import read_artifact
from experionyx.provenance import Provenance
from experionyx.registry import Registry
from experionyx.reproducibility.compare import compare_documents, diff_mappings
from experionyx.reproducibility.entities import ReproductionAttempt
from experionyx.reproducibility.spec import ENGINE_VERSION, ReproductionSpec
from experionyx.reproducibility.taxonomy import ComparisonOutcome, ReproductionMode, TargetKind


@dataclass(frozen=True)
class _EngineReplay:
    documents: tuple[str, ...]
    prefix: str
    fn: Callable[[Registry, ArtifactStore, Executor, str], dict[str, object]]


def _dispatch_table() -> dict[TargetKind, _EngineReplay]:
    from experionyx.benchmark.engine import ARTIFACTS as benchmark_docs
    from experionyx.benchmark.engine import replay_check as benchmark_replay
    from experionyx.calibration.engine import ARTIFACT_DIR as calibration_dir
    from experionyx.calibration.engine import DOCUMENTS as calibration_docs
    from experionyx.calibration.engine import replay_check as calibration_replay
    from experionyx.data_quality.engine import DOCUMENTS as quality_docs
    from experionyx.data_quality.engine import replay_check as quality_replay
    from experionyx.drift.engine import DOCUMENTS as drift_docs
    from experionyx.drift.engine import replay_check as drift_replay
    from experionyx.graph.engine import ARTIFACTS as graph_docs
    from experionyx.graph.engine import replay_check as graph_replay
    from experionyx.interactions.lifecycle import replay_check as interaction_replay
    from experionyx.reliability.engine import replay_check as reliability_replay
    from experionyx.resources.engine import ARTIFACT_DIR as resources_dir
    from experionyx.resources.engine import DOCUMENTS as resources_docs
    from experionyx.resources.engine import replay_check as resources_replay
    from experionyx.stress.engine import ARTIFACT_DIR as stress_dir
    from experionyx.stress.engine import DOCUMENTS as stress_docs
    from experionyx.stress.engine import replay_check as stress_replay

    return {
        TargetKind.BENCHMARK_RESULT: _EngineReplay(
            tuple(benchmark_docs), "benchmark", benchmark_replay
        ),
        TargetKind.STRESS_ANALYSIS: _EngineReplay(tuple(stress_docs), stress_dir, stress_replay),
        TargetKind.CALIBRATION_ANALYSIS: _EngineReplay(
            tuple(calibration_docs), calibration_dir, calibration_replay
        ),
        TargetKind.RESOURCE_ANALYSIS: _EngineReplay(
            tuple(resources_docs), resources_dir, resources_replay
        ),
        TargetKind.DRIFT_ANALYSIS: _EngineReplay(tuple(drift_docs), "drift", drift_replay),
        TargetKind.QUALITY_ANALYSIS: _EngineReplay(
            tuple(quality_docs), "data_quality", quality_replay
        ),
        TargetKind.RELIABILITY_PROFILE: _EngineReplay(
            ("profile", "sources"), "reliability", reliability_replay
        ),
        TargetKind.INTERACTION_ANALYSIS: _EngineReplay(
            ("spec", "effects", "bootstrap", "summary", "per_sample"),
            "interaction",
            interaction_replay,
        ),
        TargetKind.GRAPH_SNAPSHOT: _EngineReplay(tuple(graph_docs), "graph", graph_replay),
    }


_TARGET_ENTITY: dict[TargetKind, str] = {
    TargetKind.BENCHMARK_RESULT: "experionyx.benchmark.entities:BenchmarkResult",
    TargetKind.STRESS_ANALYSIS: "experionyx.stress.entities:StressAnalysis",
    TargetKind.CALIBRATION_ANALYSIS: "experionyx.calibration.entities:CalibrationAnalysis",
    TargetKind.RESOURCE_ANALYSIS: "experionyx.resources.entities:ResourceAnalysis",
    TargetKind.DRIFT_ANALYSIS: "experionyx.drift.entities:DriftAnalysis",
    TargetKind.QUALITY_ANALYSIS: "experionyx.data_quality.entities:QualityAnalysis",
    TargetKind.RELIABILITY_PROFILE: "experionyx.reliability.entities:ReliabilityProfile",
    TargetKind.INTERACTION_ANALYSIS: "experionyx.interactions.entities:InteractionAnalysis",
    TargetKind.GRAPH_SNAPSHOT: "experionyx.graph.entities:GraphSnapshot",
}


def _load_entity_class(path: str) -> Any:
    import importlib

    module_name, _, class_name = path.partition(":")
    return getattr(importlib.import_module(module_name), class_name)


def _investigation_for_run(registry: Registry, run_id: str) -> str:
    from experionyx.domain import Experiment

    run = registry.get(Run, run_id)
    return registry.get(Experiment, run.experiment_id).investigation_id


def resolve_target(registry: Registry, target_kind: TargetKind, target_id: str) -> dict[str, object]:  # fmt: skip
    """Confirm a target exists and report its home investigation/run, without reproducing
    anything -- the check behind `reproduce validate`."""
    if target_kind is TargetKind.RUN:
        registry.get(Run, target_id)
        return {"exists": True, "investigation_id": _investigation_for_run(registry, target_id), "run_id": target_id}  # fmt: skip
    if target_kind is TargetKind.SCHEDULE:
        from experionyx.scheduler.entities import Schedule

        registry.get(Schedule, target_id)
        return {"exists": True, "investigation_id": None, "run_id": None}
    cls = _load_entity_class(_TARGET_ENTITY[target_kind])
    target = registry.get(cls, target_id)
    return {"exists": True, "investigation_id": target.investigation_id, "run_id": target.run_id}


def verify_run_artifacts(registry: Registry, store: ArtifactStore, run_id: str) -> dict[str, object]:  # fmt: skip
    """Re-hash every persisted artifact of `run_id` -- the check behind `reproduce verify`."""
    return _artifact_verification(registry, store, run_id)


def compare_artifact_documents(
    registry: Registry,
    store: ArtifactStore,
    run_a: str,
    path_a: str,
    run_b: str,
    path_b: str,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, object]:
    """Ad hoc, tolerance-aware comparison of two stored JSON artifacts -- possibly from two
    DIFFERENT runs/targets, unlike `run_reproduction` which always compares one target against
    its own replay. The check behind `reproduce compare`."""
    x = read_artifact(registry, store, run_a, path_a)
    y = read_artifact(registry, store, run_b, path_b)
    outcome, diffs = compare_documents(
        x, y, relative_tolerance=relative_tolerance, absolute_tolerance=absolute_tolerance
    )
    return {
        "run_a": run_a, "path_a": path_a, "run_b": run_b, "path_b": path_b,
        "outcome": outcome.value, "differences": [d.to_dict() for d in diffs],
    }  # fmt: skip


def resolve_attempt(registry: Registry, ref: str) -> ReproductionAttempt:
    if ref.startswith(ReproductionAttempt.PREFIX + "_"):
        return registry.get(ReproductionAttempt, ref)
    matches = [a for a in registry.find(ReproductionAttempt) if a.id.startswith(ref)]
    if len(matches) != 1:
        raise ValidationError(f"{'no' if not matches else 'ambiguous'} reproduction attempt matches {ref!r}")  # fmt: skip
    return matches[0]


def _environment_diff(registry: Registry, original_run_id: str | None, replay_run_id: str | None) -> dict[str, object]:  # fmt: skip
    """Compare the two runs' `Provenance`/`EnvironmentSnapshot`: Python version, OS, machine,
    package versions, dependency digest, executor version, seed, source commit. Missing evidence
    (no Provenance yet, e.g. a replay that failed before completing) is reported, never guessed."""
    if original_run_id is None or replay_run_id is None:
        return {"comparable": False, "reason": "no single representative run pair for this target"}
    from experionyx.domain import EnvironmentSnapshot

    flat: dict[str, dict[str, object]] = {}
    for label, run_id in (("original", original_run_id), ("reproduced", replay_run_id)):
        found = registry.find(Provenance, run_id=run_id)
        if not found:
            return {"comparable": False, "reason": f"{label} run has no Provenance record yet"}
        p = found[0]
        env = registry.get(EnvironmentSnapshot, p.environment_id)
        fields: dict[str, object] = {
            "python_version": env.python_version,
            "os": env.os,
            "machine": env.machine,
            "dependency_digest": p.dependency_digest,
            "executor_version": p.executor_version,
            "seed": p.seed,
            "source_commit": p.source.commit,
            "source_state": p.source.state.value,
        }
        for name, ver in env.packages.items():
            fields[f"package:{name}"] = ver
        flat[label] = fields
    return {"comparable": True, **diff_mappings(flat["original"], flat["reproduced"])}


def _artifact_verification(registry: Registry, store: ArtifactStore, run_id: str | None) -> dict[str, object]:  # fmt: skip
    """Re-hash every persisted artifact of `run_id` against its recorded digest. Never regenerates
    a missing artifact; a missing or corrupted artifact is reported, not silently skipped."""
    if run_id is None:
        return {"checked": 0, "ok": [], "missing": [], "corrupted": []}
    run = registry.get(Run, run_id)
    ok: list[str] = []
    corrupted: list[str] = []
    for artifact in registry.find(Artifact, run_id=run_id):
        try:
            store.verify(run, artifact)
            ok.append(artifact.path)
        except ArtifactIntegrityError:
            corrupted.append(artifact.path)
        except OSError:
            corrupted.append(artifact.path)
    return {"checked": len(ok) + len(corrupted), "ok": sorted(ok), "missing": [], "corrupted": sorted(corrupted)}  # fmt: skip


def _classify(
    mode: ReproductionMode,
    raw: dict[str, object],
    field_differences: tuple[dict[str, object], ...],
) -> ComparisonOutcome:
    if raw.get("sources_changed"):
        return ComparisonOutcome.UNAVAILABLE
    if raw.get("replay_status") != "COMPLETED":
        return ComparisonOutcome.UNAVAILABLE
    deterministic = raw.get("deterministic")
    if mode is ReproductionMode.PROVENANCE_ONLY:
        return ComparisonOutcome.EQUAL  # outputs were never compared; identity/config match was the whole check  # fmt: skip
    if mode in (ReproductionMode.EXACT, ReproductionMode.DETERMINISTIC):
        return ComparisonOutcome.EQUAL if deterministic else ComparisonOutcome.DIFFERENT
    if mode in (ReproductionMode.NUMERIC_TOLERANCE, ReproductionMode.STATISTICAL):
        if deterministic:
            return ComparisonOutcome.EQUAL
        if not field_differences:
            # the underlying engine reported a byte-level difference but nothing was available to
            # re-diff at field granularity (binary artifact, or the artifact vanished)
            return ComparisonOutcome.DIFFERENT
        worst = {"DIFFERENT", "INCOMPARABLE"} & {d["outcome"] for d in field_differences}
        if "INCOMPARABLE" in worst:
            return ComparisonOutcome.INCOMPARABLE
        if "DIFFERENT" in worst:
            return ComparisonOutcome.DIFFERENT
        return ComparisonOutcome.APPROXIMATELY_EQUAL
    return ComparisonOutcome.UNAVAILABLE


def _tolerance_recompare(
    registry: Registry,
    store: ArtifactStore,
    original_run_id: str,
    replay_run_id: str,
    prefix: str,
    names: tuple[str, ...],
    relative_tolerance: float,
    absolute_tolerance: float,
) -> tuple[dict[str, object], ...]:
    out: list[dict[str, object]] = []
    for name in names:
        try:
            x = read_artifact(registry, store, original_run_id, f"{prefix}/{name}.json")
            y = read_artifact(registry, store, replay_run_id, f"{prefix}/{name}.json")
        except ExperionyxError:
            continue
        if x == y:
            continue
        _outcome, diffs = compare_documents(
            x, y, relative_tolerance=relative_tolerance, absolute_tolerance=absolute_tolerance
        )
        out.extend({"document": name, **d.to_dict()} for d in diffs)
    return tuple(out)


def _run_kind_replay(
    registry: Registry, store: ArtifactStore, executor: Executor, run_id: str
) -> dict[str, object]:
    """A RUN target has no fixed document schema, so it is compared by artifact digest only
    (byte-identical or not) -- never at field granularity. Use a specific analysis `TargetKind`
    for tolerance-aware, field-level comparison."""
    replay = executor.replay(run_id)
    out: dict[str, object] = {
        "original_run": run_id,
        "replay_run": replay.run.id,
        "replay_status": replay.status.value,
    }
    if replay.status is not RunStatus.COMPLETED:
        return {**out, "deterministic": False, "sources_changed": False, "differences": [f"replay run ended {replay.status.value}"]}  # fmt: skip
    originals = {a.path: a.digest for a in registry.find(Artifact, run_id=run_id)}
    reproduced = {a.path: a.digest for a in registry.find(Artifact, run_id=replay.run.id)}
    diffs = sorted(p for p in originals if originals.get(p) != reproduced.get(p))
    return {**out, "deterministic": not diffs, "sources_changed": False, "differences": diffs}


def _schedule_kind_replay(
    registry: Registry,
    store: ArtifactStore,
    open_registry: Callable[[], Registry],
    open_executor: Callable[[Registry], Executor],
    source_root: Any,
    schedule_id: str,
) -> tuple[dict[str, object], str | None]:
    from experionyx.scheduler.engine import (  # fmt: skip
        resolve_schedule,
        run_schedule,
        spec_from_schedule,
        unit_views,
    )
    from experionyx.scheduler.taxonomy import UnitKind

    sched = resolve_schedule(registry, schedule_id)
    before = {u.key: (u.status.value, u.primary_ref) for u in unit_views(registry, sched.id)}
    spec = spec_from_schedule(sched)
    result = run_schedule(open_registry, store, open_executor, spec, source_root=source_root)
    after = {u.key: (u.status.value, u.primary_ref) for u in unit_views(registry, sched.id)}
    differences = sorted(k for k in before if before.get(k) != after.get(k))
    resource_keys = {u.key for u in spec.units if u.kind is UnitKind.RESOURCE}
    non_resource_diffs = [d for d in differences if d not in resource_keys]
    return (
        {
            "original_run": None,
            "replay_run": None,
            "replay_status": result.status.value if result.status is not None else None,
            "deterministic": not non_resource_diffs,
            "sources_changed": False,
            "differences": differences,
        },
        result.schedule_run_id,
    )


@dataclass(frozen=True)
class ReproductionRunResult:
    attempt_id: str
    outcome: ComparisonOutcome
    replay_run_id: str | None


def _run_provenance_only(
    registry: Registry,
    store: ArtifactStore,
    spec: ReproductionSpec,
    attempt_number: int,
    investigation_id: str | None,
) -> ReproductionRunResult:
    """Never re-executes anything: confirms the target's OWN recorded evidence (provenance,
    identity, stored artifacts) is present and intact. This is the cheapest, always-available
    check, and the only one meaningful when the platform cannot guarantee re-execution at all."""
    if spec.target_kind is TargetKind.RUN:
        original_run_id: str | None = spec.target_id
        inv = investigation_id or _investigation_for_run(registry, spec.target_id)
    elif spec.target_kind is TargetKind.SCHEDULE:
        if investigation_id is None:
            raise ValidationError("reproducing a SCHEDULE needs --investigation")
        inv, original_run_id = investigation_id, None
    else:
        cls = _load_entity_class(_TARGET_ENTITY[spec.target_kind])
        target = registry.get(cls, spec.target_id)
        inv = investigation_id or target.investigation_id
        original_run_id = target.run_id
    registry.get(Investigation, inv)

    art_verification = _artifact_verification(registry, store, original_run_id)
    has_provenance = original_run_id is not None and bool(registry.find(Provenance, run_id=original_run_id))  # fmt: skip
    if original_run_id is None:
        outcome, note = ComparisonOutcome.UNAVAILABLE, "this target has no single representative run to check provenance for"  # fmt: skip
    elif not has_provenance:
        outcome, note = ComparisonOutcome.UNAVAILABLE, "the original run has no Provenance record"
    elif art_verification["corrupted"]:
        outcome, note = ComparisonOutcome.DIFFERENT, "one or more stored artifacts failed digest verification"  # fmt: skip
    else:
        outcome, note = ComparisonOutcome.EQUAL, "outputs were not re-executed or compared; only identity/configuration/environment/artifact integrity were checked"  # fmt: skip

    attempt = ReproductionAttempt(
        spec.target_kind, spec.target_id, spec.spec_id, spec.to_dict(), attempt_number, inv,
        original_run_id, None, None, None, outcome, False, (), (), {}, art_verification, note,
        ENGINE_VERSION, datetime.now(UTC),
    )  # fmt: skip
    registry.add(attempt)
    return ReproductionRunResult(attempt.id, outcome, None)


def run_reproduction(
    registry: Registry,
    store: ArtifactStore,
    executor: Executor,
    spec: ReproductionSpec,
    *,
    investigation_id: str | None = None,
    open_registry: Callable[[], Registry] | None = None,
    open_executor: Callable[[Registry], Executor] | None = None,
    source_root: Any = None,
) -> ReproductionRunResult:
    """Dispatch, classify and persist one reproduction attempt. Never mutates the target: every
    call adds a new, immutable `ReproductionAttempt` (attempt numbers 0, 1, 2, ... per target).
    `open_registry`/`open_executor`/`source_root` are only needed for `TargetKind.SCHEDULE`, whose
    own engine needs to open one fresh connection per worker thread."""
    existing = registry.find(ReproductionAttempt, target_id=spec.target_id)
    attempt_number = len(existing)

    if spec.mode is ReproductionMode.PROVENANCE_ONLY:
        return _run_provenance_only(registry, store, spec, attempt_number, investigation_id)

    original_run_id: str | None
    prefix: str
    names: tuple[str, ...]
    if spec.target_kind is TargetKind.RUN:
        original_run_id = spec.target_id
        inv = investigation_id or _investigation_for_run(registry, spec.target_id)
        raw = _run_kind_replay(registry, store, executor, spec.target_id)
        secondary_ref = None
        prefix, names = "", ()
    elif spec.target_kind is TargetKind.SCHEDULE:
        if investigation_id is None:
            raise ValidationError("reproducing a SCHEDULE needs --investigation")
        if open_registry is None or open_executor is None or source_root is None:
            raise ValidationError("reproducing a SCHEDULE needs open_registry/open_executor/source_root")  # fmt: skip
        inv = investigation_id
        original_run_id = None
        raw, secondary_ref = _schedule_kind_replay(
            registry, store, open_registry, open_executor, source_root, spec.target_id
        )
        prefix, names = "", ()
    else:
        cls = _load_entity_class(_TARGET_ENTITY[spec.target_kind])
        target = registry.get(cls, spec.target_id)
        inv = investigation_id or target.investigation_id
        original_run_id = target.run_id
        entry = _dispatch_table()[spec.target_kind]
        raw = entry.fn(registry, store, executor, spec.target_id)
        secondary_ref = None
        prefix, names = entry.prefix, entry.documents

    registry.get(Investigation, inv)
    replay_run_id_raw = raw.get("replay_run")
    replay_run_id = replay_run_id_raw if isinstance(replay_run_id_raw, str) else None
    replay_status = raw.get("replay_status")
    differences_raw = raw.get("differences")
    differences = tuple(str(d) for d in differences_raw) if isinstance(differences_raw, list) else ()  # fmt: skip

    field_diffs: tuple[dict[str, object], ...] = ()
    if (
        spec.mode in (ReproductionMode.NUMERIC_TOLERANCE, ReproductionMode.STATISTICAL)
        and differences
        and original_run_id is not None
        and replay_run_id is not None
        and names
    ):
        field_diffs = _tolerance_recompare(
            registry, store, original_run_id, replay_run_id, prefix, names,
            spec.relative_tolerance, spec.absolute_tolerance,
        )  # fmt: skip

    outcome = _classify(spec.mode, raw, field_diffs)
    note: str | None = None
    if raw.get("sources_changed"):
        note_raw = raw.get("note")
        note = note_raw if isinstance(note_raw, str) else "the persisted evidence changed since the target was collected; determinism cannot be judged from this replay"  # fmt: skip

    env_diff = _environment_diff(registry, original_run_id, replay_run_id)
    art_verification = _artifact_verification(registry, store, original_run_id)

    attempt = ReproductionAttempt(
        spec.target_kind, spec.target_id, spec.spec_id, spec.to_dict(), attempt_number, inv,
        original_run_id, replay_run_id, str(replay_status) if replay_status is not None else None,
        secondary_ref, outcome, bool(raw.get("sources_changed", False)), differences, field_diffs,
        env_diff, art_verification, note, ENGINE_VERSION, datetime.now(UTC),
    )  # fmt: skip
    registry.add(attempt)
    return ReproductionRunResult(attempt.id, outcome, replay_run_id)


__all__ = [
    "ReproductionRunResult",
    "compare_artifact_documents",
    "resolve_attempt",
    "resolve_target",
    "run_reproduction",
    "verify_run_artifacts",
]

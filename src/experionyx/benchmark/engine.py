"""Executing a benchmark. The orchestrator validates, then runs the expanded protocol through the
EXISTING machinery (fault laboratory, failure discovery, interaction analysis, reliability profile);
a final `collect` Run reads the persisted evidence and writes the coverage, results and artifacts,
so the benchmark itself has provenance and can be replayed. Nothing is re-implemented here."""

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from experionyx.artifacts import ArtifactStore
from experionyx.benchmark import collect as col
from experionyx.benchmark.entities import Benchmark, BenchmarkResult, BenchmarkUnit
from experionyx.benchmark.protocol import base_fault, design_for, expand, pair_bases
from experionyx.benchmark.spec import ENGINE_VERSION, BenchmarkSpec
from experionyx.benchmark.taxonomy import CoverageStatus, UnitKind, UnitStatus
from experionyx.domain import (
    Artifact,
    ClaimStatus,
    ConfigurationRef,
    DatasetRef,
    EpistemicKind,
    EvidenceRelation,
    EvidenceTarget,
    Experiment,
    ExperimentStatus,
    ModelRef,
    Run,
    RunStatus,
    to_jsonable,
)
from experionyx.drift.engine import run_drift_request
from experionyx.errors import BenchmarkError, ExperionyxError, ValidationError
from experionyx.evaluation.config import _thaw
from experionyx.execution import Executor, RunContext, resolve_procedure
from experionyx.failures.config import DiscoveryConfig
from experionyx.failures.engine import run_discovery
from experionyx.faults.design import FaultDesign, FaultLimits
from experionyx.faults.entities import FaultTrial, TrialStatus
from experionyx.faults.lab import run_fault_experiment, with_seed
from experionyx.faults.library import default_fault_registry
from experionyx.faults.report import read_artifact
from experionyx.interactions.config import InteractionSpec
from experionyx.interactions.engine import run_interaction
from experionyx.interactions.lifecycle import REPLAY_TOLERANCE, _compare
from experionyx.registry import Registry
from experionyx.reliability.engine import run_profile
from experionyx.reliability.spec import ProfileSpec
from experionyx.reliability.taxonomy import Scope
from experionyx.slices.engine import SliceAnalysisSpec, run_slice_analysis_request

PROCEDURE = "experionyx.benchmark.engine:run_benchmark_collect"
ARTIFACTS = ("spec", "units", "coverage", "results", "summary")


def _tag(spec: BenchmarkSpec) -> str:
    return spec.spec_id[4:16]


@dataclass(frozen=True)
class BenchmarkRunResult:
    benchmark_id: str | None
    result_id: str | None
    run_id: str | None
    status: RunStatus | None
    already_run: bool
    errors: Mapping[str, str]


def _plain_design(spec: BenchmarkSpec, base: Any) -> FaultDesign:
    return FaultDesign(
        base.to_dict(),
        spec.evaluation,
        seeds=spec.seeds,
        primary_metric=spec.primary_metric,
        limits=FaultLimits(max_failed_trials=spec.limits.max_failed_trials),
        aggregation_confidence=spec.aggregation_confidence,
        aggregation_resamples=spec.aggregation_resamples,
        aggregation_seed=spec.aggregation_seed,
    )


def _completed_runs(registry: Registry, fx_id: str, point_index: int) -> tuple[str, ...]:
    return tuple(
        sorted(
            t.treatment_run_id
            for t in registry.find(FaultTrial, fault_experiment_id=fx_id)
            if t.point_index == point_index
            and t.status is TrialStatus.COMPLETED
            and t.treatment_run_id
        )
    )


def _existing(registry: Registry, spec: BenchmarkSpec) -> BenchmarkResult | None:
    for b in registry.find(Benchmark, spec_id=spec.spec_id):
        found = registry.find(BenchmarkResult, benchmark_id=b.id)
        if found:
            return found[0]
    return None


def run_benchmark(
    registry: Registry,
    store: ArtifactStore,
    executor: Executor,
    spec: BenchmarkSpec,
    *,
    source_root: Path,
    discovery_config: DiscoveryConfig | None = None,
) -> BenchmarkRunResult:
    """Validate (BenchmarkRefusal before anything exists), execute, collect. Idempotent per
    definition: a benchmark whose result already exists in this registry is returned, not re-run."""
    fr = default_fault_registry()
    proto = expand(
        registry, spec, fr
    )  # raises BenchmarkRefusal with every issue, before any run exists
    done = _existing(registry, spec)
    if done is not None:
        return BenchmarkRunResult(
            done.benchmark_id, done.id, done.run_id, RunStatus.COMPLETED, True, {}
        )
    errors: dict[str, str] = {}
    fault_fx: dict[str, str] = {}
    cell_fx: dict[str, dict[str, str]] = {}
    baseline: str | None = None
    tag = _tag(spec)

    def launch(name: str, base: Any, design: FaultDesign) -> Any:
        return run_fault_experiment(
            registry,
            store,
            executor,
            model_id=spec.model,
            dataset_id=spec.dataset,
            base_spec=base,
            fault_registry=fr,
            design=design,
            name=name,
            source_root=source_root,
            baseline_run_id=baseline,
        )

    for g in spec.faults:
        if g.name in {u["grid"] for u in proto.unsupported}:
            continue
        base = base_fault(fr, g, spec.seeds[0])
        try:
            res = launch(f"benchmark {tag} {g.name}", base, design_for(spec, base, g))
            baseline = baseline or res.baseline_run_id
            fault_fx[g.name] = res.fault_experiment.id
        except ExperionyxError as exc:
            errors[f"grid:{g.name}"] = f"{type(exc).__name__}: {exc}"
    if baseline is None:
        raise BenchmarkError(
            "no baseline evaluation could be created; errors: " + json.dumps(errors)
        )
    for p in spec.interactions:
        sa, sb = pair_bases(fr, spec, p)
        compounds = {
            "AB": fr.compound(sa, sb),
            **({"BA": fr.compound(sb, sa)} if p.order_analysis else {}),
        }
        for cell, cbase in compounds.items():
            try:
                res = launch(f"benchmark {tag} {p.name} {cell}", cbase, _plain_design(spec, cbase))
                cell_fx.setdefault(p.name, {})[cell] = res.fault_experiment.id
            except ExperionyxError as exc:
                errors[f"pair:{p.name}:{cell}"] = f"{type(exc).__name__}: {exc}"
    inv = registry.get(Experiment, registry.get(Run, baseline).experiment_id).investigation_id
    discovery_run: str | None = None
    mode_ids: list[str] = []
    all_fx = [*fault_fx.values(), *(x for cells in cell_fx.values() for x in cells.values())]
    if spec.discovery:
        try:
            d = run_discovery(
                registry,
                store,
                executor,
                inv,
                [baseline],
                all_fx,
                discovery_config or DiscoveryConfig(),
            )
            if d.status is RunStatus.COMPLETED:
                discovery_run = d.run_id
                cand = read_artifact(registry, store, d.run_id, "failure-candidates.json")
                listed: Any = cand
                mode_ids = [str(m["id"]) for m in listed["modes"]]
            else:
                errors["discovery"] = f"the discovery run ended {d.status.value}"
        except ExperionyxError as exc:
            errors["discovery"] = f"{type(exc).__name__}: {exc}"
    analyses: dict[str, str] = {}
    for p in spec.interactions:
        gx = {g.name: g for g in spec.faults}
        fa, fb, cells = fault_fx.get(p.a), fault_fx.get(p.b), cell_fx.get(p.name, {})
        info = proto.pairs[p.name]
        if fa is None or fb is None or "AB" not in cells:
            errors.setdefault(
                f"interaction:{p.name}", "a required experiment of the design was not created"
            )
            continue
        try:
            cfg = replace(
                spec.interaction_config,
                order_analysis=p.order_analysis,
                primary_metric=spec.interaction_config.primary_metric or spec.primary_metric,
            )
            ispec = InteractionSpec(
                control=(baseline,),
                a=_completed_runs(registry, fa, info["a_point_index"]),
                b=_completed_runs(registry, fb, info["b_point_index"]),
                ab=_completed_runs(registry, cells["AB"], 0),
                ba=_completed_runs(registry, cells["BA"], 0) if "BA" in cells else (),
                discovery_run_id=discovery_run,
                config=cfg,
            )
            out = run_interaction(registry, store, executor, inv, ispec)
            if out.analysis_id:
                analyses[p.name] = out.analysis_id
            else:
                errors[f"interaction:{p.name}"] = f"the interaction run ended {out.status.value}"
        except (ExperionyxError, ValueError) as exc:
            errors[f"interaction:{p.name}"] = f"{type(exc).__name__}: {exc}"
        del gx
    slice_analysis_id: str | None = None
    if spec.slices:
        try:
            so = run_slice_analysis_request(
                registry, store, executor, inv,
                SliceAnalysisSpec(baseline, spec.slices, (), tuple(fault_fx.values()), tuple(mode_ids), tuple(analyses.values()), spec.slice_config),
            )  # fmt: skip
            slice_analysis_id = so.analysis_id
            if slice_analysis_id is None:
                errors["slices"] = f"the slice analysis run ended {so.status.value}"
        except (ExperionyxError, ValueError) as exc:
            errors["slices"] = f"{type(exc).__name__}: {exc}"
    drift_analysis_id: str | None = None
    if spec.drift is not None:
        try:
            do = run_drift_request(
                registry,
                store,
                executor,
                inv,
                dataclasses.replace(spec.drift, baseline_run=baseline),
            )
            drift_analysis_id = do.analysis_id
            if drift_analysis_id is None:
                errors["drift"] = f"the drift analysis run ended {do.status.value}"
        except (ExperionyxError, ValueError) as exc:
            errors["drift"] = f"{type(exc).__name__}: {exc}"
    profile_id: str | None = None
    if spec.profile:
        try:
            pr = run_profile(
                registry,
                store,
                executor,
                inv,
                ProfileSpec(
                    Scope.MODEL_DATASET_EVALUATION,
                    baseline,
                    tuple(fault_fx.values()),
                    tuple(analyses.values()),
                    tuple(mode_ids),
                    (slice_analysis_id,) if slice_analysis_id else (),
                    (drift_analysis_id,) if drift_analysis_id else (),
                ),
            )
            profile_id = pr.profile_id
            if profile_id is None:
                errors["profile"] = f"the profile run ended {pr.status.value}"
        except ExperionyxError as exc:
            errors["profile"] = f"{type(exc).__name__}: {exc}"
    ex = col.Executed(
        inv,
        baseline,
        fault_fx,
        cell_fx,
        errors,
        discovery_run,
        analyses,
        profile_id,
        slice_analysis_id,
        drift_analysis_id,
    )
    conf = ConfigurationRef(
        {
            "benchmark_collect": {
                "spec": spec.to_dict(),
                "protocol_hash": proto.protocol_hash,
                "executed": ex.to_dict(),
            }
        }
    )
    if not registry.exists(ConfigurationRef, conf.id):
        registry.add(conf)
    exp = Experiment(
        inv,
        f"benchmark {spec.name} {spec.version}",
        "Coverage and results of an executed robustness benchmark protocol",
        ModelRef("none", "0"),
        DatasetRef("none", "0"),
        conf.id,
        datetime.now(UTC),
    )
    if not registry.exists(Experiment, exp.id):
        registry.add(exp)
        registry.update_status(exp.with_status(ExperimentStatus.READY))
    result = executor.execute(
        exp.id, resolve_procedure(PROCEDURE), seed=0, procedure_name=PROCEDURE
    )
    found = _existing(registry, spec)
    return BenchmarkRunResult(
        found.benchmark_id if found else None,
        found.id if found else None,
        result.run.id,
        result.status,
        False,
        errors,
    )


def _write(ctx: RunContext, name: str, payload: object) -> str:
    directory = ctx.artifact_dir / "benchmark"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(
        json.dumps(to_jsonable(payload), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return ctx.register_artifact(
        f"benchmark/{name}.json", name=f"benchmark-{name}", media_type="application/json"
    ).id


def run_benchmark_collect(ctx: RunContext) -> None:
    params = _thaw(ctx.parameters)
    if not isinstance(params, dict) or set(params) != {"benchmark_collect"}:
        raise ValidationError(
            "a benchmark configuration must be exactly {'benchmark_collect': ...}"
        )
    body = params["benchmark_collect"]
    spec = BenchmarkSpec.from_dict(body["spec"])
    ex = col.Executed.from_dict(body["executed"])
    reg, now = ctx.registry, ctx.started_at
    proto = expand(reg, spec, default_fault_registry())
    if proto.protocol_hash != body["protocol_hash"]:
        raise BenchmarkError(
            "the protocol expands differently now than when it was executed (the fault library or engine changed); the results cannot be collected under this definition"
        )
    got = col.collect(reg, ctx.store, spec, proto, ex)
    art = {
        "spec": _write(
            ctx,
            "spec",
            {
                "spec_id": spec.spec_id,
                "spec": spec.to_dict(),
                "protocol_hash": proto.protocol_hash,
                "engine_version": ENGINE_VERSION,
                "resolved_faults": proto.resolved,
                "unsupported": list(proto.unsupported),
                "interaction_pairs": proto.pairs,
                "executed": ex.to_dict(),
            },
        ),
        "units": _write(ctx, "units", {"units": got.units}),
        "coverage": _write(ctx, "coverage", got.coverage),
        "results": _write(ctx, "results", got.results),
        "summary": _write(ctx, "summary", got.summary),
    }
    model_id, data_id = spec.model, spec.dataset
    bench = Benchmark(
        spec.name,
        spec.version,
        spec.spec_id,
        spec.to_dict(),
        proto.protocol_hash,
        ENGINE_VERSION,
        model_id,
        data_id,
        now,
    )
    result = BenchmarkResult(
        bench.id,
        ex.investigation_id,
        ctx.run.id,
        proto.protocol_hash,
        got.provenance_fingerprint,
        got.coverage_status,
        got.section_status,
        {
            "coverage_status": got.coverage_status.value,
            "incomplete_reasons": got.summary["incomplete_reasons"],
            "counts": got.summary["counts"],
            "section_status": got.section_status,
            "no_score": col.NO_SCORE,
        },
        now,
    )
    new = not reg.exists(BenchmarkResult, result.id)
    if new:
        with reg.transaction():
            if not reg.exists(Benchmark, bench.id):
                reg.add(bench)
            reg.add(result)
            for u in got.units:
                detail = {
                    k: u[k]
                    for k in (
                        "fault",
                        "fault_id",
                        "point_index",
                        "parameter",
                        "value",
                        "seed",
                        "cell",
                        "pair",
                        "reason",
                    )
                }
                reg.add(
                    BenchmarkUnit(
                        result.id,
                        u["key"],
                        UnitKind(u["kind"]),
                        UnitStatus(u["status"]),
                        u["grid"] or "-",
                        detail,
                        now,
                        u["run_id"],
                    )
                )
    ctx.observe("benchmark.new_record", int(new), kind=EpistemicKind.OBSERVATION)
    for k, n in got.summary["counts"].items():
        ctx.observe(f"benchmark.units.{k}", n, kind=EpistemicKind.DERIVED_METRIC)
    ctx.observe(
        "benchmark.coverage_complete",
        int(got.coverage_status is CoverageStatus.COMPLETE),
        kind=EpistemicKind.DERIVED_METRIC,
    )
    claim = ctx.assert_claim(
        f"[{ctx.run.id}] Benchmark {spec.name} {spec.version} was executed as protocol {proto.protocol_hash[7:19]}: {got.summary['counts']['completed']} of {got.summary['counts']['units']} unit(s) completed; coverage is {got.coverage_status.value}. This reports evidence with its coverage; it makes no assessment of robustness.",
        asserted_by=f"experionyx.benchmark/{ENGINE_VERSION}", status=ClaimStatus.SUPPORTED if got.coverage_status is CoverageStatus.COMPLETE else ClaimStatus.INSUFFICIENT_EVIDENCE,
    )  # fmt: skip
    ctx.add_evidence(
        claim, EvidenceTarget.RUN, ex.baseline_run, EvidenceRelation.CONTEXT, "baseline run"
    )
    for u in got.units:
        if u["kind"] != UnitKind.BASELINE.value and u["status"] == "COMPLETED" and u["run_id"]:
            ctx.add_evidence(
                claim,
                EvidenceTarget.RUN,
                u["run_id"],
                EvidenceRelation.SUPPORTS,
                f"unit {u['key']}",
            )
    for name in ("coverage", "results"):
        ctx.add_evidence(
            claim,
            EvidenceTarget.ARTIFACT,
            art[name],
            EvidenceRelation.SUPPORTS,
            "benchmark document",
        )


def replay_check(
    reg: Registry, store: ArtifactStore, executor: Executor, result_id: str
) -> dict[str, Any]:
    """Replay the collect Run as a NEW run and compare every artifact. If the persisted evidence
    changed since (a failure mode's lifecycle state, say) the fingerprints differ and this is
    reported as `sources_changed`, not as nondeterminism."""
    r = reg.get(BenchmarkResult, result_id)
    replay = executor.replay(r.run_id)
    out: dict[str, Any] = {
        "result_id": r.id,
        "original_run": r.run_id,
        "replay_run": replay.run.id,
        "replay_status": replay.status.value,
        "tolerance": REPLAY_TOLERANCE,
    }
    if replay.status is not RunStatus.COMPLETED:
        return {
            **out,
            "deterministic": False,
            "differences": [f"replay run ended {replay.status.value}"],
        }
    diffs: list[str] = []
    fps: list[object] = []
    for name in ARTIFACTS:
        x = read_artifact(reg, store, r.run_id, f"benchmark/{name}.json")
        y = read_artifact(reg, store, replay.run.id, f"benchmark/{name}.json")
        if name == "summary" and isinstance(x, dict) and isinstance(y, dict):
            fps = [x.get("provenance_fingerprint"), y.get("provenance_fingerprint")]
        _compare(name, x, y, diffs)
    if len(fps) == 2 and fps[0] != fps[1]:
        return {
            **out,
            "deterministic": None,
            "sources_changed": True,
            "differences": diffs[:50],
            "note": "the persisted evidence changed since this result was collected (provenance fingerprint differs); determinism cannot be judged from this replay",
        }
    return {**out, "deterministic": not diffs, "sources_changed": False, "differences": diffs[:50]}


__all__ = [
    "PROCEDURE",
    "Artifact",
    "BenchmarkRunResult",
    "replay_check",
    "run_benchmark",
    "run_benchmark_collect",
    "with_seed",
]

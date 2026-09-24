"""Command line interface. Every command reads or writes the real registry of a workspace."""

import argparse
import dataclasses
import json
import logging
import platform
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from experionyx import __version__
from experionyx.adapters.capabilities import DeviceKind, TaskType
from experionyx.adapters.records import RegisteredDataset, RegisteredModel
from experionyx.adapters.registry import default_registries
from experionyx.artifacts import LocalArtifactStore
from experionyx.benchmark.engine import replay_check as benchmark_replay_check
from experionyx.benchmark.engine import run_benchmark
from experionyx.benchmark.entities import Benchmark, BenchmarkResult
from experionyx.benchmark.protocol import validation_report as benchmark_validation
from experionyx.benchmark.registry import BenchmarkRegistry
from experionyx.benchmark.spec import BenchmarkSpec
from experionyx.calibration import engine as calibration_engine
from experionyx.calibration.entities import CalibrationAnalysis, CalibrationResult
from experionyx.calibration.registry import CalibrationRegistry
from experionyx.calibration.spec import CalibrationSpec
from experionyx.data_quality import engine as quality_engine
from experionyx.data_quality.entities import QualityAnalysis, QualityCheck
from experionyx.data_quality.registry import QualityRegistry
from experionyx.data_quality.spec import CHECK_DOCS, QualitySpec
from experionyx.demos import DEMOS, run_demo
from experionyx.domain import (
    Artifact,
    ConfigurationRef,
    Entity,
    Experiment,
    ExperimentStatus,
    Investigation,
    Observation,
    Run,
    RunStatus,
    to_jsonable,
)
from experionyx.dossier.builder import build_dossier
from experionyx.dossier.builder import build_snapshot as build_dossier_snapshot
from experionyx.dossier.entities import (
    DossierFinding,
    DossierItem,
    DossierSnapshot,
    EvidenceDossier,
)
from experionyx.dossier.render import render_markdown as render_dossier_markdown
from experionyx.dossier.spec import DossierSpec
from experionyx.dossier.taxonomy import DossierSourceKind
from experionyx.drift import engine as drift_engine
from experionyx.drift.entities import DriftAnalysis, DriftWindow
from experionyx.drift.registry import DriftRegistry
from experionyx.drift.spec import ShiftSpec
from experionyx.errors import (
    ArtifactIntegrityError,
    BenchmarkRefusal,
    DesignRefusal,
    ExperionyxError,
    NotFoundError,
    ProfileRefusal,
    SchedulerRefusal,
    ValidationError,
)
from experionyx.evaluation.compare import compare_evaluations
from experionyx.evaluation.config import (
    EvaluationConfig,
    ScoreSource,
)
from experionyx.evaluation.engine import PROCEDURE
from experionyx.evaluation.loading import load_evaluation
from experionyx.evaluation.metrics import default_metric_registry
from experionyx.evaluation.results import EvaluationResult
from experionyx.evaluation.serial import from_jsonable
from experionyx.execution import ExecutionResult, Executor, resolve_procedure, run_states
from experionyx.failures.config import DiscoveryConfig
from experionyx.failures.engine import run_discovery
from experionyx.failures.entities import (
    FailureCluster,
    FailureEvidence,
    FailureMode,
    FailureSignal,
)
from experionyx.failures.lifecycle import change_status, confirm, graph, reproduce
from experionyx.failures.taxonomy import FailureCategory, FailureStatus
from experionyx.faults.degradation import measure_degradation, measure_latency
from experionyx.faults.demos import FAULT_DEMOS, run_fault_demo
from experionyx.faults.design import FaultDesign, FaultLimits, SweepSpec
from experionyx.faults.entities import FaultExperiment, FaultTrial
from experionyx.faults.lab import FaultExperimentResult, run_fault_experiment
from experionyx.faults.library import default_fault_registry
from experionyx.faults.report import analysis_run_id, load_analysis, read_artifact, summary_rows
from experionyx.faults.spec import FaultRegistry, FaultScope, FaultSpec, ScopeKind
from experionyx.graph.diff import diff as graph_diff_snapshots
from experionyx.graph.engine import GraphRunResult, resolve_graph, resolve_snapshot, run_graph
from experionyx.graph.engine import replay_check as graph_replay_check
from experionyx.graph.entities import EvidenceGraph, GraphSnapshot
from experionyx.graph.query import (
    SnapshotIndex,
    analyses_depending_on_run,
    artifacts_for_analysis,
    dataset_to_failure_paths,
    evidence_for_claim,
    model_to_failure_paths,
    runs_contributing_to_failure_mode,
)
from experionyx.graph.query import TraversalResult as GraphTraversalResult
from experionyx.graph.spec import GraphQuery, GraphSpec
from experionyx.graph.taxonomy import NodeKind as GraphNodeKind
from experionyx.graph.taxonomy import TraversalDirection
from experionyx.interactions.config import InteractionSpec
from experionyx.interactions.design import validate_report
from experionyx.interactions.engine import run_interaction
from experionyx.interactions.entities import InteractionAnalysis
from experionyx.interactions.lifecycle import (
    change_status as interaction_change_status,
)
from experionyx.interactions.lifecycle import (
    check_reproduction,
    confirm_by_review,
    replay_check,
)
from experionyx.interactions.registry import InteractionRegistry
from experionyx.interactions.taxonomy import InteractionStatus
from experionyx.leaderboard.compare import (
    CORRECTION_METHODS,
    compare_protocol_constrained,
    compare_submissions,
    correct_family,
)
from experionyx.leaderboard.entities import (
    BenchmarkProtocol,
    BenchmarkSubmission,
    LeaderboardEntry,
    LeaderboardSnapshot,
)
from experionyx.leaderboard.protocol import register_protocol, resolve_protocol
from experionyx.leaderboard.snapshot import build_snapshot
from experionyx.leaderboard.submission import resolve_submission
from experionyx.leaderboard.submission import submit as submit_benchmark
from experionyx.provenance import Provenance, RunOutcome
from experionyx.registry import Registry
from experionyx.reliability.engine import replay_check as profile_replay_check
from experionyx.reliability.engine import run_profile
from experionyx.reliability.entities import ReliabilityProfile
from experionyx.reliability.registry import ReliabilityProfileRegistry
from experionyx.reliability.spec import ProfileSpec
from experionyx.reporting.entities import Report, ReportFinding
from experionyx.reporting.generator import builtin_template, ensure_template, generate_report
from experionyx.reporting.render import render_html, render_markdown
from experionyx.reporting.spec import ReportSpec
from experionyx.reporting.taxonomy import ReportType
from experionyx.reporting.validate import validate_report as report_validate
from experionyx.reproducibility.engine import compare_artifact_documents as reproduction_compare
from experionyx.reproducibility.engine import resolve_attempt as reproduction_resolve
from experionyx.reproducibility.engine import resolve_target as reproduction_resolve_target
from experionyx.reproducibility.engine import run_reproduction
from experionyx.reproducibility.engine import verify_run_artifacts as reproduction_verify
from experionyx.reproducibility.entities import ReproductionAttempt
from experionyx.reproducibility.spec import (
    DEFAULT_ABSOLUTE_TOLERANCE,
    DEFAULT_RELATIVE_TOLERANCE,
    ReproductionSpec,
)
from experionyx.reproducibility.taxonomy import ComparisonOutcome, ReproductionMode, TargetKind
from experionyx.resources import compare as resource_compare
from experionyx.resources import engine as resource_engine
from experionyx.resources.entities import ResourceAnalysis, ResourceTrial
from experionyx.resources.registry import ResourceRegistry
from experionyx.resources.spec import ResourceSpec
from experionyx.scheduler.engine import UnitView as SchedulerUnitView
from experionyx.scheduler.engine import (
    cancel_schedule,
    retry_unit,
    run_schedule,
    spec_from_schedule,
)
from experionyx.scheduler.engine import materialize as scheduler_materialize
from experionyx.scheduler.engine import resolve_schedule as scheduler_resolve
from experionyx.scheduler.engine import unit_views as scheduler_unit_views
from experionyx.scheduler.engine import utc_now as scheduler_utc_now
from experionyx.scheduler.entities import ScheduleRun as ScheduleRunEntity
from experionyx.scheduler.graph import expand
from experionyx.scheduler.graph import validation_report as scheduler_validation_report
from experionyx.scheduler.spec import ScheduleSpec
from experionyx.scheduler.taxonomy import ScheduleRunState
from experionyx.slices import analysis as slice_an
from experionyx.slices.data import dataset_for_run, load_baseline
from experionyx.slices.engine import SliceAnalysisSpec, run_slice_analysis_request
from experionyx.slices.entities import Slice, SliceAnalysis
from experionyx.slices.registry import SliceRegistry
from experionyx.slices.spec import SliceSpec
from experionyx.sqlite import DB_SCHEMA_VERSION, SqliteRegistry
from experionyx.stats import store as stats_store
from experionyx.stats.entities import StatisticalAnalysis
from experionyx.stress import engine as stress_engine
from experionyx.stress.entities import StressAnalysis, StressTrial
from experionyx.stress.registry import StressRegistry
from experionyx.stress.spec import FAMILIES as STRESS_FAMILIES
from experionyx.stress.spec import StressDesign

DEFAULT_WORKSPACE = ".experionyx"
REGISTRY_FILE = "registry.sqlite"
EXPERIMENTS_DIR = "experiments"


def _json(value: object) -> dict[str, object]:
    data = to_jsonable(value)
    if not isinstance(data, dict):
        raise TypeError(f"{type(value).__name__} is not a dataclass")
    return data


def _record(entity: Entity) -> dict[str, object]:
    return {"id": entity.id, **entity.to_dict()}


def _dump(document: object) -> None:
    print(json.dumps(document, indent=2, sort_keys=True))


def _open(workspace: str) -> SqliteRegistry:
    path = Path(workspace) / REGISTRY_FILE
    if not path.is_file():
        raise ExperionyxError(f"no registry at {path} (use --workspace to point at a workspace)")
    return SqliteRegistry(path)


def _executor(registry: SqliteRegistry, workspace: str) -> Executor:
    store = LocalArtifactStore(Path(workspace) / EXPERIMENTS_DIR)
    return Executor(
        registry,
        store,
        source_root=Path.cwd(),
        adapters=default_registries(entry_points=True),
        inputs_root=Path(workspace),
    )


def _cmd_info(_: argparse.Namespace) -> int:
    print(f"experionyx {__version__}")
    print(f"python {platform.python_version()} ({platform.python_implementation()})")
    print(f"platform {platform.platform()}")
    print(f"registry schema {DB_SCHEMA_VERSION}")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        print(f"workspace: {args.workspace}")
        print(f"registry schema: {DB_SCHEMA_VERSION}")
        print(f"investigations: {len(reg.find(Investigation))}")
        print(f"experiments: {len(reg.find(Experiment))}")
        print(f"configurations: {len(reg.find(ConfigurationRef))}")
        print(f"observations: {len(reg.find(Observation))}")
        print(f"artifacts: {len(reg.find(Artifact))}")
        for status, runs in run_states(reg).items():
            print(f"runs {status}: {len(runs)}")
    return 0


def _cmd_experiment(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        experiment = reg.get(Experiment, args.id)
        _dump(
            {
                "experiment": _record(experiment),
                "configuration": _record(reg.get(ConfigurationRef, experiment.configuration_id)),
                "runs": [
                    {"id": r.id, "status": r.status.value, "seed": r.seed, "attempt": r.attempt}
                    for r in reg.find(Run, experiment_id=experiment.id)
                ],
            }
        )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        run = reg.get(Run, args.id)
        outcome = reg.find(RunOutcome, run_id=run.id)
        _dump(
            {
                "run": _record(run),
                "outcome": _record(outcome[0]) if outcome else None,
                "observations": [_record(o) for o in reg.find(Observation, run_id=run.id)],
                "artifacts": [_record(a) for a in reg.find(Artifact, run_id=run.id)],
            }
        )
    return 0


def _cmd_provenance(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        run = reg.get(Run, args.id)
        found = reg.find(Provenance, run_id=run.id)
        if not found:
            raise ExperionyxError(f"run {run.id} has no provenance (it never started)")
        _dump({**_record(found[0]), "fingerprint": found[0].fingerprint})
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        run = reg.get(Run, args.id)
        store = LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR)
        bad = 0
        for artifact in reg.find(Artifact, run_id=run.id):
            try:
                store.verify(run, artifact)
                print(f"OK       {artifact.path} {artifact.digest}")
            except ArtifactIntegrityError as exc:
                bad += 1
                print(f"MISMATCH {artifact.path}: {exc}")
    return 1 if bad else 0


def _options(pairs: list[str]) -> dict[str, object]:
    """`--option key=value` (value parsed as JSON when possible, else kept as a string)."""
    out: dict[str, object] = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not (key and sep):
            raise ExperionyxError(f"--option expects key=value, got {pair!r}")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def _cmd_init(args: argparse.Namespace) -> int:
    Path(args.workspace).mkdir(parents=True, exist_ok=True)
    SqliteRegistry(Path(args.workspace) / REGISTRY_FILE).close()
    print(f"workspace ready: {args.workspace}")
    return 0


def _cmd_adapters_list(_: argparse.Namespace) -> int:
    registries = default_registries(entry_points=True)
    for kind, registry in (("model", registries.models), ("dataset", registries.datasets)):
        for status in registry.status():
            if status.info is None:
                print(f"{kind:8} {status.name:10} UNAVAILABLE  {status.reason}")
            else:
                i = status.info
                print(
                    f"{kind:8} {i.name:10} {i.version:8} {i.framework:14} "
                    + ",".join(i.capabilities)
                )
    return 0


def _cmd_adapters_inspect(args: argparse.Namespace) -> int:
    registries = default_registries(entry_points=True)
    found: dict[str, object] = {}
    for kind, registry in (("model", registries.models), ("dataset", registries.datasets)):
        for status in registry.status():
            if status.name == args.name:
                found[kind] = (
                    {"available": False, "reason": status.reason}
                    if status.info is None
                    else {"available": True, **_json(status.info)}
                )
    if not found:
        raise ExperionyxError(f"no adapter named {args.name!r}")
    _dump(found)
    return 0


def _cmd_model_inspect(args: argparse.Namespace) -> int:
    if args.target.startswith("mdl_"):
        with _open(args.workspace) as reg:
            _dump(_record(reg.get(RegisteredModel, args.target)))
        return 0
    adapter_cls = default_registries(entry_points=True).models.resolve(_need_adapter(args))
    adapter = adapter_cls.load(
        args.target,
        version=args.version,
        device=DeviceKind(args.device),
        options=_options(args.option),
    )
    _dump(
        {
            "device": _json(adapter.device),
            "load_seconds": adapter.load_seconds,
            "metadata": _json(adapter.metadata()),
        }
    )
    return 0


def _cmd_model_register(args: argparse.Namespace) -> int:
    options = _options(args.option)
    adapter_cls = default_registries(entry_points=True).models.resolve(args.adapter)
    adapter = adapter_cls.load(
        args.source, version=args.version, device=DeviceKind.CPU, options=options
    )
    record = RegisteredModel(args.name, adapter.metadata(), datetime.now(UTC), args.source, options)
    with _open(args.workspace) as reg:
        reg.add(record)
    print(f"registered model {record.id}\nfingerprint: {record.fingerprint}")
    return 0


def _cmd_dataset_inspect(args: argparse.Namespace) -> int:
    if args.target.startswith("dst_"):
        with _open(args.workspace) as reg:
            _dump(_record(reg.get(RegisteredDataset, args.target)))
        return 0
    adapter_cls = default_registries(entry_points=True).datasets.resolve(_need_adapter(args))
    adapter = adapter_cls.load(args.target, version=args.version, options=_options(args.option))
    _dump(_json(adapter.metadata(deep=args.deep)))
    return 0


def _cmd_dataset_register(args: argparse.Namespace) -> int:
    options = _options(args.option)
    adapter_cls = default_registries(entry_points=True).datasets.resolve(args.adapter)
    adapter = adapter_cls.load(args.source, version=args.version, options=options)
    record = RegisteredDataset(
        args.name, adapter.metadata(), datetime.now(UTC), args.source, options
    )
    with _open(args.workspace) as reg:
        reg.add(record)
    print(f"registered dataset {record.id}\nfingerprint: {record.fingerprint}")
    return 0


def _need_adapter(args: argparse.Namespace) -> str:
    if not args.adapter:
        raise ExperionyxError("--adapter is required unless inspecting a registered ID")
    return str(args.adapter)


def _cmd_demo(args: argparse.Namespace) -> int:
    Path(args.workspace).mkdir(parents=True, exist_ok=True)
    return _report(run_demo(args.name, Path(args.workspace), args.seed))


def _cmd_metrics_list(args: argparse.Namespace) -> int:
    registry = default_metric_registry()
    for m in (
        registry.for_task(TaskType(args.task)) if args.task else map(registry.get, registry.ids())
    ):
        tasks = ",".join(sorted(t.value for t in m.tasks))
        needs = ",".join(sorted(r.value for r in m.requires))
        print(f"{m.id:22} v{m.version} {tasks:15} requires={needs:14} scale={m.scale}  {m.name}")
    return 0


def _evaluation_config(args: argparse.Namespace) -> EvaluationConfig:
    if args.config:
        config = EvaluationConfig.from_dict(
            json.loads(Path(args.config).read_text(encoding="utf-8"))
        )
    else:
        config = EvaluationConfig()
    changes: dict[str, object] = {}
    if args.split is not None:
        changes["split"] = args.split
    if args.batch_size is not None:
        changes["batch_size"] = args.batch_size
    if args.metric:
        changes["metrics"] = tuple(args.metric)
    if args.score_source is not None:
        changes["score_source"] = ScoreSource(args.score_source)
    if args.bins is not None:
        changes["calibration"] = dataclasses.replace(config.calibration, bins=args.bins)
    if args.bootstrap_resamples is not None:
        changes["bootstrap"] = dataclasses.replace(
            config.bootstrap,
            enabled=args.bootstrap_resamples > 0,
            resamples=max(1, args.bootstrap_resamples),
        )
    return dataclasses.replace(config, **changes)  # type: ignore[arg-type]


def _run_evaluation(args: argparse.Namespace, model_id: str, dataset_id: str) -> ExecutionResult:
    config = _evaluation_config(args)
    with _open(args.workspace) as reg:
        model = reg.get(RegisteredModel, model_id)
        dataset = reg.get(RegisteredDataset, dataset_id)
        now = datetime.now(UTC)
        inv = Investigation(
            "baseline-evaluation",
            "How does each registered model behave on its baseline data?",
            now,
        )
        cfg = ConfigurationRef(config.to_parameters())
        exp = Experiment(
            inv.id,
            f"evaluate {model.name} {model.version} on {dataset.name} {dataset.version}",
            "Baseline evaluation: measurements only, no causal claim",
            model.ref(),
            dataset.ref(),
            cfg.id,
            now,
        )
        for entity in (inv, cfg, exp):
            if not reg.exists(type(entity), entity.id):
                reg.add(entity)
        if reg.get(Experiment, exp.id).status is ExperimentStatus.DRAFT:
            reg.update_status(exp.with_status(ExperimentStatus.READY))
        return _executor(reg, args.workspace).execute(
            exp.id, resolve_procedure(PROCEDURE), seed=args.seed, procedure_name=PROCEDURE
        )


def _summary(evaluation: EvaluationResult) -> dict[str, object]:
    return {
        "run": evaluation.context.run_id,
        "task": evaluation.task.value,
        "n_samples": evaluation.n_samples,
        "metrics": {
            m.metric_id: (
                m.value if m.status.value == "COMPUTED" else f"{m.status.value}: {m.reason}"
            )
            for m in evaluation.metrics
            if m.value is not None or m.status.value != "COMPUTED"
        },
        "findings": [_json(f) for f in evaluation.findings],
        "profile": _json(evaluation.profile) if evaluation.profile else None,
        "warnings": list(evaluation.warnings),
    }


def _cmd_evaluate(args: argparse.Namespace) -> int:
    result = _run_evaluation(args, args.model, args.dataset)
    code = _report(result)
    if result.status is RunStatus.COMPLETED:
        print(f"inspect with: experionyx evaluation inspect {result.run.id}")
    return code


def _cmd_autopsy(args: argparse.Namespace) -> int:
    result = _run_evaluation(args, args.model_id, args.dataset)
    if result.status is not RunStatus.COMPLETED:
        return _report(result)
    with _open(args.workspace) as reg:
        evaluation = _load_evaluation(reg, args.workspace, result.run.id)
    _dump(
        {
            "run": result.run.id,
            "profile": _json(evaluation.profile),
            "findings": [_json(f) for f in evaluation.findings],
        }
    )
    return 0


def _load_evaluation(reg: SqliteRegistry, workspace: str, run_id: str) -> EvaluationResult:
    run = reg.get(Run, run_id)
    found = [a for a in reg.find(Artifact, run_id=run.id) if a.path == "evaluation/evaluation.json"]
    if not found:
        raise ExperionyxError(f"run {run.id} has no evaluation artifact")
    store = LocalArtifactStore(Path(workspace) / EXPERIMENTS_DIR)
    store.verify(run, found[0])  # the stored evaluation must still match its recorded digest
    text = (store.run_dir(run) / "artifacts" / found[0].path).read_text(encoding="utf-8")
    result: EvaluationResult = from_jsonable(EvaluationResult, json.loads(text))
    return result


def _cmd_evaluation_inspect(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        evaluation = _load_evaluation(reg, args.workspace, args.run)
    _dump(_json(evaluation) if args.full else _summary(evaluation))
    return 0


def _cmd_evaluation_compare(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        a = _load_evaluation(reg, args.workspace, args.run_a)
        b = _load_evaluation(reg, args.workspace, args.run_b)
    _dump(_json(compare_evaluations(a, b)))
    return 0


def _cmd_faults_list(_: argparse.Namespace) -> int:
    for ft in default_fault_registry().all():
        reqs = ",".join(sorted(r.value for r in ft.requires)) or "-"
        state = "" if ft.implemented else "  [NOT IMPLEMENTED: reserved]"
        head = f"{ft.name:22} v{ft.version} {ft.category.value:20} {ft.target.value:6}"
        print(f"{head} requires={reqs:26} stochastic={ft.stochastic}{state}")
    return 0


def _cmd_fault_inspect(args: argparse.Namespace) -> int:
    ft = default_fault_registry().resolve(args.name)
    params = []
    for f in dataclasses.fields(ft.params_type):
        default = None if f.default is dataclasses.MISSING else f.default
        params.append(
            {
                "name": f.name,
                "type": getattr(f.type, "__name__", str(f.type)),
                "required": f.default is dataclasses.MISSING,
                "default": default,
                "doc": ft.param_docs.get(f.name, ""),
            }
        )
    _dump(
        {
            "name": ft.name,
            "version": ft.version,
            "category": ft.category.value,
            "target": ft.target.value,
            "description": ft.description,
            "requires": sorted(r.value for r in ft.requires),
            "stochastic": ft.stochastic,
            "implemented": ft.implemented,
            "sweepable": list(ft.sweepable()),
            "parameters": params,
        }
    )
    return 0


def _spec_from_args(args: argparse.Namespace, fr: FaultRegistry) -> FaultSpec:
    if args.spec_file:
        return fr.from_dict(json.loads(Path(args.spec_file).read_text(encoding="utf-8")))
    if not args.type:
        raise ExperionyxError("give --type (with --param key=value ...) or --spec-file")
    if args.scope_fraction is not None and args.scope_class is not None:
        raise ExperionyxError("choose one scope: --scope-fraction or --scope-class")
    scope = FaultScope()
    if args.scope_fraction is not None:
        scope = FaultScope(ScopeKind.RANDOM_SUBSET, fraction=args.scope_fraction)
    elif args.scope_class is not None:
        label = _options([f"x={args.scope_class}"])["x"]
        scope = FaultScope(
            ScopeKind.CLASS, label=label if isinstance(label, int | str) else str(label)
        )
    return fr.make(args.type, seed=args.seed, scope=scope, **_options(args.param))  # type: ignore[arg-type]


def _print_fault_result(reg: SqliteRegistry, workspace: str, result: FaultExperimentResult) -> int:
    store = LocalArtifactStore(Path(workspace) / EXPERIMENTS_DIR)
    print(f"fault experiment: {result.fault_experiment.id} ({result.fault_experiment.status})")
    print(f"baseline run:     {result.baseline_run_id}")
    print(f"analysis run:     {result.analysis_run_id} ({result.analysis_status})")
    statuses: dict[str, int] = {}
    for t in result.trials:
        statuses[t.status.value] = statuses.get(t.status.value, 0) + 1
    print(f"trials:           {statuses}")
    if result.analysis_status is not RunStatus.COMPLETED:
        return 1
    analysis = load_analysis(reg, store, result.fault_experiment.id)
    print(
        f"primary metric:   {analysis.primary_metric} ({analysis.primary_direction}), baseline {analysis.baseline_value}"
    )
    for row in summary_rows(analysis):
        value = "" if row["value"] is None else f"{row['parameter']}={row['value']:g}"
        det = row["deterioration_mean"]
        print(
            f"  {value:16} trials {row['trials']:6} faulted_mean={row['faulted_mean']}  deterioration={'n/a' if det is None else format(det, '.6g')}  {row['classification']}"
        )
    print(f"inspect with: experionyx fault experiment inspect {result.fault_experiment.id}")
    return 0


def _run_fault(args: argparse.Namespace, sweep: SweepSpec | None, seeds: tuple[int, ...]) -> int:
    fr = default_fault_registry()
    spec = _spec_from_args(args, fr)
    evaluation = _evaluation_config(args)
    limits = FaultLimits(max_failed_trials=args.max_failed_trials)
    design = FaultDesign(
        spec.to_dict(),
        evaluation,
        seeds=seeds,
        sweep=sweep,
        primary_metric=args.primary_metric,
        limits=limits,
    )
    with _open(args.workspace) as reg:
        store = LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR)
        name = args.name or f"{spec.type} {'sweep ' + sweep.parameter if sweep else 'run'}"
        result = run_fault_experiment(
            reg,
            store,
            _executor(reg, args.workspace),
            model_id=args.model,
            dataset_id=args.dataset,
            base_spec=spec,
            fault_registry=fr,
            design=design,
            name=name,
            source_root=Path.cwd(),
            baseline_run_id=args.baseline_run,
        )
        return _print_fault_result(reg, args.workspace, result)


def _cmd_fault_run(args: argparse.Namespace) -> int:
    return _run_fault(args, None, (args.seed,))


def _cmd_fault_sweep(args: argparse.Namespace) -> int:
    name, sep, raw = args.sweep.partition("=")
    if not (name and sep and raw):
        raise ExperionyxError("--sweep expects parameter=v1,v2,...")
    try:
        values = tuple(float(x) for x in raw.split(","))
        seeds = tuple(int(x) for x in args.seeds.split(","))
    except ValueError as exc:
        raise ExperionyxError(f"invalid sweep values or seeds: {exc}") from exc
    return _run_fault(args, SweepSpec(name, values), seeds)


def _cmd_fault_compare(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        store = LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR)
        base, treat = (
            load_evaluation(reg, store, args.baseline_run),
            load_evaluation(reg, store, args.treatment_run),
        )
        fault = read_artifact(reg, store, args.treatment_run, "fault/fault.json")
    _dump(
        {
            "baseline_run": args.baseline_run,
            "treatment_run": args.treatment_run,
            "fault": {
                k: fault[k]
                for k in (
                    "fault_id",
                    "fault",
                    "seed",
                    "scope",
                    "affected_samples",
                    "total_samples",
                    "timings_seconds",
                )
            }
            if isinstance(fault, dict)
            else None,
            "degradation": [_json(d) for d in measure_degradation(base, treat)],
            "latency": _json(measure_latency(base, treat)),
            "note": "deterioration > 0 means worse (direction-aware); this is a measurement, not a verdict",
        }
    )
    return 0


def _cmd_fault_experiment_inspect(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        store = LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR)
        fx = reg.get(FaultExperiment, args.id)
        trials = reg.find(FaultTrial, fault_experiment_id=fx.id)
        document: dict[str, object] = {
            "fault_experiment": _record(fx),
            "trials": [
                {
                    "point": t.point_index,
                    "repeat": t.repeat_index,
                    "parameter": t.parameter_name,
                    "value": t.parameter_value,
                    "seed": t.seed,
                    "status": t.status.value,
                    "fault_id": t.fault_id,
                    "treatment_run": t.treatment_run_id,
                    "reason": t.reason,
                }
                for t in sorted(trials, key=lambda x: (x.point_index, x.repeat_index))
            ],
        }
        try:
            analysis = load_analysis(reg, store, fx.id)
            document.update(
                {
                    "analysis_run": analysis_run_id(reg, fx.id),
                    "primary_metric": analysis.primary_metric,
                    "summary": summary_rows(analysis),
                    "warnings": list(analysis.warnings),
                }
            )
            if args.full:
                document["analysis"] = _json(analysis)
        except ExperionyxError as exc:
            document["analysis_error"] = str(exc)
    _dump(document)
    return 0


# -- failure discovery -----------------------------------------------------------------------------


def _discovery_config(path: str | None) -> DiscoveryConfig:
    if path is None:
        return DiscoveryConfig()
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExperionyxError(f"cannot read --config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ExperionyxError("--config must be a JSON object")
    return DiscoveryConfig.from_dict(data)


def _mode_row(m: FailureMode) -> dict[str, object]:
    meas = m.structured.get("measurements")
    size = meas.get("size") if isinstance(meas, Mapping) else None
    return {
        "id": m.id,
        "status": m.status.value,
        "category": m.category.value,
        "title": m.title,
        "signals": size,
        "cluster_id": m.cluster_id,
        "investigation_id": m.investigation_id,
    }


def _find_modes(reg: SqliteRegistry, args: argparse.Namespace) -> list[FailureMode]:
    filters = {
        k: v
        for k, v in (
            ("investigation_id", args.investigation),
            ("status", args.status),
            ("category", args.category),
        )
        if v
    }
    return reg.find(FailureMode, **filters)


def _cmd_failures_list(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        _dump(
            {
                "failure_modes": [_mode_row(m) for m in _find_modes(reg, args)],
                "note": "status DISCOVERED/CANDIDATE/SUPPORTED are machine-assigned from configured criteria; only CONFIRMED involved a person",
            }
        )
    return 0


def _cmd_failure_candidates(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        rows = []
        for m in _find_modes(reg, args):
            if m.status in (
                FailureStatus.DISCOVERED,
                FailureStatus.CANDIDATE,
                FailureStatus.SUPPORTED,
            ):
                rows.append({**_mode_row(m), "criteria": to_jsonable(m.structured["criteria"])})
        _dump({"candidates": rows})
    return 0


def _cmd_failure_inspect(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        prefix = args.id.split("_", 1)[0]
        cls = {"fmd": FailureMode, "fcl": FailureCluster, "fsg": FailureSignal}.get(prefix)
        if cls is None:
            raise ExperionyxError(
                "expected a failure mode (fmd_), cluster (fcl_) or signal (fsg_) ID"
            )
        _dump(_record(reg.get(cls, args.id)))
    return 0


def _cmd_failure_cluster(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        if args.id:
            cl = reg.get(FailureCluster, args.id)
            members = [reg.get(FailureSignal, i) for i in cl.signal_ids]
            _dump(
                {
                    "cluster": _record(cl),
                    "members": [
                        {
                            "id": s.id,
                            "run_id": s.run_id,
                            "kind": s.signal_kind.value,
                            "signature": s.signature,
                            "magnitude": s.magnitude,
                            "sample_count": s.sample_count,
                        }
                        for s in members
                    ],
                }
            )
        else:
            filt = {"investigation_id": args.investigation} if args.investigation else {}
            _dump(
                {
                    "clusters": [
                        {
                            "id": c.id,
                            "size": len(c.signal_ids),
                            "algorithm": c.algorithm,
                            "compactness": c.metrics.get("compactness"),
                            "stability": c.metrics.get("stability"),
                        }
                        for c in reg.find(FailureCluster, **filt)
                    ]
                }
            )
    return 0


def _cmd_failure_evidence(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        reg.get(FailureMode, args.id)
        _dump(
            {
                "mode_id": args.id,
                "evidence": [
                    _record(e) for e in reg.find(FailureEvidence, failure_mode_id=args.id)
                ],
            }
        )
    return 0


def _infer_investigation(reg: SqliteRegistry, args: argparse.Namespace) -> str:
    """The investigation a discovery is homed in. Sources may come from several investigations;
    then the home must be chosen explicitly."""
    if args.investigation:
        reg.get(Investigation, args.investigation)
        return str(args.investigation)
    found = {reg.get(FaultExperiment, i).investigation_id for i in args.fault_experiment}
    found |= {reg.get(Experiment, reg.get(Run, i).experiment_id).investigation_id for i in args.run}
    if len(found) != 1:
        raise ExperionyxError(
            f"sources span investigations {sorted(found)}; choose the home with --investigation"
        )
    return found.pop()


def _cmd_failure_discover(args: argparse.Namespace) -> int:
    cfg = _discovery_config(args.config)
    with _open(args.workspace) as reg:
        store = LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR)
        res = run_discovery(
            reg,
            store,
            _executor(reg, args.workspace),
            _infer_investigation(reg, args),
            args.run,
            args.fault_experiment,
            cfg,
            seed=args.seed,
        )
        ids: list[str] = []
        if res.status is RunStatus.COMPLETED:  # exactly the modes THIS discovery produced
            listed = read_artifact(reg, store, res.run_id, "failure-candidates.json")
            ids = [str(m["id"]) for m in listed["modes"]] if isinstance(listed, dict) else []
        _dump(
            {
                "discovery_run": res.run_id,
                "experiment": res.experiment_id,
                "status": res.status.value,
                "config_hash": cfg.config_hash,
                "failure_modes": [_mode_row(reg.get(FailureMode, i)) for i in ids],
            }
        )
        return 0 if res.status.value == "COMPLETED" else 1


def _cmd_failure_reproduce(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        store = LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR)
        r = reproduce(
            reg, store, _executor(reg, args.workspace), args.id, _discovery_config(args.config)
        )
        _dump(
            {
                "mode_id": r.mode_id,
                "attempted": r.attempted,
                "passed_count": r.passed_count,
                "passed": r.passed,
                "runs_not_replayed": r.runs_not_replayed,
                "records": to_jsonable(r.records),
                "note": "reproduction does not change the mode's status; confirmation is a separate, explicit step",
            }
        )
        return 0 if r.passed else 1


def _cmd_failure_confirm(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        _dump(_mode_row(confirm(reg, args.id, args.by, args.reason)))
    return 0


def _cmd_failure_set_status(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        _dump(_mode_row(change_status(reg, args.id, FailureStatus(args.to), args.by, args.reason)))
    return 0


def _cmd_failure_graph(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        _dump(graph(reg, args.id))
    return 0


# -- interaction analysis ---------------------------------------------------------------------------------


def _load_spec(path: str) -> InteractionSpec:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExperionyxError(f"cannot read spec {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ExperionyxError("a spec file must be a JSON object")
    if (
        "cells" not in data
    ):  # friendly form: {"control": [...], "a": [...], "b": [...], "ab": [...], ...}
        cells = {k.upper(): data.pop(k) for k in ("control", "a", "b", "ab", "ba") if k in data}
        data = {"cells": cells, **data}
    return InteractionSpec.from_dict(data)


def _emit(doc: dict[str, object], text: str, args: argparse.Namespace) -> None:
    if getattr(args, "format", "json") == "text":
        print(text)
    else:
        _dump(doc)


def _interaction_row(a: InteractionAnalysis) -> dict[str, object]:
    return {
        "id": a.id,
        "status": a.status.value,
        "primary_metric": a.primary_metric,
        "primary_class": a.primary_class.value,
        "spec_id": a.spec_id,
        "run_id": a.run_id,
        "statement": a.summary["statement"],
    }


def _cmd_interaction_validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.spec)
    with _open(args.workspace) as reg:
        store = LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR)
        report = validate_report(reg, store, spec)
    lines = [f"spec {report['spec_id']}: {'VALID' if report['valid'] else 'REFUSED'}"]
    issues: Any = report.get("issues", [])
    checks: Any = report.get("checks", [])
    lines += [
        f"  - {i['code']}: required {i['required']}; found {i['found']}; {i['why']}" for i in issues
    ]
    lines += [f"  + {c}" for c in checks]
    _emit(report, "\n".join(lines), args)
    return 0 if report["valid"] else 1


def _cmd_interaction_analyze(args: argparse.Namespace) -> int:
    spec = _load_spec(args.spec)
    with _open(args.workspace) as reg:
        store = LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR)
        inv = (
            args.investigation
            or reg.get(Experiment, reg.get(Run, spec.control[0]).experiment_id).investigation_id
        )
        try:
            res = run_interaction(
                reg, store, _executor(reg, args.workspace), inv, spec, seed=args.seed
            )
        except DesignRefusal as exc:  # refused BEFORE any run is created
            print(
                "error: design refused; nothing was analyzed and nothing was recorded",
                file=sys.stderr,
            )
            for issue in exc.issues:
                print(f"  - {issue}", file=sys.stderr)
            return 2
        doc: dict[str, object] = {
            "analysis_run": res.run_id,
            "status": res.status.value,
            "analysis_id": res.analysis_id,
        }
        if res.analysis_id:
            a = reg.get(InteractionAnalysis, res.analysis_id)
            doc["interaction"] = _interaction_row(a)
        shown = doc.get("interaction")
        said = shown["statement"] if isinstance(shown, dict) else ""
        _emit(doc, f"{res.status.value}: {said}", args)
        return 0 if res.status is RunStatus.COMPLETED else 1


def _ireg(args: argparse.Namespace, reg: SqliteRegistry) -> InteractionRegistry:
    return InteractionRegistry(reg, LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR))


def _cmd_interaction_list(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        found = _ireg(args, reg).search(
            fault=args.fault,
            metric=args.metric,
            status=args.status,
            failure_mode=args.failure_mode,
            experiment=args.experiment,
            dataset=args.dataset,
            model=args.model,
            primary_class=args.klass,
            investigation=args.investigation,
        )
        _emit(
            {
                "interactions": [_interaction_row(a) for a in found],
                "note": "labels describe an observed contrast under a design; they are not causal claims",
            },
            "\n".join(
                f"{a.id} {a.status.value:20} {a.primary_class.value:28} {a.primary_metric}"
                for a in found
            ),
            args,
        )
    return 0


def _cmd_interaction_inspect(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        ir = _ireg(args, reg)
        a = ir.get(args.id)
        doc: dict[str, object] = {
            **_record(a),
            "effects": [
                {
                    "level": e.level.value,
                    "measure": e.measure,
                    "status": e.effect_status.value,
                    "class": e.interaction_class.value,
                    "interaction_contrast": (e.record["derived"] or {}).get("interaction_contrast")
                    if isinstance(e.record["derived"], Mapping)
                    else None,
                }
                for e in ir.effects(a.id)
            ],
            "evidence": [
                {"kind": x.evidence_kind.value, "summary": x.summary} for x in ir.evidence(a.id)
            ],
        }
        if args.full:
            doc["effect_records"] = [_record(e) for e in ir.effects(a.id)]
            doc["raw_trials"] = to_jsonable(ir.raw_trials(a.id))
        _emit(
            doc,
            f"{a.id} [{a.status.value}] {a.primary_class.value}\n{a.summary['statement']}",
            args,
        )
    return 0


def _cmd_interaction_failures(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        ir = _ireg(args, reg)
        ir.get(args.id)
        try:
            doc = ir.artifact(args.id, "failure_modes")
        except ExperionyxError:
            doc = {
                "status": "NOT_AVAILABLE",
                "reason": "no failure-mode analysis was requested (the spec had no discovery_run_id)",
            }
        shown_doc = to_jsonable(doc)
        _dump(shown_doc if isinstance(shown_doc, dict) else {"data": shown_doc})
    return 0


def _cmd_interaction_replay(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        out = replay_check(
            reg,
            LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR),
            _executor(reg, args.workspace),
            args.id,
        )
        _dump(out)
        return 0 if out["deterministic"] else 1


def _cmd_interaction_export(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        ir = _ireg(args, reg)
        a = ir.get(args.id)
        bundle = {
            "analysis": _record(a),
            "provenance": ir.provenance(a.id),
            "effects": [_record(e) for e in ir.effects(a.id)],
            "evidence": [_record(e) for e in ir.evidence(a.id)],
            "artifacts": {
                name: to_jsonable(ir.artifact(a.id, name))
                for name in (
                    "spec",
                    "design_validation",
                    "trials",
                    "effects",
                    "bootstrap",
                    "summary",
                )
            },
        }
        text = json.dumps(to_jsonable(bundle), indent=2, sort_keys=True)
        if args.out:
            Path(args.out).write_text(text + "\n", encoding="utf-8")
            print(f"wrote {args.out}")
        else:
            print(text)
    return 0


def _cmd_interaction_related(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        _dump(
            {
                "analysis_id": args.id,
                "related": [
                    {
                        "analysis_id": r.analysis_id,
                        "relation": r.relation,
                        "differences": list(r.differences),
                    }
                    for r in _ireg(args, reg).related(args.id)
                ],
                "note": "pointers for a human; nothing is merged and parameter differences stay visible",
            }
        )
    return 0


def _cmd_interaction_reproduce(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        r = check_reproduction(
            reg, args.id, args.replicate, abs_tol=args.abs_tol, rel_tol=args.rel_tol
        )
        _dump(
            {
                "original": r.original_id,
                "replicate": r.replicate_id,
                "passed": r.passed,
                "checks": to_jsonable(r.checks),
            }
        )
        return 0 if r.passed else 1


def _cmd_interaction_confirm(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        _dump(_interaction_row(confirm_by_review(reg, args.id, args.by, args.reason)))
    return 0


def _cmd_interaction_set_status(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        _dump(
            _interaction_row(
                interaction_change_status(
                    reg, args.id, InteractionStatus(args.to), args.by, args.reason
                )
            )
        )
    return 0


# -- reliability profiles ---------------------------------------------------------------------------------


def _load_profile_spec(path: str) -> ProfileSpec:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExperionyxError(f"cannot read profile spec {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ExperionyxError("a profile spec file must be a JSON object")
    return ProfileSpec.from_dict(data)


def _profile_row(p: ReliabilityProfile) -> dict[str, object]:
    return {
        "id": p.id,
        "scope": p.scope.value,
        "model_fingerprint": p.model_fingerprint,
        "dataset_fingerprint": p.dataset_fingerprint,
        "split": p.split,
        "spec_id": p.spec_id,
        "dimension_status": to_jsonable(p.dimension_status),
    }


def _rreg(args: argparse.Namespace, reg: SqliteRegistry) -> ReliabilityProfileRegistry:
    return ReliabilityProfileRegistry(
        reg, LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR)
    )


def _profile_text(doc: dict[str, Any], pid: str) -> str:
    lines = [f"{pid} [{doc['scope']}]", "dimension status (no overall score):"]
    lines += [f"  {d:24} {s}" for d, s in doc["dimension_status"].items()]
    lines += ["observations (INTERPRETED, deterministic templates):"]
    lines += [f"  - {x['text']}" for x in doc["interpreted"]["statements"]]
    return "\n".join(lines)


def _cmd_reliability_profile(args: argparse.Namespace) -> int:
    spec = _load_profile_spec(args.spec)
    with _open(args.workspace) as reg:
        store = LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR)
        inv = (
            args.investigation
            or reg.get(Experiment, reg.get(Run, spec.baseline_run).experiment_id).investigation_id
        )
        try:
            res = run_profile(reg, store, _executor(reg, args.workspace), inv, spec, seed=args.seed)
        except ProfileRefusal as exc:  # refused BEFORE any run is created
            print(
                "error: profile refused; nothing was built and nothing was recorded",
                file=sys.stderr,
            )
            for issue in exc.issues:
                print(f"  - {issue}", file=sys.stderr)
            return 2
        doc: dict[str, object] = {
            "profile_run": res.run_id,
            "status": res.status.value,
            "profile_id": res.profile_id,
        }
        text = res.status.value
        if res.profile_id:
            body = _rreg(args, reg).document(res.profile_id)
            doc["profile"] = _profile_row(reg.get(ReliabilityProfile, res.profile_id))
            text = _profile_text(body, res.profile_id)
        _emit(doc, text, args)
        return 0 if res.status is RunStatus.COMPLETED else 1


def _cmd_reliability_list(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        dim = (
            (args.dimension, args.dimension_status)
            if args.dimension and args.dimension_status
            else None
        )
        found = _rreg(args, reg).search(
            model=args.model,
            dataset=args.dataset,
            scope=args.scope,
            split=args.split,
            evaluation=args.evaluation,
            investigation=args.investigation,
            ref=args.ref,
            dimension_status=dim,
        )
        _emit(
            {
                "profiles": [_profile_row(p) for p in found],
                "note": "profiles are evidence summaries; they are not ranked",
            },
            "\n".join(
                f"{p.id} {p.scope.value:26} model={p.model_fingerprint[:16]} dataset={p.dataset_fingerprint[:16]}"
                for p in found
            ),
            args,
        )
    return 0


def _cmd_reliability_inspect(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        rr = _rreg(args, reg)
        p = rr.get(args.id)
        doc = rr.document(p.id)
        if args.dimension:
            _dump(
                {
                    "profile_id": p.id,
                    "dimension": args.dimension,
                    **doc["dimensions"][args.dimension],
                }
            )
            return 0
        out: dict[str, object] = {
            **_profile_row(p),
            "interpreted": doc["interpreted"],
            "provenance_fingerprint": p.provenance_fingerprint,
        }
        if args.full:
            out["document"] = doc
        _emit(out, _profile_text(doc, p.id), args)
    return 0


def _cmd_reliability_evidence(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        rr = _rreg(args, reg)
        refs = rr.references(args.id, dimension=args.dimension)
        _dump(
            {
                "profile_id": args.id,
                "provenance": to_jsonable(rr.provenance(args.id)),
                "references": [
                    {
                        "dimension": r.dimension.value,
                        "kind": r.ref_kind.value,
                        "id": r.ref_id,
                        "note": r.note,
                    }
                    for r in refs
                ],
                "artifacts": [
                    {"path": a.path, "id": a.id, "digest": a.digest} for a in rr.artifacts(args.id)
                ],
            }
        )
    return 0


def _cmd_reliability_compare(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        try:
            out = _rreg(args, reg).compare(args.a, args.b)
        except ProfileRefusal as exc:
            print("error: profiles are not comparable; nothing was compared", file=sys.stderr)
            for issue in exc.issues:
                print(f"  - {issue}", file=sys.stderr)
            return 2
        shown: Any = to_jsonable(out)
        _emit(
            shown,
            f"compared {args.a} with {args.b} (raw differences b - a; no winner)\n"
            + "\n".join(f"  {d}: {v['a']} -> {v['b']}" for d, v in out["dimension_status"].items()),
            args,
        )
    return 0


def _cmd_reliability_replay(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        out = profile_replay_check(
            reg,
            LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR),
            _executor(reg, args.workspace),
            args.id,
        )
        _dump(out)
        return 0 if out["deterministic"] else 1


# -- robustness benchmarks --------------------------------------------------------------------------------


def _load_benchmark_spec(path: str) -> BenchmarkSpec:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExperionyxError(f"cannot read benchmark spec {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ExperionyxError("a benchmark spec file must be a JSON object")
    return BenchmarkSpec.from_dict(data)


def _breg(args: argparse.Namespace, reg: SqliteRegistry) -> BenchmarkRegistry:
    return BenchmarkRegistry(reg, LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR))


def _bench_row(b: Benchmark) -> dict[str, object]:
    return {
        "id": b.id,
        "name": b.name,
        "version": b.version,
        "spec_id": b.spec_id,
        "protocol_hash": b.protocol_hash,
        "engine_version": b.engine_version,
        "model": b.model_record_id,
        "dataset": b.dataset_record_id,
    }


def _result_row(r: BenchmarkResult) -> dict[str, object]:
    return {
        "id": r.id,
        "benchmark_id": r.benchmark_id,
        "coverage_status": r.coverage_status.value,
        "section_status": to_jsonable(r.section_status),
        "counts": to_jsonable(r.summary["counts"]),
        "run_id": r.run_id,
    }


def _coverage_text(cov: dict[str, Any], rid: str) -> str:
    t, ff = cov["trials"], cov["fault_families"]
    lines = [
        f"{rid}: coverage {'COMPLETE' if cov['complete'] else 'INCOMPLETE'} (this counts what was executed; it is not a robustness measure)",
        f"  fault families: tested {ff['tested']} / requested {ff['requested']}",
        f"  trials: {t['completed']} completed, {t['failed']} failed, {t['skipped']} skipped, {t['not_run']} not run of {t['requested']}",
        f"  parameter points: {cov['parameter_points']}",
        f"  interactions: {cov['interactions']['analyzed']}/{cov['interactions']['requested']} analyzed",
        f"  failure modes: discovery {cov['failure_modes']['discovery']}, {cov['failure_modes']['discovered']} discovered",
    ]
    lines += [f"  ! {r}" for r in cov["incomplete_reasons"]]
    return "\n".join(lines)


def _cmd_benchmark_validate(args: argparse.Namespace) -> int:
    spec = _load_benchmark_spec(args.spec)
    with _open(args.workspace) as reg:
        report = benchmark_validation(reg, spec, default_fault_registry())
    issues: Any = report.get("issues", [])
    lines = [f"spec {report['spec_id']}: {'VALID' if report['valid'] else 'REFUSED'}"]
    lines += [
        f"  - {i['code']}: required {i['required']}; found {i['found']}; {i['why']}" for i in issues
    ]
    if report["valid"]:
        lines += [
            f"  {report['units']} experiment unit(s): {report['units_by_kind']}",
            *[f"  ! {w}" for w in report["warnings"]],
        ]
    _emit(report, "\n".join(lines), args)
    return 0 if report["valid"] else 1


def _cmd_benchmark_run(args: argparse.Namespace) -> int:
    spec = _load_benchmark_spec(args.spec)
    with _open(args.workspace) as reg:
        try:
            res = run_benchmark(
                reg,
                LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR),
                _executor(reg, args.workspace),
                spec,
                source_root=Path.cwd(),
            )
        except BenchmarkRefusal as exc:  # refused BEFORE any run is created
            print(
                "error: benchmark refused; nothing was executed and nothing was recorded",
                file=sys.stderr,
            )
            for issue in exc.issues:
                print(f"  - {issue}", file=sys.stderr)
            return 2
        doc: dict[str, object] = {
            "benchmark_id": res.benchmark_id,
            "result_id": res.result_id,
            "collect_run": res.run_id,
            "status": res.status.value if res.status else None,
            "already_run": res.already_run,
            "errors": dict(res.errors),
        }
        text = f"benchmark {res.benchmark_id or '?'}: {'already executed in this registry' if res.already_run else (res.status.value if res.status else 'not collected')}"
        code = 1
        if res.result_id:
            br = _breg(args, reg)
            row = _result_row(br.result(res.result_id))
            doc["result"] = row
            cov = br.document(res.result_id, "coverage")
            text = _coverage_text(cov, res.result_id)
            code = 0 if cov["complete"] else 3  # 3: the protocol ran but the coverage is incomplete
        _emit(doc, text, args)
        return code


def _cmd_benchmark_list(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        found = _breg(args, reg).search(
            name=args.name,
            version=args.version,
            model=args.model,
            dataset=args.dataset,
            protocol=args.protocol,
            engine=args.engine,
            fault=args.fault,
            coverage=args.coverage,
        )
        _emit(
            {
                "benchmarks": [_bench_row(b) for b in found],
                "note": "benchmarks are evidence protocols; they are not ranked",
            },
            "\n".join(
                f"{b.id} {b.name} {b.version} protocol={b.protocol_hash[7:19]}" for b in found
            ),
            args,
        )
    return 0


def _cmd_benchmark_inspect(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        br = _breg(args, reg)
        if args.id.startswith("bmk_"):
            b = br.get(args.id)
            doc: dict[str, object] = {
                **_bench_row(b),
                "spec": to_jsonable(b.spec),
                "results": [_result_row(r) for r in br.results(b.id)],
            }
            _emit(
                doc,
                f"{b.id} {b.name} {b.version}: {len(br.results(b.id))} result(s), protocol {b.protocol_hash[7:19]}",
                args,
            )
        else:
            r = br.result(args.id)
            summ = br.document(r.id, "summary")
            out: dict[str, object] = {
                **_result_row(r),
                "summary": summ,
                "provenance": to_jsonable(br.provenance(r.id)),
                "artifacts": [{"path": a.path, "id": a.id} for a in br.artifacts(r.id)],
            }
            if args.full:
                out["documents"] = br.bundle(r.id)
            text = [
                f"{r.id} coverage {r.coverage_status.value}",
                "sections (no overall score):",
                *[f"  {k:20} {v}" for k, v in summ["section_status"].items()],
                "observations (INTERPRETED, deterministic templates):",
                *[f"  - {x['text']}" for x in summ["interpreted"]["statements"]],
            ]
            _emit(out, "\n".join(text), args)
    return 0


def _cmd_benchmark_coverage(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        br = _breg(args, reg)
        r = br.resolve_result(args.id)
        cov = br.document(r.id, "coverage")
        _emit({"result_id": r.id, "coverage": cov}, _coverage_text(cov, r.id), args)
        return 0 if cov["complete"] else 3


def _cmd_benchmark_compare(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        try:
            out = _breg(args, reg).compare(args.a, args.b)
        except BenchmarkRefusal as exc:
            print("error: results are not comparable; nothing was compared", file=sys.stderr)
            for issue in exc.issues:
                print(f"  - {issue}", file=sys.stderr)
            return 2
        shown: Any = to_jsonable(out)
        _emit(
            shown,
            f"compared {args.a} with {args.b} under protocol {out['protocol_hash'][7:19]} (raw differences b - a; no winner)\n"
            + "\n".join(f"  {k}: {v['a']} -> {v['b']}" for k, v in out["section_status"].items()),
            args,
        )
    return 0


def _cmd_benchmark_replay(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        r = _breg(args, reg).resolve_result(args.id)
        out = benchmark_replay_check(
            reg,
            LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR),
            _executor(reg, args.workspace),
            r.id,
        )
        _dump(out)
        return 0 if out["deterministic"] is True else 1


def _cmd_benchmark_protocol(args: argparse.Namespace) -> int:
    spec = _load_benchmark_spec(args.spec)
    with _open(args.workspace) as reg:
        protocol = register_protocol(reg, spec)
        _emit(
            protocol.to_dict() | {"id": protocol.id},
            f"{protocol.id}: protocol {protocol.name} {protocol.version} (hash {protocol.protocol_hash[7:19]})",
            args,
        )
    return 0


def _cmd_benchmark_submit(args: argparse.Namespace) -> int:
    spec = _load_benchmark_spec(args.spec)
    with _open(args.workspace) as reg:
        protocol = resolve_protocol(reg, args.protocol)
        try:
            result = submit_benchmark(
                reg,
                LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR),
                _executor(reg, args.workspace),
                protocol,
                spec,
                source_root=Path.cwd(),
            )
        except BenchmarkRefusal as exc:
            print("error: benchmark refused; nothing was submitted", file=sys.stderr)
            for issue in exc.issues:
                print(f"  - {issue}", file=sys.stderr)
            return 2
        doc: dict[str, object] = {
            "submission_id": result.submission_id,
            "benchmark_id": result.benchmark_id,
            "result_id": result.result_id,
            "status": result.status.value if result.status else None,
            "already_submitted": result.already_submitted,
        }
        _emit(doc, f"submission {result.submission_id} to protocol {protocol.id}", args)
    return 0 if result.submission_id is not None else 1


def _cmd_leaderboard_list(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        protocols = sorted(reg.find(BenchmarkProtocol), key=lambda p: p.id)
        _emit(
            {"protocols": [p.to_dict() | {"id": p.id} for p in protocols]},
            "\n".join(
                f"{p.id} {p.name} {p.version} (hash {p.protocol_hash[7:19]})" for p in protocols
            ),
            args,
        )
    return 0


def _cmd_leaderboard_inspect(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        ref = args.id
        if ref.startswith(BenchmarkProtocol.PREFIX + "_") or not any(
            ref.startswith(p + "_")
            for p in (
                BenchmarkSubmission.PREFIX,
                LeaderboardSnapshot.PREFIX,
                LeaderboardEntry.PREFIX,
            )
        ):
            protocol = resolve_protocol(reg, ref)
            submissions = sorted(
                reg.find(BenchmarkSubmission, protocol_id=protocol.id), key=lambda s: s.id
            )
            snapshots = sorted(
                reg.find(LeaderboardSnapshot, protocol_id=protocol.id), key=lambda s: s.id
            )
            doc: dict[str, object] = {
                **protocol.to_dict(),
                "id": protocol.id,
                "submissions": [s.id for s in submissions],
                "snapshots": [s.id for s in snapshots],
            }
            _emit(
                doc,
                f"{protocol.id}: {len(submissions)} submission(s), {len(snapshots)} snapshot(s)",
                args,
            )
        elif ref.startswith(BenchmarkSubmission.PREFIX + "_"):
            sub = resolve_submission(reg, ref)
            _emit(
                sub.to_dict() | {"id": sub.id},
                f"{sub.id}: model {sub.model_record_id}, result {sub.result_id}",
                args,
            )
        elif ref.startswith(LeaderboardSnapshot.PREFIX + "_"):
            snap = reg.get(LeaderboardSnapshot, ref)
            entries = sorted(reg.find(LeaderboardEntry, snapshot_id=snap.id), key=lambda e: e.id)
            doc = {
                **snap.to_dict(),
                "id": snap.id,
                "entries": [e.to_dict() | {"id": e.id} for e in entries],
            }
            _emit(doc, f"{snap.id}: {len(entries)} entrie(s), {len(snap.excluded)} excluded", args)
        else:
            entry = reg.get(LeaderboardEntry, ref)
            _emit(
                entry.to_dict() | {"id": entry.id},
                f"{entry.id}: {entry.reproducibility_state.value}",
                args,
            )
    return 0


def _cmd_leaderboard_snapshot(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        protocol = resolve_protocol(reg, args.protocol)
        snapshot = build_snapshot(
            reg,
            LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR),
            protocol,
            metric_ids=tuple(args.metric or ()),
            require_complete_coverage=args.require_complete_coverage,
            require_reproduction=args.require_reproduction,
        )
        entries = sorted(reg.find(LeaderboardEntry, snapshot_id=snapshot.id), key=lambda e: e.id)
        doc = {
            **snapshot.to_dict(),
            "id": snapshot.id,
            "entries": [e.to_dict() | {"id": e.id} for e in entries],
        }
        _emit(
            doc, f"{snapshot.id}: {len(entries)} entrie(s), {len(snapshot.excluded)} excluded", args
        )
    return 0


def _cmd_leaderboard_compare(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        try:
            compare_protocol_constrained(reg, args.a, args.b)
            out = compare_submissions(
                reg, LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR), args.a, args.b
            )
        except (ValidationError, BenchmarkRefusal) as exc:
            print("error: submissions are not comparable; nothing was compared", file=sys.stderr)
            print(f"  - {exc}", file=sys.stderr)
            return 2
        _emit(
            out,
            f"compared {args.a} with {args.b} under protocol {out['protocol_hash'][7:19]}",
            args,
        )
    return 0


def _cmd_leaderboard_correct(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        analysis = correct_family(reg, tuple(args.analysis), method=args.method, alpha=args.alpha)
        _dump(analysis.to_dict() | {"id": analysis.id})
    return 0


def _cmd_leaderboard_verify(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        sub = resolve_submission(reg, args.id)
        result = reg.get(BenchmarkResult, sub.result_id)
        out = reproduction_verify(
            reg, LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR), result.run_id
        )
        _dump(out)
    return 1 if out["corrupted"] else 0


def _cmd_leaderboard_replay(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        original = reg.get(LeaderboardSnapshot, args.id)
        protocol = reg.get(BenchmarkProtocol, original.protocol_id)
        require_coverage = bool(original.filtering_rules.get("require_complete_coverage", False))
        require_repro = bool(original.evidence_requirements.get("require_reproduction", False))
        rebuilt = build_snapshot(
            reg,
            LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR),
            protocol,
            metric_ids=original.metric_ids,
            require_complete_coverage=require_coverage,
            require_reproduction=require_repro,
        )
        same = rebuilt.id == original.id
        _emit(
            {"original": original.id, "rebuilt": rebuilt.id, "deterministic": same},
            f"replay of {original.id}: {'identical' if same else 'evidence has changed: ' + rebuilt.id}",
            args,
        )
    return 0 if same else 1


def _load_schedule_spec(path: str) -> ScheduleSpec:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExperionyxError(f"cannot read schedule spec {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ExperionyxError("a schedule spec file must be a JSON object")
    return ScheduleSpec.from_dict(data)


def _scheduler_open_registry(workspace: str) -> Callable[[], SqliteRegistry]:
    def factory() -> SqliteRegistry:
        return _open(workspace)

    return factory


def _scheduler_open_executor(workspace: str) -> Callable[[Registry], Executor]:
    def factory(reg: Registry) -> Executor:
        return Executor(
            reg,
            LocalArtifactStore(Path(workspace) / EXPERIMENTS_DIR),
            source_root=Path.cwd(),
            adapters=default_registries(entry_points=True),
            inputs_root=Path(workspace),
        )

    return factory


def _unit_row(v: SchedulerUnitView) -> dict[str, object]:
    return v.to_dict()


def _run_row(r: object) -> dict[str, object]:
    from experionyx.scheduler.entities import ScheduleRun as _SR

    assert isinstance(r, _SR)  # noqa: S101  # narrows for mypy; callers only pass ScheduleRun
    return {
        "id": r.id,
        "sequence": r.sequence,
        "status": r.status.value,
        "dry_run": r.dry_run,
        "investigation_id": r.investigation_id,
        "created_at": r.created_at.isoformat(),
    }


def _cmd_scheduler_validate(args: argparse.Namespace) -> int:
    spec = _load_schedule_spec(args.spec)
    report = scheduler_validation_report(spec)
    lines = [f"spec {report['spec_id']}: {'VALID' if report['valid'] else 'REFUSED'}"]
    issues: Any = report.get("issues", [])
    lines += [
        f"  - {i['code']}: required {i['required']}; found {i['found']}; {i['why']}" for i in issues
    ]
    if report["valid"]:
        lines.append(f"  {report['units']} unit(s), order: {report['order']}")
    _emit(report, "\n".join(lines), args)
    return 0 if report["valid"] else 1


def _cmd_scheduler_expand(args: argparse.Namespace) -> int:
    spec = _load_schedule_spec(args.spec)
    with _open(args.workspace) as reg:
        try:
            sched, plan, units = scheduler_materialize(reg, spec, scheduler_utc_now())
        except SchedulerRefusal as exc:
            print("error: schedule refused; nothing was materialized", file=sys.stderr)
            for issue in exc.issues:
                print(f"  - {issue}", file=sys.stderr)
            return 2
        doc: dict[str, object] = {
            "schedule_id": sched.id,
            "spec_id": sched.spec_id,
            "graph_hash": sched.graph_hash,
            "units": [units[u.key].id for u in plan.units],
            "order": [u.key for u in plan.units],
        }
        _emit(doc, f"schedule {sched.id}: {len(plan.units)} unit(s) expanded", args)
    return 0


def _cmd_scheduler_run(args: argparse.Namespace) -> int:
    spec = _load_schedule_spec(args.spec)
    result = run_schedule(
        _scheduler_open_registry(args.workspace),
        LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR),
        _scheduler_open_executor(args.workspace),
        spec,
        investigation_id=args.investigation,
        max_workers=args.max_workers,
        dry_run_only=args.dry_run,
        source_root=Path.cwd(),
    )
    text = f"schedule {result.schedule_id}: {result.status.value if result.status else 'DRY_RUN'} {dict(result.counts)}"
    _emit(result.to_dict(), text, args)
    if result.dry_run:
        return 0 if not result.issues else 1
    return 0 if result.status is ScheduleRunState.COMPLETED else 1


def _cmd_scheduler_resume(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        sched = scheduler_resolve(reg, args.id)
        spec = spec_from_schedule(sched)
    result = run_schedule(
        _scheduler_open_registry(args.workspace),
        LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR),
        _scheduler_open_executor(args.workspace),
        spec,
        investigation_id=args.investigation,
        max_workers=args.max_workers,
        source_root=Path.cwd(),
    )
    text = f"schedule {result.schedule_id}: resumed as {result.schedule_run_id}, {result.status.value if result.status else '?'} {dict(result.counts)}"
    _emit(result.to_dict(), text, args)
    return 0 if result.status is ScheduleRunState.COMPLETED else 1


def _cmd_scheduler_status(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        sched = scheduler_resolve(reg, args.id)
        views = scheduler_unit_views(reg, sched.id)
        runs = sorted(reg.find(ScheduleRunEntity, schedule_id=sched.id), key=lambda r: r.sequence)
        counts: dict[str, int] = {}
        for v in views:
            counts[v.status.value] = counts.get(v.status.value, 0) + 1
        doc: dict[str, object] = {
            "schedule_id": sched.id,
            "name": sched.name,
            "version": sched.version,
            "counts": counts,
            "units": [_unit_row(v) for v in views],
            "runs": [_run_row(r) for r in runs],
        }
        lines = [f"{sched.id} {sched.name} {sched.version}: {counts}"]
        lines += [
            f"  {v.key:24} {v.status.value:14} attempts={v.attempts} ref={v.primary_ref}"
            for v in views
        ]
        _emit(doc, "\n".join(lines), args)
    return 0


def _cmd_scheduler_inspect(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        sched = scheduler_resolve(reg, args.id)
        doc = {
            "id": sched.id,
            "name": sched.name,
            "version": sched.version,
            "spec_id": sched.spec_id,
            "graph_hash": sched.graph_hash,
            "engine_version": sched.engine_version,
            "spec": to_jsonable(sched.spec),
        }
        _emit(doc, f"{sched.id} {sched.name} {sched.version} (spec {sched.spec_id})", args)
    return 0


def _cmd_scheduler_graph(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        sched = scheduler_resolve(reg, args.id)
        views = {v.unit_id: v for v in scheduler_unit_views(reg, sched.id)}
        edges = sorted(
            (dep, v.key) for v in views.values() for dep in v.depends_on
        )  # (dependency key, dependent key)
        doc: dict[str, object] = {
            "schedule_id": sched.id,
            "nodes": sorted(v.key for v in views.values()),
            "edges": [list(e) for e in edges],
        }
        _emit(
            doc,
            "\n".join(f"{a} -> {b}" for a, b in edges) or "(no dependencies)",
            args,
        )
    return 0


def _cmd_scheduler_cancel(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        sched = scheduler_resolve(reg, args.id)
        counts = cancel_schedule(reg, sched.id)
        _emit(
            {"schedule_id": sched.id, "cancelled_from": counts},
            f"cancelled schedule {sched.id}: {counts}",
            args,
        )
    return 0


def _cmd_scheduler_retry(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        sched = scheduler_resolve(reg, args.id)
        unit = retry_unit(reg, sched.id, args.unit)
        _emit(
            {"schedule_id": sched.id, "unit": args.unit, "status": unit.status.value},
            f"unit {args.unit} of {sched.id}: {unit.status.value}",
            args,
        )
    return 0


def _cmd_scheduler_replay(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        sched = scheduler_resolve(reg, args.id)
        before = {v.key: (v.status, v.primary_ref) for v in scheduler_unit_views(reg, sched.id)}
        spec = spec_from_schedule(sched)
    result = run_schedule(
        _scheduler_open_registry(args.workspace),
        LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR),
        _scheduler_open_executor(args.workspace),
        spec,
        max_workers=args.max_workers,
        source_root=Path.cwd(),
    )
    with _open(args.workspace) as reg:
        after = {v.key: (v.status, v.primary_ref) for v in scheduler_unit_views(reg, sched.id)}
    differences = {
        k: {"before": [before[k][0].value, before[k][1]], "after": [after[k][0].value, after[k][1]]}
        for k in before
        if before.get(k) != after.get(k)
    }
    env_dependent = sorted(
        {u.key for u in expand(spec).units if u.kind.value == "RESOURCE"} & set(differences)
    )
    doc: dict[str, object] = {
        "schedule_id": sched.id,
        "replay_schedule_run_id": result.schedule_run_id,
        "differences": differences,
        "environment_dependent": env_dependent,
        "note": (
            "RESOURCE units measure the live machine; their differences are expected "
            "variance, not nondeterminism"
        ),
    }
    deterministic = not (set(differences) - set(env_dependent))
    _emit(
        doc,
        f"replay of {sched.id}: {len(differences)} unit(s) differ ({len(env_dependent)} environment-dependent)",
        args,
    )
    return 0 if deterministic else 1


def _load_graph_spec(path: str) -> GraphSpec:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExperionyxError(f"cannot read graph spec {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ExperionyxError("a graph spec file must be a JSON object")
    return GraphSpec.from_dict(data)


def _graph_row(g: EvidenceGraph) -> dict[str, object]:
    return {
        "id": g.id,
        "name": g.name,
        "version": g.version,
        "spec_id": g.spec_id,
        "engine_version": g.engine_version,
    }


def _snapshot_row(s: GraphSnapshot) -> dict[str, object]:
    return {
        "id": s.id,
        "graph_id": s.graph_id,
        "run_id": s.run_id,
        "node_count": s.node_count,
        "edge_count": s.edge_count,
        "unresolved_count": s.unresolved_count,
        "truncated": s.truncated,
        "source_fingerprint": s.source_fingerprint,
    }


def _cmd_graph_build(args: argparse.Namespace) -> int:
    spec = _load_graph_spec(args.spec)
    with _open(args.workspace) as reg:
        store = LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR)
        result: GraphRunResult = run_graph(
            reg, store, _executor(reg, args.workspace), spec, investigation_id=args.investigation
        )
        doc: dict[str, object] = {
            "graph_id": result.graph_id,
            "snapshot_id": result.snapshot_id,
            "run_id": result.run_id,
            "status": result.status.value if result.status else None,
            "already_built": result.already_built,
        }
        if result.snapshot_id is not None:
            doc["snapshot"] = _snapshot_row(reg.get(GraphSnapshot, result.snapshot_id))
        text = f"graph {result.graph_id}: snapshot {result.snapshot_id} ({'reused' if result.already_built else 'built'})"
        _emit(doc, text, args)
    return 0 if result.snapshot_id is not None else 1


def _cmd_graph_list(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        graphs = reg.find(EvidenceGraph)
        _emit(
            {"graphs": [_graph_row(g) for g in graphs]},
            "\n".join(f"{g.id} {g.name} {g.version} (spec {g.spec_id})" for g in graphs),
            args,
        )
    return 0


def _cmd_graph_inspect(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        if args.id.startswith(EvidenceGraph.PREFIX + "_") or not args.id.startswith(
            GraphSnapshot.PREFIX + "_"
        ):
            g = resolve_graph(reg, args.id)
            snaps = reg.find(GraphSnapshot, graph_id=g.id)
            doc: dict[str, object] = {
                **_graph_row(g),
                "spec": to_jsonable(g.spec),
                "snapshots": [_snapshot_row(s) for s in snaps],
            }
            _emit(doc, f"{g.id} {g.name} {g.version}: {len(snaps)} snapshot(s)", args)
        else:
            s = resolve_snapshot(reg, args.id)
            idx = SnapshotIndex(reg, s.id)
            doc = {
                **_snapshot_row(s),
                "summary": to_jsonable(s.summary),
                "nodes_by_kind": _count_by(n.node_kind.value for n in idx.nodes.values()),
                "edges_by_relation": _count_by(e.relation.value for e in idx.edges.values()),
            }
            _emit(
                doc,
                f"{s.id}: {s.node_count} node(s), {s.edge_count} edge(s), {s.unresolved_count} unresolved",
                args,
            )
    return 0


def _count_by(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def _graph_query_parser_kwargs(args: argparse.Namespace) -> GraphQuery:
    return GraphQuery(
        max_depth=args.max_depth,
        max_visited=args.max_visited,
        kinds=tuple(GraphNodeKind(k) for k in (args.kind or [])),
        relations=tuple(args.relation or []),
    )


def _resolve_node(idx: SnapshotIndex, ref: str) -> str:
    """`ref` is a graph node ID (gnd_...) or a referenced entity ID (e.g. run_...)."""
    if ref in idx.nodes:
        return ref
    for kind in GraphNodeKind:
        node = idx.find_by_ref(kind, ref)
        if node is not None:
            return node.id
    raise ExperionyxError(f"no node in this snapshot references {ref!r}")


def _emit_traversal(result: GraphTraversalResult, args: argparse.Namespace, label: str) -> int:
    text = f"{label}: {len(result.nodes)} node(s), {len(result.edges)} edge(s), truncated={result.truncated}"
    _emit(result.to_dict(), text, args)
    return 0


def _cmd_graph_neighbors(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        s = resolve_snapshot(reg, args.snapshot)
        idx = SnapshotIndex(reg, s.id)
        node_id = _resolve_node(idx, args.node)
        direction = TraversalDirection(args.direction)
        result = idx.neighbors(node_id, direction)
        return _emit_traversal(result, args, f"neighbors of {args.node}")


def _cmd_graph_path(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        s = resolve_snapshot(reg, args.snapshot)
        idx = SnapshotIndex(reg, s.id)
        a, b = _resolve_node(idx, args.a), _resolve_node(idx, args.b)
        query = _graph_query_parser_kwargs(args)
        direction = TraversalDirection(args.direction)
        result = idx.path(a, b, direction, query)
        _emit(
            result.to_dict(),
            f"path {args.a} -> {args.b}: {'found' if result.found else 'not found'} (truncated={result.truncated})",
            args,
        )
        return 0 if result.found else 1


_GRAPH_QUERIES: dict[str, Any] = {
    "evidence-for-claim": evidence_for_claim,
    "runs-for-failure-mode": runs_contributing_to_failure_mode,
    "analyses-for-run": analyses_depending_on_run,
    "artifacts-for-analysis": artifacts_for_analysis,
}


def _cmd_graph_query(args: argparse.Namespace) -> int:
    if args.name in _GRAPH_QUERIES:
        with _open(args.workspace) as reg:
            s = resolve_snapshot(reg, args.snapshot)
            idx = SnapshotIndex(reg, s.id)
            node_id = _resolve_node(idx, args.node)
            query = _graph_query_parser_kwargs(args)
            result = _GRAPH_QUERIES[args.name](idx, node_id, query)
            return _emit_traversal(result, args, f"{args.name} {args.node}")
    if args.name in ("model-to-failure", "dataset-to-failure"):
        if args.node2 is None:
            raise ExperionyxError(f"{args.name} needs --node2 (the failure mode ID)")
        with _open(args.workspace) as reg:
            s = resolve_snapshot(reg, args.snapshot)
            idx = SnapshotIndex(reg, s.id)
            query = _graph_query_parser_kwargs(args)
            fn = (
                model_to_failure_paths
                if args.name == "model-to-failure"
                else dataset_to_failure_paths
            )
            result = fn(idx, args.node, args.node2, query)
            _emit(
                result.to_dict(),
                f"{args.name} {args.node} -> {args.node2}: {'found' if result.found else 'not found'}",
                args,
            )
            return 0 if result.found else 1
    raise ExperionyxError(
        f"unknown query {args.name!r}; choose one of {sorted({*_GRAPH_QUERIES, 'model-to-failure', 'dataset-to-failure'})}"
    )


def _cmd_graph_diff(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        a = resolve_snapshot(reg, args.a)
        b = resolve_snapshot(reg, args.b)
        result = graph_diff_snapshots(reg, a.id, b.id)
        text = f"diff {a.id} -> {b.id}: +{len(result.added_nodes)}/-{len(result.removed_nodes)} node(s), +{len(result.added_edges)}/-{len(result.removed_edges)} edge(s)"
        _emit(result.to_dict(), text, args)
    return 0


def _cmd_graph_replay(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        s = resolve_snapshot(reg, args.id)
        out = graph_replay_check(
            reg,
            LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR),
            _executor(reg, args.workspace),
            s.id,
        )
        _dump(out)
        return 0 if out["deterministic"] is True else 1


def _attempt_row(a: ReproductionAttempt) -> dict[str, object]:
    return {
        "id": a.id,
        "target_kind": a.target_kind.value,
        "target_id": a.target_id,
        "attempt": a.attempt,
        "mode": a.spec.get("mode"),
        "outcome": a.outcome.value,
        "sources_changed": a.sources_changed,
        "replay_run_id": a.replay_run_id,
        "note": a.note,
    }


def _attempt_full(a: ReproductionAttempt) -> dict[str, object]:
    return {"id": a.id, **a.to_dict()}


def _report_row(r: Report) -> dict[str, object]:
    return {
        "id": r.id,
        "report_type": r.report_type.value,
        "investigation_id": r.investigation_id,
        "status": r.status.value,
        "title": r.title,
        "generated_at": r.generated_at.isoformat(),
        "evidence_gaps": len(r.evidence_gaps),
    }


def _cmd_report_generate(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        report_type = ReportType(args.report_type)
        template_id = args.template_id or ensure_template(reg, builtin_template(report_type))
        spec = ReportSpec(
            report_type,
            args.investigation,
            template_id,
            tuple(args.run or ()),
            tuple(args.statistical_analysis or ()),
        )
        result = generate_report(reg, spec)
        report = reg.get(Report, result.report_id)
        _emit(
            {
                "id": report.id,
                **report.to_dict(),
                "finding_ids": list(result.finding_ids),
                "warnings": list(result.warnings),
            },
            f"{report.id}: {report.status.value}",
            args,
        )
    return 0


def _cmd_report_list(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        filters = {"investigation_id": args.investigation} if args.investigation else {}
        reports = reg.find(Report, **filters)
        _dump([_report_row(r) for r in reports])
    return 0


def _cmd_report_show(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        report = reg.get(Report, args.id)
        _emit({"id": report.id, **report.to_dict()}, f"{report.id}: {report.status.value}", args)
    return 0


def _cmd_report_sections(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        report = reg.get(Report, args.id)
        _dump([s.to_dict() for s in report.sections])
    return 0


def _cmd_report_claims(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        reg.get(Report, args.id)  # NotFoundError if the report does not exist
        findings = reg.find(ReportFinding, report_id=args.id)
        _dump([{"id": f.id, **f.to_dict()} for f in findings])
    return 0


def _cmd_report_validate(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        issues = report_validate(reg, args.id)
        _dump([{"severity": i.severity, "code": i.code, "message": i.message} for i in issues])
    return 1 if any(i.severity == "ERROR" for i in issues) else 0


def _cmd_report_export(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        report = reg.get(Report, args.id)
        text = (
            render_html(reg, report)
            if args.export_format == "html"
            else render_markdown(reg, report)
        )
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


def _dossier_row(d: EvidenceDossier) -> dict[str, object]:
    return {
        "id": d.id,
        "investigation_id": d.investigation_id,
        "research_question": d.research_question,
        "source_kind": d.source_kind.value,
        "source_id": d.source_id,
        "reports": len(d.report_ids),
        "evidence_gaps": len(d.evidence_gaps),
        "created_at": d.created_at.isoformat(),
    }


def _cmd_dossier_build(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        explicit = tuple((x.split(":", 1)[0], x.split(":", 1)[1]) for x in (args.evidence or ()))
        spec = DossierSpec(
            args.investigation,
            args.question,
            DossierSourceKind(args.source_kind),
            args.source_id or args.investigation,
            explicit,
        )
        result = build_dossier(reg, spec)
        dossier = reg.get(EvidenceDossier, result.dossier_id)
        doc = {
            "id": dossier.id,
            **dossier.to_dict(),
            "item_ids": list(result.item_ids),
            "finding_ids": list(result.finding_ids),
            "warnings": list(result.warnings),
        }
        _emit(
            doc,
            f"{dossier.id}: {len(result.item_ids)} items, {len(result.finding_ids)} findings",
            args,
        )
    return 0


def _cmd_dossier_list(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        filters = {"investigation_id": args.investigation} if args.investigation else {}
        dossiers = reg.find(EvidenceDossier, **filters)
        _dump([_dossier_row(d) for d in dossiers])
    return 0


def _cmd_dossier_show(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        dossier = reg.get(EvidenceDossier, args.id)
        _emit({"id": dossier.id, **dossier.to_dict()}, f"{dossier.id}", args)
    return 0


def _cmd_dossier_items(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        reg.get(EvidenceDossier, args.id)
        items = reg.find(DossierItem, dossier_id=args.id)
        _dump([{"id": i.id, **i.to_dict()} for i in items])
    return 0


def _cmd_dossier_findings(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        reg.get(EvidenceDossier, args.id)
        findings = reg.find(DossierFinding, dossier_id=args.id)
        _dump([{"id": f.id, **f.to_dict()} for f in findings])
    return 0


def _cmd_dossier_snapshot(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        snapshot_id = build_dossier_snapshot(reg, args.id)
        snapshot = reg.get(DossierSnapshot, snapshot_id)
        _emit({"id": snapshot.id, **snapshot.to_dict()}, snapshot.id, args)
    return 0


def _cmd_dossier_validate(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        reg.get(EvidenceDossier, args.id)
        findings = reg.find(DossierFinding, dossier_id=args.id)
    unresolved = [
        f
        for f in findings
        if f.status.value
        in (
            "UNSUPPORTED",
            "MISSING_EXPECTED_EVIDENCE",
            "CONFLICTING_EVIDENCE",
            "PROVENANCE_GAP",
            "REPRODUCIBILITY_GAP",
            "STALE_EVIDENCE",
            "UNAVAILABLE_EVIDENCE",
        )
    ]
    _dump([{"id": f.id, "status": f.status.value, "statement": f.statement} for f in unresolved])
    return (
        1
        if any(f.status.value in ("UNAVAILABLE_EVIDENCE", "STALE_EVIDENCE") for f in unresolved)
        else 0
    )


def _cmd_dossier_export(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        dossier = reg.get(EvidenceDossier, args.id)
        text = render_dossier_markdown(reg, dossier)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


def _cmd_reproduce_validate(args: argparse.Namespace) -> int:
    kind = TargetKind(args.target_kind)
    with _open(args.workspace) as reg:
        try:
            info = reproduction_resolve_target(reg, kind, args.target_id)
        except NotFoundError as exc:
            _dump({"valid": False, "error": str(exc)})
            return 1
    _dump({"valid": True, **info})
    return 0


def _cmd_reproduce_run(args: argparse.Namespace) -> int:
    kind = TargetKind(args.target_kind)
    mode = ReproductionMode(args.mode)
    spec = ReproductionSpec(
        kind, args.target_id, mode, args.relative_tolerance, args.absolute_tolerance
    )
    with _open(args.workspace) as reg:
        if kind is TargetKind.SCHEDULE:
            result = run_reproduction(
                reg,
                LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR),
                _executor(reg, args.workspace),
                spec,
                investigation_id=args.investigation,
                open_registry=_scheduler_open_registry(args.workspace),
                open_executor=_scheduler_open_executor(args.workspace),
                source_root=Path.cwd(),
            )
        else:
            result = run_reproduction(
                reg,
                LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR),
                _executor(reg, args.workspace),
                spec,
                investigation_id=args.investigation,
            )
        attempt = reg.get(ReproductionAttempt, result.attempt_id)
        _emit(
            _attempt_full(attempt),
            f"{attempt.id}: {attempt.outcome.value} (mode {mode.value})",
            args,
        )
    return (
        0
        if result.outcome in (ComparisonOutcome.EQUAL, ComparisonOutcome.APPROXIMATELY_EQUAL)
        else 1
    )


def _cmd_reproduce_inspect(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        a = reproduction_resolve(reg, args.id)
        _emit(_attempt_full(a), f"{a.id}: {a.outcome.value}", args)
    return 0


def _cmd_reproduce_compare(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        out = reproduction_compare(
            reg,
            LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR),
            args.run_a,
            args.path_a,
            args.run_b,
            args.path_b,
            relative_tolerance=args.relative_tolerance,
            absolute_tolerance=args.absolute_tolerance,
        )
        _dump(out)
    return 0 if out["outcome"] in ("EQUAL", "APPROXIMATELY_EQUAL") else 1


def _cmd_reproduce_verify(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        out = reproduction_verify(
            reg, LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR), args.run_id
        )
        _dump(out)
    return 1 if out["corrupted"] else 0


def _cmd_reproduce_replay(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        prior = reproduction_resolve(reg, args.id)
        spec = ReproductionSpec.from_dict(prior.spec)
        store = LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR)
        if prior.target_kind is TargetKind.SCHEDULE:
            result = run_reproduction(
                reg,
                store,
                _executor(reg, args.workspace),
                spec,
                investigation_id=prior.investigation_id,
                open_registry=_scheduler_open_registry(args.workspace),
                open_executor=_scheduler_open_executor(args.workspace),
                source_root=Path.cwd(),
            )
        else:
            result = run_reproduction(
                reg,
                store,
                _executor(reg, args.workspace),
                spec,
                investigation_id=prior.investigation_id,
            )
        attempt = reg.get(ReproductionAttempt, result.attempt_id)
        _emit(
            _attempt_full(attempt),
            f"replay of {prior.id} -> {attempt.id}: {attempt.outcome.value}",
            args,
        )
    return (
        0
        if result.outcome in (ComparisonOutcome.EQUAL, ComparisonOutcome.APPROXIMATELY_EQUAL)
        else 1
    )


def _cmd_reproduce_diff(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        a = reproduction_resolve(reg, args.a)
        b = reproduction_resolve(reg, args.b)
        doc = {
            "a": _attempt_row(a),
            "b": _attempt_row(b),
            "outcome_changed": a.outcome != b.outcome,
            "differences_a": list(a.differences),
            "differences_b": list(b.differences),
        }
        _emit(doc, f"diff {a.id} ({a.outcome.value}) vs {b.id} ({b.outcome.value})", args)
    return 0


def _cmd_fault_demo(args: argparse.Namespace) -> int:
    ws = Path(args.workspace)
    ws.mkdir(parents=True, exist_ok=True)
    reg = SqliteRegistry(ws / REGISTRY_FILE)
    try:
        code = 0
        for result in run_fault_demo(args.name, ws):
            print(f"\n== {result.fault_experiment.name}")
            code |= _print_fault_result(reg, args.workspace, result)
        return code
    finally:
        reg.close()


def _report(result: ExecutionResult) -> int:
    print(f"run: {result.run.id}")
    print(f"status: {result.status}")
    print(f"attempt: {result.run.attempt}")
    print(f"fingerprint: {result.provenance.fingerprint}")
    if result.error:
        print(f"error: [{result.error.stage}] {result.error.error_type}: {result.error.message}")
    return 0 if result.status is RunStatus.COMPLETED else 1


def _cmd_execute(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path.cwd()))  # let `--procedure pkg.mod:fn` import project code
    procedure = resolve_procedure(args.procedure)
    with _open(args.workspace) as reg:
        result = _executor(reg, args.workspace).execute(
            args.id, procedure, seed=args.seed, procedure_name=args.procedure
        )
    return _report(result)


def _cmd_replay(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path.cwd()))
    procedure = resolve_procedure(args.procedure) if args.procedure else None
    with _open(args.workspace) as reg:
        result = _executor(reg, args.workspace).replay(args.id, procedure)
    print("replay requested (this does not verify reproduction)")
    return _report(result)


def _cmd_recover(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        recovered = _executor(reg, args.workspace).recover_interrupted()
    for run in recovered:
        print(f"recovered {run.id}: FAILED (interrupted)")
    print(f"{len(recovered)} run(s) recovered")
    return 0


# -- statistics --------------------------------------------------------------------------------------


def _floats(text: str) -> list[float]:
    try:
        return [float(x) for x in text.split(",") if x.strip()]
    except ValueError as exc:
        raise ExperionyxError(f"not a comma-separated list of numbers: {text!r}") from exc


def _stats_row(a: StatisticalAnalysis) -> dict[str, object]:
    return {
        "id": a.id,
        "kind": a.analysis_kind,
        "status": a.analysis_status,
        "source": a.sources.get("kind"),
        "input_hash": a.input_hash,
        "engine_version": a.engine_version,
        "created_at": a.created_at.isoformat(),
    }


def _stats_sources(args: argparse.Namespace) -> dict[str, object]:
    if args.file:
        doc = json.loads(Path(args.file).read_text(encoding="utf-8"))
        return {"kind": "inline", **doc}
    if getattr(args, "fault_experiment", None):
        return {
            "kind": "fault_trials",
            "fault_experiment_id": args.fault_experiment,
            "metric": args.measure,
            "point_index": args.point_index,
            "reference_field": args.reference_field,
            "treatment_field": args.treatment_field,
        }
    if getattr(args, "interaction", None):
        return {
            "kind": "interaction",
            "analysis_id": args.interaction,
            "measure": args.measure,
            "reference_cell": args.reference_cell,
            "treatment_cell": args.treatment_cell,
        }
    if getattr(args, "artifact", None):
        run_id, path = args.artifact
        out: dict[str, object] = {
            "kind": "artifact",
            "run_id": run_id,
            "path": path,
            "reference": args.reference_pointer.split("/"),
        }
        if args.treatment_pointer:
            out["treatment"] = args.treatment_pointer.split("/")
        return out
    if args.reference_values is None:
        raise ExperionyxError(
            "give values (--reference-values), a --file, an --interaction or an --artifact"
        )
    src: dict[str, object] = {"kind": "inline", "reference": _floats(args.reference_values)}
    if args.treatment_values is not None:
        src["treatment"] = _floats(args.treatment_values)
    return src


def _stats_run(
    args: argparse.Namespace, kind: str, sources: dict[str, object], cfg: dict[str, object]
) -> int:
    with _open(args.workspace) as reg:
        store = LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR)
        a, new = stats_store.create(reg, store, kind, sources, cfg)
        _emit(
            {
                **_stats_row(a),
                "new": new,
                "config": to_jsonable(a.config),
                "result": to_jsonable(a.result),
            },
            f"{a.id} {kind} {a.analysis_status}{'' if new else ' (already registered)'}",
            args,
        )
    return 0


def _stats_cfg(args: argparse.Namespace) -> dict[str, object]:
    return {
        k: getattr(args, k)
        for k in ("pairing", "estimator", "method", "confidence", "resamples", "seed")
        if getattr(args, k) is not None
    }


def _cmd_stats_compare(args: argparse.Namespace) -> int:
    cfg = _stats_cfg(args)
    if args.permutations is not None:
        cfg["permutations"] = args.permutations
    return _stats_run(args, "COMPARE", _stats_sources(args), cfg)


def _cmd_stats_bootstrap(args: argparse.Namespace) -> int:
    return _stats_run(args, "BOOTSTRAP", _stats_sources(args), _stats_cfg(args))


def _cmd_stats_proportion(args: argparse.Namespace) -> int:
    if args.failure_mode:
        src: dict[str, object] = {"kind": "failure_mode", "mode_id": args.failure_mode}
    elif args.successes is not None and args.trials is not None:
        src = {"kind": "inline", "successes": args.successes, "trials": args.trials}
    else:
        raise ExperionyxError("give --successes and --trials, or --failure-mode")
    cfg = {} if args.confidence is None else {"confidence": args.confidence}
    return _stats_run(args, "PROPORTION", src, cfg)


def _cmd_stats_correct(args: argparse.Namespace) -> int:
    if args.analysis:
        src: dict[str, object] = {"kind": "analyses", "ids": sorted(args.analysis)}
    else:
        try:
            ps = {k: float(x) for k, _, x in (p.partition("=") for p in args.p)}
        except ValueError as exc:
            raise ExperionyxError("--p needs name=value with a numeric value") from exc
        src = {"kind": "inline", "pvalues": ps}
    cfg = {k: getattr(args, k) for k in ("method", "alpha") if getattr(args, k) is not None}
    return _stats_run(args, "CORRECTION", src, cfg)


def _cmd_stats_list(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        rows = [
            _stats_row(a)
            for a in sorted(reg.find(StatisticalAnalysis), key=lambda a: a.id)
            if not args.kind or a.analysis_kind == args.kind
        ]
    _emit({"analyses": rows}, "\n".join(f"{r['id']} {r['kind']} {r['status']}" for r in rows), args)
    return 0


def _cmd_stats_inspect(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        a = reg.get(StatisticalAnalysis, args.id)
    doc = {
        **_stats_row(a),
        "config": to_jsonable(a.config),
        "sources": to_jsonable(a.sources),
        "result": to_jsonable(a.result),
        "result_hash": a.result_hash,
    }
    _emit(doc, f"{a.id} {a.analysis_kind} {a.analysis_status} inputs {a.input_hash[7:19]}", args)
    return 0


def _cmd_stats_verify(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        store = LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR)
        out = stats_store.verify(reg, store, args.id)
    _emit(out, f"{args.id} {'reproduced' if out['reproduced'] else 'NOT reproduced'}", args)
    return 0 if out["reproduced"] else 1


# -- slices -------------------------------------------------------------------------------------------------


def _slice_specs(path: str) -> list[SliceSpec]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    items = doc["slices"] if isinstance(doc, dict) and "slices" in doc else [doc]
    return [SliceSpec.from_dict(x) for x in items]


def _sreg(args: argparse.Namespace, reg: SqliteRegistry) -> SliceRegistry:
    return SliceRegistry(reg, LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR))


def _slice_row(s: Slice) -> dict[str, object]:
    return {
        "id": s.id,
        "name": s.name,
        "description": s.description,
        "fields": list(s.fields),
        "static": s.spec().static,
    }


def _summary_slices(a: SliceAnalysis) -> dict[str, Any]:
    x = a.summary.get("slices")
    return dict(x) if isinstance(x, Mapping) else {}


def _analysis_row(a: SliceAnalysis) -> dict[str, object]:
    return {
        "id": a.id,
        "status": a.analysis_status,
        "baseline_run_id": a.baseline_run_id,
        "dataset_fingerprint": a.dataset_fingerprint,
        "spec_id": a.spec_id,
        "slices": sorted(_summary_slices(a)),
    }


def _cmd_slice_validate(args: argparse.Namespace) -> int:
    doc = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    if isinstance(doc, dict) and "baseline_run" in doc:
        spec = SliceAnalysisSpec.from_dict(doc)
        out: dict[str, object] = {
            "valid": True,
            "kind": "analysis",
            "spec_id": spec.spec_id,
            "slices": [
                {"name": s.name, "slice_id": s.slice_id, "description": s.human, "static": s.static}
                for s in spec.slices
            ],
        }
    else:
        out = {
            "valid": True,
            "kind": "slices",
            "slices": [
                {
                    "name": s.name,
                    "slice_id": s.slice_id,
                    "description": s.human,
                    "fields": list(s.fields),
                    "static": s.static,
                }
                for s in _slice_specs(args.spec)
            ],
        }
    _emit(out, "VALID: " + ", ".join(f"{s['name']} = {s['slice_id']}" for s in out["slices"]), args)  # type: ignore[attr-defined]
    return 0


def _cmd_slice_register(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        sr = _sreg(args, reg)
        rows = []
        for spec in _slice_specs(args.spec):
            s, created = sr.register(spec)
            rows.append({**_slice_row(s), "created": created})
    _emit(
        {"slices": rows},
        "\n".join(
            f"{r['id']} {r['name']} {'created' if r['created'] else 'already registered'}"
            for r in rows
        ),
        args,
    )
    return 0


def _cmd_slice_list(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        sr = _sreg(args, reg)
        static = None if args.static is None else args.static == "yes"
        rows = [
            _slice_row(s)
            for s in sr.search(name=args.name, field=args.field, text=args.text, static=static)
        ]
        analyses = (
            [
                _analysis_row(a)
                for a in sr.analyses(
                    **({"baseline_run_id": args.baseline_run} if args.baseline_run else {})
                )
            ]
            if args.analyses
            else []
        )
    _emit(
        {"slices": rows, **({"analyses": analyses} if args.analyses else {})},
        "\n".join(f"{r['id']} {r['name']}: {r['description']}" for r in rows),
        args,
    )
    return 0


def _cmd_slice_inspect(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        sr = _sreg(args, reg)
        if args.id.startswith("sls_"):
            s = sr.get(args.id)
            doc: dict[str, object] = {
                **_slice_row(s),
                "condition": to_jsonable(s.condition),
                "slice_schema": s.slice_schema,
                "used_by": [
                    a.id
                    for a in sr.analyses()
                    if args.id in {x["slice_id"] for x in _summary_slices(a).values()}
                ],
            }
        else:
            a = sr.analysis(args.id)
            doc = {
                **_analysis_row(a),
                "summary": to_jsonable(a.summary),
                "provenance": sr.provenance(a.id),
                "artifacts": [{"path": x.path, "id": x.id} for x in sr.artifacts(a.id)],
            }
            if args.full:
                doc["documents"] = {
                    n: sr.document(a.id, n) for n in ("spec", "membership", "results", "summary")
                }
    _emit(doc, f"{args.id}: {doc.get('description') or doc.get('status')}", args)
    return 0


def _slice_dataset(
    args: argparse.Namespace, reg: SqliteRegistry, run_id: str, specs: Sequence[SliceSpec]
) -> object:
    """The baseline's dataset, loaded only if some slice reads a feature."""
    if not any(f.startswith("feature:") for s in specs for f in s.fields):
        return None
    return dataset_for_run(reg, default_registries(entry_points=True), Path(args.workspace), run_id)


def _cmd_slice_evaluate(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        sr = _sreg(args, reg)
        specs = _slice_specs(args.spec)
        ds = _slice_dataset(args, reg, args.baseline_run, specs)
        out = []
        for spec in specs:
            m = sr.evaluate(spec, args.baseline_run, ds)  # type: ignore[arg-type]
            row = to_jsonable(m)
            assert isinstance(row, dict)  # noqa: S101
            if not args.ids:
                row["sample_ids"] = f"{m.n_members} member IDs omitted (use --ids)"
            out.append(row)
    _emit(
        {"memberships": out},
        "\n".join(
            f"{r['name']}: {r['status']}, {r['n_members']}/{r['n_total']} samples, {r['n_unknown']} unknown"
            for r in out
        ),
        args,
    )
    return 0


def _cmd_slice_analyze(args: argparse.Namespace) -> int:
    spec = SliceAnalysisSpec.from_dict(json.loads(Path(args.spec).read_text(encoding="utf-8")))
    with _open(args.workspace) as reg:
        run = reg.get(Run, spec.baseline_run)
        inv = reg.get(Experiment, run.experiment_id).investigation_id
        out = run_slice_analysis_request(
            reg,
            LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR).__class__(
                Path(args.workspace) / EXPERIMENTS_DIR
            ),
            _executor(reg, args.workspace),
            inv,
            spec,
        )
        a = reg.get(SliceAnalysis, out.analysis_id) if out.analysis_id else None
    doc: dict[str, object] = {
        "status": out.status.value,
        "run_id": out.run_id,
        "analysis_id": out.analysis_id,
        "analysis_status": None if a is None else a.analysis_status,
        "summary": None if a is None else to_jsonable(a.summary),
    }
    _emit(doc, f"{out.status.value}: {out.analysis_id} ({doc['analysis_status']})", args)
    return 0 if a is not None and a.analysis_status == "COMPLETE" else 3


def _cmd_slice_compare(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        store = LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR)
        sr = SliceRegistry(reg, store)
        base = load_baseline(reg, store, args.baseline_run)
        (a_spec,) = _slice_specs(args.a)
        b_specs = [] if args.b in ("POPULATION", "REST") else _slice_specs(args.b)
        ds = _slice_dataset(args, reg, args.baseline_run, [a_spec, *b_specs])
        cfg = slice_an.SliceConfig(
            **{
                k: v
                for k, v in (
                    ("confidence", args.confidence),
                    ("resamples", args.resamples),
                    ("seed", args.seed),
                    ("method", args.method),
                    ("min_members", args.min_members),
                )
                if v is not None
            }
        )
        a = sr.evaluate(a_spec, args.baseline_run, ds)  # type: ignore[arg-type]
        if args.b in ("POPULATION", "REST"):
            b_name, b_ids = args.b, None
        else:
            (b_spec,) = _slice_specs(args.b)
            b = sr.evaluate(b_spec, args.baseline_run, ds)  # type: ignore[arg-type]
            b_name, b_ids = b_spec.name, list(b.sample_ids)
    if a.n_members == 0:
        raise ExperionyxError(
            f"slice {a_spec.name} has no members ({a.status.value}); nothing to compare"
        )
    if args.b == "POPULATION":
        res = slice_an.compare_to_population(base, a.sample_ids, cfg)
    else:
        other = (
            b_ids
            if b_ids is not None
            else [i for i in sorted(base.rows) if i not in set(a.sample_ids)]
        )
        res = slice_an.compare_groups(base, a.sample_ids, other, cfg)
    doc = {
        "a": {"name": a_spec.name, "slice_id": a_spec.slice_id, "n_members": a.n_members},
        "b": b_name,
        "comparison": to_jsonable(res),
    }
    _emit(doc, f"{a_spec.name} vs {b_name}: {res.get('status')}", args)
    return 0


# -- drift ---------------------------------------------------------------------------------------------------


def _drift_spec(path: str) -> ShiftSpec:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExperionyxError(f"cannot read drift spec {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise ExperionyxError("a drift spec file must be a JSON object")
    return ShiftSpec.from_dict(doc)


def _drift_dataset(args: argparse.Namespace, reg: SqliteRegistry, spec: ShiftSpec) -> object:
    """The baseline's dataset, loaded only if the spec reads a dataset column."""
    return drift_engine.baseline_dataset(
        reg, default_registries(entry_points=True), Path(args.workspace), spec
    )


def _dreg(args: argparse.Namespace, reg: SqliteRegistry) -> DriftRegistry:
    return DriftRegistry(reg, LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR))


def _pairs(a: DriftAnalysis) -> dict[str, Any]:
    x = a.summary.get("pairs")
    return dict(x) if isinstance(x, Mapping) else {}


def _drift_row(a: DriftAnalysis) -> dict[str, object]:
    return {
        "id": a.id,
        "status": a.analysis_status,
        "baseline_run_id": a.baseline_run_id,
        "dataset_fingerprint": a.dataset_fingerprint,
        "spec_id": a.spec_id,
        "provenance_fingerprint": a.provenance_fingerprint,
        "window_pairs": sorted(_pairs(a)),
    }


def _window_row(w: DriftWindow) -> dict[str, object]:
    return {
        "id": w.id,
        "name": w.name,
        "role": w.role,
        "ordering": w.ordering_field,
        "window": w.window().describe(),
    }


def _cmd_drift_validate(args: argparse.Namespace) -> int:
    spec = _drift_spec(args.spec)
    plan = spec.plan()
    oid = spec.ordering.field
    out: dict[str, object] = {
        "valid": True,
        "spec_id": spec.spec_id,
        "ordering": oid,
        "pairs": [
            {
                "key": p.key,
                "reference_window_id": p.reference.window_id(oid),
                "comparison_window_id": p.comparison.window_id(oid),
            }
            for p in plan.pairs
        ],
        "skipped": [s.to_dict(oid) for s in plan.skipped],
        "features": {f.name: f.kind for f in spec.features},
        "slices": {s.name: s.slice_id for s in spec.slices},
        "dimensions": list(spec.config.dimensions or ()),
        "config": spec.config.to_dict(),
    }
    if args.preflight:
        with _open(args.workspace) as reg:
            res = drift_engine.preflight(
                reg,
                LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR),
                spec,
                _drift_dataset(args, reg, spec),
            )
        out["preflight"] = {
            "ok": True,
            "windows": {
                k: {"n_samples": w.n, "sample_digest": w.digest} for k, w in res.windows.items()
            },
            "skipped": [s.to_dict(oid) for s in res.skipped],
        }
    _emit(
        out,
        f"VALID: {spec.spec_id}: {len(plan.pairs)} window pair(s), {len(plan.skipped)} skipped",
        args,
    )
    return 0


def _cmd_drift_list(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        dr = _dreg(args, reg)
        cols = {
            k: v
            for k, v in (("baseline_run_id", args.baseline_run), ("analysis_status", args.status))
            if v
        }
        analyses = [_drift_row(a) for a in dr.analyses(**cols)]
        windows = (
            [_window_row(w) for w in dr.windows(ordering=args.ordering)] if args.windows else []
        )
    _emit(
        {"analyses": analyses, **({"windows": windows} if args.windows else {})},
        "\n".join(f"{r['id']} {r['status']} {r['spec_id']}" for r in analyses),
        args,
    )
    return 0


def _cmd_drift_inspect(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        dr = _dreg(args, reg)
        if args.id.startswith("twn_"):
            w = dr.window(args.id)
            used = [
                a.id
                for a in dr.analyses()
                if any(
                    args.id in (p["reference_window_id"], p["comparison_window_id"])
                    for p in _pairs(a).values()
                )
            ]
            doc: dict[str, object] = {
                **_window_row(w),
                "drift_schema": w.drift_schema,
                "start": w.start,
                "end": w.end,
                "start_inclusive": w.start_inclusive,
                "end_inclusive": w.end_inclusive,
                "used_by": used,
            }
        else:
            a = dr.analysis(args.id)
            doc = {
                **_drift_row(a),
                "summary": to_jsonable(a.summary),
                "provenance": dr.provenance(a.id),
                "artifacts": [{"path": x.path, "id": x.id} for x in dr.artifacts(a.id)],
            }
            if args.full:
                doc["documents"] = {
                    n: dr.document(a.id, n)
                    for n in (
                        "spec",
                        "windows",
                        "feature_results",
                        "distribution_results",
                        "performance_results",
                        "summary",
                    )
                }
    _emit(doc, f"{args.id}: {doc.get('window') or doc.get('status')}", args)
    return 0


def _cmd_drift_windows(args: argparse.Namespace) -> int:
    spec = _drift_spec(args.spec)
    with _open(args.workspace) as reg:
        res = drift_engine.preflight(
            reg,
            LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR),
            spec,
            _drift_dataset(args, reg, spec),
        )
    oid = spec.ordering.field
    rows: list[dict[str, Any]] = []
    for w in res.windows.values():
        d: dict[str, Any] = dict(w.to_dict())
        if not args.ids:
            d["sample_ids"] = f"{w.n} sample IDs omitted (use --ids)"
        rows.append(d)
    doc: dict[str, object] = {
        "spec_id": spec.spec_id,
        "ordering": oid,
        "windows": rows,
        "pairs": [
            {"key": k, "reference_window_id": r, "comparison_window_id": c} for k, r, c in res.pairs
        ],
        "skipped": [s.to_dict(oid) for s in res.skipped],
    }
    _emit(
        doc,
        "\n".join(f"{r['window_id']} {r['window']['role']} n={r['n_samples']}" for r in rows),
        args,
    )
    return 0


def _cmd_drift_evaluate(args: argparse.Namespace) -> int:
    spec = _drift_spec(args.spec)
    with _open(args.workspace) as reg:
        run = reg.get(Run, spec.baseline_run)
        inv = reg.get(Experiment, run.experiment_id).investigation_id
        out = drift_engine.run_drift_request(
            reg,
            LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR),
            _executor(reg, args.workspace),
            inv,
            spec,
            dataset=_drift_dataset(args, reg, spec),
        )
        a = reg.get(DriftAnalysis, out.analysis_id) if out.analysis_id else None
    doc: dict[str, object] = {
        "status": out.status.value,
        "run_id": out.run_id,
        "analysis_id": out.analysis_id,
        "analysis_status": None if a is None else a.analysis_status,
        "summary": None if a is None else to_jsonable(a.summary),
    }
    _emit(doc, f"{out.status.value}: {out.analysis_id} ({doc['analysis_status']})", args)
    return 0 if a is not None and a.analysis_status == "COMPLETE" else 3


def _cmd_drift_compare(args: argparse.Namespace) -> int:
    """The same computation as `evaluate`, but nothing is stored and no run is created."""
    spec = _drift_spec(args.spec)
    with _open(args.workspace) as reg:
        store = LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR)
        drift_engine.validate(reg, spec)
        c = drift_engine.compute(reg, store, spec, _drift_dataset(args, reg, spec))
    doc: dict[str, object] = {
        "spec_id": spec.spec_id,
        "summary": drift_engine.summarize(spec, c),
        "not_stored": "nothing was stored; use `drift evaluate` to record this analysis",
    }
    if args.full:
        doc["feature_results"], doc["distribution_results"], doc["performance_results"] = (
            c.feature_results,
            c.distribution_results,
            c.performance_results,
        )
    _emit(doc, f"{len(c.resolution.pairs)} window pair(s) compared (not stored)", args)
    return 0


def _cmd_drift_replay(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        out = drift_engine.replay_check(
            reg,
            LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR),
            _executor(reg, args.workspace),
            args.id,
        )
    _emit(out, f"{'reproduced' if out['deterministic'] else 'DIFFERS'}: {args.id}", args)
    return 0 if out["deterministic"] else 1


# -- data quality --------------------------------------------------------------------------------------------


def _quality_spec(path: str) -> QualitySpec:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExperionyxError(f"cannot read quality spec {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise ExperionyxError("a quality spec file must be a JSON object")
    return QualitySpec.from_dict(doc)


def _qreg(args: argparse.Namespace, reg: SqliteRegistry) -> QualityRegistry:
    return QualityRegistry(reg, LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR))


def _quality_dataset(args: argparse.Namespace, reg: SqliteRegistry, spec: QualitySpec) -> object:
    return quality_engine.load_dataset(
        reg, default_registries(entry_points=True), Path(args.workspace), spec
    )


def _quality_row(a: QualityAnalysis) -> dict[str, object]:
    return {
        "id": a.id,
        "status": a.analysis_status,
        "dataset_id": a.dataset_id,
        "dataset_fingerprint": a.dataset_fingerprint,
        "spec_id": a.spec_id,
        "provenance_fingerprint": a.provenance_fingerprint,
        "status_counts": to_jsonable(a.summary.get("status_counts")),
    }


def _check_row(c: QualityCheck) -> dict[str, object]:
    return {
        "id": c.id,
        "check_id": c.check_id,
        "check_type": c.check_type,
        "scope": c.scope,
        "status": c.status,
        "reason": c.reason,
    }


def _cmd_quality_validate(args: argparse.Namespace) -> int:
    spec = _quality_spec(args.spec)
    out: dict[str, object] = {
        "valid": True,
        "spec_id": spec.spec_id,
        "dataset_id": spec.dataset_id,
        "checks": [{"check_id": c.check_id, "type": c.type} for c in spec.checks],
        "splits": list(spec.splits),
        "features": {f.name: f.kind for f in spec.features},
        "slices": {s.name: s.slice_id for s in spec.slices},
        "config": spec.config.to_dict(),
    }
    if args.preflight:
        with _open(args.workspace) as reg:
            tables = quality_engine.preflight(reg, spec, _quality_dataset(args, reg, spec))
        out["preflight"] = {
            "ok": True,
            "splits": {
                str(s): {"n_rows": t.n, "sample_digest": t.digest()} for s, t in tables.items()
            },
        }
    _emit(out, f"VALID: {spec.spec_id}: {len(spec.checks)} check(s)", args)
    return 0


def _cmd_quality_list(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        cols = {
            k: v
            for k, v in (("dataset_id", args.dataset_id), ("analysis_status", args.status))
            if v
        }
        rows = [_quality_row(a) for a in _qreg(args, reg).analyses(**cols)]
    _emit(
        {"analyses": rows}, "\n".join(f"{r['id']} {r['status']} {r['spec_id']}" for r in rows), args
    )
    return 0


def _cmd_quality_inspect(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        qr = _qreg(args, reg)
        if args.id.startswith("qck_"):
            c = reg.get(QualityCheck, args.id)
            doc: dict[str, object] = {
                **_check_row(c),
                "analysis_id": c.analysis_id,
                "evidence": to_jsonable(c.evidence),
            }
        else:
            a = qr.analysis(args.id)
            doc = {
                **_quality_row(a),
                "summary": to_jsonable(a.summary),
                "provenance": qr.provenance(a.id),
                "artifacts": [{"path": x.path, "id": x.id} for x in qr.artifacts(a.id)],
            }
            if args.full:
                doc["documents"] = {
                    n: qr.document(a.id, n)
                    for n in ("spec", "checks", "observations", "violations", "summary")
                }
    _emit(doc, f"{args.id}: {doc.get('status')}", args)
    return 0


def _cmd_quality_checks(args: argparse.Namespace) -> int:
    if args.catalog or not args.id:
        doc: dict[str, object] = {"check_types": CHECK_DOCS}
        _emit(doc, "\n".join(f"{k}: {v}" for k, v in CHECK_DOCS.items()), args)
        return 0
    with _open(args.workspace) as reg:
        cols = {k: v for k, v in (("check_type", args.type), ("status", args.status)) if v}
        rows = [_check_row(c) for c in _qreg(args, reg).checks(args.id, **cols)]
    _emit(
        {"analysis_id": args.id, "checks": rows},
        "\n".join(f"{r['check_type']} {r['scope']} {r['status']}" for r in rows),
        args,
    )
    return 0


def _cmd_quality_run(args: argparse.Namespace) -> int:
    spec = _quality_spec(args.spec)
    with _open(args.workspace) as reg:
        inv = quality_engine.baseline_investigation(reg, args.investigation)
        out = quality_engine.run_quality_request(
            reg,
            LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR),
            _executor(reg, args.workspace),
            inv,
            spec,
            dataset=_quality_dataset(args, reg, spec),
        )
        a = reg.get(QualityAnalysis, out.analysis_id) if out.analysis_id else None
    doc: dict[str, object] = {
        "status": out.status.value,
        "run_id": out.run_id,
        "analysis_id": out.analysis_id,
        "analysis_status": None if a is None else a.analysis_status,
        "summary": None if a is None else to_jsonable(a.summary),
    }
    _emit(doc, f"{out.status.value}: {out.analysis_id} ({doc['analysis_status']})", args)
    counts = {} if a is None else dict(a.summary.get("status_counts", {}))  # type: ignore[call-overload]
    return 0 if a is not None and a.analysis_status == "COMPLETE" and not counts.get("FAIL") else 3


def _cmd_quality_replay(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        out = quality_engine.replay_check(
            reg,
            LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR),
            _executor(reg, args.workspace),
            args.id,
        )
    _emit(out, f"{'reproduced' if out['deterministic'] else 'DIFFERS'}: {args.id}", args)
    return 0 if out["deterministic"] else 1


# -- model stress --------------------------------------------------------------------------------------------


def _stress_design(path: str) -> StressDesign:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExperionyxError(f"cannot read stress spec {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise ExperionyxError("a stress spec file must be a JSON object")
    return StressDesign.from_dict(doc)


def _sxreg(args: argparse.Namespace, reg: SqliteRegistry) -> StressRegistry:
    return StressRegistry(reg, LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR))


def _stress_row(a: StressAnalysis) -> dict[str, object]:
    return {
        "id": a.id,
        "status": a.analysis_status,
        "spec_id": a.spec_id,
        "model_id": a.model_id,
        "dataset_id": a.dataset_id,
        "baseline_run_id": a.baseline_run_id,
        "provenance_fingerprint": a.provenance_fingerprint,
        "design": a.summary.get("design"),
        "families": to_jsonable(a.summary.get("families")),
        "coverage": to_jsonable(a.summary.get("coverage")),
    }


def _trial_row(t: StressTrial) -> dict[str, object]:
    return {
        "id": t.id,
        "unit_key": t.unit_key,
        "point_index": t.point_index,
        "repeat_index": t.repeat_index,
        "cell": t.cell,
        "family": t.family,
        "stress_ids": list(t.stress_ids),
        "origin": t.origin,
        "seed": t.seed,
        "status": t.status,
        "run_id": t.run_id,
        "reason": t.reason,
    }


def _cmd_stress_families(args: argparse.Namespace) -> int:
    rows = {
        n: {
            "origin": f.origin.value,
            "target": f.target,
            "description": f.description,
            "fault_type": f.fault_type,
            "default_sweep_parameter": f.default_sweep,
            "parameters": {
                k: {"kind": p.kind, "required": p.required} for k, p in f.params.items()
            },
        }
        for n, f in sorted(STRESS_FAMILIES.items())
    }
    _emit(
        {"families": rows},
        "\n".join(f"{n}: {r['origin']} - {r['description']}" for n, r in rows.items()),
        args,
    )
    return 0


def _cmd_stress_validate(args: argparse.Namespace) -> int:
    d = _stress_design(args.spec)
    units = d.plan.units()
    out: dict[str, object] = {
        "valid": True,
        "spec_id": d.spec_id,
        "plan_id": d.plan.plan_id,
        "design": d.plan.design,
        "origin": d.plan.origin.value,
        "stress_ids": [c.stress_id for c in d.plan.components],
        "n_trials": len(units),
        "units": [
            {
                "key": u.key,
                "point_index": u.point_index,
                "repeat_index": u.repeat_index,
                "cell": u.cell,
                "seed": u.seed,
                "value": u.value,
            }
            for u in units
        ],
    }
    if args.preflight:
        with _open(args.workspace) as reg:
            out["capabilities"] = stress_engine.preflight(reg, d, _executor(reg, args.workspace))
    _emit(out, f"VALID: {d.spec_id}: {d.plan.design}, {len(units)} trial(s)", args)
    return 0


def _cmd_stress_list(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        cols = {
            k: v
            for k, v in (
                ("model_id", args.model_id),
                ("dataset_id", args.dataset_id),
                ("analysis_status", args.status),
            )
            if v
        }
        rows = [_stress_row(a) for a in _sxreg(args, reg).analyses(**cols)]
    _emit(
        {"analyses": rows},
        "\n".join(f"{r['id']} {r['status']} {r['design']} {r['spec_id']}" for r in rows),
        args,
    )
    return 0


def _cmd_stress_inspect(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        sr = _sxreg(args, reg)
        if args.id.startswith("sxt_"):
            t = reg.get(StressTrial, args.id)
            doc: dict[str, object] = {
                **_trial_row(t),
                "analysis_id": t.analysis_id,
                "lineage": to_jsonable(t.lineage),
            }
        else:
            a = sr.analysis(args.id)
            doc = {
                **_stress_row(a),
                "summary": to_jsonable(a.summary),
                "provenance": sr.provenance(a.id),
                "trials": [_trial_row(t) for t in sr.trials(a.id)],
                "artifacts": [{"path": x.path, "id": x.id} for x in sr.artifacts(a.id)],
            }
            if args.full:
                doc["documents"] = {
                    n: sr.document(a.id, n)
                    for n in (
                        "spec",
                        "plan",
                        "trials",
                        "baseline",
                        "results",
                        "analyses",
                        "summary",
                    )
                }
    _emit(doc, f"{args.id}: {doc.get('status')}", args)
    return 0


def _stress_run(args: argparse.Namespace, *, sweep_only: bool) -> int:
    d = _stress_design(args.spec)
    if sweep_only and d.plan.design not in ("SWEEP", "REPEATED"):
        raise ExperionyxError(
            f"`stress sweep` needs a plan with a sweep or several seeds (this one is {d.plan.design}); use `stress run`"
        )
    with _open(args.workspace) as reg:
        out = stress_engine.run_stress_experiment(
            reg,
            LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR),
            _executor(reg, args.workspace),
            d,
            source_root=Path.cwd(),
            investigation_id=args.investigation,
        )
        a = reg.get(StressAnalysis, out.analysis_id) if out.analysis_id else None
    doc: dict[str, object] = {
        "status": out.status.value,
        "run_id": out.run_id,
        "analysis_id": out.analysis_id,
        "analysis_status": None if a is None else a.analysis_status,
        "baseline_run_id": out.baseline_run_id,
        "capabilities": to_jsonable(list(out.capabilities)),
        "summary": None if a is None else to_jsonable(a.summary),
    }
    _emit(doc, f"{out.status.value}: {out.analysis_id} ({doc['analysis_status']})", args)
    return 0 if a is not None and a.analysis_status == "COMPLETE" else 3


def _cmd_stress_run(args: argparse.Namespace) -> int:
    return _stress_run(args, sweep_only=False)


def _cmd_stress_sweep(args: argparse.Namespace) -> int:
    return _stress_run(args, sweep_only=True)


def _cmd_stress_compare(args: argparse.Namespace) -> int:
    """Recompute the analysis from the stored trials and compare it with the stored documents; nothing is stored."""
    with _open(args.workspace) as reg:
        store = LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR)
        sr = StressRegistry(reg, store)
        a = sr.analysis(args.id)
        spec, trials, analyses = (
            sr.document(a.id, "spec"),
            sr.document(a.id, "trials"),
            sr.document(a.id, "analyses"),
        )
        design = StressDesign.from_dict(spec["design"])
        dataset = None
        try:
            from experionyx.slices.data import dataset_for_run

            dataset = dataset_for_run(
                reg, default_registries(entry_points=True), Path(args.workspace), a.baseline_run_id
            )
        except ExperionyxError:
            dataset = None
        c = stress_engine.compute(
            reg,
            store,
            design,
            a.baseline_run_id,
            [
                {k: t[k] for k in ("unit", "status", "run_id", "experiment_id", "reason", "origin")}
                for t in trials["trials"]
            ],
            analyses["links"],
            spec["capabilities"],
            dataset,
            a.investigation_id,
        )
        stored = {
            n: sr.document(a.id, n)
            for n in ("spec", "plan", "trials", "baseline", "results", "analyses", "summary")
        }
    from experionyx.interactions.lifecycle import _compare as compare_docs

    diffs: list[str] = []
    for name, doc in {**c.docs, "summary": c.summary}.items():
        compare_docs(name, stored[name], json.loads(json.dumps(to_jsonable(doc))), diffs)
    out = {
        "analysis_id": a.id,
        "reproduced": not diffs,
        "differences": diffs[:50],
        "provenance_fingerprint": {"stored": a.provenance_fingerprint, "recomputed": c.fingerprint},
        "not_stored": "nothing was stored; `stress replay` re-executes the trials",
    }
    _emit(out, f"{'reproduced' if not diffs else 'DIFFERS'}: {a.id}", args)
    return 0 if not diffs else 1


def _cmd_stress_replay(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        out = stress_engine.replay_check(
            reg,
            LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR),
            _executor(reg, args.workspace),
            args.id,
            trials=not args.no_trials,
        )
    _emit(out, f"{'reproduced' if out['deterministic'] else 'DIFFERS'}: {args.id}", args)
    return 0 if out["deterministic"] else 1


# -- calibration & uncertainty ---------------------------------------------------------------------------------


def _calibration_spec(path: str) -> CalibrationSpec:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExperionyxError(f"cannot read calibration spec {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise ExperionyxError("a calibration spec file must be a JSON object")
    return CalibrationSpec.from_dict(doc)


def _cbstore(args: argparse.Namespace) -> LocalArtifactStore:
    return LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR)


def _calibration_row(a: CalibrationAnalysis) -> dict[str, object]:
    s = a.summary
    method = s.get("method")
    return {
        "id": a.id,
        "status": a.analysis_status,
        "spec_id": a.spec_id,
        "baseline_run_id": a.baseline_run_id,
        "dataset_fingerprint": a.dataset_fingerprint,
        "provenance_fingerprint": a.provenance_fingerprint,
        "prediction_representation": s.get("prediction_representation"),
        "binning": to_jsonable(s.get("binning")),
        "method": method.get("method") if isinstance(method, Mapping) else None,
        "baseline": to_jsonable(s.get("baseline")),
    }


def _calibration_result_row(r: CalibrationResult) -> dict[str, object]:
    return {
        "id": r.id,
        "context_key": r.context_key,
        "context_kind": r.context_kind,
        "status": r.status,
        "n_samples": r.n_samples,
        "reason": r.reason,
        "headline": to_jsonable(r.headline),
        "sample_digest": r.sample_digest,
    }


def _cmd_calibration_validate(args: argparse.Namespace) -> int:
    spec = _calibration_spec(args.spec)
    out: dict[str, object] = {
        "valid": True,
        "spec_id": spec.spec_id,
        "baseline_run": spec.baseline_run,
        "prediction_source": spec.prediction_source,
        "objects": list(spec.objects),
        "binning": spec.binning.to_dict(),
        "method": spec.method,
        "slices": {s.name: s.slice_id for s in spec.slices},
        "windows": [w.name for w in spec.windows.windows] if spec.windows else [],
        "stress_analyses": list(spec.stress_analyses),
    }
    if args.preflight:
        with _open(args.workspace) as reg:
            base = calibration_engine.validate(reg, _cbstore(args), spec)
            out["preflight"] = {
                "baseline_usable": len(base.obs),
                "baseline_rows": base.n_rows,
                "invalid_rows": len(base.invalid),
                "duplicate_ids": len(base.duplicates),
                "prediction_representation": base.representation,
                "classes": list(base.classes),
            }
    _emit(out, f"VALID: {spec.spec_id}", args)
    return 0


def _cmd_calibration_list(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        cols = {
            k: v
            for k, v in (("baseline_run_id", args.baseline_run), ("analysis_status", args.status))
            if v
        }
        rows = [
            _calibration_row(a) for a in CalibrationRegistry(reg, _cbstore(args)).analyses(**cols)
        ]
    _emit(
        {"analyses": rows}, "\n".join(f"{r['id']} {r['status']} {r['spec_id']}" for r in rows), args
    )
    return 0


def _cmd_calibration_inspect(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        cr = CalibrationRegistry(reg, _cbstore(args))
        if args.id.startswith("cbr_"):
            r = reg.get(CalibrationResult, args.id)
            doc: dict[str, object] = {**_calibration_result_row(r), "analysis_id": r.analysis_id}
        else:
            a = cr.analysis(args.id)
            doc = {
                **_calibration_row(a),
                "summary": to_jsonable(a.summary),
                "provenance": cr.provenance(a.id),
                "results": [_calibration_result_row(r) for r in cr.results(a.id)],
                "artifacts": [{"path": x.path, "id": x.id} for x in cr.artifacts(a.id)],
            }
            if args.document:
                doc["document"] = cr.document(a.id, args.document)
            if args.full:
                doc["documents"] = {n: cr.document(a.id, n) for n in calibration_engine.DOCUMENTS}
    _emit(doc, f"{args.id}: {doc.get('status')}", args)
    return 0


def _cmd_calibration_evaluate(args: argparse.Namespace) -> int:
    spec = _calibration_spec(args.spec)
    with _open(args.workspace) as reg:
        inv = (
            args.investigation
            or reg.get(Experiment, reg.get(Run, spec.baseline_run).experiment_id).investigation_id
        )
        out = calibration_engine.run_calibration_request(
            reg, _cbstore(args), _executor(reg, args.workspace), inv, spec
        )
        a = reg.get(CalibrationAnalysis, out.analysis_id) if out.analysis_id else None
    doc: dict[str, object] = {
        "status": out.status.value,
        "run_id": out.run_id,
        "analysis_id": out.analysis_id,
        "analysis_status": None if a is None else a.analysis_status,
        "summary": None if a is None else to_jsonable(a.summary),
    }
    _emit(doc, f"{out.status.value}: {out.analysis_id} ({doc['analysis_status']})", args)
    return 0 if a is not None and a.analysis_status == "COMPLETE" else 3


def _cmd_calibration_compare(args: argparse.Namespace) -> int:
    """Recompute the analysis from the stored predictions and compare with the stored documents; nothing is stored."""
    with _open(args.workspace) as reg:
        store = _cbstore(args)
        cr = CalibrationRegistry(reg, store)
        a = cr.analysis(args.id)
        spec = CalibrationSpec.from_dict(a.spec)
        dataset = None
        try:
            from experionyx.slices.data import dataset_for_run

            dataset = dataset_for_run(
                reg, default_registries(entry_points=True), Path(args.workspace), a.baseline_run_id
            )
        except ExperionyxError:
            dataset = None
        c = calibration_engine.compute(reg, store, spec, dataset)
        stored = {n: cr.document(a.id, n) for n in calibration_engine.DOCUMENTS[1:]}
    from experionyx.interactions.lifecycle import _compare as compare_docs

    diffs: list[str] = []
    for name, doc in c.docs.items():
        compare_docs(name, stored[name], json.loads(json.dumps(to_jsonable(doc))), diffs)
    out = {
        "analysis_id": a.id,
        "reproduced": not diffs,
        "differences": diffs[:50],
        "provenance_fingerprint": {"stored": a.provenance_fingerprint, "recomputed": c.fingerprint},
        "not_stored": "nothing was stored; `calibration replay` re-executes the analysis run",
    }
    _emit(out, f"{'reproduced' if not diffs else 'DIFFERS'}: {a.id}", args)
    return 0 if not diffs else 1


def _cmd_calibration_replay(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        out = calibration_engine.replay_check(
            reg, _cbstore(args), _executor(reg, args.workspace), args.id
        )
    _emit(out, f"{'reproduced' if out['deterministic'] else 'DIFFERS'}: {args.id}", args)
    return 0 if out["deterministic"] else 1


# -- resource & systems reliability ----------------------------------------------------------------------------


def _resource_spec(path: str) -> ResourceSpec:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExperionyxError(f"cannot read resource spec {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise ExperionyxError("a resource spec file must be a JSON object")
    return ResourceSpec.from_dict(doc)


def _rsstore(args: argparse.Namespace) -> LocalArtifactStore:
    return LocalArtifactStore(Path(args.workspace) / EXPERIMENTS_DIR)


def _resource_investigation(reg: SqliteRegistry, args: argparse.Namespace) -> str:
    if args.investigation:
        return str(args.investigation)
    found = reg.find(Investigation)
    if len(found) != 1:
        raise ExperionyxError(
            f"the workspace has {len(found)} investigations; choose one with --investigation"
        )
    return found[0].id


def _resource_row(a: ResourceAnalysis) -> dict[str, object]:
    s = a.summary
    env = s.get("environment")
    return {
        "id": a.id,
        "status": a.analysis_status,
        "spec_id": a.spec_id,
        "run_id": a.run_id,
        "model_id": a.model_id,
        "dataset_id": a.dataset_id,
        "provenance_fingerprint": a.provenance_fingerprint,
        "batch_size": to_jsonable(a.spec.get("batch_size")),
        "workers": to_jsonable(a.spec.get("workers")),
        "trials": to_jsonable(s.get("trials")),
        "evidence_status": s.get("evidence_status"),
        "environment_id": to_jsonable(
            env.get("environment_id") if isinstance(env, Mapping) else None
        ),
    }


def _resource_trial_row(t: ResourceTrial) -> dict[str, object]:
    return {
        "id": t.id,
        "phase": t.phase,
        "trial_index": t.trial_index,
        "status": t.status,
        "planned_samples": t.planned_samples,
        "completed_samples": t.completed_samples,
        "failed_samples": t.failed_samples,
        "wall_seconds": t.wall_seconds,
        "cpu_seconds": t.cpu_seconds,
    }


def _cmd_resources_validate(args: argparse.Namespace) -> int:
    spec = _resource_spec(args.spec)
    out: dict[str, object] = {
        "valid": True,
        "spec_id": spec.spec_id,
        "model_id": spec.model_id,
        "dataset_id": spec.dataset_id,
        "split": spec.split,
        "operation": spec.operation,
        "batch_size": spec.batch_size,
        "repeats": spec.repeats,
        "warmup_trials": spec.warmup_trials,
        "workers": spec.workers,
        "timeout_seconds": spec.timeout_seconds,
        "stress_ids": list(spec.stress_ids),
        "note": "the spec identity is the experimental definition; timing is environment-specific and is never part of it",
    }
    if args.preflight:
        with _open(args.workspace) as reg:
            adapters = default_registries(entry_points=True)
            out["preflight"] = resource_engine.preflight(
                reg, _rsstore(args), adapters, Path(args.workspace), spec
            )
    _emit(out, f"VALID: {spec.spec_id}", args)
    return 0


def _cmd_resources_list(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        cols = {k: v for k, v in (("spec_id", args.spec_id), ("analysis_status", args.status)) if v}
        rows = [_resource_row(a) for a in ResourceRegistry(reg, _rsstore(args)).analyses(**cols)]
    _emit(
        {"analyses": rows}, "\n".join(f"{r['id']} {r['status']} {r['spec_id']}" for r in rows), args
    )
    return 0


def _cmd_resources_inspect(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        rr = ResourceRegistry(reg, _rsstore(args))
        if args.id.startswith("rst_"):
            t = reg.get(ResourceTrial, args.id)
            doc: dict[str, object] = {**_resource_trial_row(t), "analysis_id": t.analysis_id}
        else:
            a = rr.analysis(args.id)
            doc = {
                **_resource_row(a),
                "summary": to_jsonable(a.summary),
                "provenance": rr.provenance(a.id),
                "trials": [_resource_trial_row(t) for t in rr.trials(a.id)],
                "artifacts": [
                    {"path": x.path, "id": x.id, "digest": x.digest} for x in rr.artifacts(a.id)
                ],
            }
            if args.document:
                doc["document"] = rr.document(a.id, args.document)
            if args.full:
                doc["documents"] = {n: rr.document(a.id, n) for n in resource_engine.DOCUMENTS}
    _emit(doc, f"{args.id}: {doc.get('status')}", args)
    return 0


def _cmd_resources_run(args: argparse.Namespace) -> int:
    spec = _resource_spec(args.spec)
    with _open(args.workspace) as reg:
        out = resource_engine.run_resource_request(
            reg,
            _rsstore(args),
            _executor(reg, args.workspace),
            _resource_investigation(reg, args),
            spec,
        )
        a = reg.get(ResourceAnalysis, out.analysis_id) if out.analysis_id else None
    doc: dict[str, object] = {
        "status": out.status.value,
        "run_id": out.run_id,
        "analysis_id": out.analysis_id,
        "analysis_status": None if a is None else a.analysis_status,
        "summary": None if a is None else to_jsonable(a.summary),
        "note": "environment-specific engineering measurement",
    }
    _emit(doc, f"{out.status.value}: {out.analysis_id} ({doc['analysis_status']})", args)
    return 0 if a is not None and a.analysis_status == "COMPLETE" else 3


def _cmd_resources_sweep(args: argparse.Namespace) -> int:
    spec = _resource_spec(args.spec)
    try:
        sizes = [int(x) for x in args.batch_sizes.split(",")]
    except ValueError as exc:
        raise ExperionyxError(f"--batch-sizes must be comma-separated integers: {exc}") from exc
    with _open(args.workspace) as reg:
        units = resource_compare.run_sweep(
            reg,
            _rsstore(args),
            _executor(reg, args.workspace),
            _resource_investigation(reg, args),
            spec,
            sizes,
        )
        ids = [u["analysis_id"] for u in units]
        cmp_doc = None
        if len(ids) > 1 and all(ids):
            cmp_doc = resource_compare.compare_analyses(
                reg, _rsstore(args), ids, metric=args.metric, correction=args.correction
            )
    doc: dict[str, object] = {
        "units": units,
        "comparison": cmp_doc,
        "note": "every batch size is a separate persisted measurement; environment-specific engineering measurement",
    }
    _emit(
        doc,
        "\n".join(f"batch_size={u['batch_size']} {u['status']} {u['analysis_id']}" for u in units),
        args,
    )
    return 0 if all(u["status"] == "COMPLETED" and u["analysis_id"] for u in units) else 3


def _cmd_resources_compare(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        out = resource_compare.compare_analyses(
            reg,
            _rsstore(args),
            args.ids,
            metric=args.metric,
            correction=args.correction,
            allow_environment_mismatch=args.allow_environment_mismatch,
        )
    _emit(out, f"compared {len(args.ids)} analyses on {args.metric}", args)
    return 0


def _cmd_resources_replay(args: argparse.Namespace) -> int:
    with _open(args.workspace) as reg:
        out = resource_engine.replay_check(
            reg, _rsstore(args), _executor(reg, args.workspace), args.id
        )
    _emit(
        out,
        f"{'definition reproduced' if out['definition_reproduced'] else 'DIFFERS'}: {args.id}",
        args,
    )
    return 0 if out["definition_reproduced"] else 1


def _cmd_viz_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from experionyx.api.app import create_app
    except ModuleNotFoundError as exc:
        raise ExperionyxError(
            "the visualization UI needs optional dependencies: pip install experionyx[viz]"
        ) from exc
    app = create_app(args.workspace)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="experionyx", description="EXPERIONYX: AI Experimental Forensics & Reliability Lab"
    )
    parser.add_argument("--version", action="version", version=f"experionyx {__version__}")
    parser.add_argument(
        "--workspace",
        default=DEFAULT_WORKSPACE,
        help=f"workspace directory (default {DEFAULT_WORKSPACE})",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log lifecycle events")
    sub = parser.add_subparsers(dest="command")

    def add(
        name: str, func: Callable[[argparse.Namespace], int], help_: str
    ) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_)
        p.set_defaults(func=func)
        return p

    add("info", _cmd_info, "print package and environment information")
    add("status", _cmd_status, "summarize the registry")
    add("experiment", _cmd_experiment, "show an experiment and its runs").add_argument("id")
    add("run", _cmd_run, "show a run, its outcome, observations and artifacts").add_argument("id")
    add("provenance", _cmd_provenance, "show a run's provenance and fingerprint").add_argument("id")
    add(
        "verify", _cmd_verify, "re-hash a run's artifacts against their recorded digests"
    ).add_argument("id")
    p = add("execute", _cmd_execute, "execute a registered experiment as a new run")
    p.add_argument("id", help="experiment ID")
    p.add_argument("--procedure", required=True, help="importable 'module:function' to execute")
    p.add_argument("--seed", required=True, type=int, help="explicit random seed")
    p = add("replay", _cmd_replay, "request a replay of a finished run as a NEW run")
    p.add_argument("id", help="run ID")
    p.add_argument("--procedure", help="override the recorded procedure ('module:function')")
    add("recover", _cmd_recover, "mark RUNNING runs whose process has ended as FAILED")
    add("init", _cmd_init, "create an empty workspace registry")
    p = add("demo", _cmd_demo, "train, register and execute a small real adapter-backed experiment")
    p.add_argument("name", choices=DEMOS)
    p.add_argument("--seed", type=int, default=0)

    def group(name: str, help_: str) -> "argparse._SubParsersAction[argparse.ArgumentParser]":
        return sub.add_parser(name, help=help_).add_subparsers(
            dest=f"{name}_command", required=True
        )

    def loader_args(p: argparse.ArgumentParser, *, model: bool) -> None:
        p.add_argument(
            "--adapter", help="adapter name (required unless inspecting a registered ID)"
        )
        p.add_argument("--version", default="1", help="version to record (default 1)")
        p.add_argument("--option", action="append", default=[], help="adapter option key=value")
        if model:
            p.add_argument("--device", default="CPU", choices=[d.value for d in DeviceKind])

    adapters = group("adapters", "list and inspect model/dataset adapters")
    adapters.add_parser("list", help="list adapters and their capabilities").set_defaults(
        func=_cmd_adapters_list
    )
    p = adapters.add_parser("inspect", help="show one adapter's details")
    p.add_argument("name")
    p.set_defaults(func=_cmd_adapters_inspect)

    models = group("model", "inspect or register a model")
    p = models.add_parser("inspect", help="inspect a registered model ID or load a file")
    p.add_argument("target", help="model ID (mdl_...) or a source path")
    loader_args(p, model=True)
    p.set_defaults(func=_cmd_model_inspect)
    p = models.add_parser("register", help="load a model artifact and register it")
    p.add_argument("--name", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--version", default="1")
    p.add_argument("--option", action="append", default=[])
    p.set_defaults(func=_cmd_model_register)

    def eval_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--dataset", required=True, help="registered dataset ID (dst_...)")
        p.add_argument("--config", help="EvaluationConfig JSON file (strict: unknown fields fail)")
        p.add_argument("--split", help="dataset split to evaluate")
        p.add_argument("--batch-size", type=int)
        p.add_argument(
            "--metric", action="append", help="metric ID (repeatable); default: all applicable"
        )
        p.add_argument("--score-source", choices=[x.value for x in ScoreSource])
        p.add_argument("--bins", type=int, help="calibration bins")
        p.add_argument("--bootstrap-resamples", type=int, help="0 disables bootstrap intervals")
        p.add_argument("--seed", type=int, default=0, help="run seed (default 0)")

    p = models.add_parser(
        "autopsy", help="evaluate a registered model and print its profile and findings"
    )
    p.add_argument("model_id", help="registered model ID (mdl_...)")
    eval_args(p)
    p.set_defaults(func=_cmd_autopsy)

    p = add(
        "evaluate", _cmd_evaluate, "run a baseline evaluation of a registered model on a dataset"
    )
    p.add_argument("--model", required=True, help="registered model ID (mdl_...)")
    eval_args(p)
    metrics = group("metrics", "list the metric registry")
    p = metrics.add_parser("list", help="metrics, their tasks, requirements and scales")
    p.add_argument("--task", choices=["CLASSIFICATION", "REGRESSION"])
    p.set_defaults(func=_cmd_metrics_list)
    evals = group("evaluation", "inspect and compare stored evaluations")
    p = evals.add_parser("inspect", help="summary of a run's stored evaluation")
    p.add_argument("run")
    p.add_argument("--full", action="store_true", help="print the complete EvaluationResult")
    p.set_defaults(func=_cmd_evaluation_inspect)
    p = evals.add_parser("compare", help="structured comparison of two evaluation runs (no winner)")
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.set_defaults(func=_cmd_evaluation_compare)

    add("faults", _cmd_faults_list, "list the registered fault types")
    fault = group("fault", "fault-injection experiments")
    p = fault.add_parser("inspect", help="one fault type: parameters, requirements, semantics")
    p.add_argument("name")
    p.set_defaults(func=_cmd_fault_inspect)

    def fault_args(q: argparse.ArgumentParser, *, sweep: bool) -> None:
        q.add_argument("--model", required=True, help="registered model ID (mdl_...)")
        eval_args(q)
        q.add_argument("--type", help="fault type (see `experionyx faults`)")
        q.add_argument("--param", action="append", default=[], help="fault parameter key=value")
        q.add_argument("--spec-file", help="a complete FaultSpec JSON (needed for compound faults)")
        q.add_argument("--scope-fraction", type=float, help="apply to this share of samples")
        q.add_argument("--scope-class", help="apply to samples of this true class")
        q.add_argument("--primary-metric", help="metric the effect is classified on")
        q.add_argument(
            "--baseline-run", help="reuse a COMPLETED baseline run with identical configuration"
        )
        q.add_argument(
            "--max-failed-trials",
            type=int,
            help="early termination: skip remaining trials after N failures",
        )
        q.add_argument("--name", help="experiment name")
        if sweep:
            q.add_argument(
                "--sweep", required=True, help="parameter=v1,v2,... (each value is a real run)"
            )
            q.add_argument(
                "--seeds", default="0", help="comma-separated seeds (each is a real run per point)"
            )

    p = fault.add_parser("run", help="one fault, one seed: control run + faulted run + analysis")
    fault_args(p, sweep=False)
    p.set_defaults(func=_cmd_fault_run)
    p = fault.add_parser("sweep", help="a parameter sweep with repeated seeds")
    fault_args(p, sweep=True)
    p.set_defaults(func=_cmd_fault_sweep)
    p = fault.add_parser("compare", help="degradation of a treatment run against its baseline run")
    p.add_argument("baseline_run")
    p.add_argument("treatment_run")
    p.set_defaults(func=_cmd_fault_compare)
    p = fault.add_parser("demo", help="run real example fault experiments")
    p.add_argument("name", choices=FAULT_DEMOS)
    p.set_defaults(func=_cmd_fault_demo)
    fexp = fault.add_parser("experiment", help="fault experiment records").add_subparsers(
        dest="fexp_command", required=True
    )
    p = fexp.add_parser("inspect", help="design, trials and analysis of a fault experiment")
    p.add_argument("id", help="fault experiment ID (fxp_...)")
    p.add_argument("--full", action="store_true", help="include the complete analysis")
    p.set_defaults(func=_cmd_fault_experiment_inspect)

    def fail_filters(q: argparse.ArgumentParser) -> None:
        q.add_argument("--investigation", help="only this investigation (inv_...)")
        q.add_argument("--status", choices=[x.value for x in FailureStatus])
        q.add_argument("--category", choices=[x.value for x in FailureCategory])

    fail_filters(add("failures", _cmd_failures_list, "list registered failure modes"))
    fail = group("failure", "failure discovery and the failure registry")
    p = fail.add_parser("inspect", help="a failure mode, cluster or signal")
    p.add_argument("id")
    p.set_defaults(func=_cmd_failure_inspect)
    p = fail.add_parser(
        "discover", help="run failure discovery over stored runs and fault experiments"
    )
    p.add_argument(
        "--fault-experiment", action="append", default=[], help="fault experiment ID (repeatable)"
    )
    p.add_argument("--run", action="append", default=[], help="evaluation run ID (repeatable)")
    p.add_argument(
        "--investigation",
        help="home investigation of the result (inferred when all sources share one)",
    )
    p.add_argument("--config", help="DiscoveryConfig JSON file (default: built-in configuration)")
    p.add_argument(
        "--seed", type=int, default=0, help="run seed (the discovery itself is deterministic)"
    )
    p.set_defaults(func=_cmd_failure_discover)
    p = fail.add_parser(
        "cluster", help="list clusters, or show one cluster with its member signals"
    )
    p.add_argument("id", nargs="?")
    p.add_argument("--investigation")
    p.set_defaults(func=_cmd_failure_cluster)
    p = fail.add_parser("candidates", help="modes not yet confirmed, with their criteria checks")
    fail_filters(p)
    p.set_defaults(func=_cmd_failure_candidates)
    p = fail.add_parser("evidence", help="all retained evidence of a mode")
    p.add_argument("id")
    p.set_defaults(func=_cmd_failure_evidence)
    p = fail.add_parser("reproduce", help="replay supporting runs and compare within tolerances")
    p.add_argument("id")
    p.add_argument("--config", help="the DiscoveryConfig JSON the mode was discovered with")
    p.set_defaults(func=_cmd_failure_reproduce)
    p = fail.add_parser("confirm", help="CONFIRM a SUPPORTED, reproduced mode (explicit decision)")
    p.add_argument("id")
    p.add_argument("--by", required=True, help="who is making the decision")
    p.add_argument("--reason", required=True)
    p.set_defaults(func=_cmd_failure_confirm)
    p = fail.add_parser("set-status", help="reject or deprecate (or otherwise move) a mode")
    p.add_argument("id")
    p.add_argument(
        "--to",
        required=True,
        choices=[x.value for x in FailureStatus if x is not FailureStatus.CONFIRMED],
    )
    p.add_argument("--by", required=True)
    p.add_argument("--reason", required=True)
    p.set_defaults(func=_cmd_failure_set_status)
    p = fail.add_parser(
        "graph", help="relationships around a mode (backend graph, no visualization)"
    )
    p.add_argument("id")
    p.set_defaults(func=_cmd_failure_graph)

    ia = group("interaction", "fault interaction analysis (observed contrasts, not causal claims)")

    def fmt(q: argparse.ArgumentParser) -> None:
        q.add_argument(
            "--format",
            choices=["json", "text"],
            default="json",
            help="output format (default json)",
        )

    p = ia.add_parser(
        "validate", help="check a design; refuses invalid designs and lists every issue"
    )
    p.add_argument("spec", help="spec JSON file (cells of run IDs + config)")
    fmt(p)
    p.set_defaults(func=_cmd_interaction_validate)
    p = ia.add_parser("analyze", help="validate, then analyze a design as a real Run")
    p.add_argument("spec")
    p.add_argument("--investigation", help="home investigation (default: the control run's)")
    p.add_argument("--seed", type=int, default=0)
    fmt(p)
    p.set_defaults(func=_cmd_interaction_analyze)
    p = ia.add_parser("list", help="search registered interactions")
    for flag in (
        "fault",
        "metric",
        "status",
        "failure-mode",
        "experiment",
        "dataset",
        "model",
        "investigation",
    ):
        p.add_argument(f"--{flag}")
    p.add_argument("--class", dest="klass", help="primary-metric interaction class")
    fmt(p)
    p.set_defaults(func=_cmd_interaction_list)
    p = ia.add_parser("inspect", help="one interaction with its effects and evidence")
    p.add_argument("id")
    p.add_argument(
        "--full", action="store_true", help="include every effect record and the raw trials"
    )
    fmt(p)
    p.set_defaults(func=_cmd_interaction_inspect)
    p = ia.add_parser("failures", help="how failure modes are observed across the design's cells")
    p.add_argument("id")
    p.set_defaults(func=_cmd_interaction_failures)
    p = ia.add_parser(
        "replay",
        help="replay the analysis as a NEW run and compare every result; exit 1 on any difference",
    )
    p.add_argument("id")
    p.set_defaults(func=_cmd_interaction_replay)
    p = ia.add_parser("export", help="a self-contained JSON bundle of an interaction")
    p.add_argument("id")
    p.add_argument("--out")
    p.set_defaults(func=_cmd_interaction_export)
    p = ia.add_parser("related", help="structurally similar interactions (never merged)")
    p.add_argument("id")
    p.set_defaults(func=_cmd_interaction_related)
    p = ia.add_parser(
        "reproduce", help="check an independent replicate analysis against a SUPPORTED one"
    )
    p.add_argument("id")
    p.add_argument("--replicate", required=True)
    p.add_argument("--abs-tol", type=float, default=0.0)
    p.add_argument("--rel-tol", type=float, default=0.5)
    p.set_defaults(func=_cmd_interaction_reproduce)
    p = ia.add_parser(
        "confirm",
        help="CONFIRMED_BY_REVIEW: an explicit human decision on a REPRODUCIBLE interaction",
    )
    p.add_argument("id")
    p.add_argument("--by", required=True)
    p.add_argument("--reason", required=True)
    p.set_defaults(func=_cmd_interaction_confirm)
    p = ia.add_parser("set-status", help="reject or deprecate")
    p.add_argument("id")
    p.add_argument("--to", required=True, choices=["REJECTED", "DEPRECATED"])
    p.add_argument("--by", required=True)
    p.add_argument("--reason", required=True)
    p.set_defaults(func=_cmd_interaction_set_status)

    rl = group("reliability", "evidence-first reliability profiles (no overall score or ranking)")

    def rfmt(q: argparse.ArgumentParser) -> None:
        q.add_argument(
            "--format",
            choices=["json", "text"],
            default="json",
            help="output format (default json)",
        )

    p = rl.add_parser(
        "profile", help="build a profile from stored evidence (refused if sources are incompatible)"
    )
    p.add_argument(
        "spec",
        help="profile spec JSON: scope, baseline_run, fault_experiments, interactions, failure_modes",
    )
    p.add_argument("--investigation")
    p.add_argument("--seed", type=int, default=0)
    rfmt(p)
    p.set_defaults(func=_cmd_reliability_profile)
    p = rl.add_parser("list", help="search registered profiles")
    for flag in (
        "model",
        "dataset",
        "scope",
        "split",
        "evaluation",
        "investigation",
        "ref",
        "dimension",
        "dimension-status",
    ):
        p.add_argument(f"--{flag}")
    rfmt(p)
    p.set_defaults(func=_cmd_reliability_list)
    p = rl.add_parser("inspect", help="one profile: dimension statuses and observations")
    p.add_argument("id")
    p.add_argument("--dimension", help="show one dimension in full")
    p.add_argument("--full", action="store_true", help="include the complete profile document")
    rfmt(p)
    p.set_defaults(func=_cmd_reliability_inspect)
    p = rl.add_parser("evidence", help="source references, provenance and artifacts of a profile")
    p.add_argument("id")
    p.add_argument("--dimension")
    p.set_defaults(func=_cmd_reliability_evidence)
    p = rl.add_parser(
        "compare", help="raw per-dimension differences between two compatible profiles"
    )
    p.add_argument("a")
    p.add_argument("b")
    rfmt(p)
    p.set_defaults(func=_cmd_reliability_compare)
    p = rl.add_parser(
        "replay", help="replay the profile as a NEW run and compare; exit 1 on any difference"
    )
    p.add_argument("id")
    p.set_defaults(func=_cmd_reliability_replay)

    bm = group("benchmark", "standardized robustness benchmarks (coverage and evidence; no score)")

    def bfmt(q: argparse.ArgumentParser) -> None:
        q.add_argument(
            "--format",
            choices=["json", "text"],
            default="json",
            help="output format (default json)",
        )

    p = bm.add_parser(
        "validate",
        help="expand and check a benchmark spec; refuses invalid definitions with every issue listed",
    )
    p.add_argument("spec", help="benchmark spec JSON file")
    bfmt(p)
    p.set_defaults(func=_cmd_benchmark_validate)
    p = bm.add_parser(
        "run",
        help="execute a benchmark protocol and collect its coverage and results (exit 3 if coverage is incomplete)",
    )
    p.add_argument("spec")
    bfmt(p)
    p.set_defaults(func=_cmd_benchmark_run)
    p = bm.add_parser("list", help="search registered benchmark definitions")
    for flag in ("name", "version", "model", "dataset", "protocol", "engine", "fault", "coverage"):
        p.add_argument(f"--{flag}")
    bfmt(p)
    p.set_defaults(func=_cmd_benchmark_list)
    p = bm.add_parser("inspect", help="a benchmark definition (bmk_) or a result (brs_)")
    p.add_argument("id")
    p.add_argument("--full", action="store_true", help="include every result document")
    bfmt(p)
    p.set_defaults(func=_cmd_benchmark_inspect)
    p = bm.add_parser("coverage", help="the coverage account of a result (exit 3 if incomplete)")
    p.add_argument("id")
    bfmt(p)
    p.set_defaults(func=_cmd_benchmark_coverage)
    p = bm.add_parser(
        "compare", help="raw differences between two results of the SAME protocol (no winner)"
    )
    p.add_argument("a")
    p.add_argument("b")
    bfmt(p)
    p.set_defaults(func=_cmd_benchmark_compare)
    p = bm.add_parser(
        "replay",
        help="replay the collect run as a NEW run and compare; exit 1 unless verified deterministic",
    )
    p.add_argument("id")
    p.set_defaults(func=_cmd_benchmark_replay)
    p = bm.add_parser(
        "protocol", help="register a benchmark spec's model-independent protocol identity"
    )
    p.add_argument(
        "spec", help="benchmark spec JSON file (the model field is ignored for identity)"
    )
    bfmt(p)
    p.set_defaults(func=_cmd_benchmark_protocol)
    p = bm.add_parser(
        "submit",
        help="run a full benchmark (Phase 9) and submit its evidence to a registered protocol",
    )
    p.add_argument("spec", help="benchmark spec JSON file, including the model")
    p.add_argument(
        "--protocol", required=True, help="protocol ID (bpr_...) or a protocol_hash prefix"
    )
    bfmt(p)
    p.set_defaults(func=_cmd_benchmark_submit)

    lb = group(
        "leaderboard",
        "reporting/organization over benchmark submissions; never a universal score or ranking",
    )

    def lfmt(q: argparse.ArgumentParser) -> None:
        q.add_argument("--format", choices=["json", "text"], default="json")

    p = lb.add_parser("list", help="list registered benchmark protocols")
    lfmt(p)
    p.set_defaults(func=_cmd_leaderboard_list)
    p = lb.add_parser(
        "inspect", help="a protocol (bpr_), submission (bsb_), snapshot (lbs_) or entry (lbe_)"
    )
    p.add_argument("id")
    lfmt(p)
    p.set_defaults(func=_cmd_leaderboard_inspect)
    p = lb.add_parser(
        "snapshot",
        help="build (or reuse) an immutable snapshot of every qualifying submission to a protocol",
    )
    p.add_argument("protocol", help="protocol ID (bpr_...) or a protocol_hash prefix")
    p.add_argument(
        "--metric",
        action="append",
        help="restrict to this metric id (repeatable); default: every metric present",
    )
    p.add_argument("--require-complete-coverage", action="store_true")
    p.add_argument("--require-reproduction", action="store_true")
    lfmt(p)
    p.set_defaults(func=_cmd_leaderboard_snapshot)
    p = lb.add_parser(
        "compare", help="protocol-constrained raw comparison of two submissions (no winner)"
    )
    p.add_argument("a", help="submission ID (bsb_...)")
    p.add_argument("b", help="submission ID (bsb_...)")
    lfmt(p)
    p.set_defaults(func=_cmd_leaderboard_compare)
    p = lb.add_parser(
        "correct",
        help="multiple-comparison correction over an explicit family of registered stats analyses",
    )
    p.add_argument("analysis", nargs="+", help="sta_... COMPARE analysis IDs forming the family")
    p.add_argument("--method", choices=list(CORRECTION_METHODS), default="NONE")
    p.add_argument("--alpha", type=float, default=0.05)
    p.set_defaults(func=_cmd_leaderboard_correct)
    p = lb.add_parser("verify", help="re-hash a submission's underlying benchmark run artifacts")
    p.add_argument("id", help="submission ID (bsb_...)")
    p.set_defaults(func=_cmd_leaderboard_verify)
    p = lb.add_parser(
        "replay", help="rebuild a snapshot from current evidence; exit 1 if the evidence changed"
    )
    p.add_argument("id", help="snapshot ID (lbs_...)")
    lfmt(p)
    p.set_defaults(func=_cmd_leaderboard_replay)

    st = group(
        "stats",
        "statistical analysis: intervals, comparisons, corrections (recorded, reproducible)",
    )

    def sfmt(q: argparse.ArgumentParser) -> None:
        q.add_argument("--format", choices=["json", "text"], default="json")

    def ssource(q: argparse.ArgumentParser, *, treatment: bool) -> None:
        q.add_argument("--reference-values", help="comma-separated numbers")
        q.add_argument("--treatment-values", help="comma-separated numbers")
        q.add_argument(
            "--file",
            help='JSON {"reference": ..., "treatment": ...}; mappings key->value keep identity (needed for PAIRED)',
        )
        q.add_argument(
            "--interaction", help="an interaction analysis ID (ian_): read its trial values"
        )
        q.add_argument("--measure", help="metric name (interaction and fault sources)")
        q.add_argument(
            "--fault-experiment",
            help="a fault experiment ID (fxp_): per-seed values of its analysis",
        )
        q.add_argument("--point-index", type=int, default=0)
        q.add_argument(
            "--reference-field",
            default="baseline",
            help="fault trial field: baseline|faulted|deterioration|...",
        )
        q.add_argument("--treatment-field", default="faulted")
        q.add_argument("--reference-cell", default="CONTROL")
        q.add_argument("--treatment-cell")
        q.add_argument(
            "--artifact", nargs=2, metavar=("RUN", "PATH"), help="a run artifact holding the values"
        )
        q.add_argument("--reference-pointer", help="a/b/c path to the values inside the artifact")
        q.add_argument("--treatment-pointer")
        q.add_argument(
            "--pairing",
            choices=["PAIRED", "UNPAIRED", "UNKNOWN"],
            help="default UNKNOWN (analyzed as UNPAIRED); interaction sources default to the design's",
        )
        q.add_argument("--estimator", choices=["mean", "median"])
        q.add_argument("--method", choices=["percentile", "bca"])
        q.add_argument("--confidence", type=float)
        q.add_argument("--resamples", type=int)
        q.add_argument("--seed", type=int)
        sfmt(q)

    p = st.add_parser(
        "compare", help="treatment vs reference: effect sizes, permutation test, bootstrap CI"
    )
    ssource(p, treatment=True)
    p.add_argument("--permutations", type=int)
    p.set_defaults(func=_cmd_stats_compare)
    p = st.add_parser(
        "bootstrap",
        help="deterministic bootstrap CI (percentile or BCa) of one sample, a paired or an unpaired difference",
    )
    ssource(p, treatment=False)
    p.set_defaults(func=_cmd_stats_bootstrap)
    p = st.add_parser(
        "proportion",
        help="Wilson interval for a count of successes (or a failure mode's prevalence)",
    )
    p.add_argument("--successes", type=int)
    p.add_argument("--trials", type=int)
    p.add_argument("--failure-mode", help="a failure mode ID (fmd_)")
    p.add_argument("--confidence", type=float)
    sfmt(p)
    p.set_defaults(func=_cmd_stats_proportion)
    p = st.add_parser(
        "correct", help="explicit multiple-comparison correction over a named family of p-values"
    )
    p.add_argument("--p", action="append", default=[], metavar="NAME=P")
    p.add_argument(
        "--analysis", action="append", default=[], help="a registered COMPARE analysis (sta_)"
    )
    p.add_argument("--method", choices=["NONE", "BONFERRONI", "BENJAMINI_HOCHBERG"])
    p.add_argument("--alpha", type=float)
    sfmt(p)
    p.set_defaults(func=_cmd_stats_correct)
    p = st.add_parser("list", help="registered statistical analyses")
    p.add_argument("--kind", choices=["COMPARE", "BOOTSTRAP", "PROPORTION", "CORRECTION"])
    sfmt(p)
    p.set_defaults(func=_cmd_stats_list)
    p = st.add_parser("inspect", help="one analysis: settings, sources, result")
    p.add_argument("id")
    sfmt(p)
    p.set_defaults(func=_cmd_stats_inspect)
    p = st.add_parser(
        "verify", help="recompute from the recorded sources; exit 1 unless reproduced"
    )
    p.add_argument("id")
    sfmt(p)
    p.set_defaults(func=_cmd_stats_verify)

    sl = group(
        "slice",
        "slice and subgroup analysis (per-population evidence; no ranking, no fairness score)",
    )

    def slfmt(q: argparse.ArgumentParser) -> None:
        q.add_argument("--format", choices=["json", "text"], default="json")

    p = sl.add_parser(
        "validate",
        help="parse and normalize slice definitions or an analysis spec; prints their identities",
    )
    p.add_argument("spec")
    slfmt(p)
    p.set_defaults(func=_cmd_slice_validate)
    p = sl.add_parser(
        "register", help="register slice definitions (equivalent definitions are one record)"
    )
    p.add_argument("spec")
    slfmt(p)
    p.set_defaults(func=_cmd_slice_register)
    p = sl.add_parser("list", help="search registered slices (and optionally analyses)")
    p.add_argument("--name")
    p.add_argument("--field")
    p.add_argument("--text")
    p.add_argument(
        "--static", choices=["yes", "no"], help="yes: membership does not depend on model output"
    )
    p.add_argument("--analyses", action="store_true")
    p.add_argument("--baseline-run")
    slfmt(p)
    p.set_defaults(func=_cmd_slice_list)
    p = sl.add_parser("inspect", help="a slice (sls_) or a slice analysis (san_) with provenance")
    p.add_argument("id")
    p.add_argument(
        "--full",
        action="store_true",
        help="include the spec, membership, results and summary documents",
    )
    slfmt(p)
    p.set_defaults(func=_cmd_slice_inspect)
    p = sl.add_parser(
        "evaluate",
        help="membership of slice definitions over a baseline run's samples (nothing is stored)",
    )
    p.add_argument("spec")
    p.add_argument("--baseline-run", required=True)
    p.add_argument("--ids", action="store_true", help="list member sample IDs")
    slfmt(p)
    p.set_defaults(func=_cmd_slice_evaluate)
    p = sl.add_parser(
        "analyze",
        help="run a slice analysis as a new run (exit 3 if some evidence is insufficient or unavailable)",
    )
    p.add_argument("spec")
    slfmt(p)
    p.set_defaults(func=_cmd_slice_analyze)
    p = sl.add_parser(
        "compare",
        help="one slice vs POPULATION, REST or another slice on a baseline run (descriptive delta, effect sizes, test)",
    )
    p.add_argument("a", help="slice definition file")
    p.add_argument("b", help="slice definition file, POPULATION or REST")
    p.add_argument("--baseline-run", required=True)
    p.add_argument("--confidence", type=float)
    p.add_argument("--resamples", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--method", choices=["percentile", "bca"])
    p.add_argument("--min-members", type=int)
    slfmt(p)
    p.set_defaults(func=_cmd_slice_compare)

    dr = group(
        "drift",
        "temporal and distribution shift between explicit windows (observed differences; no drift score, no causal claim)",
    )
    p = dr.add_parser(
        "validate",
        help="parse and normalize a drift spec; prints its identity and window plan (--preflight also checks the data)",
    )
    p.add_argument("spec")
    p.add_argument(
        "--preflight",
        action="store_true",
        help="also load the baseline and dataset and refuse what the data cannot support",
    )
    slfmt(p)
    p.set_defaults(func=_cmd_drift_validate)
    p = dr.add_parser("list", help="list drift analyses (and optionally registered windows)")
    p.add_argument("--baseline-run")
    p.add_argument("--status", choices=["COMPLETE", "PARTIAL"])
    p.add_argument("--windows", action="store_true", help="also list registered windows")
    p.add_argument("--ordering", help="filter windows by ordering field")
    slfmt(p)
    p.set_defaults(func=_cmd_drift_list)
    p = dr.add_parser("inspect", help="a drift analysis (dan_) with provenance, or a window (twn_)")
    p.add_argument("id")
    p.add_argument("--full", action="store_true", help="include every stored document")
    slfmt(p)
    p.set_defaults(func=_cmd_drift_inspect)
    p = dr.add_parser(
        "windows",
        help="resolve a spec's windows against the baseline data: members, digests, skipped windows (nothing is stored)",
    )
    p.add_argument("spec")
    p.add_argument("--ids", action="store_true", help="list member sample IDs")
    slfmt(p)
    p.set_defaults(func=_cmd_drift_windows)
    p = dr.add_parser(
        "evaluate",
        help="run a drift analysis as a new run (exit 3 if some evidence is insufficient, unavailable or skipped)",
    )
    p.add_argument("spec")
    slfmt(p)
    p.set_defaults(func=_cmd_drift_evaluate)
    p = dr.add_parser("compare", help="compute a drift analysis without storing anything")
    p.add_argument("spec")
    p.add_argument("--full", action="store_true", help="include every per-feature result")
    slfmt(p)
    p.set_defaults(func=_cmd_drift_compare)
    p = dr.add_parser(
        "replay",
        help="replay the analysis as a NEW run and compare every document; exit 1 on any difference",
    )
    p.add_argument("id")
    slfmt(p)
    p.set_defaults(func=_cmd_drift_replay)

    dq = group(
        "data-quality",
        "data-quality checks on registered datasets (multidimensional evidence; no quality score, no universal verdict)",
    )
    p = dq.add_parser(
        "validate",
        help="parse and normalize a quality spec; prints its identity (--preflight also checks the data)",
    )
    p.add_argument("spec")
    p.add_argument(
        "--preflight",
        action="store_true",
        help="also load the dataset and refuse what it cannot support",
    )
    slfmt(p)
    p.set_defaults(func=_cmd_quality_validate)
    p = dq.add_parser("list", help="list quality analyses")
    p.add_argument("--dataset-id")
    p.add_argument("--status", choices=["COMPLETE", "PARTIAL"])
    slfmt(p)
    p.set_defaults(func=_cmd_quality_list)
    p = dq.add_parser(
        "inspect", help="a quality analysis (qan_) with provenance, or one check result (qck_)"
    )
    p.add_argument("id")
    p.add_argument("--full", action="store_true", help="include every stored document")
    slfmt(p)
    p.set_defaults(func=_cmd_quality_inspect)
    p = dq.add_parser("checks", help="the check catalog, or the per-check results of an analysis")
    p.add_argument("id", nargs="?", help="analysis ID (qan_); omit for the catalog")
    p.add_argument("--catalog", action="store_true")
    p.add_argument("--type")
    p.add_argument(
        "--status",
        choices=["PASS", "FAIL", "WARNING", "INCONCLUSIVE", "UNAVAILABLE", "NOT_APPLICABLE"],
    )
    slfmt(p)
    p.set_defaults(func=_cmd_quality_checks)
    p = dq.add_parser(
        "run",
        help="run a quality analysis as a new run (exit 3 if any check FAILED or evidence is incomplete)",
    )
    p.add_argument("spec")
    p.add_argument("--investigation", help="investigation ID (default: the only one)")
    slfmt(p)
    p.set_defaults(func=_cmd_quality_run)
    p = dq.add_parser(
        "replay",
        help="replay the analysis as a NEW run and compare every document; exit 1 on any difference",
    )
    p.add_argument("id")
    slfmt(p)
    p.set_defaults(func=_cmd_quality_replay)

    sx = group(
        "stress",
        "controlled model stress (input, parameter, decision-rule, evaluation-condition); observed changes, no robustness score, no causal claim",
    )
    p = sx.add_parser(
        "families",
        help="the stress families, their origin (Fault Laboratory or model-level) and parameters",
    )
    slfmt(p)
    p.set_defaults(func=_cmd_stress_families)
    p = sx.add_parser(
        "validate",
        help="parse a stress design; prints its identity and every expanded trial (--preflight checks the model, dataset and baseline)",
    )
    p.add_argument("spec")
    p.add_argument(
        "--preflight",
        action="store_true",
        help="also load the registered model and refuse what it cannot support",
    )
    slfmt(p)
    p.set_defaults(func=_cmd_stress_validate)
    p = sx.add_parser("list", help="list stress analyses")
    p.add_argument("--model-id")
    p.add_argument("--dataset-id")
    p.add_argument("--status", choices=["COMPLETE", "PARTIAL"])
    slfmt(p)
    p.set_defaults(func=_cmd_stress_list)
    p = sx.add_parser(
        "inspect",
        help="a stress analysis (sxa_) with provenance, trials and artifacts, or one trial (sxt_)",
    )
    p.add_argument("id")
    p.add_argument("--full", action="store_true", help="include every stored document")
    slfmt(p)
    p.set_defaults(func=_cmd_stress_inspect)
    for name, fn, text in (
        (
            "run",
            _cmd_stress_run,
            "run a stress design (single point, compound, factorial, repeated execution); exit 3 if evidence is incomplete",
        ),
        (
            "sweep",
            _cmd_stress_sweep,
            "run a stress design that sweeps a parameter or repeats over seeds",
        ),
    ):
        p = sx.add_parser(name, help=text)
        p.add_argument("spec")
        p.add_argument(
            "--investigation", help="investigation ID (default: a model-stress investigation)"
        )
        slfmt(p)
        p.set_defaults(func=fn)
    p = sx.add_parser(
        "compare",
        help="recompute an analysis from its stored trials and compare it with the stored documents; nothing is stored",
    )
    p.add_argument("id")
    slfmt(p)
    p.set_defaults(func=_cmd_stress_compare)
    p = sx.add_parser(
        "replay",
        help="replay the analysis and every trial as NEW runs and compare; exit 1 on any difference",
    )
    p.add_argument("id")
    p.add_argument("--no-trials", action="store_true", help="replay only the analysis run")
    slfmt(p)
    p.set_defaults(func=_cmd_stress_replay)

    cb = group(
        "calibration",
        "calibration and uncertainty of stored predictions (confidence is not automatically uncertainty; no universal score)",
    )
    p = cb.add_parser(
        "validate",
        help="parse a calibration spec; prints its identity (--preflight refuses unsupported baselines before anything runs)",
    )
    p.add_argument("spec")
    p.add_argument(
        "--preflight",
        action="store_true",
        help="also check the baseline run's stored predictions and the referenced analyses",
    )
    slfmt(p)
    p.set_defaults(func=_cmd_calibration_validate)
    p = cb.add_parser("list", help="list calibration analyses")
    p.add_argument("--baseline-run")
    p.add_argument("--status", choices=["COMPLETE", "PARTIAL"])
    slfmt(p)
    p.set_defaults(func=_cmd_calibration_list)
    p = cb.add_parser(
        "inspect",
        help="a calibration analysis (cba_) with provenance, evidence and artifacts, or one context result (cbr_)",
    )
    p.add_argument("id")
    p.add_argument(
        "--document",
        choices=list(calibration_engine.DOCUMENTS),
        help="include one stored document (evidence)",
    )
    p.add_argument("--full", action="store_true", help="include every stored document")
    slfmt(p)
    p.set_defaults(func=_cmd_calibration_inspect)
    p = cb.add_parser(
        "evaluate",
        help="run a calibration analysis as a new run; exit 3 if some evidence is insufficient, unavailable or invalid",
    )
    p.add_argument("spec")
    p.add_argument("--investigation", help="investigation ID (default: the baseline run's)")
    slfmt(p)
    p.set_defaults(func=_cmd_calibration_evaluate)
    p = cb.add_parser(
        "compare",
        help="recompute an analysis from the stored predictions and compare it with the stored documents; nothing is stored",
    )
    p.add_argument("id")
    slfmt(p)
    p.set_defaults(func=_cmd_calibration_compare)
    p = cb.add_parser(
        "replay",
        help="replay the analysis as a NEW run and compare every document; exit 1 on any difference",
    )
    p.add_argument("id")
    slfmt(p)
    p.set_defaults(func=_cmd_calibration_replay)

    rs = group(
        "resources",
        "latency, throughput and resource use of a real workload (environment-specific engineering measurements; no composite score)",
    )
    p = rs.add_parser(
        "validate",
        help="parse a resource spec; prints its identity (--preflight refuses unsupported requests before anything runs)",
    )
    p.add_argument("spec")
    p.add_argument(
        "--preflight",
        action="store_true",
        help="also load the model and dataset and check every requested capability",
    )
    slfmt(p)
    p.set_defaults(func=_cmd_resources_validate)
    p = rs.add_parser("list", help="list resource analyses")
    p.add_argument("--spec-id")
    p.add_argument("--status", choices=["COMPLETE", "PARTIAL"])
    slfmt(p)
    p.set_defaults(func=_cmd_resources_list)
    p = rs.add_parser(
        "inspect",
        help="a resource analysis (rsa_) with provenance, trials and artifacts, or one trial (rst_)",
    )
    p.add_argument("id")
    p.add_argument(
        "--document",
        choices=list(resource_engine.DOCUMENTS),
        help="include one stored document (evidence)",
    )
    p.add_argument("--full", action="store_true", help="include every stored document")
    slfmt(p)
    p.set_defaults(func=_cmd_resources_inspect)
    p = rs.add_parser(
        "run",
        help="measure a workload as a new run; exit 3 if any trial failed or timed out or evidence is insufficient",
    )
    p.add_argument("spec")
    p.add_argument("--investigation", help="investigation ID (default: the workspace's only one)")
    slfmt(p)
    p.set_defaults(func=_cmd_resources_run)
    p = rs.add_parser(
        "sweep",
        help="one separately persisted measurement per batch size, then a corrected comparison against the first",
    )
    p.add_argument("spec")
    p.add_argument("--batch-sizes", required=True, help="comma-separated, e.g. 1,8,32")
    p.add_argument("--metric", choices=list(resource_compare.METRICS), default="throughput")
    p.add_argument(
        "--correction", choices=["NONE", "BONFERRONI", "BENJAMINI_HOCHBERG"], default="BONFERRONI"
    )
    p.add_argument("--investigation")
    slfmt(p)
    p.set_defaults(func=_cmd_resources_sweep)
    p = rs.add_parser(
        "compare",
        help="compare analyses against the first (Phase 10 statistics, multiple-comparison corrected); nothing is stored",
    )
    p.add_argument("ids", nargs="+")
    p.add_argument("--metric", choices=list(resource_compare.METRICS), default="throughput")
    p.add_argument(
        "--correction", choices=["NONE", "BONFERRONI", "BENJAMINI_HOCHBERG"], default="BONFERRONI"
    )
    p.add_argument(
        "--allow-environment-mismatch",
        action="store_true",
        help="compare analyses from different environments (confounded, labelled)",
    )
    slfmt(p)
    p.set_defaults(func=_cmd_resources_compare)
    p = rs.add_parser(
        "replay",
        help="replay the measurement as a NEW run; compares the definition and the model outputs, never the timings; exit 1 on a difference",
    )
    p.add_argument("id")
    slfmt(p)
    p.set_defaults(func=_cmd_resources_replay)

    sc = group("scheduler", "durable, resumable, dependency-aware orchestration across engines")

    def scfmt(q: argparse.ArgumentParser) -> None:
        q.add_argument("--format", choices=["json", "text"], default="json")

    p = sc.add_parser(
        "validate", help="check a schedule spec's DAG (cycles, references); no registry touched"
    )
    p.add_argument("spec", help="schedule spec JSON file")
    scfmt(p)
    p.set_defaults(func=_cmd_scheduler_validate)
    p = sc.add_parser("expand", help="materialize a spec into explicit, inspectable units")
    p.add_argument("spec")
    scfmt(p)
    p.set_defaults(func=_cmd_scheduler_expand)
    p = sc.add_parser("run", help="execute a schedule's DAG (idempotent, resumable, concurrent)")
    p.add_argument("spec")
    p.add_argument("--investigation", help="investigation ID (default: the workspace's only one)")
    p.add_argument("--max-workers", type=int, help="override the spec's execution policy")
    p.add_argument(
        "--dry-run", action="store_true", help="validate and expand only; ZERO experiments run"
    )
    scfmt(p)
    p.set_defaults(func=_cmd_scheduler_run)
    p = sc.add_parser("status", help="unit statuses, attempts and run history of a schedule")
    p.add_argument("id", help="schedule ID (sch_...) or a spec_id (ssp_...) prefix")
    scfmt(p)
    p.set_defaults(func=_cmd_scheduler_status)
    p = sc.add_parser("inspect", help="a schedule's definition")
    p.add_argument("id")
    scfmt(p)
    p.set_defaults(func=_cmd_scheduler_inspect)
    p = sc.add_parser("graph", help="the dependency graph of a schedule")
    p.add_argument("id")
    scfmt(p)
    p.set_defaults(func=_cmd_scheduler_graph)
    p = sc.add_parser(
        "resume",
        help="continue a schedule from its persisted spec; already-SUCCEEDED units are not re-run",
    )
    p.add_argument("id")
    p.add_argument("--investigation")
    p.add_argument("--max-workers", type=int)
    scfmt(p)
    p.set_defaults(func=_cmd_scheduler_resume)
    p = sc.add_parser(
        "cancel", help="mark every open unit CANCELLED; completed evidence is never touched"
    )
    p.add_argument("id")
    scfmt(p)
    p.set_defaults(func=_cmd_scheduler_cancel)
    p = sc.add_parser("retry", help="explicitly request a new attempt at a FAILED/TIMED_OUT unit")
    p.add_argument("id")
    p.add_argument("--unit", required=True, help="unit key")
    scfmt(p)
    p.set_defaults(func=_cmd_scheduler_retry)
    p = sc.add_parser(
        "replay",
        help="re-run a schedule as a new attempt and diff unit outcomes; exit 1 on a "
        "non-environment-dependent difference",
    )
    p.add_argument("id")
    p.add_argument("--max-workers", type=int)
    scfmt(p)
    p.set_defaults(func=_cmd_scheduler_replay)

    gr = group("graph", "queryable evidence/failure knowledge graph over every existing entity")

    def gfmt(q: argparse.ArgumentParser) -> None:
        q.add_argument("--format", choices=["json", "text"], default="json")

    def gquery_bounds(q: argparse.ArgumentParser) -> None:
        q.add_argument("--max-depth", type=int, default=12)
        q.add_argument("--max-visited", type=int, default=5000)
        q.add_argument(
            "--kind", action="append", help="restrict traversal to this NodeKind (repeatable)"
        )
        q.add_argument(
            "--relation",
            action="append",
            help="restrict traversal to this RelationType (repeatable)",
        )

    p = gr.add_parser(
        "build", help="construct (or reuse) a graph snapshot over the current registry"
    )
    p.add_argument("spec", help="graph spec JSON file")
    p.add_argument(
        "--investigation",
        help="investigation ID (default: the workspace's only one, or the spec's)",
    )
    gfmt(p)
    p.set_defaults(func=_cmd_graph_build)
    p = gr.add_parser("list", help="list constructed graph definitions")
    gfmt(p)
    p.set_defaults(func=_cmd_graph_list)
    p = gr.add_parser("inspect", help="a graph definition (grh_) or a snapshot (gsn_)")
    p.add_argument("id")
    gfmt(p)
    p.set_defaults(func=_cmd_graph_inspect)
    p = gr.add_parser("neighbors", help="direct incoming/outgoing edges of one node")
    p.add_argument("snapshot", help="snapshot ID (gsn_...) or a spec_id prefix")
    p.add_argument("node", help="a graph node ID (gnd_...) or a referenced entity ID")
    p.add_argument("--direction", choices=["IN", "OUT", "BOTH"], default="BOTH")
    gfmt(p)
    p.set_defaults(func=_cmd_graph_neighbors)
    p = gr.add_parser("path", help="a path between two nodes, if one exists within the bounds")
    p.add_argument("snapshot")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--direction", choices=["IN", "OUT", "BOTH"], default="OUT")
    gquery_bounds(p)
    gfmt(p)
    p.set_defaults(func=_cmd_graph_path)
    p = gr.add_parser(
        "query",
        help=(
            "a named composed query: evidence-for-claim, runs-for-failure-mode, "
            "analyses-for-run, artifacts-for-analysis, model-to-failure, dataset-to-failure"
        ),
    )
    p.add_argument("snapshot")
    p.add_argument("name")
    p.add_argument("node", help="the seed node (ID or referenced entity ID)")
    p.add_argument("--node2", help="second node ID, for model-to-failure / dataset-to-failure")
    gquery_bounds(p)
    gfmt(p)
    p.set_defaults(func=_cmd_graph_query)
    p = gr.add_parser("diff", help="raw added/removed nodes and edges between two snapshots")
    p.add_argument("a")
    p.add_argument("b")
    gfmt(p)
    p.set_defaults(func=_cmd_graph_diff)
    p = gr.add_parser(
        "replay",
        help="reconstruct a snapshot as a NEW run and compare; exit 1 unless verified deterministic",
    )
    p.add_argument("id")
    p.set_defaults(func=_cmd_graph_replay)

    rp = group("reproduce", "reproducibility framework: what can and cannot be reproduced, and how")

    def rp_fmt(q: argparse.ArgumentParser) -> None:
        q.add_argument("--format", choices=["json", "text"], default="json")

    def rtol(q: argparse.ArgumentParser) -> None:
        q.add_argument("--relative-tolerance", type=float, default=DEFAULT_RELATIVE_TOLERANCE)
        q.add_argument("--absolute-tolerance", type=float, default=DEFAULT_ABSOLUTE_TOLERANCE)

    p = rp.add_parser(
        "validate",
        help="confirm a target exists and report its home investigation/run, without reproducing anything",
    )
    p.add_argument("target_kind", choices=[k.value for k in TargetKind])
    p.add_argument("target_id")
    p.set_defaults(func=_cmd_reproduce_validate)
    p = rp.add_parser(
        "run",
        help="attempt to reproduce a target against the requested mode; exit 1 unless EQUAL/APPROXIMATELY_EQUAL",
    )
    p.add_argument("target_kind", choices=[k.value for k in TargetKind])
    p.add_argument("target_id")
    p.add_argument(
        "--mode",
        choices=[m.value for m in ReproductionMode],
        default=ReproductionMode.DETERMINISTIC.value,
    )
    p.add_argument("--investigation", help="required for a SCHEDULE target")
    rtol(p)
    rp_fmt(p)
    p.set_defaults(func=_cmd_reproduce_run)
    p = rp.add_parser("inspect", help="a persisted reproduction attempt (rpa_...)")
    p.add_argument("id")
    rp_fmt(p)
    p.set_defaults(func=_cmd_reproduce_inspect)
    p = rp.add_parser(
        "compare",
        help="ad hoc tolerance-aware comparison of two stored JSON artifacts, possibly from different runs",
    )
    p.add_argument("run_a")
    p.add_argument("path_a", help="artifact path within run_a, e.g. benchmark/results.json")
    p.add_argument("run_b")
    p.add_argument("path_b")
    rtol(p)
    p.set_defaults(func=_cmd_reproduce_compare)
    p = rp.add_parser("verify", help="re-hash a run's artifacts against their recorded digests")
    p.add_argument("run_id")
    p.set_defaults(func=_cmd_reproduce_verify)
    p = rp.add_parser(
        "replay", help="re-attempt a prior reproduction attempt's exact spec as a NEW attempt"
    )
    p.add_argument("id")
    rp_fmt(p)
    p.set_defaults(func=_cmd_reproduce_replay)
    p = rp.add_parser("diff", help="structural diff between two persisted reproduction attempts")
    p.add_argument("a")
    p.add_argument("b")
    rp_fmt(p)
    p.set_defaults(func=_cmd_reproduce_diff)

    rpt = group(
        "report", "research reporting: deterministic reports assembled from persisted evidence"
    )

    def rpt_fmt(q: argparse.ArgumentParser) -> None:
        q.add_argument("--format", choices=["json", "text"], default="json")

    p = rpt.add_parser(
        "generate", help="generate a report from an investigation's persisted evidence"
    )
    p.add_argument("report_type", choices=[t.value for t in ReportType])
    p.add_argument("--investigation", required=True)
    p.add_argument(
        "--template-id",
        help="an existing report template ID; defaults to the built-in template for this report type",
    )
    p.add_argument(
        "--run",
        action="append",
        help="narrow evidence to this run (repeatable); default is every run in scope",
    )
    p.add_argument(
        "--statistical-analysis",
        action="append",
        help="include this statistical analysis ID (repeatable)",
    )
    rpt_fmt(p)
    p.set_defaults(func=_cmd_report_generate)
    p = rpt.add_parser("list", help="list generated reports")
    p.add_argument("--investigation")
    p.set_defaults(func=_cmd_report_list)
    p = rpt.add_parser("show", help="show a report (rpt_...)")
    p.add_argument("id")
    rpt_fmt(p)
    p.set_defaults(func=_cmd_report_show)
    p = rpt.add_parser("sections", help="a report's sections")
    p.add_argument("id")
    p.set_defaults(func=_cmd_report_sections)
    p = rpt.add_parser("claims", help="a report's claim/evidence matrix (its findings)")
    p.add_argument("id")
    p.set_defaults(func=_cmd_report_claims)
    p = rpt.add_parser(
        "validate",
        help="check every claim has evidence and every reference resolves; exit 1 on any ERROR",
    )
    p.add_argument("id")
    p.set_defaults(func=_cmd_report_validate)
    p = rpt.add_parser("export", help="render a report to Markdown or HTML")
    p.add_argument("id")
    p.add_argument("--export-format", choices=["markdown", "html"], default="markdown")
    p.add_argument("--output", help="write to this path instead of stdout")
    p.set_defaults(func=_cmd_report_export)

    dsr = group(
        "dossier",
        "evidence dossiers: structured, persisted research packages built on top of reports",
    )

    def dsr_fmt(q: argparse.ArgumentParser) -> None:
        q.add_argument("--format", choices=["json", "text"], default="json")

    p = dsr.add_parser(
        "build", help="build a dossier from an investigation's persisted evidence and reports"
    )
    p.add_argument("--investigation", required=True)
    p.add_argument("--question", required=True, help="the research question this dossier addresses")
    p.add_argument(
        "--source-kind", choices=[k.value for k in DossierSourceKind], default="INVESTIGATION"
    )
    p.add_argument("--source-id", help="defaults to --investigation")
    p.add_argument(
        "--evidence",
        action="append",
        help="explicit 'source_kind:source_id' evidence to include (repeatable)",
    )
    dsr_fmt(p)
    p.set_defaults(func=_cmd_dossier_build)
    p = dsr.add_parser("list", help="list built dossiers")
    p.add_argument("--investigation")
    p.set_defaults(func=_cmd_dossier_list)
    p = dsr.add_parser("show", help="show a dossier (dsr_...)")
    p.add_argument("id")
    dsr_fmt(p)
    p.set_defaults(func=_cmd_dossier_show)
    p = dsr.add_parser("items", help="a dossier's evidence items and their lineage")
    p.add_argument("id")
    p.set_defaults(func=_cmd_dossier_items)
    p = dsr.add_parser("findings", help="a dossier's sufficiency-analysis findings")
    p.add_argument("id")
    p.set_defaults(func=_cmd_dossier_findings)
    p = dsr.add_parser(
        "snapshot", help="freeze a dossier's current evidence into an immutable snapshot"
    )
    p.add_argument("id")
    dsr_fmt(p)
    p.set_defaults(func=_cmd_dossier_snapshot)
    p = dsr.add_parser(
        "validate",
        help="list unresolved sufficiency findings; exit 1 if any evidence is unavailable or stale",
    )
    p.add_argument("id")
    p.set_defaults(func=_cmd_dossier_validate)
    p = dsr.add_parser("export", help="render a dossier to Markdown")
    p.add_argument("id")
    p.add_argument("--output", help="write to this path instead of stdout")
    p.set_defaults(func=_cmd_dossier_export)

    datasets = group("dataset", "inspect or register a dataset")
    p = datasets.add_parser("inspect", help="inspect a registered dataset ID or load a source")
    p.add_argument("target", help="dataset ID (dst_...) or a source (path or builtin:name)")
    loader_args(p, model=False)
    p.add_argument("--deep", action="store_true", help="also compute expensive statistics")
    p.set_defaults(func=_cmd_dataset_inspect)
    p = datasets.add_parser("register", help="load a dataset and register it")
    p.add_argument("--name", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--version", default="1")
    p.add_argument("--option", action="append", default=[])
    p.set_defaults(func=_cmd_dataset_register)

    viz = group("viz", "serve the read-only visualization/analysis UI over a workspace")
    p = viz.add_parser(
        "serve", help="start the FastAPI+static UI (needs: pip install experionyx[viz])"
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8420)
    p.set_defaults(func=_cmd_viz_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        return int(args.func(args))
    except ExperionyxError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

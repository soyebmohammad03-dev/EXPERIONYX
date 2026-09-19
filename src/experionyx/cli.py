"""Command line interface. Every command reads or writes the real registry of a workspace."""

import argparse
import dataclasses
import json
import logging
import platform
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from experionyx import __version__
from experionyx.adapters.capabilities import DeviceKind, TaskType
from experionyx.adapters.records import RegisteredDataset, RegisteredModel
from experionyx.adapters.registry import default_registries
from experionyx.artifacts import LocalArtifactStore
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
from experionyx.errors import ArtifactIntegrityError, ExperionyxError
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
from experionyx.faults.degradation import measure_degradation, measure_latency
from experionyx.faults.demos import FAULT_DEMOS, run_fault_demo
from experionyx.faults.design import FaultDesign, FaultLimits, SweepSpec
from experionyx.faults.entities import FaultExperiment, FaultTrial
from experionyx.faults.lab import FaultExperimentResult, run_fault_experiment
from experionyx.faults.library import default_fault_registry
from experionyx.faults.report import analysis_run_id, load_analysis, read_artifact, summary_rows
from experionyx.faults.spec import FaultRegistry, FaultScope, FaultSpec, ScopeKind
from experionyx.provenance import Provenance, RunOutcome
from experionyx.sqlite import DB_SCHEMA_VERSION, SqliteRegistry

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
    _dump({"device": _json(adapter.device), "load_seconds": adapter.load_seconds,
           "metadata": _json(adapter.metadata())})  # fmt: skip
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
    _dump({"name": ft.name, "version": ft.version, "category": ft.category.value, "target": ft.target.value,
           "description": ft.description, "requires": sorted(r.value for r in ft.requires),
           "stochastic": ft.stochastic, "implemented": ft.implemented, "sweepable": list(ft.sweepable()),
           "parameters": params})  # fmt: skip
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
    _dump({
        "baseline_run": args.baseline_run, "treatment_run": args.treatment_run,
        "fault": {k: fault[k] for k in ("fault_id", "fault", "seed", "scope", "affected_samples", "total_samples", "timings_seconds")} if isinstance(fault, dict) else None,
        "degradation": [_json(d) for d in measure_degradation(base, treat)],
        "latency": _json(measure_latency(base, treat)),
        "note": "deterioration > 0 means worse (direction-aware); this is a measurement, not a verdict",
    })  # fmt: skip
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

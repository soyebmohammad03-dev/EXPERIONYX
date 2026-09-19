"""Command line interface. Every command reads or writes the real registry of a workspace."""

import argparse
import json
import logging
import platform
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from experionyx import __version__
from experionyx.artifacts import LocalArtifactStore
from experionyx.domain import (
    Artifact,
    ConfigurationRef,
    Entity,
    Experiment,
    Investigation,
    Observation,
    Run,
    RunStatus,
)
from experionyx.errors import ArtifactIntegrityError, ExperionyxError
from experionyx.execution import ExecutionResult, Executor, resolve_procedure, run_states
from experionyx.provenance import Provenance, RunOutcome
from experionyx.sqlite import DB_SCHEMA_VERSION, SqliteRegistry

DEFAULT_WORKSPACE = ".experionyx"
REGISTRY_FILE = "registry.sqlite"
EXPERIMENTS_DIR = "experiments"


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
    return Executor(registry, store, source_root=Path.cwd())


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

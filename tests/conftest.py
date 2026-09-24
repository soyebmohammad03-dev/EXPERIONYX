import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from experionyx.artifacts import LocalArtifactStore
from experionyx.domain import ConfigurationRef, Experiment, ExperimentStatus, Investigation
from experionyx.execution import Executor
from experionyx.sqlite import SqliteRegistry
from factories import T0, configuration, experiment, investigation

if TYPE_CHECKING:
    from api_world import ApiWorld


def git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com",
         "-c", "commit.gpgsign=false", *args],
        cwd=cwd, check=True, capture_output=True,
    )  # fmt: skip


def make_repo(path: Path, *, dirty: bool = False, untracked: bool = False) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q", "-b", "main")
    (path / "tracked.txt").write_text("v1\n", encoding="utf-8")
    git(path, "add", ".")
    git(path, "commit", "-q", "-m", "initial")
    if dirty:
        (path / "tracked.txt").write_text("v2\n", encoding="utf-8")
    if untracked:
        (path / "new.txt").write_text("new\n", encoding="utf-8")
    return path


@dataclass
class Lab:
    registry: SqliteRegistry
    store: LocalArtifactStore
    executor: Executor
    investigation: Investigation
    configuration: ConfigurationRef
    experiment: Experiment
    workspace: Path


def make_lab(workspace: Path, source_root: Path) -> Lab:
    workspace.mkdir(parents=True, exist_ok=True)
    registry = SqliteRegistry(workspace / "registry.sqlite")
    store = LocalArtifactStore(workspace / "experiments")
    inv, cfg = investigation(), configuration()
    exp = experiment(inv, cfg)
    for e in (inv, cfg, exp):
        registry.add(e)
    registry.update_status(exp.with_status(ExperimentStatus.READY))
    ready = registry.get(Experiment, exp.id)
    return Lab(
        registry,
        store,
        Executor(registry, store, source_root=source_root),
        inv,
        cfg,
        ready,
        workspace,
    )


@pytest.fixture
def lab(tmp_path: Path) -> Iterator[Lab]:
    """A real workspace (SQLite file + artifact directory); source root is not a git repo."""
    source = tmp_path / "src-not-a-repo"
    source.mkdir()
    lab = make_lab(tmp_path / "ws", source)
    yield lab
    lab.registry.close()


@pytest.fixture(scope="session")
def api_world(tmp_path_factory: pytest.TempPathFactory) -> "ApiWorld":
    """One real, richly populated workspace shared by tests/test_api_*.py (see api_world.py):
    expensive to build (a real sklearn baseline, fault sweeps, discovery, drift, stats,
    reliability, graph, report, dossier, reproducibility), so it is built once per session and
    every test in test_api_*.py only reads from it -- never mutates the shared registry."""
    pytest.importorskip("sklearn")
    from api_world import build_api_world

    return build_api_world(tmp_path_factory.mktemp("api"))


__all__ = ["T0", "ApiWorld", "Lab", "api_world", "git", "lab", "make_lab", "make_repo"]

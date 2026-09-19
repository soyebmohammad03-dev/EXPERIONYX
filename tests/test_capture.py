import hashlib
import importlib.metadata
import json
import platform
import random
import sys
import types
from pathlib import Path

import pytest

from conftest import make_repo
from experionyx.capture import (
    capture_environment,
    capture_source,
    installed_packages,
    seed_everything,
)
from experionyx.provenance import SourceState


def test_environment_is_captured_from_the_real_runtime() -> None:
    cap = capture_environment()
    env = cap.snapshot
    assert env.python_version == platform.python_version()
    assert env.os == platform.platform()
    assert env.machine == platform.machine()
    assert env.source_revision is None  # source lives in provenance, not environment identity
    assert "pytest" in env.packages
    assert "experionyx" in env.packages
    assert cap.runtime["python_implementation"] == platform.python_implementation()


def test_environment_identity_is_deterministic_and_digest_matches_packages() -> None:
    a, b = capture_environment(), capture_environment()
    assert a.snapshot.id == b.snapshot.id
    assert a.dependency_digest == b.dependency_digest
    canonical = json.dumps(dict(a.snapshot.packages), sort_keys=True, separators=(",", ":"))
    assert a.dependency_digest == "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def test_packages_are_sorted_normalized_records() -> None:
    packages, _ = installed_packages()
    assert list(packages) == sorted(packages)
    assert all(n == n.lower() and "_" not in n and "." not in n for n in packages)


def test_no_secrets_or_private_paths_are_captured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("EXPERIONYX_TEST_SECRET", "s3cr3t-token-value")
    monkeypatch.chdir(tmp_path)
    blob = json.dumps(
        {
            "env": capture_environment().snapshot.to_dict(),
            "rt": dict(capture_environment().runtime),
        },
        default=str,
    )
    assert "s3cr3t-token-value" not in blob
    assert str(Path.home()) not in blob
    assert str(tmp_path) not in blob
    assert platform.node() not in blob or platform.node() in ("", "localhost")


class _Meta(dict[str, str | None]):
    def __missing__(self, key: str) -> None:  # like email.message.Message: absent header -> None
        return None


class _Dist:
    def __init__(self, name: str | None, version: str | None) -> None:
        self.metadata = _Meta({"Name": name})
        self.version = version


def test_unreliable_distributions_are_reported_not_invented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dists = [
        _Dist("Good_Pkg", "1.2.3"),
        _Dist(None, "1.0"),
        _Dist("bad", "not a version"),
        _Dist("nover", None),
        _Dist("dup", "2.0"),
        _Dist("dup", "1.0"),
    ]
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: dists)
    packages, issues = installed_packages()
    assert packages == {"dup": "1.0", "good-pkg": "1.2.3"}
    assert issues == (
        "ambiguous-version:dup",
        "invalid-version:bad",
        "invalid-version:nover",
        "unnamed-distribution",
    )


# --- git --------------------------------------------------------------------------------------


def test_clean_repository_is_reproducible_source(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "r")
    src = capture_source(repo)
    assert src.state is SourceState.REPRODUCIBLE_SOURCE
    assert src.branch == "main"
    assert src.commit is not None
    assert len(src.commit) == 40


def test_dirty_repository_is_never_reported_clean(tmp_path: Path) -> None:
    clean = capture_source(make_repo(tmp_path / "a"))
    dirty = capture_source(make_repo(tmp_path / "b", dirty=True))
    assert dirty.state is SourceState.MODIFIED_WORKTREE
    assert dirty.commit is not None
    assert clean.state is SourceState.REPRODUCIBLE_SOURCE


def test_untracked_files_make_the_worktree_modified(tmp_path: Path) -> None:
    assert (
        capture_source(make_repo(tmp_path / "u", untracked=True)).state
        is SourceState.MODIFIED_WORKTREE
    )


def test_detached_head_has_no_branch(tmp_path: Path) -> None:
    from conftest import git

    repo = make_repo(tmp_path / "d")
    git(repo, "checkout", "-q", "--detach")
    src = capture_source(repo)
    assert src.branch is None
    assert src.state is SourceState.REPRODUCIBLE_SOURCE


def test_non_repository_and_missing_directory_are_unknown(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    assert capture_source(plain).state is SourceState.UNKNOWN
    assert capture_source(tmp_path / "does-not-exist").state is SourceState.UNKNOWN


def test_repository_without_commits_is_unknown(tmp_path: Path) -> None:
    from conftest import git

    (tmp_path / "empty").mkdir()
    git(tmp_path / "empty", "init", "-q")
    assert capture_source(tmp_path / "empty").state is SourceState.UNKNOWN


def test_missing_git_binary_is_unknown(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    assert capture_source(tmp_path).state is SourceState.UNKNOWN


# --- seeding ----------------------------------------------------------------------------------


def test_seed_makes_python_random_repeatable() -> None:
    seed_everything(7)
    first = [random.random() for _ in range(3)]
    seed_everything(7)
    assert [random.random() for _ in range(3)] == first
    seed_everything(8)
    assert [random.random() for _ in range(3)] != first


def test_numpy_is_seeded_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    fake = types.ModuleType("numpy")
    fake.random = types.SimpleNamespace(seed=calls.append)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "numpy", fake)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    assert seed_everything(5) == ("python.random", "numpy.random")
    assert calls == [5]


def test_only_python_random_is_reported_without_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    assert seed_everything(1) == ("python.random",)

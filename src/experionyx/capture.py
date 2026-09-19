"""Capture of the real execution environment, source revision and seeding.

Independent of the registry and executor. Privacy: nothing here reads environment variables,
usernames, hostnames, home directories or absolute paths (see docs/provenance.md).
"""

import importlib.metadata
import importlib.util
import os
import platform
import random
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from experionyx.domain import EnvironmentSnapshot
from experionyx.errors import ValidationError
from experionyx.hashing import content_hash
from experionyx.provenance import SourceRevision, SourceState
from experionyx.validation import version as validate_version

GIT_TIMEOUT_SECONDS = 10.0
_UNKNOWN = "unknown"


@dataclass(frozen=True)
class EnvironmentCapture:
    """A captured environment plus what could not be determined about it."""

    snapshot: EnvironmentSnapshot  # reproducibility-critical, content-addressed
    dependency_digest: str  # sha256 over the canonical package mapping
    dependency_issues: tuple[str, ...]  # distributions skipped or ambiguous, sorted
    runtime: Mapping[str, object]  # informational only; never part of identity or fingerprint


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()  # PEP 503


def installed_packages() -> tuple[dict[str, str], tuple[str, ...]]:
    """Installed distributions as {normalized name: version}, plus explicit issues.

    Distributions without a name or a valid version are skipped and reported; a name installed
    at several different versions is reported as ambiguous and the lowest version is kept.
    """
    found: dict[str, set[str]] = {}
    issues: list[str] = []
    for dist in importlib.metadata.distributions():
        raw_name = dist.metadata["Name"]
        if not raw_name:
            issues.append("unnamed-distribution")
            continue
        name = _normalize(raw_name)
        try:
            ver = validate_version(f"packages.{name}", dist.version)
        except ValidationError:
            issues.append(f"invalid-version:{name}")
            continue
        found.setdefault(name, set()).add(ver)
    packages: dict[str, str] = {}
    for name, versions in found.items():
        packages[name] = min(versions)
        if len(versions) > 1:
            issues.append(f"ambiguous-version:{name}")
    return dict(sorted(packages.items())), tuple(sorted(issues))


def capture_environment(seeded: tuple[str, ...] = ()) -> EnvironmentCapture:
    """Snapshot the running interpreter and installed packages. Nothing is hardcoded."""
    packages, issues = installed_packages()
    snapshot = EnvironmentSnapshot(
        python_version=platform.python_version(),
        os=platform.platform(),
        machine=platform.machine() or _UNKNOWN,
        packages=packages,
        source_revision=None,  # source is recorded in provenance, not in the environment identity
    )
    runtime: dict[str, object] = {
        "python_implementation": platform.python_implementation(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "pid": os.getpid(),
        "seeded_libraries": list(seeded),
    }
    return EnvironmentCapture(snapshot, content_hash(packages), issues, runtime)


def _git(root: Path, *args: str) -> str | None:
    """stdout of a read-only git command, or None on any failure (no git, not a repo, ...)."""
    try:
        proc = subprocess.run(  # noqa: S603  # fixed argv, no shell, read-only git subcommands
            ["git", "-C", str(root), *args],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def capture_source(root: Path) -> SourceRevision:
    """Git commit/branch/cleanliness of `root`. Any doubt yields UNKNOWN, never a clean claim."""
    commit = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain", "--untracked-files=normal")
    if commit is None or status is None:
        return SourceRevision(SourceState.UNKNOWN)
    branch = _git(root, "symbolic-ref", "--short", "-q", "HEAD")  # fails on detached HEAD
    state = SourceState.MODIFIED_WORKTREE if status.strip() else SourceState.REPRODUCIBLE_SOURCE
    return SourceRevision(state, commit.strip(), branch.strip() if branch else None)


def seed_everything(seed: int) -> tuple[str, ...]:
    """Seed the global RNGs of Python and, if installed, NumPy. Returns what was seeded.

    This records intent and seeds what it can; it does NOT make an execution deterministic
    (threads, hash randomization, hardware, and other frameworks are outside its reach).
    """
    seeded = ["python.random"]
    random.seed(seed)
    if importlib.util.find_spec("numpy") is not None:
        numpy = sys.modules.get("numpy") or importlib.import_module("numpy")
        numpy.random.seed(seed)
        seeded.append("numpy.random")
    return tuple(seeded)

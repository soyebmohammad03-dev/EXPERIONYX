"""Artifact storage: a small protocol plus a local-filesystem implementation.

Files are written by experiments directly into their run's artifact directory (no copying) and
registered in place: the digest is computed from the file's bytes, streamed.
"""

import hashlib
import json
import mimetypes
import os
import tempfile
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Protocol

from experionyx.domain import Artifact, ArtifactCategory, Run
from experionyx.errors import ArtifactError, ArtifactIntegrityError, ValidationError
from experionyx.hashing import HASH_PREFIX
from experionyx.validation import relative_path

CHUNK_BYTES = 1024 * 1024
DEFAULT_MEDIA_TYPE = "application/octet-stream"


def sha256_file(path: Path) -> tuple[str, int]:
    """(`sha256:<hex>`, size in bytes) of a file, reading it in bounded-size chunks."""
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        while chunk := f.read(CHUNK_BYTES):
            h.update(chunk)
            size += len(chunk)
    return HASH_PREFIX + h.hexdigest(), size


class ArtifactStore(Protocol):
    """Where a run's files live. Implementations may be replaced (e.g. object storage)."""

    def prepare(self, run: Run) -> Path:
        """Create the run's storage and return the directory experiments write artifacts into."""
        ...

    def register(
        self,
        run: Run,
        path: str,
        *,
        name: str,
        category: ArtifactCategory,
        media_type: str | None,
        created_at: datetime,
    ) -> Artifact:
        """Hash the file at `path` (relative to the artifact directory) and describe it."""
        ...

    def verify(self, run: Run, artifact: Artifact) -> None:
        """Raise ArtifactIntegrityError if the stored bytes no longer match the artifact."""
        ...

    def write_metadata(self, run: Run, filename: str, payload: Mapping[str, object]) -> None:
        """Atomically write a JSON metadata document for the run."""
        ...


class LocalArtifactStore:
    """`<root>/<experiment-id>/runs/<run-id>/{metadata,artifacts}/`."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def run_dir(self, run: Run) -> Path:
        return self._root / run.experiment_id / "runs" / run.id

    def prepare(self, run: Run) -> Path:
        base = self.run_dir(run)
        (base / "metadata").mkdir(parents=True, exist_ok=True)
        artifacts = base / "artifacts"
        artifacts.mkdir(exist_ok=True)
        return artifacts

    def _resolve(self, directory: Path, relative: str) -> Path:
        """Resolve inside `directory`; refuses escapes (`..`, absolute paths, symlinks out)."""
        try:
            relative_path("path", relative)
        except ValidationError as exc:
            raise ArtifactError(str(exc)) from exc
        base = directory.resolve()
        target = (base / relative).resolve()
        if not target.is_relative_to(base):
            raise ArtifactError(f"artifact path escapes the artifact directory: {relative!r}")
        return target

    def register(
        self,
        run: Run,
        path: str,
        *,
        name: str,
        category: ArtifactCategory,
        media_type: str | None,
        created_at: datetime,
    ) -> Artifact:
        target = self._resolve(self.run_dir(run) / "artifacts", path)
        if not target.is_file():
            raise ArtifactError(f"artifact file does not exist: {path!r}")
        try:
            digest, size = sha256_file(target)
        except OSError as exc:
            raise ArtifactError(f"could not hash artifact {path!r}: {exc}") from exc
        return Artifact(
            run.id,
            name,
            path,
            digest,
            size,
            media_type or mimetypes.guess_type(path)[0] or DEFAULT_MEDIA_TYPE,
            created_at,
            category,
        )

    def verify(self, run: Run, artifact: Artifact) -> None:
        if artifact.run_id != run.id:
            raise ArtifactIntegrityError(f"artifact {artifact.id} does not belong to run {run.id}")
        try:
            target = self._resolve(self.run_dir(run) / "artifacts", artifact.path)
            digest, size = sha256_file(target)
        except (ArtifactError, OSError) as exc:
            raise ArtifactIntegrityError(f"{artifact.path!r} is missing or unreadable") from exc
        if (digest, size) != (artifact.digest, artifact.size_bytes):
            raise ArtifactIntegrityError(
                f"{artifact.path!r} changed since registration: registered "
                f"{artifact.digest} ({artifact.size_bytes} B), found {digest} ({size} B)"
            )

    def write_metadata(self, run: Run, filename: str, payload: Mapping[str, object]) -> None:
        directory = self.run_dir(run) / "metadata"
        directory.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, sort_keys=True, indent=2)
                f.write("\n")
            os.replace(tmp, directory / filename)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

import hashlib
import json
from pathlib import Path

import pytest

from experionyx.artifacts import CHUNK_BYTES, LocalArtifactStore, sha256_file
from experionyx.domain import ArtifactCategory, Run
from experionyx.errors import ArtifactError, ArtifactIntegrityError
from factories import T0, configuration, environment, experiment, investigation, run


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path / "experiments")


@pytest.fixture
def a_run() -> Run:
    return run(experiment(investigation(), configuration()), environment())


def _register(store: LocalArtifactStore, r: Run, path: str = "out.txt", **kw: object):  # type: ignore[no-untyped-def]
    return store.register(
        r, path, name="out", category=ArtifactCategory.OUTPUT, media_type=None, created_at=T0, **kw
    )


def test_digest_is_of_the_bytes_and_streamed(tmp_path: Path) -> None:
    data = b"abc" * CHUNK_BYTES + b"tail"  # spans several chunks
    f = tmp_path / "big.bin"
    f.write_bytes(data)
    digest, size = sha256_file(f)
    assert digest == "sha256:" + hashlib.sha256(data).hexdigest()
    assert size == len(data)


def test_digest_ignores_filename(tmp_path: Path) -> None:
    (tmp_path / "a").write_bytes(b"same")
    (tmp_path / "b").write_bytes(b"same")
    assert sha256_file(tmp_path / "a") == sha256_file(tmp_path / "b")


def test_register_records_real_digest_size_and_media_type(
    store: LocalArtifactStore, a_run: Run
) -> None:
    directory = store.prepare(a_run)
    (directory / "out.txt").write_bytes(b"hello")
    art = _register(store, a_run)
    assert art.digest == "sha256:" + hashlib.sha256(b"hello").hexdigest()
    assert art.size_bytes == 5
    assert art.media_type == "text/plain"
    assert art.run_id == a_run.id
    store.verify(a_run, art)


def test_layout_is_deterministic_and_under_root(store: LocalArtifactStore, a_run: Run) -> None:
    directory = store.prepare(a_run)
    expected = store.run_dir(a_run)
    assert directory == expected / "artifacts"
    assert (expected / "metadata").is_dir()
    assert expected == store._root / a_run.experiment_id / "runs" / a_run.id


def test_modified_missing_and_truncated_files_are_detected(
    store: LocalArtifactStore, a_run: Run
) -> None:
    directory = store.prepare(a_run)
    target = directory / "out.txt"
    target.write_bytes(b"original")
    art = _register(store, a_run)
    target.write_bytes(b"tampered")  # same size, different bytes
    with pytest.raises(ArtifactIntegrityError, match="changed since registration"):
        store.verify(a_run, art)
    target.write_bytes(b"orig")  # different size
    with pytest.raises(ArtifactIntegrityError):
        store.verify(a_run, art)
    target.unlink()
    with pytest.raises(ArtifactIntegrityError, match="missing"):
        store.verify(a_run, art)
    target.write_bytes(b"original")  # restored bytes verify again
    store.verify(a_run, art)


def test_register_missing_file_or_directory(store: LocalArtifactStore, a_run: Run) -> None:
    directory = store.prepare(a_run)
    (directory / "sub").mkdir()
    with pytest.raises(ArtifactError, match="does not exist"):
        _register(store, a_run, "nope.txt")
    with pytest.raises(ArtifactError, match="does not exist"):
        _register(store, a_run, "sub")


@pytest.mark.parametrize("bad", ["../escape.txt", "/etc/passwd", "a/../../x", "./x", "a\\b", ""])
def test_paths_cannot_escape_the_artifact_directory(
    store: LocalArtifactStore, a_run: Run, bad: str
) -> None:
    store.prepare(a_run)
    with pytest.raises(ArtifactError):
        _register(store, a_run, bad)


def test_symlink_out_of_the_artifact_directory_is_refused(
    store: LocalArtifactStore, a_run: Run, tmp_path: Path
) -> None:
    directory = store.prepare(a_run)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    (directory / "link.txt").symlink_to(outside)
    with pytest.raises(ArtifactError, match="escapes"):
        _register(store, a_run, "link.txt")


def test_verify_rejects_another_runs_artifact(store: LocalArtifactStore, a_run: Run) -> None:
    directory = store.prepare(a_run)
    (directory / "out.txt").write_text("x")
    art = _register(store, a_run)
    other = Run(a_run.experiment_id, a_run.environment_id, 99, T0)
    with pytest.raises(ArtifactIntegrityError):
        store.verify(other, art)


def test_metadata_is_written_atomically_without_leftovers(
    store: LocalArtifactStore, a_run: Run
) -> None:
    store.prepare(a_run)
    store.write_metadata(a_run, "m.json", {"b": 1, "a": [1, 2]})
    meta = store.run_dir(a_run) / "metadata"
    assert json.loads((meta / "m.json").read_text()) == {"a": [1, 2], "b": 1}
    with pytest.raises(TypeError):
        store.write_metadata(a_run, "m.json", {"bad": object()})
    assert json.loads((meta / "m.json").read_text()) == {"a": [1, 2], "b": 1}  # untouched
    assert [p.name for p in meta.iterdir()] == ["m.json"]  # no temp files

"""Schema migration from real older-version databases (built with the historical layouts)."""

import json
import sqlite3
from pathlib import Path

import pytest

from experionyx.domain import Artifact, Entity, Experiment, Observation, Run, RunStatus
from experionyx.errors import SchemaVersionError
from experionyx.hashing import canonical_json, content_hash
from experionyx.provenance import (
    ErrorInfo,
    ExecutionParameters,
    FailureStage,
    Provenance,
    RunOutcome,
    SourceRevision,
    SourceState,
)
from experionyx.sqlite import _SPECS, DB_SCHEMA_VERSION, SqliteRegistry, _column_value, _ddl
from factories import (
    DIGEST,
    T0,
    artifact,
    claim,
    configuration,
    environment,
    evidence,
    experiment,
    investigation,
    observation,
    run,
)


def insert(
    conn: sqlite3.Connection, entity: Entity, payload: dict[str, object] | None = None
) -> None:
    """Insert `entity` the way an older release would have stored it."""
    spec = _SPECS[type(entity)]
    data = payload if payload is not None else entity.to_dict()
    cols = ["id", *spec.columns, "payload", "content_hash"]
    values = [
        entity.id,
        *(_column_value(entity, c) for c in spec.columns),
        canonical_json(data),
        content_hash(data),
    ]
    conn.execute(
        f"INSERT INTO {spec.table} ({', '.join(cols)}) "  # noqa: S608
        f"VALUES ({', '.join('?' * len(cols))})",
        values,
    )


def old_artifact_payload(a: Artifact) -> dict[str, object]:
    """Phase 1 artifacts had no logical name or category."""
    data = a.to_dict()
    del data["name"], data["category"]
    return data


def make_database(path: Path, version: int) -> dict[str, Entity]:
    inv, cfg, env = investigation(), configuration(), environment()
    exp = experiment(inv, cfg)
    r = run(exp, env)
    obs, art, clm = observation(r), artifact(r), claim(inv)
    ent = evidence(clm, obs)
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("PRAGMA foreign_keys = ON")
    for statement in _ddl(upto=version):
        conn.execute(statement)
    stored_run = r.with_status(RunStatus.RUNNING) if version >= 2 else r  # Phase 2 had provenance
    for e in (inv, cfg, env, exp, stored_run, obs, clm, ent):
        insert(conn, e)
    insert(conn, art, old_artifact_payload(art) if version == 1 else None)
    made: dict[str, Entity] = {"exp": exp, "run": r, "obs": obs, "art": art, "inv": inv}
    if version >= 2:
        prov = Provenance(
            run_id=r.id, experiment_id=exp.id, environment_id=env.id, dependency_digest=DIGEST,
            configuration_id=cfg.id, seed=r.seed,
            source=SourceRevision(SourceState.REPRODUCIBLE_SOURCE, "a" * 40, "main"),
            executor_version="0.0.1", execution=ExecutionParameters("m:f"), started_at=T0,
            runtime={"pid": 1},
        )  # fmt: skip
        payload = prov.to_dict()
        if version == 2:
            del payload["inputs"]  # Phase 2 provenance had no `inputs`
        insert(conn, prov, payload)
        made["prov"] = prov
    conn.execute(f"PRAGMA user_version = {version}")
    conn.close()
    return made


def test_phase1_database_migrates_to_current_and_keeps_all_data(tmp_path: Path) -> None:
    path = tmp_path / "v1.sqlite"
    made = make_database(path, 1)
    with SqliteRegistry(path) as reg:
        assert reg.get(Experiment, made["exp"].id) == made["exp"]
        assert reg.get(Run, made["run"].id) == made["run"]
        assert reg.get(Observation, made["obs"].id) == made["obs"]
        migrated = reg.get(Artifact, made["art"].id)  # content hash verified on read
        assert migrated.id == made["art"].id  # identity survives
        assert migrated.name == migrated.path  # v1 had no name: the path is what we know
        assert migrated.category.value == "OUTPUT"
        assert isinstance(made["art"], Artifact)
        assert migrated.digest == made["art"].digest
        assert reg.find(Provenance) == []
    with sqlite3.connect(path) as raw:
        assert raw.execute("PRAGMA user_version").fetchone()[0] == DB_SCHEMA_VERSION
        tables = {r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"provenance", "outcomes", "models", "datasets"} <= tables
    assert (tmp_path / "v1.sqlite.v1.bak").is_file()  # backup taken before migrating


def test_phase2_database_migrates_and_provenance_fingerprints_are_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "v2.sqlite"
    made = make_database(path, 2)
    prov = made["prov"]
    assert isinstance(prov, Provenance)
    expected = content_hash(
        {
            "experiment": prov.experiment_id,
            "source": {"state": "REPRODUCIBLE_SOURCE", "commit": "a" * 40},
            "environment": prov.environment_id,
            "dependencies": prov.dependency_digest,
            "configuration": prov.configuration_id,
            "seed": prov.seed,
            "executor": "0.0.1",
            "execution": {
                "procedure": "m:f",
                "resources": {"max_workers": 1, "memory_limit_mb": None, "timeout_seconds": None},
                "metadata": {},
            },
        }
    )  # the Phase 2 definition, written out independently
    with SqliteRegistry(path) as reg:
        (loaded,) = reg.find(Provenance)
        assert loaded.inputs is None
        assert loaded.fingerprint == expected
        assert loaded.id == prov.id
        # the migrated database is fully usable: a RUNNING run can be closed with an outcome
        running = reg.get(Run, made["run"].id)
        assert running.status is RunStatus.RUNNING
        reg.add(
            RunOutcome(
                running.id,
                T0,
                RunStatus.FAILED,
                0.1,
                1,
                (),
                ErrorInfo(FailureStage.EXECUTION, "x.Y", "m"),
            )
        )
        reg.update_status(running.with_status(RunStatus.FAILED))


def test_migrated_databases_keep_their_immutability_triggers(tmp_path: Path) -> None:
    path = tmp_path / "v1.sqlite"
    make_database(path, 1)
    SqliteRegistry(path).close()
    with sqlite3.connect(path) as raw:
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            raw.execute("UPDATE artifacts SET payload = '{}'")
        with pytest.raises(sqlite3.DatabaseError, match="cannot be deleted"):
            raw.execute("DELETE FROM artifacts")


def test_a_failing_migration_changes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "broken.sqlite"
    make_database(path, 1)
    with sqlite3.connect(path) as raw:
        raw.execute("DROP TRIGGER artifacts_immutable")
        raw.execute("UPDATE artifacts SET payload = 'not json'")
        raw.execute(
            "CREATE TRIGGER artifacts_immutable BEFORE UPDATE ON artifacts "
            "BEGIN SELECT RAISE(ABORT, 'registry records are immutable'); END"
        )
    with pytest.raises(ValueError, match="Expecting value"):
        SqliteRegistry(path)
    with sqlite3.connect(path) as raw:
        assert raw.execute("PRAGMA user_version").fetchone()[0] == 1  # still the old version
        tables = {r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert not {"provenance", "outcomes", "models", "datasets"} & tables  # no partial upgrade


def test_current_and_future_versions(tmp_path: Path) -> None:
    fresh = tmp_path / "fresh.sqlite"
    SqliteRegistry(fresh).close()
    assert not list(tmp_path.glob("*.bak"))  # no migration, no backup
    SqliteRegistry(fresh).close()  # reopening is a no-op
    future = tmp_path / "future.sqlite"
    with sqlite3.connect(future) as raw:
        raw.execute(f"PRAGMA user_version = {DB_SCHEMA_VERSION + 1}")
    with pytest.raises(SchemaVersionError, match="unsupported"):
        SqliteRegistry(future)


def test_in_memory_databases_need_no_backup() -> None:
    SqliteRegistry(":memory:").close()
    assert json.dumps({"ok": True})


def test_phase4_database_migrates_to_fault_tables_and_keeps_everything(tmp_path: Path) -> None:
    path = tmp_path / "v3.sqlite"
    made = make_database(path, 3)
    with SqliteRegistry(path) as reg:
        assert reg.get(Experiment, made["exp"].id) == made["exp"]
        (prov,) = reg.find(Provenance)
        assert prov.inputs is None
    with sqlite3.connect(path) as raw:
        assert raw.execute("PRAGMA user_version").fetchone()[0] == DB_SCHEMA_VERSION
        tables = {r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"fault_experiments", "fault_trials", "fault_analyses"} <= tables
    assert (tmp_path / "v3.sqlite.v3.bak").is_file()


def test_phase5_database_migrates_to_failure_tables_and_keeps_everything(tmp_path: Path) -> None:
    path = tmp_path / "v4.sqlite"
    made = make_database(path, 4)
    with SqliteRegistry(path) as reg:
        assert reg.get(Experiment, made["exp"].id) == made["exp"]
        assert reg.find(Provenance)[0].inputs is None
    with sqlite3.connect(path) as raw:
        assert raw.execute("PRAGMA user_version").fetchone()[0] == DB_SCHEMA_VERSION
        tables = {r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {
            "failure_signals",
            "failure_clusters",
            "failure_modes",
            "failure_evidence",
            "failure_relationships",
        } <= tables
        assert {
            "interaction_analyses",
            "interaction_effects",
            "interaction_evidence",
            "reliability_profiles",
            "reliability_references",
            "benchmarks",
            "benchmark_results",
            "benchmark_units",
        } <= tables
    assert (tmp_path / "v4.sqlite.v4.bak").is_file()


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5, 6, 7])
def test_every_prior_version_migrates_to_failure_tables_and_accepts_failure_records(
    tmp_path: Path, version: int
) -> None:
    from dataclasses import replace

    from experionyx.failures.entities import FailureSignal
    from failure_helpers import sig

    path = tmp_path / f"v{version}.sqlite"
    made = make_database(path, version)
    with SqliteRegistry(path) as reg:
        assert reg.get(Experiment, made["exp"].id) == made["exp"]  # existing data untouched
        run = reg.get(Run, made["run"].id)
        s = replace(sig(1), run_id=run.id, experiment_id=run.experiment_id)
        reg.add(s)  # the new tables work immediately after the stepwise migration
        assert reg.get(FailureSignal, s.id) == s
    with sqlite3.connect(path) as raw:
        assert raw.execute("PRAGMA user_version").fetchone()[0] == DB_SCHEMA_VERSION == 8
        tables = {r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {
            "failure_signals",
            "failure_clusters",
            "failure_modes",
            "failure_evidence",
            "failure_relationships",
        } <= tables
    assert (
        tmp_path / f"v{version}.sqlite.v{version}.bak"
    ).is_file()  # a backup is kept per migration
    with SqliteRegistry(path) as reg:  # reopening a current database is a no-op
        assert reg.get(FailureSignal, s.id) == s


def test_failure_migration_is_atomic_when_a_step_fails(tmp_path: Path) -> None:
    path = tmp_path / "v4.sqlite"
    make_database(path, 4)
    with sqlite3.connect(path) as raw:
        raw.execute("CREATE TABLE failure_modes (x INTEGER)")  # collides with the migration's DDL
    with pytest.raises(sqlite3.DatabaseError):
        SqliteRegistry(path)
    with sqlite3.connect(path) as raw:
        assert raw.execute("PRAGMA user_version").fetchone()[0] == 4  # still the old version
        tables = {r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "failure_signals" not in tables  # nothing half-applied


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5, 6, 7])
def test_every_prior_version_migrates_to_benchmark_tables_and_keeps_its_data(
    tmp_path: Path, version: int
) -> None:
    path = tmp_path / f"v{version}.sqlite"
    made = make_database(path, version)
    with SqliteRegistry(path) as reg:
        assert reg.get(Experiment, made["exp"].id) == made["exp"]  # existing data untouched
    with sqlite3.connect(path) as raw:
        assert raw.execute("PRAGMA user_version").fetchone()[0] == DB_SCHEMA_VERSION == 8
        tables = {r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"benchmarks", "benchmark_results", "benchmark_units"} <= tables
        assert (
            raw.execute("SELECT COUNT(*) FROM benchmark_results").fetchone()[0] == 0
        )  # new tables start empty
    assert (
        tmp_path / f"v{version}.sqlite.v{version}.bak"
    ).is_file()  # a backup is kept per migration


def test_benchmark_migration_is_atomic_when_a_step_fails(tmp_path: Path) -> None:
    path = tmp_path / "v7.sqlite"
    make_database(path, 7)
    with sqlite3.connect(path) as raw:
        raw.execute("CREATE TABLE benchmark_units (x INTEGER)")  # collides with the migration's DDL
    with pytest.raises(sqlite3.DatabaseError):
        SqliteRegistry(path)
    with sqlite3.connect(path) as raw:
        assert raw.execute("PRAGMA user_version").fetchone()[0] == 7  # still the old version
        tables = {r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "benchmarks" not in tables  # nothing half-applied

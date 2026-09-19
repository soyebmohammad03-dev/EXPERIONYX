import dataclasses
import sqlite3
from pathlib import Path

import pytest

from conftest import Lab
from experionyx.domain import (
    Artifact,
    Experiment,
    ExperimentStatus,
    Investigation,
    Run,
    RunStatus,
)
from experionyx.errors import (
    DuplicateError,
    MissingReferenceError,
    SchemaVersionError,
    ValidationError,
)
from experionyx.provenance import (
    ErrorInfo,
    ExecutionParameters,
    FailureStage,
    Provenance,
    RunOutcome,
    SourceRevision,
    SourceState,
)
from experionyx.sqlite import SqliteRegistry, _ddl
from factories import (
    DIGEST,
    T0,
    artifact,
    configuration,
    environment,
    experiment,
    investigation,
    observation,
    run,
)


def _provenance(r: Run, **over: object) -> Provenance:
    base: dict[str, object] = {
        "run_id": r.id,
        "experiment_id": r.experiment_id,
        "environment_id": r.environment_id,
        "dependency_digest": DIGEST,
        "configuration_id": experiment(investigation(), configuration()).configuration_id,
        "seed": r.seed,
        "source": SourceRevision(SourceState.UNKNOWN),
        "executor_version": "0.0.1",
        "execution": ExecutionParameters("m:f"),
        "started_at": T0,
    }
    return Provenance(**{**base, **over})  # type: ignore[arg-type]


@pytest.fixture
def stage(lab: Lab) -> tuple[Lab, Run]:
    env = environment()
    lab.registry.add(env)
    r = run(lab.experiment, env)
    lab.registry.add(r)
    return lab, r


# --- transactions -----------------------------------------------------------------------------


def test_transaction_commits_all_or_rolls_back_all(tmp_path: Path) -> None:
    reg = SqliteRegistry(tmp_path / "t.sqlite")
    inv, cfg = investigation(), configuration()

    def failing_group() -> None:
        with reg.transaction():
            reg.add(inv)
            reg.add(cfg)
            raise RuntimeError("abort")

    with pytest.raises(RuntimeError, match="abort"):
        failing_group()
    assert not reg.exists(Investigation, inv.id)
    assert reg.find(type(cfg)) == []
    with reg.transaction():
        reg.add(inv)
        reg.add(cfg)
    assert reg.exists(Investigation, inv.id)
    reg.close()


def test_nested_transactions_join_the_outer_one(tmp_path: Path) -> None:
    reg = SqliteRegistry(tmp_path / "n.sqlite")
    inv = investigation()

    def nested_then_fail() -> None:
        with reg.transaction():
            with reg.transaction():
                reg.add(inv)
            raise RuntimeError

    with pytest.raises(RuntimeError):
        nested_then_fail()
    assert not reg.exists(Investigation, inv.id)
    reg.close()


def test_failed_add_inside_transaction_does_not_leave_partial_state(tmp_path: Path) -> None:
    reg = SqliteRegistry(tmp_path / "p.sqlite")
    inv = investigation()
    reg.add(inv)
    cfg = configuration()

    def group_with_duplicate() -> None:
        with reg.transaction():
            reg.add(cfg)
            reg.add(inv)  # duplicate -> whole group rolls back

    with pytest.raises(DuplicateError):
        group_with_duplicate()
    assert not reg.exists(type(cfg), cfg.id)
    reg.close()


def test_writers_are_serialized_by_the_database_lock(tmp_path: Path) -> None:
    path = tmp_path / "lock.sqlite"
    a, b = SqliteRegistry(path), SqliteRegistry(path, timeout=0.1)
    with a.transaction():
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            b.add(investigation())
        a.add(configuration())
    b.add(investigation())  # lock released after commit
    assert a.exists(Investigation, investigation().id)  # visible across connections
    a.close()
    b.close()


def test_second_connection_sees_status_changes_and_rejects_stale_transitions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s.sqlite"
    a, b = SqliteRegistry(path), SqliteRegistry(path)
    inv, cfg = investigation(), configuration()
    exp = experiment(inv, cfg)
    for e in (inv, cfg, exp):
        a.add(e)
    a.update_status(exp.with_status(ExperimentStatus.READY))
    with pytest.raises(ValidationError):  # b's view is fresh: DRAFT->READY is no longer legal
        b.update_status(exp.with_status(ExperimentStatus.READY))
    assert b.get(Experiment, exp.id).status is ExperimentStatus.READY
    a.close()
    b.close()


# --- cross-record invariants ------------------------------------------------------------------


def test_provenance_must_match_its_run(stage: tuple[Lab, Run]) -> None:
    lab, r = stage
    for bad in (
        {"seed": r.seed + 1},
        {"environment_id": "env_" + "0" * 32},
        {"configuration_id": "cfg_" + "0" * 32},
        {"experiment_id": "exp_" + "0" * 32},
    ):
        with pytest.raises((ValidationError, MissingReferenceError)):
            lab.registry.add(_provenance(r, **bad))
    assert lab.registry.find(Provenance) == []


def test_provenance_only_while_pending_and_once(stage: tuple[Lab, Run]) -> None:
    lab, r = stage
    p = _provenance(r)
    lab.registry.add(p)
    with pytest.raises(DuplicateError):
        lab.registry.add(dataclasses.replace(p, started_at=T0.replace(year=2030)))
    lab.registry.update_status(r.with_status(RunStatus.RUNNING))
    other = Run(r.experiment_id, r.environment_id, 5, T0)
    lab.registry.add(other)
    lab.registry.update_status(other.with_status(RunStatus.RUNNING))
    with pytest.raises(ValidationError, match="PENDING"):
        lab.registry.add(_provenance(other))


def test_replay_must_stay_within_the_same_experiment(stage: tuple[Lab, Run]) -> None:
    lab, r = stage
    inv2, cfg2 = investigation(), configuration()
    other_exp = dataclasses.replace(experiment(inv2, cfg2), name="another")
    lab.registry.add(other_exp)
    foreign = Run(other_exp.id, r.environment_id, 0, T0)
    lab.registry.add(foreign)
    with pytest.raises(ValidationError, match="same experiment"):
        lab.registry.add(_provenance(r, replay_of=foreign.id))


def _failed_outcome(r: Run, **over: object) -> RunOutcome:
    base: dict[str, object] = {
        "run_id": r.id,
        "finished_at": T0,
        "status": RunStatus.FAILED,
        "duration_seconds": 0.1,
        "observation_count": 0,
        "error": ErrorInfo(FailureStage.EXECUTION, "x.Y", "m"),
    }
    return RunOutcome(**{**base, **over})  # type: ignore[arg-type]


def test_outcome_requires_matching_observation_count(stage: tuple[Lab, Run]) -> None:
    lab, r = stage
    lab.registry.add(observation(r))
    with pytest.raises(ValidationError, match="observations"):
        lab.registry.add(_failed_outcome(r, observation_count=0))
    lab.registry.add(_failed_outcome(r, observation_count=1))  # PENDING -> FAILED is legal


def test_outcome_status_must_be_a_legal_transition(stage: tuple[Lab, Run]) -> None:
    lab, r = stage
    with pytest.raises(ValidationError, match="illegal"):
        lab.registry.add(
            RunOutcome(r.id, T0, RunStatus.COMPLETED, 1.0, 0)  # PENDING cannot complete
        )
    assert lab.registry.find(RunOutcome) == []


def test_running_run_needs_provenance_before_an_outcome(stage: tuple[Lab, Run]) -> None:
    lab, r = stage
    lab.registry.update_status(r.with_status(RunStatus.RUNNING))  # bypassing provenance
    with pytest.raises(ValidationError, match="provenance"):
        lab.registry.add(_failed_outcome(r))


def test_outcome_cannot_list_another_runs_artifact(stage: tuple[Lab, Run]) -> None:
    lab, r = stage
    other = Run(r.experiment_id, r.environment_id, 9, T0)
    lab.registry.add(other)
    foreign = artifact(other)
    lab.registry.add(foreign)
    with pytest.raises(ValidationError, match="different run"):
        lab.registry.add(_failed_outcome(r, artifact_ids=(foreign.id,)))


def test_outcome_cannot_list_an_unregistered_artifact(stage: tuple[Lab, Run]) -> None:
    lab, r = stage
    never_registered = artifact(r)
    from experionyx.errors import MissingReferenceError

    with pytest.raises(MissingReferenceError):
        lab.registry.add(_failed_outcome(r, artifact_ids=(never_registered.id,)))
    assert isinstance(never_registered, Artifact)


def test_records_are_immutable_in_the_new_tables(stage: tuple[Lab, Run]) -> None:
    lab, r = stage
    lab.registry.add(_provenance(r))
    raw = lab.registry._conn
    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        raw.execute("UPDATE provenance SET payload = '{}'")
    with pytest.raises(sqlite3.DatabaseError, match="cannot be deleted"):
        raw.execute("DELETE FROM provenance")


def test_replay_of_column_is_nullable_and_queryable(stage: tuple[Lab, Run]) -> None:
    lab, r = stage
    lab.registry.add(_provenance(r))
    assert lab.registry.find(Provenance, run_id=r.id)[0].replay_of is None


# --- schema versions --------------------------------------------------------------------------


def test_unversioned_or_future_databases_are_rejected(tmp_path: Path) -> None:
    future = tmp_path / "future.sqlite"
    with sqlite3.connect(future) as raw:
        raw.execute("PRAGMA user_version = 99")
    with pytest.raises(SchemaVersionError, match="99"):
        SqliteRegistry(future)


def test_schema_contains_all_tables_and_is_versioned(tmp_path: Path) -> None:
    path = tmp_path / "v.sqlite"
    SqliteRegistry(path).close()
    with sqlite3.connect(path) as raw:
        tables = {r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert raw.execute("PRAGMA user_version").fetchone()[0] == 3
    assert {
        "provenance",
        "outcomes",
        "runs",
        "artifacts",
        "observations",
        "models",
        "datasets",
    } <= tables
    assert any(x.startswith("CREATE TABLE provenance") for x in _ddl())

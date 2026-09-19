import dataclasses
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from experionyx.domain import (
    Artifact,
    Claim,
    ConfigurationRef,
    EnvironmentSnapshot,
    Evidence,
    EvidenceRelation,
    EvidenceTarget,
    Experiment,
    ExperimentStatus,
    Investigation,
    Observation,
    Run,
    RunStatus,
)
from experionyx.errors import (
    CorruptRecordError,
    DuplicateError,
    MissingReferenceError,
    NotFoundError,
    SchemaVersionError,
    ValidationError,
)
from experionyx.registry import Registry
from experionyx.sqlite import DB_SCHEMA_VERSION, SqliteRegistry
from factories import (
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


@pytest.fixture
def reg() -> Iterator[SqliteRegistry]:
    r = SqliteRegistry(":memory:")
    yield r
    r.close()


@pytest.fixture
def base(reg: SqliteRegistry) -> tuple[Investigation, Experiment, EnvironmentSnapshot, Run]:
    inv, cfg, env = investigation(), configuration(), environment()
    exp = experiment(inv, cfg)
    r = run(exp, env)
    for e in (inv, cfg, env, exp, r):
        reg.add(e)
    return inv, exp, env, r


def test_satisfies_registry_protocol(reg: SqliteRegistry) -> None:
    protocol_view: Registry = reg
    assert protocol_view.exists(Investigation, "inv_" + "0" * 32) is False


def test_create_get_exists(reg: SqliteRegistry) -> None:
    inv = investigation()
    assert not reg.exists(Investigation, inv.id)
    reg.add(inv)
    assert reg.exists(Investigation, inv.id)
    assert reg.get(Investigation, inv.id) == inv


def test_config_and_env_round_trip_through_storage(reg: SqliteRegistry) -> None:
    cfg, env = configuration(), environment()
    reg.add(cfg)
    reg.add(env)
    assert reg.get(ConfigurationRef, cfg.id) == cfg
    assert reg.get(EnvironmentSnapshot, env.id) == env


def test_missing_entity(reg: SqliteRegistry) -> None:
    with pytest.raises(NotFoundError):
        reg.get(Investigation, "inv_" + "0" * 32)


def test_duplicate_id_rejected(reg: SqliteRegistry) -> None:
    inv = investigation()
    reg.add(inv)
    with pytest.raises(DuplicateError):
        reg.add(inv)
    # same identity, different timestamp: still the same ID, still rejected (immutable)
    with pytest.raises(DuplicateError):
        reg.add(dataclasses.replace(inv, created_at=T0.replace(year=2027)))
    assert reg.get(Investigation, inv.id).created_at == T0


def test_missing_references_rejected(reg: SqliteRegistry) -> None:
    inv, cfg, env = investigation(), configuration(), environment()
    exp = experiment(inv, cfg)
    with pytest.raises(MissingReferenceError):
        reg.add(exp)  # investigation + configuration absent
    reg.add(inv)
    with pytest.raises(MissingReferenceError):
        reg.add(exp)  # configuration still absent
    reg.add(cfg)
    reg.add(exp)
    with pytest.raises(MissingReferenceError):
        reg.add(run(exp, env))  # environment absent
    assert not reg.exists(Run, run(exp, env).id)


def test_child_records_require_their_run(reg: SqliteRegistry) -> None:
    ghost = run(experiment(investigation(), configuration()), environment())
    with pytest.raises(MissingReferenceError):
        reg.add(observation(ghost))
    with pytest.raises(MissingReferenceError):
        reg.add(artifact(ghost))


def test_evidence_requires_claim_and_target(
    reg: SqliteRegistry, base: tuple[Investigation, Experiment, EnvironmentSnapshot, Run]
) -> None:
    inv, _, _, r = base
    c, o = claim(inv), observation(r)
    with pytest.raises(MissingReferenceError):
        reg.add(evidence(c, o))  # claim missing
    reg.add(c)
    with pytest.raises(MissingReferenceError):
        reg.add(evidence(c, o))  # target observation missing
    reg.add(o)
    reg.add(evidence(c, o))


def test_evidence_can_target_each_kind(
    reg: SqliteRegistry, base: tuple[Investigation, Experiment, EnvironmentSnapshot, Run]
) -> None:
    inv, exp, _, r = base
    c, o, a = claim(inv), observation(r), artifact(r)
    for e in (c, o, a):
        reg.add(e)
    targets = [
        (EvidenceTarget.EXPERIMENT, exp.id),
        (EvidenceTarget.RUN, r.id),
        (EvidenceTarget.OBSERVATION, o.id),
        (EvidenceTarget.ARTIFACT, a.id),
    ]
    for kind, tid in targets:
        reg.add(Evidence(c.id, kind, tid, EvidenceRelation.CONTEXT, T0))
    assert len(reg.find(Evidence, claim_id=c.id)) == 4
    assert len(reg.find(Evidence, target_kind="RUN")) == 1


def test_find_filters_and_ordering(
    reg: SqliteRegistry, base: tuple[Investigation, Experiment, EnvironmentSnapshot, Run]
) -> None:
    inv, exp, env, _ = base
    more = [run(exp, env, seed=s) for s in (1, 2, 3)]
    for m in more:
        reg.add(m)
    runs = reg.find(Run, experiment_id=exp.id)
    assert len(runs) == 4
    assert [r.id for r in runs] == sorted(r.id for r in runs)
    assert reg.find(Run, experiment_id=exp.id, status="PENDING") == runs
    assert reg.find(Run, status="COMPLETED") == []
    assert reg.find(Experiment, investigation_id=inv.id) == [exp]
    assert len(reg.find(Run)) == 4
    with pytest.raises(ValidationError):
        reg.find(Run, seed="1")


def test_status_update_follows_lifecycle(
    reg: SqliteRegistry, base: tuple[Investigation, Experiment, EnvironmentSnapshot, Run]
) -> None:
    _, exp, _, r = base
    ready = exp.with_status(ExperimentStatus.READY)
    reg.update_status(ready)
    assert reg.get(Experiment, exp.id).status is ExperimentStatus.READY
    assert reg.find(Experiment, status="READY") == [ready]
    with pytest.raises(ValidationError):
        reg.update_status(dataclasses.replace(exp, status=ExperimentStatus.COMPLETED))
    running = r.with_status(RunStatus.RUNNING)
    reg.update_status(running)
    assert reg.get(Run, r.id) == running
    with pytest.raises(ValidationError):  # illegal: RUNNING -> PENDING
        reg.update_status(r)


def test_status_update_cannot_change_other_fields(
    reg: SqliteRegistry, base: tuple[Investigation, Experiment, EnvironmentSnapshot, Run]
) -> None:
    _, exp, _, _ = base
    tampered = dataclasses.replace(
        exp.with_status(ExperimentStatus.READY), created_at=T0.replace(year=2030)
    )
    with pytest.raises(ValidationError):
        reg.update_status(tampered)
    assert reg.get(Experiment, exp.id) == exp
    # changing an identity field yields a different ID, i.e. a different (absent) record
    with pytest.raises(NotFoundError):
        reg.update_status(
            dataclasses.replace(exp, hypothesis="other").with_status(ExperimentStatus.READY)
        )


def test_status_update_of_missing_record(reg: SqliteRegistry) -> None:
    exp = experiment(investigation(), configuration())
    with pytest.raises(NotFoundError):
        reg.update_status(exp.with_status(ExperimentStatus.READY))


def test_database_enforces_immutability_below_the_python_layer(
    reg: SqliteRegistry, base: tuple[Investigation, Experiment, EnvironmentSnapshot, Run]
) -> None:
    _, _, _, r = base
    raw = reg._conn
    with pytest.raises(sqlite3.DatabaseError, match="cannot be deleted"):
        raw.execute("DELETE FROM investigations")
    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        raw.execute("UPDATE investigations SET payload = '{}'")
    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        raw.execute("UPDATE runs SET experiment_id = 'x'")
    with pytest.raises(sqlite3.IntegrityError):  # FK enforced by SQLite itself
        raw.execute(
            "INSERT INTO observations (id, run_id, payload, content_hash) "
            "VALUES ('o', 'nope', '{}', 'h')"
        )
    assert reg.exists(Run, r.id)


def test_corruption_is_detected(
    reg: SqliteRegistry, base: tuple[Investigation, Experiment, EnvironmentSnapshot, Run]
) -> None:
    inv, _, _, _ = base
    raw = reg._conn
    raw.execute("DROP TRIGGER investigations_immutable")
    payload = investigation().to_dict() | {"created_at": "2030-01-01T00:00:00+00:00"}
    import json

    raw.execute("UPDATE investigations SET payload = ?", (json.dumps(payload),))
    with pytest.raises(CorruptRecordError):
        reg.get(Investigation, inv.id)


# --- schema / versioning ----------------------------------------------------------------------


def test_new_database_is_versioned_and_persistent(tmp_path: Path) -> None:
    path = tmp_path / "reg.sqlite"
    inv = investigation()
    with SqliteRegistry(path) as reg:
        reg.add(inv)
    with sqlite3.connect(path) as raw:
        assert raw.execute("PRAGMA user_version").fetchone()[0] == DB_SCHEMA_VERSION
    with SqliteRegistry(path) as reg2:  # reopen: schema not re-created, data intact
        assert reg2.get(Investigation, inv.id) == inv


def test_unsupported_schema_version_rejected(tmp_path: Path) -> None:
    path = tmp_path / "future.sqlite"
    with sqlite3.connect(path) as raw:
        raw.execute("PRAGMA user_version = 99")
    with pytest.raises(SchemaVersionError, match="99"):
        SqliteRegistry(path)


def test_foreign_nonempty_database_rejected(tmp_path: Path) -> None:
    path = tmp_path / "other.sqlite"
    with sqlite3.connect(path) as raw:
        raw.execute("CREATE TABLE unrelated (x)")
    with pytest.raises(SchemaVersionError):
        SqliteRegistry(path)


# --- full evidence chain ----------------------------------------------------------------------


def test_complete_investigation_chain_persisted_and_retrieved(tmp_path: Path) -> None:
    path = tmp_path / "lab.sqlite"
    inv, cfg, env = investigation(), configuration(), environment()
    exp = experiment(inv, cfg)
    runs = [run(exp, env, seed=s) for s in range(3)]
    obs = [
        Observation(r.id, "accuracy", 0.9 + i / 100, T0, unit="ratio") for i, r in enumerate(runs)
    ]
    art = artifact(runs[0])
    clm = claim(inv)
    evs = [evidence(clm, o) for o in obs] + [
        Evidence(clm.id, EvidenceTarget.ARTIFACT, art.id, EvidenceRelation.CONTEXT, T0, note="raw")
    ]

    with SqliteRegistry(path) as reg:
        for e in (inv, cfg, env, exp, *runs, *obs, art, clm, *evs):
            reg.add(e)
        reg.update_status(exp.with_status(ExperimentStatus.READY))

    with SqliteRegistry(path) as reg:  # fresh connection: everything comes from disk
        assert reg.get(Investigation, inv.id) == inv
        (stored_exp,) = reg.find(Experiment, investigation_id=inv.id)
        assert stored_exp.status is ExperimentStatus.READY
        assert stored_exp.model == exp.model
        assert stored_exp.dataset == exp.dataset
        assert reg.get(ConfigurationRef, stored_exp.configuration_id) == cfg
        stored_runs = reg.find(Run, experiment_id=exp.id)
        assert {r.seed for r in stored_runs} == {0, 1, 2}
        assert reg.get(EnvironmentSnapshot, stored_runs[0].environment_id) == env
        stored_obs = [o for r in stored_runs for o in reg.find(Observation, run_id=r.id)]
        assert sorted(float(o.value) for o in stored_obs) == [0.9, 0.91, 0.92]  # type: ignore[arg-type]
        assert reg.find(Artifact, run_id=runs[0].id) == [art]
        (stored_claim,) = reg.find(Claim, investigation_id=inv.id)
        stored_ev = reg.find(Evidence, claim_id=stored_claim.id)
        assert len(stored_ev) == 4
        assert {e.target_id for e in stored_ev} == {o.id for o in obs} | {art.id}
        # every evidence target resolves to a real record
        from experionyx.domain import EVIDENCE_TARGET_TYPES

        for e in stored_ev:
            assert reg.exists(EVIDENCE_TARGET_TYPES[e.target_kind], e.target_id)

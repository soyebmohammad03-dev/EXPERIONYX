import dataclasses
import json
from datetime import timedelta

import pytest

from experionyx.domain import ConfigurationRef, RunStatus
from experionyx.errors import SchemaVersionError, ValidationError
from experionyx.provenance import (
    ErrorInfo,
    ExecutionParameters,
    FailureStage,
    Provenance,
    ResourceLimits,
    RunOutcome,
    SourceRevision,
    SourceState,
    compare_provenance,
    config_diff,
)
from factories import DIGEST, T0, configuration, environment, experiment, investigation, run

COMMIT = "a" * 40
EXP = experiment(investigation(), configuration())
ENV = environment()
RUN = run(EXP, ENV)
ART = "art_" + "1" * 32


def prov(**over: object) -> Provenance:
    base: dict[str, object] = {
        "run_id": RUN.id,
        "experiment_id": EXP.id,
        "environment_id": ENV.id,
        "dependency_digest": DIGEST,
        "configuration_id": EXP.configuration_id,
        "seed": 3,
        "source": SourceRevision(SourceState.REPRODUCIBLE_SOURCE, COMMIT, "main"),
        "executor_version": "0.0.1",
        "execution": ExecutionParameters("pkg.mod:fn"),
        "started_at": T0,
        "runtime": {"pid": 1, "cpu_count": 8},
    }
    return Provenance(**{**base, **over})  # type: ignore[arg-type]


# --- validation -------------------------------------------------------------------------------


def test_source_state_invariants() -> None:
    SourceRevision(SourceState.UNKNOWN)
    with pytest.raises(ValidationError):
        SourceRevision(SourceState.UNKNOWN, COMMIT)
    with pytest.raises(ValidationError):
        SourceRevision(SourceState.REPRODUCIBLE_SOURCE)
    with pytest.raises(ValidationError):
        SourceRevision(SourceState.MODIFIED_WORKTREE, "abc123")
    with pytest.raises(ValidationError):
        SourceRevision("clean", COMMIT)  # type: ignore[arg-type]


def test_resource_limits_validation() -> None:
    ResourceLimits(2, 512, 1.5)
    for bad in ({"max_workers": 0}, {"memory_limit_mb": 0}, {"timeout_seconds": 0.0},
                {"timeout_seconds": float("nan")}):  # fmt: skip
        with pytest.raises(ValidationError):
            ResourceLimits(**bad)


def test_provenance_requires_traceable_typed_fields() -> None:
    with pytest.raises(ValidationError):
        prov(run_id="run_bad")
    with pytest.raises(ValidationError):
        prov(dependency_digest="deadbeef")
    with pytest.raises(ValidationError):
        prov(seed=-1)
    with pytest.raises(ValidationError):
        prov(source="git")
    with pytest.raises(ValidationError):
        prov(replay_of=RUN.id)  # a run cannot replay itself
    with pytest.raises(ValidationError):
        prov(started_at=T0.replace(tzinfo=None))


def test_provenance_is_immutable_including_nested_mappings() -> None:
    p = prov(execution=ExecutionParameters("m:f", metadata={"k": {"x": 1}}))
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.seed = 9  # type: ignore[misc]
    with pytest.raises(TypeError):
        p.runtime["pid"] = 2  # type: ignore[index]
    with pytest.raises(TypeError):
        p.execution.metadata["k"]["x"] = 2  # type: ignore[index]


def test_replay_marker() -> None:
    assert prov().is_replay is False
    assert prov(replay_of="run_" + "2" * 32).is_replay is True


# --- fingerprint ------------------------------------------------------------------------------


def test_fingerprint_is_stable_and_ignores_volatile_fields() -> None:
    base = prov()
    assert base.fingerprint == prov().fingerprint
    assert base.fingerprint.startswith("sha256:")
    same = [
        prov(started_at=T0 + timedelta(days=9)),  # wall clock
        prov(runtime={"pid": 999}),  # informational
        prov(run_id="run_" + "9" * 32),  # which run it was
        prov(replay_of="run_" + "8" * 32),
        prov(source=SourceRevision(SourceState.REPRODUCIBLE_SOURCE, COMMIT, "other-branch")),
    ]
    assert {p.fingerprint for p in same} == {base.fingerprint}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", 4),
        ("executor_version", "0.0.2"),
        ("dependency_digest", "sha256:" + "0" * 64),
        ("environment_id", "env_" + "0" * 32),
        ("configuration_id", "cfg_" + "0" * 32),
        ("experiment_id", "exp_" + "0" * 32),
        ("execution", ExecutionParameters("pkg.mod:other")),
        ("execution", ExecutionParameters("pkg.mod:fn", ResourceLimits(2))),
        ("execution", ExecutionParameters("pkg.mod:fn", metadata={"a": 1})),
        ("source", SourceRevision(SourceState.REPRODUCIBLE_SOURCE, "b" * 40)),
        ("source", SourceRevision(SourceState.MODIFIED_WORKTREE, COMMIT)),
        ("source", SourceRevision(SourceState.UNKNOWN)),
    ],
)
def test_fingerprint_changes_with_every_reproducibility_input(field: str, value: object) -> None:
    assert prov(**{field: value}).fingerprint != prov().fingerprint


def test_compare_provenance_names_differing_components() -> None:
    assert compare_provenance(prov(), prov(started_at=T0 + timedelta(1))) == ()
    changed = prov(seed=5, source=SourceRevision(SourceState.UNKNOWN))
    assert compare_provenance(prov(), changed) == ("source", "seed")


# --- serialization ----------------------------------------------------------------------------


def test_provenance_round_trips_through_json() -> None:
    p = prov(
        replay_of="run_" + "2" * 32,
        execution=ExecutionParameters("m:f", ResourceLimits(2, 64, 9.5), {"a": [1, 2]}),
    )
    again = Provenance.from_dict(json.loads(json.dumps(p.to_dict())))
    assert again == p
    assert again.fingerprint == p.fingerprint
    assert again.id == p.id


def test_provenance_rejects_bad_payloads() -> None:
    d = prov().to_dict()
    with pytest.raises(SchemaVersionError):
        Provenance.from_dict({**d, "schema_version": 9})
    with pytest.raises(ValidationError):
        Provenance.from_dict({**d, "surprise": 1})


# --- outcome ----------------------------------------------------------------------------------


def outcome(**over: object) -> RunOutcome:
    base: dict[str, object] = {
        "run_id": RUN.id,
        "finished_at": T0,
        "status": RunStatus.COMPLETED,
        "duration_seconds": 1.5,
        "observation_count": 2,
        "artifact_ids": (ART,),
    }
    return RunOutcome(**{**base, **over})  # type: ignore[arg-type]


def test_outcome_status_and_error_must_agree() -> None:
    err = ErrorInfo(FailureStage.EXECUTION, "builtins.ValueError", "")
    outcome()
    outcome(status=RunStatus.FAILED, error=err)
    with pytest.raises(ValidationError):
        outcome(status=RunStatus.FAILED)
    with pytest.raises(ValidationError):
        outcome(error=err)
    with pytest.raises(ValidationError):
        outcome(status=RunStatus.RUNNING)


def test_outcome_validation() -> None:
    with pytest.raises(ValidationError):
        outcome(duration_seconds=-1.0)
    with pytest.raises(ValidationError):
        outcome(artifact_ids=("art_" + "2" * 32, ART))  # unsorted
    with pytest.raises(ValidationError):
        outcome(artifact_ids=(ART, ART))  # duplicates
    with pytest.raises(ValidationError):
        outcome(artifact_ids=("nope",))
    diag = ErrorInfo(FailureStage.EXECUTION, "x.Y", "m", diagnostic_artifact_id="art_" + "3" * 32)
    with pytest.raises(ValidationError):
        outcome(status=RunStatus.FAILED, error=diag)  # diagnostic not listed


def test_outcome_allows_unknown_duration_and_empty_message() -> None:
    o = outcome(
        status=RunStatus.FAILED,
        duration_seconds=None,
        error=ErrorInfo(FailureStage.INTERRUPTED, "x.Y", ""),
    )
    assert RunOutcome.from_dict(json.loads(json.dumps(o.to_dict()))) == o


def test_outcome_round_trip_with_diagnostic() -> None:
    err = ErrorInfo(FailureStage.EXECUTION, "builtins.ValueError", "boom", ART)
    o = outcome(status=RunStatus.FAILED, error=err)
    assert RunOutcome.from_dict(json.loads(json.dumps(o.to_dict()))) == o


# --- configuration comparison -----------------------------------------------------------------


def test_config_diff_by_dotted_path() -> None:
    a = ConfigurationRef({"lr": 0.1, "opt": {"name": "sgd", "m": 0.9}, "layers": [1, 2], "gone": 1})
    b = ConfigurationRef({"lr": 0.2, "opt": {"name": "sgd", "m": 0.9}, "layers": [1, 3], "new": 1})
    d = config_diff(a, b)
    assert d.changed == ("layers", "lr")
    assert d.added == ("new",)
    assert d.removed == ("gone",)
    assert not d.identical
    assert config_diff(
        a,
        ConfigurationRef(
            {"gone": 1, "layers": [1, 2], "lr": 0.1, "opt": {"m": 0.9, "name": "sgd"}}
        ),
    ).identical

import dataclasses
from datetime import datetime, timedelta, timezone

import pytest

from experionyx.domain import (
    Claim,
    ClaimStatus,
    ConfigurationRef,
    EnvironmentSnapshot,
    Evidence,
    EvidenceRelation,
    EvidenceTarget,
    ExperimentStatus,
    Investigation,
    ModelRef,
    Observation,
    Run,
    RunStatus,
)
from experionyx.errors import SchemaVersionError, ValidationError
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

INV = investigation()
CFG = configuration()
ENV = environment()
EXP = experiment(INV, CFG)
RUN = run(EXP, ENV)
OBS = observation(RUN)


@pytest.mark.parametrize("bad", ["", " x", "x ", "  "])
def test_text_rejects_empty_or_padded(bad: str) -> None:
    with pytest.raises(ValidationError):
        Investigation(bad, "q", T0)
    with pytest.raises(ValidationError):
        Investigation("n", bad, T0)


@pytest.mark.parametrize("bad", ["", "1 0", "-1", "v 2", " 1"])
def test_malformed_versions_rejected(bad: str) -> None:
    with pytest.raises(ValidationError):
        ModelRef("m", bad)


def test_bad_digest_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelRef("m", "1", "sha256:xyz")


@pytest.mark.parametrize(
    "bad",
    [datetime(2026, 1, 1), datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=2))), "2026-01-01"],
)
def test_timestamps_must_be_utc_aware(bad: object) -> None:
    with pytest.raises(ValidationError):
        Investigation("n", "q", bad)  # type: ignore[arg-type]


def test_references_must_have_correct_prefix() -> None:
    with pytest.raises(ValidationError):
        Run("", ENV.id, 0, T0)
    with pytest.raises(ValidationError):
        Run(INV.id, ENV.id, 0, T0)  # investigation ID where experiment ID is required
    with pytest.raises(ValidationError):
        Run(EXP.id, ENV.id, -1, T0)
    with pytest.raises(ValidationError):
        Run(EXP.id, ENV.id, True, T0)  # bool is not a seed
    with pytest.raises(ValidationError):
        Observation("nope", "acc", 1.0, T0)


def test_enum_fields_are_not_coerced() -> None:
    with pytest.raises(ValidationError):
        Run(EXP.id, ENV.id, 0, T0, status="PENDING")  # type: ignore[arg-type]


def test_observation_value_rules() -> None:
    with pytest.raises(ValidationError):
        Observation(RUN.id, "acc", float("nan"), T0)
    with pytest.raises(ValidationError):
        Observation(RUN.id, "acc", [1], T0)  # type: ignore[arg-type]


def test_observation_kind_restricted() -> None:
    from experionyx.domain import EpistemicKind

    with pytest.raises(ValidationError):
        Observation(RUN.id, "acc", 1, T0, epistemic_kind=EpistemicKind.HYPOTHESIS)
    Observation(RUN.id, "acc", 1, T0, epistemic_kind=EpistemicKind.DERIVED_METRIC)


@pytest.mark.parametrize("path", ["/abs", "../up", "a/../b", "a\\b", "./x", ""])
def test_artifact_path_must_be_relative_and_normalized(path: str) -> None:
    with pytest.raises(ValidationError):
        dataclasses.replace(artifact(RUN), path=path)


def test_artifact_requires_producing_run_and_media_type() -> None:
    with pytest.raises(ValidationError):
        dataclasses.replace(artifact(RUN), run_id=EXP.id)
    with pytest.raises(ValidationError):
        dataclasses.replace(artifact(RUN), media_type="json")


def test_claim_requires_traceability() -> None:
    with pytest.raises(ValidationError):
        Claim(INV.id, "s", "", T0)
    with pytest.raises(ValidationError):
        Claim("inv_bad", "s", "me", T0)
    assert claim(INV).status is ClaimStatus.INSUFFICIENT_EVIDENCE


def test_evidence_target_id_must_match_kind() -> None:
    c = claim(INV)
    with pytest.raises(ValidationError):
        Evidence(c.id, EvidenceTarget.RUN, OBS.id, EvidenceRelation.SUPPORTS, T0)
    with pytest.raises(ValidationError):
        Evidence(c.id, EvidenceTarget.OBSERVATION, "", EvidenceRelation.SUPPORTS, T0)


def test_entities_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        RUN.seed = 9  # type: ignore[misc]


def test_configuration_is_deeply_immutable_and_isolated_from_input() -> None:
    source: dict[str, object] = {"layers": [1, 2], "opt": {"lr": 0.1}}
    cfg = ConfigurationRef(source)
    before = cfg.id
    source["layers"] = [9]
    source["opt"] = {"lr": 9}
    assert cfg.id == before
    with pytest.raises(TypeError):
        cfg.parameters["new"] = 1  # type: ignore[index]
    opt = cfg.parameters["opt"]
    with pytest.raises(TypeError):
        opt["lr"] = 2  # type: ignore[index]
    assert cfg.parameters["layers"] == (1, 2)


def test_configuration_rejects_non_json_values() -> None:
    for bad in ({"x": object()}, {"x": float("inf")}, {1: "a"}, {"": 1}):
        with pytest.raises(ValidationError):
            ConfigurationRef(bad)  # type: ignore[arg-type]


def test_environment_package_versions_validated() -> None:
    with pytest.raises(ValidationError):
        EnvironmentSnapshot("3.12", "os", "arm64", {"numpy": "not a version"})


# --- identity ---------------------------------------------------------------------------------


def test_ids_are_deterministic_and_prefixed() -> None:
    assert investigation().id == INV.id
    assert INV.id.startswith("inv_")
    assert len(INV.id) == 4 + 32
    assert [x.id[:4] for x in (EXP, RUN, OBS)] == ["exp_", "run_", "obs_"]


def test_config_id_ignores_key_order() -> None:
    assert ConfigurationRef({"a": 1, "b": 2}).id == ConfigurationRef({"b": 2, "a": 1}).id
    assert ConfigurationRef({"a": 1}).id != ConfigurationRef({"a": 2}).id


def test_identity_ignores_timestamps_and_status() -> None:
    later = dataclasses.replace(INV, created_at=T0 + timedelta(days=1))
    assert later.id == INV.id
    assert EXP.with_status(ExperimentStatus.READY).id == EXP.id
    assert later.content_hash() != INV.content_hash()  # full-record hash does see timestamps


def test_identity_changes_with_identity_fields() -> None:
    assert dataclasses.replace(EXP, name="other").id != EXP.id
    assert dataclasses.replace(RUN, seed=1).id != RUN.id
    assert dataclasses.replace(RUN, attempt=1).id != RUN.id
    assert dataclasses.replace(OBS, sequence=1).id != OBS.id
    assert dataclasses.replace(OBS, value=0.5).id == OBS.id  # value is content, not identity


def test_artifact_identity_is_content_addressed() -> None:
    a = artifact(RUN)
    assert dataclasses.replace(a, digest="sha256:" + "cd" * 32).id != a.id
    assert dataclasses.replace(a, size_bytes=1).id == a.id


# --- lifecycle --------------------------------------------------------------------------------


def test_experiment_lifecycle_transitions() -> None:
    e = EXP
    for nxt in (ExperimentStatus.READY, ExperimentStatus.RUNNING, ExperimentStatus.COMPLETED):
        e = e.with_status(nxt)
    assert e.status is ExperimentStatus.COMPLETED
    with pytest.raises(ValidationError):
        e.with_status(ExperimentStatus.RUNNING)
    with pytest.raises(ValidationError):
        EXP.with_status(ExperimentStatus.COMPLETED)  # DRAFT cannot skip to COMPLETED
    assert EXP.with_status(ExperimentStatus.CANCELLED).status is ExperimentStatus.CANCELLED


def test_run_lifecycle_transitions() -> None:
    r = RUN.with_status(RunStatus.RUNNING).with_status(RunStatus.FAILED)
    assert r.status is RunStatus.FAILED
    with pytest.raises(ValidationError):
        RUN.with_status(RunStatus.COMPLETED)
    with pytest.raises(ValidationError):
        r.with_status(RunStatus.RUNNING)


# --- serialization ----------------------------------------------------------------------------

ALL = [
    INV,
    CFG,
    ENV,
    EXP.with_status(ExperimentStatus.READY),
    RUN,
    OBS,
    artifact(RUN),
    claim(INV),
    evidence(claim(INV), OBS),
    Observation(RUN.id, "flag", True, T0),
    Observation(RUN.id, "label", "cat", T0, sequence=2),
]


@pytest.mark.parametrize("entity", ALL, ids=lambda e: type(e).__name__)
def test_round_trip(entity: object) -> None:
    import json

    cls = type(entity)
    d = entity.to_dict()  # type: ignore[attr-defined]
    assert json.loads(json.dumps(d)) == d  # plain JSON
    again = cls.from_dict(json.loads(json.dumps(d)))  # type: ignore[attr-defined]
    assert again == entity
    assert again.id == entity.id  # type: ignore[attr-defined]
    assert again.to_dict() == d


def test_payload_carries_kind_and_schema_version() -> None:
    d = RUN.to_dict()
    assert d["kind"] == "run"
    assert d["schema_version"] == 1


def test_from_dict_rejects_bad_payloads() -> None:
    d = RUN.to_dict()
    with pytest.raises(SchemaVersionError):
        Run.from_dict({**d, "schema_version": 2})
    with pytest.raises(ValidationError):
        Run.from_dict({**d, "kind": "experiment"})
    with pytest.raises(ValidationError):
        Run.from_dict({**d, "surprise": 1})
    with pytest.raises(ValidationError):
        Run.from_dict({k: x for k, x in d.items() if k != "seed"})
    with pytest.raises(ValidationError):
        Run.from_dict({**d, "seed": "0"})
    with pytest.raises(ValidationError):
        Run.from_dict({**d, "status": "BOGUS"})
    with pytest.raises(ValidationError):
        Run.from_dict({**d, "created_at": "yesterday"})
    with pytest.raises(ValidationError):
        Run.from_dict({**d, "created_at": "2026-01-01T00:00:00"})  # naive

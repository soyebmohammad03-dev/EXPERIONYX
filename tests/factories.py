"""Small builders for realistic domain objects used across tests."""

from datetime import UTC, datetime

from experionyx.domain import (
    Artifact,
    Claim,
    ConfigurationRef,
    DatasetRef,
    EnvironmentSnapshot,
    Evidence,
    EvidenceRelation,
    EvidenceTarget,
    Experiment,
    Investigation,
    ModelRef,
    Observation,
    Run,
)

T0 = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)
DIGEST = "sha256:" + "ab" * 32


def investigation() -> Investigation:
    return Investigation("seed-stability", "Does accuracy vary across seeds?", T0)


def configuration() -> ConfigurationRef:
    return ConfigurationRef({"lr": 0.1, "epochs": 5, "layers": [16, 8]})


def environment() -> EnvironmentSnapshot:
    return EnvironmentSnapshot("3.12.1", "macOS-15", "arm64", {"numpy": "2.0.1"}, "ca28b59")


def experiment(inv: Investigation, cfg: ConfigurationRef) -> Experiment:
    return Experiment(
        inv.id,
        "baseline-logreg",
        "Accuracy std across seeds is below 0.01",
        ModelRef("logreg", "1.0"),
        DatasetRef("iris", "2024-01", DIGEST),
        cfg.id,
        T0,
    )


def run(exp: Experiment, env: EnvironmentSnapshot, seed: int = 0) -> Run:
    return Run(exp.id, env.id, seed, T0)


def observation(r: Run) -> Observation:
    return Observation(r.id, "accuracy", 0.93, T0, unit="ratio")


def artifact(r: Run) -> Artifact:
    return Artifact(r.id, "metrics", "out/metrics.json", DIGEST, 128, "application/json", T0)


def claim(inv: Investigation) -> Claim:
    return Claim(inv.id, "Accuracy is stable across seeds", "soyeb", T0)


def evidence(c: Claim, o: Observation) -> Evidence:
    return Evidence(c.id, EvidenceTarget.OBSERVATION, o.id, EvidenceRelation.SUPPORTS, T0)

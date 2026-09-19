"""The profile procedure, run by the normal execution engine so that a profile is a Run with
provenance, an artifact, an observation and a claim, and can be replayed."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from experionyx.artifacts import ArtifactStore
from experionyx.domain import (
    ClaimStatus,
    ConfigurationRef,
    DatasetRef,
    EpistemicKind,
    EvidenceRelation,
    EvidenceTarget,
    Experiment,
    ExperimentStatus,
    ModelRef,
    RunStatus,
    to_jsonable,
)
from experionyx.errors import ExperionyxError, ValidationError
from experionyx.evaluation.config import _thaw
from experionyx.execution import Executor, RunContext, resolve_procedure
from experionyx.faults.report import read_artifact
from experionyx.interactions.lifecycle import REPLAY_TOLERANCE, _compare
from experionyx.registry import Registry
from experionyx.reliability import builder
from experionyx.reliability.entities import ReliabilityProfile, ReliabilityReference
from experionyx.reliability.spec import PROFILE_VERSION, ProfileSpec
from experionyx.reliability.taxonomy import DimensionStatus

PROCEDURE = "experionyx.reliability.engine:run_reliability_profile"


@dataclass(frozen=True)
class ProfileRequest:
    investigation_id: str
    spec: Mapping[str, object]

    def to_parameters(self) -> dict[str, object]:
        return {
            "reliability_profile": {
                "investigation_id": self.investigation_id,
                "spec": dict(self.spec),
            }
        }

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, object]) -> "ProfileRequest":
        if set(parameters) != {"reliability_profile"}:
            raise ValidationError(
                "a profile configuration must be exactly {'reliability_profile': ...}"
            )
        body = _thaw(parameters["reliability_profile"])
        if (
            not isinstance(body, dict)
            or set(body) != {"investigation_id", "spec"}
            or not isinstance(body["spec"], dict)
            or not isinstance(body["investigation_id"], str)
        ):
            raise ValidationError("'reliability_profile' has the wrong fields")
        return cls(body["investigation_id"], body["spec"])


def _write(ctx: RunContext, name: str, payload: object) -> str:
    directory = ctx.artifact_dir / "reliability"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(
        json.dumps(to_jsonable(payload), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return ctx.register_artifact(
        f"reliability/{name}.json", name=f"reliability-{name}", media_type="application/json"
    ).id


def run_reliability_profile(ctx: RunContext) -> None:
    req = ProfileRequest.from_parameters(ctx.parameters)
    spec = ProfileSpec.from_dict(req.spec)
    reg, now = ctx.registry, ctx.started_at
    built = builder.build(reg, ctx.store, spec)  # raises ProfileRefusal on incompatible sources
    doc: dict[str, Any] = built.document
    profile_art = _write(ctx, "profile", doc)
    _write(
        ctx,
        "sources",
        {
            "spec_id": spec.spec_id,
            "provenance_fingerprint": built.provenance_fingerprint,
            "references": [
                {"dimension": d.value, "kind": k.value, "id": i, "note": n}
                for d, k, i, n in built.references
            ],
        },
    )
    c = built.context
    profile = ReliabilityProfile(
        req.investigation_id,
        ctx.run.id,
        spec.spec_id,
        spec.to_dict(),
        spec.scope,
        c.model_fingerprint,
        c.dataset_fingerprint,
        c.split or "*",
        c.evaluation_config_hash,
        built.provenance_fingerprint,
        built.dimension_status,
        {
            "dimension_status": built.dimension_status,
            "sources": {
                "baseline_run": spec.baseline_run,
                "fault_experiments": len(spec.fault_experiments),
                "interactions": len(spec.interactions),
                "failure_modes": len(spec.failure_modes),
            },
            "no_score": doc["no_score"],
        },
        now,
    )
    new = not reg.exists(ReliabilityProfile, profile.id)
    if new:
        with reg.transaction():
            reg.add(profile)
            for d, k, i, n in built.references:
                ref = ReliabilityReference(profile.id, d, k, i, n, now)
                if not reg.exists(ReliabilityReference, ref.id):
                    reg.add(ref)
    ctx.observe("reliability.new_record", int(new), kind=EpistemicKind.OBSERVATION)
    for dim, st in built.dimension_status.items():
        ctx.observe(f"reliability.dimension.{dim.lower()}", st, kind=EpistemicKind.OBSERVATION)
    counts = _count(built.dimension_status)
    claim = ctx.assert_claim(
        f"[{ctx.run.id}] A reliability profile ({spec.scope.value}) was assembled from stored evidence: baseline run {spec.baseline_run}, {len(spec.fault_experiments)} fault experiment(s), {len(spec.interactions)} interaction analysis(es), {len(spec.failure_modes)} failure mode(s); dimension statuses {counts}. It lists observations with their sources and makes no assessment of overall reliability.",
        asserted_by=f"experionyx.reliability/{PROFILE_VERSION}", status=ClaimStatus.SUPPORTED,
    )  # fmt: skip
    ctx.add_evidence(
        claim, EvidenceTarget.RUN, spec.baseline_run, EvidenceRelation.CONTEXT, "baseline run"
    )
    ctx.add_evidence(
        claim, EvidenceTarget.ARTIFACT, profile_art, EvidenceRelation.SUPPORTS, "profile document"
    )


def _count(status: Mapping[str, str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in status.values():
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


@dataclass(frozen=True)
class ProfileRunResult:
    experiment_id: str
    run_id: str
    status: RunStatus
    profile_id: str | None


def run_profile(
    registry: Registry,
    store: ArtifactStore,
    executor: Executor,
    investigation_id: str,
    spec: ProfileSpec,
    *,
    seed: int = 0,
) -> ProfileRunResult:
    """Validate first (ProfileRefusal, nothing created), then build the profile as a Run."""
    ctx = builder.context_of(registry, store, spec)
    builder.check_compatibility(registry, store, spec, ctx)
    conf = ConfigurationRef(ProfileRequest(investigation_id, spec.to_dict()).to_parameters())
    if not registry.exists(ConfigurationRef, conf.id):
        registry.add(conf)
    exp = Experiment(
        investigation_id,
        "reliability profile",
        "Evidence-first reliability profile over stored results",
        ModelRef("none", "0"),
        DatasetRef("none", "0"),
        conf.id,
        datetime.now(UTC),
    )
    if not registry.exists(Experiment, exp.id):
        registry.add(exp)
        registry.update_status(exp.with_status(ExperimentStatus.READY))
    result = executor.execute(
        exp.id, resolve_procedure(PROCEDURE), seed=seed, procedure_name=PROCEDURE
    )
    found = [
        p
        for p in registry.find(ReliabilityProfile, spec_id=spec.spec_id)
        if p.investigation_id == investigation_id
    ]
    return ProfileRunResult(exp.id, result.run.id, result.status, found[0].id if found else None)


def replay_check(
    reg: Registry, store: ArtifactStore, executor: Executor, profile_id: str
) -> dict[str, object]:
    """Replay the profile Run as a NEW run and compare the whole profile document and its sources."""
    p = reg.get(ReliabilityProfile, profile_id)
    replay = executor.replay(p.run_id)
    out: dict[str, object] = {
        "profile_id": p.id,
        "original_run": p.run_id,
        "replay_run": replay.run.id,
        "replay_status": replay.status.value,
        "tolerance": REPLAY_TOLERANCE,
    }
    if replay.status is not RunStatus.COMPLETED:
        return {
            **out,
            "deterministic": False,
            "differences": [f"replay run ended {replay.status.value}"],
        }
    diffs: list[str] = []
    fingerprints: list[object] = []
    for name in ("profile", "sources"):
        try:
            x = read_artifact(reg, store, p.run_id, f"reliability/{name}.json")
        except ExperionyxError:
            continue
        y = read_artifact(reg, store, replay.run.id, f"reliability/{name}.json")
        if name == "sources" and isinstance(x, dict) and isinstance(y, dict):
            fingerprints = [x.get("provenance_fingerprint"), y.get("provenance_fingerprint")]
        _compare(name, x, y, diffs)
    if len(fingerprints) == 2 and fingerprints[0] != fingerprints[1]:
        # The evidence changed after the profile was built (e.g. a failure mode's lifecycle
        # state): the profile is a snapshot, so differing output is expected, not nondeterminism.
        return {
            **out,
            "deterministic": None,
            "sources_changed": True,
            "differences": diffs[:50],
            "note": "the included evidence changed since this profile was built (provenance fingerprint differs); determinism cannot be judged from this replay",
        }
    return {**out, "deterministic": not diffs, "sources_changed": False, "differences": diffs[:50]}


__all__ = [
    "PROCEDURE",
    "DimensionStatus",
    "ProfileRunResult",
    "replay_check",
    "run_profile",
    "run_reliability_profile",
]

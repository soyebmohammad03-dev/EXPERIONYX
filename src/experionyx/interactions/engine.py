"""The interaction-analysis procedure, run by the normal execution engine so that every analysis
is a Run with provenance, artifacts, observations, claims and evidence, and can be replayed."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

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
from experionyx.errors import ValidationError
from experionyx.evaluation.config import _thaw
from experionyx.execution import Executor, RunContext, resolve_procedure
from experionyx.failures.entities import FailureMode, FailureRelationship
from experionyx.failures.taxonomy import NodeKind, Predicate
from experionyx.hashing import content_hash
from experionyx.interactions import analysis as an
from experionyx.interactions import design as ds
from experionyx.interactions.config import INTERACTION_VERSION, InteractionSpec
from experionyx.interactions.entities import (
    InteractionAnalysis,
    InteractionEffect,
    InteractionEvidence,
)
from experionyx.interactions.modes import failure_modes
from experionyx.interactions.samples import per_sample
from experionyx.interactions.taxonomy import (
    EffectStatus,
    EvidenceKind,
    InteractionClass,
    InteractionStatus,
    Level,
    ModeState,
)
from experionyx.registry import Registry

PROCEDURE = "experionyx.interactions.engine:run_interaction_analysis"
ACTOR = "experionyx-interaction"
SUPPORTING = (InteractionClass.OBSERVED_INTERACTION, InteractionClass.ORDER_DEPENDENT_INTERACTION)
LANGUAGE = {
    "measures": "observed metric values from stored evaluation runs, per trial",
    "infers": "a contrast, direction-aware reading and a rule-based label under the configured thresholds",
    "human_may_conclude": "that, under this design, the combined treatment departed from the additive reference by the reported amount and interval; anything about mechanism or cause needs further, purpose-built experiments",
    "never_claimed": [
        "fault A causes fault B",
        "the faults interact mechanistically",
        "a fault is responsible for a failure",
        "the model is vulnerable because of a fault",
    ],
}


@dataclass(frozen=True)
class Result:
    design: ds.LoadedDesign
    effects: dict[str, object]
    samples: dict[str, object] | None
    modes: dict[str, object] | None
    provenance_fingerprint: str
    fault_a: str
    fault_b: str
    structural_key: str
    summary: dict[str, object]


def _fault_hash(desc: Mapping[str, object]) -> str:
    return content_hash(to_jsonable(desc))


def provenance_fingerprint(d: ds.LoadedDesign) -> str:
    """Changes if ANY scientifically meaningful input changes: the design (run IDs, cells,
    configuration), each run's own provenance fingerprint (fault, seed, model, dataset, source,
    environment), the trial seeds and the structural identity."""
    cells = {
        c: [
            {
                "run_id": t.run_id,
                "key": t.key,
                "seed": t.seed,
                "run_provenance": t.provenance_fingerprint,
            }
            for t in ts
        ]
        for c, ts in d.trials.items()
    }
    return content_hash(
        {
            "interaction_version": INTERACTION_VERSION,
            "spec": d.spec.to_dict(),
            "cells": cells,
            "structural": d.structural_key(),
        }
    )


def statement(effect: Mapping[str, object], n: Mapping[str, int]) -> str:
    """Deterministic template text from structured results (no LLM, no free text)."""
    derived, boot, interp = effect["derived"], effect["bootstrap"], effect["interpreted"]
    if not isinstance(derived, Mapping) or not isinstance(interp, Mapping):
        return f"No interaction contrast could be computed for {effect['name']} ({effect.get('undefined_reason')}). This is not evidence of the absence of an interaction."
    text = f"An interaction contrast of {derived['interaction_contrast']:.6g} was observed for {effect['name']} under the specified design (trials per cell: {dict(n)}; {derived['aggregation']} aggregation)."
    iv: Mapping[str, float] | None = None
    if isinstance(boot, Mapping) and boot.get("status") == "COMPUTED":
        ivs = boot["intervals"]
        assert isinstance(ivs, Mapping)  # noqa: S101
        iv = ivs.get("interaction_contrast")
    if iv and isinstance(boot, Mapping):
        excl = iv["lower"] > 0 or iv["upper"] < 0
        text += f" The percentile bootstrap interval [{iv['lower']:.6g}, {iv['upper']:.6g}] ({boot['confidence']:.0%}, {boot['resamples']} resamples, seed {boot['seed']}) {'excluded' if excl else 'included'} zero."
    else:
        text += " No interval was computed (insufficient data)."
    return (
        text
        + f" Label under the configured rules: {interp['class']}. This is an observed statistical contrast, not a causal or mechanistic claim."
    )


def build_result(registry: Registry, store: ArtifactStore, spec: InteractionSpec) -> Result:
    d = ds.validate(registry, store, spec)  # raises DesignRefusal before any number exists
    effects = an.analyze(d)
    ev_classes = None
    if d.task == "CLASSIFICATION":
        from experionyx.evaluation.loading import load_evaluation

        ev_classes = tuple(
            load_evaluation(registry, store, d.trials["CONTROL"][0].run_id).classes or ()
        )
    samples = per_sample(registry, store, d, ev_classes) if spec.config.per_sample else None
    modes = failure_modes(registry, store, d) if spec.discovery_run_id else None
    prov = provenance_fingerprint(d)
    fa, fb = _fault_hash(d.descriptor_a), _fault_hash(d.descriptor_b)
    listed = effects["effects"]
    assert isinstance(listed, list)  # noqa: S101
    primary = next(e for e in listed if e["name"] == effects["primary_metric"])
    n = {c: len(t) for c, t in d.trials.items()}
    summary: dict[str, object] = {
        "spec_id": spec.spec_id, "primary_metric": effects["primary_metric"], "primary_class": effects["primary_class"],
        "observed": {"primary_cell_values": primary["observed"]},
        "derived": primary["derived"], "interpreted": primary["interpreted"],
        "statement": statement(primary, n), "trials_per_cell": n, "multiplicity": effects["multiplicity"],
        "per_sample_status": None if samples is None else samples["status"],
        "failure_modes_status": None if modes is None else modes["status"],
        "warnings": list(d.warnings), "language": LANGUAGE, "interaction_version": INTERACTION_VERSION,
        "class_counts": _count(an.classes_of(effects)),
    }  # fmt: skip
    return Result(
        d,
        effects,
        samples,
        modes,
        prov,
        fa,
        fb,
        content_hash(to_jsonable(d.structural_key())),
        summary,
    )


def _count(classes: Mapping[str, str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in classes.values():
        out[c] = out.get(c, 0) + 1
    return dict(sorted(out.items()))


@dataclass(frozen=True)
class InteractionRequest:
    investigation_id: str
    spec: Mapping[str, object]

    def to_parameters(self) -> dict[str, object]:
        return {
            "interaction_analysis": {
                "investigation_id": self.investigation_id,
                "spec": dict(self.spec),
            }
        }

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, object]) -> "InteractionRequest":
        if set(parameters) != {"interaction_analysis"}:
            raise ValidationError(
                "an interaction configuration must be exactly {'interaction_analysis': ...}"
            )
        body = _thaw(parameters["interaction_analysis"])
        if (
            not isinstance(body, dict)
            or set(body) != {"investigation_id", "spec"}
            or not isinstance(body["spec"], dict)
            or not isinstance(body["investigation_id"], str)
        ):
            raise ValidationError("'interaction_analysis' has the wrong fields")
        return cls(body["investigation_id"], body["spec"])


def _write(ctx: RunContext, name: str, payload: object) -> str:
    directory = ctx.artifact_dir / "interaction"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(
        json.dumps(to_jsonable(payload), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return ctx.register_artifact(
        f"interaction/{name}.json", name=f"interaction-{name}", media_type="application/json"
    ).id


def run_interaction_analysis(ctx: RunContext) -> None:
    req = InteractionRequest.from_parameters(ctx.parameters)
    spec = InteractionSpec.from_dict(req.spec)
    reg, now = ctx.registry, ctx.started_at
    res = build_result(reg, ctx.store, spec)
    d = res.design
    art: dict[str, str] = {}
    art["spec"] = _write(
        ctx,
        "spec",
        {
            "spec_id": spec.spec_id,
            "spec": spec.to_dict(),
            "provenance_fingerprint": res.provenance_fingerprint,
            "structural_key": res.structural_key,
            "fault_a": to_jsonable(d.descriptor_a),
            "fault_b": to_jsonable(d.descriptor_b),
        },
    )
    art["design_validation"] = _write(
        ctx,
        "design_validation",
        {
            "valid": True,
            "checks": list(d.checks),
            "warnings": list(d.warnings),
            "excluded_runs": list(d.excluded),
            "pairing": d.pairing.value,
        },
    )
    art["trials"] = _write(
        ctx,
        "trials",
        {
            c: [
                {
                    "run_id": t.run_id,
                    "key": t.key,
                    "seed": t.seed,
                    "values": dict(t.values),
                    "undefined": dict(t.undefined),
                    "fault": to_jsonable(t.fault),
                    "provenance_fingerprint": t.provenance_fingerprint,
                }
                for t in ts
            ]
            for c, ts in d.trials.items()
        },
    )
    effects = res.effects["effects"]
    assert isinstance(effects, list)  # noqa: S101
    art["effects"] = _write(
        ctx,
        "effects",
        {
            "primary_metric": res.effects["primary_metric"],
            "multiplicity": res.effects["multiplicity"],
            "effects": effects,
        },
    )
    art["bootstrap"] = _write(ctx, "bootstrap", {e["name"]: e["bootstrap"] for e in effects})
    if res.samples is not None:
        art["per_sample"] = _write(ctx, "per_sample", res.samples)
    if res.modes is not None:
        art["failure_modes"] = _write(ctx, "failure_modes", res.modes)
    art["summary"] = _write(ctx, "summary", res.summary)

    primary_class = InteractionClass(str(res.effects["primary_class"]))
    analysis = InteractionAnalysis(
        req.investigation_id,
        ctx.run.id,
        spec.spec_id,
        spec.to_dict(),
        res.structural_key,
        res.fault_a,
        res.fault_b,
        d.model_fingerprint,
        d.dataset_fingerprint,
        str(res.effects["primary_metric"]),
        primary_class,
        res.provenance_fingerprint,
        {
            **{
                k: res.summary[k]
                for k in (
                    "statement",
                    "primary_class",
                    "trials_per_cell",
                    "class_counts",
                    "multiplicity",
                )
            },
            "descriptors": {
                "fault_a": to_jsonable(d.descriptor_a),
                "fault_b": to_jsonable(d.descriptor_b),
                "split": d.split,
                "evaluation_config_hash": d.evaluation_config_hash,
                "order_analysis": "BA" in d.trials,
                "aggregation": spec.config.aggregation.value,
                "normalization": spec.config.normalization.value,
            },
        },
        now,
    )
    new = not reg.exists(InteractionAnalysis, analysis.id)
    if new:
        persist(reg, analysis, effects, d, res, art, now)
    ctx.observe("interaction.new_record", int(new), kind=EpistemicKind.OBSERVATION)
    ctx.observe("interaction.hypotheses_tested", len(effects), kind=EpistemicKind.DERIVED_METRIC)
    derived = res.summary["derived"]
    if isinstance(derived, Mapping):
        ctx.observe(
            f"interaction.{res.effects['primary_metric']}.contrast",
            float(derived["interaction_contrast"]),
            kind=EpistemicKind.DERIVED_METRIC,
        )
    claim = ctx.assert_claim(
        f"[{ctx.run.id}] {res.summary['statement']}",
        asserted_by=f"experionyx.interactions/{INTERACTION_VERSION}",
        status=ClaimStatus.SUPPORTED
        if primary_class in SUPPORTING
        else ClaimStatus.INSUFFICIENT_EVIDENCE
        if primary_class is InteractionClass.INCONCLUSIVE
        else ClaimStatus.INCONCLUSIVE,
    )
    for cell, ts in d.trials.items():
        for t in ts:
            ctx.add_evidence(
                claim,
                EvidenceTarget.RUN,
                t.run_id,
                EvidenceRelation.CONTEXT if cell == "CONTROL" else EvidenceRelation.SUPPORTS,
                f"{cell} trial",
            )
    for name in ("effects", "summary"):
        ctx.add_evidence(
            claim, EvidenceTarget.ARTIFACT, art[name], EvidenceRelation.SUPPORTS, "underlying data"
        )


def persist(
    reg: Registry,
    a: InteractionAnalysis,
    effects: Sequence[Mapping[str, object]],
    d: ds.LoadedDesign,
    res: Result,
    art: Mapping[str, str],
    now: datetime,
) -> InteractionAnalysis:
    with reg.transaction():
        reg.add(a)
        for e in effects:
            interp = e["interpreted"]
            assert isinstance(interp, Mapping)  # noqa: S101
            reg.add(
                InteractionEffect(
                    a.id,
                    Level(str(e["level"])),
                    str(e["name"]),
                    EffectStatus(str(e["status"])),
                    InteractionClass(str(interp["class"])),
                    e,
                    now,
                )
            )
        reg.add(
            InteractionEvidence(
                a.id,
                EvidenceKind.DESIGN,
                None,
                "design validated before analysis",
                {
                    "checks": list(d.checks),
                    "warnings": list(d.warnings),
                    "excluded_runs": list(d.excluded),
                },
                now,
            )
        )
        reg.add(
            InteractionEvidence(
                a.id,
                EvidenceKind.OBSERVATION,
                None,
                "analysis artifacts and provenance",
                {"artifacts": dict(art), "provenance_fingerprint": res.provenance_fingerprint},
                now,
            )
        )
        _edges(reg, a, d, res, now)
        if a.primary_class in SUPPORTING:
            a = transition(
                reg,
                a,
                InteractionStatus.SUPPORTED,
                ACTOR,
                "primary-metric label meets the configured observation criteria",
                now,
                {"automatic": True, "rule": to_jsonable(_rule(effects, a.primary_metric))},
            )
    return a


def _rule(effects: Sequence[Mapping[str, object]], primary: str) -> object:
    e = next(x for x in effects if x["name"] == primary)
    interp = e["interpreted"]
    assert isinstance(interp, Mapping)  # noqa: S101
    return interp["rule"]


def _edge(
    reg: Registry,
    a: InteractionAnalysis,
    sk: NodeKind,
    sid: str,
    p: Predicate,
    ok: NodeKind,
    oid: str,
    detail: Mapping[str, object],
    now: datetime,
) -> None:
    e = FailureRelationship(a.investigation_id, sk, sid, p, ok, oid, detail, now)
    if not reg.exists(FailureRelationship, e.id):
        reg.add(e)


def _edges(
    reg: Registry, a: InteractionAnalysis, d: ds.LoadedDesign, res: Result, now: datetime
) -> None:
    for fault, label in ((a.fault_a, "A"), (a.fault_b, "B")):
        _edge(
            reg,
            a,
            NodeKind.FAULT,
            fault,
            Predicate.OBSERVED_WITH,
            NodeKind.INTERACTION,
            a.id,
            {"role": label, "note": "observed within one design; no direction implied"},
            now,
        )
    if a.primary_class is InteractionClass.ORDER_DEPENDENT_INTERACTION:
        _edge(
            reg,
            a,
            NodeKind.FAULT,
            a.fault_a,
            Predicate.ORDER_SENSITIVE_WITH,
            NodeKind.FAULT,
            a.fault_b,
            {"interaction_id": a.id},
            now,
        )
    for ts in d.trials.values():
        for t in ts:
            _edge(
                reg,
                a,
                NodeKind.INTERACTION,
                a.id,
                Predicate.SUPPORTED_BY,
                NodeKind.RUN,
                t.run_id,
                {},
                now,
            )
    if res.modes and res.modes.get("status") == "COMPUTED":
        rows = res.modes["modes"]
        assert isinstance(rows, list)  # noqa: S101
        for r in rows:
            if not reg.exists(FailureMode, r["mode_id"]):
                continue
            states = set(r["states"])
            if ModeState.OBSERVED_UNDER_COMPOUND_TREATMENT.value in states:
                _edge(
                    reg,
                    a,
                    NodeKind.FAILURE_MODE,
                    r["mode_id"],
                    Predicate.OBSERVED_WITH,
                    NodeKind.INTERACTION,
                    a.id,
                    {"states": sorted(states)},
                    now,
                )
            if ModeState.INSUFFICIENT_EVIDENCE.value in states:
                continue
            if states & {ModeState.NEWLY_OBSERVED.value, ModeState.INCREASED_PREVALENCE.value}:
                _edge(
                    reg,
                    a,
                    NodeKind.FAILURE_MODE,
                    r["mode_id"],
                    Predicate.AMPLIFIED_UNDER,
                    NodeKind.INTERACTION,
                    a.id,
                    {"states": sorted(states)},
                    now,
                )
            if ModeState.REDUCED_PREVALENCE.value in states:
                _edge(
                    reg,
                    a,
                    NodeKind.FAILURE_MODE,
                    r["mode_id"],
                    Predicate.SUPPRESSED_UNDER,
                    NodeKind.INTERACTION,
                    a.id,
                    {"states": sorted(states)},
                    now,
                )


def transition(
    reg: Registry,
    a: InteractionAnalysis,
    target: InteractionStatus,
    actor: str,
    reason: str,
    now: datetime,
    detail: Mapping[str, object] | None = None,
    refs: Sequence[str] = (),
) -> InteractionAnalysis:
    """Validated lifecycle change; previous state, new state, reason, actor, time and evidence
    references are retained as TRANSITION evidence."""
    if not actor.strip() or not reason.strip():
        raise ValidationError("a status change needs an actor and a reason")
    new = a.with_status(target)
    with reg.transaction():
        reg.update_status(new)
        reg.add(
            InteractionEvidence(
                a.id,
                EvidenceKind.TRANSITION,
                None,
                f"{a.status.value} -> {target.value}",
                {
                    "from": a.status.value,
                    "to": target.value,
                    "actor": actor,
                    "reason": reason,
                    "at": now.isoformat(),
                    "evidence_refs": list(refs),
                    **(detail or {}),
                },
                now,
            )
        )
    return new


@dataclass(frozen=True)
class InteractionRunResult:
    experiment_id: str
    run_id: str
    status: RunStatus
    analysis_id: str | None


def run_interaction(
    registry: Registry,
    store: ArtifactStore,
    executor: Executor,
    investigation_id: str,
    spec: InteractionSpec,
    *,
    seed: int = 0,
) -> InteractionRunResult:
    """Validate first (DesignRefusal, nothing created), then run the analysis as a Run."""
    ds.validate(registry, store, spec)
    conf = ConfigurationRef(InteractionRequest(investigation_id, spec.to_dict()).to_parameters())
    if not registry.exists(ConfigurationRef, conf.id):
        registry.add(conf)
    exp = Experiment(
        investigation_id,
        "interaction analysis",
        "Deterministic fault interaction analysis over stored runs",
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
    found = registry.find(InteractionAnalysis, spec_id=spec.spec_id)
    mine = [x for x in found if x.investigation_id == investigation_id]
    return InteractionRunResult(exp.id, result.run.id, result.status, mine[0].id if mine else None)

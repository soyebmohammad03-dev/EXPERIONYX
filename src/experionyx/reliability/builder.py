"""Profile construction from PERSISTED evidence. Nothing is recomputed that an earlier phase
already stored: baseline metrics come from the stored evaluation, fault responses from the stored
fault analysis, modes from the failure registry, interactions from the interaction registry.
Incompatible sources are refused. Every observation carries its source reference; there is no
score, ranking or verdict."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from experionyx.artifacts import ArtifactStore
from experionyx.domain import Artifact, Run, RunStatus, to_jsonable
from experionyx.errors import ExperionyxError, ProfileRefusal
from experionyx.evaluation.loading import load_evaluation
from experionyx.evaluation.results import EvaluationResult, Status
from experionyx.failures.entities import FailureCluster, FailureEvidence, FailureMode, FailureSignal
from experionyx.failures.extraction import label_key
from experionyx.faults.entities import FaultExperiment, FaultTrial, TrialStatus
from experionyx.faults.report import load_analysis
from experionyx.hashing import content_hash
from experionyx.interactions.design import DesignIssue
from experionyx.interactions.entities import InteractionAnalysis, InteractionEffect
from experionyx.interactions.taxonomy import InteractionStatus
from experionyx.provenance import Provenance
from experionyx.registry import Registry
from experionyx.reliability.spec import PROFILE_VERSION, ProfileSpec
from experionyx.reliability.taxonomy import Dimension, DimensionStatus, RefKind, Scope
from experionyx.slices.entities import SliceAnalysis

FAULT_PATH = "fault/fault.json"
NEVER_CLAIMED = [
    "the model is robust",
    "the model is unreliable",
    "the model is safer than another",
    "the evidence establishes general-world robustness",
    "a fault causes a failure",
]


@dataclass(frozen=True)
class Context:
    model_fingerprint: str
    dataset_fingerprint: str
    split: str | None
    evaluation_config_hash: str
    task: str
    baseline: EvaluationResult


@dataclass(frozen=True)
class Built:
    context: Context
    document: dict[str, Any]
    references: tuple[tuple[Dimension, RefKind, str, str], ...]
    provenance_fingerprint: str
    dimension_status: dict[str, str]


def g(x: object) -> str:
    return "n/a" if x is None else f"{x:.6g}" if isinstance(x, int | float) else str(x)


def _issue(code: str, required: str, found: str, why: str, cell: str | None = None) -> DesignIssue:
    return DesignIssue(code, required, found, why, cell)


def context_of(registry: Registry, store: ArtifactStore, spec: ProfileSpec) -> Context:
    """The profile's anchor: a COMPLETED baseline evaluation run that carries no fault."""
    try:
        run = registry.get(Run, spec.baseline_run)
    except ExperionyxError:
        raise ProfileRefusal(
            (
                _issue(
                    "BASELINE_MISSING",
                    "a registered baseline run",
                    spec.baseline_run,
                    "the profile has no anchor",
                ),
            )
        ) from None
    if run.status is not RunStatus.COMPLETED:
        raise ProfileRefusal(
            (
                _issue(
                    "BASELINE_NOT_COMPLETED",
                    "a COMPLETED run",
                    run.status.value,
                    "an unfinished run has no reliable results",
                ),
            )
        )
    if any(a.path == FAULT_PATH for a in registry.find(Artifact, run_id=run.id)):
        raise ProfileRefusal(
            (
                _issue(
                    "BASELINE_IS_FAULTED",
                    "a fault-free baseline evaluation",
                    f"{run.id} carries a fault artifact",
                    "a faulted run is a treatment, not a baseline",
                ),
            )
        )
    try:
        ev = load_evaluation(registry, store, run.id)
    except ExperionyxError as exc:
        raise ProfileRefusal(
            (
                _issue(
                    "BASELINE_UNUSABLE",
                    "a digest-verified stored evaluation",
                    str(exc),
                    "the baseline results cannot be read",
                ),
            )
        ) from None
    return Context(
        str(ev.context.model_fingerprint),
        str(ev.context.dataset_fingerprint),
        ev.context.split,
        content_hash(to_jsonable(ev.config)),
        ev.task.value,
        ev,
    )


def _matches(
    scope: Scope, ctx: Context, model: object, dataset: object, split: object, cfg_hash: object
) -> list[tuple[str, object, object]]:
    bad: list[tuple[str, object, object]] = []
    for label, want, got in (
        ("model fingerprint", ctx.model_fingerprint, model),
        ("dataset fingerprint", ctx.dataset_fingerprint, dataset),
    ):
        if want != got:
            bad.append((label, want, got))
    if scope in (Scope.MODEL_DATASET_SPLIT, Scope.MODEL_DATASET_EVALUATION) and ctx.split != split:
        bad.append(("split", ctx.split, split))
    if scope is Scope.MODEL_DATASET_EVALUATION and ctx.evaluation_config_hash != cfg_hash:
        bad.append(("evaluation configuration", ctx.evaluation_config_hash, cfg_hash))
    return bad


def check_compatibility(
    registry: Registry, store: ArtifactStore, spec: ProfileSpec, ctx: Context
) -> None:
    """Refuse (listing every issue) if any source belongs to a different model, dataset, or, per
    scope, split / evaluation configuration."""
    issues: list[DesignIssue] = []
    scope = spec.scope
    for fid in spec.fault_experiments:
        try:
            fx = registry.get(FaultExperiment, fid)
            ev = load_evaluation(registry, store, fx.baseline_run_id)
            load_analysis(registry, store, fid)
        except ExperionyxError as exc:
            issues.append(
                _issue(
                    "FAULT_EXPERIMENT_UNUSABLE",
                    "a fault experiment with a stored baseline evaluation and analysis",
                    f"{fid}: {exc}",
                    "its results cannot be read",
                    fid,
                )
            )
            continue
        for label, want, got in _matches(
            scope,
            ctx,
            ev.context.model_fingerprint,
            ev.context.dataset_fingerprint,
            ev.context.split,
            content_hash(to_jsonable(ev.config)),
        ):
            issues.append(
                _issue(
                    "INCOMPATIBLE_FAULT_EXPERIMENT",
                    f"{label} {want}",
                    f"{got}",
                    f"the experiment's baseline differs in {label}; sources of one profile must match at scope {scope.value}",
                    fid,
                )
            )
    for iid in spec.interactions:
        try:
            a = registry.get(InteractionAnalysis, iid)
        except ExperionyxError:
            issues.append(
                _issue(
                    "INTERACTION_MISSING",
                    "a registered interaction analysis",
                    iid,
                    "it cannot be summarized",
                    iid,
                )
            )
            continue
        d: Any = a.summary.get("descriptors", {})
        for label, want, got in _matches(
            scope,
            ctx,
            a.model_fingerprint,
            a.dataset_fingerprint,
            d.get("split"),
            d.get("evaluation_config_hash"),
        ):
            issues.append(
                _issue(
                    "INCOMPATIBLE_INTERACTION",
                    f"{label} {want}",
                    f"{got}",
                    f"the interaction was analyzed under a different {label}",
                    iid,
                )
            )
    for mid in spec.failure_modes:
        try:
            mode = registry.get(FailureMode, mid)
            sigs = [
                registry.get(FailureSignal, s)
                for s in registry.get(FailureCluster, mode.cluster_id).signal_ids
            ]
        except ExperionyxError:
            issues.append(
                _issue(
                    "FAILURE_MODE_MISSING",
                    "a registered failure mode",
                    mid,
                    "it cannot be summarized",
                    mid,
                )
            )
            continue
        for label, key, want in (
            ("model fingerprint", "model_fingerprint", ctx.model_fingerprint),
            ("dataset fingerprint", "dataset_fingerprint", ctx.dataset_fingerprint),
        ):
            got = sorted({str(s.detail.get(key)) for s in sigs} - {want})
            if got:
                issues.append(
                    _issue(
                        "INCOMPATIBLE_FAILURE_MODE",
                        f"{label} {want}",
                        f"signals with {got}",
                        "the mode is built from a different model or dataset",
                        mid,
                    )
                )
    for sid in spec.slice_analyses:
        try:
            sa = registry.get(SliceAnalysis, sid)
        except ExperionyxError:
            issues.append(
                _issue(
                    "SLICE_ANALYSIS_MISSING",
                    "a registered slice analysis",
                    sid,
                    "it cannot be summarized",
                    sid,
                )
            )
            continue
        for label, want, got in (
            ("baseline run", spec.baseline_run, sa.baseline_run_id),
            ("dataset fingerprint", ctx.dataset_fingerprint, sa.dataset_fingerprint),
        ):
            if want != got:
                issues.append(_issue("INCOMPATIBLE_SLICE_ANALYSIS", f"{label} {want}", f"{got}", f"the slice analysis was made over a different {label}", sid))  # fmt: skip
    if issues:
        raise ProfileRefusal(tuple(issues))


# -- sections ------------------------------------------------------------------------------------------


def _artifact_ids(registry: Registry, run_id: str) -> dict[str, str]:
    return {a.path: a.id for a in registry.find(Artifact, run_id=run_id)}


@dataclass
class _Acc:
    """Collects dimensions, references and interpreted statements while sections are built."""

    dims: dict[str, dict[str, Any]] = field(default_factory=dict)
    refs: list[tuple[Dimension, RefKind, str, str]] = field(default_factory=list)
    statements: list[dict[str, Any]] = field(default_factory=list)
    intervals: list[dict[str, Any]] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    def dim(
        self,
        d: Dimension,
        status: DimensionStatus,
        observations: object,
        reason: str | None = None,
        **extra: Any,
    ) -> None:
        self.dims[d.value] = {
            "status": status.value,
            "reason": reason,
            "observations": observations,
            **extra,
        }

    def ref(self, d: Dimension, kind: RefKind, rid: str, note: str) -> None:
        self.refs.append((d, kind, rid, note))

    def say(self, text: str, basis: Sequence[dict[str, str]]) -> None:
        self.statements.append({"label": "INTERPRETED", "text": text, "basis": list(basis)})


def _baseline(registry: Registry, spec: ProfileSpec, ctx: Context, acc: _Acc) -> None:
    ev = ctx.baseline
    src: dict[str, Any] = {
        "kind": "RUN",
        "id": spec.baseline_run,
        "artifacts": _artifact_ids(registry, spec.baseline_run),
    }
    metrics: list[dict[str, Any]] = []
    for m in ev.metrics:
        if m.structured is not None:
            continue
        metrics.append(
            {
                "metric_id": m.metric_id,
                "value": m.value,
                "status": m.status.value,
                "higher_is_better": m.higher_is_better,
                "n_samples": m.n_samples,
                "interval": to_jsonable(m.interval),
                "reason": m.reason,
            }
        )
        if m.interval is not None and m.interval.status is Status.COMPUTED:
            acc.intervals.append(
                {
                    "source": "baseline",
                    "measure": m.metric_id,
                    "method": m.interval.method,
                    "confidence": m.interval.confidence,
                    "lower": m.interval.lower,
                    "upper": m.interval.upper,
                    "n": m.interval.n_samples,
                    "resamples": m.interval.resamples,
                }
            )
    acc.dim(
        Dimension.BASELINE_PERFORMANCE,
        DimensionStatus.OBSERVED,
        [
            {
                "source": src,
                "n_samples": ev.n_samples,
                "task": ev.task.value,
                "split": ev.context.split,
                "metrics": metrics,
                "error_counts": dict(ev.errors.counts),
                "evaluation_findings": [f.finding_id for f in ev.findings],
                "warnings": list(ev.warnings),
            }
        ],
    )
    acc.ref(Dimension.BASELINE_PERFORMANCE, RefKind.RUN, spec.baseline_run, "baseline evaluation")
    for path, aid in sorted(src["artifacts"].items()):
        if path.startswith("evaluation/"):
            acc.ref(Dimension.BASELINE_PERFORMANCE, RefKind.ARTIFACT, aid, path)
    primary = "accuracy" if ctx.task == "CLASSIFICATION" else "rmse"
    pm = next((m for m in metrics if m["metric_id"] == primary), None)
    if pm and pm["value"] is not None:
        iv = pm["interval"]
        tail = (
            f" (interval [{g(iv['lower'])}, {g(iv['upper'])}], {iv['method']}, {iv['confidence']:.0%})"
            if iv and iv.get("status") == "COMPUTED"
            else ""
        )
        acc.say(
            f"On {ev.n_samples} samples of split {ev.context.split!r}, {primary} was {g(pm['value'])}{tail}.",
            [{"kind": "RUN", "id": spec.baseline_run}],
        )
    cal = ev.calibration
    if cal.status is Status.COMPUTED:
        acc.dim(
            Dimension.CALIBRATION,
            DimensionStatus.OBSERVED,
            [
                {
                    "source": src,
                    "ece": cal.ece,
                    "mce": cal.mce,
                    "brier_score": cal.brier_score,
                    "n_bins": cal.n_bins,
                    "n_samples": cal.n_samples,
                    "confidence_source": cal.confidence_source,
                    "binning": cal.binning,
                    "interpretation": cal.interpretation,
                }
            ],
        )
        acc.ref(Dimension.CALIBRATION, RefKind.RUN, spec.baseline_run, "baseline calibration")
    else:
        acc.dim(
            Dimension.CALIBRATION,
            DimensionStatus.UNAVAILABLE,
            [],
            reason=f"{cal.status.value}: {cal.reason}",
        )
    lat = ev.latency
    if lat.status is Status.COMPUTED:
        acc.dim(
            Dimension.LATENCY,
            DimensionStatus.OBSERVED,
            [
                {
                    "source": src,
                    "median_batch_seconds": lat.median_batch_seconds,
                    "mean_batch_seconds": lat.mean_batch_seconds,
                    "throughput_samples_per_second": lat.throughput_samples_per_second,
                    "n_batches": lat.n_batches,
                    "load_seconds": lat.load_seconds,
                    "note": "timings are noisy and hardware dependent",
                }
            ],
        )
        acc.ref(Dimension.LATENCY, RefKind.RUN, spec.baseline_run, "baseline latency")
    else:
        acc.dim(
            Dimension.LATENCY,
            DimensionStatus.UNAVAILABLE,
            [],
            reason=f"{lat.status.value}: {lat.reason}",
        )


def _faults(
    registry: Registry, store: ArtifactStore, spec: ProfileSpec, acc: _Acc
) -> dict[str, dict[str, Any]]:
    if not spec.fault_experiments:
        acc.dim(
            Dimension.FAULT_SENSITIVITY,
            DimensionStatus.UNAVAILABLE,
            [],
            reason="no fault experiment was included in the profile",
        )
        return {}
    obs: list[dict[str, Any]] = []
    info: dict[str, dict[str, Any]] = {}
    for fid in spec.fault_experiments:
        fx = registry.get(FaultExperiment, fid)
        res = load_analysis(registry, store, fid)
        trials = registry.find(FaultTrial, fault_experiment_id=fid)
        fault: Any = fx.design.get("fault", {})
        pts = []
        for p in res.points:
            pts.append(
                {
                    "point": p.point_index,
                    "parameter": p.parameter_name,
                    "value": p.parameter_value,
                    "trials": p.n_trials,
                    "completed": p.n_completed,
                    "faulted": to_jsonable(p.primary_faulted),
                    "deterioration": to_jsonable(p.primary_deterioration),
                    "effect_classification": p.assessment.classification.value,
                    "assessment": to_jsonable(p.assessment),
                    "severity": to_jsonable(p.severity),
                }
            )
            det = p.primary_deterioration
            if det.ci_lower is not None:
                acc.intervals.append(
                    {
                        "source": f"fault {fid}",
                        "measure": f"{res.primary_metric} deterioration",
                        "method": "bootstrap of the mean over seeds",
                        "confidence": det.confidence,
                        "lower": det.ci_lower,
                        "upper": det.ci_upper,
                        "n": det.n,
                        "resamples": det.resamples,
                    }
                )
            where = f" ({p.parameter_name}={g(p.parameter_value)})" if p.parameter_name else ""
            acc.say(
                f"Under fault {fault.get('type')}{where}, {res.primary_metric} was {g(p.primary_faulted.mean)} (baseline {g(res.baseline_value)}); mean deterioration {g(det.mean)}"
                + (
                    f" [{g(det.ci_lower)}, {g(det.ci_upper)}]"
                    if det.ci_lower is not None
                    else " (no interval)"
                )
                + f" over {p.n_completed} completed trial(s); diagnostic effect classification {p.assessment.classification.value}.",
                [{"kind": "FAULT_EXPERIMENT", "id": fid}],
            )
        seeds = sorted({t.seed for t in trials})
        obs.append(
            {
                "source": {
                    "kind": "FAULT_EXPERIMENT",
                    "id": fid,
                    "baseline_run_id": fx.baseline_run_id,
                },
                "fault": {
                    "type": fault.get("type"),
                    "version": fault.get("version"),
                    "parameters": fault.get("parameters"),
                    "scope": fault.get("scope"),
                    "components": [c.get("type") for c in fault.get("components", [])],
                },
                "primary_metric": res.primary_metric,
                "direction": res.primary_direction.value,
                "baseline_value": res.baseline_value,
                "points": pts,
                "trials": {
                    "total": len(trials),
                    "completed": sum(t.status is TrialStatus.COMPLETED for t in trials),
                    "failed": res.failed_trials,
                    "skipped": res.skipped_trials,
                    "seeds": seeds,
                },
                "warnings": list(res.warnings),
            }
        )
        acc.ref(
            Dimension.FAULT_SENSITIVITY, RefKind.FAULT_EXPERIMENT, fid, f"fault {fault.get('type')}"
        )
        info[fid] = obs[-1]
    ok = any(p["completed"] > 0 for o in obs for p in o["points"])
    acc.dim(
        Dimension.FAULT_SENSITIVITY,
        DimensionStatus.OBSERVED if ok else DimensionStatus.INSUFFICIENT_EVIDENCE,
        obs,
        reason=None if ok else "no fault experiment has a completed trial",
        note="fault responses are listed, never ranked",
    )
    return info


def _has_passing_reproduction(registry: Registry, mode_id: str) -> bool:
    return any(
        e.detail.get("summary") is True and e.detail.get("passed") is True
        for e in registry.find(
            FailureEvidence, failure_mode_id=mode_id, evidence_kind="REPRODUCTION"
        )
    )


def _modes(registry: Registry, spec: ProfileSpec, acc: _Acc) -> list[dict[str, Any]]:
    if not spec.failure_modes:
        for d in (Dimension.FAILURE_PREVALENCE, Dimension.FAILURE_SEVERITY):
            acc.dim(
                d,
                DimensionStatus.UNAVAILABLE,
                [],
                reason="no failure mode was included in the profile",
            )
        return []
    rows: list[dict[str, Any]] = []
    for mid in spec.failure_modes:
        m = registry.get(FailureMode, mid)
        meas: Any = to_jsonable(m.structured["measurements"])
        rep = _has_passing_reproduction(registry, mid)
        rows.append(
            {
                "source": {"kind": "FAILURE_MODE", "id": mid, "cluster_id": m.cluster_id},
                "lifecycle_state": m.status.value,
                "established": m.status.value == "CONFIRMED",
                "category": m.category.value,
                "title": m.title,
                "prevalence": meas["prevalence"],
                "impact": meas["impact"],
                "severity": meas["severity"],
                "classes": meas["classes"],
                "predicted_classes": meas["predicted_classes"],
                "slices": meas["slices"],
                "fault_types": meas["fault_types"],
                "supporting_runs": meas["runs"],
                "supporting_experiments": meas["experiments"],
                "seeds": meas["seeds"],
                "signal_count": meas["size"],
                "reproduction_check_passed": rep,
                "reproducibility": "reproduction check passed"
                if rep
                else "no passing reproduction check recorded",
            }
        )
        acc.ref(Dimension.FAILURE_PREVALENCE, RefKind.FAILURE_MODE, mid, m.status.value)
        acc.ref(Dimension.FAILURE_SEVERITY, RefKind.FAILURE_MODE, mid, m.status.value)
        pv = meas["prevalence"]
        acc.say(
            f"Failure mode {mid} ({m.category.value}) has lifecycle state {m.status.value}; {pv['runs_with_signal']} of {pv['runs_analyzed']} analyzed run(s) contained one of its {meas['size']} signal(s); "
            + (
                "a reproduction check passed. "
                if rep
                else "no passing reproduction check is recorded. "
            )
            + (
                ""
                if m.status.value == "CONFIRMED"
                else "A mode that is not CONFIRMED is a grouping of signals, not an established finding."
            ),
            [{"kind": "FAILURE_MODE", "id": mid}],
        )
    acc.dim(
        Dimension.FAILURE_PREVALENCE,
        DimensionStatus.DERIVED,
        [
            {
                "source": r["source"],
                "lifecycle_state": r["lifecycle_state"],
                "category": r["category"],
                "established": r["established"],
                "prevalence": r["prevalence"],
                "classes": r["classes"],
                "slices": r["slices"],
                "fault_types": r["fault_types"],
                "supporting_experiments": r["supporting_experiments"],
            }
            for r in rows
        ],
        note="derived by failure discovery; prevalence denominators are the runs that discovery analyzed",
    )
    acc.dim(
        Dimension.FAILURE_SEVERITY,
        DimensionStatus.DERIVED,
        [
            {
                "source": r["source"],
                "lifecycle_state": r["lifecycle_state"],
                "severity": r["severity"],
                "impact": r["impact"],
                "reproducibility": r["reproducibility"],
            }
            for r in rows
        ],
        note="severity is a vector; no single severity score exists",
    )
    return rows


def _interactions(registry: Registry, spec: ProfileSpec, acc: _Acc) -> list[dict[str, Any]]:
    if not spec.interactions:
        acc.dim(
            Dimension.INTERACTION_SENSITIVITY,
            DimensionStatus.UNAVAILABLE,
            [],
            reason="no interaction analysis was included in the profile",
        )
        return []
    rows: list[dict[str, Any]] = []
    for iid in spec.interactions:
        a = registry.get(InteractionAnalysis, iid)
        (eff,) = registry.find(InteractionEffect, analysis_id=iid, measure=a.primary_metric)
        rec: Any = to_jsonable(eff.record)
        d = rec.get("derived") or {}
        b = rec.get("bootstrap") or {}
        iv = (b.get("intervals") or {}).get("interaction_contrast")
        desc: Any = to_jsonable(a.summary["descriptors"])
        rep = any(
            e.detail.get("passed") is True
            for e in registry.find(
                __import__(
                    "experionyx.interactions.entities", fromlist=["InteractionEvidence"]
                ).InteractionEvidence,
                analysis_id=iid,
                evidence_kind="REPRODUCTION",
            )
        )
        rows.append(
            {
                "source": {"kind": "INTERACTION", "id": iid, "run_id": a.run_id},
                "lifecycle_state": a.status.value,
                "components": [desc["fault_a"]["type"], desc["fault_b"]["type"]],
                "fault_a": desc["fault_a"],
                "fault_b": desc["fault_b"],
                "metric": a.primary_metric,
                "label": a.primary_class.value,
                "interaction_contrast": d.get("interaction_contrast"),
                "normalized_contrast": d.get("normalized_contrast"),
                "effect_a": d.get("effect_a"),
                "effect_b": d.get("effect_b"),
                "combined_effect": d.get("combined_effect"),
                "interval": iv,
                "bootstrap": {
                    k: b.get(k)
                    for k in (
                        "status",
                        "method",
                        "resamples",
                        "seed",
                        "confidence",
                        "n_trials",
                        "pairing",
                        "warnings",
                    )
                },
                "order_effect": d.get("order_effect"),
                "order_dependent": a.primary_class.value == "ORDER_DEPENDENT_INTERACTION",
                "trials_per_cell": to_jsonable(a.summary["trials_per_cell"]),
                "reproduction_passed": rep,
                "structural_key": a.structural_key,
            }
        )
        acc.ref(Dimension.INTERACTION_SENSITIVITY, RefKind.INTERACTION, iid, a.status.value)
        if iv:
            acc.intervals.append(
                {
                    "source": f"interaction {iid}",
                    "measure": f"{a.primary_metric} interaction contrast",
                    "method": b.get("method"),
                    "confidence": b.get("confidence"),
                    "lower": iv["lower"],
                    "upper": iv["upper"],
                    "n": min((b.get("n_trials") or {"x": 0}).values()),
                    "resamples": b.get("resamples"),
                }
            )
        for w in b.get("warnings") or []:
            if w not in acc.caveats:
                acc.caveats.append(w)
        acc.say(
            f"An interaction contrast of {g(d.get('interaction_contrast'))} was observed for {a.primary_metric} under design {a.spec_id}"
            + (f" (interval [{g(iv['lower'])}, {g(iv['upper'])}])" if iv else " (no interval)")
            + f"; label {a.primary_class.value}; lifecycle state {a.status.value}. This is an observed contrast, not a causal claim.",
            [{"kind": "INTERACTION", "id": iid}],
        )
    acc.dim(
        Dimension.INTERACTION_SENSITIVITY,
        DimensionStatus.DERIVED,
        rows,
        note="contrasts under stated designs; lifecycle states are shown, none is a causal claim",
    )
    return rows


def _slices(registry: Registry, spec: ProfileSpec, ctx: Context, acc: _Acc) -> None:
    ev = ctx.baseline
    src: dict[str, Any] = {"kind": "RUN", "id": spec.baseline_run}
    slices = [
        {
            "name": s.name,
            "conditions": to_jsonable(s.conditions),
            "n_samples": s.n_samples,
            "fraction_of_samples": s.fraction_of_samples,
            "deltas_vs_overall": to_jsonable(s.deltas_vs_overall),
        }
        for s in ev.slices
    ]
    classes = [
        {
            "class": label_key(c.label),
            "support": c.support,
            "predicted": c.predicted,
            "precision": c.precision,
            "recall": c.recall,
        }
        for c in (ev.confusion.per_class if ev.confusion is not None else ())
    ]
    class_effects = []
    for iid in spec.interactions:
        for e in registry.find(InteractionEffect, analysis_id=iid):
            if e.level.value in ("CLASS", "SLICE"):
                d: Any = to_jsonable(e.record.get("derived"))
                class_effects.append(
                    {
                        "source": {"kind": "INTERACTION", "id": iid},
                        "level": e.level.value,
                        "measure": e.measure,
                        "label": e.interaction_class.value,
                        "interaction_contrast": (d or {}).get("interaction_contrast"),
                    }
                )
    san_rows: list[dict[str, Any]] = []
    for sid in spec.slice_analyses:
        sa = registry.get(SliceAnalysis, sid)
        san_rows.append({"source": {"kind": "SLICE_ANALYSIS", "id": sid, "spec_id": sa.spec_id, "provenance_fingerprint": sa.provenance_fingerprint}, "analysis_status": sa.analysis_status, "measure": to_jsonable(sa.summary.get("measure")), "slices": to_jsonable(sa.summary.get("slices")), "fault_status_counts": to_jsonable(sa.summary.get("fault_status_counts")), "failure_mode_status_counts": to_jsonable(sa.summary.get("failure_mode_status_counts")), "interaction_status_counts": to_jsonable(sa.summary.get("interaction_status_counts")), "note": "slice statuses are copied as recorded; a slice without enough evidence stays INSUFFICIENT_EVIDENCE or UNAVAILABLE"})  # fmt: skip
        acc.ref(Dimension.SLICE_SENSITIVITY, RefKind.SLICE_ANALYSIS, sid, sa.analysis_status)
    if not (slices or classes or class_effects or san_rows):
        acc.dim(
            Dimension.SLICE_SENSITIVITY,
            DimensionStatus.UNAVAILABLE,
            [],
            reason="the evaluation configured no slices and has no class breakdown",
        )
        return
    acc.dim(
        Dimension.SLICE_SENSITIVITY,
        DimensionStatus.OBSERVED,
        [
            {
                "source": src,
                "slices": slices
                if slices
                else {"status": "UNAVAILABLE", "reason": "the evaluation configured no slices"},
                "classes": classes
                if classes
                else {
                    "status": "UNAVAILABLE",
                    "reason": "no class breakdown (regression or none stored)",
                },
                "slice_analyses": san_rows
                if san_rows
                else {
                    "status": "UNAVAILABLE",
                    "reason": "no slice analysis was referenced by the profile",
                },
                "interaction_class_or_slice_effects": class_effects
                if class_effects
                else {
                    "status": "UNAVAILABLE",
                    "reason": "no class- or slice-level interaction effects included",
                },
            }
        ],
        note="existing slice/class representations are reused; nothing is assumed uniform",
    )
    acc.ref(Dimension.SLICE_SENSITIVITY, RefKind.RUN, spec.baseline_run, "baseline slices/classes")


def _reproducibility(
    registry: Registry,
    spec: ProfileSpec,
    faults: Mapping[str, Any],
    modes: Sequence[Any],
    inters: Sequence[Any],
    acc: _Acc,
    runs: Sequence[str],
) -> None:
    if not (faults or modes or inters):
        acc.dim(
            Dimension.REPRODUCIBILITY,
            DimensionStatus.INSUFFICIENT_EVIDENCE,
            [],
            reason="no fault experiment, failure mode or interaction to assess",
        )
        return
    envs, unresolved = set(), []
    for r in runs:
        p = registry.find(Provenance, run_id=r)
        if p:
            envs.add(p[0].environment_id)
    fault_rows = [
        {
            "fault_experiment": fid,
            "completed_trials": o["trials"]["completed"],
            "distinct_seeds": len(o["trials"]["seeds"]),
            "failed_trials": o["trials"]["failed"],
            "skipped_trials": o["trials"]["skipped"],
        }
        for fid, o in faults.items()
    ]
    for fr in fault_rows:
        if fr["distinct_seeds"] < 3:
            unresolved.append(
                f"fault experiment {fr['fault_experiment']} has {fr['distinct_seeds']} distinct seed(s); its aggregate rests on few repeats"
            )
        if fr["failed_trials"] or fr["skipped_trials"]:
            unresolved.append(
                f"fault experiment {fr['fault_experiment']} has {fr['failed_trials']} failed and {fr['skipped_trials']} skipped trial(s)"
            )
    for m in modes:
        if not m["reproduction_check_passed"]:
            unresolved.append(
                f"failure mode {m['source']['id']} ({m['lifecycle_state']}) has no passing reproduction check"
            )
    for i in inters:
        if i["lifecycle_state"] not in (
            InteractionStatus.REPRODUCIBLE.value,
            InteractionStatus.CONFIRMED_BY_REVIEW.value,
        ):
            unresolved.append(
                f"interaction {i['source']['id']} is {i['lifecycle_state']} (not independently reproduced)"
            )
    if len(envs) > 1:
        unresolved.append(f"sources ran in {len(envs)} different environments")
    acc.dim(
        Dimension.REPRODUCIBILITY,
        DimensionStatus.DERIVED,
        [
            {
                "fault_experiments": fault_rows,
                "failure_modes": {
                    "total": len(modes),
                    "with_passing_reproduction": sum(
                        bool(m["reproduction_check_passed"]) for m in modes
                    ),
                    "confirmed": sum(bool(m["established"]) for m in modes),
                },
                "interactions": {
                    "total": len(inters),
                    "by_lifecycle_state": _count(i["lifecycle_state"] for i in inters),
                    "with_passing_reproduction": sum(
                        bool(i["reproduction_passed"]) for i in inters
                    ),
                },
                "replay": {
                    "status": "NOT_RUN",
                    "note": "profile construction does not replay runs; use `reliability replay` to replay the profile itself",
                },
                "provenance_consistency": {
                    "distinct_environments": len(envs),
                    "distinct_model_fingerprints": 1,
                    "distinct_dataset_fingerprints": 1,
                    "note": "model/dataset agreement is enforced by the compatibility check",
                },
                "unresolved_issues": unresolved,
            }
        ],
        note="counts and statuses only; there is no reliability percentage",
    )


def _count(items: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items()))


def _uncertainty(acc: _Acc) -> None:
    if not acc.intervals:
        acc.dim(
            Dimension.UNCERTAINTY,
            DimensionStatus.INSUFFICIENT_EVIDENCE,
            [],
            reason="no stored interval exists in the included evidence",
        )
        return
    caveats = [
        "intervals are descriptive spreads under each analysis's own method and are not comparable across methods",
        *acc.caveats,
    ]
    acc.dim(
        Dimension.UNCERTAINTY,
        DimensionStatus.DERIVED,
        [{"intervals": acc.intervals, "caveats": caveats}],
        note="collected from the sources; not recomputed and not combined",
    )


def provenance_fingerprint(
    registry: Registry, spec: ProfileSpec, ctx: Context, runs: Sequence[str]
) -> str:
    """Changes if the spec, any included run's provenance, an interaction's provenance, or a failure
    mode's full record (including its lifecycle state) changes."""
    prov: dict[str, Provenance | None] = {}
    for r in sorted(set(runs)):
        found = registry.find(Provenance, run_id=r)
        prov[r] = found[0] if found else None
    return content_hash(
        {
            "profile_version": PROFILE_VERSION,
            "spec": spec.to_dict(),
            "context": [
                ctx.model_fingerprint,
                ctx.dataset_fingerprint,
                ctx.split,
                ctx.evaluation_config_hash,
            ],
            "runs": {r: (p.fingerprint if p else None) for r, p in prov.items()},
            "faults": {
                f: content_hash(to_jsonable(registry.get(FaultExperiment, f).design))
                for f in spec.fault_experiments
            },
            "interactions": {
                i: registry.get(InteractionAnalysis, i).provenance_fingerprint
                for i in spec.interactions
            },
            "modes": {m: registry.get(FailureMode, m).content_hash() for m in spec.failure_modes},
            **(
                {
                    "slice_analyses": {
                        s: registry.get(SliceAnalysis, s).provenance_fingerprint
                        for s in spec.slice_analyses
                    }
                }
                if spec.slice_analyses
                else {}
            ),
        }
    )


def build(registry: Registry, store: ArtifactStore, spec: ProfileSpec) -> Built:
    ctx = context_of(registry, store, spec)
    check_compatibility(registry, store, spec, ctx)
    acc = _Acc()
    _baseline(registry, spec, ctx, acc)
    faults = _faults(registry, store, spec, acc)
    modes = _modes(registry, spec, acc)
    inters = _interactions(registry, spec, acc)
    _slices(registry, spec, ctx, acc)
    runs = [spec.baseline_run]
    for fid in spec.fault_experiments:
        runs += [
            t.treatment_run_id
            for t in registry.find(FaultTrial, fault_experiment_id=fid)
            if t.treatment_run_id
        ]
    _reproducibility(registry, spec, faults, modes, inters, acc, runs)
    _uncertainty(acc)
    missing = [d.value for d in Dimension if d.value not in acc.dims]
    assert not missing, missing  # noqa: S101  # every dimension is always present with an explicit status
    fp = provenance_fingerprint(registry, spec, ctx, runs)
    status = {d: v["status"] for d, v in sorted(acc.dims.items())}
    doc = {
        "profile_version": PROFILE_VERSION, "spec": spec.to_dict(), "spec_id": spec.spec_id, "scope": spec.scope.value,
        "context": {"model_fingerprint": ctx.model_fingerprint, "dataset_fingerprint": ctx.dataset_fingerprint, "split": ctx.split, "evaluation_config_hash": ctx.evaluation_config_hash, "task": ctx.task, "baseline_run": spec.baseline_run},
        "provenance_fingerprint": fp, "dimensions": dict(sorted(acc.dims.items())), "dimension_status": status,
        "interpreted": {"label": "INTERPRETED", "statements": acc.statements, "note": "deterministic templates over the observations above; the labels OBSERVED, DERIVED and INTERPRETED are kept apart", "never_claimed": NEVER_CLAIMED},
        "language": {"measures": "OBSERVED values read from stored evaluations", "derived": "values computed by earlier phases (fault aggregation, failure discovery, interaction analysis) and read back, not recomputed", "infers": "nothing beyond deterministic templates", "human_may_conclude": "which evidence exists for this evaluated system under these sources and what it shows, without generalizing beyond the evaluated model, dataset and configuration"},
        "no_score": "no overall score, ranking or verdict is computed; dimensions are independent",
    }  # fmt: skip
    return Built(ctx, doc, tuple(acc.refs), fp, status)

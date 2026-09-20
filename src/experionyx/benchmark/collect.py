"""Turns the PERSISTED evidence of an executed protocol into the coverage account and the
structured results. Nothing is measured here: values are read from stored evaluations, fault
analyses, interaction analyses, failure modes and the reliability profile. Coverage reports what
was and was not executed; it is never a measure of robustness, and missing coverage is never
represented as robustness."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from experionyx.artifacts import ArtifactStore
from experionyx.benchmark.protocol import Protocol, Unit
from experionyx.benchmark.spec import ENGINE_VERSION, BenchmarkSpec
from experionyx.benchmark.taxonomy import CoverageStatus, UnitKind, UnitStatus
from experionyx.domain import Run, RunStatus, to_jsonable
from experionyx.drift.entities import DriftAnalysis
from experionyx.errors import ExperionyxError
from experionyx.evaluation.loading import load_evaluation
from experionyx.evaluation.results import Status
from experionyx.failures.entities import FailureCluster, FailureEvidence, FailureMode
from experionyx.faults.entities import FaultExperiment, FaultTrial, TrialStatus
from experionyx.faults.report import load_analysis, read_artifact
from experionyx.hashing import content_hash
from experionyx.interactions.entities import InteractionAnalysis, InteractionEffect
from experionyx.provenance import Provenance
from experionyx.registry import Registry
from experionyx.reliability.entities import ReliabilityProfile
from experionyx.reliability.taxonomy import DimensionStatus
from experionyx.slices.entities import SliceAnalysis
from experionyx.stress.entities import StressAnalysis

NO_SCORE = "no overall score, ranking or verdict is computed; coverage counts what was executed and is not a measure of robustness"
NEVER_CLAIMED = [
    "the model is robust",
    "the model is not robust",
    "the model is safer than another",
    "coverage of the protocol establishes robustness",
    "a fault causes a failure",
]
LANGUAGE = {
    "measures": "OBSERVED values read from stored evaluations of runs",
    "derived": "values computed by earlier phases (fault aggregation, interaction analysis, failure discovery, profile assembly) and read back, not recomputed",
    "interpreted": "deterministic sentence templates over the values above, each with its basis",
    "human_may_conclude": "which evidence exists under this protocol and what it shows, limited to the evaluated model, dataset, faults, seeds and configuration",
}


@dataclass(frozen=True)
class Executed:
    """What the orchestrator did (identifiers of persisted records), passed to the collect Run."""

    investigation_id: str
    baseline_run: str
    fault_experiments: Mapping[str, str]  # grid -> fault experiment ID
    cell_experiments: Mapping[str, Mapping[str, str]]  # pair -> {AB|BA -> fault experiment ID}
    errors: Mapping[str, str]  # unit/experiment key -> why it could not be created or run
    discovery_run: str | None
    interaction_analyses: Mapping[str, str]  # pair -> interaction analysis ID
    profile_id: str | None
    slice_analysis: str | None = None
    drift_analysis: str | None = None
    stress_analysis: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "investigation_id": self.investigation_id,
            "baseline_run": self.baseline_run,
            "fault_experiments": dict(self.fault_experiments),
            "cell_experiments": {k: dict(v) for k, v in self.cell_experiments.items()},
            "errors": dict(self.errors),
            "discovery_run": self.discovery_run,
            "interaction_analyses": dict(self.interaction_analyses),
            "profile_id": self.profile_id,
            **({"slice_analysis": self.slice_analysis} if self.slice_analysis else {}),
            **({"drift_analysis": self.drift_analysis} if self.drift_analysis else {}),
            **({"stress_analysis": self.stress_analysis} if self.stress_analysis else {}),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Executed":
        return cls(
            str(d["investigation_id"]),
            str(d["baseline_run"]),
            dict(d["fault_experiments"]),
            {k: dict(v) for k, v in d["cell_experiments"].items()},
            dict(d["errors"]),
            d.get("discovery_run"),
            dict(d["interaction_analyses"]),
            d.get("profile_id"),
            d.get("slice_analysis"),
            d.get("drift_analysis"),
            d.get("stress_analysis"),
        )


@dataclass(frozen=True)
class Collected:
    units: list[dict[str, Any]]  # one per protocol unit, with status and run
    coverage: dict[str, Any]
    results: dict[str, Any]
    summary: dict[str, Any]
    section_status: dict[str, str]
    coverage_status: CoverageStatus
    provenance_fingerprint: str


def _g(x: object) -> str:
    return "n/a" if x is None else f"{x:.6g}" if isinstance(x, int | float) else str(x)


def _trial_units(registry: Registry, fx_id: str | None) -> dict[tuple[int, int], FaultTrial]:
    if fx_id is None:
        return {}
    return {
        (t.point_index, t.seed): t for t in registry.find(FaultTrial, fault_experiment_id=fx_id)
    }


def _prov(registry: Registry, run_id: str | None) -> str | None:
    if run_id is None:
        return None
    found = registry.find(Provenance, run_id=run_id)
    return found[0].fingerprint if found else None


def resolve_units(registry: Registry, protocol: Protocol, ex: Executed) -> list[dict[str, Any]]:
    """Map every protocol unit to its persisted trial/run and status. A unit whose experiment could
    not be created is NOT_RUN with the reason; failed and skipped trials keep theirs."""
    fx_trials = {g: _trial_units(registry, fx) for g, fx in ex.fault_experiments.items()}
    cell_trials = {
        p: {c: _trial_units(registry, fx) for c, fx in cells.items()}
        for p, cells in ex.cell_experiments.items()
    }
    out: list[dict[str, Any]] = []
    for u in protocol.units:
        row: dict[str, Any] = {
            **u.to_dict(),
            "status": UnitStatus.NOT_RUN.value,
            "run_id": None,
            "reason": None,
        }
        if u.kind is UnitKind.BASELINE:
            run = registry.get(Run, ex.baseline_run)
            row.update(
                status=(
                    UnitStatus.COMPLETED if run.status is RunStatus.COMPLETED else UnitStatus.FAILED
                ).value,
                run_id=run.id,
            )
        else:
            if u.kind is UnitKind.FAULT_TRIAL:
                trial = fx_trials.get(u.grid or "", {}).get((u.point_index or 0, u.seed or 0))
                err = ex.errors.get(f"grid:{u.grid}")
            else:
                trial = (
                    cell_trials.get(u.pair or "", {}).get(u.cell or "", {}).get((0, u.seed or 0))
                )
                err = ex.errors.get(f"pair:{u.pair}:{u.cell}")
            if trial is not None:
                row.update(
                    status={
                        TrialStatus.COMPLETED: UnitStatus.COMPLETED,
                        TrialStatus.FAILED: UnitStatus.FAILED,
                        TrialStatus.SKIPPED: UnitStatus.SKIPPED,
                    }[trial.status].value,
                    run_id=trial.treatment_run_id,
                    reason=trial.reason,
                )
            else:
                row["reason"] = err or "the experiment holding this unit was not created"
        out.append(row)
    return out


def collect(
    registry: Registry, store: ArtifactStore, spec: BenchmarkSpec, protocol: Protocol, ex: Executed
) -> Collected:
    units = resolve_units(registry, protocol, ex)
    fault_units = [u for u in units if u["kind"] == UnitKind.FAULT_TRIAL.value]
    cell_units = [u for u in units if u["kind"] == UnitKind.INTERACTION_TRIAL.value]
    stmts: list[dict[str, Any]] = []
    reasons: list[str] = []

    # -- baseline -------------------------------------------------------------------------------------
    baseline: dict[str, Any] = {
        "status": DimensionStatus.UNAVAILABLE.value,
        "source": {"kind": "RUN", "id": ex.baseline_run},
    }
    metrics_cov: dict[str, Any] = {
        "requested": list(spec.evaluation.metrics)
        or "all registered metrics compatible with the task",
        "available": [],
        "undefined": {},
    }
    slices_cov: dict[str, Any] = {
        "configured": [s.name for s in spec.evaluation.slices],
        "present": [],
        "missing": [],
    }
    try:
        ev = load_evaluation(registry, store, ex.baseline_run)
        ms = [m for m in ev.metrics if m.structured is None]
        baseline = {
            "status": DimensionStatus.OBSERVED.value,
            "source": {"kind": "RUN", "id": ex.baseline_run},
            "n_samples": ev.n_samples,
            "task": ev.task.value,
            "split": ev.context.split,
            "metrics": [
                {
                    "metric_id": m.metric_id,
                    "value": m.value,
                    "status": m.status.value,
                    "higher_is_better": m.higher_is_better,
                    "interval": to_jsonable(m.interval),
                }
                for m in ms
            ],
            "calibration": {"status": ev.calibration.status.value, "ece": ev.calibration.ece},
            "error_counts": dict(ev.errors.counts),
        }
        metrics_cov["available"] = sorted(
            m.metric_id for m in ms if m.status is Status.COMPUTED and m.value is not None
        )
        metrics_cov["undefined"] = {
            m.metric_id: f"{m.status.value}: {m.reason}"
            for m in ms
            if not (m.status is Status.COMPUTED and m.value is not None)
        }
        slices_cov["present"] = sorted(s.name for s in ev.slices)
        slices_cov["missing"] = sorted(set(slices_cov["configured"]) - set(slices_cov["present"]))
        primary = spec.primary_metric or (
            "accuracy" if ev.task.value == "CLASSIFICATION" else "rmse"
        )
        pm = next((m for m in ms if m.metric_id == primary), None)
        if pm is not None and pm.value is not None:
            stmts.append(
                {
                    "label": "INTERPRETED",
                    "text": f"At baseline, {primary} was {_g(pm.value)} on {ev.n_samples} samples of split {ev.context.split!r}.",
                    "basis": [{"kind": "RUN", "id": ex.baseline_run}],
                }
            )
    except ExperionyxError as exc:
        reasons.append(f"the baseline evaluation could not be read: {exc}")
        metrics_cov["undefined"] = {"*": f"baseline unreadable: {exc}"}

    # -- fault responses (persisted Phase 5 analyses) ---------------------------------------------------------------------
    fam: dict[str, Any] = {}
    responses: list[dict[str, Any]] = []
    for g in spec.faults:
        gu = [u for u in fault_units if u["grid"] == g.name]
        unsup = next((x for x in protocol.unsupported if x["grid"] == g.name), None)
        if unsup:
            fam[g.name] = {
                "status": UnitStatus.UNSUPPORTED.value,
                "reason": unsup["reason"],
                "points_requested": len(g.points()),
            }
            continue
        counts = {s.value: sum(u["status"] == s.value for u in gu) for s in UnitStatus}
        pts: dict[Any, list[str]] = {}
        for u in gu:
            pts.setdefault(u["point_index"], []).append(u["status"])
        fam[g.name] = {
            "fault": g.fault,
            "resolved_version": protocol.resolved[g.name]["version"],
            "points_requested": len(pts),
            "points_with_any_completed": sum("COMPLETED" in v for v in pts.values()),
            "points_fully_completed": sum(all(s == "COMPLETED" for s in v) for v in pts.values()),
            "trials_requested": len(gu),
            "completed": counts["COMPLETED"],
            "failed": counts["FAILED"],
            "skipped": counts["SKIPPED"],
            "not_run": counts["NOT_RUN"],
        }
        fx_id = ex.fault_experiments.get(g.name)
        resp: dict[str, Any] = {
            "grid": g.name,
            "fault": {
                "type": g.fault,
                "version": protocol.resolved[g.name]["version"],
                "parameters": to_jsonable(g.parameters),
                "sweep_parameter": g.sweep_parameter,
                "scope_fraction": g.scope_fraction,
            },
            "fault_experiment": fx_id,
            "status": DimensionStatus.UNAVAILABLE.value,
            "points": [],
        }
        if fx_id is None:
            resp["reason"] = ex.errors.get(f"grid:{g.name}", "the fault experiment was not created")
        else:
            try:
                res = load_analysis(registry, store, fx_id)
                resp.update(
                    primary_metric=res.primary_metric,
                    direction=res.primary_direction.value,
                    baseline_value=res.baseline_value,
                    warnings=list(res.warnings),
                )
                for p in res.points:
                    row: dict[str, Any] = {
                        "point": p.point_index,
                        "parameter": p.parameter_name,
                        "value": p.parameter_value,
                        "trials_requested": p.n_trials,
                        "completed": p.n_completed,
                        "faulted": to_jsonable(p.primary_faulted),
                        "deterioration": to_jsonable(p.primary_deterioration),
                        "effect_classification": p.assessment.classification.value,
                        "source": {"kind": "FAULT_EXPERIMENT", "id": fx_id},
                    }
                    resp["points"].append(row)
                    if p.n_completed:
                        d = p.primary_deterioration
                        where = (
                            f" ({p.parameter_name}={_g(p.parameter_value)})"
                            if p.parameter_name
                            else ""
                        )
                        stmts.append(
                            {
                                "label": "INTERPRETED",
                                "text": f"Under fault family {g.fault}{where} across {p.n_completed} completed trial(s), {res.primary_metric} changed from {_g(res.baseline_value)} to {_g(p.primary_faulted.mean)}; mean deterioration {_g(d.mean)}"
                                + (
                                    f" [{_g(d.ci_lower)}, {_g(d.ci_upper)}]"
                                    if d.ci_lower is not None
                                    else " (no interval)"
                                )
                                + f"; diagnostic effect classification {p.assessment.classification.value}.",
                                "basis": [{"kind": "FAULT_EXPERIMENT", "id": fx_id}],
                            }
                        )
                resp["status"] = (
                    DimensionStatus.OBSERVED.value
                    if any(p["completed"] for p in resp["points"])
                    else DimensionStatus.INSUFFICIENT_EVIDENCE.value
                )
            except ExperionyxError as exc:
                resp["reason"] = f"the fault analysis could not be read: {exc}"
        responses.append(resp)
    # -- interactions (persisted Phase 7 analyses) ------------------------------------------------------------------------------------
    inter: list[dict[str, Any]] = []
    for pair in spec.interactions:
        cu = [u for u in cell_units if u["pair"] == pair.name]
        row = {
            "pair": pair.name,
            "a": pair.a,
            "b": pair.b,
            "a_value": pair.a_value,
            "b_value": pair.b_value,
            "order_analysis": pair.order_analysis,
            "compound_trials": {
                c: sum(u["status"] == "COMPLETED" for u in cu if u["cell"] == c)
                for c in sorted({u["cell"] for u in cu})
            },
            "analysis_id": ex.interaction_analyses.get(pair.name),
            "status": DimensionStatus.UNAVAILABLE.value,
        }
        aid = row["analysis_id"]
        if aid is None:
            row["reason"] = ex.errors.get(
                f"interaction:{pair.name}", "the interaction analysis was not run"
            )
        else:
            a = registry.get(InteractionAnalysis, aid)
            (eff,) = registry.find(InteractionEffect, analysis_id=aid, measure=a.primary_metric)
            rec: Any = to_jsonable(eff.record)
            derived, boot = rec.get("derived") or {}, rec.get("bootstrap") or {}
            iv = (boot.get("intervals") or {}).get("interaction_contrast")
            row.update(
                metric=a.primary_metric,
                label=a.primary_class.value,
                lifecycle_state=a.status.value,
                interaction_contrast=derived.get("interaction_contrast"),
                interval=iv,
                order_effect=derived.get("order_effect"),
                trials_per_cell=to_jsonable(a.summary["trials_per_cell"]),
                bootstrap={
                    k: boot.get(k)
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
                status=(
                    DimensionStatus.INSUFFICIENT_EVIDENCE
                    if a.primary_class.value == "INCONCLUSIVE"
                    else DimensionStatus.DERIVED
                ).value,
                source={"kind": "INTERACTION", "id": aid},
            )
            stmts.append(
                {
                    "label": "INTERPRETED",
                    "text": f"An interaction contrast of {_g(derived.get('interaction_contrast'))} was observed for {a.primary_metric} between {pair.a} and {pair.b} under this design"
                    + (
                        f" (interval [{_g(iv['lower'])}, {_g(iv['upper'])}])"
                        if iv
                        else " (no interval: insufficient trials)"
                    )
                    + f"; label {a.primary_class.value}. This is an observed contrast, not a causal claim.",
                    "basis": [{"kind": "INTERACTION", "id": aid}],
                }
            )
        inter.append(row)
    # -- failure modes (persisted Phase 6 discovery) ---------------------------------------------------------------------------------------
    modes: list[dict[str, Any]] = []
    fm_status = DimensionStatus.UNAVAILABLE.value
    fm_reason: str | None = (
        "failure discovery was disabled in this benchmark"
        if not spec.discovery
        else ex.errors.get("discovery", "failure discovery did not run")
    )
    mode_hashes: dict[str, str] = {}
    if ex.discovery_run:
        try:
            cand = read_artifact(registry, store, ex.discovery_run, "failure-candidates.json")
            assert isinstance(cand, dict)  # noqa: S101
            for entry in cand["modes"]:
                m = registry.get(FailureMode, str(entry["id"]))
                meas: Any = to_jsonable(m.structured["measurements"])
                rep = any(
                    e.detail.get("summary") is True and e.detail.get("passed") is True
                    for e in registry.find(
                        FailureEvidence, failure_mode_id=m.id, evidence_kind="REPRODUCTION"
                    )
                )
                modes.append(
                    {
                        "mode_id": m.id,
                        "lifecycle_state": m.status.value,
                        "established": m.status.value == "CONFIRMED",
                        "category": m.category.value,
                        "prevalence": meas["prevalence"],
                        "severity": meas["severity"],
                        "classes": meas["classes"],
                        "slices": meas["slices"],
                        "fault_types": meas["fault_types"],
                        "signal_count": meas["size"],
                        "reproduction_check_passed": rep,
                        "cluster_id": registry.get(FailureCluster, m.cluster_id).id,
                        "source": {"kind": "FAILURE_MODE", "id": m.id},
                    }
                )
                mode_hashes[m.id] = m.content_hash()
            fm_status, fm_reason = DimensionStatus.DERIVED.value, None
            if modes:
                stmts.append(
                    {
                        "label": "INTERPRETED",
                        "text": f"Failure discovery over this protocol's runs produced {len(modes)} grouping(s) of failure signals; "
                        + f"{sum(x['established'] for x in modes)} are CONFIRMED. A mode that is not CONFIRMED is a grouping of signals, not an established finding.",
                        "basis": [{"kind": "RUN", "id": ex.discovery_run}],
                    }
                )
        except ExperionyxError as exc:
            fm_reason = f"the discovery artifacts could not be read: {exc}"
    # -- reliability profile (reference) -----------------------------------------------------------------------------------------------------
    profile: dict[str, Any] = {
        "status": DimensionStatus.UNAVAILABLE.value,
        "reason": "profile assembly was disabled in this benchmark"
        if not spec.profile
        else ex.errors.get("profile", "the profile was not built"),
    }
    prof_fp: str | None = None
    if ex.profile_id:
        pr = registry.get(ReliabilityProfile, ex.profile_id)
        prof_fp = pr.provenance_fingerprint
        profile = {
            "status": DimensionStatus.DERIVED.value,
            "profile_id": pr.id,
            "scope": pr.scope.value,
            "dimension_status": to_jsonable(pr.dimension_status),
            "source": {"kind": "RELIABILITY_PROFILE", "id": pr.id},
        }

    # -- slice analysis (only when the benchmark explicitly requested one) --------------------------------------------------------------------
    slice_block: dict[str, Any] = {"status": DimensionStatus.UNAVAILABLE.value, "reason": "no slice analysis was requested by this benchmark"}  # fmt: skip
    slice_fp: str | None = None
    if spec.slices:
        slice_block = {"status": DimensionStatus.UNAVAILABLE.value, "requested": [s.name for s in spec.slices], "reason": ex.errors.get("slices", "the slice analysis did not run")}  # fmt: skip
        if ex.slice_analysis:
            sa = registry.get(SliceAnalysis, ex.slice_analysis)
            slice_fp = sa.provenance_fingerprint
            slice_block = {"status": DimensionStatus.DERIVED.value, "requested": [s.name for s in spec.slices], "analysis_id": sa.id, "analysis_status": sa.analysis_status, "slices": to_jsonable(sa.summary.get("slices")), "fault_status_counts": to_jsonable(sa.summary.get("fault_status_counts")), "failure_mode_status_counts": to_jsonable(sa.summary.get("failure_mode_status_counts")), "interaction_status_counts": to_jsonable(sa.summary.get("interaction_status_counts")), "source": {"kind": "SLICE_ANALYSIS", "id": sa.id}}  # fmt: skip
            if sa.analysis_status != "COMPLETE":
                reasons.append(
                    "the slice analysis has insufficient or unavailable evidence for some requested slice or source"
                )
        else:
            reasons.append("the requested slice analysis did not run")

    # -- drift analysis (only when the benchmark explicitly requested one) ---------------------------------------------------------------------
    drift_block: dict[str, Any] = {"status": DimensionStatus.UNAVAILABLE.value, "reason": "no drift analysis was requested by this benchmark"}  # fmt: skip
    drift_fp: str | None = None
    if spec.drift is not None:
        drift_block = {"status": DimensionStatus.UNAVAILABLE.value, "reason": ex.errors.get("drift", "the drift analysis did not run")}  # fmt: skip
        if ex.drift_analysis:
            da = registry.get(DriftAnalysis, ex.drift_analysis)
            drift_fp = da.provenance_fingerprint
            drift_block = {"status": DimensionStatus.DERIVED.value, "analysis_id": da.id, "analysis_status": da.analysis_status, "ordering": to_jsonable(da.summary.get("ordering")), "n_window_pairs": da.summary.get("n_pairs"), "skipped": to_jsonable(da.summary.get("skipped")), "status_counts": to_jsonable(da.summary.get("status_counts")), "pairs": to_jsonable(da.summary.get("pairs")), "source": {"kind": "DRIFT_ANALYSIS", "id": da.id}, "note": "observed distribution differences between the declared windows of the baseline evaluation; not a fault, not a cause"}  # fmt: skip
            if da.analysis_status != "COMPLETE":
                reasons.append("the drift analysis has insufficient, unavailable or skipped evidence for some window or result")  # fmt: skip
        else:
            reasons.append("the requested drift analysis did not run")

    # -- stress analysis (only when the benchmark explicitly requested one) -------------------------------------------------------------------
    stress_block: dict[str, Any] = {"status": DimensionStatus.UNAVAILABLE.value, "reason": "no stress analysis was requested by this benchmark"}  # fmt: skip
    stress_fp: str | None = None
    if spec.stress is not None:
        stress_block = {"status": DimensionStatus.UNAVAILABLE.value, "planned": len(spec.stress.units()), "reason": ex.errors.get("stress", "the stress analysis did not run")}  # fmt: skip
        if ex.stress_analysis:
            xa = registry.get(StressAnalysis, ex.stress_analysis)
            stress_fp = xa.provenance_fingerprint
            cov: dict[str, Any] = dict(xa.summary.get("coverage") or {})  # type: ignore[call-overload]
            insufficient = bool(
                cov.get("failed")
                or cov.get("skipped")
                or cov.get("unsupported")
                or cov.get("insufficient_evidence")
            )
            stress_block = {"status": DimensionStatus.DERIVED.value, "analysis_id": xa.id, "analysis_status": xa.analysis_status, "planned": len(spec.stress.units()), "coverage": cov, "trial_status_counts": to_jsonable(xa.summary.get("trial_status_counts")), "primary_status_counts": to_jsonable(xa.summary.get("primary_status_counts")), "source": {"kind": "STRESS_ANALYSIS", "id": xa.id}, "note": "the stress analysis is referenced, not duplicated: planned, executed, unsupported, failed and insufficient-evidence units are counted in its coverage"}  # fmt: skip
            if xa.analysis_status != "COMPLETE" or insufficient:
                reasons.append(
                    "the stress analysis has failed, skipped, unsupported or insufficient-evidence trials"
                )
        else:
            reasons.append("the requested stress analysis did not run")

    # -- coverage ---------------------------------------------------------------------------------------------------------------------------------
    trials = {s.value: sum(u["status"] == s.value for u in fault_units) for s in UnitStatus}
    pts_total = sum(f.get("points_requested", 0) for f in fam.values())
    tested = sorted(k for k, f in fam.items() if f.get("completed", 0) > 0)
    inter_cov: dict[str, Any] = {
        "requested": len(spec.interactions),
        "analyzed": sum(r["analysis_id"] is not None for r in inter),
        "with_interval": sum(
            bool((r.get("bootstrap") or {}).get("status") == "COMPUTED") for r in inter
        ),
        "inconclusive": [r["pair"] for r in inter if r.get("label") == "INCONCLUSIVE"],
        "not_analyzed": {r["pair"]: r.get("reason") for r in inter if r["analysis_id"] is None},
        "compound_units": {
            "requested": len(cell_units),
            "completed": sum(u["status"] == "COMPLETED" for u in cell_units),
        },
    }
    failed = [
        {"unit_key": u["key"], "kind": u["kind"], "status": u["status"], "reason": u["reason"]}
        for u in units
        if u["status"] != "COMPLETED"
    ]
    if failed:
        reasons.append(f"{len(failed)} unit(s) did not complete")
    if protocol.unsupported:
        reasons.append(f"{len(protocol.unsupported)} unsupported fault grid(s) were excluded")
    if set(fam) - set(tested) - {k for k, f in fam.items() if f.get("status") == "UNSUPPORTED"}:
        reasons.append("some fault families completed no trial")
    if metrics_cov["undefined"]:
        reasons.append(f"{len(metrics_cov['undefined'])} metric(s) undefined")
    if slices_cov["missing"]:
        reasons.append("some configured slices are absent from the baseline results")
    if inter_cov["analyzed"] < inter_cov["requested"]:
        reasons.append("some interaction analyses did not run")
    if inter_cov["inconclusive"]:
        reasons.append("some interactions were inconclusive (too few trials)")
    if spec.discovery and fm_status != DimensionStatus.DERIVED.value:
        reasons.append("failure discovery did not produce results")
    if spec.profile and profile["status"] != DimensionStatus.DERIVED.value:
        reasons.append("the reliability profile was not built")
    complete = not reasons
    cov_status = CoverageStatus.COMPLETE if complete else CoverageStatus.INCOMPLETE
    coverage = {
        "fault_families": {"requested": [g.name for g in spec.faults], "tested": tested, "not_tested": sorted(set(fam) - set(tested)), "by_family": fam},
        "parameter_points": {"requested": pts_total, "with_any_completed": sum(f.get("points_with_any_completed", 0) for f in fam.values()), "fully_completed": sum(f.get("points_fully_completed", 0) for f in fam.values())},
        "trials": {"requested": len(fault_units), "completed": trials["COMPLETED"], "failed": trials["FAILED"], "skipped": trials["SKIPPED"], "not_run": trials["NOT_RUN"]},
        "seeds": {"requested": list(spec.seeds), "distinct_seeds_with_a_completed_trial": sorted({u["seed"] for u in fault_units if u["status"] == "COMPLETED"})},
        "metrics": metrics_cov, "slices": {**slices_cov, **({"slice_analysis": {"requested": slice_block.get("requested"), "status": slice_block["status"], "analysis_status": slice_block.get("analysis_status")}} if spec.slices else {})}, "interactions": inter_cov,
        "failure_modes": {"discovery": "DISABLED" if not spec.discovery else "RAN" if ex.discovery_run else "DID_NOT_RUN", "discovered": len(modes), "by_lifecycle_state": _count(m["lifecycle_state"] for m in modes), "unavailable_reason": fm_reason},
        "failed_or_undefined": failed, "unsupported_combinations": list(protocol.unsupported),
        "complete": complete, "incomplete_reasons": reasons, "note": NO_SCORE,
    }  # fmt: skip
    if not complete:
        stmts.insert(
            0,
            {
                "label": "INTERPRETED",
                "text": "Coverage is INCOMPLETE: "
                + "; ".join(reasons)
                + ". Missing coverage is not evidence of robustness.",
                "basis": [{"kind": "COVERAGE", "id": "coverage"}],
            },
        )

    # -- sections, provenance, results ------------------------------------------------------------------------------------------------------------
    ok_resp = [r for r in responses if r["status"] == "OBSERVED"]
    intervals = sum(
        1 for r in responses for p in r["points"] if p["deterioration"].get("ci_lower") is not None
    ) + sum(1 for r in inter if r.get("interval"))
    section = {
        "baseline": baseline["status"], "fault_responses": DimensionStatus.OBSERVED.value if ok_resp else DimensionStatus.INSUFFICIENT_EVIDENCE.value if responses else DimensionStatus.UNAVAILABLE.value,
        "interactions": DimensionStatus.UNAVAILABLE.value if not inter else DimensionStatus.DERIVED.value if any(r["status"] == "DERIVED" for r in inter) else DimensionStatus.INSUFFICIENT_EVIDENCE.value,
        "failure_modes": fm_status, "reliability_profile": profile["status"],
        **({"slice_analysis": slice_block["status"]} if spec.slices else {}),
        **({"drift_analysis": drift_block["status"]} if spec.drift is not None else {}),
        **({"stress_analysis": stress_block["status"]} if spec.stress is not None else {}),
        "uncertainty": DimensionStatus.DERIVED.value if intervals else DimensionStatus.INSUFFICIENT_EVIDENCE.value,
        "reproducibility": DimensionStatus.DERIVED.value if responses else DimensionStatus.INSUFFICIENT_EVIDENCE.value,
    }  # fmt: skip
    results = {
        "baseline": baseline, "fault_responses": {"status": section["fault_responses"], "grids": responses},
        "interactions": {"status": section["interactions"], "pairs": inter},
        "failure_modes": {"status": fm_status, "reason": fm_reason, "modes": modes, "discovery_run": ex.discovery_run},
        "reliability_profile": profile,
        **({"slice_analysis": slice_block} if spec.slices else {}),
        **({"drift_analysis": drift_block} if spec.drift is not None else {}),
        **({"stress_analysis": stress_block} if spec.stress is not None else {}),
        "uncertainty": {"status": section["uncertainty"], "intervals_available": intervals, "caveats": ["intervals are descriptive spreads under each analysis's own method and are not comparable across methods", *sorted({w for r in inter for w in ((r.get("bootstrap") or {}).get("warnings") or [])})]},
        "reproducibility": {"status": section["reproducibility"], "trials_per_family": {k: {"requested": f.get("trials_requested"), "completed": f.get("completed")} for k, f in fam.items() if "trials_requested" in f}, "distinct_seeds": len(spec.seeds), "interactions_by_lifecycle_state": _count(r.get("lifecycle_state") for r in inter if r.get("lifecycle_state")), "unresolved": list(reasons), "replay": "the collect Run can be replayed (`benchmark replay`); re-executing the whole protocol in another registry is the reproduction test"},
    }  # fmt: skip
    fp = content_hash(
        {
            "engine": ENGINE_VERSION,
            "spec_id": spec.spec_id,
            "protocol_hash": protocol.protocol_hash,
            "units": [[u["key"], u["status"], _prov(registry, u["run_id"])] for u in units],
            "interactions": {
                r["pair"]: registry.get(
                    InteractionAnalysis, r["analysis_id"]
                ).provenance_fingerprint
                for r in inter
                if r["analysis_id"]
            },
            "modes": mode_hashes,
            "profile": prof_fp,
            **({"slice_analysis": slice_fp} if spec.slices else {}),
            **({"drift_analysis": drift_fp} if spec.drift is not None else {}),
            **({"stress_analysis": stress_fp} if spec.stress is not None else {}),
        }
    )
    summary = {
        "benchmark": {"name": spec.name, "version": spec.version, "spec_id": spec.spec_id, "protocol_hash": protocol.protocol_hash, "engine_version": ENGINE_VERSION},
        "coverage_status": cov_status.value, "incomplete_reasons": reasons, "section_status": section,
        "counts": {"units": len(units), "completed": sum(u["status"] == "COMPLETED" for u in units), "failed": sum(u["status"] == "FAILED" for u in units), "skipped": sum(u["status"] == "SKIPPED" for u in units), "not_run": sum(u["status"] == "NOT_RUN" for u in units), "unsupported_grids": len(protocol.unsupported)},
        "interpreted": {"label": "INTERPRETED", "statements": stmts, "never_claimed": NEVER_CLAIMED}, "language": LANGUAGE, "no_score": NO_SCORE, "provenance_fingerprint": fp,
    }  # fmt: skip
    return Collected(units, coverage, results, summary, section, cov_status, fp)


def _count(items: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items()))


__all__ = ["Collected", "Executed", "FaultExperiment", "Unit", "collect", "resolve_units"]

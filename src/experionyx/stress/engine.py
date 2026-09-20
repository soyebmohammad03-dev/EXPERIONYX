"""The stress laboratory's procedures and orchestrator, run by the normal execution engine.

Trials: input stress (Fault Laboratory families) runs through the Fault Laboratory orchestrator
itself (`run_fault_experiment`): same procedure, same runs, same FaultExperiment/FaultTrial records,
so failure discovery, interaction analysis and every other consumer of fault experiments see them
unchanged. Model-level and evaluation-condition stress runs through `run_stress_evaluation`, which
evaluates a stressed derivative of the bound model with the same evaluation engine. EVERY point,
repeat and cell is its own Run; nothing loops out of sight.

Analysis (`run_stress_analysis`, a Run of its own): compares each trial with the baseline, records the
paired/unpaired Phase 10 statistics, slices, input-distribution and data-quality evidence and the
links to failure discovery and interaction analysis, writes seven documents and the records, and can
be replayed. There is no composite score and no causal statement anywhere."""

import json
import platform
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from experionyx.adapters.records import RegisteredDataset, RegisteredModel
from experionyx.artifacts import ArtifactStore, sha256_file
from experionyx.domain import (
    Artifact,
    ConfigurationRef,
    EpistemicKind,
    Experiment,
    Investigation,
    Observation,
    Run,
    RunStatus,
    to_jsonable,
)
from experionyx.drift import measures as drift_measures
from experionyx.errors import ArtifactIntegrityError, ExperionyxError, ValidationError
from experionyx.evaluation.config import EvaluationConfig, _thaw
from experionyx.evaluation.engine import PROCEDURE as EVALUATION_PROCEDURE
from experionyx.evaluation.engine import evaluate
from experionyx.evaluation.loading import load_evaluation
from experionyx.execution import Executor, RunContext, resolve_procedure
from experionyx.failures.engine import run_discovery
from experionyx.faults.apply import FaultedDataset
from experionyx.faults.design import FaultDesign, FaultLimits, SweepSpec
from experionyx.faults.entities import FaultTrial, TrialStatus
from experionyx.faults.lab import (
    RUN_SEED,
    _new_experiment,
    run_fault_experiment,
)
from experionyx.faults.lab import (
    preflight as fault_preflight,
)
from experionyx.faults.library import default_fault_registry
from experionyx.faults.report import read_artifact
from experionyx.hashing import content_hash
from experionyx.interactions.config import InteractionConfig, InteractionSpec
from experionyx.interactions.engine import run_interaction
from experionyx.interactions.lifecycle import REPLAY_TOLERANCE, _compare
from experionyx.provenance import RunOutcome
from experionyx.registry import Registry
from experionyx.slices import analysis as slice_an
from experionyx.slices.data import SliceDataError, load_baseline, run_rows, sample_table
from experionyx.slices.evaluate import evaluate as slice_membership
from experionyx.slices.spec import SliceSpec
from experionyx.stats import core as st
from experionyx.stress import analysis as an
from experionyx.stress.capability import (
    StressUnsupported,
    capability_report,
    load_model,
    refuse_unavailable,
)
from experionyx.stress.entities import StressAnalysis, StressTrial
from experionyx.stress.spec import (
    STRESS_SCHEMA,
    STRESS_VERSION,
    Origin,
    StressDesign,
    StressSpec,
    StressUnit,
)
from experionyx.stress.transform import build_stressed_model

PROCEDURE_TRIAL = "experionyx.stress.engine:run_stress_evaluation"
PROCEDURE_ANALYSIS = "experionyx.stress.engine:run_stress_analysis"
DOCUMENTS = ("spec", "plan", "trials", "baseline", "results", "analyses", "summary")
ARTIFACT_DIR = "stress"
DETERMINISM_NOTE = (
    "Seeded stress derives randomness from (seed, array name) with numpy's PCG64; deterministic "
    "stress (scaling, thresholds, batch size) uses no randomness and takes no seed. Bitwise "
    "identity across numpy versions or hardware is not claimed."
)


class StressBaselineError(ExperionyxError):
    """The baseline run is not compatible with the stress design (reason attached)."""


# -- documents on disk ---------------------------------------------------------------------------------------


def _write(ctx: RunContext, name: str, payload: object) -> str:
    directory = ctx.artifact_dir / ARTIFACT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(
        json.dumps(to_jsonable(payload), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return ctx.register_artifact(
        f"{ARTIFACT_DIR}/{name}.json", name=f"stress-{name}", media_type="application/json"
    ).id


# -- trial procedure (model-level and evaluation-condition stress) ----------------------------------------------


def trial_parameters(evaluation: EvaluationConfig, unit: StressUnit) -> dict[str, object]:
    return {
        "stress_evaluation": {
            "evaluation": to_jsonable(evaluation),
            "unit": unit.to_dict(),
            "stress_schema": STRESS_SCHEMA,
        }
    }


def run_stress_evaluation(ctx: RunContext) -> None:
    if set(ctx.parameters) != {"stress_evaluation"}:
        raise ValidationError(
            "a stress trial's configuration must be exactly {'stress_evaluation': ...}"
        )
    body = _thaw(ctx.parameters["stress_evaluation"])
    if not isinstance(body, dict) or set(body) != {"evaluation", "unit", "stress_schema"}:
        raise ValidationError("'stress_evaluation' has the wrong fields")
    evaluation = EvaluationConfig.from_dict(body["evaluation"])
    unit = body["unit"]
    components = tuple(StressSpec.from_dict(c) for c in unit["components"])
    if ctx.model is None or ctx.dataset is None or ctx.inputs is None or ctx.inputs.dataset is None:
        raise StressUnsupported(
            "a stress trial needs a registered model and dataset bound to the run"
        )
    if any(c.info.origin is Origin.FAULT_LABORATORY for c in components):
        raise StressUnsupported(
            "input stress runs through the Fault Laboratory, not the stress trial procedure"
        )
    for c in components:
        if c.family == "BATCH_SIZE":
            evaluation = replace(evaluation, batch_size=int(c.parameters["batch_size"]))  # type: ignore[call-overload]
    started = time.perf_counter()
    build = build_stressed_model(ctx.model, components)
    build_seconds = time.perf_counter() - started
    started = time.perf_counter()
    result = evaluate(ctx, evaluation, model=build.model)
    seconds = time.perf_counter() - started
    pred = ctx.artifact_dir / "evaluation" / "predictions.jsonl"
    digest = sha256_file(pred)[0] if pred.is_file() else None
    doc = {
        "stress_version": STRESS_VERSION,
        "unit": unit,
        "unit_key": unit["key"],
        "stress_ids": unit["stress_ids"],
        "components": [c.to_dict() for c in components],
        "origins": [c.info.origin.value for c in components],
        "targets": [c.info.target for c in components],
        "seed": unit["seed"],
        "model_fingerprint": ctx.inputs.model.fingerprint if ctx.inputs.model else None,
        "dataset_fingerprint": ctx.inputs.dataset.fingerprint,
        "split": evaluation.split,
        "batch_size": evaluation.batch_size,
        "build": build.record,
        "predictions_digest": digest,
        "n_samples": result.n_samples,
        "determinism": DETERMINISM_NOTE,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "timings_seconds": {"build": build_seconds, "evaluation": seconds},
    }
    _write(ctx, "stress", doc)
    ctx.observe("stress.evaluation_seconds", seconds, unit="s")
    ctx.observe("stress.n_samples", result.n_samples, kind=EpistemicKind.OBSERVATION)


# -- baseline ---------------------------------------------------------------------------------------------------


def baseline_parameters(design: StressDesign) -> dict[str, object]:
    return design.evaluation.to_parameters()


def _strip(params: object, ignore_batch: bool) -> object:
    plain = _thaw(params)
    if ignore_batch and isinstance(plain, dict) and isinstance(plain.get("evaluation"), dict):
        plain["evaluation"] = {k: v for k, v in plain["evaluation"].items() if k != "batch_size"}
    return plain


def validate_baseline(
    reg: Registry,
    run_id: str,
    design: StressDesign,
    model: RegisteredModel,
    data: RegisteredDataset,
) -> dict[str, Any]:
    """A baseline must be the SAME model, dataset, split, evaluation and metric configuration as the
    stressed evaluations (only an evaluation condition the plan deliberately stresses may differ)."""
    run = reg.get(Run, run_id)
    exp = reg.get(Experiment, run.experiment_id)
    cfg = reg.get(ConfigurationRef, exp.configuration_id)
    ignore = any(c.family == "BATCH_SIZE" for c in design.plan.components)
    checks = {
        "completed": run.status is RunStatus.COMPLETED,
        "same_model": exp.model == model.ref(),
        "same_dataset": exp.dataset == data.ref(),
        "same_evaluation_configuration": _strip(cfg.parameters, ignore)
        == _strip(design.evaluation.to_parameters(), ignore),
    }
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        why = {
            "completed": f"the baseline run is {run.status.value}, not COMPLETED",
            "same_model": "the baseline was run on a different model",
            "same_dataset": "the baseline was run on a different dataset",
            "same_evaluation_configuration": "the baseline used a different evaluation configuration (split, metrics, bootstrap, ...); a control may differ from the stressed runs only by the stress",
        }
        raise StressBaselineError("; ".join(why[k] for k in bad))
    return {
        "run_id": run_id,
        "checks": checks,
        "ignored_for_stressed_condition": ["batch_size"] if ignore else [],
        "evaluation_configuration": _thaw(cfg.parameters),
    }


# -- input evidence (before / after a Fault Laboratory transformation) ----------------------------------------


def input_evidence(
    ctx: Any, trial_run_id: str, split: str | None, max_features: int = 50
) -> dict[str, Any]:
    """The input distribution before and after the recorded fault, and its data-quality effect. It is
    stress-induced by construction: the two sets are the SAME samples before and after, so no
    significance test applies and nothing here is natural drift."""
    fault = read_artifact(ctx.registry, ctx.store, trial_run_id, "fault/fault.json")
    assert isinstance(fault, dict)  # noqa: S101
    registry = default_fault_registry()
    spec = registry.from_dict(fault["fault"])
    assert ctx.dataset is not None  # noqa: S101
    faulted = FaultedDataset(ctx.dataset, spec, registry, split=split)
    before, after = [], []
    for b, a in zip(ctx.dataset.batches(1024, split), faulted.batches(1024, split), strict=True):
        before.append(np.asarray(b.inputs, dtype=float))
        after.append(np.asarray(a.inputs, dtype=float))
    if not before:
        return {"status": "UNAVAILABLE", "reason": "no samples"}
    x, y = np.concatenate(before), np.concatenate(after)
    doc: dict[str, Any] = {
        "status": "DERIVED",
        "label": "STRESS_INDUCED",
        "note": "the same samples before and after a deliberately applied, recorded transformation; this is not naturally occurring drift and no test is applied",
        "fault_id": fault.get("fault_id"),
        "fault_type": fault["fault"]["type"],
        "n_samples": int(x.shape[0]),
        "shape": list(x.shape[1:]),
    }
    quality = {
        "missing_or_nonfinite_before": int((~np.isfinite(x)).sum()),
        "missing_or_nonfinite_after": int((~np.isfinite(y)).sum()),
        "n_values": int(x.size),
    }
    doc["data_quality"] = quality
    if x.ndim != 2:
        fin = np.isfinite(x) & np.isfinite(y)
        doc["global"] = {
            "mean_before": float(x[np.isfinite(x)].mean()) if np.isfinite(x).any() else None,
            "mean_after": float(y[np.isfinite(y)].mean()) if np.isfinite(y).any() else None,
            "fraction_changed": float((x[fin] != y[fin]).mean()) if fin.any() else None,
        }
        doc["features"] = {
            "status": "UNAVAILABLE",
            "reason": "per-feature distributions are reported for 2-D tabular inputs only",
        }
        return doc
    feats: dict[str, Any] = {}
    for j in range(min(x.shape[1], max_features)):
        b_valid, _ = drift_measures.clean(x[:, j].tolist(), "NUMERIC")
        a_valid, ca = drift_measures.clean(y[:, j].tolist(), "NUMERIC")
        rec: dict[str, Any] = {
            "n_valid_before": len(b_valid),
            "n_valid_after": len(a_valid),
            "n_missing_or_nonfinite_after": ca.n_missing + ca.n_nonfinite,
        }
        if b_valid and a_valid:
            rec |= {
                "mean_before": float(np.mean(b_valid)),
                "mean_after": float(np.mean(a_valid)),
                "ks_statistic": drift_measures.ks_statistic(b_valid, a_valid),
                "wasserstein_1": drift_measures.wasserstein_1(b_valid, a_valid),
            }
        feats[str(j)] = rec
    doc["features"] = feats
    if x.shape[1] > max_features:
        doc["features_truncated"] = x.shape[1]
    return doc


# -- analysis -------------------------------------------------------------------------------------------------------


def _read_json(reg: Registry, store: ArtifactStore, run_id: str, path: str) -> dict[str, Any]:
    doc = read_artifact(reg, store, run_id, path)
    if not isinstance(doc, dict):
        raise ExperionyxError(f"{path} of run {run_id} is not an object")
    return doc


def _artifact_digests(reg: Registry, run_id: str) -> dict[str, str]:
    return {
        a.path: a.digest for a in sorted(reg.find(Artifact, run_id=run_id), key=lambda a: a.path)
    }


def _obs(reg: Registry, run_id: str, name: str) -> float | None:
    for o in reg.find(Observation, run_id=run_id):
        if o.name == name and isinstance(o.value, int | float) and not isinstance(o.value, bool):
            return float(o.value)
    return None


@dataclass
class Computed:
    docs: dict[str, Any]
    status: str
    fingerprint: str
    summary: dict[str, Any]


def compute(
    reg: Registry,
    store: ArtifactStore,
    design: StressDesign,
    baseline_run: str,
    trials: Sequence[Mapping[str, Any]],
    links: Mapping[str, Any],
    capabilities: Sequence[Mapping[str, Any]],
    dataset: Any | None,
    investigation_id: str,
) -> Computed:
    """Every number of an analysis, deterministic given the registry content and the dataset."""
    plan = design.plan
    base_ev = load_evaluation(reg, store, baseline_run)
    base_rows = run_rows(reg, store, baseline_run)
    task = base_ev.task
    measure_name, hib = slice_an.measure_of(task)
    model = reg.get(RegisteredModel, design.model_id)
    data = reg.get(RegisteredDataset, design.dataset_id)
    compat = validate_baseline(reg, baseline_run, design, model, data)
    stress_kw: dict[str, Any] = {
        "min_members": design.min_members,
        "confidence": design.confidence,
        "resamples": design.resamples,
        "permutations": design.permutations,
        "seed": design.seed,
    }
    # slices (explicit only); static membership so it is the same set in the baseline and every trial
    slice_defs: dict[str, dict[str, Any]] = {}
    for sd in design.slices:
        spec = SliceSpec.from_dict(sd)
        if not spec.static:
            slice_defs[spec.name] = {
                "slice_id": spec.slice_id,
                "status": "REFUSED",
                "reason": "the slice reads model-output fields (predicted/correct/confidence), so its membership would differ between the baseline and a stressed run",
            }
            continue
        try:
            mem = slice_membership(
                spec, sample_table(load_baseline(reg, store, baseline_run), dataset, spec.fields)
            )
            slice_defs[spec.name] = {
                "slice_id": spec.slice_id,
                "status": mem.status.value,
                "n_members": mem.n_members,
                "n_total": mem.n_total,
                "prevalence": mem.prevalence,
                "membership_digest": mem.membership_digest,
                "ids": [int(i) for i in mem.sample_ids],
            }
        except SliceDataError as exc:
            slice_defs[spec.name] = {
                "slice_id": spec.slice_id,
                "status": "UNAVAILABLE",
                "reason": str(exc),
            }
    results: dict[str, Any] = {}
    analyses: dict[str, Any] = {}
    trial_docs: list[dict[str, Any]] = []
    fp_trials: list[Any] = []
    rows_by_key: dict[str, Mapping[int, Mapping[str, Any]]] = {}
    for t in trials:
        u = t["unit"]
        key = u["key"]
        rec = {
            "unit": u,
            "status": t["status"],
            "run_id": t.get("run_id"),
            "experiment_id": t.get("experiment_id"),
            "reason": t.get("reason"),
            "origin": t["origin"],
        }
        if t["status"] != "COMPLETED" or not t.get("run_id"):
            results[key] = {"status": t["status"], "reason": t.get("reason")}
            trial_docs.append({**rec, "evidence": None})
            fp_trials.append([key, t["status"]])
            continue
        rid = t["run_id"]
        ev = load_evaluation(reg, store, rid)
        rows = run_rows(reg, store, rid)
        rows_by_key[key] = rows
        origin_doc = (
            "fault/fault.json"
            if t["origin"]["kind"] == "FAULT_LABORATORY"
            else "stress/stress.json"
        )
        od = _read_json(reg, store, rid, origin_doc)
        lineage = {
            k: od.get(k)
            for k in (
                "fault_id",
                "fault_family_id",
                "affected_fraction",
                "resulting_content_digest",
                "faulted_dataset_identity",
                "predictions_digest",
                "stress_ids",
                "build",
            )
        }
        digests = _artifact_digests(reg, rid)
        # a Fault Laboratory trial evaluates a DERIVED (faulted) view whose own identity differs; the
        # dataset it was derived from is what must match the baseline's
        source_dataset = (
            od.get("source_dataset_fingerprint")
            if t["origin"]["kind"] == "FAULT_LABORATORY"
            else ev.context.dataset_fingerprint
        )
        evidence = {
            "trial_key": key,
            "run_id": rid,
            "baseline_run_id": baseline_run,
            "stress_ids": u["stress_ids"],
            "origin": t["origin"],
            "model_fingerprint": ev.context.model_fingerprint,
            "dataset_fingerprint": source_dataset,
            "evaluated_dataset_identity": ev.context.dataset_fingerprint,
            "split": ev.context.split,
            "seed": u["seed"],
            "transformation": {k: v for k, v in lineage.items() if v is not None},
            "artifact_digests": digests,
        }
        if (
            ev.context.model_fingerprint != base_ev.context.model_fingerprint
            or source_dataset != base_ev.context.dataset_fingerprint
        ):
            results[key] = {
                "status": "INCOMPATIBLE",
                "reason": "the trial and the baseline evaluated a different model or dataset",
            }
            trial_docs.append({**rec, "evidence": evidence})
            fp_trials.append([key, "INCOMPATIBLE"])
            continue
        metrics = an.metric_deltas(base_ev, ev)
        measure = an.paired_measure(base_rows, rows, task, None, **stress_kw)
        common = sorted(set(base_rows) & set(rows))
        agree = (
            sum(base_rows[i]["predicted"] == rows[i]["predicted"] for i in common) / len(common)
            if common
            else None
        )
        sdiffs = [
            max(abs(a - b) for a, b in zip(base_rows[i]["scores"], rows[i]["scores"], strict=True))
            for i in common
            if base_rows[i].get("scores") and rows[i].get("scores")
        ]
        results[key] = {
            "status": "COMPLETED",
            "metrics": metrics,
            "primary_measure": {
                k: measure[k]
                for k in (
                    "measure",
                    "mean_baseline",
                    "mean_stressed",
                    "absolute_change",
                    "relative_change",
                    "relative_reason",
                    "deterioration",
                    "direction",
                )
            },
            "outputs_vs_baseline": {
                "n_common_samples": len(common),
                "prediction_agreement": agree,
                "max_score_difference": max(sdiffs) if sdiffs else None,
            },
            "latency": {
                "baseline_evaluation_seconds": _obs(
                    reg, baseline_run, "evaluation.inference_seconds"
                ),
                "stressed_evaluation_seconds": _obs(reg, rid, "stress.evaluation_seconds")
                or _obs(reg, rid, "fault.evaluation_seconds"),
            },
            "note": an.NOTE,
        }
        per_slice: dict[str, Any] = {}
        for name, sd in slice_defs.items():
            if sd["status"] != "COMPUTED":
                per_slice[name] = {
                    "status": "UNAVAILABLE"
                    if sd["status"] in ("REFUSED", "UNAVAILABLE", "MISSING_FIELD")
                    else "INSUFFICIENT_EVIDENCE",
                    "reason": sd.get("reason") or f"the slice has no members ({sd['status']})",
                    "n_members": sd.get("n_members", 0),
                }
                continue
            per_slice[name] = {
                "slice_id": sd["slice_id"],
                "n_members": sd["n_members"],
                **an.paired_measure(base_rows, rows, task, sd["ids"], **stress_kw),
            }
        analyses[key] = {"primary": measure, "slices": per_slice}
        if (
            t["origin"]["kind"] == "FAULT_LABORATORY"
            and design.input_evidence
            and dataset is not None
        ):
            try:
                results[key]["input_evidence"] = input_evidence(
                    _Ctx(reg, store, dataset), rid, ev.context.split
                )
            except ExperionyxError as exc:
                results[key]["input_evidence"] = {"status": "UNAVAILABLE", "reason": str(exc)}
        elif t["origin"]["kind"] != "FAULT_LABORATORY":
            results[key]["input_evidence"] = {
                "status": "INPUTS_UNCHANGED",
                "note": "this stress does not modify the inputs; the input distribution is the baseline's",
            }
        trial_docs.append({**rec, "evidence": evidence})
        fp_trials.append(
            [
                key,
                "COMPLETED",
                digests.get("evaluation/evaluation.json"),
                lineage.get("predictions_digest")
                or (lineage.get("resulting_content_digest") and "fault"),
            ]
        )
    # multiple comparisons: one family, the per-trial primary tests
    entries = [(k, a["primary"]) for k, a in analyses.items() if "primary" in a]
    family = an.correct(entries, design.correction, design.alpha) if entries else {}
    # per point aggregates over repeats/seeds (raw values kept)
    aggregates: dict[str, Any] = {}
    groups: dict[tuple[str, int], list[str]] = {}
    for t in trials:
        u = t["unit"]
        if u["key"] in analyses:
            groups.setdefault((u["cell"], u["point_index"]), []).append(u["key"])
    for (cell, p), keys in sorted(groups.items()):
        vals = [
            analyses[k]["primary"]["deterioration"]
            for k in keys
            if analyses[k]["primary"].get("deterioration") is not None
        ]
        u0 = next(t["unit"] for t in trials if t["unit"]["key"] == keys[0])
        aggregates[f"{cell or 'ALL'}|{p}"] = {
            "cell": cell,
            "point_index": p,
            "parameter": u0["parameter"],
            "value": u0["value"],
            "trial_keys": keys,
            "metric": measure_name,
            "deterioration": an.aggregate(
                vals,
                confidence=design.confidence,
                resamples=design.resamples,
                seed=design.seed,
                method="percentile",
            )
            if vals
            else {"n": 0},
        }
    stability: dict[str, Any] | None = None
    if plan.components[0].family == "REPEATED_EXECUTION":
        done = [t for t in trials if t["status"] == "COMPLETED"]
        if len(done) >= 2:
            rows_l = [rows_by_key[t["unit"]["key"]] for t in done]
            digs = [
                _read_json(reg, store, t["run_id"], "stress/stress.json").get("predictions_digest")
                for t in done
            ]
            mvals = [
                {m["metric_id"]: m["stressed_value"] for m in results[t["unit"]["key"]]["metrics"]}
                for t in done
            ]
            stability = an.stability(rows_l, digs, mvals)
            secs = [
                results[t["unit"]["key"]]["latency"]["stressed_evaluation_seconds"] for t in done
            ]
            stability["latency_variation"] = (
                an.latency_variation([s for s in secs if s is not None])
                if all(s is not None for s in secs)
                else {"status": "UNAVAILABLE"}
            )
        else:
            stability = {
                "observed": "INSUFFICIENT_EVIDENCE",
                "reason": "fewer than two completed repeats",
            }
    counts = Counter(t["status"] for t in trials)
    statuses = Counter(a["primary"]["status"] for a in analyses.values())
    slice_status = Counter(s.get("status") for a in analyses.values() for s in a["slices"].values())
    partial = (
        any(t["status"] != "COMPLETED" for t in trials)
        or bool(statuses.get("INSUFFICIENT_EVIDENCE"))
        or any(k in slice_status for k in ("INSUFFICIENT_EVIDENCE", "UNAVAILABLE"))
        or any(r.get("status") == "INCOMPATIBLE" for r in results.values())
        or any(
            x.get("status") in ("REFUSED", "UNAVAILABLE", "FAILED")
            for x in links.values()
            if isinstance(x, Mapping)
        )
    )
    units = [t["unit"] for t in trials]
    coverage = {
        "requested": len(units),
        "executed": counts.get("COMPLETED", 0) + counts.get("FAILED", 0),
        "completed": counts.get("COMPLETED", 0),
        "failed": counts.get("FAILED", 0),
        "skipped": counts.get("SKIPPED", 0),
        "unsupported": counts.get("UNSUPPORTED", 0),
        "insufficient_evidence": statuses.get("INSUFFICIENT_EVIDENCE", 0),
        "seeds": sorted({u["seed"] for u in units}),
    }
    docs: dict[str, Any] = {
        "spec": {
            "spec_id": design.spec_id,
            "design": design.to_dict(),
            "stress_version": STRESS_VERSION,
            "model": {"id": design.model_id, "fingerprint": base_ev.context.model_fingerprint},
            "dataset": {
                "id": design.dataset_id,
                "fingerprint": base_ev.context.dataset_fingerprint,
            },
            "split": base_ev.context.split,
            "evaluation_config_hash": content_hash(to_jsonable(base_ev.config)),
            "baseline_run": baseline_run,
            "capabilities": list(capabilities),
            "seeds": coverage["seeds"],
            "versions": versions(),
            "stress_vs_fault": "input stress is executed by the Fault Laboratory and identified as FAULT_LABORATORY in every trial's origin; model-level stress is new and identified as MODEL or EVALUATION",
            "determinism": DETERMINISM_NOTE,
            "source_revision": "recorded in the run's Provenance record (see `stress inspect`)",
        },
        "plan": {
            "plan": plan.to_dict(),
            "plan_id": plan.plan_id,
            "design": plan.design,
            "units": units,
            "coverage": coverage,
        },
        "trials": {"trials": trial_docs},
        "baseline": {
            "run_id": baseline_run,
            "compatibility": compat,
            "task": task.value,
            "n_samples": base_ev.n_samples,
            "measure": measure_name,
            "higher_is_better": hib,
            "metrics": [
                {
                    "metric_id": m.metric_id,
                    "value": m.value,
                    "status": m.status.value,
                    "higher_is_better": m.higher_is_better,
                    "n_samples": m.n_samples,
                    "interval": an._interval(m),
                }
                for m in base_ev.metrics
                if m.structured is None
            ],
            "context": to_jsonable(base_ev.context),
        },
        "results": {
            "results": results,
            "aggregates": aggregates,
            "stability": stability,
            "note": an.NOTE,
        },
        "analyses": {
            "analyses": analyses,
            "multiple_comparisons": family,
            "slices": {
                n: {k: v for k, v in d.items() if k != "ids"} for n, d in slice_defs.items()
            },
            "links": dict(links),
            "pairing_note": "PAIRED means the same samples with the same ground truth were evaluated in both runs; the difference is then per sample",
        },
    }
    summary = {
        "spec_id": design.spec_id,
        "plan_id": plan.plan_id,
        "design": plan.design,
        "families": [c.family for c in plan.components],
        "origins": sorted({c.info.origin.value for c in plan.components}),
        "baseline_run_id": baseline_run,
        "measure": measure_name,
        "coverage": coverage,
        "trial_status_counts": dict(sorted(counts.items())),
        "primary_status_counts": dict(sorted(statuses.items())),
        "slice_status_counts": {
            str(k): v for k, v in sorted(slice_status.items(), key=lambda kv: str(kv[0]))
        },
        "points": {
            k: {
                "parameter": v["parameter"],
                "value": v["value"],
                "n": v["deterioration"].get("n"),
                "mean_deterioration": (v["deterioration"].get("summary") or {})
                .get("mean", {})
                .get("value")
                if isinstance(v["deterioration"].get("summary"), dict)
                else None,
            }
            for k, v in aggregates.items()
        },
        "stability_observed": None if stability is None else stability.get("observed"),
        "links": dict(links),
        "correction": {"method": design.correction, "alpha": design.alpha},
        "note": an.NOTE + "; no composite stress or robustness score is computed",
    }
    fp = content_hash(
        {
            "design": design.to_dict(),
            "baseline": [
                baseline_run,
                content_hash(to_jsonable(base_ev.config)),
                base_ev.context.model_fingerprint,
                base_ev.context.dataset_fingerprint,
                base_ev.context.split,
            ],
            "baseline_digests": _artifact_digests(reg, baseline_run).get(
                "evaluation/evaluation.json"
            ),
            "trials": fp_trials,
            "slices": {n: d.get("membership_digest") for n, d in slice_defs.items()},
            "versions": {k: v for k, v in versions().items() if k != "python"},
        }
    )
    docs["spec"]["provenance_fingerprint"] = fp
    return Computed(docs, "PARTIAL" if partial else "COMPLETE", fp, summary)


class _Ctx:
    """The minimal context `input_evidence` needs, so it also runs outside a procedure."""

    def __init__(self, registry: Registry, store: ArtifactStore, dataset: Any) -> None:
        self.registry, self.store, self.dataset = registry, store, dataset


def versions() -> dict[str, str]:
    return {
        "stress": STRESS_VERSION,
        "stress_schema": str(STRESS_SCHEMA),
        "statistics": st.STATS_VERSION,
        "slice_analysis": slice_an.ANALYSIS_VERSION,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def analysis_parameters(
    investigation_id: str,
    design: StressDesign,
    baseline_run: str,
    trials: Sequence[Mapping[str, Any]],
    links: Mapping[str, Any],
    capabilities: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    return {
        "stress_analysis": {
            "investigation_id": investigation_id,
            "design": design.to_dict(),
            "baseline_run": baseline_run,
            "trials": [dict(t) for t in trials],
            "links": dict(links),
            "capabilities": [dict(c) for c in capabilities],
        }
    }


def run_stress_analysis(ctx: RunContext) -> None:
    if set(ctx.parameters) != {"stress_analysis"}:
        raise ValidationError(
            "a stress analysis configuration must be exactly {'stress_analysis': ...}"
        )
    body = _thaw(ctx.parameters["stress_analysis"])
    if not isinstance(body, dict) or set(body) != {
        "investigation_id",
        "design",
        "baseline_run",
        "trials",
        "links",
        "capabilities",
    }:
        raise ValidationError("'stress_analysis' has the wrong fields")
    design = StressDesign.from_dict(body["design"])
    reg, now = ctx.registry, ctx.started_at
    c = compute(
        reg,
        ctx.store,
        design,
        body["baseline_run"],
        body["trials"],
        body["links"],
        body["capabilities"],
        ctx.dataset,
        body["investigation_id"],
    )
    art = {name: _write(ctx, name, c.docs[name]) for name in DOCUMENTS if name != "summary"}
    art["summary"] = _write(ctx, "summary", c.summary)
    a = StressAnalysis(
        body["investigation_id"],
        ctx.run.id,
        design.spec_id,
        design.to_dict(),
        design.model_id,
        design.dataset_id,
        body["baseline_run"],
        c.fingerprint,
        c.status,
        {**c.summary, "artifacts": art},
        now,
    )
    new = not reg.exists(StressAnalysis, a.id)
    if new:
        with reg.transaction():
            reg.add(a)
            for t in body["trials"]:
                u = t["unit"]
                lineage = {
                    "origin": t["origin"],
                    "experiment_id": t.get("experiment_id"),
                    "fault_trial_id": (t["origin"] or {}).get("fault_trial_id"),
                    "parameter": u["parameter"],
                    "value": u["value"],
                }
                rec = StressTrial(
                    a.id,
                    u["key"],
                    u["point_index"],
                    u["repeat_index"],
                    u["cell"],
                    u["components"][0]["family"],
                    tuple(u["stress_ids"]),
                    t["origin"]["kind"],
                    u["seed"],
                    t["status"],
                    t.get("run_id"),
                    t.get("reason"),
                    lineage,
                    now,
                )
                if not reg.exists(StressTrial, rec.id):
                    reg.add(rec)
    ctx.observe("stress.new_record", int(new), kind=EpistemicKind.OBSERVATION)
    ctx.observe("stress.trials", len(body["trials"]), kind=EpistemicKind.OBSERVATION)
    ctx.observe(
        "stress.completed_trials",
        c.summary["coverage"]["completed"],
        kind=EpistemicKind.OBSERVATION,
    )


# -- orchestration --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StressRunResult:
    analysis_id: str | None
    run_id: str
    status: RunStatus
    baseline_run_id: str
    investigation_id: str
    capabilities: tuple[Mapping[str, Any], ...]


def preflight(
    reg: Registry,
    design: StressDesign,
    executor: Executor | None = None,
    adapters: Any = None,
    inputs_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Everything that can be refused without running: registered model and dataset, capabilities,
    Fault Laboratory compatibility of every unit, sample and trial budgets, baseline compatibility."""
    model = reg.get(RegisteredModel, design.model_id)
    data = reg.get(RegisteredDataset, design.dataset_id)
    adapters = (
        adapters if adapters is not None else (None if executor is None else executor.adapters)
    )
    root = (
        inputs_root
        if inputs_root is not None
        else (None if executor is None else executor.inputs_root)
    )
    adapter = load_model(reg, adapters, root, design.model_id)
    report = capability_report(adapter, design.plan, data)
    refuse_unavailable(report)
    if design.plan.origin is Origin.FAULT_LABORATORY:
        for u in design.plan.units():
            fault_preflight(design.plan.fault_spec_of(u), default_fault_registry(), data)
    split = design.evaluation.split
    n = data.metadata.splits.get(split) if split else data.metadata.num_samples
    if isinstance(n, int):
        if n > design.evaluation.retention.max_samples:
            raise StressUnsupported(f"{n} samples exceed the evaluation's retention.max_samples")
        if len(design.plan.units()) * n > 5_000_000:
            raise StressUnsupported(
                f"{len(design.plan.units())} trials x {n} samples exceed the sample-evaluation budget"
            )
    if design.baseline_run is not None:
        validate_baseline(reg, design.baseline_run, design, model, data)
    _ = model
    return report


def _fault_design(
    design: StressDesign, base: Any, seeds: tuple[int, ...], sweep: SweepSpec | None
) -> FaultDesign:
    return FaultDesign(
        base.to_dict(),
        design.evaluation,
        seeds=seeds,
        sweep=sweep,
        aggregation_resamples=min(design.resamples, 1000),
        limits=FaultLimits(),
    )


def _fault_trials(reg: Registry, fx_id: str) -> dict[tuple[int, int], FaultTrial]:
    return {
        (t.point_index, t.repeat_index): t for t in reg.find(FaultTrial, fault_experiment_id=fx_id)
    }


def _trial_dict(
    unit: StressUnit,
    status: str,
    run_id: str | None,
    experiment_id: str | None,
    reason: str | None,
    origin: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "unit": unit.to_dict(),
        "status": status,
        "run_id": run_id,
        "experiment_id": experiment_id,
        "reason": reason,
        "origin": dict(origin),
    }


def run_stress_experiment(
    registry: Registry,
    store: ArtifactStore,
    executor: Executor,
    design: StressDesign,
    *,
    source_root: Path,
    investigation_id: str | None = None,
    name: str | None = None,
) -> StressRunResult:
    """Preflight (refusals create nothing), baseline, every trial as its own run, optional failure
    discovery and interaction analysis, then the analysis run."""
    report = preflight(registry, design, executor)
    model = registry.get(RegisteredModel, design.model_id)
    data = registry.get(RegisteredDataset, design.dataset_id)
    plan = design.plan
    inv = (
        registry.get(Investigation, investigation_id)
        if investigation_id
        else Investigation(
            "model-stress",
            "How does measured behaviour change under controlled stress?",
            datetime.now(UTC),
        )
    )
    if not registry.exists(Investigation, inv.id):
        registry.add(inv)
    if design.baseline_run is not None:
        baseline_run = design.baseline_run
    else:
        bexp = _new_experiment(
            registry,
            inv,
            f"baseline {model.name} {model.version} on {data.name} {data.version}",
            "Baseline evaluation (control condition of a stress experiment)",
            model,
            data,
            baseline_parameters(design),
        )
        res = executor.execute(
            bexp.id,
            resolve_procedure(EVALUATION_PROCEDURE),
            seed=RUN_SEED,
            procedure_name=EVALUATION_PROCEDURE,
        )
        if res.status is not RunStatus.COMPLETED:
            raise StressBaselineError(f"the baseline evaluation failed: {res.error}")
        baseline_run = res.run.id
    validate_baseline(registry, baseline_run, design, model, data)
    label = name or f"stress {plan.plan_id[:12]}"
    trials: list[dict[str, Any]] = []
    fault_experiments: list[str] = []
    links: dict[str, Any] = {}
    interaction_cells: dict[str, list[str]] = {"A": [], "B": [], "AB": []}
    if plan.origin is Origin.FAULT_LABORATORY:
        fr = default_fault_registry()
        jobs: list[tuple[str, Any, tuple[int, ...], SweepSpec | None, list[StressUnit]]] = []
        units = plan.units()
        if plan.factorial:
            for cell, comp in (("A", plan.components[0]), ("B", plan.components[1])):
                jobs.append(
                    (
                        cell,
                        comp.fault_spec(),
                        plan.seeds,
                        None,
                        [u for u in units if u.cell == cell],
                    )
                )
            ab = fr.compound(plan.components[0].fault_spec(), plan.components[1].fault_spec())
            jobs.append(("AB", ab, plan.seeds, None, [u for u in units if u.cell == "AB"]))
        else:
            sweep = (
                None
                if plan.sweep is None
                else SweepSpec(plan.sweep.parameter, tuple(float(v) for v in plan.sweep.values))
            )
            base = (
                plan.components[0].fault_spec()
                if len(plan.components) == 1
                else fr.compound(*[c.fault_spec() for c in plan.components])
            )
            jobs.append(("", base, plan.seeds, sweep, list(units)))
        for cell, base_spec, seeds, sweep, job_units in jobs:
            fx = run_fault_experiment(
                registry,
                store,
                executor,
                model_id=design.model_id,
                dataset_id=design.dataset_id,
                base_spec=base_spec,
                fault_registry=fr,
                design=_fault_design(design, base_spec, seeds, sweep),
                name=f"{label} {cell}".strip(),
                source_root=source_root,
                baseline_run_id=baseline_run,
            )
            fault_experiments.append(fx.fault_experiment.id)
            ft = _fault_trials(registry, fx.fault_experiment.id)
            for u in job_units:
                t = ft.get((u.point_index, u.repeat_index))
                origin = {
                    "kind": "FAULT_LABORATORY",
                    "fault_experiment_id": fx.fault_experiment.id,
                    "fault_trial_id": None if t is None else t.id,
                    "fault_id": None if t is None else t.fault_id,
                    "procedure": "experionyx.faults.engine:run_fault_evaluation",
                }
                if t is None:
                    trials.append(
                        _trial_dict(
                            u,
                            "FAILED",
                            None,
                            None,
                            "the Fault Laboratory recorded no trial for this unit",
                            origin,
                        )
                    )
                    continue
                status = (
                    "COMPLETED"
                    if t.status is TrialStatus.COMPLETED
                    else "SKIPPED"
                    if t.status is TrialStatus.SKIPPED
                    else "FAILED"
                )
                trials.append(
                    _trial_dict(
                        u, status, t.treatment_run_id, t.treatment_experiment_id, t.reason, origin
                    )
                )
                if status == "COMPLETED" and u.cell and t.treatment_run_id:
                    interaction_cells[u.cell].append(t.treatment_run_id)
    else:
        for u in plan.units():
            sorigin: dict[str, Any] = {
                "kind": "STRESS_LABORATORY",
                "procedure": PROCEDURE_TRIAL,
                "families": [c.family for c in u.components],
            }
            cfg = trial_parameters(design.evaluation, u)
            try:
                texp = _new_experiment(
                    registry,
                    inv,
                    f"stress {'+'.join(c.family for c in u.components)} unit {u.key[4:16]} of {label}",
                    "Stressed evaluation",
                    model,
                    data,
                    cfg,
                )
                res = executor.execute(
                    texp.id,
                    resolve_procedure(PROCEDURE_TRIAL),
                    seed=RUN_SEED,
                    procedure_name=PROCEDURE_TRIAL,
                )
                ok = res.status is RunStatus.COMPLETED
                reason = None
                if not ok:
                    outs = registry.find(RunOutcome, run_id=res.run.id)
                    err = outs[0].error if outs else None
                    reason = (
                        f"{err.error_type}: {err.message}" if err else "the run did not complete"
                    )
                trials.append(
                    _trial_dict(
                        u, "COMPLETED" if ok else "FAILED", res.run.id, texp.id, reason, sorigin
                    )
                )
            except ExperionyxError as exc:
                trials.append(
                    _trial_dict(u, "FAILED", None, None, f"{type(exc).__name__}: {exc}", sorigin)
                )
    done_runs = [t["run_id"] for t in trials if t["status"] == "COMPLETED" and t["run_id"]]
    if design.discover_failures:
        try:
            dr = run_discovery(
                registry,
                store,
                executor,
                inv.id,
                [] if plan.origin is Origin.FAULT_LABORATORY else done_runs,
                fault_experiments,
            )
            links["failure_discovery"] = {
                "status": "COMPLETED" if dr.status is RunStatus.COMPLETED else "FAILED",
                "run_id": dr.run_id,
                "sources": {
                    "fault_experiments": fault_experiments,
                    "runs": [] if plan.origin is Origin.FAULT_LABORATORY else done_runs,
                },
                "note": "modes are discovered evidence groupings; none is confirmed by this analysis",
            }
        except ExperionyxError as exc:
            links["failure_discovery"] = {
                "status": "FAILED",
                "reason": f"{type(exc).__name__}: {exc}",
            }
    if plan.factorial:
        try:
            io = run_interaction(
                registry,
                store,
                executor,
                inv.id,
                InteractionSpec(
                    control=(baseline_run,),
                    a=tuple(interaction_cells["A"]),
                    b=tuple(interaction_cells["B"]),
                    ab=tuple(interaction_cells["AB"]),
                    config=InteractionConfig(bootstrap_resamples=design.resamples),
                ),
            )
            links["interaction"] = {
                "status": "COMPLETED" if io.analysis_id else "FAILED",
                "analysis_id": io.analysis_id,
                "run_id": io.run_id,
                "cells": {k: list(v) for k, v in interaction_cells.items()},
            }
        except ExperionyxError as exc:
            links["interaction"] = {
                "status": "REFUSED",
                "reason": f"{type(exc).__name__}: {exc}",
                "cells": {k: list(v) for k, v in interaction_cells.items()},
            }
    if fault_experiments:
        links["fault_experiments"] = {
            "status": "COMPLETED",
            "ids": fault_experiments,
            "note": "input stress trials are Fault Laboratory runs; see these experiments for the fault-lab analyses",
        }
    cfg = analysis_parameters(inv.id, design, baseline_run, trials, links, report)
    aexp = _new_experiment(
        registry,
        inv,
        f"analysis of {label}",
        "Stress analysis over stored trial evaluations",
        model,
        data,
        cfg,
    )
    aex = Executor(
        registry,
        store,
        source_root=source_root,
        adapters=executor.adapters,
        inputs_root=executor.inputs_root,
    )
    ares = aex.execute(
        aexp.id,
        resolve_procedure(PROCEDURE_ANALYSIS),
        seed=RUN_SEED,
        procedure_name=PROCEDURE_ANALYSIS,
    )
    found = [
        a
        for a in registry.find(StressAnalysis, spec_id=design.spec_id)
        if a.investigation_id == inv.id
    ]
    return StressRunResult(
        found[0].id if found else None,
        ares.run.id,
        ares.status,
        baseline_run,
        inv.id,
        tuple(report),
    )


# -- replay -----------------------------------------------------------------------------------------------------------


_VOLATILE = ("timings_seconds", "numpy_version", "python_version")


def replay_check(
    reg: Registry,
    store: ArtifactStore,
    executor: Executor,
    analysis_id: str,
    *,
    trials: bool = True,
) -> dict[str, object]:
    """Replay the analysis Run AND every completed trial Run as new runs; compare all seven documents,
    each trial's identity (unit, stress ids, parameters, seed) and its metric values within
    REPLAY_TOLERANCE, and its predictions digest where the stress records one (deterministic)."""
    a = reg.get(StressAnalysis, analysis_id)
    replay = executor.replay(a.run_id)
    out: dict[str, object] = {
        "analysis_id": a.id,
        "original_run": a.run_id,
        "replay_run": replay.run.id,
        "replay_status": replay.status.value,
        "tolerance": REPLAY_TOLERANCE,
    }
    if replay.status is not RunStatus.COMPLETED:
        return {
            **out,
            "deterministic": False,
            "differences": [f"the analysis replay ended {replay.status.value}"],
        }
    diffs: list[str] = []
    for name in DOCUMENTS:
        try:
            x = read_artifact(reg, store, a.run_id, f"stress/{name}.json")
        except ArtifactIntegrityError:
            raise
        except ExperionyxError:
            diffs.append(f"{name}: missing from the original run")
            continue
        _compare(name, x, read_artifact(reg, store, replay.run.id, f"stress/{name}.json"), diffs)
    replayed: list[dict[str, str]] = []
    if trials:
        for t in sorted(
            reg.find(StressTrial, analysis_id=a.id),
            key=lambda x: (x.point_index, x.repeat_index, x.cell),
        ):
            if t.status != "COMPLETED" or t.run_id is None:
                continue
            r = executor.replay(t.run_id)
            replayed.append(
                {
                    "unit_key": t.unit_key,
                    "original_run": t.run_id,
                    "replay_run": r.run.id,
                    "status": r.status.value,
                }
            )
            if r.status is not RunStatus.COMPLETED:
                diffs.append(f"trial {t.unit_key}: replay ended {r.status.value}")
                continue
            path = "fault/fault.json" if t.origin == "FAULT_LABORATORY" else "stress/stress.json"
            x, y = _read_json(reg, store, t.run_id, path), _read_json(reg, store, r.run.id, path)
            _compare(
                f"trial {t.unit_key} {path}",
                {k: v for k, v in x.items() if k not in _VOLATILE and k != "timings_seconds"},
                {k: v for k, v in y.items() if k not in _VOLATILE and k != "timings_seconds"},
                diffs,
            )
            ex, ey = load_evaluation(reg, store, t.run_id), load_evaluation(reg, store, r.run.id)
            ma = {m.metric_id: m.value for m in ex.metrics if m.structured is None}
            mb = {m.metric_id: m.value for m in ey.metrics if m.structured is None}
            _compare(f"trial {t.unit_key} metrics", ma, mb, diffs)
    return {
        **out,
        "deterministic": not diffs,
        "differences": diffs[:50],
        "compared": {"documents": list(DOCUMENTS), "trials_replayed": replayed},
    }


__all__ = [
    "DOCUMENTS",
    "PROCEDURE_ANALYSIS",
    "PROCEDURE_TRIAL",
    "StressBaselineError",
    "StressRunResult",
    "compute",
    "preflight",
    "replay_check",
    "run_stress_analysis",
    "run_stress_evaluation",
    "run_stress_experiment",
    "validate_baseline",
]

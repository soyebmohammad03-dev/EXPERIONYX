"""The calibration & uncertainty procedure, run by the normal execution engine so each analysis is a
Run with provenance, artifacts and observations and can be replayed.

Inputs are STORED predictions of finished runs (the baseline evaluation, optionally a calibration run,
and the completed trials of referenced stress analyses); the model is never loaded, called or changed.
Data that cannot be used is surfaced as evidence (invalid probabilities, duplicate sample IDs, missing
scores, insufficient samples) and never silently dropped or repaired."""

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from experionyx.adapters.capabilities import TaskType
from experionyx.artifacts import ArtifactStore
from experionyx.calibration import analysis as an
from experionyx.calibration import measures as m
from experionyx.calibration.entities import CalibrationAnalysis, CalibrationResult
from experionyx.calibration.spec import CalibrationSpec
from experionyx.data_quality.entities import QualityAnalysis
from experionyx.domain import (
    ConfigurationRef,
    EpistemicKind,
    Experiment,
    ExperimentStatus,
    ModelRef,
    Run,
    RunStatus,
    to_jsonable,
)
from experionyx.drift.data import order_keys, resolve_window
from experionyx.errors import ArtifactIntegrityError, ExperionyxError, ValidationError
from experionyx.evaluation.config import _thaw
from experionyx.evaluation.loading import load_evaluation
from experionyx.execution import Executor, RunContext, resolve_procedure
from experionyx.faults.report import read_artifact
from experionyx.hashing import content_hash
from experionyx.interactions.lifecycle import REPLAY_TOLERANCE, _compare
from experionyx.interactions.samples import AlignmentError
from experionyx.interactions.samples import _rows as prediction_rows
from experionyx.registry import Registry
from experionyx.slices.data import Baseline, SliceDataError, feature_columns, sample_table
from experionyx.slices.evaluate import MembershipStatus, evaluate
from experionyx.stress.entities import StressAnalysis, StressTrial

PROCEDURE = "experionyx.calibration.engine:run_calibration_analysis"
ARTIFACT_DIR = "calibration"
DOCUMENTS = ("spec", "predictions", "bins", "metrics", "uncertainty", "comparisons", "candidates", "summary")  # fmt: skip
ANALYSIS_VERSION = "1"
BASELINE, CALIBRATED = "baseline", "calibrated"
MAX_LISTED = 50


class CalibrationError(ExperionyxError):
    """The requested analysis cannot be made from the stored evidence (reason attached)."""


# -- reading a run's stored predictions ---------------------------------------------------------------------


@dataclass
class Loaded:
    run_id: str
    classes: tuple[Any, ...]
    representation: str  # PREDICT_PROBA | SOFTMAX_LOGITS
    split: str | None
    dataset_id: str | None
    dataset_fingerprint: str | None
    model_id: str | None
    model_fingerprint: str | None
    evaluation_config_hash: str
    n_rows: int
    excluded_by_evaluation: int
    obs: dict[int, m.Obs]
    invalid: list[dict[str, Any]]  # rows that could not be used, with the reason
    duplicates: list[int]  # IDs listed more than once (all their rows are set aside)
    base: Baseline  # the stored rows (without duplicates), for slice / window field access
    rows: dict[int, dict[str, Any]] = field(default_factory=dict)

    @property
    def digest(self) -> str:
        return content_hash({"rows": [[i, r["true"], r["predicted"], _plain(r.get("scores"))] for i, r in sorted(self.rows.items())]})  # fmt: skip


def _plain(x: Any) -> Any:
    """A stored score vector in a hashable, JSON-safe form: non-finite numbers (which invalid rows may
    contain) become their names, so a damaged row is hashed, not crashed on."""
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    return [_plain(v) for v in x] if isinstance(x, list | tuple) else x


def representation_of(source: str | None) -> str | None:
    if source is None:
        return None
    return "PREDICT_PROBA" if source.startswith("predict_proba") else "SOFTMAX_LOGITS" if source.startswith("softmax") else None  # fmt: skip


def load_run(reg: Registry, store: ArtifactStore, run_id: str) -> Loaded:
    """A run's usable observations plus the evidence about the unusable ones. Raises CalibrationError
    for a model output calibration is not defined for (regression, no scores, unknown score source)."""
    ev = load_evaluation(reg, store, run_id)
    if ev.task is not TaskType.CLASSIFICATION:
        raise CalibrationError(f"UNAVAILABLE: run {run_id} is a {ev.task.value} evaluation; the model exposes no documented uncertainty representation for it, so none is fabricated")  # fmt: skip
    rep = representation_of(ev.score_source)
    if rep is None:
        raise CalibrationError(f"UNAVAILABLE: run {run_id} stored no probability scores ({ev.confidence.reason or 'the model exposes no probabilities'}); calibration needs class probabilities")  # fmt: skip
    if ev.classes is None:
        raise CalibrationError(f"UNAVAILABLE: run {run_id} does not declare its class order")
    try:
        raw = list(prediction_rows(reg, store, run_id))
    except AlignmentError as exc:
        raise CalibrationError(f"UNAVAILABLE: {exc}") from exc
    seen: dict[Any, int] = {}
    for r in raw:
        seen[r["index"]] = seen.get(r["index"], 0) + 1
    dups = sorted(i for i, c in seen.items() if c > 1 and isinstance(i, int) and not isinstance(i, bool))  # fmt: skip
    obs: dict[int, m.Obs] = {}
    invalid: list[dict[str, Any]] = []
    rows: dict[int, dict[str, Any]] = {}
    for r in raw:
        i = r["index"]
        if isinstance(i, bool) or not isinstance(i, int):
            invalid.append({"id": i, "reason": "NON_INTEGER_SAMPLE_ID"})
            continue
        if i in dups:
            continue
        rows[i] = r
        o = m.to_obs(i, r["true"], r["predicted"], r.get("scores"), ev.classes)
        if isinstance(o, str):
            invalid.append({"id": i, "reason": o})
        else:
            obs[i] = o
    invalid.sort(key=lambda x: (str(x["id"]), x["reason"]))
    base = Baseline(run_id, ev.task, ev.classes, ev.context.split, ev.context.dataset_id, ev.context.dataset_fingerprint, content_hash(to_jsonable(ev.config)), rows)  # fmt: skip
    return Loaded(run_id, tuple(ev.classes), rep, ev.context.split, ev.context.dataset_id, ev.context.dataset_fingerprint, ev.context.model_id, ev.context.model_fingerprint, base.evaluation_config_hash, len(raw), ev.excluded.count, obs, invalid, dups, base, rows)  # fmt: skip


def dq_block(ld: Loaded, obs: Sequence[m.Obs] | None = None, n_members: int | None = None) -> dict[str, Any]:  # fmt: skip
    """What was wrong with the data behind a context, as evidence."""
    by: dict[str, int] = {}
    for x in ld.invalid:
        by[x["reason"]] = by.get(x["reason"], 0) + 1
    usable = len(ld.obs) if obs is None else len(obs)
    members = ld.n_rows if n_members is None else n_members
    return {
        "n_rows": ld.n_rows,
        "n_members": members,
        "n_usable": usable,
        "n_unusable_members": members - usable
        if obs is not None
        else len(ld.invalid) + len(ld.duplicates),
        "invalid_by_reason": dict(sorted(by.items())),
        "invalid_examples": ld.invalid[:MAX_LISTED],
        "n_duplicate_ids": len(ld.duplicates),
        "duplicate_ids": ld.duplicates[:MAX_LISTED],
        "excluded_by_evaluation": ld.excluded_by_evaluation,
        "affected": bool(ld.invalid or ld.duplicates or ld.excluded_by_evaluation),
    }


# -- the post-hoc calibration protocol ----------------------------------------------------------------------


def split_ids(ids: Sequence[int], fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Deterministic calibration / evaluation partition of sample IDs: order by a seeded hash of the
    ID (independent of row order), the first `fraction` calibrate, the rest evaluate."""
    ranked = sorted(ids, key=lambda i: (hashlib.sha256(f"{seed}:{i}".encode()).hexdigest(), i))
    k = min(max(round(fraction * len(ranked)), 1), len(ranked) - 1)
    return sorted(ranked[:k]), sorted(ranked[k:])


def check_calibration_data(base: Loaded, cal: Loaded) -> None:
    """Refuse a calibration run that is not usable, or that IS the evaluation data (leakage)."""
    if cal.run_id == base.run_id:
        raise CalibrationError("LEAKAGE: the calibration run is the evaluated baseline run")
    if (cal.dataset_fingerprint, cal.split) == (base.dataset_fingerprint, base.split):
        raise CalibrationError(f"LEAKAGE: the calibration run {cal.run_id} covers the same dataset fingerprint and split ({base.split!r}) as the evaluation, so the calibrator would be fitted on the samples it is evaluated on")  # fmt: skip
    for label, a, b in (("model fingerprint", base.model_fingerprint, cal.model_fingerprint), ("classes", base.classes, cal.classes), ("prediction representation", base.representation, cal.representation)):  # fmt: skip
        if a != b:
            raise CalibrationError(f"the calibration run differs from the baseline in {label}: {b!r} vs {a!r}")  # fmt: skip


def fit_calibrator(method: str, obs: Sequence[m.Obs], min_samples: int) -> tuple[dict[str, Any] | None, str | None]:  # fmt: skip
    """(parameters, None) or (None, why not). A calibrator is not fitted on too little data or on a
    single outcome class."""
    pairs = m.top_pairs(obs)
    hits = sum(e for _, e in pairs)
    if len(pairs) < min_samples:
        return None, f"{len(pairs)} calibration observation(s) < min_samples={min_samples}"
    if hits in (0, len(pairs)):
        return None, "the calibration data has a single outcome (all predictions correct, or all incorrect): the calibrator is not identifiable"  # fmt: skip
    if method == "PLATT":
        p = m.fit_platt(pairs)
        return (p, None) if p["converged"] else (None, "Platt scaling did not converge")
    return m.fit_isotonic(pairs), None


def method_block(spec: CalibrationSpec, base: Loaded, cal_run: Loaded | None) -> tuple[dict[str, Any], list[m.Obs], list[m.Obs]]:  # fmt: skip
    """(record, evaluation obs raw, evaluation obs calibrated). The record carries the protocol, the
    identity of the calibration and evaluation data, the parameters and the leakage statement."""
    f, cfg = spec.fit, spec.statistics
    usable = sorted(base.obs)
    rec: dict[str, Any] = {"method": spec.method, "protocol": {"mode": f.mode}, "status": an.COMPUTED, "reason": None, "parameters": None, "model": {"fingerprint": base.model_fingerprint, "immutable": "the model is never loaded or modified: calibration is a separate function of the stored top-label confidence"}}  # fmt: skip
    if f.mode == "SPLIT":
        if len(usable) < 2:
            return {**rec, "status": an.INSUFFICIENT, "reason": "fewer than 2 usable observations to split"}, [], []  # fmt: skip
        cal_ids, ev_ids = split_ids(usable, f.fraction, f.seed)
        fit_obs = [base.obs[i] for i in cal_ids]
        rec["protocol"].update(fraction=f.fraction, seed=f.seed, calibration_data={"source": f"run {base.run_id}, split by seeded ID hash", "n": len(cal_ids), "ids_digest": an.digest_of(cal_ids), "dataset_fingerprint": base.dataset_fingerprint, "split": base.split}, evaluation_data={"source": f"run {base.run_id}, held-out part", "n": len(ev_ids), "ids_digest": an.digest_of(ev_ids), "dataset_fingerprint": base.dataset_fingerprint, "split": base.split}, overlap=len(set(cal_ids) & set(ev_ids)))  # fmt: skip
        rec["protocol"]["calibration_ids"], rec["protocol"]["evaluation_ids"] = cal_ids, ev_ids
        eval_obs = [base.obs[i] for i in ev_ids]
    else:
        assert cal_run is not None  # noqa: S101  # validated before execution
        check_calibration_data(base, cal_run)
        fit_obs = [cal_run.obs[i] for i in sorted(cal_run.obs)]
        eval_obs = [base.obs[i] for i in usable]
        rec["protocol"].update(calibration_data={"source": f"run {cal_run.run_id}", "n": len(fit_obs), "ids_digest": an.digest_of(sorted(cal_run.obs)), "dataset_fingerprint": cal_run.dataset_fingerprint, "split": cal_run.split, "predictions_digest": cal_run.digest}, evaluation_data={"source": f"run {base.run_id}", "n": len(eval_obs), "ids_digest": an.digest_of(usable), "dataset_fingerprint": base.dataset_fingerprint, "split": base.split, "predictions_digest": base.digest}, overlap=0)  # fmt: skip
    rec["protocol"]["leakage"] = "NONE: the calibrator is fitted only on the calibration data; every reported 'after' number uses evaluation data disjoint from it"  # fmt: skip
    params, why = fit_calibrator(spec.method, fit_obs, cfg.min_samples)
    if params is None:
        return {**rec, "status": an.INSUFFICIENT, "reason": why}, eval_obs, []
    rec["parameters"] = params
    calibrated = [replace(o, cal=m.apply_calibrator(spec.method, params, o.conf)) for o in eval_obs]
    return rec, eval_obs, calibrated


# -- stress trials ------------------------------------------------------------------------------------------


def stress_contexts(reg: Registry, store: ArtifactStore, spec: CalibrationSpec, base: Loaded) -> list[dict[str, Any]]:  # fmt: skip
    """One entry per trial of every referenced stress analysis: {key, trial, loaded | reason}."""
    out: list[dict[str, Any]] = []
    for xid in spec.stress_analyses:
        xa = reg.get(StressAnalysis, xid)
        trials = sorted(reg.find(StressTrial, analysis_id=xid), key=lambda t: (t.point_index, t.repeat_index, t.cell, t.id))  # fmt: skip
        for t in trials:
            lineage = {"stress_analysis_id": xid, "trial_id": t.id, "unit_key": t.unit_key, "stress_ids": list(t.stress_ids), "family": t.family, "origin": t.origin, "seed": t.seed, "point_index": t.point_index, "repeat_index": t.repeat_index, "cell": t.cell, "trial_status": t.status, "trial_run_id": t.run_id, "stress_analysis_fingerprint": xa.provenance_fingerprint}  # fmt: skip
            entry: dict[str, Any] = {"key": f"stress:{t.unit_key}", "lineage": lineage, "loaded": None, "reason": None}  # fmt: skip
            if t.status != "COMPLETED" or t.run_id is None:
                entry["reason"] = f"the stress trial is {t.status}" + (f": {t.reason}" if t.reason else "")  # fmt: skip
            else:
                try:
                    ld = load_run(reg, store, t.run_id)
                    # the dataset (input stress) or model (parameter stress) is DELIBERATELY derived, so
                    # their fingerprints are recorded as the trial's own, not required to equal the baseline's
                    lineage.update(trial_dataset_fingerprint=ld.dataset_fingerprint, trial_model_fingerprint=ld.model_fingerprint)  # fmt: skip
                    if ld.classes != base.classes or ld.representation != base.representation:
                        entry["reason"] = "the trial's classes or score representation differ from the baseline"  # fmt: skip
                    else:
                        entry["loaded"] = ld
                except (ExperionyxError, KeyError) as exc:
                    entry["reason"] = f"the trial's predictions cannot be read: {exc}"
            out.append(entry)
    return out


# -- computation --------------------------------------------------------------------------------------------


@dataclass
class Computed:
    base: Loaded
    results: dict[str, dict[str, Any]]  # context key -> context result
    method: dict[str, Any] | None
    comparisons: dict[str, dict[str, Any]]
    corrections: dict[str, Any]
    docs: dict[str, Any]
    summary: dict[str, Any]
    fingerprint: str
    partial: bool
    lineage: dict[str, Any] = field(default_factory=dict)


def _subset(ld: Loaded, ids: Sequence[int]) -> list[m.Obs]:
    return [ld.obs[i] for i in sorted(ids) if i in ld.obs]


def _ctx(ld: Loaded, spec: CalibrationSpec, key: str, kind: str, ids: Sequence[int] | None = None, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:  # fmt: skip
    members = sorted(ld.rows) if ids is None else sorted(ids)
    obs = _subset(ld, members)
    dq = dq_block(ld, obs, len(members)) if ids is not None else dq_block(ld)
    return an.context_result(obs, spec, ld.classes, key=key, kind=kind, n_members=len(members), dq=dq, extra=extra)  # fmt: skip


def check_expectations(spec: CalibrationSpec, base: Loaded) -> None:
    if spec.prediction_source != base.representation:
        raise CalibrationError(f"the spec declares prediction_source {spec.prediction_source} but run {base.run_id} stored {base.representation} scores: scores are never treated as another representation")  # fmt: skip
    for label, want, got in (("model_id", spec.model_id, base.model_id), ("dataset_id", spec.dataset_id, base.dataset_id), ("split", spec.split, base.split)):  # fmt: skip
        if want is not None and want != got:
            raise CalibrationError(f"the spec expects {label} {want!r} but the baseline run has {got!r}")  # fmt: skip


def compute(reg: Registry, store: ArtifactStore, spec: CalibrationSpec, dataset: Any | None) -> Computed:  # fmt: skip
    """Every number of an analysis, deterministic given the registry content (reads only)."""
    base = load_run(reg, store, spec.baseline_run)
    check_expectations(spec, base)
    cfg = spec.statistics
    results: dict[str, dict[str, Any]] = {BASELINE: _ctx(base, spec, BASELINE, "BASELINE")}
    all_obs = [base.obs[i] for i in sorted(base.obs)]
    comps: dict[str, dict[str, Any]] = {}
    lineage: dict[str, Any] = {"calibration_run": None, "stress": {}, "quality": {}, "windows": {}, "slices": {}}  # fmt: skip

    # post-hoc calibration: separate calibration and evaluation data, before vs after ---------------
    method: dict[str, Any] | None = None
    if spec.method != "NONE":
        cal_run = None
        if spec.fit.mode == "RUN":
            cal_run = load_run(reg, store, str(spec.fit.calibration_run))
            lineage["calibration_run"] = {"run_id": cal_run.run_id, "predictions_digest": cal_run.digest, "dataset_fingerprint": cal_run.dataset_fingerprint, "split": cal_run.split}  # fmt: skip
        method, raw_eval, cal_eval = method_block(spec, base, cal_run)
        if method["status"] == an.COMPUTED:
            dq = dq_block(base, raw_eval, len(raw_eval))
            before = an.context_result(raw_eval, spec, base.classes, key="calibration_before", kind="BASELINE", n_members=len(raw_eval), dq=dq)  # fmt: skip
            after = an.context_result(cal_eval, spec, base.classes, key=CALIBRATED, kind="CALIBRATED", n_members=len(cal_eval), dq=dq, calibrated=True, extra={"method": spec.method})  # fmt: skip
            method.update(before=before, after=after)
            results[CALIBRATED] = after
            comps["before vs after calibration"] = an.compare_contexts(raw_eval, cal_eval, spec, key_a="calibration_before", key_b=CALIBRATED, family="POST_HOC_CALIBRATION")  # fmt: skip
        else:
            results[CALIBRATED] = {"key": CALIBRATED, "kind": "CALIBRATED", "status": an.INSUFFICIENT, "reason": method["reason"], "n_samples": 0, "sample_digest": an.digest_of([]), "n_members": 0}  # fmt: skip

    # slices (only the explicitly requested ones) ---------------------------------------------------
    for s in spec.slices:
        key = f"slice:{s.name}"
        try:
            mem = evaluate(s, sample_table(base.base, dataset, s.fields))
        except SliceDataError as exc:
            results[key] = {"key": key, "kind": "SLICE", "status": an.UNAVAILABLE, "reason": str(exc), "n_samples": 0, "sample_digest": an.digest_of([]), "n_members": 0}  # fmt: skip
            continue
        if mem.status is not MembershipStatus.COMPUTED:
            results[key] = {"key": key, "kind": "SLICE", "status": an.UNAVAILABLE if mem.status is MembershipStatus.MISSING_FIELD else an.INSUFFICIENT, "reason": mem.reason or f"membership is {mem.status.value}", "n_samples": 0, "sample_digest": an.digest_of([]), "n_members": 0}  # fmt: skip
            continue
        ids = [int(i) for i in mem.sample_ids]
        results[key] = _ctx(base, spec, key, "SLICE", ids, {"slice_id": s.slice_id, "membership": {"digest": mem.membership_digest, "n_members": mem.n_members, "n_unknown": mem.n_unknown, "prevalence": mem.prevalence, "sample_ids": ids}})  # fmt: skip
        lineage["slices"][s.name] = {"slice_id": s.slice_id, "membership_digest": mem.membership_digest}  # fmt: skip
        rest = [base.obs[i] for i in sorted(base.obs) if i not in set(ids)]
        comps[f"{key} vs REST"] = an.compare_contexts(_subset(base, ids), rest, spec, key_a=key, key_b="REST", family="SLICE")  # fmt: skip

    # temporal / distribution windows ----------------------------------------------------------------
    if spec.windows is not None:
        ws = spec.windows
        order: dict[int, float] = {}
        excluded: dict[str, list[int]] = {}
        window_ids: dict[str, list[int]] = {}
        try:
            raw = {}
            if ws.ordering.field != "index":
                if dataset is None:
                    raise SliceDataError(f"the ordering {ws.ordering.field!r} needs the dataset, which is not available")  # fmt: skip
                raw = feature_columns(dataset, base.split, [ws.ordering.field], set(base.rows), raw=True)[ws.ordering.field]  # fmt: skip
            order, excluded = order_keys(sorted(base.rows), ws.ordering.field, raw)
            for w in ws.windows:
                rw = resolve_window(w, ws.ordering.field, order)
                window_ids[str(w.name)] = list(rw.sample_ids)
                key = f"window:{w.name}"
                results[key] = _ctx(base, spec, key, "WINDOW", rw.sample_ids, {"membership": {"digest": rw.digest, "n_members": rw.n, "sample_ids": list(rw.sample_ids)}, "window": {"window_id": rw.window_id, "definition": w.to_dict(), "ordering": ws.ordering.field, "sample_digest": rw.digest, "order_min": rw.order_min, "order_max": rw.order_max, "n_excluded_no_key": {k: len(v) for k, v in excluded.items() if v}}})  # fmt: skip
                lineage["windows"][str(w.name)] = {
                    "window_id": rw.window_id,
                    "sample_digest": rw.digest,
                }
        except SliceDataError as exc:
            for w in ws.windows:
                results[f"window:{w.name}"] = {"key": f"window:{w.name}", "kind": "WINDOW", "status": an.UNAVAILABLE, "reason": str(exc), "n_samples": 0, "sample_digest": an.digest_of([]), "n_members": 0}  # fmt: skip
        for a, b in ws.pairs():
            if a in window_ids and b in window_ids:
                comps[f"window:{a} vs window:{b}"] = an.compare_contexts(_subset(base, window_ids[a]), _subset(base, window_ids[b]), spec, key_a=f"window:{a}", key_b=f"window:{b}", family="WINDOW")  # fmt: skip
            else:
                comps[f"window:{a} vs window:{b}"] = {"a": f"window:{a}", "b": f"window:{b}", "family": "WINDOW", "status": an.UNAVAILABLE, "reason": "a window could not be resolved", "metrics": {}}  # fmt: skip

    # stress conditions -------------------------------------------------------------------------------
    for e in stress_contexts(reg, store, spec, base):
        ld = e["loaded"]
        if ld is None:
            results[e["key"]] = {"key": e["key"], "kind": "STRESS", "status": an.UNAVAILABLE, "reason": e["reason"], "n_samples": 0, "sample_digest": an.digest_of([]), "n_members": 0, "lineage": e["lineage"]}  # fmt: skip
            continue
        results[e["key"]] = _ctx(ld, spec, e["key"], "STRESS", None, {"lineage": e["lineage"], "predictions_digest": ld.digest})  # fmt: skip
        lineage["stress"][e["key"]] = {**e["lineage"], "predictions_digest": ld.digest}
        comps[f"baseline vs {e['key']}"] = an.compare_contexts(all_obs, [ld.obs[i] for i in sorted(ld.obs)], spec, key_a=BASELINE, key_b=e["key"], family="STRESS")  # fmt: skip
    for qid in spec.quality_analyses:
        qa = reg.get(QualityAnalysis, qid)
        lineage["quality"][qid] = {"provenance_fingerprint": qa.provenance_fingerprint, "analysis_status": qa.analysis_status}  # fmt: skip
    if len(comps) > 200:
        raise CalibrationError(f"{len(comps)} comparisons exceed the limit of 200")
    corrections = an.correct_families(comps, spec)
    cands = an.candidates(all_obs, spec)
    return _assemble(spec, base, results, method, comps, corrections, cands, lineage, cfg.seed)


def _assemble(spec: CalibrationSpec, base: Loaded, results: dict[str, dict[str, Any]], method: dict[str, Any] | None, comps: dict[str, dict[str, Any]], corrections: dict[str, Any], cands: dict[str, Any], lineage: dict[str, Any], seed: int) -> Computed:  # fmt: skip
    ctx_docs = results
    rows = []
    part = {}
    if method and method.get("protocol", {}).get("calibration_ids") is not None:
        part = dict.fromkeys(method["protocol"]["calibration_ids"], "calibration") | dict.fromkeys(method["protocol"]["evaluation_ids"], "evaluation")  # fmt: skip
    cal_conf = {}
    if method and method.get("after") is not None:
        cal = spec_calibrated_confidences(spec, method, base)
        cal_conf = cal
    for i in sorted(base.rows):
        r = base.rows[i]
        o = base.obs.get(i)
        row: dict[str, Any] = {"id": i, "true": r["true"], "predicted": r["predicted"], "usable": o is not None, "confidence": None if o is None else o.conf, "correct": None if o is None else o.correct}  # fmt: skip
        if o is not None:
            row.update(entropy_nats=m.entropy(o.probs), normalized_entropy=m.normalized_entropy(o.probs), top2_margin=m.margin(o.probs), scores=list(o.probs))  # fmt: skip
        if part:
            row["calibration_part"] = part.get(i)
        if i in cal_conf:
            row["calibrated_confidence"] = cal_conf[i]
        rows.append(row)
    docs: dict[str, Any] = {}
    docs["predictions"] = {"classes": list(base.classes), "prediction_representation": base.representation, "rows": rows, "invalid": base.invalid, "duplicate_ids": base.duplicates, "contexts": {k: {"sample_digest": r.get("sample_digest"), "n_samples": r.get("n_samples"), **({"sample_ids": r["membership"]["sample_ids"]} if r.get("membership") else {})} for k, r in results.items()}}  # fmt: skip
    docs["bins"] = {k: r.get("bins") for k, r in ctx_docs.items() if r.get("bins")}
    docs["metrics"] = {"contexts": {k: {"status": r["status"], "reason": r.get("reason"), "n_samples": r.get("n_samples"), "n_members": r.get("n_members"), "binning": r.get("binning"), "metrics": r.get("metrics"), "data_quality": r.get("data_quality"), "kind": r.get("kind")} for k, r in ctx_docs.items()}, "calibration_method": None if method is None else {k: v for k, v in method.items() if k not in ("before", "after")} | {"before": _lite(method.get("before")), "after": _lite(method.get("after"))}, "definitions": DEFINITIONS}  # fmt: skip
    docs["uncertainty"] = {"contexts": {k: {"status": r["status"], "uncertainty": r.get("uncertainty"), "prediction_uncertainty": r.get("prediction_uncertainty")} for k, r in ctx_docs.items()}, "note": "statistical uncertainty of the metrics (intervals) and predictive dispersion are separate things; neither is an epistemic/aleatoric claim"}  # fmt: skip
    docs["comparisons"] = {"comparisons": comps, "corrections": corrections, "note": an.NOTE}
    docs["candidates"] = cands
    fp = content_hash({"spec_id": spec.spec_id, "baseline": {"run_id": base.run_id, "dataset_fingerprint": base.dataset_fingerprint, "split": base.split, "evaluation_config_hash": base.evaluation_config_hash, "model_fingerprint": base.model_fingerprint, "representation": base.representation, "predictions_digest": base.digest}, "lineage": lineage, "contexts": {k: r.get("sample_digest") for k, r in sorted(results.items())}, "method": None if method is None else {"protocol": {k: v for k, v in method["protocol"].items() if k not in ("calibration_ids", "evaluation_ids")}, "parameters": method.get("parameters")}, "analysis_version": ANALYSIS_VERSION})  # fmt: skip
    counts: dict[str, int] = {}
    for c in comps.values():
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    b = results[BASELINE]
    summary = {
        "baseline_run_id": base.run_id,
        "dataset_fingerprint": base.dataset_fingerprint,
        "split": base.split,
        "prediction_representation": base.representation,
        "n_rows": base.n_rows,
        "n_usable": len(base.obs),
        "binning": spec.binning.to_dict(),
        "objects": list(spec.objects),
        "baseline": {
            "status": b["status"],
            "reason": b.get("reason"),
            "metrics": (b.get("metrics") or {}).get("top_label"),
            "probability_vector": (b.get("metrics") or {}).get("probability_vector"),
        },
        "contexts": {
            k: {
                "kind": r.get("kind"),
                "status": r["status"],
                "n_samples": r.get("n_samples"),
                "reason": r.get("reason"),
                "headline": (r.get("metrics") or {}).get("top_label"),
            }
            for k, r in sorted(results.items())
        },
        "comparison_status_counts": dict(sorted(counts.items())),
        "corrections": corrections,
        "method": None
        if method is None
        else {
            "method": method["method"],
            "status": method["status"],
            "reason": method["reason"],
            "protocol": {
                k: v
                for k, v in method["protocol"].items()
                if k not in ("calibration_ids", "evaluation_ids")
            },
            "parameters": method.get("parameters"),
        },
        "data_quality": dq_block(base),
        "candidate_counts": {
            "high_confidence_incorrect": len(cands["high_confidence_incorrect"]),
            "low_confidence_correct": len(cands["low_confidence_correct"]),
        },
        "prediction_uncertainty": {k: v["status"] for k, v in UNSUPPORTED.items()},
        "note": an.NOTE,
    }
    partial = (
        any(r["status"] != an.COMPUTED for r in results.values())
        or any(c["status"] != an.COMPUTED for c in comps.values())
        or (method is not None and method["status"] != an.COMPUTED)
        or bool(base.invalid or base.duplicates)
    )
    docs["summary"] = summary
    return Computed(base, results, method, comps, corrections, docs, summary, fp, partial, lineage)


UNSUPPORTED = an.UNSUPPORTED_UNCERTAINTY

DEFINITIONS = {
    "top_label": "confidence = probability of the PREDICTED class; event = prediction correct",
    "classwise": "per class k: p_k against the event 'true class is k'",
    "probability_vector": "multiclass Brier and negative log-likelihood of the whole vector (scoring rules, not curves)",
    "bins": "[lower, upper), last bin closed; membership by bisection on the reported edges; empty bins are listed",
    "ece": "sum_b (n_b/n) |accuracy_b - mean_confidence_b| over non-empty bins",
    "mce": "max_b |accuracy_b - mean_confidence_b| over non-empty bins",
    "brier_top_label": "mean (confidence - correct)^2",
    "log_loss_top_label": "-mean [c ln(conf) + (1-c) ln(1-conf)], probabilities clipped to [1e-15, 1-1e-15]",
    "brier_multiclass": "mean over samples of sum_k (p_k - 1[y=k])^2",
    "nll_multiclass": "-mean ln p_y (clipped)",
    "caveat": "ECE and MCE depend on the binning and the sample; no value here is a universal calibration or uncertainty score",
}


def _lite(r: dict[str, Any] | None) -> dict[str, Any] | None:
    return None if r is None else {k: r.get(k) for k in ("key", "kind", "status", "reason", "n_samples", "sample_digest", "metrics")}  # fmt: skip


def spec_calibrated_confidences(spec: CalibrationSpec, method: dict[str, Any], base: Loaded) -> dict[int, float]:  # fmt: skip
    ids = method["protocol"].get("evaluation_ids")
    if ids is None:
        ids = sorted(base.obs)
    return {i: m.apply_calibrator(spec.method, method["parameters"], base.obs[i].conf) for i in ids if i in base.obs}  # fmt: skip


# -- the procedure -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationRequest:
    investigation_id: str
    spec: Mapping[str, object]

    def to_parameters(self) -> dict[str, object]:
        return {"calibration_analysis": {"investigation_id": self.investigation_id, "spec": dict(self.spec)}}  # fmt: skip

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, object]) -> "CalibrationRequest":
        if set(parameters) != {"calibration_analysis"}:
            raise ValidationError("a calibration configuration must be exactly {'calibration_analysis': ...}")  # fmt: skip
        body = _thaw(parameters["calibration_analysis"])
        if not isinstance(body, dict) or set(body) != {"investigation_id", "spec"} or not isinstance(body["spec"], dict):  # fmt: skip
            raise ValidationError("'calibration_analysis' has the wrong fields")
        return cls(str(body["investigation_id"]), body["spec"])


def _write(ctx: RunContext, name: str, payload: object) -> str:
    directory = ctx.artifact_dir / ARTIFACT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(json.dumps(to_jsonable(payload), sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")  # fmt: skip
    return ctx.register_artifact(f"{ARTIFACT_DIR}/{name}.json", name=f"calibration-{name}", media_type="application/json").id  # fmt: skip


def run_calibration_analysis(ctx: RunContext) -> None:
    req = CalibrationRequest.from_parameters(ctx.parameters)
    spec = CalibrationSpec.from_dict(req.spec)
    reg, now = ctx.registry, ctx.started_at
    c = compute(reg, ctx.store, spec, ctx.dataset)
    if ctx.dataset is not None and c.base.dataset_fingerprint is not None and ctx.dataset.fingerprint() != c.base.dataset_fingerprint:  # fmt: skip
        raise CalibrationError("the loaded dataset's fingerprint differs from the baseline evaluation's")  # fmt: skip
    spec_doc = {
        "spec_id": spec.spec_id,
        "spec": spec.to_dict(),
        "provenance_fingerprint": c.fingerprint,
        "model": {"model_id": c.base.model_id, "fingerprint": c.base.model_fingerprint},
        "dataset": {"dataset_id": c.base.dataset_id, "fingerprint": c.base.dataset_fingerprint},
        "split": c.base.split,
        "evaluation_config_hash": c.base.evaluation_config_hash,
        "prediction_representation": c.base.representation,
        "score_assumption": "scores are the stored predict_proba output"
        if c.base.representation == "PREDICT_PROBA"
        else "scores are softmax of stored model outputs, ASSUMING those outputs are logits",
        "target_representation": {"target": spec.target, "classes": list(c.base.classes)},
        "binning": spec.binning.to_dict(),
        "calibration_method": spec.method,
        "calibration_split": None if spec.method == "NONE" else spec.fit.to_dict(),
        "bootstrap": spec.statistics.to_dict(),
        "lineage": c.lineage,
        "source_runs": {"baseline": spec.baseline_run, "predictions_digest": c.base.digest},
        "analysis_version": ANALYSIS_VERSION,
    }
    art = {"spec": _write(ctx, "spec", spec_doc)}
    for name in DOCUMENTS[1:]:
        art[name] = _write(ctx, name, c.docs[name])
    analysis = CalibrationAnalysis(
        req.investigation_id, ctx.run.id, spec.spec_id, spec.to_dict(), spec.baseline_run,
        str(c.base.dataset_fingerprint), c.fingerprint, "PARTIAL" if c.partial else "COMPLETE",
        {**c.summary, "artifacts": art}, now,
    )  # fmt: skip
    new = not reg.exists(CalibrationAnalysis, analysis.id)
    if new:
        with reg.transaction():
            reg.add(analysis)
            for key, r in sorted(c.results.items()):
                head = (r.get("metrics") or {}).get("top_label") or {}
                reg.add(CalibrationResult(analysis.id, key, r["kind"], int(r.get("n_samples") or 0), r["status"], r.get("reason"), head, r["sample_digest"], now))  # fmt: skip
    ctx.observe("calibration.new_record", int(new), kind=EpistemicKind.OBSERVATION)
    ctx.observe("calibration.contexts", len(c.results), kind=EpistemicKind.OBSERVATION)
    ctx.observe("calibration.comparisons", len(c.comparisons), kind=EpistemicKind.DERIVED_METRIC)


# -- entry points ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationRunResult:
    experiment_id: str
    run_id: str
    status: RunStatus
    analysis_id: str | None


def validate(reg: Registry, store: ArtifactStore, spec: CalibrationSpec) -> Loaded:
    """Refuse before anything is created: the baseline must be a completed classification evaluation
    with stored probability scores of the declared representation; a calibration run must not be the
    evaluation data; every referenced stress / quality analysis must exist and fit the baseline."""
    run = reg.get(Run, spec.baseline_run)
    if run.status is not RunStatus.COMPLETED:
        raise ValidationError(f"baseline run {run.id} is {run.status.value}, not COMPLETED")
    base = load_run(reg, store, spec.baseline_run)
    check_expectations(spec, base)
    if spec.method != "NONE" and spec.fit.mode == "RUN":
        check_calibration_data(base, load_run(reg, store, str(spec.fit.calibration_run)))
    for xid in spec.stress_analyses:
        xa = reg.get(StressAnalysis, xid)
        if xa.baseline_run_id != spec.baseline_run:
            raise ValidationError(f"stress analysis {xid} was made over baseline {xa.baseline_run_id}, not {spec.baseline_run}")  # fmt: skip
    for qid in spec.quality_analyses:
        reg.get(QualityAnalysis, qid)
    return base


def run_calibration_request(
    registry: Registry, store: ArtifactStore, executor: Executor, investigation_id: str,
    spec: CalibrationSpec, *, seed: int = 0,
) -> CalibrationRunResult:  # fmt: skip
    validate(registry, store, spec)
    base_run = registry.get(Run, spec.baseline_run)
    base_exp = registry.get(Experiment, base_run.experiment_id)
    conf = ConfigurationRef(CalibrationRequest(investigation_id, spec.to_dict()).to_parameters())
    if not registry.exists(ConfigurationRef, conf.id):
        registry.add(conf)
    exp = Experiment(
        investigation_id, "calibration analysis",
        "Deterministic calibration and uncertainty analysis over stored predictions",
        ModelRef("none", "0"), base_exp.dataset, conf.id, datetime.now(UTC),
    )  # fmt: skip
    if not registry.exists(Experiment, exp.id):
        registry.add(exp)
        registry.update_status(exp.with_status(ExperimentStatus.READY))
    result = executor.execute(exp.id, resolve_procedure(PROCEDURE), seed=seed, procedure_name=PROCEDURE)  # fmt: skip
    found = [a for a in registry.find(CalibrationAnalysis, spec_id=spec.spec_id) if a.investigation_id == investigation_id]  # fmt: skip
    return CalibrationRunResult(exp.id, result.run.id, result.status, found[0].id if found else None)  # fmt: skip


def replay_check(reg: Registry, store: ArtifactStore, executor: Executor, analysis_id: str) -> dict[str, object]:  # fmt: skip
    """Replay the analysis Run as a NEW run and compare every stored document within REPLAY_TOLERANCE."""
    a = reg.get(CalibrationAnalysis, analysis_id)
    replay = executor.replay(a.run_id)
    out: dict[str, object] = {"analysis_id": a.id, "original_run": a.run_id, "replay_run": replay.run.id, "replay_status": replay.status.value, "tolerance": REPLAY_TOLERANCE}  # fmt: skip
    if replay.status is not RunStatus.COMPLETED:
        return {**out, "deterministic": False, "differences": [f"replay run ended {replay.status.value}"]}  # fmt: skip
    diffs: list[str] = []
    for name in DOCUMENTS:
        try:
            x = read_artifact(reg, store, a.run_id, f"{ARTIFACT_DIR}/{name}.json")
        except ArtifactIntegrityError:
            raise
        except ExperionyxError:
            diffs.append(f"{name}: missing from the original run")
            continue
        _compare(name, x, read_artifact(reg, store, replay.run.id, f"{ARTIFACT_DIR}/{name}.json"), diffs)  # fmt: skip
    return {**out, "deterministic": not diffs, "differences": diffs[:50], "compared": list(DOCUMENTS)}  # fmt: skip


__all__ = ["PROCEDURE", "CalibrationError", "CalibrationRunResult", "compute", "replay_check", "run_calibration_analysis", "run_calibration_request", "validate"]  # fmt: skip

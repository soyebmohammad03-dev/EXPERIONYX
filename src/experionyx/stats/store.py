"""Run a statistical analysis on persisted evidence and register it. Inputs come from an explicit
`sources` document; the analysis record keeps the method, settings, source references and result,
and `verify` recomputes it from the sources to prove it is reproducible.

Source kinds:
  inline       {"kind": "inline", "reference": <values>, "treatment": <values>}
  interaction  {"kind": "interaction", "analysis_id"|"run_id", "measure", "reference_cell",
                "treatment_cell"}: trial values of an interaction analysis (its trials artifact)
  fault_trials {"kind": "fault_trials", "fault_experiment_id"|"run_id", "metric", "point_index",
                "reference_field", "treatment_field"}: per-trial baseline/faulted/... values of a
                fault experiment's analysis, keyed by seed (PAIRED by construction)
  artifact     {"kind": "artifact", "run_id", "path", "reference": [pointer], "treatment": [pointer]}:
                a mapping key -> number, or a list of numbers, inside any digest-verified artifact
  failure_mode {"kind": "failure_mode", "mode_id"}: runs with the signal / runs analyzed (PROPORTION)
  analyses     {"kind": "analyses", "ids": [sta_...]}: p-values of registered COMPARE analyses
               (CORRECTION); {"kind": "inline", "pvalues": {name: p}} also works
Only inline inputs are stored in the record; everything else is referenced and re-read on verify."""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from experionyx.artifacts import ArtifactStore
from experionyx.domain import to_jsonable
from experionyx.errors import DuplicateError, ValidationError
from experionyx.faults.report import analysis_run_id, read_artifact
from experionyx.hashing import content_hash
from experionyx.interactions.entities import InteractionAnalysis
from experionyx.interactions.taxonomy import Pairing
from experionyx.registry import Registry
from experionyx.stats import core
from experionyx.stats.entities import StatisticalAnalysis

DEFAULTS: Mapping[str, object] = {
    "pairing": "UNKNOWN", "estimator": "mean", "method": "percentile", "confidence": 0.95,
    "resamples": 2000, "permutations": 10_000, "seed": 0,
}  # fmt: skip
CORRECTION_DEFAULTS: Mapping[str, object] = {"method": "BONFERRONI", "alpha": 0.05}


def _sample(node: object, what: str) -> dict[str, float] | list[float]:
    if isinstance(node, Mapping):
        return {str(k): x for k, x in node.items()}
    if isinstance(node, list | tuple):
        return list(node)
    raise ValidationError(f"{what} must be a mapping key -> number or a list of numbers")


def _walk(doc: object, pointer: Sequence[str]) -> object:
    for step in pointer:
        if isinstance(doc, Mapping) and step in doc:
            doc = doc[step]
        elif isinstance(doc, list) and step.isdigit() and int(step) < len(doc):
            doc = doc[int(step)]
        else:
            raise ValidationError(f"pointer step {step!r} not found")
    return doc


def _interaction_run(reg: Registry, sources: Mapping[str, Any]) -> str:
    if "run_id" in sources:
        return str(sources["run_id"])
    return reg.get(InteractionAnalysis, str(sources["analysis_id"])).run_id


def resolve_samples(
    reg: Registry, store: ArtifactStore | None, sources: Mapping[str, Any], paired: bool = False
) -> tuple[Any, Any, str | None, tuple[str, ...]]:
    """(reference, treatment, pairing the evidence itself declares or None, notes on excluded data)."""
    kind = sources.get("kind")
    if kind == "inline":
        trt = sources.get("treatment")
        return (
            _sample(sources["reference"], "reference"),
            None if trt is None else _sample(trt, "treatment"),
            None,
            (),
        )
    if store is None:
        raise ValidationError("an artifact store is needed to read persisted evidence")
    if kind == "interaction":
        run_id = _interaction_run(reg, sources)
        trials = read_artifact(reg, store, run_id, "interaction/trials.json")
        design = read_artifact(reg, store, run_id, "interaction/design_validation.json")
        assert isinstance(trials, dict)  # noqa: S101
        assert isinstance(design, dict)  # noqa: S101
        measure = str(sources["measure"])
        cells = {}
        for role in ("reference_cell", "treatment_cell"):
            cell = str(sources[role])
            if cell not in trials:
                raise ValidationError(f"the interaction has no cell {cell!r}: {sorted(trials)}")
            cells[role] = {
                t["key"]: t["values"][measure] for t in trials[cell] if measure in t["values"]
            }
        pairing = str(design["pairing"])
        notes: tuple[str, ...] = ()
        if paired:  # only keys present in both cells are pairs; the rest are recorded, not used
            common = set(cells["reference_cell"]) & set(cells["treatment_cell"])
            if not common:
                raise ValidationError(
                    "paired analysis was requested but the two cells share no trial key, so there are no "
                    "pairs (a single control run is not paired with treatment trials); use "
                    "--pairing UNPAIRED to compare them as independent samples"
                )
            left = sorted(set(cells["reference_cell"]) ^ set(cells["treatment_cell"]))
            if left:
                notes = (
                    f"{len(left)} trial key(s) present in only one cell were excluded: {left}",
                )
            cells = {k: {x: y for x, y in c.items() if x in common} for k, c in cells.items()}
        return cells["reference_cell"], cells["treatment_cell"], pairing, notes
    if kind == "fault_trials":
        run_id = (
            str(sources["run_id"])
            if "run_id" in sources
            else analysis_run_id(reg, str(sources["fault_experiment_id"]))
        )
        trials = read_artifact(reg, store, run_id, "fault/trials.json")
        if not isinstance(trials, list):
            raise ValidationError("fault/trials.json is not a list of trials")
        metric, point = str(sources["metric"]), int(sources.get("point_index", 0))
        fields = (str(sources["reference_field"]), str(sources["treatment_field"]))
        got: tuple[dict[str, float], dict[str, float]] = ({}, {})
        for t in trials:
            if t["status"] != "COMPLETED" or t["point_index"] != point:
                continue
            row = next((d for d in t["degradations"] if d["metric_id"] == metric), None)
            key = f"seed{t['seed']}/repeat{t['repeat_index']}"
            if row is not None and all(row.get(f) is not None for f in fields):
                got[0][key], got[1][key] = row[fields[0]], row[fields[1]]
        if not got[0]:
            raise ValidationError(f"no completed trial at point {point} has metric {metric!r}")
        return got[0], got[1], "PAIRED", ()
    if kind == "artifact":
        doc = read_artifact(reg, store, str(sources["run_id"]), str(sources["path"]))
        ref = _sample(_walk(doc, sources["reference"]), "reference")
        tp = sources.get("treatment")
        trt = None if tp is None else _sample(_walk(doc, tp), "treatment")
        return ref, trt, None, ()
    raise ValidationError(f"unknown source kind {kind!r}")


def _config(kind: str, given: Mapping[str, Any] | None) -> dict[str, object]:
    base = dict(CORRECTION_DEFAULTS if kind == "CORRECTION" else DEFAULTS)
    extra = set(given or {}) - set(base)
    if extra:
        raise ValidationError(f"unknown setting(s) {sorted(extra)}; known: {sorted(base)}")
    return {**base, **(given or {})}


def compute(
    reg: Registry, store: ArtifactStore | None, kind: str, sources: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[object, object, str]:  # fmt: skip
    """(result document, exact inputs used, resolved pairing) for one analysis."""
    if kind in ("COMPARE", "BOOTSTRAP"):
        pairing = Pairing(str(config["pairing"]))
        ref, trt, _, notes = resolve_samples(reg, store, sources, pairing is Pairing.PAIRED)
        kw: dict[str, Any] = {
            "estimator": config["estimator"], "method": config["method"],
            "confidence": config["confidence"], "resamples": config["resamples"], "seed": config["seed"],
        }  # fmt: skip
        if kind == "COMPARE":
            if trt is None:
                raise ValidationError("a comparison needs a reference and a treatment sample")
            res: object = core.compare(
                ref, trt, pairing=pairing, permutations=config["permutations"], **kw
            )
        else:
            res = core.bootstrap_interval(ref, trt, pairing=pairing, **kw)
        doc = to_jsonable(res)
        assert isinstance(doc, dict)  # noqa: S101
        if notes:
            doc["source_notes"] = list(notes)
        return doc, {"reference": ref, "treatment": trt}, pairing.value
    if kind == "PROPORTION":
        if sources.get("kind") == "failure_mode":
            k, n = _failure_counts(reg, str(sources["mode_id"]))
        else:
            k, n = sources["successes"], sources["trials"]
        pdoc = to_jsonable(core.proportion_interval(k, n, float(config["confidence"])))
        assert isinstance(pdoc, dict)  # noqa: S101
        if sources.get("kind") == "failure_mode":
            pdoc["source_notes"] = [
                "runs are treated as independent Bernoulli trials; the runs of a discovery span "
                "different faults, severities and seeds, so this describes THIS set of runs, not "
                "a population failure rate"
            ]
        return pdoc, {"successes": k, "trials": n}, "n/a"
    if kind == "CORRECTION":
        if sources.get("kind") == "analyses":
            ps = {}
            for i in sources["ids"]:
                a = reg.get(StatisticalAnalysis, i)
                if a.analysis_kind != "COMPARE":
                    raise ValidationError(f"{i} is a {a.analysis_kind} analysis, not COMPARE")
                test = a.result["test"]
                ps[i] = test["p_value"] if isinstance(test, Mapping) else None
        else:
            ps = dict(sources["pvalues"])
        res = core.adjust_pvalues(ps, method=str(config["method"]), alpha=float(config["alpha"]))
        return to_jsonable(res), {"pvalues": ps}, "n/a"
    raise ValidationError(f"unknown analysis kind {kind!r}")


def _failure_counts(reg: Registry, mode_id: str) -> tuple[int, int]:
    from experionyx.failures.entities import FailureCluster, FailureMode

    mode = reg.get(FailureMode, mode_id)
    prev = reg.get(FailureCluster, mode.cluster_id).metrics.get("prevalence")
    if not isinstance(prev, Mapping) or not isinstance(prev.get("runs_analyzed"), int):
        raise ValidationError(f"mode {mode_id} carries no prevalence with a denominator")
    return int(prev["runs_with_signal"]), int(prev["runs_analyzed"])


def create(
    reg: Registry, store: ArtifactStore | None, kind: str, sources: Mapping[str, Any],
    config: Mapping[str, Any] | None = None, *, now: datetime | None = None,
) -> tuple[StatisticalAnalysis, bool]:  # fmt: skip
    """Compute, register and return (analysis, is_new). The same inputs and settings give the same
    record (same ID), so re-running is idempotent."""
    cfg = _config(kind, config)
    if kind in ("COMPARE", "BOOTSTRAP") and (config is None or "pairing" not in config):
        declared = resolve_samples(reg, store, sources)[2]  # what the evidence itself declares
        if declared is not None:
            cfg["pairing"] = declared
    result, inputs, _ = compute(reg, store, kind, sources, cfg)
    stored = dict(sources) if sources.get("kind") != "inline" else {**sources, "kind": "inline"}
    a = StatisticalAnalysis(
        kind, content_hash(to_jsonable(inputs)), core.STATS_VERSION, _status(result), cfg, stored,
        result if isinstance(result, dict) else {"result": result}, content_hash(result),
        now or datetime.now(UTC),
    )  # fmt: skip
    if reg.exists(StatisticalAnalysis, a.id):
        return reg.get(StatisticalAnalysis, a.id), False
    try:
        reg.add(a)
    except DuplicateError:  # lost a race; the stored record is the same content
        return reg.get(StatisticalAnalysis, a.id), False
    return a, True


def _status(result: object) -> str:
    return str(result["status"]) if isinstance(result, dict) and "status" in result else "DERIVED"


def verify(reg: Registry, store: ArtifactStore | None, analysis_id: str) -> dict[str, object]:
    """Recompute from the recorded sources and settings; reproduced only if the result hash and the
    input hash both match. A missing/changed source is reported, never silently accepted."""
    a = reg.get(StatisticalAnalysis, analysis_id)
    result, inputs, _ = compute(reg, store, a.analysis_kind, a.sources, a.config)
    input_hash, result_hash = content_hash(to_jsonable(inputs)), content_hash(result)
    return {
        "analysis_id": a.id,
        "reproduced": input_hash == a.input_hash and result_hash == a.result_hash,
        "inputs_match": input_hash == a.input_hash,
        "result_match": result_hash == a.result_hash,
        "engine_version": a.engine_version,
        "current_engine_version": core.STATS_VERSION,
    }

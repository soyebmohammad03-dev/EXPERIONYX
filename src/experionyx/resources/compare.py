"""Comparing resource analyses, and the batch-size sweep.

Comparison reuses Phase 10 (`stats.core`): summaries, effect sizes, a permutation (or paired sign-flip)
test and a bootstrap interval of the difference, then a multiple-comparison correction across the
family of treatments against the reference. Pairing is declared, never assumed:

  * per-sample latency (batch size 1, identical sample IDs): PAIRED by sample ID;
  * trial-level values (throughput, trial seconds): UNPAIRED. Repeated trials of two configurations
    are separate executions, not matched observations.

Trials on one machine are repeated measurements, not independent samples of a hardware population, so
an interval or p-value here describes measurement and sampling variability under these conditions
only. Analyses from different environments are refused unless the caller explicitly allows the
mismatch, and then they are labelled: an environment difference is confounded with everything else.
Nothing here produces a composite score."""

import dataclasses
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from experionyx.artifacts import ArtifactStore
from experionyx.domain import to_jsonable
from experionyx.errors import ValidationError
from experionyx.interactions.taxonomy import Pairing
from experionyx.registry import Registry
from experionyx.resources import engine as eng
from experionyx.resources.entities import ResourceAnalysis
from experionyx.resources.registry import ResourceRegistry
from experionyx.resources.spec import ResourceSpec
from experionyx.stats import core as st

METRICS = ("throughput", "trial_seconds", "per_sample_seconds")
MAX_SWEEP = 50


def _values(rr: ResourceRegistry, a: ResourceAnalysis, metric: str) -> tuple[Any, int]:
    """(values, n_samples per trial) from the analysis' preserved raw trials: completed MEASURED only."""
    trials = [
        t
        for t in rr.document(a.id, "trials")["trials"]
        if t["phase"] == "MEASURED" and t["status"] == "COMPLETED"
    ]
    repeats = a.spec.get("repeats")
    eng.check_trials(
        rr.document(a.id, "trials")["trials"], repeats if isinstance(repeats, int) else 0
    )
    if not trials:
        raise ValidationError(f"{a.id} has no completed measured trials to compare")
    n = trials[0]["completed_samples"]
    if metric == "throughput":
        return [t["throughput"]["samples_per_second"] for t in trials], n
    if metric == "trial_seconds":
        return [t["wall_seconds"] for t in trials], n
    if any(not t["per_sample_seconds"] for t in trials):
        raise ValidationError(f"{a.id} has no per-sample latency (measurable only at batch size 1)")
    per: dict[str, list[float]] = {}
    for t in trials:
        for sid, sec in t["per_sample_seconds"]:
            per.setdefault(str(sid), []).append(sec)
    return {k: statistics.fmean(v) for k, v in per.items()}, n


def _environment(rr: ResourceRegistry, a: ResourceAnalysis) -> dict[str, Any]:
    e = rr.document(a.id, "environment")
    return {k: e.get(k) for k in ("environment_id", "device", "measurement_backends")}


def compare_analyses(
    reg: Registry,
    store: ArtifactStore,
    ids: Sequence[str],
    *,
    metric: str = "throughput",
    correction: str = "BONFERRONI",
    alpha: float = 0.05,
    confidence: float = 0.95,
    resamples: int = 1000,
    permutations: int = 2000,
    seed: int = 0,
    allow_environment_mismatch: bool = False,
) -> dict[str, Any]:
    """The first id is the reference. Nothing is stored."""
    if metric not in METRICS:
        raise ValidationError(f"metric must be one of {list(METRICS)}")
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValidationError("comparison needs at least two distinct analyses")
    rr = ResourceRegistry(reg, store)
    an = [rr.analysis(i) for i in ids]
    envs = [_environment(rr, a) for a in an]
    mismatch = any(e != envs[0] for e in envs[1:])
    if mismatch and not allow_environment_mismatch:
        raise ValidationError(
            "the analyses were measured in different environments (environment, device or measurement backend); pass allow_environment_mismatch to compare them anyway, labelled as confounded"
        )
    ref = an[0]
    data = [_values(rr, a, metric) for a in an]
    if metric == "trial_seconds" and len({n for _, n in data}) != 1:
        raise ValidationError(
            "trial_seconds is comparable only between analyses that processed the same number of samples per trial; compare throughput instead"
        )
    paired = metric == "per_sample_seconds"
    if paired and any(set(d[0]) != set(data[0][0]) for d in data[1:]):
        raise ValidationError(
            "per-sample latencies are paired by sample ID and need identical sample IDs in every analysis"
        )
    rows: list[dict[str, Any]] = []
    for a, (vals, _) in zip(an[1:], data[1:], strict=True):
        c = st.compare(
            data[0][0],
            vals,
            pairing=Pairing.PAIRED if paired else Pairing.UNPAIRED,
            confidence=confidence,
            resamples=resamples,
            permutations=permutations,
            seed=seed,
        )
        ref_v = sorted(data[0][0].values() if isinstance(data[0][0], Mapping) else data[0][0])
        tr_v = sorted(vals.values() if isinstance(vals, Mapping) else vals)
        rows.append(
            {
                "analysis_id": a.id,
                "spec_id": a.spec_id,
                "differs_from_reference": [
                    k
                    for k in sorted(set(ref.spec) | set(a.spec))
                    if ref.spec.get(k) != a.spec.get(k)
                ],
                "median_ratio_vs_reference": statistics.median(tr_v) / statistics.median(ref_v)
                if statistics.median(ref_v)
                else None,
                "comparison": to_jsonable(c),
            }
        )
    corr = st.adjust_pvalues(
        {r["analysis_id"]: r["comparison"]["test"]["p_value"] for r in rows},
        method=correction,
        alpha=alpha,
    )
    for r in rows:
        r["adjusted_p_value"] = corr.adjusted.get(r["analysis_id"])
        r["significant_after_correction"] = corr.rejected.get(r["analysis_id"])
    return {
        "metric": metric,
        "reference": ref.id,
        "pairing": "PAIRED" if paired else "UNPAIRED",
        "environment_match": not mismatch,
        "environments": {a.id: e for a, e in zip(an, envs, strict=True)},
        "comparisons": rows,
        "correction": to_jsonable(corr),
        "caveats": [
            "trials on one machine are repeated measurements, not independent samples of a hardware population",
            "an observed difference is an observation under these conditions, not a causal claim",
            "a p-value says nothing about practical importance; read the effect sizes and the median ratio",
            *(
                [
                    "the analyses come from DIFFERENT environments: the environment difference is confounded with every other difference"
                ]
                if mismatch
                else []
            ),
        ],
    }


def run_sweep(
    reg: Registry,
    store: ArtifactStore,
    executor: Any,
    investigation_id: str,
    base: ResourceSpec,
    batch_sizes: Sequence[int],
) -> list[dict[str, Any]]:
    """One separately persisted measurement (its own Run) per batch size. Every spec is validated
    and preflighted BEFORE the first one runs, so an invalid sweep executes nothing."""
    if not batch_sizes or len(batch_sizes) > MAX_SWEEP or len(set(batch_sizes)) != len(batch_sizes):
        raise ValidationError(f"a sweep needs 1..{MAX_SWEEP} distinct batch sizes")
    specs = [dataclasses.replace(base, batch_size=b) for b in batch_sizes]  # validates each size
    for s in specs:
        eng.preflight(reg, store, executor.adapters, executor.inputs_root, s)
    out: list[dict[str, Any]] = []
    for s in specs:
        r = eng.run_resource_request(reg, store, executor, investigation_id, s)
        out.append(
            {
                "batch_size": s.batch_size,
                "spec_id": s.spec_id,
                "run_id": r.run_id,
                "status": r.status.value,
                "analysis_id": r.analysis_id,
            }
        )
    return out

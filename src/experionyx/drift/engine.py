"""The drift-analysis procedure, run by the normal execution engine so each analysis is a Run with
provenance, artifacts and observations and can be replayed.

Reuse, not reinvention: baseline predictions, sample tables and slice membership come from the
Phase 11 slice data layer; window metrics, their bootstrap intervals and the unpaired comparison of
the per-sample measure come from the slice/evaluation/Phase 10 code; p-value correction is the
Phase 10 `adjust_pvalues`; failure-mode evidence links reuse the slice failure integration. What is
new is the temporal layer (windows, ordering) and the distribution measures.

Nothing here scores, ranks or explains. Every result is one measurement between two named windows
under an explicit method, with its sample counts and an evidence status."""

import json
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from experionyx.adapters.capabilities import TaskType
from experionyx.artifacts import ArtifactStore
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
from experionyx.drift import measures
from experionyx.drift.data import (
    DriftDataError,
    Frame,
    Resolution,
    ResolvedWindow,
    load_frame,
    resolve,
)
from experionyx.drift.entities import DriftAnalysis, DriftWindow
from experionyx.drift.results import NOTE, DriftEvidence, Evidence, PerformanceDriftResult
from experionyx.drift.spec import ANALYSIS_VERSION, DRIFT_SCHEMA, ShiftSpec
from experionyx.errors import ArtifactIntegrityError, ExperionyxError, ValidationError
from experionyx.evaluation.config import _thaw
from experionyx.evaluation.loading import load_evaluation
from experionyx.execution import Executor, RunContext, resolve_procedure
from experionyx.failures.entities import FailureMode
from experionyx.faults.report import read_artifact
from experionyx.hashing import content_hash
from experionyx.interactions.lifecycle import REPLAY_TOLERANCE, _compare
from experionyx.registry import Registry
from experionyx.slices import analysis as slice_an
from experionyx.slices.data import (
    Baseline,
    SliceDataError,
    dataset_for_run,
    load_baseline,
    sample_table,
)
from experionyx.slices.evaluate import Membership, MembershipStatus, evaluate
from experionyx.stats import core as st

PROCEDURE = "experionyx.drift.engine:run_drift_analysis"
POPULATION = "POPULATION"
OUTPUTS = ("label", "prediction", "confidence")
DOCUMENTS = (
    "spec",
    "windows",
    "feature_results",
    "distribution_results",
    "performance_results",
    "summary",
)
GROUPS = {
    "covariate": "covariate",
    "label": "target_and_output",
    "prediction": "target_and_output",
    "confidence": "target_and_output",
    "performance": "performance",
}


# -- request ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftRequest:
    investigation_id: str
    spec: Mapping[str, object]

    def to_parameters(self) -> dict[str, object]:
        return {
            "drift_analysis": {"investigation_id": self.investigation_id, "spec": dict(self.spec)}
        }

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, object]) -> "DriftRequest":
        if set(parameters) != {"drift_analysis"}:
            raise ValidationError("a drift configuration must be exactly {'drift_analysis': ...}")
        body = _thaw(parameters["drift_analysis"])
        if (
            not isinstance(body, dict)
            or set(body) != {"investigation_id", "spec"}
            or not isinstance(body["spec"], dict)
        ):
            raise ValidationError("'drift_analysis' has the wrong fields")
        return cls(str(body["investigation_id"]), body["spec"])


# -- computation --------------------------------------------------------------------------------------------


@dataclass
class Computed:
    base: Baseline
    frame: Frame
    resolution: Resolution
    model_fingerprint: str | None
    model_id: str | None
    dataset_id: str | None
    memberships: dict[str, dict[str, Membership]]  # slice name -> window id -> membership
    unavailable: dict[str, str]  # slice name -> why it could not be evaluated
    feature_results: dict[str, Any]
    distribution_results: dict[str, Any]
    performance_results: dict[str, Any]
    failure_evidence: dict[str, Any]
    stubs: int  # populations that could not be evaluated for a pair


def _membership_of_window(rw: ResolvedWindow, n_total: int) -> Membership:
    """A window seen as a membership, so the slice failure-mode integration can be reused."""
    return Membership(
        slice_id=rw.window_id,
        name=rw.window.describe(),
        description=f"samples with an ordering key in {rw.window.describe()}",
        fields=(),
        status=MembershipStatus.COMPUTED if rw.n else MembershipStatus.EMPTY,
        sample_ids=rw.sample_ids,
        n_members=rw.n,
        n_total=n_total,
        n_unknown=0,
        unknown_sample_ids=(),
        prevalence=rw.n / n_total if n_total else None,
        missing_fields=(),
        unseen_values={},
        membership_digest=rw.digest,
    )


def _stub(status: Evidence, code: str, reason: str, n_ref: int, n_cmp: int) -> dict[str, Any]:
    return {
        "status": status.value,
        "reason_code": code,
        "reason": reason,
        "n_reference": n_ref,
        "n_comparison": n_cmp,
        "note": NOTE,
    }


def _put(d: dict[str, Any], key: str, pop: str, val: Any) -> None:
    d.setdefault(key, {"populations": {}})["populations"][pop] = val


def _kind(task: TaskType) -> str:
    return "CATEGORICAL" if task is TaskType.CLASSIFICATION else "NUMERIC"


def _distribution(
    base: Baseline, frame: Frame, spec: ShiftSpec, ref: Sequence[int], cmp: Sequence[int]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(covariate results by feature, label/prediction/confidence results) for one pair of sample sets."""
    cfg, dims = spec.config, spec.config.dimensions or ()
    cov: dict[str, Any] = {}
    if "covariate" in dims:
        for f in spec.features:
            col = frame.columns[f.name]
            cov[f.name] = measures.shift(
                "COVARIATE",
                f.name,
                f.kind,
                [col.get(i) for i in ref],
                [col.get(i) for i in cmp],
                cfg,
            ).to_dict()
    out: dict[str, Any] = {}
    kind = _kind(base.task)
    rows = base.rows
    if "label" in dims:
        out["label"] = measures.shift(
            "LABEL",
            "target",
            kind,
            [rows[i]["true"] for i in ref],
            [rows[i]["true"] for i in cmp],
            cfg,
        ).to_dict()
    if "prediction" in dims:
        out["prediction"] = measures.shift(
            "PREDICTION",
            "predicted",
            kind,
            [rows[i]["predicted"] for i in ref],
            [rows[i]["predicted"] for i in cmp],
            cfg,
        ).to_dict()
        if all(rows[i].get("confidence") is None for i in (*ref, *cmp)):
            out["confidence"] = {
                "dimension": "CONFIDENCE",
                "subject": "confidence",
                "status": Evidence.UNAVAILABLE.value,
                "reason": "the baseline stored no per-sample confidence",
                "note": NOTE,
            }
        else:
            out["confidence"] = measures.shift(
                "CONFIDENCE",
                "confidence",
                "NUMERIC",
                [rows[i].get("confidence") for i in ref],
                [rows[i].get("confidence") for i in cmp],
                cfg,
            ).to_dict()
    return cov, out


def _performance(
    base: Baseline,
    ref: Sequence[int],
    cmp: Sequence[int],
    cfg_slice: slice_an.SliceConfig,
    min_samples: int,
) -> dict[str, Any]:
    """Window metrics (OBSERVED) and their differences (DERIVED), reusing the slice metric engine."""
    r, c = (
        slice_an.slice_metrics(base, ref, cfg_slice),
        slice_an.slice_metrics(base, cmp, cfg_slice),
    )
    by_ref = {m["metric_id"]: m for m in r["metrics"]}
    rows: list[dict[str, Any]] = []
    for m in c["metrics"]:
        a = by_ref.get(m["metric_id"])
        row: dict[str, Any] = {
            "metric_id": m["metric_id"],
            "higher_is_better": m["higher_is_better"],
            "reference_value": None if a is None else a["value"],
            "comparison_value": m["value"],
            "reference_status": None if a is None else a["status"],
            "comparison_status": m["status"],
            "reference_interval": None if a is None else a["interval"],
            "comparison_interval": m["interval"],
            "difference": None,
            "change": None,
            "worse_on_this_metric": None,
        }
        if a is not None and a["status"] == "COMPUTED" and m["status"] == "COMPUTED":
            d = m["value"] - a["value"]
            row.update(
                difference=d,
                change="EQUAL" if d == 0 else "HIGHER" if d > 0 else "LOWER",
                worse_on_this_metric=None if d == 0 else (d < 0) == m["higher_is_better"],
            )
        else:
            row["reason"] = (
                (a or {}).get("reason") or m.get("reason") or "a window value is undefined"
            )
        rows.append(row)
    name, hib = slice_an.measure_of(base.task)
    inf = slice_an.compare_groups(base, cmp, ref, cfg_slice)  # a = comparison, b = reference
    thin = min(len(ref), len(cmp)) < min_samples
    return PerformanceDriftResult(
        task=base.task.value,
        measure=name,
        status=Evidence.INSUFFICIENT_EVIDENCE
        if thin
        else Evidence.INCONCLUSIVE
        if inf.get("status") == st.Status.INCONCLUSIVE.value
        else Evidence.DERIVED,
        reference={**r, "status": Evidence.OBSERVED.value},
        comparison={**c, "status": Evidence.OBSERVED.value},
        metrics=tuple(rows),
        inference={**inf, "higher_is_better": hib},
        reason=f"fewer than min_samples={min_samples} in a window ({len(ref)} reference, {len(cmp)} comparison): values are reported, inference is not reliable"
        if thin
        else None,
    ).to_dict()


def _p(doc: Mapping[str, Any]) -> float | None:
    if "measures" in doc or "test" in doc:
        t = doc.get("test")
        return t.get("p_value") if isinstance(t, Mapping) else None
    inf = (doc.get("inference") or {}).get("inference")
    t = inf.get("test") if isinstance(inf, Mapping) else None
    p = t.get("p_value") if isinstance(t, Mapping) else None
    return p if (doc.get("inference") or {}).get("status") != slice_an.INSUFFICIENT else None


def correct(entries: list[tuple[str, str, str, dict[str, Any]]], spec: ShiftSpec) -> dict[str, Any]:
    """Phase 10 multiple-comparison correction. Each entry is (pair, population, group, doc); the
    family is one (pair, population, group) - or (population, group) across every window pair - and
    exactly its hypothesis-test p-values. Results without a valid p-value are listed, not dropped."""
    cfg = spec.config
    fams: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for pair, pop, group, doc in entries:
        key = (
            f"{pair} | {pop} | {group}"
            if cfg.correction_scope == "PER_WINDOW_PAIR"
            else f"{pop} | {group}"
        )
        fams.setdefault(key, []).append((f"{pair} | {doc.get('subject', 'performance')}", doc))
    out: dict[str, Any] = {}
    for key, members in sorted(fams.items()):
        ps = {m: _p(doc) for m, doc in members}
        corr = st.adjust_pvalues(ps, method=cfg.correction, alpha=cfg.alpha)
        for m, doc in members:
            doc["multiplicity"] = {
                "family": key,
                "method": cfg.correction,
                "alpha": cfg.alpha,
                "raw_p": ps[m],
                "adjusted_p": corr.adjusted.get(m),
                "adjusted_p_below_alpha": corr.rejected.get(m)
                if cfg.correction != "NONE"
                else None,
            }
        cdoc = to_jsonable(corr)
        assert isinstance(cdoc, dict)  # noqa: S101
        out[key] = {
            **cdoc,
            "members": [m for m, _ in members],
            "scope": cfg.correction_scope,
            "definition": "the hypothesis tests of one comparison family: "
            + (
                "one window pair, one population, one result group"
                if cfg.correction_scope == "PER_WINDOW_PAIR"
                else "every window pair, one population, one result group"
            ),
        }
    return out


def compute(reg: Registry, store: ArtifactStore, spec: ShiftSpec, dataset: Any | None) -> Computed:
    """All numbers of an analysis, deterministic given the registry content and the dataset. Reads only."""
    base = load_baseline(reg, store, spec.baseline_run)
    if base.dataset_fingerprint is None:
        raise DriftDataError("the baseline evaluation records no dataset fingerprint")
    if dataset is not None and dataset.fingerprint() != base.dataset_fingerprint:
        raise DriftDataError(
            "the loaded dataset's fingerprint differs from the baseline evaluation's"
        )
    ev = load_evaluation(reg, store, spec.baseline_run)
    frame = load_frame(base, dataset, spec)
    res = resolve(spec, frame)
    cfg, sc = spec.config, spec.config.slice_config()
    dims = cfg.dimensions or ()
    mems: dict[str, dict[str, Membership]] = {}
    unavailable: dict[str, str] = {}
    for s in spec.slices:
        try:
            table = sample_table(base, dataset, s.fields)
        except SliceDataError as exc:
            unavailable[s.name] = str(exc)
            continue
        mems[s.name] = {
            wid: evaluate(s, {i: table[i] for i in rw.sample_ids})
            for wid, rw in res.windows.items()
        }
    populations = [POPULATION, *(s.name for s in spec.slices)]
    feat: dict[str, Any] = {}
    dist: dict[str, Any] = {}
    perf: dict[str, Any] = {}
    fam_cov: list[tuple[str, str, str, dict[str, Any]]] = []
    fam_out: list[tuple[str, str, str, dict[str, Any]]] = []
    fam_perf: list[tuple[str, str, str, dict[str, Any]]] = []
    stubs = 0
    for key, rid, cid in res.pairs:
        rw, cw = res.windows[rid], res.windows[cid]
        for pop in populations:
            if pop == POPULATION:
                ref_ids, cmp_ids = list(rw.sample_ids), list(cw.sample_ids)
            elif pop in unavailable:
                stubs += 1
                stub = _stub(Evidence.UNAVAILABLE, "SLICE_UNAVAILABLE", unavailable[pop], 0, 0)
                for d in (feat, dist, perf):
                    _put(d, key, pop, stub)
                continue
            else:
                mr, mc = mems[pop][rid], mems[pop][cid]
                bad = next((m for m in (mr, mc) if m.status is not MembershipStatus.COMPUTED), None)
                if bad is not None:
                    stubs += 1
                    empty = bad.status is MembershipStatus.EMPTY
                    stub = _stub(
                        Evidence.UNDEFINED,
                        "NO_MEMBERS" if empty else bad.status.value,
                        f"the slice {pop!r} has no members in the {'reference' if bad is mr else 'comparison'} window ({bad.status.value}); this is an absence of samples, not a low-sample result",
                        mr.n_members,
                        mc.n_members,
                    )
                    for d in (feat, dist, perf):
                        _put(d, key, pop, stub)
                    continue
                ref_ids, cmp_ids = [int(i) for i in mr.sample_ids], [int(i) for i in mc.sample_ids]
            cov, outs = _distribution(base, frame, spec, ref_ids, cmp_ids)
            if "covariate" in dims:
                _put(feat, key, pop, cov)
                fam_cov += [(key, pop, "covariate", d) for d in cov.values()]
            if outs:
                _put(dist, key, pop, outs)
                fam_out += [
                    (key, pop, "target_and_output", d) for d in outs.values() if "test" in d
                ]
            if "performance" in dims:
                pr = _performance(base, ref_ids, cmp_ids, sc, cfg.min_samples)
                _put(perf, key, pop, pr)
                fam_perf.append((key, pop, "performance", pr))
            if pop == POPULATION:  # the exact inputs of this pair, once
                ev_doc = DriftEvidence(
                    key,
                    rid,
                    cid,
                    rw.digest,
                    cw.digest,
                    rw.n,
                    cw.n,
                    str(base.dataset_fingerprint),
                    spec.baseline_run,
                    spec.ordering.field,
                ).to_dict()
                for d in (feat, dist, perf):
                    d.setdefault(key, {"populations": {}})["evidence"] = ev_doc
    families = {
        "covariate": correct(fam_cov, spec) if fam_cov else {},
        "target_and_output": correct(fam_out, spec) if fam_out else {},
        "performance": correct(fam_perf, spec) if fam_perf else {},
    }
    failure_evidence: dict[str, Any] = {}
    if spec.failure_modes:
        for wid, rw in res.windows.items():
            failure_evidence[wid] = {
                "window": rw.window.describe(),
                **slice_an.failure_slice(
                    reg,
                    store,
                    base,
                    _membership_of_window(rw, len(base.rows)),
                    spec.failure_modes,
                    sc,
                ),
            }
    feat = {"results": feat, "multiple_comparisons": families["covariate"]}
    dist = {"results": dist, "multiple_comparisons": families["target_and_output"]}
    perf = {
        "results": perf,
        "multiple_comparisons": families["performance"],
        "failure_mode_evidence": failure_evidence,
        "failure_mode_note": "observed failure prevalence within each window: an evidence link between a window and a failure mode, not an explanation of either",
    }
    return Computed(
        base,
        frame,
        res,
        ev.context.model_fingerprint,
        ev.context.model_id,
        ev.context.dataset_id,
        mems,
        unavailable,
        feat,
        dist,
        perf,
        failure_evidence,
        stubs,
    )


# -- documents ----------------------------------------------------------------------------------------------


def _statuses(block: Mapping[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}

    def walk(x: Any) -> None:
        if isinstance(x, Mapping):
            if "status" in x and "note" in x and ("dimension" in x or "measure" in x):
                out[x["status"]] = out.get(x["status"], 0) + 1
                return
            for v in x.values():
                walk(v)

    walk(
        {
            k: v
            for k, v in block.items()
            if k not in ("multiple_comparisons", "failure_mode_evidence")
        }
    )
    return dict(sorted(out.items()))


def windows_doc(spec: ShiftSpec, c: Computed) -> dict[str, Any]:
    oid = spec.ordering.field
    excl = c.frame.order_excluded
    return {
        "ordering": spec.ordering.to_dict(),
        "ordering_note": "the declared ordering; nothing is inferred from row position. Samples without a usable ordering key belong to no window",
        "n_baseline_samples": len(c.base.rows),
        "n_with_ordering_key": len(c.frame.order),
        "order_excluded": {k: {"n": len(v), "sample_ids": v[:20]} for k, v in excl.items()},
        "windows": {wid: rw.to_dict() for wid, rw in c.resolution.windows.items()},
        "pairs": [
            {"key": k, "reference_window_id": r, "comparison_window_id": q}
            for k, r, q in c.resolution.pairs
        ],
        "skipped": [s.to_dict(oid) for s in c.resolution.skipped],
        "slices": {
            s.name: {
                "slice_id": s.slice_id,
                "description": s.human,
                "windows": {
                    wid: {
                        "status": m.status.value,
                        "n_members": m.n_members,
                        "n_window_samples": m.n_total,
                        "n_unknown": m.n_unknown,
                        "prevalence": m.prevalence,
                        "membership_digest": m.membership_digest,
                        "warnings": list(m.warnings),
                    }
                    for wid, m in c.memberships[s.name].items()
                }
                if s.name in c.memberships
                else {"status": "UNAVAILABLE", "reason": c.unavailable[s.name]},
            }
            for s in spec.slices
        },
        "membership_note": "slice membership is evaluated independently inside each window; it is not assumed to be the same set over time",
    }


def spec_doc(spec: ShiftSpec, c: Computed, fingerprint: str) -> dict[str, Any]:
    cfg = spec.config
    return {
        "spec_id": spec.spec_id,
        "spec": spec.to_dict(),
        "provenance_fingerprint": fingerprint,
        "baseline_run": spec.baseline_run,
        "model_id": c.model_id,
        "model_fingerprint": c.model_fingerprint,
        "dataset_id": c.dataset_id,
        "dataset_fingerprint": c.base.dataset_fingerprint,
        "split": c.base.split,
        "evaluation_config_hash": c.base.evaluation_config_hash,
        "ordering": spec.ordering.to_dict(),
        "window_ids": {k: rw.window.describe() for k, rw in c.resolution.windows.items()},
        "features": {f.name: f.kind for f in spec.features},
        "metrics": list(cfg.metrics) or "every scalar metric of the task",
        "methods": {
            "numeric": [
                "ks_statistic",
                "wasserstein_1",
                "location (Phase 10 effect sizes and bootstrap interval)",
                "ks_permutation test",
            ],
            "categorical": [
                "proportions with Wilson intervals (Phase 10)",
                "jensen_shannon_divergence (base 2)",
                "total_variation_distance",
                "jensen_shannon_permutation test",
            ],
            "performance": [
                "window metrics (slice/evaluation engine)",
                "unpaired comparison of the per-sample measure (Phase 10)",
            ],
            "correction": cfg.correction,
            "correction_scope": cfg.correction_scope,
            "alpha": cfg.alpha,
        },
        "random_seed": cfg.seed,
        "slice_ids": {s.name: s.slice_id for s in spec.slices},
        "failure_modes": list(spec.failure_modes),
        "dimensions": list(cfg.dimensions or ()),
        "versions": versions(),
        "source_revision": "recorded in the run's Provenance record (see `drift inspect`)",
    }


def versions() -> dict[str, str]:
    return {
        "drift_analysis": ANALYSIS_VERSION,
        "drift_schema": str(DRIFT_SCHEMA),
        "statistics": st.STATS_VERSION,
        "slice_analysis": slice_an.ANALYSIS_VERSION,
        "python": platform.python_version(),
    }


def summarize(spec: ShiftSpec, c: Computed) -> dict[str, Any]:
    pairs: dict[str, Any] = {}
    for key, rid, cid in c.resolution.pairs:
        rw, cw = c.resolution.windows[rid], c.resolution.windows[cid]
        pops: dict[str, Any] = {}
        for pop in (POPULATION, *(s.name for s in spec.slices)):
            f = c.feature_results["results"].get(key, {}).get("populations", {}).get(pop, {})
            d = c.distribution_results["results"].get(key, {}).get("populations", {}).get(pop, {})
            p = c.performance_results["results"].get(key, {}).get("populations", {}).get(pop, {})
            if "reason_code" in (p or f or d):
                pops[pop] = next(x for x in (p, f, d) if "reason_code" in x)
                continue
            head = (p.get("inference") or {}).get("descriptive") or {}
            pops[pop] = {
                "covariate_status_counts": _statuses({"x": f}),
                "features": {
                    n: {
                        "status": r["status"],
                        "statistic": _headline(r),
                        "adjusted_p": (r.get("multiplicity") or {}).get("adjusted_p"),
                    }
                    for n, r in f.items()
                },
                "label": None
                if "label" not in d
                else {
                    "status": d["label"]["status"],
                    "statistic": _headline(d["label"]),
                },
                "prediction": None
                if "prediction" not in d
                else {
                    "status": d["prediction"]["status"],
                    "statistic": _headline(d["prediction"]),
                },
                "performance": None
                if not p
                else {
                    "status": p["status"],
                    "measure": p["measure"],
                    "difference": head.get("difference"),
                },
            }
        pairs[key] = {
            "reference_window_id": rid,
            "comparison_window_id": cid,
            "n_reference": rw.n,
            "n_comparison": cw.n,
            "populations": pops,
        }
    counts = _statuses(
        {
            "f": c.feature_results["results"],
            "d": c.distribution_results["results"],
            "p": c.performance_results["results"],
        }
    )
    return {
        "baseline_run_id": spec.baseline_run,
        "dataset_fingerprint": c.base.dataset_fingerprint,
        "ordering": spec.ordering.field,
        "n_pairs": len(c.resolution.pairs),
        "n_skipped_windows": len(c.resolution.skipped),
        "skipped": [
            {"window": s.window.describe(), "reason": s.reason} for s in c.resolution.skipped
        ],
        "measure": slice_an.measure_of(c.base.task)[0],
        "pairs": pairs,
        "status_counts": counts,
        "correction": {
            "method": spec.config.correction,
            "alpha": spec.config.alpha,
            "scope": spec.config.correction_scope,
        },
        "stubs": c.stubs,
        "note": NOTE + "; no composite drift score or overall verdict is computed",
    }


def _headline(r: Mapping[str, Any]) -> Any:
    m = r.get("measures") or {}
    return m.get("ks_statistic", m.get("jensen_shannon_divergence"))


def provenance_fingerprint(spec: ShiftSpec, c: Computed, reg: Registry) -> str:
    return content_hash(
        {
            "dataset_fingerprint": c.base.dataset_fingerprint,
            "split": c.base.split,
            "evaluation_config_hash": c.base.evaluation_config_hash,
            "model_fingerprint": c.model_fingerprint,
            "baseline_run": spec.baseline_run,
            "ordering": spec.ordering.to_dict(),
            "windows": {k: rw.digest for k, rw in c.resolution.windows.items()},
            "pairs": [list(p) for p in c.resolution.pairs],
            "skipped": [[s.window.describe(), s.reason] for s in c.resolution.skipped],
            "order_excluded": dict(c.frame.order_excluded),
            "features": {f.name: f.kind for f in spec.features},
            "config": spec.config.to_dict(),
            "slices": {
                s.name: {
                    "slice_id": s.slice_id,
                    "membership": {
                        w: m.membership_digest for w, m in c.memberships.get(s.name, {}).items()
                    },
                }
                for s in spec.slices
            },
            "modes": {m: reg.get(FailureMode, m).content_hash() for m in spec.failure_modes},
            "versions": {k: v for k, v in versions().items() if k != "python"},
        }
    )


def _write(ctx: RunContext, name: str, payload: object) -> str:
    directory = ctx.artifact_dir / "drift"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(
        json.dumps(to_jsonable(payload), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return ctx.register_artifact(
        f"drift/{name}.json", name=f"drift-{name}", media_type="application/json"
    ).id


def run_drift_analysis(ctx: RunContext) -> None:
    req = DriftRequest.from_parameters(ctx.parameters)
    spec = ShiftSpec.from_dict(req.spec)
    reg, now = ctx.registry, ctx.started_at
    c = compute(reg, ctx.store, spec, ctx.dataset)
    summary = summarize(spec, c)
    fp = provenance_fingerprint(spec, c, reg)
    art = {
        "spec": _write(ctx, "spec", spec_doc(spec, c, fp)),
        "windows": _write(ctx, "windows", windows_doc(spec, c)),
        "feature_results": _write(ctx, "feature_results", c.feature_results),
        "distribution_results": _write(ctx, "distribution_results", c.distribution_results),
        "performance_results": _write(ctx, "performance_results", c.performance_results),
        "summary": _write(ctx, "summary", summary),
    }
    counts = summary["status_counts"]
    partial = (
        bool(c.resolution.skipped)
        or bool(c.stubs)
        or any(k != Evidence.DERIVED.value and k != Evidence.OBSERVED.value for k in counts)
    )
    analysis = DriftAnalysis(
        req.investigation_id,
        ctx.run.id,
        spec.spec_id,
        spec.to_dict(),
        spec.baseline_run,
        str(c.base.dataset_fingerprint),
        fp,
        "PARTIAL" if partial else "COMPLETE",
        {**summary, "artifacts": art},
        now,
    )
    new = not reg.exists(DriftAnalysis, analysis.id)
    if new:
        with reg.transaction():
            for rw in c.resolution.windows.values():
                rec = DriftWindow.of(rw.window, spec.ordering.field, now)
                if not reg.exists(DriftWindow, rec.id):
                    reg.add(rec)
            reg.add(analysis)
    ctx.observe("drift.new_record", int(new), kind=EpistemicKind.OBSERVATION)
    ctx.observe("drift.window_pairs", len(c.resolution.pairs), kind=EpistemicKind.OBSERVATION)
    ctx.observe("drift.skipped_windows", len(c.resolution.skipped), kind=EpistemicKind.OBSERVATION)
    ctx.observe("drift.populations", 1 + len(spec.slices), kind=EpistemicKind.DERIVED_METRIC)


# -- entry points -------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftRunResult:
    experiment_id: str
    run_id: str
    status: RunStatus
    analysis_id: str | None


def validate(reg: Registry, spec: ShiftSpec) -> Run:
    """Refuse before anything is created: the baseline must be a completed run and every referenced
    failure mode must exist. (Structural problems were already refused when the spec was built.)"""
    run = reg.get(Run, spec.baseline_run)
    if run.status is not RunStatus.COMPLETED:
        raise ValidationError(f"baseline run {run.id} is {run.status.value}, not COMPLETED")
    for m in spec.failure_modes:
        reg.get(FailureMode, m)
    return run


def needs_dataset(spec: ShiftSpec) -> bool:
    return bool(spec.features) or spec.ordering.field != "index" or any(f.startswith("feature:") for s in spec.slices for f in s.fields)  # fmt: skip


def baseline_dataset(reg: Registry, adapters: Any, inputs_root: Any, spec: ShiftSpec) -> Any | None:
    """The baseline's dataset (fingerprint-verified), loaded only if the spec reads a dataset
    column; a spec that needs one the baseline cannot provide is refused here."""
    if not needs_dataset(spec):
        return None
    if adapters is None:
        raise DriftDataError("the spec reads dataset columns but no adapter registry is available to load the dataset")  # fmt: skip
    ds = dataset_for_run(reg, adapters, inputs_root, spec.baseline_run)
    if ds is None:
        raise DriftDataError(f"baseline run {spec.baseline_run} has no registered dataset, but the spec reads dataset columns")  # fmt: skip
    return ds


def preflight(
    reg: Registry, store: ArtifactStore, spec: ShiftSpec, dataset: Any | None
) -> Resolution:
    """The data-dependent refusals, without running anything: the features and ordering exist, keys
    are unique when required, explicit windows are non-empty, slices are evaluable."""
    validate(reg, spec)
    base = load_baseline(reg, store, spec.baseline_run)
    frame = load_frame(base, dataset, spec)
    for s in spec.slices:
        sample_table(base, dataset, s.fields)
    return resolve(spec, frame)


def run_drift_request(
    registry: Registry,
    store: ArtifactStore,
    executor: Executor,
    investigation_id: str,
    spec: ShiftSpec,
    *,
    dataset: Any | None = None,
    seed: int = 0,
) -> DriftRunResult:
    validate(registry, spec)
    if dataset is not None or not (spec.features or spec.slices or spec.ordering.field != "index"):
        preflight(
            registry, store, spec, dataset
        )  # nothing is created if the data cannot support it
    run = registry.get(Run, spec.baseline_run)
    base_exp = registry.get(Experiment, run.experiment_id)
    conf = ConfigurationRef(DriftRequest(investigation_id, spec.to_dict()).to_parameters())
    if not registry.exists(ConfigurationRef, conf.id):
        registry.add(conf)
    exp = Experiment(
        investigation_id,
        "drift analysis",
        "Deterministic temporal and distribution shift analysis over a stored run",
        ModelRef("none", "0"),
        base_exp.dataset,
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
        a
        for a in registry.find(DriftAnalysis, spec_id=spec.spec_id)
        if a.investigation_id == investigation_id
    ]
    return DriftRunResult(exp.id, result.run.id, result.status, found[0].id if found else None)


def replay_check(
    reg: Registry, store: ArtifactStore, executor: Executor, analysis_id: str
) -> dict[str, object]:
    """Replay the analysis Run as a NEW run and compare every stored document within REPLAY_TOLERANCE:
    window identities, membership digests, drift statistics, metric results and evidence references."""
    a = reg.get(DriftAnalysis, analysis_id)
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
            "differences": [f"replay run ended {replay.status.value}"],
        }
    diffs: list[str] = []
    for name in DOCUMENTS:
        try:
            x = read_artifact(reg, store, a.run_id, f"drift/{name}.json")
        except ArtifactIntegrityError:
            raise  # a document that no longer matches its digest is not a replay difference; refuse
        except ExperionyxError:
            diffs.append(f"{name}: missing from the original run")
            continue
        _compare(name, x, read_artifact(reg, store, replay.run.id, f"drift/{name}.json"), diffs)
    return {
        **out,
        "deterministic": not diffs,
        "differences": diffs[:50],
        "compared": list(DOCUMENTS),
    }


__all__ = [
    "PROCEDURE",
    "DriftRunResult",
    "compute",
    "preflight",
    "replay_check",
    "run_drift_analysis",
    "run_drift_request",
    "validate",
]

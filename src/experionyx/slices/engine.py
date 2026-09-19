"""The slice-analysis procedure, run by the normal execution engine so each analysis is a Run with
provenance, artifacts and observations and can be replayed."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

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
from experionyx.errors import ExperionyxError, ValidationError
from experionyx.evaluation.config import _thaw
from experionyx.execution import Executor, RunContext, resolve_procedure
from experionyx.failures.entities import FailureMode
from experionyx.faults.entities import FaultExperiment
from experionyx.hashing import HASH_PREFIX, content_hash
from experionyx.interactions.entities import InteractionAnalysis
from experionyx.registry import Registry
from experionyx.slices import analysis as an
from experionyx.slices.data import Baseline, SliceDataError, load_baseline, sample_table
from experionyx.slices.entities import Slice, SliceAnalysis
from experionyx.slices.evaluate import Membership, MembershipStatus, evaluate
from experionyx.slices.spec import SliceSpec

PROCEDURE = "experionyx.slices.engine:run_slice_analysis"
POPULATION, REST = "POPULATION", "REST"


def _ids(name: str, ids: Sequence[str], prefix: str) -> tuple[str, ...]:
    import experionyx.validation as v

    if isinstance(ids, str) or not all(isinstance(i, str) for i in ids):
        raise ValidationError(f"{name} must be a list of IDs")
    for i in ids:
        v.ref(name, i, prefix)
    return tuple(sorted(set(ids)))


@dataclass(frozen=True)
class SliceAnalysisSpec:
    baseline_run: str
    slices: tuple[SliceSpec, ...]
    comparisons: tuple[tuple[str, str], ...] = ()  # (a, b): slice names, REST or POPULATION
    fault_experiments: tuple[str, ...] = ()
    failure_modes: tuple[str, ...] = ()
    interactions: tuple[str, ...] = ()
    config: an.SliceConfig = field(default_factory=an.SliceConfig)

    def __post_init__(self) -> None:
        import experionyx.validation as v

        v.ref("baseline_run", self.baseline_run, "run")
        if not self.slices:
            raise ValidationError("a slice analysis needs at least one slice")
        names = [s.name for s in self.slices]
        ids = [s.slice_id for s in self.slices]
        if len(set(names)) != len(names) or (names and any(n in (POPULATION, REST) for n in names)):
            raise ValidationError(f"slice names must be unique and not {POPULATION}/{REST}")
        if len(set(ids)) != len(ids):
            raise ValidationError("two slices in one analysis have the same definition")
        object.__setattr__(self, "slices", tuple(sorted(self.slices, key=lambda s: s.slice_id)))
        for a, b in self.comparisons:
            for x in (a, b):
                if x not in (*names, POPULATION, REST):
                    raise ValidationError(f"comparison names unknown slice {x!r}")
            if a == b or a in (POPULATION, REST):
                raise ValidationError(
                    f"a comparison needs a slice as its first side and two different sides, got ({a}, {b})"
                )
        object.__setattr__(self, "comparisons", tuple(sorted(set(self.comparisons))))
        object.__setattr__(
            self, "fault_experiments", _ids("fault_experiments", self.fault_experiments, "fxp")
        )
        object.__setattr__(self, "failure_modes", _ids("failure_modes", self.failure_modes, "fmd"))
        object.__setattr__(self, "interactions", _ids("interactions", self.interactions, "ian"))

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline_run": self.baseline_run,
            "slices": [s.to_dict() for s in self.slices],
            "comparisons": [list(c) for c in self.comparisons],
            "fault_experiments": list(self.fault_experiments),
            "failure_modes": list(self.failure_modes),
            "interactions": list(self.interactions),
            "config": self.config.to_dict(),
            "analysis_version": an.ANALYSIS_VERSION,
        }

    @property
    def spec_id(self) -> str:
        # descriptions are labels; names appear in comparisons and outputs, so they are included
        d = self.to_dict()
        d["slices"] = [{"name": s.name, "slice_id": s.slice_id} for s in self.slices]
        return "ssp_" + content_hash(d)[len(HASH_PREFIX) :][:32]

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "SliceAnalysisSpec":
        allowed = {
            "baseline_run",
            "slices",
            "comparisons",
            "fault_experiments",
            "failure_modes",
            "interactions",
            "config",
            "analysis_version",
        }
        if set(d) - allowed or "baseline_run" not in d or "slices" not in d:
            raise ValidationError(
                f"malformed slice analysis spec (unexpected {sorted(set(d) - allowed)}; baseline_run and slices are required)"
            )
        raw = d["slices"]
        if not isinstance(raw, list | tuple):
            raise ValidationError("slices must be a list")

        def lst(k: str) -> tuple[str, ...]:
            x = d.get(k, [])
            if not isinstance(x, list | tuple):
                raise ValidationError(f"{k} must be a list")
            return tuple(str(i) for i in x)

        comps = d.get("comparisons", [])
        if not isinstance(comps, list | tuple) or any(
            not isinstance(c, list | tuple) or len(c) != 2 for c in comps
        ):
            raise ValidationError("comparisons must be a list of [a, b] pairs")
        cfg = d.get("config", {})
        if not isinstance(cfg, Mapping):
            raise ValidationError("config must be an object")
        return cls(
            str(d["baseline_run"]),
            tuple(SliceSpec.from_dict(s) for s in raw),
            tuple((str(c[0]), str(c[1])) for c in comps),
            lst("fault_experiments"),
            lst("failure_modes"),
            lst("interactions"),
            an.SliceConfig.from_dict(cfg),
        )


@dataclass(frozen=True)
class SliceRequest:
    investigation_id: str
    spec: Mapping[str, object]

    def to_parameters(self) -> dict[str, object]:
        return {
            "slice_analysis": {"investigation_id": self.investigation_id, "spec": dict(self.spec)}
        }

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, object]) -> "SliceRequest":
        if set(parameters) != {"slice_analysis"}:
            raise ValidationError("a slice configuration must be exactly {'slice_analysis': ...}")
        body = _thaw(parameters["slice_analysis"])
        if (
            not isinstance(body, dict)
            or set(body) != {"investigation_id", "spec"}
            or not isinstance(body["spec"], dict)
        ):
            raise ValidationError("'slice_analysis' has the wrong fields")
        return cls(str(body["investigation_id"]), body["spec"])


# -- computation ----------------------------------------------------------------------------------------------


@dataclass
class Computed:
    baseline: Baseline
    memberships: dict[str, Membership]  # by slice name
    results: dict[str, Any]
    unavailable: dict[str, str]  # slice name -> why it could not be evaluated


def compute(
    reg: Registry, store: ArtifactStore, spec: SliceAnalysisSpec, dataset: Any | None
) -> Computed:
    """All numbers of an analysis, deterministic given the registry content. Pure with respect to
    the registry (reads only)."""
    base = load_baseline(reg, store, spec.baseline_run)
    if base.dataset_fingerprint is None:
        raise SliceDataError("the baseline evaluation records no dataset fingerprint")
    mems: dict[str, Membership] = {}
    unavailable: dict[str, str] = {}
    for s in spec.slices:
        try:
            table = sample_table(base, dataset, s.fields)
            mems[s.name] = evaluate(s, table)
        except SliceDataError as exc:
            unavailable[s.name] = str(exc)
    cfg = spec.config
    per_slice: dict[str, Any] = {}
    for s in spec.slices:
        if s.name in unavailable:
            per_slice[s.name] = {"status": "UNAVAILABLE", "reason": unavailable[s.name]}
            continue
        m = mems[s.name]
        if m.status is MembershipStatus.COMPUTED:
            per_slice[s.name] = {"status": "COMPUTED", **an.slice_metrics(base, m.sample_ids, cfg)}
        else:
            per_slice[s.name] = {
                "status": "UNDEFINED"
                if m.status is MembershipStatus.MISSING_FIELD
                else m.status.value,
                "n_samples": 0,
                "reason": m.reason or f"membership is {m.status.value}",
            }
    pairs = spec.comparisons or tuple((s.name, POPULATION) for s in spec.slices)
    comps: dict[str, dict[str, Any]] = {}
    for a, b in pairs:
        key = f"{a} vs {b}"
        if a not in mems or (b not in (POPULATION, REST) and b not in mems):
            comps[key] = {"status": "UNAVAILABLE", "reason": "a slice could not be evaluated"}
            continue
        if mems[a].status is not MembershipStatus.COMPUTED:
            comps[key] = {
                "status": an.INSUFFICIENT,
                "reason": f"{a} has no members ({mems[a].status.value})",
            }
            continue
        if b == POPULATION:
            comps[key] = an.compare_to_population(base, mems[a].sample_ids, cfg)
            comps[key]["status"] = comps[key]["vs_rest"].get("status", "UNDEFINED")
            comps[key]["inference"] = comps[key]["vs_rest"].get("inference")
        else:
            rest = [i for i in sorted(base.rows) if i not in set(mems[a].sample_ids)]
            other = rest if b == REST else list(mems[b].sample_ids)
            comps[key] = an.compare_groups(base, mems[a].sample_ids, other, cfg)
    correction = an.correct_family(comps, cfg)
    faults: dict[str, Any] = {}
    modes: dict[str, Any] = {}
    inter: dict[str, Any] = {}
    for s in spec.slices:
        if s.name in unavailable:
            continue
        m = mems[s.name]
        if spec.fault_experiments:
            faults[s.name] = [
                an.fault_slice(reg, store, base, m, f, cfg)
                if s.static
                else {
                    "fault_experiment_id": f,
                    "slice_id": s.slice_id,
                    "status": "REFUSED",
                    "reason": "the slice reads model-output fields (predicted/correct/confidence), so its membership differs between runs",
                }
                for f in spec.fault_experiments
            ]
        if spec.failure_modes:
            modes[s.name] = an.failure_slice(reg, store, base, m, spec.failure_modes, cfg)
        if spec.interactions:
            inter[s.name] = [
                an.interaction_slice(reg, store, base, m, i, cfg)
                if s.static
                else {
                    "interaction_id": i,
                    "slice_id": s.slice_id,
                    "status": "REFUSED",
                    "reason": "the slice reads model-output fields, so its membership differs between runs",
                }
                for i in spec.interactions
            ]
    results = {
        "measure": an.measure_of(base.task)[0],
        "slices": per_slice,
        "comparisons": comps,
        "multiple_comparisons": correction,
        "faults": faults,
        "failure_modes": modes,
        "interactions": inter,
        "note": an.ENGINE_NOTE,
    }
    return Computed(base, mems, results, unavailable)


def summarize(spec: SliceAnalysisSpec, c: Computed) -> dict[str, Any]:
    slices = {}
    for s in spec.slices:
        r = c.results["slices"][s.name]
        m = c.memberships.get(s.name)
        headline = None
        if r.get("status") == "COMPUTED":
            headline = next(
                (
                    x
                    for x in r["metrics"]
                    if x["metric_id"] == c.results["measure"] and x["status"] == "COMPUTED"
                ),
                None,
            )
        slices[s.name] = {
            "slice_id": s.slice_id,
            "status": r["status"],
            "n_members": None if m is None else m.n_members,
            "prevalence": None if m is None else m.prevalence,
            "evidence": r.get("evidence"),
            "headline": None
            if headline is None
            else {
                "metric_id": headline["metric_id"],
                "value": headline["value"],
                "interval": headline["interval"],
            },
        }

    def statuses(block: Mapping[str, Any]) -> dict[str, int]:
        out: dict[str, int] = {}
        for v in block.values():
            for x in v if isinstance(v, list) else [v]:
                out[x["status"]] = out.get(x["status"], 0) + 1
        return dict(sorted(out.items()))

    return {
        "baseline_run_id": spec.baseline_run,
        "dataset_fingerprint": c.baseline.dataset_fingerprint,
        "measure": c.results["measure"],
        "n_samples": len(c.baseline.rows),
        "slices": slices,
        "comparisons": {
            k: {
                "status": v.get("status"),
                "difference": (
                    v.get("descriptive") or v.get("descriptive_vs_population") or {}
                ).get("difference"),
            }
            for k, v in c.results["comparisons"].items()
        },
        "fault_status_counts": statuses(c.results["faults"]),
        "failure_mode_status_counts": statuses(
            {k: v["modes"] if v.get("modes") else v for k, v in c.results["failure_modes"].items()}
        ),
        "interaction_status_counts": statuses(c.results["interactions"]),
        "correction": {
            k: c.results["multiple_comparisons"][k] for k in ("method", "alpha", "n_hypotheses")
        },
        "note": an.ENGINE_NOTE,
    }


def _write(ctx: RunContext, name: str, payload: object) -> str:
    directory = ctx.artifact_dir / "slice"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(
        json.dumps(to_jsonable(payload), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return ctx.register_artifact(
        f"slice/{name}.json", name=f"slice-{name}", media_type="application/json"
    ).id


def provenance_fingerprint(spec: SliceAnalysisSpec, c: Computed) -> str:
    return content_hash(
        {
            "dataset_fingerprint": c.baseline.dataset_fingerprint,
            "split": c.baseline.split,
            "evaluation_config_hash": c.baseline.evaluation_config_hash,
            "baseline_run": spec.baseline_run,
            "slices": {
                s.name: {
                    "slice_id": s.slice_id,
                    "membership": (
                        c.memberships[s.name].membership_digest if s.name in c.memberships else None
                    ),
                }
                for s in spec.slices
            },
            "metric_config": spec.config.to_dict(),
            "sources": {
                "faults": list(spec.fault_experiments),
                "modes": list(spec.failure_modes),
                "interactions": list(spec.interactions),
            },
            "analysis_version": an.ANALYSIS_VERSION,
        }
    )


def run_slice_analysis(ctx: RunContext) -> None:
    req = SliceRequest.from_parameters(ctx.parameters)
    spec = SliceAnalysisSpec.from_dict(req.spec)
    reg, now = ctx.registry, ctx.started_at
    if ctx.dataset is not None:
        base_fp = reg.get(Run, spec.baseline_run)  # existence check before the heavy work
        del base_fp
    c = compute(reg, ctx.store, spec, ctx.dataset)
    if ctx.dataset is not None and ctx.dataset.fingerprint() != c.baseline.dataset_fingerprint:
        raise SliceDataError(
            "the loaded dataset's fingerprint differs from the baseline evaluation's"
        )
    summary = summarize(spec, c)
    prov = provenance_fingerprint(spec, c)
    art = {
        "spec": _write(
            ctx,
            "spec",
            {
                "spec_id": spec.spec_id,
                "spec": spec.to_dict(),
                "slice_ids": {s.name: s.slice_id for s in spec.slices},
                "provenance_fingerprint": prov,
                "dataset_fingerprint": c.baseline.dataset_fingerprint,
                "split": c.baseline.split,
                "evaluation_config_hash": c.baseline.evaluation_config_hash,
                "source_runs": {
                    "baseline": spec.baseline_run,
                    "fault_experiments": list(spec.fault_experiments),
                    "failure_modes": list(spec.failure_modes),
                    "interactions": list(spec.interactions),
                },
            },
        ),
        "membership": _write(
            ctx,
            "membership",
            {n: {**asdict(m), "sample_ids": list(m.sample_ids)} for n, m in c.memberships.items()}
            | {n: {"status": "UNAVAILABLE", "reason": why} for n, why in c.unavailable.items()},
        ),
        "results": _write(ctx, "results", c.results),
        "summary": _write(ctx, "summary", summary),
    }
    partial = (
        bool(c.unavailable)
        or any(
            k != "COMPUTED"
            for blk in (
                "fault_status_counts",
                "failure_mode_status_counts",
                "interaction_status_counts",
            )
            for k in summary[blk]
        )
        or any(
            s["status"] != "COMPUTED" or s["evidence"] == an.INSUFFICIENT
            for s in summary["slices"].values()
        )
    )
    analysis = SliceAnalysis(
        req.investigation_id,
        ctx.run.id,
        spec.spec_id,
        spec.to_dict(),
        spec.baseline_run,
        str(c.baseline.dataset_fingerprint),
        prov,
        "PARTIAL" if partial else "COMPLETE",
        {**summary, "artifacts": art},
        now,
    )
    new = not reg.exists(SliceAnalysis, analysis.id)
    if new:
        with reg.transaction():
            for s in spec.slices:
                sl = Slice.of(s, now)
                if not reg.exists(Slice, sl.id):
                    reg.add(sl)
            reg.add(analysis)
    ctx.observe("slice.new_record", int(new), kind=EpistemicKind.OBSERVATION)
    ctx.observe("slice.slices", len(spec.slices), kind=EpistemicKind.OBSERVATION)
    ctx.observe(
        "slice.comparisons", len(c.results["comparisons"]), kind=EpistemicKind.DERIVED_METRIC
    )


# -- entry point -------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SliceRunResult:
    experiment_id: str
    run_id: str
    status: RunStatus
    analysis_id: str | None


def validate(reg: Registry, spec: SliceAnalysisSpec) -> Run:
    """Refuse before anything is created: the baseline must be a completed run with an evaluation,
    and every referenced fault experiment / mode / interaction must exist."""
    run = reg.get(Run, spec.baseline_run)
    if run.status is not RunStatus.COMPLETED:
        raise ValidationError(f"baseline run {run.id} is {run.status.value}, not COMPLETED")
    for cls_, ids in (
        (FaultExperiment, spec.fault_experiments),
        (FailureMode, spec.failure_modes),
        (InteractionAnalysis, spec.interactions),
    ):
        for i in ids:
            reg.get(cls_, i)
    return run


def run_slice_analysis_request(
    registry: Registry,
    store: ArtifactStore,
    executor: Executor,
    investigation_id: str,
    spec: SliceAnalysisSpec,
    *,
    seed: int = 0,
) -> SliceRunResult:
    run = validate(registry, spec)
    base_exp = registry.get(Experiment, run.experiment_id)
    conf = ConfigurationRef(SliceRequest(investigation_id, spec.to_dict()).to_parameters())
    if not registry.exists(ConfigurationRef, conf.id):
        registry.add(conf)
    exp = Experiment(
        investigation_id,
        "slice analysis",
        "Deterministic slice and subgroup analysis over stored runs",
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
        for a in registry.find(SliceAnalysis, spec_id=spec.spec_id)
        if a.investigation_id == investigation_id
    ]
    return SliceRunResult(exp.id, result.run.id, result.status, found[0].id if found else None)


__all__ = [
    "PROCEDURE",
    "ExperionyxError",
    "SliceAnalysisSpec",
    "compute",
    "run_slice_analysis",
    "run_slice_analysis_request",
    "validate",
]

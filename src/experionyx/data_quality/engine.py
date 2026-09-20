"""The quality-analysis procedure, run by the normal execution engine so each analysis is a Run with
provenance, artifacts and observations and can be replayed.

Reuse: the dataset comes from the registered dataset adapter (fingerprint-verified), slice membership
from the Phase 11 evaluator, windows from Phase 12, cleaning and target-distribution comparison from
the Phase 12 measures, and every interval, effect size, test and correction from Phase 10.

No check is combined with another into a score; the summary counts statuses per check type and scope
and says nothing more."""

import json
import platform
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from experionyx.adapters.records import RegisteredDataset
from experionyx.artifacts import ArtifactStore
from experionyx.data_quality import checks as ck
from experionyx.data_quality.data import QualityDataError, Table, dataset_columns, load_table
from experionyx.data_quality.entities import QualityAnalysis, QualityCheck
from experionyx.data_quality.results import NOTE, CheckResult, Status
from experionyx.data_quality.spec import ANALYSIS_VERSION, PER_TABLE, QUALITY_SCHEMA, QualitySpec
from experionyx.domain import (
    ConfigurationRef,
    EpistemicKind,
    Experiment,
    ExperimentStatus,
    Investigation,
    ModelRef,
    RunStatus,
    to_jsonable,
)
from experionyx.errors import ArtifactIntegrityError, ExperionyxError, ValidationError
from experionyx.evaluation.config import _thaw
from experionyx.execution import Executor, RunContext, resolve_procedure
from experionyx.faults.report import read_artifact
from experionyx.hashing import content_hash
from experionyx.interactions.lifecycle import REPLAY_TOLERANCE, _compare
from experionyx.registry import Registry
from experionyx.slices.data import SliceDataError, load_dataset_record
from experionyx.slices.evaluate import Membership, MembershipStatus, build_table, evaluate
from experionyx.stats import core as st

PROCEDURE = "experionyx.data_quality.engine:run_quality_analysis"
DOCUMENTS = ("spec", "checks", "observations", "violations", "summary")


@dataclass(frozen=True)
class QualityRequest:
    investigation_id: str
    spec: Mapping[str, object]

    def to_parameters(self) -> dict[str, object]:
        return {
            "quality_analysis": {"investigation_id": self.investigation_id, "spec": dict(self.spec)}
        }

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, object]) -> "QualityRequest":
        if set(parameters) != {"quality_analysis"}:
            raise ValidationError(
                "a quality configuration must be exactly {'quality_analysis': ...}"
            )
        body = _thaw(parameters["quality_analysis"])
        if (
            not isinstance(body, dict)
            or set(body) != {"investigation_id", "spec"}
            or not isinstance(body["spec"], dict)
        ):
            raise ValidationError("'quality_analysis' has the wrong fields")
        return cls(str(body["investigation_id"]), body["spec"])


@dataclass
class Computed:
    record: RegisteredDataset
    tables: dict[str | None, Table]
    columns: tuple[str, ...]
    results: list[CheckResult]
    slice_membership: dict[str, dict[str, str]]  # slice name -> split -> membership digest / status


def _splits(spec: QualitySpec, dataset: Any) -> list[str | None]:
    declared = tuple(dataset.splits())
    want: list[str | None] = [*spec.splits] or [*declared] or [None]
    missing = [s for s in want if s is not None and s not in declared]
    if missing:
        raise QualityDataError(
            f"split(s) {missing} are not declared by the dataset (declared: {list(declared)})"
        )
    return want


def load_tables(spec: QualitySpec, dataset: Any) -> tuple[tuple[str, ...], dict[str | None, Table]]:
    columns = dataset_columns(dataset)
    tables = {s: load_table(dataset, s, columns) for s in _splits(spec, dataset)}
    named = set(columns)
    need: dict[str, str] = {f.name: "a declared feature" for f in spec.features}
    if spec.id_column:
        need[spec.id_column] = "the id_column"
    if spec.ordering and spec.ordering.field != "index":
        need[spec.ordering.field[len("feature:") :]] = "the ordering feature"
    for name, what in need.items():
        if name not in named:
            raise QualityDataError(
                f"{name!r} ({what}) is not a column of this dataset (columns: {list(columns)})"
            )
    return columns, tables


def _slice_table(t: Table, fields: tuple[str, ...]) -> Any:
    cols = {f: t.column(f[len("feature:") :]) for f in fields if f.startswith("feature:")}
    rows = []
    for p, i in enumerate(t.ids):
        row: dict[str, object] = {f: c[p] for f, c in cols.items()}
        if t.target is not None and "target" in fields:
            row["target"] = t.target[p]
        rows.append((i, row))
    return build_table(rows)


def compute(spec: QualitySpec, record: RegisteredDataset, dataset: Any) -> Computed:
    """All results of an analysis, deterministic given the dataset. Pure with respect to the registry."""
    if dataset.fingerprint() != record.fingerprint:
        raise QualityDataError("the loaded dataset's fingerprint differs from the registered one")
    columns, tables = load_tables(spec, dataset)
    ctx = ck.Ctx(spec, tables, frozenset(spec.missing_values))
    results: list[CheckResult] = []
    mem_info: dict[str, dict[str, str]] = {s.name: {} for s in spec.slices}
    slice_tabs: dict[tuple[str, str | None], tuple[Table | None, Status, str | None]] = {}
    for s in spec.slices:
        for split, t in tables.items():
            key = (s.name, split)
            try:
                m: Membership = evaluate(s, _slice_table(t, s.fields))
            except ExperionyxError as exc:
                slice_tabs[key] = (
                    None,
                    Status.UNAVAILABLE,
                    f"slice membership could not be evaluated: {exc}",
                )
                mem_info[s.name][str(split)] = "UNAVAILABLE"
                continue
            mem_info[s.name][str(split)] = f"{m.status.value}:{m.membership_digest}"
            if m.status is MembershipStatus.COMPUTED:
                pos = {i: p for p, i in enumerate(t.ids)}
                slice_tabs[key] = (t.select([pos[int(i)] for i in m.sample_ids]), Status.PASS, None)
            elif m.status is MembershipStatus.MISSING_FIELD:
                slice_tabs[key] = (
                    None,
                    Status.UNAVAILABLE,
                    f"the slice reads a field this dataset does not provide: {m.reason}",
                )
            else:
                slice_tabs[key] = (
                    None,
                    Status.NOT_APPLICABLE,
                    f"NO_MEMBERS: the slice has no members in this split ({m.status.value})",
                )
    for cs in spec.checks:
        if cs.type in PER_TABLE:
            fn = ck.PER_TABLE_FUNCS[cs.type]
            for split, t in tables.items():
                results.append(fn(cs, ctx, t, {"split": split}))
                if cs.type in spec.config.slice_checks:
                    for s in spec.slices:
                        sub, why_status, why = slice_tabs[(s.name, split)]
                        scope = {"split": split, "slice": s.name, "slice_id": s.slice_id}
                        if sub is None:
                            results.append(
                                CheckResult(cs.check_id, cs.type, scope, why_status, why)
                            )
                            continue
                        r = fn(cs, ctx, sub, scope)
                        if sub.n < spec.config.min_members and r.status in (
                            Status.PASS,
                            Status.WARNING,
                        ):
                            r = CheckResult(
                                r.check_id,
                                r.check_type,
                                r.scope,
                                Status.INCONCLUSIVE,
                                f"the slice has {sub.n} row(s) in this split, fewer than min_members={spec.config.min_members}: values are reported, not reliable",
                                r.observations,
                                r.violations,
                                r.thresholds,
                                r.evidence,
                                r.statistics,
                            )
                        results.append(r)
        else:
            results.append(ck.CROSS_FUNCS[cs.type](cs, ctx))
    return Computed(record, tables, columns, results, mem_info)


# -- documents ----------------------------------------------------------------------------------------------


def versions() -> dict[str, str]:
    return {
        "data_quality": ANALYSIS_VERSION,
        "quality_schema": str(QUALITY_SCHEMA),
        "statistics": st.STATS_VERSION,
        "python": platform.python_version(),
    }


def _scope_key(r: CheckResult) -> str:
    return r.scope_key


def summarize(spec: QualitySpec, c: Computed) -> dict[str, Any]:
    by_type: dict[str, dict[str, int]] = {}
    overall: Counter[str] = Counter()
    for r in c.results:
        by_type.setdefault(r.check_type, Counter())[r.status.value] += 1
        overall[r.status.value] += 1
    sev = Counter(v["severity"] for r in c.results for v in r.violations)
    return {
        "dataset_id": spec.dataset_id,
        "dataset_fingerprint": c.record.fingerprint,
        "n_checks": len(spec.checks),
        "n_results": len(c.results),
        "status_counts": dict(sorted(overall.items())),
        "by_check_type": {k: dict(sorted(v.items())) for k, v in sorted(by_type.items())},
        "results": [
            {
                "check_id": r.check_id,
                "check_type": r.check_type,
                "scope": r.scope_key,
                "status": r.status.value,
            }
            for r in sorted(c.results, key=lambda r: (r.check_type, r.scope_key, r.check_id))
        ],
        "violation_counts": dict(sorted(sev.items())),
        "splits": {str(s): t.n for s, t in c.tables.items()},
        "note": NOTE
        + "; there is no data-quality score, no ranking and no overall verdict: quality is multidimensional and every status is scoped to one check and one scope",
    }


def provenance_fingerprint(spec: QualitySpec, c: Computed) -> str:
    return content_hash(
        {
            "dataset_fingerprint": c.record.fingerprint,
            "adapter": [c.record.adapter, c.record.adapter_version],
            "dataset_version": c.record.version,
            "columns": list(c.columns),
            "splits": {str(s): [t.n, t.digest()] for s, t in c.tables.items()},
            "spec": {k: v for k, v in spec.to_dict().items() if k != "slices"},
            "check_ids": [x.check_id for x in spec.checks],
            "slices": {
                s.name: {"slice_id": s.slice_id, "membership": c.slice_membership[s.name]}
                for s in spec.slices
            },
            "versions": {k: v for k, v in versions().items() if k != "python"},
        }
    )


def spec_doc(spec: QualitySpec, c: Computed, fp: str) -> dict[str, Any]:
    rec = c.record
    return {
        "spec_id": spec.spec_id,
        "spec": spec.to_dict(),
        "provenance_fingerprint": fp,
        "dataset": {
            "id": rec.id,
            "name": rec.name,
            "version": rec.version,
            "fingerprint": rec.fingerprint,
            "adapter": rec.adapter,
            "adapter_version": rec.adapter_version,
        },
        "columns": list(c.columns),
        "splits": {
            str(s): {"n_rows": t.n, "sample_digest": t.digest()} for s, t in c.tables.items()
        },
        "check_ids": [{"check_id": x.check_id, "type": x.type} for x in spec.checks],
        "thresholds": {
            x.check_id: {k: v for k, v in x.config.items() if v not in (None, [], {}, "")}
            for x in spec.checks
        },
        "random_seed": spec.config.seed,
        "versions": versions(),
        "missing_values": list(spec.missing_values),
        "slice_ids": {s.name: s.slice_id for s in spec.slices},
        "slice_membership": c.slice_membership,
        "source_revision": "recorded in the run's Provenance record (see `data-quality inspect`)",
    }


def checks_doc(spec: QualitySpec, c: Computed) -> dict[str, Any]:
    return {
        "checks": [
            {
                **x.to_dict(),
                "check_id": x.check_id,
                "results": [
                    {
                        "scope": r.scope_key,
                        "status": r.status.value,
                        "reason": r.reason,
                        "thresholds": r.thresholds,
                        "evidence": r.evidence,
                        "n_violations": len(r.violations),
                    }
                    for r in c.results
                    if r.check_id == x.check_id
                ],
            }
            for x in spec.checks
        ]
    }


def observations_doc(c: Computed) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for r in c.results:
        out.setdefault(r.check_id, {})[r.scope_key] = {
            "observations": r.observations,
            "statistics": r.statistics,
            "note": r.note,
        }
    return out


def violations_doc(c: Computed) -> dict[str, Any]:
    rows = [
        {"check_id": r.check_id, "check_type": r.check_type, "scope": r.scope_key, **v}
        for r in c.results
        for v in r.violations
    ]
    return {
        "violations": rows,
        "note": "counts are exact; affected_rows lists at most `examples` sample IDs (truncated says whether more exist); a violation is a finding under a configured rule or an observation needing interpretation, not a diagnosis",
    }


def _write(ctx: RunContext, name: str, payload: object) -> str:
    directory = ctx.artifact_dir / "data_quality"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(
        json.dumps(to_jsonable(payload), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return ctx.register_artifact(
        f"data_quality/{name}.json", name=f"data-quality-{name}", media_type="application/json"
    ).id


def run_quality_analysis(ctx: RunContext) -> None:
    req = QualityRequest.from_parameters(ctx.parameters)
    spec = QualitySpec.from_dict(req.spec)
    reg, now = ctx.registry, ctx.started_at
    if ctx.dataset is None:
        raise QualityDataError("the experiment has no loaded dataset")
    record = reg.get(RegisteredDataset, spec.dataset_id)
    c = compute(spec, record, ctx.dataset)
    summary = summarize(spec, c)
    fp = provenance_fingerprint(spec, c)
    art = {
        "spec": _write(ctx, "spec", spec_doc(spec, c, fp)),
        "checks": _write(ctx, "checks", checks_doc(spec, c)),
        "observations": _write(ctx, "observations", observations_doc(c)),
        "violations": _write(ctx, "violations", violations_doc(c)),
        "summary": _write(ctx, "summary", summary),
    }
    partial = any(r.status in (Status.INCONCLUSIVE, Status.UNAVAILABLE) for r in c.results)
    analysis = QualityAnalysis(
        req.investigation_id,
        ctx.run.id,
        spec.spec_id,
        spec.to_dict(),
        spec.dataset_id,
        record.fingerprint,
        fp,
        "PARTIAL" if partial else "COMPLETE",
        {**summary, "artifacts": art},
        now,
    )
    new = not reg.exists(QualityAnalysis, analysis.id)
    if new:
        with reg.transaction():
            reg.add(analysis)
            for r in c.results:
                q = QualityCheck(
                    analysis.id,
                    r.check_id,
                    r.check_type,
                    r.scope_key,
                    r.status.value,
                    r.reason,
                    r.evidence,
                    now,
                )
                if not reg.exists(QualityCheck, q.id):
                    reg.add(q)
    ctx.observe("quality.new_record", int(new), kind=EpistemicKind.OBSERVATION)
    ctx.observe("quality.results", len(c.results), kind=EpistemicKind.OBSERVATION)
    ctx.observe(
        "quality.failed_results",
        sum(r.status is Status.FAIL for r in c.results),
        kind=EpistemicKind.DERIVED_METRIC,
    )


# -- entry points -------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class QualityRunResult:
    experiment_id: str
    run_id: str
    status: RunStatus
    analysis_id: str | None


def baseline_investigation(reg: Registry, requested: str | None) -> str:
    """The investigation a quality analysis belongs to: the requested one, or the only one there is."""
    found = sorted(i.id for i in reg.find(Investigation))
    if requested is not None:
        reg.get(Investigation, requested)
        return requested
    if len(found) != 1:
        raise ValidationError(
            f"{len(found)} investigations exist; say which one with --investigation"
        )
    return found[0]


def validate(reg: Registry, spec: QualitySpec) -> RegisteredDataset:
    return reg.get(RegisteredDataset, spec.dataset_id)


def load_dataset(reg: Registry, adapters: Any, inputs_root: Any, spec: QualitySpec) -> Any:
    if adapters is None:
        raise QualityDataError("no adapter registry is available to load the dataset")
    try:
        return load_dataset_record(adapters, inputs_root, validate(reg, spec))
    except SliceDataError as exc:
        raise QualityDataError(str(exc)) from exc


def preflight(reg: Registry, spec: QualitySpec, dataset: Any) -> dict[str | None, Table]:
    """Data-dependent refusals without running anything: the dataset loads and matches its
    registered fingerprint, the declared splits, features, id column and ordering exist."""
    rec = validate(reg, spec)
    if dataset.fingerprint() != rec.fingerprint:
        raise QualityDataError("the loaded dataset's fingerprint differs from the registered one")
    return load_tables(spec, dataset)[1]


def run_quality_request(
    registry: Registry,
    store: ArtifactStore,
    executor: Executor,
    investigation_id: str,
    spec: QualitySpec,
    *,
    dataset: Any | None = None,
    seed: int = 0,
) -> QualityRunResult:
    rec = validate(registry, spec)
    registry.get(Investigation, investigation_id)
    if dataset is None:
        dataset = load_dataset(registry, executor.adapters, executor.inputs_root, spec)
    preflight(registry, spec, dataset)  # nothing is created if the data cannot support the request
    conf = ConfigurationRef(QualityRequest(investigation_id, spec.to_dict()).to_parameters())
    if not registry.exists(ConfigurationRef, conf.id):
        registry.add(conf)
    exp = Experiment(
        investigation_id,
        "data quality analysis",
        "Deterministic data-quality checks over a registered dataset",
        ModelRef("none", "0"),
        rec.ref(),
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
        for a in registry.find(QualityAnalysis, spec_id=spec.spec_id)
        if a.investigation_id == investigation_id
    ]
    return QualityRunResult(exp.id, result.run.id, result.status, found[0].id if found else None)


def replay_check(
    reg: Registry, store: ArtifactStore, executor: Executor, analysis_id: str
) -> dict[str, object]:
    """Replay the analysis Run as a NEW run and compare every stored document within REPLAY_TOLERANCE."""
    a = reg.get(QualityAnalysis, analysis_id)
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
            x = read_artifact(reg, store, a.run_id, f"data_quality/{name}.json")
        except ArtifactIntegrityError:
            raise
        except ExperionyxError:
            diffs.append(f"{name}: missing from the original run")
            continue
        _compare(
            name, x, read_artifact(reg, store, replay.run.id, f"data_quality/{name}.json"), diffs
        )
    return {
        **out,
        "deterministic": not diffs,
        "differences": diffs[:50],
        "compared": list(DOCUMENTS),
    }


__all__ = [
    "PROCEDURE",
    "QualityRunResult",
    "compute",
    "preflight",
    "replay_check",
    "run_quality_analysis",
    "run_quality_request",
    "validate",
]

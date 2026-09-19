"""Protocol expansion and validation. A BenchmarkSpec becomes an explicit list of experiment
units (baseline, one per fault grid x parameter point x seed, and one per interaction cell x
seed); nothing is hidden, interpolated or skipped. Every problem is found BEFORE any run exists
and reported with what was required, what was found and why."""

from dataclasses import dataclass
from typing import Any

from experionyx.adapters.records import RegisteredDataset, RegisteredModel
from experionyx.benchmark.spec import ENGINE_VERSION, BenchmarkSpec, FaultGrid, InteractionPair
from experionyx.benchmark.taxonomy import UnitKind
from experionyx.domain import to_jsonable
from experionyx.errors import (
    BenchmarkRefusal,
    ExperionyxError,
    FaultCompatibilityError,
    FaultError,
    ValidationError,
)
from experionyx.evaluation.metrics import default_metric_registry
from experionyx.faults.design import FaultDesign, FaultLimits, SweepSpec
from experionyx.faults.lab import preflight, trial_spec, with_seed
from experionyx.faults.spec import FaultRegistry, FaultScope, FaultSpec, ScopeKind
from experionyx.hashing import content_hash
from experionyx.interactions.design import DesignIssue
from experionyx.registry import Registry


@dataclass(frozen=True)
class Unit:
    key: str  # deterministic, human-readable identity within the benchmark
    kind: UnitKind
    grid: str | None = None
    pair: str | None = None
    cell: str | None = None  # A | B | AB | BA for interaction trials
    point_index: int | None = None
    parameter: str | None = None
    value: float | None = None
    seed: int | None = None
    fault: dict[str, Any] | None = None  # the exact, seeded fault specification
    fault_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "kind": self.kind.value,
            "grid": self.grid,
            "pair": self.pair,
            "cell": self.cell,
            "point_index": self.point_index,
            "parameter": self.parameter,
            "value": self.value,
            "seed": self.seed,
            "fault_id": self.fault_id,
            "fault": self.fault,
        }


@dataclass(frozen=True)
class Protocol:
    spec_id: str
    protocol_hash: str
    units: tuple[Unit, ...]
    resolved: dict[str, dict[str, str]]  # grid -> {type, version} actually used
    unsupported: tuple[dict[str, str], ...]
    warnings: tuple[str, ...]
    pairs: dict[
        str, dict[str, Any]
    ]  # interaction pair -> which grid points its A and B cells REUSE
    engine_version: str = ENGINE_VERSION

    def by_kind(self, kind: UnitKind) -> list[Unit]:
        return [u for u in self.units if u.kind is kind]


def base_fault(fr: FaultRegistry, g: FaultGrid, seed: int) -> FaultSpec:
    scope = (
        FaultScope(ScopeKind.RANDOM_SUBSET, fraction=g.scope_fraction)
        if g.scope_fraction is not None
        else None
    )
    params = dict(g.parameters)
    if (
        g.sweep_parameter is not None
    ):  # the swept parameter needs a placeholder; every point replaces it explicitly
        params[g.sweep_parameter] = g.values[0]
    return fr.make(g.fault, seed=seed, scope=scope, version=g.fault_version, **params)


def point_fault(fr: FaultRegistry, g: FaultGrid, value: float | None, seed: int) -> FaultSpec:
    """The exact fault for one grid point and seed (identical to what the fault laboratory builds)."""
    return trial_spec(base_fault(fr, g, seed), g.sweep_parameter, value, seed)


def design_for(spec: BenchmarkSpec, base: FaultSpec, g: FaultGrid) -> FaultDesign:
    return FaultDesign(
        base.to_dict(), spec.evaluation, seeds=spec.seeds,
        sweep=SweepSpec(g.sweep_parameter, g.values) if g.sweep_parameter else None,
        primary_metric=spec.primary_metric, limits=FaultLimits(max_failed_trials=spec.limits.max_failed_trials),
        aggregation_confidence=spec.aggregation_confidence, aggregation_resamples=spec.aggregation_resamples,
        aggregation_seed=spec.aggregation_seed,
    )  # fmt: skip


def pair_bases(
    fr: FaultRegistry, spec: BenchmarkSpec, p: InteractionPair
) -> tuple[FaultSpec, FaultSpec]:
    grids = {g.name: g for g in spec.faults}
    ga, gb = grids[p.a], grids[p.b]
    sa = trial_spec(base_fault(fr, ga, spec.seeds[0]), ga.sweep_parameter, p.a_value, spec.seeds[0])
    sb = trial_spec(base_fault(fr, gb, spec.seeds[0]), gb.sweep_parameter, p.b_value, spec.seeds[0])
    return sa, sb


def expand(registry: Registry, spec: BenchmarkSpec, fr: FaultRegistry) -> Protocol:
    """Validate the spec against the registry and fault library and expand it. Raises
    BenchmarkRefusal listing EVERY issue; nothing has been executed or recorded."""
    issues: list[DesignIssue] = []
    warnings: list[str] = []
    unsupported: list[dict[str, str]] = []

    def issue(code: str, required: str, found: str, why: str, where: str | None = None) -> None:
        issues.append(DesignIssue(code, required, found, why, where))

    try:
        registry.get(RegisteredModel, spec.model)
    except ExperionyxError:
        issue(
            "MODEL_MISSING",
            "a registered model",
            spec.model,
            "the benchmark has nothing to evaluate",
        )
    data: RegisteredDataset | None = None
    try:
        data = registry.get(RegisteredDataset, spec.dataset)
    except ExperionyxError:
        issue(
            "DATASET_MISSING",
            "a registered dataset",
            spec.dataset,
            "the benchmark has nothing to evaluate on",
        )
    metrics = default_metric_registry()
    for m in filter(None, [spec.primary_metric, *spec.evaluation.metrics]):
        try:
            metrics.get(m)
        except ExperionyxError:
            issue(
                "UNKNOWN_METRIC", "a registered metric", m, f"registered: {sorted(metrics.ids())}"
            )
    if spec.interaction_config.order_analysis:
        issue(
            "ORDER_ANALYSIS_ON_CONFIG",
            "order_analysis set per interaction pair",
            "set on interaction_config",
            "the pair decides whether its reverse order is run",
        )

    resolved: dict[str, dict[str, str]] = {}
    ok_grids: dict[str, FaultGrid] = {}
    for g in spec.faults:
        try:
            ft = fr.resolve(g.fault, g.fault_version)
        except ExperionyxError as exc:
            issue(
                "UNKNOWN_FAULT",
                "a registered fault type and version",
                f"{g.fault} {g.fault_version or ''}".strip(),
                str(exc),
                g.name,
            )
            continue
        if ft.apply is None:
            issue(
                "FAULT_NOT_IMPLEMENTED",
                "an implemented fault",
                g.fault,
                "this fault is reserved vocabulary and cannot run",
                g.name,
            )
            continue
        try:
            for value in g.points():
                for seed in spec.seeds[:1]:
                    fault = point_fault(fr, g, value, seed)
                    if data is not None:
                        preflight(fault, fr, data)
            design = design_for(spec, base_fault(fr, g, spec.seeds[0]), g)
            samples = (
                data.metadata.splits.get(spec.evaluation.split)
                if data is not None and spec.evaluation.split
                else (data.metadata.num_samples if data is not None else None)
            )
            if isinstance(samples, int):
                design.check_sample_budget(samples)
        except FaultCompatibilityError as exc:
            if spec.limits.allow_unsupported:
                unsupported.append({"grid": g.name, "fault": g.fault, "reason": str(exc)})
                resolved[g.name] = {"type": ft.name, "version": ft.version}
                continue
            issue(
                "UNSUPPORTED_FAULT",
                "a fault the dataset can support",
                g.fault,
                f"{exc} (set limits.allow_unsupported to record it in the coverage instead)",
                g.name,
            )
            continue
        except (ValidationError, FaultError, ExperionyxError) as exc:
            issue(
                "INVALID_GRID",
                "a valid fault specification for every parameter point",
                f"{g.fault} {dict(g.parameters)} sweep {g.sweep_parameter}={list(g.values)}",
                str(exc),
                g.name,
            )
            continue
        resolved[g.name] = {"type": ft.name, "version": ft.version}
        ok_grids[g.name] = g
    grids = {g.name: g for g in spec.faults}
    for p in spec.interactions:
        for role, gname, val in (("a", p.a, p.a_value), ("b", p.b, p.b_value)):
            gg = grids.get(gname)
            if gg is None:
                issue(
                    "INTERACTION_UNKNOWN_GRID",
                    "a fault grid defined in this benchmark",
                    gname,
                    f"interaction {p.name}: {role} names no grid",
                    p.name,
                )
            elif gname not in ok_grids:
                issue(
                    "INTERACTION_GRID_UNUSABLE",
                    "a supported grid",
                    gname,
                    f"interaction {p.name}: the grid is unsupported or invalid",
                    p.name,
                )
            elif gg.values and (val is None or val not in gg.values):
                issue(
                    "INTERACTION_VALUE",
                    f"{role}_value one of {list(gg.values)}",
                    str(val),
                    f"interaction {p.name}: the point must be one the grid tests",
                    p.name,
                )
            elif not gg.values and val is not None:
                issue(
                    "INTERACTION_VALUE",
                    "no value for an unswept grid",
                    str(val),
                    f"interaction {p.name}: grid {gname!r} has no sweep",
                    p.name,
                )
        if p.a in ok_grids and p.b in ok_grids and not [i for i in issues if i.cell == p.name]:
            try:
                sa, sb = pair_bases(fr, spec, p)
                fr.compound(sa, sb)
            except (ValidationError, FaultError) as exc:
                issue(
                    "INTERACTION_INVALID", "a composable pair of faults", p.name, str(exc), p.name
                )
        if len(spec.seeds) < spec.interaction_config.min_trials:
            warnings.append(
                f"interaction {p.name}: {len(spec.seeds)} seed(s) < min_trials={spec.interaction_config.min_trials}: no interval can be produced (the label will be INCONCLUSIVE)"
            )
    if issues:
        raise BenchmarkRefusal(tuple(issues))

    units: list[Unit] = [Unit("baseline", UnitKind.BASELINE)]
    for g in spec.faults:
        if g.name not in ok_grids:
            continue
        for i, value in enumerate(g.points()):
            for seed in spec.seeds:
                f = point_fault(fr, g, value, seed)
                units.append(
                    Unit(
                        f"fault:{g.name}:{i}:{seed}",
                        UnitKind.FAULT_TRIAL,
                        grid=g.name,
                        point_index=i,
                        parameter=g.sweep_parameter,
                        value=value,
                        seed=seed,
                        fault=f.to_dict(),
                        fault_id=f.id,
                    )
                )
    pairs: dict[str, dict[str, Any]] = {}
    for p in spec.interactions:
        sa, sb = pair_bases(fr, spec, p)
        ga, gb = grids[p.a], grids[p.b]
        # Cells A and B are the SAME fault specs (same seeds) as the grid points they name, so the
        # interaction reuses those grid trials instead of re-running identical evaluations. Only the
        # compound cells are new units.
        pairs[p.name] = {
            "a_grid": p.a,
            "a_point_index": ga.values.index(p.a_value)
            if ga.values and p.a_value is not None
            else 0,
            "b_grid": p.b,
            "b_point_index": gb.values.index(p.b_value)
            if gb.values and p.b_value is not None
            else 0,
            "order_analysis": p.order_analysis,
            "reuses_grid_trials": True,
        }
        compounds = {
            "AB": fr.compound(sa, sb),
            **({"BA": fr.compound(sb, sa)} if p.order_analysis else {}),
        }
        for cell, base in compounds.items():
            for seed in spec.seeds:
                f = with_seed(base, seed)
                units.append(
                    Unit(
                        f"interaction:{p.name}:{cell}:{seed}",
                        UnitKind.INTERACTION_TRIAL,
                        pair=p.name,
                        cell=cell,
                        seed=seed,
                        fault=f.to_dict(),
                        fault_id=f.id,
                    )
                )
    if len(units) > spec.limits.max_units:
        raise BenchmarkRefusal(
            (
                DesignIssue(
                    "MAX_UNITS",
                    f"at most {spec.limits.max_units} experiment units",
                    f"{len(units)} units",
                    "the expanded protocol exceeds limits.max_units; nothing was executed",
                ),
            )
        )
    keys = [u.key for u in units]
    assert len(keys) == len(set(keys))  # noqa: S101  # unit keys are unique by construction
    proto_hash = content_hash(
        {
            "engine": ENGINE_VERSION,
            "protocol_key": to_jsonable(spec.protocol_key()),
            "faults": resolved,
            "unsupported": unsupported,
            "units": [[u.key, u.fault_id] for u in units],
            "pairs": pairs,
        }
    )
    return Protocol(
        spec.spec_id, proto_hash, tuple(units), resolved, tuple(unsupported), tuple(warnings), pairs
    )


def validation_report(registry: Registry, spec: BenchmarkSpec, fr: FaultRegistry) -> dict[str, Any]:
    """Non-raising form for the CLI."""
    try:
        p = expand(registry, spec, fr)
    except BenchmarkRefusal as exc:
        return {
            "valid": False,
            "spec_id": spec.spec_id,
            "issues": [i.to_dict() for i in exc.issues if isinstance(i, DesignIssue)],
        }
    counts: dict[str, int] = {}
    for u in p.units:
        counts[u.kind.value] = counts.get(u.kind.value, 0) + 1
    return {
        "valid": True,
        "spec_id": spec.spec_id,
        "protocol_hash": p.protocol_hash,
        "engine_version": p.engine_version,
        "units": len(p.units),
        "units_by_kind": counts,
        "resolved_faults": p.resolved,
        "unsupported": list(p.unsupported),
        "warnings": list(p.warnings),
        "unit_keys": [u.key for u in p.units],
    }

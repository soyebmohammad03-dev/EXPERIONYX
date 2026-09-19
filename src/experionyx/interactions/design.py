"""Loading and validating the experimental design. A design is refused, with EVERY problem
listed, before any number is computed; there is no partial analysis of an invalid design."""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from experionyx.artifacts import ArtifactStore
from experionyx.domain import Artifact, Experiment, Run, RunStatus, to_jsonable
from experionyx.errors import DesignRefusal, ExperionyxError
from experionyx.evaluation.loading import load_evaluation
from experionyx.evaluation.results import EvaluationResult, Status
from experionyx.failures.extraction import label_key
from experionyx.failures.sources import load_fault
from experionyx.hashing import content_hash
from experionyx.interactions.config import InteractionSpec
from experionyx.interactions.taxonomy import Pairing
from experionyx.provenance import Provenance
from experionyx.registry import Registry

FAULT_PATH = "fault/fault.json"


@dataclass(frozen=True)
class DesignIssue:
    code: str
    required: str
    found: str
    why: str
    cell: str | None = None

    def __str__(self) -> str:
        where = f"[{self.cell}] " if self.cell else ""
        return f"{where}{self.code}: required {self.required}; found {self.found}; {self.why}"

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "cell": self.cell,
            "required": self.required,
            "found": self.found,
            "why": self.why,
        }


@dataclass(frozen=True)
class Trial:
    cell: str
    run_id: str
    key: str  # the pairing key: the fault seed (treatment) or the run ID (control)
    seed: int
    values: Mapping[str, float]  # measure name -> value (scalar metrics, per-class recall, slices)
    undefined: Mapping[str, str]  # measure name -> why it has no value in this trial
    fault: Mapping[str, object] | None
    provenance_fingerprint: str
    environment_id: str
    n_samples: int


@dataclass(frozen=True)
class LoadedDesign:
    spec: InteractionSpec
    trials: Mapping[str, tuple[Trial, ...]]
    excluded: tuple[
        Mapping[str, object], ...
    ]  # runs left out (not COMPLETED, unusable) with reasons
    direction: Mapping[str, bool | None]  # measure -> higher_is_better
    task: str
    model_fingerprint: str
    dataset_fingerprint: str
    split: str | None
    evaluation_config_hash: str
    descriptor_a: Mapping[str, object]
    descriptor_b: Mapping[str, object]
    pairing: Pairing
    warnings: tuple[str, ...] = ()
    checks: tuple[str, ...] = field(
        default_factory=tuple
    )  # human-readable list of what was verified

    def structural_key(self) -> dict[str, object]:
        """What makes two designs 'the same interaction structure' (seeds and run IDs excluded)."""
        return {
            "fault_a": self.descriptor_a,
            "fault_b": self.descriptor_b,
            "model_fingerprint": self.model_fingerprint,
            "dataset_fingerprint": self.dataset_fingerprint,
            "split": self.split,
            "evaluation_config_hash": self.evaluation_config_hash,
            "order_analysis": "BA" in self.trials,
        }


def descriptor(fault: Mapping[str, object]) -> dict[str, object]:
    """A fault without its seed: type, version, parameters, scope (identity of the fault FAMILY)."""
    return {
        "type": fault["type"],
        "version": fault["version"],
        "parameters": fault.get("parameters", {}),
        "scope": fault.get("scope", {}),
    }


def _components(fault: Mapping[str, object]) -> list[dict[str, object]]:
    comps = fault.get("components") or []
    assert isinstance(comps, list | tuple)  # noqa: S101
    return [descriptor(c) for c in comps] if comps else [descriptor(fault)]


def _measures(
    ev: EvaluationResult,
) -> tuple[dict[str, float], dict[str, str], dict[str, bool | None]]:
    vals: dict[str, float] = {}
    bad: dict[str, str] = {}
    direction: dict[str, bool | None] = {}

    def put(name: str, value: float | None, hib: bool | None, why: str | None = None) -> None:
        direction[name] = hib
        if value is not None and isinstance(value, int | float) and math.isfinite(value):
            vals[name] = float(value)
        else:
            bad[name] = why or "value is missing or not finite"

    for m in ev.metrics:
        if m.structured is None:
            put(
                m.metric_id,
                m.value if m.status is Status.COMPUTED else None,
                m.higher_is_better,
                None if m.status is Status.COMPUTED else f"{m.status.value}: {m.reason}",
            )
    if ev.confusion is not None:
        for c in ev.confusion.per_class:
            put(
                f"class:{label_key(c.label)}:recall",
                c.recall,
                True,
                "recall undefined (no samples of this class)",
            )
    for s in ev.slices:
        for m in s.metrics:
            put(
                f"slice:{s.name}:{m.metric_id}",
                m.value if m.status is Status.COMPUTED else None,
                m.higher_is_better,
            )
    return vals, bad, direction


def _has_fault(registry: Registry, run_id: str) -> bool:
    return any(a.path == FAULT_PATH for a in registry.find(Artifact, run_id=run_id))


def validate(registry: Registry, store: ArtifactStore, spec: InteractionSpec) -> LoadedDesign:
    """Load every cell and check the design. Raises DesignRefusal listing all issues."""
    cfg = spec.config
    issues: list[DesignIssue] = []
    excluded: list[dict[str, object]] = []
    checks: list[str] = []
    per_cell: dict[
        str, list[tuple[Run, EvaluationResult, Provenance | None, Mapping[str, object] | None]]
    ] = {}

    def issue(code: str, required: str, found: str, why: str, cell: str | None = None) -> None:
        issues.append(DesignIssue(code, required, found, why, cell))

    for cell, run_ids in spec.cells().items():
        loaded: list[
            tuple[Run, EvaluationResult, Provenance | None, Mapping[str, object] | None]
        ] = []
        for rid in run_ids:
            try:
                run = registry.get(Run, rid)
            except ExperionyxError:
                issue(
                    "RUN_MISSING",
                    "a registered run",
                    rid,
                    "the cell references a run that does not exist",
                    cell,
                )
                continue
            if run.status is not RunStatus.COMPLETED:
                excluded.append(
                    {"cell": cell, "run_id": rid, "reason": f"run is {run.status.value}"}
                )
                continue
            try:
                ev = load_evaluation(registry, store, rid)
                fault = load_fault(registry, store, rid) if _has_fault(registry, rid) else None
            except ExperionyxError as exc:
                excluded.append({"cell": cell, "run_id": rid, "reason": f"unusable results: {exc}"})
                continue
            provs = registry.find(Provenance, run_id=rid)
            loaded.append((run, ev, provs[0] if provs else None, fault))
        per_cell[cell] = loaded
        if not loaded:
            issue(
                "CELL_EMPTY",
                "at least one COMPLETED run with a stored evaluation",
                f"0 usable of {len(run_ids)}",
                "a cell without usable runs cannot be compared",
                cell,
            )
    if issues:
        raise DesignRefusal(tuple(issues))
    checks.append("every cell has at least one COMPLETED run with a verified stored evaluation")

    # faults: control has none; treatments have one, consistent within the cell
    for cell, rows in per_cell.items():
        for run, _, _, fault in rows:
            if cell == "CONTROL" and fault is not None:
                issue(
                    "CONTROL_HAS_FAULT",
                    "a control run without a fault",
                    f"{run.id} carries a fault artifact",
                    "a faulted run cannot serve as the control",
                    cell,
                )
            if cell != "CONTROL" and fault is None:
                issue(
                    "TREATMENT_WITHOUT_FAULT",
                    "a fault-evaluation run with fault/fault.json",
                    run.id,
                    "the applied fault cannot be identified",
                    cell,
                )
    if issues:
        raise DesignRefusal(tuple(issues))

    def fdesc(cell: str) -> list[list[dict[str, object]]]:
        seen: list[list[dict[str, object]]] = []
        for _, _, _, fault in per_cell[cell]:
            assert fault is not None  # noqa: S101
            spec_d = fault["fault"]
            assert isinstance(spec_d, Mapping)  # noqa: S101
            comps = _components(spec_d)
            if comps not in seen:
                seen.append(comps)
        return seen

    seen = {c: fdesc(c) for c in per_cell if c != "CONTROL"}
    for cell, s in seen.items():
        if len(s) != 1:
            issue(
                "CELL_MIXES_FAULTS",
                "one fault definition per cell (seeds may differ)",
                f"{len(s)} different definitions",
                "trials of different faults cannot be aggregated",
                cell,
            )
    if issues:
        raise DesignRefusal(tuple(issues))
    da, db = seen["A"][0], seen["B"][0]
    if len(da) != 1 or len(db) != 1:
        issue(
            "COMPONENT_NOT_SINGLE",
            "cells A and B each a single (non-compound) fault",
            f"A has {len(da)}, B has {len(db)} components",
            "the components of the design must be the single faults",
        )
    else:
        for cell, want in (("AB", [da[0], db[0]]), ("BA", [db[0], da[0]])):
            if cell in seen and seen[cell][0] != want:
                issue(
                    "COMPOUND_MISMATCH",
                    f"{cell} = compound of {[w['type'] for w in want]} in that order with the parameters and scopes of the single-fault cells",
                    f"{[c['type'] for c in seen[cell][0]]} (parameters/scope/order compared)",
                    "the compound treatment is not the ordered composition of the single faults",
                    cell,
                )
        if da[0] == db[0]:
            issue(
                "SAME_FAULT",
                "two different fault definitions",
                "A and B are identical",
                "an interaction needs two distinct faults",
            )
    if cfg.order_analysis and "BA" not in per_cell:
        issue(
            "BA_MISSING",
            "a BA cell for order analysis",
            "none",
            "order effects cannot be assessed without the reverse order",
        )
    if issues:
        raise DesignRefusal(tuple(issues))
    checks.append(
        "A and B are single faults; AB (and BA) are their ordered compositions with identical parameters and scope"
    )

    # comparability of every run
    rows_all = [(c, r) for c, rows in per_cell.items() for r in rows]
    _, ref_ev, ref_prov, _ = per_cell["CONTROL"][0]
    ref_cfg = content_hash(to_jsonable(ref_ev.config))
    ref_ds = ref_ev.context.dataset_fingerprint
    for cell, (run, ev, prov, fault) in rows_all:
        ds = (
            ev.context.dataset_fingerprint
            if fault is None
            else (fault.get("source_dataset_fingerprint") if fault else None)
        )
        for label, req, got in (
            ("MODEL_MISMATCH", ref_ev.context.model_fingerprint, ev.context.model_fingerprint),
            ("DATASET_MISMATCH", ref_ds, ds),
            ("SPLIT_MISMATCH", ref_ev.context.split, ev.context.split),
            ("EVALUATION_CONFIG_MISMATCH", ref_cfg, content_hash(to_jsonable(ev.config))),
            ("TASK_MISMATCH", ref_ev.task, ev.task),
            ("SAMPLE_COUNT_MISMATCH", ref_ev.n_samples, ev.n_samples),
            ("CLASSES_MISMATCH", ref_ev.classes, ev.classes),
        ):
            if req != got:
                issue(
                    label,
                    f"{req}",
                    f"{got} in {run.id}",
                    "runs compared in one interaction must share model, source dataset, split, evaluation configuration, task and samples",
                    cell,
                )
        if (
            cfg.require_same_environment
            and prov is not None
            and ref_prov is not None
            and prov.environment_id != ref_prov.environment_id
        ):
            issue(
                "ENVIRONMENT_MISMATCH",
                ref_prov.environment_id,
                prov.environment_id,
                "numerical libraries may differ (set require_same_environment=false to accept this explicitly)",
                cell,
            )
        if prov is None:
            issue(
                "PROVENANCE_MISSING",
                "a provenance record",
                run.id,
                "the run's inputs cannot be verified",
                cell,
            )
    if issues:
        raise DesignRefusal(tuple(issues))
    checks.append(
        "all runs share model fingerprint, source dataset fingerprint, split, evaluation configuration, task, sample count and classes"
        + (" and environment" if cfg.require_same_environment else "")
    )

    # measures and their directions
    trials: dict[str, list[Trial]] = {c: [] for c in per_cell}
    direction: dict[str, bool | None] = {}
    for cell, rows in per_cell.items():
        for run, ev, prov, fault in rows:
            vals, bad, dirs = _measures(ev)
            for name, d in dirs.items():
                if name in direction and direction[name] != d:
                    issue(
                        "DIRECTION_MISMATCH",
                        f"one direction for {name}",
                        f"{direction[name]} vs {d}",
                        "a metric's direction must agree across runs",
                        cell,
                    )
                direction.setdefault(name, d)
            assert prov is not None  # noqa: S101
            seed = int(str(fault["seed"])) if fault is not None else run.seed
            trials[cell].append(
                Trial(
                    cell,
                    run.id,
                    str(seed) if fault is not None else run.id,
                    seed,
                    vals,
                    bad,
                    dict(fault) if fault is not None else None,
                    prov.fingerprint,
                    prov.environment_id,
                    ev.n_samples,
                )
            )
    for cell, ts in trials.items():
        keys = [t.key for t in ts]
        if len(keys) != len(set(keys)):
            issue(
                "DUPLICATE_TRIAL_KEY",
                "one run per trial seed in a cell",
                f"repeated {sorted({k for k in keys if keys.count(k) > 1})}",
                "duplicate seeds are repeats, not independent trials, and cannot be paired",
                cell,
            )
    control_ev = per_cell["CONTROL"][0][1]
    primary = cfg.primary_metric or (
        "accuracy" if control_ev.task.value == "CLASSIFICATION" else "rmse"
    )
    wanted = [primary, *cfg.metrics]
    have = set().union(*(t.values.keys() for t in trials["CONTROL"]))
    for m in wanted:
        if m not in have:
            issue(
                "METRIC_MISSING",
                f"metric {m!r} computed in the control",
                "not present or undefined",
                "an interaction contrast needs a defined baseline value",
            )
    if cfg.pairing is Pairing.PAIRED:
        if len(trials["CONTROL"]) != 1:
            issue(
                "PAIRING_CONTROL",
                "a single control run for paired analysis",
                f"{len(trials['CONTROL'])} runs",
                "paired analysis with several controls is not supported",
            )
        keysets = {c: {t.key for t in trials[c]} for c in trials if c != "CONTROL"}
        if len({frozenset(k) for k in keysets.values()}) != 1:
            issue(
                "PAIRING_KEYS",
                "identical trial-seed sets in every treatment cell",
                json.dumps({c: sorted(k) for c, k in keysets.items()}),
                "pairing by seed is only justified when the same seeds were run in every cell (never paired by position)",
            )
    if issues:
        raise DesignRefusal(tuple(issues))
    checks.append(
        f"pairing declared as {cfg.pairing.value}"
        + (
            ": identical trial-seed sets in every treatment cell, paired by seed key"
            if cfg.pairing is Pairing.PAIRED
            else ""
        )
    )
    warns: list[str] = []
    for cell, ts in trials.items():
        if cell != "CONTROL" and len(ts) < cfg.min_trials:
            warns.append(
                f"cell {cell} has {len(ts)} usable trial(s) (< min_trials={cfg.min_trials}): no interval will be produced"
            )
    if excluded:
        warns.append(
            f"{len(excluded)} run(s) were excluded and are listed in the design validation"
        )
    first = next(iter(seen["A"][0]))
    del first
    return LoadedDesign(
        spec,
        {c: tuple(sorted(ts, key=lambda t: t.key)) for c, ts in trials.items()},
        tuple(excluded),
        direction,
        control_ev.task.value,
        str(ref_ev.context.model_fingerprint),
        str(ref_ds),
        ref_ev.context.split,
        ref_cfg,
        seen["A"][0][0],
        seen["B"][0][0],
        cfg.pairing,
        tuple(warns),
        tuple(checks),
    )


def validate_report(
    registry: Registry, store: ArtifactStore, spec: InteractionSpec
) -> dict[str, object]:
    """Non-raising form for the CLI: {'valid': bool, 'issues': [...], 'checks': [...]}."""
    try:
        d = validate(registry, store, spec)
    except DesignRefusal as exc:
        return {
            "valid": False,
            "spec_id": spec.spec_id,
            "issues": [i.to_dict() for i in exc.issues if isinstance(i, DesignIssue)],
        }
    return {
        "valid": True,
        "spec_id": spec.spec_id,
        "checks": list(d.checks),
        "warnings": list(d.warnings),
        "excluded": list(d.excluded),
        "trials": {c: len(t) for c, t in d.trials.items()},
    }


def experiment_of(registry: Registry, run_id: str) -> Experiment:
    return registry.get(Experiment, registry.get(Run, run_id).experiment_id)


def all_run_ids(d: LoadedDesign) -> Sequence[str]:
    return [t.run_id for ts in d.trials.values() for t in ts]

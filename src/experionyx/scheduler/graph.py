"""Deterministic expansion of a `ScheduleSpec` into an explicit, validated DAG of planned units.

Everything a schedule could refuse -- a cycle, a missing reference, a duplicate key -- is found
HERE, before any `ScheduleUnit` is persisted and before anything executes (see docs/scheduler.md).
`ScheduleSpec.__post_init__` already rejects duplicate keys and references to unknown keys; this
module adds the one check that needs real graph analysis: cycles."""

from dataclasses import dataclass

from experionyx.errors import SchedulerRefusal
from experionyx.hashing import content_hash
from experionyx.interactions.design import DesignIssue
from experionyx.scheduler.spec import (
    ENGINE_VERSION,
    ResourceRequirement,
    RetryPolicy,
    ScheduleSpec,
    UnitDef,
)
from experionyx.scheduler.taxonomy import UnitKind


@dataclass(frozen=True)
class PlannedUnit:
    """One expanded unit, ordered as it will be considered for dispatch."""

    key: str
    kind: UnitKind
    parameters: dict[str, object]
    depends_on: tuple[str, ...]
    retry: RetryPolicy
    resources: ResourceRequirement | None
    timeout_seconds: float | None
    priority: int


@dataclass(frozen=True)
class Plan:
    spec_id: str
    graph_hash: str
    units: tuple[PlannedUnit, ...]  # deterministic topological order (see `_topological_order`)
    edges: tuple[tuple[str, str], ...]  # (dependency_key, dependent_key), sorted
    engine_version: str = ENGINE_VERSION

    def unit(self, key: str) -> PlannedUnit:
        for u in self.units:
            if u.key == key:
                return u
        raise KeyError(key)


def _topological_order(spec: ScheduleSpec) -> list[UnitDef]:
    """Kahn's algorithm with a deterministic tie-break (higher priority, then key) among units that
    become available at the same step. Raises SchedulerRefusal naming every unit left in a cycle."""
    by_key = {u.key: u for u in spec.units}
    indegree = {u.key: len(u.depends_on) for u in spec.units}
    dependents: dict[str, list[str]] = {u.key: [] for u in spec.units}
    for u in spec.units:
        for d in u.depends_on:
            dependents[d].append(u.key)
    available = sorted(
        (k for k, n in indegree.items() if n == 0), key=lambda k: (-by_key[k].priority, k)
    )
    ordered: list[UnitDef] = []
    while available:
        available.sort(key=lambda k: (-by_key[k].priority, k))
        key = available.pop(0)
        ordered.append(by_key[key])
        for nxt in dependents[key]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                available.append(nxt)
    if len(ordered) != len(spec.units):
        remaining = sorted(set(by_key) - {u.key for u in ordered})
        raise SchedulerRefusal(
            (
                DesignIssue(
                    "CYCLE",
                    "a directed acyclic graph of unit dependencies",
                    f"cycle involving {remaining}",
                    "unit(s) cannot be ordered: each has an unsatisfied dependency on another "
                    "member of the same cycle; nothing was expanded or executed",
                ),
            )
        )
    return ordered


def expand(spec: ScheduleSpec) -> Plan:
    """Validate and expand `spec`. Raises SchedulerRefusal (before anything exists) if the
    dependency graph contains a cycle; every other structural problem is already rejected by
    `ScheduleSpec.__post_init__` when the spec itself is constructed."""
    ordered = _topological_order(spec)
    units = tuple(
        PlannedUnit(
            u.key,
            u.kind,
            dict(u.parameters),
            u.depends_on,
            u.retry or spec.policy.default_retry,
            u.resources,
            u.timeout_seconds
            if u.timeout_seconds is not None
            else spec.policy.default_timeout_seconds,
            u.priority,
        )
        for u in ordered
    )
    edges = tuple(sorted((d, u.key) for u in spec.units for d in u.depends_on))
    graph_hash = content_hash(
        {
            "engine": ENGINE_VERSION,
            "spec_id": spec.spec_id,
            "order": [u.key for u in units],
            "edges": list(edges),
        }
    )
    return Plan(spec.spec_id, graph_hash, units, edges)


def dependents_of(plan: Plan, key: str) -> tuple[str, ...]:
    return tuple(sorted(b for a, b in plan.edges if a == key))


def validation_report(spec: ScheduleSpec) -> dict[str, object]:
    """Non-raising form for the CLI: validate and expand, or report every refusal issue."""
    try:
        plan = expand(spec)
    except SchedulerRefusal as exc:
        return {
            "valid": False,
            "spec_id": spec.spec_id,
            "issues": [i.to_dict() for i in exc.issues if isinstance(i, DesignIssue)],
        }
    return {
        "valid": True,
        "spec_id": spec.spec_id,
        "graph_hash": plan.graph_hash,
        "engine_version": plan.engine_version,
        "units": len(plan.units),
        "order": [u.key for u in plan.units],
        "edges": [list(e) for e in plan.edges],
    }

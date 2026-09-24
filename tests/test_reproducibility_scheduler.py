"""Scheduler integration: a REPRODUCTION unit dispatches through `run_reproduction`, resolves its
target via the normal `$dep:` mechanism (so it can never run before the evidence it reproduces
exists), and its outcome is visible as the unit's `primary_ref` (a `rpa_...` id) -- see
docs/reproducibility.md."""

from pathlib import Path

import pytest

from experionyx.artifacts import LocalArtifactStore
from experionyx.domain import Investigation
from experionyx.execution import Executor
from experionyx.graph.engine import run_graph
from experionyx.graph.spec import GraphSpec
from experionyx.registry import Registry
from experionyx.reproducibility.entities import ReproductionAttempt
from experionyx.scheduler.engine import run_schedule, utc_now
from experionyx.scheduler.spec import ScheduleSpec, UnitDef
from experionyx.scheduler.taxonomy import ScheduleRunState, UnitKind, UnitState
from experionyx.sqlite import SqliteRegistry


def open_registry(ws: Path) -> SqliteRegistry:
    return SqliteRegistry(ws / "registry.sqlite")


def open_executor(ws: Path, reg: Registry) -> Executor:
    return Executor(reg, LocalArtifactStore(ws / "experiments"), source_root=Path.cwd())


def run(ws: Path, spec: ScheduleSpec) -> object:
    return run_schedule(
        lambda: open_registry(ws),
        LocalArtifactStore(ws / "experiments"),
        lambda reg: open_executor(ws, reg),
        spec,
        source_root=Path.cwd(),
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    reg = SqliteRegistry(ws / "registry.sqlite")
    reg.add(Investigation("repro-scheduler", "can a REPRODUCTION unit be scheduled?", utc_now()))
    reg.close()
    return ws


def _spec(target_run_id: str) -> ScheduleSpec:
    # "base" is an unrelated real unit whose only job is to prove the scheduler's own dependency
    # mechanism blocks "check" until it SUCCEEDS -- exactly the guarantee item 12 asks for.
    base = UnitDef(
        key="base",
        kind=UnitKind.STATISTICAL_ANALYSIS,
        parameters={
            "kind": "BOOTSTRAP",
            "sources": {"kind": "inline", "reference": [1.0, 2.0, 3.0, 4.0]},
        },
    )
    check = UnitDef(
        key="check",
        kind=UnitKind.REPRODUCTION,
        parameters={"target_kind": "RUN", "target_id": target_run_id, "mode": "DETERMINISTIC"},
        depends_on=("base",),
    )
    return ScheduleSpec("repro-chain", "1.0.0", (base, check))


def test_a_reproduction_unit_depends_on_and_reproduces_its_target(workspace: Path) -> None:
    with open_registry(workspace) as reg:
        store = LocalArtifactStore(workspace / "experiments")
        graph_result = run_graph(reg, store, open_executor(workspace, reg), GraphSpec("g", "1.0.0"))
        assert graph_result.run_id is not None
        target_run_id = graph_result.run_id

    result = run(workspace, _spec(target_run_id))
    assert result.status is ScheduleRunState.COMPLETED  # type: ignore[attr-defined]
    assert dict(result.counts) == {"SUCCEEDED": 2}  # type: ignore[attr-defined]

    with open_registry(workspace) as reg:
        from experionyx.scheduler.engine import unit_views

        views = {v.key: v for v in unit_views(reg, result.schedule_id)}  # type: ignore[attr-defined]
        assert views["base"].status is UnitState.SUCCEEDED
        assert views["check"].status is UnitState.SUCCEEDED
        attempt_id = views["check"].primary_ref
        assert attempt_id is not None and attempt_id.startswith("rpa_")
        attempt = reg.get(ReproductionAttempt, attempt_id)
        assert attempt.target_id == target_run_id
        # "base" (which "check" depends on) adds a new StatisticalAnalysis to the registry before
        # "check" runs, so the graph collect run's replay genuinely sees a grown registry and
        # differs -- real evidence of registry drift, not a reproduction bug.
        assert attempt.outcome.value == "DIFFERENT"
        assert "graph/nodes.json" in attempt.differences


def test_a_reproduction_unit_cannot_target_a_schedule(workspace: Path) -> None:
    bad = UnitDef(
        key="check",
        kind=UnitKind.REPRODUCTION,
        parameters={
            "target_kind": "SCHEDULE",
            "target_id": "sch_" + "1" * 32,
            "mode": "DETERMINISTIC",
        },
    )
    result = run(workspace, ScheduleSpec("bad-repro", "1.0.0", (bad,)))
    assert dict(result.counts) == {"FAILED": 1}  # type: ignore[attr-defined]

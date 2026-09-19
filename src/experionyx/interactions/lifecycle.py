"""Lifecycle operations. Nothing is confirmed automatically: REPRODUCIBLE needs an independent
replicate analysis that agrees, CONFIRMED_BY_REVIEW needs a named person and a reason."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from experionyx.artifacts import ArtifactStore
from experionyx.domain import RunStatus
from experionyx.errors import ExperionyxError, InteractionError, ValidationError
from experionyx.execution import Executor
from experionyx.faults.report import read_artifact
from experionyx.interactions.engine import ACTOR, SUPPORTING, transition
from experionyx.interactions.entities import (
    InteractionAnalysis,
    InteractionEffect,
    InteractionEvidence,
)
from experionyx.interactions.taxonomy import EvidenceKind, InteractionStatus
from experionyx.registry import Registry


@dataclass(frozen=True)
class ReproductionCheck:
    original_id: str
    replicate_id: str
    passed: bool
    checks: tuple[Mapping[str, object], ...]


def _primary(reg: Registry, a: InteractionAnalysis) -> Mapping[str, object]:
    (e,) = reg.find(InteractionEffect, analysis_id=a.id, measure=a.primary_metric)
    derived = e.record["derived"]
    assert isinstance(derived, Mapping)  # noqa: S101
    return derived


def _treatment_runs(a: InteractionAnalysis) -> set[str]:
    cells: Any = a.spec["cells"]
    return {str(r) for k, runs in cells.items() if k != "CONTROL" for r in runs}


def check_reproduction(
    reg: Registry,
    original_id: str,
    replicate_id: str,
    *,
    abs_tol: float = 0.0,
    rel_tol: float = 0.5,
    now: datetime | None = None,
) -> ReproductionCheck:
    """Compare an analysis with an INDEPENDENT replicate (same structure, disjoint treatment runs).
    Records the outcome as evidence on the original; if it passes and the original is SUPPORTED,
    moves it to REPRODUCIBLE. Tolerances are arguments and are recorded."""
    now = now or datetime.now(UTC)
    o, r = reg.get(InteractionAnalysis, original_id), reg.get(InteractionAnalysis, replicate_id)
    if o.status is not InteractionStatus.SUPPORTED:
        raise ValidationError(
            f"only a SUPPORTED interaction can be checked for reproduction (this one is {o.status.value})"
        )
    co, cr = _primary(reg, o), _primary(reg, r)
    vo, vr = float(str(co["interaction_contrast"])), float(str(cr["interaction_contrast"]))
    shared = _treatment_runs(o) & _treatment_runs(r)
    checks: list[dict[str, object]] = [
        {"name": "same_structure", "required": o.structural_key, "observed": r.structural_key, "passed": o.structural_key == r.structural_key},
        {"name": "same_primary_metric", "required": o.primary_metric, "observed": r.primary_metric, "passed": o.primary_metric == r.primary_metric},
        {"name": "independent_treatment_runs", "required": "no shared treatment run", "observed": sorted(shared)[:10], "passed": not shared},
        {"name": "replicate_shows_observed_interaction", "required": [c.value for c in SUPPORTING], "observed": r.primary_class.value, "passed": r.primary_class in SUPPORTING},
        {"name": "same_sign", "required": "sign(replicate) == sign(original)", "observed": [vo, vr], "passed": vo != 0 and vr != 0 and (vo > 0) == (vr > 0)},
        {"name": "magnitude_within_tolerance", "required": f"|diff| <= max({abs_tol}, {rel_tol} * |original|)", "observed": abs(vr - vo), "passed": abs(vr - vo) <= max(abs_tol, rel_tol * abs(vo))},
    ]  # fmt: skip
    passed = all(bool(c["passed"]) for c in checks)
    detail = {
        "replicate_id": replicate_id,
        "original_contrast": vo,
        "replicate_contrast": vr,
        "abs_tol": abs_tol,
        "rel_tol": rel_tol,
        "checks": checks,
        "passed": passed,
        "protocol": "independent replicate analysis: same structure, disjoint treatment runs, agreeing sign and magnitude; repeatability under the declared design, not a proof of generality",
    }
    with reg.transaction():
        reg.add(
            InteractionEvidence(
                o.id,
                EvidenceKind.REPRODUCTION,
                replicate_id,
                f"replicate {'agrees' if passed else 'does not agree'}",
                detail,
                now,
            )
        )
        if passed:
            transition(
                reg,
                o,
                InteractionStatus.REPRODUCIBLE,
                ACTOR,
                "an independent replicate reproduced the primary-metric contrast within tolerance",
                now,
                refs=[replicate_id],
            )
    return ReproductionCheck(o.id, replicate_id, passed, tuple(checks))


def confirm_by_review(
    reg: Registry, analysis_id: str, actor: str, reason: str, now: datetime | None = None
) -> InteractionAnalysis:
    a = reg.get(InteractionAnalysis, analysis_id)
    if actor.strip() == ACTOR:
        raise ValidationError(
            "confirmation must be made by a named person, not the automatic actor"
        )
    if a.status is not InteractionStatus.REPRODUCIBLE:
        raise ValidationError(
            f"only a REPRODUCIBLE interaction can be confirmed (this one is {a.status.value})"
        )
    return transition(
        reg,
        a,
        InteractionStatus.CONFIRMED_BY_REVIEW,
        actor,
        reason,
        now or datetime.now(UTC),
        {"automatic": False},
    )


def change_status(
    reg: Registry,
    analysis_id: str,
    target: InteractionStatus,
    actor: str,
    reason: str,
    now: datetime | None = None,
) -> InteractionAnalysis:
    """Reject or deprecate (never to CONFIRMED_BY_REVIEW/REPRODUCIBLE, which have their own gates)."""
    if target in (
        InteractionStatus.CONFIRMED_BY_REVIEW,
        InteractionStatus.REPRODUCIBLE,
        InteractionStatus.SUPPORTED,
    ):
        raise InteractionError(f"{target.value} is reached only through its own checks")
    return transition(
        reg,
        reg.get(InteractionAnalysis, analysis_id),
        target,
        actor,
        reason,
        now or datetime.now(UTC),
    )


REPLAY_TOLERANCE = 1e-12  # the same interpreter and libraries reproduce floats exactly; this only allows for platform rounding


def replay_check(
    reg: Registry, store: ArtifactStore, executor: Executor, analysis_id: str
) -> dict[str, object]:
    """Replay the analysis Run as a NEW Run and compare its results with the original's:
    specification identity, provenance fingerprint, every effect's contrast, aggregates and
    bootstrap interval, and the per-sample results, within REPLAY_TOLERANCE."""
    a = reg.get(InteractionAnalysis, analysis_id)
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
    for name in ("spec", "effects", "bootstrap", "summary", "per_sample"):
        try:
            x = read_artifact(reg, store, a.run_id, f"interaction/{name}.json")
        except ExperionyxError:
            continue
        y = read_artifact(reg, store, replay.run.id, f"interaction/{name}.json")
        _compare(name, x, y, diffs)
    return {
        **out,
        "deterministic": not diffs,
        "differences": diffs[:50],
        "compared": [
            "spec identity",
            "provenance fingerprint",
            "effects",
            "bootstrap",
            "summary",
            "per-sample results",
        ],
    }


def _compare(path: str, x: object, y: object, diffs: list[str]) -> None:
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            if k not in x or k not in y:
                diffs.append(f"{path}.{k}: present in one run only")
            else:
                _compare(f"{path}.{k}", x[k], y[k], diffs)
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            diffs.append(f"{path}: length {len(x)} vs {len(y)}")
        for i, (p, q) in enumerate(zip(x, y, strict=False)):
            _compare(f"{path}[{i}]", p, q, diffs)
    elif isinstance(x, float) and isinstance(y, float):
        if abs(x - y) > REPLAY_TOLERANCE * max(1.0, abs(x)):
            diffs.append(f"{path}: {x!r} vs {y!r}")
    elif x != y:
        diffs.append(f"{path}: {x!r} vs {y!r}")

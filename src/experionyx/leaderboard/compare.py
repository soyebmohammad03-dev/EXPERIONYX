"""Protocol-constrained comparison. `compare_submissions` never produces a winner: it wraps
`BenchmarkRegistry.compare`, which already refuses two results of different protocols
(`BenchmarkRefusal`) and reports only raw per-metric differences, direction and uncertainty.
`correct_family` reuses Phase 10's `stats.store` CORRECTION kind directly over an explicitly named
family of already-registered statistical comparisons -- multiple-comparison correction is never
applied silently; the caller names the method and the exact family (see docs/leaderboard.md,
docs/statistics.md)."""

from typing import Any

from experionyx.artifacts import ArtifactStore
from experionyx.benchmark.registry import BenchmarkRegistry
from experionyx.errors import ValidationError
from experionyx.leaderboard.entities import BenchmarkSubmission
from experionyx.leaderboard.submission import resolve_submission
from experionyx.registry import Registry
from experionyx.stats import store as stats_store
from experionyx.stats.entities import StatisticalAnalysis

CORRECTION_METHODS = ("NONE", "BONFERRONI", "BENJAMINI_HOCHBERG")


def compare_submissions(
    registry: Registry, store: ArtifactStore, submission_a: str, submission_b: str
) -> dict[str, Any]:
    """Raw comparison of two submissions' underlying `BenchmarkResult`s. Refuses (does not
    compute anything) if they were not produced under the identical protocol."""
    a, b = resolve_submission(registry, submission_a), resolve_submission(registry, submission_b)
    breg = BenchmarkRegistry(registry, store)
    out = breg.compare(a.result_id, b.result_id)
    return {"submission_a": a.id, "submission_b": b.id, **out}


def compare_protocol_constrained(registry: Registry, a: str, b: str) -> None:
    """Raise ValidationError unless both submissions belong to the same protocol -- checked
    before any comparison is attempted, so a protocol mismatch is refused, not silently compared."""  # fmt: skip
    sa, sb = registry.get(BenchmarkSubmission, a), registry.get(BenchmarkSubmission, b)
    if sa.protocol_id != sb.protocol_id:
        raise ValidationError(
            f"submissions belong to different protocols ({sa.protocol_id} vs {sb.protocol_id}); "
            "comparisons are protocol-specific"
        )


def correct_family(
    registry: Registry,
    analysis_ids: tuple[str, ...],
    *,
    method: str = "NONE",
    alpha: float = 0.05,
) -> StatisticalAnalysis:
    """Multiple-comparison correction over an EXPLICITLY named family of already-registered COMPARE
    analyses (e.g. one per pairwise submission comparison on a chosen metric, built with
    `experionyx stats compare`). Reuses `stats.core.adjust_pvalues` end to end via the existing
    CORRECTION analysis kind; nothing here computes a p-value itself."""
    if method not in CORRECTION_METHODS:
        raise ValidationError(f"method must be one of {list(CORRECTION_METHODS)}")
    if not analysis_ids:
        raise ValidationError("a correction family needs at least one analysis id")
    for aid in analysis_ids:
        registry.get(StatisticalAnalysis, aid)  # raises NotFoundError for an unknown id
    analysis, _new = stats_store.create(
        registry, None, "CORRECTION", {"kind": "analyses", "ids": list(analysis_ids)},
        {"method": method, "alpha": alpha},
    )  # fmt: skip
    return analysis


__all__ = [
    "CORRECTION_METHODS",
    "compare_protocol_constrained",
    "compare_submissions",
    "correct_family",
]

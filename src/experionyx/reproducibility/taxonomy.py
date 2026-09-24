"""Controlled vocabulary for the reproducibility framework (see docs/reproducibility.md).

The framework never promises bit-for-bit reproduction the platform cannot guarantee: `mode` is
requested by the caller and the achieved `ComparisonOutcome` is reported honestly, never inflated
to match what was asked for.
"""

from enum import StrEnum


class ReproductionMode(StrEnum):
    """What kind of agreement a reproduction attempt is being asked to check. These are NOT
    ordered degrees of the same thing -- each is measured differently, so they are never
    collapsed into one score."""

    EXACT = "EXACT"  # every stored artifact byte-identical
    DETERMINISTIC = "DETERMINISTIC"  # the engine's own replay_check verdict (excludes timing/environment-only noise the source engine already excludes, e.g. resource measurements)  # fmt: skip
    NUMERIC_TOLERANCE = "NUMERIC_TOLERANCE"  # every differing artifact re-compared field-by-field within `tolerance`  # fmt: skip
    STATISTICAL = "STATISTICAL"  # agreement judged by confidence/bootstrap interval overlap, not point equality  # fmt: skip
    PROVENANCE_ONLY = "PROVENANCE_ONLY"  # outputs are not re-executed or compared at all; only identity/configuration/environment are checked  # fmt: skip


class ComparisonOutcome(StrEnum):
    EQUAL = "EQUAL"
    APPROXIMATELY_EQUAL = "APPROXIMATELY_EQUAL"
    DIFFERENT = "DIFFERENT"
    UNAVAILABLE = "UNAVAILABLE"  # nothing to compare against (missing evidence, failed replay)
    INCOMPARABLE = "INCOMPARABLE"  # the two sides are not the same shape/type; no meaningful comparison exists  # fmt: skip


class TargetKind(StrEnum):
    """What kind of persisted record a `ReproductionSpec` targets. Every member maps to an
    existing, already-audited replay path (`graph.build.ENTITY_TYPES`-style reuse, never a new
    replay implementation)."""

    RUN = "RUN"
    BENCHMARK_RESULT = "BENCHMARK_RESULT"
    SCHEDULE = "SCHEDULE"
    STRESS_ANALYSIS = "STRESS_ANALYSIS"
    CALIBRATION_ANALYSIS = "CALIBRATION_ANALYSIS"
    RESOURCE_ANALYSIS = "RESOURCE_ANALYSIS"
    DRIFT_ANALYSIS = "DRIFT_ANALYSIS"
    QUALITY_ANALYSIS = "QUALITY_ANALYSIS"
    RELIABILITY_PROFILE = "RELIABILITY_PROFILE"
    INTERACTION_ANALYSIS = "INTERACTION_ANALYSIS"
    GRAPH_SNAPSHOT = "GRAPH_SNAPSHOT"


PREFIX_BY_KIND: dict[TargetKind, str] = {
    TargetKind.RUN: "run",
    TargetKind.BENCHMARK_RESULT: "brs",
    TargetKind.SCHEDULE: "sch",
    TargetKind.STRESS_ANALYSIS: "sxa",
    TargetKind.CALIBRATION_ANALYSIS: "cba",
    TargetKind.RESOURCE_ANALYSIS: "rsa",
    TargetKind.DRIFT_ANALYSIS: "dan",
    TargetKind.QUALITY_ANALYSIS: "qan",
    TargetKind.RELIABILITY_PROFILE: "rpf",
    TargetKind.INTERACTION_ANALYSIS: "ian",
    TargetKind.GRAPH_SNAPSHOT: "gsn",
}

__all__ = ["PREFIX_BY_KIND", "ComparisonOutcome", "ReproductionMode", "TargetKind"]

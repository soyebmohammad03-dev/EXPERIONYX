"""Controlled vocabulary for the leaderboard layer (see docs/leaderboard.md)."""

from enum import StrEnum


class ReproducibilityState(StrEnum):
    """An evidence state about a submission's `BenchmarkResult`, never a quality rating. Derived
    from the most recent `ReproductionAttempt` (see docs/reproducibility.md) targeting it, if any."""

    REPRODUCED = "REPRODUCED"  # the most recent attempt's outcome was EQUAL
    PARTIALLY_REPRODUCED = "PARTIALLY_REPRODUCED"  # APPROXIMATELY_EQUAL
    NOT_REPRODUCED = "NOT_REPRODUCED"  # DIFFERENT or INCOMPARABLE
    UNAVAILABLE = "UNAVAILABLE"  # no attempt exists, or the attempt itself reported UNAVAILABLE


__all__ = ["ReproducibilityState"]

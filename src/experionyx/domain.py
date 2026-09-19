"""Domain vocabulary that is already fixed by the methodology (docs/methodology.md).

Only enumerations live here. Entities (Model, Experiment, Run, ...) arrive in Phase 1.
"""

from enum import StrEnum


class ClaimStatus(StrEnum):
    """Outcome of evaluating a claim against evidence. Negative results are first-class."""

    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    CONFOUNDED = "CONFOUNDED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class EpistemicKind(StrEnum):
    """What kind of statement a piece of information is; never conflate these."""

    OBSERVATION = "OBSERVATION"
    DERIVED_METRIC = "DERIVED_METRIC"
    INTERPRETATION = "INTERPRETATION"
    HYPOTHESIS = "HYPOTHESIS"

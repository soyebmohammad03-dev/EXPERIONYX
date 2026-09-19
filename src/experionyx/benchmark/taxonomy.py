"""Vocabulary of the benchmark engine."""

from enum import StrEnum


class UnitKind(StrEnum):
    BASELINE = "BASELINE"
    FAULT_TRIAL = "FAULT_TRIAL"  # one (fault grid, parameter point, seed) evaluation
    INTERACTION_TRIAL = "INTERACTION_TRIAL"  # one (pair, cell, seed) evaluation


class UnitStatus(StrEnum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"  # the run failed; its reason is kept
    SKIPPED = "SKIPPED"  # not executed because a limit was reached
    NOT_RUN = "NOT_RUN"  # its experiment could not be created or run
    UNSUPPORTED = (
        "UNSUPPORTED"  # excluded because the dataset/model cannot support it (recorded, not hidden)
    )


class CoverageStatus(StrEnum):
    COMPLETE = "COMPLETE"  # every requested unit completed and every requested analysis ran
    INCOMPLETE = "INCOMPLETE"  # something requested is missing; the coverage document says what

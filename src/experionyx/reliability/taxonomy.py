"""Vocabulary of reliability profiles. A profile has several DIMENSIONS, each with an explicit
status. There is no overall score, ranking, or verdict anywhere in this package."""

from enum import StrEnum


class Dimension(StrEnum):
    BASELINE_PERFORMANCE = "BASELINE_PERFORMANCE"
    FAULT_SENSITIVITY = "FAULT_SENSITIVITY"
    FAILURE_PREVALENCE = "FAILURE_PREVALENCE"
    FAILURE_SEVERITY = "FAILURE_SEVERITY"
    INTERACTION_SENSITIVITY = "INTERACTION_SENSITIVITY"
    SLICE_SENSITIVITY = "SLICE_SENSITIVITY"
    REPRODUCIBILITY = "REPRODUCIBILITY"
    UNCERTAINTY = "UNCERTAINTY"
    LATENCY = "LATENCY"
    CALIBRATION = "CALIBRATION"


class DimensionStatus(StrEnum):
    OBSERVED = "OBSERVED"  # values read directly from stored evaluation results
    DERIVED = (
        "DERIVED"  # computed by an earlier phase (discovery, aggregation, interaction analysis)
    )
    UNAVAILABLE = "UNAVAILABLE"  # the evidence does not exist (reason stated)
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"  # evidence exists but is too thin to summarize


class Scope(StrEnum):
    """What must match across every source. There is no cross-dataset scope: aggregation across
    unrelated datasets is deliberately not offered (extension point: add a scope AND its
    compatibility rules together)."""

    MODEL_DATASET = "MODEL_DATASET"  # same model and dataset fingerprints
    MODEL_DATASET_SPLIT = "MODEL_DATASET_SPLIT"  # ... and the same evaluated split
    MODEL_DATASET_EVALUATION = (
        "MODEL_DATASET_EVALUATION"  # ... and the same evaluation configuration
    )


class RefKind(StrEnum):
    RUN = "RUN"
    ARTIFACT = "ARTIFACT"
    FAULT_EXPERIMENT = "FAULT_EXPERIMENT"
    FAILURE_MODE = "FAILURE_MODE"
    INTERACTION = "INTERACTION"
    SLICE_ANALYSIS = "SLICE_ANALYSIS"

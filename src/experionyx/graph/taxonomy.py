"""Controlled, versioned vocabulary for the evidence/failure knowledge graph (see docs/graph.md).

`NodeKind` mirrors every registry entity's `Entity.PREFIX` (a `GraphNode` never copies a record;
it only references one by kind + ID), plus a small set of *synthetic* kinds for concepts the
registry itself does not address individually (a dataset split, a failure-relationship
descriptor such as a class or slice label). `RelationType` is the closed set of edge labels a
graph may use; it is a superset of the vocabulary requested for this phase plus the two
`EvidenceRelation` members (`REFUTES`, `CONTEXTUALIZES`) needed to represent evidence faithfully
-- an edge is never mislabeled `SUPPORTS` when the source record said otherwise.
"""

from enum import StrEnum


class NodeKind(StrEnum):
    """One member per registry entity `PREFIX`, plus synthetic (non-registry) concepts."""

    INVESTIGATION = "INVESTIGATION"  # inv
    CONFIGURATION = "CONFIGURATION"  # cfg
    ENVIRONMENT = "ENVIRONMENT"  # env
    EXPERIMENT = "EXPERIMENT"  # exp
    RUN = "RUN"  # run
    OBSERVATION = "OBSERVATION"  # obs
    ARTIFACT = "ARTIFACT"  # art
    CLAIM = "CLAIM"  # clm
    EVIDENCE = "EVIDENCE"  # evd
    PROVENANCE = "PROVENANCE"  # prv
    RUN_OUTCOME = "RUN_OUTCOME"  # out
    MODEL = "MODEL"  # mdl
    DATASET = "DATASET"  # dst
    FAULT_EXPERIMENT = "FAULT_EXPERIMENT"  # fxp
    FAULT_TRIAL = "FAULT_TRIAL"  # ftr
    FAULT_ANALYSIS = "FAULT_ANALYSIS"  # fan
    FAILURE_SIGNAL = "FAILURE_SIGNAL"  # fsg
    FAILURE_CLUSTER = "FAILURE_CLUSTER"  # fcl
    FAILURE_MODE = "FAILURE_MODE"  # fmd
    FAILURE_EVIDENCE = "FAILURE_EVIDENCE"  # fev
    FAILURE_RELATIONSHIP = "FAILURE_RELATIONSHIP"  # frl
    INTERACTION_ANALYSIS = "INTERACTION_ANALYSIS"  # ian
    INTERACTION_EFFECT = "INTERACTION_EFFECT"  # ief
    INTERACTION_EVIDENCE = "INTERACTION_EVIDENCE"  # iev
    RELIABILITY_PROFILE = "RELIABILITY_PROFILE"  # rpf
    RELIABILITY_REFERENCE = "RELIABILITY_REFERENCE"  # rrf
    BENCHMARK = "BENCHMARK"  # bmk
    BENCHMARK_RESULT = "BENCHMARK_RESULT"  # brs
    BENCHMARK_UNIT = "BENCHMARK_UNIT"  # bun
    STATISTICAL_ANALYSIS = "STATISTICAL_ANALYSIS"  # sta
    SLICE = "SLICE"  # sls
    SLICE_ANALYSIS = "SLICE_ANALYSIS"  # san
    TEMPORAL_WINDOW = "TEMPORAL_WINDOW"  # twn
    DRIFT_ANALYSIS = "DRIFT_ANALYSIS"  # dan
    QUALITY_ANALYSIS = "QUALITY_ANALYSIS"  # qan
    QUALITY_CHECK = "QUALITY_CHECK"  # qck
    STRESS_ANALYSIS = "STRESS_ANALYSIS"  # sxa
    STRESS_TRIAL = "STRESS_TRIAL"  # sxt
    CALIBRATION_ANALYSIS = "CALIBRATION_ANALYSIS"  # cba
    CALIBRATION_RESULT = "CALIBRATION_RESULT"  # cbr
    RESOURCE_ANALYSIS = "RESOURCE_ANALYSIS"  # rsa
    RESOURCE_TRIAL = "RESOURCE_TRIAL"  # rst
    SCHEDULE = "SCHEDULE"  # sch
    SCHEDULE_RUN = "SCHEDULE_RUN"  # scr
    SCHEDULE_UNIT = "SCHEDULE_UNIT"  # sun
    EXECUTION_ATTEMPT = "EXECUTION_ATTEMPT"  # att
    UNIT_STATE_TRANSITION = "UNIT_STATE_TRANSITION"  # utr
    REPRODUCTION_ATTEMPT = "REPRODUCTION_ATTEMPT"  # rpa
    # synthetic (graph-only; never registry entities, never given a registry PREFIX)
    SPLIT = "SPLIT"  # a (dataset_fingerprint, split name) pair
    DESCRIPTOR = "DESCRIPTOR"  # a non-registry FailureRelationship endpoint (class/slice/fault)
    UNRESOLVED = "UNRESOLVED"  # a referenced ID that does not exist in this registry


# Entity.PREFIX -> NodeKind, for every registry-backed kind (synthetic kinds have no prefix).
PREFIX_TO_KIND: dict[str, NodeKind] = {
    "inv": NodeKind.INVESTIGATION, "cfg": NodeKind.CONFIGURATION, "env": NodeKind.ENVIRONMENT,
    "exp": NodeKind.EXPERIMENT, "run": NodeKind.RUN, "obs": NodeKind.OBSERVATION,
    "art": NodeKind.ARTIFACT, "clm": NodeKind.CLAIM, "evd": NodeKind.EVIDENCE,
    "prv": NodeKind.PROVENANCE, "out": NodeKind.RUN_OUTCOME, "mdl": NodeKind.MODEL,
    "dst": NodeKind.DATASET, "fxp": NodeKind.FAULT_EXPERIMENT, "ftr": NodeKind.FAULT_TRIAL,
    "fan": NodeKind.FAULT_ANALYSIS, "fsg": NodeKind.FAILURE_SIGNAL, "fcl": NodeKind.FAILURE_CLUSTER,
    "fmd": NodeKind.FAILURE_MODE, "fev": NodeKind.FAILURE_EVIDENCE, "frl": NodeKind.FAILURE_RELATIONSHIP,
    "ian": NodeKind.INTERACTION_ANALYSIS, "ief": NodeKind.INTERACTION_EFFECT,
    "iev": NodeKind.INTERACTION_EVIDENCE, "rpf": NodeKind.RELIABILITY_PROFILE,
    "rrf": NodeKind.RELIABILITY_REFERENCE, "bmk": NodeKind.BENCHMARK, "brs": NodeKind.BENCHMARK_RESULT,
    "bun": NodeKind.BENCHMARK_UNIT, "sta": NodeKind.STATISTICAL_ANALYSIS, "sls": NodeKind.SLICE,
    "san": NodeKind.SLICE_ANALYSIS, "twn": NodeKind.TEMPORAL_WINDOW, "dan": NodeKind.DRIFT_ANALYSIS,
    "qan": NodeKind.QUALITY_ANALYSIS, "qck": NodeKind.QUALITY_CHECK, "sxa": NodeKind.STRESS_ANALYSIS,
    "sxt": NodeKind.STRESS_TRIAL, "cba": NodeKind.CALIBRATION_ANALYSIS, "cbr": NodeKind.CALIBRATION_RESULT,
    "rsa": NodeKind.RESOURCE_ANALYSIS, "rst": NodeKind.RESOURCE_TRIAL, "sch": NodeKind.SCHEDULE,
    "scr": NodeKind.SCHEDULE_RUN, "sun": NodeKind.SCHEDULE_UNIT, "att": NodeKind.EXECUTION_ATTEMPT,
    "utr": NodeKind.UNIT_STATE_TRANSITION, "rpa": NodeKind.REPRODUCTION_ATTEMPT,
}  # fmt: skip


class RelationType(StrEnum):
    """The closed edge vocabulary. An edge label is evidence/provenance metadata; it is never an
    automatically-inferred causal claim (see docs/graph.md)."""

    DERIVED_FROM = "DERIVED_FROM"
    USES_MODEL = "USES_MODEL"
    USES_DATASET = "USES_DATASET"
    USES_SPLIT = "USES_SPLIT"
    PRODUCED_RUN = "PRODUCED_RUN"
    PRODUCED_ARTIFACT = "PRODUCED_ARTIFACT"
    SUPPORTS = "SUPPORTS"
    REFUTES = "REFUTES"
    CONTEXTUALIZES = "CONTEXTUALIZES"
    EVIDENCE_FOR = "EVIDENCE_FOR"
    OBSERVED_IN = "OBSERVED_IN"
    MEMBER_OF = "MEMBER_OF"
    DEPENDS_ON = "DEPENDS_ON"
    TRIGGERED_BY = "TRIGGERED_BY"
    REPRODUCED_BY = "REPRODUCED_BY"
    COMPARED_WITH = "COMPARED_WITH"
    DISCOVERED_FROM = "DISCOVERED_FROM"
    ANALYZED_BY = "ANALYZED_BY"
    REFERENCES = "REFERENCES"


class TraversalDirection(StrEnum):
    OUT = "OUT"
    IN = "IN"
    BOTH = "BOTH"

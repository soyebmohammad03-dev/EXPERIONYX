"""Vocabulary of fault interaction analysis. OBSERVED values come from runs, DERIVED values are
computed from them by fixed formulas, INTERPRETED labels apply configured rules to derived values.
None of these labels is a causal claim."""

from collections.abc import Mapping
from enum import StrEnum


class Cell(StrEnum):
    CONTROL = "CONTROL"  # no fault (Y0)
    A = "A"  # fault A only (YA)
    B = "B"  # fault B only (YB)
    AB = "AB"  # A then B (YAB)
    BA = "BA"  # B then A (YBA); only for order-sensitive analysis


class Pairing(StrEnum):
    PAIRED = "PAIRED"  # trials are matched by an explicit key (the trial seed), never by position
    UNPAIRED = "UNPAIRED"  # cells are independent samples of trials
    UNKNOWN = "UNKNOWN"  # not declared; analyzed as UNPAIRED and reported as such


class Normalization(StrEnum):
    NONE = "NONE"
    BASELINE_MAGNITUDE = "BASELINE_MAGNITUDE"  # |Y0|
    EXPECTED_ADDITIVE_MAGNITUDE = "EXPECTED_ADDITIVE_MAGNITUDE"  # |effect_A + effect_B|
    MAX_COMPONENT_MAGNITUDE = "MAX_COMPONENT_MAGNITUDE"  # max(|effect_A|, |effect_B|)


class Aggregation(StrEnum):
    MEAN = "MEAN"
    MEDIAN = "MEDIAN"


class EffectStatus(StrEnum):
    COMPUTED = "COMPUTED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    UNDEFINED = "UNDEFINED"


class Level(StrEnum):
    METRIC = "METRIC"
    CLASS = "CLASS"  # per-class recall
    SLICE = "SLICE"


class InteractionClass(StrEnum):
    """Interpretation of a contrast under the configured rules. NO_EVIDENCE is not evidence of
    absence. None of these says that one fault causes, or mechanistically acts on, another."""

    NO_EVIDENCE = "NO_EVIDENCE"
    POSSIBLE_INTERACTION = "POSSIBLE_INTERACTION"
    OBSERVED_INTERACTION = "OBSERVED_INTERACTION"
    ORDER_DEPENDENT_INTERACTION = "ORDER_DEPENDENT_INTERACTION"
    INCONCLUSIVE = "INCONCLUSIVE"
    UNDEFINED = "UNDEFINED"


class Relation(StrEnum):
    """Direction-aware reading of the raw contrast (the raw contrast is always kept as is)."""

    SUPER_ADDITIVE_DETERIORATION = "SUPER_ADDITIVE_DETERIORATION"  # combined worse than additive
    SUB_ADDITIVE_DETERIORATION = "SUB_ADDITIVE_DETERIORATION"  # combined better than additive
    ADDITIVE = "ADDITIVE"  # contrast is exactly zero
    UNKNOWN_DIRECTION = "UNKNOWN_DIRECTION"  # the metric has no declared direction


class InteractionStatus(StrEnum):
    DISCOVERED = "DISCOVERED"  # a valid interaction contrast was computed
    SUPPORTED = "SUPPORTED"  # the configured statistical requirements are met
    REPRODUCIBLE = "REPRODUCIBLE"  # an independent replicate analysis reproduced it
    CONFIRMED_BY_REVIEW = "CONFIRMED_BY_REVIEW"  # an explicit human decision
    DEPRECATED = "DEPRECATED"
    REJECTED = "REJECTED"


S = InteractionStatus
INTERACTION_TRANSITIONS: Mapping[InteractionStatus, frozenset[InteractionStatus]] = {
    S.DISCOVERED: frozenset({S.SUPPORTED, S.REJECTED, S.DEPRECATED}),
    S.SUPPORTED: frozenset({S.REPRODUCIBLE, S.REJECTED, S.DEPRECATED}),
    S.REPRODUCIBLE: frozenset({S.CONFIRMED_BY_REVIEW, S.REJECTED, S.DEPRECATED}),
    S.CONFIRMED_BY_REVIEW: frozenset({S.DEPRECATED}),
    S.DEPRECATED: frozenset(),
    S.REJECTED: frozenset(),
}


class ModeState(StrEnum):
    """How a Phase 6 failure mode is observed across the design's cells (never 'caused by')."""

    NEWLY_OBSERVED = "NEWLY_OBSERVED"  # in the compound cell, in neither single-fault cell
    OBSERVED_UNDER_COMPOUND_TREATMENT = "OBSERVED_UNDER_COMPOUND_TREATMENT"
    PERSISTING = "PERSISTING"
    INCREASED_PREVALENCE = "INCREASED_PREVALENCE"
    REDUCED_PREVALENCE = "REDUCED_PREVALENCE"  # includes fully absent under the compound cell
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class EvidenceKind(StrEnum):
    DESIGN = "DESIGN"
    OBSERVATION = "OBSERVATION"
    REPRODUCTION = "REPRODUCTION"
    TRANSITION = "TRANSITION"
    CLAIM = "CLAIM"

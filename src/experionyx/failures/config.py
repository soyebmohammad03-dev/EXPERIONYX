"""Typed, strictly validated, content-hashed configuration of failure discovery.

Every threshold lives here (nothing is hardcoded in the pipeline). The configuration hash and the
algorithm versions are recorded in every discovery run, cluster and mode.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Self

from experionyx.domain import to_jsonable
from experionyx.errors import ValidationError
from experionyx.evaluation.config import _thaw
from experionyx.evaluation.serial import from_jsonable
from experionyx.hashing import content_hash

DISCOVERY_SCHEMA_VERSION = 1
EXTRACTOR_VERSION = "1.0.0"
SIMILARITY_VERSION = "1.0.0"
CLUSTERING_VERSION = "1.0.0"


def _unit(name: str, v: float) -> None:
    if not (math.isfinite(v) and 0.0 <= v <= 1.0):
        raise ValidationError(f"{name} must be in [0, 1]")


def _nonneg(name: str, v: float) -> None:
    if not (math.isfinite(v) and v >= 0.0):
        raise ValidationError(f"{name} must be a finite number >= 0")


@dataclass(frozen=True)
class ExtractionConfig:
    """When an analysis result is turned into a failure signal, and how it is bucketed."""

    class_recall_floor: float = 0.8  # PER_CLASS_RECALL_LOW if recall below this
    confusion_pair_rate_min: float = 0.15  # share of a class predicted as another
    high_confidence_error_rate_min: float = 0.05
    ece_min: float = 0.10
    regression_tail_ratio_min: float = 5.0  # max |residual| / MAE
    regression_bias_ratio_min: float = 0.5  # |mean signed error| / MAE
    slice_deterioration_min: float = 0.10  # absolute, in metric units
    fault_effect_min: float = 0.02  # metric deterioration needed for a fault signal
    class_recall_drop_min: float = 0.10
    ece_increase_min: float = 0.03
    severity_edges: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50)  # signature severity bands
    fraction_edges: tuple[float, ...] = (0.25, 0.50, 0.75)  # affected-fraction bands
    max_sample_ids_per_signal: int = 200  # bounded; truncation is recorded
    max_signals: int = 20000  # bounded; exceeding it is an explicit error, never truncation

    def __post_init__(self) -> None:
        for name in (
            "class_recall_floor",
            "confusion_pair_rate_min",
            "high_confidence_error_rate_min",
            "ece_min",
            "class_recall_drop_min",
        ):
            _unit(name, getattr(self, name))
        for name in (
            "regression_tail_ratio_min",
            "regression_bias_ratio_min",
            "slice_deterioration_min",
            "fault_effect_min",
            "ece_increase_min",
        ):
            _nonneg(name, getattr(self, name))
        for name in ("severity_edges", "fraction_edges"):
            edges = getattr(self, name)
            if list(edges) != sorted(set(edges)) or any(
                not math.isfinite(e) or e < 0 for e in edges
            ):
                raise ValidationError(f"{name} must be strictly increasing, finite and >= 0")
        if self.max_sample_ids_per_signal < 0 or self.max_signals < 1:
            raise ValidationError("max_sample_ids_per_signal >= 0 and max_signals >= 1 required")


@dataclass(frozen=True)
class SimilarityWeights:
    """Weights of the interpretable similarity dimensions. Dimensions that do not apply to a
    pair (e.g. neither signal names a class) are excluded from the average, not scored as 0."""

    error_type: float = 2.0
    class_label: float = 3.0
    slice: float = 2.0
    fault_type: float = 2.0
    parameter_proximity: float = 1.0
    affected_fraction: float = 1.0
    severity: float = 1.0
    model: float = 0.5
    dataset: float = 0.5

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _nonneg(name, getattr(self, name))
        if sum(getattr(self, n) for n in self.__dataclass_fields__) <= 0:
            raise ValidationError("at least one similarity weight must be positive")


@dataclass(frozen=True)
class SimilarityConfig:
    weights: SimilarityWeights = field(default_factory=SimilarityWeights)
    threshold: float = 0.8  # pairs at or above this are linked
    parameter_relative_tolerance: float = 0.5  # |a-b| / max(|a|,|b|) at or below => proximate
    fraction_tolerance: float = 0.25
    max_pairwise_comparisons: int = (
        500_000  # per discovery; beyond it blocks fall back to exact signatures
    )

    def __post_init__(self) -> None:
        _unit("threshold", self.threshold)
        _nonneg("parameter_relative_tolerance", self.parameter_relative_tolerance)
        _unit("fraction_tolerance", self.fraction_tolerance)
        if self.max_pairwise_comparisons < 1:
            raise ValidationError("max_pairwise_comparisons must be >= 1")


class ClusteringAlgorithm(StrEnum):
    SIGNATURE_GROUPING = "SIGNATURE_GROUPING"  # rule-based: identical signature => same cluster
    SIMILARITY_COMPONENTS = "SIMILARITY_COMPONENTS"  # connected components of the similarity graph


@dataclass(frozen=True)
class ClusteringConfig:
    algorithm: ClusteringAlgorithm = ClusteringAlgorithm.SIMILARITY_COMPONENTS
    sensitivity_delta: float = 0.1  # re-cluster at threshold +/- delta to test stability
    max_clusters: int = 1000
    min_cluster_size_for_mode: int = 2  # smaller clusters are recorded but not registered as modes

    def __post_init__(self) -> None:
        _unit("sensitivity_delta", self.sensitivity_delta)
        if self.max_clusters < 1 or self.min_cluster_size_for_mode < 1:
            raise ValidationError("max_clusters and min_cluster_size_for_mode must be >= 1")


@dataclass(frozen=True)
class EvidenceCriteria:
    """Requirements a cluster must meet for a lifecycle status (configuration, not code)."""

    min_signals: int = 3
    min_runs: int = 2
    min_experiments: int = 1
    min_seeds: int = 1  # distinct seeds among fault-derived signals (ignored if none are)
    min_mean_effect: float = 0.05  # mean signal magnitude
    direction_consistency_min: float = 1.0  # share of signals with the majority direction
    require_interval_excludes_null: bool = (
        False  # CI of the mean magnitude must exclude null_region
    )
    null_region: float = 0.02  # magnitudes at or below this count as "no effect"
    min_compactness: float = 0.8  # mean intra-cluster similarity
    min_threshold_stability: float = 0.5  # membership overlap when the threshold moves

    def __post_init__(self) -> None:
        for name in ("min_signals", "min_runs", "min_experiments", "min_seeds"):
            if getattr(self, name) < 1:
                raise ValidationError(f"{name} must be >= 1")
        for name in ("direction_consistency_min", "min_compactness", "min_threshold_stability"):
            _unit(name, getattr(self, name))
        _nonneg("min_mean_effect", self.min_mean_effect)
        _nonneg("null_region", self.null_region)


@dataclass(frozen=True)
class EvidenceConfig:
    candidate: EvidenceCriteria = field(default_factory=EvidenceCriteria)
    supported: EvidenceCriteria = field(
        default_factory=lambda: EvidenceCriteria(
            min_signals=6,
            min_runs=4,
            min_experiments=2,
            min_seeds=3,
            min_mean_effect=0.05,
            require_interval_excludes_null=True,
            min_compactness=0.85,
            min_threshold_stability=0.7,
        )
    )
    bootstrap_resamples: int = 1000
    bootstrap_confidence: float = 0.95
    bootstrap_seed: int = 0

    def __post_init__(self) -> None:
        if (
            self.bootstrap_resamples < 1
            or not 0.0 < self.bootstrap_confidence < 1.0
            or self.bootstrap_seed < 0
        ):
            raise ValidationError("invalid bootstrap settings")
        c, s = self.candidate, self.supported
        if (
            s.min_signals < c.min_signals
            or s.min_runs < c.min_runs
            or s.min_experiments < c.min_experiments
        ):
            raise ValidationError(
                "supported requirements must not be weaker than candidate requirements"
            )


@dataclass(frozen=True)
class ReproductionConfig:
    magnitude_tolerance_abs: float = 0.05  # a reproduced magnitude may differ by this much...
    magnitude_tolerance_relative: float = (
        0.25  # ...or by this share of the original (either passes)
    )
    min_pass_fraction: float = 0.8
    max_runs: int = 5  # bounded: replays at most this many supporting runs (recorded)

    def __post_init__(self) -> None:
        _nonneg("magnitude_tolerance_abs", self.magnitude_tolerance_abs)
        _nonneg("magnitude_tolerance_relative", self.magnitude_tolerance_relative)
        _unit("min_pass_fraction", self.min_pass_fraction)
        if self.max_runs < 1:
            raise ValidationError("max_runs must be >= 1")


@dataclass(frozen=True)
class DiscoveryConfig:
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    reproduction: ReproductionConfig = field(default_factory=ReproductionConfig)
    max_source_runs: int = 500
    schema_version: int = DISCOVERY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DISCOVERY_SCHEMA_VERSION:
            raise ValidationError(f"unsupported discovery config schema {self.schema_version}")
        if self.max_source_runs < 1:
            raise ValidationError("max_source_runs must be >= 1")

    def to_dict(self) -> dict[str, object]:
        data = to_jsonable(self)
        assert isinstance(data, dict)  # noqa: S101
        return data

    @property
    def config_hash(self) -> str:
        return content_hash(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        result: Self = from_jsonable(cls, _thaw(dict(data)))
        return result

"""The registry entity of a statistical analysis: an immutable, content-addressed record of what was
computed, with which method and settings, from which evidence, and what came out. Raw observations
that already live in digest-verified run artifacts are referenced (run ID, artifact path, selection),
never copied; inline inputs, which exist nowhere else, are stored so the analysis is reproducible."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Self

import experionyx.validation as v
from experionyx.domain import Entity, open_payload
from experionyx.errors import ValidationError
from experionyx.stats.core import Status

ANALYSIS_KINDS = ("COMPARE", "BOOTSTRAP", "PROPORTION", "CORRECTION")


@dataclass(frozen=True)
class StatisticalAnalysis(Entity):
    KIND: ClassVar[str] = "statistical_analysis"
    PREFIX: ClassVar[str] = "sta"
    analysis_kind: str
    input_hash: str  # digest of the exact inputs the analysis used (values, not references)
    engine_version: str
    analysis_status: str  # a stats.core.Status value
    config: Mapping[str, object]  # method, confidence, resamples, seed, pairing, correction, ...
    sources: Mapping[str, object]  # where the inputs came from (see stats.store)
    result: Mapping[str, object]
    result_hash: str
    created_at: datetime

    def __post_init__(self) -> None:
        if self.analysis_kind not in ANALYSIS_KINDS:
            raise ValidationError(f"analysis_kind must be one of {list(ANALYSIS_KINDS)}")
        v.digest("input_hash", self.input_hash)
        v.text("engine_version", self.engine_version)
        v.member("analysis_status", Status(self.analysis_status), Status)
        for name in ("config", "sources", "result"):
            object.__setattr__(self, name, v.freeze_mapping(name, getattr(self, name)))
        v.digest("result_hash", self.result_hash)
        v.timestamp("created_at", self.created_at)

    def _identity(self) -> Mapping[str, object]:
        return {
            "analysis_kind": self.analysis_kind,
            "input_hash": self.input_hash,
            "config": self.config,
            "sources": self.sources,  # two modes with equal counts are still different evidence
            "engine_version": self.engine_version,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        names = (
            "analysis_kind", "input_hash", "engine_version", "analysis_status", "config",
            "sources", "result", "result_hash", "created_at",
        )  # fmt: skip
        open_payload(d, cls.KIND, names)
        return cls(
            v.get_str(d, "analysis_kind"),
            v.get_str(d, "input_hash"),
            v.get_str(d, "engine_version"),
            v.get_str(d, "analysis_status"),
            v.get_mapping(d, "config"),
            v.get_mapping(d, "sources"),
            v.get_mapping(d, "result"),
            v.get_str(d, "result_hash"),
            v.get_time(d, "created_at"),
        )

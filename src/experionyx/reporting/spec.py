"""The typed, immutable, content-addressed request for one report generation (see
docs/reporting.md). Identity is the content hash of the canonical form, so a meaningful change
(scope, template) changes `spec_id` and therefore the resulting report's identity."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

import experionyx.validation as v
from experionyx.errors import ValidationError
from experionyx.hashing import HASH_PREFIX, content_hash
from experionyx.reporting.taxonomy import ReportType

REPORT_ENGINE_VERSION = "1.0.0"  # the generation methodology; bump when assembly semantics change
SPEC_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ReportSpec:
    report_type: ReportType
    investigation_id: str
    template_id: str
    run_ids: tuple[str, ...] = ()  # narrows evidence to these runs; empty = every run in scope
    statistical_analysis_ids: tuple[
        str, ...
    ] = ()  # explicit: StatisticalAnalysis is not investigation-scoped
    schema_version: int = SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SPEC_SCHEMA_VERSION:
            raise ValidationError(f"unsupported report spec schema {self.schema_version}")
        v.member("report_type", self.report_type, ReportType)
        v.ref("investigation_id", self.investigation_id, "inv")
        v.ref("template_id", self.template_id, "rtp")
        for name, ids, prefix in (
            ("run_ids", self.run_ids, "run"),
            ("statistical_analysis_ids", self.statistical_analysis_ids, "sta"),
        ):
            if not isinstance(ids, tuple) or not all(isinstance(x, str) for x in ids):
                raise ValidationError(f"{name} must be a tuple of strings")
            for x in ids:
                v.ref(name, x, prefix)
        if len(set(self.run_ids)) != len(self.run_ids):
            raise ValidationError("run_ids must not repeat")
        if len(set(self.statistical_analysis_ids)) != len(self.statistical_analysis_ids):
            raise ValidationError("statistical_analysis_ids must not repeat")

    def to_dict(self) -> dict[str, object]:
        return {
            "report_type": self.report_type.value,
            "investigation_id": self.investigation_id,
            "template_id": self.template_id,
            "run_ids": list(self.run_ids),
            "statistical_analysis_ids": list(self.statistical_analysis_ids),
            "schema_version": self.schema_version,
        }

    @property
    def spec_id(self) -> str:
        return "rsp_" + content_hash(self.to_dict())[len(HASH_PREFIX) :][:32]

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        known = {
            "report_type",
            "investigation_id",
            "template_id",
            "run_ids",
            "statistical_analysis_ids",
            "schema_version",
        }
        extra = set(d) - known
        missing = {"report_type", "investigation_id", "template_id"} - set(d)
        if extra or missing:
            raise ValidationError(
                f"malformed report spec (unexpected {sorted(extra)}, missing {sorted(missing)})"
            )
        run_ids = d.get("run_ids", ())
        sa_ids = d.get("statistical_analysis_ids", ())
        if not isinstance(run_ids, list | tuple) or not isinstance(sa_ids, list | tuple):
            raise ValidationError("run_ids and statistical_analysis_ids must be lists")
        return cls(
            ReportType(str(d["report_type"])),
            str(d["investigation_id"]),
            str(d["template_id"]),
            tuple(str(x) for x in run_ids),
            tuple(str(x) for x in sa_ids),
            int(d.get("schema_version", SPEC_SCHEMA_VERSION)),  # type: ignore[call-overload]
        )


__all__ = ["REPORT_ENGINE_VERSION", "ReportSpec"]

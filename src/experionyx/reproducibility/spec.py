"""The typed, immutable, content-addressed request for one reproduction attempt. Identity is the
content hash of the canonical form, so a meaningful change (target, mode, tolerance, engine
version) changes `spec_id` (see docs/reproducibility.md)."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

import experionyx.validation as v
from experionyx.errors import ValidationError
from experionyx.hashing import HASH_PREFIX, content_hash
from experionyx.reproducibility.taxonomy import PREFIX_BY_KIND, ReproductionMode, TargetKind

ENGINE_VERSION = "1.0.0"  # the reproducibility methodology; bump when comparison semantics change
SPEC_SCHEMA_VERSION = 1
DEFAULT_RELATIVE_TOLERANCE = 1e-6
DEFAULT_ABSOLUTE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ReproductionSpec:
    target_kind: TargetKind
    target_id: str
    mode: ReproductionMode
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE  # only used by NUMERIC_TOLERANCE
    absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE  # only used by NUMERIC_TOLERANCE
    schema_version: int = SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SPEC_SCHEMA_VERSION:
            raise ValidationError(f"unsupported reproduction spec schema {self.schema_version}")
        v.member("target_kind", self.target_kind, TargetKind)
        v.member("mode", self.mode, ReproductionMode)
        v.ref("target_id", self.target_id, PREFIX_BY_KIND[self.target_kind])
        for name, val in (("relative_tolerance", self.relative_tolerance), ("absolute_tolerance", self.absolute_tolerance)):  # fmt: skip
            if isinstance(val, bool) or not isinstance(val, int | float) or val < 0:
                raise ValidationError(f"{name} must be a non-negative number")

    def to_dict(self) -> dict[str, object]:
        return {
            "target_kind": self.target_kind.value,
            "target_id": self.target_id,
            "mode": self.mode.value,
            "relative_tolerance": self.relative_tolerance,
            "absolute_tolerance": self.absolute_tolerance,
            "schema_version": self.schema_version,
        }

    @property
    def spec_id(self) -> str:
        return "rsc_" + content_hash(self.to_dict())[len(HASH_PREFIX) :][:32]

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        known = {"target_kind", "target_id", "mode", "relative_tolerance", "absolute_tolerance", "schema_version"}  # fmt: skip
        extra = set(d) - known
        missing = {"target_kind", "target_id", "mode"} - set(d)
        if extra or missing:
            raise ValidationError(f"malformed reproduction spec (unexpected {sorted(extra)}, missing {sorted(missing)})")  # fmt: skip
        return cls(
            TargetKind(str(d["target_kind"])),
            str(d["target_id"]),
            ReproductionMode(str(d["mode"])),
            float(d.get("relative_tolerance", DEFAULT_RELATIVE_TOLERANCE)),  # type: ignore[arg-type]
            float(d.get("absolute_tolerance", DEFAULT_ABSOLUTE_TOLERANCE)),  # type: ignore[arg-type]
            int(d.get("schema_version", SPEC_SCHEMA_VERSION)),  # type: ignore[call-overload]
        )


__all__ = [
    "DEFAULT_ABSOLUTE_TOLERANCE",
    "DEFAULT_RELATIVE_TOLERANCE",
    "ENGINE_VERSION",
    "ReproductionSpec",
]

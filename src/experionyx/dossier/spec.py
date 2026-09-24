"""The typed, immutable, content-addressed request for one dossier construction (see
docs/dossier.md). Identity is the content hash of the canonical form: a meaningful change (scope,
source, explicit additions) changes `spec_id` and therefore the resulting dossier's identity."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

import experionyx.validation as v
from experionyx.dossier.taxonomy import DOSSIER_ENGINE_VERSION, DossierSourceKind
from experionyx.errors import ValidationError
from experionyx.hashing import HASH_PREFIX, content_hash

SPEC_SCHEMA_VERSION = 1

# source_kind -> the registry ID prefix its source_id must carry.
_SOURCE_PREFIX: dict[DossierSourceKind, str] = {
    DossierSourceKind.INVESTIGATION: "inv",
    DossierSourceKind.REPORT: "rpt",
    DossierSourceKind.RUN: "run",
    DossierSourceKind.FAILURE_MODE: "fmd",
    DossierSourceKind.BENCHMARK_RESULT: "brs",
    DossierSourceKind.RELIABILITY_PROFILE: "rpf",
    DossierSourceKind.GRAPH_SNAPSHOT: "gsn",
    DossierSourceKind.EXPLICIT_EVIDENCE_IDS: "inv",  # the investigation still anchors the scope
}


@dataclass(frozen=True)
class DossierSpec:
    investigation_id: str
    research_question: str
    source_kind: DossierSourceKind
    source_id: str
    explicit_evidence: tuple[
        tuple[str, str], ...
    ] = ()  # (source_kind, source_id) pairs, added regardless of source_kind
    schema_version: int = SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SPEC_SCHEMA_VERSION:
            raise ValidationError(f"unsupported dossier spec schema {self.schema_version}")
        v.ref("investigation_id", self.investigation_id, "inv")
        v.text("research_question", self.research_question)
        v.member("source_kind", self.source_kind, DossierSourceKind)
        v.ref("source_id", self.source_id, _SOURCE_PREFIX[self.source_kind])
        if not isinstance(self.explicit_evidence, tuple):
            raise ValidationError("explicit_evidence must be a tuple")
        for i, pair in enumerate(self.explicit_evidence):
            if (
                not isinstance(pair, tuple)
                or len(pair) != 2
                or not all(isinstance(x, str) for x in pair)
            ):
                raise ValidationError(
                    f"explicit_evidence[{i}] must be a (source_kind, source_id) pair"
                )
        if len(set(self.explicit_evidence)) != len(self.explicit_evidence):
            raise ValidationError("explicit_evidence must not repeat")

    def to_dict(self) -> dict[str, object]:
        return {
            "investigation_id": self.investigation_id,
            "research_question": self.research_question,
            "source_kind": self.source_kind.value,
            "source_id": self.source_id,
            "explicit_evidence": [list(p) for p in self.explicit_evidence],
            "schema_version": self.schema_version,
        }

    @property
    def spec_id(self) -> str:
        return "dsp_" + content_hash(self.to_dict())[len(HASH_PREFIX) :][:32]

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        known = {
            "investigation_id",
            "research_question",
            "source_kind",
            "source_id",
            "explicit_evidence",
            "schema_version",
        }
        extra = set(d) - known
        missing = {"investigation_id", "research_question", "source_kind", "source_id"} - set(d)
        if extra or missing:
            raise ValidationError(
                f"malformed dossier spec (unexpected {sorted(extra)}, missing {sorted(missing)})"
            )
        raw_pairs = d.get("explicit_evidence", [])
        if not isinstance(raw_pairs, list):
            raise ValidationError("explicit_evidence must be a list")
        pairs: list[tuple[str, str]] = []
        for p in raw_pairs:
            if not isinstance(p, list | tuple) or len(p) != 2:
                raise ValidationError("each explicit_evidence entry must be a 2-item list")
            pairs.append((str(p[0]), str(p[1])))
        return cls(
            str(d["investigation_id"]),
            str(d["research_question"]),
            DossierSourceKind(str(d["source_kind"])),
            str(d["source_id"]),
            tuple(pairs),
            int(d.get("schema_version", SPEC_SCHEMA_VERSION)),  # type: ignore[call-overload]
        )


__all__ = ["DOSSIER_ENGINE_VERSION", "DossierSpec"]

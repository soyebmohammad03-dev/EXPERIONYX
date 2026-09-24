"""The typed, immutable, content-addressed definition of one graph construction request. Identity
is the content hash of the canonical form, so a meaningful change (scope, included kinds, bounds,
engine version) changes `spec_id`. It says nothing about WHICH registry rows exist -- that is
captured separately, per construction, in `GraphSnapshot.source_fingerprint` (see docs/graph.md)."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

import experionyx.validation as v
from experionyx.errors import ValidationError
from experionyx.graph.taxonomy import NodeKind
from experionyx.hashing import HASH_PREFIX, content_hash

ENGINE_VERSION = "1.0.0"  # the graph construction/traversal methodology; bump when either changes
SPEC_SCHEMA_VERSION = 1
DEFAULT_MAX_NODES = 20_000
DEFAULT_MAX_EDGES = 100_000
DEFAULT_MAX_TRAVERSAL_DEPTH = 12
DEFAULT_MAX_VISITED = 5_000


@dataclass(frozen=True)
class GraphSpec:
    """A graph construction request. `investigation_id=None` scopes to the whole registry;
    otherwise only entities that themselves carry a matching `investigation_id` are included --
    entities with no such attribute (a Run, an Artifact, ...) are included regardless of scope
    (see docs/graph.md, "known limitations"). `included_kinds=()` means every known kind."""

    name: str
    version: str  # the DEFINITION version (major.minor.patch); snapshots keep the exact spec
    investigation_id: str | None = None
    included_kinds: tuple[NodeKind, ...] = ()
    max_nodes: int = DEFAULT_MAX_NODES
    max_edges: int = DEFAULT_MAX_EDGES
    schema_version: int = SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SPEC_SCHEMA_VERSION:
            raise ValidationError(f"unsupported graph spec schema {self.schema_version}")
        v.text("name", self.name)
        if not re.fullmatch(r"\d+\.\d+\.\d+", self.version):
            raise ValidationError(f"graph version must be major.minor.patch, got {self.version!r}")
        if self.investigation_id is not None:
            v.ref("investigation_id", self.investigation_id, "inv")
        kinds = tuple(sorted(set(self.included_kinds), key=lambda k: k.value))
        object.__setattr__(self, "included_kinds", kinds)
        if self.max_nodes < 1 or self.max_edges < 1:
            raise ValidationError("max_nodes and max_edges must be >= 1")

    def includes(self, kind: NodeKind) -> bool:
        return not self.included_kinds or kind in self.included_kinds

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "investigation_id": self.investigation_id,
            "included_kinds": [k.value for k in self.included_kinds],
            "max_nodes": self.max_nodes,
            "max_edges": self.max_edges,
            "schema_version": self.schema_version,
        }

    @property
    def spec_id(self) -> str:
        return "gsp_" + content_hash(self.to_dict())[len(HASH_PREFIX) :][:32]

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        known = {"name", "version", "investigation_id", "included_kinds", "max_nodes", "max_edges", "schema_version"}  # fmt: skip
        extra = set(d) - known
        missing = {"name", "version"} - set(d)
        if extra or missing:
            raise ValidationError(f"malformed graph spec (unexpected {sorted(extra)}, missing {sorted(missing)})")  # fmt: skip
        kinds = d.get("included_kinds", [])
        if not isinstance(kinds, list | tuple):
            raise ValidationError("included_kinds must be a list")
        return cls(
            str(d["name"]),
            str(d["version"]),
            None if d.get("investigation_id") is None else str(d["investigation_id"]),
            tuple(NodeKind(str(k)) for k in kinds),
            int(d.get("max_nodes", DEFAULT_MAX_NODES)),  # type: ignore[call-overload]
            int(d.get("max_edges", DEFAULT_MAX_EDGES)),  # type: ignore[call-overload]
            int(d.get("schema_version", SPEC_SCHEMA_VERSION)),  # type: ignore[call-overload]
        )


@dataclass(frozen=True)
class GraphQuery:
    """Bounds for one traversal call. Every traversal stops at whichever bound it hits first and
    reports `truncated=True` in its result; nothing is silently cut off (see docs/graph.md)."""

    max_depth: int = DEFAULT_MAX_TRAVERSAL_DEPTH
    max_visited: int = DEFAULT_MAX_VISITED
    kinds: tuple[NodeKind, ...] = ()  # restrict traversal to these kinds; () = unrestricted
    relations: tuple[str, ...] = ()  # restrict to these relation values; () = unrestricted

    def __post_init__(self) -> None:
        if self.max_depth < 0 or self.max_visited < 1:
            raise ValidationError("max_depth must be >= 0 and max_visited must be >= 1")

    def to_dict(self) -> dict[str, object]:
        return {
            "max_depth": self.max_depth,
            "max_visited": self.max_visited,
            "kinds": [k.value for k in self.kinds],
            "relations": list(self.relations),
        }


DEFAULT_QUERY = GraphQuery()


__all__ = [
    "DEFAULT_MAX_EDGES",
    "DEFAULT_MAX_NODES",
    "DEFAULT_MAX_TRAVERSAL_DEPTH",
    "DEFAULT_MAX_VISITED",
    "DEFAULT_QUERY",
    "ENGINE_VERSION",
    "GraphQuery",
    "GraphSpec",
]

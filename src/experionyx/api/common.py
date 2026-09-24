"""Shared serialization and pagination helpers for the API routers. No analysis logic lives here."""

from collections.abc import Sequence
from typing import Any

from experionyx.domain import Entity

LIMIT_DEFAULT = 50
LIMIT_MAX = 500


def record(entity: Entity) -> dict[str, object]:
    """An entity as JSON: its id plus its own `to_dict()`. Never a hand-rolled reshaping."""
    return {"id": entity.id, **entity.to_dict()}


def clamp_pagination(limit: int | None, offset: int | None) -> tuple[int, int]:
    lim = LIMIT_DEFAULT if limit is None else max(1, min(limit, LIMIT_MAX))
    off = 0 if offset is None else max(0, offset)
    return lim, off


def page(items: Sequence[Entity], limit: int | None, offset: int | None) -> dict[str, Any]:
    """Bounded page over an already-fetched, deterministically ordered sequence of entities."""
    lim, off = clamp_pagination(limit, offset)
    ordered = sorted(items, key=lambda e: e.id)
    window = ordered[off : off + lim]
    return {
        "items": [record(e) for e in window],
        "total": len(ordered),
        "limit": lim,
        "offset": off,
    }


__all__ = ["LIMIT_DEFAULT", "LIMIT_MAX", "clamp_pagination", "page", "record"]

"""Canonical serialization and content hashing.

Canonical form: UTF-8 JSON, sorted keys, no whitespace, ASCII-escaped, NaN/Infinity rejected.
"""

import hashlib
import json

HASH_PREFIX = "sha256:"


def canonical_json(payload: object) -> str:
    """Deterministic JSON text for a JSON-compatible payload."""
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def content_hash(payload: object) -> str:
    """`sha256:<hex>` of the canonical JSON of `payload`."""
    return HASH_PREFIX + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

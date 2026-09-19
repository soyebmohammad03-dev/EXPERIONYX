"""Field validation and strict payload parsing. Invalid input is rejected, never coerced."""

import math
import re
from collections.abc import Mapping
from datetime import datetime, timedelta
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import TypeVar

from experionyx.errors import ValidationError

E = TypeVar("E", bound=Enum)
T = TypeVar("T")

_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_REF = re.compile(r"([a-z]{3})_[0-9a-f]{32}")


def text(field: str, value: object) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{field} must be a string, got {type(value).__name__}")
    if not value or value != value.strip():
        raise ValidationError(f"{field} must be non-empty without surrounding whitespace")
    return value


def version(field: str, value: object) -> str:
    v = text(field, value)
    if not _VERSION.fullmatch(v):
        raise ValidationError(f"{field} is not a valid version: {v!r}")
    return v


def digest(field: str, value: object) -> str:
    v = text(field, value)
    if not _DIGEST.fullmatch(v):
        raise ValidationError(f"{field} must be 'sha256:<64 lowercase hex>': {v!r}")
    return v


def ref(field: str, value: object, prefix: str) -> str:
    """An entity ID reference such as `exp_<32 hex>`."""
    v = text(field, value)
    m = _REF.fullmatch(v)
    if not m or m.group(1) != prefix:
        raise ValidationError(f"{field} must be a '{prefix}_<32 hex>' ID: {v!r}")
    return v


def timestamp(field: str, value: object) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() != timedelta(0):
        raise ValidationError(f"{field} must be a timezone-aware UTC datetime")
    return value


def non_negative_int(field: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValidationError(f"{field} must be a non-negative integer: {value!r}")
    return value


def member(field: str, value: object, cls: type[T]) -> T:
    """Require an instance of `cls` (used for enums and value objects; no coercion)."""
    if not isinstance(value, cls):
        raise ValidationError(f"{field} must be a {cls.__name__}, got {value!r}")
    return value


def relative_path(field: str, value: object) -> str:
    v = text(field, value)
    parts = PurePosixPath(v).parts
    if v.startswith("/") or "\\" in v or "/".join(parts) != v or {"..", "."} & set(parts):
        raise ValidationError(f"{field} must be a normalized relative POSIX path: {v!r}")
    return v


def freeze(field: str, value: object) -> object:
    """Deep-copy a JSON-compatible value into an immutable form (mappings read-only, tuples)."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError(f"{field} must be finite")
        return value
    if isinstance(value, Mapping):
        out: dict[str, object] = {}
        for k, v in value.items():
            if not isinstance(k, str) or not k:
                raise ValidationError(f"{field} keys must be non-empty strings: {k!r}")
            out[k] = freeze(f"{field}.{k}", v)
        return MappingProxyType(out)
    if isinstance(value, list | tuple):
        return tuple(freeze(f"{field}[{i}]", v) for i, v in enumerate(value))
    raise ValidationError(f"{field} contains a non-JSON value of type {type(value).__name__}")


def freeze_mapping(field: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{field} must be a mapping, got {type(value).__name__}")
    frozen = freeze(field, value)
    assert isinstance(frozen, Mapping)  # noqa: S101  # freeze() of a Mapping is a Mapping
    return frozen


# --- strict readers for deserialization -------------------------------------------------------


def get_raw(d: Mapping[str, object], key: str) -> object:
    if key not in d:
        raise ValidationError(f"missing field {key!r}")
    return d[key]


def get_str(d: Mapping[str, object], key: str) -> str:
    v = get_raw(d, key)
    if not isinstance(v, str):
        raise ValidationError(f"{key} must be a string, got {type(v).__name__}")
    return v


def get_opt_str(d: Mapping[str, object], key: str) -> str | None:
    return None if get_raw(d, key) is None else get_str(d, key)


def get_int(d: Mapping[str, object], key: str) -> int:
    v = get_raw(d, key)
    if not isinstance(v, int) or isinstance(v, bool):
        raise ValidationError(f"{key} must be an integer, got {type(v).__name__}")
    return v


def get_mapping(d: Mapping[str, object], key: str) -> Mapping[str, object]:
    v = get_raw(d, key)
    if not isinstance(v, Mapping):
        raise ValidationError(f"{key} must be an object, got {type(v).__name__}")
    return v


def get_time(d: Mapping[str, object], key: str) -> datetime:
    try:
        return datetime.fromisoformat(get_str(d, key))
    except ValueError as exc:
        raise ValidationError(f"{key} is not an ISO-8601 timestamp") from exc


def get_enum(d: Mapping[str, object], key: str, enum: type[E]) -> E:
    v = get_str(d, key)
    try:
        return enum(v)
    except ValueError as exc:
        raise ValidationError(f"{key}: {v!r} is not a valid {enum.__name__}") from exc

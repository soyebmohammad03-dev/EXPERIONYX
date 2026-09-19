"""Strict, type-hint-driven JSON round-tripping for the evaluation dataclasses.

`to_jsonable` (experionyx.domain) writes; `from_jsonable` reads and REJECTS unknown or missing
fields, wrong types, and invalid enum values, so nothing silently disappears or is coerced.
"""

import collections.abc
import types
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from typing import Any, Union, get_args, get_origin, get_type_hints

from experionyx.errors import ValidationError


def _fail(path: str, message: str) -> ValidationError:
    return ValidationError(f"{path or '<root>'}: {message}")


def from_jsonable(tp: object, data: object, path: str = "") -> Any:
    """Build a value of type `tp` from JSON data. Raises ValidationError on any mismatch."""
    origin = get_origin(tp)
    if tp is object or tp is Any:
        return data
    if origin in (Union, types.UnionType):
        members = get_args(tp)
        if data is None and type(None) in members:
            return None
        errors = []
        for member in members:
            if member is type(None):
                continue
            try:
                return from_jsonable(member, data, path)
            except ValidationError as exc:
                errors.append(str(exc))
        raise _fail(path, f"no union member matched ({'; '.join(errors)})")
    if isinstance(tp, type) and issubclass(tp, Enum):
        try:
            return tp(data)
        except ValueError:
            raise _fail(path, f"{data!r} is not a valid {tp.__name__}") from None
    if tp is bool:
        if isinstance(data, bool):
            return data
        raise _fail(path, f"expected bool, got {data!r}")
    if tp is int:
        if isinstance(data, int) and not isinstance(data, bool):
            return data
        raise _fail(path, f"expected int, got {data!r}")
    if tp is float:
        if isinstance(data, int | float) and not isinstance(data, bool):
            return float(data)
        raise _fail(path, f"expected number, got {data!r}")
    if tp is str:
        if isinstance(data, str):
            return data
        raise _fail(path, f"expected str, got {data!r}")
    if is_dataclass(tp) and isinstance(tp, type):
        if not isinstance(data, dict):
            raise _fail(path, f"expected object for {tp.__name__}, got {type(data).__name__}")
        hints = get_type_hints(tp)
        known = {f.name for f in fields(tp)}
        extra = set(data) - known
        if extra:
            raise _fail(path, f"unknown field(s) {sorted(extra)} for {tp.__name__}")
        kwargs = {}
        for f in fields(tp):
            if f.name in data:
                kwargs[f.name] = from_jsonable(hints[f.name], data[f.name], f"{path}.{f.name}")
            elif f.default is MISSING and f.default_factory is MISSING:
                raise _fail(path, f"missing field {f.name!r} for {tp.__name__}")
        return tp(**kwargs)
    if origin is tuple:
        if not isinstance(data, list | tuple):
            raise _fail(path, f"expected a list, got {type(data).__name__}")
        (item,) = (a for a in get_args(tp) if a is not Ellipsis)
        return tuple(from_jsonable(item, x, f"{path}[{i}]") for i, x in enumerate(data))
    if origin in (dict, collections.abc.Mapping):
        if not isinstance(data, dict):
            raise _fail(path, f"expected an object, got {type(data).__name__}")
        _, value_type = get_args(tp)
        return {k: from_jsonable(value_type, x, f"{path}.{k}") for k, x in data.items()}
    raise _fail(path, f"unsupported type {tp!r}")

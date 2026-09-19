"""A typed, immutable slice definition with a deterministic identity. No expression evaluation:
a condition is a small tree of explicit operators. Equivalent spellings are normalized (children of
AND/OR flattened, de-duplicated and sorted, a one-value IN becomes EQ, NOT NOT x becomes x, integral
floats become ints) so they share one identity. Logical equivalence beyond that (De Morgan, range
containment) is NOT decided; two such definitions remain distinct slices.

Fields are names in a sample table: the built-ins `target`, `predicted`, `correct`, `confidence`
and any dataset feature as `feature:<name or column index>`. Outcome fields (`predicted`,
`correct`, `confidence`) depend on the model run, so a slice using them has no run-independent
membership and cannot be used across runs (see `SliceSpec.static`)."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

from experionyx.errors import ValidationError
from experionyx.hashing import HASH_PREFIX, canonical_json, content_hash

SLICE_SCHEMA = 1
Scalar = bool | int | float | str
OUTCOME_FIELDS = frozenset({"predicted", "correct", "confidence"})
MAX_DEPTH = 16
MAX_ARGS = 64
_LEAF = ("eq", "in", "range", "is_true")
_COMBINE = ("and", "or")


def _scalar(x: object) -> Scalar:
    if isinstance(x, bool | str):
        return x
    if isinstance(x, int | float):
        if not math.isfinite(x):
            raise ValidationError("condition values must be finite")
        return int(x) if isinstance(x, float) and x.is_integer() else x
    raise ValidationError(
        f"condition values must be bool, number or string, got {type(x).__name__}"
    )


def _key(x: object) -> str:
    return canonical_json(x)


@dataclass(frozen=True)
class Condition:
    op: str
    field: str | None = None
    values: tuple[Scalar, ...] = ()
    low: float | None = None
    high: float | None = None
    low_inclusive: bool = True
    high_inclusive: bool = False
    args: tuple["Condition", ...] = ()

    # -- canonical form -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        if self.op in _COMBINE or self.op == "not":
            return {"op": self.op, "args": [a.to_dict() for a in self.args]}
        if self.op == "range":
            return {
                "op": "range", "field": self.field, "low": self.low, "high": self.high,
                "low_inclusive": self.low_inclusive, "high_inclusive": self.high_inclusive,
            }  # fmt: skip
        if self.op == "is_true":
            return {"op": "is_true", "field": self.field}
        return {"op": self.op, "field": self.field, "values": list(self.values)}

    def fields(self) -> frozenset[str]:
        if self.field is not None:
            return frozenset({self.field})
        return frozenset().union(*(a.fields() for a in self.args)) if self.args else frozenset()

    def describe(self) -> str:
        f = self.field
        if self.op == "eq":
            return f"{f} == {self.values[0]!r}"
        if self.op == "in":
            return f"{f} in {{{', '.join(repr(v) for v in self.values)}}}"
        if self.op == "is_true":
            return f"{f} is true"
        if self.op == "range":
            lo = "" if self.low is None else f"{self.low:g} {'<=' if self.low_inclusive else '<'} "
            hi = (
                ""
                if self.high is None
                else f" {'<=' if self.high_inclusive else '<'} {self.high:g}"
            )
            return f"{lo}{f}{hi}"
        if self.op == "not":
            return f"NOT ({self.args[0].describe()})"
        joiner = " AND " if self.op == "and" else " OR "
        return joiner.join(f"({a.describe()})" for a in self.args)

    @classmethod
    def from_dict(cls, d: Mapping[str, object], _depth: int = 0) -> "Condition":
        """Strictly parse and normalize; unknown operators and stray keys are rejected."""
        if _depth > MAX_DEPTH:
            raise ValidationError(f"a slice condition may nest at most {MAX_DEPTH} levels")
        op = d.get("op")
        if not isinstance(op, str) or op not in (*_LEAF, *_COMBINE, "not"):
            raise ValidationError(f"unknown slice operator {op!r}")
        allowed = {
            "eq": {"op", "field", "values"}, "in": {"op", "field", "values"},
            "is_true": {"op", "field"},
            "range": {"op", "field", "low", "high", "low_inclusive", "high_inclusive"},
            "and": {"op", "args"}, "or": {"op", "args"}, "not": {"op", "args"},
        }[op]  # fmt: skip
        if set(d) - allowed:
            raise ValidationError(f"{op}: unexpected keys {sorted(set(d) - allowed)}")
        if op in _COMBINE or op == "not":
            raw = d.get("args")
            if not isinstance(raw, list | tuple) or isinstance(raw, str):
                raise ValidationError(f"{op} needs a list of `args`")
            kids = [cls.from_dict(a, _depth + 1) for a in raw if isinstance(a, Mapping)]
            if len(kids) != len(raw):
                raise ValidationError(f"{op}: every arg must be a condition object")
            return all_of(*kids) if op == "and" else any_of(*kids) if op == "or" else negate(*kids)
        f = d.get("field")
        if not isinstance(f, str) or not f or f != f.strip():
            raise ValidationError(f"{op} needs a non-empty `field` without surrounding whitespace")
        if op == "is_true":
            return is_true(f)
        if op == "range":
            return between(
                f, _bound(d.get("low")), _bound(d.get("high")),
                low_inclusive=_flag(d, "low_inclusive", True),
                high_inclusive=_flag(d, "high_inclusive", False),
            )  # fmt: skip
        vals = d.get("values")
        if not isinstance(vals, list | tuple) or isinstance(vals, str):
            raise ValidationError(f"{op} needs a list of `values`")
        return isin(f, vals) if op == "in" else _eq_from_list(f, vals)


def _flag(d: Mapping[str, object], k: str, default: bool) -> bool:
    x = d.get(k, default)
    if not isinstance(x, bool):
        raise ValidationError(f"{k} must be a boolean")
    return x


def _bound(x: object) -> float | None:
    if x is None:
        return None
    if isinstance(x, bool) or not isinstance(x, int | float) or not math.isfinite(x):
        raise ValidationError("range bounds must be finite numbers")
    return float(x)


def _eq_from_list(f: str, vals: Sequence[object]) -> Condition:
    if len(vals) != 1:
        raise ValidationError("eq needs exactly one value")
    return eq(f, vals[0])  # type: ignore[arg-type]


# -- constructors (all normalize) ------------------------------------------------------------------


def eq(field: str, value: Scalar) -> Condition:
    return Condition("eq", _field(field), (_scalar(value),))


def isin(field: str, values: Sequence[Scalar]) -> Condition:
    vs = sorted({_key(_scalar(v)): _scalar(v) for v in values}.items())
    if not vs:
        raise ValidationError("in needs at least one value")
    if len(vs) > MAX_ARGS * 16:
        raise ValidationError("too many values")
    if len(vs) == 1:
        return Condition("eq", _field(field), (vs[0][1],))
    return Condition("in", _field(field), tuple(v for _, v in vs))


def between(
    field: str,
    low: float | None,
    high: float | None,
    *,
    low_inclusive: bool = True,
    high_inclusive: bool = False,
) -> Condition:
    low, high = _bound(low), _bound(high)
    if low is None and high is None:
        raise ValidationError("a range needs a low and/or a high bound")
    if (
        low is not None
        and high is not None
        and (low > high or (low == high and not (low_inclusive and high_inclusive)))
    ):
        raise ValidationError(
            "a range needs low < high (or low == high with both bounds inclusive)"
        )
    return Condition(
        "range", _field(field), low=low, high=high,
        low_inclusive=low_inclusive if low is not None else True,
        high_inclusive=high_inclusive if high is not None else False,
    )  # fmt: skip


def is_true(field: str) -> Condition:
    return Condition("is_true", _field(field))


def label(value: Scalar) -> Condition:
    """A class-label slice: the true target equals `value`."""
    return eq("target", value)


def all_of(*items: Condition) -> Condition:
    return _combine("and", items)


def any_of(*items: Condition) -> Condition:
    return _combine("or", items)


def negate(*items: Condition) -> Condition:
    if len(items) != 1:
        raise ValidationError("not takes exactly one condition")
    (c,) = items
    return c.args[0] if c.op == "not" else Condition("not", args=(c,))


def _combine(op: str, items: Sequence[Condition]) -> Condition:
    flat: dict[str, Condition] = {}
    for c in items:
        for k in c.args if c.op == op else (c,):
            flat[_key(k.to_dict())] = k
    if not flat:
        raise ValidationError(f"{op} needs at least one condition")
    if len(flat) > MAX_ARGS:
        raise ValidationError(f"{op} takes at most {MAX_ARGS} conditions")
    if len(flat) == 1:
        return next(iter(flat.values()))
    return Condition(op, args=tuple(flat[k] for k in sorted(flat)))


def _field(f: str) -> str:
    if not isinstance(f, str) or not f or f != f.strip():
        raise ValidationError("a field must be a non-empty string without surrounding whitespace")
    return f


# -- the slice ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SliceSpec:
    """A named slice. Identity is the normalized condition (and schema version): the name and the
    description are labels and do not change it."""

    name: str
    condition: Condition
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise ValidationError("slice name must be non-empty without surrounding whitespace")

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(sorted(self.condition.fields()))

    @property
    def static(self) -> bool:
        """True if membership does not depend on any model output, so it is the same in every run
        of the same dataset (unless a fault changes the fields it reads; see fault analysis)."""
        return not (self.condition.fields() & OUTCOME_FIELDS)

    @property
    def human(self) -> str:
        return self.description or self.condition.describe()

    @property
    def slice_id(self) -> str:
        # the same derivation as Entity.id for the registered Slice, so the two always agree
        h = content_hash({"kind": "slice", "identity": self.identity()})
        return "sls_" + h[len(HASH_PREFIX) :][:32]

    def identity(self) -> dict[str, object]:
        return {"slice_schema": SLICE_SCHEMA, "condition": self.condition.to_dict()}

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "condition": self.condition.to_dict(), "description": self.description, "slice_schema": SLICE_SCHEMA}  # fmt: skip

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        extra = set(d) - {"name", "condition", "description", "slice_schema"}
        cond = d.get("condition")
        if extra or not isinstance(cond, Mapping) or not isinstance(d.get("name"), str):
            raise ValidationError(
                f"malformed slice (unexpected {sorted(extra)}; name and condition are required)"
            )
        if d.get("slice_schema", SLICE_SCHEMA) != SLICE_SCHEMA:
            raise ValidationError(f"unsupported slice_schema {d.get('slice_schema')!r}")
        desc = d.get("description")
        if desc is not None and not isinstance(desc, str):
            raise ValidationError("description must be a string")
        return cls(str(d["name"]), Condition.from_dict(cond), desc)

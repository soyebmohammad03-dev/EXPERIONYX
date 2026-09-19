"""Slice membership over a sample table, with three-valued logic.

A condition on a sample is TRUE, FALSE or UNKNOWN. UNKNOWN means the sample has no usable value
for a field the condition reads (the field is absent, None, NaN or infinite, or of a different
kind than the condition compares against). UNKNOWN is never a match: those samples are counted and
reported, not silently included, dropped or treated as False. Kleene logic: AND is false if any
argument is false, OR is true if any argument is true, NOT unknown is unknown.

Membership is a function of the table's CONTENT: it is keyed by sample ID, iterated in sorted ID
order, and independent of the order rows were supplied in."""

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from experionyx.errors import ValidationError
from experionyx.hashing import content_hash
from experionyx.slices.spec import Condition, SliceSpec

SampleId = int | str
SampleTable = Mapping[SampleId, Mapping[str, object]]
MAX_LISTED = 20


class MembershipStatus(StrEnum):
    COMPUTED = "COMPUTED"  # at least one member
    EMPTY = "EMPTY"  # a valid slice with no members (a result, not an error)
    MISSING_FIELD = "MISSING_FIELD"  # a field the slice reads exists in no sample: undefined
    NO_SAMPLES = "NO_SAMPLES"  # the table itself is empty


class DuplicateSampleError(ValidationError):
    """A sample ID occurs more than once; membership would be ambiguous."""


def build_table(
    rows: Iterable[tuple[SampleId, Mapping[str, object]]],
) -> dict[SampleId, Mapping[str, object]]:
    table: dict[SampleId, Mapping[str, object]] = {}
    dupes: list[SampleId] = []
    for sid, fields in rows:
        if isinstance(sid, bool) or not isinstance(sid, int | str):
            raise ValidationError(f"sample IDs must be int or str, got {sid!r}")
        if sid in table:
            dupes.append(sid)
        table[sid] = fields
    if dupes:
        raise DuplicateSampleError(
            f"{len(set(dupes))} duplicate sample ID(s), e.g. {sorted(set(dupes), key=_ord)[:5]}"
        )
    return table


def _ord(sid: SampleId) -> tuple[int, Any]:
    return (isinstance(sid, str), sid)


def _kind(x: object) -> str | None:
    if isinstance(x, bool):
        return "bool"
    if isinstance(x, int | float):
        return "num" if math.isfinite(x) else None
    if isinstance(x, str):
        return "str"
    return None  # None, NaN handled above, lists, dicts: unusable


def _eq(c: Condition, row: Mapping[str, object]) -> bool | None:
    assert c.field is not None  # noqa: S101
    if c.field not in row:
        return None
    v = row[c.field]
    k = _kind(v)
    if k is None:
        return None
    results = []
    for target in c.values:
        tk = _kind(target)
        results.append(None if tk != k else v == target)
    if any(r is True for r in results):
        return True
    return None if any(r is None for r in results) else False


def _range(c: Condition, row: Mapping[str, object]) -> bool | None:
    assert c.field is not None  # noqa: S101
    v = row.get(c.field)
    if _kind(v) != "num":
        return None
    x = float(v)  # type: ignore[arg-type]
    if c.low is not None and not (x >= c.low if c.low_inclusive else x > c.low):
        return False
    return not (c.high is not None and not (x <= c.high if c.high_inclusive else x < c.high))


def matches(c: Condition, row: Mapping[str, object]) -> bool | None:
    if c.op in ("eq", "in"):
        return _eq(c, row)
    if c.op == "range":
        return _range(c, row)
    if c.op == "is_true":
        assert c.field is not None  # noqa: S101
        v = row.get(c.field)
        return v if isinstance(v, bool) else None
    if c.op == "not":
        r = matches(c.args[0], row)
        return None if r is None else not r
    rs = [matches(a, row) for a in c.args]
    if c.op == "and":
        return (
            False if any(r is False for r in rs) else None if any(r is None for r in rs) else True
        )
    return True if any(r is True for r in rs) else None if any(r is None for r in rs) else False


@dataclass(frozen=True)
class Membership:
    slice_id: str
    name: str
    description: str
    fields: tuple[str, ...]
    status: MembershipStatus
    sample_ids: tuple[SampleId, ...]
    n_members: int
    n_total: int
    n_unknown: int  # samples whose condition could not be decided (never counted as members)
    unknown_sample_ids: tuple[SampleId, ...]  # first MAX_LISTED, sorted
    prevalence: float | None  # n_members / n_total; None if the table is empty
    missing_fields: tuple[str, ...]  # fields present in no sample
    unseen_values: Mapping[str, tuple[object, ...]]  # eq/in values never observed in the field
    membership_digest: str  # over the sorted member IDs
    warnings: tuple[str, ...] = ()
    reason: str | None = None


def _leaves(c: Condition) -> list[Condition]:
    return [c] if not c.args else [x for a in c.args for x in _leaves(a)]


def evaluate(spec: SliceSpec, table: SampleTable) -> Membership:
    ids = sorted(table, key=_ord)
    n = len(ids)
    fields = spec.fields
    present = {f for f in fields if any(f in table[i] for i in ids)}
    missing = tuple(f for f in fields if f not in present)
    base: dict[str, Any] = {
        "slice_id": spec.slice_id, "name": spec.name, "description": spec.human, "fields": fields,
        "n_total": n, "missing_fields": missing,
    }  # fmt: skip

    def result(
        status: MembershipStatus, members: list[SampleId], unknown: list[SampleId], **kw: Any
    ) -> Membership:
        return Membership(
            status=status, sample_ids=tuple(members), n_members=len(members), n_unknown=len(unknown),
            unknown_sample_ids=tuple(unknown[:MAX_LISTED]),
            prevalence=len(members) / n if n else None,
            membership_digest=content_hash({"ids": [str(i) if isinstance(i, str) else i for i in members]}),
            unseen_values=kw.pop("unseen", {}), **base, **kw,
        )  # fmt: skip

    if n == 0:
        return result(MembershipStatus.NO_SAMPLES, [], [], reason="the sample table is empty")
    if missing:
        return result(
            MembershipStatus.MISSING_FIELD, [], ids,
            reason=f"field(s) {list(missing)} exist in no sample; membership is undefined, not empty",
        )  # fmt: skip
    members: list[SampleId] = []
    unknown: list[SampleId] = []
    for sid in ids:
        r = matches(spec.condition, table[sid])
        (members if r else unknown if r is None else []).append(sid)
    unseen: dict[str, tuple[object, ...]] = {}
    for leaf in _leaves(spec.condition):
        if leaf.op in ("eq", "in") and leaf.field:
            seen = {
                table[i].get(leaf.field) for i in ids if _kind(table[i].get(leaf.field)) is not None
            }
            gone = tuple(v for v in leaf.values if v not in seen)
            if gone:
                unseen[leaf.field] = tuple(dict.fromkeys((*unseen.get(leaf.field, ()), *gone)))
    warns = []
    if unknown:
        warns.append(f"{len(unknown)} of {n} samples have no usable value for a field this slice reads; they are excluded from membership and counted as unknown")  # fmt: skip
    if unseen:
        warns.append(
            f"value(s) never observed in the data: {dict(unseen)}; check the category names"
        )
    status = MembershipStatus.COMPUTED if members else MembershipStatus.EMPTY
    return result(status, members, unknown, unseen=unseen, warnings=tuple(warns))

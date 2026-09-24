"""Generic, tolerance-aware comparison of two JSON-shaped documents (predictions, probabilities,
metrics, per-sample outputs, artifact bodies, ...). Never engine-specific: it walks whatever
structure it is given, the same way `graph.build` walks whatever dataclass fields it is given,
so it works uniformly across every `TargetKind`'s stored artifacts (see docs/reproducibility.md).
"""

import math
from dataclasses import dataclass

from experionyx.reproducibility.taxonomy import ComparisonOutcome


@dataclass(frozen=True)
class FieldDifference:
    path: str
    original: object
    reproduced: object
    outcome: ComparisonOutcome

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "original": self.original,
            "reproduced": self.reproduced,
            "outcome": self.outcome.value,
        }


def _numbers_close(a: float, b: float, rel_tol: float, abs_tol: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return False
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def _walk(
    path: str, a: object, b: object, rel_tol: float, abs_tol: float, out: list[FieldDifference]
) -> None:
    if isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            out.append(FieldDifference(path, a, b, ComparisonOutcome.DIFFERENT))
        return
    if isinstance(a, int | float) and isinstance(b, int | float):
        if a == b:
            return
        if _numbers_close(float(a), float(b), rel_tol, abs_tol):
            out.append(FieldDifference(path, a, b, ComparisonOutcome.APPROXIMATELY_EQUAL))
        else:
            out.append(FieldDifference(path, a, b, ComparisonOutcome.DIFFERENT))
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out.append(FieldDifference(f"{path}.{key}", a.get(key), b.get(key), ComparisonOutcome.DIFFERENT))  # fmt: skip
                continue
            _walk(f"{path}.{key}", a[key], b[key], rel_tol, abs_tol, out)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(FieldDifference(path, f"length {len(a)}", f"length {len(b)}", ComparisonOutcome.DIFFERENT))  # fmt: skip
            return
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _walk(f"{path}[{i}]", x, y, rel_tol, abs_tol, out)
        return
    if type(a) is not type(b):
        out.append(FieldDifference(path, a, b, ComparisonOutcome.INCOMPARABLE))
        return
    if a != b:
        out.append(FieldDifference(path, a, b, ComparisonOutcome.DIFFERENT))


def compare_documents(
    original: object, reproduced: object, *, relative_tolerance: float, absolute_tolerance: float
) -> tuple[ComparisonOutcome, tuple[FieldDifference, ...]]:
    """Recursively compare two JSON-shaped documents. Numbers within tolerance are
    `APPROXIMATELY_EQUAL`; anything else that differs is `DIFFERENT`; a type mismatch at the same
    path is `INCOMPARABLE`. The overall outcome is the worst individual outcome found (INCOMPARABLE
    > DIFFERENT > APPROXIMATELY_EQUAL > EQUAL)."""
    diffs: list[FieldDifference] = []
    _walk("$", original, reproduced, relative_tolerance, absolute_tolerance, diffs)
    if not diffs:
        return ComparisonOutcome.EQUAL, ()
    rank = {ComparisonOutcome.APPROXIMATELY_EQUAL: 1, ComparisonOutcome.DIFFERENT: 2, ComparisonOutcome.INCOMPARABLE: 3}  # fmt: skip
    worst = max(diffs, key=lambda d: rank.get(d.outcome, 0)).outcome
    return worst, tuple(diffs)


def diff_mappings(a: dict[str, object], b: dict[str, object]) -> dict[str, object]:
    """A shallow, human-readable diff of two flat mappings (environment fields, package
    versions, ...): what changed, what was added, what was removed. Used for environment/identity
    comparisons, never for tolerance-based output comparison."""
    changed = {k: {"original": a[k], "reproduced": b[k]} for k in a.keys() & b.keys() if a[k] != b[k]}  # fmt: skip
    return {
        "changed": changed,
        "only_in_original": sorted(a.keys() - b.keys()),
        "only_in_reproduced": sorted(b.keys() - a.keys()),
    }


__all__ = ["FieldDifference", "compare_documents", "diff_mappings"]

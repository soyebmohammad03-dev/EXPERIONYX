"""Applying a FaultSpec: scope selection, ordered component application, and a dataset wrapper.

Determinism: all randomness is derived from (seed, sample position, component index), so the same
spec on the same data yields the same perturbation regardless of batch size or evaluation order.
It is deterministic given numpy's documented Generator/PCG64 behaviour; bitwise identity across
numpy versions, CPUs or platforms is NOT claimed.

Mutation: nothing here mutates caller-owned arrays; the source dataset is never modified. The
faulted dataset is a *derived* object with its own identity, computed from
(source fingerprint, fault id, split).
"""

import hashlib
import json
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import ClassVar, Self

import numpy as np

from experionyx.adapters.base import AdapterInfo, Batch, DatasetAdapter, Sample
from experionyx.adapters.capabilities import DatasetCapability
from experionyx.adapters.metadata import DatasetMetadata
from experionyx.domain import to_jsonable
from experionyx.errors import FaultCompatibilityError, FaultError
from experionyx.faults.spec import (
    FaultContext,
    FaultRegistry,
    FaultSpec,
    FaultTarget,
    Requirement,
    ScopeKind,
)


@dataclass(frozen=True)
class ComponentReport:
    fault_type: str
    version: str
    fault_id: str
    target: str
    scope: Mapping[str, object]
    seed: int
    affected_samples: int  # samples selected by the scope and passed to the fault
    changed_labels: int | None  # label faults: labels that actually differ afterwards


def _canon(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def faulted_identity(source_fingerprint: str, spec: FaultSpec, split: str | None) -> str:
    """Identity of the derived dataset: source identity + fault id (type, version, parameters,
    scope, seeds, component order) + split. Not affected by timestamps or batch size."""
    body = {"source": source_fingerprint, "fault": spec.id, "split": split}
    return "sha256:" + hashlib.sha256(_canon(body)).hexdigest()


class FaultApplier:
    """Applies the leaves of a spec, in order, to batches of (inputs, labels)."""

    def __init__(
        self,
        spec: FaultSpec,
        registry: FaultRegistry,
        *,
        n_total: int,
        classes: tuple[int | str, ...] | None,
    ) -> None:
        self.spec, self.registry, self.n_total, self.classes = spec, registry, n_total, classes
        self._leaves = spec.flatten()
        self._types = [registry.resolve(leaf.type, leaf.version) for leaf in self._leaves]
        self._selected: dict[int, np.ndarray] = {}
        for c, leaf in enumerate(self._leaves):
            if leaf.scope.kind is ScopeKind.RANDOM_SUBSET:
                assert leaf.scope.fraction is not None  # noqa: S101  # validated by FaultScope
                k = round(leaf.scope.fraction * n_total)
                rng = np.random.default_rng(np.random.SeedSequence([leaf.seed, c, 0x5C0E]))
                chosen = np.zeros(n_total, dtype=bool)
                chosen[rng.permutation(n_total)[:k]] = True
                self._selected[c] = chosen
        self._affected = [0] * len(self._leaves)
        self._changed: list[int | None] = [None] * len(self._leaves)

    def _mask(self, c: int, positions: np.ndarray, y: np.ndarray | None) -> np.ndarray:
        scope = self._leaves[c].scope
        if scope.kind is ScopeKind.ALL:
            return np.ones(len(positions), dtype=bool)
        if scope.kind is ScopeKind.RANDOM_SUBSET:
            return self._selected[c][positions]
        if y is None:
            raise FaultCompatibilityError("class-targeted scope needs the true labels")
        return np.asarray(y == scope.label)

    @staticmethod
    def _check(name: str, requires: frozenset[Requirement], x: np.ndarray) -> None:
        if Requirement.FLOAT_ARRAY in requires and x.dtype.kind != "f":
            raise FaultCompatibilityError(
                f"{name} needs floating-point inputs, got dtype {x.dtype}"
            )
        if Requirement.TABULAR in requires and x.ndim != 2:
            raise FaultCompatibilityError(
                f"{name} needs 2-D (samples, features) inputs, got {x.ndim}-D"
            )
        if Requirement.IMAGE in requires and x.ndim < 3:
            raise FaultCompatibilityError(
                f"{name} needs image-shaped inputs (>= 3-D), got {x.ndim}-D"
            )

    def apply(
        self, x: np.ndarray | None, y: np.ndarray | None, positions: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        n = len(positions)
        if (x is not None and len(x) != n) or (y is not None and len(y) != n):
            rows_x = None if x is None else len(x)
            rows_y = None if y is None else len(y)
            raise FaultError(f"shape mismatch: {n} positions, {rows_x} input rows, {rows_y} labels")
        for c, (leaf, ft) in enumerate(zip(self._leaves, self._types, strict=True)):
            if ft.apply is None:
                raise FaultCompatibilityError(f"fault {ft.name} is not implemented")
            ctx = FaultContext(leaf.seed, c, positions, self.classes)
            if ft.target is FaultTarget.INPUT:
                if x is None:
                    raise FaultError("no inputs to perturb")
                self._check(ft.name, ft.requires, x)
                mask = self._mask(c, positions, y)
                x = ft.apply(x, mask, leaf.parameters, ctx) if n else x
                self._affected[c] += int(mask.sum())
            else:
                if y is None:
                    raise FaultCompatibilityError(f"{ft.name} needs labels")
                mask = self._mask(c, positions, y)
                new_y = ft.apply(y, mask, leaf.parameters, ctx) if n else y
                self._affected[c] += int(mask.sum())
                self._changed[c] = (self._changed[c] or 0) + int((new_y != y).sum())
                y = new_y
        return x, y

    def reports(self) -> tuple[ComponentReport, ...]:
        return tuple(
            ComponentReport(
                leaf.type,
                leaf.version,
                leaf.id,
                ft.target.value,
                to_jsonable(leaf.scope),  # type: ignore[arg-type]
                leaf.seed,
                self._affected[c],
                self._changed[c],
            )
            for c, (leaf, ft) in enumerate(zip(self._leaves, self._types, strict=True))
        )


def apply_to_arrays(
    spec: FaultSpec,
    registry: FaultRegistry,
    x: np.ndarray | None,
    y: np.ndarray | None = None,
    *,
    positions: np.ndarray | None = None,
    n_total: int | None = None,
    classes: tuple[int | str, ...] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, tuple[ComponentReport, ...]]:
    """Convenience: apply a spec to whole arrays (used by tests and quick experiments)."""
    n = len(x) if x is not None else len(y)  # type: ignore[arg-type]
    pos = np.arange(n) if positions is None else np.asarray(positions)
    applier = FaultApplier(spec, registry, n_total=n_total or n, classes=classes)
    x2, y2 = applier.apply(x, y, pos)
    return x2, y2, applier.reports()


def _to_numpy(obj: object) -> tuple[np.ndarray, str]:
    if hasattr(obj, "detach"):
        return obj.detach().cpu().numpy(), "torch"  # type: ignore[attr-defined,no-any-return]
    return np.asarray(obj), "numpy"


def _back(arr: np.ndarray, kind: str) -> object:
    if kind == "torch":
        import torch

        return torch.from_numpy(np.ascontiguousarray(arr))
    return arr


@dataclass
class FaultedDataset:
    """A DatasetAdapter view of `source` with a fault applied on the fly.

    The registered dataset is untouched. Perturbed batches are produced lazily (bounded memory);
    `identity` is the derived dataset's fingerprint and `content_digest` a SHA-256 over the
    perturbed bytes actually streamed (platform dependent; available after iteration).
    """

    NAME: ClassVar[str] = "faulted"
    VERSION: ClassVar[str] = "1.0.0"
    FRAMEWORK: ClassVar[str] = "experionyx-faults"

    source: DatasetAdapter
    spec: FaultSpec
    registry: FaultRegistry
    split: str | None = None
    apply_seconds: float = field(default=0.0, init=False)
    _digest: "hashlib._Hash" = field(default_factory=hashlib.sha256, init=False, repr=False)
    _applier: FaultApplier | None = field(default=None, init=False, repr=False)
    _iterated: bool = field(default=False, init=False)

    @classmethod
    def info(cls) -> AdapterInfo:
        return AdapterInfo(
            "dataset", cls.NAME, cls.VERSION, cls.FRAMEWORK, (), "derived, faulted view"
        )

    @classmethod
    def load(cls, source: str | Path, *, version: str, options: Mapping[str, object]) -> Self:
        raise FaultError("a faulted dataset is derived from a registered dataset, not loaded")

    @property
    def capabilities(self) -> frozenset[DatasetCapability]:
        return self.source.capabilities

    @property
    def identity(self) -> str:
        return faulted_identity(self.source.fingerprint(), self.spec, self.split)

    def fingerprint(self) -> str:
        return self.identity

    def splits(self) -> tuple[str, ...]:
        return self.source.splits()

    def num_samples(self, split: str | None = None) -> int:
        return self.source.num_samples(split)

    def metadata(self, *, deep: bool = False) -> DatasetMetadata:
        meta = self.source.metadata(deep=deep)
        note = {
            "fault_id": self.spec.id,
            "fault_family_id": self.spec.family_id,
            "source_fingerprint": meta.fingerprint,
            "split": self.split,
        }
        return replace(meta, fingerprint=self.identity, source={**meta.source, "fault": note})

    def sample(self, index: int, split: str | None = None) -> Sample:
        raise FaultError("sample access on a faulted dataset is not supported; iterate batches")

    def _classes(self) -> tuple[int | str, ...] | None:
        schema = self.source.metadata().target_schema
        return tuple(schema.class_labels) if schema is not None and schema.class_labels else None

    @property
    def content_digest(self) -> str | None:
        return "sha256:" + self._digest.hexdigest() if self._iterated else None

    def batches(self, batch_size: int, split: str | None = None) -> Iterator[Batch]:
        n_total = self.source.num_samples(split)
        self._applier = FaultApplier(
            self.spec, self.registry, n_total=n_total, classes=self._classes()
        )
        offset = 0
        for batch in self.source.batches(batch_size, split):
            started = time.perf_counter()
            x, x_kind = _to_numpy(batch.inputs)
            y, y_kind = (None, "") if batch.target is None else _to_numpy(batch.target)
            positions = np.arange(offset, offset + len(batch.indices))
            x2, y2 = self._applier.apply(x, y, positions)
            out_x = batch.inputs if x2 is x else _back(x2, x_kind)
            out_y = batch.target if y2 is y or y2 is None else _back(y2, y_kind)
            self._digest.update(np.ascontiguousarray(x2).tobytes())
            if y2 is not None:
                self._digest.update(np.ascontiguousarray(y2).tobytes())
            self.apply_seconds += time.perf_counter() - started
            offset += len(batch.indices)
            yield Batch(batch.indices, out_x, out_y)
        self._iterated = True

    def report(self) -> dict[str, object]:
        reports = self._applier.reports() if self._applier else ()
        return {
            "components": [to_jsonable(r) for r in reports],
            "apply_seconds": self.apply_seconds,
            "content_digest": self.content_digest,
        }


def total_affected(reports: Sequence[ComponentReport]) -> int:
    """Samples affected by at least the largest component (an upper bound for compound faults)."""
    return max((r.affected_samples for r in reports), default=0)

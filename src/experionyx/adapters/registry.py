"""Adapter registries: register, discover, resolve and inspect adapters by name.

Framework-specific code lives only inside adapter modules. Built-in adapters are registered
*lazily* (by import path) so listing works without their framework installed, and third-party
adapters plug in through `register()` or the entry-point groups `experionyx.model_adapters` /
`experionyx.dataset_adapters`. Registries are plain objects: no module-level mutable state.
"""

import importlib
import importlib.metadata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from experionyx.adapters.base import AdapterInfo, DatasetAdapter, ModelAdapter
from experionyx.errors import (
    AdapterNotFoundError,
    AdapterUnavailableError,
    DuplicateAdapterError,
    ValidationError,
)

MODEL_ENTRY_POINTS = "experionyx.model_adapters"
DATASET_ENTRY_POINTS = "experionyx.dataset_adapters"


class _Described(Protocol):
    @classmethod
    def info(cls) -> AdapterInfo: ...


T = TypeVar("T", bound=_Described)


@dataclass(frozen=True)
class AdapterStatus:
    name: str
    available: bool
    info: AdapterInfo | None  # None when unavailable
    reason: str | None = None  # why it is unavailable


class AdapterRegistry(Generic[T]):
    def __init__(self, kind: str, extra_hint: Callable[[str], str] | None = None) -> None:
        self._kind = kind
        self._loaders: dict[str, Callable[[], type[T]]] = {}
        self._hint = extra_hint or (lambda name: "")

    def register(self, name: str, adapter: type[T]) -> None:
        self._add(name, lambda: adapter)

    def register_lazy(self, name: str, target: str) -> None:
        """Register `module:Class`; the module is imported only when resolved or inspected."""

        def load() -> type[T]:
            module_name, _, attr = target.partition(":")
            try:
                cls = getattr(importlib.import_module(module_name), attr)
            except ImportError as exc:
                message = f"{self._kind} adapter {name!r} is unavailable: {exc}. {self._hint(name)}"
                raise AdapterUnavailableError(message.strip()) from exc
            return cls  # type: ignore[no-any-return]

        self._add(name, load)

    def _add(self, name: str, loader: Callable[[], type[T]]) -> None:
        if not name or name != name.strip():
            raise ValidationError(f"invalid adapter name {name!r}")
        if name in self._loaders:
            raise DuplicateAdapterError(f"{self._kind} adapter {name!r} is already registered")
        self._loaders[name] = loader

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._loaders))

    def resolve(self, name: str) -> type[T]:
        try:
            loader = self._loaders[name]
        except KeyError:
            raise AdapterNotFoundError(
                f"no {self._kind} adapter named {name!r} (registered: {list(self.names())})"
            ) from None
        return loader()

    def status(self) -> list[AdapterStatus]:
        out = []
        for name in self.names():
            try:
                out.append(AdapterStatus(name, True, self.resolve(name).info()))
            except AdapterUnavailableError as exc:
                out.append(AdapterStatus(name, False, None, str(exc)))
        return out

    def load_entry_points(self, group: str) -> list[str]:
        """Register third-party adapters advertised via entry points; returns the names added."""
        added = []
        for ep in importlib.metadata.entry_points(group=group):
            if ep.name not in self._loaders:
                self.register_lazy(ep.name, ep.value)
                added.append(ep.name)
        return added


@dataclass(frozen=True)
class AdapterRegistries:
    models: AdapterRegistry[ModelAdapter]
    datasets: AdapterRegistry[DatasetAdapter]


def default_registries(*, entry_points: bool = False) -> AdapterRegistries:
    """A fresh pair of registries with the built-in sklearn and torch adapters (lazily loaded)."""
    models: AdapterRegistry[ModelAdapter] = AdapterRegistry("model", _hint_by_name)
    datasets: AdapterRegistry[DatasetAdapter] = AdapterRegistry("dataset", _hint_by_name)
    models.register_lazy("sklearn", "experionyx.adapters.sklearn_adapter:SklearnModelAdapter")
    models.register_lazy("torch", "experionyx.adapters.torch_adapter:TorchModelAdapter")
    datasets.register_lazy("sklearn", "experionyx.adapters.sklearn_adapter:SklearnDatasetAdapter")
    datasets.register_lazy("torch", "experionyx.adapters.torch_adapter:TorchDatasetAdapter")
    if entry_points:
        models.load_entry_points(MODEL_ENTRY_POINTS)
        datasets.load_entry_points(DATASET_ENTRY_POINTS)
    return AdapterRegistries(models, datasets)


def _hint_by_name(name: str) -> str:
    return f"Install it with: pip install 'experionyx[{name}]'"

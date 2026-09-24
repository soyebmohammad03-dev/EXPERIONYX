"""FastAPI dependency accessors. `app.create_app` records the registry path and opens the
artifact store once; `get_registry` opens a fresh `SqliteRegistry` per request and closes it when
the request finishes, because sqlite3 connections cannot cross the worker threads FastAPI runs
synchronous endpoint functions in. Routers never open their own connection."""

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request

from experionyx.artifacts import ArtifactStore
from experionyx.sqlite import SqliteRegistry


def get_registry(request: Request) -> Iterator[SqliteRegistry]:
    registry = SqliteRegistry(request.app.state.registry_path)
    try:
        yield registry
    finally:
        registry.close()


def get_store(request: Request) -> ArtifactStore:
    store: ArtifactStore = request.app.state.store
    return store


RegistryDep = Annotated[SqliteRegistry, Depends(get_registry)]
StoreDep = Annotated[ArtifactStore, Depends(get_store)]

__all__ = ["RegistryDep", "StoreDep", "get_registry", "get_store"]

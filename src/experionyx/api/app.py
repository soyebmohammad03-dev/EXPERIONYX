"""FastAPI application factory: opens the workspace registry/artifact store once, mounts every
read-only router under `/api`, and serves the static vanilla-JS SPA at `/` (see docs/api.md)."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from experionyx.api import errors
from experionyx.api.routers import (
    benchmark,
    dossier,
    drift,
    failures,
    faults,
    graph,
    investigations,
    reliability,
    reporting,
    reproducibility,
    stats,
    viz,
)
from experionyx.artifacts import LocalArtifactStore
from experionyx.errors import ExperionyxError

REGISTRY_FILE = "registry.sqlite"
EXPERIMENTS_DIR = "experiments"
_STATIC_DIR = Path(__file__).parent / "static"

_ROUTERS = (
    investigations.router,
    reliability.router,
    failures.router,
    faults.router,
    drift.router,
    stats.router,
    benchmark.router,
    reproducibility.router,
    graph.router,
    reporting.router,
    dossier.router,
    viz.router,
)


def create_app(workspace: str) -> FastAPI:
    path = Path(workspace) / REGISTRY_FILE
    if not path.is_file():
        raise ExperionyxError(f"no registry at {path} (use --workspace to point at a workspace)")
    path.stat()  # fail fast if the path is somehow unreadable before serving any request

    app = FastAPI(title="EXPERIONYX", description="Read-only research analysis interface")
    # A fresh SqliteRegistry per request (see deps.get_registry): endpoint functions run in a
    # worker thread pool, and sqlite3 connections cannot cross threads. Opening one is cheap and
    # matches the CLI's own pattern of a fresh connection per command.
    app.state.registry_path = path
    app.state.store = LocalArtifactStore(Path(workspace) / EXPERIMENTS_DIR)

    errors.install(app)
    for router in _ROUTERS:
        app.include_router(router)

    if _STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")

    return app


__all__ = ["create_app"]

"""Maps the existing `ExperionyxError` hierarchy to JSON error responses. No new error taxonomy."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from experionyx.errors import (
    BenchmarkRefusal,
    DesignRefusal,
    ExperionyxError,
    NotFoundError,
    ProfileRefusal,
    SchedulerRefusal,
    ValidationError,
)

_REFUSAL_TYPES = (DesignRefusal, ProfileRefusal, BenchmarkRefusal, SchedulerRefusal)


def _body(exc: Exception) -> dict[str, object]:
    return {"error": type(exc).__name__, "detail": str(exc)}


def install(app: FastAPI) -> None:
    @app.exception_handler(NotFoundError)
    async def _not_found(_request: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content=_body(exc))

    @app.exception_handler(ValidationError)
    async def _invalid(_request: Request, exc: ValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content=_body(exc))

    async def _refused(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=422, content=_body(exc))

    for refusal_type in _REFUSAL_TYPES:
        app.add_exception_handler(refusal_type, _refused)

    @app.exception_handler(ExperionyxError)
    async def _other(_request: Request, exc: ExperionyxError) -> JSONResponse:
        return JSONResponse(status_code=500, content=_body(exc))


__all__ = ["install"]

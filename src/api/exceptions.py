"""Centralized exception handler registration."""

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


async def _value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    logger.warning(f"ValueError at {request.url}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )


async def _runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
    logger.error(f"RuntimeError at {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred."},
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ValueError, _value_error_handler)
    app.add_exception_handler(RuntimeError, _runtime_error_handler)

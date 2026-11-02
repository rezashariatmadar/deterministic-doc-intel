"""FastAPI custom middleware for request tracing and structured logging."""

import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.exceptions import (
    AppException,
    ConfigurationError,
    DocumentNotFoundError,
    DuplicateDocumentError,
    IngestionError,
)
from app.core.logging import correlation_id_ctx


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Ensure every HTTP request has an X-Request-ID header and logging context."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        corr_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        token = correlation_id_ctx.set(corr_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = corr_id
            return response
        finally:
            correlation_id_ctx.reset(token)


def register_exception_handlers(app: FastAPI) -> None:
    """Register custom exception handlers mapping domain exceptions to HTTP responses."""

    @app.exception_handler(DocumentNotFoundError)
    async def document_not_found_handler(
        request: Request, exc: DocumentNotFoundError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"error": "DocumentNotFound", "message": exc.message},
        )

    @app.exception_handler(DuplicateDocumentError)
    async def duplicate_document_handler(
        request: Request, exc: DuplicateDocumentError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"error": "DuplicateDocument", "message": exc.message},
        )

    @app.exception_handler(IngestionError)
    async def ingestion_error_handler(request: Request, exc: IngestionError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": "IngestionError", "message": exc.message},
        )

    @app.exception_handler(ConfigurationError)
    async def configuration_error_handler(
        request: Request, exc: ConfigurationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"error": "ConfigurationError", "message": exc.message},
        )

    @app.exception_handler(AppException)
    async def general_app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.error_code, "message": exc.message},
        )

"""Main entry point for the Hybrid & Agentic RAG FastAPI application."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.middleware import CorrelationIdMiddleware, register_exception_handlers
from app.api.v1.router import api_v1_router
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging

settings = get_settings()
setup_logging(log_level=settings.LOG_LEVEL)
logger = get_logger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager handling startup and shutdown events."""
    logger.info(
        "Starting up RAG application",
        extra={
            "app_name": settings.APP_NAME,
            "env": settings.APP_ENV,
            "embedding_provider": settings.EMBEDDING_PROVIDER,
            "llm_provider": settings.LLM_PROVIDER,
        },
    )
    yield
    logger.info("Shutting down RAG application")


app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    description="Enterprise-grade production Hybrid & Agentic RAG Platform.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# 1. Custom correlation ID tracing middleware
app.add_middleware(CorrelationIdMiddleware)

# 2. Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Cache-Hit", "X-Request-ID", "Retry-After"],
)

# 3. Register domain exception handlers
register_exception_handlers(app)

# 4. Include v1 API router
app.include_router(api_v1_router)

# 5. Mount frontend dashboard if built
_dashboard_dist = Path(__file__).resolve().parent.parent / "dashboard" / "dist"
if _dashboard_dist.is_dir():
    _assets_dir = _dashboard_dist / "assets"
    if _assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="dashboard_assets")
    app.mount(
        "/dashboard", StaticFiles(directory=str(_dashboard_dist), html=True), name="dashboard"
    )


@app.get("/", tags=["Root"])
async def root() -> dict[str, str]:
    """Root metadata endpoint."""
    return {
        "app": settings.APP_NAME,
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/v1/health",
        "dashboard": "/dashboard" if _dashboard_dist.is_dir() else "not_built",
    }

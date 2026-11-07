"""Health check endpoint verifying database and Redis connectivity."""

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db
from app.core.config import get_settings

router = APIRouter()
settings = get_settings()


def determine_health_status(db_status: str, redis_status: str) -> tuple[str, int]:
    """Determine overall health status string and corresponding HTTP status code."""
    is_healthy = db_status == "connected" and redis_status == "connected"
    status_str = "healthy" if is_healthy else "degraded"
    http_code = status.HTTP_200_OK if is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    return status_str, http_code


@router.get("/health")
async def health_check(
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Check connectivity to PostgreSQL and Redis Stack."""
    # 1. Database check
    db_status = "connected"
    try:
        await db.execute(text("SELECT 1"))
    except Exception:
        db_status = "disconnected"

    # 2. Redis check
    redis_status = "connected"
    try:
        r = aioredis.from_url(settings.REDIS_URL, socket_timeout=2.0)
        await r.ping()
        await r.aclose()
    except Exception:
        redis_status = "disconnected"

    overall_status, http_code = determine_health_status(db_status, redis_status)
    response.status_code = http_code

    return {
        "status": overall_status,
        "database": db_status,
        "redis": redis_status,
        "version": "0.1.0",
    }

"""Health check endpoint verifying database and Redis connectivity."""

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db
from app.core.config import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/health")
async def health_check(
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

    overall_status = (
        "healthy" if db_status == "connected" and redis_status == "connected" else "degraded"
    )

    return {
        "status": overall_status,
        "database": db_status,
        "redis": redis_status,
        "version": "0.1.0",
    }

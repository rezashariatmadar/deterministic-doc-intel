"""Metrics endpoints for monitoring query telemetry, latencies, and costs."""

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db
from app.db.repositories.telemetry import TelemetryRepository
from app.schemas.metrics import MetricsSummaryResponse

router = APIRouter()


@router.get("/metrics/summary", response_model=MetricsSummaryResponse)
async def get_metrics_summary(
    start_time: datetime | None = Query(
        default=None,
        description="Filter metrics created on or after this timestamp",
    ),
    db: AsyncSession = Depends(get_db),
) -> MetricsSummaryResponse:
    """
    Retrieve system-wide aggregated telemetry metrics including latency percentiles,
    hit ratios, and costs.
    """
    repo = TelemetryRepository(db)
    return await repo.get_summary_metrics(start_time=start_time)

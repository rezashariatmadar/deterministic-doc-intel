"""Central API v1 router combining all endpoint modules."""

from fastapi import APIRouter

from app.api.v1.endpoints import documents, health, metrics, query

api_v1_router = APIRouter(prefix="/api/v1")

api_v1_router.include_router(health.router, tags=["Health"])
api_v1_router.include_router(documents.router, tags=["Documents"])
api_v1_router.include_router(query.router, tags=["Query"])
api_v1_router.include_router(metrics.router, tags=["Metrics"])

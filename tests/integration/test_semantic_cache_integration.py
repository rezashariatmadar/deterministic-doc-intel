"""Integration tests for Redis semantic cache layer and cascade invalidation."""

import os
import uuid
from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_embedding, get_llm, get_redis_client
from app.core.config import get_settings
from app.db.models.query_log import QueryLog
from app.main import app
from tests.integration.test_api_endpoints import DummyEmbeddingProvider, DummyLLMProvider

settings = get_settings()


@pytest.fixture
async def integration_client() -> AsyncGenerator[AsyncClient, None]:
    """Client for testing semantic cache with dummy embedding and LLM providers."""
    app.dependency_overrides[get_embedding] = lambda: DummyEmbeddingProvider(
        settings.EMBEDDING_DIMENSION
    )
    app.dependency_overrides[get_llm] = lambda: DummyLLMProvider()

    # Flush Redis keys before testing
    redis = get_redis_client()
    try:
        keys = await redis.keys("semcache:*")
        if keys:
            await redis.delete(*keys)
    except Exception:
        pass

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_semantic_cache_hit_and_cascade_document_invalidation(
    integration_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    pdf_path = "data/raw_corpus/nvidia_10k_sample.pdf"
    assert os.path.exists(pdf_path)

    # 1. Ingest document
    with open(pdf_path, "rb") as f:
        ingest_resp = await integration_client.post(
            "/api/v1/documents/ingest",
            files={"file": (f"test_cache_{uuid.uuid4().hex[:8]}.pdf", f, "application/pdf")},
            data={"category": "financial_report"},
        )
    assert ingest_resp.status_code in (200, 201)
    doc_id = ingest_resp.json()["id"]

    query_payload = {
        "query": "What was the Data Center revenue in fiscal 2024?",
        "top_k": 3,
        "search_mode": "hybrid",
    }

    # 2. First query: cache miss
    resp_1 = await integration_client.post("/api/v1/query", json=query_payload)
    assert resp_1.status_code == 200
    assert resp_1.headers.get("X-Cache-Hit") == "false"
    data_1 = resp_1.json()
    assert "Data Center revenue was $47.5B" in data_1["answer"]

    # Verify telemetry query log for first query is cache_hit=False
    stmt_1 = (
        select(QueryLog)
        .where(QueryLog.query_text == query_payload["query"])
        .order_by(QueryLog.created_at.desc())
    )
    res_1 = await db_session.execute(stmt_1)
    logs_1 = res_1.scalars().all()
    assert len(logs_1) >= 1
    assert logs_1[0].cache_hit is False

    # 3. Second query: cache hit
    resp_2 = await integration_client.post("/api/v1/query", json=query_payload)
    assert resp_2.status_code == 200
    assert resp_2.headers.get("X-Cache-Hit") == "true"
    data_2 = resp_2.json()
    assert data_2["answer"] == data_1["answer"]
    assert data_2["confidence_score"] == data_1["confidence_score"]

    # Verify telemetry query log for second query is cache_hit=True
    res_2 = await db_session.execute(stmt_1)
    logs_2 = res_2.scalars().all()
    assert len(logs_2) >= 2
    assert logs_2[0].cache_hit is True
    assert logs_2[0].rewrite_ms == 0.0
    assert logs_2[0].dense_search_ms == 0.0
    assert logs_2[0].prompt_tokens == 0
    assert logs_2[0].completion_tokens == 0

    # 4. Delete document
    del_resp = await integration_client.delete(f"/api/v1/documents/{doc_id}")
    assert del_resp.status_code == 204

    # 5. Third query after deletion: cache miss because document was invalidated
    resp_3 = await integration_client.post("/api/v1/query", json=query_payload)
    assert resp_3.status_code == 200
    assert resp_3.headers.get("X-Cache-Hit") == "false"

    res_3 = await db_session.execute(stmt_1)
    logs_3 = res_3.scalars().all()
    assert len(logs_3) >= 3
    assert logs_3[0].cache_hit is False


@pytest.mark.asyncio
async def test_semantic_cache_disabled_fallback(
    integration_client: AsyncClient,
) -> None:
    from app.api.dependencies import get_semantic_cache

    app.dependency_overrides[get_semantic_cache] = lambda: None

    query_payload = {
        "query": "What was the Data Center revenue in fiscal 2024?",
        "top_k": 3,
        "search_mode": "hybrid",
    }

    resp = await integration_client.post("/api/v1/query", json=query_payload)
    assert resp.status_code == 200
    assert resp.headers.get("X-Cache-Hit") == "false"

"""Integration tests for Telemetry persistence, Document task fields, and Metrics endpoint."""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_embedding, get_llm
from app.db.models.document import Document
from app.db.models.query_log import QueryLog
from app.db.repositories.telemetry import TelemetryRepository
from app.main import app
from app.schemas.metrics import MetricsSummaryResponse, QueryLogCreate
from tests.integration.test_api_endpoints import DummyEmbeddingProvider, DummyLLMProvider


@pytest.mark.asyncio
async def test_document_task_fields_persistence(db_session):
    """Verify that Document model supports task_id and processing_stage columns."""
    file_hash = uuid.uuid4().hex
    doc = Document(
        filename="async_doc.pdf",
        file_hash=file_hash,
        title="Async Ingestion Test Document",
        total_pages=1,
        status="PENDING",
        task_id="celery-task-999",
        processing_stage="QUEUED",
    )
    db_session.add(doc)
    await db_session.commit()
    await db_session.refresh(doc)

    assert doc.id is not None
    assert doc.task_id == "celery-task-999"
    assert doc.processing_stage == "QUEUED"


@pytest.mark.asyncio
async def test_query_log_persistence_and_repository(db_session):
    """Verify QueryLog creation and summary metrics calculation via TelemetryRepository."""
    repo = TelemetryRepository(db_session)

    req_id_1 = str(uuid.uuid4())
    req_id_2 = str(uuid.uuid4())

    log_1 = QueryLogCreate(
        request_id=req_id_1,
        query_text="What was Apple's total revenue in 2023?",
        search_mode="hybrid",
        cache_hit=False,
        is_answerable=True,
        confidence_score=0.92,
        top_reranker_score=0.88,
        rewrite_ms=45.0,
        dense_search_ms=12.0,
        sparse_search_ms=8.0,
        rerank_ms=25.0,
        llm_ttft_ms=120.0,
        total_latency_ms=210.0,
        prompt_tokens=450,
        completion_tokens=80,
        embedding_tokens=15,
        estimated_cost_usd=0.000115,
    )
    log_2 = QueryLogCreate(
        request_id=req_id_2,
        query_text="What was Apple's total revenue in 2023?",
        search_mode="hybrid",
        cache_hit=True,
        is_answerable=True,
        confidence_score=0.92,
        top_reranker_score=None,
        rewrite_ms=0.0,
        dense_search_ms=0.0,
        sparse_search_ms=0.0,
        rerank_ms=0.0,
        llm_ttft_ms=None,
        total_latency_ms=2.5,
        prompt_tokens=0,
        completion_tokens=0,
        embedding_tokens=15,
        estimated_cost_usd=0.0000003,
    )

    created_1 = await repo.log_query(log_1)
    created_2 = await repo.log_query(log_2)

    assert created_1.id is not None
    assert created_1.request_id == req_id_1
    assert created_2.id is not None
    assert created_2.cache_hit is True

    summary = await repo.get_summary_metrics()
    assert isinstance(summary, MetricsSummaryResponse)
    assert summary.total_queries >= 2
    assert 0.0 <= summary.cache_hit_ratio <= 1.0
    assert summary.latencies.p50_ms >= 0.0
    assert summary.latencies.p95_ms >= summary.latencies.p50_ms


@pytest.mark.asyncio
async def test_metrics_summary_api_endpoint(db_session):
    """Verify GET /api/v1/metrics/summary returns valid JSON and schema."""
    repo = TelemetryRepository(db_session)
    await repo.log_query(
        QueryLogCreate(
            request_id=str(uuid.uuid4()),
            query_text="Summary API test query",
            search_mode="dense",
            cache_hit=False,
            is_answerable=True,
            confidence_score=0.85,
            rewrite_ms=0.0,
            dense_search_ms=15.0,
            sparse_search_ms=0.0,
            rerank_ms=0.0,
            total_latency_ms=85.0,
            prompt_tokens=100,
            completion_tokens=20,
            embedding_tokens=10,
            estimated_cost_usd=0.000027,
        )
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/metrics/summary")
        assert response.status_code == 200
        data = response.json()
        assert "total_queries" in data
        assert "cache_hit_ratio" in data
        assert "abstention_rate" in data
        assert "total_cost_usd" in data
        assert "avg_cost_per_1k_usd" in data
        assert "latencies" in data
        assert "p50_ms" in data["latencies"]
        assert "p95_ms" in data["latencies"]
        assert "p99_ms" in data["latencies"]
        assert "stage_latencies_avg" in data


@pytest.mark.asyncio
async def test_query_endpoint_persists_telemetry(db_session):
    """Verify that calling POST /api/v1/query generates and persists a QueryLog."""
    # Override dependencies with dummy providers
    dim = 1024
    app.dependency_overrides[get_embedding] = lambda: DummyEmbeddingProvider(dimension=dim)
    app.dependency_overrides[get_llm] = lambda: DummyLLMProvider()

    query_text = f"What was the growth in Data Center revenue? {uuid.uuid4().hex}"

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/query",
                json={
                    "query": query_text,
                    "search_mode": "hybrid",
                    "top_k": 3,
                },
            )
            assert resp.status_code == 200

        # Query the database to verify a QueryLog was persisted
        from sqlalchemy import select

        stmt = (
            select(QueryLog)
            .where(QueryLog.query_text == query_text)
            .order_by(QueryLog.created_at.desc())
        )
        result = await db_session.execute(stmt)
        log_entry = result.scalars().first()

        assert log_entry is not None
        assert log_entry.search_mode == "hybrid"
        assert log_entry.total_latency_ms > 0
        assert log_entry.prompt_tokens >= 0
        assert log_entry.completion_tokens >= 0
        assert log_entry.embedding_tokens > 0
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_sparse_query_zero_embedding_tokens(db_session):
    """Verify that sparse search queries account for 0 embedding tokens."""
    dim = 1024
    app.dependency_overrides[get_embedding] = lambda: DummyEmbeddingProvider(dimension=dim)
    app.dependency_overrides[get_llm] = lambda: DummyLLMProvider()

    query_text = f"What was the sparse query revenue? {uuid.uuid4().hex}"

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/query",
                json={
                    "query": query_text,
                    "search_mode": "sparse",
                    "top_k": 3,
                },
            )
            assert resp.status_code == 200

        from sqlalchemy import select

        stmt = (
            select(QueryLog)
            .where(QueryLog.query_text == query_text)
            .order_by(QueryLog.created_at.desc())
        )
        result = await db_session.execute(stmt)
        log_entry = result.scalars().first()

        assert log_entry is not None
        assert log_entry.search_mode == "sparse"
        assert log_entry.embedding_tokens == 0
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_stream_query_endpoint_persists_telemetry(db_session):
    """Verify that calling POST /api/v1/query/stream generates and persists a QueryLog."""
    dim = 1024
    app.dependency_overrides[get_embedding] = lambda: DummyEmbeddingProvider(dimension=dim)
    app.dependency_overrides[get_llm] = lambda: DummyLLMProvider()

    query_text = f"What was the stream revenue growth? {uuid.uuid4().hex}"

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/query/stream",
                json={
                    "query": query_text,
                    "search_mode": "hybrid",
                    "top_k": 3,
                },
            )
            assert resp.status_code == 200
            # Read full stream content to let sse_event_publisher complete
            _ = resp.text

        from sqlalchemy import select

        stmt = (
            select(QueryLog)
            .where(QueryLog.query_text == query_text)
            .order_by(QueryLog.created_at.desc())
        )
        result = await db_session.execute(stmt)
        log_entry = result.scalars().first()

        assert log_entry is not None
        assert log_entry.search_mode == "hybrid"
        assert log_entry.total_latency_ms > 0
        assert log_entry.llm_ttft_ms is not None
        assert log_entry.prompt_tokens >= 0
        assert log_entry.completion_tokens >= 0
    finally:
        app.dependency_overrides.clear()

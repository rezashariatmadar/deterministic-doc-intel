"""Unit tests verifying Phase 5 Milestone M0 Backend Fixes (B1 - B9)."""

import json
from datetime import UTC
from unittest.mock import patch
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_embedding, get_llm
from app.db.repositories.telemetry import TelemetryRepository
from app.main import app
from app.schemas.metrics import QueryLogCreate
from app.schemas.query import (
    Citation,
    CitationSourceType,
    QueryRequest,
    RAGResponse,
)
from app.services.agent.external_search import ExternalSearchTool
from tests.integration.test_api_endpoints import DummyEmbeddingProvider, DummyLLMProvider


def test_b1_b9_query_request_fields():
    """B1 & B9: QueryRequest includes bypass_cache and resolve_parents fields."""
    req = QueryRequest(query="Test query with options", bypass_cache=True, resolve_parents=False)
    assert req.bypass_cache is True
    assert req.resolve_parents is False

    req_default = QueryRequest(query="Default options query")
    assert req_default.bypass_cache is False
    assert req_default.resolve_parents is True


def test_b1_expanded_canonical_scope_hash():
    """B1: canonical_scope_hash incorporates all execution parameters deterministically."""
    from unittest.mock import AsyncMock

    from app.schemas.query import IntentRoute
    from app.services.cache.semantic_cache import SemanticCacheService

    cache = SemanticCacheService(redis=AsyncMock())

    base_hash = cache.canonical_scope_hash("hybrid", top_k=5)

    # Test each parameter variation creates a distinct hash
    assert cache.canonical_scope_hash("hybrid", top_k=5, resolve_parents=False) != base_hash
    assert cache.canonical_scope_hash("hybrid", top_k=5, enable_agent=True) != base_hash
    assert (
        cache.canonical_scope_hash(
            "hybrid", top_k=5, route_override=IntentRoute.STRUCTURED_SQL_ANALYTICS.value
        )
        != base_hash
    )
    assert cache.canonical_scope_hash("hybrid", top_k=5, enable_query_rewrite=True) != base_hash
    assert cache.canonical_scope_hash("hybrid", top_k=5, enable_hyde=True) != base_hash
    assert cache.canonical_scope_hash("hybrid", top_k=5, allow_external_fallback=True) != base_hash
    assert cache.canonical_scope_hash("hybrid", top_k=5, max_crag_retries=1) != base_hash
    assert cache.canonical_scope_hash("hybrid", top_k=5, weight_dense=0.7) != base_hash
    assert cache.canonical_scope_hash("hybrid", top_k=5, weight_sparse=0.3) != base_hash


def test_b9_agent_state_resolve_parents():
    """B9: AgentState carries resolve_parents and defaults to True."""
    from app.services.agent.orchestrator import AgentState

    state_default = AgentState(query="test", original_query="test")
    assert state_default.resolve_parents is True

    state_custom = AgentState(query="test", original_query="test", resolve_parents=False)
    assert state_custom.resolve_parents is False


def test_b5_rag_response_stage_latencies():
    """B5: RAGResponse supports stage_latencies dictionary or None."""
    resp = RAGResponse(
        answer="Sample answer",
        confidence_score=0.9,
        latency_ms=120.5,
        stage_latencies={"dense_search_ms": 15.2, "rerank_ms": 25.0},
    )
    assert resp.stage_latencies is not None
    assert resp.stage_latencies["dense_search_ms"] == 15.2
    assert resp.stage_latencies["rerank_ms"] == 25.0

    resp_none = RAGResponse(
        answer="Sample answer",
        confidence_score=0.9,
        latency_ms=120.5,
        stage_latencies=None,
    )
    assert resp_none.stage_latencies is None

    resp_default = RAGResponse(
        answer="Sample answer",
        confidence_score=0.9,
        latency_ms=120.5,
    )
    assert resp_default.stage_latencies is None


def test_b8_citation_match_tier_and_rehomed_from():
    """B8: Citation model supports match_tier and rehomed_from attributes."""
    citation = Citation(
        source_id=2,
        document_title="Sample Doc",
        page_start=1,
        page_end=1,
        section_path="Intro",
        supporting_quote="Key revenue was $50M.",
        match_tier="normalized",
        rehomed_from=1,
    )
    assert citation.match_tier == "normalized"
    assert citation.rehomed_from == 1
    assert citation.source_type == CitationSourceType.DOCUMENT


@pytest.mark.asyncio
async def test_b4_cors_expose_headers():
    """B4: CORS middleware exposes custom headers to frontend clients."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/api/v1/health",
            headers={"Origin": "http://localhost:5173"},
        )
        assert resp.status_code == 200
        exposed = resp.headers.get("access-control-expose-headers", "")
        # Should include X-Cache-Hit, X-Request-ID, Retry-After
        assert "X-Cache-Hit" in exposed
        assert "X-Request-ID" in exposed
        assert "Retry-After" in exposed


@pytest.mark.asyncio
async def test_m1_fastapi_dashboard_mount():
    """M1: FastAPI mounts the built dashboard at /dashboard and root metadata references it."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        root_resp = await client.get("/")
        assert root_resp.status_code == 200
        root_data = root_resp.json()
        assert "dashboard" in root_data

        dash_resp = await client.get("/dashboard")
        # Should either redirect to /dashboard/ or return 200 index.html
        assert dash_resp.status_code in (200, 307)
        dash_index_resp = await client.get("/dashboard/")
        assert dash_index_resp.status_code == 200
        assert "داشبورد مهندسی رگ" in dash_index_resp.text


@pytest.mark.asyncio
async def test_b6_telemetry_stage_latency_averages_filter_non_cache():
    """B6: Summary metrics average stage latencies filter out cache hits and zero-stage logs."""
    from datetime import datetime

    from app.db.session import async_session_maker

    test_start = datetime.now(UTC)

    async with async_session_maker() as session:
        repo = TelemetryRepository(session)
        # Log a non-cache query with dense search 40ms
        await repo.log_query(
            QueryLogCreate(
                request_id=str(uuid4()),
                query_text="Non-cache query",
                search_mode="dense",
                cache_hit=False,
                is_answerable=True,
                confidence_score=0.9,
                rewrite_ms=10.0,
                dense_search_ms=40.0,
                sparse_search_ms=0.0,
                rerank_ms=20.0,
                total_latency_ms=100.0,
                prompt_tokens=50,
                completion_tokens=20,
                embedding_tokens=10,
                estimated_cost_usd=0.0001,
            )
        )
        # Log 5 cache hits with 0ms stage latencies
        for _ in range(5):
            await repo.log_query(
                QueryLogCreate(
                    request_id=str(uuid4()),
                    query_text="Cache hit query",
                    search_mode="dense",
                    cache_hit=True,
                    is_answerable=True,
                    confidence_score=0.9,
                    rewrite_ms=0.0,
                    dense_search_ms=0.0,
                    sparse_search_ms=0.0,
                    rerank_ms=0.0,
                    total_latency_ms=5.0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    embedding_tokens=10,
                    estimated_cost_usd=0.00001,
                )
            )

        summary = await repo.get_summary_metrics(start_time=test_start)
        # With 1 non-cache log of 40ms and 5 cache hits of 0ms, average should be 40.0, NOT 6.66ms!
        assert summary.total_queries == 6
        assert summary.stage_latencies_avg.dense_search_ms == 40.0
        assert summary.stage_latencies_avg.rerank_ms == 20.0
        assert summary.stage_latencies_avg.rewrite_ms == 10.0


@pytest.mark.asyncio
async def test_b7_external_search_tool_returns_empty_on_failure():
    """B7: When external search encounters errors, return [] instead of synthetic chunk."""
    tool = ExternalSearchTool(timeout_seconds=0.5)
    with patch("httpx.AsyncClient.get", side_effect=Exception("Network failure")):
        results = await tool.search("Any query", allow_external_fallback=True)
        assert results == []


@pytest.mark.asyncio
async def test_b1_b2_b3_sse_stream_verification_payload_and_bypass_cache():
    """B1, B2, B3: Verify SSE stream verification payload includes full sources and metadata."""
    dim = 1024
    app.dependency_overrides[get_embedding] = lambda: DummyEmbeddingProvider(dimension=dim)
    app.dependency_overrides[get_llm] = lambda: DummyLLMProvider()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Non-cached stream request with bypass_cache=True
        req_body = {
            "query": "What is the capital of France and what are its features?",
            "search_mode": "hybrid",
            "stream": True,
            "bypass_cache": True,
            "resolve_parents": False,
        }
        resp = await client.post("/api/v1/query/stream", json=req_body)
        assert resp.status_code == 200
        assert resp.headers.get("X-Cache-Hit") == "false"

        events = []
        for line in resp.text.split("\n"):
            if line.startswith("data: ") and not line.startswith("data: [DONE]"):
                payload = json.loads(line[len("data: ") :])
                events.append(payload)

        verif_events = [e for e in events if e.get("event") == "verification"]
        assert len(verif_events) == 1
        verif_data = verif_events[0]["data"]

        # Check full sources in verification payload (B2)
        assert "sources" in verif_data
        assert isinstance(verif_data["sources"], list)
        # Check stage_latencies in verification payload (B5)
        assert "stage_latencies" in verif_data
        assert isinstance(verif_data["stage_latencies"], dict)
        # Check citations have source_type (B2 & B8)
        for cit in verif_data["citations"]:
            assert "source_type" in cit
            assert "match_tier" in cit
            assert "rehomed_from" in cit

        # 2. Synchronous query with bypass_cache=True (B1)
        sync_req_body = {
            "query": "What is the capital of France and what are its features?",
            "search_mode": "hybrid",
            "bypass_cache": True,
            "resolve_parents": False,
        }
        sync_resp = await client.post("/api/v1/query", json=sync_req_body)
        assert sync_resp.status_code == 200
        assert sync_resp.headers.get("X-Cache-Hit") == "false"
        sync_data = sync_resp.json()
        assert "stage_latencies" in sync_data
        assert "sources" in sync_data

    app.dependency_overrides.clear()

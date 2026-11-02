"""Integration tests for FastAPI application endpoints."""

import os
from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_embedding, get_llm
from app.core.config import get_settings
from app.main import app
from app.services.providers.base import EmbeddingProvider, LLMProvider

settings = get_settings()


class DummyEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider for API integration tests."""

    def __init__(self, dimension: int) -> None:
        super().__init__(model_name="dummy-embed", dimension=dimension)
        # Unit vector along first dimension
        self.vec = [1.0] + [0.0] * (dimension - 1)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.vec for _ in texts]

    async def embed_query(self, query: str) -> list[float]:
        return self.vec


class DummyLLMProvider(LLMProvider):
    """Deterministic LLM provider for API integration tests."""

    def __init__(self) -> None:
        super().__init__(model_name="dummy-llm")

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        return "NVIDIA Data Center revenue was $47.5B in fiscal 2024 [1]."

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> AsyncGenerator[str, None]:
        tokens = ["NVIDIA", " Data", " Center", " revenue", " was", " $47.5B", " [1]."]
        for t in tokens:
            yield t


@pytest.fixture
async def api_client() -> AsyncGenerator[AsyncClient, None]:
    # Override providers with deterministic mock providers
    app.dependency_overrides[get_embedding] = lambda: DummyEmbeddingProvider(
        settings.EMBEDDING_DIMENSION
    )
    app.dependency_overrides[get_llm] = lambda: DummyLLMProvider()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_health_check_endpoint(api_client: AsyncClient) -> None:
    response = await api_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
    assert data["redis"] == "connected"


@pytest.mark.asyncio
async def test_document_ingestion_list_query_delete_flow(
    api_client: AsyncClient,
) -> None:
    pdf_path = "data/raw_corpus/nvidia_10k_sample.pdf"
    assert os.path.exists(pdf_path)

    # 1. Ingest document via POST /api/v1/documents/ingest
    with open(pdf_path, "rb") as f:
        upload_resp = await api_client.post(
            "/api/v1/documents/ingest",
            files={"file": ("nvidia_sample.pdf", f, "application/pdf")},
            data={"category": "financial_report"},
        )

    assert upload_resp.status_code in [200, 201]
    doc_data = upload_resp.json()
    assert doc_data["title"] is not None
    assert doc_data["total_pages"] == 3
    assert doc_data["total_chunks"] > 0
    assert doc_data["status"] == "indexed"
    doc_id = doc_data["id"]

    try:
        # 2. Re-ingesting same file should return existing document (idempotent)
        with open(pdf_path, "rb") as f:
            dup_resp = await api_client.post(
                "/api/v1/documents/ingest",
                files={"file": ("nvidia_sample.pdf", f, "application/pdf")},
                data={"category": "financial_report"},
            )
        assert dup_resp.status_code == 200
        assert dup_resp.json()["id"] == doc_id

        # 3. List documents via GET /api/v1/documents
        list_resp = await api_client.get("/api/v1/documents")
        assert list_resp.status_code == 200
        items = list_resp.json()
        assert any(item["id"] == doc_id for item in items)

        # 4. Standard query via POST /api/v1/query
        query_resp = await api_client.post(
            "/api/v1/query",
            json={
                "query": "What was the Data Center revenue in fiscal 2024?",
                "top_k": 3,
            },
        )
        assert query_resp.status_code == 200
        query_data = query_resp.json()
        assert "Data Center revenue was $47.5B" in query_data["answer"]
        assert len(query_data["citations"]) >= 1
        assert query_data["confidence_score"] > 0.0
        assert len(query_data["sources"]) > 0

        # 5. Streaming query via POST /api/v1/query/stream
        stream_resp = await api_client.post(
            "/api/v1/query/stream",
            json={
                "query": "What was the Data Center revenue in fiscal 2024?",
                "top_k": 3,
            },
        )
        assert stream_resp.status_code == 200
        assert "text/event-stream" in stream_resp.headers["content-type"]
        body_text = stream_resp.text
        assert "data: " in body_text

    finally:
        # 6. Delete document via DELETE /api/v1/documents/{id}
        del_resp = await api_client.delete(f"/api/v1/documents/{doc_id}")
        assert del_resp.status_code in [200, 204]

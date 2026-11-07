"""Integration tests for synchronous and asynchronous document ingestion.

Covers Celery worker execution, event loop isolation, and live task tracking.
"""

import math
import os
from collections.abc import AsyncGenerator
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_embedding
from app.core.config import get_settings
from app.main import app
from app.services.providers.base import EmbeddingProvider
from app.workers.tasks import ingest_document_task

settings = get_settings()


class DummyEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider for integration tests."""

    def __init__(self, dimension: int) -> None:
        super().__init__(model_name="dummy-embed", dimension=dimension)
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        vec[0] = 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    async def embed_query(self, query: str) -> list[float]:
        return self._embed(query)


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    app.dependency_overrides[get_embedding] = lambda: DummyEmbeddingProvider(
        settings.EMBEDDING_DIMENSION
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_synchronous_ingestion_backward_compatibility(async_client: AsyncClient) -> None:
    """Verify that default ingestion (async_mode=False) completes synchronously with 201 Created."""
    pdf_path = "data/raw_corpus/nvidia_10k_sample.pdf"
    assert os.path.exists(pdf_path)

    # Use unique content so test is isolated and independent of database state
    with open(pdf_path, "rb") as f:
        content = f.read() + f"\n% sync-{uuid4()}\n".encode()

    filename = f"nvidia_sync_{uuid4().hex[:8]}.pdf"
    resp = await async_client.post(
        "/api/v1/documents/ingest",
        files={"file": (filename, content, "application/pdf")},
        data={"category": "test_sync"},
    )
    assert resp.status_code == 201
    doc = resp.json()
    assert doc["status"] == "COMPLETED"
    assert doc["processing_stage"] == "COMPLETED"
    assert doc["total_chunks"] > 0

    # Cleanup
    await async_client.delete(f"/api/v1/documents/{doc['id']}")


@pytest.mark.asyncio
async def test_asynchronous_ingestion_dispatch_and_polling(async_client: AsyncClient) -> None:
    """Verify async_mode=True returns 202, stages upload, and tracks via GET /tasks/{task_id}."""
    pdf_path = "data/raw_corpus/nvidia_10k_sample.pdf"
    assert os.path.exists(pdf_path)

    with open(pdf_path, "rb") as f:
        content = f.read() + f"\n% async-{uuid4()}\n".encode()

    filename = f"test_async_{uuid4().hex[:8]}.pdf"
    fake_task_id = f"task-{uuid4()}"

    # Mock ingest_document_task.delay so worker daemon is not required during test
    mock_async_result = MagicMock()
    mock_async_result.id = fake_task_id

    patch_target = "app.api.v1.endpoints.documents.ingest_document_task.delay"
    with patch(patch_target, return_value=mock_async_result) as mock_delay:
        resp = await async_client.post(
            "/api/v1/documents/ingest?async_mode=true",
            files={"file": (filename, content, "application/pdf")},
            data={"category": "test_async"},
        )

        assert resp.status_code == 202
        data = resp.json()
        assert data["task_id"] == fake_task_id
        assert data["status"] == "PENDING"
        assert data["processing_stage"] == "QUEUED"
        doc_id = data["id"]
        mock_delay.assert_called_once()

    # Verify task polling endpoint GET /api/v1/documents/tasks/{task_id}
    try:
        poll_resp = await async_client.get(f"/api/v1/documents/tasks/{fake_task_id}")
        assert poll_resp.status_code == 200
        poll_data = poll_resp.json()
        assert poll_data["task_id"] == fake_task_id
        assert poll_data["document_id"] == doc_id
        assert poll_data["status"] == "PENDING"
        assert poll_data["processing_stage"] == "QUEUED"

        # Also test /api/v1/tasks/{task_id} alias
        alias_resp = await async_client.get(f"/api/v1/tasks/{fake_task_id}")
        assert alias_resp.status_code == 200
        assert alias_resp.json()["task_id"] == fake_task_id

        # Now execute the task directly to simulate Celery worker completion
        staged_path = mock_delay.call_args[0][1]
        assert os.path.exists(staged_path)

        # Run Celery task function with DummyEmbeddingProvider
        with patch(
            "app.services.ingestion.pipeline.get_embedding_provider",
            return_value=DummyEmbeddingProvider(settings.EMBEDDING_DIMENSION),
        ):
            result = ingest_document_task(doc_id, staged_path)

        assert result["status"] == "COMPLETED"
        assert result["processing_stage"] == "COMPLETED"
        assert result["chunk_count"] > 0

        # Verify polling endpoint reflects COMPLETED
        completed_resp = await async_client.get(f"/api/v1/documents/tasks/{fake_task_id}")
        assert completed_resp.status_code == 200
        completed_data = completed_resp.json()
        assert completed_data["status"] == "COMPLETED"
        assert completed_data["processing_stage"] == "COMPLETED"
        assert completed_data["chunk_count"] > 0
    finally:
        # Cleanup document
        await async_client.delete(f"/api/v1/documents/{doc_id}")


@pytest.mark.asyncio
async def test_task_polling_nonexistent_task(async_client: AsyncClient) -> None:
    """Verify task polling for an unknown task ID returns 404."""
    unknown_task_id = f"unknown-task-{uuid4()}"
    resp = await async_client.get(f"/api/v1/documents/tasks/{unknown_task_id}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_async_ingestion_worker_failure_state(async_client: AsyncClient) -> None:
    """Verify that when Celery worker fails, polling returns status FAILED with error details."""
    pdf_path = "data/raw_corpus/nvidia_10k_sample.pdf"
    assert os.path.exists(pdf_path)

    with open(pdf_path, "rb") as f:
        content = f.read() + f"\n% async-fail-{uuid4()}\n".encode()

    filename = f"test_async_fail_{uuid4().hex[:8]}.pdf"
    fake_task_id = f"task-{uuid4()}"

    mock_async_result = MagicMock()
    mock_async_result.id = fake_task_id

    patch_target = "app.api.v1.endpoints.documents.ingest_document_task.delay"
    with patch(patch_target, return_value=mock_async_result):
        resp = await async_client.post(
            "/api/v1/documents/ingest?async_mode=true",
            files={"file": (filename, content, "application/pdf")},
            data={"category": "test_async_fail"},
        )
        assert resp.status_code == 202
        doc_id = resp.json()["id"]

    try:
        # Mock Celery AsyncResult returning FAILURE state before DB update
        mock_celery_result = MagicMock()
        mock_celery_result.state = "FAILURE"
        mock_celery_result.result = RuntimeError("Worker out of memory")

        with patch("app.api.v1.endpoints.documents.AsyncResult", return_value=mock_celery_result):
            poll_resp = await async_client.get(f"/api/v1/documents/tasks/{fake_task_id}")
            assert poll_resp.status_code == 200
            poll_data = poll_resp.json()
            assert poll_data["task_id"] == fake_task_id
            assert poll_data["status"] == "FAILED"
            assert poll_data["processing_stage"] == "FAILED"
            assert "Worker out of memory" in poll_data["error"]
    finally:
        await async_client.delete(f"/api/v1/documents/{doc_id}")


@pytest.mark.asyncio
async def test_async_ingestion_dispatch_failure_resilience(async_client: AsyncClient) -> None:
    """Verify dispatch exception marks document FAILED and allows subsequent re-ingestion."""
    pdf_path = "data/raw_corpus/nvidia_10k_sample.pdf"
    assert os.path.exists(pdf_path)

    with open(pdf_path, "rb") as f:
        content = f.read() + f"\n% dispatch-fail-{uuid4()}\n".encode()

    filename = f"test_dispatch_fail_{uuid4().hex[:8]}.pdf"

    patch_target = "app.api.v1.endpoints.documents.ingest_document_task.delay"
    with patch(patch_target, side_effect=RuntimeError("Redis connection refused")):
        resp = await async_client.post(
            "/api/v1/documents/ingest?async_mode=true",
            files={"file": (filename, content, "application/pdf")},
            data={"category": "test_dispatch_fail"},
        )
        assert resp.status_code == 500
        assert "Failed to dispatch background ingestion" in resp.json()["detail"]

    # Verify doc in DB is marked FAILED, so re-ingestion does NOT return 409 Conflict
    fake_task_id = f"task-recovered-{uuid4()}"
    mock_async_result = MagicMock()
    mock_async_result.id = fake_task_id

    with patch(patch_target, return_value=mock_async_result):
        retry_resp = await async_client.post(
            "/api/v1/documents/ingest?async_mode=true",
            files={"file": (filename, content, "application/pdf")},
            data={"category": "test_dispatch_fail"},
        )
        assert retry_resp.status_code == 202
        recovered_doc_id = retry_resp.json()["id"]

    await async_client.delete(f"/api/v1/documents/{recovered_doc_id}")

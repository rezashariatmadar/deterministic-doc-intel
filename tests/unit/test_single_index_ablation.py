"""Unit tests for single-index ablation parameter in VectorSearchService."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.services.retrieval.vector_search import VectorSearchService


@pytest.fixture
def mock_session() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_embedding_provider() -> AsyncMock:
    provider = AsyncMock()
    provider.embed_query.return_value = [0.1] * 1536
    return provider


def create_mock_child_chunk(parent_id: Any = None) -> MagicMock:
    chunk_id = uuid4()
    p_id = parent_id or uuid4()
    doc_id = uuid4()

    chunk = MagicMock()
    chunk.id = chunk_id
    chunk.parent_chunk_id = p_id
    chunk.document_id = doc_id
    chunk.raw_text = "Child chunk text with exact detail."
    chunk.contextualized_text = "Child contextualized text."
    chunk.page_start = 12
    chunk.page_end = 12
    chunk.section_path = "Item 7 > Overview"
    chunk.element_type = "prose"
    chunk.token_count = 15
    chunk.chunk_metadata = {}
    chunk.metadata = {}
    chunk.document = MagicMock()
    chunk.document.title = "Apple_10-K-Q4-2023-As-Filed.pdf"
    return chunk


def create_mock_parent_chunk(parent_id: Any) -> MagicMock:
    parent = MagicMock()
    parent.id = parent_id
    parent.raw_text = "Parent expanded broad context encompassing pages 11 to 13."
    parent.contextualized_text = "Parent contextualized broad text."
    parent.page_start = 11
    parent.page_end = 13
    parent.section_path = "Item 7 > Itemized Discussion"
    parent.token_count = 60
    return parent


@pytest.mark.asyncio
async def test_single_index_ablation_resolve_parents_false(
    mock_session: AsyncMock,
    mock_embedding_provider: AsyncMock,
) -> None:
    """When resolve_parents=False, search returns child chunks without parent resolution."""
    parent_id = uuid4()
    child = create_mock_child_chunk(parent_id=parent_id)

    service = VectorSearchService(
        session=mock_session,
        embedding_provider=mock_embedding_provider,
        reranker=None,
    )
    # Mock chunk repository methods
    service.chunk_repo = MagicMock()
    service.chunk_repo.search_similar = AsyncMock(return_value=[(child, 0.95)])
    service.chunk_repo.get_parents_by_ids = AsyncMock()

    results = await service.search(
        query="What was the revenue?",
        search_mode="dense",
        top_k=5,
        resolve_parents=False,
    )

    assert len(results) == 1
    # Should keep child chunk properties
    assert results[0].text == "Child chunk text with exact detail."
    assert results[0].page_start == 12
    assert results[0].page_end == 12
    # Parent resolution was skipped
    service.chunk_repo.get_parents_by_ids.assert_not_called()


@pytest.mark.asyncio
async def test_single_index_ablation_resolve_parents_true_default(
    mock_session: AsyncMock,
    mock_embedding_provider: AsyncMock,
) -> None:
    """When resolve_parents=True (or defaulted), search resolves parent chunks."""
    parent_id = uuid4()
    child = create_mock_child_chunk(parent_id=parent_id)
    parent = create_mock_parent_chunk(parent_id=parent_id)

    service = VectorSearchService(
        session=mock_session,
        embedding_provider=mock_embedding_provider,
        reranker=None,
    )
    service.chunk_repo = MagicMock()
    service.chunk_repo.search_similar = AsyncMock(return_value=[(child, 0.95)])
    service.chunk_repo.get_parents_by_ids = AsyncMock(return_value={parent_id: parent})

    results = await service.search(
        query="What was the revenue?",
        search_mode="dense",
        top_k=5,
        resolve_parents=True,
    )

    assert len(results) == 1
    # Text and page range should be expanded to parent's
    assert results[0].text == "Parent expanded broad context encompassing pages 11 to 13."
    assert results[0].page_start == 11
    assert results[0].page_end == 13
    service.chunk_repo.get_parents_by_ids.assert_called_once_with([parent_id])

"""Integration tests for VectorSearchService over real PostgreSQL and pgvector."""

import hashlib
import uuid
from collections.abc import AsyncGenerator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.repositories.chunk_repo import ChunkRepository
from app.db.repositories.document_repo import DocumentRepository
from app.db.session import async_session_maker
from app.schemas.document import ChunkCreate, DocumentCreate
from app.schemas.query import QueryFilters
from app.services.providers.base import EmbeddingProvider
from app.services.retrieval.vector_search import VectorSearchService

settings = get_settings()


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic mock embedding provider for tests."""

    def __init__(self, query_vec: list[float]) -> None:
        super().__init__(model_name="mock-model", dimension=len(query_vec))
        self.query_vec = query_vec

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.query_vec for _ in texts]

    async def embed_query(self, query: str) -> list[float]:
        return self.query_vec


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session


@pytest.mark.asyncio
async def test_vector_search_executes_and_filters(db_session: AsyncSession) -> None:
    doc_repo = DocumentRepository(db_session)
    chunk_repo = ChunkRepository(db_session)

    dim = settings.EMBEDDING_DIMENSION
    # Query vector: unit vector along dimension 0
    query_vec = [1.0] + [0.0] * (dim - 1)
    # Similar vector: [1.0, 0.0, ...] -> cosine similarity 1.0
    similar_vec = [1.0] + [0.0] * (dim - 1)
    # Orthogonal vector: [0.0, 1.0, ...] -> cosine similarity 0.0
    dissimilar_vec = [0.0, 1.0] + [0.0] * (dim - 2)

    unique_hash = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
    doc = await doc_repo.create(
        DocumentCreate(
            title="NVIDIA 10-K Fiscal 2024",
            filename="nvidia_sample.pdf",
            file_hash=unique_hash,
            total_pages=2,
            category="financial_report",
        )
    )

    try:
        await chunk_repo.bulk_create(
            [
                ChunkCreate(
                    document_id=doc.id,
                    chunk_index=0,
                    raw_text="Data Center revenue rose 217% to $47.5B in fiscal 2024.",
                    contextualized_text="[Doc: NVIDIA 10-K] Data Center rose 217%.",
                    token_count=18,
                    page_start=45,
                    page_end=45,
                    section_path="Item 7 > Financial Results",
                    element_type="prose",
                    embedding_model=settings.EMBEDDING_MODEL,
                    embedding=similar_vec,
                ),
                ChunkCreate(
                    document_id=doc.id,
                    chunk_index=1,
                    raw_text="The board consists of 12 directors serving annual terms.",
                    contextualized_text="[Doc: NVIDIA 10-K] Board structure details.",
                    token_count=10,
                    page_start=80,
                    page_end=80,
                    section_path="Item 10 > Governance",
                    element_type="table",
                    embedding_model=settings.EMBEDDING_MODEL,
                    embedding=dissimilar_vec,
                ),
            ]
        )

        provider = MockEmbeddingProvider(query_vec=query_vec)
        service = VectorSearchService(session=db_session, embedding_provider=provider)

        # 1. Search with similarity threshold 0.35 (should only retrieve the similar chunk)
        results = await service.search(
            query="What was Data Center revenue?",
            filters=QueryFilters(min_similarity=0.35),
            top_k=5,
        )

        assert len(results) == 1
        top_chunk = results[0]
        assert top_chunk.source_id == 1
        assert top_chunk.document_title == "NVIDIA 10-K Fiscal 2024"
        assert top_chunk.similarity_score > 0.95
        assert "Data Center revenue rose 217%" in top_chunk.text
        assert top_chunk.element_type == "prose"

        # 2. Search filtering for element_type "table" (similar chunk is "prose")
        table_results = await service.search(
            query="What was Data Center revenue?",
            filters=QueryFilters(min_similarity=0.35, element_types=["table"]),
            top_k=5,
        )
        assert len(table_results) == 0

    finally:
        await doc_repo.delete(doc.id)

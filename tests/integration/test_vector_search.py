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
                    metadata={
                        "bbox": {"x0": 10.0, "y0": 20.0, "x1": 30.0, "y1": 40.0},
                        "block_index": 3,
                    },
                ),
                ChunkCreate(
                    document_id=doc.id,
                    chunk_index=1,
                    raw_text=(
                        "Overview of Board: "
                        "The board consists of 12 directors serving annual terms."
                    ),
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
        await doc_repo.update_status(doc.id, status="COMPLETED", total_chunks=2)

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
        assert top_chunk.token_count == 18
        assert top_chunk.bbox == {"x0": 10.0, "y0": 20.0, "x1": 30.0, "y1": 40.0}
        assert top_chunk.metadata == {
            "bbox": {"x0": 10.0, "y0": 20.0, "x1": 30.0, "y1": 40.0},
            "block_index": 3,
        }

        # 2. Search filtering for element_type "table" (similar chunk is "prose")
        table_results = await service.search(
            query="What was Data Center revenue?",
            filters=QueryFilters(min_similarity=0.35, element_types=["table"]),
            top_k=5,
        )
        assert len(table_results) == 0

        # 3. Context window budgeting test
        budgeted_results = await service.search(
            query="What was Data Center revenue?",
            filters=QueryFilters(min_similarity=0.35),
            top_k=5,
            max_tokens=5,  # Token limit lower than chunk's 18 tokens
        )
        assert len(budgeted_results) == 0

    finally:
        await doc_repo.delete(doc.id)


@pytest.mark.asyncio
async def test_search_lexical_and_parent_retrieval(db_session: AsyncSession) -> None:
    from app.schemas.document import ParentChunkCreate
    from app.services.retrieval.rrf import reciprocal_rank_fusion

    doc_repo = DocumentRepository(db_session)
    chunk_repo = ChunkRepository(db_session)

    unique_hash = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
    doc = await doc_repo.create(
        DocumentCreate(
            title="Lexical & Parent Integration Doc",
            filename="lexical_doc.pdf",
            file_hash=unique_hash,
            total_pages=2,
            category="financial_report",
        )
    )

    try:
        # Create parent chunk
        parents = await chunk_repo.bulk_create_parents(
            [
                ParentChunkCreate(
                    document_id=doc.id,
                    parent_index=0,
                    raw_text=(
                        "Overview of Board: "
                        "The board consists of 12 directors serving annual terms."
                    ),
                    contextualized_text="[Doc: Lexical] The board consists of 12 directors.",
                    token_count=25,
                    page_start=1,
                    page_end=1,
                    section_path="Item 10 > Governance",
                )
            ]
        )
        assert len(parents) == 1
        parent_id = parents[0].id

        # Create child chunks
        dim = settings.EMBEDDING_DIMENSION
        dummy_vec = [0.1] * dim
        created_chunks = await chunk_repo.bulk_create(
            [
                ChunkCreate(
                    document_id=doc.id,
                    parent_chunk_id=parent_id,
                    chunk_index=0,
                    raw_text=(
                        "Overview of Board: "
                        "The board consists of 12 directors serving annual terms."
                    ),
                    contextualized_text=(
                        "[Doc: Lexical | Section: Item 10 > Governance] "
                        "The board consists of 12 directors."
                    ),
                    token_count=15,
                    page_start=1,
                    page_end=1,
                    section_path="Item 10 > Governance",
                    element_type="prose",
                    embedding_model=settings.EMBEDDING_MODEL,
                    embedding=dummy_vec,
                ),
                ChunkCreate(
                    document_id=doc.id,
                    chunk_index=1,
                    raw_text="Cloud compute infrastructure reached $47.5B in quarterly billing.",
                    contextualized_text=(
                        "[Doc: Lexical | Section: Item 7] "
                        "Cloud compute infrastructure reached $47.5B in quarterly billing."
                    ),
                    token_count=15,
                    page_start=2,
                    page_end=2,
                    section_path="Item 7 > Financial",
                    element_type="prose",
                    embedding_model=settings.EMBEDDING_MODEL,
                    embedding=dummy_vec,
                ),
            ]
        )
        assert len(created_chunks) == 2
        await doc_repo.update_status(doc.id, status="COMPLETED", total_chunks=2)

        # 1. Lexical search for specific term
        lex_res = await chunk_repo.search_lexical(query="directors", top_k=5)
        assert len(lex_res) >= 1
        assert lex_res[0][0].id == created_chunks[0].id
        assert lex_res[0][1] > 0.0

        # 2. Punctuation and symbols test (websearch_to_tsquery resilience)
        punctuated_query = "What about Item 10: (Governance) & 12 directors?"
        punc_res = await chunk_repo.search_lexical(query=punctuated_query, top_k=5)
        assert len(punc_res) >= 1
        assert punc_res[0][0].id == created_chunks[0].id

        # 3. Batch parent resolution
        parent_map = await chunk_repo.get_parents_by_ids([parent_id])
        assert parent_id in parent_map
        assert "Overview of Board" in parent_map[parent_id].raw_text

        # 4. RRF fusion test with actual chunk objects
        dense_list = [(created_chunks[1], 0.9)]
        sparse_list = [(created_chunks[0], 0.8), (created_chunks[1], 0.4)]
        fused = reciprocal_rank_fusion(dense_list, sparse_list)
        assert len(fused) == 2
        # created_chunks[1] is present in both lists
        assert fused[0][0].id == created_chunks[1].id

        # 5. Multi-query / sub-queries retrieval test
        from unittest.mock import AsyncMock

        mock_llm = AsyncMock()
        mock_llm.generate.return_value = (
            '{"semantic_query": "Overview", "lexical_query": "billing", '
            '"sub_queries": ["directors", "quarterly billing"]}'
        )
        provider = MockEmbeddingProvider(query_vec=dummy_vec)
        multi_service = VectorSearchService(
            session=db_session,
            embedding_provider=provider,
            llm_provider=mock_llm,
        )
        sub_results = await multi_service.search(
            query="Tell me about board and billing",
            search_mode="sparse",
            enable_query_rewrite=True,
            top_k=5,
        )
        # Both chunks (one for directors, one for quarterly billing) should be retrieved
        found_ids = {c.chunk_id for c in sub_results}
        assert created_chunks[0].id in found_ids
        assert created_chunks[1].id in found_ids

        fused_ids = [entry[0].id for entry in fused]
        assert created_chunks[1].id in fused_ids
        assert created_chunks[0].id in fused_ids

    finally:
        await doc_repo.delete(doc.id)

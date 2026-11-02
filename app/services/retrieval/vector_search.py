"""Dense vector retrieval service integrating EmbeddingProvider and ChunkRepository."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.repositories.chunk_repo import ChunkRepository
from app.schemas.query import QueryFilters, RetrievedChunk
from app.services.providers.base import EmbeddingProvider


class VectorSearchService:
    """Orchestrates query embedding generation and similarity retrieval over pgvector."""

    def __init__(
        self,
        session: AsyncSession,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self.session = session
        self.embedding_provider = embedding_provider
        self.chunk_repo = ChunkRepository(session)
        self.settings = get_settings()

    async def search(
        self,
        query: str,
        filters: QueryFilters | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Embed the user query and retrieve top-k semantically relevant chunks
        passing all metadata pre-filters and minimum similarity threshold.
        """
        if not query.strip():
            return []

        # 1. Embed query vector
        query_vector = await self.embedding_provider.embed_query(query)

        # 2. Extract filter criteria
        min_similarity = (
            filters.min_similarity
            if filters and filters.min_similarity is not None
            else self.settings.SIMILARITY_THRESHOLD
        )
        doc_id = filters.document_id if filters else None
        category = filters.category if filters else None
        element_types = filters.element_types if filters else None
        page_min = filters.page_min if filters else None
        page_max = filters.page_max if filters else None

        # 3. Execute vector search in PostgreSQL HNSW index
        matched_chunks = await self.chunk_repo.search_similar(
            query_vector=query_vector,
            top_k=top_k,
            threshold=min_similarity,
            document_id=doc_id,
            category=category,
            element_types=element_types,
            page_min=page_min,
            page_max=page_max,
        )

        # 4. Map to enriched RetrievedChunk schemas with 1-based source indices
        results: list[RetrievedChunk] = []
        for idx, (chunk, score) in enumerate(matched_chunks, start=1):
            doc_title = (
                chunk.document.title
                if chunk.document and chunk.document.title
                else "Unknown Document"
            )
            results.append(
                RetrievedChunk(
                    source_id=idx,
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    document_title=doc_title,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section_path=chunk.section_path,
                    element_type=chunk.element_type,
                    text=chunk.raw_text,
                    contextualized_text=chunk.contextualized_text,
                    similarity_score=round(score, 4),
                )
            )

        return results

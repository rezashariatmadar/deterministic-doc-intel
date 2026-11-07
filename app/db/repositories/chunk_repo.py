"""Repository pattern for DocumentChunk database and pgvector search operations."""

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models.chunk import DocumentChunk
from app.db.models.document import Document
from app.db.models.parent_chunk import ParentChunk
from app.schemas.document import ChunkCreate, ParentChunkCreate


class ChunkRepository:
    """Encapsulates database operations for document chunks and vector similarity search."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def bulk_create(self, chunks_in: list[ChunkCreate]) -> list[DocumentChunk]:
        """Atomically insert a batch of chunks into PostgreSQL."""
        records = [
            DocumentChunk(
                document_id=c.document_id,
                parent_chunk_id=c.parent_chunk_id,
                chunk_index=c.chunk_index,
                raw_text=c.raw_text,
                contextualized_text=c.contextualized_text,
                token_count=c.token_count,
                page_start=c.page_start,
                page_end=c.page_end,
                section_path=c.section_path,
                element_type=c.element_type,
                embedding_model=c.embedding_model,
                embedding=c.embedding,
                chunk_metadata=c.metadata,
            )
            for c in chunks_in
        ]
        if records:
            self.session.add_all(records)
            await self.session.commit()
        return records

    async def bulk_create_parents(self, parents_in: list[ParentChunkCreate]) -> list[ParentChunk]:
        """Atomically insert a batch of parent chunks into PostgreSQL."""
        records = [
            ParentChunk(
                document_id=p.document_id,
                parent_index=p.parent_index,
                raw_text=p.raw_text,
                contextualized_text=p.contextualized_text,
                token_count=p.token_count,
                page_start=p.page_start,
                page_end=p.page_end,
                section_path=p.section_path,
                chunk_metadata=p.metadata,
            )
            for p in parents_in
        ]
        if records:
            self.session.add_all(records)
            await self.session.commit()
            for r in records:
                await self.session.refresh(r)
        return records

    async def search_similar(
        self,
        query_vector: list[float],
        top_k: int = 5,
        threshold: float = 0.35,
        document_id: UUID | None = None,
        category: str | None = None,
        element_types: list[str] | None = None,
        page_min: int | None = None,
        page_max: int | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Execute cosine similarity search over pgvector HNSW index.
        Returns list of tuples: (DocumentChunk, similarity_score) sorted descending by similarity.
        Cosine similarity = 1.0 - cosine_distance.
        """
        cosine_distance = DocumentChunk.embedding.cosine_distance(query_vector)
        similarity_score = (1.0 - cosine_distance).label("similarity_score")

        stmt = (
            select(DocumentChunk, similarity_score)
            .join(Document, DocumentChunk.document_id == Document.id)
            .options(selectinload(DocumentChunk.document))
            .where(Document.status == "COMPLETED")
        )

        if category:
            stmt = stmt.where(Document.category == category)

        if document_id:
            stmt = stmt.where(DocumentChunk.document_id == document_id)

        if element_types:
            stmt = stmt.where(DocumentChunk.element_type.in_(element_types))

        if page_min is not None:
            stmt = stmt.where(DocumentChunk.page_start >= page_min)

        if page_max is not None:
            stmt = stmt.where(DocumentChunk.page_end <= page_max)

        # Filter by similarity threshold floor
        stmt = stmt.where(similarity_score >= threshold)

        # Order by closest distance (highest similarity first)
        stmt = stmt.order_by(cosine_distance.asc()).limit(top_k)

        result = await self.session.execute(stmt)
        rows = result.all()
        return [(row[0], float(row[1])) for row in rows]

    async def search_lexical(
        self,
        query: str,
        top_k: int = 30,
        document_id: UUID | None = None,
        category: str | None = None,
        element_types: list[str] | None = None,
        page_min: int | None = None,
        page_max: int | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Execute full-text lexical search using PostgreSQL TSVECTOR, GIN index, and ts_rank_cd.
        Uses func.websearch_to_tsquery('english', query) for robust syntax handling.
        """
        if not query or not query.strip():
            return []

        ts_query = func.websearch_to_tsquery("english", query)
        lexical_score = func.ts_rank_cd(DocumentChunk.search_vector, ts_query, 32).label(
            "lexical_score"
        )

        stmt = (
            select(DocumentChunk, lexical_score)
            .join(Document, DocumentChunk.document_id == Document.id)
            .options(selectinload(DocumentChunk.document))
            .where(Document.status == "COMPLETED")
            .where(DocumentChunk.search_vector.op("@@")(ts_query))
        )

        if category:
            stmt = stmt.where(Document.category == category)

        if document_id:
            stmt = stmt.where(DocumentChunk.document_id == document_id)

        if element_types:
            stmt = stmt.where(DocumentChunk.element_type.in_(element_types))

        if page_min is not None:
            stmt = stmt.where(DocumentChunk.page_start >= page_min)

        if page_max is not None:
            stmt = stmt.where(DocumentChunk.page_end <= page_max)

        stmt = stmt.order_by(lexical_score.desc()).limit(top_k)

        result = await self.session.execute(stmt)
        rows = result.all()
        return [(row[0], float(row[1])) for row in rows]

    async def get_parents_by_ids(self, parent_ids: list[UUID]) -> dict[UUID, ParentChunk]:
        """Fetch parent chunks in batch by their UUIDs."""
        if not parent_ids:
            return {}
        stmt = select(ParentChunk).where(ParentChunk.id.in_(parent_ids))
        result = await self.session.execute(stmt)
        return {p.id: p for p in result.scalars().all()}

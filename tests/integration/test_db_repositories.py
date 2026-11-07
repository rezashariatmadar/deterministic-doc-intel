"""Integration tests for Document and Chunk repositories with pgvector."""

import uuid

import pytest

from app.core.config import get_settings
from app.db.repositories.chunk_repo import ChunkRepository
from app.db.repositories.document_repo import DocumentRepository
from app.schemas.document import ChunkCreate, DocumentCreate


@pytest.mark.asyncio
async def test_document_and_chunk_lifecycle(db_session):
    settings = get_settings()
    doc_repo = DocumentRepository(db_session)
    chunk_repo = ChunkRepository(db_session)

    # 1. Create document with unique hash for test isolation
    unique_hash = (uuid.uuid4().hex + uuid.uuid4().hex)[:64]
    doc = await doc_repo.create(
        DocumentCreate(
            filename="test_nvidia_10k.pdf",
            file_hash=unique_hash,
            title="NVIDIA 10-K Report 2024",
            total_pages=2,
        )
    )
    assert doc.id is not None
    assert doc.status == "PENDING"

    try:
        # 2. Verify hash lookup (idempotency check)
        existing = await doc_repo.get_by_hash(unique_hash)
        assert existing is not None
        assert existing.id == doc.id

        # 3. Create distinct embedding vectors with different directions
        dim = settings.EMBEDDING_DIMENSION
        half = dim // 2
        vec1 = [1.0] * half + [0.0] * (dim - half)
        vec2 = [0.0] * half + [1.0] * (dim - half)

        chunks = await chunk_repo.bulk_create(
            [
                ChunkCreate(
                    document_id=doc.id,
                    chunk_index=0,
                    raw_text="Data Center revenue rose 217% to $47.5B in fiscal 2024.",
                    contextualized_text=(
                        "[Doc: NVIDIA 10-K | Section: Item 7] Data Center rose 217%."
                    ),
                    token_count=18,
                    page_start=45,
                    page_end=45,
                    section_path="Item 7. MD&A > Financial Results",
                    element_type="prose",
                    embedding_model=settings.EMBEDDING_MODEL,
                    embedding=vec1,
                ),
                ChunkCreate(
                    document_id=doc.id,
                    chunk_index=1,
                    raw_text="Gaming revenue was $10.4B, up 15% due to GeForce RTX 40 Series.",
                    contextualized_text=(
                        "[Doc: NVIDIA 10-K | Section: Item 7 | Page: 46] Gaming revenue was $10.4B."
                    ),
                    token_count=16,
                    page_start=46,
                    page_end=46,
                    section_path="Item 7. MD&A > Gaming",
                    element_type="prose",
                    embedding_model=settings.EMBEDDING_MODEL,
                    embedding=vec2,
                ),
            ]
        )
        assert len(chunks) == 2

        await doc_repo.update_status(doc.id, status="COMPLETED", total_chunks=2)

        # 4. Search by similarity (cosine distance)
        # Query vector points in the same direction as vec1
        query_vec = [1.0] * half + [0.0] * (dim - half)
        results = await chunk_repo.search_similar(query_vector=query_vec, top_k=5, threshold=0.35)
        assert len(results) >= 1
        top_chunk, score = results[0]
        assert top_chunk.id == chunks[0].id
        assert score >= 0.35
    finally:
        # 5. Clean up
        await doc_repo.delete(doc.id)
        deleted = await doc_repo.get_by_id(doc.id)
        assert deleted is None


@pytest.mark.asyncio
async def test_parent_chunk_and_search_vector_generation(db_session):
    from sqlalchemy import select

    from app.db.models.chunk import DocumentChunk
    from app.db.models.parent_chunk import ParentChunk

    settings = get_settings()
    doc_repo = DocumentRepository(db_session)
    chunk_repo = ChunkRepository(db_session)

    unique_hash = (uuid.uuid4().hex + uuid.uuid4().hex)[:64]
    doc = await doc_repo.create(
        DocumentCreate(
            filename="test_hierarchical.pdf",
            file_hash=unique_hash,
            title="NVIDIA 10-K Hierarchical",
            total_pages=5,
        )
    )

    try:
        # 1. Insert parent chunk
        parent = ParentChunk(
            document_id=doc.id,
            parent_index=0,
            raw_text="Full Section: Data Center revenue surged due to accelerated computing demand.",
            contextualized_text="[Doc: NVIDIA 10-K] Full Section: Data Center revenue surged.",
            token_count=100,
            page_start=1,
            page_end=2,
            section_path="Item 7 > Overview",
            chunk_metadata={"section": "overview"},
        )
        db_session.add(parent)
        await db_session.commit()
        await db_session.refresh(parent)
        assert parent.id is not None

        # 2. Insert child chunk linked to parent_chunk_id
        dim = settings.EMBEDDING_DIMENSION
        vec = [0.1] * dim
        child_chunks = await chunk_repo.bulk_create(
            [
                ChunkCreate(
                    document_id=doc.id,
                    chunk_index=0,
                    parent_chunk_id=parent.id,
                    raw_text="Data Center revenue surged.",
                    contextualized_text="[Doc: NVIDIA 10-K] Data Center revenue surged.",
                    token_count=10,
                    page_start=1,
                    page_end=1,
                    section_path="Item 7 > Overview",
                    element_type="prose",
                    embedding_model=settings.EMBEDDING_MODEL,
                    embedding=vec,
                )
            ]
        )
        assert len(child_chunks) == 1
        assert child_chunks[0].parent_chunk_id == parent.id

        # 3. Verify search_vector was generated automatically in PostgreSQL
        stmt = select(DocumentChunk).where(DocumentChunk.id == child_chunks[0].id)
        result = await db_session.execute(stmt)
        saved_chunk = result.scalar_one()

        assert saved_chunk.search_vector is not None
        # TSVECTOR string representation in Postgres contains stemmed lexemes like 'center', 'revenu', 'surg'
        sv_str = str(saved_chunk.search_vector)
        assert "center" in sv_str
        assert "revenu" in sv_str

    finally:
        await doc_repo.delete(doc.id)
        # Verify cascade deletion of parent chunk
        parent_stmt = select(ParentChunk).where(ParentChunk.id == parent.id)
        parent_res = await db_session.execute(parent_stmt)
        assert parent_res.scalar_one_or_none() is None

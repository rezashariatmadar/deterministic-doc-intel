"""Document ingestion pipeline processing."""

import logging
from pathlib import Path
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.exceptions import DocumentNotFoundError
from app.db.models.document import Document
from app.db.repositories.chunk_repo import ChunkRepository
from app.db.repositories.document_repo import DocumentRepository
from app.schemas.document import ChunkCreate, ParentChunkCreate
from app.services.ingestion.chunker import DocumentChunker
from app.services.ingestion.parser import PDFLayoutParser
from app.services.providers.base import EmbeddingProvider
from app.services.providers.factory import get_embedding_provider

logger = logging.getLogger(__name__)


async def process_document_ingestion(
    document_id: UUID,
    file_path: Path,
    session: AsyncSession,
    task_id: str | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> Document:
    """Execute complete layout-aware parsing, chunking, embedding, and indexing.

    Transitions document state across stages:
    PARSING -> CHUNKING -> EMBEDDING -> INDEXING -> COMPLETED (or FAILED).
    """
    doc_repo = DocumentRepository(session)
    chunk_repo = ChunkRepository(session)

    doc = await doc_repo.get_by_id(document_id)
    if not doc:
        raise DocumentNotFoundError(f"Document with ID {document_id} not found.")

    # 1. Transition to PROCESSING / PARSING
    await doc_repo.update_status(
        doc_id=doc.id,
        status="PROCESSING",
        processing_stage="PARSING",
        task_id=task_id,
    )

    try:
        # 2. Stage: PARSING
        parser = PDFLayoutParser()
        parsed_doc = parser.parse(file_path)

        doc_title = parsed_doc.title or doc.title or doc.filename
        total_pages = parsed_doc.total_pages

        table_count = sum(1 for el in parsed_doc.elements if el.element_type == "table")
        heading_count = sum(1 for el in parsed_doc.elements if el.element_type == "heading")
        extra_metadata = dict(doc.extra_metadata or {})
        extra_metadata.update(
            {
                "table_count": table_count,
                "heading_count": heading_count,
                "total_structural_elements": len(parsed_doc.elements),
                "parser": "PDFLayoutParser",
            }
        )

        # 3. Stage: CHUNKING
        await doc_repo.update_status(
            doc_id=doc.id,
            status="PROCESSING",
            processing_stage="CHUNKING",
            title=doc_title,
            total_pages=total_pages,
            extra_metadata=extra_metadata,
        )

        settings = get_settings()
        chunker = DocumentChunker(
            target_tokens=settings.CHILD_CHUNK_SIZE,
            overlap_tokens=settings.CHILD_CHUNK_OVERLAP,
        )
        hierarchical_result = chunker.chunk_document_hierarchical(
            parsed_doc,
            parent_tokens=settings.PARENT_CHUNK_SIZE,
            child_tokens=settings.CHILD_CHUNK_SIZE,
            child_overlap=settings.CHILD_CHUNK_OVERLAP,
        )

        parent_id_map: dict[int, UUID] = {}
        if hierarchical_result.parents:
            parent_creates = [
                ParentChunkCreate(
                    document_id=doc.id,
                    parent_index=p_idx,
                    raw_text=p.raw_text,
                    contextualized_text=p.contextualized_text,
                    token_count=p.token_count,
                    page_start=p.page_start,
                    page_end=p.page_end,
                    section_path=p.section_path,
                    metadata=p.metadata,
                )
                for p_idx, p in enumerate(hierarchical_result.parents)
            ]
            saved_parents = await chunk_repo.bulk_create_parents(parent_creates)
            parent_id_map = {p.parent_index: p.id for p in saved_parents}

        child_drafts = [c for _, c in hierarchical_result.children]

        # 4. Stage: EMBEDDING
        await doc_repo.update_status(
            doc_id=doc.id,
            status="PROCESSING",
            processing_stage="EMBEDDING",
        )

        provider = embedding_provider or get_embedding_provider()
        embeddings: list[list[float]] = []

        if child_drafts:
            texts_to_embed = [c.contextualized_text for c in child_drafts]
            batch_size = settings.EMBEDDING_BATCH_SIZE

            for i in range(0, len(texts_to_embed), batch_size):
                batch = texts_to_embed[i : i + batch_size]
                vectors = await provider.embed_texts(batch)
                embeddings.extend(vectors)

        # 5. Stage: INDEXING
        await doc_repo.update_status(
            doc_id=doc.id,
            status="PROCESSING",
            processing_stage="INDEXING",
        )

        if child_drafts:
            chunk_creates = [
                ChunkCreate(
                    document_id=doc.id,
                    parent_chunk_id=parent_id_map.get(p_idx),
                    chunk_index=draft.chunk_index,
                    raw_text=draft.raw_text,
                    contextualized_text=draft.contextualized_text,
                    token_count=draft.token_count,
                    page_start=draft.page_start,
                    page_end=draft.page_end,
                    section_path=draft.section_path,
                    element_type=draft.element_type,
                    embedding_model=provider.model_name,
                    embedding=embeddings[idx],
                    metadata=draft.metadata,
                )
                for idx, (p_idx, draft) in enumerate(hierarchical_result.children)
            ]
            await chunk_repo.bulk_create(chunk_creates)

        # 6. Stage: COMPLETED
        updated_doc = await doc_repo.update_status(
            doc_id=doc.id,
            status="COMPLETED",
            processing_stage="COMPLETED",
            total_chunks=len(child_drafts),
        )
        assert updated_doc is not None
        return updated_doc

    except Exception as exc:
        logger.exception("Document ingestion failed for %s: %s", document_id, exc)
        try:
            await session.rollback()
            await doc_repo.update_status(
                doc_id=document_id,
                status="FAILED",
                processing_stage="FAILED",
                error_message=str(exc),
            )
        except Exception as inner_exc:
            logger.exception("Failed to mark document %s as FAILED: %s", document_id, inner_exc)
        raise

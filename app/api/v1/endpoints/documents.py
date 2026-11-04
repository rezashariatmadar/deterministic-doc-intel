"""Document ingestion, retrieval, and management endpoints."""

import hashlib
import os
import tempfile
from pathlib import Path
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db, get_embedding
from app.core.config import get_settings
from app.core.exceptions import AppException, DocumentNotFoundError
from app.db.repositories.chunk_repo import ChunkRepository
from app.db.repositories.document_repo import DocumentRepository
from app.schemas.document import ChunkCreate, DocumentCreate, DocumentRead
from app.services.ingestion.chunker import DocumentChunker
from app.services.ingestion.parser import PDFLayoutParser
from app.services.providers.base import EmbeddingProvider

router = APIRouter()


@router.post(
    "/documents/ingest",
    response_model=DocumentRead,
    status_code=status.HTTP_201_CREATED,
)
async def ingest_document(
    response: Response,
    file: UploadFile = File(...),
    category: str | None = Form(default=None),
    db: AsyncSession = Depends(get_db),
    embedding_provider: EmbeddingProvider = Depends(get_embedding),
) -> DocumentRead:
    """
    Ingest a PDF document idempotently.
    Parses layout and tables, chunks with contextual breadcrumbs, embeds via pgvector.
    If the document SHA-256 hash was already indexed, returns the existing document.
    """
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    file_hash = hashlib.sha256(content).hexdigest()
    doc_repo = DocumentRepository(db)
    chunk_repo = ChunkRepository(db)

    # 1. Idempotency check: return existing document if already ingested
    existing_doc = await doc_repo.get_by_hash(file_hash)
    if existing_doc:
        if existing_doc.status == "COMPLETED":
            response.status_code = status.HTTP_200_OK
            return DocumentRead.model_validate(existing_doc)
        # Previous attempt failed or was incomplete; purge it to reprocess cleanly
        await doc_repo.delete(existing_doc.id)

    # 2. Save temporary file to run layout-aware parser
    filename = file.filename or "uploaded_document.pdf"
    suffix = os.path.splitext(filename)[1] or ".pdf"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(content)
        tmp_path = tmp_file.name

    doc = None
    try:
        parser = PDFLayoutParser()
        parsed_doc = parser.parse(Path(tmp_path))

        doc_title = parsed_doc.title or filename
        total_pages = parsed_doc.total_pages

        # Extract document-level structural metadata from parser
        table_count = sum(1 for el in parsed_doc.elements if el.element_type == "table")
        heading_count = sum(1 for el in parsed_doc.elements if el.element_type == "heading")
        extra_metadata = {
            "table_count": table_count,
            "heading_count": heading_count,
            "total_structural_elements": len(parsed_doc.elements),
            "parser": "PDFLayoutParser",
        }

        # 3. Create Document record in DB
        doc = await doc_repo.create(
            DocumentCreate(
                filename=filename,
                title=doc_title,
                source_type="pdf",
                category=category,
                file_hash=file_hash,
                total_pages=total_pages,
                extra_metadata=extra_metadata,
            )
        )

        # 4. Generate structural chunks
        settings = get_settings()
        chunker = DocumentChunker(
            target_tokens=settings.CHUNK_SIZE,
            overlap_tokens=settings.CHUNK_OVERLAP,
        )
        draft_chunks = chunker.chunk_document(parsed_doc)

        if draft_chunks:
            # 5. Compute vector embeddings in batches
            texts_to_embed = [c.contextualized_text for c in draft_chunks]
            batch_size = settings.EMBEDDING_BATCH_SIZE
            embeddings: list[list[float]] = []

            for i in range(0, len(texts_to_embed), batch_size):
                batch = texts_to_embed[i : i + batch_size]
                vectors = await embedding_provider.embed_texts(batch)
                embeddings.extend(vectors)

            # 6. Bulk insert chunks into PostgreSQL
            chunk_creates = [
                ChunkCreate(
                    document_id=doc.id,
                    chunk_index=draft.chunk_index,
                    raw_text=draft.raw_text,
                    contextualized_text=draft.contextualized_text,
                    token_count=draft.token_count,
                    page_start=draft.page_start,
                    page_end=draft.page_end,
                    section_path=draft.section_path,
                    element_type=draft.element_type,
                    embedding_model=embedding_provider.model_name,
                    embedding=embeddings[idx],
                    metadata=draft.metadata,
                )
                for idx, draft in enumerate(draft_chunks)
            ]
            await chunk_repo.bulk_create(chunk_creates)

        # 7. Update document status to COMPLETED
        updated_doc = await doc_repo.update_status(
            doc_id=doc.id,
            status="COMPLETED",
            total_chunks=len(draft_chunks),
        )

        assert updated_doc is not None
        return DocumentRead.model_validate(updated_doc)

    except AppException as exc:
        if doc is not None:
            await doc_repo.update_status(
                doc_id=doc.id,
                status="FAILED",
                error_message=str(exc),
            )
        raise HTTPException(
            status_code=exc.status_code,
            detail=f"Document ingestion failed: {exc.message}",
        ) from exc
    except Exception as exc:
        if doc is not None:
            await doc_repo.update_status(
                doc_id=doc.id,
                status="FAILED",
                error_message=str(exc),
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {exc}",
        ) from exc
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.get("/documents", response_model=list[DocumentRead])
async def list_documents(
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
) -> list[DocumentRead]:
    """List indexed documents with pagination."""
    doc_repo = DocumentRepository(db)
    docs = await doc_repo.list_all(skip=skip, limit=limit)
    return [DocumentRead.model_validate(d) for d in docs]


@router.get("/documents/{document_id}", response_model=DocumentRead)
async def get_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> DocumentRead:
    """Retrieve details of a specific document by UUID."""
    doc_repo = DocumentRepository(db)
    doc = await doc_repo.get_by_id(document_id)
    if not doc:
        raise DocumentNotFoundError(f"Document with ID {document_id} not found.")
    return DocumentRead.model_validate(doc)


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a document and its cascading chunks from PostgreSQL."""
    doc_repo = DocumentRepository(db)
    deleted = await doc_repo.delete(document_id)
    if not deleted:
        raise DocumentNotFoundError(f"Document with ID {document_id} not found.")

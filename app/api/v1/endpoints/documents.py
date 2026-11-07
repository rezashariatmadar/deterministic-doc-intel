"""Document ingestion, retrieval, and management endpoints."""

import hashlib
import os
import tempfile
from pathlib import Path
from uuid import UUID

from celery.result import AsyncResult
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db, get_embedding, get_semantic_cache
from app.core.exceptions import AppException, DocumentNotFoundError
from app.db.repositories.document_repo import DocumentRepository
from app.schemas.document import (
    DocumentCreate,
    DocumentRead,
    TaskStatusResponse,
)
from app.services.cache.semantic_cache import SemanticCacheService
from app.services.ingestion.pipeline import process_document_ingestion
from app.services.providers.base import EmbeddingProvider
from app.workers.celery_app import celery_app
from app.workers.tasks import ingest_document_task

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
    async_mode: bool = Query(default=False),
    db: AsyncSession = Depends(get_db),
    embedding_provider: EmbeddingProvider = Depends(get_embedding),
    semantic_cache: SemanticCacheService | None = Depends(get_semantic_cache),
) -> DocumentRead:
    """
    Ingest a PDF document idempotently.
    Parses layout and tables, chunks with contextual breadcrumbs, embeds via pgvector.
    Supports synchronous execution (default) or background execution via Celery (async_mode=True).
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

    # 1. Idempotency check: return existing document if already ingested
    existing_doc = await doc_repo.get_by_hash(file_hash)
    if existing_doc:
        if existing_doc.status == "COMPLETED":
            response.status_code = status.HTTP_200_OK
            if existing_doc.processing_stage is None:
                existing_doc.processing_stage = "COMPLETED"
            return DocumentRead.model_validate(existing_doc)
        if existing_doc.status in ("PENDING", "PROCESSING"):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Ingestion is currently in progress (status: {existing_doc.status}).",
            )
        # Previous attempt failed or was incomplete; purge it to reprocess cleanly
        await doc_repo.delete(existing_doc.id)
        if semantic_cache is not None:
            await semantic_cache.invalidate_document(existing_doc.id)

    filename = file.filename or "uploaded_document.pdf"
    suffix = os.path.splitext(filename)[1] or ".pdf"

    if async_mode:
        doc = await doc_repo.create(
            DocumentCreate(
                filename=filename,
                title=filename,
                source_type="pdf",
                category=category,
                file_hash=file_hash,
                total_pages=0,
                processing_stage="QUEUED",
            )
        )
        staging_dir = Path("data/uploads")
        staging_dir.mkdir(parents=True, exist_ok=True)
        staging_path = staging_dir / f"{doc.id}{suffix}"

        try:
            staging_path.write_bytes(content)
            task = ingest_document_task.delay(str(doc.id), str(staging_path))
            updated_doc = await doc_repo.update_status(
                doc_id=doc.id,
                status="PENDING",
                processing_stage="QUEUED",
                task_id=task.id,
            )
            assert updated_doc is not None
            response.status_code = status.HTTP_202_ACCEPTED
            return DocumentRead.model_validate(updated_doc)
        except Exception as exc:
            if staging_path.exists():
                staging_path.unlink()
            await doc_repo.update_status(
                doc_id=doc.id,
                status="FAILED",
                processing_stage="FAILED",
                error_message=str(exc),
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to dispatch background ingestion: {exc}",
            ) from exc

    # Synchronous ingestion (default)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(content)
        tmp_path = Path(tmp_file.name)

    try:
        doc = await doc_repo.create(
            DocumentCreate(
                filename=filename,
                title=filename,
                source_type="pdf",
                category=category,
                file_hash=file_hash,
                total_pages=0,
                processing_stage="PARSING",
            )
        )
        completed_doc = await process_document_ingestion(
            document_id=doc.id,
            file_path=tmp_path,
            session=db,
            task_id=None,
            embedding_provider=embedding_provider,
        )
        return DocumentRead.model_validate(completed_doc)
    except AppException as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=f"Document ingestion failed: {exc.message}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {exc}",
        ) from exc
    finally:
        if tmp_path.exists():
            os.remove(tmp_path)


@router.get(
    "/documents/tasks/{task_id}",
    response_model=TaskStatusResponse,
)
@router.get(
    "/tasks/{task_id}",
    response_model=TaskStatusResponse,
    include_in_schema=False,
)
async def get_task_status(
    task_id: str,
    db: AsyncSession = Depends(get_db),
) -> TaskStatusResponse:
    """Retrieve execution status and stage for a background ingestion task."""
    doc_repo = DocumentRepository(db)
    doc = await doc_repo.get_by_task_id(task_id)

    async_result = AsyncResult(task_id, app=celery_app)

    if doc is not None:
        if doc.status in ("PENDING", "PROCESSING") and async_result.state == "FAILURE":
            return TaskStatusResponse(
                task_id=task_id,
                status="FAILED",
                processing_stage="FAILED",
                document_id=str(doc.id),
                error=doc.error_message or str(async_result.result),
                chunk_count=doc.total_chunks,
                page_count=doc.total_pages,
            )
        return TaskStatusResponse(
            task_id=task_id,
            status=doc.status,
            processing_stage=doc.processing_stage or "QUEUED",
            document_id=str(doc.id),
            error=doc.error_message,
            chunk_count=doc.total_chunks,
            page_count=doc.total_pages,
        )

    if async_result.state == "SUCCESS" and isinstance(async_result.result, dict):
        res = async_result.result
        return TaskStatusResponse(
            task_id=task_id,
            status=res.get("status", "COMPLETED"),
            processing_stage=res.get("processing_stage", "COMPLETED"),
            document_id=res.get("document_id"),
            error=res.get("error"),
            chunk_count=res.get("chunk_count", 0),
            page_count=res.get("page_count", 0),
        )

    if async_result.state == "FAILURE":
        return TaskStatusResponse(
            task_id=task_id,
            status="FAILED",
            processing_stage="FAILED",
            document_id=None,
            error=str(async_result.result),
            chunk_count=0,
            page_count=0,
        )

    if async_result.state == "PENDING" and async_result.info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found.",
        )

    stage = "QUEUED"
    if isinstance(async_result.info, dict):
        stage = async_result.info.get("stage", "QUEUED")

    return TaskStatusResponse(
        task_id=task_id,
        status=async_result.state,
        processing_stage=stage,
        document_id=None,
        error=None,
        chunk_count=0,
        page_count=0,
    )


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
    semantic_cache: SemanticCacheService | None = Depends(get_semantic_cache),
) -> None:
    """Delete a document and its cascading chunks from PostgreSQL."""
    doc_repo = DocumentRepository(db)
    deleted = await doc_repo.delete(document_id)
    if not deleted:
        raise DocumentNotFoundError(f"Document with ID {document_id} not found.")
    if semantic_cache is not None:
        await semantic_cache.invalidate_document(document_id)

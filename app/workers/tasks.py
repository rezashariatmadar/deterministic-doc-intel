"""Celery tasks for background processing."""

import asyncio
import concurrent.futures
import contextlib
import logging
import os
from pathlib import Path
from typing import Any
from uuid import UUID

from celery import Task

from app.db.models.document import Document
from app.db.session import get_task_scoped_session
from app.services.ingestion.pipeline import process_document_ingestion
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

__all__ = ["celery_app", "get_task_scoped_session", "ingest_document_task"]


@celery_app.task(bind=True, acks_late=True, name="app.workers.tasks.ingest_document_task")  # type: ignore[untyped-decorator]
def ingest_document_task(self: Task, document_id: str, file_path: str) -> dict[str, Any]:
    """Celery task to ingest a document asynchronously using an isolated event loop."""
    task_id = self.request.id or None
    logger.info("Executing ingest_document_task: task_id=%s, doc_id=%s", task_id, document_id)

    async def _run() -> Document:
        async with get_task_scoped_session() as session:
            return await process_document_ingestion(
                document_id=UUID(document_id),
                file_path=Path(file_path),
                session=session,
                task_id=task_id,
            )

    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                doc = pool.submit(asyncio.run, _run()).result()
        else:
            doc = asyncio.run(_run())

        return {
            "task_id": task_id or doc.task_id or "",
            "status": doc.status,
            "processing_stage": doc.processing_stage,
            "document_id": str(doc.id),
            "error": None,
            "chunk_count": doc.total_chunks,
            "page_count": doc.total_pages,
        }
    except Exception as exc:
        logger.exception("Ingestion task failed for doc_id=%s: %s", document_id, exc)
        raise
    finally:
        path_obj = Path(file_path)
        if ("uploads" in path_obj.parts or "tmp" in path_obj.parts) and path_obj.exists():
            with contextlib.suppress(OSError):
                os.remove(file_path)

"""Unit tests for Celery worker engine isolation and task-scoped sessions."""

import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.pool import NullPool

from app.workers.tasks import get_task_scoped_session, ingest_document_task


def test_task_scoped_session_uses_nullpool():
    """Verify that get_task_scoped_session instantiates an AsyncEngine with NullPool."""

    async def _check_pool() -> bool:
        async with get_task_scoped_session() as session:
            assert isinstance(session.bind, AsyncEngine)
            return isinstance(session.bind.pool, NullPool)

    is_null_pool = asyncio.run(_check_pool())
    assert is_null_pool is True


def test_task_scoped_session_executes_queries_in_separate_event_loops():
    """Verify that separate asyncio.run() loops can execute queries without cross-loop errors."""

    async def _run_query() -> int:
        async with get_task_scoped_session() as session:
            result = await session.execute(text("SELECT 1"))
            return int(result.scalar_one())

    # First event loop
    res1 = asyncio.run(_run_query())
    assert res1 == 1

    # Second event loop (new loop, simulating another worker task)
    res2 = asyncio.run(_run_query())
    assert res2 == 1


def test_ingest_document_task_registered():
    """Verify Celery task decorator and signature."""
    assert ingest_document_task.name == "app.workers.tasks.ingest_document_task"

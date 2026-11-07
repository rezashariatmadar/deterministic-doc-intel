"""Async database session management using SQLAlchemy and asyncpg."""

import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.core.config import get_settings

settings = get_settings()

is_testing = (
    "pytest" in sys.modules
    or bool(os.environ.get("PYTEST_CURRENT_TEST"))
    or settings.APP_ENV == "test"
)

engine_kwargs: dict[str, Any] = {
    "echo": settings.DEBUG,
}

if is_testing:
    engine_kwargs["poolclass"] = NullPool
else:
    engine_kwargs["pool_pre_ping"] = True
    engine_kwargs["pool_size"] = 10
    engine_kwargs["max_overflow"] = 20

engine: AsyncEngine = create_async_engine(
    settings.DATABASE_URL,
    **engine_kwargs,
)

async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for yielding an async database session per request."""
    async with async_session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_task_scoped_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield a task-scoped AsyncSession backed by an isolated NullPool engine.

    Instantiates a dedicated AsyncEngine with poolclass=NullPool inside the
    current asyncio event loop to ensure complete loop isolation across
    Celery task executions.
    """
    task_engine: AsyncEngine = create_async_engine(
        settings.DATABASE_URL,
        poolclass=NullPool,
        echo=settings.DEBUG,
    )
    task_session_maker = async_sessionmaker(
        task_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )
    async with task_session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
    await task_engine.dispose()

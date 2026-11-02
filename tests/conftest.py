"""Pytest fixtures for integration tests."""

from collections.abc import AsyncGenerator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import async_session_maker


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an isolated async database session."""
    async with async_session_maker() as session:
        yield session

"""Pytest fixtures for integration tests."""

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import async_session_maker


@pytest.fixture
def sample_pdf_path() -> Path:
    """Return path to sample 10-K PDF for parsing and chunking tests."""
    return Path("data/raw_corpus/nvidia_10k_sample.pdf")


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an isolated async database session."""
    async with async_session_maker() as session:
        yield session

"""FastAPI dependency injection providers for database and AI services."""

from collections.abc import AsyncGenerator
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.session import async_session_maker
from app.services.providers.base import EmbeddingProvider, LLMProvider
from app.services.providers.factory import get_embedding_provider, get_llm_provider
from app.services.retrieval.reranker import CrossEncoderReranker


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency that yields a scoped async database session."""
    async with async_session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


def get_embedding() -> EmbeddingProvider:
    """Dependency returning the configured embedding provider singleton/instance."""
    return get_embedding_provider()


def get_llm() -> LLMProvider:
    """Dependency returning the configured LLM provider instance."""
    return get_llm_provider()


@lru_cache
def get_reranker() -> CrossEncoderReranker | None:
    """Dependency returning the shared CrossEncoderReranker instance if enabled."""
    settings = get_settings()
    if settings.ENABLE_RERANKER:
        return CrossEncoderReranker()
    return None

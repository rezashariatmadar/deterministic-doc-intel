"""FastAPI dependency injection providers for database, cache, and AI services."""

from collections.abc import AsyncGenerator
from functools import lru_cache

import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.session import async_session_maker
from app.services.cache.semantic_cache import SemanticCacheService
from app.services.providers.base import EmbeddingProvider, LLMProvider
from app.services.providers.factory import get_embedding_provider, get_llm_provider
from app.services.retrieval.reranker import CrossEncoderReranker

_redis_client: aioredis.Redis | None = None


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


def get_redis_client() -> aioredis.Redis:
    """Return a shared singleton async Redis client instance."""
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=False,
        )
    assert _redis_client is not None
    return _redis_client


def get_semantic_cache() -> SemanticCacheService | None:
    """Dependency returning the SemanticCacheService instance if enabled."""
    settings = get_settings()
    if not settings.SEMANTIC_CACHE_ENABLED:
        return None
    redis_client = get_redis_client()
    return SemanticCacheService(
        redis=redis_client,
        threshold=settings.SEMANTIC_CACHE_THRESHOLD,
        ttl=settings.SEMANTIC_CACHE_TTL,
    )

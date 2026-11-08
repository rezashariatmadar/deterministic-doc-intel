"""FastAPI dependency injection providers for database, cache, security, and AI services."""

import asyncio
from collections.abc import AsyncGenerator
from functools import lru_cache

import redis.asyncio as aioredis
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.session import async_session_maker
from app.services.agent.memory import ConversationMemoryService
from app.services.cache.semantic_cache import SemanticCacheService
from app.services.providers.base import EmbeddingProvider, LLMProvider
from app.services.providers.factory import get_embedding_provider, get_llm_provider
from app.services.retrieval.reranker import CrossEncoderReranker
from app.services.security.guardrails import (
    PIIRedactor,
    PromptInjectionDetector,
    RateLimiter,
)

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
    """Return a shared singleton async Redis client instance, refreshing if event loop closed."""
    global _redis_client

    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    if _redis_client is not None:
        try:
            pool = getattr(_redis_client, "connection_pool", None)
            pool_loop = getattr(pool, "_loop", None)
            if pool_loop is not None and (
                pool_loop.is_closed() or (current_loop and pool_loop is not current_loop)
            ):
                _redis_client = None
        except Exception:
            _redis_client = None

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
    try:
        redis_client = get_redis_client()
        return SemanticCacheService(
            redis=redis_client,
            threshold=settings.SEMANTIC_CACHE_THRESHOLD,
            ttl=settings.SEMANTIC_CACHE_TTL,
        )
    except Exception:
        return None


def get_rate_limiter() -> RateLimiter:
    """Dependency returning RateLimiter instance configured from application settings."""
    settings = get_settings()
    try:
        redis_client = get_redis_client()
    except Exception:
        redis_client = None
    return RateLimiter(
        redis_client=redis_client,
        limit=settings.RATE_LIMIT_REQUESTS,
        window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
    )


def get_prompt_injection_detector() -> PromptInjectionDetector:
    """Dependency returning PromptInjectionDetector instance."""
    return PromptInjectionDetector()


def get_pii_redactor() -> PIIRedactor:
    """Dependency returning PIIRedactor instance."""
    return PIIRedactor()


def get_conversation_memory(
    llm: LLMProvider = Depends(get_llm),
) -> ConversationMemoryService:
    """Dependency returning ConversationMemoryService with Redis persistence and LLM resolution."""
    settings = get_settings()
    try:
        redis_client = get_redis_client()
    except Exception:
        redis_client = None
    return ConversationMemoryService(
        redis_client=redis_client,
        llm_provider=llm,
        max_turns=settings.CHAT_SESSION_MAX_TURNS,
        ttl=settings.CHAT_SESSION_TTL,
    )

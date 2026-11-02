"""Pluggable provider interfaces and factory implementations."""

from app.services.providers.base import EmbeddingProvider, LLMProvider
from app.services.providers.cloud_provider import (
    OpenAIEmbeddingProvider,
    OpenAILLMProvider,
)
from app.services.providers.factory import get_embedding_provider, get_llm_provider
from app.services.providers.local_provider import (
    LocalHttpEmbeddingProvider,
    LocalHttpLLMProvider,
)

__all__ = [
    "EmbeddingProvider",
    "LLMProvider",
    "LocalHttpEmbeddingProvider",
    "LocalHttpLLMProvider",
    "OpenAIEmbeddingProvider",
    "OpenAILLMProvider",
    "get_embedding_provider",
    "get_llm_provider",
]

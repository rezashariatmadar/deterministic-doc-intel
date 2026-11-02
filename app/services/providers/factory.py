"""Factory functions to instantiate configured embedding and LLM providers."""

from app.core.config import get_settings
from app.core.exceptions import ConfigurationError
from app.services.providers.base import EmbeddingProvider, LLMProvider
from app.services.providers.cloud_provider import (
    OpenAIEmbeddingProvider,
    OpenAILLMProvider,
)
from app.services.providers.local_provider import (
    LocalHttpEmbeddingProvider,
    LocalHttpLLMProvider,
)


def get_embedding_provider(
    provider_type: str | None = None,
    model_name: str | None = None,
    dimension: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> EmbeddingProvider:
    """Instantiate and return the appropriate EmbeddingProvider."""
    settings = get_settings()
    ptype = (provider_type or settings.EMBEDDING_PROVIDER).lower()
    dim = dimension or settings.EMBEDDING_DIMENSION

    if ptype == "openai":
        key = api_key or settings.OPENAI_API_KEY
        mname = model_name or settings.OPENAI_EMBEDDING_MODEL
        return OpenAIEmbeddingProvider(
            api_key=key,
            model_name=mname,
            dimension=dim,
        )

    if ptype == "local":
        url = base_url or settings.LOCAL_LLM_BASE_URL
        mname = model_name or "nomic-embed-text"
        return LocalHttpEmbeddingProvider(
            base_url=url,
            model_name=mname,
            dimension=dim,
        )

    raise ConfigurationError(
        f"Unsupported embedding provider '{ptype}'. Allowed options: 'openai', 'local'."
    )


def get_llm_provider(
    provider_type: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMProvider:
    """Instantiate and return the appropriate LLMProvider."""
    settings = get_settings()
    ptype = (provider_type or settings.LLM_PROVIDER).lower()

    if ptype == "openai":
        key = api_key or settings.OPENAI_API_KEY
        mname = model_name or settings.OPENAI_LLM_MODEL
        return OpenAILLMProvider(
            api_key=key,
            model_name=mname,
        )

    if ptype == "local":
        url = base_url or settings.LOCAL_LLM_BASE_URL
        mname = model_name or settings.LOCAL_LLM_MODEL
        return LocalHttpLLMProvider(
            base_url=url,
            model_name=mname,
        )

    raise ConfigurationError(
        f"Unsupported LLM provider '{ptype}'. Allowed options: 'openai', 'local'."
    )

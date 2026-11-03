"""Unit tests for pluggable embedding and LLM provider abstractions."""

import pytest
import respx

from app.core.exceptions import ConfigurationError
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


def test_openai_embedding_provider_initialization() -> None:
    provider = OpenAIEmbeddingProvider(
        api_key="sk-test-key",
        model_name="text-embedding-3-small",
        dimension=1024,
    )
    assert isinstance(provider, EmbeddingProvider)
    assert provider.dimension == 1024
    assert provider.model_name == "text-embedding-3-small"


def test_provider_initialization_missing_key_raises() -> None:
    with pytest.raises(ConfigurationError, match="OpenAI API key"):
        OpenAIEmbeddingProvider(api_key=None)

    with pytest.raises(ConfigurationError, match="OpenAI API key"):
        OpenAILLMProvider(api_key=None)


@pytest.mark.asyncio
async def test_openai_embedding_provider_embed_texts_and_query() -> None:
    provider = OpenAIEmbeddingProvider(
        api_key="sk-test-key",
        model_name="text-embedding-3-small",
        dimension=4,
    )

    mock_vec1 = [0.1, 0.2, 0.3, 0.4]
    mock_vec2 = [0.5, 0.6, 0.7, 0.8]

    with respx.mock(base_url="https://api.openai.com/v1") as respx_mock:
        respx_mock.post("/embeddings").respond(
            status_code=200,
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "embedding": mock_vec1, "index": 0},
                    {"object": "embedding", "embedding": mock_vec2, "index": 1},
                ],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 10, "total_tokens": 10},
            },
        )

        embeddings = await provider.embed_texts(["First text", "Second text"])
        assert len(embeddings) == 2
        assert embeddings[0] == mock_vec1
        assert embeddings[1] == mock_vec2

        query_res = await provider.embed_query("Search query")
        assert len(query_res) == 4


@pytest.mark.asyncio
async def test_openai_llm_provider_generate_and_stream() -> None:
    provider = OpenAILLMProvider(
        api_key="sk-test-key",
        model_name="gpt-4o-mini",
    )
    assert isinstance(provider, LLMProvider)

    # 1. Non-streaming generate
    with respx.mock(base_url="https://api.openai.com/v1") as respx_mock:
        respx_mock.post("/chat/completions").respond(
            status_code=200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Data Center revenue was $47.5B [1].",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

        response = await provider.generate(
            prompt="What was Data Center revenue?",
            system_prompt="Answer factually.",
        )
        assert "Data Center revenue was $47.5B [1]." in response

    # 2. Streaming generate
    sse_chunks = (
        'data: {"choices":[{"delta":{"content":"Data"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":" Center"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":" revenue"}}]}\n\n'
        "data: [DONE]\n\n"
    )

    with respx.mock(base_url="https://api.openai.com/v1") as respx_mock:
        respx_mock.post("/chat/completions").respond(
            status_code=200,
            headers={"Content-Type": "text/event-stream"},
            content=sse_chunks.encode("utf-8"),
        )

        streamed_parts = []
        async for chunk in provider.generate_stream(prompt="Stream query"):
            streamed_parts.append(chunk)

        assert "".join(streamed_parts) == "Data Center revenue"


def test_local_http_provider_initialization() -> None:
    provider = LocalHttpEmbeddingProvider(
        base_url="http://localhost:11434/v1",
        model_name="nomic-embed-text",
        dimension=768,
    )
    assert provider.dimension == 768

    llm = LocalHttpLLMProvider(
        base_url="http://localhost:11434/v1",
        model_name="mistral",
    )
    assert llm.model_name == "mistral"


def test_factory_creates_expected_providers() -> None:
    openai_embed = get_embedding_provider(
        provider_type="openai",
        api_key="sk-test",
        dimension=1024,
    )
    assert isinstance(openai_embed, OpenAIEmbeddingProvider)

    openai_llm = get_llm_provider(
        provider_type="openai",
        api_key="sk-test",
    )
    assert isinstance(openai_llm, OpenAILLMProvider)

    local_embed = get_embedding_provider(
        provider_type="local",
        base_url="http://localhost:11434/v1",
        dimension=1024,
    )
    assert isinstance(local_embed, LocalHttpEmbeddingProvider)

    local_llm = get_llm_provider(
        provider_type="local",
        base_url="http://localhost:11434/v1",
    )
    assert isinstance(local_llm, LocalHttpLLMProvider)

    with pytest.raises(ConfigurationError, match="Unsupported embedding provider"):
        get_embedding_provider(provider_type="unknown")


def test_local_embedding_provider_respects_settings_model() -> None:
    from app.core.config import get_settings
    settings = get_settings()
    provider = get_embedding_provider(provider_type="local")
    assert provider.model_name == settings.EMBEDDING_MODEL


def test_gemini_configuration_handling() -> None:
    with pytest.raises(ConfigurationError, match="Gemini provider is not yet installed"):
        get_embedding_provider(provider_type="gemini")
    with pytest.raises(ConfigurationError, match="Gemini provider is not yet installed"):
        get_llm_provider(provider_type="gemini")


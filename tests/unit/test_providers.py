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


def test_factory_creates_gemini_providers() -> None:
    embed = get_embedding_provider(
        provider_type="gemini",
        api_key="gemini-test-key",
        dimension=1024,
    )
    assert embed.model_name == "gemini-embedding-001"
    assert embed.dimension == 1024

    llm = get_llm_provider(
        provider_type="gemini",
        api_key="gemini-test-key",
    )
    assert llm.model_name == "gemini-2.5-flash"


@pytest.mark.asyncio
async def test_gemini_embedding_provider_embed_texts_and_query() -> None:
    from app.services.providers.cloud_provider import GeminiEmbeddingProvider

    provider = GeminiEmbeddingProvider(
        api_key="test-key",
        model_name="gemini-embedding-001",
        dimension=4,
    )
    mock_vec = [0.1, 0.2, 0.3, 0.4]

    with respx.mock(base_url="https://generativelanguage.googleapis.com/v1beta") as respx_mock:
        route = respx_mock.post("/models/gemini-embedding-001:batchEmbedContents").respond(
            status_code=200,
            json={"embeddings": [{"values": mock_vec}]},
        )
        res = await provider.embed_texts(["hello"])
        assert len(res) == 1
        assert res[0] == mock_vec
        assert route.calls.last.request.headers["x-goog-api-key"] == "test-key"

        respx_mock.post("/models/gemini-embedding-001:batchEmbedContents").respond(
            status_code=200,
            json={"embeddings": [{"values": mock_vec}]},
        )
        query_res = await provider.embed_query("hello")
        assert query_res == mock_vec


@pytest.mark.asyncio
async def test_gemini_llm_provider_generate_and_stream() -> None:
    from app.services.providers.cloud_provider import GeminiLLMProvider

    provider = GeminiLLMProvider(
        api_key="test-key",
        model_name="gemini-2.5-flash",
    )

    with respx.mock(base_url="https://generativelanguage.googleapis.com/v1beta") as respx_mock:
        gen_route = respx_mock.post("/models/gemini-2.5-flash:generateContent").respond(
            status_code=200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": "Data Center revenue was $47.5B [1]."}]}}
                ]
            },
        )
        resp = await provider.generate(prompt="Question", system_prompt="Answer factually.")
        assert "Data Center revenue was $47.5B" in resp
        assert gen_route.calls.last.request.headers["x-goog-api-key"] == "test-key"

        sse_data = (
            'data: {"candidates": [{"content": {"parts": [{"text": "Data Center"}]}}]}\n\n'
            'data: {"candidates": [{"content": {"parts": [{"text": " revenue"}]}}]}\n\n'
        )
        stream_route = respx_mock.post("/models/gemini-2.5-flash:streamGenerateContent").respond(
            status_code=200,
            headers={"Content-Type": "text/event-stream"},
            content=sse_data.encode("utf-8"),
        )
        stream_tokens = [tok async for tok in provider.generate_stream(prompt="Question")]
        assert "".join(stream_tokens) == "Data Center revenue"
        assert stream_route.calls.last.request.headers["x-goog-api-key"] == "test-key"

"""Local provider implementations (OpenAI-compatible HTTP servers and Sentence Transformers)."""

import json
from collections.abc import AsyncGenerator

import httpx

from app.core.exceptions import ConfigurationError
from app.services.providers.base import EmbeddingProvider, LLMProvider


class LocalHttpEmbeddingProvider(EmbeddingProvider):
    """Local OpenAI-compatible embedding provider (Ollama, vLLM, llama.cpp, TEI)."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model_name: str = "BAAI/bge-m3",
        dimension: int = 1024,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(model_name=model_name, dimension=dimension)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload = {
            "model": self.model_name,
            "input": texts,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/embeddings",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]

    async def embed_query(self, query: str) -> list[float]:
        results = await self.embed_texts([query])
        return results[0]


class LocalHttpLLMProvider(LLMProvider):
    """Local OpenAI-compatible LLM provider (Ollama, vLLM, LM Studio, llama.cpp)."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model_name: str = "mistral",
        timeout: float = 90.0,
    ) -> None:
        super().__init__(model_name=model_name)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _build_messages(self, prompt: str, system_prompt: str | None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        messages = self._build_messages(prompt, system_prompt)
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        content = data["choices"][0]["message"]["content"]
        return content or ""

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> AsyncGenerator[str, None]:
        messages = self._build_messages(prompt, system_prompt)
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }

        async with (
            httpx.AsyncClient(timeout=self.timeout) as client,
            client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
            ) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                clean_line = line.strip()
                if not clean_line or clean_line.startswith(":"):
                    continue
                if clean_line.startswith("data: "):
                    data_part = clean_line[6:].strip()
                    if data_part == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(data_part)
                        delta = chunk_data["choices"][0].get("delta", {})
                        delta_content = delta.get("content")
                        if delta_content:
                            yield delta_content
                    except json.JSONDecodeError:
                        continue


class LocalCudaEmbeddingProvider(EmbeddingProvider):
    """Optional in-process PyTorch / SentenceTransformers embedding provider."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        dimension: int = 1024,
        device: str = "cuda",
    ) -> None:
        super().__init__(model_name=model_name, dimension=dimension)
        self.device = device
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as err:
            raise ConfigurationError(
                "sentence-transformers is not installed. Run 'uv pip install torch' "
                "or switch EMBEDDING_PROVIDER to 'openai' or 'local' (HTTP)."
            ) from err

        self._model = SentenceTransformer(self.model_name, device=self.device)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._ensure_loaded()
        assert self._model is not None
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    async def embed_query(self, query: str) -> list[float]:
        results = await self.embed_texts([query])
        return results[0]

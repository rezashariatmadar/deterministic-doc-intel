"""Cloud provider implementations (OpenAI & Gemini) for embeddings and LLMs."""

import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from app.core.exceptions import ConfigurationError
from app.services.providers.base import EmbeddingProvider, LLMProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embeddings client supporting text-embedding-3 models."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "text-embedding-3-small",
        dimension: int = 1536,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise ConfigurationError("OpenAI API key must be provided via settings or argument.")
        super().__init__(model_name=model_name, dimension=dimension)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload: dict[str, object] = {
            "model": self.model_name,
            "input": texts,
        }
        # text-embedding-3 supports dimension shortening
        if "text-embedding-3" in self.model_name and self.dimension:
            payload["dimensions"] = self.dimension

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        # Sort results by index to ensure order matches input
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]

    async def embed_query(self, query: str) -> list[float]:
        results = await self.embed_texts([query])
        return results[0]


class OpenAILLMProvider(LLMProvider):
    """OpenAI chat completions provider supporting standard and streaming generation."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise ConfigurationError("OpenAI API key must be provided via settings or argument.")
        super().__init__(model_name=model_name)
        self.api_key = api_key
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
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
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
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with (
            httpx.AsyncClient(timeout=self.timeout) as client,
            client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=headers,
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


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Google Gemini embeddings client supporting gemini-embedding models."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-embedding-001",
        dimension: int = 1024,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise ConfigurationError("Gemini API key must be provided via settings or argument.")
        super().__init__(model_name=model_name, dimension=dimension)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        clean_model = self.model_name.removeprefix("models/")
        requests = [
            {
                "model": f"models/{clean_model}",
                "content": {"parts": [{"text": t}]},
                "outputDimensionality": self.dimension,
            }
            for t in texts
        ]

        url = f"{self.base_url}/models/{clean_model}:batchEmbedContents"
        params = {"key": self.api_key}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, params=params, json={"requests": requests})
            response.raise_for_status()
            data = response.json()

        return [item["values"] for item in data.get("embeddings", [])]

    async def embed_query(self, query: str) -> list[float]:
        results = await self.embed_texts([query])
        return results[0]


class GeminiLLMProvider(LLMProvider):
    """Google Gemini LLM client supporting gemini models with streaming."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-3.5-flash",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise ConfigurationError("Gemini API key must be provided via settings or argument.")
        super().__init__(model_name=model_name)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        clean_model = self.model_name.removeprefix("models/")
        url = f"{self.base_url}/models/{clean_model}:generateContent"
        params = {"key": self.api_key}

        payload: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, params=params, json=payload)
            response.raise_for_status()
            data = response.json()

        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts)

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> AsyncGenerator[str, None]:
        clean_model = self.model_name.removeprefix("models/")
        url = f"{self.base_url}/models/{clean_model}:streamGenerateContent"
        params = {"key": self.api_key, "alt": "sse"}

        payload: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        async with (
            httpx.AsyncClient(timeout=self.timeout) as client,
            client.stream("POST", url, params=params, json=payload) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                clean_line = line.strip()
                if not clean_line or clean_line.startswith(":"):
                    continue
                if clean_line.startswith("data: "):
                    data_part = clean_line[6:].strip()
                    try:
                        chunk_data = json.loads(data_part)
                        candidates = chunk_data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            for p in parts:
                                text_piece = p.get("text")
                                if text_piece:
                                    yield text_piece
                    except json.JSONDecodeError:
                        continue

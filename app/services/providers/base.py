"""Base abstract classes for embedding and LLM providers."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator


class EmbeddingProvider(ABC):
    """Abstract interface for text embedding providers."""

    def __init__(self, model_name: str, dimension: int) -> None:
        self.model_name = model_name
        self.dimension = dimension

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Compute vector embeddings for a list of text strings."""

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        """Compute vector embedding for a single search query."""


class LLMProvider(ABC):
    """Abstract interface for Large Language Model generation providers."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Generate a complete text response for the given prompt."""

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> AsyncGenerator[str, None]:
        """Stream chunks of the generated response asynchronously."""

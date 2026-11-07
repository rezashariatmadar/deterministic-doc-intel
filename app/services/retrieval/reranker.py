"""Neural Cross-Encoder reranker with asynchronous execution and circuit breaker."""

import asyncio
import logging
import math
from typing import Any

from app.core.config import get_settings
from app.schemas.query import RetrievedChunk

logger = logging.getLogger(__name__)

# Process-level model cache to avoid reloading PyTorch weights on every HTTP request
_MODEL_CACHE: dict[tuple[str, str | None], Any] = {}


def sigmoid(x: float) -> float:
    """Map unbounded logit to [0.0, 1.0] probability."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


class CrossEncoderReranker:
    """
    Reranks candidate retrieved chunks using a cross-encoder model.
    Runs inference off the event loop and features a circuit breaker timeout.
    """

    def __init__(
        self,
        model_name: str | None = None,
        timeout_ms: int | None = None,
        device: str | None = None,
        model_instance: Any = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.RERANKER_MODEL
        self.timeout_ms = timeout_ms or settings.RERANKER_TIMEOUT_MS
        self.device = device
        self._model = model_instance

    def _ensure_loaded(self) -> Any:
        if self._model is not None:
            return self._model
        cache_key = (self.model_name, self.device)
        if cache_key in _MODEL_CACHE:
            self._model = _MODEL_CACHE[cache_key]
            return self._model
        try:
            from sentence_transformers import CrossEncoder

            loaded = CrossEncoder(self.model_name, device=self.device)
            _MODEL_CACHE[cache_key] = loaded
            self._model = loaded
            return self._model
        except Exception as exc:
            logger.warning("Failed to initialize CrossEncoder %s: %s", self.model_name, exc)
            raise

    def _score_pairs_sync(self, pairs: list[tuple[str, str]]) -> list[float]:
        model = self._ensure_loaded()
        raw_logits = model.predict(pairs)
        return [sigmoid(float(logit)) for logit in raw_logits]

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: int = 5,
        threshold: float = 0.05,
    ) -> list[RetrievedChunk]:
        """
        Rerank retrieved chunks against the query.
        Returns up to top_n chunks above threshold, re-indexed with source_id 1..N.
        Falls back to input chunks on timeout or error (circuit breaker).
        """
        if not chunks or not query.strip():
            return []

        # Pairs of (query, chunk_context)
        pairs = [(query, c.contextualized_text or c.text) for c in chunks]

        try:
            scores = await asyncio.wait_for(
                asyncio.to_thread(self._score_pairs_sync, pairs),
                timeout=self.timeout_ms / 1000.0,
            )
        except Exception as exc:
            logger.warning(
                "CrossEncoder reranking failed or timed out (%s). Falling back to RRF ordering.",
                exc,
            )
            # Circuit breaker: graceful degradation to input chunks
            fallback_chunks = chunks[:top_n]
            for idx, c in enumerate(fallback_chunks, start=1):
                c.source_id = idx
            return fallback_chunks

        # Attach rerank scores and sort descending
        scored_chunks: list[RetrievedChunk] = []
        for chunk, score in zip(chunks, scores, strict=True):
            chunk.rerank_score = round(score, 4)
            # Keep similarity_score normalized and bounded for downstream confidence estimation
            chunk.similarity_score = max(chunk.similarity_score, round(score, 4))
            if score >= threshold:
                scored_chunks.append(chunk)

        scored_chunks.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)
        final_chunks = scored_chunks[:top_n]

        # Re-index source_id from 1 to N
        for idx, c in enumerate(final_chunks, start=1):
            c.source_id = idx

        return final_chunks

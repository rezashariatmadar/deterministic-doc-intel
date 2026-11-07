"""Unit tests for CrossEncoderReranker, Sigmoid normalization, and Circuit Breaker."""

import time
import uuid
from unittest.mock import MagicMock

import pytest

from app.schemas.query import RetrievedChunk
from app.services.retrieval.reranker import CrossEncoderReranker, sigmoid


def make_chunk(
    source_id: int,
    text: str,
    similarity_score: float = 0.5,
) -> RetrievedChunk:
    return RetrievedChunk(
        source_id=source_id,
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_title="Sample Doc",
        page_start=1,
        page_end=1,
        section_path="Section 1",
        element_type="prose",
        text=text,
        contextualized_text=f"[Doc: Sample] {text}",
        similarity_score=similarity_score,
    )


def test_sigmoid_math_and_bounds() -> None:
    # Negative logit must be strictly positive and < 0.5
    assert 0.0 < sigmoid(-10.0) < 0.001
    # 0.0 logit must be exactly 0.5
    assert pytest.approx(sigmoid(0.0)) == 0.5
    # Positive logit must be > 0.5 and < 1.0
    assert 0.999 < sigmoid(10.0) < 1.0


@pytest.mark.asyncio
async def test_reranker_reorders_and_reindexes_chunks() -> None:
    c1 = make_chunk(1, "Vague mention of networking hardware.", similarity_score=0.85)
    c2 = make_chunk(2, "Data Center revenue was $47.5B in fiscal 2024.", similarity_score=0.75)
    c3 = make_chunk(3, "Unrelated paragraph about board elections.", similarity_score=0.65)

    # Mock model where c2 gets highest logit, c1 gets medium, c3 gets very negative
    mock_model = MagicMock()
    # Logits: c1 -> 0.0 (prob 0.5), c2 -> 4.0 (prob ~0.98), c3 -> -5.0 (prob ~0.006)
    mock_model.predict.return_value = [0.0, 4.0, -5.0]

    reranker = CrossEncoderReranker(model_instance=mock_model, timeout_ms=500)
    reranked = await reranker.rerank(
        query="What was Data Center revenue?",
        chunks=[c1, c2, c3],
        top_n=2,
        threshold=0.05,
    )

    # c2 must be boosted to rank 1 and re-indexed to source_id 1
    assert len(reranked) == 2
    assert reranked[0].text == c2.text
    assert reranked[0].source_id == 1
    assert reranked[0].rerank_score > 0.9

    # c1 must be rank 2 and re-indexed to source_id 2
    assert reranked[1].text == c1.text
    assert reranked[1].source_id == 2
    assert reranked[1].rerank_score == pytest.approx(0.5, rel=1e-3)

    # c3 was below threshold (0.05) and was pruned


@pytest.mark.asyncio
async def test_reranker_circuit_breaker_on_timeout() -> None:
    c1 = make_chunk(1, "First chunk", similarity_score=0.9)
    c2 = make_chunk(2, "Second chunk", similarity_score=0.8)

    # Mock model that sleeps longer than timeout_ms
    mock_model = MagicMock()

    def slow_predict(pairs: list) -> list[float]:
        time.sleep(0.15)
        return [2.0, 1.0]

    mock_model.predict.side_effect = slow_predict

    # Set timeout to 50ms
    reranker = CrossEncoderReranker(model_instance=mock_model, timeout_ms=50)

    # Must NOT crash; should trigger circuit breaker and return input chunks
    fallback_chunks = await reranker.rerank(
        query="test query",
        chunks=[c1, c2],
        top_n=2,
    )

    assert len(fallback_chunks) == 2
    assert fallback_chunks[0].text == "First chunk"
    assert fallback_chunks[0].source_id == 1
    assert fallback_chunks[1].text == "Second chunk"
    assert fallback_chunks[1].source_id == 2


@pytest.mark.asyncio
async def test_reranker_circuit_breaker_on_exception() -> None:
    c1 = make_chunk(1, "Chunk text", similarity_score=0.9)

    mock_model = MagicMock()
    mock_model.predict.side_effect = RuntimeError("CUDA out of memory")

    reranker = CrossEncoderReranker(model_instance=mock_model, timeout_ms=500)
    fallback = await reranker.rerank(query="query", chunks=[c1], top_n=1)

    assert len(fallback) == 1
    assert fallback[0].text == "Chunk text"
    assert fallback[0].source_id == 1

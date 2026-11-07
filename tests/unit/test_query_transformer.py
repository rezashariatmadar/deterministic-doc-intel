"""Unit tests for QueryTransformer single-call optimization and error fallbacks."""

from unittest.mock import AsyncMock

import pytest

from app.services.retrieval.query_transformer import QueryTransformer, TransformedQuery


@pytest.mark.asyncio
async def test_query_transformer_parses_valid_json() -> None:
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = """
    {
        "semantic_query": "What was NVIDIA Data Center revenue and growth rate in fiscal 2024?",
        "lexical_query": "NVIDIA \\"Data Center\\" revenue fiscal 2024 growth",
        "sub_queries": [
            "What was Data Center revenue in 2024?",
            "What was Data Center growth rate?"
        ]
    }
    """

    transformer = QueryTransformer(llm_provider=mock_llm)
    res = await transformer.transform("How did data center do in 24?")

    assert isinstance(res, TransformedQuery)
    assert res.original_query == "How did data center do in 24?"
    assert "Data Center" in res.semantic_query
    assert '"Data Center"' in res.lexical_query
    assert len(res.sub_queries) == 2


@pytest.mark.asyncio
async def test_query_transformer_fallback_on_invalid_json() -> None:
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "Sorry, I am an AI and cannot process this request."

    transformer = QueryTransformer(llm_provider=mock_llm)
    raw_query = "Compare Q1 and Q2 revenue"
    res = await transformer.transform(raw_query)

    assert isinstance(res, TransformedQuery)
    assert res.semantic_query == raw_query
    assert res.lexical_query == raw_query
    assert res.sub_queries == [raw_query]


@pytest.mark.asyncio
async def test_query_transformer_fallback_when_llm_is_none() -> None:
    transformer = QueryTransformer(llm_provider=None)
    raw_query = "What is the operating margin?"
    res = await transformer.transform(raw_query)

    assert res.semantic_query == raw_query
    assert res.lexical_query == raw_query
    assert res.sub_queries == [raw_query]


@pytest.mark.asyncio
async def test_query_transformer_with_hyde_enabled() -> None:
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = """
    {
        "semantic_query": "NVIDIA fiscal 2024 GAAP gross margin percentage",
        "lexical_query": "NVIDIA GAAP gross margin 2024",
        "sub_queries": ["What was gross margin?"],
        "hypothetical_passage": "In fiscal 2024, gross margin was 72.7%, up from 56.9%."
    }
    """

    transformer = QueryTransformer(llm_provider=mock_llm)
    res = await transformer.transform("gross margin", enable_hyde=True)

    assert res.hypothetical_passage is not None
    assert "72.7%" in res.hypothetical_passage

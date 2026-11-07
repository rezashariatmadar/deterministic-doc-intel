"""Unit tests for CostEstimator token counting and cost calculation."""

from unittest.mock import patch

import pytest

from app.services.telemetry.cost import CostEstimator


def test_estimate_tokens_standard():
    text = "Artificial intelligence and hybrid vector search."
    tokens = CostEstimator.estimate_tokens(text, model="gpt-4o-mini")
    assert isinstance(tokens, int)
    assert tokens > 0


def test_estimate_tokens_empty_string():
    assert CostEstimator.estimate_tokens("", model="gpt-4o-mini") == 0


def test_estimate_tokens_unknown_model_fallback():
    text = "Retrieval augmented generation with parent-child chunking."
    # Model not natively in tiktoken model dict (e.g. gemini-2.5-flash) falls back to cl100k_base
    tokens = CostEstimator.estimate_tokens(text, model="gemini-2.5-flash")
    assert isinstance(tokens, int)
    assert tokens > 0


def test_estimate_tokens_graceful_character_fallback():
    text = "Fallback character estimation test string"
    with (
        patch("tiktoken.get_encoding", side_effect=Exception("tiktoken failed")),
        patch("tiktoken.encoding_for_model", side_effect=Exception("tiktoken failed")),
    ):
        tokens = CostEstimator.estimate_tokens(text, model="gpt-4o-mini")
        expected = max(1, len(text) // 4)
        assert tokens == expected


def test_estimate_cost_gpt_4o_mini():
    # gpt-4o-mini: $0.15 per 1M prompt tokens, $0.60 per 1M completion tokens
    cost = CostEstimator.estimate_cost(
        model="gpt-4o-mini",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        embedding_tokens=0,
    )
    assert pytest.approx(cost, rel=1e-5) == 0.75


def test_estimate_cost_gemini_flash():
    # gemini-2.5-flash: $0.075 per 1M prompt tokens, $0.30 per 1M completion tokens
    cost = CostEstimator.estimate_cost(
        model="gemini-2.5-flash",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        embedding_tokens=0,
    )
    assert pytest.approx(cost, rel=1e-5) == 0.375


def test_estimate_cost_with_embeddings():
    # text-embedding-3-small: $0.02 per 1M tokens
    cost = CostEstimator.estimate_cost(
        model="gpt-4o-mini",
        prompt_tokens=1_000_000,
        completion_tokens=0,
        embedding_tokens=1_000_000,
    )
    # prompt ($0.15) + embedding ($0.02) = $0.17
    assert pytest.approx(cost, rel=1e-5) == 0.17


def test_estimate_cost_zero_tokens():
    cost = CostEstimator.estimate_cost(
        model="gpt-4o-mini",
        prompt_tokens=0,
        completion_tokens=0,
        embedding_tokens=0,
    )
    assert cost == 0.0


def test_estimate_cost_unknown_model_fallback():
    cost = CostEstimator.estimate_cost(
        model="unknown-custom-model",
        prompt_tokens=1_000,
        completion_tokens=1_000,
        embedding_tokens=1_000,
    )
    assert isinstance(cost, float)
    assert cost > 0.0

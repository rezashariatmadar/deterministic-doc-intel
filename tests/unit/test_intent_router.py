"""Unit tests for the 2-Tier Semantic Intent Router."""

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock

import pytest

from app.schemas.query import IntentRoute
from app.services.agent.router import SemanticIntentRouter
from app.services.providers.base import LLMProvider


class MockRouterLLM(LLMProvider):
    def __init__(self) -> None:
        super().__init__(model_name="mock-router-llm")
        self.generate_mock = AsyncMock()

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        res = await self.generate_mock(prompt, system_prompt=system_prompt, temperature=temperature)
        return str(res)

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> AsyncGenerator[str, None]:
        yield "mock"


class TestSemanticIntentRouter:
    """Test suite for Tier 1 deterministic and Tier 2 LLM routing."""

    @pytest.fixture
    def router(self) -> SemanticIntentRouter:
        return SemanticIntentRouter(llm_provider=None)

    @pytest.mark.parametrize(
        "query",
        [
            "hello!",
            "Hi there",
            "Good morning",
            "Who are you?",
            "What can you do?",
            "Thank you so much",
            "tell me a joke please",
        ],
    )
    def test_tier1_chitchat(self, router: SemanticIntentRouter, query: str) -> None:
        decision = router.classify_tier1(query)
        assert decision is not None
        assert decision.route == IntentRoute.CHITCHAT_OR_OUT_OF_DOMAIN
        assert decision.tier == "tier1_rules"

    @pytest.mark.parametrize(
        "query",
        [
            "SELECT revenue FROM financial_metrics WHERE fiscal_year = 2023",
            "Query the database table for Apple's average revenue",
            "What was the sum of NVIDIA net income in the database?",
            "Show me the highest revenue in the tabular data",
        ],
    )
    def test_tier1_sql(self, router: SemanticIntentRouter, query: str) -> None:
        decision = router.classify_tier1(query)
        assert decision is not None
        assert decision.route == IntentRoute.STRUCTURED_SQL_ANALYTICS
        assert decision.tier == "tier1_rules"

    @pytest.mark.parametrize(
        "query",
        [
            "Compare revenue of Apple and NVIDIA and explain why NVIDIA grew faster",
            "Show financial metrics and narrative strategy from the 10-K",
            "Compare gross margins between 2022 and 2023 and summarize the reasons",
        ],
    )
    def test_tier1_hybrid_sql_rag(self, router: SemanticIntentRouter, query: str) -> None:
        decision = router.classify_tier1(query)
        assert decision is not None
        assert decision.route == IntentRoute.HYBRID_SQL_AND_RAG
        assert decision.tier == "tier1_rules"

    @pytest.mark.parametrize(
        "query",
        [
            "step by step how did supply chain constraints affect production",
            "how did export controls impact GPU sales and what was the management response",
        ],
    )
    def test_tier1_multi_hop(self, router: SemanticIntentRouter, query: str) -> None:
        decision = router.classify_tier1(query)
        assert decision is not None
        assert decision.route == IntentRoute.MULTI_HOP_AGENTIC_RAG
        assert decision.tier == "tier1_rules"

    @pytest.mark.asyncio
    async def test_fallback_fast_track_when_no_llm(self, router: SemanticIntentRouter) -> None:
        query = "What is Apple's policy on intellectual property rights?"
        decision = await router.classify(query)
        assert decision.route == IntentRoute.FAST_TRACK_HYBRID_RAG

    @pytest.mark.asyncio
    async def test_tier2_llm_classification(self) -> None:
        mock_llm = MockRouterLLM()
        mock_llm.generate_mock.return_value = (
            '{"route": "MULTI_HOP_AGENTIC_RAG", "confidence": 0.94, '
            '"reasoning": "Involves multi-step analysis"}'
        )
        router = SemanticIntentRouter(llm_provider=mock_llm)

        decision = await router.classify("Investigate cross-company trends over three cycles")
        assert decision.route == IntentRoute.MULTI_HOP_AGENTIC_RAG
        assert decision.tier == "tier2_llm"
        assert decision.confidence == 0.94

    @pytest.mark.asyncio
    async def test_tier2_llm_failure_defaults_to_fast_track(self) -> None:
        mock_llm = MockRouterLLM()
        mock_llm.generate_mock.side_effect = RuntimeError("LLM error")
        router = SemanticIntentRouter(llm_provider=mock_llm)

        decision = await router.classify("Ambiguous query without clear rules")
        assert decision.route == IntentRoute.FAST_TRACK_HYBRID_RAG

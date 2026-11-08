"""Unit tests for multi-turn conversational memory and coreference contextualizer."""

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.agent.memory import ConversationMemoryService
from app.services.providers.base import LLMProvider


class MockLLM(LLMProvider):
    def __init__(self) -> None:
        super().__init__(model_name="mock-llm")
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


class TestConversationMemoryService:
    """Test suite for conversation history persistence, sliding window, and contextualization."""

    @pytest.fixture
    def memory_service(self) -> ConversationMemoryService:
        return ConversationMemoryService(redis_client=None, max_turns=6, ttl=3600)

    @pytest.mark.asyncio
    async def test_add_turn_and_sliding_window(
        self, memory_service: ConversationMemoryService
    ) -> None:
        session_id = "test_sess_1"

        # Add 8 turns, should keep only last 6
        for i in range(8):
            await memory_service.add_turn(
                session_id=session_id,
                user_query=f"Question {i}",
                assistant_response=f"Answer {i}",
            )

        history = await memory_service.get_history(session_id)
        assert len(history) == 6
        assert history[0]["user"] == "Question 2"
        assert history[-1]["user"] == "Question 7"

    @pytest.mark.asyncio
    async def test_clear_session(self, memory_service: ConversationMemoryService) -> None:
        session_id = "test_sess_clear"
        await memory_service.add_turn(session_id, "Hello", "Hi there")
        assert len(await memory_service.get_history(session_id)) == 1

        await memory_service.clear(session_id)
        assert len(await memory_service.get_history(session_id)) == 0

    @pytest.mark.asyncio
    async def test_contextualize_no_session_or_no_history(
        self, memory_service: ConversationMemoryService
    ) -> None:
        q = "What is its revenue?"
        assert await memory_service.contextualize_query(None, q) == q
        assert await memory_service.contextualize_query("empty_session", q) == q

    @pytest.mark.asyncio
    async def test_contextualize_unambiguous_query_unchanged(
        self, memory_service: ConversationMemoryService
    ) -> None:
        session_id = "test_sess_unambiguous"
        await memory_service.add_turn(session_id, "Tell me about Apple", "Apple is a tech company")

        q = "What was NVIDIA revenue in 2024?"
        # Does not contain ambiguous pronouns
        res = await memory_service.contextualize_query(session_id, q)
        assert res == q

    @pytest.mark.asyncio
    async def test_contextualize_with_llm(self) -> None:
        mock_llm = MockLLM()
        mock_llm.generate_mock.return_value = "What was Apple's total revenue in fiscal year 2023?"
        service = ConversationMemoryService(redis_client=None, llm_provider=mock_llm)

        session_id = "test_llm_sess"
        await service.add_turn(
            session_id,
            "What were Apple's key products in 2023?",
            "Apple makes iPhones, Macs, and iPads.",
        )

        res = await service.contextualize_query(session_id, "What was its revenue?")
        assert res == "What was Apple's total revenue in fiscal year 2023?"
        mock_llm.generate_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_contextualize_heuristic_fallback(
        self, memory_service: ConversationMemoryService
    ) -> None:
        session_id = "test_heuristic_sess"
        await memory_service.add_turn(
            session_id,
            "Can you describe Apple's financial health?",
            "Apple maintains a strong balance sheet.",
        )

        # Heuristic should replace "its" with "Apple's"
        res = await memory_service.contextualize_query(session_id, "What was its revenue in 2023?")
        assert "Apple's revenue in 2023?" in res

        # Heuristic should replace "their" with "Apple's"
        res_their = await memory_service.contextualize_query(
            session_id, "What was their net income in 2023?"
        )
        assert "Apple's net income in 2023?" in res_their

    @pytest.mark.asyncio
    async def test_contextualize_multi_turn_entity_in_earlier_turn(
        self, memory_service: ConversationMemoryService
    ) -> None:
        session_id = "test_multi_turn_sess"
        # Turn 1: Mentions NVIDIA
        await memory_service.add_turn(
            session_id,
            "Let's look at NVIDIA's AI hardware.",
            "NVIDIA designs H100 GPUs.",
        )
        # Turn 2: Intervening question without entity name
        await memory_service.add_turn(
            session_id,
            "What architecture do they use?",
            "They use the Hopper architecture.",
        )
        # Turn 3: Follow-up using 'their'
        res = await memory_service.contextualize_query(
            session_id, "What was their total revenue in 2024?"
        )
        assert "NVIDIA's total revenue in 2024?" in res

    @pytest.mark.asyncio
    async def test_redis_persistence(self) -> None:
        mock_redis = MagicMock()
        mock_pipe = MagicMock()
        mock_pipe.execute = AsyncMock(return_value=[True, True, True])
        mock_redis.pipeline.return_value = mock_pipe
        mock_redis.lrange = AsyncMock(
            return_value=[
                b'{"user": "Q1", "assistant": "A1", "timestamp": 12345}',
                b'{"user": "Q2", "assistant": "A2", "timestamp": 12346}',
            ]
        )

        service = ConversationMemoryService(redis_client=mock_redis, max_turns=6)
        await service.add_turn("sess_redis", "Q3", "A3")
        mock_redis.pipeline.assert_called_once()

        history = await service.get_history("sess_redis")
        assert len(history) == 2
        assert history[0]["user"] == "Q1"
        assert history[1]["assistant"] == "A2"

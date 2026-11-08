"""Unit tests for CRAG Grader, Reformulator, Fact Verifier, and Agent Orchestrator state machine."""

import uuid
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock

import pytest

from app.db.session import async_session_maker
from app.schemas.query import (
    CitationSourceType,
    ExtractionStatus,
    IntentRoute,
    QueryFilters,
    RetrievedChunk,
)
from app.services.agent.grader import (
    AdaptiveQueryReformulator,
    FactExtractorAndVerifier,
    RetrievalGrader,
)
from app.services.agent.orchestrator import AgentOrchestrator, AgentState
from app.services.providers.base import EmbeddingProvider, LLMProvider


class MockAgentLLM(LLMProvider):
    def __init__(self) -> None:
        super().__init__(model_name="mock-agent-llm")
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


class MockAgentEmbedding(EmbeddingProvider):
    def __init__(self) -> None:
        super().__init__(model_name="mock-embedding", dimension=1024)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 1024 for _ in texts]

    async def embed_query(self, query: str) -> list[float]:
        return [0.1] * 1024


def make_chunk(
    source_id: int = 1,
    text: str = "Sample chunk text",
    similarity: float = 0.85,
    source_type: CitationSourceType = CitationSourceType.DOCUMENT,
) -> RetrievedChunk:
    return RetrievedChunk(
        source_id=source_id,
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_title="Sample Document",
        page_start=1,
        page_end=1,
        section_path="Section 1",
        element_type="text",
        text=text,
        contextualized_text=text,
        similarity_score=similarity,
        source_type=source_type,
        metadata={"source_type": source_type.value},
    )


class TestCRAGGraderAndReformulator:
    """Test suite for RetrievalGrader and AdaptiveQueryReformulator."""

    @pytest.mark.asyncio
    async def test_grader_accepts_high_similarity_and_synthetic(self) -> None:
        grader = RetrievalGrader(llm_provider=None, min_relevance_threshold=0.35)
        c1 = make_chunk(similarity=0.85)
        c2 = make_chunk(similarity=0.10, source_type=CitationSourceType.SQL_TABLE)
        c3 = make_chunk(similarity=0.15)  # Should be rejected

        rel, irrel = await grader.grade_chunks("Sample query", [c1, c2, c3])
        assert len(rel) == 2
        assert len(irrel) == 1
        assert c3 in irrel

    @pytest.mark.asyncio
    async def test_reformulator_heuristic(self) -> None:
        reformulator = AdaptiveQueryReformulator(llm_provider=None)
        q = "What was AAPL revenue and NVDA EPS in FY2023?"
        expanded = await reformulator.reformulate(q, retry_count=0)
        assert "Apple Inc." in expanded
        assert "NVIDIA Corporation" in expanded
        assert "earnings per share" in expanded

    @pytest.mark.asyncio
    async def test_fact_verifier(self) -> None:
        verifier = FactExtractorAndVerifier(llm_provider=None)
        chunk = make_chunk(text="Apple revenue was $383,285 million in fiscal year 2023.")
        answer = "Apple total revenue was $383,285 million in fiscal 2023 [1]."

        facts = await verifier.extract_and_verify("revenue", answer, [chunk], is_answerable=True)
        assert len(facts) >= 1
        assert facts[0].status == ExtractionStatus.VERIFIED
        assert 1 in facts[0].source_ids

    @pytest.mark.asyncio
    async def test_fact_verifier_granular_statuses(self) -> None:
        verifier = FactExtractorAndVerifier(llm_provider=None)
        chunk = make_chunk(text="Apple revenue was $383,285 million in fiscal year 2023.")

        # Test extraction_failed on unverified tag
        unverified_answer = "Apple released a secret VR headset in 2021 [unverified: 2]."
        facts_unverified = await verifier.extract_and_verify(
            "headset", unverified_answer, [chunk], is_answerable=True
        )
        assert len(facts_unverified) >= 1
        assert facts_unverified[0].status == ExtractionStatus.EXTRACTION_FAILED

        # Test conflicting status
        conflicting_answer = (
            "The reports contradict each other regarding total European sales figures."
        )
        facts_conflict = await verifier.extract_and_verify(
            "sales", conflicting_answer, [chunk], is_answerable=True
        )
        assert len(facts_conflict) >= 1
        assert facts_conflict[0].status == ExtractionStatus.CONFLICTING

        # Test ambiguous status
        ambiguous_answer = (
            "It is unclear whether regulatory scrutiny might impact future operations."
        )
        facts_ambiguous = await verifier.extract_and_verify(
            "scrutiny", ambiguous_answer, [chunk], is_answerable=True
        )
        assert len(facts_ambiguous) >= 1
        assert facts_ambiguous[0].status == ExtractionStatus.AMBIGUOUS


class TestAgentOrchestratorStateMachine:
    """Test suite for AgentOrchestrator state transitions, CRAG loop, and SQL integration."""

    @pytest.mark.asyncio
    async def test_chitchat_route_execution(self) -> None:
        async with async_session_maker() as session:
            orchestrator = AgentOrchestrator(
                db=session,
                embedding_provider=MockAgentEmbedding(),
                llm_provider=MockAgentLLM(),
            )
            state = AgentState(
                query="Hello there!",
                original_query="Hello there!",
                route_override=IntentRoute.CHITCHAT_OR_OUT_OF_DOMAIN,
            )
            response = await orchestrator.run(state)
            assert response.is_answerable is True
            assert response.confidence_score == 1.0
            assert len(response.citations) == 0
            assert response.route == IntentRoute.CHITCHAT_OR_OUT_OF_DOMAIN
            assert len(response.execution_trace) >= 1
            assert any(t.step == "chitchat_node" for t in response.execution_trace)

    @pytest.mark.asyncio
    async def test_structured_sql_analytics_route_execution(self) -> None:
        async with async_session_maker() as session:
            mock_llm = MockAgentLLM()

            async def mock_generate(
                prompt: str, system_prompt: str | None = None, temperature: float = 0.0
            ) -> str:
                if "financial_metrics" in prompt or "Text-to-SQL" in prompt:
                    return (
                        "SELECT company, fiscal_year, metric_name, metric_value "
                        "FROM financial_metrics WHERE company = 'Apple' AND fiscal_year = 2023"
                    )
                return "In fiscal year 2023, Apple reported total revenue of $383,285 million [1]."

            mock_llm.generate_mock.side_effect = mock_generate

            orchestrator = AgentOrchestrator(
                db=session,
                embedding_provider=MockAgentEmbedding(),
                llm_provider=mock_llm,
            )
            sql_q = "What was Apple's total revenue in fiscal year 2023 in the database?"
            state = AgentState(
                query=sql_q,
                original_query=sql_q,
                route_override=IntentRoute.STRUCTURED_SQL_ANALYTICS,
            )
            response = await orchestrator.run(state)
            assert response.is_answerable is True
            assert response.route == IntentRoute.STRUCTURED_SQL_ANALYTICS
            assert response.sql_query_executed is not None
            assert "financial_metrics" in response.sql_query_executed
            # Trace should contain sql_node and generate_node
            steps = [t.step for t in response.execution_trace]
            assert "sql_node" in steps
            assert "generate_node" in steps

    @pytest.mark.asyncio
    async def test_crag_retry_and_filter_relaxation(self) -> None:
        """CRAG Trap 2 & Trap 8: Verify rewrite loop and filter relaxation on retry #2."""
        async with async_session_maker() as session:
            mock_llm = MockAgentLLM()
            mock_llm.generate_mock.return_value = (
                "I cannot find sufficient evidence in the documents to answer this question."
            )

            orchestrator = AgentOrchestrator(
                db=session,
                embedding_provider=MockAgentEmbedding(),
                llm_provider=mock_llm,
            )
            # Force empty retrieval to trigger CRAG retries
            orchestrator.vector_search.search = AsyncMock(return_value=[])  # type: ignore[method-assign]

            initial_filters = QueryFilters(
                element_types=["table"],
                page_min=1,
                page_max=5,
                min_similarity=0.60,
            )

            state = AgentState(
                query="obscure term not found",
                original_query="obscure term not found",
                filters=initial_filters,
                max_crag_retries=2,
                allow_external_fallback=False,
                route_override=IntentRoute.FAST_TRACK_HYBRID_RAG,
            )

            response = await orchestrator.run(state)

            # Verified Trap 2: retried 2 times
            assert response.crag_retry_count == 2
            assert response.rewritten_query is not None
            assert response.rewritten_query != "obscure term not found"
            # Verified Trap 8: filters relaxed on retry #2
            assert state.filters is not None
            assert state.filters.element_types is None
            assert state.filters.page_min is None
            assert state.filters.page_max is None
            assert state.filters.min_similarity == 0.25

            # Execution trace should record rewrite_node twice
            rewrite_steps = [t for t in response.execution_trace if t.step == "rewrite_node"]
            assert len(rewrite_steps) == 2

    @pytest.mark.asyncio
    async def test_crag_external_fallback_when_enabled(self) -> None:
        """Verify fallback to external search when corpus search exhausts retries."""
        async with async_session_maker() as session:
            mock_llm = MockAgentLLM()
            mock_llm.generate_mock.return_value = (
                "According to public web records, the answer is verified [1]."
            )

            orchestrator = AgentOrchestrator(
                db=session,
                embedding_provider=MockAgentEmbedding(),
                llm_provider=mock_llm,
            )
            orchestrator.vector_search.search = AsyncMock(return_value=[])  # type: ignore[method-assign]

            state = AgentState(
                query="latest global corporate news",
                original_query="latest global corporate news",
                max_crag_retries=1,
                allow_external_fallback=True,
                route_override=IntentRoute.FAST_TRACK_HYBRID_RAG,
            )

            response = await orchestrator.run(state)
            steps = [t.step for t in response.execution_trace]
            assert "external_search_node" in steps
            assert len(state.external_chunks) > 0
            assert state.external_chunks[0].source_type == CitationSourceType.EXTERNAL_WEB

    @pytest.mark.asyncio
    async def test_multi_hop_enables_query_rewrite(self) -> None:
        """Verify retrieve_node sets enable_query_rewrite=True for MULTI_HOP_AGENTIC_RAG."""
        async with async_session_maker() as session:
            mock_llm = MockAgentLLM()
            mock_llm.generate_mock.return_value = "Multi-hop answer [1]."
            orchestrator = AgentOrchestrator(
                db=session,
                embedding_provider=MockAgentEmbedding(),
                llm_provider=mock_llm,
            )
            orchestrator.vector_search.search = AsyncMock(  # type: ignore[method-assign]
                return_value=[make_chunk(text="Relevant evidence chunk", similarity=0.9)]
            )

            state = AgentState(
                query="Compare revenue and profit across years",
                original_query="Compare revenue and profit across years",
                route_override=IntentRoute.MULTI_HOP_AGENTIC_RAG,
            )
            response = await orchestrator.run(state)
            assert response.route == IntentRoute.MULTI_HOP_AGENTIC_RAG
            call_kwargs = orchestrator.vector_search.search.call_args.kwargs
            assert call_kwargs.get("enable_query_rewrite") is True

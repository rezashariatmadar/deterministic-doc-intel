"""Explicit Pure-Python State Machine Orchestrator for Self-Correcting Agentic RAG."""

import logging
import time
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.query import (
    AgentTraceStep,
    CandidateFact,
    IntentRoute,
    QueryFilters,
    RAGResponse,
    RetrievedChunk,
)
from app.services.agent.external_search import ExternalSearchTool
from app.services.agent.grader import (
    AdaptiveQueryReformulator,
    FactExtractorAndVerifier,
    RetrievalGrader,
)
from app.services.agent.router import SemanticIntentRouter
from app.services.agent.sql_tool import SafeSQLTool
from app.services.providers.base import EmbeddingProvider, LLMProvider
from app.services.retrieval.generator import GroundedGenerator
from app.services.retrieval.reranker import CrossEncoderReranker
from app.services.retrieval.vector_search import VectorSearchService

logger = logging.getLogger(__name__)


class AgentState(BaseModel):
    """Execution state passed through the agent state machine nodes."""

    query: str
    original_query: str
    session_id: str | None = None
    filters: QueryFilters | None = None
    top_k: int = 5
    search_mode: Literal["hybrid", "dense", "sparse"] = "hybrid"
    weight_dense: float | None = None
    weight_sparse: float | None = None
    enable_reranker: bool | None = None
    resolve_parents: bool = True
    allow_external_fallback: bool = False
    route: IntentRoute | None = None
    route_override: IntentRoute | None = None
    crag_retries: int = 0
    max_crag_retries: int = 2
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    graded_chunks: list[RetrievedChunk] = Field(default_factory=list)
    sql_chunks: list[RetrievedChunk] = Field(default_factory=list)
    external_chunks: list[RetrievedChunk] = Field(default_factory=list)
    candidate_facts: list[CandidateFact] = Field(default_factory=list)
    execution_trace: list[AgentTraceStep] = Field(default_factory=list)
    sql_query_executed: str | None = None
    response: RAGResponse | None = None
    error: str | None = None


class AgentOrchestrator:
    """Explicit Pure-Python State Machine with Async Nodes and Bounded Conditional Transitions."""

    def __init__(
        self,
        db: AsyncSession,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
        reranker: CrossEncoderReranker | None = None,
        router: SemanticIntentRouter | None = None,
        sql_tool: SafeSQLTool | None = None,
        external_tool: ExternalSearchTool | None = None,
        grader: RetrievalGrader | None = None,
        reformulator: AdaptiveQueryReformulator | None = None,
        fact_verifier: FactExtractorAndVerifier | None = None,
        generator: GroundedGenerator | None = None,
    ) -> None:
        self.db = db
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider
        self.reranker = reranker

        self.vector_search = VectorSearchService(
            session=db,
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            reranker=reranker,
        )
        self.router = router or SemanticIntentRouter(llm_provider=llm_provider)
        self.sql_tool = sql_tool or SafeSQLTool(llm_provider=llm_provider)
        self.external_tool = external_tool or ExternalSearchTool()
        self.grader = grader or RetrievalGrader(llm_provider=llm_provider)
        self.reformulator = reformulator or AdaptiveQueryReformulator(llm_provider=llm_provider)
        self.fact_verifier = fact_verifier or FactExtractorAndVerifier(llm_provider=llm_provider)
        self.generator = generator or GroundedGenerator(llm_provider=llm_provider)

    def _record_trace(
        self,
        state: AgentState,
        step: str,
        action: str,
        input_data: Any = None,
        output_data: Any = None,
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        state.execution_trace.append(
            AgentTraceStep(
                step=step,
                action=action,
                input=input_data,
                output=output_data,
                duration_ms=round(duration_ms, 2) if duration_ms is not None else None,
                metadata=metadata or {},
            )
        )

    async def router_node(self, state: AgentState) -> str:
        """Route query via IntentRouter or honor manual route override."""
        start = time.perf_counter()
        if state.route_override is not None:
            state.route = state.route_override
            reasoning = "Route specified via route_override parameter."
            tier = "manual"
        else:
            decision = await self.router.classify(state.query)
            state.route = decision.route
            reasoning = decision.reasoning
            tier = decision.tier

        duration = (time.perf_counter() - start) * 1000.0
        self._record_trace(
            state,
            step="router_node",
            action="classify_intent",
            input_data={"query": state.query},
            output_data={"route": state.route.value, "reasoning": reasoning},
            duration_ms=duration,
            metadata={"tier": tier},
        )

        # Transition mapping
        if state.route == IntentRoute.CHITCHAT_OR_OUT_OF_DOMAIN:
            return "chitchat_node"
        elif state.route in (
            IntentRoute.STRUCTURED_SQL_ANALYTICS,
            IntentRoute.HYBRID_SQL_AND_RAG,
        ):
            return "sql_node"
        else:
            # FAST_TRACK_HYBRID_RAG or MULTI_HOP_AGENTIC_RAG
            return "retrieve_node"

    async def chitchat_node(self, state: AgentState) -> str:
        """Handle conversational greetings or out-of-domain small talk."""
        start = time.perf_counter()
        canned_answer = (
            "Hello! I am your AI financial and corporate intelligence assistant. "
            "I can answer factual questions about SEC 10-K filings, analyze metrics, "
            "or compare annual performance for Apple (FY22-23) and NVIDIA (FY23-24)."
        )
        duration = (time.perf_counter() - start) * 1000.0
        self._record_trace(
            state,
            step="chitchat_node",
            action="generate_greeting",
            output_data={"answer": canned_answer},
            duration_ms=duration,
        )

        state.response = RAGResponse(
            answer=canned_answer,
            is_answerable=True,
            citations=[],
            sources=[],
            confidence_score=1.0,
            latency_ms=round(duration, 2),
            route=state.route,
            candidate_facts=[],
            execution_trace=state.execution_trace,
            crag_retry_count=state.crag_retries,
            crag_retries=state.crag_retries,
            sql_query_executed=state.sql_query_executed,
            rewritten_query=state.query if state.query != state.original_query else None,
        )
        return "END"

    async def sql_node(self, state: AgentState) -> str:
        """Translate natural language to safe SQL and execute against financial_metrics sandbox."""
        start = time.perf_counter()
        try:
            sql_query = await self.sql_tool.generate_sql(state.query)
            state.sql_query_executed = sql_query
            cols, rows = await self.sql_tool.execute_query(self.db, sql_query)
            if getattr(self.sql_tool, "last_executed_sql", None):
                state.sql_query_executed = self.sql_tool.last_executed_sql
            executed_sql = state.sql_query_executed or sql_query
            sql_chunk = self.sql_tool.format_as_chunk(
                cols, rows, executed_sql, source_id=len(state.sql_chunks) + 1
            )
            state.sql_chunks.append(sql_chunk)

            duration = (time.perf_counter() - start) * 1000.0
            self._record_trace(
                state,
                step="sql_node",
                action="execute_sql_query",
                input_data={"generated_sql": sql_query},
                output_data={"row_count": len(rows), "columns": cols},
                duration_ms=duration,
            )
        except Exception as exc:
            duration = (time.perf_counter() - start) * 1000.0
            logger.warning("SafeSQLTool failed execution: %s", exc)
            self._record_trace(
                state,
                step="sql_node",
                action="execute_sql_query_failed",
                output_data={"error": str(exc)},
                duration_ms=duration,
            )

        if state.route == IntentRoute.HYBRID_SQL_AND_RAG:
            return "retrieve_node"
        return "generate_node"

    async def retrieve_node(self, state: AgentState) -> str:
        """Execute vector search over corporate filing chunks."""
        start = time.perf_counter()
        top_k = (
            max(state.top_k, 8) if state.route == IntentRoute.MULTI_HOP_AGENTIC_RAG else state.top_k
        )
        chunks = await self.vector_search.search(
            query=state.query,
            filters=state.filters,
            top_k=top_k,
            search_mode=state.search_mode,
            weight_dense=state.weight_dense,
            weight_sparse=state.weight_sparse,
            enable_reranker=state.enable_reranker,
            enable_query_rewrite=(state.route == IntentRoute.MULTI_HOP_AGENTIC_RAG),
            resolve_parents=state.resolve_parents,
        )
        state.retrieved_chunks = chunks

        duration = (time.perf_counter() - start) * 1000.0
        self._record_trace(
            state,
            step="retrieve_node",
            action="vector_search",
            input_data={"query": state.query, "retry": state.crag_retries},
            output_data={"retrieved_chunk_count": len(chunks)},
            duration_ms=duration,
        )
        return "grade_node"

    async def grade_node(self, state: AgentState) -> str:
        """CRAG Step: Filter chunks by relevance and decide next transition (Trap 2)."""
        start = time.perf_counter()
        rel_chunks, irrel_chunks = await self.grader.grade_chunks(
            state.query, state.retrieved_chunks
        )
        state.graded_chunks = rel_chunks

        duration = (time.perf_counter() - start) * 1000.0
        self._record_trace(
            state,
            step="grade_node",
            action="evaluate_chunk_relevance",
            output_data={
                "relevant_count": len(rel_chunks),
                "irrelevant_count": len(irrel_chunks),
                "crag_retries": state.crag_retries,
                "max_crag_retries": state.max_crag_retries,
            },
            duration_ms=duration,
        )

        # Trap 2: If no relevant chunks and retries remaining -> rewrite_node
        if len(rel_chunks) == 0 and state.crag_retries < state.max_crag_retries:
            return "rewrite_node"

        # Trap 2: If no relevant chunks after max retries -> fallback to external search or generate
        if len(rel_chunks) == 0 and state.crag_retries >= state.max_crag_retries:
            if state.allow_external_fallback:
                return "external_search_node"
            return "generate_node"

        return "generate_node"

    async def rewrite_node(self, state: AgentState) -> str:
        """CRAG Step: Increment retry, reformulate query, and relax filters (Trap 8)."""
        start = time.perf_counter()
        state.crag_retries += 1

        reformulated = await self.reformulator.reformulate(
            query=state.query,
            retry_count=state.crag_retries,
            failed_chunks=state.retrieved_chunks,
        )
        state.query = reformulated

        # Trap 8: Broaden overly restrictive filters on CRAG retry #2
        if state.crag_retries >= 2:
            logger.info("CRAG Retry #2 reached: relaxing filters and threshold to 0.25")
            if state.filters is not None:
                state.filters.element_types = None
                state.filters.page_min = None
                state.filters.page_max = None
                state.filters.min_similarity = 0.25
            else:
                state.filters = QueryFilters(min_similarity=0.25)

        duration = (time.perf_counter() - start) * 1000.0
        self._record_trace(
            state,
            step="rewrite_node",
            action="reformulate_query",
            output_data={
                "new_query": state.query,
                "retry_count": state.crag_retries,
                "filters_relaxed": state.crag_retries >= 2,
            },
            duration_ms=duration,
        )
        return "retrieve_node"

    async def external_search_node(self, state: AgentState) -> str:
        """Retrieve external web knowledge when corporate corpus has no grounding evidence."""
        start = time.perf_counter()
        external_chunks = await self.external_tool.search(
            query=state.original_query,
            allow_external_fallback=state.allow_external_fallback,
            top_k=3,
            start_source_id=len(state.sql_chunks) + len(state.graded_chunks) + 1,
        )
        state.external_chunks = external_chunks

        duration = (time.perf_counter() - start) * 1000.0
        self._record_trace(
            state,
            step="external_search_node",
            action="external_web_search",
            output_data={"external_chunk_count": len(external_chunks)},
            duration_ms=duration,
        )
        return "generate_node"

    async def generate_node(self, state: AgentState) -> str:
        """Synthesize response, extract verified candidate facts, and segregate citations."""
        start = time.perf_counter()

        # Combine all valid context chunks
        combined_chunks: list[RetrievedChunk] = []
        combined_chunks.extend(state.sql_chunks)
        combined_chunks.extend(state.graded_chunks)
        combined_chunks.extend(state.external_chunks)

        # Re-index source_ids so [1], [2] are strictly 1-based sequential
        for idx, chunk in enumerate(combined_chunks, start=1):
            chunk.source_id = idx

        # Generate grounded answer
        rag_response = await self.generator.generate(
            query=state.original_query,
            chunks=combined_chunks,
        )

        # Trap 4: Segregate citations by matching source_type from corresponding chunk
        chunk_source_map = {c.source_id: c.source_type for c in combined_chunks}
        for citation in rag_response.citations:
            if citation.source_id in chunk_source_map:
                citation.source_type = chunk_source_map[citation.source_id]
                citation.metadata["source_type"] = citation.source_type.value

        # Extract and verify Candidate Facts
        candidate_facts = await self.fact_verifier.extract_and_verify(
            query=state.original_query,
            answer=rag_response.answer,
            chunks=combined_chunks,
            is_answerable=rag_response.is_answerable,
        )

        duration = (time.perf_counter() - start) * 1000.0
        self._record_trace(
            state,
            step="generate_node",
            action="grounded_generation_and_verification",
            output_data={
                "is_answerable": rag_response.is_answerable,
                "citation_count": len(rag_response.citations),
                "candidate_facts_count": len(candidate_facts),
            },
            duration_ms=duration,
        )

        rag_response.candidate_facts = candidate_facts
        rag_response.execution_trace = state.execution_trace
        rag_response.route = state.route
        rag_response.stage_latencies = self.vector_search.last_stage_latencies
        rag_response.crag_retry_count = state.crag_retries
        rag_response.crag_retries = state.crag_retries
        rag_response.sql_query_executed = state.sql_query_executed
        if state.query != state.original_query:
            rag_response.rewritten_query = state.query
        state.response = rag_response

        return "END"

    async def run(self, initial_state: AgentState) -> RAGResponse:
        """Execute state machine until terminal END transition."""
        current_node = "router_node"
        state = initial_state

        max_steps = 15  # Safety boundary preventing infinite transition loops
        steps_executed = 0

        while current_node != "END" and steps_executed < max_steps:
            steps_executed += 1
            if current_node == "router_node":
                current_node = await self.router_node(state)
            elif current_node == "chitchat_node":
                current_node = await self.chitchat_node(state)
            elif current_node == "sql_node":
                current_node = await self.sql_node(state)
            elif current_node == "retrieve_node":
                current_node = await self.retrieve_node(state)
            elif current_node == "grade_node":
                current_node = await self.grade_node(state)
            elif current_node == "rewrite_node":
                current_node = await self.rewrite_node(state)
            elif current_node == "external_search_node":
                current_node = await self.external_search_node(state)
            elif current_node == "generate_node":
                current_node = await self.generate_node(state)
            else:
                logger.error("Unknown state machine node '%s'", current_node)
                break

        assert state.response is not None, (
            "Agent state machine finished without producing a RAGResponse"
        )
        return state.response

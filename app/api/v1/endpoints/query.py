"""Query endpoints for grounded answers and real-time Server-Sent Events (SSE) streaming."""

import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, BackgroundTasks, Depends, Request, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import (
    get_conversation_memory,
    get_db,
    get_embedding,
    get_llm,
    get_pii_redactor,
    get_prompt_injection_detector,
    get_rate_limiter,
    get_reranker,
    get_semantic_cache,
)
from app.core.logging import get_correlation_id
from app.db.repositories.telemetry import TelemetryRepository
from app.db.session import async_session_maker
from app.schemas.metrics import QueryLogCreate
from app.schemas.query import QueryRequest, RAGResponse, RetrievedChunk
from app.services.agent.memory import ConversationMemoryService
from app.services.agent.orchestrator import AgentOrchestrator, AgentState
from app.services.cache.semantic_cache import SemanticCacheService
from app.services.providers.base import EmbeddingProvider, LLMProvider
from app.services.retrieval.generator import GroundedGenerator
from app.services.retrieval.reranker import CrossEncoderReranker
from app.services.retrieval.vector_search import VectorSearchService
from app.services.security.guardrails import (
    PIIRedactor,
    PromptInjectionDetector,
    RateLimiter,
)
from app.services.telemetry.cost import CostEstimator

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=RAGResponse)
async def query_rag(
    request: QueryRequest,
    response: Response,
    background_tasks: BackgroundTasks,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
    embedding_provider: EmbeddingProvider = Depends(get_embedding),
    llm_provider: LLMProvider = Depends(get_llm),
    reranker: CrossEncoderReranker | None = Depends(get_reranker),
    semantic_cache: SemanticCacheService | None = Depends(get_semantic_cache),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    prompt_detector: PromptInjectionDetector = Depends(get_prompt_injection_detector),
    pii_redactor: PIIRedactor = Depends(get_pii_redactor),
    conversation_memory: ConversationMemoryService = Depends(get_conversation_memory),
) -> RAGResponse:
    """Execute vector search and generate grounded answer with citations and confidence score."""
    start_time = time.perf_counter()

    # 1. Enforce rate limiting per client
    forwarded = http_request.headers.get("x-forwarded-for", "")
    client_host = http_request.client.host if http_request.client else None
    client_id = forwarded.split(",")[0].strip() or client_host or "127.0.0.1"
    await rate_limiter.enforce(client_id)

    # 2. Guard against prompt injection attacks (HTTP 400)
    prompt_detector.validate(request.query)

    # 2b. Redact sensitive PII from user query (Step 4.6)
    sanitized_query = pii_redactor.redact(request.query)

    # 3. Contextualize query with conversational memory before cache lookup (Trap 1)
    effective_query = sanitized_query
    if request.session_id:
        effective_query = await conversation_memory.contextualize_query(
            request.session_id, sanitized_query
        )

    query_vector: list[float] | None = None
    scope_hash: str | None = None

    # 4. Semantic cache lookup using contextualized query
    if semantic_cache is not None:
        try:
            scope_hash = semantic_cache.canonical_scope_hash(
                search_mode=request.search_mode,
                filters=request.filters,
                top_k=request.top_k,
                enable_reranker=request.enable_reranker,
            )
            query_vector = await embedding_provider.embed_query(effective_query)
            cached_response, _similarity = await semantic_cache.lookup(
                query_vector=query_vector,
                scope_hash=scope_hash,
            )
            if cached_response is not None:
                response.headers["X-Cache-Hit"] = "true"
                total_latency_ms = (time.perf_counter() - start_time) * 1000.0
                cached_response.latency_ms = round(total_latency_ms, 2)
                if effective_query != sanitized_query and cached_response.rewritten_query is None:
                    cached_response.rewritten_query = effective_query

                # Persist turn to conversational memory
                if request.session_id:
                    await conversation_memory.add_turn(
                        request.session_id, sanitized_query, cached_response.answer
                    )

                model_name = getattr(llm_provider, "model_name", "gpt-4o-mini")
                embedding_tokens = CostEstimator.estimate_tokens(effective_query)
                cost_usd = CostEstimator.estimate_cost(
                    model=model_name,
                    prompt_tokens=0,
                    completion_tokens=0,
                    embedding_tokens=embedding_tokens,
                )

                log_entry = QueryLogCreate(
                    request_id=get_correlation_id() or str(uuid.uuid4()),
                    session_id=request.session_id,
                    query_text=effective_query,
                    search_mode=request.search_mode,
                    cache_hit=True,
                    is_answerable=cached_response.is_answerable,
                    confidence_score=cached_response.confidence_score,
                    top_reranker_score=None,
                    rewrite_ms=0.0,
                    dense_search_ms=0.0,
                    sparse_search_ms=0.0,
                    rerank_ms=0.0,
                    llm_ttft_ms=None,
                    total_latency_ms=round(total_latency_ms, 2),
                    prompt_tokens=0,
                    completion_tokens=0,
                    embedding_tokens=embedding_tokens,
                    estimated_cost_usd=cost_usd,
                )
                try:
                    telemetry_repo = TelemetryRepository(db)
                    await telemetry_repo.log_query(log_entry)
                except Exception as exc:
                    logger.warning("Failed to persist query telemetry: %s", exc)

                return cached_response
        except Exception as exc:
            logger.warning("Semantic cache lookup encountered error: %s", exc)

    response.headers["X-Cache-Hit"] = "false"

    stage_latencies: dict[str, float] = {}
    chunks: list[RetrievedChunk] = []

    # 5. Routing: Execute multi-agent state machine if enable_agent or route_override
    if request.enable_agent or request.route_override is not None:
        orchestrator = AgentOrchestrator(
            db=db,
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            reranker=reranker,
        )
        initial_state = AgentState(
            query=effective_query,
            original_query=effective_query,
            session_id=request.session_id,
            filters=request.filters,
            top_k=request.top_k,
            search_mode=request.search_mode,
            weight_dense=request.weight_dense,
            weight_sparse=request.weight_sparse,
            enable_reranker=request.enable_reranker,
            allow_external_fallback=request.allow_external_fallback,
            route_override=request.route_override,
            max_crag_retries=request.max_crag_retries,
        )
        rag_response = await orchestrator.run(initial_state)
        chunks = rag_response.sources
    else:
        # Standard fast-path pipeline
        search_service = VectorSearchService(
            session=db,
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            reranker=reranker,
        )
        chunks = await search_service.search(
            query=effective_query,
            filters=request.filters,
            top_k=request.top_k,
            search_mode=request.search_mode,
            weight_dense=request.weight_dense,
            weight_sparse=request.weight_sparse,
            enable_reranker=request.enable_reranker,
            enable_query_rewrite=request.enable_query_rewrite,
            enable_hyde=request.enable_hyde,
        )
        generator = GroundedGenerator(llm_provider=llm_provider)
        rag_response = await generator.generate(query=effective_query, chunks=chunks)
        stage_latencies = search_service.last_stage_latencies

    total_latency_ms = (time.perf_counter() - start_time) * 1000.0
    rag_response.latency_ms = round(total_latency_ms, 2)
    if effective_query != sanitized_query and rag_response.rewritten_query is None:
        rag_response.rewritten_query = effective_query

    # 6. Save turn to conversational memory
    if request.session_id:
        try:
            await conversation_memory.add_turn(
                request.session_id, sanitized_query, rag_response.answer
            )
        except Exception as exc:
            logger.warning("Failed to save turn to conversational memory: %s", exc)

    # 7. Async store into semantic cache
    if semantic_cache is not None:
        try:
            if scope_hash is None:
                scope_hash = semantic_cache.canonical_scope_hash(
                    search_mode=request.search_mode,
                    filters=request.filters,
                    top_k=request.top_k,
                    enable_reranker=request.enable_reranker,
                )
            if query_vector is None:
                query_vector = await embedding_provider.embed_query(effective_query)
            background_tasks.add_task(
                semantic_cache.store,
                query_vector=query_vector,
                scope_hash=scope_hash,
                response=rag_response,
            )
        except Exception as exc:
            logger.warning("Failed to schedule semantic cache store: %s", exc)

    # 8. Extract stage latencies and top reranker score
    top_reranker_score: float | None = None
    if chunks:
        rerank_scores = [c.rerank_score for c in chunks if c.rerank_score is not None]
        if rerank_scores:
            top_reranker_score = max(rerank_scores)

    model_name = getattr(llm_provider, "model_name", "gpt-4o-mini")
    generator_helper = GroundedGenerator(llm_provider=llm_provider)
    context_xml = generator_helper.build_context_xml(chunks) if chunks else ""
    prompt_text = f"{context_xml}\n{effective_query}"
    prompt_tokens = CostEstimator.estimate_tokens(prompt_text, model=model_name)
    completion_tokens = CostEstimator.estimate_tokens(rag_response.answer, model=model_name)
    embedding_tokens = (
        CostEstimator.estimate_tokens(effective_query)
        if request.search_mode in ("hybrid", "dense")
        else 0
    )

    cost_usd = CostEstimator.estimate_cost(
        model=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        embedding_tokens=embedding_tokens,
    )

    log_entry = QueryLogCreate(
        request_id=get_correlation_id() or str(uuid.uuid4()),
        session_id=request.session_id,
        query_text=effective_query,
        search_mode=request.search_mode,
        cache_hit=False,
        is_answerable=rag_response.is_answerable,
        confidence_score=rag_response.confidence_score,
        top_reranker_score=top_reranker_score,
        rewrite_ms=stage_latencies.get("rewrite_ms", 0.0),
        dense_search_ms=stage_latencies.get("dense_search_ms", 0.0),
        sparse_search_ms=stage_latencies.get("sparse_search_ms", 0.0),
        rerank_ms=stage_latencies.get("rerank_ms", 0.0),
        llm_ttft_ms=None,
        total_latency_ms=round(total_latency_ms, 2),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        embedding_tokens=embedding_tokens,
        estimated_cost_usd=cost_usd,
    )

    try:
        telemetry_repo = TelemetryRepository(db)
        await telemetry_repo.log_query(log_entry)
    except Exception as exc:
        logger.warning("Failed to persist query telemetry: %s", exc)

    return rag_response


@router.post("/query/stream")
async def stream_query_rag(
    request: QueryRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
    embedding_provider: EmbeddingProvider = Depends(get_embedding),
    llm_provider: LLMProvider = Depends(get_llm),
    reranker: CrossEncoderReranker | None = Depends(get_reranker),
    semantic_cache: SemanticCacheService | None = Depends(get_semantic_cache),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    prompt_detector: PromptInjectionDetector = Depends(get_prompt_injection_detector),
    pii_redactor: PIIRedactor = Depends(get_pii_redactor),
    conversation_memory: ConversationMemoryService = Depends(get_conversation_memory),
) -> StreamingResponse:
    """Stream grounded answer tokens via Server-Sent Events (SSE)."""
    request_t0 = time.perf_counter()

    forwarded = http_request.headers.get("x-forwarded-for", "")
    client_host = http_request.client.host if http_request.client else None
    client_id = forwarded.split(",")[0].strip() or client_host or "127.0.0.1"
    await rate_limiter.enforce(client_id)
    prompt_detector.validate(request.query)

    # Redact sensitive PII from query (Step 4.6)
    sanitized_query = pii_redactor.redact(request.query)

    effective_query = sanitized_query
    if request.session_id:
        effective_query = await conversation_memory.contextualize_query(
            request.session_id, sanitized_query
        )

    scope_hash: str | None = None
    query_vector: list[float] | None = None

    if semantic_cache is not None:
        try:
            scope_hash = semantic_cache.canonical_scope_hash(
                search_mode=request.search_mode,
                filters=request.filters,
                top_k=request.top_k,
                enable_reranker=request.enable_reranker,
            )
            query_vector = await embedding_provider.embed_query(effective_query)
            cached_response, _similarity = await semantic_cache.lookup(
                query_vector=query_vector,
                scope_hash=scope_hash,
            )
            if cached_response is not None:
                await db.close()
                if effective_query != sanitized_query and cached_response.rewritten_query is None:
                    cached_response.rewritten_query = effective_query

                if request.session_id:
                    await conversation_memory.add_turn(
                        request.session_id, sanitized_query, cached_response.answer
                    )

                async def sse_cached_publisher() -> AsyncGenerator[str, None]:
                    sources_payload = [
                        {
                            "source_id": s.source_id,
                            "document_title": s.document_title,
                            "page": s.page_start,
                            "section": s.section_path,
                            "similarity_score": s.similarity_score,
                        }
                        for s in cached_response.sources
                    ]
                    yield f"data: {json.dumps({'event': 'sources', 'data': sources_payload})}\n\n"
                    token_msg = json.dumps({"event": "token", "content": cached_response.answer})
                    yield f"data: {token_msg}\n\n"
                    citations_payload = [
                        {
                            "source_id": c.source_id,
                            "document_id": str(c.document_id) if c.document_id else None,
                            "document_title": c.document_title,
                            "page_start": c.page_start,
                            "page_end": c.page_end,
                            "section_path": c.section_path,
                            "supporting_quote": c.supporting_quote,
                            "bbox": c.bbox,
                            "metadata": c.metadata,
                        }
                        for c in cached_response.citations
                    ]
                    verification_payload = {
                        "is_answerable": cached_response.is_answerable,
                        "verified_answer": cached_response.answer,
                        "citations": citations_payload,
                        "confidence_score": cached_response.confidence_score,
                        "has_unverified": "[unverified:" in cached_response.answer,
                        "latency_ms": cached_response.latency_ms,
                        "candidate_facts": [
                            f.model_dump() for f in cached_response.candidate_facts
                        ],
                        "execution_trace": [
                            t.model_dump() for t in cached_response.execution_trace
                        ],
                        "route": cached_response.route.value if cached_response.route else None,
                        "crag_retry_count": cached_response.crag_retry_count,
                        "sql_query_executed": cached_response.sql_query_executed,
                        "rewritten_query": cached_response.rewritten_query,
                    }
                    verif_msg = json.dumps({"event": "verification", "data": verification_payload})
                    yield f"data: {verif_msg}\n\n"
                    yield "data: [DONE]\n\n"

                    # Persist streaming query cache-hit telemetry asynchronously
                    try:
                        async with async_session_maker() as stream_db:
                            model_name = getattr(llm_provider, "model_name", "gpt-4o-mini")
                            embedding_tokens = CostEstimator.estimate_tokens(effective_query)
                            cost_usd = CostEstimator.estimate_cost(
                                model=model_name,
                                prompt_tokens=0,
                                completion_tokens=0,
                                embedding_tokens=embedding_tokens,
                            )
                            total_latency_ms = round((time.perf_counter() - request_t0) * 1000.0, 2)
                            log_entry = QueryLogCreate(
                                request_id=get_correlation_id() or str(uuid.uuid4()),
                                session_id=request.session_id,
                                query_text=effective_query,
                                search_mode=request.search_mode,
                                cache_hit=True,
                                is_answerable=cached_response.is_answerable,
                                confidence_score=cached_response.confidence_score,
                                top_reranker_score=None,
                                rewrite_ms=0.0,
                                dense_search_ms=0.0,
                                sparse_search_ms=0.0,
                                rerank_ms=0.0,
                                llm_ttft_ms=None,
                                total_latency_ms=total_latency_ms,
                                prompt_tokens=0,
                                completion_tokens=0,
                                embedding_tokens=embedding_tokens,
                                estimated_cost_usd=cost_usd,
                            )
                            telemetry_repo = TelemetryRepository(stream_db)
                            await telemetry_repo.log_query(log_entry)
                    except Exception as exc:
                        logger.warning("Failed to persist streaming query cache telemetry: %s", exc)

                return StreamingResponse(
                    sse_cached_publisher(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                        "X-Cache-Hit": "true",
                    },
                )
        except Exception as exc:
            logger.warning("Semantic cache lookup encountered error: %s", exc)

    # Multi-agent CRAG state machine execution when requested
    if request.enable_agent or request.route_override is not None:
        orchestrator = AgentOrchestrator(
            db=db,
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            reranker=reranker,
        )
        initial_state = AgentState(
            query=effective_query,
            original_query=effective_query,
            session_id=request.session_id,
            filters=request.filters,
            top_k=request.top_k,
            search_mode=request.search_mode,
            weight_dense=request.weight_dense,
            weight_sparse=request.weight_sparse,
            enable_reranker=request.enable_reranker,
            allow_external_fallback=request.allow_external_fallback,
            route_override=request.route_override,
            max_crag_retries=request.max_crag_retries,
        )
        try:
            agent_response = await orchestrator.run(initial_state)
        finally:
            await db.close()

        total_latency_ms = round((time.perf_counter() - request_t0) * 1000.0, 2)
        agent_response.latency_ms = total_latency_ms

        if effective_query != sanitized_query and agent_response.rewritten_query is None:
            agent_response.rewritten_query = effective_query

        async def sse_agent_publisher() -> AsyncGenerator[str, None]:
            sources_payload = [
                {
                    "source_id": s.source_id,
                    "document_title": s.document_title,
                    "page": s.page_start,
                    "section": s.section_path,
                    "similarity_score": s.similarity_score,
                }
                for s in agent_response.sources
            ]
            yield f"data: {json.dumps({'event': 'sources', 'data': sources_payload})}\n\n"
            token_msg = json.dumps({"event": "token", "content": agent_response.answer})
            yield f"data: {token_msg}\n\n"
            citations_payload = [
                {
                    "source_id": c.source_id,
                    "document_id": str(c.document_id) if c.document_id else None,
                    "document_title": c.document_title,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "section_path": c.section_path,
                    "supporting_quote": c.supporting_quote,
                    "bbox": c.bbox,
                    "metadata": c.metadata,
                }
                for c in agent_response.citations
            ]
            verification_payload = {
                "is_answerable": agent_response.is_answerable,
                "verified_answer": agent_response.answer,
                "citations": citations_payload,
                "confidence_score": agent_response.confidence_score,
                "has_unverified": "[unverified:" in agent_response.answer,
                "latency_ms": agent_response.latency_ms,
                "candidate_facts": [f.model_dump() for f in agent_response.candidate_facts],
                "execution_trace": [t.model_dump() for t in agent_response.execution_trace],
                "route": agent_response.route.value if agent_response.route else None,
                "crag_retry_count": agent_response.crag_retry_count,
                "sql_query_executed": agent_response.sql_query_executed,
                "rewritten_query": agent_response.rewritten_query,
            }
            verif_msg = json.dumps({"event": "verification", "data": verification_payload})
            yield f"data: {verif_msg}\n\n"
            yield "data: [DONE]\n\n"

            if request.session_id:
                try:
                    await conversation_memory.add_turn(
                        request.session_id, sanitized_query, agent_response.answer
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to save agent stream turn to conversational memory: %s", exc
                    )

            if semantic_cache is not None:
                try:
                    sc_hash = semantic_cache.canonical_scope_hash(
                        search_mode=request.search_mode,
                        filters=request.filters,
                        top_k=request.top_k,
                        enable_reranker=request.enable_reranker,
                    )
                    q_vec = await embedding_provider.embed_query(effective_query)
                    await semantic_cache.store(
                        query_vector=q_vec,
                        scope_hash=sc_hash,
                        response=agent_response,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to store agent stream query response to semantic cache: %s", exc
                    )

            try:
                async with async_session_maker() as stream_db:
                    model_name = getattr(llm_provider, "model_name", "gpt-4o-mini")
                    generator_helper = GroundedGenerator(llm_provider=llm_provider)
                    context_xml = (
                        generator_helper.build_context_xml(agent_response.sources)
                        if agent_response.sources
                        else ""
                    )
                    prompt_text = f"{context_xml}\n{effective_query}"
                    prompt_tokens = CostEstimator.estimate_tokens(prompt_text, model=model_name)
                    completion_tokens = CostEstimator.estimate_tokens(
                        agent_response.answer, model=model_name
                    )
                    embedding_tokens = (
                        CostEstimator.estimate_tokens(effective_query)
                        if request.search_mode in ("hybrid", "dense")
                        else 0
                    )
                    cost_usd = CostEstimator.estimate_cost(
                        model=model_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        embedding_tokens=embedding_tokens,
                    )

                    total_latency_ms = agent_response.latency_ms
                    top_reranker_score: float | None = None
                    if agent_response.sources:
                        rerank_scores = [
                            c.rerank_score
                            for c in agent_response.sources
                            if c.rerank_score is not None
                        ]
                        if rerank_scores:
                            top_reranker_score = max(rerank_scores)

                    log_entry = QueryLogCreate(
                        request_id=get_correlation_id() or str(uuid.uuid4()),
                        session_id=request.session_id,
                        query_text=effective_query,
                        search_mode=request.search_mode,
                        cache_hit=False,
                        is_answerable=agent_response.is_answerable,
                        confidence_score=agent_response.confidence_score,
                        top_reranker_score=top_reranker_score,
                        rewrite_ms=0.0,
                        dense_search_ms=0.0,
                        sparse_search_ms=0.0,
                        rerank_ms=0.0,
                        llm_ttft_ms=None,
                        total_latency_ms=total_latency_ms,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        embedding_tokens=embedding_tokens,
                        estimated_cost_usd=cost_usd,
                    )
                    telemetry_repo = TelemetryRepository(stream_db)
                    await telemetry_repo.log_query(log_entry)
            except Exception as exc:
                logger.warning("Failed to persist agent streaming query telemetry: %s", exc)

        return StreamingResponse(
            sse_agent_publisher(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Cache-Hit": "false",
            },
        )

    # Standard vector search pipeline
    search_service = VectorSearchService(
        session=db,
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        reranker=reranker,
    )
    try:
        chunks = await search_service.search(
            query=effective_query,
            filters=request.filters,
            top_k=request.top_k,
            search_mode=request.search_mode,
            weight_dense=request.weight_dense,
            weight_sparse=request.weight_sparse,
            enable_reranker=request.enable_reranker,
            enable_query_rewrite=request.enable_query_rewrite,
            enable_hyde=request.enable_hyde,
        )
    finally:
        await db.close()

    stage_latencies = search_service.last_stage_latencies
    top_reranker_score: float | None = None
    if chunks:
        rerank_scores = [c.rerank_score for c in chunks if c.rerank_score is not None]
        if rerank_scores:
            top_reranker_score = max(rerank_scores)

    generator = GroundedGenerator(llm_provider=llm_provider)

    async def sse_event_publisher() -> AsyncGenerator[str, None]:
        sources_payload = [
            {
                "source_id": s.source_id,
                "document_title": s.document_title,
                "page": s.page_start,
                "section": s.section_path,
                "similarity_score": s.similarity_score,
            }
            for s in chunks
        ]
        yield f"data: {json.dumps({'event': 'sources', 'data': sources_payload})}\n\n"

        accumulated_tokens: list[str] = []
        first_token_time: float | None = None

        async for token in generator.generate_stream(query=effective_query, chunks=chunks):
            if first_token_time is None:
                first_token_time = time.perf_counter()
            accumulated_tokens.append(token)
            token_msg = json.dumps({"event": "token", "content": token})
            yield f"data: {token_msg}\n\n"

        raw_text = "".join(accumulated_tokens)
        rag_response = generator.finalize_stream_response(raw_text, chunks, request_t0)
        if effective_query != sanitized_query and rag_response.rewritten_query is None:
            rag_response.rewritten_query = effective_query

        citations_payload = [
            {
                "source_id": c.source_id,
                "document_id": str(c.document_id) if c.document_id else None,
                "document_title": c.document_title,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "section_path": c.section_path,
                "supporting_quote": c.supporting_quote,
                "bbox": c.bbox,
                "metadata": c.metadata,
            }
            for c in rag_response.citations
        ]

        verification_payload = {
            "is_answerable": rag_response.is_answerable,
            "verified_answer": rag_response.answer,
            "citations": citations_payload,
            "confidence_score": rag_response.confidence_score,
            "has_unverified": "[unverified:" in rag_response.answer,
            "latency_ms": rag_response.latency_ms,
            "candidate_facts": [f.model_dump() for f in rag_response.candidate_facts],
            "execution_trace": [t.model_dump() for t in rag_response.execution_trace],
            "route": rag_response.route.value if rag_response.route else None,
            "crag_retry_count": rag_response.crag_retry_count,
            "sql_query_executed": rag_response.sql_query_executed,
            "rewritten_query": rag_response.rewritten_query,
        }
        verif_msg = json.dumps({"event": "verification", "data": verification_payload})
        yield f"data: {verif_msg}\n\n"
        yield "data: [DONE]\n\n"

        if request.session_id:
            try:
                await conversation_memory.add_turn(
                    request.session_id, sanitized_query, rag_response.answer
                )
            except Exception as exc:
                logger.warning("Failed to save stream turn to conversational memory: %s", exc)

        if semantic_cache is not None:
            try:
                sc_hash = semantic_cache.canonical_scope_hash(
                    search_mode=request.search_mode,
                    filters=request.filters,
                    top_k=request.top_k,
                    enable_reranker=request.enable_reranker,
                )
                q_vec = await embedding_provider.embed_query(effective_query)
                await semantic_cache.store(
                    query_vector=q_vec,
                    scope_hash=sc_hash,
                    response=rag_response,
                )
            except Exception as exc:
                logger.warning("Failed to store stream query response to semantic cache: %s", exc)

        try:
            async with async_session_maker() as stream_db:
                ttft_ms = (
                    (first_token_time - request_t0) * 1000.0
                    if first_token_time is not None
                    else None
                )
                model_name = getattr(llm_provider, "model_name", "gpt-4o-mini")
                context_xml = generator.build_context_xml(chunks) if chunks else ""
                prompt_text = f"{context_xml}\n{effective_query}"
                prompt_tokens = CostEstimator.estimate_tokens(prompt_text, model=model_name)
                completion_tokens = CostEstimator.estimate_tokens(
                    rag_response.answer, model=model_name
                )
                embedding_tokens = (
                    CostEstimator.estimate_tokens(effective_query)
                    if request.search_mode in ("hybrid", "dense")
                    else 0
                )
                cost_usd = CostEstimator.estimate_cost(
                    model=model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    embedding_tokens=embedding_tokens,
                )

                total_latency_ms = rag_response.latency_ms
                log_entry = QueryLogCreate(
                    request_id=get_correlation_id() or str(uuid.uuid4()),
                    session_id=request.session_id,
                    query_text=effective_query,
                    search_mode=request.search_mode,
                    cache_hit=False,
                    is_answerable=rag_response.is_answerable,
                    confidence_score=rag_response.confidence_score,
                    top_reranker_score=top_reranker_score,
                    rewrite_ms=stage_latencies.get("rewrite_ms", 0.0),
                    dense_search_ms=stage_latencies.get("dense_search_ms", 0.0),
                    sparse_search_ms=stage_latencies.get("sparse_search_ms", 0.0),
                    rerank_ms=stage_latencies.get("rerank_ms", 0.0),
                    llm_ttft_ms=round(ttft_ms, 2) if ttft_ms is not None else None,
                    total_latency_ms=total_latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    embedding_tokens=embedding_tokens,
                    estimated_cost_usd=cost_usd,
                )
                telemetry_repo = TelemetryRepository(stream_db)
                await telemetry_repo.log_query(log_entry)
        except Exception as exc:
            logger.warning("Failed to persist streaming query telemetry: %s", exc)

    return StreamingResponse(
        sse_event_publisher(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Cache-Hit": "false",
        },
    )

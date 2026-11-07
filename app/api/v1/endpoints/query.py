"""Query endpoints for grounded answers and real-time Server-Sent Events (SSE) streaming."""

import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, BackgroundTasks, Depends, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import (
    get_db,
    get_embedding,
    get_llm,
    get_reranker,
    get_semantic_cache,
)
from app.core.logging import get_correlation_id
from app.db.repositories.telemetry import TelemetryRepository
from app.db.session import async_session_maker
from app.schemas.metrics import QueryLogCreate
from app.schemas.query import QueryRequest, RAGResponse
from app.services.cache.semantic_cache import SemanticCacheService
from app.services.providers.base import EmbeddingProvider, LLMProvider
from app.services.retrieval.generator import GroundedGenerator
from app.services.retrieval.reranker import CrossEncoderReranker
from app.services.retrieval.vector_search import VectorSearchService
from app.services.telemetry.cost import CostEstimator

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=RAGResponse)
async def query_rag(
    request: QueryRequest,
    response: Response,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    embedding_provider: EmbeddingProvider = Depends(get_embedding),
    llm_provider: LLMProvider = Depends(get_llm),
    reranker: CrossEncoderReranker | None = Depends(get_reranker),
    semantic_cache: SemanticCacheService | None = Depends(get_semantic_cache),
) -> RAGResponse:
    """Execute vector search and generate grounded answer with citations and confidence score."""
    start_time = time.perf_counter()
    query_vector: list[float] | None = None
    scope_hash: str | None = None

    if semantic_cache is not None:
        try:
            scope_hash = semantic_cache.canonical_scope_hash(
                search_mode=request.search_mode,
                filters=request.filters,
                top_k=request.top_k,
                enable_reranker=request.enable_reranker,
            )
            query_vector = await embedding_provider.embed_query(request.query)
            cached_response, _similarity = await semantic_cache.lookup(
                query_vector=query_vector,
                scope_hash=scope_hash,
            )
            if cached_response is not None:
                response.headers["X-Cache-Hit"] = "true"
                total_latency_ms = (time.perf_counter() - start_time) * 1000.0
                cached_response.latency_ms = round(total_latency_ms, 2)

                model_name = getattr(llm_provider, "model_name", "gpt-4o-mini")
                embedding_tokens = CostEstimator.estimate_tokens(request.query)
                cost_usd = CostEstimator.estimate_cost(
                    model=model_name,
                    prompt_tokens=0,
                    completion_tokens=0,
                    embedding_tokens=embedding_tokens,
                )

                log_entry = QueryLogCreate(
                    request_id=get_correlation_id() or str(uuid.uuid4()),
                    query_text=request.query,
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

    search_service = VectorSearchService(
        session=db,
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        reranker=reranker,
    )
    chunks = await search_service.search(
        query=request.query,
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
    rag_response = await generator.generate(query=request.query, chunks=chunks)
    total_latency_ms = (time.perf_counter() - start_time) * 1000.0

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
                query_vector = await embedding_provider.embed_query(request.query)
            background_tasks.add_task(
                semantic_cache.store,
                query_vector=query_vector,
                scope_hash=scope_hash,
                response=rag_response,
            )
        except Exception as exc:
            logger.warning("Failed to schedule semantic cache store: %s", exc)

    # Extract stage latencies and top reranker score
    stage_latencies = search_service.last_stage_latencies
    top_reranker_score: float | None = None
    if chunks:
        rerank_scores = [c.rerank_score for c in chunks if c.rerank_score is not None]
        if rerank_scores:
            top_reranker_score = max(rerank_scores)

    model_name = getattr(llm_provider, "model_name", "gpt-4o-mini")
    context_xml = generator.build_context_xml(chunks) if chunks else ""
    prompt_text = f"{context_xml}\n{request.query}"
    prompt_tokens = CostEstimator.estimate_tokens(prompt_text, model=model_name)
    completion_tokens = CostEstimator.estimate_tokens(rag_response.answer, model=model_name)
    embedding_tokens = (
        CostEstimator.estimate_tokens(request.query)
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
        query_text=request.query,
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
    db: AsyncSession = Depends(get_db),
    embedding_provider: EmbeddingProvider = Depends(get_embedding),
    llm_provider: LLMProvider = Depends(get_llm),
    reranker: CrossEncoderReranker | None = Depends(get_reranker),
    semantic_cache: SemanticCacheService | None = Depends(get_semantic_cache),
) -> StreamingResponse:
    """Stream grounded answer tokens via Server-Sent Events (SSE)."""
    request_t0 = time.perf_counter()

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
            query_vector = await embedding_provider.embed_query(request.query)
            cached_response, _similarity = await semantic_cache.lookup(
                query_vector=query_vector,
                scope_hash=scope_hash,
            )
            if cached_response is not None:
                await db.close()

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
                    }
                    verif_msg = json.dumps({"event": "verification", "data": verification_payload})
                    yield f"data: {verif_msg}\n\n"
                    yield "data: [DONE]\n\n"

                    # Persist streaming query cache-hit telemetry asynchronously
                    try:
                        async with async_session_maker() as stream_db:
                            model_name = getattr(llm_provider, "model_name", "gpt-4o-mini")
                            embedding_tokens = CostEstimator.estimate_tokens(request.query)
                            cost_usd = CostEstimator.estimate_cost(
                                model=model_name,
                                prompt_tokens=0,
                                completion_tokens=0,
                                embedding_tokens=embedding_tokens,
                            )
                            total_latency_ms = round((time.perf_counter() - request_t0) * 1000.0, 2)
                            log_entry = QueryLogCreate(
                                request_id=get_correlation_id() or str(uuid.uuid4()),
                                query_text=request.query,
                                search_mode=request.search_mode,
                                cache_hit=True,
                                is_answerable=cached_response.is_answerable,
                                confidence_score=cached_response.confidence_score,
                                top_reranker_score=None,
                                rewrite_ms=0.0,
                                dense_search_ms=0.0,
                                sparse_search_ms=0.0,
                                rerank_ms=0.0,
                                llm_ttft_ms=total_latency_ms,
                                total_latency_ms=total_latency_ms,
                                prompt_tokens=0,
                                completion_tokens=0,
                                embedding_tokens=embedding_tokens,
                                estimated_cost_usd=cost_usd,
                            )
                            telemetry_repo = TelemetryRepository(stream_db)
                            await telemetry_repo.log_query(log_entry)
                    except Exception as exc:
                        logger.warning(
                            "Failed to persist streaming query cache hit telemetry: %s", exc
                        )

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
            logger.warning("Redis error during streaming semantic cache lookup: %s", exc)

    search_service = VectorSearchService(
        session=db,
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        reranker=reranker,
    )
    chunks = await search_service.search(
        query=request.query,
        filters=request.filters,
        top_k=request.top_k,
        search_mode=request.search_mode,
        weight_dense=request.weight_dense,
        weight_sparse=request.weight_sparse,
        enable_reranker=request.enable_reranker,
        enable_query_rewrite=request.enable_query_rewrite,
        enable_hyde=request.enable_hyde,
    )
    stage_latencies = dict(search_service.last_stage_latencies)
    top_reranker_score: float | None = None
    if chunks:
        rerank_scores = [c.rerank_score for c in chunks if c.rerank_score is not None]
        if rerank_scores:
            top_reranker_score = max(rerank_scores)

    # Release DB connection immediately so connection pool is not starved during LLM streaming
    await db.close()

    generator = GroundedGenerator(llm_provider=llm_provider)
    stream_gen = generator.generate_stream(query=request.query, chunks=chunks)

    async def sse_event_publisher() -> AsyncGenerator[str, None]:
        nonlocal query_vector, scope_hash
        start_time = time.perf_counter()
        # Initial event emitting retrieved sources metadata
        sources_payload = [
            {
                "source_id": c.source_id,
                "document_title": c.document_title,
                "page": c.page_start,
                "section": c.section_path,
                "similarity_score": c.similarity_score,
            }
            for c in chunks
        ]
        yield f"data: {json.dumps({'event': 'sources', 'data': sources_payload})}\n\n"

        # Stream generated response tokens while accumulating full answer
        accumulated_tokens: list[str] = []
        ttft_ms: float | None = None
        async for token in stream_gen:
            if ttft_ms is None and len(accumulated_tokens) == 0:
                ttft_ms = (time.perf_counter() - start_time) * 1000.0
            accumulated_tokens.append(token)
            yield f"data: {json.dumps({'event': 'token', 'content': token})}\n\n"

        # Post-generation citation and evidence verification
        full_raw_text = "".join(accumulated_tokens)
        rag_response = generator.finalize_stream_response(full_raw_text, chunks, start_time)

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
        }
        yield f"data: {json.dumps({'event': 'verification', 'data': verification_payload})}\n\n"
        yield "data: [DONE]\n\n"

        # Persist response to semantic cache asynchronously
        if semantic_cache is not None:
            try:
                if query_vector is None:
                    query_vector = await embedding_provider.embed_query(request.query)
                if scope_hash is None:
                    scope_hash = semantic_cache.canonical_scope_hash(
                        search_mode=request.search_mode,
                        filters=request.filters,
                        top_k=request.top_k,
                        enable_reranker=request.enable_reranker,
                    )
                await semantic_cache.store(
                    query_vector=query_vector,
                    scope_hash=scope_hash,
                    response=rag_response,
                )
            except Exception as exc:
                logger.warning("Failed to store streaming query in semantic cache: %s", exc)

        # Persist streaming query telemetry asynchronously
        try:
            async with async_session_maker() as stream_db:
                model_name = getattr(llm_provider, "model_name", "gpt-4o-mini")
                context_xml = generator.build_context_xml(chunks) if chunks else ""
                prompt_text = f"{context_xml}\n{request.query}"
                prompt_tokens = CostEstimator.estimate_tokens(prompt_text, model=model_name)
                completion_tokens = CostEstimator.estimate_tokens(
                    rag_response.answer, model=model_name
                )
                embedding_tokens = (
                    CostEstimator.estimate_tokens(request.query)
                    if request.search_mode in ("hybrid", "dense")
                    else 0
                )

                cost_usd = CostEstimator.estimate_cost(
                    model=model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    embedding_tokens=embedding_tokens,
                )

                total_latency_ms = round((time.perf_counter() - request_t0) * 1000.0, 2)
                log_entry = QueryLogCreate(
                    request_id=get_correlation_id() or str(uuid.uuid4()),
                    query_text=request.query,
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
        },
    )

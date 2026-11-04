"""Query endpoints for grounded answers and real-time Server-Sent Events (SSE) streaming."""

import json
import time
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db, get_embedding, get_llm
from app.schemas.query import QueryRequest, RAGResponse
from app.services.providers.base import EmbeddingProvider, LLMProvider
from app.services.retrieval.generator import GroundedGenerator
from app.services.retrieval.vector_search import VectorSearchService

router = APIRouter()


@router.post("/query", response_model=RAGResponse)
async def query_rag(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
    embedding_provider: EmbeddingProvider = Depends(get_embedding),
    llm_provider: LLMProvider = Depends(get_llm),
) -> RAGResponse:
    """Execute vector search and generate grounded answer with citations and confidence score."""
    search_service = VectorSearchService(
        session=db,
        embedding_provider=embedding_provider,
    )
    chunks = await search_service.search(
        query=request.query,
        filters=request.filters,
        top_k=request.top_k,
    )

    generator = GroundedGenerator(llm_provider=llm_provider)
    return await generator.generate(query=request.query, chunks=chunks)


@router.post("/query/stream")
async def stream_query_rag(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
    embedding_provider: EmbeddingProvider = Depends(get_embedding),
    llm_provider: LLMProvider = Depends(get_llm),
) -> StreamingResponse:
    """Stream grounded answer tokens via Server-Sent Events (SSE)."""
    search_service = VectorSearchService(
        session=db,
        embedding_provider=embedding_provider,
    )
    chunks = await search_service.search(
        query=request.query,
        filters=request.filters,
        top_k=request.top_k,
    )

    generator = GroundedGenerator(llm_provider=llm_provider)
    stream_gen = generator.generate_stream(query=request.query, chunks=chunks)

    async def sse_event_publisher() -> AsyncGenerator[str, None]:
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
        async for token in stream_gen:
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
            }
            for c in rag_response.citations
        ]
        verification_payload = {
            "is_answerable": rag_response.is_answerable,
            "citations": citations_payload,
            "confidence_score": rag_response.confidence_score,
            "has_unverified": "[unverified:" in rag_response.answer,
            "latency_ms": rag_response.latency_ms,
        }
        yield f"data: {json.dumps({'event': 'verification', 'data': verification_payload})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        sse_event_publisher(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

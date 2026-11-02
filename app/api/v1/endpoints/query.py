"""Query endpoints for grounded answers and real-time Server-Sent Events (SSE) streaming."""

import json
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

        # Stream generated response tokens
        async for token in stream_gen:
            yield f"data: {json.dumps({'event': 'token', 'content': token})}\n\n"

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

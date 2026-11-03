"""Pydantic schemas for search query requests, retrieved chunks, and RAG responses."""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class QueryFilters(BaseModel):
    """Metadata pre-filters applied during vector search."""

    document_id: UUID | None = None
    category: str | None = None
    element_types: list[str] | None = None
    page_min: int | None = None
    page_max: int | None = None
    min_similarity: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity cutoff floor (0.0 to 1.0)",
    )


class RetrievedChunk(BaseModel):
    """Encapsulates a retrieved chunk enriched with similarity score and citation metadata."""

    source_id: int = Field(
        description="1-based ordinal source index for inline citations [1], [2], etc."
    )
    chunk_id: UUID
    document_id: UUID
    document_title: str
    page_start: int
    page_end: int
    section_path: str
    element_type: str
    text: str
    contextualized_text: str
    similarity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Cosine similarity score (1 - cosine distance)",
    )


class QueryRequest(BaseModel):
    """Input payload for RAG query endpoints."""

    query: str = Field(min_length=3, description="User question or search query")
    filters: QueryFilters | None = None
    top_k: int = Field(default=5, ge=1, le=20, description="Max chunks to retrieve")
    stream: bool = Field(
        default=False, description="Whether to stream the generated answer via SSE"
    )


class Citation(BaseModel):
    """Ground truth citation linking an answer statement to a retrieved source."""

    source_id: int
    document_id: UUID | None = None
    document_title: str
    page_start: int
    page_end: int
    section_path: str
    supporting_quote: str | None = None
    quote: str | None = None

    def model_post_init(self, __context: Any) -> None:
        if self.supporting_quote is None and self.quote is not None:
            self.supporting_quote = self.quote
        elif self.quote is None and self.supporting_quote is not None:
            self.quote = self.supporting_quote


class RAGResponse(BaseModel):
    """Final grounded RAG answer with verified citations and metrics."""

    answer: str
    is_answerable: bool = Field(
        default=True,
        description="Whether the retrieved context provides sufficient evidence",
    )
    citations: list[Citation] = Field(default_factory=list)
    sources: list[RetrievedChunk] = Field(default_factory=list)
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Aggregate confidence score based on retrieval similarity and citation density",
    )
    latency_ms: float = Field(description="Total query and generation latency in milliseconds")

"""Pydantic schemas for search query requests, retrieved chunks, and RAG responses."""

import contextlib
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class IntentRoute(StrEnum):
    """5-tier semantic intent routes for query dispatch."""

    CHITCHAT_OR_OUT_OF_DOMAIN = "CHITCHAT_OR_OUT_OF_DOMAIN"
    FAST_TRACK_HYBRID_RAG = "FAST_TRACK_HYBRID_RAG"
    MULTI_HOP_AGENTIC_RAG = "MULTI_HOP_AGENTIC_RAG"
    STRUCTURED_SQL_ANALYTICS = "STRUCTURED_SQL_ANALYTICS"
    HYBRID_SQL_AND_RAG = "HYBRID_SQL_AND_RAG"


class ExtractionStatus(StrEnum):
    """Granular verification status for candidate facts extracted from responses."""

    VERIFIED = "verified"
    NOT_FOUND = "not_found"
    EXTRACTION_FAILED = "extraction_failed"
    AMBIGUOUS = "ambiguous"
    CONFLICTING = "conflicting"
    UNSUPPORTED = "unsupported"


class CitationSourceType(StrEnum):
    """Origin source type for citation segregation."""

    DOCUMENT = "document"
    SQL_TABLE = "sql_table"
    EXTERNAL_WEB = "external_web"


class AgentTraceStep(BaseModel):
    """Step in the agent execution trace recording node actions, observations, and latency."""

    step: str = ""
    action: str = ""
    step_name: str = ""
    detail: str = ""
    status: str = "completed"
    input: Any = None
    output: Any = None
    duration_ms: float | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if not self.step and self.step_name:
            self.step = self.step_name
        elif not self.step_name and self.step:
            self.step_name = self.step
        if not self.action and self.detail:
            self.action = self.detail
        elif not self.detail and self.action:
            self.detail = self.action
        if self.duration_ms is None and self.latency_ms > 0:
            self.duration_ms = self.latency_ms
        elif self.duration_ms is not None and self.latency_ms == 0.0:
            self.latency_ms = self.duration_ms


class CandidateFact(BaseModel):
    """Structured atomic candidate fact extracted from answers with verification status."""

    fact: str = ""
    claim_text: str = ""
    status: ExtractionStatus = ExtractionStatus.VERIFIED
    evidence: str | None = None
    supporting_quote: str | None = None
    source_id: int | None = None
    source_ids: list[int] = Field(default_factory=list)
    document_title: str | None = None
    page_number: int | None = None
    bbox: dict[str, Any] | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    notes: str | None = None

    def model_post_init(self, __context: Any) -> None:
        if not self.fact and self.claim_text:
            self.fact = self.claim_text
        elif not self.claim_text and self.fact:
            self.claim_text = self.fact
        if self.evidence is None and self.supporting_quote is not None:
            self.evidence = self.supporting_quote
        elif self.supporting_quote is None and self.evidence is not None:
            self.supporting_quote = self.evidence
        if self.source_id is None and self.source_ids:
            self.source_id = self.source_ids[0]
        elif self.source_id is not None and not self.source_ids:
            self.source_ids = [self.source_id]


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
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    parent_chunk_id: UUID | None = None
    token_count: int | None = None
    bbox: dict[str, Any] | None = None
    source_type: CitationSourceType = CitationSourceType.DOCUMENT
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """Input payload for RAG query endpoints."""

    query: str = Field(min_length=3, description="User question or search query")
    filters: QueryFilters | None = None
    top_k: int = Field(default=5, ge=1, le=20, description="Max chunks to retrieve")
    stream: bool = Field(
        default=False, description="Whether to stream the generated answer via SSE"
    )
    search_mode: Literal["hybrid", "dense", "sparse"] = "hybrid"
    weight_dense: float | None = Field(default=None, ge=0.0, le=1.0)
    weight_sparse: float | None = Field(default=None, ge=0.0, le=1.0)
    enable_reranker: bool | None = None
    enable_query_rewrite: bool = False
    enable_hyde: bool = False

    # Phase 4 additions with full backward compatibility defaults
    session_id: str | None = Field(
        default=None,
        max_length=128,
        description="Optional conversational session ID for multi-turn history",
    )
    enable_agent: bool = Field(
        default=False,
        description="Whether to route through the multi-agent CRAG state machine",
    )
    enable_agentic: bool = Field(
        default=False,
        description="Alias for enable_agent (Phase 4 Agentic Router, CRAG loop, and SQL tools)",
    )
    allow_external_fallback: bool = Field(
        default=False,
        description="Whether to permit fallback to external web search when corpus lacks evidence",
    )
    max_crag_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum corrective RAG query reformulation retries",
    )
    route_override: IntentRoute | None = Field(
        default=None,
        description="Optional manual override for intent router",
    )
    resolve_parents: bool = Field(
        default=True,
        description="Whether to resolve child chunks to larger parent chunks for context expansion",
    )
    bypass_cache: bool = Field(
        default=False,
        description="Whether to bypass semantic cache lookup and execute a fresh query",
    )

    def model_post_init(self, __context: Any) -> None:
        if self.enable_agentic and not self.enable_agent:
            self.enable_agent = True
        elif self.enable_agent and not self.enable_agentic:
            self.enable_agentic = True


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
    bbox: dict[str, Any] | None = None
    source_type: CitationSourceType = CitationSourceType.DOCUMENT
    match_tier: str | None = None
    rehomed_from: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

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

    # Phase 4 additions with full backward compatibility defaults
    route: IntentRoute | None = Field(
        default=None,
        description="Routing classification decided by intent router",
    )
    route_taken: IntentRoute | str = Field(
        default="FAST_TRACK_HYBRID_RAG",
        description="Route taken by router or orchestrator",
    )
    rewritten_query: str | None = None
    crag_retries: int = Field(
        default=0,
        description="Number of CRAG query reformulation retries executed",
    )
    sql_query_executed: str | None = None
    candidate_facts: list[CandidateFact] = Field(
        default_factory=list,
        description="Structured candidate facts extracted and verified from the answer",
    )
    execution_trace: list[AgentTraceStep] = Field(
        default_factory=list,
        description="State machine trace of agent steps, decisions, and timings",
    )
    crag_retry_count: int = Field(
        default=0,
        description="Number of CRAG query reformulation retries executed",
    )
    stage_latencies: dict[str, float] | None = Field(
        default=None,
        description="Latency breakdown by retrieval and processing stages in milliseconds",
    )

    def model_post_init(self, __context: Any) -> None:
        if self.route is not None:
            self.route_taken = self.route
        elif self.route_taken and self.route_taken != "FAST_TRACK_HYBRID_RAG":
            with contextlib.suppress(Exception):
                self.route = IntentRoute(self.route_taken)
        if self.crag_retry_count > 0 and self.crag_retries == 0:
            self.crag_retries = self.crag_retry_count
        elif self.crag_retries > 0 and self.crag_retry_count == 0:
            self.crag_retry_count = self.crag_retries

"""Pydantic schemas for query telemetry, cost accounting, and metrics summary."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class QueryLogCreate(BaseModel):
    """Payload for logging query execution metrics."""

    request_id: str = Field(description="Unique correlation ID for the request")
    session_id: str | None = Field(default=None, description="Optional client/session identifier")
    query_text: str = Field(description="The input user query")
    search_mode: str = Field(
        default="hybrid", description="Retrieval mode: hybrid, dense, or sparse"
    )
    cache_hit: bool = Field(default=False, description="Whether the answer was served from cache")
    is_answerable: bool = Field(
        default=True, description="Whether the question was deemed answerable"
    )
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    top_reranker_score: float | None = Field(default=None)

    # Latencies in milliseconds
    rewrite_ms: float = Field(default=0.0, ge=0.0)
    dense_search_ms: float = Field(default=0.0, ge=0.0)
    sparse_search_ms: float = Field(default=0.0, ge=0.0)
    rerank_ms: float = Field(default=0.0, ge=0.0)
    llm_ttft_ms: float | None = Field(default=None, ge=0.0)
    total_latency_ms: float = Field(default=0.0, ge=0.0)

    # Token counts
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    embedding_tokens: int = Field(default=0, ge=0)

    # Estimated cost in USD
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)


class QueryLogRead(QueryLogCreate):
    """Schema for returning query log database records."""

    id: UUID
    created_at: datetime

    model_config = {"from_attributes": True}


class LatencyPercentiles(BaseModel):
    """P50, P95, and P99 latency summary in milliseconds."""

    p50_ms: float = Field(default=0.0, description="50th percentile total latency")
    p95_ms: float = Field(default=0.0, description="95th percentile total latency")
    p99_ms: float = Field(default=0.0, description="99th percentile total latency")


class StageLatencyBreakdown(BaseModel):
    """Average latency across individual pipeline stages in milliseconds."""

    rewrite_ms: float = Field(default=0.0, description="Query rewrite / HyDE latency")
    dense_search_ms: float = Field(default=0.0, description="Embedding + pgvector dense search")
    sparse_search_ms: float = Field(default=0.0, description="BM25 TSVECTOR sparse search")
    rerank_ms: float = Field(default=0.0, description="Cross-Encoder neural rerank latency")
    llm_ttft_ms: float | None = Field(default=None, description="Time to first token from LLM")


class MetricsSummaryResponse(BaseModel):
    """System-wide telemetry and performance summary metrics."""

    total_queries: int = Field(default=0, description="Total number of queries executed")
    cache_hit_ratio: float = Field(
        default=0.0, description="Proportion of queries served from cache"
    )
    abstention_rate: float = Field(default=0.0, description="Proportion of unanswerable queries")
    total_cost_usd: float = Field(default=0.0, description="Total estimated cost in USD")
    avg_cost_per_1k_usd: float = Field(
        default=0.0, description="Average estimated cost per 1,000 queries in USD"
    )
    latencies: LatencyPercentiles = Field(default_factory=LatencyPercentiles)
    stage_latencies_avg: StageLatencyBreakdown = Field(default_factory=StageLatencyBreakdown)

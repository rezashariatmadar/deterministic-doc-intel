"""SQLAlchemy model for query execution telemetry and cost accounting."""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class QueryLog(Base):
    """Stores per-query execution metrics, stage latency breakdowns, token usage, and cost."""

    __tablename__ = "query_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    request_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="Unique correlation request ID",
    )
    session_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        comment="Optional user/client session ID",
    )
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    search_mode: Mapped[str] = mapped_column(
        String(20),
        default="hybrid",
        nullable=False,
        comment="hybrid | dense | sparse",
    )
    cache_hit: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
    )
    is_answerable: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )
    confidence_score: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
    )
    top_reranker_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    # Stage latencies in milliseconds
    rewrite_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    dense_search_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    sparse_search_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    rerank_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    llm_ttft_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_latency_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # Token counts
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    embedding_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Cost accounting
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

"""Repository pattern for telemetry and query log database operations."""

from datetime import datetime

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.query_log import QueryLog
from app.schemas.metrics import (
    LatencyPercentiles,
    MetricsSummaryResponse,
    QueryLogCreate,
    StageLatencyBreakdown,
)


class TelemetryRepository:
    """Encapsulates all database persistence and aggregation for telemetry query logs."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def log_query(self, log_entry: QueryLogCreate) -> QueryLog:
        """Persist a new QueryLog record."""
        log = QueryLog(
            request_id=log_entry.request_id,
            session_id=log_entry.session_id,
            query_text=log_entry.query_text,
            search_mode=log_entry.search_mode,
            cache_hit=log_entry.cache_hit,
            is_answerable=log_entry.is_answerable,
            confidence_score=log_entry.confidence_score,
            top_reranker_score=log_entry.top_reranker_score,
            rewrite_ms=log_entry.rewrite_ms,
            dense_search_ms=log_entry.dense_search_ms,
            sparse_search_ms=log_entry.sparse_search_ms,
            rerank_ms=log_entry.rerank_ms,
            llm_ttft_ms=log_entry.llm_ttft_ms,
            total_latency_ms=log_entry.total_latency_ms,
            prompt_tokens=log_entry.prompt_tokens,
            completion_tokens=log_entry.completion_tokens,
            embedding_tokens=log_entry.embedding_tokens,
            estimated_cost_usd=log_entry.estimated_cost_usd,
        )
        self.session.add(log)
        await self.session.commit()
        await self.session.refresh(log)
        return log

    async def get_summary_metrics(
        self, start_time: datetime | None = None
    ) -> MetricsSummaryResponse:
        """Compute aggregated latency percentiles, hit ratios, and stage latency metrics."""
        stmt = select(
            func.count(QueryLog.id),
            func.count(case((QueryLog.cache_hit.is_(True), 1))),
            func.count(case((QueryLog.is_answerable.is_(False), 1))),
            func.coalesce(func.sum(QueryLog.estimated_cost_usd), 0.0),
            func.coalesce(func.percentile_cont(0.50).within_group(QueryLog.total_latency_ms), 0.0),
            func.coalesce(func.percentile_cont(0.95).within_group(QueryLog.total_latency_ms), 0.0),
            func.coalesce(func.percentile_cont(0.99).within_group(QueryLog.total_latency_ms), 0.0),
            func.coalesce(func.avg(QueryLog.rewrite_ms), 0.0),
            func.coalesce(func.avg(QueryLog.dense_search_ms), 0.0),
            func.coalesce(func.avg(QueryLog.sparse_search_ms), 0.0),
            func.coalesce(func.avg(QueryLog.rerank_ms), 0.0),
            func.avg(QueryLog.llm_ttft_ms),
        )

        if start_time is not None:
            stmt = stmt.where(QueryLog.created_at >= start_time)

        result = await self.session.execute(stmt)
        row = result.one()

        (
            total_queries,
            cache_hits,
            abstentions,
            total_cost,
            p50,
            p95,
            p99,
            avg_rewrite,
            avg_dense,
            avg_sparse,
            avg_rerank,
            avg_ttft,
        ) = row

        if not total_queries:
            return MetricsSummaryResponse()

        cache_hit_ratio = round(cache_hits / total_queries, 4)
        abstention_rate = round(abstentions / total_queries, 4)
        total_cost_usd = round(float(total_cost), 6)
        avg_cost_per_1k = round((float(total_cost) / total_queries) * 1000.0, 6)

        return MetricsSummaryResponse(
            total_queries=total_queries,
            cache_hit_ratio=cache_hit_ratio,
            abstention_rate=abstention_rate,
            total_cost_usd=total_cost_usd,
            avg_cost_per_1k_usd=avg_cost_per_1k,
            latencies=LatencyPercentiles(
                p50_ms=round(float(p50), 2),
                p95_ms=round(float(p95), 2),
                p99_ms=round(float(p99), 2),
            ),
            stage_latencies_avg=StageLatencyBreakdown(
                rewrite_ms=round(float(avg_rewrite), 2),
                dense_search_ms=round(float(avg_dense), 2),
                sparse_search_ms=round(float(avg_sparse), 2),
                rerank_ms=round(float(avg_rerank), 2),
                llm_ttft_ms=round(float(avg_ttft), 2) if avg_ttft is not None else None,
            ),
        )

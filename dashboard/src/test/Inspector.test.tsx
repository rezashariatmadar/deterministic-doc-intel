import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Inspector } from '../components/Inspector';
import { MetricsSummaryResponse, RAGResponse } from '../types';

describe('Inspector Component', () => {
  const mockResponse: RAGResponse = {
    answer: 'Sample answer [1].',
    is_answerable: true,
    confidence_score: 0.92,
    latency_ms: 210.5,
    sql_query_executed: 'SELECT revenue FROM financial_metrics WHERE company = "Apple"',
    citations: [
      {
        source_id: 1,
        document_title: 'Apple 10-K',
        page_start: 10,
        page_end: 10,
        section_path: 'Overview',
        supporting_quote: 'Apple revenue was $383B.',
        source_type: 'document',
      },
    ],
    sources: [
      {
        source_id: 1,
        chunk_id: 'c-1',
        document_id: 'd-1',
        document_title: 'Apple 10-K',
        page_start: 10,
        page_end: 10,
        section_path: 'Overview',
        element_type: 'text',
        text: 'Apple revenue was $383B in fiscal 2023.',
        contextualized_text: 'Contextualized full parent: Apple revenue was $383B in fiscal 2023.',
        similarity_score: 0.91,
        dense_rank: 1,
        sparse_rank: 2,
        rrf_score: 0.032,
        rerank_score: 0.94,
        parent_chunk_id: 'p-1',
        source_type: 'document',
      },
    ],
    candidate_facts: [
      {
        fact: 'Apple revenue was $383B.',
        claim_text: 'Apple revenue was $383B.',
        status: 'verified',
        confidence: 0.98,
        source_id: 1,
      },
    ],
    execution_trace: [
      {
        step: 'router_node',
        action: 'Classified query as HYBRID_SQL_AND_RAG',
        duration_ms: 45.2,
      },
      {
        step: 'sql_tool_node',
        action: 'Executed read-only safe SQL',
        duration_ms: 120.1,
      },
    ],
  };

  const mockMetrics: MetricsSummaryResponse = {
    total_queries: 150,
    cache_hit_ratio: 0.35,
    abstention_rate: 0.08,
    total_cost_usd: 0.0125,
    avg_cost_per_1k_usd: 0.0833,
    latencies: {
      p50_ms: 120.0,
      p95_ms: 350.0,
      p99_ms: 480.0,
    },
    stage_latencies_avg: {
      rewrite_ms: 30.0,
      dense_search_ms: 15.0,
      sparse_search_ms: 10.0,
      rerank_ms: 22.0,
      llm_ttft_ms: 110.0,
    },
  };

  it('renders Trace tab with SQL query block and ordered steps', () => {
    render(
      <Inspector
        currentResponse={mockResponse}
        selectedSourceId={null}
        metrics={mockMetrics}
        metricsLoading={false}
        onRefreshMetrics={vi.fn()}
        activeTab="trace"
        onChangeTab={vi.fn()}
      />
    );

    expect(screen.getByText(/کوئری پایگاه داده مالی اجراشده/)).toBeInTheDocument();
    expect(screen.getByText(/SELECT revenue FROM financial_metrics/)).toBeInTheDocument();
    expect(screen.getByText('router_node')).toBeInTheDocument();
    expect(screen.getByText('sql_tool_node')).toBeInTheDocument();
  });

  it('renders Evidence tab with candidate facts and chunk ranks', () => {
    render(
      <Inspector
        currentResponse={mockResponse}
        selectedSourceId={null}
        metrics={mockMetrics}
        metricsLoading={false}
        onRefreshMetrics={vi.fn()}
        activeTab="evidence"
        onChangeTab={vi.fn()}
      />
    );

    // Candidate facts
    expect(screen.getByText('Apple revenue was $383B.')).toBeInTheDocument();
    expect(screen.getByText(/تاییدشده \(Verified\)/)).toBeInTheDocument();

    // Retrieved Chunks
    expect(screen.getByText('Dense #1')).toBeInTheDocument();
    expect(screen.getByText('BM25 #2')).toBeInTheDocument();
    expect(screen.getByText('Rerank: 0.94')).toBeInTheDocument();
    expect(screen.getByText('گسترش به متن کامل والد')).toBeInTheDocument();
  });

  it('renders Metrics tab with percentiles and stage latency chart', () => {
    render(
      <Inspector
        currentResponse={mockResponse}
        selectedSourceId={null}
        metrics={mockMetrics}
        metricsLoading={false}
        onRefreshMetrics={vi.fn()}
        activeTab="metrics"
        onChangeTab={vi.fn()}
      />
    );

    expect(screen.getByText('150')).toBeInTheDocument();
    expect(screen.getByText('35%')).toBeInTheDocument();
    expect(screen.getByText('120 ms')).toBeInTheDocument();
    expect(screen.getByText('$0.0125')).toBeInTheDocument();
    expect(screen.getByText(/جستجوی برداری \(Dense\):/)).toBeInTheDocument();
  });
});

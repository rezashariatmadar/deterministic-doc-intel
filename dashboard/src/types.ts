/**
 * Domain types matching FastAPI backend models and Phase 5 dashboard requirements.
 */

export type IntentRoute =
  | 'CHITCHAT_OR_OUT_OF_DOMAIN'
  | 'FAST_TRACK_HYBRID_RAG'
  | 'MULTI_HOP_AGENTIC_RAG'
  | 'STRUCTURED_SQL_ANALYTICS'
  | 'HYBRID_SQL_AND_RAG';

export type ExtractionStatus =
  | 'verified'
  | 'not_found'
  | 'extraction_failed'
  | 'ambiguous'
  | 'conflicting'
  | 'unsupported';

export type CitationSourceType = 'document' | 'sql_table' | 'external_web';

export interface BoundingBox {
  x0?: number;
  y0?: number;
  x1?: number;
  y1?: number;
  page?: number;
  [key: string]: unknown;
}

export interface CandidateFact {
  fact?: string;
  claim_text?: string;
  status: ExtractionStatus;
  evidence?: string | null;
  supporting_quote?: string | null;
  source_id?: number | null;
  source_ids?: number[];
  document_title?: string | null;
  page_number?: number | null;
  bbox?: BoundingBox | null;
  confidence: number;
  notes?: string | null;
}

export interface Citation {
  source_id: number;
  document_id?: string | null;
  document_title: string;
  page_start: number;
  page_end: number;
  section_path: string;
  supporting_quote?: string | null;
  quote?: string | null;
  bbox?: BoundingBox | null;
  source_type: CitationSourceType;
  match_tier?: string | null;
  rehomed_from?: number | null;
  metadata?: Record<string, unknown>;
}

export interface RetrievedChunk {
  source_id: number;
  chunk_id: string;
  document_id: string;
  document_title: string;
  page_start: number;
  page_end: number;
  section_path: string;
  element_type: string;
  text: string;
  contextualized_text: string;
  similarity_score: number;
  dense_rank?: number | null;
  sparse_rank?: number | null;
  rrf_score?: number | null;
  rerank_score?: number | null;
  parent_chunk_id?: string | null;
  token_count?: number | null;
  bbox?: BoundingBox | null;
  source_type: CitationSourceType;
  metadata?: Record<string, unknown>;
}

export interface AgentTraceStep {
  step: string;
  action: string;
  step_name?: string;
  detail?: string;
  status?: string;
  input?: unknown;
  output?: unknown;
  duration_ms?: number | null;
  latency_ms?: number;
  metadata?: Record<string, unknown>;
}

export interface RAGResponse {
  answer: string;
  is_answerable: boolean;
  citations: Citation[];
  sources: RetrievedChunk[];
  confidence_score: number;
  latency_ms: number;
  route?: IntentRoute | null;
  route_taken?: IntentRoute | string;
  rewritten_query?: string | null;
  crag_retries?: number;
  crag_retry_count?: number;
  sql_query_executed?: string | null;
  candidate_facts: CandidateFact[];
  execution_trace: AgentTraceStep[];
  stage_latencies?: Record<string, number>;
}

export interface QueryFilters {
  document_id?: string | null;
  category?: string | null;
  element_types?: string[] | null;
  page_min?: number | null;
  page_max?: number | null;
  min_similarity?: number;
}

export interface QueryRequest {
  query: string;
  filters?: QueryFilters | null;
  top_k?: number;
  stream?: boolean;
  search_mode?: 'hybrid' | 'dense' | 'sparse';
  weight_dense?: number | null;
  weight_sparse?: number | null;
  enable_reranker?: boolean | null;
  enable_query_rewrite?: boolean;
  enable_hyde?: boolean;
  session_id?: string | null;
  enable_agent?: boolean;
  enable_agentic?: boolean;
  allow_external_fallback?: boolean;
  max_crag_retries?: number;
  route_override?: IntentRoute | null;
  resolve_parents?: boolean;
  bypass_cache?: boolean;
}

export interface DocumentItem {
  id: string;
  filename: string;
  file_hash: string;
  title: string;
  total_pages: number;
  total_chunks: number;
  status: 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED';
  processing_stage?: 'QUEUED' | 'PARSING' | 'CHUNKING' | 'EMBEDDING' | 'INDEXING' | 'COMPLETED' | 'FAILED' | null;
  task_id?: string | null;
  error_message?: string | null;
  created_at: string;
  updated_at: string;
}

export interface HealthStatus {
  status: string;
  database: 'connected' | 'disconnected' | string;
  redis: 'connected' | 'disconnected' | string;
  timestamp: string;
}

export interface LatencyPercentiles {
  p50_ms: number;
  p95_ms: number;
  p99_ms: number;
}

export interface StageLatencyBreakdown {
  rewrite_ms: number;
  dense_search_ms: number;
  sparse_search_ms: number;
  rerank_ms: number;
  llm_ttft_ms?: number | null;
}

export interface MetricsSummaryResponse {
  total_queries: number;
  cache_hit_ratio: number;
  abstention_rate: number;
  total_cost_usd: number;
  avg_cost_per_1k_usd: number;
  latencies: LatencyPercentiles;
  stage_latencies_avg: StageLatencyBreakdown;
}

export interface Preset {
  id: string;
  label: string;
  description: string;
  query: string;
  options: Partial<QueryRequest>;
  turn2Query?: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  originalQuery?: string;
  rewrittenQuery?: string | null;
  response?: RAGResponse | null;
  cacheHit?: boolean | null;
  ttftMs?: number | null;
  isLoading?: boolean;
  error?: string | null;
  blockedPattern?: string | null;
  isAbstain?: boolean;
  aborted?: boolean;
}

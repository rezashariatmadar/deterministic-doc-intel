# Hybrid and Agentic RAG Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20%2B%20pgvector-336791.svg)](https://github.com/pgvector/pgvector)
[![ParadeDB](https://img.shields.io/badge/ParadeDB-pg__search%20BM25-orange.svg)](https://www.paradedb.com)
[![Redis](https://img.shields.io/badge/Redis-Stack%207.4-DC382D.svg)](https://redis.io)
[![Celery](https://img.shields.io/badge/Celery-5.4-37814A.svg)](https://docs.celeryq.dev)
[![React](https://img.shields.io/badge/React-18%20%2B%20Vite-61DAFB.svg)](https://react.dev)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![CI Gate](https://github.com/example/rag/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

A hybrid and agentic retrieval-augmented generation (RAG) platform for financial and technical documents, including SEC 10-K filings, technical specifications, and structured tables.

---

## 1. Problem statement and architecture

Standard fixed-chunk RAG pipelines often break on corporate filings:

1. Fixed-size chunking splits tables across chunks, separating column headers from values and corrupting calculations.
2. Chunks between 150 and 250 tokens match queries well in vector search but lack enough context for generation. Chunks over 1,000 tokens add irrelevant text to the attention window.
3. Language models invent citations `[1]` or attribute claims to the wrong passage.
4. Quantitative questions (such as comparing year-over-year revenue across companies) require exact SQL aggregations over database rows, which vector search cannot calculate reliably.
5. Missing information causes the model to fabricate answers instead of abstaining or searching externally.

### System approach

- Hierarchical indexing: extracts layout blocks, stores compact 200-token child chunks for dense and sparse search, and maps each child to a 1,200-token parent chunk for LLM generation context.
- Hybrid search with normalized RRF: combines `pgvector` HNSW cosine similarity and ParadeDB BM25 rankings, scaled to $[0.0, 1.0]$.
- Agentic routing: directs queries across fast-track retrieval, text-to-SQL with AST validation, multi-hop decomposition, and web search fallback, with reflection loops to filter out irrelevant chunks.
- Deterministic quote verifier: checks every citation against source text using exact match, normalized offsets, token fuzzy matching ($\ge 80\%$), cross-chunk re-homing, and text slicing, then attaches bounding boxes (`[x0, y0, x1, y1]`) to verified citations.
- Observability dashboard: includes step-by-step trace views, an evidence inspector with PDF bounding boxes, SSE streaming, and Redis semantic caching.

---

## 2. System diagrams

### Diagram 1: Asynchronous Celery Ingestion & Hierarchical Indexing Pipeline

```mermaid
flowchart TD
    Client(["Client / API User"]) -->|Multipart PDF Upload| API["FastAPI Ingestion Endpoint<br/><code>POST /api/v1/documents/upload?async_mode=true</code>"]
    API -->|Generate UUID & SHA-256| DB_Doc[("PostgreSQL: <code>documents</code><br/>Status: PENDING")]
    API -->|Dispatch Task| CeleryBroker[("Redis Task Queue<br/>Broker")]
    API -->|Return 202 Accepted| Client

    CeleryBroker -->|Consume Task| Worker["Celery Worker Engine<br/><code>app.workers.tasks.ingest_document</code>"]
    
    subgraph Pipeline ["6-Stage Ingestion Worker Pipeline"]
        Worker -->|Stage 1| Parser["PyMuPDF Structural Parser<br/>Extract Layout Blocks, Tables & Bounding Boxes"]
        Parser -->|Stage 2| Chunker["Hierarchical Chunker<br/>1200-tok Parent Chunks + 200-tok Child Chunks"]
        Chunker -->|Stage 3| Breadcrumb["Contextual Breadcrumb Injection<br/><code>[Document: ... | Section: ... | Page: ...]</code>"]
        Breadcrumb -->|Stage 4| BatchEmbed["Batch Cloud Embedding<br/>1024-dim Vectorization"]
        BatchEmbed -->|Stage 5| DB_Store[("PostgreSQL Storage<br/><code>document_chunks</code> table")]
        DB_Store -->|Build HNSW Index| HNSW["pgvector HNSW Cosine Index"]
        DB_Store -->|Build BM25 Index| BM25["ParadeDB pg_search BM25 Index"]
        DB_Store -->|Stage 6| CacheInvalidate["Redis Semantic Cache<br/>Scope Invalidation via <code>semcache:doc_scope_map</code>"]
    end

    CacheInvalidate -->|Update Document Status| DB_Done[("PostgreSQL: <code>documents</code><br/>Status: COMPLETED")]
```

### Diagram 2: Online Self-Correcting Agentic State Machine

```mermaid
flowchart TD
    Start(["User Query"]) --> Memory["Contextual Memory Service<br/>Multi-turn Pronoun Resolution & Session History"]
    Memory --> CacheCheck{"Redis Semantic Cache<br/>Cosine Similarity &ge; 0.94?"}
    
    CacheCheck -->|Cache HIT| CacheHit["Instant Return Cached Answer<br/><code>X-Cache-Hit: true</code> | Latency &lt; 10ms"]
    CacheCheck -->|Cache MISS| IntentRouter["Semantic Intent Router<br/>Pydantic Structured Classification"]

    IntentRouter -->|STRUCTURED_SQL_ANALYTICS| SQLValidator["Safe Text-to-SQL Tool<br/>AST Validation, Read-Only Enforced"]
    SQLValidator -->|Execute Query| SQLDb[("PostgreSQL Financial Metrics")]
    SQLDb -->|Format Rows| FormatChunk["Format as RetrievedChunk"]

    IntentRouter -->|FAST_TRACK_HYBRID_RAG| ParallelSearch
    IntentRouter -->|MULTI_HOP_AGENTIC_RAG| QueryRewrite["Query Transformation & Decomposition"]
    QueryRewrite --> ParallelSearch

    subgraph SearchEngine ["Two-Stage Hybrid Search & Provenance Engine"]
        ParallelSearch["Parallel Search Execution"]
        ParallelSearch --> DenseSearch["pgvector Dense Search<br/>HNSW Cosine Distance"]
        ParallelSearch --> SparseSearch["ParadeDB BM25 Lexical Search<br/>True Okapi Saturation"]
        DenseSearch --> RRF["Normalized Reciprocal Rank Fusion<br/>Score Scaled to 0.0 - 1.0"]
        SparseSearch --> RRF
        RRF --> SiblingResolve["Parent-Child Sibling Expansion<br/>Batch Load Full Chunks + BBoxes"]
        SiblingResolve --> Rerank["Cross-Encoder Neural Reranker<br/>Top-K Candidate Selection"]
    end

    Rerank --> CRAGJudge{"CRAG Relevance Grader<br/>Are Chunks Relevant?"}
    FormatChunk --> GroundedGen

    CRAGJudge -->|RELEVANT| GroundedGen["Grounded Generator<br/>Strict XML Context & Bracketed Citations <code>[N]</code>"]
    CRAGJudge -->|AMBIGUOUS & Retries &lt; 2| Reformulate["Query Reformulation Loop<br/>Strip Ambiguities & Re-search"]
    Reformulate --> ParallelSearch
    CRAGJudge -->|INCORRECT / No Corpus Evidence| WebFallback["External Web Search Fallback<br/>Wikipedia & DuckDuckGo API"]
    WebFallback --> GroundedGen

    GroundedGen --> Verifier["6-Tier Deterministic Verifier Ladder<br/>Exact &rarr; Normalized &rarr; Fuzzy &ge; 80% &rarr; Cross-chunk &rarr; Slicing"]
    Verifier --> Telemetry["Async Telemetry & Redis Cache Store<br/>Log QueryLog & Cache Scope Hash"]
    Telemetry --> Output(["Final Grounded Answer + Verified Provenance"])
```

---

## 3. Benchmark results

A 28-query benchmark on SEC 10-K filings (NVIDIA, Apple, Microsoft, Amazon, and Alphabet) evaluated dense, sparse, hybrid, and hierarchical retrieval alongside generation faithfulness.

### Table 1: Retrieval ablation (`evals/reports/retrieval_ablation.md`)

| Retrieval Configuration | Recall@3 | Recall@5 | Recall@10 | Precision@5 | MRR@10 | NDCG@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense Vector Only (Flat)** | 0.7309 | 0.7309 | 0.7743 | 0.1583 | 0.3958 | 0.4192 |
| **Sparse BM25 Only (ParadeDB)** | 0.7552 | 0.7552 | 0.7812 | 0.1583 | 0.3958 | 0.4293 |
| **Hybrid RRF (Flat)** | 0.9392 | 0.9826 | 0.9826 | 0.2500 | 1.0000 | 0.9256 |
| **Hybrid + Hierarchical + Reranker** | **0.9826** | **0.9826** | **0.9826** | **0.2500** | **1.0000** | 0.9459 |
| **Full Pipeline (Transform + Hybrid + Hierarchical + Rerank)** | **0.9826** | **0.9826** | **0.9826** | **0.2500** | **1.0000** | **0.9502** |

Combining dense and sparse search via RRF increased Recall@5 from 73.1% to 98.3% (+25.2 percentage points) and MRR@10 from 0.3958 to 1.0000. Adding the cross-encoder neural reranker raised NDCG@5 to 0.9502.

### Table 2: Generation benchmark (`evals/reports/generation_benchmark.md`)

| Category Archetype | Queries | Faithfulness | Answer Relevance | Correctness | Citation Precision | Abstention Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exact Keyword** | 6 | 1.0000 | 0.7417 | 0.8833 | 1.0000 | 1.0000 |
| **Single-Hop Semantic** | 6 | 1.0000 | 0.7364 | 0.9514 | 1.0000 | 1.0000 |
| **Multi-Hop Comparative** | 6 | 1.0000 | 0.8493 | 0.9417 | 1.0000 | 1.0000 |
| **Tabular Numerical** | 6 | 1.0000 | 0.8808 | 0.9417 | 1.0000 | 1.0000 |
| **Unanswerable Adversarial** | 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Overall Macro Average** | **28** | **1.0000** | **0.8303** | **0.9396** | **1.0000** | **1.0000** |

The verifier ladder eliminated hallucinated citations, achieving 100% citation precision across all 28 queries. On unanswerable queries with no supporting evidence, the system abstained consistently (100% abstention accuracy).

---

## 4. Evaluation modes

The repository includes both an offline dry-run suite for CI and live benchmark scripts for testing against the 5-document SEC corpus.

### Dry-run mode (CI)

```bash
uv run python -m evals.run_retrieval_eval --dry-run --output /tmp/ci_ablation.md
```

Runs mock retrievers against stored fixtures to test scoring logic and regression baselines without calling external model APIs.

### Live benchmark execution

```bash
# 1. Start containers and ingest the 5-document SEC corpus
docker compose up -d
uv run alembic upgrade head
uv run python -m scripts.ingest_corpus --manifest data/raw_corpus/manifest.json

# 2. Run live retrieval ablation (connects to pgvector & ParadeDB BM25)
uv run python -m evals.run_retrieval_eval --output evals/reports/retrieval_ablation.md

# 3. Run live LLM generation & hallucination evaluation
uv run python -m evals.run_generation_eval --output evals/reports/generation_benchmark.md
```

*Note:* Live evaluation requires model API credentials in `.env`.

---

## 5. Architectural decisions

Detailed justifications are documented in [`docs/architecture/ROADMAP_DEVIATIONS.md`](docs/architecture/ROADMAP_DEVIATIONS.md):

- **ADR-001 (Cloud APIs vs. in-container models):** Uses Gemini and OpenAI APIs instead of running local PyTorch models inside container images. This avoids 3GB+ image bloat and GPU dependencies, while keeping interfaces compatible with local runtimes (Ollama or vLLM).
- **ADR-002 (ParadeDB for BM25):** Deploys `paradedb/paradedb:latest` (Postgres 16 with pgvector and pg_search) instead of standard PostgreSQL `tsvector`, providing Okapi BM25 term frequency saturation and document length normalization (`k1` and `b`) within the same database instance.
- **ADR-003 (Deterministic verification ladder):** Validates generated citations through exact match, punctuation-normalized match, token fuzzy similarity ($\ge 80\%$), cross-chunk re-homing, and quote slicing.
- **ADR-004 (Parent-child sibling lookup):** Resolves child chunks to parent documents and loads sibling records, providing exact bounding-box coordinates (`[x0, y0, x1, y1]`) and page numbers for citations even when quotes span adjacent child chunks.
- **ADR-005 (Normalized Reciprocal Rank Fusion):** Normalizes RRF scores with $\frac{w_{\text{dense}} + w_{\text{sparse}}}{k + 1}$ to scale values into $[0.0, 1.0]$, preventing compressed scores from distorting downstream generation thresholds.

---

## 6. Frontend dashboard

The web interface is built with React 18, Vite, and Tailwind CSS, with right-to-left Persian typography:

- Direction isolation: uses CSS logical properties (`ms-*`, `me-*`, `ps-*`, `pe-*`, `start-*`, `end-*`), Vazirmatn font, and `<bdi dir="ltr">` tags for technical terms.
- Citation inspector: clicking an inline citation chip `[1]` opens an evidence drawer with PDF bounding boxes, section hierarchy, similarity scores, and verification tier.
- Execution traces: shows execution steps across the intent router, safe text-to-SQL tool, CRAG grader, and reranker with latency measurements.
- Presets: includes sample queries covering hybrid search, multi-hop reasoning, SQL analytics, conversation history, and prompt injection defense.
- Real-time streaming: token streaming via server-sent events (SSE) with cache-hit indicators.

When built, the dashboard is served by FastAPI at `/dashboard/`.

---

## 7. Portfolio summary templates

Engineers presenting this project in portfolios or interviews can adapt the following descriptions:

- **Hybrid retrieval and provenance:**
  > *"Architected a two-stage Hybrid RAG platform in Python (FastAPI, PostgreSQL pgvector, ParadeDB BM25) featuring Parent-Child hierarchical indexing, Reciprocal Rank Fusion (RRF), and Cross-Encoder neural reranking, improving retrieval **Recall@5 from 73.1% to 98.3% (+25.2 pp)** and **MRR@10 from 0.3958 to 1.0000** across a 28-query SEC 10-K financial benchmark."*

- **Agentic state machine and text-to-SQL:**
  > *"Engineered a self-correcting Agentic RAG state machine with dynamic intent routing, multi-hop query decomposition, Corrective RAG (CRAG) reflection grading, and an AST-sandboxed Text-to-SQL tool, achieving **100% Faithfulness**, **100% Citation Precision**, and **100% Adversarial Abstention Accuracy** with zero hallucinated figures."*

- **Asynchronous ingestion and telemetry:**
  > *"Built an asynchronous document ingestion pipeline using Celery and Redis alongside an invalidation-aware Redis vector semantic cache and rate limiter, cutting **P95 latency to < 10ms on cache hits**, and designed an RTL Evidence Ledger observability dashboard with real-time SSE streaming and execution tracing."*

---

## 8. Quickstart

### 1. Start containers

```bash
docker compose up -d
```

### 2. Install dependencies and run migrations

```bash
# Sync Python virtual environment
uv sync

# Run Alembic migrations
uv run alembic upgrade head
```

### 3. Build dashboard and run tests

```bash
# Run backend test suite
uv run pytest tests/unit -v

# Build frontend dashboard
cd dashboard
npm install
npm test
npm run build
cd ..
```

### 4. Start the application

```bash
# Start FastAPI application
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- API documentation: `http://localhost:8000/docs`
- Embedded dashboard: `http://localhost:8000/dashboard/`
- Vite development server: `cd dashboard && npm run dev` (`http://localhost:5173/`)

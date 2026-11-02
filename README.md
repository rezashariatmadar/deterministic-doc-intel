# Production-Grade Hybrid & Agentic RAG Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20%2B%20pgvector-336791.svg)](https://github.com/pgvector/pgvector)
[![Redis](https://img.shields.io/badge/Redis-Stack%207.4-DC382D.svg)](https://redis.io)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](http://mypy-lang.org/)

An enterprise-ready, evaluated Hybrid and Agentic Retrieval-Augmented Generation (RAG) platform designed for mission-critical financial and corporate document intelligence.

---

## 🏛️ Architecture Highlights

- **Layout-Aware Ingestion Pipeline**: Extracts structured text and complex multi-column tables using PyMuPDF, preserving document hierarchy and table column headers across splits.
- **Contextual Breadcrumb Injection**: Prepends hierarchical metadata (`[Document: ... | Section: ... | Page: ...]`) to every chunk before vectorization, resolving lost-context failure modes.
- **Sub-millisecond Vector Retrieval**: PostgreSQL 16 with `pgvector` HNSW indexes (`m=16`, `ef_construction=64`, cosine distance) combined with metadata pre-filtering (GIN index on JSONB).
- **Pluggable Provider Architecture**: Decoupled interface supporting both cloud APIs (OpenAI `text-embedding-3-small`, `gpt-4o-mini`, Gemini) and local OpenAI-compatible endpoints (Ollama, vLLM, TEI) or local CUDA acceleration (RTX 4060 Ti).
- **Strict Grounded Generation & Citation Verification**: XML-constrained prompts enforce bracketed citations `[N]`. All citations are verified against source IDs, flagging hallucinations as `[unverified: N]`.
- **Real-Time Streaming**: Server-Sent Events (SSE) stream answer tokens with low first-token latency alongside source attribution metadata.

---

## 📂 Project Directory Structure

```text
rag/
├── app/
│   ├── api/
│   │   ├── dependencies.py       # Scoped DB sessions & provider injection
│   │   ├── middleware.py         # Correlation ID (X-Request-ID) & domain error handlers
│   │   └── v1/
│   │       ├── router.py         # Aggregated v1 API routing
│   │       └── endpoints/
│   │           ├── health.py     # PostgreSQL & Redis connectivity ping
│   │           ├── documents.py  # Ingest, list, retrieve, and delete documents
│   │           └── query.py      # Grounded generation & SSE token streaming
│   ├── core/
│   │   ├── config.py             # Pydantic Settings & environment variables
│   │   ├── exceptions.py         # Strongly-typed domain exception hierarchy
│   │   └── logging.py            # Structured JSON logger with correlation IDs
│   ├── db/
│   │   ├── base.py               # DeclarativeBase with UUID & timestamp mixins
│   │   ├── session.py            # Async engine & sessionmaker (NullPool in tests)
│   │   ├── models/               # SQLAlchemy ORM models (Document, DocumentChunk)
│   │   └── repositories/         # Repository pattern for database operations
│   ├── schemas/                  # Pydantic DTOs (document, chunk, query, citations)
│   └── services/
│       ├── ingestion/            # Normalizer, layout parser & structural chunker
│       ├── providers/            # Pluggable cloud & local embedding/LLM adapters
│       └── retrieval/            # Vector search service & grounded generator
├── data/
│   └── raw_corpus/               # SEC 10-K test PDFs and verification manifest
├── docker/                       # PostgreSQL initialization SQL scripts
├── docker-compose.yml            # PostgreSQL 16 + pgvector + Redis Stack
├── pyproject.toml                # Dependencies, ruff, mypy, and pytest configs
└── scripts/
    └── verify_baseline.py        # End-to-end baseline benchmark script
```

---

## ⚡ Quickstart

### 1. Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (fast Python package manager)
- Docker & Docker Compose

### 2. Start Infrastructure Containers

```bash
docker compose up -d
```

Verify that PostgreSQL with pgvector and Redis Stack are healthy:

```bash
docker compose ps
```

### 3. Install Dependencies & Apply Migrations

```bash
# Sync virtual environment
uv sync

# Apply Alembic database migrations
uv run alembic upgrade head
```

### 4. Configure Environment Variables

Copy the example `.env` file:

```bash
cp .env.example .env
```

To use cloud providers, set your API key:

```env
OPENAI_API_KEY="sk-..."
EMBEDDING_PROVIDER="openai"
LLM_PROVIDER="openai"
```

To run against a local server (e.g. Ollama or vLLM):

```env
EMBEDDING_PROVIDER="local"
LLM_PROVIDER="local"
LOCAL_LLM_BASE_URL="http://localhost:11434/v1"
```

---

## 🧪 Testing & Verification

Run the full automated test suite:

```bash
uv run pytest -v
```

Run code formatting and strict type checks:

```bash
uv run ruff check app tests
uv run mypy app tests
```

Execute the end-to-end baseline verification script:

```bash
uv run python scripts/verify_baseline.py
```

---

## 🌐 API Endpoints Overview

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | PostgreSQL and Redis connectivity status |
| `POST` | `/api/v1/documents/ingest` | Multipart PDF upload, parsing, chunking & vector indexing |
| `GET` | `/api/v1/documents` | List indexed documents with pagination |
| `DELETE` | `/api/v1/documents/{id}` | Delete document and cascade chunks |
| `POST` | `/api/v1/query` | Grounded search with verified citations and confidence score |
| `POST` | `/api/v1/query/stream` | Real-time Server-Sent Events (SSE) token streaming |

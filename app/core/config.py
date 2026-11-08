"""Central application configuration using Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and defaults."""

    # Application settings
    APP_NAME: str = "Hybrid & Agentic RAG Platform"
    APP_ENV: Literal["development", "staging", "production", "test"] = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # Database & Cache
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/rag_db"
    REDIS_URL: str = "redis://localhost:6379/0"
    SEMANTIC_CACHE_ENABLED: bool = True
    SEMANTIC_CACHE_THRESHOLD: float = 0.94
    SEMANTIC_CACHE_TTL: int = 86400

    # Celery Background Tasks
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # Embedding Provider Configuration
    EMBEDDING_PROVIDER: Literal["local", "openai", "gemini"] = "local"
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DIMENSION: int = 1024
    EMBEDDING_DEVICE: str = "cuda"
    EMBEDDING_BATCH_SIZE: int = 32

    # LLM Provider Configuration
    LLM_PROVIDER: Literal["local", "openai", "gemini"] = "local"
    LOCAL_LLM_BASE_URL: str = "http://localhost:11434/v1"
    LOCAL_LLM_MODEL: str = "mistral"

    # Cloud Provider Settings
    OPENAI_API_KEY: str | None = None
    OPENAI_LLM_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    GEMINI_API_KEY: str | None = None
    GEMINI_LLM_MODEL: str = "gemini-2.5-flash"
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"

    # Ingestion & Chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    PARENT_CHUNK_SIZE: int = 1200
    CHILD_CHUNK_SIZE: int = 200
    CHILD_CHUNK_OVERLAP: int = 30

    # Retrieval
    RETRIEVAL_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.35
    MAX_CONTEXT_TOKENS: int = 4000

    # Phase 2: Hybrid Search & RRF
    HYBRID_SEARCH_CANDIDATES: int = 30
    RRF_K: int = 60
    WEIGHT_DENSE: float = 0.65
    WEIGHT_SPARSE: float = 0.35

    # Phase 2: Cross-Encoder Reranker
    ENABLE_RERANKER: bool = True
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_TOP_K: int = 5
    RERANKER_THRESHOLD: float = 0.05
    RERANKER_TIMEOUT_MS: int = 500

    # Phase 4: Self-Correcting Agentic RAG, Text-to-SQL, Memory & Guardrails
    SQL_EXECUTION_TIMEOUT_MS: int = 3000
    MAX_SQL_CORRECTION_RETRIES: int = 1
    MAX_CRAG_RETRIES: int = 2
    CRAG_FALLBACK_SIMILARITY: float = 0.25
    ENABLE_EXTERNAL_FALLBACK: bool = False
    CHAT_SESSION_TTL: int = 86400
    CHAT_SESSION_MAX_TURNS: int = 6
    RATE_LIMIT_REQUESTS: int = 60
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    PROMPT_INJECTION_THRESHOLD: float = 0.8

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached singleton instance of application settings."""
    return Settings()

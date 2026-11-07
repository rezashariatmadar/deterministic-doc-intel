import json
import logging

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    AppException,
    ConfigurationError,
    DocumentNotFoundError,
    DocumentParsingError,
    DuplicateDocumentError,
    EmptyRetrievalContextError,
    ProviderError,
)
from app.core.logging import (
    JSONFormatter,
    get_correlation_id,
    set_correlation_id,
    setup_logging,
)


def test_settings_load_defaults():
    settings = Settings(
        DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/rag_db",
        REDIS_URL="redis://localhost:6379/0",
    )
    assert settings.EMBEDDING_PROVIDER == "local"
    assert settings.EMBEDDING_MODEL == "BAAI/bge-m3"
    assert settings.EMBEDDING_DIMENSION == 1024
    assert settings.LLM_PROVIDER == "local"
    assert settings.LOCAL_LLM_BASE_URL == "http://localhost:11434/v1"


def test_settings_custom_overrides():
    settings = Settings(
        DATABASE_URL="postgresql+asyncpg://user:pass@remote:5432/custom_db",
        REDIS_URL="redis://remote:6379/1",
        EMBEDDING_PROVIDER="openai",
        EMBEDDING_MODEL="text-embedding-3-small",
        EMBEDDING_DIMENSION=1536,
        LLM_PROVIDER="openai",
        OPENAI_API_KEY="sk-test-key",
        RETRIEVAL_TOP_K=10,
        SIMILARITY_THRESHOLD=0.5,
    )
    assert settings.DATABASE_URL == "postgresql+asyncpg://user:pass@remote:5432/custom_db"
    assert settings.REDIS_URL == "redis://remote:6379/1"
    assert settings.EMBEDDING_PROVIDER == "openai"
    assert settings.EMBEDDING_MODEL == "text-embedding-3-small"
    assert settings.EMBEDDING_DIMENSION == 1536
    assert settings.LLM_PROVIDER == "openai"
    assert settings.OPENAI_API_KEY == "sk-test-key"
    assert settings.RETRIEVAL_TOP_K == 10
    assert settings.SIMILARITY_THRESHOLD == 0.5


def test_get_settings_cached():
    get_settings.cache_clear()
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_domain_exceptions_hierarchy():
    base_err = AppException("Generic error", status_code=500, error_code="GENERIC_ERROR")
    assert base_err.status_code == 500
    assert base_err.to_dict()["error"] == "GENERIC_ERROR"

    parse_err = DocumentParsingError("Corrupt PDF", filename="report.pdf", page_number=3)
    assert parse_err.status_code == 422
    assert parse_err.error_code == "DOCUMENT_PARSING_ERROR"
    assert parse_err.filename == "report.pdf"
    assert parse_err.page_number == 3
    assert parse_err.to_dict()["details"]["filename"] == "report.pdf"

    dup_err = DuplicateDocumentError(
        "Already indexed", file_hash="abc123hash", existing_document_id="doc-456"
    )
    assert dup_err.status_code == 409
    assert dup_err.error_code == "DUPLICATE_DOCUMENT"
    assert dup_err.file_hash == "abc123hash"
    assert dup_err.existing_document_id == "doc-456"

    prov_err = ProviderError("Ollama timeout", provider="local", status_code=504)
    assert prov_err.status_code == 504
    assert prov_err.error_code == "PROVIDER_ERROR"
    assert prov_err.provider == "local"

    empty_err = EmptyRetrievalContextError(
        "No matching context", query="what is net income?", similarity_threshold=0.35
    )
    assert empty_err.status_code == 404
    assert empty_err.error_code == "EMPTY_RETRIEVAL_CONTEXT"
    assert empty_err.query == "what is net income?"

    doc_not_found = DocumentNotFoundError("Document not found")
    assert doc_not_found.status_code == 404

    cfg_err = ConfigurationError("Missing key")
    assert cfg_err.status_code == 500


def test_logging_json_formatter_and_correlation_id():
    formatter = JSONFormatter()
    logger = logging.getLogger("test_logger")

    # Set correlation ID in context
    test_req_id = "req-test-uuid-1234"
    set_correlation_id(test_req_id)
    assert get_correlation_id() == test_req_id

    record = logger.makeRecord(
        name="test_logger",
        level=logging.INFO,
        fn="test_config.py",
        lno=42,
        msg="Testing structured log with %s",
        args=("context",),
        exc_info=None,
        extra={"custom_field": "custom_val"},
    )

    formatted = formatter.format(record)
    parsed = json.loads(formatted)

    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test_logger"
    assert parsed["message"] == "Testing structured log with context"
    assert parsed["correlation_id"] == test_req_id
    assert parsed["extra"]["custom_field"] == "custom_val"
    assert "timestamp" in parsed

    # Clean up correlation ID
    set_correlation_id(None)
    assert get_correlation_id() is None


def test_setup_logging():
    setup_logging(log_level="DEBUG", json_format=True)
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) > 0
    assert isinstance(root.handlers[0].formatter, JSONFormatter)


def test_cors_configuration_validity():
    from starlette.middleware.cors import CORSMiddleware

    from app.main import app

    for middleware in app.user_middleware:
        if getattr(middleware, "cls", None) == CORSMiddleware:
            kwargs = getattr(middleware, "kwargs", {})
            origins = kwargs.get("allow_origins", [])
            assert isinstance(origins, list)
            assert not ("*" in origins and kwargs.get("allow_credentials") is True)


def test_phase2_settings_and_query_schemas():
    import uuid
    from datetime import datetime

    from app.schemas.document import ParentChunkCreate, ParentChunkRead
    from app.schemas.query import QueryRequest, RetrievedChunk

    settings = Settings(
        DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/rag_db",
        REDIS_URL="redis://localhost:6379/0",
    )
    assert settings.PARENT_CHUNK_SIZE == 1200
    assert settings.CHILD_CHUNK_SIZE == 200
    assert settings.CHILD_CHUNK_OVERLAP == 30
    assert settings.HYBRID_SEARCH_CANDIDATES == 30
    assert settings.RRF_K == 60
    assert settings.WEIGHT_DENSE == 0.65
    assert settings.WEIGHT_SPARSE == 0.35
    assert settings.ENABLE_RERANKER is True
    assert settings.RERANKER_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert settings.RERANKER_TIMEOUT_MS == 500

    # QueryRequest phase 2 controls
    req = QueryRequest(
        query="What was revenue?",
        search_mode="hybrid",
        weight_dense=0.7,
        weight_sparse=0.3,
        enable_reranker=True,
        enable_query_rewrite=True,
        enable_hyde=True,
    )
    assert req.search_mode == "hybrid"
    assert req.weight_dense == 0.7
    assert req.enable_query_rewrite is True

    # RetrievedChunk with phase 2 rank/score metadata
    p_id = uuid.uuid4()
    c = RetrievedChunk(
        source_id=1,
        chunk_id=uuid.uuid4(),
        parent_chunk_id=p_id,
        document_id=uuid.uuid4(),
        document_title="Report",
        page_start=1,
        page_end=1,
        section_path="Intro",
        element_type="prose",
        text="Sample text",
        contextualized_text="[Doc] Sample text",
        similarity_score=0.92,
        dense_rank=1,
        sparse_rank=3,
        rrf_score=0.032,
        rerank_score=0.89,
    )
    assert c.parent_chunk_id == p_id
    assert c.dense_rank == 1
    assert c.rrf_score == 0.032
    assert c.rerank_score == 0.89

    # ParentChunkCreate and ParentChunkRead
    p_create = ParentChunkCreate(
        document_id=uuid.uuid4(),
        parent_index=0,
        raw_text="Parent text",
        contextualized_text="[Doc] Parent text",
        token_count=100,
        page_start=1,
        page_end=2,
        section_path="Overview",
    )
    assert p_create.parent_index == 0
    p_read = ParentChunkRead(
        id=p_id,
        document_id=p_create.document_id,
        parent_index=0,
        raw_text="Parent text",
        contextualized_text="[Doc] Parent text",
        token_count=100,
        page_start=1,
        page_end=2,
        section_path="Overview",
        created_at=datetime.now(),
    )
    assert p_read.id == p_id

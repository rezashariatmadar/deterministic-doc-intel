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

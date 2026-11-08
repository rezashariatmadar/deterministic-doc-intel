"""Custom domain exceptions for the Hybrid & Agentic RAG Platform."""

from typing import Any


class AppException(Exception):
    """Base domain exception for all application errors."""

    status_code: int = 500
    error_code: str = "INTERNAL_SERVER_ERROR"

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        if error_code is not None:
            self.error_code = error_code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to structured dictionary representation."""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
        }


class DocumentParsingError(AppException):
    """Raised when document extraction or layout parsing fails."""

    status_code: int = 422
    error_code: str = "DOCUMENT_PARSING_ERROR"

    def __init__(
        self,
        message: str,
        filename: str | None = None,
        page_number: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if filename:
            details["filename"] = filename
        if page_number is not None:
            details["page_number"] = page_number
        super().__init__(message=message, details=details)
        self.filename = filename
        self.page_number = page_number


class IngestionError(AppException):
    """Raised when an ingestion pipeline step fails."""

    status_code: int = 422
    error_code: str = "INGESTION_ERROR"

    def __init__(
        self,
        message: str,
        filename: str | None = None,
        page_number: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if filename:
            details["filename"] = filename
        if page_number is not None:
            details["page_number"] = page_number
        super().__init__(message=message, details=details)
        self.filename = filename
        self.page_number = page_number


class DuplicateDocumentError(AppException):
    """Raised when an uploaded document has already been indexed."""

    status_code: int = 409
    error_code: str = "DUPLICATE_DOCUMENT"

    def __init__(
        self,
        message: str,
        file_hash: str | None = None,
        existing_document_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if file_hash:
            details["file_hash"] = file_hash
        if existing_document_id:
            details["existing_document_id"] = existing_document_id
        super().__init__(message=message, details=details)
        self.file_hash = file_hash
        self.existing_document_id = existing_document_id


class ProviderError(AppException):
    """Raised when an external or local AI model provider fails or times out."""

    status_code: int = 502
    error_code: str = "PROVIDER_ERROR"

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if provider:
            details["provider"] = provider
        super().__init__(
            message=message,
            status_code=status_code or self.status_code,
            details=details,
        )
        self.provider = provider


class EmptyRetrievalContextError(AppException):
    """Raised when similarity search returns no matching chunks meeting threshold."""

    status_code: int = 404
    error_code: str = "EMPTY_RETRIEVAL_CONTEXT"

    def __init__(
        self,
        message: str,
        query: str | None = None,
        similarity_threshold: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if query:
            details["query"] = query
        if similarity_threshold is not None:
            details["similarity_threshold"] = similarity_threshold
        super().__init__(message=message, details=details)
        self.query = query
        self.similarity_threshold = similarity_threshold


class DocumentNotFoundError(AppException):
    """Raised when a requested document ID does not exist."""

    status_code: int = 404
    error_code: str = "DOCUMENT_NOT_FOUND"


class ConfigurationError(AppException):
    """Raised when application configuration is invalid or missing required keys."""

    status_code: int = 500
    error_code: str = "CONFIGURATION_ERROR"


class PromptInjectionError(AppException):
    """Raised when a prompt injection attempt is detected in user input."""

    status_code: int = 400
    error_code: str = "PROMPT_INJECTION_DETECTED"

    def __init__(
        self,
        message: str = "Prompt injection attempt detected.",
        pattern: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if pattern:
            details["pattern"] = pattern
        super().__init__(message=message, details=details)
        self.pattern = pattern


class RateLimitExceededError(AppException):
    """Raised when client exceeds the allowed request rate limit."""

    status_code: int = 429
    error_code: str = "RATE_LIMIT_EXCEEDED"

    def __init__(
        self,
        message: str = "Rate limit exceeded. Please try again later.",
        retry_after: int = 60,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        details["retry_after"] = retry_after
        super().__init__(message=message, details=details)
        self.retry_after = retry_after


class SafeSQLError(AppException):
    """Raised when a generated SQL query fails security checks or execution validation."""

    status_code: int = 400
    error_code: str = "SAFE_SQL_ERROR"

    def __init__(
        self,
        message: str,
        query: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if query:
            details["query"] = query
        super().__init__(message=message, details=details)
        self.query = query


class AgentOrchestrationError(AppException):
    """Raised when agent execution state machine encounters an unrecoverable failure."""

    status_code: int = 500
    error_code: str = "AGENT_ORCHESTRATION_ERROR"


class ExternalSearchError(AppException):
    """Raised when external web fallback retrieval fails or is rejected."""

    status_code: int = 502
    error_code: str = "EXTERNAL_SEARCH_ERROR"

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

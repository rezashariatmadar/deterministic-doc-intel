"""Structured JSON logging with correlation ID support."""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

# Context variable for async-safe request/correlation ID tracking
correlation_id_ctx: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def get_correlation_id() -> str | None:
    """Retrieve current correlation ID from context."""
    return correlation_id_ctx.get()


def set_correlation_id(correlation_id: str | None) -> None:
    """Set correlation ID in context for current execution trace."""
    correlation_id_ctx.set(correlation_id)


class JSONFormatter(logging.Formatter):
    """Custom formatter outputting structured JSON logs with correlation ID."""

    def format(self, record: logging.LogRecord) -> str:
        log_payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Include correlation ID from context or record attribute
        corr_id = getattr(record, "correlation_id", None) or get_correlation_id()
        if corr_id:
            log_payload["correlation_id"] = corr_id

        # Attach exception details if present
        if record.exc_info:
            log_payload["exception"] = self.formatException(record.exc_info)

        # Include any extra non-standard attributes passed via extra={}
        standard_attrs = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
            "correlation_id",
        }
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in standard_attrs and not k.startswith("_")
        }
        if extras:
            log_payload["extra"] = extras

        return json.dumps(log_payload, default=str)


def setup_logging(log_level: str = "INFO", json_format: bool = True) -> None:
    """Configure root and application loggers with structured output.

    Args:
        log_level: Desired minimum log level (e.g. 'DEBUG', 'INFO', 'WARNING', 'ERROR').
        json_format: Whether to format records as JSON (True) or standard text (False).
    """
    root_logger = logging.getLogger()
    level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicate log outputs
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        standard_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
        )
        handler.setFormatter(standard_formatter)

    root_logger.addHandler(handler)

    # Set appropriate levels for external third-party loggers
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access", "asyncpg", "httpx"):
        third_party_logger = logging.getLogger(logger_name)
        third_party_logger.setLevel(level)


def get_logger(name: str = "rag") -> logging.Logger:
    """Get a named logger instance."""
    return logging.getLogger(name)

"""Production security guardrails: Prompt Injection Detection, PII Redaction, and Rate Limiting."""

import logging
import re
import time
from typing import Any, ClassVar

from app.core.exceptions import PromptInjectionError, RateLimitExceededError

logger = logging.getLogger(__name__)


class PromptInjectionDetector:
    """Detects adversarial prompt injection, jailbreaks, and directive override attempts."""

    # Compile regex patterns covering common prompt injection vectors
    INJECTION_PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        re.compile(
            r"\b(?:ignore|disregard|forget|override|bypass)\b[\s\w]{0,25}"
            r"\b(?:previous|all|prior|system)\b[\s\w]{0,25}"
            r"\b(?:instructions|directives|rules|prompts|commands|constraints)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:system\s+prompt|system\s+message|system\s+instructions)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be)\b[\s\w]{0,25}"
            r"\b(?:DAN|unrestricted|jailbreak|evil|developer\s+mode|root|admin)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:developer\s+mode|jailbreak|jailbroken)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:output|reveal|repeat|print|dump|show)\b[\s\w]{0,20}"
            r"\b(?:the\s+above|your|the\s+original)\b[\s\w]{0,20}"
            r"\b(?:prompt|instructions|system)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:do\s+anything\s+now|mode\s+enabled)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"<!--\s*#system|\[SYSTEM\]|<\|im_start\|>system|<<SYS>>|\[INST\]",
            re.IGNORECASE,
        ),
    ]

    def detect(self, text: str) -> tuple[bool, str | None]:
        """Check if the provided text contains any known prompt injection patterns."""
        if not text:
            return False, None
        normalized = text.strip()
        for pattern in self.INJECTION_PATTERNS:
            match = pattern.search(normalized)
            if match:
                return True, match.group(0)
        return False, None

    def is_injection(self, text: str) -> bool:
        """Return True if prompt injection pattern is detected, False otherwise."""
        detected, _ = self.detect(text)
        return detected

    def validate(self, text: str) -> None:
        """Validate input query and raise PromptInjectionError if malicious injection detected."""
        detected, pattern = self.detect(text)
        if detected:
            logger.warning("Prompt injection detected matching pattern '%s'", pattern)
            raise PromptInjectionError(
                message=f"Prompt injection pattern detected: '{pattern}'",
                pattern=pattern,
            )


class PIIRedactor:
    """Redacts PII (Emails, Phones, SSNs, Cards, Keys) while preserving numbers and years."""

    # 1. Emails: standard RFC 5322 compatible pattern
    EMAIL_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    )

    # 2. US Social Security Numbers: 3 digits - 2 digits - 4 digits (strict boundary)
    SSN_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"
    )

    # 3. Credit Card Numbers: 13-19 digits, separated by dashes or spaces, or 16 continuous digits
    CREDIT_CARD_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(?:\d{4}[- ]){3}\d{4}\b|\b(?:\d{4}[- ]){2}\d{4}[- ]\d{3}\b|"
        r"\b(?:4\d{15}|5[1-5]\d{14}|6011\d{12}|3[47]\d{13})\b"
    )

    # 4. Phone Numbers: US/International telephone numbers with mandatory area code formatting
    PHONE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|"
        r"([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)"
        r"([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})\b"
    )

    # 5. API Keys / Secrets / Tokens:
    API_KEY_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(?:sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{36}|AIza[0-9A-Za-z-_]{35}|"
        r"xox[baprs]-[0-9a-zA-Z]{10,48}|key-[0-9a-zA-Z]{32})\b"
    )

    def redact(self, text: str) -> str:
        """Replace detected PII with canonical redaction tags, preserving numbers and years."""
        if not text:
            return text

        redacted = text
        # Redact API keys first (to avoid partial match by emails or tokens)
        redacted = self.API_KEY_PATTERN.sub("[REDACTED_API_KEY]", redacted)
        # Redact emails
        redacted = self.EMAIL_PATTERN.sub("[REDACTED_EMAIL]", redacted)
        # Redact SSNs
        redacted = self.SSN_PATTERN.sub("[REDACTED_SSN]", redacted)
        # Redact Credit Cards
        redacted = self.CREDIT_CARD_PATTERN.sub("[REDACTED_CREDIT_CARD]", redacted)
        # Redact Phone Numbers
        redacted = self.PHONE_PATTERN.sub("[REDACTED_PHONE]", redacted)

        return redacted

    def contains_pii(self, text: str) -> bool:
        """Check if any PII pattern exists in the input text."""
        if not text:
            return False
        return (
            bool(self.API_KEY_PATTERN.search(text))
            or bool(self.EMAIL_PATTERN.search(text))
            or bool(self.SSN_PATTERN.search(text))
            or bool(self.CREDIT_CARD_PATTERN.search(text))
            or bool(self.PHONE_PATTERN.search(text))
        )


class RateLimiter:
    """Fixed-window rate limiter with fail-open resiliency when Redis is offline or None."""

    def __init__(
        self,
        redis_client: Any = None,
        limit: int = 60,
        window_seconds: int = 60,
    ) -> None:
        self.redis = redis_client
        self.limit = limit
        self.window_seconds = window_seconds

    async def check_rate_limit(self, client_id: str) -> tuple[bool, int]:
        """Check if client_id is within rate limit.

        Returns (allowed: bool, retry_after: int).
        Fail-open: returns (True, 0) if Redis is None or fails.
        """
        if self.redis is None:
            return True, 0

        try:
            current_window = int(time.time() // self.window_seconds)
            key = f"rate_limit:{client_id}:{current_window}"

            pipe = self.redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, self.window_seconds + 1)
            results = await pipe.execute()

            request_count = results[0]
            if request_count > self.limit:
                # Calculate remaining seconds in current window
                elapsed = int(time.time() % self.window_seconds)
                retry_after = max(1, self.window_seconds - elapsed)
                return False, retry_after

            return True, 0
        except Exception as exc:
            logger.warning("RateLimiter failed open due to Redis error: %s", exc)
            return True, 0

    async def enforce(self, client_id: str) -> None:
        """Enforce rate limit, raising RateLimitExceededError if limit is breached."""
        allowed, retry_after = await self.check_rate_limit(client_id)
        if not allowed:
            logger.warning(
                "Rate limit exceeded for client %s, retry_after=%ds",
                client_id,
                retry_after,
            )
            raise RateLimitExceededError(
                message=f"Rate limit of {self.limit} requests per {self.window_seconds}s exceeded.",
                retry_after=retry_after,
            )

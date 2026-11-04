"""Text hygiene and normalization utilities for document ingestion."""

import re
import unicodedata

# Regex for hyphenated word split across line breaks (e.g. "oper-\nating" -> "operating")
HYPHENATED_LINE_BREAK_REGEX = re.compile(r"(\b[A-Za-z]+)-\s*[\r\n]+\s*([A-Za-z]+\b)")

# Non-printable control characters excluding standard tab (\t) and newline (\n)
CONTROL_CHARACTERS_REGEX = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Multiple consecutive horizontal spaces (excluding newlines)
HORIZONTAL_WHITESPACE_REGEX = re.compile(r"[^\S\r\n]+")

# Excessive vertical whitespace (more than 2 consecutive newlines)
EXCESSIVE_NEWLINES_REGEX = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """
    Clean raw extracted text from PDF/documents:
    1. Removes null bytes and non-printable control characters.
    2. Reconnects words split across line breaks by hyphens.
    3. Normalizes Unicode quotes, dashes, and standard characters.
    4. Cleans excessive horizontal whitespace and normalizes paragraphs.
    """
    if not text:
        return ""

    # 1. Strip null bytes & control chars
    cleaned = CONTROL_CHARACTERS_REGEX.sub("", text)

    # 2. Normalize unicode characters (NFKC handles special characters)
    cleaned = unicodedata.normalize("NFKC", cleaned)

    # 3. Normalize stylized quotes, apostrophes, and dashes to ASCII standards
    quote_replacements = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "—": " - ",
        "–": "-",
        "…": "...",
    }
    for orig, rep in quote_replacements.items():
        cleaned = cleaned.replace(orig, rep)

    # 4. Reconnect hyphenated words split across line breaks
    cleaned = HYPHENATED_LINE_BREAK_REGEX.sub(r"\1\2", cleaned)

    # 5. Normalize whitespace: reduce multiple horizontal spaces, keep paragraph spacing
    cleaned = "\n".join(
        HORIZONTAL_WHITESPACE_REGEX.sub(" ", line).strip() for line in cleaned.splitlines()
    )

    # 6. Reduce more than 2 consecutive newlines to double newline
    cleaned = EXCESSIVE_NEWLINES_REGEX.sub("\n\n", cleaned)

    return cleaned.strip()


def normalize_table_cell(cell: str) -> str:
    """Normalize a table cell text, stripping whitespace and escaping markdown pipes."""
    if not cell:
        return ""
    # Strip null bytes and non-printable control characters
    cleaned = CONTROL_CHARACTERS_REGEX.sub("", cell)
    cleaned = unicodedata.normalize("NFKC", cleaned)
    # Replace internal newlines with spaces and clean whitespace
    normalized = " ".join(cleaned.split())
    # Escape pipe characters so they don't break markdown tables
    normalized = normalized.replace("|", r"\|")
    return normalized

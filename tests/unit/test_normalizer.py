"""Unit tests for text hygiene and normalization."""

from app.services.ingestion.normalizer import clean_text, normalize_table_cell


def test_clean_text_removes_hyphenation_and_control_chars() -> None:
    dirty = "The oper-\nating income for Q3 was \x0015 million."
    cleaned = clean_text(dirty)
    assert cleaned == "The operating income for Q3 was 15 million."


def test_clean_text_normalizes_whitespace() -> None:
    dirty = "Revenue   rose     significantly.\n\n\nOperating  margin expanded."
    cleaned = clean_text(dirty)
    assert cleaned == "Revenue rose significantly.\n\nOperating margin expanded."


def test_clean_text_normalizes_unicode_quotes() -> None:
    dirty = "“NVIDIA’s Hopper” — ‘Next Gen’"
    cleaned = clean_text(dirty)
    assert cleaned == '"NVIDIA\'s Hopper" - \'Next Gen\''


def test_normalize_table_cell() -> None:
    assert normalize_table_cell("  $47,525 \n ") == "$47,525"
    assert normalize_table_cell("|") == "\\|"

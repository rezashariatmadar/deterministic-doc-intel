"""Unit tests for layout-aware PDF and table parsing."""

from pathlib import Path

import pymupdf as fitz
import pytest

from app.core.exceptions import DocumentParsingError
from app.services.ingestion.parser import ParsedDocument, PDFLayoutParser


def test_parser_extracts_structural_elements() -> None:
    pdf_path = Path("data/raw_corpus/nvidia_10k_sample.pdf")
    parser = PDFLayoutParser()
    parsed: ParsedDocument = parser.parse(pdf_path)

    assert parsed.total_pages == 3
    assert len(parsed.elements) > 0
    assert "NVIDIA" in parsed.title

    # Check structural element types
    element_types = {el.element_type for el in parsed.elements}
    assert "heading" in element_types
    assert "table" in element_types
    assert "prose" in element_types

    # Verify table element properties
    tables = [el for el in parsed.elements if el.element_type == "table"]
    assert len(tables) >= 2, "Must extract at least two financial tables"

    # Verify table content contains key numbers
    segment_table = tables[0]
    assert "Data Center" in segment_table.text
    assert "47,525" in segment_table.text
    assert segment_table.page_number in [2, 45]

    # Verify heading hierarchy extraction
    headings = [el for el in parsed.elements if el.element_type == "heading"]
    assert any("Item 1" in h.text or "Item 7" in h.text for h in headings)


def test_parser_handles_empty_or_nonexistent_gracefully() -> None:
    parser = PDFLayoutParser()
    empty_doc = parser.parse_text_blocks([], title="Empty Document", total_pages=0)
    assert empty_doc.total_pages == 0
    assert len(empty_doc.elements) == 0


def test_parser_extracts_bbox_and_block_index(sample_pdf_path: Path) -> None:
    parser = PDFLayoutParser()
    doc = parser.parse(sample_pdf_path)
    assert len(doc.elements) > 0
    prose_elements = [el for el in doc.elements if el.element_type == "prose"]
    assert len(prose_elements) > 0
    first = prose_elements[0]
    assert "bbox" in first.metadata
    bbox = first.metadata["bbox"]
    assert all(k in bbox for k in ("x0", "y0", "x1", "y1"))
    assert bbox["x1"] > bbox["x0"]
    assert "block_index" in first.metadata
    assert isinstance(first.metadata["block_index"], int)


def test_parser_preserves_paragraph_cohesion_and_hyphenation() -> None:
    parser = PDFLayoutParser()
    block = (
        0,
        0,
        100,
        100,
        "This is an oper-\nating expense and capital investment\nfor data center computing.",
        0,
        0,
    )
    parsed = parser.parse_text_blocks([block])
    assert len(parsed.elements) == 1
    el = parsed.elements[0]
    assert el.element_type == "prose"
    assert "operating" in el.text
    assert "\n\n" not in el.text


def test_parser_rejects_password_protected_pdf(tmp_path: Path) -> None:
    # Create an encrypted PDF requiring a password
    pdf_path = tmp_path / "encrypted.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(
        str(pdf_path),
        encryption=fitz.PDF_ENCRYPT_AES_256,
        user_pw="secret123",
        owner_pw="master123",
    )
    doc.close()

    parser = PDFLayoutParser()
    with pytest.raises(DocumentParsingError) as exc_info:
        parser.parse(pdf_path)
    err = str(exc_info.value).lower()
    assert "password-protected" in err or "encrypted" in err

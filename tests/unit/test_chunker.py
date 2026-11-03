"""Unit tests for structure-preserving chunking and contextual header injection."""

from pathlib import Path

from app.services.ingestion.chunker import DocumentChunker
from app.services.ingestion.parser import ParsedDocument, PDFLayoutParser, StructuralElement


def test_chunker_injects_breadcrumbs_and_preserves_tables() -> None:
    elements = [
        StructuralElement(
            element_type="heading",
            text="Item 7. Management's Discussion",
            page_number=1,
            heading_level=1,
        ),
        StructuralElement(
            element_type="prose",
            text="Revenue rose 206% year-over-year driven by compute networking.",
            page_number=1,
        ),
        StructuralElement(
            element_type="table",
            text="| Segment | Revenue |\n|---|---|\n| Data Center | $47.5B |",
            page_number=1,
        ),
    ]
    doc = ParsedDocument(title="NVIDIA 10-K", total_pages=1, elements=elements)
    chunker = DocumentChunker(target_tokens=512, overlap_tokens=50)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.token_count <= 512
        assert chunk.contextualized_text.startswith(
            "[Document: NVIDIA 10-K | Section: Item 7. Management's Discussion | Page: 1]"
        )
        assert "Revenue rose" in chunk.raw_text or "Data Center" in chunk.raw_text
        assert chunk.chunk_index >= 0


def test_chunker_splits_large_table_repeating_headers() -> None:
    # Build a table with 50 rows that exceeds 100 tokens
    header = "| Metric ID | Description | Value |"
    separator = "|---|---|---|"
    rows = [
        f"| METRIC_{i:03d} | Key operational metric number {i} | ${i * 100}M |" for i in range(50)
    ]
    table_text = "\n".join([header, separator, *rows])

    elements = [
        StructuralElement(
            element_type="heading",
            text="Item 8. Financial Disclosures",
            page_number=2,
            heading_level=1,
        ),
        StructuralElement(
            element_type="table",
            text=table_text,
            page_number=2,
        ),
    ]
    doc = ParsedDocument(title="NVIDIA 10-K", total_pages=2, elements=elements)
    # Set small target tokens to force table split
    chunker = DocumentChunker(target_tokens=150, overlap_tokens=20)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2, "Large table must be split into multiple chunks"
    for chunk in chunks:
        # Every continuation chunk of the table must retain the markdown table header
        assert "| Metric ID | Description | Value |" in chunk.raw_text
        assert "|---|---|---|" in chunk.raw_text


def test_chunker_on_real_pdf_fixture() -> None:
    pdf_path = Path("data/raw_corpus/nvidia_10k_sample.pdf")
    parser = PDFLayoutParser()
    doc = parser.parse(pdf_path)

    chunker = DocumentChunker(target_tokens=512, overlap_tokens=50)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    for idx, c in enumerate(chunks):
        assert c.chunk_index == idx
        assert c.token_count <= 512
        assert c.token_count > 0
        assert c.page_start >= 1
        assert c.page_end >= c.page_start
        assert c.section_path != ""
        assert c.contextualized_text.startswith("[Document:")


def test_chunker_transfers_bbox_metadata(sample_pdf_path: Path) -> None:
    parser = PDFLayoutParser()
    doc = parser.parse(sample_pdf_path)
    chunker = DocumentChunker()
    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 0
    assert "bbox" in chunks[0].metadata
    assert all(k in chunks[0].metadata["bbox"] for k in ("x0", "y0", "x1", "y1"))


def test_chunker_multipage_prose_stores_per_page_bboxes() -> None:
    el1 = StructuralElement(
        element_type="prose",
        text="This is sentence one on page one of a multi-page section.",
        page_number=1,
        metadata={"bbox": {"x0": 50.0, "y0": 100.0, "x1": 200.0, "y1": 150.0}, "block_index": 1},
    )
    el2 = StructuralElement(
        element_type="prose",
        text="This is continuation sentence two on page two of the section.",
        page_number=2,
        metadata={"bbox": {"x0": 60.0, "y0": 80.0, "x1": 220.0, "y1": 140.0}, "block_index": 2},
    )
    doc = ParsedDocument(title="Multi-page Doc", total_pages=2, elements=[el1, el2])
    chunker = DocumentChunker(target_tokens=500)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.page_start == 1
    assert chunk.page_end == 2
    assert "bboxes" in chunk.metadata
    assert len(chunk.metadata["bboxes"]) == 2
    assert chunk.metadata["bboxes"][0]["page"] == 1
    assert chunk.metadata["bboxes"][0]["bbox"] == {
        "x0": 50.0,
        "y0": 100.0,
        "x1": 200.0,
        "y1": 150.0,
    }
    assert chunk.metadata["bboxes"][1]["page"] == 2
    assert chunk.metadata["bboxes"][1]["bbox"] == {
        "x0": 60.0,
        "y0": 80.0,
        "x1": 220.0,
        "y1": 140.0,
    }


def test_chunker_sliding_window_prose_overlap() -> None:
    p1 = StructuralElement(
        element_type="prose",
        text="Alpha apple apricot avocado banana blueberry blackberry cantaloupe cherry.",
        page_number=1,
    )
    p2 = StructuralElement(
        element_type="prose",
        text="Date elderberry fig grape grapefruit guava honeydew kiwi lemon lime mango.",
        page_number=1,
    )
    doc = ParsedDocument(title="Doc", total_pages=1, elements=[p1, p2])
    # Target tokens small enough to split p1 and p2 into separate chunks
    chunker = DocumentChunker(target_tokens=15, overlap_tokens=5)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2
    # With sliding window overlap, trailing tokens from chunk 0 must appear in chunk 1
    words_chunk0 = set(chunks[0].raw_text.lower().split())
    words_chunk1 = set(chunks[1].raw_text.lower().split())
    common = words_chunk0.intersection(words_chunk1)
    assert len(common) > 0


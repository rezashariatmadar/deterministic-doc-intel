"""Layout-aware PDF document and table parser."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf as fitz  # PyMuPDF

from app.core.exceptions import DocumentParsingError
from app.services.ingestion.normalizer import clean_text, normalize_table_cell


@dataclass
class StructuralElement:
    """Represents a discrete structural unit (heading, table, prose) extracted from a document."""

    element_type: str  # "heading" | "table" | "prose" | "list"
    text: str
    page_number: int
    heading_level: int = 0  # 1 for H1 (Item/Part), 2 for H2, 3 for H3
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """Represents a fully parsed document preserving structural hierarchy."""

    title: str
    total_pages: int
    elements: list[StructuralElement] = field(default_factory=list)


class PDFLayoutParser:
    """Parses PDF documents into typed structural elements preserving tables and hierarchy."""

    # Heading patterns typical in SEC 10-K and technical reports
    HEADING_H1_REGEX = re.compile(
        r"^(PART\s+[IVX]+|Item\s+\d+[A-Z]?\.\s+.*|FORM\s+10-K|UNITED\s+STATES\s+SECURITIES.*)$",
        re.IGNORECASE,
    )
    HEADING_H2_REGEX = re.compile(
        r"^(Revenue\s+by\s+Reportable\s+Segment|Consolidated\s+Statements\s+of\s+.*|Management's\s+Discussion.*)$",
        re.IGNORECASE,
    )

    # Patterns for repetitive headers and footers that should be stripped
    REPETITIVE_HEADER_FOOTER_REGEX = re.compile(
        r"^(Page\s+\d+|NVIDIA\s+CORPORATION\s+\|\s+\d{4}\s+ANNUAL\s+REPORT)$",
        re.IGNORECASE,
    )

    def parse(self, pdf_path: Path) -> ParsedDocument:
        """Parse a PDF file from disk into a ParsedDocument."""
        if not pdf_path.exists():
            raise DocumentParsingError(
                f"File not found: {pdf_path}",
                filename=pdf_path.name,
            )

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            raise DocumentParsingError(
                f"Failed to open PDF document: {e}",
                filename=pdf_path.name,
            ) from e

        if doc.needs_pass:
            doc.close()
            raise DocumentParsingError(
                "PDF is password-protected or encrypted.",
                filename=pdf_path.name,
            )

        try:
            total_pages = len(doc)
            elements: list[StructuralElement] = []
            doc_title = self._extract_title(doc, pdf_path.stem)

            for page_idx in range(total_pages):
                page = doc[page_idx]
                page_num = page_idx + 1

                # Extract text blocks: tuple (x0, y0, x1, y1, text, block_no, block_type)
                blocks = page.get_text("blocks")
                page_elements = self._process_page_blocks(blocks, page_num)
                elements.extend(page_elements)

            return ParsedDocument(
                title=doc_title,
                total_pages=total_pages,
                elements=elements,
            )
        finally:
            doc.close()

    def parse_text_blocks(
        self,
        blocks: list[Any],
        title: str = "Document",
        total_pages: int = 1,
    ) -> ParsedDocument:
        """Helper method for parsing raw blocks (useful for unit testing / empty documents)."""
        elements = self._process_page_blocks(blocks, page_num=1) if blocks else []
        return ParsedDocument(
            title=title,
            total_pages=total_pages,
            elements=elements,
        )

    def _extract_title(self, doc: fitz.Document, default_title: str) -> str:
        """Determine document title from metadata or page 1 prominent text."""
        meta_title = doc.metadata.get("title") if doc.metadata else None
        if meta_title and len(str(meta_title).strip()) > 3:
            return str(meta_title).strip()

        if len(doc) > 0:
            first_page_text = doc[0].get_text()
            has_form_10k = "FORM 10-K" in first_page_text.upper()
            company_name = ""
            for line in first_page_text.splitlines():
                stripped = line.strip()
                is_nvidia = "NVIDIA" in stripped.upper()
                is_corp = "CORPORATION" in stripped.upper() or "CORP" in stripped.upper()
                if is_nvidia and is_corp:
                    company_name = stripped
                    break

            if company_name:
                return f"{company_name} Form 10-K" if has_form_10k else company_name

            if has_form_10k:
                return "Form 10-K Annual Report"

        return default_title.replace("_", " ").title()

    def _extract_block_metadata(self, block: Any) -> dict[str, Any]:
        """Extract bounding box and block index from PyMuPDF block tuple."""
        if len(block) >= 6:
            try:
                x0, y0, x1, y1 = (
                    float(block[0]),
                    float(block[1]),
                    float(block[2]),
                    float(block[3]),
                )
                block_no = int(block[5])
                return {
                    "bbox": {
                        "x0": round(x0, 2),
                        "y0": round(y0, 2),
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                    },
                    "block_index": block_no,
                }
            except (ValueError, TypeError, IndexError):
                pass
        return {}

    def _copy_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Deep copy metadata dictionary with bbox."""
        if not metadata:
            return {}
        copied = dict(metadata)
        if "bbox" in copied and isinstance(copied["bbox"], dict):
            copied["bbox"] = dict(copied["bbox"])
        return copied

    def _create_prose_element(
        self,
        lines: list[str],
        page_num: int,
        block_metadata: dict[str, Any],
    ) -> StructuralElement | None:
        """Create a single cohesive prose element from lines, normalizing hyphenation."""
        if not lines:
            return None
        joined = "\n".join(lines)
        cleaned = clean_text(joined)
        sub_lines = cleaned.splitlines()
        is_list = any(sl.strip().startswith(("-", "*", "•", "1.", "2.")) for sl in sub_lines)
        if not is_list:
            cleaned = " ".join(sub_lines)
        if not cleaned:
            return None
        return StructuralElement(
            element_type="prose",
            text=cleaned,
            page_number=page_num,
            heading_level=0,
            metadata=self._copy_metadata(block_metadata),
        )

    def _process_page_blocks(
        self,
        blocks: list[Any],
        page_num: int,
    ) -> list[StructuralElement]:
        """Process text blocks of a page, classifying them into headings, tables, or prose."""
        page_elements: list[StructuralElement] = []

        for block in blocks:
            # block[4] contains the block text
            block_text = block[4].strip()
            if not block_text:
                continue

            # Strip repeating page footers/headers (e.g. "Page 45", "NVIDIA CORP | 2024 REPORT")
            if self.REPETITIVE_HEADER_FOOTER_REGEX.match(block_text):
                continue

            block_metadata = self._extract_block_metadata(block)

            # 1. Detect Tables: ASCII table markers (+--- or | col |)
            if self._is_table_block(block_text):
                table_md = self._convert_ascii_table_to_markdown(block_text)
                if table_md:
                    table_meta: dict[str, Any] = {"format": "markdown_table"}
                    if block_metadata:
                        table_meta.update(self._copy_metadata(block_metadata))
                    page_elements.append(
                        StructuralElement(
                            element_type="table",
                            text=table_md,
                            page_number=page_num,
                            heading_level=0,
                            metadata=table_meta,
                        )
                    )
                continue

            # 2. Process block lines: accumulate cohesive prose paragraphs and detect headings
            lines = [line.strip() for line in block_text.splitlines() if line.strip()]
            prose_lines: list[str] = []

            for line in lines:
                # Skip repeating header/footer line-level
                if self.REPETITIVE_HEADER_FOOTER_REGEX.match(line):
                    continue

                if self.HEADING_H1_REGEX.match(line):
                    prose_el = self._create_prose_element(prose_lines, page_num, block_metadata)
                    if prose_el:
                        page_elements.append(prose_el)
                    prose_lines.clear()
                    page_elements.append(
                        StructuralElement(
                            element_type="heading",
                            text=clean_text(line),
                            page_number=page_num,
                            heading_level=1,
                            metadata=self._copy_metadata(block_metadata),
                        )
                    )
                elif self.HEADING_H2_REGEX.match(line):
                    prose_el = self._create_prose_element(prose_lines, page_num, block_metadata)
                    if prose_el:
                        page_elements.append(prose_el)
                    prose_lines.clear()
                    page_elements.append(
                        StructuralElement(
                            element_type="heading",
                            text=clean_text(line),
                            page_number=page_num,
                            heading_level=2,
                            metadata=self._copy_metadata(block_metadata),
                        )
                    )
                else:
                    prose_lines.append(line)

            prose_el = self._create_prose_element(prose_lines, page_num, block_metadata)
            if prose_el:
                page_elements.append(prose_el)

        return page_elements

    def _is_table_block(self, text: str) -> bool:
        """Check if a text block contains a structured table (ASCII grid or pipe delimited)."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 3:
            return False

        pipe_lines = sum(1 for line in lines if line.startswith("|") and line.endswith("|"))
        border_lines = sum(1 for line in lines if line.startswith("+--") or line.startswith("+=="))

        return pipe_lines >= 2 or (border_lines >= 2 and pipe_lines >= 1)

    def _convert_ascii_table_to_markdown(self, raw_table_text: str) -> str:
        """Convert an ASCII bordered table into clean GitHub-Flavored Markdown."""
        lines = [line.strip() for line in raw_table_text.splitlines() if line.strip()]
        data_rows: list[list[str]] = []

        for line in lines:
            # Skip horizontal separator rows (+---+---+)
            if line.startswith("+") and ("-" in line or "=" in line):
                continue

            if line.startswith("|") and line.endswith("|"):
                # Split columns by pipe and normalize cell content
                raw_cells = line[1:-1].split("|")
                cleaned_cells = [normalize_table_cell(c) for c in raw_cells]
                data_rows.append(cleaned_cells)

        if not data_rows:
            return ""

        # Format as Markdown table: header row + separator row + data rows
        header_row = "| " + " | ".join(data_rows[0]) + " |"
        col_count = len(data_rows[0])
        separator_row = "| " + " | ".join(["---"] * col_count) + " |"

        body_rows = ["| " + " | ".join(row) + " |" for row in data_rows[1:]]

        return "\n".join([header_row, separator_row, *body_rows])

"""Structure-preserving document chunking engine with contextual header injection."""

import re
from dataclasses import dataclass, field
from typing import Any

import tiktoken

from app.services.ingestion.parser import ParsedDocument


@dataclass
class DocumentChunkDraft:
    """Represents a chunk ready for embedding and persistence."""

    chunk_index: int
    raw_text: str
    contextualized_text: str
    token_count: int
    page_start: int
    page_end: int
    section_path: str
    element_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentChunker:
    """Segments parsed structural documents into retrieval units preserving tables and hierarchy."""

    def __init__(
        self,
        target_tokens: int = 512,
        overlap_tokens: int = 50,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count exact tokens using tiktoken."""
        return len(self.tokenizer.encode(text))

    def chunk_document(self, doc: ParsedDocument) -> list[DocumentChunkDraft]:
        """Convert a ParsedDocument into a sequence of contextualized chunks."""
        if not doc.elements:
            return []

        chunks: list[DocumentChunkDraft] = []
        heading_stack: dict[int, str] = {}  # Map level -> heading text
        chunk_idx = 0

        # Buffer for accumulating prose within the same section
        prose_buffer: list[str] = []
        prose_pages: list[int] = []
        current_section = "General"

        for el in doc.elements:
            # 1. Handle Heading elements: update the hierarchical breadcrumb stack
            if el.element_type == "heading":
                # Flush existing prose before changing sections
                if prose_buffer:
                    chunk = self._flush_prose_buffer(
                        prose_buffer,
                        prose_pages,
                        doc.title,
                        current_section,
                        chunk_idx,
                    )
                    if chunk:
                        chunks.append(chunk)
                        chunk_idx += 1
                    prose_buffer = []
                    prose_pages = []

                level = el.heading_level or 1
                heading_stack[level] = el.text
                # Clear deeper heading levels when a shallower level arrives
                keys_to_clear = [k for k in heading_stack if k > level]
                for k in keys_to_clear:
                    del heading_stack[k]

                # Format breadcrumb section path: "Item 7 > Revenue by Segment"
                current_section = " > ".join(heading_stack[k] for k in sorted(heading_stack))
                continue

            # 2. Handle Table elements: preserve atomicity or split large tables
            if el.element_type == "table":
                # Flush existing prose before table
                if prose_buffer:
                    chunk = self._flush_prose_buffer(
                        prose_buffer,
                        prose_pages,
                        doc.title,
                        current_section,
                        chunk_idx,
                    )
                    if chunk:
                        chunks.append(chunk)
                        chunk_idx += 1
                    prose_buffer = []
                    prose_pages = []

                table_chunks = self._chunk_table(
                    table_text=el.text,
                    doc_title=doc.title,
                    section_path=current_section,
                    page_num=el.page_number,
                    start_chunk_idx=chunk_idx,
                )
                chunks.extend(table_chunks)
                chunk_idx += len(table_chunks)
                continue

            # 3. Handle Prose elements
            el_tokens = self.count_tokens(el.text)
            current_buffer_tokens = self.count_tokens(" ".join(prose_buffer))

            if current_buffer_tokens + el_tokens <= self.target_tokens:
                prose_buffer.append(el.text)
                prose_pages.append(el.page_number)
            else:
                # Flush buffer
                if prose_buffer:
                    chunk = self._flush_prose_buffer(
                        prose_buffer,
                        prose_pages,
                        doc.title,
                        current_section,
                        chunk_idx,
                    )
                    if chunk:
                        chunks.append(chunk)
                        chunk_idx += 1
                    prose_buffer = []
                    prose_pages = []

                # If single paragraph exceeds target tokens, split by sentences
                if el_tokens > self.target_tokens:
                    sentence_chunks = self._split_large_text(
                        text=el.text,
                        doc_title=doc.title,
                        section_path=current_section,
                        page_num=el.page_number,
                        start_chunk_idx=chunk_idx,
                    )
                    chunks.extend(sentence_chunks)
                    chunk_idx += len(sentence_chunks)
                else:
                    prose_buffer.append(el.text)
                    prose_pages.append(el.page_number)

        # Flush any remaining prose
        if prose_buffer:
            chunk = self._flush_prose_buffer(
                prose_buffer,
                prose_pages,
                doc.title,
                current_section,
                chunk_idx,
            )
            if chunk:
                chunks.append(chunk)

        return chunks

    def _flush_prose_buffer(
        self,
        buffer: list[str],
        pages: list[int],
        doc_title: str,
        section_path: str,
        chunk_idx: int,
    ) -> DocumentChunkDraft | None:
        """Create a DocumentChunkDraft from buffered prose elements."""
        raw_text = "\n\n".join(buffer).strip()
        if not raw_text:
            return None

        page_start = min(pages) if pages else 1
        page_end = max(pages) if pages else page_start

        breadcrumb = f"[Document: {doc_title} | Section: {section_path} | Page: {page_start}]"
        contextualized_text = f"{breadcrumb} {raw_text}"
        tokens = self.count_tokens(raw_text)

        return DocumentChunkDraft(
            chunk_index=chunk_idx,
            raw_text=raw_text,
            contextualized_text=contextualized_text,
            token_count=tokens,
            page_start=page_start,
            page_end=page_end,
            section_path=section_path,
            element_type="prose",
        )

    def _chunk_table(
        self,
        table_text: str,
        doc_title: str,
        section_path: str,
        page_num: int,
        start_chunk_idx: int,
    ) -> list[DocumentChunkDraft]:
        """Split a table while keeping headers on continuation slices."""
        lines = [line.strip() for line in table_text.splitlines() if line.strip()]
        if len(lines) < 3:
            # Small or invalid table, wrap as single chunk
            tokens = self.count_tokens(table_text)
            breadcrumb = f"[Document: {doc_title} | Section: {section_path} | Page: {page_num}]"
            return [
                DocumentChunkDraft(
                    chunk_index=start_chunk_idx,
                    raw_text=table_text,
                    contextualized_text=f"{breadcrumb} {table_text}",
                    token_count=tokens,
                    page_start=page_num,
                    page_end=page_num,
                    section_path=section_path,
                    element_type="table",
                )
            ]

        # Extract markdown table header and separator row
        header_row = lines[0]
        separator_row = lines[1]
        data_rows = lines[2:]

        table_tokens = self.count_tokens(table_text)
        if table_tokens <= self.target_tokens:
            # Table fits cleanly in one chunk
            breadcrumb = f"[Document: {doc_title} | Section: {section_path} | Page: {page_num}]"
            return [
                DocumentChunkDraft(
                    chunk_index=start_chunk_idx,
                    raw_text=table_text,
                    contextualized_text=f"{breadcrumb} {table_text}",
                    token_count=table_tokens,
                    page_start=page_num,
                    page_end=page_num,
                    section_path=section_path,
                    element_type="table",
                )
            ]

        # Table exceeds target tokens: slice data rows and repeat headers on every slice
        table_chunks: list[DocumentChunkDraft] = []
        current_rows: list[str] = []
        base_header_tokens = self.count_tokens(f"{header_row}\n{separator_row}")
        curr_idx = start_chunk_idx

        for row in data_rows:
            candidate_tokens = base_header_tokens + self.count_tokens(
                "\n".join([*current_rows, row])
            )

            if candidate_tokens <= self.target_tokens or not current_rows:
                current_rows.append(row)
            else:
                # Flush slice
                slice_text = "\n".join([header_row, separator_row, *current_rows])
                breadcrumb = f"[Document: {doc_title} | Section: {section_path} | Page: {page_num}]"
                table_chunks.append(
                    DocumentChunkDraft(
                        chunk_index=curr_idx,
                        raw_text=slice_text,
                        contextualized_text=f"{breadcrumb} {slice_text}",
                        token_count=self.count_tokens(slice_text),
                        page_start=page_num,
                        page_end=page_num,
                        section_path=section_path,
                        element_type="table",
                    )
                )
                curr_idx += 1
                current_rows = [row]

        if current_rows:
            slice_text = "\n".join([header_row, separator_row, *current_rows])
            breadcrumb = f"[Document: {doc_title} | Section: {section_path} | Page: {page_num}]"
            table_chunks.append(
                DocumentChunkDraft(
                    chunk_index=curr_idx,
                    raw_text=slice_text,
                    contextualized_text=f"{breadcrumb} {slice_text}",
                    token_count=self.count_tokens(slice_text),
                    page_start=page_num,
                    page_end=page_num,
                    section_path=section_path,
                    element_type="table",
                )
            )

        return table_chunks

    def _split_large_text(
        self,
        text: str,
        doc_title: str,
        section_path: str,
        page_num: int,
        start_chunk_idx: int,
    ) -> list[DocumentChunkDraft]:
        """Split oversized prose text along sentence boundaries with token budgeting."""
        # Simple sentence splitter
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[DocumentChunkDraft] = []
        curr_sentences: list[str] = []
        curr_idx = start_chunk_idx

        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue

            candidate = " ".join([*curr_sentences, s_clean])
            if self.count_tokens(candidate) <= self.target_tokens or not curr_sentences:
                curr_sentences.append(s_clean)
            else:
                raw_chunk = " ".join(curr_sentences)
                breadcrumb = f"[Document: {doc_title} | Section: {section_path} | Page: {page_num}]"
                chunks.append(
                    DocumentChunkDraft(
                        chunk_index=curr_idx,
                        raw_text=raw_chunk,
                        contextualized_text=f"{breadcrumb} {raw_chunk}",
                        token_count=self.count_tokens(raw_chunk),
                        page_start=page_num,
                        page_end=page_num,
                        section_path=section_path,
                        element_type="prose",
                    )
                )
                curr_idx += 1
                curr_sentences = [s_clean]

        if curr_sentences:
            raw_chunk = " ".join(curr_sentences)
            breadcrumb = f"[Document: {doc_title} | Section: {section_path} | Page: {page_num}]"
            chunks.append(
                DocumentChunkDraft(
                    chunk_index=curr_idx,
                    raw_text=raw_chunk,
                    contextualized_text=f"{breadcrumb} {raw_chunk}",
                    token_count=self.count_tokens(raw_chunk),
                    page_start=page_num,
                    page_end=page_num,
                    section_path=section_path,
                    element_type="prose",
                )
            )

        return chunks

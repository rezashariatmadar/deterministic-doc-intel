"""Structure-preserving document chunking engine with contextual header injection."""

import re
from dataclasses import dataclass, field
from typing import Any

import tiktoken

from app.services.ingestion.parser import ParsedDocument, StructuralElement


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
        prose_elements: list[StructuralElement] = []
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
                        elements=prose_elements,
                    )
                    if chunk:
                        chunks.append(chunk)
                        chunk_idx += 1
                    prose_buffer = []
                    prose_pages = []
                    prose_elements = []

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
                        elements=prose_elements,
                    )
                    if chunk:
                        chunks.append(chunk)
                        chunk_idx += 1
                    prose_buffer = []
                    prose_pages = []
                    prose_elements = []

                table_chunks = self._chunk_table(
                    table_text=el.text,
                    doc_title=doc.title,
                    section_path=current_section,
                    page_num=el.page_number,
                    start_chunk_idx=chunk_idx,
                    metadata=el.metadata,
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
                prose_elements.append(el)
            else:
                # Flush buffer
                if prose_buffer:
                    chunk = self._flush_prose_buffer(
                        prose_buffer,
                        prose_pages,
                        doc.title,
                        current_section,
                        chunk_idx,
                        elements=prose_elements,
                    )
                    if chunk:
                        chunks.append(chunk)
                        chunk_idx += 1
                    overlap_buf, overlap_pgs, overlap_els = self._get_overlap_tail(
                        prose_buffer, prose_pages, prose_elements
                    )
                    prose_buffer = overlap_buf
                    prose_pages = overlap_pgs
                    prose_elements = overlap_els

                # If single paragraph exceeds target tokens, split by sentences
                if el_tokens > self.target_tokens:
                    sentence_chunks = self._split_large_text(
                        text=el.text,
                        doc_title=doc.title,
                        section_path=current_section,
                        page_num=el.page_number,
                        start_chunk_idx=chunk_idx,
                        metadata=el.metadata,
                    )
                    chunks.extend(sentence_chunks)
                    chunk_idx += len(sentence_chunks)
                    if sentence_chunks and self.overlap_tokens:
                        last_c = sentence_chunks[-1]
                        toks = self.tokenizer.encode(last_c.raw_text)
                        ov_text = (
                            self.tokenizer.decode(toks[-self.overlap_tokens :]).strip()
                            if len(toks) > self.overlap_tokens
                            else last_c.raw_text
                        )
                        prose_buffer = [ov_text]
                        prose_pages = [last_c.page_end]
                        prose_elements = [
                            StructuralElement(
                                element_type="prose",
                                text=ov_text,
                                page_number=last_c.page_end,
                                metadata=self._copy_metadata(last_c.metadata),
                            )
                        ]
                else:
                    prose_buffer.append(el.text)
                    prose_pages.append(el.page_number)
                    prose_elements.append(el)

        # Flush any remaining prose
        if prose_buffer:
            chunk = self._flush_prose_buffer(
                prose_buffer,
                prose_pages,
                doc.title,
                current_section,
                chunk_idx,
                elements=prose_elements,
            )
            if chunk:
                chunks.append(chunk)

        return chunks

    def _merge_element_metadata(self, elements: list[StructuralElement]) -> dict[str, Any]:
        """Merge bounding boxes and block provenance from accumulated structural elements."""
        merged: dict[str, Any] = {}
        bboxes: list[dict[str, float]] = []
        page_bboxes: dict[int, list[dict[str, float]]] = {}
        block_indices: list[int] = []

        pages = [el.page_number for el in elements if el.page_number is not None]
        page_start = min(pages) if pages else 1
        page_end = max(pages) if pages else 1
        is_multipage = page_start != page_end

        for el in elements:
            if not el.metadata:
                continue
            for k, v in el.metadata.items():
                if k not in ("bbox", "bboxes", "block_index", "block_indices"):
                    merged[k] = v
            if "bbox" in el.metadata and isinstance(el.metadata["bbox"], dict):
                b = el.metadata["bbox"]
                if all(coord in b for coord in ("x0", "y0", "x1", "y1")):
                    bboxes.append(b)
                    page_bboxes.setdefault(el.page_number, []).append(b)
            if (
                "block_index" in el.metadata
                and isinstance(el.metadata["block_index"], int)
                and el.metadata["block_index"] not in block_indices
            ):
                block_indices.append(el.metadata["block_index"])

        if bboxes:
            merged["bbox"] = {
                "x0": round(min(float(b["x0"]) for b in bboxes), 2),
                "y0": round(min(float(b["y0"]) for b in bboxes), 2),
                "x1": round(max(float(b["x1"]) for b in bboxes), 2),
                "y1": round(max(float(b["y1"]) for b in bboxes), 2),
            }

        # If elements span across multiple pages, store per-page bounding boxes
        if (is_multipage or len(page_bboxes) > 1) and page_bboxes:
            merged["bboxes"] = [
                {
                    "page": p,
                    "bbox": {
                        "x0": round(min(float(b["x0"]) for b in p_boxes), 2),
                        "y0": round(min(float(b["y0"]) for b in p_boxes), 2),
                        "x1": round(max(float(b["x1"]) for b in p_boxes), 2),
                        "y1": round(max(float(b["y1"]) for b in p_boxes), 2),
                    },
                }
                for p, p_boxes in sorted(page_bboxes.items())
            ]

        if block_indices:
            merged["block_index"] = block_indices[0]
            if len(block_indices) > 1:
                merged["block_indices"] = block_indices

        return merged

    def _copy_metadata(self, metadata: dict[str, Any] | None) -> dict[str, Any]:
        """Deep copy metadata dictionary with bbox and bboxes."""
        if not metadata:
            return {}
        copied = dict(metadata)
        if "bbox" in copied and isinstance(copied["bbox"], dict):
            copied["bbox"] = dict(copied["bbox"])
        if "bboxes" in copied and isinstance(copied["bboxes"], list):
            copied["bboxes"] = [
                {
                    "page": b["page"],
                    "bbox": dict(b["bbox"]) if isinstance(b.get("bbox"), dict) else {},
                }
                for b in copied["bboxes"]
                if isinstance(b, dict) and "page" in b
            ]
        return copied

    def _get_overlap_tail(
        self,
        buffer: list[str],
        pages: list[int],
        elements: list[StructuralElement],
    ) -> tuple[list[str], list[int], list[StructuralElement]]:
        """Extract trailing tokens from buffered prose to seed sliding-window overlap."""
        if not self.overlap_tokens or not buffer:
            return [], [], []

        full_text = " ".join(buffer).strip()
        if not full_text:
            return [], [], []

        toks = self.tokenizer.encode(full_text)
        if len(toks) <= self.overlap_tokens:
            overlap_text = full_text
        else:
            overlap_text = self.tokenizer.decode(toks[-self.overlap_tokens :]).strip()

        if not overlap_text:
            return [], [], []

        last_page = pages[-1] if pages else 1
        last_el = (
            elements[-1]
            if elements
            else StructuralElement(element_type="prose", text=overlap_text, page_number=last_page)
        )
        overlap_el = StructuralElement(
            element_type="prose",
            text=overlap_text,
            page_number=last_page,
            metadata=self._copy_metadata(last_el.metadata),
        )
        return [overlap_text], [last_page], [overlap_el]

    def _flush_prose_buffer(
        self,
        buffer: list[str],
        pages: list[int],
        doc_title: str,
        section_path: str,
        chunk_idx: int,
        elements: list[StructuralElement] | None = None,
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
        metadata = self._merge_element_metadata(elements or [])

        return DocumentChunkDraft(
            chunk_index=chunk_idx,
            raw_text=raw_text,
            contextualized_text=contextualized_text,
            token_count=tokens,
            page_start=page_start,
            page_end=page_end,
            section_path=section_path,
            element_type="prose",
            metadata=metadata,
        )

    def _chunk_table(
        self,
        table_text: str,
        doc_title: str,
        section_path: str,
        page_num: int,
        start_chunk_idx: int,
        metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunkDraft]:
        """Split a table while keeping headers on continuation slices."""
        chunk_metadata = self._copy_metadata(metadata)
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
                    metadata=self._copy_metadata(chunk_metadata),
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
                    metadata=self._copy_metadata(chunk_metadata),
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
                        metadata=self._copy_metadata(chunk_metadata),
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
                    metadata=self._copy_metadata(chunk_metadata),
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
        metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunkDraft]:
        """Split oversized prose text along sentence boundaries with token budgeting."""
        chunk_metadata = self._copy_metadata(metadata)
        # Simple sentence splitter
        raw_sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences: list[str] = []
        step_budget = (
            max(1, self.target_tokens - self.overlap_tokens)
            if self.overlap_tokens
            else self.target_tokens
        )
        for s in raw_sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            if self.count_tokens(s_clean) <= step_budget:
                sentences.append(s_clean)
            else:
                # Decompose oversized sentence into word-based chunks that fit within step_budget
                words = s_clean.split()
                sub_buf: list[str] = []
                for w in words:
                    cand_words = " ".join([*sub_buf, w])
                    if self.count_tokens(cand_words) <= step_budget:
                        sub_buf.append(w)
                    else:
                        if sub_buf:
                            sentences.append(" ".join(sub_buf))
                        sub_buf = [w]
                if sub_buf:
                    sentences.append(" ".join(sub_buf))

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
                        metadata=self._copy_metadata(chunk_metadata),
                    )
                )
                curr_idx += 1
                if self.overlap_tokens and curr_sentences:
                    toks = self.tokenizer.encode(raw_chunk)
                    if len(toks) > self.overlap_tokens:
                        overlap_s = self.tokenizer.decode(toks[-self.overlap_tokens :]).strip()
                    else:
                        overlap_s = raw_chunk
                    if self.count_tokens(f"{overlap_s} {s_clean}") <= self.target_tokens:
                        curr_sentences = [overlap_s, s_clean]
                    else:
                        curr_sentences = [s_clean]
                else:
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
                    metadata=self._copy_metadata(chunk_metadata),
                )
            )

        return chunks

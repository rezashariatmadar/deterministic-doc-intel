"""Grounded generation engine with strict citation enforcement and hallucination detection."""

import re
import time
from collections.abc import AsyncGenerator

from app.schemas.query import Citation, RAGResponse, RetrievedChunk
from app.services.providers.base import LLMProvider

GROUNDED_SYSTEM_PROMPT = """You are an accurate, strictly factual financial analysis assistant.
Answer the user's question using ONLY the provided sources inside <sources> XML tags.

CRITICAL GUIDELINES:
1. Base every factual claim or statistic directly on the provided context.
2. Every factual statement MUST be followed by its bracketed citation, e.g. [1] or [1][2].
3. If the provided sources do not contain sufficient evidence to answer the question, state:
   "The provided documents do not contain sufficient information to answer this question."
4. Never extrapolate, assume, or hallucinate citations not present in <sources>.
"""


class GroundedGenerator:
    """Orchestrates XML prompt assembly, citation verification, and confidence scoring."""

    def __init__(self, llm_provider: LLMProvider | None = None) -> None:
        self.llm_provider = llm_provider

    def build_context_xml(self, chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks into structured XML with clear source IDs."""
        xml_parts: list[str] = ["<sources>"]
        for chunk in chunks:
            part = (
                f'  <source id="{chunk.source_id}" '
                f'document="{chunk.document_title}" '
                f'section="{chunk.section_path}" '
                f'pages="{chunk.page_start}-{chunk.page_end}">\n'
                f"    {chunk.contextualized_text}\n"
                f"  </source>"
            )
            xml_parts.append(part)
        xml_parts.append("</sources>")
        return "\n".join(xml_parts)

    def verify_and_extract_citations(
        self, raw_answer: str, chunks: list[RetrievedChunk]
    ) -> tuple[str, list[Citation]]:
        """
        Verify all bracketed citations [N] against retrieved chunks.
        Valid citations are extracted into structured Citation models.
        Hallucinated citations are rewritten as [unverified: N].
        """
        chunk_map = {c.source_id: c for c in chunks}
        seen_source_ids: set[int] = set()
        citations: list[Citation] = []

        def replace_citation(match: re.Match[str]) -> str:
            source_id = int(match.group(1))
            if source_id in chunk_map:
                if source_id not in seen_source_ids:
                    seen_source_ids.add(source_id)
                    chunk = chunk_map[source_id]
                    # Extract first sentence of text as candidate quote
                    first_sent = chunk.text.split(".")[0].strip()
                    quote = (
                        f"{first_sent}."
                        if first_sent and not first_sent.endswith(".")
                        else first_sent
                    )
                    citations.append(
                        Citation(
                            source_id=source_id,
                            document_title=chunk.document_title,
                            page_start=chunk.page_start,
                            page_end=chunk.page_end,
                            section_path=chunk.section_path,
                            quote=quote if quote else None,
                        )
                    )
                return f"[{source_id}]"
            # Flag citation hallucinated by LLM
            return f"[unverified: {source_id}]"

        verified_answer = re.sub(r"\[(\d+)\]", replace_citation, raw_answer)
        return verified_answer, citations

    def calculate_confidence(
        self,
        chunks: list[RetrievedChunk],
        citations: list[Citation],
        has_unverified: bool,
        answer: str,
    ) -> float:
        """
        Calculate aggregate answer confidence (0.0 to 1.0).
        Penalizes answers missing citations or containing unverified citations.
        """
        if not chunks or not answer.strip():
            return 0.0

        if "do not contain sufficient information" in answer.lower():
            return 0.0

        # Base retrieval score: average similarity of top retrieved chunks
        avg_retrieval_similarity = sum(c.similarity_score for c in chunks) / len(chunks)

        score = avg_retrieval_similarity

        # Reward verified citation density or penalize missing citations
        score = min(1.0, score + 0.1) if citations else max(0.1, score - 0.25)

        # Heavily penalize hallucinated citations
        if has_unverified:
            score = max(0.0, score - 0.35)

        return round(min(1.0, max(0.0, score)), 4)

    async def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> RAGResponse:
        """Generate grounded answer, verify citations, and assemble full RAGResponse."""
        start_time = time.perf_counter()

        if not chunks:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            unanswerable_msg = (
                "The provided documents do not contain sufficient information "
                "to answer this question."
            )
            return RAGResponse(
                answer=unanswerable_msg,
                citations=[],
                sources=[],
                confidence_score=0.0,
                latency_ms=round(latency_ms, 2),
            )

        if self.llm_provider is None:
            raise RuntimeError("LLMProvider must be supplied to GroundedGenerator.generate")

        context_xml = self.build_context_xml(chunks)
        prompt = (
            f"Context:\n{context_xml}\n\n"
            f"User Question: {query}\n\n"
            f"Provide a concise, factual answer referencing source IDs in brackets (e.g. [1]):"
        )

        raw_answer = await self.llm_provider.generate(
            prompt=prompt,
            system_prompt=GROUNDED_SYSTEM_PROMPT,
            temperature=0.0,
        )

        verified_answer, citations = self.verify_and_extract_citations(raw_answer, chunks)
        has_unverified = "[unverified:" in verified_answer
        confidence = self.calculate_confidence(chunks, citations, has_unverified, verified_answer)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return RAGResponse(
            answer=verified_answer,
            citations=citations,
            sources=chunks,
            confidence_score=confidence,
            latency_ms=round(latency_ms, 2),
        )

    def generate_stream(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> AsyncGenerator[str, None]:
        """Stream generated tokens directly from the LLM provider."""
        if not chunks:

            async def _empty_stream() -> AsyncGenerator[str, None]:
                yield (
                    "The provided documents do not contain sufficient information "
                    "to answer this question."
                )

            return _empty_stream()

        if self.llm_provider is None:
            raise RuntimeError("LLMProvider must be supplied to GroundedGenerator.generate_stream")

        context_xml = self.build_context_xml(chunks)
        prompt = (
            f"Context:\n{context_xml}\n\n"
            f"User Question: {query}\n\n"
            f"Provide a concise, factual answer referencing source IDs in brackets (e.g. [1]):"
        )

        return self.llm_provider.generate_stream(
            prompt=prompt,
            system_prompt=GROUNDED_SYSTEM_PROMPT,
            temperature=0.0,
        )

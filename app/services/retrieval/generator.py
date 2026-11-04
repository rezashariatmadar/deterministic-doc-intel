"""Grounded generation engine with strict citation enforcement and hallucination detection."""

import re
import time
from collections.abc import AsyncGenerator

from app.schemas.query import Citation, RAGResponse, RetrievedChunk
from app.services.providers.base import LLMProvider
from app.services.retrieval.verifier import QuoteVerifier

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

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        quote_verifier: QuoteVerifier | None = None,
    ) -> None:
        self.llm_provider = llm_provider
        self.quote_verifier = (
            quote_verifier if quote_verifier is not None else QuoteVerifier()
        )

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
        Verify all bracketed citations [N] against retrieved chunks with QuoteVerifier.
        Valid citations extract the exact slice from chunk text as Citation.quote.
        Citations pointing to wrong chunks are automatically re-homed.
        Hallucinated citations or ungrounded claims are rewritten as [unverified: N].
        """
        matches = list(re.finditer(r"\[(\d+)\]", raw_answer))
        if not matches:
            return raw_answer, []

        chunk_map = {c.source_id: c for c in chunks}
        seen_source_ids: set[int] = set()
        citations: list[Citation] = []
        replacements: list[str] = []

        for i, m in enumerate(matches):
            orig_source_id = int(m.group(1))
            prev_end = matches[i - 1].end() if i > 0 else 0
            next_start = (
                matches[i + 1].start() if i + 1 < len(matches) else len(raw_answer)
            )

            # Extract preceding text back to sentence/paragraph boundary
            preceding = raw_answer[prev_end : m.start()]
            claim_text = ""
            if preceding.strip(" \t\r\n,;:-"):
                boundaries = [0]
                for bm in re.finditer(r"[\r\n]+|(?<!\d)(?<=[.!?])\s+", preceding):
                    boundaries.append(bm.end())
                boundaries.append(len(preceding))
                sents = [
                    preceding[boundaries[k] : boundaries[k + 1]].strip()
                    for k in range(len(boundaries) - 1)
                ]
                sents = [s for s in sents if s]
                claim_text = sents[-1] if sents else preceding.strip()

            claim_text = claim_text.strip(" \t\r\n,;:-")

            # If claim_text is purely an introductory phrase, inspect text following citation
            intro_phrases = {
                "according to",
                "as reported in",
                "as stated in",
                "as noted in",
                "as reported by",
                "as described in",
                "per",
                "based on",
            }
            if claim_text.lower() in intro_phrases:
                following = raw_answer[m.end() : next_start]
                boundaries = [0]
                for bm in re.finditer(r"[\r\n]+|(?<!\d)(?<=[.!?])\s+", following):
                    boundaries.append(bm.end())
                boundaries.append(len(following))
                sents = [
                    following[boundaries[k] : boundaries[k + 1]].strip()
                    for k in range(len(boundaries) - 1)
                ]
                sents = [s for s in sents if s]
                claim_text = sents[0] if sents else following.strip()

            claim_text = claim_text.strip(" \t\r\n,;:-")

            # Fallback when claim_text is empty (e.g. clustered [1][2], [1], [2], or leading [1])
            if not claim_text:
                if i > 0:
                    text_before = raw_answer[: m.start()]
                    # Strip trailing citations and any separating punctuation or whitespace
                    stripped_before = re.sub(r"(\[\d+\][\s,;:-]*)+$", "", text_before)
                    boundaries = [0]
                    for bm in re.finditer(
                        r"[\r\n]+|(?<!\d)(?<=[.!?])\s+", stripped_before
                    ):
                        boundaries.append(bm.end())
                    boundaries.append(len(stripped_before))
                    sents = [
                        stripped_before[boundaries[k] : boundaries[k + 1]].strip()
                        for k in range(len(boundaries) - 1)
                    ]
                    sents = [s for s in sents if s]
                    if sents:
                        claim_text = sents[-1]
                # Fallback: citation at start of sentence [1] ... -> check text following citation
                if not claim_text:
                    following = raw_answer[m.end() : next_start]
                    boundaries = [0]
                    for bm in re.finditer(r"[\r\n]+|(?<!\d)(?<=[.!?])\s+", following):
                        boundaries.append(bm.end())
                    boundaries.append(len(following))
                    sents = [
                        following[boundaries[k] : boundaries[k + 1]].strip()
                        for k in range(len(boundaries) - 1)
                    ]
                    sents = [s for s in sents if s]
                    if sents:
                        claim_text = sents[0]

            claim_text = claim_text.strip(" \t\r\n,;:-")

            # Check for paired inner quotes (double or curly quotes; avoid contractions)
            quote_match = (
                re.search(r'"([^"]{3,})"', claim_text)
                or re.search(r'“([^”]{3,})”', claim_text)
                or re.search(r'‘([^’]{3,})’', claim_text)
            )
            inner_quote = quote_match.group(1).strip() if quote_match else None

            if orig_source_id in chunk_map:
                primary_chunk = chunk_map[orig_source_id]
                res = None
                if inner_quote:
                    res = self.quote_verifier.verify_citation(
                        inner_quote, primary_chunk, chunks
                    )
                if not res or not res.is_verified:
                    res = self.quote_verifier.verify_citation(
                        claim_text, primary_chunk, chunks
                    )
            else:
                res = None

            if res is not None and res.is_verified:
                effective_id = (
                    res.rehomed_source_id
                    if (
                        res.rehomed_source_id is not None
                        and res.rehomed_source_id != orig_source_id
                    )
                    else orig_source_id
                )
                if effective_id in chunk_map:
                    chunk = chunk_map[effective_id]
                    if effective_id not in seen_source_ids:
                        seen_source_ids.add(effective_id)
                        citations.append(
                            Citation(
                                source_id=effective_id,
                                document_id=chunk.document_id,
                                document_title=chunk.document_title,
                                page_start=chunk.page_start,
                                page_end=chunk.page_end,
                                section_path=chunk.section_path,
                                supporting_quote=res.verified_quote,
                                quote=res.verified_quote,
                            )
                        )
                    replacements.append(f"[{effective_id}]")
                else:
                    replacements.append(f"[unverified: {orig_source_id}]")
            else:
                replacements.append(f"[unverified: {orig_source_id}]")

        parts: list[str] = []
        last_idx = 0
        for m, rep in zip(matches, replacements, strict=True):
            parts.append(raw_answer[last_idx : m.start()])
            parts.append(rep)
            last_idx = m.end()
        parts.append(raw_answer[last_idx:])
        verified_answer = "".join(parts)
        # Collapse duplicate adjacent brackets after rehoming
        verified_answer = re.sub(r"\[(\d+)\](?:\s*\[\1\])+", r"[\1]", verified_answer)
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
                is_answerable=False,
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

        return self.finalize_stream_response(raw_answer, chunks, start_time)

    def finalize_stream_response(
        self,
        raw_answer: str,
        chunks: list[RetrievedChunk],
        start_time: float,
    ) -> RAGResponse:
        """Process raw model output through verification and build final RAGResponse."""
        verified_answer, citations = self.verify_and_extract_citations(raw_answer, chunks)
        has_unverified = "[unverified:" in verified_answer
        confidence = self.calculate_confidence(chunks, citations, has_unverified, verified_answer)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        is_answerable = "do not contain sufficient information" not in verified_answer.lower()

        # Guard: If answer claimed factual citations but every citation failed verification,
        # refuse to pass unverified assertions as factually answerable.
        has_cited_brackets = bool(re.search(r"\[\d+\]", raw_answer))
        if is_answerable and has_cited_brackets and len(citations) == 0:
            is_answerable = False
            confidence = 0.0

        return RAGResponse(
            answer=verified_answer,
            is_answerable=is_answerable,
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

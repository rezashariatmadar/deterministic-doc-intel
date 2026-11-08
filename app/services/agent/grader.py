"""Corrective RAG (CRAG) Retrieval Grader, Adaptive Reformulator, and Candidate Fact Verifier."""

import logging
import re
from typing import ClassVar

from app.schemas.query import (
    CandidateFact,
    ExtractionStatus,
    RetrievedChunk,
)
from app.services.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class RetrievalGrader:
    """Evaluates retrieved chunks for relevance to the user question (CRAG Step)."""

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        min_relevance_threshold: float = 0.35,
    ) -> None:
        self.llm_provider = llm_provider
        self.min_relevance_threshold = min_relevance_threshold

    async def grade_chunk(self, query: str, chunk: RetrievedChunk) -> bool:
        """Grade a single chunk as relevant (True) or irrelevant (False)."""
        # Always accept synthetic chunks from SQL or external web
        if chunk.metadata.get("source_type") in ("sql_table", "external_web"):
            return True

        # High similarity chunks are immediately accepted
        if chunk.similarity_score >= 0.70:
            return True

        # Extremely low similarity chunks are filtered
        if chunk.similarity_score < self.min_relevance_threshold:
            return False

        # If LLM available, perform semantic binary grading
        if self.llm_provider is not None:
            prompt = (
                "You are an objective document relevance evaluator.\n"
                f'User Question: "{query}"\n'
                f'Retrieved Document Text:\n"""{chunk.text[:500]}"""\n\n'
                "Determine if the document contains relevant information to answer the question.\n"
                "Respond ONLY with 'YES' or 'NO'."
            )
            try:
                verdict = await self.llm_provider.generate(
                    prompt=prompt,
                    system_prompt="You evaluate retrieval document relevance strictly.",
                    temperature=0.0,
                )
                clean_verdict = verdict.strip().upper()
                return "YES" in clean_verdict
            except Exception as exc:
                logger.warning("LLM chunk grading failed, falling back to heuristic: %s", exc)

        # Heuristic term overlap check
        query_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
        chunk_words = set(re.findall(r"\b\w{3,}\b", chunk.text.lower()))
        overlap = query_words.intersection(chunk_words)
        return len(overlap) >= 1

    async def grade_chunks(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
        """Filter a list of retrieved chunks into relevant and irrelevant partitions."""
        relevant: list[RetrievedChunk] = []
        irrelevant: list[RetrievedChunk] = []

        for chunk in chunks:
            is_rel = await self.grade_chunk(query, chunk)
            if is_rel:
                relevant.append(chunk)
            else:
                irrelevant.append(chunk)

        logger.info(
            "CRAG Graded %d chunks: %d relevant, %d irrelevant",
            len(chunks),
            len(relevant),
            len(irrelevant),
        )
        return relevant, irrelevant


class AdaptiveQueryReformulator:
    """Reformulates queries on retrieval failure or low relevance."""

    ACRONYMS: ClassVar[dict[str, str]] = {
        "AAPL": "Apple Inc.",
        "NVDA": "NVIDIA Corporation",
        "EPS": "earnings per share",
        "FY": "fiscal year",
        "OPEX": "operating expenses",
        "CAPEX": "capital expenditures",
        "COGS": "cost of goods sold",
        "SEC": "Securities and Exchange Commission",
    }

    def __init__(self, llm_provider: LLMProvider | None = None) -> None:
        self.llm_provider = llm_provider

    async def reformulate(
        self,
        query: str,
        retry_count: int,
        failed_chunks: list[RetrievedChunk] | None = None,
    ) -> str:
        """Generate an improved search query using LLM rewriting or heuristic expansion."""
        if self.llm_provider is not None:
            prompt = (
                "You are an expert search query reformulator for corporate SEC 10-K filings.\n"
                f'Original Query: "{query}"\n'
                f"Attempt Number: {retry_count + 1}\n\n"
                "The previous retrieval attempt failed to find relevant documents.\n"
                "Reformulate the query into a clearer, more standard financial query.\n"
                "Expand acronyms, introduce synonyms, remove conversational phrasing.\n"
                "Return ONLY the rewritten query text."
            )
            try:
                reformulated = await self.llm_provider.generate(
                    prompt=prompt,
                    system_prompt="You reformulate queries for search. Return only query text.",
                    temperature=0.0,
                )
                cleaned = reformulated.strip().strip('"').strip("'")
                if cleaned and len(cleaned) >= 5:
                    logger.info("Reformulated query '%s' -> '%s'", query, cleaned)
                    return cleaned
            except Exception as exc:
                logger.warning("LLM query reformulation failed: %s", exc)

        # Deterministic heuristic expansion
        expanded = query
        for acronym, full_term in self.ACRONYMS.items():
            pattern = re.compile(rf"\b{acronym}\b", re.IGNORECASE)
            expanded = pattern.sub(f"{acronym} ({full_term})", expanded)

        # On retry 2, simplify query to noun keywords
        if retry_count >= 1:
            keywords = re.findall(r"\b[A-Za-z0-9_]{3,}\b", expanded)
            return " ".join(keywords)

        return expanded


class FactExtractorAndVerifier:
    """Extracts atomic candidate facts from answers and discriminates verification status."""

    def __init__(self, llm_provider: LLMProvider | None = None) -> None:
        self.llm_provider = llm_provider

    async def extract_and_verify(
        self,
        query: str,
        answer: str,
        chunks: list[RetrievedChunk],
        is_answerable: bool,
    ) -> list[CandidateFact]:
        """Extract atomic claims and assign verification status."""
        facts: list[CandidateFact] = []

        if not is_answerable or "cannot find" in answer.lower() or "insufficient" in answer.lower():
            return [
                CandidateFact(
                    fact="The requested information was not found in corporate documents.",
                    status=ExtractionStatus.NOT_FOUND,
                    confidence=0.95,
                    notes="Corpus lacks sufficient grounding evidence.",
                )
            ]

        # Split answer into sentences / atomic clauses
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", answer)
            if s.strip() and len(s.strip()) > 15
        ]

        chunk_texts = [c.text.lower() for c in chunks]

        for s in sentences:
            # Check for failed verification markers
            if "[unverified:" in s.lower():
                cleaned_fact = re.sub(r"\[(?:unverified:\s*\d+|\d+)\]", "", s).strip()
                facts.append(
                    CandidateFact(
                        fact=cleaned_fact,
                        status=ExtractionStatus.EXTRACTION_FAILED,
                        confidence=0.0,
                        notes="Citation verification ladder failed to find ground truth evidence.",
                    )
                )
                continue

            # Check for inline citation markers [1], [2]
            citation_refs = re.findall(r"\[(\d+)\]", s)
            source_ids = [int(num) for num in citation_refs]

            # Clean citation brackets from fact statement
            cleaned_fact = re.sub(r"\[\d+\]", "", s).strip()

            # Check for conflicting evidence mentions
            if any(w in s.lower() for w in ("conflict", "contradict", "discrepancy")):
                facts.append(
                    CandidateFact(
                        fact=cleaned_fact,
                        status=ExtractionStatus.CONFLICTING,
                        source_ids=source_ids,
                        confidence=0.50,
                        notes="Statement indicates conflicting or contradictory information.",
                    )
                )
                continue

            # Verify presence of key terms in chunks
            key_terms = re.findall(r"\b[A-Za-z0-9$,.%]{3,}\b", cleaned_fact.lower())
            supported_count = 0
            matching_evidence: str | None = None

            for c_idx, c_text in enumerate(chunk_texts):
                matches = sum(1 for term in key_terms if term in c_text)
                if matches >= min(3, len(key_terms)):
                    supported_count += 1
                    if matching_evidence is None and chunks:
                        matching_evidence = chunks[c_idx].text[:200]

            if source_ids or supported_count > 0:
                facts.append(
                    CandidateFact(
                        fact=cleaned_fact,
                        status=ExtractionStatus.VERIFIED,
                        evidence=matching_evidence,
                        source_ids=source_ids,
                        confidence=0.95,
                    )
                )
            elif "might" in s.lower() or "possibly" in s.lower() or "unclear" in s.lower():
                facts.append(
                    CandidateFact(
                        fact=cleaned_fact,
                        status=ExtractionStatus.AMBIGUOUS,
                        confidence=0.60,
                        notes="Statement expresses hedging or ambiguity.",
                    )
                )
            else:
                facts.append(
                    CandidateFact(
                        fact=cleaned_fact,
                        status=ExtractionStatus.UNSUPPORTED,
                        confidence=0.50,
                        notes="No direct grounding evidence in retrieved chunks.",
                    )
                )

        return facts

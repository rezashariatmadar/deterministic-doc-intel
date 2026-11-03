"""Unit tests for GroundedGenerator, context formatting, citation verification, and confidence."""

import uuid
from collections.abc import AsyncGenerator

import pytest

from app.schemas.query import RetrievedChunk
from app.services.providers.base import LLMProvider
from app.services.retrieval.generator import GroundedGenerator


class MockLLMProvider(LLMProvider):
    """Simple mock LLM provider returning predetermined response."""

    def __init__(self, response_text: str) -> None:
        super().__init__(model_name="mock-llm")
        self.response_text = response_text

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        return self.response_text

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> AsyncGenerator[str, None]:
        yield self.response_text


@pytest.fixture
def sample_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            source_id=1,
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            document_title="NVIDIA 10-K Fiscal 2024",
            page_start=45,
            page_end=45,
            section_path="Item 7 > Financial Results",
            element_type="prose",
            text="Data Center revenue rose 217% to $47.5B in fiscal 2024.",
            contextualized_text="[Doc: NVIDIA 10-K] Data Center revenue rose 217%.",
            similarity_score=0.88,
        ),
        RetrievedChunk(
            source_id=2,
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            document_title="NVIDIA 10-K Fiscal 2024",
            page_start=46,
            page_end=46,
            section_path="Item 7 > Gaming Segment",
            element_type="prose",
            text="Gaming revenue was $10.4B, an increase of 15% year-over-year.",
            contextualized_text="[Doc: NVIDIA 10-K] Gaming revenue was $10.4B.",
            similarity_score=0.76,
        ),
    ]


def test_build_context_xml(sample_chunks: list[RetrievedChunk]) -> None:
    generator = GroundedGenerator(llm_provider=None)
    xml_context = generator.build_context_xml(sample_chunks)

    assert "<sources>" in xml_context
    assert "</source>" in xml_context
    assert 'source id="1"' in xml_context
    assert 'source id="2"' in xml_context
    assert "Data Center revenue rose 217%" in xml_context


def test_citation_verification_detects_hallucinations(
    sample_chunks: list[RetrievedChunk],
) -> None:
    generator = GroundedGenerator(llm_provider=None)
    # LLM hallucinates source [99] which is not in sample_chunks
    raw_answer = "NVIDIA Data Center revenue grew to $47.5B [1], and future outlook is $100B [99]."

    verified_answer, citations = generator.verify_and_extract_citations(raw_answer, sample_chunks)

    assert "[1]" in verified_answer
    assert "[unverified: 99]" in verified_answer
    assert len(citations) == 1
    assert citations[0].source_id == 1
    assert citations[0].document_title == "NVIDIA 10-K Fiscal 2024"
    assert citations[0].page_start == 45


@pytest.mark.asyncio
async def test_grounded_generation_returns_rag_response(
    sample_chunks: list[RetrievedChunk],
) -> None:
    llm = MockLLMProvider(
        response_text="Data Center revenue rose 217% to $47.5B in fiscal 2024 [1]."
    )
    generator = GroundedGenerator(llm_provider=llm)

    response = await generator.generate(
        query="How much did Data Center revenue grow?",
        chunks=sample_chunks,
    )

    assert response.answer.startswith("Data Center revenue rose")
    assert len(response.citations) == 1
    assert response.citations[0].source_id == 1
    assert response.citations[0].quote == "Data Center revenue rose 217% to $47.5B in fiscal 2024"
    assert response.confidence_score > 0.8
    assert response.latency_ms >= 0.0
    assert len(response.sources) == 2


@pytest.mark.asyncio
async def test_grounded_generation_empty_chunks_returns_unanswerable() -> None:
    llm = MockLLMProvider(response_text="Should not be called")
    generator = GroundedGenerator(llm_provider=llm)

    response = await generator.generate(
        query="What is quantum teleportation?",
        chunks=[],
    )

    assert "do not contain sufficient information" in response.answer
    assert len(response.citations) == 0
    assert response.confidence_score == 0.0


def test_citation_verification_extracts_exact_haystack_slice(
    sample_chunks: list[RetrievedChunk],
) -> None:
    generator = GroundedGenerator(llm_provider=None)
    raw_answer = "Data Center revenue rose 217% to $47.5B in fiscal 2024 [1]."

    verified_answer, citations = generator.verify_and_extract_citations(raw_answer, sample_chunks)

    assert "[1]" in verified_answer
    assert len(citations) == 1
    assert citations[0].source_id == 1
    assert citations[0].quote == "Data Center revenue rose 217% to $47.5B in fiscal 2024"


def test_citation_verification_rehomes_wrong_chunk(
    sample_chunks: list[RetrievedChunk],
) -> None:
    generator = GroundedGenerator(llm_provider=None)
    # LLM incorrectly cited [1] for Gaming revenue which is actually in chunk [2]
    raw_answer = "Gaming revenue was $10.4B, an increase of 15% year-over-year [1]."

    verified_answer, citations = generator.verify_and_extract_citations(raw_answer, sample_chunks)

    assert "[2]" in verified_answer
    assert "[1]" not in verified_answer
    assert len(citations) == 1
    assert citations[0].source_id == 2
    assert citations[0].section_path == "Item 7 > Gaming Segment"
    assert citations[0].page_start == 46
    assert citations[0].quote == "Gaming revenue was $10.4B, an increase of 15% year-over-year"


def test_citation_verification_rewrites_hallucination_to_unverified(
    sample_chunks: list[RetrievedChunk],
) -> None:
    generator = GroundedGenerator(llm_provider=None)
    # Source [1] exists, but the claim is completely fabricated
    raw_answer = "Quantum computing revenue reached $50B in 2024 [1]."

    verified_answer, citations = generator.verify_and_extract_citations(raw_answer, sample_chunks)

    assert "[unverified: 1]" in verified_answer
    assert "[1]" not in verified_answer
    assert len(citations) == 0


def test_generator_accepts_optional_quote_verifier(
    sample_chunks: list[RetrievedChunk],
) -> None:
    from app.services.retrieval.verifier import QuoteVerifier

    custom_verifier = QuoteVerifier()
    generator = GroundedGenerator(llm_provider=None, quote_verifier=custom_verifier)
    assert generator.quote_verifier is custom_verifier


def test_citation_verification_clustered_and_punctuation_separated_citations() -> None:
    c1 = RetrievedChunk(
        source_id=1,
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_title="NVIDIA 10-K",
        page_start=45,
        page_end=45,
        section_path="Item 7 > Financial Results",
        element_type="prose",
        text="Total revenue increased significantly across all core segments in fiscal 2024.",
        contextualized_text="[Doc: NVIDIA 10-K] Total revenue increased.",
        similarity_score=0.92,
    )
    c2 = RetrievedChunk(
        source_id=2,
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_title="NVIDIA 10-K",
        page_start=46,
        page_end=46,
        section_path="Item 7 > Segment Analysis",
        element_type="prose",
        text="Total revenue increased significantly across all core segments in fiscal 2024.",
        contextualized_text="[Doc: NVIDIA 10-K] Total revenue increased.",
        similarity_score=0.89,
    )
    chunks = [c1, c2]
    generator = GroundedGenerator(llm_provider=None)

    # 1. Immediate clustered citations [1][2]
    raw_1 = "Total revenue increased significantly across all core segments in fiscal 2024 [1][2]."
    ans_1, cits_1 = generator.verify_and_extract_citations(raw_1, chunks)
    assert "[1][2]" in ans_1
    assert len(cits_1) == 2
    assert {c.source_id for c in cits_1} == {1, 2}
    assert all(
        c.quote == "Total revenue increased significantly across all core segments in fiscal 2024"
        for c in cits_1
    )

    # 2. Punctuation-separated clustered citations [1], [2]
    raw_2 = (
        "Total revenue increased significantly across all core segments in fiscal 2024 [1], [2]."
    )
    ans_2, cits_2 = generator.verify_and_extract_citations(raw_2, chunks)
    assert "[1], [2]" in ans_2
    assert len(cits_2) == 2
    assert {c.source_id for c in cits_2} == {1, 2}
    assert all(
        c.quote == "Total revenue increased significantly across all core segments in fiscal 2024"
        for c in cits_2
    )

    # 3. Semicolon-separated clustered citations [1]; [2]
    raw_3 = (
        "Total revenue increased significantly across all core segments in fiscal 2024 [1]; [2]."
    )
    ans_3, cits_3 = generator.verify_and_extract_citations(raw_3, chunks)
    assert "[1]; [2]" in ans_3
    assert len(cits_3) == 2
    assert {c.source_id for c in cits_3} == {1, 2}



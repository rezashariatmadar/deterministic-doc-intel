"""Unit tests for RAGJudge hybrid LLM-as-a-judge evaluation suite."""

import uuid
from collections.abc import AsyncGenerator

import pytest

from app.schemas.query import Citation, RAGResponse, RetrievedChunk
from app.services.providers.base import LLMProvider
from evals.judge import JudgeResult, RAGJudge
from evals.schemas import GoldenQuery


class MockLLMProvider(LLMProvider):
    """Mock LLMProvider for testing judge responses and error handling."""

    def __init__(self, output: str) -> None:
        super().__init__(model_name="mock-gemini-judge")
        self.output = output
        self.calls: list[dict[str, str]] = []

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt or "",
                "temperature": str(temperature),
            }
        )
        return self.output

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> AsyncGenerator[str, None]:
        yield self.output


@pytest.fixture
def sample_answerable_query() -> GoldenQuery:
    return GoldenQuery(
        id="q01",
        category="exact_keyword",
        query="What was Apple's total commercial paper liabilities as of September 30, 2023?",
        expected_documents=["Apple_10-K-Q4-2023-As-Filed.pdf"],
        expected_pages=[33],
        required_substrings=["commercial paper", "5,985"],
        ground_truth_answer=(
            "As of September 30, 2023, Apple recorded $5,985 million in commercial "
            "paper liabilities."
        ),
        is_answerable=True,
    )


@pytest.fixture
def sample_unanswerable_query() -> GoldenQuery:
    return GoldenQuery(
        id="q25",
        category="unanswerable_adversarial",
        query="What was Tesla's total automotive gross margin in Q3 2023?",
        expected_documents=[],
        expected_pages=[],
        required_substrings=[],
        ground_truth_answer=(
            "The provided documents do not contain information regarding Tesla's Cybertruck "
            "deliveries or automotive gross margin."
        ),
        is_answerable=False,
    )


@pytest.fixture
def sample_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            source_id=1,
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            document_title="Apple_10-K-Q4-2023-As-Filed.pdf",
            page_start=33,
            page_end=33,
            section_path="Item 8 > Commercial Paper",
            element_type="table",
            text=(
                "Commercial paper contractual maturities had liabilities of "
                "$5,985 million as of September 30, 2023."
            ),
            contextualized_text=(
                "Commercial paper contractual maturities had liabilities of $5,985 million."
            ),
            similarity_score=0.92,
        )
    ]


def test_judge_result_schema() -> None:
    result = JudgeResult(
        faithfulness=0.95,
        answer_relevance=1.0,
        ground_truth_correctness=0.9,
        citation_precision=1.0,
        abstention_accuracy=1.0,
        reasoning="Well grounded answer.",
    )
    assert result.faithfulness == 0.95
    assert result.citation_precision == 1.0
    assert result.abstention_accuracy == 1.0
    assert "Well grounded" in result.reasoning


@pytest.mark.asyncio
async def test_deterministic_abstention_accuracy_on_unanswerable(
    sample_unanswerable_query: GoldenQuery,
) -> None:
    judge = RAGJudge(llm_provider=None)

    # 1. Correct abstention: is_answerable=False
    refusal_resp = RAGResponse(
        answer=(
            "The provided documents do not contain sufficient information to answer this question."
        ),
        is_answerable=False,
        citations=[],
        sources=[],
        confidence_score=0.0,
        latency_ms=10.0,
    )
    res1 = await judge.evaluate_response(
        sample_unanswerable_query, refusal_resp, mode="deterministic"
    )
    assert res1.abstention_accuracy == 1.0
    assert res1.faithfulness == 1.0
    assert res1.citation_precision == 1.0

    # 2. Failed abstention: answerable=True with hallucinated claim
    hallucinated_resp = RAGResponse(
        answer="Tesla automotive gross margin was 17.9% in Q3 2023 [1].",
        is_answerable=True,
        citations=[],
        sources=[],
        confidence_score=0.8,
        latency_ms=12.0,
    )
    res2 = await judge.evaluate_response(
        sample_unanswerable_query, hallucinated_resp, mode="deterministic"
    )
    assert res2.abstention_accuracy == 0.0
    assert res2.faithfulness < 0.5


@pytest.mark.asyncio
async def test_deterministic_abstention_accuracy_on_answerable(
    sample_answerable_query: GoldenQuery,
    sample_chunks: list[RetrievedChunk],
) -> None:
    judge = RAGJudge(llm_provider=None)

    # 1. Answered correctly: is_answerable=True
    answered_resp = RAGResponse(
        answer="Apple recorded $5,985 million in commercial paper liabilities [1].",
        is_answerable=True,
        citations=[
            Citation(
                source_id=1,
                document_title="Apple_10-K-Q4-2023-As-Filed.pdf",
                page_start=33,
                page_end=33,
                section_path="Item 8",
            )
        ],
        sources=sample_chunks,
        confidence_score=0.9,
        latency_ms=15.0,
    )
    res1 = await judge.evaluate_response(
        sample_answerable_query, answered_resp, mode="deterministic"
    )
    assert res1.abstention_accuracy == 1.0

    # 2. Falsely abstained on answerable query
    false_refusal_resp = RAGResponse(
        answer="I cannot answer this question from the context.",
        is_answerable=False,
        citations=[],
        sources=sample_chunks,
        confidence_score=0.0,
        latency_ms=10.0,
    )
    res2 = await judge.evaluate_response(
        sample_answerable_query, false_refusal_resp, mode="deterministic"
    )
    assert res2.abstention_accuracy == 0.0


@pytest.mark.asyncio
async def test_deterministic_citation_precision(
    sample_answerable_query: GoldenQuery,
    sample_chunks: list[RetrievedChunk],
) -> None:
    judge = RAGJudge(llm_provider=None)

    # 1. 100% precision: 1 verified, 0 unverified
    resp_perfect = RAGResponse(
        answer="Apple had $5,985 million liabilities [1].",
        is_answerable=True,
        citations=[
            Citation(
                source_id=1,
                document_title="Apple_10-K-Q4-2023-As-Filed.pdf",
                page_start=33,
                page_end=33,
                section_path="Item 8",
            )
        ],
        sources=sample_chunks,
        confidence_score=0.95,
        latency_ms=10.0,
    )
    res_perfect = await judge.evaluate_response(
        sample_answerable_query, resp_perfect, mode="deterministic"
    )
    assert res_perfect.citation_precision == 1.0

    # 2. 50% precision: 1 verified [1] and 1 unverified [unverified: 2]
    resp_mixed = RAGResponse(
        answer="Apple had $5,985 million [1] and an unbacked claim [unverified: 2].",
        is_answerable=True,
        citations=[
            Citation(
                source_id=1,
                document_title="Apple_10-K-Q4-2023-As-Filed.pdf",
                page_start=33,
                page_end=33,
                section_path="Item 8",
            )
        ],
        sources=sample_chunks,
        confidence_score=0.6,
        latency_ms=10.0,
    )
    res_mixed = await judge.evaluate_response(
        sample_answerable_query, resp_mixed, mode="deterministic"
    )
    assert res_mixed.citation_precision == 0.5

    # 3. 0% precision: 0 verified, 1 unverified
    resp_zero = RAGResponse(
        answer="Completely hallucinated citation [unverified: 99].",
        is_answerable=True,
        citations=[],
        sources=sample_chunks,
        confidence_score=0.2,
        latency_ms=10.0,
    )
    res_zero = await judge.evaluate_response(
        sample_answerable_query, resp_zero, mode="deterministic"
    )
    assert res_zero.citation_precision == 0.0


@pytest.mark.asyncio
async def test_deterministic_lexical_and_numeric_correctness(
    sample_answerable_query: GoldenQuery,
    sample_chunks: list[RetrievedChunk],
) -> None:
    judge = RAGJudge(llm_provider=None)

    # Contains exact number '5,985' and key phrase 'commercial paper'
    good_resp = RAGResponse(
        answer=(
            "Apple recorded $5,985 million in commercial paper liabilities "
            "as of September 30, 2023 [1]."
        ),
        is_answerable=True,
        citations=[
            Citation(
                source_id=1,
                document_title="Apple_10-K-Q4-2023-As-Filed.pdf",
                page_start=33,
                page_end=33,
                section_path="Item 8",
            )
        ],
        sources=sample_chunks,
        confidence_score=0.95,
        latency_ms=10.0,
    )
    res_good = await judge.evaluate_response(
        sample_answerable_query, good_resp, mode="deterministic"
    )
    assert res_good.ground_truth_correctness >= 0.8
    assert res_good.faithfulness >= 0.8
    assert res_good.answer_relevance >= 0.8

    # Missing exact number and key phrases
    bad_resp = RAGResponse(
        answer="The company has some short-term liabilities in general.",
        is_answerable=True,
        citations=[],
        sources=sample_chunks,
        confidence_score=0.3,
        latency_ms=10.0,
    )
    res_bad = await judge.evaluate_response(sample_answerable_query, bad_resp, mode="deterministic")
    assert res_bad.ground_truth_correctness < 0.5


@pytest.mark.asyncio
async def test_llm_judge_mode_with_valid_json(
    sample_answerable_query: GoldenQuery,
    sample_chunks: list[RetrievedChunk],
) -> None:
    json_response = """
    ```json
    {
      "faithfulness": 0.98,
      "answer_relevance": 0.95,
      "ground_truth_correctness": 0.92,
      "reasoning": "Fully factual response with explicit citation to commercial paper table."
    }
    ```
    """
    mock_llm = MockLLMProvider(output=json_response)
    judge = RAGJudge(llm_provider=mock_llm)

    resp = RAGResponse(
        answer="Apple recorded $5,985 million in commercial paper liabilities [1].",
        is_answerable=True,
        citations=[
            Citation(
                source_id=1,
                document_title="Apple_10-K-Q4-2023-As-Filed.pdf",
                page_start=33,
                page_end=33,
                section_path="Item 8",
            )
        ],
        sources=sample_chunks,
        confidence_score=0.95,
        latency_ms=15.0,
    )

    res = await judge.evaluate_response(sample_answerable_query, resp, mode="llm")
    assert res.faithfulness == 0.98
    assert res.answer_relevance == 0.95
    assert res.ground_truth_correctness == 0.92
    assert res.citation_precision == 1.0
    assert res.abstention_accuracy == 1.0
    assert "commercial paper" in res.reasoning
    assert len(mock_llm.calls) == 1


@pytest.mark.asyncio
async def test_llm_judge_fallback_on_corrupted_json(
    sample_answerable_query: GoldenQuery,
    sample_chunks: list[RetrievedChunk],
) -> None:
    corrupted_response = "I cannot evaluate this in JSON format: error 500 occurred."
    mock_llm = MockLLMProvider(output=corrupted_response)
    judge = RAGJudge(llm_provider=mock_llm)

    resp = RAGResponse(
        answer="Apple recorded $5,985 million in commercial paper liabilities [1].",
        is_answerable=True,
        citations=[
            Citation(
                source_id=1,
                document_title="Apple_10-K-Q4-2023-As-Filed.pdf",
                page_start=33,
                page_end=33,
                section_path="Item 8",
            )
        ],
        sources=sample_chunks,
        confidence_score=0.95,
        latency_ms=15.0,
    )

    res = await judge.evaluate_response(sample_answerable_query, resp, mode="llm")
    # Must fallback gracefully to deterministic metrics rather than crashing
    assert 0.0 <= res.faithfulness <= 1.0
    assert 0.0 <= res.answer_relevance <= 1.0
    assert res.ground_truth_correctness >= 0.6
    assert res.citation_precision == 1.0
    assert res.abstention_accuracy == 1.0
    assert "Fallback" in res.reasoning

"""Unit tests for Information Retrieval (IR) evaluation metrics."""

import math
from typing import Any
from uuid import uuid4

import pytest

from app.schemas.query import RetrievedChunk
from evals.metrics import (
    RetrievalEvaluator,
    calculate_mrr_at_k,
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
    is_relevant_hit,
)
from evals.schemas import GoldenQuery


def make_chunk(
    doc_title: str = "Apple_10-K-Q4-2023-As-Filed.pdf",
    page_start: int = 33,
    page_end: int = 33,
    text: str = "commercial paper liabilities of 5,985 million",
    source_id: int = 1,
) -> RetrievedChunk:
    """Helper to create test RetrievedChunk."""
    return RetrievedChunk(
        source_id=source_id,
        chunk_id=uuid4(),
        document_id=uuid4(),
        document_title=doc_title,
        page_start=page_start,
        page_end=page_end,
        section_path="Item 8",
        element_type="text",
        text=text,
        contextualized_text=text,
        similarity_score=0.9,
    )


@pytest.fixture
def base_query() -> GoldenQuery:
    return GoldenQuery(
        id="q01",
        category="exact_keyword",
        query="What was Apple's total commercial paper liabilities?",
        expected_documents=["Apple_10-K-Q4-2023-As-Filed.pdf", "Apple 10-K"],
        expected_pages=[33],
        required_substrings=["commercial paper", "5,985"],
        ground_truth_answer="$5,985 million",
        is_answerable=True,
    )


@pytest.fixture
def unanswerable_query() -> GoldenQuery:
    return GoldenQuery(
        id="q25",
        category="unanswerable_adversarial",
        query="What were Apple's quantum computing revenues in 2023?",
        expected_documents=[],
        expected_pages=[],
        required_substrings=[],
        ground_truth_answer="Not answered in corpus.",
        is_answerable=False,
    )


class TestHitCriterion:
    def test_relevant_hit_exact(self, base_query: GoldenQuery) -> None:
        chunk = make_chunk(
            doc_title="Apple_10-K-Q4-2023-As-Filed.pdf",
            page_start=33,
            page_end=33,
            text="Total commercial paper was 5,985.",
        )
        assert is_relevant_hit(chunk, base_query) is True

    def test_relevant_hit_case_folding_and_doc_alias(self, base_query: GoldenQuery) -> None:
        chunk = make_chunk(
            doc_title="apple 10-k",
            page_start=30,
            page_end=35,
            text="COMMERCIAL PAPER is listed as 5,985.",
        )
        assert is_relevant_hit(chunk, base_query) is True

    def test_irrelevant_wrong_document(self, base_query: GoldenQuery) -> None:
        chunk = make_chunk(
            doc_title="nvidia_10k_sample.pdf",
            page_start=33,
            page_end=33,
            text="commercial paper liabilities of 5,985 million",
        )
        assert is_relevant_hit(chunk, base_query) is False

    def test_irrelevant_page_disjoint(self, base_query: GoldenQuery) -> None:
        chunk = make_chunk(
            doc_title="Apple_10-K-Q4-2023-As-Filed.pdf",
            page_start=10,
            page_end=12,
            text="commercial paper liabilities of 5,985 million",
        )
        assert is_relevant_hit(chunk, base_query) is False

    def test_irrelevant_missing_substring(self, base_query: GoldenQuery) -> None:
        chunk = make_chunk(
            doc_title="Apple_10-K-Q4-2023-As-Filed.pdf",
            page_start=33,
            page_end=33,
            text="unrelated text about cash and marketable securities",
        )
        assert is_relevant_hit(chunk, base_query) is False

    def test_unanswerable_query_never_hits(self, unanswerable_query: GoldenQuery) -> None:
        chunk = make_chunk()
        assert is_relevant_hit(chunk, unanswerable_query) is False


class TestRecallAtK:
    def test_recall_perfect_hit(self, base_query: GoldenQuery) -> None:
        chunk = make_chunk()
        assert calculate_recall_at_k([chunk], base_query, k=3) == 1.0

    def test_recall_partial_pages_hit(self) -> None:
        query = GoldenQuery(
            id="multi_page",
            category="multi_hop_comparative",
            query="Compare pages 2 and 3",
            expected_documents=["doc.pdf"],
            expected_pages=[2, 3],
            required_substrings=["evidence"],
            ground_truth_answer="Answer",
            is_answerable=True,
        )
        c1 = make_chunk(doc_title="doc.pdf", page_start=2, page_end=2, text="evidence from page 2")
        c_irrelevant = make_chunk(doc_title="other.pdf", page_start=3, page_end=3, text="other")

        # Top 1 chunk covers page 2 (1 of 2 pages)
        recall = calculate_recall_at_k([c1, c_irrelevant], query, k=2)
        assert recall == 0.5

        # Adding chunk covering page 3 completes recall to 1.0
        c2 = make_chunk(doc_title="doc.pdf", page_start=3, page_end=3, text="evidence from page 3")
        assert calculate_recall_at_k([c1, c2], query, k=2) == 1.0

    def test_recall_partial_substrings_hit(self) -> None:
        query = GoldenQuery(
            id="multi_sub",
            category="exact_keyword",
            query="Query requiring two terms",
            expected_documents=["doc.pdf"],
            expected_pages=[1],
            required_substrings=["term_one", "term_two"],
            ground_truth_answer="Answer",
            is_answerable=True,
        )
        c1 = make_chunk(doc_title="doc.pdf", page_start=1, page_end=1, text="includes term_one")
        assert calculate_recall_at_k([c1], query, k=1) == 0.5

        c2 = make_chunk(doc_title="doc.pdf", page_start=1, page_end=1, text="includes term_two")
        assert calculate_recall_at_k([c1, c2], query, k=2) == 1.0

    def test_recall_zero_hits(self, base_query: GoldenQuery) -> None:
        c_bad = make_chunk(doc_title="other.pdf")
        assert calculate_recall_at_k([c_bad], base_query, k=5) == 0.0

    def test_recall_empty_retrieved(self, base_query: GoldenQuery) -> None:
        assert calculate_recall_at_k([], base_query, k=5) == 0.0

    def test_recall_unanswerable_query(self, unanswerable_query: GoldenQuery) -> None:
        c = make_chunk()
        assert calculate_recall_at_k([c], unanswerable_query, k=5) == 0.0

    def test_recall_k_zero_or_negative(self, base_query: GoldenQuery) -> None:
        c = make_chunk()
        assert calculate_recall_at_k([c], base_query, k=0) == 0.0
        assert calculate_recall_at_k([c], base_query, k=-1) == 0.0


class TestPrecisionAtK:
    def test_precision_perfect_hits(self, base_query: GoldenQuery) -> None:
        c1 = make_chunk(source_id=1)
        c2 = make_chunk(source_id=2)
        assert calculate_precision_at_k([c1, c2], base_query, k=2) == 1.0

    def test_precision_partial_hits(self, base_query: GoldenQuery) -> None:
        c1 = make_chunk(source_id=1)
        c2 = make_chunk(doc_title="irrelevant.pdf", source_id=2)
        # 1 relevant hit out of k=2 => 0.5
        assert calculate_precision_at_k([c1, c2], base_query, k=2) == 0.5
        # 1 relevant hit out of k=5 => 0.2
        assert calculate_precision_at_k([c1, c2], base_query, k=5) == 0.2

    def test_precision_zero_hits(self, base_query: GoldenQuery) -> None:
        c1 = make_chunk(doc_title="irrelevant.pdf")
        assert calculate_precision_at_k([c1], base_query, k=5) == 0.0

    def test_precision_empty_or_unanswerable(
        self, base_query: GoldenQuery, unanswerable_query: GoldenQuery
    ) -> None:
        assert calculate_precision_at_k([], base_query, k=5) == 0.0
        assert calculate_precision_at_k([make_chunk()], unanswerable_query, k=5) == 0.0
        assert calculate_precision_at_k([make_chunk()], base_query, k=0) == 0.0


class TestMRRAtK:
    def test_mrr_hit_at_rank_1(self, base_query: GoldenQuery) -> None:
        c1 = make_chunk()
        assert calculate_mrr_at_k([c1], base_query, k=10) == 1.0

    def test_mrr_hit_at_rank_2(self, base_query: GoldenQuery) -> None:
        c_bad = make_chunk(doc_title="irrelevant.pdf")
        c_good = make_chunk()
        assert calculate_mrr_at_k([c_bad, c_good], base_query, k=10) == 0.5

    def test_mrr_hit_at_rank_3(self, base_query: GoldenQuery) -> None:
        c_bad1 = make_chunk(doc_title="irrelevant.pdf")
        c_bad2 = make_chunk(doc_title="irrelevant2.pdf")
        c_good = make_chunk()
        assert calculate_mrr_at_k([c_bad1, c_bad2, c_good], base_query, k=10) == pytest.approx(
            1.0 / 3.0, 1e-4
        )

    def test_mrr_hit_outside_k(self, base_query: GoldenQuery) -> None:
        c_bad = [make_chunk(doc_title="irrelevant.pdf") for _ in range(5)]
        c_good = make_chunk()
        assert calculate_mrr_at_k([*c_bad, c_good], base_query, k=5) == 0.0

    def test_mrr_empty_and_unanswerable(
        self, base_query: GoldenQuery, unanswerable_query: GoldenQuery
    ) -> None:
        assert calculate_mrr_at_k([], base_query, k=10) == 0.0
        assert calculate_mrr_at_k([make_chunk()], unanswerable_query, k=10) == 0.0


class TestNDCGAtK:
    def test_ndcg_rank_1(self, base_query: GoldenQuery) -> None:
        c1 = make_chunk()
        assert calculate_ndcg_at_k([c1], base_query, k=5) == 1.0

    def test_ndcg_degraded_rank(self, base_query: GoldenQuery) -> None:
        c_bad = make_chunk(doc_title="irrelevant.pdf")
        c_good = make_chunk()
        # Rank 2 hit out of 1 relevant item total
        # DCG = 1 / log2(3) = 0.63092975
        # IDCG = 1 / log2(2) = 1.0
        expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
        assert calculate_ndcg_at_k([c_bad, c_good], base_query, k=5) == pytest.approx(
            expected, 1e-4
        )

    def test_ndcg_multiple_hits(self, base_query: GoldenQuery) -> None:
        c_good1 = make_chunk(source_id=1)
        c_bad = make_chunk(doc_title="irrelevant.pdf", source_id=2)
        c_good2 = make_chunk(source_id=3)
        # 2 relevant items retrieved: at rank 1 and rank 3
        # DCG = 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5
        # IDCG with ideal_k = 2 => 1/log2(2) + 1/log2(3) = 1 + 0.63092975 = 1.63092975
        dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        expected = dcg / idcg
        assert calculate_ndcg_at_k([c_good1, c_bad, c_good2], base_query, k=5) == pytest.approx(
            expected, 1e-4
        )

    def test_ndcg_zero_hits_and_empty(
        self, base_query: GoldenQuery, unanswerable_query: GoldenQuery
    ) -> None:
        assert calculate_ndcg_at_k([], base_query, k=5) == 0.0
        assert calculate_ndcg_at_k([make_chunk(doc_title="bad.pdf")], base_query, k=5) == 0.0
        assert calculate_ndcg_at_k([make_chunk()], unanswerable_query, k=5) == 0.0

    def test_ndcg_multi_hop_partial_retrieval_below_one(self) -> None:
        multi_hop_query = GoldenQuery(
            id="q13",
            category="multi_hop_comparative",
            query="Compare NVIDIA revenue with Apple net sales.",
            expected_documents=["nvidia_10k_sample.pdf", "Apple_10-K-Q4-2023-As-Filed.pdf"],
            expected_pages=[2, 31],
            required_substrings=["60,922", "383,285"],
            ground_truth_answer="Revenue was 60,922 vs 383,285.",
            is_answerable=True,
        )
        # Only 1 of 2 targets retrieved at rank 1, second chunk is irrelevant
        c_hit = make_chunk(
            doc_title="nvidia_10k_sample.pdf",
            page_start=2,
            page_end=2,
            text="NVIDIA revenue was 60,922 million.",
            source_id=1,
        )
        c_miss = make_chunk(doc_title="irrelevant.pdf", source_id=2)

        # In a 2-target query, retrieving only 1 target must not give NDCG=1.0!
        # Ideal DCG for 2 items is 1/log2(2) + 1/log2(3) = 1.6309.
        # DCG achieved is 1/log2(2) = 1.0.
        # NDCG should be 1.0 / 1.6309 = 0.6131.
        score = calculate_ndcg_at_k([c_hit, c_miss], multi_hop_query, k=5)
        assert score < 1.0
        assert score == pytest.approx(1.0 / (1.0 + (1.0 / math.log2(3))), 1e-4)


class TestRetrievalEvaluator:
    @pytest.mark.asyncio
    async def test_evaluator_ablation_mock(self, base_query: GoldenQuery) -> None:
        class MockSearchService:
            async def search(self, **kwargs: Any) -> list[RetrievedChunk]:
                return [make_chunk()]

        evaluator = RetrievalEvaluator(search_service=MockSearchService())
        results = await evaluator.run_ablation([base_query], config_name="dense")
        assert "recall_at_3" in results
        assert "recall_at_5" in results
        assert "recall_at_10" in results
        assert "precision_at_5" in results
        assert "mrr_at_10" in results
        assert "ndcg_at_5" in results
        assert results["recall_at_3"] == 1.0
        assert results["mrr_at_10"] == 1.0

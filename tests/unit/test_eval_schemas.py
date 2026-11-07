"""Unit tests for Golden Evaluation Dataset schemas and validation."""

import json
import re
from pathlib import Path

import pymupdf

from evals.schemas import GoldenDataset

DATASET_PATH = Path("evals/datasets/golden_v1.json")
CORPUS_DIR = Path("data/raw_corpus")


def test_golden_dataset_file_exists() -> None:
    """Ensure the golden_v1.json file exists at expected path."""
    assert DATASET_PATH.exists(), f"Golden dataset file not found at {DATASET_PATH}"


def test_golden_dataset_schema_validation() -> None:
    """Validate golden dataset against Pydantic schema."""
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)

    dataset = GoldenDataset.model_validate(data)
    assert 25 <= len(dataset.queries) <= 30, f"Expected 25-30 queries, got {len(dataset.queries)}"

    query_ids = [q.id for q in dataset.queries]
    assert len(query_ids) == len(set(query_ids)), "Query IDs must be unique"


def test_golden_dataset_category_distribution() -> None:
    """Ensure all 5 required query categories are represented with sufficient coverage."""
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)

    dataset = GoldenDataset.model_validate(data)
    categories = {q.category for q in dataset.queries}
    expected_categories = {
        "exact_keyword",
        "single_hop_semantic",
        "multi_hop_comparative",
        "tabular_numerical",
        "unanswerable_adversarial",
    }
    diff = expected_categories - categories
    assert categories == expected_categories, f"Missing categories: {diff}"

    counts: dict[str, int] = {}
    for q in dataset.queries:
        counts[q.category] = counts.get(q.category, 0) + 1

    for cat in expected_categories:
        cnt = counts.get(cat, 0)
        assert cnt >= 4, f"Category {cat} has only {cnt} items (minimum 4 required)"


def test_golden_dataset_coordinate_integrity() -> None:
    """Verify ground-truth coordinates are valid and immutable."""
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)

    dataset = GoldenDataset.model_validate(data)
    for q in dataset.queries:
        assert len(q.query.strip()) > 5, f"Query '{q.id}' text is too short"
        assert len(q.ground_truth_answer.strip()) > 5, f"Ground truth answer '{q.id}' is too short"

        if q.is_answerable:
            assert len(q.expected_documents) >= 1, (
                f"Answerable query '{q.id}' must specify expected_documents"
            )
            assert len(q.expected_pages) >= 1, (
                f"Answerable query '{q.id}' must specify expected_pages"
            )
            assert all(p >= 1 for p in q.expected_pages), (
                f"Query '{q.id}' page numbers must be 1-based (>= 1)"
            )
            assert len(q.required_substrings) >= 1, (
                f"Answerable query '{q.id}' must specify required_substrings"
            )
            for sub in q.required_substrings:
                assert len(sub.strip()) >= 3, f"Query '{q.id}' substring '{sub}' is too short"
        else:
            assert q.category == "unanswerable_adversarial", (
                f"Unanswerable query '{q.id}' must be adversarial"
            )
            assert len(q.expected_pages) == 0, (
                f"Adversarial query '{q.id}' must have empty expected_pages"
            )
            assert len(q.required_substrings) == 0, (
                f"Adversarial query '{q.id}' must have empty required_substrings"
            )


def test_golden_dataset_substrings_exist_in_corpus() -> None:
    """Verify answerable queries substrings exist on expected pages in actual PDFs."""
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)

    dataset = GoldenDataset.model_validate(data)

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.lower())

    for q in dataset.queries:
        if not q.is_answerable:
            continue

        # Find the physical file among expected documents
        candidate_paths = [
            CORPUS_DIR / doc_name
            for doc_name in q.expected_documents
            if (CORPUS_DIR / doc_name).exists() and doc_name.endswith(".pdf")
        ]
        assert candidate_paths, f"Query '{q.id}' has no matching PDF files in {CORPUS_DIR}"

        # Combine text across the candidate files and expected pages
        combined_text = ""
        for p in candidate_paths:
            doc = pymupdf.open(p)  # type: ignore[no-untyped-call]
            for page_num in q.expected_pages:
                if 1 <= page_num <= len(doc):
                    combined_text += " " + doc[page_num - 1].get_text()  # type: ignore[no-untyped-call]

        norm_combined = norm(combined_text)
        for sub in q.required_substrings:
            assert norm(sub) in norm_combined, (
                f"Query '{q.id}' substring '{sub}' not found on expected pages {q.expected_pages}"
            )

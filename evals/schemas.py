"""Evaluation dataset schemas and data structures."""

from collections.abc import Iterator
from typing import Any, Literal

from pydantic import BaseModel, Field


class GoldenQuery(BaseModel):
    """Represents a single curated ground-truth query in the golden evaluation dataset."""

    id: str = Field(description="Unique identifier for the evaluation query (e.g. q01)")
    category: Literal[
        "exact_keyword",
        "single_hop_semantic",
        "multi_hop_comparative",
        "tabular_numerical",
        "unanswerable_adversarial",
    ] = Field(description="Evaluation category archetype")
    query: str = Field(min_length=3, description="Evaluated query string")
    expected_documents: list[str] = Field(
        default_factory=list,
        description="List of expected matching document filenames or canonical titles",
    )
    expected_pages: list[int] = Field(
        default_factory=list,
        description="1-based page numbers where supporting evidence appears",
    )
    required_substrings: list[str] = Field(
        default_factory=list,
        description="Exact case-insensitive substrings present in the source pages",
    )
    ground_truth_answer: str = Field(
        min_length=3,
        description="Reference ground-truth answer for generation evaluation",
    )
    is_answerable: bool = Field(
        default=True,
        description="Whether the query is answerable from the corpus (False for adversarial)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata or tags",
    )


class GoldenDataset(BaseModel):
    """Encapsulates a versioned collection of GoldenQuery instances."""

    version: str = Field(default="v1", description="Dataset version identifier")
    description: str = Field(default="", description="Description of dataset domain and coverage")
    queries: list[GoldenQuery] = Field(
        default_factory=list,
        description="List of golden queries",
    )

    def __iter__(self) -> Iterator[GoldenQuery]:  # type: ignore[override]
        return iter(self.queries)

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, index: int) -> GoldenQuery:
        return self.queries[index]

    def get_by_category(self, category: str) -> list[GoldenQuery]:
        """Filter queries by category archetype."""
        return [q for q in self.queries if q.category == category]

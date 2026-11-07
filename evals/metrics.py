"""Information Retrieval (IR) metrics and ablation evaluation service."""

import logging
import math
from typing import Any

from evals.schemas import GoldenDataset, GoldenQuery

logger = logging.getLogger(__name__)

CONFIG_ALIASES: dict[str, str] = {
    "dense": "dense",
    "dense_only": "dense",
    "dense_flat": "dense",
    "sparse": "sparse",
    "sparse_only": "sparse",
    "sparse_bm25": "sparse",
    "hybrid_flat": "hybrid_flat",
    "hybrid_rrf_flat": "hybrid_flat",
    "hybrid_hierarchical_rerank": "hybrid_hierarchical_rerank",
    "hybrid_hierarchical": "hybrid_hierarchical_rerank",
    "full_pipeline": "full_pipeline",
    "full": "full_pipeline",
}

ABLATION_CONFIGS: dict[str, dict[str, Any]] = {
    "dense": {
        "search_mode": "dense",
        "resolve_parents": False,
        "enable_reranker": False,
        "enable_query_rewrite": False,
    },
    "sparse": {
        "search_mode": "sparse",
        "resolve_parents": False,
        "enable_reranker": False,
        "enable_query_rewrite": False,
    },
    "hybrid_flat": {
        "search_mode": "hybrid",
        "resolve_parents": False,
        "enable_reranker": False,
        "enable_query_rewrite": False,
    },
    "hybrid_hierarchical_rerank": {
        "search_mode": "hybrid",
        "resolve_parents": True,
        "enable_reranker": True,
        "enable_query_rewrite": False,
    },
    "full_pipeline": {
        "search_mode": "hybrid",
        "resolve_parents": True,
        "enable_reranker": True,
        "enable_query_rewrite": True,
    },
}


def _matches_document(candidate: str, expected: str) -> bool:
    """Check if candidate document name/title matches an expected document entry."""
    c = candidate.strip().lower()
    e = expected.strip().lower()
    if not c or not e:
        return False
    if c == e:
        return True
    c_base = c.rsplit("/", 1)[-1]
    e_base = e.rsplit("/", 1)[-1]
    if c_base == e_base:
        return True
    return c_base.removesuffix(".pdf") == e_base.removesuffix(".pdf")


def _get_chunk_doc_names(chunk: Any) -> list[str]:
    """Extract candidate document names and titles from a retrieved chunk object."""
    names: list[str] = []
    for attr in ("document_title", "filename", "doc_title", "document_name", "title"):
        val = getattr(chunk, attr, None)
        if isinstance(val, str) and val.strip():
            names.append(val.strip())

    meta = getattr(chunk, "metadata", None)
    if isinstance(meta, dict):
        for key in ("filename", "file_name", "document_title", "title", "doc_title"):
            val = meta.get(key)
            if isinstance(val, str) and val.strip():
                names.append(val.strip())

    doc_obj = getattr(chunk, "document", None)
    if doc_obj is not None:
        for attr in ("title", "filename"):
            val = getattr(doc_obj, attr, None)
            if isinstance(val, str) and val.strip():
                names.append(val.strip())

    return names


def _matches_expected_documents(chunk: Any, expected_documents: list[str]) -> bool:
    """Check whether chunk matches any expected document."""
    if not expected_documents:
        return False
    chunk_names = _get_chunk_doc_names(chunk)
    for name in chunk_names:
        for exp in expected_documents:
            if _matches_document(name, exp):
                return True
    return False


def _get_page_range(chunk: Any) -> tuple[int | None, int | None]:
    """Extract 1-based start and end page numbers from chunk."""
    p_start = getattr(chunk, "page_start", None)
    p_end = getattr(chunk, "page_end", None)
    if p_start is None:
        p_start = getattr(chunk, "page", None) or getattr(chunk, "page_number", None)
    if p_end is None:
        p_end = p_start
    return p_start, p_end


def _get_chunk_text(chunk: Any) -> str:
    """Extract raw or contextual text from chunk."""
    text = getattr(chunk, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    raw = getattr(chunk, "raw_text", None)
    if isinstance(raw, str) and raw.strip():
        return raw
    return ""


def is_relevant_hit(chunk: Any, query: GoldenQuery) -> bool:
    """
    Deterministic IR Hit Criterion:
    A retrieved chunk is relevant for a GoldenQuery iff:
    1. Chunk's document matches query.expected_documents (case-insensitive filename / doc title).
    2. Chunk's page range (page_start <= p <= page_end) intersects with query.expected_pages.
    3. Any required substring in query.required_substrings is present in chunk.text (case-folded).
    """
    if not query.is_answerable:
        return False

    if not _matches_expected_documents(chunk, query.expected_documents):
        return False

    if query.expected_pages:
        p_start, p_end = _get_page_range(chunk)
        if p_start is None or p_end is None:
            return False
        if not any(p_start <= p <= p_end for p in query.expected_pages):
            return False

    if query.required_substrings:
        text = _get_chunk_text(chunk).casefold()
        if not any(sub.casefold() in text for sub in query.required_substrings):
            return False

    return True


def calculate_recall_at_k(retrieved: list[Any], query: GoldenQuery, k: int) -> float:
    """
    Fraction of expected evidence covered in top K.
    - 0.0 for unanswerable queries or empty retrieved lists.
    - Ratio of covered expected pages / required substrings for multi-evidence queries.
    - 1.0 if any relevant evidence hit is retrieved for single-evidence queries.
    """
    if k <= 0 or not retrieved or not query.is_answerable:
        return 0.0

    top_k = retrieved[:k]

    covered_pages: set[int] = set()
    covered_substrings: set[str] = set()

    for chunk in top_k:
        if not _matches_expected_documents(chunk, query.expected_documents):
            continue

        p_start, p_end = _get_page_range(chunk)
        text = _get_chunk_text(chunk).casefold()

        # Check pages covered
        if query.expected_pages and p_start is not None and p_end is not None:
            for p in query.expected_pages:
                if p_start <= p <= p_end and (
                    not query.required_substrings
                    or any(sub.casefold() in text for sub in query.required_substrings)
                ):
                    covered_pages.add(p)

        # Check substrings covered
        if query.required_substrings:
            for sub in query.required_substrings:
                if sub.casefold() in text and (
                    not query.expected_pages
                    or (
                        p_start is not None
                        and p_end is not None
                        and any(p_start <= p <= p_end for p in query.expected_pages)
                    )
                ):
                    covered_substrings.add(sub)

    has_multi_pages = len(query.expected_pages) > 1
    has_multi_subs = len(query.required_substrings) > 1

    if has_multi_pages and has_multi_subs:
        page_ratio = len(covered_pages) / len(query.expected_pages)
        sub_ratio = len(covered_substrings) / len(query.required_substrings)
        return round((page_ratio + sub_ratio) / 2.0, 4)
    elif has_multi_pages:
        return round(len(covered_pages) / len(query.expected_pages), 4)
    elif has_multi_subs:
        return round(len(covered_substrings) / len(query.required_substrings), 4)
    else:
        # Single evidence query: 1.0 if any valid relevant hit retrieved
        return 1.0 if any(is_relevant_hit(c, query) for c in top_k) else 0.0


def calculate_precision_at_k(retrieved: list[Any], query: GoldenQuery, k: int = 5) -> float:
    """
    Precision@K: Relevant chunks in top K divided by K.
    Safely returns 0.0 for k <= 0, empty list, or unanswerable queries.
    """
    if k <= 0 or not retrieved or not query.is_answerable:
        return 0.0

    top_k = retrieved[:k]
    relevant_count = sum(1 for c in top_k if is_relevant_hit(c, query))
    return round(relevant_count / float(k), 4)


def calculate_mrr_at_k(retrieved: list[Any], query: GoldenQuery, k: int = 10) -> float:
    """
    Mean Reciprocal Rank (MRR@K): Reciprocal rank 1 / rank of first relevant hit within top K.
    Returns 0.0 if no relevant hit within top K or unanswerable.
    """
    if k <= 0 or not retrieved or not query.is_answerable:
        return 0.0

    top_k = retrieved[:k]
    for rank, chunk in enumerate(top_k, start=1):
        if is_relevant_hit(chunk, query):
            return round(1.0 / float(rank), 4)

    return 0.0


def _get_expected_target_count(query: GoldenQuery) -> int:
    """
    Compute the number of distinct relevant targets expected in ideal ranking.
    For multi-hop comparative queries, at least 2 distinct evidence targets
    (or count of distinct expected documents/pages) are required.
    For single-hop queries, 1 ideal chunk covers the required evidence.
    """
    if query.category == "multi_hop_comparative":
        distinct_docs: list[str] = []
        for doc in query.expected_documents:
            if not any(_matches_document(doc, d) for d in distinct_docs):
                distinct_docs.append(doc)
        if len(distinct_docs) > 1:
            return len(distinct_docs)
        if len(query.expected_pages) > 1:
            return len(query.expected_pages)
        return 2
    return 1


def calculate_ndcg_at_k(retrieved: list[Any], query: GoldenQuery, k: int = 5) -> float:
    """
    Normalized Discounted Cumulative Gain (NDCG@K) with binary relevance and log2(i + 1) discount.
    Returns 0.0 for empty retrieval, zero hits, or unanswerable queries.
    """
    if k <= 0 or not retrieved or not query.is_answerable:
        return 0.0

    top_k = retrieved[:k]
    relevance_scores = [1.0 if is_relevant_hit(c, query) else 0.0 for c in top_k]

    dcg = sum(
        rel / math.log2(rank + 1) for rank, rel in enumerate(relevance_scores, start=1) if rel > 0.0
    )

    if dcg <= 0.0:
        return 0.0

    hits_in_retrieved = sum(1 for c in retrieved if is_relevant_hit(c, query))
    expected_targets = _get_expected_target_count(query)
    total_relevant = max(hits_in_retrieved, expected_targets)
    ideal_k = min(total_relevant, k)
    if ideal_k <= 0:
        return 0.0

    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_k + 1))
    return round(dcg / idcg, 4)


class RetrievalEvaluator:
    """Executes multi-configuration retrieval ablation against a GoldenDataset."""

    def __init__(self, search_service: Any = None) -> None:
        self.search_service = search_service

    async def run_ablation(
        self,
        dataset: list[GoldenQuery] | GoldenDataset,
        config_name: str,
        top_k: int = 10,
    ) -> dict[str, float]:
        """
        Run retrieval ablation for a specific configuration across all answerable queries.
        Returns macro-averaged metrics:
        Recall@3, Recall@5, Recall@10, Precision@5, MRR@10, NDCG@5.
        """
        canonical_config = CONFIG_ALIASES.get(config_name.lower(), config_name.lower())
        if canonical_config not in ABLATION_CONFIGS:
            raise ValueError(
                f"Unknown configuration '{config_name}'. Allowed: {list(ABLATION_CONFIGS.keys())}"
            )

        config_params = ABLATION_CONFIGS[canonical_config]
        queries = dataset.queries if isinstance(dataset, GoldenDataset) else list(dataset)

        answerable_queries = [q for q in queries if q.is_answerable]
        if not answerable_queries:
            logger.warning("No answerable queries in dataset for retrieval evaluation.")
            return {
                "recall_at_3": 0.0,
                "recall_at_5": 0.0,
                "recall_at_10": 0.0,
                "precision_at_5": 0.0,
                "mrr_at_10": 0.0,
                "ndcg_at_5": 0.0,
            }

        recalls_3: list[float] = []
        recalls_5: list[float] = []
        recalls_10: list[float] = []
        precisions_5: list[float] = []
        mrrs_10: list[float] = []
        ndcgs_5: list[float] = []

        for q in answerable_queries:
            if self.search_service is not None:
                retrieved = await self.search_service.search(
                    query=q.query,
                    top_k=top_k,
                    **config_params,
                )
            else:
                retrieved = []

            recalls_3.append(calculate_recall_at_k(retrieved, q, k=3))
            recalls_5.append(calculate_recall_at_k(retrieved, q, k=5))
            recalls_10.append(calculate_recall_at_k(retrieved, q, k=10))
            precisions_5.append(calculate_precision_at_k(retrieved, q, k=5))
            mrrs_10.append(calculate_mrr_at_k(retrieved, q, k=10))
            ndcgs_5.append(calculate_ndcg_at_k(retrieved, q, k=5))

        n = len(answerable_queries)
        return {
            "recall_at_3": round(sum(recalls_3) / n, 4),
            "recall_at_5": round(sum(recalls_5) / n, 4),
            "recall_at_10": round(sum(recalls_10) / n, 4),
            "precision_at_5": round(sum(precisions_5) / n, 4),
            "mrr_at_10": round(sum(mrrs_10) / n, 4),
            "ndcg_at_5": round(sum(ndcgs_5) / n, 4),
        }

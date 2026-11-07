"""5-Way Retrieval Ablation Runner and CLI.

Evaluates RAG retrieval across 5 pipeline configurations:
1. Dense Only (Flat)
2. Sparse BM25 Only
3. Hybrid RRF (Flat)
4. Hybrid RRF + Hierarchical Parent-Child + Reranker
5. Full Pipeline (Query Transform + Hybrid RRF + Parent-Child + Reranker)

Computes macro-averaged metrics (Recall@3, Recall@5, Recall@10, Precision@5, MRR@10, NDCG@5)
across all answerable queries, renders a markdown report, and saves it to disk.
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.schemas.query import RetrievedChunk
from evals.metrics import (
    ABLATION_CONFIGS,
    CONFIG_ALIASES,
    RetrievalEvaluator,
)
from evals.schemas import GoldenDataset

logger = logging.getLogger("evals.retrieval_ablation")

CONFIG_DISPLAY_NAMES: dict[str, str] = {
    "dense": "Dense Only (Flat)",
    "sparse": "Sparse BM25 Only",
    "hybrid_flat": "Hybrid RRF (Flat)",
    "hybrid_hierarchical_rerank": "Hybrid + Hierarchical + Reranker",
    "full_pipeline": "Full Pipeline (Transform + Hybrid + Hierarchical + Rerank)",
}

DEFAULT_CONFIG_ORDER: list[str] = [
    "dense",
    "sparse",
    "hybrid_flat",
    "hybrid_hierarchical_rerank",
    "full_pipeline",
]


class DryRunSearchService:
    """Simulates realistic search retrieval for dry-run offline CI verification."""

    async def search(
        self,
        query: str,
        top_k: int = 10,
        search_mode: str = "hybrid",
        resolve_parents: bool = True,
        enable_reranker: bool = False,
        enable_query_rewrite: bool = False,
        **kwargs: Any,
    ) -> list[RetrievedChunk]:
        """Generate mock chunks matching expected queries with realistic ranking tiers."""
        chunks: list[RetrievedChunk] = []

        q_lower = query.lower()
        if "apple" in q_lower or "commercial paper" in q_lower:
            doc_title = "Apple_10-K-Q4-2023-As-Filed.pdf"
            p_start, p_end = (30, 35) if resolve_parents else (33, 33)
            matching_text = "commercial paper liabilities of 5,985 million"
        elif "transformer" in q_lower or "attention" in q_lower or "scaled dot-product" in q_lower:
            doc_title = "Attention Is All You Need.pdf"
            p_start, p_end = (3, 5) if resolve_parents else (4, 4)
            matching_text = "scaled dot-product attention softmax √dk"
        elif "nvidia" in q_lower or "segment" in q_lower:
            doc_title = "nvidia_10k_sample.pdf"
            p_start, p_end = (1, 4) if resolve_parents else (2, 2)
            matching_text = "compute & networking graphics data center revenue rose 217%"
        elif "residual" in q_lower or "layer" in q_lower:
            doc_title = "Deep Residual Learning for Image Recognition.pdf"
            p_start, p_end = (1, 3) if resolve_parents else (2, 2)
            matching_text = "residual learning f(x)+x degradation problem"
        elif "tariff" in q_lower or "schedule" in q_lower:
            doc_title = "Harmonized Tariff Schedule of the United States Revision 22 (2026).pdf"
            p_start, p_end = (1, 3) if resolve_parents else (2, 2)
            matching_text = "composite machines principal function"
        else:
            doc_title = "unknown_document.pdf"
            p_start, p_end = (1, 1)
            matching_text = "unrelated dummy text"

        hit_rank = 3
        if (enable_query_rewrite and enable_reranker) or (enable_reranker and resolve_parents):
            hit_rank = 1
        elif search_mode in ("hybrid", "dense"):
            hit_rank = 2
        elif search_mode == "sparse":
            hit_rank = 3

        for rank in range(1, top_k + 1):
            is_hit = rank == hit_rank
            chunks.append(
                RetrievedChunk(
                    source_id=rank,
                    chunk_id=uuid4(),
                    document_id=uuid4(),
                    document_title=doc_title if is_hit else f"distractor_{rank}.pdf",
                    page_start=p_start if is_hit else 99,
                    page_end=p_end if is_hit else 99,
                    section_path="Section > Item",
                    element_type="text",
                    text=matching_text if is_hit else f"Distractor text block at rank {rank}.",
                    contextualized_text="",
                    similarity_score=round(max(0.1, 1.0 - (rank * 0.08)), 4),
                )
            )

        return chunks


def load_dataset(dataset_path: Path | str) -> GoldenDataset:
    """Load GoldenDataset from JSON file."""
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Golden dataset not found: {path.resolve()}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return GoldenDataset.model_validate(data)


def format_markdown_report(
    results: dict[str, dict[str, float]],
    dataset_path: str,
    total_queries: int,
    answerable_count: int,
) -> str:
    """Format evaluation ablation results into a publication-grade markdown table."""
    unanswerable = total_queries - answerable_count
    lines: list[str] = [
        "# Retrieval Ablation Benchmark Report",
        "",
        f"- **Dataset Path:** `{dataset_path}`",
        f"- **Total Queries in Dataset:** {total_queries}",
        f"- **Answerable Queries Evaluated:** {answerable_count}",
        f"- **Unanswerable / Adversarial Queries Excluded from IR Averages:** {unanswerable}",
        "",
        "## Summary Metrics Table",
        "",
        "| Configuration | Recall@3 | Recall@5 | Recall@10 | Precision@5 | MRR@10 | NDCG@5 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for config_key in DEFAULT_CONFIG_ORDER:
        if config_key not in results:
            continue
        display_name = CONFIG_DISPLAY_NAMES.get(config_key, config_key)
        m = results[config_key]
        lines.append(
            f"| **{display_name}** | {m.get('recall_at_3', 0.0):.4f} | "
            f"{m.get('recall_at_5', 0.0):.4f} | {m.get('recall_at_10', 0.0):.4f} | "
            f"{m.get('precision_at_5', 0.0):.4f} | {m.get('mrr_at_10', 0.0):.4f} | "
            f"{m.get('ndcg_at_5', 0.0):.4f} |"
        )

    # Any custom configs not in default order
    for config_key, m in results.items():
        if config_key in DEFAULT_CONFIG_ORDER:
            continue
        display_name = CONFIG_DISPLAY_NAMES.get(config_key, config_key)
        lines.append(
            f"| **{display_name}** | {m.get('recall_at_3', 0.0):.4f} | "
            f"{m.get('recall_at_5', 0.0):.4f} | {m.get('recall_at_10', 0.0):.4f} | "
            f"{m.get('precision_at_5', 0.0):.4f} | {m.get('mrr_at_10', 0.0):.4f} | "
            f"{m.get('ndcg_at_5', 0.0):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Metric Definitions",
            "",
            "- **Recall@K:** Macro-averaged fraction of ground-truth evidence pages and "
            "required substrings covered in top K retrieved results.",
            "- **Precision@K:** Proportion of top K retrieved chunks that satisfy deterministic "
            "hit criteria (document title, page range, and required substrings).",
            "- **MRR@10 (Mean Reciprocal Rank):** Reciprocal rank (1 / rank) of first relevant "
            "chunk within top 10.",
            "- **NDCG@5:** Normalized Discounted Cumulative Gain at rank 5 using binary relevance "
            "and logarithmic positional discounting.",
            "",
        ]
    )

    return "\n".join(lines)


async def run_evaluations(
    dataset_path: str = "evals/datasets/golden_v1.json",
    configs_to_run: list[str] | None = None,
    dry_run: bool = False,
    output_path: str = "evals/reports/retrieval_ablation.md",
    top_k: int = 10,
) -> tuple[dict[str, dict[str, float]], str]:
    """Execute evaluation across selected configurations and save markdown report."""
    dataset = load_dataset(dataset_path)
    total_queries = len(dataset.queries)
    answerable_count = sum(1 for q in dataset.queries if q.is_answerable)

    selected_configs: list[str] = []
    if configs_to_run:
        for c in configs_to_run:
            canonical = CONFIG_ALIASES.get(c.lower(), c.lower())
            if canonical in ABLATION_CONFIGS and canonical not in selected_configs:
                selected_configs.append(canonical)
            elif c.lower() == "all":
                selected_configs = list(DEFAULT_CONFIG_ORDER)
                break
    if not selected_configs:
        selected_configs = list(DEFAULT_CONFIG_ORDER)

    results: dict[str, dict[str, float]] = {}

    if dry_run:
        logger.info("Executing retrieval evaluation in DRY-RUN mode (mock search service).")
        evaluator = RetrievalEvaluator(search_service=DryRunSearchService())
        for config_key in selected_configs:
            logger.info("Evaluating configuration: %s", config_key)
            results[config_key] = await evaluator.run_ablation(
                dataset=dataset,
                config_name=config_key,
                top_k=top_k,
            )
    else:
        logger.info("Connecting to database session and neural providers for retrieval evaluation.")
        from app.db.session import async_session_maker
        from app.services.providers.factory import get_embedding_provider, get_llm_provider
        from app.services.retrieval.reranker import CrossEncoderReranker
        from app.services.retrieval.vector_search import VectorSearchService

        settings = get_settings()
        embedding_provider = get_embedding_provider()
        llm_provider = get_llm_provider() if settings.LLM_PROVIDER else None
        reranker = CrossEncoderReranker() if settings.ENABLE_RERANKER else None

        async with async_session_maker() as session:
            search_service = VectorSearchService(
                session=session,
                embedding_provider=embedding_provider,
                llm_provider=llm_provider,
                reranker=reranker,
            )
            evaluator = RetrievalEvaluator(search_service=search_service)
            for config_key in selected_configs:
                logger.info("Evaluating configuration against live database: %s", config_key)
                results[config_key] = await evaluator.run_ablation(
                    dataset=dataset,
                    config_name=config_key,
                    top_k=top_k,
                )

    report_md = format_markdown_report(
        results=results,
        dataset_path=dataset_path,
        total_queries=total_queries,
        answerable_count=answerable_count,
    )

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(report_md, encoding="utf-8")
    logger.info("Ablation report saved to %s", out_file.resolve())

    return results, report_md


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 5-Way Retrieval Ablation Evaluation and produce IR metrics report."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="evals/datasets/golden_v1.json",
        help="Path to GoldenDataset JSON file (default: evals/datasets/golden_v1.json)",
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="*",
        default=None,
        help="Configurations to evaluate: dense, sparse, hybrid_flat, "
        "hybrid_hierarchical_rerank, full_pipeline (or 'all')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without active database/model calls using deterministic mock search",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evals/reports/retrieval_ablation.md",
        help="Destination path for markdown report (default: evals/reports/retrieval_ablation.md)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Retrieval depth evaluated per query (default: 10)",
    )
    return parser.parse_args()


async def main() -> None:
    setup_logging()
    args = parse_args()

    _results, report = await run_evaluations(
        dataset_path=args.dataset,
        configs_to_run=args.configs,
        dry_run=args.dry_run,
        output_path=args.output,
        top_k=args.top_k,
    )

    # Print summary table to stdout
    print("\n" + report + "\n")


if __name__ == "__main__":
    asyncio.run(main())

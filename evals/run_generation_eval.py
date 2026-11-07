"""Generation & Hallucination Benchmark CLI.

Evaluates end-to-end grounded RAG generation using a Hybrid LLM-as-a-Judge:
- Zero-cost deterministic metrics: Abstention Accuracy and Citation Precision.
- Single-pass LLM Judge: Faithfulness (1 - Hallucination rate), Answer Relevance,
  and Ground Truth Correctness.
- Offline deterministic fallback mode for CI and reproducible verification.

Computes macro averages across categories and overall metrics,
renders a summary table, and writes evals/reports/generation_benchmark.md.
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
from app.schemas.query import Citation, RAGResponse, RetrievedChunk
from evals.judge import JudgeResult, RAGJudge
from evals.schemas import GoldenDataset, GoldenQuery

logger = logging.getLogger("evals.generation_benchmark")

CATEGORY_DISPLAY_NAMES: dict[str, str] = {
    "exact_keyword": "Exact Keyword",
    "single_hop_semantic": "Single-Hop Semantic",
    "multi_hop_comparative": "Multi-Hop Comparative",
    "tabular_numerical": "Tabular Numerical",
    "unanswerable_adversarial": "Unanswerable Adversarial",
}

DEFAULT_CATEGORIES: list[str] = [
    "exact_keyword",
    "single_hop_semantic",
    "multi_hop_comparative",
    "tabular_numerical",
    "unanswerable_adversarial",
]


def load_dataset(dataset_path: str) -> GoldenDataset:
    """Load and validate the golden evaluation dataset JSON."""
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Golden dataset file not found at: {path.resolve()}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return GoldenDataset.model_validate(data)


def generate_dry_run_response(query: GoldenQuery) -> RAGResponse:
    """Simulate a realistic RAG pipeline response for dry-run offline verification."""
    if not query.is_answerable:
        return RAGResponse(
            answer=(
                "The provided documents do not contain sufficient information "
                "to answer this question."
            ),
            is_answerable=False,
            citations=[],
            sources=[],
            confidence_score=0.0,
            latency_ms=25.0,
        )

    doc_title = query.expected_documents[0] if query.expected_documents else "Golden_Document.pdf"
    page = query.expected_pages[0] if query.expected_pages else 1
    chunk_id = uuid4()
    doc_id = uuid4()

    chunk = RetrievedChunk(
        source_id=1,
        chunk_id=chunk_id,
        document_id=doc_id,
        document_title=doc_title,
        page_start=page,
        page_end=page,
        section_path="Main Content",
        element_type="prose",
        text=query.ground_truth_answer,
        contextualized_text=f"[Doc: {doc_title}] {query.ground_truth_answer}",
        similarity_score=0.92,
    )
    citation = Citation(
        source_id=1,
        document_id=doc_id,
        document_title=doc_title,
        page_start=page,
        page_end=page,
        section_path="Main Content",
        supporting_quote=query.ground_truth_answer,
        quote=query.ground_truth_answer,
    )
    return RAGResponse(
        answer=f"{query.ground_truth_answer} [1]",
        is_answerable=True,
        citations=[citation],
        sources=[chunk],
        confidence_score=0.94,
        latency_ms=120.0,
    )


def compute_macro_averages(results: list[dict[str, Any]]) -> dict[str, float]:
    """Calculate mean metric scores across a collection of evaluation item records."""
    if not results:
        return {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "ground_truth_correctness": 0.0,
            "citation_precision": 0.0,
            "abstention_accuracy": 0.0,
        }

    n = len(results)
    return {
        "faithfulness": round(sum(r["judge"].faithfulness for r in results) / n, 4),
        "answer_relevance": round(sum(r["judge"].answer_relevance for r in results) / n, 4),
        "ground_truth_correctness": round(
            sum(r["judge"].ground_truth_correctness for r in results) / n, 4
        ),
        "citation_precision": round(sum(r["judge"].citation_precision for r in results) / n, 4),
        "abstention_accuracy": round(sum(r["judge"].abstention_accuracy for r in results) / n, 4),
    }


def format_markdown_report(
    summary_by_category: dict[str, dict[str, Any]],
    overall_summary: dict[str, float],
    dataset_path: str,
    judge_mode: str,
    total_queries: int,
    answerable_count: int,
    unanswerable_count: int,
    detailed_records: list[dict[str, Any]],
) -> str:
    """Format evaluation metrics into a publication-ready markdown benchmark report."""
    lines: list[str] = [
        "# Generation & Hallucination Benchmark Report",
        "",
        f"- **Dataset Path:** `{dataset_path}`",
        f"- **Judge Mode:** `{judge_mode}`",
        f"- **Total Queries Evaluated:** {total_queries}",
        f"- **Answerable Queries:** {answerable_count}",
        f"- **Unanswerable / Adversarial Queries:** {unanswerable_count}",
        "",
        "## Summary Metrics Table",
        "",
        "| Category Archetype | Queries | Faithfulness | Relevance | Correctness | "
        "Citation Precision | Abstention Accuracy |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for cat in DEFAULT_CATEGORIES:
        display = CATEGORY_DISPLAY_NAMES.get(cat, cat)
        if cat in summary_by_category:
            s = summary_by_category[cat]
            cnt = s["count"]
            m = s["metrics"]
            lines.append(
                f"| **{display}** | {cnt} | {m['faithfulness']:.4f} | "
                f"{m['answer_relevance']:.4f} | {m['ground_truth_correctness']:.4f} | "
                f"{m['citation_precision']:.4f} | {m['abstention_accuracy']:.4f} |"
            )

    lines.append(
        f"| **Overall Macro Average** | {total_queries} | {overall_summary['faithfulness']:.4f} | "
        f"{overall_summary['answer_relevance']:.4f} | "
        f"{overall_summary['ground_truth_correctness']:.4f} | "
        f"{overall_summary['citation_precision']:.4f} | "
        f"{overall_summary['abstention_accuracy']:.4f} |"
    )

    lines.extend(
        [
            "",
            "## Metric Definitions",
            "",
            "- **Faithfulness (1 - Hallucination):** Degree to which generated assertions "
            "are grounded in retrieved source context.",
            "- **Answer Relevance:** Degree to which the response directly answers the user prompt "
            "without drifting or extraneous content.",
            "- **Ground Truth Correctness:** Semantic and factual agreement between "
            "generated answer and curated reference ground truth.",
            "- **Citation Precision:** Ratio of successfully verified citations (`[N]`) to total "
            "attempted citations (`verified / (verified + unverified)`).",
            "- **Abstention Accuracy:** Accuracy of model abstention on unanswerable / "
            "adversarial queries (and answering on answerable queries).",
            "",
            "## Detailed Evaluation Results",
            "",
            "| Query ID | Category | Faithfulness | Relevance | Correctness | "
            "Citation Prec | Abstention Acc | Latency (ms) | Reasoning |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]
    )

    for item in detailed_records:
        q: GoldenQuery = item["query"]
        r: RAGResponse = item["response"]
        j: JudgeResult = item["judge"]
        reasoning_snippet = j.reasoning.replace("\n", " ")[:80] if j.reasoning else "N/A"
        lines.append(
            f"| `{q.id}` | {q.category} | {j.faithfulness:.4f} | {j.answer_relevance:.4f} | "
            f"{j.ground_truth_correctness:.4f} | {j.citation_precision:.4f} | "
            f"{j.abstention_accuracy:.4f} | {r.latency_ms:.1f} | {reasoning_snippet} |"
        )

    lines.append("")
    return "\n".join(lines)


async def run_generation_benchmark(
    dataset_path: str = "evals/datasets/golden_v1.json",
    judge_mode: str | None = None,
    dry_run: bool = False,
    output_path: str = "evals/reports/generation_benchmark.md",
    top_k: int = 5,
) -> tuple[dict[str, Any], str]:
    """Execute generation and hallucination evaluation pipeline and save report."""
    dataset = load_dataset(dataset_path)
    total_queries = len(dataset.queries)
    answerable_count = sum(1 for q in dataset.queries if q.is_answerable)
    unanswerable_count = total_queries - answerable_count

    settings = get_settings()

    # Determine judge mode if not explicitly supplied
    effective_judge_mode = judge_mode
    if not effective_judge_mode:
        if settings.GEMINI_API_KEY or settings.OPENAI_API_KEY:
            effective_judge_mode = "llm"
        else:
            effective_judge_mode = "deterministic"

    logger.info(
        "Initializing Generation Benchmark: %d queries, judge_mode=%s, dry_run=%s",
        total_queries,
        effective_judge_mode,
        dry_run,
    )

    llm_provider = None
    if not dry_run or effective_judge_mode == "llm":
        from app.services.providers.factory import get_llm_provider

        try:
            llm_provider = get_llm_provider() if settings.LLM_PROVIDER else None
        except Exception as exc:
            logger.warning("Could not initialize LLMProvider (%s); using deterministic judge.", exc)
            effective_judge_mode = "deterministic"

    judge = RAGJudge(llm_provider=llm_provider, default_mode=effective_judge_mode)

    detailed_records: list[dict[str, Any]] = []

    if dry_run:
        logger.info("Executing benchmark in DRY-RUN mode (mock generation responses).")
        for q in dataset.queries:
            resp = generate_dry_run_response(q)
            res = await judge.evaluate_response(q, resp, mode=effective_judge_mode)
            detailed_records.append({"query": q, "response": resp, "judge": res})
    else:
        logger.info("Executing benchmark against live database and neural retrieval pipeline.")
        from app.db.session import async_session_maker
        from app.services.providers.factory import get_embedding_provider
        from app.services.retrieval.generator import GroundedGenerator
        from app.services.retrieval.reranker import CrossEncoderReranker
        from app.services.retrieval.vector_search import VectorSearchService

        embedding_provider = get_embedding_provider()
        reranker = CrossEncoderReranker() if getattr(settings, "RERANKER_MODEL", None) else None

        async with async_session_maker() as session:
            search_service = VectorSearchService(
                session=session,
                embedding_provider=embedding_provider,
                llm_provider=llm_provider,
                reranker=reranker,
            )
            generator = GroundedGenerator(llm_provider=llm_provider)

            for q in dataset.queries:
                logger.info("Evaluating query [%s]: %s", q.id, q.query[:60])
                chunks = await search_service.search(
                    query=q.query,
                    top_k=top_k,
                    search_mode="hybrid",
                    resolve_parents=True,
                    enable_reranker=True,
                    enable_query_rewrite=True,
                )
                resp = await generator.generate(query=q.query, chunks=chunks)
                res = await judge.evaluate_response(q, resp, mode=effective_judge_mode)
                detailed_records.append({"query": q, "response": resp, "judge": res})

    # Group metrics by category
    summary_by_category: dict[str, dict[str, Any]] = {}
    for cat in DEFAULT_CATEGORIES:
        cat_items = [item for item in detailed_records if item["query"].category == cat]
        if cat_items:
            summary_by_category[cat] = {
                "count": len(cat_items),
                "metrics": compute_macro_averages(cat_items),
            }

    overall_metrics = compute_macro_averages(detailed_records)

    report_md = format_markdown_report(
        summary_by_category=summary_by_category,
        overall_summary=overall_metrics,
        dataset_path=dataset_path,
        judge_mode=effective_judge_mode,
        total_queries=total_queries,
        answerable_count=answerable_count,
        unanswerable_count=unanswerable_count,
        detailed_records=detailed_records,
    )

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(report_md, encoding="utf-8")
    logger.info("Generation benchmark report written to: %s", out_file.resolve())

    payload = {
        "summary_by_category": summary_by_category,
        "overall_summary": overall_metrics,
        "total_queries": total_queries,
        "answerable_count": answerable_count,
        "unanswerable_count": unanswerable_count,
    }
    return payload, report_md


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for run_generation_eval."""
    parser = argparse.ArgumentParser(
        description="Run Generation and Hallucination Evaluation Suite with Hybrid LLM-as-a-Judge."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="evals/datasets/golden_v1.json",
        help="Path to GoldenDataset JSON file (default: evals/datasets/golden_v1.json)",
    )
    parser.add_argument(
        "--judge-mode",
        type=str,
        choices=["llm", "deterministic"],
        default=None,
        help="Judge mode: 'llm' or 'deterministic' "
        "(default: 'llm' if API key available else 'deterministic')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate generation responses without live neural search or model generation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evals/reports/generation_benchmark.md",
        help="Destination path for markdown report "
        "(default: evals/reports/generation_benchmark.md)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Retrieval depth evaluated per query (default: 5)",
    )
    return parser.parse_args()


async def main() -> None:
    """Entry point for generation benchmark CLI."""
    setup_logging()
    args = parse_args()

    _payload, report_md = await run_generation_benchmark(
        dataset_path=args.dataset,
        judge_mode=args.judge_mode,
        dry_run=args.dry_run,
        output_path=args.output,
        top_k=args.top_k,
    )

    print("\n" + report_md + "\n")


if __name__ == "__main__":
    asyncio.run(main())

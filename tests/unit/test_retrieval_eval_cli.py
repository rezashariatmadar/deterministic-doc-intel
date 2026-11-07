"""Unit tests for the retrieval ablation runner CLI."""

from pathlib import Path

import pytest

from evals.run_retrieval_eval import format_markdown_report, load_dataset, run_evaluations


def test_load_dataset() -> None:
    dataset = load_dataset("evals/datasets/golden_v1.json")
    assert len(dataset.queries) == 28
    assert sum(1 for q in dataset.queries if q.is_answerable) == 24


def test_format_markdown_report() -> None:
    results = {
        "dense": {
            "recall_at_3": 0.5,
            "recall_at_5": 0.6,
            "recall_at_10": 0.7,
            "precision_at_5": 0.2,
            "mrr_at_10": 0.5,
            "ndcg_at_5": 0.55,
        },
        "full_pipeline": {
            "recall_at_3": 0.8,
            "recall_at_5": 0.9,
            "recall_at_10": 0.95,
            "precision_at_5": 0.4,
            "mrr_at_10": 0.85,
            "ndcg_at_5": 0.82,
        },
    }
    report = format_markdown_report(
        results=results,
        dataset_path="evals/datasets/golden_v1.json",
        total_queries=28,
        answerable_count=24,
    )
    assert "# Retrieval Ablation Benchmark Report" in report
    assert "Dense Only (Flat)" in report
    assert "Full Pipeline" in report
    assert "0.5000" in report
    assert "0.8500" in report


@pytest.mark.asyncio
async def test_run_evaluations_dry_run(tmp_path: Path) -> None:
    out_file = tmp_path / "report.md"
    results, _report = await run_evaluations(
        dataset_path="evals/datasets/golden_v1.json",
        configs_to_run=["dense", "hybrid_flat"],
        dry_run=True,
        output_path=str(out_file),
        top_k=5,
    )
    assert "dense" in results
    assert "hybrid_flat" in results
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "Dense Only (Flat)" in content
    assert "Hybrid RRF (Flat)" in content

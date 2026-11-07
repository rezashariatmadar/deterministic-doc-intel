"""Unit tests for the generation & hallucination benchmark runner CLI."""

from pathlib import Path
from unittest.mock import patch

import pytest

from evals.judge import JudgeResult
from evals.run_generation_eval import (
    compute_macro_averages,
    format_markdown_report,
    generate_dry_run_response,
    load_dataset,
    parse_args,
    run_generation_benchmark,
)
from evals.schemas import GoldenDataset, GoldenQuery


def test_load_dataset() -> None:
    dataset = load_dataset("evals/datasets/golden_v1.json")
    assert isinstance(dataset, GoldenDataset)
    assert len(dataset.queries) == 28
    assert sum(1 for q in dataset.queries if q.is_answerable) == 24
    assert sum(1 for q in dataset.queries if not q.is_answerable) == 4


def test_generate_dry_run_response() -> None:
    answerable_q = GoldenQuery(
        id="q01",
        category="exact_keyword",
        query="What were the liabilities?",
        ground_truth_answer="Liabilities were $5,985 million.",
        is_answerable=True,
    )
    resp_ans = generate_dry_run_response(answerable_q)
    assert resp_ans.is_answerable is True
    assert "5,985" in resp_ans.answer
    assert "[1]" in resp_ans.answer
    assert len(resp_ans.citations) == 1
    assert len(resp_ans.sources) == 1

    unanswerable_q = GoldenQuery(
        id="q25",
        category="unanswerable_adversarial",
        query="What is out-of-corpus data?",
        ground_truth_answer="The provided documents do not contain information.",
        is_answerable=False,
    )
    resp_unans = generate_dry_run_response(unanswerable_q)
    assert resp_unans.is_answerable is False
    assert "do not contain sufficient information" in resp_unans.answer
    assert len(resp_unans.citations) == 0


def test_compute_macro_averages() -> None:
    records = [
        {
            "judge": JudgeResult(
                faithfulness=1.0,
                answer_relevance=0.8,
                ground_truth_correctness=0.9,
                citation_precision=1.0,
                abstention_accuracy=1.0,
                reasoning="test",
            )
        },
        {
            "judge": JudgeResult(
                faithfulness=0.8,
                answer_relevance=1.0,
                ground_truth_correctness=0.7,
                citation_precision=0.5,
                abstention_accuracy=1.0,
                reasoning="test",
            )
        },
    ]
    avg = compute_macro_averages(records)
    assert avg["faithfulness"] == 0.9
    assert avg["answer_relevance"] == 0.9
    assert avg["ground_truth_correctness"] == 0.8
    assert avg["citation_precision"] == 0.75
    assert avg["abstention_accuracy"] == 1.0


def test_format_markdown_report() -> None:
    summary_by_cat = {
        "exact_keyword": {
            "count": 6,
            "metrics": {
                "faithfulness": 0.95,
                "answer_relevance": 0.92,
                "ground_truth_correctness": 0.90,
                "citation_precision": 1.0,
                "abstention_accuracy": 1.0,
            },
        }
    }
    overall = {
        "faithfulness": 0.95,
        "answer_relevance": 0.92,
        "ground_truth_correctness": 0.90,
        "citation_precision": 1.0,
        "abstention_accuracy": 1.0,
    }
    query = GoldenQuery(
        id="q01",
        category="exact_keyword",
        query="Sample question?",
        ground_truth_answer="Sample answer.",
        is_answerable=True,
    )
    response = generate_dry_run_response(query)
    judge = JudgeResult(
        faithfulness=0.95,
        answer_relevance=0.92,
        ground_truth_correctness=0.90,
        citation_precision=1.0,
        abstention_accuracy=1.0,
        reasoning="Good answer",
    )
    detailed = [{"query": query, "response": response, "judge": judge}]

    report = format_markdown_report(
        summary_by_category=summary_by_cat,
        overall_summary=overall,
        dataset_path="evals/datasets/golden_v1.json",
        judge_mode="deterministic",
        total_queries=28,
        answerable_count=24,
        unanswerable_count=4,
        detailed_records=detailed,
    )

    assert "# Generation & Hallucination Benchmark Report" in report
    assert "**Judge Mode:** `deterministic`" in report
    assert "Exact Keyword" in report
    assert "Overall Macro Average" in report
    assert "`q01`" in report


def test_parse_args() -> None:
    test_args = [
        "run_generation_eval.py",
        "--dataset",
        "custom_dataset.json",
        "--judge-mode",
        "deterministic",
        "--dry-run",
        "--output",
        "custom_report.md",
        "--top-k",
        "8",
    ]
    with patch("sys.argv", test_args):
        args = parse_args()
        assert args.dataset == "custom_dataset.json"
        assert args.judge_mode == "deterministic"
        assert args.dry_run is True
        assert args.output == "custom_report.md"
        assert args.top_k == 8


@pytest.mark.asyncio
async def test_run_generation_benchmark_dry_run(tmp_path: Path) -> None:
    out_file = tmp_path / "gen_report.md"
    payload, report_md = await run_generation_benchmark(
        dataset_path="evals/datasets/golden_v1.json",
        judge_mode="deterministic",
        dry_run=True,
        output_path=str(out_file),
        top_k=5,
    )

    assert payload["total_queries"] == 28
    assert payload["answerable_count"] == 24
    assert payload["unanswerable_count"] == 4
    assert "summary_by_category" in payload
    assert "overall_summary" in payload

    # Check overall metrics
    overall = payload["overall_summary"]
    assert overall["faithfulness"] >= 0.8
    assert overall["answer_relevance"] >= 0.8
    assert overall["ground_truth_correctness"] >= 0.8
    assert overall["citation_precision"] >= 0.9
    assert overall["abstention_accuracy"] == 1.0

    # Check output file
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "# Generation & Hallucination Benchmark Report" in content
    assert "Unanswerable Adversarial" in content
    assert report_md == content

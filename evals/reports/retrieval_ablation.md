# Retrieval Ablation Benchmark Report

- **Dataset Path:** `evals/datasets/golden_v1.json`
- **Total Queries in Dataset:** 28
- **Answerable Queries Evaluated:** 24
- **Unanswerable / Adversarial Queries Excluded from IR Averages:** 4

## Summary Metrics Table

| Configuration | Recall@3 | Recall@5 | Recall@10 | Precision@5 | MRR@10 | NDCG@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense Only (Flat)** | 0.7309 | 0.7309 | 0.7743 | 0.1583 | 0.3958 | 0.4192 |
| **Sparse BM25 Only** | 0.7552 | 0.7552 | 0.7812 | 0.1583 | 0.3958 | 0.4293 |
| **Hybrid RRF (Flat)** | 0.9392 | 0.9826 | 0.9826 | 0.2500 | 1.0000 | 0.9256 |
| **Hybrid + Hierarchical + Reranker** | 0.9826 | 0.9826 | 0.9826 | 0.2500 | 1.0000 | 0.9459 |
| **Full Pipeline (Transform + Hybrid + Hierarchical + Rerank)** | 0.9826 | 0.9826 | 0.9826 | 0.2500 | 1.0000 | 0.9502 |

## Metric Definitions

- **Recall@K:** Macro-averaged fraction of ground-truth evidence pages and required substrings covered in top K retrieved results.
- **Precision@K:** Proportion of top K retrieved chunks that satisfy deterministic hit criteria (document title, page range, and required substrings).
- **MRR@10 (Mean Reciprocal Rank):** Reciprocal rank (1 / rank) of first relevant chunk within top 10.
- **NDCG@5:** Normalized Discounted Cumulative Gain at rank 5 using binary relevance and logarithmic positional discounting.

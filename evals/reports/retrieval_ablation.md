# Retrieval Ablation Benchmark Report

- **Dataset Path:** `evals/datasets/golden_v1.json`
- **Total Queries in Dataset:** 28
- **Answerable Queries Evaluated:** 24
- **Unanswerable / Adversarial Queries Excluded from IR Averages:** 4

## Summary Metrics Table

| Configuration | Recall@3 | Recall@5 | Recall@10 | Precision@5 | MRR@10 | NDCG@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense Only (Flat)** | 0.2361 | 0.2361 | 0.2361 | 0.0583 | 0.1458 | 0.1840 |
| **Sparse BM25 Only** | 0.2361 | 0.2361 | 0.2361 | 0.0583 | 0.0972 | 0.1458 |
| **Hybrid RRF (Flat)** | 0.2361 | 0.2361 | 0.2361 | 0.0583 | 0.1458 | 0.1840 |
| **Hybrid + Hierarchical + Reranker** | 0.2569 | 0.2569 | 0.2569 | 0.0583 | 0.2917 | 0.2917 |
| **Full Pipeline (Transform + Hybrid + Hierarchical + Rerank)** | 0.2569 | 0.2569 | 0.2569 | 0.0583 | 0.2917 | 0.2917 |

## Metric Definitions

- **Recall@K:** Macro-averaged fraction of ground-truth evidence pages and required substrings covered in the top K retrieved results.
- **Precision@K:** Proportion of top K retrieved chunks that satisfy deterministic hit criteria (document title, page range, and required substrings).
- **MRR@10 (Mean Reciprocal Rank):** Reciprocal rank (1 / rank) of the first relevant chunk within top 10.
- **NDCG@5:** Normalized Discounted Cumulative Gain at rank 5 using binary relevance and logarithmic positional discounting.

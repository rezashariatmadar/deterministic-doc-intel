# Generation & Hallucination Benchmark Report

- **Dataset Path:** `evals/datasets/golden_v1.json`
- **Judge Mode:** `deterministic`
- **Total Queries Evaluated:** 28
- **Answerable Queries:** 24
- **Unanswerable / Adversarial Queries:** 4

## Summary Metrics Table

| Category Archetype | Queries | Faithfulness | Relevance | Correctness | Citation Precision | Abstention Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exact Keyword** | 6 | 1.0000 | 0.7417 | 0.8833 | 1.0000 | 1.0000 |
| **Single-Hop Semantic** | 6 | 1.0000 | 0.7364 | 0.9514 | 1.0000 | 1.0000 |
| **Multi-Hop Comparative** | 6 | 1.0000 | 0.8493 | 0.9417 | 1.0000 | 1.0000 |
| **Tabular Numerical** | 6 | 1.0000 | 0.8808 | 0.9417 | 1.0000 | 1.0000 |
| **Unanswerable Adversarial** | 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Overall Macro Average** | 28 | 1.0000 | 0.8303 | 0.9396 | 1.0000 | 1.0000 |

## Metric Definitions

- **Faithfulness (1 - Hallucination):** Degree to which generated assertions are grounded in retrieved source context.
- **Answer Relevance:** Degree to which the response directly answers the user prompt without drifting or extraneous content.
- **Ground Truth Correctness:** Semantic and factual agreement between generated answer and curated reference ground truth.
- **Citation Precision:** Ratio of successfully verified citations (`[N]`) to total attempted citations (`verified / (verified + unverified)`).
- **Abstention Accuracy:** Accuracy of model abstention on unanswerable / adversarial queries (and answering on answerable queries).

## Detailed Evaluation Results

| Query ID | Category | Faithfulness | Relevance | Correctness | Citation Prec | Abstention Acc | Latency (ms) | Reasoning |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `q01` | exact_keyword | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 120.0 | Evaluated via deterministic offline heuristics. |
| `q02` | exact_keyword | 1.0000 | 0.6500 | 0.8833 | 1.0000 | 1.0000 | 120.0 | Evaluated via deterministic offline heuristics. |
| `q03` | exact_keyword | 1.0000 | 0.7500 | 1.0000 | 1.0000 | 1.0000 | 120.0 | Evaluated via deterministic offline heuristics. |
| `q04` | exact_keyword | 1.0000 | 0.4500 | 0.6500 | 1.0000 | 1.0000 | 120.0 | Evaluated via deterministic offline heuristics. |
| `q05` | exact_keyword | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 120.0 | Evaluated via deterministic offline heuristics. |
| `q06` | exact_keyword | 1.0000 | 0.6000 | 0.7667 | 1.0000 | 1.0000 | 120.0 | Evaluated via deterministic offline heuristics. |
| `q07` | single_hop_semantic | 1.0000 | 0.8000 | 1.0000 | 1.0000 | 1.0000 | 120.0 | Evaluated via deterministic offline heuristics. |
| `q08` | single_hop_semantic | 1.0000 | 0.8182 | 1.0000 | 1.0000 | 1.0000 | 120.0 | Evaluated via deterministic offline heuristics. |
| `q09` | single_hop_semantic | 1.0000 | 0.6500 | 0.8833 | 1.0000 | 1.0000 | 120.0 | Evaluated via deterministic offline heuristics. |
| `q10` | single_hop_semantic | 1.0000 | 0.7000 | 0.8250 | 1.0000 | 1.0000 | 120.0 | Evaluated via deterministic offline heuristics. |
| `q11` | single_hop_semantic | 1.0000 | 0.7500 | 1.0000 | 1.0000 | 1.0000 | 120.0 | Evaluated via deterministic offline heuristics. |
| `q12` | single_hop_semantic | 1.0000 | 0.7000 | 1.0000 | 1.0000 | 1.0000 | 120.0 | Evaluated via deterministic offline heuristics. |
| `q13` | multi_hop_comparative | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 120.0 | Evaluated via deterministic offline heuristics. |
| `q14` | multi_hop_comparative | 1.0000 | 0.7500 | 1.0000 | 1.0000 | 1.0000 | 120.0 | Evaluated via deterministic offline heuristics. |
| `q15` | multi_hop_comparative | 1.0000 | 0.5182 | 0.6500 | 1.0000 | 1.0000 | 120.0 | Evaluated via deterministic offline heuristics. |
| `q16` | multi_hop_comparative | 1.0000 | 0.9545 | 1.0000 | 1.0000 | 1.0000 | 120.0 | Evaluated via deterministic offline heuristics. |
| `q17` | multi_hop_comparative | 1.0000 | 0.9500 | 1.0000 | 1.0000 | 1.0000 | 120.0 | Evaluated via deterministic offline heuristics. |
| `q18` | multi_hop_comparative | 1.0000 | 0.9231 | 1.0000 | 1.0000 | 1.0000 | 120.0 | Evaluated via deterministic offline heuristics. |
| `q19` | tabular_numerical | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 120.0 | Evaluated via deterministic offline heuristics. |
| `q20` | tabular_numerical | 1.0000 | 0.8500 | 1.0000 | 1.0000 | 1.0000 | 120.0 | Evaluated via deterministic offline heuristics. |
| `q21` | tabular_numerical | 1.0000 | 0.9167 | 1.0000 | 1.0000 | 1.0000 | 120.0 | Evaluated via deterministic offline heuristics. |
| `q22` | tabular_numerical | 1.0000 | 0.9091 | 1.0000 | 1.0000 | 1.0000 | 120.0 | Evaluated via deterministic offline heuristics. |
| `q23` | tabular_numerical | 1.0000 | 0.6091 | 0.6500 | 1.0000 | 1.0000 | 120.0 | Evaluated via deterministic offline heuristics. |
| `q24` | tabular_numerical | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 120.0 | Evaluated via deterministic offline heuristics. |
| `q25` | unanswerable_adversarial | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 25.0 | Evaluated via deterministic offline heuristics. |
| `q26` | unanswerable_adversarial | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 25.0 | Evaluated via deterministic offline heuristics. |
| `q27` | unanswerable_adversarial | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 25.0 | Evaluated via deterministic offline heuristics. |
| `q28` | unanswerable_adversarial | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 25.0 | Evaluated via deterministic offline heuristics. |

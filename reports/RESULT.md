# RAG evaluation results

## Run information

| Field | Value |
| --- | --- |
| Evaluation date | 2026-09-25 (Asia/Saigon) |
| Framework and version | Deterministic offline evaluator v1 (`src/evaluate.py`) |
| Evaluator | Token-overlap faithfulness/relevance/recall and source precision |
| Generator model | `local-extractive-v1` for both A/B configs; product LLM is configured as `mistral-small-latest` but is not used by this API-free evaluation |
| Embedding model | Effective model: deterministic `hashing-384` for both configs (`EMBEDDING_PROVIDER=hashing`) |
| Corpus version/commit | 8 official EU sources; repository `a23df34-dirty`; corpus SHA-256 `bea1d6c72ea4ba049ca6d0a34ad47b9e62c3cb6697b768d344d7e2fac2f1505f` |
| Golden dataset size | 15 grounded questions |
| `top_k` | 5 |
| Fallback threshold and calibration | Product threshold `0.18` cosine; fallback is held out of this A/B run so the only changed variable is dense-only versus hybrid + RRF |

The evaluator is deliberately API-free and reproducible. Faithfulness measures answer-token support by retrieved context; answer relevance is token F1 against the reference answer; context recall measures expected-context token coverage; context precision is the proportion of retrieved chunks from the annotated source. Scores are bounded from 0 to 1. A future benchmark can substitute RAGAS judges without changing the dataset or retrieval configurations.

## Configurations

- **Config A - dense-only:** `hashing-384`, cosine Chroma retrieval, deterministic dense candidate pool 10, `top_k=5`.
- **Config B - hybrid + RRF:** the exact same dense candidate pool plus 10 BM25 candidates, fused once with RRF (`k=60`), `top_k=5`.

Both configurations use the same corpus, chunking (`800/100`), golden dataset, answer extractor, metrics and `top_k`; only retrieval changes.

## Overall scores

| Metric | Config A | Config B | Delta B-A |
| --- | ---: | ---: | ---: |
| Faithfulness | 1.000 | 1.000 | +0.000 |
| Answer relevance | 0.177 | 0.282 | +0.105 |
| Context recall | 0.777 | 0.927 | +0.150 |
| Context precision | 0.453 | 0.573 | +0.120 |
| **Average** | **0.602** | **0.696** | **+0.094** |

## A/B comparison

- Cấu hình tốt hơn: **hybrid + RRF**, tăng 0.094 điểm trung bình (15.6% tương đối).
- Evidence: context recall tăng +0.150, context precision tăng +0.120 và answer relevance tăng +0.105. BM25 bổ sung tín hiệu từ khóa chính xác như “Model Documentation Form”, “Article 4” và “copyright policy”.
- Trade-off về latency/cost: lần chạy cuối đo được dense 84.7 ms/query và hybrid 38.5 ms/query. Config B chạy sau nên được hưởng OS/Chroma cache; số đo không chứng minh hybrid vốn nhanh hơn. Hybrid có thêm BM25 và RRF nhưng không phát sinh API cost.

## Worst performers

| # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
| --: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | Give three examples of prohibited AI practices. | Hybrid | 1.000 | 0.053 | 0.333 | 0.400 | retrieval | The list spans a long source chunk and generic query terms also match definition/literacy pages. |
| 2 | What obligations apply to high-risk AI systems before market placement? | Hybrid | 1.000 | 0.098 | 0.765 | 0.600 | retrieval/generation | The expected answer is a long enumeration; the extractive baseline selects only three sentences. |
| 3 | What does the AI literacy obligation require providers and deployers to do? | Hybrid | 1.000 | 0.105 | 1.000 | 1.000 | generation | Retrieval is correct, but the extractive generator includes more wording than the concise reference answer. |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| ---: | --- | --- | --- | --- |
| 1 | Enable Mistral generation and keep the citation-only prompt | Correct context is present but answer relevance is only 0.282 | Better synthesis of lists and purpose statements without changing retrieval | Re-run all 15 cases with the same retrieval results and an LLM-based relevance judge |
| 2 | Add heading-aware list chunks for long policy pages | Prohibited-practices recall is 0.333 | Preserve complete enumerations inside a chunk | Assert all prohibited items appear in top-5 context for case 3 |
| 3 | Calibrate dense threshold on a larger in/out-domain set | Current 0.18 threshold is based on this small corpus | More reliable fallback and refusal behavior | Add at least 10 out-of-domain questions and plot false accept/refusal rates |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| --- | --- | ---: | ---: | --- |
| Hybrid BM25 + RRF | Dense-only | +0.094 average | No API cost; small local CPU overhead | Keep hybrid as the default retrieval strategy. |

## Reproduction

```bash
python -m src.task4_chunking_indexing
python -m src.evaluate
pytest -q
```

Raw per-question metrics are printed as JSON by `src.evaluate`, making every aggregate above auditable.

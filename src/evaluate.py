"""Reproducible offline A/B evaluation for dense versus hybrid retrieval.

The four bounded [0, 1] metrics are lexical proxies so the lab can be rerun
without an evaluator API key.  For a publication-grade run, replace only the
scorer with RAGAS/LLM judges while preserving the same golden set and configs.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path

from .task10_generation import _local_grounded_answer, reorder_for_llm
from .task4_chunking_indexing import get_collection
from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf


ROOT = Path(__file__).parent.parent
GOLDEN_PATH = ROOT / "group_project" / "evaluation" / "golden_dataset.json"
STOP = {"the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "be", "for", "with", "what", "how"}


def tokens(text: str) -> set[str]:
    return {item for item in re.findall(r"(?u)\b\w\w+\b", text.lower()) if item not in STOP}


def recall(reference: str, candidate: str) -> float:
    expected = tokens(reference)
    return len(expected & tokens(candidate)) / len(expected) if expected else 1.0


def f1(reference: str, candidate: str) -> float:
    expected, actual = tokens(reference), tokens(candidate)
    if not expected or not actual:
        return 0.0
    overlap = len(expected & actual)
    precision, rec = overlap / len(actual), overlap / len(expected)
    return 2 * precision * rec / (precision + rec) if precision + rec else 0.0


def retrieve_config(question: str, top_k: int, hybrid: bool) -> list[dict]:
    # Query all vectors, then let semantic_search's deterministic (score, id)
    # ordering select the candidate pool.  This avoids HNSW tie jitter at the
    # top-2k boundary for the intentionally collision-prone hashing baseline.
    dense = semantic_search(question, top_k=get_collection().count())[: top_k * 2]
    if not hybrid:
        return dense[:top_k]
    sparse = lexical_search(question, top_k=top_k * 2)
    return rerank_rrf([dense, sparse], top_k=top_k)


def evaluate_configuration(dataset: list[dict], *, hybrid: bool, top_k: int = 5) -> dict:
    rows, latencies = [], []
    for case in dataset:
        started = time.perf_counter()
        chunks = retrieve_config(case["question"], top_k, hybrid)
        ordered = reorder_for_llm(chunks)
        answer = _local_grounded_answer(case["question"], ordered)
        latencies.append((time.perf_counter() - started) * 1000)
        context = " ".join(chunk["content"] for chunk in chunks)
        source_hits = [chunk["metadata"]["source"] == case.get("expected_source") for chunk in chunks]
        answer_plain = re.sub(r"\[S\d+\]", "", answer)
        rows.append({
            "question": case["question"],
            "faithfulness": recall(answer_plain, context),
            "answer_relevance": f1(case["expected_answer"], answer_plain),
            "context_recall": recall(case["expected_context"], context),
            "context_precision": sum(source_hits) / len(chunks) if chunks else 0.0,
            "retrieved_sources": [chunk["metadata"]["source"] for chunk in chunks],
        })
    metrics = {
        key: statistics.fmean(row[key] for row in rows)
        for key in ("faithfulness", "answer_relevance", "context_recall", "context_precision")
    }
    metrics["average"] = statistics.fmean(metrics.values())
    return {"metrics": metrics, "mean_latency_ms": statistics.fmean(latencies), "cases": rows}


def main() -> None:
    dataset = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    result = {
        "dataset_size": len(dataset),
        "top_k": 5,
        "dense_only": evaluate_configuration(dataset, hybrid=False),
        "hybrid_rrf": evaluate_configuration(dataset, hybrid=True),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

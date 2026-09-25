"""Hybrid retrieval with a dense-confidence vectorless fallback."""

from __future__ import annotations

import os

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD") or "0.18")
DEFAULT_TOP_K = 5


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    if not query.strip() or top_k <= 0:
        return []
    candidate_k = max(top_k * 2, top_k)
    dense = semantic_search(query, top_k=candidate_k)
    sparse = lexical_search(query, top_k=candidate_k)
    hybrid = rerank_rrf([dense, sparse], top_k=top_k) if use_reranking else dense[:top_k]
    best_dense_score = float(dense[0]["score"]) if dense else 0.0
    if best_dense_score < score_threshold:
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback
        except Exception:
            pass
    return hybrid[:top_k]


if __name__ == "__main__":
    for result in retrieve("What obligations apply to GPAI providers?", top_k=3):
        print(result["retrieval_method"], result["score"], result["metadata"]["source"])

"""BM25 retrieval over the exact same chunks used by dense retrieval."""

from __future__ import annotations

import re


CORPUS: list[dict] = []


def _tokenize(text: str) -> list[str]:
    return re.findall(r"(?u)\b\w\w+\b", text.lower())


def _corpus() -> list[dict]:
    if CORPUS:
        return CORPUS
    from .task4_chunking_indexing import chunk_documents, load_documents

    return chunk_documents(load_documents())


def build_bm25_index(corpus: list[dict]):
    from rank_bm25 import BM25Okapi

    return BM25Okapi([_tokenize(item["content"]) for item in corpus])


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    corpus = _corpus()
    if not query.strip() or top_k <= 0 or not corpus:
        return []
    scores = build_bm25_index(corpus).get_scores(_tokenize(query))
    ranked = sorted(range(len(corpus)), key=lambda index: (-float(scores[index]), corpus[index]["id"]))
    results = []
    for index in ranked:
        score = float(scores[index])
        item = corpus[index]
        results.append({**item, "score": score, "retrieval_method": "bm25"})
        if len(results) >= top_k:
            break
    return results


if __name__ == "__main__":
    for result in lexical_search("general-purpose AI transparency", top_k=3):
        print(result["score"], result["metadata"]["source"])

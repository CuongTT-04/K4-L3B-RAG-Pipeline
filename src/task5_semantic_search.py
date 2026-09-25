"""Dense cosine retrieval over the shared Chroma collection."""

from .task4_chunking_indexing import embed_texts, get_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    query = query.strip()
    if not query or top_k <= 0:
        return []
    collection = get_collection()
    count = collection.count() if hasattr(collection, "count") else top_k
    if count == 0:
        return []
    response = collection.query(
        query_embeddings=[embed_texts([query])[0]],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )
    seen, results = set(), []
    for item_id, content, metadata, distance in zip(
        response["ids"][0], response["documents"][0], response["metadatas"][0], response["distances"][0]
    ):
        if item_id in seen:
            continue
        seen.add(item_id)
        results.append({
            "id": item_id,
            "content": content,
            "score": float(max(-1.0, min(1.0, 1.0 - distance))),
            "metadata": metadata,
            "retrieval_method": "dense",
        })
    return sorted(results, key=lambda item: (-item["score"], item["id"]))[:top_k]


if __name__ == "__main__":
    for result in semantic_search("What is a general-purpose AI model?", top_k=3):
        print(result["score"], result["metadata"]["source"])

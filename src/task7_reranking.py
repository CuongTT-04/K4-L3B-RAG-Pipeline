"""Reciprocal Rank Fusion for heterogeneous retrieval scores."""


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    if top_k <= 0:
        return []
    if k < 0:
        raise ValueError("k must be non-negative")
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for ranked_list in ranked_lists:
        seen_in_list = set()
        for rank, item in enumerate(ranked_list, 1):
            item_id = item["id"]
            if item_id in seen_in_list:
                continue
            seen_in_list.add(item_id)
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            items.setdefault(item_id, item)
    ranked_ids = sorted(scores, key=lambda item_id: (-scores[item_id], item_id))
    return [
        {**items[item_id], "score": scores[item_id], "retrieval_method": "hybrid"}
        for item_id in ranked_ids[:top_k]
    ]


if __name__ == "__main__":
    print("RRF is ready; run pytest tests/test_contracts.py -q")

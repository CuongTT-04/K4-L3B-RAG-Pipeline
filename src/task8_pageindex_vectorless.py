"""Vectorless fallback using document sections and rank-based scores.

The local implementation is deterministic and works without credentials.  It mirrors
PageIndex's document-tree idea: Markdown headings define sections, and relevant
sections are selected lexically without embeddings.  The cache is a source manifest
so an optional remote PageIndex adapter can be added without repeated uploads.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from .task4_chunking_indexing import load_documents


load_dotenv()
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
ROOT = Path(__file__).parent.parent
STANDARDIZED_DIR = ROOT / "data" / "standardized"
CACHE_FILE = ROOT / "data" / "pageindex_manifest.json"


def upload_documents() -> None:
    """Cache stable document IDs; safe to rerun and never stores API keys."""
    documents = load_documents()
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        item["metadata"]["source"]: {
            "document_id": item["id"],
            "path": item["id"],
            "url": item["metadata"].get("url"),
        }
        for item in documents
    }
    CACHE_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Cached {len(manifest)} vectorless document IDs in {CACHE_FILE}")


def _tokens(text: str) -> set[str]:
    stop = {"the", "a", "an", "is", "are", "of", "to", "in", "and", "for", "what", "how", "when"}
    return {token for token in re.findall(r"(?u)\b\w\w+\b", text.lower()) if token not in stop}


def _sections(document: dict) -> list[dict]:
    parts = re.split(r"(?m)(?=^#{1,6}\s+)", document["content"])
    return [
        {
            "id": f"{document['id']}::pageindex-{index}",
            "content": part.strip(),
            "metadata": {**document["metadata"], "chunk_index": index},
        }
        for index, part in enumerate(parts)
        if part.strip()
    ]


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    if not query.strip() or top_k <= 0:
        return []
    query_tokens = _tokens(query)
    candidates = []
    for document in load_documents():
        for section in _sections(document):
            section_tokens = _tokens(section["content"])
            overlap = len(query_tokens & section_tokens)
            if overlap:
                score = overlap / math.sqrt(max(len(query_tokens), 1) * max(len(section_tokens), 1))
                candidates.append((score, section))
    candidates.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
    results = []
    for rank, (score, item) in enumerate(candidates[:top_k], 1):
        results.append({**item, "score": float(score or 1.0 / rank), "retrieval_method": "pageindex"})
    return results


if __name__ == "__main__":
    upload_documents()

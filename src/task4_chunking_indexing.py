"""Load, chunk, embed, and idempotently index the normalized corpus."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from .contracts import validate_document


load_dotenv()
ROOT = Path(__file__).parent.parent
STANDARDIZED_DIR = ROOT / "data" / "standardized"
CHROMA_DIR = ROOT / "chroma_db"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
CHUNKING_METHOD = "recursive"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "hashing").lower()
_DEFAULT_EMBEDDING_MODEL = (
    "hashing-384"
    if EMBEDDING_PROVIDER == "hashing"
    else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL)
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
COLLECTION_NAME = "rag_documents"


def _hashing_embeddings(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        vector = [0.0] * EMBEDDING_DIM
        tokens = re.findall(r"(?u)\b\w\w+\b", text.lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            number = int.from_bytes(digest, "little")
            vector[number % EMBEDDING_DIM] += 1.0 if number & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        vectors.append([value / norm for value in vector])
    return vectors


@lru_cache(maxsize=1)
def _sentence_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if EMBEDDING_PROVIDER == "hashing":
        return _hashing_embeddings(texts)
    if EMBEDDING_PROVIDER == "sentence_transformers":
        return _sentence_model().encode(texts, normalize_embeddings=True).tolist()
    if EMBEDDING_PROVIDER == "openai":
        from openai import OpenAI

        response = OpenAI().embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [item.embedding for item in response.data]
    if EMBEDDING_PROVIDER == "gemini":
        from google import genai

        response = genai.Client(api_key=os.getenv("GEMINI_API_KEY")).models.embed_content(model=EMBEDDING_MODEL, contents=texts)
        return [list(item.values) for item in response.embeddings]
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")


def get_collection():
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def _parse_markdown(path: Path) -> tuple[str, dict]:
    text = path.read_text(encoding="utf-8")
    metadata: dict[str, object] = {}
    if text.startswith("---\n") and "\n---\n" in text[4:]:
        raw_header, text = text[4:].split("\n---\n", 1)
        for line in raw_header.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                try:
                    metadata[key.strip()] = json.loads(value.strip())
                except json.JSONDecodeError:
                    metadata[key.strip()] = value.strip()
    return text.strip(), metadata


def load_documents() -> list[dict]:
    documents = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if path.name.startswith("."):
            continue
        content, header = _parse_markdown(path)
        relative = path.relative_to(STANDARDIZED_DIR).as_posix()
        doc_type = str(header.get("doc_type") or ("legal" if "legal" in path.parts else "news"))
        document = {
            "id": relative,
            "content": content,
            "metadata": {
                "source": str(header.get("source") or path.name),
                "title": str(header.get("title") or path.stem.replace("_", " ").title()),
                "doc_type": doc_type,
                "url": header.get("url") if isinstance(header.get("url"), str) else None,
            },
        }
        validate_document(document)
        documents.append(document)
    return documents


def _split_text(text: str) -> list[str]:
    if len(text) <= CHUNK_SIZE:
        return [text.strip()] if text.strip() else []
    chunks, start = [], 0
    separators = ("\n\n", "\n", ". ", " ")
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            floor = start + max(CHUNK_SIZE // 2, 1)
            candidates = [text.rfind(separator, floor, end) for separator in separators]
            boundary = max(candidates)
            if boundary > start:
                end = boundary + (2 if text[boundary:boundary + 2] == ". " else 0)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    chunks = []
    for document in documents:
        validate_document(document)
        for index, text in enumerate(_split_text(document["content"])):
            chunk = {
                "id": f"{document['id']}::chunk-{index}",
                "content": text,
                "metadata": {**document["metadata"], "chunk_index": index},
            }
            validate_document(chunk, require_chunk=True)
            chunks.append(chunk)
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    vectors = embed_texts([chunk["content"] for chunk in chunks])
    if len(vectors) != len(chunks):
        raise ValueError("embedding provider returned the wrong number of vectors")
    return [{**chunk, "embedding": vector} for chunk, vector in zip(chunks, vectors)]


def index_to_vectorstore(chunks: list[dict]) -> None:
    if not chunks:
        raise ValueError("No chunks to index. Run data collection and conversion first.")
    collection = get_collection()
    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )


def run_pipeline() -> None:
    chunks = chunk_documents(load_documents())
    index_to_vectorstore(embed_chunks(chunks))
    print(f"Indexed {len(chunks)} chunks with {EMBEDDING_PROVIDER}/{EMBEDDING_MODEL}")


if __name__ == "__main__":
    run_pipeline()

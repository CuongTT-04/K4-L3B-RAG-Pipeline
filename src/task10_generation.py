"""Grounded answer generation with verifiable source labels."""

from __future__ import annotations

import os
import re

from dotenv import load_dotenv

from .contracts import validate_generation_result
from .task9_retrieval_pipeline import retrieve


load_dotenv()
TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.2
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "")
SAFE_REFUSAL = "Tôi không thể xác minh thông tin này từ các nguồn hiện có."
QUERY_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "and",
    "or", "for", "with", "what", "which", "who", "when", "where", "how", "does",
    "do", "did", "must", "should", "can", "could", "would", "give",
}
SYSTEM_PROMPT = """Answer only from the supplied context. Cite factual sentences with [S1], [S2], etc. Never invent a citation. If the evidence is insufficient, answer exactly: Tôi không thể xác minh thông tin này từ các nguồn hiện có."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    if len(chunks) <= 2:
        return list(chunks)
    reordered = [None] * len(chunks)
    left, right = 0, len(chunks) - 1
    for index, chunk in enumerate(chunks):
        if index % 2 == 0:
            reordered[left] = chunk
            left += 1
        else:
            reordered[right] = chunk
            right -= 1
    return reordered


def format_context(chunks: list[dict]) -> str:
    parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk["metadata"]
        citation_index = chunk.get("_citation_index", index)
        parts.append(
            f"[S{citation_index} | Title: {metadata['title']} | Source: {metadata['source']} | URL: {metadata.get('url') or 'local'}]\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    if LLM_PROVIDER == "mistral":
        import requests

        model = LLM_MODEL or "mistral-small-latest"
        base_url = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1").rstrip("/")
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('MISTRAL_API_KEY', '')}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    if LLM_PROVIDER == "openai":
        from openai import OpenAI

        model = LLM_MODEL or "gpt-4.1-mini"
        response = OpenAI(api_key=os.getenv("OPENAI_API_KEY")).responses.create(
            model=model,
            instructions=system_prompt,
            input=user_message,
            temperature=TEMPERATURE,
        )
        return response.output_text.strip()
    if LLM_PROVIDER == "gemini":
        from google import genai
        from google.genai import types

        model = LLM_MODEL or "gemini-2.0-flash"
        response = genai.Client(api_key=os.getenv("GEMINI_API_KEY")).models.generate_content(
            model=model,
            contents=user_message,
            config=types.GenerateContentConfig(system_instruction=system_prompt, temperature=TEMPERATURE, top_p=TOP_P),
        )
        return (response.text or "").strip()
    if LLM_PROVIDER == "anthropic":
        from anthropic import Anthropic

        model = LLM_MODEL or "claude-3-5-haiku-latest"
        response = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")).messages.create(
            model=model,
            max_tokens=800,
            temperature=TEMPERATURE,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return "".join(block.text for block in response.content if hasattr(block, "text")).strip()
    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


def _local_grounded_answer(query: str, chunks: list[dict]) -> str:
    query_terms = set(re.findall(r"(?u)\b\w\w+\b", query.lower())) - QUERY_STOPWORDS
    sentences = []
    for source_index, chunk in enumerate(chunks, 1):
        citation_index = chunk.get("_citation_index", source_index)
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", chunk["content"]):
            clean = sentence.strip(" #-\t")
            if 35 <= len(clean) <= 420:
                overlap = len(query_terms & set(re.findall(r"(?u)\b\w\w+\b", clean.lower())))
                if overlap:
                    sentences.append((overlap, citation_index, clean))
    sentences.sort(key=lambda item: (-item[0], item[1], len(item[2])))
    chosen, seen = [], set()
    for _, source_index, sentence in sentences:
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        chosen.append(f"{sentence} [S{source_index}]")
        if len(chosen) == 3:
            break
    return "\n\n".join(chosen) if chosen else SAFE_REFUSAL


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    chunks = retrieve(query, top_k=top_k)
    query_terms = set(re.findall(r"(?u)\b\w\w+\b", query.lower())) - QUERY_STOPWORDS
    evidence_terms = set(re.findall(r"(?u)\b\w\w+\b", " ".join(chunk["content"] for chunk in chunks).lower()))
    required_overlap = min(2, len(query_terms))
    if not chunks or len(query_terms & evidence_terms) < required_overlap:
        result = {"answer": SAFE_REFUSAL, "sources": [], "retrieval_source": "none"}
        validate_generation_result(result)
        return result
    reordered = reorder_for_llm([
        {**chunk, "_citation_index": index} for index, chunk in enumerate(chunks, 1)
    ])
    configured = {
        "mistral": bool(os.getenv("MISTRAL_API_KEY")),
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "gemini": bool(os.getenv("GEMINI_API_KEY")),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
    }.get(LLM_PROVIDER, False)
    try:
        answer = call_llm(SYSTEM_PROMPT, f"Context:\n{format_context(reordered)}\n\nQuestion: {query}") if configured else _local_grounded_answer(query, reordered)
    except Exception:
        answer = _local_grounded_answer(query, reordered)
    if not answer.strip():
        answer = SAFE_REFUSAL
    # Sources follow the same order as citation labels in the context/answer.
    retrieval_source = "pageindex" if chunks[0]["retrieval_method"] == "pageindex" else "hybrid"
    result = {"answer": answer, "sources": chunks, "retrieval_source": retrieval_source}
    validate_generation_result(result)
    return result


if __name__ == "__main__":
    print(generate_with_citation("What is a general-purpose AI model?"))

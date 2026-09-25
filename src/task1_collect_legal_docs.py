"""Collect the official policy PDFs used by the EU AI Act assistant."""

from __future__ import annotations

from pathlib import Path

import requests


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

DOCUMENT_SOURCES = {
    "ai_system_definition_guidelines.pdf": "https://ec.europa.eu/newsroom/dae/redirection/document/112455",
    "gpai_code_transparency.pdf": "https://ec.europa.eu/newsroom/dae/redirection/document/118120",
    "gpai_code_copyright.pdf": "https://ec.europa.eu/newsroom/dae/redirection/document/118115",
}


def setup_directory() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_documents() -> None:
    """Download missing PDFs atomically and validate their signatures."""
    setup_directory()
    for filename, url in DOCUMENT_SOURCES.items():
        output = DATA_DIR / filename
        if output.exists() and output.stat().st_size > 1024:
            print(f"Exists: {output}")
            continue
        response = requests.get(
            url,
            timeout=60,
            headers={"User-Agent": "K4-RAG-Lab/1.0 (educational project)"},
        )
        response.raise_for_status()
        payload = response.content
        if len(payload) <= 1024 or not payload.startswith(b"%PDF"):
            raise ValueError(f"{url} did not return a valid PDF")
        temporary = output.with_suffix(".pdf.part")
        temporary.write_bytes(payload)
        temporary.replace(output)
        print(f"Saved: {output} ({len(payload):,} bytes)")


if __name__ == "__main__":
    download_documents()

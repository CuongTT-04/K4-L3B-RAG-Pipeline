"""Normalize landing PDFs and article JSON files to provenance-rich Markdown."""

from __future__ import annotations

import json
from pathlib import Path

from .task1_collect_legal_docs import DOCUMENT_SOURCES


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _front_matter(*, title: str, source: str, doc_type: str, url: str | None) -> str:
    return (
        "---\n"
        f"title: {json.dumps(title, ensure_ascii=False)}\n"
        f"source: {json.dumps(source, ensure_ascii=False)}\n"
        f"doc_type: {doc_type}\n"
        f"url: {json.dumps(url, ensure_ascii=False)}\n"
        "---\n\n"
    )


def _extract_pdf(path: Path) -> str:
    try:
        from markitdown import MarkItDown

        text = MarkItDown().convert(str(path)).text_content
    except Exception:
        from pypdf import PdfReader

        text = "\n\n".join((page.extract_text() or "") for page in PdfReader(path).pages)
    text = text.strip()
    if len(text) < 200:
        raise ValueError(f"Could not extract sufficient text from {path}")
    return text


def convert_legal_docs() -> None:
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted((LANDING_DIR / "legal").iterdir()):
        if path.suffix.lower() not in {".pdf", ".doc", ".docx"}:
            continue
        title = path.stem.replace("_", " ").title()
        header = _front_matter(title=title, source=path.name, doc_type="legal", url=DOCUMENT_SOURCES.get(path.name))
        (output_dir / f"{path.stem}.md").write_text(header + _extract_pdf(path) + "\n", encoding="utf-8")


def convert_news_articles() -> None:
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted((LANDING_DIR / "news").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for field in ("url", "title", "date_crawled", "content_markdown"):
            if not str(data.get(field, "")).strip():
                raise ValueError(f"{path}: missing {field}")
        header = _front_matter(title=data["title"], source=path.name, doc_type="news", url=data["url"])
        provenance = f"Crawled: {data['date_crawled']}\n\n"
        (output_dir / f"{path.stem}.md").write_text(header + provenance + data["content_markdown"].strip() + "\n", encoding="utf-8")


def convert_all() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    convert_legal_docs()
    convert_news_articles()
    print(f"Saved Markdown to: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()

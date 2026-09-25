"""Crawl official European Commission pages for the EU AI Act corpus."""

from __future__ import annotations

import asyncio
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import requests


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"
ARTICLE_URLS = [
    "https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai",
    "https://digital-strategy.ec.europa.eu/en/policies/ai-talent-skills-and-literacy",
    "https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai",
    "https://digital-strategy.ec.europa.eu/en/library/commission-publishes-guidelines-ai-system-definition-facilitate-first-ai-acts-rules-application",
    "https://digital-strategy.ec.europa.eu/en/news/general-purpose-ai-code-practice-now-available",
]


def _html_to_markdown(raw_html: str) -> tuple[str, str]:
    title_match = re.search(r"<title[^>]*>(.*?)</title>", raw_html, re.I | re.S)
    title = html.unescape(re.sub(r"\s+", " ", title_match.group(1))).strip() if title_match else "Untitled"
    body = re.sub(r"<(script|style|nav|footer|header)[^>]*>.*?</\1>", " ", raw_html, flags=re.I | re.S)
    body = re.sub(r"<h([1-6])[^>]*>(.*?)</h\1>", lambda m: f"\n\n{'#' * int(m.group(1))} {m.group(2)}\n\n", body, flags=re.I | re.S)
    body = re.sub(r"<li[^>]*>(.*?)</li>", r"\n- \1", body, flags=re.I | re.S)
    body = re.sub(r"<(p|div|section|article|br)[^>]*>", "\n", body, flags=re.I)
    body = re.sub(r"<[^>]+>", " ", body)
    body = html.unescape(body).replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in body.splitlines()]
    content = "\n".join(line for line in lines if line)
    content = re.sub(r"\n{3,}", "\n\n", content).strip()
    if len(content) < 200:
        raise ValueError("extracted article is unexpectedly short")
    return title, content


async def crawl_article(url: str) -> dict:
    def fetch() -> str:
        response = requests.get(url, timeout=45, headers={"User-Agent": "K4-RAG-Lab/1.0 (educational project)"})
        response.raise_for_status()
        return response.text

    raw_html = await asyncio.to_thread(fetch)
    title, content = _html_to_markdown(raw_html)
    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "content_markdown": content,
    }


async def crawl_all() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results = await asyncio.gather(*(crawl_article(url) for url in ARTICLE_URLS), return_exceptions=True)
    failures = []
    for index, (url, article) in enumerate(zip(ARTICLE_URLS, results), 1):
        if isinstance(article, Exception):
            failures.append(f"{url}: {article}")
            continue
        output = DATA_DIR / f"article_{index:02d}.json"
        output.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {output}")
    if failures:
        raise RuntimeError("Failed articles:\n" + "\n".join(failures))


if __name__ == "__main__":
    asyncio.run(crawl_all())

from typing import List, Dict, Any, Set
from datetime import datetime, timedelta, timezone
import re

import trafilatura

from api.claim_verification.retrieval.news_client import TheNewsAPIClient


HIGH_TRUST_DOMAINS: Set[str] = {
    "apnews.com",
    "reuters.com",
    "bbc.com",
    "bbc.co.uk",
    "nytimes.com",
    "wsj.com",
    "theguardian.com",
    "npr.org",
    "cbc.ca",
    "cbsnews.com",
    "abcnews.go.com",
    "washingtonpost.com",
    "politico.com",
    "bloomberg.com",
    "ft.com",
    "rte.ie",
    "theverge.com",
    "techcrunch.com",
    "wired.com",
    "arstechnica.com",
    "engadget.com",
    "venturebeat.com",
    "axios.com",
    "forbes.com",
    "cnbc.com",
    "businessinsider.com",
    "time.com",
}


def clean_text(text: str) -> str:
    text = text or ""
    return re.sub(r"\s+", " ", text).strip()


def simple_chunk(text: str, max_words: int = 120, overlap: int = 30) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + max_words)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - overlap)

    return chunks


def get_published_after_iso(hours_back: int = 72) -> str:
    dt = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def dedupe_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []

    for a in articles:
        key = (
            (a.get("url") or "").strip().lower(),
            (a.get("title") or "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(a)

    return deduped


def fetch_full_article_text(url: str) -> str:
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return ""
        extracted = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )
        return clean_text(extracted or "")
    except Exception:
        return ""


def article_to_passages(article: Dict[str, Any], max_chunks: int = 8) -> List[Dict[str, Any]]:
    title = clean_text(article.get("title", ""))
    description = clean_text(article.get("description", ""))
    snippet = clean_text(article.get("content", ""))
    full_text = fetch_full_article_text(article.get("url", ""))

    combined = "\n".join(x for x in [title, description, snippet, full_text] if x).strip()
    if not combined:
        return []

    chunks = simple_chunk(combined, max_words=120, overlap=30)[:max_chunks]

    passages: List[Dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        passages.append(
            {
                "docid": f"web:{article.get('provider', 'news')}:{article.get('url', 'unknown')}#{i:05d}",
                "contents": chunk,
                "score": 0.0,
                "source_type": "web",
                "provider": article.get("provider"),
                "url": article.get("url"),
                "domain": article.get("domain"),
                "published_at": article.get("published_at"),
                "title": article.get("title"),
                "news_source": article.get("source"),
            }
        )

    return passages


def live_news_passages(
    query: str,
    *,
    language: str = "en",
    limit: int = 20,
    hours_back: int = 168,
    trusted_only: bool = True,
    max_chunks_per_article: int = 8,
) -> List[Dict[str, Any]]:
    client = TheNewsAPIClient()

    domains = sorted(HIGH_TRUST_DOMAINS) if trusted_only else None
    published_after = get_published_after_iso(hours_back=hours_back)

    articles = client.search(
        query,
        language=language,
        limit=limit,
        published_after=published_after,
        domains=domains,
    )

    english_articles = [a for a in articles if (a.get("language") or "").lower() == "en"]
    if english_articles:
        articles = english_articles

    articles = dedupe_articles(articles)

    passages: List[Dict[str, Any]] = []
    for article in articles:
        passages.extend(article_to_passages(article, max_chunks=max_chunks_per_article))

    return passages
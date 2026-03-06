import os
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

import requests


class NewsAPIError(Exception):
    pass


def _domain_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


class TheNewsAPIClient:
    BASE_URL = "https://api.thenewsapi.com/v1/news/all"

    def __init__(self, api_token: Optional[str] = None, timeout: int = 20):
        self.api_token = api_token or os.getenv("THENEWSAPI_TOKEN")
        self.timeout = timeout
        if not self.api_token:
            raise NewsAPIError("Missing THENEWSAPI_TOKEN environment variable.")

    def search(
        self,
        query: str,
        *,
        language: str = "en",
        limit: int = 10,
        published_after: Optional[str] = None,
        domains: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        params = {
            "api_token": self.api_token,
            "search": query,
            "language": language,
            "limit": min(max(limit, 1), 50),
            "sort": "relevance_score",
        }

        if published_after:
            params["published_after"] = published_after

        if domains:
            params["domains"] = ",".join(domains)

        resp = requests.get(self.BASE_URL, params=params, timeout=self.timeout)
        if resp.status_code != 200:
            raise NewsAPIError(f"TheNewsAPI error {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        articles = data.get("data", [])

        normalized: List[Dict[str, Any]] = []
        for a in articles:
            url = a.get("url") or ""
            normalized.append(
                {
                    "provider": "thenewsapi",
                    "title": a.get("title") or "",
                    "description": a.get("description") or "",
                    "content": a.get("snippet") or a.get("description") or "",
                    "url": url,
                    "domain": _domain_from_url(url),
                    "published_at": a.get("published_at"),
                    "source": a.get("source") or "",
                    "language": a.get("language") or language,
                    "categories": a.get("categories") or [],
                    "relevance_score": a.get("relevance_score"),
                }
            )

        return normalized
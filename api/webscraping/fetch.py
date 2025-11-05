from dataclasses import dataclass
from typing import Optional
import time
import re
import urllib.parse as up
from curl_cffi import requests as cureq

DEFAULT_TIMEOUT = 10
ALLOWED_SCHEMES = {"http", "https"}

class FetchResult:
    url: str
    final_url: str
    status: int
    html: str
    elapsed_ms: int
    error: Optional[str] = None


def canonicalize_url(url: str):
    u = up.urlsplit(url.strip())
    if u.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError("Unsupported URL scheme")
    netloc = u.netloc.lower()
    path = re.sub(r"/+", "/", u.path or "/")
    return up.urlunsplit((u.scheme.lower(), netloc, path, u.query, ""))


def fetch(url: str, timeout: int = DEFAULT_TIMEOUT):
    start = time.time()
    try:
        url_c = canonicalize_url(url)
        print(f"Fetching: {url_c}")
    except Exception as e:
        return FetchResult(url=url, final_url=url, status=0, html="", elapsed_ms=0, error=str(e))
    
    try:
        resp = cureq.get(url_c, impersonate="chrome", timeout=timeout, allow_redirects=True)
        html = resp.text if resp.ok else ""
        return FetchResult(
            url=url_c,
            final_url=str(resp.url),
            status=resp.status_code,
            html=html,
            elapsed_ms=int((time.time() - start) * 1000),
            error=None if resp.ok else f"HTTP {resp.status_code}",
        )
    except Exception as e:
        return FetchResult(
            url=url_c,
            final_url=url_c,
            status=0,
            html="",
            elapsed_ms=int((time.time() - start) * 1000),
            error=str(e),
        )


if __name__ == "__main__":
    # quick manual test when running module directly
    try:
        from trafilatura import bare_extraction
    except Exception:
        bare_extraction = None

    test_urls = [
        "https://www.forbes.com/sites/technology/article/how-to-know-if-your-phone-is-hacked/",
        "https://www.forbes.com/sites/tylerroush/2025/10/10/trump-accuses-china-of-sinister-order-curbing-rare-earth-exports-threatens-massive-tariffs/",
    ]
    print("Testing fetch.py")
    for u in test_urls:
        res = fetch(u)
        if bare_extraction and res.html:
            data = bare_extraction(res.html, favor_recall=True, include_comments=False, with_metadata=True, as_dict=True)
            print(data.get("title"))
            print(data.get("author"))
            print((data.get("text") or "")[:200])
        else:
            print(f"Fetched {res.status} bytes={len(res.html)} error={res.error}")

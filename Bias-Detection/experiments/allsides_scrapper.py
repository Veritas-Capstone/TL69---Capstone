"""
AllSides News Scraper
=====================
Scrapes headline roundups from https://www.allsides.com/unbiased-balanced-news
and collects article URLs for each perspective (Left, Center, Right),
then extracts the title, body text, bias, and original source URL from each
article page.

Install ONE of these (tried in order, pick whichever works on your machine):

    Option A (recommended — impersonates browser TLS fingerprint):
        pip install curl_cffi beautifulsoup4 lxml

    Option B (Cloudflare bypass):
        pip install cloudscraper beautifulsoup4 lxml

    Option C (full headless browser — nuclear option, always works):
        pip install playwright beautifulsoup4 lxml
        playwright install chromium
"""

import json
import time
import sys
import re
from dataclasses import dataclass, field, asdict
from typing import Optional
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://www.allsides.com"
NEWS_URL = f"{BASE_URL}/unbiased-balanced-news"

REQUEST_DELAY = 1.5  # polite delay between requests (seconds)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ArticleSide:
    bias: str                          # "left", "center", or "right"
    source: str                        # e.g. "Washington Times"
    headline: str                      # headline text from roundup card
    url: str                           # AllSides article page URL
    title: Optional[str] = None        # <h1> from article page
    body: Optional[str] = None         # body text from article page
    bias_detail: Optional[str] = None  # e.g. "Lean Right", "Left", "Center"
    original_url: Optional[str] = None # external link (e.g. CNN article)


@dataclass
class HeadlineRoundup:
    main_headline: str
    main_summary: str
    sides: list[ArticleSide] = field(default_factory=list)


# ---------------------------------------------------------------------------
# HTTP fetcher — tries multiple backends to beat anti-bot protection
# ---------------------------------------------------------------------------

class Fetcher:
    """
    Tries three HTTP strategies in order:
      1. curl_cffi  — impersonates Chrome's TLS fingerprint
      2. cloudscraper — solves Cloudflare JS challenges
      3. playwright  — launches a real headless Chromium browser
    """

    def __init__(self):
        self._backend = None
        self._session = None
        self._browser = None
        self._init_backend()

    def _init_backend(self):
        # --- Try curl_cffi first ---
        try:
            from curl_cffi import requests as cffi_requests
            # Quick test to see if the import actually works
            self._session = cffi_requests.Session(impersonate="chrome")
            self._backend = "curl_cffi"
            print(f"[Fetcher] Using curl_cffi (Chrome TLS impersonation)")
            return
        except ImportError:
            pass

        # --- Try cloudscraper ---
        try:
            import cloudscraper
            self._session = cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "windows", "desktop": True}
            )
            self._backend = "cloudscraper"
            print(f"[Fetcher] Using cloudscraper")
            return
        except ImportError:
            pass

        # --- Try playwright ---
        try:
            from playwright.sync_api import sync_playwright
            pw = sync_playwright().start()
            self._browser = pw.chromium.launch(headless=True)
            self._session = self._browser.new_page()
            self._session.set_extra_http_headers({
                "Accept-Language": "en-US,en;q=0.9",
            })
            self._backend = "playwright"
            print(f"[Fetcher] Using playwright (headless Chromium)")
            return
        except ImportError:
            pass

        # --- Bare requests as last resort ---
        import requests
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
        })
        self._backend = "requests"
        print(f"[Fetcher] Using bare requests (may get 403!)")

    def fetch(self, url: str) -> Optional[BeautifulSoup]:
        try:
            if self._backend == "playwright":
                self._session.goto(url, wait_until="domcontentloaded", timeout=20000)
                html = self._session.content()
            elif self._backend == "curl_cffi":
                resp = self._session.get(url, timeout=20)
                resp.raise_for_status()
                html = resp.text
            else:
                # cloudscraper or bare requests — same API
                resp = self._session.get(url, timeout=20)
                resp.raise_for_status()
                html = resp.text

            return BeautifulSoup(html, "lxml")

        except Exception as e:
            print(f"  [ERROR] {url}: {e}", file=sys.stderr)
            return None

    def close(self):
        if self._backend == "playwright" and self._browser:
            self._browser.close()


# ---------------------------------------------------------------------------
# Step 1 — Scrape the main balanced-news page
# ---------------------------------------------------------------------------

def scrape_roundups(soup: BeautifulSoup) -> list[HeadlineRoundup]:
    """
    Parse the balanced-news landing page.

    Each headline roundup shows 3 article cards (right / center / left).
    The cards are <a href="/news/..."> links inside containers with bias
    classes like class="news-item flex-1 left".
    """
    roundups: list[HeadlineRoundup] = []

    # Grab all /news/ article links
    news_links = soup.select("a[href^='/news/']")
    if not news_links:
        news_links = soup.select("a[href*='/news/2']")

    sides: list[ArticleSide] = []
    seen_urls: set[str] = set()

    for link in news_links:
        href = link.get("href", "")
        if not href or href == "#":
            continue

        headline = link.get_text(strip=True)
        if not headline or len(headline) < 10:
            continue

        full_url = href if href.startswith("http") else BASE_URL + href
        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)

        bias, source = _detect_bias_and_source_from_card(link)

        sides.append(ArticleSide(
            bias=bias,
            source=source,
            headline=headline,
            url=full_url,
        ))

    # Group every 3 consecutive links into a roundup
    for i in range(0, len(sides), 3):
        group = sides[i : i + 3]
        if not group:
            break
        roundups.append(HeadlineRoundup(
            main_headline=group[0].headline,
            main_summary="",
            sides=group,
        ))

    return roundups


def _detect_bias_and_source_from_card(link_tag) -> tuple[str, str]:
    """Detect bias + source from a news card on the landing page."""
    bias = "unknown"
    source = ""

    # The card container has classes like: "news-item flex-1 left"
    for parent in link_tag.parents:
        if parent.name is None:
            break
        classes = " ".join(parent.get("class", []))

        if re.search(r"\bright\b", classes) and "left" not in classes:
            bias = "right"; break
        elif re.search(r"\bleft\b", classes) and "right" not in classes:
            bias = "left"; break
        elif re.search(r"\bcenter\b", classes):
            bias = "center"; break

    # Source name from nearby <p class="news-source-title">
    container = link_tag.find_parent("div", class_="news-item") or link_tag.find_parent("div")
    if container:
        src_tag = container.select_one(".news-source-title")
        if src_tag:
            source = src_tag.get_text(strip=True)

        if bias == "unknown":
            img = container.select_one("img[src*='bias-']")
            if img:
                s = img.get("src", "")
                if "right" in s: bias = "right"
                elif "left" in s: bias = "left"
                elif "center" in s: bias = "center"

    return bias, source


# ---------------------------------------------------------------------------
# Step 2 — Scrape an individual article page
# ---------------------------------------------------------------------------

def scrape_article(fetcher: Fetcher, article: ArticleSide) -> None:
    """
    Visit the AllSides article page and fill in:
      title, body, bias, bias_detail, source, original_url

    Actual HTML structure (verified against live site):

        <h1 class="my-2 font-bold text-xlg">Title</h1>

        <div class="my-2 text-lg font-bold border-b-4 text-color-left">
          From the Left
        </div>

        <div class="text-2xl font-bold">CNN Digital</div>

        <a href="/media-bias/left-center" class="text-color-left">Lean Left</a>

        <div class="body">
          <p>Article excerpt paragraph 1...</p>
          <p>Paragraph 2...</p>
        </div>

        <a id="Read-Full-Story--..." href="https://www.cnn.com/...">
          Read Full Story
        </a>
    """
    soup = fetcher.fetch(article.url)
    if soup is None:
        return

    # --- Title ---
    h1 = soup.select_one("h1")
    if h1:
        article.title = h1.get_text(strip=True)

    # --- Bias: "From the Left / Center / Right" banner ---
    for div in soup.find_all("div"):
        text = div.get_text(strip=True)
        if re.match(r"^From the (Left|Center|Right)$", text, re.IGNORECASE):
            if "Right" in text:   article.bias = "right"
            elif "Center" in text: article.bias = "center"
            elif "Left" in text:   article.bias = "left"
            break

    # --- Bias detail: "Lean Left", "Right", etc. ---
    for a in soup.select("a[href*='/media-bias/']"):
        text = a.get_text(strip=True)
        if text in ("Left", "Lean Left", "Center", "Lean Right", "Right"):
            article.bias_detail = text
            break

    # --- Source name ---
    src_div = soup.select_one("div.text-2xl.font-bold")
    if src_div:
        article.source = src_div.get_text(strip=True)
    if not article.source:
        src_link = soup.select_one("a[href*='/news-source/']")
        if src_link:
            article.source = src_link.get_text(strip=True)

    # --- Body ---
    body_div = soup.select_one("div.body")
    if body_div:
        paragraphs = body_div.find_all("p")
        article.body = "\n\n".join(
            p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
        )

    # Fallback: grab long <p> tags from the article
    if not article.body:
        article_tag = soup.select_one("article")
        if article_tag:
            parts = [
                p.get_text(strip=True)
                for p in article_tag.find_all("p")
                if len(p.get_text(strip=True)) > 60
            ]
            if parts:
                article.body = "\n\n".join(parts)

    # --- Original source URL ---
    read_full = soup.select_one("a[id^='Read-Full-Story']")
    if read_full:
        article.original_url = read_full.get("href")
    else:
        for a in soup.select("a[target='_blank']"):
            href = a.get("href", "")
            if href.startswith("http") and "allsides.com" not in href:
                article.original_url = href
                break


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    max_roundups: int = 14,
    scrape_articles: bool = True,
    output_file: str = "allsides_data_2_21.json",
):
    fetcher = Fetcher()

    print(f"\nFetching: {NEWS_URL}")
    soup = fetcher.fetch(NEWS_URL)
    if soup is None:
        print("Could not fetch the main AllSides page. Exiting.")
        fetcher.close()
        sys.exit(1)

    roundups = scrape_roundups(soup)
    if max_roundups:
        roundups = roundups[:max_roundups]

    print(f"Found {len(roundups)} headline roundup(s).\n")

    for i, roundup in enumerate(roundups, 1):
        print(f"[{i}] {roundup.main_headline}")
        for side in roundup.sides:
            print(f"    {side.bias:>8}: {side.headline}")
            print(f"             → {side.url}")

            if scrape_articles:
                time.sleep(REQUEST_DELAY)
                scrape_article(fetcher, side)
                if side.title:
                    print(f"             title: {side.title[:80]}")
                if side.bias_detail:
                    print(f"             bias:  {side.bias} ({side.bias_detail})")
                if side.source:
                    print(f"             src:   {side.source}")
                if side.body:
                    preview = side.body[:100].replace("\n", " ")
                    print(f"             body:  {preview}…")
                if side.original_url:
                    print(f"             orig:  {side.original_url[:80]}")
        print()

    # Save
    data = [asdict(r) for r in roundups]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(roundups)} roundup(s) to {output_file}")

    fetcher.close()
    return roundups


if __name__ == "__main__":
    run()
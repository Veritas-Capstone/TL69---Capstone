"""
AllSides News Scraper
Collects article URLs for each perspective (Left, Center, Right), then extracts
the title, body text, bias detail, and original source URL from each article page.
"""

import json
import time
import sys
import re
from dataclasses import dataclass, field, asdict
from typing import Optional
from bs4 import BeautifulSoup

BASE_URL = "https://www.allsides.com"
NEWS_URL = f"{BASE_URL}/unbiased-balanced-news"
REQUEST_DELAY = 1.5  # seconds between requests
OUTPUT_FILE = "allsides_data.json"
MAX_ROUNDUPS = 14


@dataclass
class ArticleSide:
    bias: str
    source: str
    headline: str
    url: str
    title: Optional[str] = None
    body: Optional[str] = None
    bias_detail: Optional[str] = None
    original_url: Optional[str] = None


@dataclass
class HeadlineRoundup:
    main_headline: str
    main_summary: str
    sides: list[ArticleSide] = field(default_factory=list)


# http fetcher — uses curl_cffi to impersonate Chrome TLS fingerprint
# requires: pip install curl_cffi
class Fetcher:
    def __init__(self):
        from curl_cffi import requests as cffi_requests
        self._session = cffi_requests.Session(impersonate="chrome")
        print("[Fetcher] Using curl_cffi")

    def fetch(self, url):
        try:
            resp = self._session.get(url, timeout=20)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "lxml")
        except Exception as e:
            print(f"  [ERROR] {url}: {e}", file=sys.stderr)
            return None


# parse the balanced-news landing page into roundups
def scrape_roundups(soup):
    roundups = []
    news_links = soup.select("a[href^='/news/']")
    if not news_links:
        news_links = soup.select("a[href*='/news/2']")

    sides = []
    seen_urls = set()

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

        bias, source = _detect_bias_and_source(link)
        sides.append(ArticleSide(bias=bias, source=source, headline=headline, url=full_url))

    # group every 3 links into a roundup (left/center/right)
    for i in range(0, len(sides), 3):
        group = sides[i:i+3]
        if not group:
            break
        roundups.append(HeadlineRoundup(
            main_headline=group[0].headline, main_summary="", sides=group))

    return roundups


def _detect_bias_and_source(link_tag):
    bias = "unknown"
    source = ""

    # card container has classes like "news-item flex-1 left"
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


# scrape an individual article page to get title, body, bias detail, original url
def scrape_article(fetcher, article):
    soup = fetcher.fetch(article.url)
    if soup is None:
        return

    # title
    h1 = soup.select_one("h1")
    if h1:
        article.title = h1.get_text(strip=True)

    # bias direction from "From the Left/Center/Right" banner
    for div in soup.find_all("div"):
        text = div.get_text(strip=True)
        if re.match(r"^From the (Left|Center|Right)$", text, re.IGNORECASE):
            if "Right" in text: article.bias = "right"
            elif "Center" in text: article.bias = "center"
            elif "Left" in text: article.bias = "left"
            break

    # bias detail: "Lean Left", "Right", etc
    for a in soup.select("a[href*='/media-bias/']"):
        text = a.get_text(strip=True)
        if text in ("Left", "Lean Left", "Center", "Lean Right", "Right"):
            article.bias_detail = text
            break

    # source name
    src_div = soup.select_one("div.text-2xl.font-bold")
    if src_div:
        article.source = src_div.get_text(strip=True)
    if not article.source:
        src_link = soup.select_one("a[href*='/news-source/']")
        if src_link:
            article.source = src_link.get_text(strip=True)

    # body text
    body_div = soup.select_one("div.body")
    if body_div:
        paragraphs = body_div.find_all("p")
        article.body = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

    if not article.body:
        article_tag = soup.select_one("article")
        if article_tag:
            parts = [p.get_text(strip=True) for p in article_tag.find_all("p") if len(p.get_text(strip=True)) > 60]
            if parts:
                article.body = "\n\n".join(parts)

    # original source URL
    read_full = soup.select_one("a[id^='Read-Full-Story']")
    if read_full:
        article.original_url = read_full.get("href")
    else:
        for a in soup.select("a[target='_blank']"):
            href = a.get("href", "")
            if href.startswith("http") and "allsides.com" not in href:
                article.original_url = href
                break


def run():
    fetcher = Fetcher()

    print(f"\nFetching: {NEWS_URL}")
    soup = fetcher.fetch(NEWS_URL)
    if soup is None:
        print("Could not fetch AllSides page.")
        sys.exit(1)

    roundups = scrape_roundups(soup)[:MAX_ROUNDUPS]
    print(f"Found {len(roundups)} headline roundups\n")

    for i, roundup in enumerate(roundups, 1):
        print(f"[{i}] {roundup.main_headline}")
        for side in roundup.sides:
            print(f"    {side.bias:>8}: {side.headline}")

            time.sleep(REQUEST_DELAY)
            scrape_article(fetcher, side)

            if side.title:
                print(f"             title: {side.title[:80]}")
            if side.bias_detail:
                print(f"             bias:  {side.bias} ({side.bias_detail})")
            if side.body:
                print(f"             body:  {side.body[:100].replace(chr(10), ' ')}...")
        print()

    # save
    data = [asdict(r) for r in roundups]
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(roundups)} roundups to {OUTPUT_FILE}")


if __name__ == "__main__":
    run()
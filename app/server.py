"""
news_mcp_server/server.py
──────────────────────────
MCP server that fetches Google News RSS and returns structured articles.

Tools exposed:
  • fetch_news   — search Google News RSS for a query, return articles
  • fetch_by_url — fetch and parse a specific RSS feed URL

Run:
  python server.py

Google News RSS format:
  https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en
"""

import json
import re
import html
import requests
from datetime import datetime
from mcp.server.fastmcp import FastMCP
from bs4 import BeautifulSoup
from dateutil import parser as dateparser

mcp = FastMCP(
    name="google-news-rss",
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}

GOOGLE_NEWS_BASE = "https://news.google.com/rss/search"


# ── Tool 1: fetch_news ────────────────────────────────────────────────────────

@mcp.tool()
def fetch_news(
    query: str,
    max_results: int = 10,
    days_back: int = 7,
) -> str:
    """
    Fetch news articles from Google News RSS for a given search query.

    Args:
        query:       Search query e.g. 'Suzlon Energy NSE stock'
        max_results: Maximum number of articles to return (default 10)
        days_back:   How many days back to search (default 7)

    Returns:
        JSON string with list of articles:
        [{
            "title":       "Article headline",
            "url":         "https://...",
            "source":      "Economic Times",
            "date":        "2024-03-15",
            "date_display":"Mar 15, 2024",
            "snippet":     "Short description from RSS"
        }]
    """
    try:
        # Build RSS URL — add India locale for NSE-relevant news
        params = {
            "q":    query,
            "hl":   "en-IN",
            "gl":   "IN",
            "ceid": "IN:en",
        }
        url  = GOOGLE_NEWS_BASE
        resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()

        articles = _parse_rss(resp.text, max_results, days_back)

        return json.dumps({
            "success":  True,
            "query":    query,
            "count":    len(articles),
            "articles": articles,
        })

    except Exception as e:
        return json.dumps({
            "success":  False,
            "error":    str(e),
            "articles": [],
        })


# ── Tool 2: fetch_by_url ──────────────────────────────────────────────────────

@mcp.tool()
def fetch_by_url(rss_url: str, max_results: int = 10) -> str:
    """
    Fetch and parse a specific RSS feed URL.

    Args:
        rss_url:     Full RSS URL to fetch
        max_results: Maximum articles to return

    Returns:
        JSON string with list of articles (same schema as fetch_news)
    """
    try:
        resp = requests.get(rss_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        articles = _parse_rss(resp.text, max_results, days_back=365)
        return json.dumps({
            "success":  True,
            "url":      rss_url,
            "count":    len(articles),
            "articles": articles,
        })
    except Exception as e:
        return json.dumps({
            "success":  False,
            "error":    str(e),
            "articles": [],
        })


# ── RSS Parser ────────────────────────────────────────────────────────────────

def _parse_rss(xml_text: str, max_results: int, days_back: int) -> list[dict]:
    """
    Parses Google News RSS XML and returns clean article dicts.
    Google News RSS item structure:
      <item>
        <title>Headline - Source Name</title>
        <link>https://...</link>
        <pubDate>Thu, 14 Mar 2024 10:30:00 GMT</pubDate>
        <description>...</description>
        <source url="...">Source Name</source>
      </item>
    """
    soup    = BeautifulSoup(xml_text, "lxml-xml")
    items   = soup.find_all("item")
    now     = datetime.utcnow()
    results = []

    for item in items:
        if len(results) >= max_results:
            break

        # ── Title & source ────────────────────────────────────────────────────
        raw_title  = item.find("title")
        raw_title  = raw_title.get_text(strip=True) if raw_title else ""

        # Google News appends " - Source Name" to the title
        source_tag = item.find("source")
        source     = source_tag.get_text(strip=True) if source_tag else ""

        # Strip source from title if present
        title = raw_title
        if source and title.endswith(f" - {source}"):
            title = title[: -(len(source) + 3)].strip()

        # ── URL ───────────────────────────────────────────────────────────────
        link = item.find("link")
        url  = ""
        if link:
            # lxml-xml puts link text as next sibling
            url = (link.get_text(strip=True)
                   or link.get("href", "")
                   or "")
        if not url:
            guid = item.find("guid")
            url  = guid.get_text(strip=True) if guid else ""

        # ── Date ──────────────────────────────────────────────────────────────
        pub_date_tag = item.find("pubDate")
        date_str     = ""
        date_display = ""
        if pub_date_tag:
            raw_date = pub_date_tag.get_text(strip=True)
            try:
                dt           = dateparser.parse(raw_date)
                age_days     = (now - dt.replace(tzinfo=None)).days
                if age_days > days_back:
                    continue   # too old
                date_str     = dt.strftime("%Y-%m-%d")
                date_display = dt.strftime("%b %d, %Y")
            except Exception:
                date_str     = raw_date[:10]
                date_display = raw_date[:16]

        # ── Snippet ───────────────────────────────────────────────────────────
        desc = item.find("description")
        snippet = ""
        if desc:
            raw = desc.get_text(strip=True)
            # Strip HTML tags that sometimes appear in description
            snippet = re.sub(r"<[^>]+>", "", html.unescape(raw)).strip()
            snippet = snippet[:300]

        if not title or not url:
            continue

        results.append({
            "title":        title,
            "url":          url,
            "source":       source,
            "date":         date_str,
            "date_display": date_display,
            "snippet":      snippet,
        })

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")

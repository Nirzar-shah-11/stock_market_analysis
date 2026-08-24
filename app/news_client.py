"""
news_mcp_server/news_client.py
───────────────────────────────
MCP client that calls the Google News RSS server,
then uses Llama 3 to generate 2 concise bullet points per article.

Returns: list of enriched article dicts ready for UI display and
         the prediction node context.
"""

import json
import subprocess
import sys
import time
import requests
from pathlib import Path
from colorama import Fore, Style

# Import fetch_news directly instead of using subprocess due to stdio buffering issues with MCP subprocess calls.
try:
    from server import fetch_news as server_fetch_news
    USE_DIRECT_IMPORT = True
except ImportError:
    USE_DIRECT_IMPORT = False

SERVER_SCRIPT = Path(__file__).parent / "server.py"

OLLAMA_MODEL = "llama3:8b"
OLLAMA_API_URL = "http://localhost:11434/api/chat"

SUMMARISE_SYSTEM = """You are a financial news analyst specialising in Indian equity markets.

Given a news article title and snippet, produce exactly 2 bullet points:
  • Point 1: The core fact / event (what happened)
  • Point 2: The market implication for the stock (why it matters to investors)

Each point must be under 20 words.
Return ONLY valid JSON — no markdown, no preamble:
{"point1": "...", "point2": "..."}"""


class NewsClient:
    """Calls the Google News RSS MCP server and enriches articles with Llama 3 summaries."""

    def __init__(self, hf_token: str, server_script: Path = SERVER_SCRIPT):
        self.hf_token      = hf_token
        self.server_script = str(server_script)
        self.hf_headers    = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type":  "application/json",
        }

    # ── Public ────────────────────────────────────────────────────────────────

    def fetch_and_summarise(
        self,
        company_name: str,
        symbol:       str,
        max_articles: int = 20,
        days_back:    int = 7,
    ) -> list[dict]:
        """
        Fetches news via MCP server, then generates 2 bullet points per article.

        Returns list of dicts:
        {
            title, url, source, date, date_display, snippet,
            point1, point2,
            sentiment_hint: "positive" | "negative" | "neutral"
        }
        """
        # Build a targeted query for Indian financial news
        query = f"{company_name} {symbol} NSE stock"

        _log(f"Fetching Google News RSS for '{query}'…")
        articles = self._call_mcp("fetch_news", {
            "query":       query,
            "max_results": max_articles,
            "days_back":   days_back,
        })

        if not articles:
            _log("No articles found", warn=True)
            return []

        _log(f"✓ {len(articles)} articles found — generating summaries…", success=True)

        enriched = []
        for i, art in enumerate(articles, 1):
            _log(f"  [{i}/{len(articles)}] Summarising: {art['title'][:60]}…")
            summary = self._summarise(art, company_name, symbol)
            enriched.append({
                **art,
                "point1":         summary.get("point1", ""),
                "point2":         summary.get("point2", ""),
                "sentiment_hint": self._infer_sentiment(art["title"] + " " + art.get("snippet","")),
            })
            time.sleep(0.8)   # HF rate-limit courtesy

        _log(f"✓ {len(enriched)} articles enriched with summaries", success=True)
        return enriched

    # ── Direct function call (replaces MCP subprocess) ─────────────────────────

    def _call_mcp(self, tool_name: str, arguments: dict) -> list[dict]:
        """
        Direct call to fetch_news (bypasses MCP subprocess which has stdio issues)
        """
        if not USE_DIRECT_IMPORT:
            _log("fetch_news not available", warn=True)
            return []
        
        try:
            if tool_name == "fetch_news":
                result_json = server_fetch_news(
                    query=arguments.get("query", ""),
                    max_results=arguments.get("max_results", 10),
                    days_back=arguments.get("days_back", 7),
                )
                data = json.loads(result_json)
                return data.get("articles", []) if data.get("success") else []
            else:
                _log(f"Unknown tool: {tool_name}", warn=True)
                return []
        except Exception as e:
            _log(f"Direct call error: {e}", warn=True)
            return []

    # ── Llama 3 summariser ────────────────────────────────────────────────────

    def _summarise(self, article: dict, company_name: str, symbol: str) -> dict:
        title   = article.get("title", "")
        snippet = article.get("snippet", "")[:400]

        user_prompt = (
            f"Company: {company_name} ({symbol})\n"
            f"Article title: {title}\n"
            f"Snippet: {snippet}\n\n"
            f"Generate 2 bullet points as instructed."
        )

        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": SUMMARISE_SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
            "max_tokens": 120,
            "temperature": 0.1,
            "stream": False,
        }

        for attempt in range(3):
            try:
                # Ollama local API doesn't need HF headers
                r = requests.post(OLLAMA_API_URL,
                                  json=payload, timeout=45)
                if r.status_code == 503:
                    time.sleep(r.json().get("estimated_time", 20) + 2)
                    continue
                if r.status_code == 429:
                    time.sleep(60)
                    continue
                r.raise_for_status()
                # Ollama response format: {message: {content: "..."}}
                raw = r.json()["message"]["content"].strip()
                return self._parse_summary(raw)
            except Exception:
                time.sleep(5 * (attempt + 1))

        return {"point1": "", "point2": ""}

    @staticmethod
    def _parse_summary(raw: str) -> dict:
        text = raw.strip()
        if "```" in text:
            text = "\n".join(l for l in text.split("\n")
                             if not l.strip().startswith("```")).strip()
        try:
            return json.loads(text)
        except Exception:
            import re
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try: return json.loads(m.group())
                except Exception: pass
        return {"point1": "", "point2": ""}

    # ── Sentiment hint (keyword-based, no LLM call) ───────────────────────────

    @staticmethod
    def _infer_sentiment(text: str) -> str:
        text = text.lower()
        pos_kw = ["surge","rally","gain","profit","win","record","upgrade",
                  "growth","beat","acquisition","order","approval","buyback"]
        neg_kw = ["fall","drop","loss","penalty","downgrade","default","probe",
                  "crash","decline","miss","concern","risk","warning","debt"]
        pos = sum(1 for k in pos_kw if k in text)
        neg = sum(1 for k in neg_kw if k in text)
        return "positive" if pos > neg else "negative" if neg > pos else "neutral"


def _log(msg, warn=False, success=False):
    tag   = f"{Fore.CYAN}[News Client]{Style.RESET_ALL}"
    color = Fore.GREEN if success else (Fore.YELLOW if warn else Fore.WHITE)
    print(f"{tag} {color}{msg}{Style.RESET_ALL}")

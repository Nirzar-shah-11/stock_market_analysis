"""
nodes/news_node.py
───────────────────
Fetches Google News RSS via MCP server, summarises each article
into 2 bullet points using Llama 3, and writes to state.

Runs in parallel with hints_node and oi_node (after price_node).

Writes: state.news_articles → list of enriched article dicts
"""

from colorama import Fore, Style
from state import PredictionState
from news_client import NewsClient


def news_node(state: PredictionState) -> dict:
    company_name = state["company_name"]
    symbol       = state["symbol"]

    _log(f"Fetching Google News RSS for {company_name} ({symbol})…")

    try:
        # Use empty token — Ollama doesn't require HF headers
        client   = NewsClient(hf_token="")
        articles = client.fetch_and_summarise(
            company_name=company_name,
            symbol=symbol,
            max_articles=8,
            days_back=7,
        )
        _log(f"✓ {len(articles)} articles with summaries", success=True)
        return {"news_articles": articles}

    except Exception as e:
        _log(f"News node error: {e}", warn=True)
        return {"news_articles": [], "errors": [f"news_node: {e}"]}


def _log(msg, warn=False, success=False):
    tag   = f"{Fore.BLUE}[News Node]{Style.RESET_ALL}"
    color = Fore.GREEN if success else (Fore.YELLOW if warn else Fore.WHITE)
    print(f"{tag} {color}{msg}{Style.RESET_ALL}")

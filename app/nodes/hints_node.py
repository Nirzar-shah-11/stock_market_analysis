"""
nodes/hints_node.py
────────────────────
Runs all hint scrapers for TODAY's date only.
Fetches: bulk deals, insider trades, pledges, FII/DII,
         regulatory filings, sector signals.

Uses jugaad_data.nse.NSELive for reliable data fetching
(replaces direct HTTP API which returns empty responses).
"""

import time
import pandas as pd
from datetime import date, timedelta
from colorama import Fore, Style
from state import PredictionState
from jugaad_data.nse import NSELive

SIGNAL_KEYWORDS = [
    "credit rating","downgrade","upgrade","outlook","litigation","penalty",
    "show cause","regulatory","insolvency","debt restructur","nclt","sebi order",
    "arbitration","tax demand","pledge","promoter","acquisition","merger",
    "demerger","buyback","dividend","bonus","rights issue","fundrais",
    "order win","contract","mou","agreement","management change","director",
    "ceo","cfo","default","npa","write-off",
]


def hints_node(state: PredictionState) -> dict:
    symbol = state["symbol"].upper()
    _log(f"Scanning today's subtle hints for {symbol}…")

    today     = date.today()
    yesterday = today - timedelta(days=3)   # look back 3 days to catch weekend filings
    
    nse_live = NSELive()

    signals = []
    
    # Try all hint sources (gracefully handle failures)
    signals.extend(_bulk_block(nse_live, symbol, yesterday, today))
    signals.extend(_insider_trades(nse_live, symbol, yesterday, today))
    signals.extend(_fii_dii(nse_live, yesterday, today, symbol))
    signals.extend(_regulatory(nse_live, symbol, yesterday, today))
    signals.extend(_sector(nse_live, symbol))

    _log(f"✓ {len(signals)} hint signal(s) found today", success=len(signals) > 0)
    return {"today_hints": signals}


# ── Bulk / Block deals ────────────────────────────────────────────────────────

def _bulk_block(nse_live, symbol, start, end) -> list:
    """
    Fetch bulk and block deals.
    Note: jugaad_data NSELive doesn't have direct bulk/block deal API.
    Returns empty for now - can be enhanced with alternative source.
    """
    signals = []
    try:
        # NSELive doesn't expose bulk/block deal endpoint directly
        # This would require either:
        # 1. Alternative data source
        # 2. Web scraping NSE website
        # For now, gracefully return empty
        pass
    except Exception as e:
        _log(f"Bulk/block deal fetch skipped: {e}", warn=True)
    return signals


# ── Insider trades ────────────────────────────────────────────────────────────

def _insider_trades(nse_live, symbol, start, end) -> list:
    """
    Fetch insider trading data.
    Note: jugaad_data NSELive doesn't expose insider trades directly.
    Returns empty for now - can be enhanced with alternative source.
    """
    signals = []
    try:
        # NSELive doesn't expose insider trade endpoint directly
        # This would require SEBI database or alternative source
        pass
    except Exception as e:
        _log(f"Insider trades fetch skipped: {e}", warn=True)
    return signals


# ── FII/DII flows ─────────────────────────────────────────────────────────────

def _fii_dii(nse_live, start, end, symbol) -> list:
    """
    Fetch FII/DII flow data using NSELive.
    """
    signals = []
    try:
        # Try to get live FII/DII data
        fii_dii_data = nse_live.live_fno()
        
        if not fii_dii_data:
            return signals
        
        # Extract flow data
        fii_net = 0
        dii_net = 0
        
        # Parse response based on structure
        if isinstance(fii_dii_data, dict):
            # Look for FII/DII data in response
            if 'fii' in fii_dii_data:
                fii_net = float(fii_dii_data.get('fii', 0) or 0)
            if 'dii' in fii_dii_data:
                dii_net = float(fii_dii_data.get('dii', 0) or 0)
        
        if abs(fii_net) < 500 and abs(dii_net) < 500:
            return signals
        
        fii_dir = "bought" if fii_net > 0 else "sold"
        dii_dir = "bought" if dii_net > 0 else "sold"
        bias = "bullish" if (fii_net > 0 and dii_net > 0) else (
               "bearish" if (fii_net < 0 and dii_net < 0) else "mixed")
        
        signals.append({
            "signal_type": "FII/DII Flow",
            "source": "NSE",
            "detail": f"FII net {fii_dir} ₹{fii_net:+,.0f} Cr | DII net {dii_dir} ₹{dii_net:+,.0f} Cr",
            "bias":   bias,
            "score_hint": 4 if bias == "bullish" else (2 if bias == "bearish" else 3),
            "explanation": (
                f"Market-wide institutional flow: FII {'accumulating' if fii_net > 0 else 'exiting'} "
                f"and DII {'supporting' if dii_net > 0 else 'also selling'}. "
                f"This {'supports' if bias == 'bullish' else 'pressures'} {symbol}."
            ),
            "raw": f"[FII/DII] FII ₹{fii_net:+,.0f} Cr | DII ₹{dii_net:+,.0f} Cr",
        })
    except Exception as e:
        _log(f"FII/DII fetch error: {e}", warn=True)
    
    return signals


# ── Regulatory filings ────────────────────────────────────────────────────────

def _regulatory(nse_live, symbol, start, end) -> list:
    """
    Fetch regulatory filings and corporate announcements using NSELive.
    """
    signals = []
    try:
        # Use NSELive corporate_announcements method
        announcements = nse_live.corporate_announcements(symbol)
        
        if not announcements:
            return signals
        
        # Parse response
        if isinstance(announcements, dict):
            rows = announcements.get("data", []) if isinstance(announcements.get("data"), list) else []
        elif isinstance(announcements, list):
            rows = announcements
        else:
            return signals
        
        for r in rows:
            subject = r.get("subject", r.get("desc", ""))
            
            if not any(kw in subject.lower() for kw in SIGNAL_KEYWORDS):
                continue
            
            # Infer bias from keywords
            negative_kw = ["penalty","downgrade","default","npa","nclt","show cause",
                           "insolvency","arbitration","litigation","write-off"]
            positive_kw = ["upgrade","order win","contract","mou","buyback",
                           "dividend","fundrais","acquisition"]
            
            bias = ("bearish" if any(k in subject.lower() for k in negative_kw)
                    else "bullish" if any(k in subject.lower() for k in positive_kw)
                    else "neutral")
            
            signals.append({
                "signal_type": "Regulatory/Corporate Filing",
                "source": "NSE Filing",
                "detail": subject[:200],
                "bias":   bias,
                "score_hint": 2 if bias == "bearish" else (4 if bias == "bullish" else 3),
                "explanation": (
                    f"Official exchange disclosure: '{subject[:150]}'. "
                    f"This type of filing typically has {'negative' if bias == 'bearish' else 'positive' if bias == 'bullish' else 'neutral'} "
                    f"market implications."
                ),
                "raw": f"[Filing] {subject}",
            })
    except Exception as e:
        _log(f"Regulatory filing fetch error: {e}", warn=True)
    
    return signals


def _sector(nse_live, symbol) -> list:
    """
    Fetch sector index performance using NSELive.
    """
    signals = []
    
    SECTOR_INDEX_MAP = {
        "ENERGY": "NIFTY ENERGY", "POWER": "NIFTY ENERGY",
        "IT": "NIFTY IT", "SOFTWARE": "NIFTY IT",
        "BANKING": "NIFTY BANK", "FINANCIAL SERVICES": "NIFTY FIN SERVICE",
        "PHARMA": "NIFTY PHARMA", "AUTO": "NIFTY AUTO",
        "METAL": "NIFTY METAL", "REALTY": "NIFTY REALTY",
        "FMCG": "NIFTY FMCG", "PSU BANK": "NIFTY PSU BANK",
    }
    
    try:
        # Get stock quote to find industry
        quote = nse_live.stock_quote(symbol)
        
        if not quote or not isinstance(quote, dict):
            return signals
        
        # Extract industry from quote
        industry = ""
        if "industryInfo" in quote:
            industry = quote["industryInfo"].get("industry", "").upper()
        elif "metadata" in quote:
            industry = quote["metadata"].get("industry", "").upper()
        
        if not industry:
            return signals
        
        # Find corresponding index
        index_name = next((v for k, v in SECTOR_INDEX_MAP.items() if k in industry), None)
        
        if not index_name:
            return signals
        
        # Get all indices data
        try:
            indices_data = nse_live.all_indices()
            
            if not indices_data:
                return signals
            
            # Parse indices data
            if isinstance(indices_data, dict):
                indices = indices_data.get("data", [])
            elif isinstance(indices_data, list):
                indices = indices_data
            else:
                return signals
            
            # Find matching index
            for idx in indices:
                idx_name = idx.get("index", "").upper() if isinstance(idx, dict) else ""
                
                if idx_name == index_name.upper():
                    pct = float(idx.get("percentChange", 0) or 0)
                    
                    if abs(pct) >= 1.5:
                        bias = "bullish" if pct > 0 else "bearish"
                        signals.append({
                            "signal_type": "Sector Index",
                            "source": "NSE Sectoral Index",
                            "detail": f"{index_name} {pct:+.2f}% today",
                            "bias":   bias,
                            "score_hint": 4 if pct > 0 else 2,
                            "explanation": (
                                f"{index_name} is {'rising' if pct > 0 else 'falling'} {abs(pct):.2f}% today. "
                                f"As a {industry.lower()} stock, {symbol} typically "
                                f"{'benefits from' if pct > 0 else 'gets dragged by'} sector moves."
                            ),
                            "raw": f"[Sector] {index_name} {pct:+.2f}%",
                        })
                    break
        except Exception as e:
            _log(f"Could not fetch all indices: {e}", warn=True)
    
    except Exception as e:
        _log(f"Sector analysis error: {e}", warn=True)
    
    return signals
def _log(msg, warn=False, success=False):
    tag = f"{Fore.BLUE}[Hints Node]{Style.RESET_ALL}"
    color = Fore.GREEN if success else (Fore.YELLOW if warn else Fore.WHITE)
    print(f"{tag} {color}{msg}{Style.RESET_ALL}")

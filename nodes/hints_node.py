"""
nodes/hints_node.py
────────────────────
Runs all 6 scrapers for TODAY's date only.
Fetches: bulk deals, insider trades, pledges, FII/DII,
         regulatory filings, sector signals.
"""

import time
import requests
import pandas as pd
from datetime import date, timedelta
from colorama import Fore, Style
from state import PredictionState

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.nseindia.com/",
    "Accept":  "application/json, text/plain, */*",
}

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
    session   = _session()

    signals = []
    sources = []   
    signals.extend(_bulk_block(session, symbol, yesterday, today))
    signals.extend(_insider_trades(session, symbol, yesterday, today))
    signals.extend(_fii_dii(session, yesterday, today, symbol))
    signals.extend(_regulatory(session, symbol, yesterday, today))
    signals.extend(_sector(session, symbol))

    _log(f"✓ {len(signals)} hint signal(s) found today", success=len(signals) > 0)
    return {"today_hints": signals}


# ── Bulk / Block deals ────────────────────────────────────────────────────────

def _bulk_block(session, symbol, start, end) -> list:
    signals = []
    fmt = "%d-%m-%Y"
    for label, url_tpl in [
        ("Bulk Deal",  "https://www.nseindia.com/api/bulk-deal-archives?from={f}&to={t}&category=bulk_deals"),
        ("Block Deal", "https://www.nseindia.com/api/block-deal-archives?from={f}&to={t}&category=block_deals"),
    ]:
        try:
            url  = url_tpl.format(f=start.strftime(fmt), t=end.strftime(fmt))
            rows = session.get(url, timeout=12).json().get("data", [])
            for r in rows:
                sym = (r.get("symbol","") or r.get("SYMBOL","")).upper().strip()
                if sym != symbol: continue
                client   = r.get("clientName", r.get("BD_CLIENT_NAME","Unknown"))
                qty      = r.get("quantityTraded", r.get("BD_QTY_TRD",0))
                price    = r.get("tradePrice", r.get("BD_TP_WATP",0))
                buy_sell = r.get("buySell", r.get("BD_BUY_SELL",""))
                signals.append({
                    "signal_type": label,
                    "source": "NSE",
                    "detail": f"{buy_sell} {qty:,} shares @ ₹{price} by {client}",
                    "bias":   "bullish" if "BUY" in str(buy_sell).upper() else "bearish",
                    "score_hint": 4 if "BUY" in str(buy_sell).upper() else 2,
                    "explanation": (
                        f"A large {label.lower()} by {client} indicates "
                        f"{'institutional accumulation' if 'BUY' in str(buy_sell).upper() else 'institutional exit'}. "
                        f"Quantity: {qty:,} shares at ₹{price}."
                    ),
                    "raw": f"[{label}] {client} {buy_sell} {qty} shares of {symbol} @ ₹{price}",
                })
        except Exception:
            pass
        time.sleep(0.3)
    return signals


# ── Insider trades ────────────────────────────────────────────────────────────

def _insider_trades(session, symbol, start, end) -> list:
    signals = []
    fmt = "%d-%m-%Y"
    try:
        url  = (f"https://www.nseindia.com/api/corporates-pit"
                f"?index=equities&symbol={symbol}"
                f"&from_date={start.strftime(fmt)}&to_date={end.strftime(fmt)}")
        rows = session.get(url, timeout=12).json().get("data", [])
        for r in rows:
            person = r.get("personName", r.get("acqName","Unknown"))
            cat    = r.get("category","")
            qty    = r.get("noOfSharesAcquired", r.get("secAcq",0))
            price  = r.get("acquiredFromPrice", r.get("secVal",0))
            trans  = r.get("transactionType","Buy")
            is_buy = "buy" in str(trans).lower()
            signals.append({
                "signal_type": "Insider Trade",
                "source": "SEBI/NSE",
                "detail": f"{person} ({cat}) {trans} {qty:,} shares @ ₹{price}",
                "bias":   "bullish" if is_buy else "bearish",
                "score_hint": 4 if is_buy else 2,
                "explanation": (
                    f"{'Promoter/insider buying is a strong confidence signal — ' if is_buy else 'Insider selling may indicate loss of confidence — '}"
                    f"{person} ({cat}) {'acquired' if is_buy else 'sold'} {qty:,} shares."
                ),
                "raw": f"[Insider] {person} ({cat}) {trans} {qty} shares of {symbol} @ ₹{price}",
            })
    except Exception:
        pass
    return signals


# ── FII/DII flows ─────────────────────────────────────────────────────────────

def _fii_dii(session, start, end, symbol) -> list:
    signals = []
    fmt = "%d-%m-%Y"
    try:
        url  = (f"https://www.nseindia.com/api/fiidiiTradeReact"
                f"?from={start.strftime(fmt)}&to={end.strftime(fmt)}")
        rows = session.get(url, timeout=12).json()
        rows = rows if isinstance(rows, list) else rows.get("data", [])
        for r in rows:
            fii_net = float(r.get("fiiBuy",0) or 0) - float(r.get("fiiSell",0) or 0)
            dii_net = float(r.get("diiBuy",0) or 0) - float(r.get("diiSell",0) or 0)
            if abs(fii_net) < 500 and abs(dii_net) < 500:
                continue
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
    except Exception:
        pass
    return signals


# ── Regulatory filings ────────────────────────────────────────────────────────

def _regulatory(session, symbol, start, end) -> list:
    signals = []
    fmt = "%d-%m-%Y"
    try:
        url  = (f"https://www.nseindia.com/api/corp-announcements"
                f"?index=equities&symbol={symbol}"
                f"&from_date={start.strftime(fmt)}&to_date={end.strftime(fmt)}")
        rows = session.get(url, timeout=12).json()
        rows = rows if isinstance(rows, list) else rows.get("data", [])
        for r in rows:
            subject = r.get("subject", r.get("desc",""))
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
    except Exception:
        pass
    return signals


# ── Sector signals ────────────────────────────────────────────────────────────

SECTOR_INDEX_MAP = {
    "ENERGY": "NIFTY ENERGY", "POWER": "NIFTY ENERGY",
    "IT": "NIFTY IT", "SOFTWARE": "NIFTY IT",
    "BANKING": "NIFTY BANK", "FINANCIAL SERVICES": "NIFTY FIN SERVICE",
    "PHARMA": "NIFTY PHARMA", "AUTO": "NIFTY AUTO",
    "METAL": "NIFTY METAL", "REALTY": "NIFTY REALTY",
    "FMCG": "NIFTY FMCG", "PSU BANK": "NIFTY PSU BANK",
}

def _sector(session, symbol) -> list:
    signals = []
    try:
        # Get sector
        resp = session.get(
            f"https://www.nseindia.com/api/quote-equity?symbol={symbol}", timeout=10)
        industry = resp.json().get("industryInfo", {}).get("industry", "").upper()
        index_name = next((v for k, v in SECTOR_INDEX_MAP.items() if k in industry), None)
        if not index_name:
            return signals

        # Get index quote
        enc = index_name.replace(" ", "%20")
        r2  = session.get(
            f"https://www.nseindia.com/api/allIndices", timeout=10)
        indices = r2.json().get("data", [])
        for idx in indices:
            if idx.get("index", "").upper() == index_name.upper():
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
    except Exception:
        pass
    return signals


def _session():
    s = requests.Session()
    s.headers.update(NSE_HEADERS)
    try:
        s.get("https://www.nseindia.com", timeout=8)
        time.sleep(0.5)
    except Exception:
        pass
    return s

def _log(msg, warn=False, success=False):
    tag = f"{Fore.BLUE}[Hints Node]{Style.RESET_ALL}"
    color = Fore.GREEN if success else (Fore.YELLOW if warn else Fore.WHITE)
    print(f"{tag} {color}{msg}{Style.RESET_ALL}")

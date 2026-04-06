"""
nodes/price_node.py
────────────────────
Fetches:
  1. Today's live quote (open, high, low, close, volume, % change)
  2. Last 60 days of OHLCV history (for pattern matching)
"""

import time, requests
import pandas as pd
from datetime import date, timedelta
from colorama import Fore, Style
from state import PredictionState

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.nseindia.com/",
    "Accept": "application/json, text/plain, */*",
}

def price_node(state: PredictionState) -> dict:
    symbol = state["symbol"].upper()
    _log(f"Fetching price data for {symbol}…")

    session = _session()

    today_price = _fetch_live_quote(session, symbol)
    history_df  = _fetch_history(symbol)

    if today_price:
        _log(f"✓ Live: ₹{today_price.get('close','?')} ({today_price.get('pct_change','?'):+.2f}%)", success=True)
    else:
        _log("Live quote unavailable — using last close from history", warn=True)

    if history_df is not None and not history_df.empty:
        _log(f"✓ History: {len(history_df)} days", success=True)
        trading_dates = sorted(history_df["Date"].tolist())
        # Fill today_price from last row if live quote failed
        if not today_price and len(history_df):
            last = history_df.iloc[-1]
            prev = history_df.iloc[-2] if len(history_df) > 1 else last
            pct  = (last["Close"] - prev["Close"]) / prev["Close"] * 100 if prev["Close"] else 0
            today_price = {
                "open": last["Open"], "high": last["High"],
                "low":  last["Low"],  "close": last["Close"],
                "volume": last["Volume"], "pct_change": round(pct, 2),
            }
    else:
        trading_dates = []
        _log("History unavailable", warn=True)

    return {
        "today_price":   today_price or {},
        "history_df":    history_df,
        "trading_dates": trading_dates,
    }

# ── Live quote ────────────────────────────────────────────────────────────────

def _fetch_live_quote(session, symbol: str) -> dict | None:
    try:
        url  = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        pd_  = data.get("priceInfo", {})
        return {
            "open":       pd_.get("open", 0),
            "high":       pd_.get("intraDayHighLow", {}).get("max", 0),
            "low":        pd_.get("intraDayHighLow", {}).get("min", 0),
            "close":      pd_.get("lastPrice", 0),
            "prev_close": pd_.get("previousClose", 0),
            "volume":     data.get("marketDeptOrderBook", {}).get("tradeInfo", {}).get("totalTradedVolume", 0),
            "pct_change": pd_.get("pChange", 0),
        }
    except Exception:
        return None

# ── 60-day history ────────────────────────────────────────────────────────────

def _fetch_history(symbol: str) -> pd.DataFrame | None:
    end   = date.today()
    start = end - timedelta(days=90)  # fetch 90 to get ~60 trading days

    # Try jugaad first
    try:
        from jugaad_data.nse import stock_df
        df = stock_df(symbol=symbol, from_date=start, to_date=end, series="EQ")
        if df is not None and not df.empty:
            return _normalize(df, symbol)
    except Exception:
        pass

    # Fallback: NSE API
    try:
        session = _session()
        fmt = "%d-%m-%Y"
        url = (
            "https://www.nseindia.com/api/historical/cm/equity"
            f"?symbol={symbol}&series=[%22EQ%22]"
            f"&from={start.strftime(fmt)}&to={end.strftime(fmt)}"
        )
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if data:
            return _normalize(pd.DataFrame(data), symbol)
    except Exception:
        pass

    return None

def _normalize(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    col_map = {
        "CH_TIMESTAMP": "Date", "CH_OPENING_PRICE": "Open",
        "CH_TRADE_HIGH_PRICE": "High", "CH_TRADE_LOW_PRICE": "Low",
        "CH_CLOSING_PRICE": "Close", "CH_TOT_TRADED_QTY": "Volume",
        "COP_DELIV_QTY": "Deliverable_Volume",
        "openPrice": "Open", "highPrice": "High", "lowPrice": "Low",
        "ltp": "Close", "tradedDate": "Date", "mTRADED_QTY": "Volume",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    for col in ["Open","High","Low","Close","Volume"]:
        if col not in df.columns: df[col] = None
    if "Deliverable_Volume" not in df.columns: df["Deliverable_Volume"] = None
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df["Symbol"] = symbol
    for col in ["Open","High","Low","Close","Volume","Deliverable_Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("Date").reset_index(drop=True).tail(60)

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
    tag = f"{Fore.CYAN}[Price Node]{Style.RESET_ALL}"
    color = Fore.GREEN if success else (Fore.YELLOW if warn else Fore.WHITE)
    print(f"{tag} {color}{msg}{Style.RESET_ALL}")

"""
nodes/price_node.py
────────────────────
Fetches:
  1. Today's live quote (open, high, low, close, volume, % change)
  2. Last 60 days of OHLCV history (for pattern matching)

Uses jugaad_data.nse.NSELive and stock_df for reliable data fetching.
"""

import pandas as pd
from datetime import date, timedelta
from colorama import Fore, Style
from state import PredictionState
from jugaad_data.nse import NSELive, stock_df

def price_node(state: PredictionState) -> dict:
    symbol = state["symbol"].upper()
    _log(f"Fetching price data for {symbol}…")

    nse_live = NSELive()

    today_price = _fetch_live_quote(nse_live, symbol)
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

def _fetch_live_quote(nse_live, symbol: str) -> dict | None:
    """
    Fetch live quote using NSELive.
    NSELive.stock_quote returns nested dict with priceInfo section.
    """
    try:
        quote = nse_live.stock_quote(symbol)
        
        if not quote or not isinstance(quote, dict):
            return None
        
        # NSELive stock_quote response structure has priceInfo nested
        price_info = quote.get("priceInfo", {}) or {}
        pre_open = quote.get("preOpenMarket", {}) or {}
        
        # Extract high/low from intraDayHighLow
        intra_high_low = price_info.get("intraDayHighLow", {}) or {}
        
        # Extract values
        open_price = float(price_info.get("open", 0) or 0)
        close_price = float(price_info.get("lastPrice", 0) or 0)
        high_price = float(intra_high_low.get("max", 0) or 0)
        low_price = float(intra_high_low.get("min", 0) or 0)
        prev_close = float(price_info.get("previousClose", 0) or 0)
        volume = float(pre_open.get("totalTradedVolume", 0) or 0)  # From preOpenMarket
        pct_change = float(price_info.get("pChange", 0) or 0)
        
        return {
            "open":       open_price,
            "high":       high_price,
            "low":        low_price,
            "close":      close_price,
            "prev_close": prev_close,
            "volume":     volume,
            "pct_change": pct_change,
        }
    except Exception as e:
        _log(f"Quote fetch error: {e}", warn=True)
        return None

# ── 60-day history ────────────────────────────────────────────────────────────

def _fetch_history(symbol: str) -> pd.DataFrame | None:
    """
    Fetch last 60 trading days of OHLCV using jugaad_data.
    """
    end   = date.today()
    start = end - timedelta(days=90)  # fetch 90 to get ~60 trading days

    try:
        df = stock_df(symbol=symbol, from_date=start, to_date=end, series="EQ")
        
        if df is not None and not df.empty:
            _log(f"Got {len(df)} days from stock_df", success=True)
            normalized = _normalize(df, symbol)
            if normalized is not None:
                _log(f"Normalized to {len(normalized)} days", success=True)
                return normalized
            else:
                _log("Normalization returned None", warn=True)
        else:
            _log("Empty dataframe from stock_df", warn=True)
    except Exception as e:
        _log(f"stock_df error: {e}", warn=True)

    return None

def _normalize(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Normalize jugaad_data stock_df columns to standard format.
    jugaad_data returns: DATE, SERIES, OPEN, HIGH, LOW, CLOSE, VOLUME, etc.
    """
    try:
        # Handle duplicate Close: both CLOSE and LTP columns exist
        # Use CLOSE if available, otherwise use LTP, but not both
        if "CLOSE" in df.columns and "LTP" in df.columns:
            df = df.drop(columns=["LTP"])
        
        # Map jugaad_data columns (uppercase) to standard format
        col_map = {
            "DATE":         "Date",
            "OPEN":         "Open",
            "HIGH":         "High",
            "LOW":          "Low",
            "CLOSE":        "Close",
            "VOLUME":       "Volume",
            "PREV. CLOSE":  "Prev_Close",
            "DELIVERY QTY": "Deliverable_Volume",
        }
        
        # Rename columns that exist
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        
        # Ensure all required columns exist FIRST
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                df[col] = 0.0
        
        # Add symbol column
        df["Symbol"] = symbol
        
        # Convert numeric columns FIRST (before touching dates)
        for col in ["Open", "High", "Low", "Close"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        
        if "Volume" in df.columns:
            df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0).astype(int)
        
        if "Deliverable_Volume" in df.columns:
            df["Deliverable_Volume"] = pd.to_numeric(df["Deliverable_Volume"], errors="coerce").fillna(0).astype(int)
        
        # NOW convert DATE column to date object AFTER all other conversions
        if "Date" in df.columns:
            try:
                # Convert datetime64[ms] to datetime, then to date
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.sort_values("Date", na_position='last').reset_index(drop=True)
                df["Date"] = df["Date"].dt.date
            except Exception as e:
                _log(f"Date handling failed: {e}", warn=True)
                df = df.reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)
        
        return df.tail(60)
    
    except Exception as e:
        _log(f"Normalize error: {e}", warn=True)
        import traceback
        traceback.print_exc()
        return None

def _log(msg, warn=False, success=False):
    tag = f"{Fore.CYAN}[Price Node]{Style.RESET_ALL}"
    color = Fore.GREEN if success else (Fore.YELLOW if warn else Fore.WHITE)
    print(f"{tag} {color}{msg}{Style.RESET_ALL}")

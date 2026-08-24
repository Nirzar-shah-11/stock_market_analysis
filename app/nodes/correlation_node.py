"""
nodes/correlation_node.py
──────────────────────────
The genuinely new agent vs the dataset builder.

Answers: "How are peer stocks and sector leaders affecting this stock TODAY?"

Logic:
  1. Looks up the stock's sector from NSE
  2. Identifies peer stocks (same sector, from Nifty indices)
  3. Fetches today's % change for each peer
  4. Computes historical correlation (from 60-day history)
  5. Flags peers with significant moves that likely impact this stock
"""

import time
import pandas as pd
import numpy as np
from datetime import date, timedelta
from colorama import Fore, Style
from state import PredictionState
from jugaad_data.nse import NSELive

# Peer map: sector keyword → list of (symbol, name)
SECTOR_PEERS = {
    "ENERGY":   [("TATAPOWER","Tata Power"),("ADANIGREEN","Adani Green"),
                 ("NTPC","NTPC"),("POWERGRID","Power Grid"),("TORNTPOWER","Torrent Power")],
    "POWER":    [("TATAPOWER","Tata Power"),("NTPC","NTPC"),
                 ("POWERGRID","Power Grid"),("CESC","CESC")],
    "IT":       [("TCS","TCS"),("INFY","Infosys"),("WIPRO","Wipro"),
                 ("HCLTECH","HCL Tech"),("TECHM","Tech Mahindra")],
    "SOFTWARE": [("TCS","TCS"),("INFY","Infosys"),("WIPRO","Wipro"),("HCLTECH","HCL Tech")],
    "BANKING":  [("HDFCBANK","HDFC Bank"),("ICICIBANK","ICICI Bank"),
                 ("SBIN","SBI"),("AXISBANK","Axis Bank"),("KOTAKBANK","Kotak Bank")],
    "FINANCIAL":[("HDFCBANK","HDFC Bank"),("BAJFINANCE","Bajaj Finance"),
                 ("ICICIBANK","ICICI Bank"),("SBIN","SBI")],
    "PHARMA":   [("SUNPHARMA","Sun Pharma"),("DRREDDY","Dr. Reddy"),
                 ("CIPLA","Cipla"),("DIVISLAB","Divi's Labs")],
    "AUTO":     [("MARUTI","Maruti"),("TATAMOTORS","Tata Motors"),
                 ("M&M","M&M"),("BAJAJ-AUTO","Bajaj Auto")],
    "METAL":    [("TATASTEEL","Tata Steel"),("JSWSTEEL","JSW Steel"),
                 ("HINDALCO","Hindalco"),("SAIL","SAIL")],
    "REALTY":   [("DLF","DLF"),("GODREJPROP","Godrej Properties"),
                 ("PRESTIGE","Prestige"),("OBEROIRLTY","Oberoi Realty")],
    "FMCG":     [("HINDUNILVR","HUL"),("ITC","ITC"),
                 ("NESTLEIND","Nestle"),("DABUR","Dabur")],
}

# Market leaders that affect everything
MARKET_LEADERS = [
    ("RELIANCE",  "Reliance Industries"),
    ("HDFCBANK",  "HDFC Bank"),
    ("TCS",       "TCS"),
    ("INFY",      "Infosys"),
]

SIGNIFICANT_PEER_MOVE = 2.0   # flag peer moves > ±2%
CORR_THRESHOLD        = 0.4   # minimum correlation to flag


def correlation_node(state: PredictionState) -> dict:
    symbol     = state["symbol"].upper()
    history_df = state.get("history_df")
    _log(f"Correlation analysis for {symbol}…")

    nse_live = NSELive()
    signals = []

    # ── Get sector ────────────────────────────────────────────────────────────
    sector = _get_sector(nse_live, symbol)
    _log(f"  Sector: {sector or 'unknown'}")

    # ── Find peers ────────────────────────────────────────────────────────────
    peers = _find_peers(symbol, sector)
    _log(f"  Peers identified: {[p[0] for p in peers]}")

    # ── Historical correlation ─────────────────────────────────────────────────
    corr_map = _compute_correlations(symbol, peers, history_df)

    # ── Today's peer moves ────────────────────────────────────────────────────
    for peer_sym, peer_name in peers:
        if peer_sym == symbol:
            continue
        try:
            pct = _get_today_pct(nse_live, peer_sym)
            if pct is None:
                continue

            corr = corr_map.get(peer_sym, 0.0)

            # Only flag significant moves
            if abs(pct) < SIGNIFICANT_PEER_MOVE and abs(corr) < CORR_THRESHOLD:
                continue

            direction = "rising" if pct > 0 else "falling"
            impact    = "positive" if pct > 0 else "negative"
            corr_str  = f"{corr:.2f}" if corr else "unknown"

            signals.append({
                "peer_symbol":   peer_sym,
                "peer_name":     peer_name,
                "peer_pct":      round(pct, 2),
                "correlation":   round(corr, 3) if corr else None,
                "signal_type":   "Correlation",
                "source":        "Peer Stock Analysis",
                "detail":        f"{peer_name} ({peer_sym}) {pct:+.2f}% | historical corr={corr_str}",
                "bias":          "bullish" if pct > 0 else "bearish",
                "score_hint":    4 if pct > 0 else 2,
                "explanation":   (
                    f"{peer_name} is {direction} {abs(pct):.2f}% today. "
                    f"Historical correlation with {symbol} = {corr_str}. "
                    f"This typically has a {impact} spillover effect on {symbol} "
                    f"{'given strong sector linkage.' if abs(corr) > 0.6 else 'though correlation is moderate.'}"
                ),
                "raw": f"[Correlation] {peer_name} {pct:+.2f}% | corr={corr_str}",
            })

            time.sleep(0.2)

        except Exception:
            continue

    _log(f"✓ {len(signals)} correlation signal(s)", success=len(signals) > 0)
    return {"correlation_signals": signals}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_sector(nse_live: NSELive, symbol: str) -> str:
    """Get sector using NSELive"""
    try:
        quote = nse_live.stock_quote(symbol)
        sector = quote.get("industryInfo", {}).get("industry", "").upper()
        return sector if sector else ""
    except Exception:
        return ""

def _find_peers(symbol: str, sector: str) -> list[tuple]:
    peers = list(MARKET_LEADERS)   # always check market leaders
    for key, peer_list in SECTOR_PEERS.items():
        if key in sector:
            for p in peer_list:
                if p not in peers and p[0] != symbol:
                    peers.append(p)
            break
    return peers[:8]   # cap at 8 peers

def _get_today_pct(nse_live: NSELive, symbol: str) -> float | None:
    """Get today's % change using NSELive"""
    try:
        quote = nse_live.stock_quote(symbol)
        pct = float(quote.get("priceInfo", {}).get("pChange", 0) or 0)
        return pct if pct != 0 else None
    except Exception:
        return None

def _compute_correlations(
    symbol: str,
    peers: list[tuple],
    history_df,
) -> dict[str, float]:
    """
    Computes rolling correlation between symbol and each peer
    using the last 60 days of history.
    """
    corr_map = {}
    if history_df is None or (hasattr(history_df, "empty") and history_df.empty):
        return corr_map

    try:
        base_returns = history_df.set_index("Date")["Close"].pct_change().dropna()
        end   = date.today()
        start = end - timedelta(days=90)

        for peer_sym, _ in peers:
            if peer_sym == symbol:
                continue
            try:
                from jugaad_data.nse import stock_df
                peer_df = stock_df(symbol=peer_sym, from_date=start,
                                   to_date=end, series="EQ")
                if peer_df is None or peer_df.empty:
                    continue
                # Normalize column names
                date_col  = next((c for c in peer_df.columns if "TIMESTAMP" in c or "Date" in c), None)
                close_col = next((c for c in peer_df.columns if "CLOSING" in c or "Close" in c), None)
                if not date_col or not close_col:
                    continue
                peer_df = peer_df.rename(columns={date_col: "Date", close_col: "Close"})
                peer_df["Date"]  = pd.to_datetime(peer_df["Date"]).dt.date
                peer_returns = peer_df.set_index("Date")["Close"].pct_change().dropna()
                aligned = pd.concat([base_returns, peer_returns], axis=1, join="inner")
                if len(aligned) > 10:
                    corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                    corr_map[peer_sym] = round(float(corr), 3) if not np.isnan(corr) else 0.0
            except Exception:
                continue
    except Exception:
        pass

    return corr_map

def _log(msg, warn=False, success=False):
    tag = f"{Fore.GREEN}[Correlation Node]{Style.RESET_ALL}"
    color = Fore.GREEN if success else (Fore.YELLOW if warn else Fore.WHITE)
    print(f"{tag} {color}{msg}{Style.RESET_ALL}")

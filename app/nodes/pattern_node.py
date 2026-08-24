"""
nodes/pattern_node.py
──────────────────────
Looks back through 60-day history and finds dates where a similar
cluster of signals occurred — then reports what happened next (1d/3d/7d).

This is the "on the fly past impact check" — no model training needed.
The LLM sees: "Last time you had weak rally + high FII selling + bearish RSI,
the stock fell 4.2% over 3 days."
"""

import numpy as np
import pandas as pd
from datetime import date
from colorama import Fore, Style
from state import PredictionState


def pattern_node(state: PredictionState) -> dict:
    history_df  = state.get("history_df")
    today_tech  = state.get("today_tech", [])
    today_hints = state.get("today_hints", [])
    symbol      = state["symbol"]

    if history_df is None or (hasattr(history_df, "empty") and history_df.empty):
        _log("No history — skipping pattern match", warn=True)
        return {"similar_patterns": []}

    _log(f"Searching 60-day history for similar signal patterns…")

    df = history_df.copy().sort_values("Date").reset_index(drop=True)
    if len(df) < 15:
        return {"similar_patterns": []}

    # ── Compute derived columns ───────────────────────────────────────────────
    df["pct_change"]   = df["Close"].pct_change()
    df["vol_avg20"]    = df["Volume"].rolling(20).mean()
    df["vol_ratio"]    = df["Volume"] / df["vol_avg20"]
    df["candle_range"] = df["High"] - df["Low"]
    df["upper_wick"]   = df["High"] - df[["Open","Close"]].max(axis=1)
    df["rsi"]          = _rsi(df["Close"])
    df["atr"]          = _atr(df)
    df["atr_ratio"]    = df["atr"] / df["atr"].rolling(20).mean()

    # Build today's signal fingerprint
    today_fingerprint = _fingerprint(today_tech, today_hints, state.get("today_price", {}), df)
    similar = []

    # Scan all historical dates (except last 8 — need future data)
    for i in range(10, len(df) - 8):
        row  = df.iloc[i]
        past = df.iloc[i - 5:i]

        hist_fingerprint = _historical_fingerprint(row, past)
        similarity = _match_score(today_fingerprint, hist_fingerprint)
        overlap = len(today_fingerprint & hist_fingerprint)
        context_score = _context_similarity(state.get("today_price", {}), row, df)
        effective_similarity = max(similarity, context_score)

        if effective_similarity < 0.25 and overlap < 1:
            continue

        # Compute outcomes
        close_now  = row["Close"]
        close_1d   = df.iloc[i + 1]["Close"] if i + 1 < len(df) else None
        close_3d   = df.iloc[i + 3]["Close"] if i + 3 < len(df) else None
        close_7d   = df.iloc[i + 7]["Close"] if i + 7 < len(df) else None

        def pct(future, now):
            return round((future - now) / now * 100, 2) if future and now else None

        outcome_1d = pct(close_1d, close_now)
        outcome_3d = pct(close_3d, close_now)
        outcome_7d = pct(close_7d, close_now)

        if outcome_1d is None:
            continue

        similar.append({
            "date":         str(row["Date"]),
            "similarity":   round(effective_similarity, 2),
            "matching_signals": hist_fingerprint,
            "close_at_date": round(float(close_now), 2),
            "outcome_1d":   outcome_1d,
            "outcome_3d":   outcome_3d,
            "outcome_7d":   outcome_7d,
            "description": (
                f"On {row['Date']}: similar signals present. "
                f"Stock {'rose' if outcome_1d > 0 else 'fell'} {abs(outcome_1d):.2f}% next day"
                + (f", {abs(outcome_3d):.2f}% in 3 days" if outcome_3d else "")
                + (f", {abs(outcome_7d):.2f}% in 7 days" if outcome_7d else "")
                + "."
            ),
        })

    # Sort by similarity and return top 5
    similar.sort(key=lambda x: x["similarity"], reverse=True)
    top = similar[:5]

    if top:
        avg_1d = sum(x["outcome_1d"] for x in top) / len(top)
        avg_3d = sum(x["outcome_3d"] for x in top if x["outcome_3d"]) / max(1, sum(1 for x in top if x["outcome_3d"]))
        _log(f"✓ {len(top)} similar past patterns found | avg 1d outcome: {avg_1d:+.2f}%", success=True)
    else:
        _log("No strong historical matches found")

    return {"similar_patterns": top}


# ── Signal fingerprinting ─────────────────────────────────────────────────────

def _fingerprint(tech_signals: list, hint_signals: list, today_price: dict | None = None, history_df: pd.DataFrame | None = None) -> set:
    """Converts today's signals into a set of labels for matching."""
    fp = set()
    for s in tech_signals:
        name = s.get("signal_type","").replace("Technical: ","").lower()
        fp.add(name)
    for s in hint_signals:
        bias = s.get("bias","")
        stype = s.get("signal_type","").lower()
        if "fii" in stype:
            fp.add(f"fii_{bias}")
        elif "insider" in stype:
            fp.add(f"insider_{bias}")
        elif "bulk" in stype or "block" in stype:
            fp.add(f"bulk_{bias}")
        elif "sector" in stype:
            fp.add(f"sector_{bias}")

    if today_price:
        pct_change = today_price.get("pct_change")
        volume = today_price.get("volume")
        if history_df is not None and not history_df.empty and volume is not None:
            vol_avg20 = history_df["Volume"].tail(20).mean()
            if pd.notna(vol_avg20) and vol_avg20:
                vol_ratio = volume / vol_avg20
            else:
                vol_ratio = None
        else:
            vol_ratio = None

        if pct_change is None and history_df is not None and len(history_df) >= 2:
            prev_close = history_df["Close"].iloc[-2]
            last_close = history_df["Close"].iloc[-1]
            if pd.notna(prev_close) and prev_close:
                pct_change = (last_close - prev_close) / prev_close

        if pct_change is not None:
            if pct_change > 0.005 and (vol_ratio is None or vol_ratio < 0.75):
                fp.add("weak rally")
            if pct_change < -0.005 and (vol_ratio is None or vol_ratio > 1.4):
                fp.add("strong selling")
            if vol_ratio is not None and abs(pct_change) < 0.01 and vol_ratio > 1.2:
                fp.add("sideways distribution")

    return fp

def _historical_fingerprint(row: pd.Series, past: pd.DataFrame) -> set:
    """Generates a signal fingerprint for a historical date."""
    fp = set()
    pct   = row.get("pct_change", 0) or 0
    vol_r = row.get("vol_ratio", 1.0) or 1.0
    c_range = row.get("candle_range", 0) or 0
    up_wick = row.get("upper_wick", 0) or 0
    rsi     = row.get("rsi")
    atr_r   = row.get("atr_ratio")

    if pct > 0.005 and vol_r < 0.75:
        fp.add("weak rally")
    if pct < -0.005 and vol_r > 1.4:
        fp.add("strong selling")
    if c_range > 0 and (up_wick / c_range) > 0.55:
        fp.add("long upper wick")
    if pd.notna(rsi):
        if len(past) >= 3:
            past_prices = past["Close"]
            past_rsi    = past["rsi"].dropna()
            if len(past_rsi) >= 2:
                if row["Close"] > past_prices.max() and rsi < past_rsi.max():
                    fp.add("bearish rsi divergence")
                elif row["Close"] < past_prices.min() and rsi > past_rsi.min():
                    fp.add("bullish rsi divergence")
    if pd.notna(atr_r) and atr_r < 0.5:
        fp.add("volatility squeeze")
    if c_range > 0 and (c_range / row["Close"]) < 0.015 and vol_r > 1.3:
        fp.add("sideways distribution")
    return fp

def _match_score(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    overlap = len(a & b)
    union = len(a | b)
    if not union:
        return 0.0
    jaccard = overlap / union
    if overlap >= 1:
        return max(jaccard, overlap / 3.0)
    return jaccard


def _context_similarity(today_price: dict | None, row: pd.Series, history_df: pd.DataFrame) -> float:
    if not today_price:
        return 0.0

    pct_change = today_price.get("pct_change")
    volume = today_price.get("volume")
    if pct_change is None and history_df is not None and len(history_df) >= 2:
        prev_close = history_df["Close"].iloc[-2]
        last_close = history_df["Close"].iloc[-1]
        if pd.notna(prev_close) and prev_close:
            pct_change = (last_close - prev_close) / prev_close

    hist_pct = row.get("pct_change")
    hist_vol_ratio = row.get("vol_ratio")

    if pct_change is None or hist_pct is None:
        return 0.0

    score = 0.0
    if (pct_change > 0 and hist_pct > 0) or (pct_change < 0 and hist_pct < 0):
        score += 0.45
    elif abs(pct_change) > 0.01 and abs(hist_pct) > 0.01:
        score += 0.2

    magnitude_gap = abs(abs(pct_change) - abs(hist_pct))
    if magnitude_gap < 0.02:
        score += 0.35
    elif magnitude_gap < 0.05:
        score += 0.2

    if volume is not None and pd.notna(hist_vol_ratio) and pd.notna(volume):
        if history_df is not None and not history_df.empty:
            vol_avg20 = history_df["Volume"].tail(20).mean()
            if pd.notna(vol_avg20) and vol_avg20:
                vol_ratio = volume / vol_avg20
                if pd.notna(vol_ratio) and pd.notna(hist_vol_ratio):
                    if abs(vol_ratio - hist_vol_ratio) < 0.25:
                        score += 0.2

    return min(1.0, score)


# ── Indicators ────────────────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    loss  = (-delta).clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"]  - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period-1, min_periods=period).mean()


def _log(msg, warn=False, success=False):
    tag = f"{Fore.BLUE}[Pattern Node]{Style.RESET_ALL}"
    color = Fore.GREEN if success else (Fore.YELLOW if warn else Fore.WHITE)
    print(f"{tag} {color}{msg}{Style.RESET_ALL}")

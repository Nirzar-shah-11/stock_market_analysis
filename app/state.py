"""
state.py — Prediction Graph State
───────────────────────────────────
All nodes read/write this single TypedDict.
"""

from __future__ import annotations
from typing import Any
from datetime import date
from typing_extensions import TypedDict, Annotated
import operator


def _merge(a: dict, b: dict) -> dict:
    m = dict(a)
    for k, v in b.items():
        if k in m and isinstance(m[k], list) and isinstance(v, list):
            m[k] = m[k] + v
        else:
            m[k] = v
    return m


class PredictionState(TypedDict):
    # ── Inputs ────────────────────────────────────────────────────────────────
    company_name:    str
    symbol:          str

    # ── Routing ───────────────────────────────────────────────────────────────
    is_fo_stock:     bool

    # ── Price data ────────────────────────────────────────────────────────────
    today_price:     dict          # {open, high, low, close, volume, pct_change}
    history_df:      Any           # pd.DataFrame — last 60 days OHLCV
    trading_dates:   list[date]    # sorted dates in history

    # ── Today's signals ───────────────────────────────────────────────────────
    today_tech:      list[dict]    # technical signals for today
    today_oi:        list[dict]    # OI signals for today
    today_hints:     list[dict]    # subtle hints for today
    news_articles:   list[dict]    # recent company news

    # ── Historical pattern matching ───────────────────────────────────────────
    similar_patterns: list[dict]   # past dates with similar signal clusters
    # Each: {date, signals, outcome_1d, outcome_3d, outcome_7d, description}

    # ── Correlation signals ───────────────────────────────────────────────────
    correlation_signals: list[dict]
    # Each: {peer_symbol, peer_name, peer_pct_change, correlation, impact_desc}

    # ── LLM prediction output ─────────────────────────────────────────────────
    prediction: dict
    # {
    #   horizon_1d: {direction, magnitude_pct, confidence, reasoning},
    #   horizon_3d: {direction, magnitude_pct, confidence, reasoning},
    #   horizon_7d: {direction, magnitude_pct, confidence, reasoning},
    #   signal_trace: [{signal, interpretation, weight, direction}],
    #   pattern_match_summary: str,
    #   correlation_summary: str,
    #   overall_bias: "Bullish" | "Bearish" | "Neutral" | "Mixed",
    #   key_drivers: [str, str, str],
    #   risks: [str, str],
    # }

    # ── Metadata ──────────────────────────────────────────────────────────────
    errors:          Annotated[list[str], operator.add]
    run_timestamp:   str
    sources: list[str]


def fresh_state(company_name: str, symbol: str) -> PredictionState:
    from datetime import datetime
    return PredictionState(
        company_name=company_name,
        symbol=symbol.upper(),
        is_fo_stock=False,
        today_price={},
        history_df=None,
        trading_dates=[],
        today_tech=[],
        today_oi=[],
        today_hints=[],
        news_articles=[],
        similar_patterns=[],
        correlation_signals=[],
        prediction={},
        errors=[],
        run_timestamp=datetime.now().isoformat(),
        sources=[],
    )

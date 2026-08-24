"""
nodes/technical_node.py
────────────────────────
Computes all 10 technical signals on today's data using 60-day history.
"""

import numpy as np
import pandas as pd
from colorama import Fore, Style
from state import PredictionState

UPPER_WICK_RATIO      = 0.55
VOLUME_WEAK_RATIO     = 0.75
VOLUME_STRONG_RATIO   = 1.40
SIDEWAYS_RANGE_PCT    = 0.015
SIDEWAYS_VOL_RATIO    = 1.30
ATR_LOW_RATIO         = 0.50
RSI_PERIOD            = 14
RSI_LOOKBACK          = 5
BREAKOUT_LOOKBACK     = 20
BREAKOUT_REVERSAL_PCT = 0.015


def technical_node(state: PredictionState) -> dict:
    df = state.get("history_df")
    if df is None or (hasattr(df, "empty") and df.empty):
        _log("No history — skipping technical signals", warn=True)
        return {"today_tech": []}

    symbol = state["symbol"]
    _log(f"Computing 10 technical signals for {symbol}…")

    df = df.copy().sort_values("Date").reset_index(drop=True)

    # ── Derived columns ───────────────────────────────────────────────────────
    df["pct_change"]      = df["Close"].pct_change()
    df["vol_avg20"]       = df["Volume"].rolling(20).mean()
    df["vol_ratio"]       = df["Volume"] / df["vol_avg20"]
    df["candle_range"]    = df["High"] - df["Low"]
    df["upper_wick"]      = df["High"] - df[["Open","Close"]].max(axis=1)
    df["rsi"]             = _rsi(df["Close"], RSI_PERIOD)
    df["atr"]             = _atr(df, 14)
    df["atr_avg20"]       = df["atr"].rolling(20).mean()
    df["atr_ratio"]       = df["atr"] / df["atr_avg20"]
    df["rolling_high20"]  = df["High"].rolling(BREAKOUT_LOOKBACK).max().shift(1)

    if len(df) < RSI_PERIOD + 5:
        return {"today_tech": []}

    # Use last row = today
    i   = len(df) - 1
    row = df.iloc[i]
    signals = []

    pct   = row["pct_change"]
    vol_r = row["vol_ratio"]
    c_range = row["candle_range"]
    up_wick = row["upper_wick"]
    rsi     = row["rsi"]

    # 1. Weak rally
    if pct > 0.005 and pd.notna(vol_r) and vol_r < VOLUME_WEAK_RATIO:
        signals.append(_s("Weak Rally",
            f"Price +{pct*100:.2f}% but only {vol_r:.2f}x avg volume — rally lacks conviction",
            "bearish_hint", 2,
            "Price is rising but volume is below average. Institutions are not participating. "
            "This type of move often reverses when retail momentum fades."))

    # 2. Strong selling
    if pct < -0.005 and pd.notna(vol_r) and vol_r > VOLUME_STRONG_RATIO:
        signals.append(_s("Strong Selling",
            f"Price {pct*100:.2f}% on {vol_r:.2f}x avg volume — institutional distribution",
            "bearish", 1,
            "Heavy volume on a down day signals institutions are actively exiting. "
            "This is one of the most reliable bearish signals."))

    # 3. Long upper wick
    if c_range > 0 and (up_wick / c_range) > UPPER_WICK_RATIO:
        signals.append(_s("Long Upper Wick",
            f"Upper wick = {up_wick/c_range*100:.0f}% of candle — bulls rejected at highs",
            "bearish_hint", 2,
            "The stock tried to go higher but sellers pushed it back down by close. "
            "This shows strong overhead resistance and selling pressure."))

    # 4. Failed breakout
    prev_high = row.get("rolling_high20")
    if pd.notna(prev_high) and row["High"] > prev_high and row["Close"] < prev_high * (1 - BREAKOUT_REVERSAL_PCT):
        signals.append(_s("Failed Breakout / Bull Trap",
            f"Broke ₹{prev_high:.2f} resistance intraday but closed below — bull trap",
            "bearish", 1,
            "The stock crossed a key resistance level attracting breakout buyers, "
            "then reversed hard. Late buyers are now trapped. Expect further downside "
            "as they exit."))

    # 7. RSI divergence
    if i >= RSI_LOOKBACK:
        past_prices = df["Close"].iloc[i - RSI_LOOKBACK:i]
        past_rsi    = df["rsi"].iloc[i - RSI_LOOKBACK:i]
        if pd.notna(rsi) and len(past_rsi.dropna()) >= 3:
            if row["Close"] > past_prices.max() and rsi < past_rsi.max():
                signals.append(_s("Bearish RSI Divergence",
                    f"Price at new high ₹{row['Close']:.2f} but RSI={rsi:.1f} < recent peak {past_rsi.max():.1f}",
                    "bearish_hint", 2,
                    "Price is making higher highs but momentum (RSI) is making lower highs. "
                    "This divergence warns the uptrend is weakening and a reversal may be near."))
            elif row["Close"] < past_prices.min() and rsi > past_rsi.min():
                signals.append(_s("Bullish RSI Divergence",
                    f"Price at new low ₹{row['Close']:.2f} but RSI={rsi:.1f} > recent low {past_rsi.min():.1f}",
                    "bullish_hint", 4,
                    "Price is making lower lows but RSI is making higher lows — hidden buying. "
                    "This bullish divergence often precedes a reversal or bounce."))

    # 8. Sideways distribution
    price_range_pct = c_range / row["Close"] if row["Close"] else 0
    if price_range_pct < SIDEWAYS_RANGE_PCT and pd.notna(vol_r) and vol_r > SIDEWAYS_VOL_RATIO:
        signals.append(_s("Sideways Distribution",
            f"Tight {price_range_pct*100:.2f}% range with {vol_r:.2f}x avg volume",
            "bearish_hint", 2,
            "High volume on a sideways day means smart money is distributing (selling) "
            "into retail demand without moving the price — a classic exit strategy."))

    # 10. Volatility squeeze
    atr_r = row.get("atr_ratio")
    if pd.notna(atr_r) and atr_r < ATR_LOW_RATIO:
        signals.append(_s("Volatility Squeeze",
            f"ATR at {atr_r:.2f}x of 20-day average — abnormally low volatility",
            "neutral_alert", 3,
            "When volatility compresses this much, a large move is statistically imminent. "
            "Direction unknown — watch for a breakout trigger in next 2-3 sessions."))

    _log(f"✓ {len(signals)} technical signal(s) detected", success=True)
    return {"today_tech": signals}


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l = loss.ewm(com=period - 1, min_periods=period).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl  = df["High"] - df["Low"]
    hc  = (df["High"] - df["Close"].shift()).abs()
    lc  = (df["Low"]  - df["Close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()

def _s(name, detail, bias, score_hint, explanation="") -> dict:
    return {
        "signal_type":  f"Technical: {name}",
        "source":       "Price/Volume",
        "detail":       detail,
        "bias":         bias,
        "score_hint":   score_hint,
        "explanation":  explanation,
        "raw":          f"[Technical] {name}: {detail}",
    }

def _log(msg, warn=False, success=False):
    tag = f"{Fore.YELLOW}[Technical Node]{Style.RESET_ALL}"
    color = Fore.GREEN if success else (Fore.YELLOW if warn else Fore.WHITE)
    print(f"{tag} {color}{msg}{Style.RESET_ALL}")

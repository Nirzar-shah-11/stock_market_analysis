"""
nodes/oi_node.py
─────────────────
Fetches today's live options chain from NSE for F&O stocks.
Computes: PCR, max pain, OI buildup direction.
"""

import time, requests
import pandas as pd
from colorama import Fore, Style
from state import PredictionState

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.nseindia.com/",
    "Accept": "application/json, text/plain, */*",
}


def oi_node(state: PredictionState) -> dict:
    if not state.get("is_fo_stock"):
        _log("Not F&O — skipping OI node")
        return {"today_oi": []}

    symbol = state["symbol"].upper()
    _log(f"Fetching live options chain for {symbol}…")

    signals = []
    try:
        session = _session()
        url  = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        data   = resp.json()
        records = data.get("records", {}).get("data", [])
        spot    = data.get("records", {}).get("underlyingValue", 0)

        if not records or not spot:
            return {"today_oi": []}

        # Build CE/PE OI table
        rows = []
        for r in records:
            strike = r.get("strikePrice", 0)
            ce = r.get("CE", {})
            pe = r.get("PE", {})
            if ce or pe:
                rows.append({
                    "strike":    strike,
                    "ce_oi":     ce.get("openInterest", 0) or 0,
                    "pe_oi":     pe.get("openInterest", 0) or 0,
                    "ce_chg_oi": ce.get("changeinOpenInterest", 0) or 0,
                    "pe_chg_oi": pe.get("changeinOpenInterest", 0) or 0,
                })

        df = pd.DataFrame(rows)
        if df.empty:
            return {"today_oi": []}

        # ── PCR (Put-Call Ratio) ───────────────────────────────────────────────
        total_ce_oi = df["ce_oi"].sum()
        total_pe_oi = df["pe_oi"].sum()
        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 1.0

        # PCR > 1.2 = bullish (more puts = hedging = market optimistic)
        # PCR < 0.8 = bearish (more calls being written = market cautious)
        if pcr > 1.2:
            signals.append(_s("High PCR (Bullish)",
                f"PCR = {pcr:.2f} — put OI ({total_pe_oi:,}) > call OI ({total_ce_oi:,})",
                "bullish", 4,
                f"A PCR above 1.2 indicates heavy put writing — option sellers (smart money) "
                f"are selling puts, implying they expect the stock to hold or rise. "
                f"Current PCR of {pcr:.2f} is bullish."))
        elif pcr < 0.8:
            signals.append(_s("Low PCR (Bearish)",
                f"PCR = {pcr:.2f} — call OI ({total_ce_oi:,}) > put OI ({total_pe_oi:,})",
                "bearish", 2,
                f"A PCR below 0.8 means call OI dominates — suggesting bearish positioning. "
                f"Option writers are selling calls, implying they don't expect the stock to rise."))

        # ── Max Pain ──────────────────────────────────────────────────────────
        max_pain = _calc_max_pain(df)
        if max_pain and spot:
            diff_pct = (max_pain - spot) / spot * 100
            signals.append(_s("Max Pain Level",
                f"Max pain at ₹{max_pain:.0f} vs spot ₹{spot:.0f} ({diff_pct:+.1f}%)",
                "bullish" if diff_pct > 1 else ("bearish" if diff_pct < -1 else "neutral"), 3,
                f"Max pain (₹{max_pain:.0f}) is the price where option writers lose least. "
                f"Stock is ₹{spot:.0f} — {'below' if diff_pct > 0 else 'above'} max pain by {abs(diff_pct):.1f}%. "
                f"Prices tend to gravitate toward max pain near expiry."))

        # ── OI buildup at ATM ─────────────────────────────────────────────────
        atm_strike = _nearest_strike(df, spot)
        atm_row = df[df["strike"] == atm_strike]
        if not atm_row.empty:
            ce_chg = atm_row["ce_chg_oi"].values[0]
            pe_chg = atm_row["pe_chg_oi"].values[0]
            if abs(ce_chg) > 1000 or abs(pe_chg) > 1000:
                if ce_chg > pe_chg:
                    signals.append(_s("ATM Call OI Buildup",
                        f"ATM ₹{atm_strike} call OI +{ce_chg:,} vs put OI +{pe_chg:,}",
                        "bearish_hint", 2,
                        f"Fresh call writing at ATM strike ₹{atm_strike} — option sellers "
                        f"are capping upside at this level. Resistance signal."))
                else:
                    signals.append(_s("ATM Put OI Buildup",
                        f"ATM ₹{atm_strike} put OI +{pe_chg:,} vs call OI +{ce_chg:,}",
                        "bullish_hint", 4,
                        f"Fresh put writing at ATM strike ₹{atm_strike} — option sellers "
                        f"are protecting this level as support. Bullish signal."))

    except Exception as e:
        _log(f"OI fetch error: {e}", warn=True)

    _log(f"✓ {len(signals)} OI signal(s)", success=len(signals) > 0)
    return {"today_oi": signals}


def _calc_max_pain(df: pd.DataFrame) -> float | None:
    try:
        strikes = df["strike"].tolist()
        pains   = []
        for s in strikes:
            # Loss for call writers + loss for put writers at this expiry price
            call_loss = ((s - df["strike"]).clip(lower=0) * df["ce_oi"]).sum()
            put_loss  = ((df["strike"] - s).clip(lower=0) * df["pe_oi"]).sum()
            pains.append(call_loss + put_loss)
        return float(strikes[pains.index(min(pains))])
    except Exception:
        return None

def _nearest_strike(df: pd.DataFrame, spot: float) -> float:
    return df["strike"].iloc[(df["strike"] - spot).abs().argsort()[:1]].values[0]

def _s(name, detail, bias, score_hint, explanation="") -> dict:
    return {
        "signal_type":  f"OI: {name}",
        "source":       "NSE Options Chain",
        "detail":       detail,
        "bias":         bias,
        "score_hint":   score_hint,
        "explanation":  explanation,
        "raw":          f"[OI] {name}: {detail}",
    }

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
    tag = f"{Fore.MAGENTA}[OI Node]{Style.RESET_ALL}"
    color = Fore.GREEN if success else (Fore.YELLOW if warn else Fore.WHITE)
    print(f"{tag} {color}{msg}{Style.RESET_ALL}")

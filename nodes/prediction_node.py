"""
nodes/prediction_node.py
─────────────────────────
The final LLM node. Takes ALL signals + patterns + correlations
and produces a full prediction with reasoning trace for 1d/3d/7d.
"""

import json, time
from colorama import Fore, Style
from state import PredictionState
from llm_client import LLMClient


SYSTEM_PROMPT = """You are a senior quantitative analyst at a top Indian equity hedge fund with 20 years of NSE/BSE experience.

You will receive a complete picture of a stock's situation:
  - Today's technical signals (price/volume patterns)
  - Today's OI signals (options market positioning)
  - Today's subtle hints (insider trades, FII/DII, filings, sector)
  - Correlation signals (how peer stocks are moving today)
  - Historical pattern matches (what happened last time similar signals appeared)

Your job: produce a detailed, honest prediction with full reasoning.

OUTPUT FORMAT — return ONLY valid JSON, no markdown, no preamble:
{
  "overall_bias": "Bullish" | "Bearish" | "Neutral" | "Mixed",
  "key_drivers": ["driver 1", "driver 2", "driver 3"],
  "risks": ["risk 1", "risk 2"],
  "horizon_1d": {
    "direction": "Up" | "Down" | "Sideways",
    "magnitude_pct": <float, e.g. 2.5>,
    "confidence": <integer 0-100>,
    "reasoning": "<2-3 sentences explaining this specific horizon>"
  },
  "horizon_3d": {
    "direction": "Up" | "Down" | "Sideways",
    "magnitude_pct": <float>,
    "confidence": <integer 0-100>,
    "reasoning": "<2-3 sentences>"
  },
  "horizon_7d": {
    "direction": "Up" | "Down" | "Sideways",
    "magnitude_pct": <float>,
    "confidence": <integer 0-100>,
    "reasoning": "<2-3 sentences>"
  },
  "signal_trace": [
    {
      "signal": "<signal name>",
      "interpretation": "<what this means for this stock specifically>",
      "weight": "High" | "Medium" | "Low",
      "direction": "Bullish" | "Bearish" | "Neutral"
    }
  ],
  "pattern_match_summary": "<what historical similar patterns suggest>",
  "correlation_summary": "<how peer stock moves are influencing this stock today>"
}

Be specific. Use the actual stock name, actual numbers from the signals, actual peer names.
If signals conflict, explain the conflict. Confidence should reflect genuine uncertainty."""


def prediction_node(state: PredictionState) -> dict:
    company_name = state["company_name"]
    symbol       = state["symbol"]

    _log(f"Generating full prediction for {company_name} ({symbol})…")

    llm    = LLMClient()
    prompt = _build_prompt(state)
    raw    = llm.chat(SYSTEM_PROMPT, prompt, max_tokens=2500, temperature=0.2)
    result = _parse(raw)

    if result:
        bias  = result.get("overall_bias","?")
        conf1 = result.get("horizon_1d",{}).get("confidence","?")
        conf3 = result.get("horizon_3d",{}).get("confidence","?")
        d1    = result.get("horizon_1d",{}).get("direction","?")
        _log(f"✓ Bias={bias} | 1d={d1} (conf={conf1}%) | 3d conf={conf3}%", success=True)
    else:
        _log("LLM parse failed — using empty prediction", warn=True)
        result = _empty_prediction()

    return {"prediction": result}


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(state: PredictionState) -> str:
    symbol       = state["symbol"]
    company_name = state["company_name"]
    today_price  = state.get("today_price", {})
    tech         = state.get("today_tech", [])
    oi           = state.get("today_oi", [])
    hints        = state.get("today_hints", [])
    corr         = state.get("correlation_signals", [])
    patterns     = state.get("similar_patterns", [])

    lines = [
        f"STOCK: {company_name} ({symbol})",
        f"DATE:  {state.get('run_timestamp','today')[:10]}",
        "",
        "═══ TODAY'S PRICE ═══",
        f"Open:  ₹{today_price.get('open','N/A')}",
        f"High:  ₹{today_price.get('high','N/A')}",
        f"Low:   ₹{today_price.get('low','N/A')}",
        f"Close: ₹{today_price.get('close','N/A')}",
        f"Change: {today_price.get('pct_change','N/A'):+}%",
        f"Volume: {today_price.get('volume','N/A'):,}" if isinstance(today_price.get('volume'), (int, float)) else f"Volume: {today_price.get('volume','N/A')}",
        "",
    ]

    # Technical signals
    lines.append("═══ TECHNICAL SIGNALS ═══")
    if tech:
        for s in tech:
            lines.append(f"[{s['bias'].upper()}] {s['signal_type']}")
            lines.append(f"  Detail: {s['detail']}")
            lines.append(f"  Score hint: {s['score_hint']}/5")
            lines.append(f"  Explanation: {s.get('explanation','')}")
            lines.append("")
    else:
        lines.append("No technical signals detected today.")
        lines.append("")

    # OI signals
    lines.append("═══ OPTIONS / OI SIGNALS ═══")
    if oi:
        for s in oi:
            lines.append(f"[{s['bias'].upper()}] {s['signal_type']}")
            lines.append(f"  Detail: {s['detail']}")
            lines.append(f"  Explanation: {s.get('explanation','')}")
            lines.append("")
    else:
        lines.append("No OI signals (not F&O or data unavailable).")
        lines.append("")

    # Hint signals
    lines.append("═══ SUBTLE HINT SIGNALS ═══")
    if hints:
        for s in hints:
            lines.append(f"[{s['bias'].upper()}] {s['signal_type']}")
            lines.append(f"  Detail: {s['detail']}")
            lines.append(f"  Explanation: {s.get('explanation','')}")
            lines.append("")
    else:
        lines.append("No subtle hint signals found today.")
        lines.append("")

    # Correlation signals
    lines.append("═══ CORRELATION / PEER SIGNALS ═══")
    if corr:
        for s in corr:
            lines.append(f"[{s['bias'].upper()}] {s['peer_name']} ({s['peer_symbol']}): {s['peer_pct']:+.2f}%")
            lines.append(f"  Historical correlation with {symbol}: {s.get('correlation','N/A')}")
            lines.append(f"  Explanation: {s.get('explanation','')}")
            lines.append("")
    else:
        lines.append("No significant peer moves detected.")
        lines.append("")

    # News articles
    news = state.get("news_articles", [])
    lines.append("═══ RECENT NEWS (Google News RSS) ═══")
    if news:
        for art in news[:5]:   # top 5 for context
            lines.append(f"[{art.get('sentiment_hint','neutral').upper()}] {art.get('source','')} | {art.get('date_display','')}")
            lines.append(f"  Headline: {art.get('title','')}")
            if art.get("point1"): lines.append(f"  • {art['point1']}")
            if art.get("point2"): lines.append(f"  • {art['point2']}")
            lines.append("")
    else:
        lines.append("No recent news found.")
        lines.append("")
        
    # Historical patterns
    lines.append("═══ HISTORICAL PATTERN MATCHES ═══")
    if patterns:
        for p in patterns[:3]:   # top 3
            lines.append(f"Date: {p['date']} | Similarity: {p['similarity']*100:.0f}%")
            lines.append(f"  Matching signals: {', '.join(p['matching_signals'])}")
            lines.append(f"  Outcome: 1d={p['outcome_1d']:+.2f}%"
                         + (f", 3d={p['outcome_3d']:+.2f}%" if p.get('outcome_3d') else "")
                         + (f", 7d={p['outcome_7d']:+.2f}%" if p.get('outcome_7d') else ""))
            lines.append("")
    else:
        lines.append("No strong historical pattern matches found.")
        lines.append("")

    lines.append("Based on ALL the above, provide your full prediction JSON.")

    return "\n".join(lines)


# ── Parse + fallback ──────────────────────────────────────────────────────────

def _parse(raw: str) -> dict | None:
    if not raw: return None
    text = raw.strip()
    if "```" in text:
        text = "\n".join(l for l in text.split("\n")
                         if not l.strip().startswith("```")).strip()
    try:
        return json.loads(text)
    except Exception:
        import re
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try: return json.loads(m.group())
            except Exception: pass
    return None

def _empty_prediction() -> dict:
    empty_h = {"direction":"Sideways","magnitude_pct":0.0,"confidence":0,"reasoning":"No data available."}
    return {
        "overall_bias": "Neutral",
        "key_drivers":  ["Insufficient data"],
        "risks":        ["Data unavailable"],
        "horizon_1d":   empty_h,
        "horizon_3d":   empty_h,
        "horizon_7d":   empty_h,
        "signal_trace": [],
        "pattern_match_summary": "No historical patterns matched.",
        "correlation_summary":   "No correlation data available.",
    }

def _log(msg, warn=False, success=False):
    tag = f"{Fore.MAGENTA}[Prediction Node]{Style.RESET_ALL}"
    color = Fore.GREEN if success else (Fore.YELLOW if warn else Fore.WHITE)
    print(f"{tag} {color}{msg}{Style.RESET_ALL}")

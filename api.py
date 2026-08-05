"""
api.py — FastAPI Server
  uvicorn api:app --reload --port 8000
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from state import fresh_state
from graph import graph
import warnings
warnings.filterwarnings("ignore", message="no explicit representation of timezones")

try:
    import numpy as np
except Exception:
    np = None

try:
    import pandas as pd
except Exception:
    pd = None

load_dotenv()

app = FastAPI(title="Stock Predictor API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# Serve static files (the web UI)
app.mount("/static", StaticFiles(directory="static"), name="static")


class PredictRequest(BaseModel):
    company: str
    symbol:  str


def generate_recommendation(prediction: dict) -> dict:
    """Generate buy/sell/hold recommendation based on prediction data."""
    if not prediction:
        return {"action": "HOLD", "confidence": 0, "reasoning": "Insufficient data"}

    overall_bias = prediction.get("overall_bias", "Neutral")
    horizon_1d = prediction.get("horizon_1d", {})
    horizon_3d = prediction.get("horizon_3d", {})
    horizon_7d = prediction.get("horizon_7d", {})
    risks = prediction.get("risks", [])
    signal_trace = prediction.get("signal_trace", [])

    confidences = [
        horizon_1d.get("confidence", 0),
        horizon_3d.get("confidence", 0),
        horizon_7d.get("confidence", 0),
    ]
    avg_confidence = sum(confidences) / len([c for c in confidences if c > 0]) if any(confidences) else 0

    bullish_count = sum(1 for s in signal_trace if s.get("direction") == "Bullish")
    bearish_count = sum(1 for s in signal_trace if s.get("direction") == "Bearish")
    total_signals = len(signal_trace) or 1

    horizon_votes = []
    for horizon in [horizon_1d, horizon_3d, horizon_7d]:
        direction = (horizon or {}).get("direction")
        confidence = (horizon or {}).get("confidence", 0) or 0
        if direction == "Up":
            horizon_votes.append(("BUY", confidence))
        elif direction == "Down":
            horizon_votes.append(("SELL", confidence))

    buy_confidence = sum(conf for action, conf in horizon_votes if action == "BUY")
    sell_confidence = sum(conf for action, conf in horizon_votes if action == "SELL")
    net_direction = "BUY" if buy_confidence > sell_confidence else "SELL" if sell_confidence > buy_confidence else None

    action = "HOLD"
    reasoning = f"Mixed/Neutral signals ({overall_bias}), awaiting clearer direction"

    if net_direction == "BUY" and buy_confidence >= 60:
        action = "BUY"
        reasoning = f"Bullish horizon signals dominate (conf: {buy_confidence:.0f} / {buy_confidence + sell_confidence:.0f})"
    elif net_direction == "SELL" and sell_confidence >= 60:
        action = "SELL"
        reasoning = f"Bearish horizon signals dominate (conf: {sell_confidence:.0f} / {buy_confidence + sell_confidence:.0f})"
    elif overall_bias == "Bullish" and avg_confidence >= 60 and bullish_count >= max(1, total_signals // 2):
        action = "BUY"
        reasoning = f"Bullish bias ({overall_bias}) detected with {bullish_count}/{total_signals} bullish signals"
    elif overall_bias == "Bearish" and avg_confidence >= 60 and bearish_count >= max(1, total_signals // 2):
        action = "SELL"
        reasoning = f"Bearish bias ({overall_bias}) detected with {bearish_count}/{total_signals} bearish signals"

    if risks and len(risks) > 1:
        action = "HOLD"
        reasoning += f". Multiple risks identified: {', '.join(risks[:2])}"

    return {
        "action": action,
        "confidence": round(min(0.95, max(0.0, avg_confidence / 100)), 2),
        "bullish_signals": bullish_count,
        "bearish_signals": bearish_count,
        "reasoning": reasoning,
        "risks": risks,
    }


def _json_safe(value):
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if pd is not None and isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in value]
    return value


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("templates/index.html") as f:
        return f.read()


@app.post("/predict")
async def predict(req: PredictRequest):
    try:
        state = fresh_state(req.company, req.symbol.upper())
        final = graph.invoke(state)
        pred  = final.get("prediction", {})
        errors = final.get("errors", [])
        
        recommendation = generate_recommendation(pred)

        response = {
            "symbol":         req.symbol.upper(),
            "company":        req.company,
            "run_timestamp":  final.get("run_timestamp", ""),
            "today_price":    final.get("today_price", {}),
            "signal_counts": {
                "technical":   len(final.get("today_tech", [])),
                "oi":          len(final.get("today_oi", [])),
                "hints":       len(final.get("today_hints", [])),
                "news":        len(final.get("news_articles", [])),
                "correlation": len(final.get("correlation_signals", [])),
                "patterns":    len(final.get("similar_patterns", [])),
            },
            "news_articles": final.get("news_articles", []),
            "prediction":    pred,
            "recommendation": recommendation,
            "errors":        errors,
        }
        return _json_safe(response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}

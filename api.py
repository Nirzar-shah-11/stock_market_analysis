"""
api.py — FastAPI Server
  uvicorn api:app --reload --port 8000
"""

import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
import numpy as np
from state import fresh_state
from graph import graph
import warnings
warnings.filterwarnings("ignore", message="no explicit representation of timezones")

load_dotenv()

app = FastAPI(title="Stock Predictor API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

APP_DIR = Path(__file__).resolve().parent
HISTORY_FILE = APP_DIR / "history.json"
HISTORY_LIMIT = 50

# Serve static files (the web UI)
STATIC_DIR = APP_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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
    reasoning = "The signals are mixed right now, so it is better to wait for a clearer trend."

    if net_direction == "BUY" and buy_confidence >= 60:
        action = "BUY"
        reasoning = "The overall picture looks positive, and this stock seems more likely to rise in the short term."
    elif net_direction == "SELL" and sell_confidence >= 60:
        action = "SELL"
        reasoning = "The overall picture looks negative, and this stock seems more likely to fall in the short term."
    elif overall_bias == "Bullish" and avg_confidence >= 60 and bullish_count >= max(1, total_signals // 2):
        action = "BUY"
        reasoning = "The signs are mostly positive, so a rise looks more likely than a drop."
    elif overall_bias == "Bearish" and avg_confidence >= 60 and bearish_count >= max(1, total_signals // 2):
        action = "SELL"
        reasoning = "The signs are mostly negative, so a drop looks more likely than a rise."

    if risks and len(risks) > 1:
        action = "HOLD"
        reasoning = "There are several warning signs, so staying cautious is the safer choice."

    return {
        "action": action,
        "confidence": round(min(0.95, max(0.0, avg_confidence / 100)), 2),
        "bullish_signals": bullish_count,
        "bearish_signals": bearish_count,
        "reasoning": reasoning,
        "risks": risks,
    }

def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []

    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    return []


def _write_history(entries: list[dict]) -> None:
    with HISTORY_FILE.open("w", encoding="utf-8") as handle:
        json.dump(entries[-HISTORY_LIMIT:], handle, indent=2, ensure_ascii=False)


def _append_history(entry: dict) -> None:
    history = _read_history()
    history.append(entry)
    _write_history(history)


def _history_entry(response: dict) -> dict:
    return {
        "symbol": response.get("symbol", ""),
        "company": response.get("company", ""),
        "run_timestamp": response.get("run_timestamp", ""),
        "action": response.get("recommendation", {}).get("action", "HOLD"),
        "confidence": response.get("recommendation", {}).get("confidence", 0),
        "bias": response.get("prediction", {}).get("overall_bias", "Neutral"),
        "today_price": response.get("today_price", {}),
        "signal_counts": response.get("signal_counts", {}),
        "prediction": {
            "overall_bias": response.get("prediction", {}).get("overall_bias", "Neutral"),
            "horizon_1d": response.get("prediction", {}).get("horizon_1d", {}),
            "horizon_3d": response.get("prediction", {}).get("horizon_3d", {}),
            "horizon_7d": response.get("prediction", {}).get("horizon_7d", {}),
        },
        "errors": response.get("errors", []),
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    with (APP_DIR / "templates" / "index.html").open(encoding="utf-8") as f:
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
        safe_response = _json_safe(response)
        _append_history(_history_entry(safe_response))
        return safe_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
async def history(limit: int = 20, symbol: str | None = None):
    entries = list(reversed(_read_history()))
    if symbol:
        target = symbol.upper()
        entries = [entry for entry in entries if entry.get("symbol", "").upper() == target]
    return _json_safe(entries[: max(1, min(limit, HISTORY_LIMIT))])


@app.get("/health")
async def health():
    return {"status": "ok"}

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
    
    # Calculate average confidence
    confidences = [
        horizon_1d.get("confidence", 0),
        horizon_3d.get("confidence", 0),
        horizon_7d.get("confidence", 0),
    ]
    avg_confidence = sum(confidences) / len([c for c in confidences if c > 0]) if any(confidences) else 0
    
    # Count bullish/bearish signals
    bullish_count = sum(1 for s in signal_trace if s.get("direction") == "Bullish")
    bearish_count = sum(1 for s in signal_trace if s.get("direction") == "Bearish")
    total_signals = len(signal_trace) or 1
    bullish_ratio = bullish_count / total_signals
    
    # Determine action
    action = "HOLD"
    reasoning = ""
    
    if overall_bias == "Bullish" and avg_confidence > 0.6 and bullish_ratio > 0.6:
        action = "BUY"
        reasoning = f"Strong bullish bias ({overall_bias}) with {bullish_count}/{total_signals} bullish signals (conf: {avg_confidence:.0%})"
    elif overall_bias == "Bearish" and avg_confidence > 0.6 and bullish_ratio < 0.4:
        action = "SELL"
        reasoning = f"Strong bearish bias ({overall_bias}) with {bearish_count}/{total_signals} bearish signals (conf: {avg_confidence:.0%})"
    elif overall_bias == "Bullish":
        action = "BUY"
        reasoning = f"Bullish bias ({overall_bias}) detected (conf: {avg_confidence:.0%})"
    elif overall_bias == "Bearish":
        action = "SELL"
        reasoning = f"Bearish bias ({overall_bias}) detected (conf: {avg_confidence:.0%})"
    else:
        action = "HOLD"
        reasoning = f"Mixed/Neutral signals ({overall_bias}), awaiting clearer direction"
    
    # Adjust for high-risk scenarios
    if risks and len(risks) > 1:
        action = "HOLD"
        reasoning += f". Multiple risks identified: {', '.join(risks[:2])}"
    
    return {
        "action": action,
        "confidence": round(avg_confidence / 100, 2),  # Normalize to 0-1 for frontend
        "bullish_signals": bullish_count,
        "bearish_signals": bearish_count,
        "reasoning": reasoning,
        "risks": risks,
    }


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

        return {
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}

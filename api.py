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
            "errors":        errors,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}

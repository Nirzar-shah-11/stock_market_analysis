# Stock Market Predictor

A multi-agent stock price prediction system for NSE (National Stock Exchange) India that leverages **LangGraph**, **Llama 3 LLM**, technical analysis, options data, and historical pattern matching to generate buy/sell/hold recommendations.

## Overview

This project combines multiple data sources and analytical methods to predict short-term (1d/3d/7d) stock price movements:

- **Technical Signals** — 10+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Options Chain (OI)** — Put/call ratios, support/resistance from open interest
- **Subtle Hints** — Bulk deals, insider trades, FII/DII flows, SEC filings
- **Correlation Analysis** — Peer stock movements and market correlation
- **Historical Patterns** — Matches today's signals against past similar clusters
- **LLM Synthesis** — Llama 3 synthesizes all signals into a unified prediction
- **Recommendation** — Final BUY/SELL/HOLD action with confidence score

## Architecture

### FastAPI Server (`api.py`)
- **POST `/predict`** — Takes company name + NSE symbol, returns full prediction
- **GET `/`** — Serves web UI (HTML frontend)
- **GET `/health`** — Health check

### Prediction Graph (`graph.py`)
A LangGraph workflow orchestrating multiple analysis nodes sequentially:

```
Price Node → Technical Node → OI Node → Hints Node 
   ↓             ↓              ↓          ↓
[Price Data]  [Tech Signals] [OI Signals] [Hints]
   ↓             ↓              ↓          ↓
            ↓ Correlation Node ↓ Pattern Node ↓ News Node
                   ↓                 ↓              ↓
                [Correlation]   [Patterns]    [News Articles]
                   ↓                 ↓              ↓
            → Prediction Node (Llama) →
                   ↓
            [Prediction + Recommendation]
```

### Analysis Nodes (`nodes/`)

| Node | Purpose |
|------|---------|
| `price_node.py` | Fetches today's OHLCV and 60-day history |
| `technical_node.py` | Computes 10 technical indicators |
| `oi_node.py` | Analyzes options chain (put/call ratios) |
| `hints_node.py` | Extracts bulk deals, insider trades, FII/DII, filings |
| `correlation_node.py` | Identifies peer stocks and correlation signals |
| `pattern_node.py` | Matches today's signals against historical patterns |
| `news_node.py` | Fetches recent news articles |
| `prediction_node.py` | Sends all data to Llama 3 for final synthesis |
| `orchestrator_node.py` | Handles routing and node sequencing |

### State Management (`state.py`)
A `PredictionState` TypedDict holds all data flowing through the graph:
- Inputs: company_name, symbol
- Price data: today_price, history_df, trading_dates
- Today's signals: today_tech, today_oi, today_hints
- Analysis: similar_patterns, correlation_signals
- Output: prediction, recommendation, errors

### Frontend (`templates/index.html`)
Interactive web UI displaying:
- Price & bias badge
- Signal counts (technical, OI, hints, correlation, patterns)
- **Recommendation card** (BUY/SELL/HOLD with confidence)
- 1d/3d/7d predictions with direction & magnitude
- Signal trace with full reasoning
- Key drivers & risks
- Historical patterns & correlation context

## 🚀 Setup

### Prerequisites
- Python 3.12+
- Virtual environment (`venv` or `conda`)
- Ollama with Llama 3.8B (for local LLM) OR API key for cloud LLM

### Installation

1. **Clone & Navigate**
   ```bash
   cd stock_market_analysis
   ```

2. **Activate Virtual Environment**
   ```bash
   source stockmarketanalysys/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment**
   Create `.env` file:
   ```env
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3:8b
   ```

5. **Start Ollama** (if using local model)
   ```bash
   ollama run llama3:8b
   ```

6. **Run Server**
   ```bash
   uvicorn api:app --reload --port 8000
   ```

7. **Access Web UI**
   Open `http://localhost:8000` in browser

## 📊 Usage

### API Endpoint

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"company": "Reliance Industries", "symbol": "RELIANCE"}'
```

**Response:**
```json
{
  "symbol": "RELIANCE",
  "company": "Reliance Industries",
  "run_timestamp": "2026-05-07T14:30:45.123456",
  "today_price": {
    "open": 2850,
    "high": 2865,
    "low": 2840,
    "close": 2855,
    "volume": 45000000,
    "pct_change": 0.18
  },
  "signal_counts": {
    "technical": 6,
    "oi": 3,
    "hints": 2,
    "news": 4,
    "correlation": 2,
    "patterns": 1
  },
  "prediction": {
    "horizon_1d": {
      "direction": "Up",
      "magnitude_pct": 1.5,
      "confidence": 72,
      "reasoning": "RSI overbought but volume breakout suggests continuation"
    },
    "horizon_3d": {
      "direction": "Up",
      "magnitude_pct": 2.8,
      "confidence": 65,
      "reasoning": "3 matching patterns from past 18 months all led to +2-3% moves"
    },
    "horizon_7d": {
      "direction": "Sideways",
      "magnitude_pct": 0.5,
      "confidence": 58,
      "reasoning": "Strong peer correlation suggests sector consolidation"
    },
    "overall_bias": "Bullish",
    "key_drivers": ["Volume breakout", "RSI recovery", "Positive peer correlation"],
    "risks": ["Profit booking likely", "Global indices weakness"],
    "signal_trace": [
      {
        "signal": "RSI(14)",
        "direction": "Bullish",
        "weight": "High",
        "interpretation": "RSI moved above 60, showing momentum"
      }
    ]
  },
  "recommendation": {
    "action": "BUY",
    "confidence": 0.69,
    "bullish_signals": 6,
    "bearish_signals": 2,
    "reasoning": "Strong bullish bias (Bullish) with 6/8 bullish signals (conf: 69%)"
  },
  "errors": []
}
```

### Web UI
1. Enter company name (e.g., "Suzlon Energy")
2. Enter NSE symbol (e.g., "SUZLON")
3. Click "Predict"
4. View results with recommendation prominently displayed

## How It Works

### Data Collection Pipeline
1. **Price Data** — Last 60 days OHLCV from data source (jugaad-data)
2. **Technical Analysis** — 10 indicators computed on historical data
3. **Options Chain** — Current put/call open interest and volumes
4. **Hints** — Bulk deals, insider trades, FII/DII flows
5. **Correlation** — Fetches peer stocks (e.g., HDFCBANK, TCS for RELIANCE)
6. **Patterns** — Searches historical data for similar signal clusters
7. **News** — Latest articles mentioning the stock

### LLM Synthesis
The prediction node constructs a detailed prompt with all signals and sends to Llama 3:

```
You are a stock market analyst. Given these signals for [STOCK]:
- Technical: [RSI, MACD, Bollinger Bands data]
- Options: [Put/Call ratios]
- Hints: [Bulk deals, insider trades]
- Correlation: [Peer stocks moving X%]
- Patterns: [3 similar clusters from past]

Predict 1d, 3d, 7d direction and magnitude. Explain your reasoning.
```

### Recommendation Logic
- **BUY**: Bullish bias + high confidence + >60% bullish signal ratio
- **SELL**: Bearish bias + high confidence + >60% bearish signal ratio
- **HOLD**: Mixed/Neutral signals OR multiple risks identified

## Dependencies

Key packages (see `requirements.txt`):
- `fastapi` — Web framework
- `langgraph` — Multi-agent orchestration
- `langchain_mcp` — LLM integration
- `httpx` — HTTP client for APIs
- `jugaad_data` — NSE data fetching
- `beautifulsoup4` — Web scraping for news

## Configuration

Edit `api.py` or `.env` to customize:
- LLM model (local Ollama or cloud API)
- Technical indicator parameters
- OI thresholds
- Correlation peer selection
- Pattern matching window

## Sample Predictions

### Example 1: RELIANCE (May 7, 2026)
- **Bias**: Bullish
- **1d**: Up ~1.5% (72% conf)
- **Recommendation**: **BUY** (69% conf)
- **Drivers**: Volume breakout, RSI recovery, positive peer correlation
- **Risks**: Profit booking, global weakness

### Example 2: INFY (May 7, 2026)
- **Bias**: Bearish
- **1d**: Down ~0.8% (61% conf)
- **Recommendation**: **SELL** (58% conf)
- **Drivers**: Bearish divergence, OI distribution suggests weakness
- **Risks**: Support at MA200, potential reversal

## Warnings & Disclaimers

- **Not financial advice** — Use as research tool only
- **Market volatility** — Predictions are probabilistic, not guaranteed
- **Data latency** — Some signals are end-of-day or delayed
- **Model limitations** — Llama 3 has knowledge cutoff; recent events may affect accuracy
- **Testing required** — Backtest strategies before live trading

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No explicit representation of timezones" warning | Already filtered in api.py (line 12) |
| API times out | Increase timeout, check Ollama is running |
| "Symbol not found" | Verify NSE symbol (e.g., RELIANCE not REL) |
| Empty signals | Check data source availability during market hours |

## Project Files

```
stock_market_analysis/
├── api.py                 # FastAPI server + recommendation logic
├── graph.py               # LangGraph prediction workflow
├── state.py               # Prediction state definition
├── llm_client.py          # LLM integration
├── news_client.py         # News fetcher
├── nodes/                 # Analysis nodes
│   ├── price_node.py
│   ├── technical_node.py
│   ├── oi_node.py
│   ├── hints_node.py
│   ├── correlation_node.py
│   ├── pattern_node.py
│   ├── prediction_node.py
│   └── orchestrator_node.py
├── templates/
│   └── index.html         # Web UI
├── static/                # CSS, JS assets
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Future Enhancements

- [ ] Backtesting engine with historical accuracy metrics
- [ ] Portfolio-level predictions (not just single stocks)
- [ ] Sentiment analysis from news/social media
- [ ] Risk-adjusted position sizing recommendations
- [ ] Integration with live broker APIs for execution
- [ ] Real-time signal updates (WebSocket)
- [ ] Export predictions to CSV/database

## Support

For issues or questions:
1. Check logs in terminal
2. Verify `.env` configuration
3. Ensure data sources are accessible during market hours
4. Test with quick symbols (RELIANCE, INFY, TCS)

---

**Last Updated**: July 7, 2026  
**Version**: 1.1  
**Author**: Stock Market Analysis Team

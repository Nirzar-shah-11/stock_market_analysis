"""nodes/orchestrator_node.py — F&O check + routing decisions"""
import time, requests
from colorama import Fore, Style
from state import PredictionState

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.nseindia.com/",
    "Accept": "application/json, text/plain, */*",
}

KNOWN_FO = {
    "RELIANCE","INFY","TCS","HDFCBANK","ICICIBANK","SBIN","HINDUNILVR",
    "BAJFINANCE","MARUTI","TATAMOTORS","WIPRO","AXISBANK","KOTAKBANK",
    "LT","TITAN","SUNPHARMA","ADANIENT","ADANIPORTS","ULTRACEMCO",
    "ASIANPAINT","HCLTECH","TECHM","POWERGRID","NTPC","ONGC",
    "ZOMATO","SUZLON","TATAPOWER","IRCTC","NYKAA","PAYTM",
}

def orchestrator_node(state: PredictionState) -> dict:
    symbol = state["symbol"].upper()
    _log(f"Orchestrator: {state['company_name']} ({symbol})")
    is_fo = _check_fo(symbol)
    _log(f"F&O stock: {is_fo}")
    return {"is_fo_stock": is_fo}

def _check_fo(symbol: str) -> bool:
    try:
        s = requests.Session()
        s.headers.update(NSE_HEADERS)
        s.get("https://www.nseindia.com", timeout=8)
        time.sleep(0.5)
        r = s.get(f"https://www.nseindia.com/api/quote-derivative?symbol={symbol}", timeout=10)
        if r.status_code == 200:
            d = r.json()
            return bool(d.get("stocks") or d.get("fut_timestamp"))
    except Exception:
        pass
    return symbol in KNOWN_FO

def route_after_orchestrator(state: PredictionState) -> str:
    return "price_node"

def route_after_merge(state: PredictionState) -> str:
    has = bool(state.get("today_tech") or state.get("today_hints") or state.get("today_oi"))
    return "prediction_node" if has else "prediction_node"  # always predict

def _log(msg, warn=False):
    print(f"{Fore.CYAN}[Orchestrator] {Fore.YELLOW if warn else Fore.CYAN}{msg}{Style.RESET_ALL}")

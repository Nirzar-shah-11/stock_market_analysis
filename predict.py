"""
predict.py — CLI Entry Point
  python predict.py --company "Suzlon Energy" --symbol SUZLON
"""

import os, sys, json, argparse
from dotenv import load_dotenv
from colorama import Fore, Style, init as colorama_init
from state import fresh_state
from graph import graph

colorama_init(autoreset=True)
load_dotenv()

def banner():
    print(f"""
{Fore.CYAN}════════════════════════════════════════════════════════════════
   Stock Price Predictor  ·  LangGraph  ·  Ollama Llama 3.8b
   Technical · OI · Hints · Correlation · Pattern Match
════════════════════════════════════════════════════════════════{Style.RESET_ALL}
""")

def print_prediction(pred: dict, symbol: str):
    if not pred:
        print(f"{Fore.RED}No prediction generated.{Style.RESET_ALL}")
        return

    bias = pred.get("overall_bias","?")
    bias_color = (Fore.GREEN if bias == "Bullish" else
                  Fore.RED   if bias == "Bearish" else Fore.YELLOW)

    print(f"\n{Fore.CYAN}{'═'*60}{Style.RESET_ALL}")
    print(f"  {symbol}  —  Overall Bias: {bias_color}{bias}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'═'*60}{Style.RESET_ALL}\n")

    for horizon, label in [("horizon_1d","1 Day"), ("horizon_3d","3 Days"), ("horizon_7d","7 Days")]:
        h = pred.get(horizon, {})
        d = h.get("direction","?")
        m = h.get("magnitude_pct","?")
        c = h.get("confidence","?")
        r = h.get("reasoning","")
        dcolor = Fore.GREEN if d == "Up" else (Fore.RED if d == "Down" else Fore.YELLOW)
        print(f"  {Fore.WHITE}{label:8s}{Style.RESET_ALL}  {dcolor}{d:8s}{Style.RESET_ALL}  "
              f"~{m}%  confidence={c}%")
        print(f"           {Fore.WHITE}{r}{Style.RESET_ALL}\n")

    print(f"{Fore.CYAN}Key Drivers:{Style.RESET_ALL}")
    for d in pred.get("key_drivers",[]):
        print(f"  • {d}")

    print(f"\n{Fore.YELLOW}Risks:{Style.RESET_ALL}")
    for r in pred.get("risks",[]):
        print(f"  • {r}")

    print(f"\n{Fore.CYAN}Signal Trace:{Style.RESET_ALL}")
    for s in pred.get("signal_trace",[]):
        w     = s.get("weight","?")
        dirn  = s.get("direction","?")
        dcolor = Fore.GREEN if dirn == "Bullish" else (Fore.RED if dirn == "Bearish" else Fore.YELLOW)
        print(f"  [{w:6s}] {dcolor}{dirn:8s}{Style.RESET_ALL}  {s.get('signal','')}")
        print(f"           {s.get('interpretation','')}")

    if pred.get("pattern_match_summary"):
        print(f"\n{Fore.CYAN}Historical Patterns:{Style.RESET_ALL}")
        print(f"  {pred['pattern_match_summary']}")

    if pred.get("correlation_summary"):
        print(f"\n{Fore.CYAN}Peer Correlation:{Style.RESET_ALL}")
        print(f"  {pred['correlation_summary']}")

    print(f"\n{Fore.CYAN}{'═'*60}{Style.RESET_ALL}\n")

if __name__ == "__main__":
    banner()
    p = argparse.ArgumentParser()
    p.add_argument("--company", "-c", required=True)
    p.add_argument("--symbol",  "-s", required=True)
    p.add_argument("--json",    action="store_true", help="Output raw JSON")
    args = p.parse_args()

    state  = fresh_state(args.company, args.symbol.upper())
    final  = graph.invoke(state)
    pred   = final.get("prediction", {})

    if args.json:
        print(json.dumps(pred, indent=2))
    else:
        print_prediction(pred, args.symbol.upper())

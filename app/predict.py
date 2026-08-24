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
def print_news(articles: list):
    if not articles:
        return

    print(f"\n{Fore.CYAN}{'═'*60}{Style.RESET_ALL}")
    print(f"  NEWS SOURCES REFERENCED  ({len(articles)} articles)")
    print(f"{Fore.CYAN}{'═'*60}{Style.RESET_ALL}")

    for i, art in enumerate(articles, 1):
        sent     = art.get("sentiment_hint", "neutral")
        color    = SENT_COLOR.get(sent, Fore.WHITE)
        icon     = SENT_ICON.get(sent, "◆")
        source   = art.get("source", "Unknown")
        date_d   = art.get("date_display", art.get("date", ""))
        title    = art.get("title", "")
        url      = art.get("url", "")
        point1   = art.get("point1", "")
        point2   = art.get("point2", "")

        print(f"\n  {Fore.WHITE}{i:2d}.{Style.RESET_ALL} "
              f"{Fore.BLUE}{source:<28}{Style.RESET_ALL} "
              f"{Fore.WHITE}{date_d}{Style.RESET_ALL}  "
              f"{color}{icon} {sent.capitalize()}{Style.RESET_ALL}")

        # Title — truncate if too long for terminal
        title_display = title if len(title) <= 70 else title[:67] + "…"
        print(f"      {Fore.WHITE}{title_display}{Style.RESET_ALL}")

        # URL — dimmed
        if url:
            url_display = url if len(url) <= 72 else url[:69] + "…"
            print(f"      {Fore.WHITE}\033[2m{url_display}\033[0m{Style.RESET_ALL}")

        # Bullet points
        if point1:
            print(f"      {Fore.CYAN}•{Style.RESET_ALL} {point1}")
        if point2:
            print(f"      {Fore.CYAN}•{Style.RESET_ALL} {point2}")

    print(f"\n{Fore.CYAN}{'═'*60}{Style.RESET_ALL}")
    
# ── Prediction section ────────────────────────────────────────────────────────

def print_prediction(pred: dict, symbol: str):
    if not pred:
        print(f"{Fore.RED}No prediction generated.{Style.RESET_ALL}")
        return

    bias       = pred.get("overall_bias", "?")
    bias_color = (Fore.GREEN  if bias == "Bullish" else
                  Fore.RED    if bias == "Bearish" else Fore.YELLOW)

    print(f"\n{Fore.CYAN}{'═'*60}{Style.RESET_ALL}")
    print(f"  {symbol}  —  Overall Bias: {bias_color}{bias}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'═'*60}{Style.RESET_ALL}\n")

    # Horizons
    for horizon, label in [
        ("horizon_1d", "1 Day "),
        ("horizon_3d", "3 Days"),
        ("horizon_7d", "7 Days"),
    ]:
        h      = pred.get(horizon, {})
        d      = h.get("direction", "?")
        m      = h.get("magnitude_pct", "?")
        c      = h.get("confidence", "?")
        r      = h.get("reasoning", "")
        dcolor = (Fore.GREEN  if d == "Up"       else
                  Fore.RED    if d == "Down"     else Fore.YELLOW)
        icon   = "▲" if d == "Up" else ("▼" if d == "Down" else "◆")
        print(f"  {Fore.WHITE}{label}{Style.RESET_ALL}  "
              f"{dcolor}{icon} {d:<9}{Style.RESET_ALL}  "
              f"~{m}%   confidence {c}%")
        if r:
            # Word-wrap reasoning at 56 chars
            words, line = r.split(), ""
            for w in words:
                if len(line) + len(w) + 1 > 56:
                    print(f"           {Fore.WHITE}{line}{Style.RESET_ALL}")
                    line = w
                else:
                    line = (line + " " + w).strip()
            if line:
                print(f"           {Fore.WHITE}{line}{Style.RESET_ALL}")
        print()

    # Key drivers
    drivers = pred.get("key_drivers", [])
    if drivers:
        print(f"{Fore.CYAN}Key Drivers:{Style.RESET_ALL}")
        for d in drivers:
            print(f"  {Fore.GREEN}▲{Style.RESET_ALL} {d}")

    # Risks
    risks = pred.get("risks", [])
    if risks:
        print(f"\n{Fore.YELLOW}Risks:{Style.RESET_ALL}")
        for r in risks:
            print(f"  {Fore.RED}⚠{Style.RESET_ALL} {r}")

    # Signal trace
    trace = pred.get("signal_trace", [])
    if trace:
        print(f"\n{Fore.CYAN}Signal Trace:{Style.RESET_ALL}")
        print(f"  {'Weight':<8} {'Direction':<10} Signal")
        print(f"  {'─'*8} {'─'*10} {'─'*35}")
        for s in trace:
            w      = s.get("weight", "?")
            dirn   = s.get("direction", "?")
            sig    = s.get("signal", "")
            interp = s.get("interpretation", "")
            dcolor = (Fore.GREEN  if dirn == "Bullish" else
                      Fore.RED    if dirn == "Bearish" else Fore.YELLOW)
            print(f"  {Fore.WHITE}{w:<8}{Style.RESET_ALL} "
                  f"{dcolor}{dirn:<10}{Style.RESET_ALL} {sig}")
            if interp:
                print(f"           {Fore.WHITE}\033[2m{interp[:65]}\033[0m{Style.RESET_ALL}")

    # Pattern + correlation summaries
    if pred.get("pattern_match_summary"):
        print(f"\n{Fore.CYAN}Historical Patterns:{Style.RESET_ALL}")
        print(f"  {pred['pattern_match_summary']}")

    if pred.get("correlation_summary"):
        print(f"\n{Fore.CYAN}Peer Correlation:{Style.RESET_ALL}")
        print(f"  {pred['correlation_summary']}")

    print(f"\n{Fore.CYAN}{'═'*60}{Style.RESET_ALL}\n")

if __name__ == "__main__":
    banner()

    p = argparse.ArgumentParser(description="NSE Stock Predictor")
    p.add_argument("--company", "-c", required=True,
                   help='Company name e.g. "Suzlon Energy"')
    p.add_argument("--symbol",  "-s", required=True,
                   help="NSE symbol e.g. SUZLON")
    p.add_argument("--json", action="store_true",
                   help="Dump raw JSON output instead of formatted display")
    args = p.parse_args()


    print(f"  Company : {args.company}")
    print(f"  Symbol  : {args.symbol.upper()}\n")

    state = fresh_state(args.company, args.symbol.upper())
    final = graph.invoke(state)

    pred     = final.get("prediction", {})
    articles = final.get("news_articles", [])
    errors   = final.get("errors", [])
    price    = final.get("today_price", {})

    if args.json:
        print(json.dumps({
            "symbol":        args.symbol.upper(),
            "company":       args.company,
            "today_price":   price,
            "news_articles": articles,
            "prediction":    pred,
            "errors":        errors,
        }, indent=2))
    else:
        # Print today's price summary
        if price:
            pct = price.get("pct_change", 0) or 0
            pc  = Fore.GREEN if pct >= 0 else Fore.RED
            print(f"  Today: ₹{price.get('close','?')}  "
                  f"{pc}{pct:+.2f}%{Style.RESET_ALL}  "
                  f"Vol: {price.get('volume', 0):,}")

        # 1. News articles with sources + bullet points
        print_news(articles)

        # 2. Prediction
        print_prediction(pred, args.symbol.upper())

    # Errors / warnings
    if errors:
        print(f"{Fore.YELLOW}Warnings:{Style.RESET_ALL}")
        for e in errors:
            print(f"  • {e}")

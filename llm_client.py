"""llm_client.py — Ollama Llama 3.8B wrapper"""
import time, requests
from colorama import Fore, Style

OLLAMA_MODEL = "llama3:70b"
OLLAMA_API_URL = "http://localhost:11434/api/chat"

class LLMClient:
    def __init__(self, api_token: str = None):
        # api_token is not used for Ollama, kept for compatibility
        pass

    def chat(self, system_prompt: str, user_prompt: str,
             max_tokens: int = 2000, temperature: float = 0.2,
             retries: int = 3) -> str:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user",   "content": user_prompt}],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        for attempt in range(1, retries + 1):
            try:
                r = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
                r.raise_for_status()
                return r.json()["message"]["content"].strip()
            except requests.exceptions.ConnectionError:
                print(f"{Fore.YELLOW}[LLM] Ollama not running on localhost:11434{Style.RESET_ALL}")
                time.sleep(5 * attempt)
            except requests.exceptions.Timeout:
                print(f"{Fore.YELLOW}[LLM] Timeout, retrying…{Style.RESET_ALL}")
                time.sleep(15 * attempt)
            except Exception as e:
                print(f"{Fore.YELLOW}[LLM] Error {attempt}/{retries}: {e}{Style.RESET_ALL}")
                time.sleep(5 * attempt)
        return ""

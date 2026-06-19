import json
import requests
from typing import Optional

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_LLM = "phi4-mini"
DEFAULT_EMBED = "nomic-embed-text"


def llm_complete(prompt: str, model: str = DEFAULT_LLM, system: Optional[str] = None,
                 temperature: float = 0, max_tokens: int = 512) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json={"model": model, "messages": messages, "stream": False,
              "options": {"temperature": temperature, "num_predict": max_tokens,
                          "keep_alive": "10m"}},
        timeout=600
    )
    return resp.json()["message"]["content"]


def embed(texts: list[str], model: str = DEFAULT_EMBED) -> list[list[float]]:
    resp = requests.post(
        f"{OLLAMA_BASE}/api/embed",
        json={"model": model, "input": texts},
        timeout=120
    )
    return resp.json()["embeddings"]

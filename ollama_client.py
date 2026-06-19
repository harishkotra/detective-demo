import json
import os
import requests
from typing import Optional

INFERENCE_PROVIDER = os.environ.get("INFERENCE_PROVIDER", "lm_studio")

PROVIDERS = {
    "lm_studio": {
        "base_url": "http://127.0.0.1:1234",
        "needs_api_key": False,
        "default_llm": "qwen/qwen3.5-9b",
        "default_llm_small": "google/gemma-4-e4b",
        "default_embed": "text-embedding-nomic-embed-text-v1.5:2",
    },
    "ollama": {
        "base_url": "http://127.0.0.1:11434",
        "needs_api_key": False,
        "default_llm": "llama3.2",
        "default_llm_small": "llama3.2:1b",
        "default_embed": "nomic-embed-text",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "needs_api_key": True,
        "default_llm": "gpt-4o-mini",
        "default_llm_small": "gpt-4o-mini",
        "default_embed": "text-embedding-3-small",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "needs_api_key": True,
        "default_llm": "meta-llama/llama-3.2-3b-instruct:free",
        "default_llm_small": "meta-llama/llama-3.2-3b-instruct:free",
        "default_embed": None,
    },
    "featherless": {
        "base_url": "https://api.featherless.ai/v1",
        "needs_api_key": True,
        "default_llm": "meta-llama/Llama-3.2-3B-Instruct",
        "default_llm_small": "meta-llama/Llama-3.2-3B-Instruct",
        "default_embed": None,
    },
}

_cfg = {
    "provider": INFERENCE_PROVIDER,
    "base_url": None,
    "api_key": None,
    "llm_model": None,
    "llm_model_small": None,
    "embed_model": None,
}


def configure(
    provider: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    llm_model: str | None = None,
    llm_model_small: str | None = None,
    embed_model: str | None = None,
):
    if provider is not None:
        _cfg["provider"] = provider
    if base_url is not None:
        _cfg["base_url"] = base_url
    if api_key is not None:
        _cfg["api_key"] = api_key
    if llm_model is not None:
        _cfg["llm_model"] = llm_model
    if llm_model_small is not None:
        _cfg["llm_model_small"] = llm_model_small
    if embed_model is not None:
        _cfg["embed_model"] = embed_model


def _resolve(key: str, default_key: str) -> str:
    explicit = _cfg.get(key)
    if explicit:
        return explicit
    env_val = os.environ.get(
        {
            "base_url": "INFERENCE_BASE_URL",
            "api_key": None,
            "llm_model": "LLM_MODEL",
            "llm_model_small": "LLM_MODEL_SMALL",
            "embed_model": "EMBED_MODEL",
        }.get(key, "")
    )
    if env_val:
        return env_val
    provider = _cfg.get("provider", INFERENCE_PROVIDER)
    provider_cfg = PROVIDERS.get(provider, PROVIDERS["lm_studio"])
    return provider_cfg.get(default_key)


def _base_url() -> str:
    return _resolve("base_url", "base_url")


def _api_key() -> str | None:
    explicit = _cfg.get("api_key")
    if explicit:
        return explicit
    provider = _cfg.get("provider", INFERENCE_PROVIDER)
    provider_cfg = PROVIDERS.get(provider, PROVIDERS["lm_studio"])
    env_key_name = {
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "featherless": "FEATHERLESS_API_KEY",
    }.get(provider)
    if env_key_name:
        return os.environ.get(env_key_name)
    return None


def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    key = _api_key()
    if key:
        h["Authorization"] = f"Bearer {key}"
    if _cfg.get("provider", INFERENCE_PROVIDER) == "openrouter":
        h["HTTP-Referer"] = "https://github.com/harishkotra/detective-demo"
        h["X-Title"] = "Detective RAG Demo"
    return h


def llm_complete(
    prompt: str,
    model: str | None = None,
    system: Optional[str] = None,
    temperature: float = 0,
    max_tokens: int = 2048,
) -> str:
    if model is None:
        model = _resolve("llm_model", "default_llm")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = requests.post(
        f"{_base_url()}/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=600,
        headers=_headers(),
    )
    msg = resp.json()["choices"][0]["message"]
    content = msg.get("content", "")
    if not content and "reasoning_content" in msg:
        content = msg["reasoning_content"]
    return content.strip()


def embed(texts: list[str], model: str | None = None) -> list[list[float]]:
    if isinstance(texts, str):
        texts = [texts]
    if model is None:
        model = _resolve("embed_model", "default_embed")
    if model is None:
        raise ValueError(
            f"Provider '{_cfg.get('provider')}' has no embedding model configured. "
            "Set one via configure(embed_model=...) or EMBED_MODEL env var."
        )
    resp = requests.post(
        f"{_base_url()}/v1/embeddings",
        json={"model": model, "input": texts},
        timeout=120,
        headers=_headers(),
    )
    data = resp.json()
    embeddings = [d["embedding"] for d in data["data"]]
    return embeddings

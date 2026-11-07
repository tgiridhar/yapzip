"""Model call. Today: HTTP to a local llama-server. The native binary swaps this for linked libllama."""

import os

import httpx

URL = os.environ.get("YAPZIP_URL", "http://127.0.0.1:8080")


def complete(msgs: list[dict], max_tokens: int, grammar: str | None = None) -> tuple[str, dict]:
    body = {
        "messages": msgs,
        "max_tokens": max_tokens,
        "temperature": 0,
        "cache_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
        # DRY penalizes repeating recent sequences (stops line loops); the short window
        # keeps it from penalizing copies of the source text further back in the prompt.
        "dry_multiplier": 0.8,
        "dry_base": 1.75,
        "dry_allowed_length": 4,
        "dry_penalty_last_n": 96,
    }
    if grammar:
        body["grammar"] = grammar
    r = httpx.post(f"{URL}/v1/chat/completions", json=body, timeout=300)
    r.raise_for_status()
    d = r.json()
    text = d["choices"][0]["message"].get("content") or ""
    return text, d.get("timings", {})

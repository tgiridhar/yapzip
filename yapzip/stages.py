"""Two-stage path from the design: (1) keep/drop read from KEEP vs DROP next-token probabilities,
no text generated; (2) rewrite only the kept lines, one at a time."""

import math
import os

import httpx

from .backend import URL

# yes/no framing: AUC 0.92 on the test lines vs 0.66 for a KEEP/DROP framing (bench/keepdrop_auc.py)
KEEP_SYSTEM = "Answer yes or no."
KEEP_USER = "Does this line contain a useful fact (a number, date, decision, plan, problem or request)?\n\n{line}"

REWRITE_SYSTEM = """Rewrite the line as a telegraph note for another AI. Keep facts, drop talk.
- Max 15 words. No articles. No filler ("I think", "basically", "just", "honestly").
- Copy names, numbers, dates, prices and versions exactly.
- Write NOT for every negation. Write IF / ONLY IF for conditions. Write -> for a change.
- Keep the speaker name if the line has one.
- Output only the note."""

SHORT_WORDS = int(os.environ.get("YAPZIP_SHORT_WORDS", "12"))   # kept lines this short pass through as-is


def _chat(system: str, user: str, **kw) -> dict:
    body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0, "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False}, **kw}
    r = httpx.post(f"{URL}/v1/chat/completions", json=body, timeout=120)
    r.raise_for_status()
    return r.json()


def p_keep(line: str, prev: str = "") -> float:
    """P(yes) / (P(yes) + P(no)) from the first output token's top logprobs. No text is generated."""
    d = _chat(KEEP_SYSTEM, KEEP_USER.format(line=line), max_tokens=1, logprobs=True, top_logprobs=20)
    top = d["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    py = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower() == "yes")
    pn = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower() == "no")
    return py / (py + pn) if py + pn else 0.5


def rewrite(line: str) -> tuple[str, dict]:
    if len(line.split()) <= SHORT_WORDS:
        return line, {}
    d = _chat(REWRITE_SYSTEM, line, max_tokens=60,
              dry_multiplier=0.8, dry_base=1.75, dry_allowed_length=4, dry_penalty_last_n=64)
    text = (d["choices"][0]["message"].get("content") or "").strip().splitlines()
    return (text[0].strip() if text else line), d.get("timings", {})

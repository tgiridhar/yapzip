"""How well does untuned Qwen separate fact lines from chatter? ROC-AUC per scoring method.

Gold per line: positive if it contains a must-keep fact string, negative if it contains a
drop span and no fact. Other lines are unlabeled and skipped.
"""

import json
import math
import re
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.common import GOLD, RAW, RESULTS  # noqa: E402

URL = "http://127.0.0.1:8080/v1/chat/completions"
PROMPTS = {
    "keepdrop": ("You label lines. KEEP = has a fact, number, date, name, decision or request. "
                 "DROP = greeting, thanks, filler, joke, reaction, agreement only. Answer KEEP or DROP.",
                 "Line: {l}", ("KEEP", "DROP")),
    "yesno": ("Answer yes or no.",
              "Does this line contain a useful fact (a number, date, decision, plan, problem or request)?\n\n{l}",
              ("yes", "no")),
}


def prob(system, user, a, b):
    d = httpx.post(URL, json={"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                              "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20,
                              "chat_template_kwargs": {"enable_thinking": False}}, timeout=60).json()
    top = d["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    match = lambda w, t: bool(t) and w.lower().startswith(t)  # noqa: E731
    pa = sum(math.exp(t["logprob"]) for t in top if match(a, t["token"].strip().lower()))
    pb = sum(math.exp(t["logprob"]) for t in top if match(b, t["token"].strip().lower()))
    return pa / (pa + pb) if pa + pb else 0.5


def heuristic(line):
    return (1.0 if re.search(r"\d", line) else 0.0) + min(len(line.split()), 30) / 30


def auc(pos, neg):
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main():
    rows = []
    for g in sorted(GOLD.glob("*.json")):
        gold = json.loads(g.read_text())
        lines = [l.strip() for l in (RAW / f"{g.stem}.txt").read_text().splitlines() if l.strip()]
        musts = [m for f in gold["facts"] for m in f["must"]]
        for l in lines:
            y = 1 if any(m in l for m in musts) else 0 if any(s in l for s in gold["drop_spans"]) else None
            if y is not None:
                rows.append({"file": g.stem, "line": l, "y": y})
    print(f"{len(rows)} labeled lines: {sum(r['y'] for r in rows)} fact, {sum(1 - r['y'] for r in rows)} chatter")
    t0 = time.time()
    for r in rows:
        r["heuristic"] = heuristic(r["line"])
        for name, (s, u, (a, b)) in PROMPTS.items():
            r[name] = prob(s, u.format(l=r["line"]), a, b)
    per_call = (time.time() - t0) / (len(rows) * len(PROMPTS))
    out = {}
    for m in ["heuristic", *PROMPTS]:
        out[m] = round(auc([r[m] for r in rows if r["y"]], [r[m] for r in rows if not r["y"]]), 3)
    out["ms_per_line"] = round(per_call * 1000)
    print(json.dumps(out))
    (RESULTS / "keepdrop_auc.json").write_text(json.dumps({"auc": out, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()

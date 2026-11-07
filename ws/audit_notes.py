"""Automatic quality audit of training notes: is each kept note faithful to its unit?

    python3 ws/audit_notes.py FILE.jsonl [FILE2.jsonl ...] [--show N]

Per file it reports, over kept units whose note is not "=":
- invented: the note has a content word, number, negation, condition or speaker the unit doesn't have (ws/halluc.py)
- lost number: a number in the unit is missing from the note
- lost negation: the unit has a negation (not, never, no, n't, without...) and the note has none
- code not copied: a code-like unit (fence, shell prompt, config/JSON/log line) whose note is not "="
- size: note chars / unit chars (median)
"""

import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ws.halluc import NEG, NUM, check  # noqa: E402

CODE = re.compile(r"^(```|\$ |> |#!|[\w.-]+\s*[:=]\s*\S+$|\{|\[|<\w|[A-Z]{2,}\s+\d|\d{4}-\d\d-\d\d[ T]\d)")


def code_like(text):
    t = text.strip()
    return t.startswith("```") or bool(CODE.match(t)) and len(t.split()) <= 12


def audit(units):
    kept = [u for u in units if u.get("keep")]
    prose = [u for u in kept if u.get("note") and u["note"].strip() != "="]
    out = {"units": len(units), "kept": len(kept), "copy(=)": len(kept) - len(prose), "notes": len(prose)}
    bad = {"invented": [], "lost number": [], "lost negation": [], "code not copied": []}
    ratios = []
    for u in prose:
        t, n = u["text"], u["note"]
        if check(t, n):
            bad["invented"].append((t, n, check(t, n)))
        nums = [x for x in NUM.findall(t) if len(x) > 1 or x not in "0123456789"]
        if any(x not in n and x.replace(",", "") not in n.replace(",", "") for x in nums):
            bad["lost number"].append((t, n, [x for x in nums if x not in n]))
        if NEG.search(t) and not re.search(r"\b(NOT|NEVER|NO|NONE|not|never|no|none|w/o|without|nobody|nothing|"
                                           r"n't|cannot|can't|don't|won't|isn't|neither|nor|zero|stop|avoid)\b|n't",
                                           n, re.I):
            bad["lost negation"].append((t, n, []))
        ratios.append(len(n) / max(1, len(t)))
    for u in kept:
        if code_like(u["text"]) and (u.get("note") or "").strip() != "=":
            bad["code not copied"].append((u["text"], u.get("note"), []))
    for k, v in bad.items():
        out[k] = f"{len(v)} ({100 * len(v) / max(1, len(prose) if k != 'code not copied' else len(kept)):.1f}%)"
    out["size (median note/unit)"] = round(statistics.median(ratios), 2) if ratios else None
    return out, bad


def main():
    files = [a for a in sys.argv[1:] if not a.startswith("--")]
    show = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--show=")), 0))
    for f in files:
        units = []
        for line in open(f):
            c = json.loads(line)
            units += c["units"]
        out, bad = audit(units)
        print(f"== {f}")
        for k, v in out.items():
            print(f"   {k:26s} {v}")
        for k, v in bad.items():
            for t, n, x in v[:show]:
                print(f"   [{k}] {t[:150]!r}\n      -> {str(n)[:150]!r} {x if x else ''}")


if __name__ == "__main__":
    main()

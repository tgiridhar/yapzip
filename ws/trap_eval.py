"""Hallucination traps: short documents whose target line hides a meaning that compression can flip
(more/less, only if/unless, subject phrases, units, hedges, sarcasm, speakers, swapped numbers...).

Each trap (data/traps/*.jsonl): {"id", "type", "lines": [...], "target": i, "must": [[alt, ...], ...], "must_not": [...]}.
The binary runs on each trap on its own; the target line's note is then checked:
  flip     a must_not word or phrase appears (the meaning changed: a hallucination)
  garble   a must group is missing from a written note (a meaning word was dropped or swapped)
  dropped  the target line was not kept (lost, not wrong)
  verbatim the note was '=' (copied: cannot be wrong)
  ok       all checks pass

    .venv/bin/python ws/trap_eval.py TAG [--model models/ws/x.gguf]
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ws.dev_eval import align, run  # noqa: E402

TRAPS = ROOT / "data" / "traps"


def load_traps():
    out = []
    for f in sorted(TRAPS.glob("*.jsonl")):
        out += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return out


def has(text, s):
    return re.search(r"(?<![a-z0-9])" + re.escape(s.lower()) + r"(?![a-z0-9])", text.lower()) is not None


def judge(t, note):
    if note is None:
        return "dropped", []
    if note.strip() == "=":
        return "verbatim", []
    flips = [s for s in t.get("must_not", []) if has(note, s)]
    missing = [g for g in t["must"] if not any(a.lower() in note.lower() for a in g)]
    if flips:
        return "flip", flips
    if missing:
        return "garble", missing
    return "ok", []


def one(t, model):
    txt = "\n".join(t["lines"]) + "\n"
    out, p = run(txt, model)
    notes = align(t["lines"], p, out)
    note = notes[t["target"]]
    verdict, why = judge(t, note)
    return dict(id=t["id"], type=t["type"], verdict=verdict, why=why, note=note, p=p[t["target"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--model")
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    traps = load_traps()
    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(lambda t: one(t, a.model), traps))
    out = ROOT / "results" / "ws" / "traps" / f"{a.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=0))
    tot = {}
    for r in res:
        tot[r["verdict"]] = tot.get(r["verdict"], 0) + 1
    by_type = {}
    for r in res:
        d = by_type.setdefault(r["type"], {})
        d[r["verdict"]] = d.get(r["verdict"], 0) + 1
    print(f"{a.tag}: {len(res)} traps  " + "  ".join(f"{k} {v}" for k, v in sorted(tot.items())))
    for ty, d in sorted(by_type.items()):
        print(f"   {ty:14s} " + "  ".join(f"{k} {v}" for k, v in sorted(d.items())))


if __name__ == "__main__":
    main()

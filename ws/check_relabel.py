"""Validate a relabeled training file against its source (relabel v4, data/STYLE.md "Quality bar").

    python3 ws/check_relabel.py SOURCE.jsonl RELABELED.jsonl [--show N]   (or SOURCE.json RELABELED.json for one document)

Hard errors (exit 1): chunk count, genre, unit ids or texts changed; keep not a bool; a kept unit without a note,
or a dropped unit with one; a fenced code block not kept as "=".
Soft flags (printed): numbers/IDs in the unit missing from the note (vote counts like (+33) excluded), words in
the note the unit doesn't support, a unit negation the note lost. Each flag is worth a second look, not a verdict.
"""

import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ws.halluc import NEG, NUM, check  # noqa: E402

VOTES = re.compile(r"\(\s*[+-]?\d[\d,.]*k?\s*\)|·\s*[\d,.]+k?\s*(?:points?|upvotes?|likes?|votes?)\b|"
                   r"commented\s*·\s*[A-Z][a-z]{2,8}\.? \d{1,2}, \d{4}", re.I)
NOTE_NEG = re.compile(r"\b(NOT|NEVER|NO|NONE|not|never|no|none|nobody|nothing|without|w/o|cannot|zero|neither|nor|"
                      r"stop|avoid|don['’]t|doesn['’]t|didn['’]t|won['’]t|can['’]t|isn['’]t|aren['’]t|wasn['’]t|cant|dont)\b|n['’]t",
                      re.I)


def load(p):
    if str(p).endswith(".json"):          # one labeled document (data/slop_dev, data/pasted): a single chunk
        return [json.load(open(p))]
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    src, out = load(sys.argv[1]), load(sys.argv[2])
    show = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--show=")), 3))
    errors, flags = [], {"lost number": [], "invented": [], "lost negation": []}
    if len(src) != len(out):
        errors.append(f"chunk count {len(src)} -> {len(out)}")
    kept = notes = 0
    ratios = []
    for ci, (a, b) in enumerate(zip(src, out)):
        if a.get("genre") != b.get("genre"):
            errors.append(f"chunk {ci}: genre changed")
        ua, ub = a["units"], b["units"]
        if [(u["id"], u["text"]) for u in ua] != [(u["id"], u["text"]) for u in ub]:
            errors.append(f"chunk {ci}: unit ids/texts changed")
            continue
        for u in ub:
            k, n = u.get("keep"), u.get("note")
            if not isinstance(k, bool):
                errors.append(f"chunk {ci} u{u['id']}: keep not bool")
            elif k and not (isinstance(n, str) and n.strip()):
                errors.append(f"chunk {ci} u{u['id']}: kept without note")
            elif not k and n is not None:
                errors.append(f"chunk {ci} u{u['id']}: dropped unit has a note")
            if u["text"].lstrip().startswith("```") and not (k and n == "="):
                errors.append(f"chunk {ci} u{u['id']}: fenced code not kept as '='")
            if not k or n == "=":
                kept += bool(k)
                continue
            kept += 1
            notes += 1
            t = u["text"]
            ratios.append(len(n) / max(1, len(t)))
            body = VOTES.sub("", t)
            nums = [x for x in NUM.findall(body) if len(x) > 1]
            missing = [x for x in nums if x not in n and x.replace(",", "") not in n.replace(",", "")]
            tag = re.match(r"\s*\[[^\]]*\]", t)
            if tag:   # chat timestamps in the tag are copied with the tag; don't double-count them
                missing = [x for x in missing if x not in tag.group(0)]
            if missing:
                flags["lost number"].append((t, n, missing))
            inv = [w for k2, w in check(t, n)]
            if inv:
                flags["invented"].append((t, n, inv))
            if NEG.search(t) and not NOTE_NEG.search(n):
                flags["lost negation"].append((t, n, []))
    print(f"{sys.argv[2]}: {len(out)} chunks, {kept} kept, {notes} notes, "
          f"median note/unit {statistics.median(ratios) if ratios else 0:.2f}")
    for k, v in flags.items():
        print(f"   {k:14s} {len(v):4d} ({100 * len(v) / max(1, notes):.1f}% of notes)")
        for t, n, x in v[:show]:
            print(f"      {t[:140]!r}\n        -> {n[:140]!r} {x if x else ''}")
    if errors:
        print(f"   HARD ERRORS ({len(errors)}):", *errors[:20], sep="\n      ")
        sys.exit(1)
    print("   OK: structure valid")


if __name__ == "__main__":
    main()

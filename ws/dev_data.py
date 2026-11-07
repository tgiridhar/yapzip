"""Dev articles for weight edits: data/slop_dev/<name>.json, long AI-written articles labeled unit by unit.

These are for building and tuning edits. The test articles (data/raw/slop_*.txt) and the 14 eval inputs stay
untouched until the final check.

    python3 ws/dev_data.py check NAME     # validate data/slop_dev/NAME.json and write NAME.txt next to it
    python3 ws/dev_data.py stats          # label counts for every dev article

Each JSON file: {"genre": str, "title": str, "units": [{"id", "text", "kind", "keep", "note"}]}.
`text` is one line of the article (a paragraph, heading, bullet, step or table row) or one whole ``` fenced block.
The article is the unit texts joined by newlines; the binary splits it back into exactly these units.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEV = ROOT / "data" / "slop_dev"
KINDS = {"fact", "code", "rhetoric", "repeat", "heading", "boilerplate", "story", "cta", "question"}


def split_units(text):
    """Same as ws/model.split_units and native/yapzip.cpp: one unit per non-empty line; a ``` block is one unit."""
    units, fence, in_fence = [], "", False
    for line in text.split("\n"):
        if line.endswith("\r"):
            line = line[:-1]
        t = line.strip()
        if in_fence:
            fence += "\n" + line
            if t.startswith("```"):
                in_fence = False
                units.append(fence)
            continue
        if t.startswith("```"):
            in_fence, fence = True, t
            continue
        if t:
            units.append(t)
    if in_fence:
        units.append(fence)
    return units


def load(name):
    return json.loads((DEV / f"{name}.json").read_text())


def text(d):
    return "\n".join(u["text"] for u in d["units"]) + "\n"


def names():
    return sorted(p.stem for p in DEV.glob("*.json"))


def check(name):
    d = load(name)
    units = d["units"]
    errs = []
    split = split_units(text(d))
    want = [u["text"] for u in units]
    if split != want:
        i = next((i for i, (a, b) in enumerate(zip(split, want)) if a != b), min(len(split), len(want)))
        errs.append(f"units don't round-trip through the splitter at unit {i + 1}: "
                    f"got {split[i][:80] if i < len(split) else None!r}, want {want[i][:80] if i < len(want) else None!r}")
    for i, u in enumerate(units, 1):
        if u.get("id") != i:
            errs.append(f"unit {i}: id should be {i}")
        if u.get("kind") not in KINDS:
            errs.append(f"unit {i}: kind {u.get('kind')!r} not in {sorted(KINDS)}")
        if u.get("keep") is True and not u.get("note"):
            errs.append(f"unit {i}: kept but no note")
        if u.get("keep") is False and u.get("note") is not None:
            errs.append(f"unit {i}: dropped but has a note")
        if u.get("keep") not in (True, False):
            errs.append(f"unit {i}: keep must be true or false")
        if u.get("kind") == "code" and u.get("keep") and u.get("note") != "=":
            errs.append(f"unit {i}: code unit kept with a note other than '='")
        note = u.get("note") or ""
        if note not in ("", "="):
            src_nums = set(re.findall(r"\d+(?:[.,:]\d+)*", u["text"]))
            bad = [n for n in re.findall(r"\d+(?:[.,:]\d+)*", note) if n not in src_nums]
            if bad:
                errs.append(f"unit {i}: note has numbers not in the unit: {bad}")
    (DEV / f"{name}.txt").write_text(text(d))
    words = len(text(d).split())
    kept = sum(u["keep"] for u in units)
    kinds = {}
    for u in units:
        kinds[u["kind"]] = kinds.get(u["kind"], 0) + 1
    print(f"{name}: {len(units)} units, {words} words, {kept} kept; kinds {kinds}")
    print("errors:", errs if errs else "none")
    return not errs


def main():
    if sys.argv[1] == "check":
        sys.exit(0 if check(sys.argv[2]) else 1)
    for n in names():
        d = load(n)
        u = d["units"]
        print(f"{n:22s} units {len(u):3d}  kept {sum(x['keep'] for x in u):3d}  words {len(text(d).split()):5d}")


if __name__ == "__main__":
    main()

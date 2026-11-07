"""Un-blind judge verdicts and print a per-system table (old model vs new model vs Opus).

Usage: python3 bench/judge_score.py JUDGE_DIR [JUDGE_DIR ...]
Each dir has key.json (letter -> system per packet) and <name>.verdict.json files.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def words(p):
    return len(p.read_text().split()) if p.exists() else 0


def unfence(body):
    lines = body.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


def sections(packet):
    """Outputs and original text from a packet, split on the section headers (not on fences: outputs and
    originals can contain their own ``` code blocks)."""
    head, *rest = re.split(r"^## OUTPUT (\w)\n", packet, flags=re.M)
    outs = {letter: unfence(body) for letter, body in zip(rest[0::2], rest[1::2])}
    src = re.split(r"^## (?:KEY FACTS|QUESTIONS)", head.split("## ORIGINAL TEXT\n", 1)[1], flags=re.M)[0]
    return outs, unfence(src)


def main():
    tot = defaultdict(lambda: defaultdict(int))
    per_file = []
    errors = []
    for d in map(Path, sys.argv[1:]):
        key = json.loads((d / "key.json").read_text())
        for name, slots in sorted(key.items()):
            vf = d / f"{name}.verdict.json"
            if not vf.exists():
                print("missing verdict:", vf)
                continue
            v = json.loads(vf.read_text())
            packet = (d / f"{name}.packet.md").read_text()
            outs, src = sections(packet)
            row = {"file": name}
            for letter, system in ((l, s) for l, slot in slots.items() for s in slot.split("+")):
                g = v[letter]
                labels = list(g["facts"].values())
                t = tot[system]
                t["files"] += 1
                t["words"] += len(outs.get(letter, "").split())
                t["in_words"] += len(src.split())
                t["facts"] += len(labels)
                for lab in ("kept", "partial", "wrong", "missing"):
                    t[lab] += sum(l == lab for l in labels)
                t["errors"] += len(g.get("errors", []))
                t["err_meaning"] += sum(e.get("severity") == "meaning" for e in g.get("errors", []))
                t["err_minor"] += sum(e.get("severity") == "minor" for e in g.get("errors", []))
                t["noise"] += int(g.get("noise", 0))
                t["repeats"] += int(g.get("repeats", 0))
                if g.get("code_intact") in ("yes", "no"):
                    t["code_n"] += 1
                    t["code_ok"] += g["code_intact"] == "yes"
                qa = g.get("qa") or {}
                t["qa_n"] += len(qa)
                t["qa_ok"] += sum(bool(q.get("correct")) for q in qa.values())
                t["wins"] += v["ranking"][0] == letter
                t["rank_sum"] += v["ranking"].index(letter) + 1
                row[system] = f"{sum(l == 'kept' for l in labels)}/{len(labels)} kept, {len(g.get('errors', []))} err"
                errors += [(name, system, e.get("quote", ""), e.get("why", "")) for e in g.get("errors", [])]
            row["winner"] = slots[v["ranking"][0]]
            per_file.append(row)

    systems = sorted((s for s in tot if s != "opus"), key=str) + (["opus"] if "opus" in tot else [])
    print(f"{'':22s}" + "".join(f"{s:>12s}" for s in systems))

    def line(label, f):
        print(f"{label:22s}" + "".join(f"{f(tot[s]):>12s}" for s in systems))

    line("files judged", lambda t: str(t["files"]))
    line("output size", lambda t: f"{t['words'] / t['in_words']:.0%}")
    line("facts kept / 100 words", lambda t: f"{100 * t['kept'] / max(1, t['words']):.1f}")
    line("facts kept", lambda t: f"{t['kept'] / t['facts']:.0%}")
    line("facts partial", lambda t: f"{t['partial'] / t['facts']:.0%}")
    line("facts wrong", lambda t: str(t["wrong"]))
    line("facts missing", lambda t: f"{t['missing'] / t['facts']:.0%}")
    line("errors (invented etc)", lambda t: str(t["errors"]))
    if any(tot[s]["err_meaning"] + tot[s]["err_minor"] for s in systems):
        line("  meaning-changing", lambda t: str(t["err_meaning"]))
        line("  minor", lambda t: str(t["err_minor"]))
    line("noise lines", lambda t: str(t["noise"]))
    if any(tot[s]["repeats"] for s in systems):
        line("repeat lines", lambda t: str(t["repeats"]))
    line("code intact", lambda t: f"{t['code_ok']}/{t['code_n']}")
    line("QA correct", lambda t: f"{t['qa_ok']}/{t['qa_n']}" if t["qa_n"] else "-")
    line("ranked 1st", lambda t: str(t["wins"]))
    line("avg rank", lambda t: f"{t['rank_sum'] / t['files']:.2f}")

    print("\nper file:")
    for r in per_file:
        print(f"  {r['file']:22s} winner {r['winner']:5s} " + " | ".join(f"{s}: {r[s]}" for s in systems if s in r))
    print("\nerrors:")
    for name, system, q, why in errors:
        print(f"  [{system}] {name}: {q[:90]!r} -- {why[:140]}")


if __name__ == "__main__":
    main()

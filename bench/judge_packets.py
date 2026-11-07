"""Build blind judge packets: the original plus unlabeled outputs from several systems, in random order.

Usage: python3 bench/judge_packets.py OUT_DIR INPUTS_DIR name=outputs_dir [name=outputs_dir ...]
Each input <x>.txt in INPUTS_DIR gets OUT_DIR/<x>.packet.md; <x>.opus.txt next to it is added as system "opus".
Answer keys from data/gold/<x>.json are included when present. OUT_DIR/key.json maps letters to systems;
byte-identical outputs share one letter, named "sys1+sys2".
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data/gold"


def main():
    out, inputs = Path(sys.argv[1]), Path(sys.argv[2])
    dirs = dict(a.split("=", 1) for a in sys.argv[3:])
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(11)
    key = {}
    for src in sorted(p for p in inputs.glob("*.txt") if not p.stem.endswith(".opus")):
        name = src.stem
        texts = {s: (Path(d) / f"{name}.txt").read_text() for s, d in dirs.items()}
        opus = inputs / f"{name}.opus.txt"
        if opus.exists():
            texts["opus"] = opus.read_text()
        systems = {}   # byte-identical outputs share one slot, named "a+b"; judge_score credits each system
        for s, t in texts.items():
            same = next((k for k, v in systems.items() if v == t), None)
            if same:
                systems[f"{same}+{s}"] = systems.pop(same)
            else:
                systems[s] = t
        order = list(systems)
        rng.shuffle(order)
        letters = "ABCDE"[: len(order)]
        key[name] = dict(zip(letters, order))
        parts = [f"# Judge packet: {name}", "", "## ORIGINAL TEXT", "```text", src.read_text().strip(), "```", ""]
        gold = GOLD / f"{name}.json"
        if gold.exists():
            g = json.loads(gold.read_text())
            parts += ["## KEY FACTS (from the original)"] + [f"F{i + 1}. {f['fact']}" for i, f in enumerate(g["facts"])] + [""]
            parts += ["## QUESTIONS (with gold answers)"] + [f"Q{i + 1}. {q['q']}\n    gold answer: {q['a']}" for i, q in enumerate(g["qa"])] + [""]
        else:
            parts += ["## KEY FACTS", "(none provided: first list the key facts of the original yourself as F1..Fn, then grade against them)", ""]
        for letter, system in zip(letters, order):
            parts += [f"## OUTPUT {letter}", "```text", systems[system].strip() or "(empty)", "```", ""]
        (out / f"{name}.packet.md").write_text("\n".join(parts))
    (out / "key.json").write_text(json.dumps(key, indent=1))
    print(f"{len(key)} packets in {out}")


if __name__ == "__main__":
    main()

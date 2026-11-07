"""Deslop test: run yapzip on the long AI-written articles (data/raw/slop_*.txt) and score them.

    python3 bench/slop_eval.py v2=results/eval/v2 v2ws=results/eval/v2ws:models/ws/v2ws.gguf

Each argument is NAME=OUT_DIR, or NAME=OUT_DIR:MODEL.gguf to run build/yapzip with -m MODEL. Outputs go to
OUT_DIR/<article>.txt (existing outputs are reused; pass --force to rerun). The reference compressions in
results/eval/slop_inputs/<article>.opus.txt are scored too. Per article it prints key facts kept (the automated
check in bench/common.fact_kept against data/gold) and output size as a share of the input's words.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.common import GOLD, RAW, ROOT, fact_kept  # noqa: E402

BIN = ROOT / "build" / "yapzip"
REF = ROOT / "results" / "eval" / "slop_inputs"


def words(s):
    return len(s.split())


def run(src, out, model, force):
    if out.exists() and not force:
        return out.read_text(), None
    h = subprocess.run([str(BIN), "-h"], capture_output=True, text=True)
    cmd = [str(BIN)] + (["--flat"] if "--flat" in h.stderr + h.stdout else []) + (["-m", model] if model else [])
    t = time.time()
    text = subprocess.run(cmd, input=src, capture_output=True, text=True, check=True).stdout
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return text, time.time() - t


def main():
    force = "--force" in sys.argv
    only = next((x.split("=", 1)[1].split(",") for x in sys.argv[1:] if x.startswith("--only=")), None)
    systems = {}
    for a in (x for x in sys.argv[1:] if not x.startswith("--")):
        name, spec = a.split("=", 1)
        d, _, model = spec.partition(":")
        systems[name] = (Path(d), model or None)
    names = sorted(p.stem for p in RAW.glob("slop_*.txt") if (GOLD / f"{p.stem}.json").exists())
    names = [n for n in names if only is None or n in only]
    tot = {s: [0, 0, 0, 0] for s in [*systems, "opus"]}   # facts kept, facts, out words, in words
    print(f"{'article':22s} {'words':>6s}" + "".join(f"{s:>22s}" for s in tot))
    for n in names:
        src = (RAW / f"{n}.txt").read_text()
        facts = json.loads((GOLD / f"{n}.json").read_text())["facts"]
        row = f"{n:22s} {words(src):6d}"
        for s in tot:
            if s == "opus":
                p = REF / f"{n}.opus.txt"
                if not p.exists():
                    row += f"{'-':>22s}"
                    continue
                out, dt = p.read_text(), None
            else:
                d, model = systems[s]
                out, dt = run(src, d / f"{n}.txt", model, force)
            k = sum(fact_kept(f["must"], out) for f in facts)
            t = tot[s]
            t[0] += k; t[1] += len(facts); t[2] += words(out); t[3] += words(src)
            cell = f"{k}/{len(facts)} {words(out) / words(src):.0%}" + (f" {dt:.0f}s" if dt else "")
            row += f"{cell:>22s}"
        print(row, flush=True)
    print(f"{'ALL':22s} {'':6s}" + "".join(
        f"{(f'{t[0] / t[1]:.0%} facts, {t[2] / t[3]:.0%} size' if t[1] else '-'):>22s}" for t in tot.values()))


if __name__ == "__main__":
    main()

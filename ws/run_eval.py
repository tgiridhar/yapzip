"""Run the 14 eval inputs through the MLX copy of a yapzip GGUF (optionally with an edit applied) and save,
per input, the output text plus per-unit records (P(keep), note) as JSON.

    python ws/run_eval.py OUT_DIR [--gguf models/ws/v4c.gguf] [--edit results/ws/edit.npz] [--only a,b]
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ws.model import Yap, chunks, split_units  # noqa: E402

INPUTS = ROOT / "results/eval/inputs"


def eval_names():
    return sorted(p.stem for p in INPUTS.glob("*.txt") if not p.name.endswith(".opus.txt"))


def run(y, name):
    recs, out = [], []
    for c, ch in enumerate(chunks(y, split_units((INPUTS / f"{name}.txt").read_text()))):
        r = y.process(ch)
        out.append(r["out"])
        for i, u in enumerate(ch):
            n = r["notes"].get(i)
            recs.append(dict(chunk=c, i=i, unit=u, p=round(float(r["p"][i]), 5), note=n["note"] if n else None))
    return "".join(out), recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--gguf", default=str(ROOT / "models/ws/v4c.gguf"))
    ap.add_argument("--edit", default=None)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    y = Yap(a.gguf)
    if a.edit:
        from ws.edits import apply_npz
        apply_npz(y, a.edit)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in (a.only.split(",") if a.only else eval_names()):
        t0 = time.time()
        text, recs = run(y, name)
        (out / f"{name}.txt").write_text(text)
        (out / f"{name}.json").write_text(json.dumps(recs, indent=1, ensure_ascii=False))
        print(f"{name}: {len(recs)} units, {sum(r['note'] is not None for r in recs)} kept, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
